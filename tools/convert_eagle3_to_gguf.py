#!/usr/bin/env python3
"""Convert HuggingFace LlamaForCausalLMEagle3 checkpoints to GGUF v3 for RustingStrata.

Preserves 2D linear weights in native BF16 (matching RustingStrata's Vulkan/CPU
BF16 kernels with zero dequantization loss), converts 1D norms to F32, keeps Q/K
projections in HuggingFace NeoX half-split order (`eagle3.rope.neox = true`),
and stores `d2t` as absolute target vocabulary IDs (I32).
"""

import argparse
import json
import os
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
ALIGNMENT = 32

# GGUF metadata value types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9

# GGML tensor types (from src/quant/mod.rs & ggml.h)
GGML_TYPE_F32 = 0
GGML_TYPE_I32 = 26
GGML_TYPE_BF16 = 30

TENSOR_MAP = {
    "fc.weight": ("fc.weight", GGML_TYPE_BF16),
    "midlayer.input_layernorm.weight": ("blk.0.attn_norm.weight", GGML_TYPE_F32),
    "midlayer.hidden_norm.weight": ("blk.0.attn_norm_2.weight", GGML_TYPE_F32),
    "midlayer.self_attn.q_proj.weight": ("blk.0.attn_q.weight", GGML_TYPE_BF16),
    "midlayer.self_attn.k_proj.weight": ("blk.0.attn_k.weight", GGML_TYPE_BF16),
    "midlayer.self_attn.v_proj.weight": ("blk.0.attn_v.weight", GGML_TYPE_BF16),
    "midlayer.self_attn.o_proj.weight": ("blk.0.attn_output.weight", GGML_TYPE_BF16),
    "midlayer.post_attention_layernorm.weight": ("blk.0.ffn_norm.weight", GGML_TYPE_F32),
    "midlayer.mlp.gate_proj.weight": ("blk.0.ffn_gate.weight", GGML_TYPE_BF16),
    "midlayer.mlp.up_proj.weight": ("blk.0.ffn_up.weight", GGML_TYPE_BF16),
    "midlayer.mlp.down_proj.weight": ("blk.0.ffn_down.weight", GGML_TYPE_BF16),
    "norm.weight": ("output_norm.weight", GGML_TYPE_F32),
    "lm_head.weight": ("output.weight", GGML_TYPE_BF16),
}


def bf16_bytes_to_f32(raw: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32).copy()


def f32_to_bf16_bytes(arr: np.ndarray) -> bytes:
    u32 = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    u16 = (u32 >> 16).astype(np.uint16)
    return u16.tobytes()


def f16_to_f32(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.float16).astype(np.float32)


def load_safetensors(path: Path) -> Dict[str, Tuple[Tuple[int, ...], str, bytes]]:
    out = {}
    with open(path, "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hdr_len))
        data_off = 8 + hdr_len
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            start, end = info["data_offsets"]
            f.seek(data_off + start)
            out[name] = (tuple(info["shape"]), info["dtype"], f.read(end - start))
    return out


def load_torch_bin(path: Path) -> Dict[str, Tuple[Tuple[int, ...], str, bytes]]:
    import torch

    sd = torch.load(path, map_location="cpu", weights_only=True)
    dtype_map = {
        torch.bfloat16: "BF16",
        torch.float16: "F16",
        torch.float32: "F32",
        torch.int64: "I64",
        torch.int32: "I32",
        torch.bool: "BOOL",
    }
    out = {}
    for name, tensor in sd.items():
        t = tensor.contiguous().cpu()
        dt = dtype_map[t.dtype]
        if dt == "BF16":
            raw = t.view(torch.uint16).numpy().tobytes()
        else:
            raw = t.numpy().tobytes()
        out[name] = (tuple(t.shape), dt, raw)
    return out


def load_checkpoint(src_dir: Path) -> Dict[str, Tuple[Tuple[int, ...], str, bytes]]:
    st_path = src_dir / "model.safetensors"
    pt_path = src_dir / "pytorch_model.bin"
    if st_path.exists():
        return load_safetensors(st_path)
    if pt_path.exists():
        return load_torch_bin(pt_path)
    raise FileNotFoundError(f"neither model.safetensors nor pytorch_model.bin found in {src_dir}")


def pack_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def pack_kv(key: str, vtype: int, payload: bytes) -> bytes:
    return pack_str(key) + struct.pack("<I", vtype) + payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src_dir", type=Path, help="HF EAGLE3 checkpoint directory")
    ap.add_argument("out_gguf", type=Path, help="Output GGUF file path")
    ap.add_argument("--target-layers", type=int, default=48, help="Target model layer count (default: 48)")
    ap.add_argument("--extract-layers", type=str, default=None, help="Comma-separated target layer indices, e.g. 2,24,45")
    args = ap.parse_args()

    with open(args.src_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if args.extract_layers:
        extract_layers = [int(x) for x in args.extract_layers.split(",")]
    elif cfg.get("eagle_aux_hidden_state_layer_ids"):
        extract_layers = [int(x) for x in cfg["eagle_aux_hidden_state_layer_ids"]]
    else:
        n = args.target_layers
        extract_layers = [2, n // 2, n - 3]
    assert len(extract_layers) == 3, f"expected 3 extract_layers, got {extract_layers}"

    raw_tensors = load_checkpoint(args.src_dir)
    n_embd = int(cfg["hidden_size"])
    n_ff = int(cfg["intermediate_size"])
    n_head = int(cfg["num_attention_heads"])
    n_head_kv = int(cfg["num_key_value_heads"])
    head_dim = int(cfg.get("head_dim", n_embd // n_head))
    eps = float(cfg.get("rms_norm_eps", 1e-6))
    rope_theta = float(cfg.get("rope_theta", 1000000.0))
    norm_before_residual = bool(cfg.get("norm_before_residual", False))
    draft_vocab = int(cfg.get("draft_vocab_size", 32000))

    prepared: List[Tuple[str, Tuple[int, ...], int, bytes]] = []
    for hf_name, (gguf_name, dst_ty) in TENSOR_MAP.items():
        if hf_name not in raw_tensors:
            raise KeyError(f"missing expected EAGLE3 tensor {hf_name}")
        shape, src_dt, raw = raw_tensors[hf_name]
        # GGUF dims are reversed relative to C row-major shape: [in_features, out_features] for 2D.
        gguf_dims = tuple(reversed(shape))
        if dst_ty == GGML_TYPE_BF16:
            if src_dt == "BF16":
                out_bytes = raw
            elif src_dt == "F32":
                out_bytes = f32_to_bf16_bytes(np.frombuffer(raw, dtype=np.float32))
            elif src_dt == "F16":
                out_bytes = f32_to_bf16_bytes(f16_to_f32(raw))
            else:
                raise ValueError(f"{hf_name}: unsupported dtype {src_dt} -> BF16")
        elif dst_ty == GGML_TYPE_F32:
            if src_dt == "F32":
                out_bytes = raw
            elif src_dt == "BF16":
                out_bytes = np.ascontiguousarray(bf16_bytes_to_f32(raw), dtype=np.float32).tobytes()
            elif src_dt == "F16":
                out_bytes = np.ascontiguousarray(f16_to_f32(raw), dtype=np.float32).tobytes()
            else:
                raise ValueError(f"{hf_name}: unsupported dtype {src_dt} -> F32")
        else:
            raise ValueError(f"unexpected dst_ty {dst_ty}")
        prepared.append((gguf_name, gguf_dims, dst_ty, out_bytes))

    # d2t: HF stores relative offsets so target_id[i] = i + d2t[i]. Store absolute IDs as I32.
    d2t_shape, d2t_dt, d2t_raw = raw_tensors["d2t"]
    assert len(d2t_shape) == 1 and d2t_shape[0] == draft_vocab, f"unexpected d2t shape {d2t_shape}"
    if d2t_dt == "I64":
        offsets = np.frombuffer(d2t_raw, dtype=np.int64)
    elif d2t_dt == "I32":
        offsets = np.frombuffer(d2t_raw, dtype=np.int32).astype(np.int64)
    else:
        raise ValueError(f"unexpected d2t dtype {d2t_dt}")
    abs_ids = (np.arange(draft_vocab, dtype=np.int64) + offsets).astype(np.int32)
    prepared.append(("d2t", (draft_vocab,), GGML_TYPE_I32, np.ascontiguousarray(abs_ids).tobytes()))

    kvs: List[bytes] = [
        pack_kv("general.architecture", GGUF_TYPE_STRING, pack_str("eagle3")),
        pack_kv("general.alignment", GGUF_TYPE_UINT32, struct.pack("<I", ALIGNMENT)),
        pack_kv("eagle3.block_count", GGUF_TYPE_UINT32, struct.pack("<I", 1)),
        pack_kv("eagle3.embedding_length", GGUF_TYPE_UINT32, struct.pack("<I", n_embd)),
        pack_kv("eagle3.feed_forward_length", GGUF_TYPE_UINT32, struct.pack("<I", n_ff)),
        pack_kv("eagle3.attention.head_count", GGUF_TYPE_UINT32, struct.pack("<I", n_head)),
        pack_kv("eagle3.attention.head_count_kv", GGUF_TYPE_UINT32, struct.pack("<I", n_head_kv)),
        pack_kv("eagle3.attention.key_length", GGUF_TYPE_UINT32, struct.pack("<I", head_dim)),
        pack_kv("eagle3.attention.layer_norm_rms_epsilon", GGUF_TYPE_FLOAT32, struct.pack("<f", eps)),
        pack_kv("eagle3.rope.dimension_count", GGUF_TYPE_UINT32, struct.pack("<I", head_dim)),
        pack_kv("eagle3.rope.freq_base", GGUF_TYPE_FLOAT32, struct.pack("<f", rope_theta)),
        pack_kv("eagle3.rope.neox", GGUF_TYPE_BOOL, struct.pack("<B", 1)),
        pack_kv("eagle3.norm_before_residual", GGUF_TYPE_BOOL, struct.pack("<B", int(norm_before_residual))),
        pack_kv(
            "eagle3.target_layers",
            GGUF_TYPE_ARRAY,
            struct.pack("<IQ", GGUF_TYPE_INT32, len(extract_layers))
            + b"".join(struct.pack("<i", v) for v in extract_layers),
        ),
    ]

    tensor_infos: List[bytes] = []
    cur_off = 0
    for name, dims, ty, data in prepared:
        cur_off = (cur_off + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        info = (
            pack_str(name)
            + struct.pack("<I", len(dims))
            + b"".join(struct.pack("<Q", d) for d in dims)
            + struct.pack("<IQ", ty, cur_off)
        )
        tensor_infos.append(info)
        cur_off += len(data)

    header = GGUF_MAGIC + struct.pack("<IQQ", GGUF_VERSION, len(prepared), len(kvs))
    meta_block = header + b"".join(kvs) + b"".join(tensor_infos)
    pad = (-len(meta_block)) % ALIGNMENT

    args.out_gguf.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = args.out_gguf.with_suffix(".gguf.tmp")
    with open(tmp_out, "wb") as f:
        f.write(meta_block)
        f.write(b"\x00" * pad)
        written = 0
        for _, _, _, data in prepared:
            tpad = (-written) % ALIGNMENT
            if tpad:
                f.write(b"\x00" * tpad)
                written += tpad
            f.write(data)
            written += len(data)
    os.replace(tmp_out, args.out_gguf)
    print(
        f"wrote {args.out_gguf} ({os.path.getsize(args.out_gguf) / (1024 * 1024):.1f} MiB): "
        f"extract_layers={extract_layers}, heads={n_head}x{head_dim}, kv={n_head_kv}, ff={n_ff}, draft_vocab={draft_vocab}"
    )


if __name__ == "__main__":
    main()
