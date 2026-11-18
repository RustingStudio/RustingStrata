#!/usr/bin/env python3
"""Requantize 2D+ Q8_0 tensors in a GGUF v3 file (e.g., mtp-Q8_0.gguf) to Q4_0.

Block layout matches llama.cpp / rustingstrata `src/quant/mod.rs::q5_codes`:
- Q8_0 (type 8, 34 B / 32 elems): f16 d, 32 x int8 qs
- Q4_0 (type 2, 18 B / 32 elems): f16 d, 16 B nibbles (low nibble = elem 0..15, high nibble = elem 16..31), code - 8
"""

import argparse
import math
import os
import struct
import sys
import numpy as np

GGML_TYPE_F32 = 0
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8

# (block_elems, block_bytes)
TYPE_SIZES = {
    0: (1, 4),     # F32
    1: (1, 2),     # F16
    2: (32, 18),   # Q4_0
    3: (32, 20),   # Q4_1
    6: (32, 22),   # Q5_0
    7: (32, 24),   # Q5_1
    8: (32, 34),   # Q8_0
    10: (256, 84), # Q2_K
    11: (256, 110),# Q3_K
    12: (256, 144),# Q4_K
    13: (256, 176),# Q5_K
    14: (256, 210),# Q6_K
    30: (1, 2),    # BF16
}

KV_SCALARS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def align_up(n: int, align: int) -> int:
    return (n + align - 1) // align * align


def read_str(f) -> tuple[bytes, str]:
    n = struct.unpack("<Q", f.read(8))[0]
    raw = f.read(n)
    return raw, raw.decode("utf-8", errors="replace")


def read_kv_val(f, vtype: int):
    if vtype in KV_SCALARS:
        sz = KV_SCALARS[vtype]
        raw = f.read(sz)
        if vtype == 4:
            return struct.unpack("<I", raw)[0]
        if vtype == 5:
            return struct.unpack("<i", raw)[0]
        if vtype == 10:
            return struct.unpack("<Q", raw)[0]
        return None
    if vtype == 8:
        return read_str(f)[1]
    if vtype == 9:
        et, ln = struct.unpack("<IQ", f.read(12))
        for _ in range(ln):
            read_kv_val(f, et)
        return None
    raise ValueError(f"Unsupported GGUF KV value type: {vtype}")


def q8_0_to_q4_0(raw: bytes) -> bytes:
    """Vectorized Q8_0 -> Q4_0 requantization (34 B -> 18 B per 32 weights)."""
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
    d8 = arr[:, :2].copy().view(np.float16).reshape(-1).astype(np.float32)
    qs8 = arr[:, 2:].view(np.int8).astype(np.float32)
    v = d8[:, None] * qs8

    idx = np.argmax(np.abs(v), axis=1)
    maxv = v[np.arange(len(v)), idx]
    d4 = (maxv / -8.0).astype(np.float16)
    df = d4.astype(np.float32)

    inv_d = np.divide(1.0, df, out=np.zeros_like(df), where=df != 0.0)[:, None]
    q = np.clip(np.floor(v * inv_d + 8.5), 0, 15).astype(np.uint8)
    packed = (q[:, :16] & 0x0F) | ((q[:, 16:] & 0x0F) << 4)

    out = np.empty((len(arr), 18), dtype=np.uint8)
    out[:, :2] = d4.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = packed
    return out.tobytes()


def requant_gguf(src: str, dst: str, strip_base: bool = False) -> None:
    alignment = 32
    with open(src, "rb") as fin:
        magic, ver, n_tensors, n_kv = struct.unpack("<4sIQQ", fin.read(24))
        if magic != b"GGUF" or ver != 3:
            raise ValueError(f"Expected GGUF v3, got magic={magic!r} ver={ver}")

        kv_start = fin.tell()
        for _ in range(n_kv):
            _, key = read_str(fin)
            vtype = struct.unpack("<I", fin.read(4))[0]
            val = read_kv_val(fin, vtype)
            if key == "general.alignment" and isinstance(val, int) and val > 0:
                alignment = val
        kv_end = fin.tell()

        tensors = []
        for _ in range(n_tensors):
            raw_name, name = read_str(fin)
            ndim = struct.unpack("<I", fin.read(4))[0]
            dims = struct.unpack(f"<{ndim}Q", fin.read(8 * ndim))
            ty, off = struct.unpack("<IQ", fin.read(12))
            tensors.append((raw_name, name, dims, ty, off))

        old_data_start = align_up(fin.tell(), alignment)
        fin.seek(kv_start)
        kv_blob = fin.read(kv_end - kv_start)

        if strip_base:
            tensors = [t for t in tensors if t[1] not in ("token_embd.weight", "output.weight", "output_norm.weight")]

        new_tensors = []
        cur_off = 0
        for raw_name, name, dims, ty, old_off in tensors:
            if ty not in TYPE_SIZES:
                raise ValueError(f"Unsupported tensor type {ty} for {name}")
            be, bb = TYPE_SIZES[ty]
            n_elems = math.prod(dims)
            if n_elems % be != 0:
                raise ValueError(f"{name}: n_elems={n_elems} not divisible by block_elems={be}")
            old_bytes = (n_elems // be) * bb
            convert = (ty == GGML_TYPE_Q8_0) and (len(dims) >= 2) and (dims[0] % 32 == 0)
            new_ty = GGML_TYPE_Q4_0 if convert else ty
            nbe, nbb = TYPE_SIZES[new_ty]
            new_bytes = (n_elems // nbe) * nbb
            new_tensors.append((raw_name, name, dims, ty, new_ty, old_off, old_bytes, cur_off, new_bytes))
            cur_off = align_up(cur_off + new_bytes, alignment)

        hdr = bytearray()
        hdr += struct.pack("<4sIQQ", b"GGUF", 3, len(new_tensors), n_kv)
        hdr += kv_blob
        for raw_name, _, dims, _, new_ty, _, _, new_off, _ in new_tensors:
            hdr += struct.pack("<Q", len(raw_name)) + raw_name
            hdr += struct.pack("<I", len(dims))
            hdr += struct.pack(f"<{len(dims)}Q", *dims)
            hdr += struct.pack("<IQ", new_ty, new_off)

        pad_len = align_up(len(hdr), alignment) - len(hdr)
        hdr += b"\x00" * pad_len

        tmp = f"{dst}.tmp"
        with open(tmp, "wb") as fout:
            fout.write(hdr)
            for raw_name, name, dims, old_ty, new_ty, old_off, old_bytes, new_off, new_bytes in new_tensors:
                fin.seek(old_data_start + old_off)
                raw = fin.read(old_bytes)
                if len(raw) != old_bytes:
                    raise EOFError(f"Short read for {name}: {len(raw)} < {old_bytes}")
                out_bytes = q8_0_to_q4_0(raw) if new_ty != old_ty else raw
                if len(out_bytes) != new_bytes:
                    raise AssertionError(f"Size mismatch for {name}: {len(out_bytes)} != {new_bytes}")
                fout.write(out_bytes)
                rem = align_up(new_bytes, alignment) - new_bytes
                if rem:
                    fout.write(b"\x00" * rem)
        os.replace(tmp, dst)

    src_mb = os.path.getsize(src) / (1024 * 1024)
    dst_mb = os.path.getsize(dst) / (1024 * 1024)
    blk_old = sum(t[6] for t in new_tensors if t[1].startswith("blk.")) / (1024 * 1024)
    blk_new = sum(t[8] for t in new_tensors if t[1].startswith("blk.")) / (1024 * 1024)
    print(f"requant_mtp_q4_0: {src} ({src_mb:.1f} MiB) -> {dst} ({dst_mb:.1f} MiB)")
    print(f"MTP blk.* tensors: {blk_old:.1f} MiB -> {blk_new:.1f} MiB ({len(new_tensors)} tensors)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Requantize Q8_0 GGUF tensors to Q4_0 for fast MTP loading")
    ap.add_argument("src", help="Input GGUF path")
    ap.add_argument("dst", help="Output GGUF path")
    ap.add_argument("--strip-base", action="store_true", help="Omit token_embd/output weights duplicated from base model")
    args = ap.parse_args()
    requant_gguf(args.src, args.dst, strip_base=args.strip_base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
