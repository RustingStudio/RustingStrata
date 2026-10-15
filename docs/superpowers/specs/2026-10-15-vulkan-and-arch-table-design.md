# RustingStrata: Vulkan backend and architecture table — Design

Date: 2026-10-15
Status: approved in brainstorming

## Goal

1. Run on AMD (and other non-NVIDIA) GPUs through Vulkan.
2. Make a new Qwen architecture cost a table entry and a parity run, not a new copy of the forward pass.

Today the model graph exists in three places (`src/model.rs` CPU oracle, `src/gpu_model.rs` decode,
`src/gpu_model/prefill.rs` prefill) and is welded to cudarc. A second backend would double the GPU copies, and every new
architecture would have to be added to all of them. The work below removes the backend coupling first, then the
architecture coupling, then adds Vulkan.

## Decisions

- **Approach A:** a `Backend` trait that mirrors the current `Gpu` API. CUDA kernels stay as they are (no risk to the
  v1 speed numbers). Vulkan kernels are separate GLSL compute shaders, compiled offline to SPIR-V and committed.
  Rejected: Slang single source (rewrites tuned CUDA with inline PTX, adds `slangc`), HIP (ROCm/Linux only, does not
  cover the RDNA2 iGPU we test on, no Windows/Intel).
- **Architectures, in order:** `qwen3` (dense), `qwen3moe`, `qwen3next`. Not now: `qwen2` (q/k/v bias), vision.
- **Hardware for AMD testing:** the Ryzen 7600X RDNA2 iGPU, enabled in BIOS, driven by RADV. It has no WMMA, so it
  exercises the fallback paths. Development happens on the RTX 3060 through its Vulkan driver. Speed on discrete AMD
  cards is "reasonable, reported by users"; there is no AMD speed gate.
- The CPU reference (`src/model.rs`) stays a separate, independent implementation. It is the oracle and must not share
  graph code with the GPU path.

## Part 1: Backend seam (CUDA only, no behavior change)

New `src/backend.rs`:

```rust
pub struct Buf { pub addr: u64, pub len: usize, /* owner handle, freed on drop */ }

pub trait Backend {
    // Memory
    fn alloc(&self, bytes: usize) -> anyhow::Result<Buf>;
    fn htod(&self, src: &[u8], dst: &Buf, off: usize) -> anyhow::Result<()>;
    fn dtoh(&self, src: &Buf, off: usize, dst: &mut [u8]) -> anyhow::Result<()>;
    fn dtod(&self, src: &Buf, dst: &Buf) -> anyhow::Result<()>;
    fn memset(&self, dst: &Buf) -> anyhow::Result<()>;
    fn mem_info(&self) -> anyhow::Result<(usize, usize)>;
    fn upload(&self, t: &Tensor) -> anyhow::Result<GpuTensor>;
    fn sync(&self) -> anyhow::Result<()>;
    // Replay (CUDA graph capture / Vulkan command buffer)
    fn record(&self, key: u64, f: &mut dyn FnMut() -> anyhow::Result<()>) -> anyhow::Result<()>;
    fn replay(&self, key: u64) -> anyhow::Result<bool>; // false: not recorded yet
    // Host-mapped CPU expert mailbox: (host pointer, device address)
    fn host_mailbox(&self, words: usize) -> anyhow::Result<(*mut u32, u64)>;
    // Ops: every current public `Gpu` method, same u64-address signatures
    // (gemv, gemvn, gemvq, quant_xq, moe_gemv, gemm, gemm_dense, map_hits, rms_norm, rms_norm_q, argmax_rows,
    //  l2_norm, rope_neox, add, swiglu, ssm_gates, conv_silu, conv_seq, deltanet, attn_prefill, attn_decode,
    //  attn_decode_gq, kv_store, kv_words, router_topk, moe_publish, moe_combine, moe_combine_seq, norm_gate, ...)
}
```

- `Buf` replaces `CudaSlice` in `gpu_model.rs`, `gpu_model/prefill.rs`, `expert_cache.rs`. `cpu_worker.rs` gets its
  mailbox from `host_mailbox` instead of `PinnedHostSlice`. After part 1, `cudarc` is imported only under `src/cuda/`.
- `GpuModel` holds `Box<dyn Backend>`. Dynamic dispatch is negligible: the forward replays from recorded graphs.
- The exact method list is whatever `Gpu` exposes when part 1 starts; the trait is extracted, not redesigned.

**Gate:** `strata bench --gpu --pp 512,4096 --tg 128 --reps 3` inside the measured v1.0.0 ranges (prefill 4K 2089-2096,
decode after 4K 95.4-95.8, short decode 100.6-101.5 tok/s, same CPU-load caveats as `docs/LOOP_LOG.md`);
`tools/parity.py` agreement unchanged (98.9%); `cargo test --release` passes.

## Part 2: Architecture table

The table is one `match arch` in `src/config.rs`. No per-arch modules. New `ModelConfig` fields:

- `layers: Vec<LayerKind>`, `enum LayerKind { Delta, Attn }`. Hybrid archs (`qwen35`, `qwen35moe`, `qwen3next`) build it
  from `full_attention_interval`; `qwen3` and `qwen3moe` are all `Attn`. Replaces `is_recurrent()`. KV size counts
  `Attn` layers instead of `n_layer / full_attention_interval`. `ssm_*` keys become optional (required only when a
  `Delta` layer exists).
- `q_gate: bool`. `qwen35*` and `qwen3next` have `attn_q` of width `2 * n_head * head_dim` with a sigmoid output gate;
  `qwen3*` has plain `q` and no gate. `attn_decode`, `attn_decode_gq` and `attn_prefill` take a flag (gate pointer 0 =
  no gate).
- `aliases: &'static [(&'static str, &'static str)]`: canonical tensor suffix to the arch's suffix. Known entries:
  `post_attention_norm` is `ffn_norm` in `qwen3`/`qwen3moe` and `attn_post_norm` in `qwen3next`. `Weights::layer` tries
  the alias. All code keeps using canonical names.
- Fused `qwen3next` tensors: `ssm_beta_alpha` (beta and alpha in one matrix) and the legacy `ssm_in` (q, k, v, z in one
  matrix, present instead of `attn_qkv` + `attn_gate` in older GGUFs). One GEMV into a scratch buffer, then a reorder
  kernel splits it following llama.cpp `src/models/qwen3next.cpp`. The CPU oracle implements the split first and the
  GPU is checked against it.
- `router_topk` handles at most 256 experts (`TOPK_R`); add the variant for 512 (`qwen3next`).
- Shared expert stays optional (already `Option<Shared>`). MTP stays conditional on `nextn` layers.

Order: `qwen3`, then `qwen3moe`, then `qwen3next`. Each goes into the CPU oracle first, then CUDA.

**Gate per architecture:** CPU oracle and CUDA produce tokens identical to llama.cpp for "The capital of France is"
(24 tokens, greedy); `tools/parity.py` greedy agreement with llama.cpp at least 98.5% on the 20-prompt set (small
models have more near-ties). Models: Qwen3 dense (a small one, e.g. 4B/8B), Qwen3-30B-A3B, Qwen3-Next-80B-A3B (the
smallest quant that fits; 80B at Q2 is about 27-30 GB against 32 GB RAM, so this may need IQ1/IQ2 quants).

## Part 3: Vulkan backend

`src/vulkan/` implements `Backend` with `ash` (dynamically loaded `libvulkan`, no SDK at build time). `--gpu` uses CUDA
when available, otherwise Vulkan; `--backend cuda|vulkan` and `--device N` override. No cargo features.

### Device requirements

Required (refuse at load naming the missing feature and the device):
- Vulkan 1.2+, `bufferDeviceAddress`, `storageBuffer8BitAccess`, `storageBuffer16BitAccess`, `shaderInt8`,
  `shaderFloat16`.
- Subgroup size 32 via `VK_EXT_subgroup_size_control` / Vulkan 1.3 (`requiredSubgroupSize = 32`). Wave64-only GCN
  (Polaris, Vega) is not supported.

Optional:
- `shaderIntegerDotProduct` for the dp4a equivalent; else unpack bytes in the shader.
- `VK_KHR_cooperative_matrix` for the prefill GEMM (replaces `mma.sync`); else a tiled dp4a GEMM (RDNA2 iGPU).
- `VK_EXT_external_memory_host` for expert streaming from the mmap; else staging-buffer copies.

### Shaders

- GLSL compute shaders in `src/vulkan/shaders/*.comp`. Pointers come in push constants as `u64` through
  `GL_EXT_buffer_reference`, so every op keeps the CUDA signature.
- Quant type and tile sizes are specialization constants. A `switch` on the type constant replaces the `QI<T>` /
  `deq8<T>` template specializations; the driver removes dead branches.
- Warp shuffles become `subgroupShuffleXor` / `subgroupAdd`. `cp.async` becomes plain loads into shared memory.
- `tools/gen_spirv.sh` runs `glslc` and writes `src/vulkan/spv/*.spv`, which are committed and loaded with
  `include_bytes!`. A test regenerates them when `glslc` is installed and fails if the committed files differ.
- Pipeline cache on disk in `~/.cache/rusting_strata/`.

### Memory and replay

- One `VkBuffer` (with device-address usage) and one allocation per `Buf`; about 600 allocations, under the 4096
  `maxMemoryAllocationCount`. `ponytail:` one allocation per buffer; a sub-allocator if a driver is slow with it.
- `record` records a command buffer with a compute-to-compute memory barrier between dispatches; `replay` submits it and
  waits on a fence. Per-token values are already read from `step_buf`, so recorded buffers replay unchanged.

### CPU expert mailbox

- Host-visible, host-coherent, device-addressable buffer. `moe_combine` spins on the done flag as the CUDA kernel does.
- Vulkan does not guarantee forward progress for a spinning shader, and Windows TDR can reset the device. The spin is
  bounded; on timeout the shader writes an error flag and the host reports it.
- Fallback only if spinning fails on RADV: split the command buffer at each MoE layer and synchronize with timeline
  semaphores (about 40 extra submits per token).

### Steps (each ends runnable)

- **V1 dense decode** (Qwen3.5-2B): all GEMV quant types, norms, rope, `attn_decode`, `deltanet`, conv, replay.
  Gate: greedy tokens identical to CUDA on the 3060 (Vulkan driver) and on the iGPU (RADV).
- **V2 MoE:** CPU mailbox, expert cache and adaptation (Qwen3.6-35B-A3B). Gate: greedy tokens identical to CUDA.
- **V3 prefill and MTP:** cooperative-matrix GEMM plus the dp4a fallback, `attn_prefill`, `conv_seq`, `deltanet_seq`,
  MTP verify and rollback. Gate: `tools/parity.py --extra "--gpu --backend vulkan"` at least 98.5%; MTP on and off give
  identical greedy tokens.
- **V4 speed pass:** compare with llama.cpp's Vulkan backend on the 3060 (stand-in for AMD) and record iGPU numbers. Goal:
  not slower than llama.cpp Vulkan on the same file. No AMD discrete gate.

## Testing

- `tests/cuda.rs` becomes `tests/backend.rs`: each kernel test runs for every backend available at run time against the
  same CPU reference and tolerance. A missing backend is skipped with a printed note.
- `tests/gpu_model.rs` runs per available backend the same way.
- `tools/parity.py` needs no change (`--extra` passes backend flags).
- New architectures: CPU oracle vs llama.cpp, GPU vs oracle, GPU vs llama.cpp.

## Error handling

- Missing required Vulkan feature: refuse at load, naming feature and device.
- `--gpu` with no usable backend: error naming the libraries that were not found.
- Device lost or mailbox spin timeout: the request fails (server: HTTP 500) and session state is reset, as CUDA errors
  do today.
- Unknown architecture: refuse and list the supported ones.

## Docs

README: build section per backend, a backend column in the numbers tables, supported architectures list.
`docs/LOOP_LOG.md`: one entry per step with measurements, as before.

## Order

Part 1, then part 2 and part 3 (independent of each other; part 2 first because it is smaller). Each part gets its own
implementation plan in `docs/superpowers/plans/`.
