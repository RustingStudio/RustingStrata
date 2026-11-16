# HIP (ROCm) backend

`--backend hip` runs the same `src/cuda/kernels.cu` on an AMD card through HIP, without `hipcc`, ROCm headers or a C++
build step: the kernels are compiled at run time by hiprtc and launched through the HIP runtime, both loaded by name
(`libamdhip64.so`, `libhiprtc.so`), the way `src/cuda` loads NVRTC. `cargo build --release` is the whole build.

**Status: experimental.** Brought up on an RX 9070 XT (gfx1201, ROCm 7.2.3). With `--fixed-experts` (deterministic
placement) the greedy output is **bit-identical to the Vulkan backend** on the four local models tried (Qwen3.6-35B-A3B,
Ornith-1.5-35B-A3B, Qwen3.5-9B dense, Qwen3.8-27B dense) and to its own CUDA-core prefill on 5 of 6 prompts — the odd
one out differs at a near-tie, where the f16 tensor-core summation order picks another token (the same class of
difference the CUDA and Vulkan backends have; with adaptive placement, greedy tokens can also change between runs, as
they do on the other backends). Not yet ported: the host-side pipeline details of the CUDA backend (pinned ring staging
for `htod`, side stream for `beside`).

## What runs

Everything the Vulkan backend runs, through the same kernels:

| | HIP | Vulkan (same machine, same model) |
|---|---|---|
| Prefill, pp 1024, calibrated | 1577-2072 tok/s | 1344 tok/s |
| Prefill, pp 4096 | 2106 tok/s | (not run) |
| Decode, tg 48, calibrated | 188-222 tok/s | 195 tok/s |
| Qwen3.8-27B dense, prefill / decode | 705 / 50.3 tok/s | 497 / 55.5 tok/s |

Qwen3.6-35B-A3B UD-Q4_K_XL (first rows) and Qwen3.8-27B UD-IQ3_S, `--ctx 4096 --vram-margin 256`, one process each,
`calibrate` run for each backend (chunk 4096; HIP picks `--mtp 3`, Vulkan `--mtp 1`). Prefill runs on the card's matrix
cores (below) and is ahead of Vulkan; decode is within a few percent in either direction — the spread over repeated runs
is larger than the difference, because each backend finds its own expert placement and that moves the MTP draft
acceptance. A dense model's prefill is 1.4x ahead and its decode 10% behind (memory-bandwidth bound).

## How it works

- `src/hip/compat.h` is prepended to `kernels.cu`: it supplies the CUDA intrinsics the source uses (`__shfl_*_sync`,
  `__ballot_sync`, `__dp4a`, `__byte_perm`, `__vsub4`, `__vadd4`, `__nanosleep`, `__trap`, the f16 conversions) with
  their AMD equivalents, and sets `__CUDA_ARCH__` to the pre-Volta level so the source takes its portable paths:
  per-lane shared loads instead of `ldmatrix`, plain copies instead of `cp.async`, and the CUDA-core multiplicands
  instead of `mma.sync`.
- The kernels compile as **wave32** (lane = `threadIdx.x & 31`), which is the default wavefront size on RDNA and the
  assumption the CUDA source already makes about a warp. The backend refuses a device that reports a 64-wide wave.
- **Matrix cores**: the prefill GEMM (`gemm_q` in `kernels.cu`) takes an AMD branch under `GEMM_WMMA`: `D[16x16] +=
  A * B^T` through `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` (the gfx11 form on RDNA3). Both operands are rows
  of 16 halves, which is exactly how the kernel stages X (positions) and W (output rows) in shared memory — no
  transpose, and the same tile loop and epilogue shape as the CUDA path. A warp owns 32x32 outputs (2x2 tiles) in the
  big-tile kernel and 16x16 in the small-tile one; a lane supplies 8 halves of its own row of each operand (gfx12
  splits the k dimension across the two half-waves) and holds 8 results, column `lane & 15`, rows `8*(lane>>4) + i`.
  `STRATA_HIP_WMMA=0` falls back to the CUDA-core multiplications: 737 -> 1684 tok/s prefill at pp 1024 on the RX 9070
  XT, and 5 of 6 prompts bit-identical between the two.
- `src/cuda/kernels.cu` keeps one source for both backends: the few places that cannot be shared are behind
  `#if STRATA_HIP` (f16 conversions, the per-lane fragment loads, `mma884`, the KV f16x2 math, the L2 prefetch hint),
  so the CUDA build's preprocessed source is unchanged.
- The code object is cached in `$XDG_CACHE_HOME/rusting_strata/<hash>.amdgcn`, keyed by the source, the `gfx` target
  and the hiprtc version.
- `calibrate --backend hip` measures chunk, MTP drafts and the short-prompt threshold on this machine and writes
  `strata-<model>-hip.json`; later runs read it, as they do for the other backends.
- Large `htod`s go through a 64 MB pinned ring (the DMA reads it while the host fills the next region), so expert
  streaming does not stop the CPU side; small copies are queued asynchronously on the same stream, and `memset` is
  stream-ordered (the null-stream `hipMemset` could land after a copy queued for the same buffer).

## Requirements

- A ROCm runtime with hiprtc: `libamdhip64.so` and `libhiprtc.so` on the loader path, in `/opt/rocm/lib`, or named by
  `STRATA_HIP` and `STRATA_HIPRTC`. The card must be RDNA (wave32, gfx10+); head_dim 256 prefill attention uses a
  16-key tile because a 32-key tile needs 72 KiB of shared memory and a block may use 64 KiB.
- The device's `gfx` target is read from the driver's device properties; `STRATA_HIP_ARCH=gfx1201` overrides it.

## Switches

- `STRATA_HIP_SYNC=1` waits after every launch, so a hardware exception names the kernel that raised it.
- `STRATA_HIP_TRACE=1` names every launch before it goes out (grid, block, shared memory).
- `STRATA_HIP_GRAPH=0` runs the forward every time instead of capturing it (capture is on by default, +5% decode).
- `STRATA_NO_GRAPH=1` turns capture off for both GPU backends.
- `STRATA_HIP_WMMA=0` uses the CUDA-core multiplications in the prefill GEMM instead of the matrices cores.

## Why the prefill GEMM does not use int8 matrix cores

The CUDA backend runs integer weight types as int8 `mma.sync` because NVIDIA's int8 tensor cores are 2-4x its f16 ones
and its layout puts each lane's 4 results over 2 rows and 2 columns, so the per-16-weight scale and min the K-quants
carry cost 2 multiplications a result. On RDNA4 both premises fail:

- `v_wmma_i32_16x16x16_iu8` is only **2x** `v_wmma_f32_16x16x16_f16` (137 vs 57 TMAC/s on gfx1201,
  `tools/hip/wmma_rate.cpp`).
- AMD's WMMA layout gives a lane 8 results in 8 *different rows*, so the scale of each result is different and the
  rescale costs ~8 scalar operations per result — per 16-key group, not once per GEMM.

Measured in the kernel's own inner-loop shape (`tools/hip/wmma_rescale.cpp`): int8 without its epilogue runs in 0.35x
the time of f16, with a vector-friendly rescale 6.6x *slower*, with the layout the source uses 13-16x slower. The
integer path cannot pay for itself here, so `GEMM_TC` stays 0 on HIP and the prefill GEMM dequantizes to f16 in shared
memory and runs on the f16 matrix cores, which is what puts it ahead of the Vulkan backend.

## Next

1. The int8 WMMA form (`__builtin_amdgcn_wmma_i32_16x16x16_iu8`) for the integer-weight prefill GEMM, which today takes
   the f16 path (`GEMM_TC 0`): less shared-memory traffic per output than the sm_80-style int8 `gemmi_*` kernels.
2. A side stream for `beside`, so expert uploads overlap with compute in prefill.

## Credits

The compatibility shim follows the structure and several implementations of Strata's
`include/strata/hip_compat/intrinsics.hpp` (MIT, (c) 2025 Niko1221, https://github.com/Niko1221/Strata), which runs
CUDA-shaped kernels on the same wave32 targets. See `THANKS.md`.
