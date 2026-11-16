# HIP (ROCm) backend

`--backend hip` runs the same `src/cuda/kernels.cu` on an AMD card through HIP, without `hipcc`, ROCm headers or a C++
build step: the kernels are compiled at run time by hiprtc and launched through the HIP runtime, both loaded by name
(`libamdhip64.so`, `libhiprtc.so`), the way `src/cuda` loads NVRTC. `cargo build --release` is the whole build.

**Status: experimental.** Brought up on Qwen3.6-35B-A3B (RX 9070 XT, gfx1201, ROCm 7.2.3). Greedy output matches the
other backends: identical to the Vulkan backend on 3 of 4 prompts, and to its own CUDA-core prefill on 5 of 6, with the
difference at a near-tie where the f16 tensor-core summation order picks another token (the same class of difference the
CUDA and Vulkan backends have). Not yet ported: graph capture (off by default) and the host-side pipeline details of the
CUDA backend (pinned ring staging, side stream).

## What runs

Everything the Vulkan backend runs, through the same kernels:

| | HIP | Vulkan (same machine, same model) |
|---|---|---|
| Prefill, pp 1024, `--mtp 2` | 1569 tok/s | 1428 tok/s |
| Prefill, pp 4096, `--mtp 2` | 2106 tok/s | (not run) |
| Decode, tg 48, `--mtp 2` | 173 tok/s | 231 tok/s |

Qwen3.6-35B-A3B UD-Q4_K_XL, `--ctx 4096 --vram-margin 256`, RX 9070 XT, Ryzen 7 5700X, one process each. Prefill runs on
the card's matrix cores (below) and is at parity with Vulkan; decode is 1.3x behind, which is the host side (no graph
capture, no side stream) rather than the kernels.

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

## Requirements

- A ROCm runtime with hiprtc: `libamdhip64.so` and `libhiprtc.so` on the loader path, in `/opt/rocm/lib`, or named by
  `STRATA_HIP` and `STRATA_HIPRTC`. The card must be RDNA (wave32, gfx10+); head_dim 256 prefill attention uses a
  16-key tile because a 32-key tile needs 72 KiB of shared memory and a block may use 64 KiB.
- The device's `gfx` target is read from the driver's device properties; `STRATA_HIP_ARCH=gfx1201` overrides it.

## Switches

- `STRATA_HIP_SYNC=1` waits after every launch, so a hardware exception names the kernel that raised it.
- `STRATA_HIP_TRACE=1` names every launch before it goes out (grid, block, shared memory).
- `STRATA_HIP_GRAPH=1` turns on graph capture once the forward allocates nothing inside a step (HIP refuses an
  allocation on a capturing stream; CUDA's stream-ordered allocator allows it).
- `STRATA_NO_GRAPH=1` turns capture off for both GPU backends.
- `STRATA_HIP_WMMA=0` uses the CUDA-core multiplications in the prefill GEMM instead of the matrices cores.

## Next

1. Graph capture: move the per-step allocations (q8 cache growth, rope table) out of the captured region; that and the
   side stream are where the remaining decode gap is.
2. The int8 WMMA form (`__builtin_amdgcn_wmma_i32_16x16x16_iu8`) for the integer-weight prefill GEMM, which today takes
   the f16 path (`GEMM_TC 0`): less shared-memory traffic per output than the sm_80-style int8 `gemmi_*` kernels.
3. A pinned ring for `htod` so expert streaming during prefill does not sync the stream per tensor.

## Credits

The compatibility shim follows the structure and several implementations of Strata's
`include/strata/hip_compat/intrinsics.hpp` (MIT, (c) 2025 Niko1221, https://github.com/Niko1221/Strata), which runs
CUDA-shaped kernels on the same wave32 targets. See `THANKS.md`.
