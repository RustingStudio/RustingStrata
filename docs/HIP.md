# HIP (ROCm) backend

`--backend hip` runs the same `src/cuda/kernels.cu` on an AMD card through HIP, without `hipcc`, ROCm headers or a C++
build step: the kernels are compiled at run time by hiprtc and launched through the HIP runtime, both loaded by name
(`libamdhip64.so`, `libhiprtc.so`), the way `src/cuda` loads NVRTC. `cargo build --release` is the whole build.

**Status: experimental.** Brought up on Qwen3.6-35B-A3B (RX 9070 XT, gfx1201, ROCm 7.2). Greedy output is identical to
the Vulkan backend on the prompts tried. Not yet ported: matrix cores (prefill runs the f16 CUDA-core path, not WMMA),
graph capture (off by default), and the host-side pipeline details of the CUDA backend (ring staging, side stream).

## What runs

Everything the Vulkan backend runs, through the same kernels:

| | HIP | Vulkan (same machine, same model) |
|---|---|---|
| Greedy tokens, 3 prompts | identical | identical |
| Prefill, pp 1024 | 492 tok/s | 1458 tok/s |
| Decode, tg 48, `--mtp 2` | 159 tok/s | 232 tok/s |

Qwen3.6-35B-A3B UD-Q4_K_XL, `--ctx 4096 --vram-margin 256`, RX 9070 XT, Ryzen 7 5700X. Decode is within 1.5x of
Vulkan; prefill is 3x behind because the prefill GEMM has no matrix-core path on HIP yet (see below).

## How it works

- `src/hip/compat.h` is prepended to `kernels.cu`: it supplies the CUDA intrinsics the source uses (`__shfl_*_sync`,
  `__ballot_sync`, `__dp4a`, `__byte_perm`, `__vsub4`, `__vadd4`, `__nanosleep`, `__trap`, the f16 conversions) with
  their AMD equivalents, and sets `__CUDA_ARCH__` to the pre-Volta level so the source takes its portable paths:
  per-lane shared loads instead of `ldmatrix`, plain copies instead of `cp.async`, and the CUDA-core multiplicands
  instead of `mma.sync`.
- The kernels compile as **wave32** (lane = `threadIdx.x & 31`), which is the default wavefront size on RDNA and the
  assumption the CUDA source already makes about a warp. The backend refuses a device that reports a 64-wide wave.
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

## Next

1. gfx12 WMMA (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` and its int8 form) for the prefill GEMM: the 3x
   prefill gap is exactly the missing matrix-core path.
2. Graph capture: move the per-step allocations (q8 cache growth, rope table) out of the captured region.
3. The CUDA host-side pipeline: a pinned ring for `htod` (expert streaming during prefill) and a side stream for
   `beside`.

## Credits

The compatibility shim follows the structure and several implementations of Strata's
`include/strata/hip_compat/intrinsics.hpp` (MIT, (c) 2025 Niko1221, https://github.com/Niko1221/Strata), which runs
CUDA-shaped kernels on the same wave32 targets. See `THANKS.md`.
