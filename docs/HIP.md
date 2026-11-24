# HIP (ROCm) backend

`--backend hip` runs the same `src/cuda/kernels.cu` on an AMD card through HIP, without `hipcc`, ROCm headers or a C++
build step: the kernels are compiled at run time by hiprtc and launched through the HIP runtime, both loaded by name
(`libamdhip64.so`, `libhiprtc.so`), the way `src/cuda` loads NVRTC. `cargo build --release` is the whole build.

**Status: experimental.** Brought up on an RX 9070 XT (gfx1201, ROCm 7.2.3). With `--fixed-experts --mtp 0`
(deterministic placement, no speculative verify) the greedy output is **bit-identical to the Vulkan backend** on the
four local models tried (Qwen3.6-35B-A3B, Ornith-1.5-35B-A3B, Qwen3.5-9B dense, Qwen3.8-27B dense) for the decode
path, and on the batched prefill path 4 of 5 long prose prompts (64-79 tokens) agree token for token on the 35B
(head_dim 128), 2 of 2 on the dense 9B and 1 of 2 on the dense 27B: the fifth 35B prompt flips at a near-tie after
ten tokens, because the matrix cores sum a 16-deep dot product in a different order than Vulkan does (all 5 agree
with the P·V product left on the CUDA cores), and the 27B flips on its first token with either attention path — at
head_dim 256 it takes 16-key tiles where Vulkan goes deeper, so its softmax and P·V sum over more steps. With the
calibrated MTP drafts on (3 on this machine, 1 on Vulkan) the two backends verify different batch shapes, so 1 of
those 7 probes flips too. HIP's prefill attention is pinned to a CPU f32 reference to 2e-4 by
`hip::tests::attn_prefill_matches_cpu`. Not yet ported: the host-side pipeline details of the CUDA backend (pinned
ring staging for `htod`, side stream for `beside`).

## What runs

Everything the Vulkan backend runs, through the same kernels:

| | HIP | Vulkan (same machine, same model) |
|---|---|---|
| Prefill, pp 1024 / 4096, calibrated | 1104 / 2993 tok/s | 1097 / 2325 tok/s |
| Decode, tg 48, calibrated | 209 / 219 tok/s | 199 / 163 tok/s |
| Qwen3.8-27B dense, pp 1024, prefill / decode | 784 / 39.1 tok/s | 508 / 35.3 tok/s |

Qwen3.6-35B-A3B UD-Q4_K_XL (first rows) and Qwen3.8-27B UD-IQ3_S, `--ctx 4096 --vram-margin 256`, one process each,
`calibrate` run for each backend (chunk 4096; HIP picks `--mtp 3`, Vulkan `--mtp 1`). Each cell is pp / the decode that
follows it, so the second value is throughput at that depth. The pp 1024 row runs first in the process, before expert
placement has settled, so it understates both backends (1577-2072 tok/s was measured for HIP there when it was not the
first row); pp 4096 is the honest short-prompt number and is 1.29x Vulkan's. Decode is within a few percent in either
direction — the spread over repeated runs is larger than the difference, because each backend finds its own expert
placement and that moves the MTP draft acceptance. A dense model's prefill is 1.5x ahead and its decode 11% ahead.

`serve` works too, checked on Qwen3.5-9B: `GET /v1/models`, non-streaming and streaming (`stream: true`) chat
completions, and the prompt cache across turns (`prompt_tokens_details.cached_tokens` on a follow-up request).

## Long prompts

Prefill degrades with prompt depth on both backends (attention is the only depth-dependent term), and the prefill
attention now runs both of its products on the card's matrix cores (below). Its per-position cost is now below
Vulkan's, which is where the depth-dependent deficit used to be.

Qwen3.6-35B-A3B UD-Q4_K_XL, `--mtp 0 --reps 1 --vram-margin 256`, KV `f16`, `--fixed-experts` (two runs where a
range is given, one process each):

| pp | HIP | Vulkan | HIP, everything on the CUDA cores (`STRATA_HIP_WMMA=0`) |
|---|---|---|---|
| 8192 | 2647-2666 tok/s | 2148-2152 tok/s | 1303-1308 tok/s |
| 32768 | 2026-2029 tok/s | 1661 tok/s | 776-794 tok/s |
| 131072 | 872 tok/s | 731 tok/s | (not run) |
| decode at 8k / 32k / 128k | 68-73 / 64 / 48 tok/s | 71-72 / 68 / 44 tok/s | 74 / 64-65 / - |

Fitting the per-token cost against depth gives a slope of 4.7 ns per position over 8k -> 32k (6.3 over 8k -> 128k)
for HIP against 5.6 (7.3) for Vulkan and 20-21 for HIP with the attention on the CUDA cores: **3.4x on prefill at
depth** (pp 32k 794 -> 2029 tok/s, pp 8k 1308 -> 2647, same placement in the same build), the score product worth 2x
of it (pp 32k 794 -> 1677) and the P·V product the rest (1677 -> 2029). HIP is now ahead of Vulkan at every depth
tried — 1.23x at 8k, 1.22x at 32k, 1.19x at 128k — and at parity or ahead on decode; the earlier numbers in this file
(pp 32k 720 tok/s, pp 128k 173, gen 80.8, slope 33 ns) were taken on the CUDA cores and are the ones to compare
against. `--chunk 0`-style token-by-token prefill is still not the answer, but a shorter `--ctx` and `--kv q8` both
buy expert slots back.

Two caveats on the numbers: `--fixed-experts` only means "same residents, no re-placement during the run", so the
spread between two runs of the same build was up to 10% before the P·V port (1502 vs 1677 at 32k) — expert placement
and host page-cache state, not the kernel — and has been under 1% since (2029 vs 2026), which is itself a sign of how
much of the prefill used to wait on the CPU-side copies; and a prompt's depth cost is the fit over the two depths,
not either point. A 128k prompt costs more VRAM in
the KV cache than a 16 GB card has left over for experts, so the run above streams most of them from host memory.

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
- **Matrix-core prefill attention** (`attn_prefill_t`): the same WMMA computes both products. Q·Kᵀ is one 16-query
  x 16-key instruction against the 16-half `qs`/`ks` rows; P·V takes A = P from a half staging tile (the `ps` tile the
  Volta path already had, unused on HIP until now, written from the softmax registers as half pairs) and B = Vᵀ read
  down a column of the `vs` tile (`strata_ld_wmma_t`, one half a key apart, since that operand's rows are head
  dimensions).
  Both results land in AMD's layout — 8 a lane, rows `8*(lane>>4) + i`, column `lane & 15` — rather than the m16n8k16
  layout the softmax, the causal mask and the epilogue are written for, so each 16x16 result tile is staged in f32
  and read back as `s[ng][2h + e] == S[warp*16 + g + 8h][ng*8 + 2t + e]` (scores) or `o[d][2r + e]` (P·V). One warp
  staging tile serves both stages (`FA_BQ` x (`FA_BKL` + 8) floats, 6-10 KiB added to the block's dynamic shared
  memory), and it is why head_dim 256 prefill takes 16 keys a step (`FA_BK_FOR`, as on Turing's 64 KB blocks): 32
  keys would need 78 KiB of a block's 64. `STRATA_HIP_WMMA=0` runs the same numerics on the CUDA cores; the
  CPU-reference test covers both.
- The CUDA-core P·V path (`ldsm4t` + `mma16816`) is still there for `STRATA_HIP_WMMA=0` and for CUDA, and it is
  worth knowing how it is shaped: on HIP it reads the two halves of a transposed register as *adjacent* keys, which is
  what the `ldmatrix.trans` fragments mean. An emulation that read them two rows apart (every even key counted twice,
  every odd one never) shipped with the first HIP commit: it moved no token on the short prompts the parity checks
  used, and on the batched prefill it made every long prompt disagree with Vulkan. `hip::tests::
  attn_prefill_matches_cpu` pins it now.
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
- `STRATA_HIP_WMMA=0` uses the CUDA-core multiplications in the prefill GEMM and in both products of the prefill
  attention instead of the matrix cores.

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

That verdict is about GGUF K-quants dequantized on the fly, and there is a known way out of it: **repack the weights**.
Strata runs int8 matrix cores for prompt experts on gfx12 (its `STRATA_PF_FUSED`, on by default for its own IQ3_S,
IQ3_XXS and Q2_0 packs) and measures a 4K prompt 1,612 -> 1,353 ms (+19.1%, 8-8.5% at 16K/32K), because a pack can be
laid out for the fragment (A = 16 weight rows, so a lane's eight results share one token's activation scale) instead
of the per-16-weight K-quant rescale that kills it here. Doing that in this engine would mean a repack pass and our own
pack format — a project, not a kernel — but it is the shape of the evidence that would reopen `GEMM_TC`.

## What is left, measured

Profiled on the same 35B prefill (`rocprofv3 --kernel-trace`, pp 4096): 54% of the kernel time is the quantized
prefill GEMM (already on the matrix cores), 12% the gated delta-rule scan `deltanet_seq`, 10% the prefill attention
(ported), and nothing else above 3%. Of a decode step's wall time, ~90% is not GPU at all — `moe_combine` spends most
of its time waiting on the CPU expert worker, which is an expert-tier question, not a kernel one.

`deltanet_seq` is the one kernel left worth a look, and it has now been measured properly rather than guessed at
(`tools/hip/deltanet.cpp`, which times the scan alone at the 35B's shape):

- its per-token latency (~1750 cycles) is **not** memory: a variant that re-reads token 0's q/k/v every step, so the
  reads stay in L1, runs just as fast (1.00x), even though the scan as written re-reads them from 128 blocks.
- no rearrangement of the scan wins: 32 floats a lane instead of 16 (half the blocks, one wave on 60 CUs) is 0.81x,
  hoisting the loop's `tok * stride` address math into pointer increments is 0.93x, splitting the dot-product
  accumulators is 0.96x, combinations worse.
- the parts with headroom are the two dependent warp-reduction rounds (1.33x if both are removed, wrong results) and
  the per-token `expf(g)` for the decay (1.12x). Both are properties of the *sequential* recurrence: they only go
  away with the chunked WY/UT-transform form (a small triangular inverse plus a few matmuls over a chunk), which
  wants f32 for the state and therefore does not get tensor cores.

Since the whole kernel is 12% of prefill, that rewrite's ceiling is ~4% of prefill. Left alone deliberately — and
Strata, which did write it (chunked scans on the gfx11 matrix cores in `gdn_tc.cuh` and the gfx12 ones in
`gdn_wmma12.inc`, both MIT), measures exactly that: their recurrence is 1.6x faster on an RDNA3 iGPU for a **1.4-2.1%**
gain in whole-prompt time (docs/DETAILS.md), it is off by default because the rounding differs from the token-by-token
FP32 scan, and on CUDA it is 0.73-0.87x, i.e. slower.

## Next

1. A side stream for `beside`, so expert uploads overlap with compute in prefill (the run-to-run spread collapsing
   from 10% to 1% with the faster attention says the copies now wait on the CPU less than they used to).
2. Decode on an MoE model: the expert residency policy, not a kernel. That is where the multi-x is (see the reference
   numbers from Strata in the handoff).

## Inkling 2 decode

Inkling 2 Mini UD-Q4_K_XL, 9070 XT, X470 (PCIe 3.0) with dual-channel DDR4 (36 GB/s read), `bench --pp 512 --tg 64`.
The core layers (6..18, run three times) and the prelude keep all 128 experts in VRAM as slabs. What does not fit runs
as hybrid layers: the host routes, the GPU runs the cached experts and the CPU the rest. These are the coda, layers
20-23 of which are Q8_0 at 8.5 MB an expert.

| | 18 slabs (desktop holds 3.3 GB) | 22 slabs (desktop idle) |
|---|---|---|
| before | 34.5 tok/s | 48.2 tok/s |
| after | 39.5 tok/s | 51.8 tok/s |

A token at 18 slabs: 13.5 ms of kernels, then ~9 ms of CPU experts (32 of the coda's 48, 260 MB at 29 GB/s, near the
DRAM limit). What changed, largest first:

- Copying a missed expert up for one use cost 2.5 ms and the CPU 0.25 ms. The share controller balanced the two sides'
  times, but a layer has about six misses, so it settled on one copy in most layers and a sync in every one. `plan`
  now picks the split from the measured cost of one expert on each side (`stage_count`): here, no copies.
- The VRAM cache froze with what prefill touched first. `Dev::admit` gives a slot to an expert used more often than
  the coldest resident one. The coda routes are skewed (100 experts out of 768 take 60-67% of the uses), and layers
  18-19 went from about 9 to 15 of their 16 uses on the GPU. Q8_0 layers cannot take NVFP4 slots, and no VRAM is left.
- The core passes run as one graph in the hybrid step (`core_graph`): 0.8 ms.

### The expert pool

Since then, every layer keeps only its most used experts in VRAM (`inkling2/forward/pool.rs`), and a decode token
never comes back to the host:

- `ink_route` routes on the device, and `ink_slots` looks each expert up in the layer's slot table. An expert that
  is not in VRAM goes to the CPU worker (`worker.rs`) through the mailbox of `cpu_worker`, with the routing and
  the activation. `ink_ffn_sum_mb` waits for the worker's sum only in a layer run that missed. The token is one
  graph.
- After each token the routing counts (decayed per token) decide the placement. Up to 4 missed experts go into free
  slots and 2 into the slots of less used ones. The long counts are saved as
  `~/.cache/rusting_strata/inkling2-profile-*.f32` when the model goes, and they place the next load.
- A prompt chunk runs from the same slots. The chunk's other experts are copied into 32 staging slots a pool, in
  rounds, before the GEMMs run.

| 12 GB weight budget, desktop holds 3 GB | pp 512 | pp 4096 | decode |
|---|---|---|---|
| slabs and hybrid layers (above) | 443 tok/s | 438 tok/s | 39.8 / 38.9 tok/s |
| pool, no profile yet | 329 | 378 | 49.7 / 55.7 |
| pool, profile of one earlier run | 436 | 426 | 60.5 / 58.9 |

Decode after the prompt of each column. At this budget the pool holds 68% of each layout's experts and keeps 98% of
a token's expert uses in VRAM; the CPU worker takes 1.8 ms a token. Greedy text is no longer bit-identical to
Vulkan: the placement decides which experts round on the CPU. `--fixed-experts` keeps the slabs and is.
`STRATA_INK_SLABS=1` keeps the slabs and pools only the layers they leave out.

### Prompt chunks of 2048 rows

A prompt chunk copies each layer run's missing experts into the staging slots, and a chunk of 256 rows already
touches most of a layer's 128 experts, so the copies repeated every 256 tokens. Prompt chunks now have 2048 rows
(`STRATA_DS_PF_TOKENS`); the scratch costs about 0.55 GB of VRAM per 1024 rows, and the chunk halves when it does
not fit. 9.5 GB weight budget, desktop at 2.2 GB, `bench --pp 512,4096 --tg 32`:

| rows | pp 512 | pp 4096 | decode | peak VRAM, card |
|---|---|---|---|---|
| 256 | 265 tok/s | 278 tok/s | 49.5 tok/s | 13.47 GB |
| 2048 | 387 | 767 | 48.9 | 14.61 GB |
| 4096 | 393 | 891 | 48.8 | 15.48 GB |

At 2048 rows a core layer run still misses 40-45 of the ~125 experts it uses (6-7% of the uses); most of those
serve 9 or more rows.

Left: `ink_mix` runs 120 times a token at 12 µs in one workgroup (about 1 ms, but a faster reduction order breaks the
bit parity with Vulkan), and a first run without a profile.

## Credits

The compatibility shim follows the structure and several implementations of Strata's
`include/strata/hip_compat/intrinsics.hpp` (MIT, (c) 2025 Niko1221, https://github.com/Niko1221/Strata), which runs
CUDA-shaped kernels on the same wave32 targets. See `THANKS.md`.

The two matrix-core ports in this backend were written independently and turned out to match work already in Strata, in
a different design:

- **Prompt attention on the matrix cores**: their `src/kernels/cuda/qsa_prompt_attn.cu` has the same instruction for
  gfx11 and gfx12 with the fragment layout noted here (`A` lane l holds `A[l % 16][(l / 16) * 8 + i]`, gfx11
  `D[2i + l / 16][l % 16]`), default on gfx12 Linux but tied to an int8 KV cache and to keeping q and p as FP16
  hi+lo parts; it is 7.2-7.5x their portable kernel and +31% prompt throughput at 8.8K tokens. This backend's version
  works on an f16 KV cache and reuses the CUDA-era m16n8k16 softmax by staging the scores, at ~2x the prefill
  attention and +22-35% prefill at depth.
- **A chunked gated delta-rule scan** (`gdn_chunk.cu`, `gdn_tc.cuh`, `gdn_wmma12.inc`): the rewrite this backend
  measured and declined, with their own numbers agreeing with the estimate (see "What is left, measured").
