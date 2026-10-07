# Improvement loop log

Each iteration: one verified item. Newest entry last. Loop instructions: `/home/vasylt/Rusting/loop-prompts/strata.md`.

## State

- Milestone: M2 done. M3 expert cache partly in (commit 90db54a). M4 first half (chunked GPU prefill) committed.
- Speed at 4K (3 clean runs): prefill **1366 tok/s** (1363-1369; gate 1017, passed), decode 49.6 tok/s (gate 67).
  See `docs/m4-results.md`.
- Parity (20 prompts x 32, GPU, 3 runs): prefill 628/630/629 = mean 98.28% +- 0.16; token-by-token prompt
  (`--chunk 0`) 631/627/631 = mean 98.39% +- 0.36. Gate 99%. Decode is not deterministic run to run (adaptive
  cache moves experts between GPU int8 and CPU kernels), so single parity runs move by +-2 misses.

## Iterations

### 2026-10-06 #1 — GPU busy, triage only
- GPU busy: `forever_bear` (1051 MiB) on the GPU. No GPU tests or benchmarks run.
- Found the prefill work above uncommitted. Release build OK (`CARGO_TARGET_DIR=target/loop`); CPU suites
  (lib, gguf, ops, quant, tokenizer, cpu_moe) pass. Kept the work: it cannot be verified or rejected without the GPU.
- No numbers moved. No commit of code.

### 2026-10-06 #2 — GPU busy again
- `forever_bear` (790 MiB) on the GPU, started by `rusting test . tests --release` from the RustingGames loop
  (short-lived, relaunched often). No GPU work run.
- Parity status from cached results: `target/parity-result-gpu.json` 631/640 = 98.59%,
  `target/parity-result-gpu-q8.json` 630/640 = 98.44% (dp4a Q8 GEMV did not close the gap). Gate needs <= 6 misses.
  Two misses are at position 0 (right after the prompt): llama.cpp builds the reference with batched prompt
  processing, so the batched prefill path may change these. Check after prefill lands.
- **Question for owner:** the RustingGames test loop keeps a game process on the GPU most of the time. May the
  Strata loop run GPU tests and benchmarks while only that process (< 1.1 GB) holds the GPU? Benchmarks would be
  noisy; correctness tests would not.

### 2026-10-07 #3 — chunked GPU prefill verified and committed
- Old test compared prefill to GPU decode and failed (max |dlogit| 2.0-2.3, limit 5% of range). Against the CPU f32
  reference, prefill is off by 0.41, decode by 1.9-2.3: decode's int8 activations are the larger error. Test
  replaced by `prefill_matches_reference` (both paths vs CPU; prefill must match CPU argmax and not exceed decode
  error). Passes at chunk 48 and 1024. `tests/cuda.rs` and `gpu_forward_matches_cpu_reference` pass.
- Removed `DBG_LAYERS` hook: limiting layers deadlocks decode (CPU worker waits for every MoE layer).
- Parity: see State. Prefill does not move agreement beyond run-to-run noise.
- `tools/parity.py` now prints `strata score` stderr on failure (one `--chunk 0` run died once with stderr
  swallowed; not reproduced on retry; forever_bear held 795 MiB at the time).
- Prefill speed not benchmarked: forever_bear on the GPU. Test timing only: 73 tokens in 1.06 s (cold).
- The new prefill kernels (attn_prefill, deltanet_seq, conv_seq/conv_state, moe_combine_seq, gemm act 1/2) are covered
  only end to end by `prefill_matches_reference`, not by per-kernel tests. Debt.

### 2026-10-07 #4 — GEMM accumulators and DeltaNet state kept in registers (76a57e2)
- Profile (nsys, 4K prefill): kernels 8.6 s of 8.9 s wall; gemm_iq2_xs 2.87 s, gemm_q5_k 1.92, gemm_iq3_xxs 1.30,
  deltanet_seq 1.26, gemm_q6_k 0.65, attn_prefill_256 0.29. H2D staging only 0.54 s GPU time: not the bottleneck.
- ptxas showed 128 B stack in the GEMMs (`acc[2][4][4]` spilled: loops not unrolled) and 64 B in deltanet_seq (loops
  over runtime `per`). Fix in `src/cuda/kernels.cu`: `#pragma unroll` on the GEMM tile loops; deltanet loops run to
  compile-time MAXK with `if (t < per)`. Now 0 stack, ~128 / 46 registers.
- GEMM throughput (`gemm_throughput` test): 3.8-5.6 to 9.8-14.5 TFLOP/s (qkv Q5_K 36.1 to 14.1 ms).
- 4K prefill: before ~444 tok/s (428/433/470, partly contaminated), after **961 / 1020 / 1022 = 1001 tok/s**.
  Decode 49.1 / 49.9 / 49.9 = 49.6 tok/s (before 43-47).
- Parity: 628 / 628 / 629 = 98.18% (before 628/630/629 = 98.28%): within run-to-run noise.
- Benchmark method: `bench.sh`-style wait for an idle GPU and a watcher that marks a run dirty if any other process
  touches the GPU during it; dirty runs are discarded. All 3 runs were clean.

### 2026-10-07 #5 — default prefill chunk 4096 (8d0c3c4)
- Profile (nsys, 4K prefill after #4): kernels 4.2 s; gemm_iq2_xs 1.22 s, gemm_q5_k 0.79, deltanet_seq 0.77,
  gemm_iq3_xxs 0.53, attn_prefill_256 0.29, gemm_q6_k 0.29. H2D in the prefill window 80 ms GPU time.
- Expert gate/up GEMMs ran at ~4.4 TFLOP/s against ~13 in the dense `gemm_throughput` test: at chunk 1024 each expert
  gets ~32 positions, so 128-position tiles are mostly empty. Larger chunk fills them.
- Clean 4K runs (3 each, idle GPU): chunk 1024 919/1012/1013, 2048 1222/1221/1218, **4096 1368/1363/1369** tok/s.
  Decode 49-50 tok/s in all. Default `--chunk` now 4096; `GpuModel::load` caps it at `ctx`. Scratch 118 to 470 MB.
  Loads and runs at `--ctx 32768` (resident experts 8160/10240, prefill 1331 tok/s single run).
- Parity, interleaved runs: chunk 4096 627/624/627/630/627/632 (mean 98.09%), chunk 1024 630/629/629/633/628
  (mean 98.41%; earlier runs 628-630). Difference ~1.4 misses, inside the 624-633 run-to-run range; the only prompt
  whose prefill changes (1367 tokens, now one chunk) scored the same. Several runs had forever_bear on the GPU, which
  changes expert placement. Accepted as noise; recheck when decode accuracy work lands.
- 4 parity runs died with `CUDA_ERROR_OUT_OF_MEMORY` while forever_bear and a RustingEngine test held VRAM (both
  chunk sizes). Load plans from free VRAM at start; a process that arrives later can starve us. Not fixed.

## Tried and rejected

(none yet)

## Next candidates

1. Decode accuracy (parity 98.1-98.4% vs gate 99%): int8 activations in decode give 1.9-2.3 logit error vs 0.4 for the
   f16 prefill path; results also depend on which experts are resident. Try Q8_1-with-sum like llama.cpp or f16
   activations for the GEMVs; measure with `gpu_forward_matches_cpu_reference` dlogit first.
2. Decode 49.6 vs 67 tok/s: profile one decode step with nsys (not yet done since M3).
3. deltanet_seq 0.77 s of 4.2 s prefill (sequential scan); chunked WY form. Plus per-kernel tests for prefill kernels.
