# Improvement loop log

Each iteration: one verified item. Newest entry last. Loop instructions: `/home/vasylt/Rusting/loop-prompts/strata.md`.

## State

- Milestone: M2 done. M3 expert cache partly in (commit 90db54a). M4 first half (chunked GPU prefill) committed.
- Speed at 4K (3 clean runs): prefill **1366 tok/s** (1363-1369; gate 1017, passed), decode **78.6 tok/s** (gate 67, passed).
  Short-prompt decode (8-token prompt, -n 128, like llama-bench tg128): 86 tok/s. Next: 32K context and MTP.
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
  only end to end by `prefill_matches_reference`, not by per-kernel tests. Debt (paid in #6 except moe_combine_seq
  and gemm act 1/2).

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

### 2026-10-07 #6 — per-kernel tests for prefill kernels (7352742)
- GPU busy at start (forever_bear 803 MiB): no benchmarks or parity; only sub-second kernel tests.
- `tests/cuda.rs`: `attn_prefill_matches_decode` (hd 64/128/256, 40 cached keys, 100-token chunk; rel err 5.1e-5,
  7.1e-5, 6.4e-5, limit 5e-4), `deltanet_chunk_matches_steps` (37 tokens vs 37 single steps: bit-exact output and
  state), `conv_seq_matches_steps` (T = 2 and 10 vs conv_silu steps: exact). Full `cuda` suite passes.
- No speed or parity change.
- Correction to candidate 1 of #5: M2 found every GPU miss equals the CPU f32 reference miss (same position, same
  top-2 gap), so int8 decode error is not what separates us from llama.cpp. The 99% gap is llama.cpp's own numerics.

### 2026-10-07 #7 — parity experiment: emulate llama.cpp activation quantization (rejected, no commit)
- GPU busy (forever_bear 805 MiB). CPU-only experiment.
- The reference came from `llama-server -ngl 99 --n-cpu-moe 20 -fa on` (plan M0/M1): llama.cpp quantizes matmul
  activations to int8 (Q8_1, 32 per block, on CUDA; Q8_K, 256 per block, for the 20 CPU expert layers).
- Temporary `STRATA_ACTQ=<block>` in `Tensor::matvec` (CPU reference) rounded x to int8 per block for every
  quantized weight. 14 short prompts (index 0-13), 32 positions each, top-1 agreement with the reference:
  f32 **445**/448 (misses: p4@0, p9@12, p13@17), block 32 443/448, block 256 439/448. Reverted.
- GPU on the same 14 prompts: 441-446 (6 runs). The 6 long prompts: 186-187/192 every run, the same misses.
- Reading: on short prompts GPU decode is within noise of f32; the f32 reference itself misses 3 of 448. Plain
  activation rounding moves the model further from llama.cpp, not closer. Matching llama.cpp to 99% would need its
  exact kernels (accumulation order, Q8 sums, per-layer CPU/GPU split), not just its rounding.
- **Question for owner:** the f32 reference agrees 98.3-98.6% with this llama.cpp reference, and the GPU path the
  same. Is the 99% gate meant against this llama.cpp run, or should parity use a reference with matching numerics
  (e.g. a llama.cpp CPU-only run, or agreement with our own f32 reference)?

### 2026-10-07 #8 — split decode attention across position blocks (1866f5e)
- GPU idle (gnome-clocks 41 MiB only).
- Profile first. Short-prompt decode (5 clean runs, "Write a long story about a dragon.", -n 128): **85.8 tok/s**
  (84.9-86.6), already above the 67 gate under llama tg128 conditions. After the 4K prompt decode is 47-50. nsys at
  4K (`--cuda-graph-trace=node`): `attn_decode` 0.87 ms x 10 layers = 8.7 ms per token, 20% of GPU time; grid was
  one block per query head (32 blocks on a 28-SM GPU, warps walking positions).
- Change: `attn_decode` grid (heads, 16); block s covers ceil(n_pos/16) positions and writes unnormalized
  [acc | max | sum] to a new `attp` scratch; new `attn_merge` kernel combines the splits and applies the gate. Split
  size comes from step[0] inside the kernel, so the decode CUDA graph still captures once. Empty splits write
  max = -inf, sum = 0.
- Test: `attn_prefill_matches_decode` now starts at position 5, so early decode steps have empty splits; rel err
  6.4-7.4e-5. Full cuda suite passes.
- Files: `src/cuda/kernels.cu`, `src/cuda/mod.rs`, `src/gpu_model.rs`, `tests/cuda.rs`.
- 4K (3 clean): prefill 1026/1316/1360 (first run cold page cache), decode **60.9** (59.8/60.4/62.4), was 49.6.
  Parity 630/628/629 (98.28%), same band as before. Short-prompt decode 83.3 (81.3-84.6, 5 clean runs after one
  noisy set of 5: 50-83); before 85.8. The ~3% short-context drop is at the edge of run-to-run noise (expert cache
  placement); the merge adds 10 small launches per token.

### 2026-10-07 #9 — vector loads in decode attention (059fd1e)
- GPU idle (gnome-clocks 41 MiB only).
- Profile (nsys, 4K prompt + 128 decode, `--cuda-graph-trace=node`, per token): attn_decode 4.1 ms (408 us x 10)
  of 16.4 ms, then gemv_q5_k 3.1 ms, gemv_q6_k 1.6, moe_gemv_iq2_xs 1.5, moe_combine 1.1, gemv_q4_k (output head)
  1.1, moe_gemv_iq3_xxs 1.0, gemv_f32 0.7, rms_norm 0.6, router_topk 0.6.
- Change: in `attn_decode` each lane owns hd/32 consecutive dims and loads K and V rows with one vector load
  (`ldh_vec<PER>`, uint4 for hd 256) instead of hd/32 scalar f16 loads; kernel templated on PER, dispatch on hd
  (64/128/256/512). attn_decode 408 -> 88 us per call.
- Test: `attn_prefill_matches_decode` (hd 64/128/256) passes; full cuda suite passes.
- Files: `src/cuda/kernels.cu`, `src/cuda/mod.rs`.
- 4K (3 clean): decode **78.6** (76.9/79.0/80.0), was 60.9; prefill 1002/1350/1138 (kernel unchanged; machine
  noise, load average 2-7, swap 14 GB). Parity 631/631/629 (98.49%). Short-prompt decode 86.1 (83.7-87.3, 5 clean).
- Speed gate (decode > 67, prefill >= 1017 at 4K) now passed on both. Parity gate (99%) still open; see #7 question.

## Tried and rejected

- #7: int8 activation rounding in the CPU reference to mimic llama.cpp (block 32: 443/448, block 256: 439/448,
  f32: 445/448 on prompts 0-13). Worse. Redo only with llama.cpp's exact dot-product kernels.
- #8: nothing rejected. Not tried: choosing the split count from n_pos (needs a fixed grid for graph capture;
  could use a max grid and let extra blocks exit, which is what empty splits already do).

## Next candidates

1. 32K context: measure prefill and decode at 32K vs llama.cpp (`docs/baseline.md`); attention and deltanet costs
   grow there. Needs idle GPU.
2. deltanet_seq 0.66 s of ~3 s 4K prefill (sequential scan, 21.9 ms x 30 layers); chunked WY form.
3. Decode GEMVs: gemv_q5_k 3.1 ms/token, gemv_q6_k 1.6, moe_combine 1.1 (28 us x 40, looks high for its work),
   output head gemv_q4_k 1.1. Check each against memory bandwidth.
4. Parity: score the 6 long prompts with the CPU f32 reference (~15 min CPU) to confirm their misses are shared.
