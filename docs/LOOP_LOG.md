# Improvement loop log

Each iteration: one verified item. Newest entry last. Loop instructions: `/home/vasylt/Rusting/loop-prompts/strata.md`.

## State

- Milestone: M2-M5 done (MTP speculative decoding, `generate --mtp`, rollback snapshots). M6: `chat`, `serve`
  (sampling, `--api-key`, history reuse 99fbd11), `bench` and `calibrate` in (a336559). v1 gate run done
  (#34, `docs/m6-results.md`). M6 done (calibration measured in #35, defaults kept). M3 expert cache adaptive.
- MTP k = 1 decode ~87-92 tok/s (81.7% accepted, #25); k = 2-3 slower (rejected for now).
- 4K prefill with int8 MMQ: **1745 tok/s** (1736-1753, #32, 3754e65).
- Speed at 4K (3 clean runs): prefill 1581 tok/s (1538-1610; gate 1017, passed), prefill 1608 at 2dcae33, decode **81.0 tok/s** at a02b4ec (76.8-81.8; gate 67, passed).
  Short-prompt decode (8-token prompt, -n 128, like llama-bench tg128): 88.1 tok/s at a02b4ec (88.06-88.22, 100% expert hits). 32K (31768-token prompt): prefill ~1290 tok/s, decode best 62.5, noisy (llama.cpp -d 32768: pp512 814, tg128 56.4).
  See `docs/m4-results.md`.
- Parity: MMQ prefill mean 98.39% +- 0.32 over 6 runs (#32; f16 98.28% +- 0.20 interleaved). Earlier (20 prompts x 32, GPU,
  3 runs): prefill 628/630/629 = mean 98.28% +- 0.16; token-by-token prompt
  (`--chunk 0`) 631/627/631 = mean 98.39% +- 0.36. Gate 99%. Decode is not deterministic run to run (adaptive
  cache moves experts between GPU int8 and CPU kernels), so single parity runs move by +-2 misses.
- Parity ceiling: llama.cpp batch vs its own token-by-token reference is 98.28% (#33); Strata vs reference-or-batch 99.17%.
  The reference is llama.cpp's CPU build; against its CUDA build Strata scores 97.34% (#34).
- MTP k = 1 with GPU argmax: 92.4 tok/s after a 512-token prompt (caa5f4b).
- At d24402d (#50): decode 99.1 after 512, 91.6 after 4096; MTP k = 1 105.1 / 94.0; 4K prefill ~2070 tok/s.
  Parity 98.59% (3 runs).
- At 2a51d2e (#52): decode after 4096 93.9, MTP k = 1 94.5; after 16384 87.0. Parity 98.36% over 6 runs, with an
  interleaved control showing ec8200a at 98.28% under the same conditions. Parity gate per change: interleaved
  3-run control against the previous binary, since single 3-run means move by +-0.3%.

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

### 2026-10-07 #10 — 32K measured; ldmatrix in prefill attention (f4cf273, 09edad5)
- GPU idle at start; machine loaded during benches (another loop's rustc at 430% CPU, load average 5-14, swap 15 GB,
  short GPU users that marked most runs dirty; 3 parity runs hit CUDA_ERROR_OUT_OF_MEMORY and were redone).
- New: `strata generate --prompt -` reads stdin (32K prompt is over the 128 KiB argument limit). 32K prompt =
  p4096 text x 8 = 31768 tokens, `--ctx 32768 -n 128`.
- Before: 32K prefill 818 tok/s (average over the whole prompt; llama.cpp's 814 is pp512 at depth 32K, a marginal
  rate, so ours is not directly comparable and is the harder number), decode 54.0 (llama 56.4). nsys: attn_prefill_256
  16.3 s of ~39 s prefill; deltanet_seq 5.0 s, gemm_q5_k 4.9, gemm_iq2_xs 4.5; decode attn_decode 0.59 ms x 10.
- Cause: V^T tile written with scalar u16 stores at a 640-byte thread stride (all one bank), mma fragments from 32-bit
  shared loads. Change: V row-major like K, fragments via `ldmatrix` x4 (`.trans` for V); helpers `ldsm4`/`ldsm4t`.
- Test: `attn_prefill_matches_decode` hd 64/128/256 unchanged errors (6.4-7.4e-5).
- After: attn_prefill_256 5.4 s (~16 TFLOPS). Clean runs: 32K prefill 1007/1204/1190, decode 26.3/55.6/54.3 (noisy;
  CPU expert work suffers under load); 4K prefill 835/1434/1220, decode 64-78. Parity 628/629/627 (98.13%).
- Decode numbers this iteration are not trustworthy (load). Re-measure 4K and short decode when the machine is quiet;
  the change does not touch the decode path.

### 2026-10-07 #11 — DeltaNet scan: 16 state floats per lane (b57b791)
- GPU idle; CPU loaded (other loop's rustc, load average up to 29), so kernel times come from nsys.
- Profile (#10): deltanet_seq 0.66 s of 4K prefill (largest kernel), 5.0 s at 32K. One state row per warp, 4 floats
  per lane: each token paid two 5-shuffle reductions per row and reloaded q/k per row; 4096 warps (3 waves).
- Change: a row lives in dk/16 lanes with 16 floats each (float4 loads), so a warp holds 4 rows at dk 128 and the
  reductions take 3 shuffles; 1024 warps (one wave). State layout in memory unchanged. dk limited to 32/64/128/256.
- Test: `deltanet_chunk_matches_steps` also checks a CPU scan: out 1.96e-7, state 2.02e-7; chunk vs steps exact.
- Files: `src/cuda/kernels.cu`, `src/cuda/mod.rs`, `tests/cuda.rs`.
- After: deltanet_seq 4K 673 -> 320 ms, 32K 5.0 -> 2.5 s; decode call median 10.7 -> 10.2 us. 4K prefill (3 clean)
  **1581** (1610/1595/1538), was 1434 best. 32K prefill 1288 under nsys. Parity 630/629/631 (98.44%).
  Decode 76/76/48 (4K) and 61-74 short: CPU load (the short set had a 16 tok/s run); not attributable to this change,
  whose decode kernel got faster. Needs the quiet re-measure (candidate 1).

### 2026-10-07 #12 — decode attention shares K/V across a GQA group (2dcae33)
- GPU idle apart from short RustingEngine test runs; CPU loaded (load average 4-12, swap 14 GB).
- Profile (b57b791, 32K): attn_decode 0.59 ms x 10 layers per token; each of the 16 query heads re-read the K/V of
  its KV head (8 heads per KV head).
- Change: one block per (KV head, split) computes all G query heads of that KV head; next K/V row prefetched;
  per-head merge in the block. ATT_SPLITS 16 -> 64 to keep the grid full; attn_merge keeps split weights in shared
  memory. Wrapper picks G from 8/4/2/1 with G*hd <= 2048.
- Test: `attn_prefill_matches_decode` now loops (nh,nkv,hd) over (4,2,64), (4,2,128), (4,2,256), (16,2,256),
  (8,8,128); rel err 6.1e-5 to 7.4e-5. Full cuda suite passes.
- Files: `src/cuda/kernels.cu`, `src/cuda/mod.rs`, `tests/cuda.rs`.
- After (nsys): attn_decode 32K 590 -> 314 us, 4K 88 -> 93 us (+3 us merge). 4K (3 clean): prefill 1561/1631/1633,
  decode **79.8/80.7/80.9** (mean 80.5, was 78.6). 32K: decode 62.5 in the one undisturbed run (llama.cpp 56.4);
  the other two (20, 12) ran under host load. Short decode 14-70, load noise; nsys shows attention is 0.6 ms of
  19 ms GPU time per token there, and moe_combine 5.8 ms/token (it waits on CPU experts, 35% of gen time).
- Parity 627/629/630 (98.23%), in band.

### 2026-10-07 #13 — prefill GEMM tile experiments (rejected, no commit)
- GPU idle apart from short bursts from other processes; host load 11-18, swap 22 GB. Wall clock unusable, so all
  numbers are nsys GEMM kernel totals for one 4K prefill (3971 tokens, -n 4), A/B runs interleaved by
  `ab.sh` in the scratchpad, each run waiting for an idle GPU.
- Baseline (7cdc0fc): GEMM total 1759/1839/1953/1779/1947 ms (q5_k 635-824, iq2_xs 561-582, iq3_xxs ~259,
  q6_k ~260). GEMMs are ~75% of 4K prefill GPU time.
- Tried, all with GEMM_BM/BK edits only or a small K-loop change:
  - GEMM_BM 256 (512 threads, 46 KB smem): 2549 ms. iq2_xs 978, iq3_xxs 460 (expert tiles half empty).
  - GEMM_BM 64: 2082 ms (q6_k 461).
  - Register prefetch of the next K step (dequantized weights as packed halves + raw float4 activations held across
    the MMA loop): 2884/2753/3312 ms. Likely register pressure / spills; 1.6x slower.
  - GEMM_BK 32: 2066/1789 ms. `__launch_bounds__(256, 2)`: 2038/1929 ms. Inside the noise, no gain.
- Conclusion: the 128x64x64 tile is at its local optimum. Further prefill GEMM gains need a different kernel
  (int8 MMQ with weights kept quantized in shared memory, as llama.cpp does), not tile tuning.

### 2026-10-07 #14 — short-context regression from #12 fixed (e087210)
- GPU idle; host load 1-13 (other loop's rustc, swap 22-25 GB). Quiet re-measure first (candidate 1): 4K decode
  76.7/77.6 (+ one 38.8), short 58-77, 32K 11-60. Short decode was below the 86 of #9, so A/B under nsys.
- Profile: attn_decode per call at a short prompt (~136 positions) was 7.9 us at 228b148 and 50 us at 2dcae33.
  64 splits x 8 heads per block spread 2-3 positions per block, and every block paid the 8-head merge.
- Change: a split covers at least 16 positions; empty splits write their empty partial and exit. Minimums 16/32/64
  gave 22/23/25 us. 4K and longer are unaffected (64 splits already hold >= 64 positions there).
- Files: `src/cuda/kernels.cu`. Test `attn_prefill_matches_decode` (up to 105 positions: full, partial and empty
  splits) passes.
- After: attn_decode short 50 -> 22 us, 4K 96 -> 91 us, 32K 315 us. Short decode (5 clean) 6.4 (load spike), 78.6,
  81.5, 82.7, 82.9. 4K clean 75.7/77.4/80.5 decode, 1144-1638 prefill (loaded). Parity 625/627/630 (one OOM rerun).
- Still 14 us slower than 228b148 at short context (~0.14 ms/token over 10 layers); the 8-head merge is serial.
  Worth it for 32K (590 -> 315 us).

### 2026-10-07 #15 — one-head decode attention below 4096 positions (b9d9547)
- GPU idle; host load 7-13. Kernel times from nsys (`--cuda-graph-trace=node`).
- Profile: after #14 short-prompt attention was still 22 + 5 us per layer and token (228b148: 7.9 + 3.2).
  `ptxas -v`: attn_decode used 247 registers (the G = 8 instantiation), so one 256-thread block per SM; a short
  context paid for waves of near-empty blocks. Forcing G = 1 inside the same kernel did not help (31 us), since the
  register count is per kernel.
- Change: separate entry points. `attn_decode` (one head per block, 16 splits, 104 registers) works below
  ATT_GQA_MIN = 4096 positions; `attn_decode_gqa` (G heads per block, 64 splits) from there on. Both are launched
  in the graph; the idle one exits (~2 us). attn_merge picks the split count from the step.
  `Gpu::attn_decode_gq` takes the threshold so the test can force either kernel.
- Threshold: at 4K the two are even (gqa 92.8 + 7.1 + 1.9 idle vs one-head 98 + 3.8 + 2 us); at 32K one-head with 64
  splits was 689 us vs gqa 319.
- Test: `attn_prefill_matches_decode` runs every (nh, nkv, hd) case with both kernels; rel err 6.1e-5 to 7.4e-5.
- Files: `src/cuda/kernels.cu`, `src/cuda/mod.rs`, `tests/cuda.rs`.
- After: short 27 -> 11.4 us, 4K ~98 -> ~102 us, 32K 319 us. 4K (3 clean) decode 77.5/80.4/80.8, prefill
  1251/1628/1652. Short decode 56-74 (host load 7.5, one prompt at 27 tok/s: not usable). Parity 627/632/631.

### 2026-10-07 #16 — short decode re-measured; CPU expert threads (no code change)
- GPU idle; host load 4-13 (other loop's rustc/rust-lld). Ryzen 5 7600X, 6 cores / 12 threads.
- Question: does the CPU expert worker (rayon over all 12 threads, `batch_rows` par_chunks of 16 rows) lose time when
  the host is loaded? A/B with `RAYON_NUM_THREADS` 12 (default) / 8 / 6 / 4 / 3 / 2, interleaved, short prompt -n 128.
- Finding: the hit rate decides it, and it varies run to run: 97.2% -> 100% over consecutive runs, even with
  `--vram-margin 2000` (181 -> 0 misses). Placement follows free VRAM at start, which other processes change.
  The thread-count A/B is confounded by that and showed no consistent winner (per-run CPU share 0-9%).
- Short decode at b9d9547 with <= 11 misses: 86.40 / 86.69 / 86.93 (12 / 6 / 3 threads), 85.5-86.7 in the first
  set at 99.9% hits. Back to the 86.1 of 059fd1e; the #12 regression is fully gone.
- The 5.8 ms/token moe_combine of #12 was a 91% hit-rate run (3864 misses), i.e. a VRAM-short start, not a
  worker problem.
- No code change. A fair thread-count test needs a fixed miss set (e.g. a flag that caps GPU expert slots).

### 2026-10-07 #17 — decode GEMV bandwidth; MoE gate and up in one launch (a02b4ec)
- GPU idle at the runs; host load 7-10.
- Measured (nsys per launch, short prompt, -n 64): dense q5_k 226-257 GB/s, output head q4_k 252, ssm_out q6_k 185,
  MoE iq2_xs / iq3_xxs experts ~122, small shared-expert matrices 114-129 (peak ~360). Per token: dense GEMVs
  ~5.9 ms, MoE GEMVs ~2.55 ms. gemv_q5_k: 40-60 registers (full occupancy), ~55 SASS per 8 values per lane,
  ~40% of issue rate, so neither occupancy nor issue alone explains it. ncu needs root (RmProfilingAdminOnly=1).
- Change: `moe_gemv` takes `pair: Option<(w2, y2)>`; with it, grid.z = 2 and z = 1 blocks run the matrices at byte
  offset w2 into the output at byte offset y2. The decode MoE step runs gate and up as one launch when both have the
  same type and shape.
- Test: `moe_gemv_selects_experts` also checks the paired launch against separate launches, bit for bit.
- Files: `src/cuda/kernels.cu`, `src/cuda/mod.rs`, `src/gpu_model.rs`, `tests/cuda.rs`.
- After: MoE GEMV 2530 -> 2405 us per token (interleaved nsys, 2 rounds: 2523/2532 vs 2373/2437). 4K (3 clean)
  decode 76.8/81.0/81.8 (was 77.5/80.4/80.8), prefill 1199/1632/1641. Parity 629/630/632. Short decode 22-74
  (host load 8, not usable).

### 2026-10-07 #18 — paired dense GEMVs (rejected); clean short decode 88.1 (no code change)
- GPU idle; host load 2.7-3.2 for the A/B, then 5-12.
- Tried: `Gpu::gemv2` runs two quantized matrices with the same type and columns as one launch (grid.y = 2, extra
  blocks exit), used for DeltaNet qkv + gate and beta + alpha, attention k + v, shared-expert gate + up. 80 fewer
  launches per token (1062 -> 982), bit-identical outputs (test passed).
- Result: no gain. Interleaved nsys dense GEMV 5946-6012 vs 5878-5998 us per token. Interleaved short decode without
  the profiler, 8 rounds: 4 rounds at 100% hits give 88.08/88.20/88.06/88.22 (base) vs 88.16/88.23/88.06/88.18;
  earlier rounds with 4-131 misses 86.2-87.9 for both. Reverted.
- Short decode at a02b4ec (100% hits, load ~2.8): **88.1 tok/s** (88.06-88.22, 4 runs). Was 86.7 at b9d9547.
- 32K at a02b4ec (3 clean): prefill 1228/1051/1050, decode 60.3/8.1/60.1 (the 8.1 run under host load 12).

### 2026-10-07 #19 — M5 start: MTP draft head in the CPU reference (6379338)
- GPU idle at start (forever_bear joined during the last test); host load 3.5.
- GGUF: block_count 41, nextn_predict_layers 1. blk.40 is a full-attention block (q 8192 with gate, k/v 512,
  q5_k / q6_k) with Q2_K experts (gate/up/down type 10, router bf16), plus nextn.eh_proj (q8_0, 4096 -> 2048),
  nextn.enorm, nextn.hnorm, nextn.shared_head_norm. No separate embedding or head: uses token_embd and output.
- Change: `Model::mtp_step(&mut Mtp, hidden, token)`: x = eh_proj [enorm(embed(token)), hnorm(hidden)], then the
  block (own KV cache in `Mtp`, position counter), shared_head_norm, output head. `step_h` also returns the last hidden
  state (before output_norm). `block` and `head` factored out of `step` (same arithmetic).
- Test: ignored `mtp_draft_acceptance` (CPU, ~35 s): greedy, 24 generated tokens of "Write a short story about a
  dragon.": **24/25 drafts** equal the main model's next-next token. Feeding the post-output_norm hidden state also
  gave 24/25 (hnorm absorbs the difference), so the pre-norm state is used. Asserts >= 60%.
- Other tests: `greedy_continuation_is_sane` (Paris), `tests/gpu_model.rs` 2/2 pass.
- Speed and parity unchanged (CPU reference only; GPU path untouched).

### 2026-10-07 #20 — M5 step 1: MTP draft head on the GPU decode path (be8f72b)
- GPU idle at start (gnome-clocks only); host load 6-13 (other loop's rustc).
- Change: `GpuModel::load(.., mtp: bool)`. With it, blk.40 loads as an attention layer with its own KV cache and its
  own `CpuWorker` (mailbox for layer 40 only); its Q2_K routed experts join the expert cache (`ExpertCache::new` takes
  the layer count; a 40-layer profile is padded, a 41-layer one truncated). `mtp_draft(token)` runs enorm/hnorm,
  eh_proj, the block and shared_head_norm + output head on the last step's hidden state, and records routing for
  layer 40. The per-layer body of `forward()` moved into `layer()` (same kernel sequence). `main.rs` passes false.
- Test `mtp_draft_matches_reference`: CPU `mtp_step` gets the GPU hidden state (`GpuModel::hidden`), so it checks the
  head alone. 9/9 positions same draft token, max |dlogit| 0.16-0.24 (range 24-39). A first version fed the CPU its own
  hidden state: main-model drift (up to 0.83) flipped a 0.26 top-2 gap at position 3.
- GPU tests now hold `common::gpu_lock()`: with three GPU tests, parallel test threads loaded two models and the
  second load failed (out of VRAM). Full suite passes.
- Speed, MTP off (interleaved short decode, load 7-9, not clean): HEAD a02b4ec 61.9 / 68.5 / 74.9, this 70.6 / 73.6 /
  75.2 tok/s. No regression. Parity not re-run: the main path issues the same kernels; `gpu_forward_matches_cpu_reference` passes.
- Draft cost not measured yet (not graph-captured; one extra worker spin thread per draft).

### 2026-10-07 #21 — M5 step 2a: gemvn, one weight pass for up to 4 GEMV inputs (0991789)
- GPU idle at start (gnome-clocks only); host load 4-6.
- Why: a 2-token verify forward only pays if each weight is read once for both tokens. Decode GEMVs are ~70% of a token.
- Change: `Gpu::gemvn(w, x, x_stride, y, y_stride, nt)`, nt 1-4: integer types quantize all inputs into shared memory,
  decode each weight block once and dp4a it against every input (nt is a template argument; a first version with a
  runtime nt and `unroll 2` was ~10% slower at nt 2). Float types (bf16 router, f32) run nt plain GEMVs.
- Test `gemvn_matches_gemv`: nt 1-4 bit for bit equal to per-input `gemv`, every 2-D integer type in the model.
- Timing (`gemv_bandwidth`, us for 1 / 2 / 3 / 4 inputs): q5_k 4096x2048 24.5 / 34.9 / 37.2 / 43.8; q6_k 2048x512
  7.4 / 9.1 / 10.6 / 12.1; q8_0 512x2048 5.3 / 6.7 / 7.9 / 9.1; q4_k head 248320x2048 1147 / 1645 / 2028 / 2376.
  So 2 tokens cost ~1.42x one token in dense GEMVs, 4 tokens ~1.8-2.1x.
- Estimate for k = 1 (2-token verify): dense GEMV 5.9 -> ~8.4 ms, MoE ~2.4 -> up to 4.8 (expert union, no sharing
  yet), rest ~3 -> ~4.5, plus the draft (~1.6 ms, of which the 248K-row head is 1.15) = ~19 ms per ~1.92 tokens vs
  11.4 ms per token: ~+15%. The draft head is the largest fixed cost; k = 2-3 with a 3-4 token verify amortizes better.
- No speed or parity change (kernel not used by decode yet).

### 2026-10-07 #22 — M5 step 2b: multi-token verify forward (5dd7576)
- GPU idle (gnome-clocks only); host load 1-11 (other loop's rustc).
- Change: `GpuModel::verify(&[tokens])`, 1-4 tokens at pos.. in one forward: dense matrices through `gemvn`
  (weights read once), q/k rope and `kv_store` over all tokens, then one `attn_decode` per token (causal via its
  step entry), `conv_seq` + `deltanet` over the tokens, MoE routing for all tokens in one `router_topk`/`map_hits`,
  `moe_gemv` with `xg` (token = slot / k), `moe_combine` grid.y = token. CPU worker mailbox holds `TMAX` = 4 tokens
  (word 1 = token count). `step_buf` holds [pos, seq] per token. `save_state`/`restore_state` copy conv and DeltaNet
  state (KV needs only `pos` rewind) for rejected drafts. `mtp_draft(token, row)` reads hidden row `row`.
- Test `verify_matches_steps`: verify of 4 and 2 tokens vs the same tokens stepped after `restore_state`: max |dlogit|
  0.0000 (bit-identical), next step after verify also identical; replay after restore identical.
- `mtp_draft_matches_reference` was flaky: one position sometimes at |dlogit| ~1.1 with a flipped argmax. HEAD
  0991789 shows the same 1.14 at pos 3 (argmax held there). Cause: near-tie in the draft block's expert routing.
  Test now allows one such position of 9.
- Speed, MTP off (interleaved short decode, host load 7-11, not clean): head 64.0/73.4/75.9, this 71.2/75.5/79.1.
  No regression. Parity not re-run: the nt = 1 path issues the same kernels with the same arguments.
- Verify cost not measured yet (not graph-captured).

### 2026-10-07 #23 — M5 step 2c: `generate --mtp`, speculative decoding k = 1 (8c53c70)
- Host load 4-11 (other loops), GPU shared through the lock.
- Change: `GpuModel::spec_step(t, row)`: MTP draft from hidden row `row`, `save_state`, verify [t, draft]. Accept:
  run the draft block on the draft token without the output head (fills its KV entry), next draft reads row 1.
  Reject: `restore_state` + plain step. Falls back to a plain step at the context end. `generate --mtp` uses it and
  prints the acceptance. Counter `drafts: (made, accepted)`.
- Fix: `prefill` returned 4 x vocab logits since 5dd7576 (logits buffer grew for verify); now sliced to n_vocab, and
  the last hidden state goes to row 0 for the first draft.
- Fix (hypothesis): two early runs died with `CUDA_ERROR_LAUNCH_FAILED` (first run of a batch, ~96% expert hits).
  Likely path: the CPU worker is preempted, the GPU moves on and overwrites a later layer's request with a newer
  seq, the worker spins forever on the old seq, and the next miss traps in `moe_combine`'s wait. The worker now skips
  a layer whose request is newer than its seq. Not reproduced since (6 warm runs, 3 with the model evicted from the
  page cache); compute-sanitizer showed 0 memory errors. Not proven.
- Test `mtp_spec_matches_greedy`: 64 greedy tokens with spec equal plain steps on the same instance (adapt off),
  28/36 drafts accepted. All gpu_model tests and the cuda suite pass.
- Speed (short dragon prompt, -n 128, interleaved, 100% hits): plain 81.6 / 84.9 / 65.9 (load) / 84.5, MTP 78.8 /
  47.6 (load 10) / 75.9 / 75.5. Acceptance 81.7-85.5%. MTP k = 1 is ~10% slower than plain for now.
- nsys of an MTP run: GPU busy 86% of the decode window (gaps 14%: verify and draft run uncaptured). Largest MTP
  costs: the 248K-row draft head (gemv_q4_k, 1.2 ms per draft) and the 2-token verify head (gemvn_q4_k, 1.6 ms).
  The 25K HtoD copies are expert uploads at load, not decode.
- Parity not re-run: the plain decode path is unchanged.

### 2026-10-07 #24 — draft vocab subset (rejected); verify and draft head graph-captured (eb7c540)
- Host load 7-28 (other loops); GPU shared through the lock.
- Tried: draft head scores only the first N vocab rows (low BPE ids taken as the frequent tokens), MTP short decode
  -n 128, 2 runs each: 248K 80.9 / 80.2 tok/s, 81.7% accepted; 65K 78.7 / 79.2, 76.7%; 32K 78.2 / 81.4, 79.2%;
  16K 73.3 / 74.1, 67.5%. The head time saved is lost to rejected drafts. Rejected; a subset needs real token
  frequencies (calibration), not id order.
- Change: `verify` keeps one CUDA graph per token count, the draft head one per (hidden row, output head on).
- Speed (MTP, interleaved, 100% hits, load 9-20): no graph 79.9 / 72.7 / 78.8 (mean 77.1), graph 81.7 / 72.4 / 83.6
  (mean 79.2), about +3%, inside the noise. Acceptance 58/71 in all six runs.
- Test fix: `prefill_matches_reference` failed once in the full suite (prefill error 1.3997, identical to HEAD;
  the decode yardstick moved to 1.2455 with adaptive expert moves). It now sets `adapt_every = 0`: full suite
  5/5 twice, prefill 1.3997 vs decode 1.4964 (thin margin).
- Parity not re-run: plain decode numerics unchanged (same kernels, now launched from a graph for nt > 1 only).
- MTP k = 1 (~79) is still level with plain decode (~81-85). The per-cycle fixed costs remain: draft head 1.2 ms
  and the second draft-block pass after an accept.

### 2026-10-07 #25 — MTP rollback from state snapshots; k chained drafts (6bd2b7c)
- Host load 4-25 (other loops); GPU shared through the lock.
- Verify cost at load 3.8: 1 token 11.81 ms (best 11.52), 2 tokens 16.25, 3 tokens 20.67, 4 tokens 25.21. That is
  linear at +4.45 ms per extra token. The draft head is 1.94 ms at best.
- Change 1: during `verify`, `conv_state` and `deltanet_seq` write the recurrent state after each of the first k
  tokens into per-layer snapshot buffers. `rollback(start, keep)` copies snapshot keep-1 back. This replaces
  restore plus re-stepping the accepted tokens. Kernel tests check the snapshots (deltanet: 0.00e0 vs steps);
  new `rollback_matches_step` test.
- Speed, `--mtp 1`, dragon prompt -n 128, 3 interleaved runs each: old 83.7 / 67.0 / 80.7 (mean 77.1), rollback
  93.8 / 87.2 / 94.4 (mean 91.8), plain 78.5 / 85.7 / 74.6 (mean 79.6). Acceptance 58/71 (81.7%) in every MTP run.
- Change 2: `--mtp N` chains N drafts (each from the draft head's own hidden state) and verifies N+1 tokens.
  `mtp_spec_matches_greedy` checks k = 1, 2, 3 against plain greedy (equal); acceptance 28/36, 35/58, 39/75.
- Speed per k (3 interleaved runs, load 7-22): k0 87.4 / 75.1 / 76.7 (mean 79.7); k1 97.4 / 85.1 / 79.7 (87.4);
  k2 64.1 / 73.1 / 74.1 (70.4, 62% accepted); k3 62.6 / 64.8 / 63.1 (63.5, 48% accepted). k >= 2 also touches more
  experts (one k2 run had 32 misses, one k3 run 110). Acceptance falls faster than verify cost per token, and each
  accepted draft re-runs the draft head. k = 1 stays the default; k = 2-3 stay available.
- Bug found by the k bench: in `load`, `let k = n_expert_used` shadowed the draft count and was stored as
  `self.k` (8), so the CLI asked for 9-token verifies. Tests set `gm.k` by hand and missed it; they now assert it.
- Parity not re-run: plain decode numerics unchanged (snapshot writes are off with k = 0).

### 2026-10-07 #26 — M6: `chat` and `serve` (8263342)
- `strata chat`: stdin turns in the model's ChatML template. Thinking is off by default (`--think` on), and the
  whole conversation is prefilled again each turn (no prefix reuse yet).
- `strata serve`: OpenAI-compatible, std::net only (no new crate). POST /v1/chat/completions, streamed (SSE, final
  chunk carries usage) or not; GET /v1/models; one request at a time (later ones wait in the listen backlog).
  Greedy only: sampling fields are ignored. `chat_template_kwargs.enable_thinking` turns thinking on.
- `generate`, `chat` and `serve` share one decode loop. `generate --mtp` text is byte-identical to the previous
  binary (dragon prompt, 64 tokens).
- Live check (GPU, `--mtp`): 2-turn chat "2+2" -> "4", "And times 3?" -> "12"; non-streamed and streamed
  completions correct; 404 and 400 paths answer JSON errors. `tests/serve.rs` covers the template and request parsing.
- Not done from the spec: sampling (temperature / top-k / top-p), prefix reuse, `--api-key`, HTTP 500 on a failed
  request.

### 2026-10-08 #27 — M6: `calibrate`, `bench`, sampling, `--api-key` (a336559)
- Found this work uncommitted from the previous session; finished, verified and committed it.
- `strata calibrate`: times prefill chunk 512/1024/2048/4096 (best of 3), batched prefill vs steps for 8-128 token
  prompts (sets `short`, the batched-prefill threshold, was a fixed 32), and decode at MTP k = 0..3 on a 128-token
  greedy run. Writes `~/.cache/rusting_strata/strata-<model>.json`; `--chunk`, `--mtp`, `--threads` override it.
  Result on the reference machine (load 1.5 -> 13 during the run): chunk 4096 (748 / 1075 / 1144 / 1544 tok/s),
  short 16 (8 tok: steps 111 ms vs batched 146; 16 tok: 234 vs 184), mtp 1 (k0 78.0, k1 86.0, k2 83.3, k3 67.3).
- Not calibrated: threads (rayon's global pool is fixed once built, so it needs one process per setting) and the
  expert-cache split (placement is already VRAM-driven). Split out as a follow-up.
- `strata bench --pp 512,4096 --tg 128 --reps 3`: prefill and decode tok/s mean (min-max); `--ctx` grows to fit.
  First run at load 13 (other loop): pp512 559 (340-681), tg128 67.5 (62.6-74.9). Not a clean number.
- Sampling (`ops::Sampler`): temperature, top-k, top-p, seed (xorshift). Greedy when temp is 0, which keeps the
  MTP path; temp > 0 decodes one token per step. Test `sampler_cuts_and_matches_softmax` (20000 draws within 2% of
  the softmax, top-p and top-k cuts). Live: seeds 1 and 2 at temp 1.2 give different text.
- `serve`: request fields temperature / top_k / top_p / seed, `--api-key` (401 without `Bearer <key>`, constant-time
  compare, test `checks_api_key`), HTTP 500 JSON on a failed non-streamed request. Live: 401 checked, greedy and
  temp 0.8 completions correct.
- GPU suite `tests/gpu_model.rs` 6/6 pass (incl. `mtp_spec_matches_greedy`). Parity not re-run: greedy picks are
  `argmax` as before.

### 2026-10-08 #28 — security fix for `--api-key` (2e9bf2a); prefix reuse across requests (99fbd11)
- Background security review of a336559: an empty `--api-key ""` matched requests with no `Authorization` header,
  and a key on the command line shows in the process list. Fix: empty key refused (at start and in `key_ok`), key
  also read from `STRATA_API_KEY`. Test `checks_api_key` covers the empty case.
- Prefix reuse: `GpuModel::prefill_cached(tokens, keep)` saves the recurrent state (existing `save_state`) after
  `tokens[..keep]`, the message history without the assistant header. The next call whose history starts with the
  saved tokens restores it and prefills only the rest (KV entries before the saved position are untouched by the
  reply; the draft head restarts at 0). `reset` forgets the saved prefix because it zeroes the KV. A hybrid model
  cannot rewind its recurrent state to an arbitrary token, so reuse works only at this saved point.
- `chat` and `serve` use it; `serve` reports `usage.prompt_tokens_details.cached_tokens`.
- Test `prefill_cached_reuses_history`: logits bit-identical to a fresh run split at the same points, with decode
  steps between calls; a prompt that does not extend the saved history starts over. GPU suite 7/7 pass.
- Live (serve, `max_tokens` 1, load 28): 2320-token second turn with 2295 reused 0.52 / 0.38 / 0.56 s, the same
  request fresh 1.96 / 1.95 / 1.97 s.
- Parity not re-run: plain prefill and decode are unchanged.

### 2026-10-08 #29 — README (35f81d4); security review note on prefix reuse
- README with build, usage, and the measured numbers vs llama.cpp (from this log and `docs/m4-results.md`).
- Background security review of 99fbd11: prefix reuse is a cross-request side channel (`cached_tokens` and the
  response time show whether the previous request had the same history). Accepted as designed: the server has one
  API key and one user, like llama.cpp's prompt cache. Documented in the README; no code change.
- Host load 22-29 (another loop runs 5 `forever_bear` processes, swap 28.6 of 31 GB): no benchmarks taken.

### 2026-10-08 #30 — prefill GEMM bound (ablation); cp.async f16 X (rejected, no code commit)
- `gemm_throughput` ablation (2 runs each, load 21-36, ms for attn_gate Q5_K 4096x2048x4096 / ssm_out Q6_K /
  ffn_down_exps IQ3_XXS): base 5.1-5.3 / 6.4-6.6 / 0.59; X loads replaced by constants 3.7-3.9 / 6.3-6.6 /
  0.40-0.57; weight dequant replaced by constants 3.9-4.2 / 4.1-4.2 / 0.55-0.63; both 2.8-3.1 / 2.8-3.1 / 0.37
  (22-25 TFLOP/s, the f16 mma peak of ~25.6). So f16 tensor cores are within 1.7-2.3x of the current kernel, and
  X loads plus dequant are not hidden behind the mma.
- Tried: X pre-converted to f16, double-buffered in shared memory with cp.async (46 KB smem per block). 9.7-13.7
  TFLOP/s vs 10.4-14.5 base (3 runs, load 31): no gain, reverted. Register prefetch of the next X step re-measured:
  8-12.7 TFLOP/s, slower again (as in #13).
- Conclusion: the only lever left above ~25 TFLOP/s is int8 tensor cores (MMQ), which also removes the f32 -> f16
  X conversion. Next: int8 MMQ, split into (a) X quantized to int8 per 32 with scale and sum, (b) one weight type
  with a parity test against the f16 GEMM, (c) the other types.

### 2026-10-08 #31 — GPU held by YOLO training; parity reference lost (no commit)
- `target/` was deleted at 18:17 (disk 90% full), with `target/parity-ref.json` and all cached parity results.
- GPU held from 18:47 to ~19:55 by `flock .gpu.lock yolo detect train` (~/cs2), then by RustingCS2 `train-bc` runs.

### 2026-10-08 #32 — int8 MMQ prefill GEMM (3754e65)
- Reference rebuilt 20:04-20:11: `llama-server -ngl 99 --n-cpu-moe 20 -fa on`, 20 prompts x 32 tokens.
- Prefill GEMM for quantized weights: X quantized once to q8_1 per 32 (`quant_q8`), weights decoded to int8 in
  shared memory, dp4a (`gemmi_*`). IQ2_XS stays on f16 (`Gpu::mmq`). `STRATA_MMQ=<types>|none` overrides,
  `STRATA_LLAMA_NUMERICS=1` uses llama.cpp's activation scales and puts IQ2_XS on MMQ too.
- CUDA tests 7 pass (incl. `gemm_matches_reference_for_every_type` against the f16 GEMM).
- Parity (GPU, 6 runs each, f16 and MMQ interleaved in 20:22-20:39, load 1-6): f16 629/628/631/628/628/630 =
  mean 98.28% +- 0.20; MMQ 626/629/630/630/632/631 = mean **98.39% +- 0.32**. Same prompts miss in both (9, 15,
  17, 19 most). Llama numerics 627/640 = 97.97% (one run, stays opt-in).
- IQ2_XS on MMQ too (`STRATA_MMQ=<all types>`): 628/628/628 = 98.12% (adds 2 misses on prompt 18 every run),
  prefill 1808.0 (1802.3-1811.1) vs 1744.5 (1730.4-1752.5) default at 20:31. Rejected for parity.
- 4K bench (`--pp 4096 --tg 64 --reps 3`, 20:25, load 2.2-1.8): prefill f16 1609.3 (1572.7-1630.0), MMQ
  **1745.4** (1736.2-1753.1, +8.5%), llama numerics 1795.3 (1788.5-1798.8). Decode 62.3 / 66.4 / 60.5 tok/s at
  tg 64 (short run, not comparable with the 81.0 decode number; decode does not use this GEMM).

### 2026-10-08 #33 — llama.cpp parity ceiling measured; frequency draft vocab rejected (no code commit)
- Question: is 99% top-1 agreement reachable at all against the token-by-token llama-server reference?
- Method (`selfparity.py`, scratchpad): llama-server (same build and flags as the reference: `-ngl 99 --n-cpu-moe 20
  -fa on`) re-scored all 640 continuation positions with the whole context as one prompt batch (`cache_prompt`
  off, `n_predict 1`). 2 positions refused by the server's output-format check (HTTP 500), excluded.
  Saved as `target/llama-batch-argmax.json`.
- **llama.cpp batch vs its own token-by-token reference: 627 / 638 = 98.28%.** llama.cpp does not reach 99%
  against itself; the misses are near-ties that flip with the batch shape (prompt 13: 29/32, prompt 19: 30/32).
- Strata (#32 runs) vs llama batch: MMQ 623 / 626 / 629 / 627 / 629 / 626 of 638 (mean 98.22%), f16 625-626
  (98.04%), all-MMQ 621-622.
- Strata matches the reference **or** llama batch: MMQ 631 / 634 / 636 / 635 / 637 / 635 of 640 (mean 99.17%),
  f16 634 / 633.
- So Strata's 98.39% against the reference already exceeds llama.cpp's own 98.28% run-mode consistency. The
  remaining misses sit inside llama.cpp's numerical noise. See "Questions for owner".
- Draft-head vocab subset by measured frequency (Next candidate 1), checked offline first: corpus of 3.09M tokens
  (local docs and code, 44K distinct). Coverage of the reference continuations: top 16K 77.5% (id order 75.6%),
  top 32K 84.2% (id order 81.7%), top 32K plus prompt and generated tokens 86.1%. Only 2.5 points over the id
  order subset rejected in #24; the uk/ru/zh prompts fall outside any such list. Rejected without a GPU run.

### 2026-10-08 #34 — M6 v1 gate run; parity reference was llama.cpp CPU; GPU argmax for MTP (caa5f4b)
- Gate run (21:38-21:46, `docs/m6-results.md`), Strata at 671050e vs llama-bench (CUDA build 5d8ccdf9d), same
  session, 3 reps each. Strata: 4K prefill 1715.6 (1618.4-1765.5), decode after 512 81.7 (80.0-83.3), MTP k = 1
  88.9 (85.7-90.8); 32K prefill 1413.0, decode 67.8 (MTP 64.9). llama.cpp n-cpu-moe 10: pp512 1046.4, pp4096
  1048.1, tg128 69.8; n-cpu-moe 14 at d32768: pp512 775.7, tg128 57.9. Speed gate passed (1.64x prefill, 1.17x
  decode). MTP helps only at short context (+7.2 / +0.4 / -2.9 tok/s at 512 / 4K / 32K).
- Found: `~/llama.cpp/build` is a CPU-only build ("no usable GPU found, --gpu-layers option will be ignored").
  The first gate script used its llama-bench (pp512 55 tok/s, killed and rerun on `build-cuda`). Every parity
  reference so far (`target/parity-ref.json`, #33 self-parity) came from that build's llama-server, so the 99%
  gate has been measured against llama.cpp's CPU numerics.
- New reference from the CUDA build (`tools/parity.py --ref target/parity-ref-cuda.json`, same flags): its greedy
  continuations match the CPU reference at 459/640 positions (divergence carries forward). Strata GPU vs the CUDA
  reference: 624 / 621 / 624 = mean 97.34%, vs 98.39% against the CPU reference. Strata is closer to llama.cpp
  CPU, so the CPU reference stays the gate yardstick; the owner question below now covers both.
- GPU argmax (caa5f4b): `argmax_rows` kernel after the draft head and verify; copy one int per row instead of
  248K-float rows. MTP k = 1, `bench --pp 512 --tg 128 --reps 3`, interleaved: HEAD 92.2 / 91.0 / 90.2 (mean
  91.1), argmax 92.5 / 93.0 / 91.7 (mean 92.4, +1.4%). New test `argmax_rows_matches_ops_argmax`; all cuda and
  gpu_model tests pass, including `mtp_spec_matches_greedy`. `prefill_matches_reference` failed on its error
  bound at HEAD code paths (prefill 1.67-1.74 vs decode 1.43-1.97, argmax equal): int8 MMQ prefill; bound now
  1.5x decode error. First attempt used `INFINITY`, which NVRTC does not define before line 898.

### 2026-10-08 #35 — M6 calibration: threads and expert split measured; defaults kept (no code commit)
- Sweep at caa5f4b, `bench --pp 512,4096 --tg 128 --reps 3 --mtp 0`, two rounds, load 1.4-3.2. Round 1 is skewed
  by the adaptive expert cache warming up (hit rate 97.3% -> 100%); round 2 numbers below.
- `--threads` 4 / 6 / 8 / 12: prefill 512 907.4 / 907.1 / 907.0 / 907.7, 4K 1799.6 / 1789.0 / 1797.3 / 1795.3,
  decode after 4K 81.6 / 81.2 / 81.5 / 81.4. All within 0.6%: with ~100% expert hits the CPU does almost no
  expert work. No thread calibration added.
- `--vram-margin` (the expert split) 256 / 512 (default, threads rows) / 1024 / 2048 MB: resident experts
  8906 / 8644 / 8109 / 7071 of 10240; prefill 512 928.1 / ~907 / 877.5 / 800.8, 4K 1812.9 / ~1795 / 1767.9 /
  1708.9; decode 81.5 / ~81.4 / 81.3 / 81.0. More resident experts help prefill only.
- 32K (`--ctx 33280 --mtp 1 --reps 2`, two rounds): margin 512 prefill 1420.5 / 1424.3, decode 64.3 / 66.5;
  256 prefill 1432.2 / 1432.5, decode 65.9 / 65.9. Margin 128: CUDA_ERROR_OUT_OF_MEMORY at load, at 4K and 32K
  (desktop using 899 MiB), so 128-256 MB of allocations sit outside the plan.
- Default stays 512 MB: 256 gains +0.9-2.3% prefill but leaves little room for desktop VRAM that grows mid-run.
  `--vram-margin 256` is the documented option for a dedicated GPU.

### 2026-10-08 #36 — VRAM outside the plan: driver reserve, not Strata (no code commit)
- Question from #35: why does `--vram-margin 128` fail at load? Temporary `mem_get_info` prints (reverted):
  after load free 508 MB at margin 512; after the 512 and 4096 prefill+decode runs 500 / 468 MB (MTP on: 498 /
  466). Strata grows only ~40 MB after load (CUDA graphs, attention splits).
- The OOM is in `ExpertCache::new`, third pool: 291 MB requested with 444 MB free fails; 280 MB with 796 MB free
  works. Per-layer `table_dev` allocations cost 0 MB (checked by charging them to the budget first). The driver
  refuses allocations that would leave roughly 150 MB or less free on this GPU (it also drives the desktop).
- So the plan already accounts for Strata's own memory; the floor is the driver reserve plus ~40 MB of runtime
  growth. 256 MB is the practical minimum margin; 512 stays the default (#35). No VRAM to reclaim here.

### 2026-10-08 #37 — MTP step profile; one attention launch for verify (6438b34)

Profile (nsys, `bench --pp 512 --tg 256`, MTP k = 1, 82.9% accepted): a verify step of 2 tokens costs 21.1 ms of
GPU kernel time against 12.2 ms for one plain token, so each accepted draft saves less than it should. Most of the
extra is MoE: two tokens route to ~2x the experts (IQ2_XS expert GEMV 2.6 vs 1.3 ms, IQ3_XXS 1.8 vs 1.0 ms), which
is inherent to speculative decoding on a MoE. The draft head's full-vocab output GEMV costs 1.1 ms per step.
`moe_combine` looked 3x dearer, but its median is unchanged (3.0 vs 3.2 us); the mean is skewed by spin waits on
CPU expert misses (max 9.3 ms). `deltanet_seq` median 19.4 vs 10.4 us for 2 tokens, as expected.

Change: verify called `attn_decode` (decode, GQA and merge kernels) once per token. Now one launch with grid.z =
tokens serves all of them, so their blocks read the same K/V rows close in time (L2), with 3 launches per
attention layer instead of 3 per token. Test: `attn_prefill_matches_decode` also runs the batched launch and
asserts bit-identical output against per-token launches. cuda and gpu_model suites pass.

A/B, 3 interleaved rounds, `bench --gpu --mtp 1 --tg 128` (3 reps short, 2 reps 32K), load 0.8-1.4:

| Decode after | caa5f4b | 6438b34 |
|---|---|---|
| 512 | 91.5 (91.3-91.8) | 92.6 (92.0-93.3) |
| 4096 | 83.1 (83.0-83.1; r1 warm-up 79.5 left out) | 83.3 (83.3-83.3) |
| 32768 | 64.4 (63.9-65.2) | 65.1 (64.8-65.3) |

Prefill unchanged. The 32K gain is smaller than the ~2 ms per step that reading the KV once would save; L2 (3 MB)
shares little of a 65 MB-per-layer KV between the two tokens. Reading each K/V row once for both tokens needs both
queries in one block, which doubles the GQA kernel's registers (~250 already). MTP still loses at 32K (65.1 vs
67.8 without), so next candidate 5 stands.

### 2026-10-08 #38 — MTP break-even by context; verify attention blocks side by side (8e2b27e)

Break-even sweep at d5b4ba7 (`bench --gpu --tg 128 --reps 2`, 2 rounds, load 0.8-1.4), decode tok/s:

| Prompt | MTP off | MTP k = 1 | Accepted |
|---|---|---|---|
| 8192 | 79.2 (79.1-79.3) | 77.1 (77.1-77.1) | 60-61% |
| 16384 | 75.2 (75.1-75.3) | 72.7 (72.2-73.2) | 60-62% |
| 24576 | 71.8 (71.7-71.8) | 69.6 (68.7-70.4) | 62-66% |

MTP already lost at 8K. nsys at 8K: the batched verify attention (#37) cost 1.85x a single token per launch. The
GQA block uses ~250 registers x 256 threads, a whole SM, so with tokens in grid.z all of token 0's blocks ran
before any of token 1's and K/V came from DRAM twice. Change: the token index is the fastest part of grid.x
(block x = head block x / nt, token x % nt), so both tokens' blocks for a split run together and share K/V rows in
L2. Bit-identical (same test as #37); cuda and gpu_model suites pass.

A/B, 3 interleaved rounds, MTP k = 1, decode tok/s (`--reps 2`):

| Prompt | d5b4ba7 | 8e2b27e |
|---|---|---|
| 512 | 91.5 (91.3-91.6) | 91.9 (91.7-92.0) |
| 8192 | 78.9 (78.4-79.3) | 79.8 (79.7-79.9) |
| 32768 | 65.7 (65.2-65.9) | 66.3 (65.8-66.6) |

Every round improved. MTP now breaks even at 8K and still loses at 32K (66.3 vs 67.8).

### 2026-10-08 #39 — MTP off above a context length: rejected (no code commit)

Sweep at 8e2b27e, `bench --gpu --pp 4096,8192,12288,16384 --ctx 16896 --tg 128 --reps 2`, 2 rounds per mode
(load 1.4-4.8 at the start of round 1, 1.4-2.0 after), decode tok/s:

| Prompt | MTP off | MTP k = 1 |
|---|---|---|
| 4096 | 81.4 (81.3-81.5) | 79.6 (79.4-79.7) |
| 8192 | 79.2 (79.0-79.4) | 79.4 (79.3-79.5) |
| 12288 | 77.1 (77.0-77.1) | 78.6 (78.2-78.9) |
| 16384 | 75.2 (75.0-75.3) | 74.6 (73.7-75.5) |

Accepted 61.5-63.2%. With `--ctx 16896` MTP loses at 4K, where it won by 1.8 with `--ctx 4096`: the larger KV
leaves fewer resident experts, and a 2-token verify touches ~2x the experts, so misses cost MTP more. It wins at
12K. The payoff follows acceptance and expert hits, not position, so a fixed length gate would be wrong for some
prompts. Cost model from #37/#38: a k = 1 step costs 1.475 plain tokens at 512 and 1.66 at 32K, so MTP pays above
~48% acceptance short and ~66% at 32K. An adaptive gate (time MTP steps against occasional plain steps, keep the
faster) would follow both; candidate 5 now says that.

### 2026-10-08 #40 — gemmi decodes the next weight step during the mma (b91e912)

4K prefill profile at 8e2b27e (nsys, `--pp 4096 --tg 1 --reps 3`), GPU time per 4096-token rep: `gemm_iq2_xs`
(f16, experts) 548 ms, `gemmi_q5_k` 503 ms (17.8 TOP/s), `deltanet_seq` 395 ms, `gemmi_q6_k` 209 ms,
`gemmi_iq3_xxs` 199 ms, `attn_prefill_256` 106 ms, `quant_q8` 61 ms. Sum ~2.2 s, so prefill is GPU-bound.
2000 tok/s needs ~190 ms less per rep.

`gemmi_q` loaded X with cp.async double buffering but read and decoded the weights synchronously at the top of
each 64-deep k-step. Change: step kb + 1's weights are decoded into 8 registers right after the sync that opens
step kb, so their global loads overlap step kb's mma, and are stored to shared memory after the closing sync.
Same arithmetic; `gemm_matches_reference_for_every_type`, cuda and gpu_model suites pass.

A/B, 3 interleaved rounds, `bench --gpu --pp 4096 --tg 1 --reps 3 --mtp 0`, load 2.9-5.5 (falling):

| | 8e2b27e | b91e912 |
|---|---|---|
| 4K prefill | 1787.5 (1787.4-1787.6; r1 warm-up 1690.5 left out) | **1817.2** (1816.4-1817.8) |

Rejected, same session: the same pipelining in the f16 `gemm_q` (IQ2_XS experts). W and X both through registers
(24 more per thread): 1594.4 vs 1821.4 (-12.5%, occupancy). W only (8 registers): 1798.5 vs 1823.3 (-1.4%).

### 2026-10-08 #41 — deltanet_seq prefetches the next token's inputs (216468b)

`deltanet_seq` took 395 ms of a 4K prefill rep (#40 profile, 13.2 ms per layer call, ~3.2 us per token): each
token of the serial scan waited on global loads of k, q, gate, beta and v before its math. Those loads do not
depend on the state, so token t + 1's now load into registers while token t computes. Same arithmetic in the same
order; `deltanet_chunk_matches_steps`, cuda and gpu_model suites pass.

A/B against b91e912, 3 interleaved rounds, `bench --gpu --pp 4096 --tg 1 --reps 3 --mtp 0`, load ~1.0:

| | b91e912 | 216468b |
|---|---|---|
| 4K prefill | 1822.8 (1818.7-1826.3) | **1977.9** (1972.3-1984.9) |

2 rounds, `--pp 512,32768 --ctx 33280 --tg 128 --reps 2`: 512 prefill 857.7 vs 886.7, 32K prefill 1435.2 vs
**1532.9** (+6.8%), decode unchanged (86.2 vs 86.2 short, 67.4 vs 67.3 after 32K).

### 2026-10-08 #42 — rope_neox angles once per token (8dbdfed)

Profile at 216468b (4K prefill): `deltanet_seq` now 6.9 ms per call (was 13.2), `rope_neox` 1.86 ms per call for
~64 MB of traffic. The rotation angles use f64 `pow`, `sin` and `cos` (1/64 rate on GA106) and were computed in
every head's block. Now one block per token computes them once and rotates all heads of that token (16 for q, 2
for k). Same values; cuda and gpu_model suites pass.

A/B against 216468b, 3 interleaved rounds, `bench --gpu --pp 4096 --tg 1 --reps 3 --mtp 0`, load 0.7-0.9:

| | 216468b | 8dbdfed |
|---|---|---|
| 4K prefill | 1962.4 (1952.7-1973.9) | **1997.5** (1994.5-2003.1) |

### 2026-10-08 #44 — parity check; router_topk in registers (5dfb5d3)

Parity at 8dbdfed (`tools/parity.py --extra=--gpu`, 3 runs, 23:07-23:10, load 0.8-1.5): 629 / 631 / 630 of 640 =
mean **98.44%**, not below #32's 98.39%. The #40-#42 kernel changes keep parity.

Decode profile (nsys, `bench --pp 512 --tg 256 --mtp 0`, 99.6% expert hits), GPU time per token: `gemv_q5_k` 3.09
ms (~754 MB, ~244 GB/s), `gemv_q6_k` 1.63 ms (~259 MB, ~159 GB/s), expert GEMVs 2.35 ms, output `gemv_q4_k` 1.13
ms, `gemv_f32` 0.71 ms, `router_topk` 0.54 ms (13.4 us per call for top 8 of 256), `deltanet_seq` 0.43 ms,
`rms_norm` 0.41 ms. Ideal at 360 GB/s for the ~1.73 GB read per token: ~4.8 ms; measured ~11.9 ms.

Change: `router_topk` re-read all logits and the taken ids from global memory for each of the k picks. Each lane
now keeps its logits in registers (expert lane + 32 r, up to 512 experts) with a taken bitmask. Same arithmetic
and tie order: a temporary test showed bit-identical ids and weights against the old kernel over 4096 tokens. New
test `router_topk_matches_ops_top_k` checks ids and weights against `ops::softmax` + `ops::top_k`, ties included.
cuda (9) and gpu_model (7) suites pass.

A/B against 8dbdfed, 3 interleaved rounds, `bench --gpu --pp 512,4096 --tg 128 --reps 3`, load 1.4-2.5 (r1 of
8dbdfed plain was cold, left out):

| Decode | 8dbdfed | 5dfb5d3 |
|---|---|---|
| after 512 | 86.8 (86.7-86.8) | **89.5** (88.9-90.3) |
| after 4096 | 81.2 (81.1-81.2) | **83.3** (82.9-83.7) |
| after 512, MTP k = 1 | 90.9 (90.0-91.5) | **93.0** (92.9-93.2) |
| after 4096, MTP k = 1 | 80.8 (80.4-81.4) | **82.0** (80.7-82.6) |

Prefill unchanged (4K ~2000-2020 in both).

### 2026-10-08 #45 — expert GEMVs with 2 rows per warp (871aa18)

Per-shape decode profile at 8dbdfed: expert GEMVs run at ~140 GB/s (IQ2_XS gate + up, 34.4 us for 8 experts x 2
matrices, ~4.9 MB) and ~125 GB/s (IQ3_XXS down), against ~245-265 GB/s for the large dense Q5_K GEMVs.
`gemv_q6_k` on `ssm_out` (2048 x 4096) runs at ~190 GB/s. `STRATA_RPW` sweep for all GEMVs (1, 2, 4): the default
(2048 / cols) stays best (89.4 vs 88.1 / 87.7 / 79.8 decode after 512).

Change: gate/up expert GEMVs (cols 2048, so 1 row per warp, 8 per block) quantized an 8 KB x per block for 4.7 KB
of weights. Expert GEMVs now take at least 2 rows per warp. Rows per warp 4 for gate/up and 2 or 8 for down
(default 4) were no better in a sweep. cuda and gpu_model suites pass; arithmetic per row unchanged.

A/B against 5dfb5d3, 3 interleaved rounds, `bench --gpu --pp 512,4096 --tg 128 --reps 3`, load 1.2-1.5:

| Decode | 5dfb5d3 | 871aa18 |
|---|---|---|
| after 512 | 89.2 (88.7-89.5) | 89.7 (89.6-89.9) |
| after 4096 | 82.9 (82.2-83.3) | 83.4 (83.0-83.7) |
| after 512, MTP k = 1 | 92.6 (92.2-92.9) | **93.8** (93.7-94.1) |
| after 4096, MTP k = 1 | 81.9 (80.6-82.6) | **82.8** (82.1-83.2) |

### 2026-10-08 #46 — integer Q3_K GEMV decode (23a5912)

MTP step profile at 871aa18 (nsys, `bench --pp 512 --tg 256 --mtp 1`, 82.9% accepted), per step (~21.7 ms GPU):
verify output head `gemvn_q4_k` 1.6 ms for 2 tokens, draft head output `gemv_q4_k` 1.1 ms, draft head down experts
`moe_gemv_q3_k` 166 us. `gemv_bandwidth` showed Q3_K experts at 20 GB/s (178 us): its `QI<11>::get` dequantized to
floats and divided by the scale to recover the integers (marked ponytail).

Change: `QI<11>::get` builds the integers from the 2 low bits and the high-bit mask with byte-wise SIMD, like Q6_K.
Same integers: `dequant_and_gemv_match_reference_for_every_type` gives the same Q3_K GEMV error before and after
(3.68e-7). cuda and gpu_model suites pass.

Kernel: 178.1 -> 39.7 us in `gemv_bandwidth` (20 -> 91 GB/s); in the MTP profile 166 -> 39 us per draft step.
End to end that is ~0.6% of an MTP step. Whole-bench A/B runs at 23:30-23:39 were too noisy to show it (desktop in
use, 22 GB swapped, decode spreads of 70-85 tok/s for both binaries), so no end-to-end number is claimed.

### 2026-10-08 #47 — float GEMVs one row per block (03bf6d7)

Decode profile by shape at 871aa18: `gemv_f32` 0.79 ms per token, of which ssm alpha and beta (32 x 2048, 60 calls
per token) 4.2 us each at grid 4 (4 SMs, ~62 GB/s), the router (256 x 2048) 8.5 us at grid 32, the shared-expert
gate (1 x 2048) 2.8 us.

Change: float GEMVs launch one 32-thread block per row (was 8 rows per 256-thread block). Same per-row
arithmetic; cuda and gpu_model suites pass.

nsys, `bench --pp 512 --tg 256 --mtp 0`, median per call (total per token):

| Shape | 23a5912 | 2 rows per block | 03bf6d7 |
|---|---|---|---|
| 32 x 2048 | 4.22 us (0.304 ms) | 3.01 us (0.195 ms) | 3.04 us (0.187 ms) |
| 256 x 2048 | 8.48 us (0.371 ms) | 7.65 us (0.350 ms) | 7.58 us (0.326 ms) |
| 1 x 2048 | 2.75 us (0.119 ms) | 2.82 us (0.121 ms) | 2.82 us (0.118 ms) |

0.79 -> 0.63 ms of GPU time per token (~1.3% of a decode step). The desktop was in use (swap 22 GB), so whole-bench
A/B numbers were too noisy to report; see #46.

### 2026-10-09 #48 — Q6_K and Q3_K GEMVs with 4-byte loads (78732c3)

`gemv_bandwidth` per shape at 720031d (it now times every distinct shape, not the first tensor of each type):
Q6_K `ssm_out` (2048 x 4096) 194 GB/s and `attn_q` (8192 x 2048) 219 GB/s, against 226-260 GB/s for Q5_K at the
same shapes. Q6_K (210 bytes) and Q3_K (110 bytes) superblocks are only 2-aligned, so `QI<14>` and `QI<11>` read
their quants with 8 two-byte loads per lane.

Change: `ld8a2` reads 8 bytes with two 4-byte loads when the address is 4-aligned, else three 4-byte loads around
it and a funnel shift. The alignment is the superblock's, so the branch is uniform across a warp. Same integers:
GEMV error unchanged (Q6_K 4.18e-7, Q3_K 3.68e-7); cuda and gpu_model suites pass.

`gemv_bandwidth`, 3 interleaved rounds (identical across rounds to 0.4 us):

| Tensor | 720031d | 78732c3 |
|---|---|---|
| Q6_K ssm_out 2048 x 4096 | 35.5 us | **27.8 us** (246 GB/s) |
| Q6_K attn_q 8192 x 2048 | 62.8 us | **48.7 us** (283 GB/s) |
| Q6_K 2048 x 512 / 512 x 2048 | 7.4 / 7.1 us | 6.1 / 5.7 us |
| Q3_K experts (moe8) | 39.2 us | 34.7 us |

`bench --gpu --pp 512,4096 --tg 128 --reps 3`, 3 interleaved rounds, load 4.3-6.4 (old round 3 ran at load 12.6
with 50-85 tok/s spreads and is left out; the new round 3 ran at load 15-20):

| | 720031d | 78732c3 |
|---|---|---|
| Decode after 512 | 89.9 (89.9-89.9) | **92.5** (91.8-92.9) |
| Decode after 4096 | 84.5 (84.4-84.5) | **86.1** (85.5-86.4) |
| Decode after 512, MTP k = 1 | 95.9 (95.3-96.5) | 95.2 (94.6-96.2) |
| Decode after 4096, MTP k = 1 | 82.4 (80.5-84.3) | **85.1** (83.4-86.0) |
| 4K prefill | 2014.4 (2009.7-2019.1) | **2044.4** (2034.9-2049.7) |

Prefill gains too: the MMQ prefill GEMM reads Q6_K through the same `QI<14>::get`.

### 2026-10-09 #49 — 16-bit and 32-bit loads for quant headers (78422f8)

`ld16` built 2 bytes from two byte loads (`p[0] | p[1] << 8`), so every half-precision scale (`ldh`) and every
2-byte quant read cost two loads. Every field it reads sits at an even offset in a superblock with an even size,
so it is now one 16-bit load. Q4_K and Q5_K also read d and dmin with one 4-byte load (`k4_dm`); Q4_K reads its 12
scale bytes as three 4-byte loads too (in `gemv_bandwidth` that helps the Q4_K output head, 975 -> 948-960 us, and
slows the Q5_K shapes, so Q5_K keeps byte loads for scales). GEMV error is unchanged for all 11 types; cuda and
gpu_model suites pass.

`gemv_bandwidth`, 2 interleaved rounds, us:

| Tensor | e1a1aa0 | 16-bit `ld16` | 78422f8 |
|---|---|---|---|
| Q4_K output 248320 x 2048 | 1150 | 976 | **948-960** (~300 GB/s) |
| Q5_K token_embd 248320 x 2048 | 1287 | 1169 | 1129-1139 |
| Q5_K attn_qkv 8192 x 2048 | 44.5-44.8 | 41.5 | 40.8 |
| Q5_K attn_gate 4096 x 2048 | 24.4 | 22.9 | 22.7 |
| Q5_K attn_output 2048 x 4096 | 25.7 | 24.1 | 23.8 |
| IQ3_XXS gate experts | 24.0 | 19.4 | 19.4 |
| IQ2_XS gate experts | 18.7 | 17.0 | 17.0 |
| Q8_0 / BF16 512-256 rows | 5.3 / 5.2 | 4.1-4.3 / 4.1 | same |

`bench --gpu --pp 512,4096 --tg 128 --reps 3`, 3 interleaved rounds, load 2.0-14.5:

| | e1a1aa0 | 78422f8 |
|---|---|---|
| Decode after 512 | 93.0 (92.7-93.4) | **96.1** (94.5-97.0) |
| Decode after 4096 | 86.5 (86.0-86.9) | **90.4** (89.5-91.2) |
| Decode after 512, MTP k = 1 | 95.5 (92.7-97.0) | **100.5** (98.3-102.3) |
| Decode after 4096, MTP k = 1 | 84.7 (79.7-87.2) | **90.7** (88.7-91.8) |
| 4K prefill | 2061.9 (2057.6-2070.4) | **2085.7** (2077.5-2095.5) |

Parity at 78422f8 (`tools/parity.py --extra=--gpu`, 3 runs, 00:56-01:20, load 1.5-6.6): 633 / 629 / 631 of 640 =
mean **98.59%**, not below #44's 98.44%.

### 2026-10-09 #50 — dense GEMV rows per warp by row count (d24402d)

`STRATA_RPW` sweep (1, 2, 4) at 78422f8 with `gemv_bandwidth`, 2 rounds: after #48-#49 the GEMVs are less
load-bound, and 2 rows per warp now wins on matrices with 2048 rows or more (output head 954 -> 912-922 us,
2048 x 4096 Q6_K / Q5_K 27.6 / 23.8 -> 26.2 / 22.7 us, 2048 x 512 Q6_K / Q8_0 5.3-5.8 / 4.1-4.3 -> 5.2 / 3.9 us)
and loses on 512-row matrices (Q6_K attn_k 5.6 -> 6.4 us, Q5_K shared gate 4.9 -> 5.6 us). The old default was
2048 / cols (1 for 2048 columns and more, 4 for 512).

Change: dense GEMVs (and gemvn) use 2 rows per warp from 2048 rows, else 1. Expert GEMVs keep their rule (#45).
cuda and gpu_model suites pass; per-row arithmetic unchanged.

`gemv_bandwidth`, 2 interleaved rounds: output head 954-961 -> 918 us, its 2-input gemvn (MTP verify)
1369-1373 -> 1156 us, Q5_K attn_qkv gemvn 2 inputs 57.1 -> 50.9 us.

`bench --gpu --pp 512,4096 --tg 128 --reps 3`, 3 interleaved rounds, load 4.0-17.3:

| | 780367d | d24402d |
|---|---|---|
| Decode after 512 | 97.6 (96.2-98.4) | **99.1** (99.1-99.2) |
| Decode after 4096 | 89.7 (89.0-91.1) | **91.6** (91.1-91.9) |
| Decode after 512, MTP k = 1 | 101.7 (100.7-103.5) | **105.1** (105.0-105.3) |
| Decode after 4096, MTP k = 1 | 90.8 (89.4-91.5) | **94.0** (93.8-94.1) |

Prefill unchanged (4K ~2050-2077 in both).

### 2026-10-09 #52 — chunked decode attention (2a51d2e)

Profile at ec8200a (4K prompt, MTP k = 1): `attn_decode_gqa` 167 us per call, ~12 calls per step (~1.7 ms of
~18.9 ms). New ignored test `attn_decode_bandwidth` (16 query heads, 2 KV heads, head_dim 256, both kernels,
300 ms warm-up) showed it far from bandwidth at every length (4K, 1 / 2 query tokens: 102 / 183 us; 8 MB of K/V
is ~23 us at 360 GB/s), and 2 query tokens cost twice 1 token, so the limit was per-block work, not DRAM.

Change: positions in chunks of 32 (4 per warp). Warps compute the chunk's scores for all G heads, reading q from
shared memory once per chunk and reducing the 32 (position, head) partials with one scatter reduction (31
shuffles instead of 160); warp h then runs the online softmax of head h over the chunk; warps add the weighted V
rows. All warps share the running max, so the end-of-block merge is a plain sum (no expf). The first version
kept the scatter's array in local memory (392 LDL/STL in SASS, the loop with a halving counter did not unroll);
recursion on a template parameter fixed it. `ATT_GQA_MIN` 4096 -> 2048 (new break-even: 2K g1 51.9 / gqa 49.8 us).

`attn_decode_bandwidth`, gqa kernel, 1 / 2 query tokens (us):

| Positions | ec8200a | 2a51d2e |
|---|---|---|
| 4096 | 97-102 / 177-183 | 64-68 / 107-121 |
| 16384 | 199-200 / 367-368 | 154-158 / 253-257 |
| 32768 | 332-336 / 620 | 257-261 / 423-427 |

Rejected variants: next-chunk register prefetch (no gain, registers 168 -> 252), G <= 4 per block (slower at
every length), `__launch_bounds__(256, 2)` (spills; only 32K with 2 tokens gains).

`bench --gpu --pp 512,4096 --tg 128 --reps 3`, 3 interleaved rounds, load 4.6-21.4; round 2 hit a load spike
(MTP spreads 54-107), so the means are rounds 1 and 3:

| | ec8200a | 2a51d2e |
|---|---|---|
| Decode after 4096 | 91.6 (91.3-91.8) | **93.9** (93.8-93.9) |
| Decode after 4096, MTP k = 1 | 91.8 (91.6-91.9) | **94.5** (93.8-95.1) |
| Decode after 512 (both MTP modes), 4K prefill | unchanged | unchanged |

`--pp 16384 --tg 128 --reps 3`, 2 rounds: decode 81.9 (80.8-82.9) -> **87.0** (86.7-87.3); MTP k = 1 noisy
(old 80.3 / 81.0, new 63.7 / 82.5, spreads up to 57-77) and not above MTP off at 16K for either binary.

Parity: new binary 628 / 631 / 632 = 98.49%, below the 98.59% logged at #49. Interleaved control under the same
conditions (04:17-04:23): ec8200a 628 / 632 / 627 = 98.28%, 2a51d2e 628 / 629 / 629 = 98.23%. Equal within run-to-run
noise (single runs of the unchanged binary span 627-633); the 98.59% at #49 was a high draw. New binary over 6
runs: 3777 / 3840 = 98.36%. cuda and gpu_model suites pass (attention parity test covers both kernels, batched
tokens and every head_dim).

## Tried and rejected

- #51 (measured, not built): expert GEMVs with the per-block x quantization removed (upper bound for quantizing
  x once per token) in `gemv_bandwidth`: IQ2_XS gate+up 17.0 -> 14.9 us, IQ3_XXS down 24.2 -> 21.6 us, others
  within noise. At most ~0.15 ms per token (~1.5% of decode) before the cost of reading int8 x back; skipped.
- #43: deltanet_seq with 8 state floats per lane (16 before; twice the warps, shorter dependency chains):
  4K prefill 1903.5 vs 1995.9 at 8dbdfed (3 interleaved rounds). Splitting its dot products into 4 partial sums:
  2002 vs 1993, noise. Both reverted.
- #40: register pipelining of W (and X) loads in the f16 prefill `gemm_q`: -1.4% (W) and -12.5% (W + X).
- #32: `STRATA_LLAMA_NUMERICS` as default: parity 97.97% vs 98.39%. IQ2_XS on MMQ: 98.12% (3 runs), +3.6%
  prefill.
- #25: MTP k = 2-3 chained drafts: correct, but 70.4 / 63.5 tok/s vs 87.4 for k = 1 (acceptance 62% / 48%).

- #7: int8 activation rounding in the CPU reference to mimic llama.cpp (block 32: 443/448, block 256: 439/448,
  f32: 445/448 on prompts 0-13). Worse. Redo only with llama.cpp's exact dot-product kernels.
- #12: first GQA version without prefetch and with merge weights in global memory: 32K 410 us, 4K 101 + 13 us.
  Replaced by the committed version.
- #13: GEMM_BM 256 / 64, GEMM_BK 32, launch bounds (256, 2), register prefetch of the next K step. None beat
  128x64x64; prefetch was 1.6x slower. See the #13 entry.
- #30: f16 X double-buffered with cp.async in the prefill GEMM: no gain (9.7-13.7 vs 10.4-14.5 TFLOP/s).
- #15: runtime G = 1 inside the single GQA kernel at short context (31 us, worse than G = 8): the kernel keeps 247
  registers whatever G a block takes. Fixed by a separate kernel.
- #8: nothing rejected. Not tried: choosing the split count from n_pos (needs a fixed grid for graph capture;
  could use a max grid and let extra blocks exit, which is what empty splits already do).
- #17 (decode GEMV, interleaved nsys, us per token dense / MoE, base ~5900 / ~2560): rows per warp 2 / 4 / 8
  (6100-10100 / 2580-2940); `#pragma unroll 8` (6040 / 2720); 16 values per lane with half-warps on alternate
  super-blocks (5890 / 2700); warp copies 4 super-blocks of its row into shared memory with 16-byte loads, then
  decodes (6590 / 2840). Upper bound for quantizing x once instead of per block: ~4% (x loads zeroed: 5690 / 2380).
- #24: draft head over the first N vocab rows (65K / 32K / 16K): acceptance drops 82% -> 77-68%, no speed gain.
- #18: dense GEMV pairs in one launch (qkv + gate, beta + alpha, k + v, shared gate + up): 80 fewer launches per
  token, short decode 88.16 vs 88.14 at 100% hits. Graph launch gaps are not a cost at this size.

### 2026-10-09 #53 — accepted drafts join the next draft pass (1b4dba2)

- Before: after an accepted draft, `spec_step` re-ran the draft block once per accepted token (no output head, one
  sync each) to rebuild its draft head entry from the main model's hidden row. Now those tokens wait in
  `MtpHead::pend`. The next `spec_step` runs them with the new token as one nt = n + 1 batch over main rows 0..=n,
  with the output head on the last row only. If an intervening `step` or prefill overwrote the rows, the pending
  entries are dropped.
- Draft numerics are unchanged (acceptance is equal: k1 297/471 in both binaries), and greedy output is unchanged
  (`mtp_spec_matches_greedy` k = 1-3 passes; gpu_model 7/7). Main model kernels are unchanged, so parity was not
  re-run.
- Bench, `--pp 512,4096 --tg 128 --reps 3`, 3 interleaved rounds against 2a51d2e. Round 1 ran at load 3.4-5.3 and
  was noisy; rounds 2-3 ran at load 2.0-2.9. Decode tok/s, rounds 2-3:

| | 2a51d2e | 1b4dba2 |
|---|---|---|
| k1, 512 | 106.4 / 106.2 | 107.0 / 107.0 |
| k1, 4096 | 94.9 / 94.6 | 95.2 / 95.2 |
| k2, 512 | 99.5 / 99.6 | 99.0 / 98.8 |
| k2, 4096 | 85.0 / 85.1 | 85.8 / 85.9 |

- The gain is small (+0.6% for k = 1): a 2-token draft block reads two expert sets, so the batch saves only the
  attention and dense weight reads plus one sync. k = 2 still loses to k = 1 (49% accepted). 4K with MTP: 95.2;
  the 100 tok/s target is not met.

### 2026-10-09 #54 — float GEMVs batch verify tokens (0272e25)

- Profile of 4K MTP decode at 1b4dba2 (nsys, `--pp 4096 --tg 128 --mtp 1`): `gemv_f32` ran 276 times per
  verify cycle (~1.0 ms of ~13.5 ms). For F32/BF16, `gemvn` looped `gemv` per token. So the router (2 MB),
  ssm beta/alpha and the shared-expert gate were read twice per 2-token verify, with twice the launches.
- Change: `gemv_row` takes NT inputs and keeps the same per-token arithmetic. New `gemvn_f32`/`gemvn_bf16`
  kernels. `gemvn_matches_gemv` now also covers the float types and is bit-exact. cuda 9/9, gpu_model 7/7.
  Plain decode (nt = 1) and verify outputs are unchanged, so parity was not re-run.
- Bench `--mtp 1 --pp 512,4096 --tg 128 --reps 3`, 3 interleaved rounds against 1b4dba2, load 3.8-6.6:

| | 1b4dba2 | this |
|---|---|---|
| 512 | 106.6 / 107.2 / 107.1 | 109.8 / 109.9 / 109.6 |
| 4096 | 95.3 / 95.3 / 95.1 | 97.5 / 97.3 / 96.2 |

  +2.6% at 512 and +1.8% at 4K. Acceptance is the same (297/471).

### 2026-10-09 #55 — greedy spec_step skips the logits copy (2f416f9)

- Graph-level nsys of 4K MTP decode at 0272e25 (`--cuda-graph-trace=graph`): per verify cycle the GPU runs
  ~16.8 ms and sits idle ~750 µs (4.5%). The idle time is ~170 µs after each draft graph (mostly the verify
  `cuGraphLaunch`, 140 µs) and ~540 µs on average after each verify (median 382). That second gap covers the
  argmax read, the 1 MB logits row copy, expert-cache bookkeeping and, on rejects, the 30 × 2 MB DeltaNet rollback
  copies. Node-level tracing inflates `cuGraphLaunch` to ~370 µs, so launch costs come from graph-level traces only.
- Change: `spec_step` returns the argmax of row n (already on the host) and no longer copies that row's logits.
  `decode` uses it as the next token. Greedy `Sampler::pick` is `ops::argmax`, and `argmax_rows` breaks ties the
  same way (tested). gpu_model 7/7.
- Bench `--mtp 1`, interleaved against 0272e25: 512, 3 rounds, `--reps 3`: 110.1 / 110.7 / 110.5 vs
  111.9 / 111.8 / 111.4 (+1.2%). 4096, 3 rounds, `--reps 5`, load 2.0-4.7: 97.4 / 97.6 / 97.6 vs
  98.3 / 98.5 / 98.4 (+0.9%).
- Remaining host idle: the verify launch after each draft (needs the draft token's embedding on the GPU to enqueue
  draft and verify back to back) and the rollback copies (the next forward could read the snapshot in place).

### 2026-10-09 #56 — rejected: GPU-side token embedding for back-to-back draft and verify

- Idea from #55: embed token ids on the GPU so the draft token never goes to the host, and launch the verify graph
  right after the draft graph without a sync. `embed_<type>` kernel (grid = tokens) read `token_embd` (Q5_K,
  ~350 MB) zero-copy from page-locked host memory; drafts were copied device to device into the verify's token
  buffer. The GPU rows were bit-identical to the CPU `row` (new test), and gpu_model passed 7/7.
- `cuMemHostRegister` refuses the GGUF mmap and the `READ_ONLY` flag on this machine (`CUDA_ERROR_NOT_SUPPORTED`),
  so the tensor had to be copied into an owned 350 MB pinned buffer.
- Bench `--mtp 1`, interleaved against 2f416f9, 3 rounds `--reps 3`, load 2.4-3.5: 4096 99.5 / 99.3 / 98.2 vs
  99.6 / 99.2 / 97.9; 512 112.0 / 112.2 / 111.6 vs 112.3 / 112.1 / 108.4. An earlier run at load 4-15 was also
  flat (4096 97.4 vs 97.2). There is no gain to pay for 350 MB of pinned RAM and the extra code, so it is reverted
  (the diff is not kept in the tree). The ~170 µs gap after the draft is likely hidden already: the host builds the
  verify inputs while the draft graph runs, so the launch cost overlaps GPU work.

### 2026-10-09 #57 — vocab head GEMV takes 8 rows per warp (3e2ad06)

- `gemv_bandwidth` (load ~5): gemvn costs 25% more with 2 inputs than with 1 (output head Q4_K 915 -> 1158 us,
  attn_qkv 41 -> 51 us), although both read the weights once. Each block quantizes all nt inputs into shared memory
  before it starts, and the output head runs 15520 blocks of 16 rows.
- `STRATA_RPW` sweep, output head nt = 2: rpw 2 / 4 / 8 / 16 = 1158 / 1062 / 1027 / 1008 us; nt = 1 stays
  915-920 up to rpw 8 (942 at 16). The layer matrices (2048-8192 rows) get slower above rpw 2-4 (ssm_out 26 -> 46 us
  at rpw 8), so only matrices with 65536 rows or more take rpw 8.
- Row results are bit-identical (same warp code and order; only the block assignment changes). cuda 9/9,
  gpu_model 7/7. Parity not re-run.
- Bench `--mtp 1`, interleaved against 2f416f9, 3 rounds `--reps 3`, load 4-9: 4096 96.7 / 98.7 / 98.5 vs
  99.3 / 99.4 / 99.5 (+0.8% on the quiet rounds 2-3); 512 110.4 / 111.3 / 111.0 vs 111.6 / 111.9 / 111.9 (+0.7%).
- Next lever in the same place: quantize x once per launch into global memory instead of once per block. That would
  remove the per-block quantize and `__syncthreads` from every quantized GEMV, small layer matrices included.

### 2026-10-09 #58 — norm outputs are quantized once, not per GEMV block (245dd92)

- Measurement (`gemv_bandwidth` with `quant_x` stubbed out, load ~3): gemvn with 2 inputs drops 10-22% (attn_qkv
  50.3 -> 44.3 us, attn_output 28.3 -> 23.6, attn_k 6.9 -> 5.4, output head 1027 -> 950); MoE expert GEMVs with
  1 input drop 8-9% (IQ2_XS gate 16.9 -> 15.4, IQ3_XXS down 24.1 -> 22.2).
- Change: new `quant_xq` kernel (one block per token, same `quant_x` code) writes `h` into a "hq" buffer after the
  attention norm, the post norm and the output norm. `gemvn` and `moe_gemv` take an optional `xq` pointer and
  skip their per-block quantize and its `__syncthreads`. Every GEMV fed from `h` uses it, including nt = 1 steps
  (through gemvn, which equals gemv bit for bit). Inputs not from `h` (DeltaNet and attention outputs, expert
  down projections) still quantize per block.
- Tests: `gemvn_matches_gemv` and `moe_gemv_selects_experts` assert the `xq` path is bit-identical for every type.
  cuda 9/9, gpu_model 7/7. Parity not re-run (bit-identical).
- Bench `--mtp 1`, interleaved against 3e2ad06, 3 rounds `--reps 3`, load 5-15: 4096 99.3 / 99.1 / 99.1 vs
  99.8 / 99.7 / 99.7 (+0.6%); 512 112.4 / 111.9 / 111.7 vs 112.8 / 112.5 / 112.5 (+0.5%). A first run (load up to
  28 in rounds 2-3) agreed on its clean round: `--mtp 0` 4096 94.1 -> 94.5, 512 99.2 -> 99.7.
- The gain is about a third of the stub's upper bound: the 2 extra launches per layer (81 per forward) cost part of it. Folding
  the quantize into `rms_norm` (one kernel) would remove them; the DeltaNet `o` and attention `att` inputs could
  get the same treatment from `norm_gate` and the attention kernel.

### 2026-10-09 #59 — rms_norm writes the quantized copy itself (b0006cb)

- Follow-up to #58: `rms_norm` takes an optional `xq` pointer and, after a block barrier, runs `quant_x` on the row
  it just wrote. The attention norm, post norm and output norm use it (`rms_norm_q`), so the 81 `quant_xq`
  launches per forward are gone. `quant_xq` stays as the reference in tests.
- Test `rms_norm_q_matches_quant_xq`: the fused bytes equal `rms_norm` then `quant_xq`. cuda 10/10, gpu_model 7/7.
  Bit-identical, so parity was not re-run.
- Bench `--mtp 1`, interleaved against 245dd92, 3 rounds `--reps 3`, load 2.9-6.7: 4096 99.5 / 99.8 / 99.9 vs
  100.8 / 100.4 / 100.3 (+0.8%); 512 112.4 / 113.2 / 112.6 vs 113.7 / 113.6 / 113.2 (+0.7%).
- 4K decode with MTP is now at the 100 tok/s target on this machine (100.5 mean).
- 32K at b0006cb (`--pp 32768 --tg 128 --reps 3`, load 1.6-2.7): `--mtp 0` prefill 1582 (1567-1591), decode 78.8
  (77.2-79.6); `--mtp 1` decode 82.7 (80.1-84.0), 62.0% accepted. The 32K decode target (70) is met, and MTP now
  wins at 32K (it lost 66.3 vs 67.8 before #52-#59).

### 2026-10-09 #60 — `--llama-numerics` flag; llama.cpp CPU attention numerics in that mode (d6d4bf1)

- The mode existed only as `STRATA_LLAMA_NUMERICS`; `--llama-numerics` now sets it (goal 2 asks for the flag).
- llama.cpp's CPU decode attention (`flash_attn_ext_f16_one_chunk`, used token by token for the reference
  continuations) rounds q to f16, walks positions in order and keeps the V accumulator in f16; the KV length is
  padded to 256 and, from 512 positions, split into one chunk per thread (6 here) merged in f32. Its prefill path
  (tiled) is f32. Strata used f32 decode and f16 tensor-core prefill (q and softmax weights rounded to f16).
- Change, only under `LLAMA_Q8`: `attn_decode_llama` reproduces the decode algorithm; `attn_prefill_llama` is an
  f32 prefill attention. Both are slow (one position or one query at a time), for parity runs.
- Test `tests/llama_numerics.rs` (own binary, so the env var cannot leak into other tests): decode is 1.5-5e-5 from
  a CPU model of llama.cpp's algorithm vs 3-6e-4 for the default kernel; prefill is ~2e-7 from f64 vs 6-7e-5 for the
  tensor-core kernel. cuda 10/10, gpu_model 7/7. The default path is unchanged (compiled out).
- Parity (3 runs per binary, interleaved, load 1.6-11.8), `--llama-numerics`: before 629 / 627 / 630 and
  629 / 628 / 625 (mean 98.07%); with both kernels 634 / 630 / 629 and 633 / 631 / 630 (mean **98.62%**, best
  99.06%). Decode attention alone: 628 / 628 / 629, no measurable change. Default mode in the same session:
  627 / 630 / 628 (98.18%).
- Rejected: f32 GEMV instead of the f16 GEMM for float weights in prefill (router, DeltaNet alpha / beta), as
  llama.cpp's CPU does: 627 / 627 / 629 vs 633 / 631 / 630 for the binary without it. Within noise, but not better.
- Misses move between prompts from run to run (adaptive expert cache: GPU int8 vs CPU kernels), so a 3-run mean
  moves by about +-0.3%. 99% is not yet reached on average.

### 2026-10-09 #61 — `--fixed-experts`: repeatable parity runs (f687b4b)

Parity misses moved between prompts from run to run because the adaptive expert cache swaps experts between the
GPU and the CPU kernels, and each run reads and rewrites the shared routing profile. `--fixed-experts` places
experts in layer order, never swaps, and neither reads nor writes the profile.

- `--gpu --llama-numerics --fixed-experts`, 3 runs (load 3.3-4.4): 632/632/632 of 640 (98.75%), per-prompt rows
  identical in all three. Before (adaptive): 634/630/629 and 633/631/630, spread 5 tokens.
- Default runs are unchanged (flag off). gpu_model tests 7 passed.
- Single numerics changes can now be measured with one run each against 632.

### 2026-10-09 #62 — CPU q8_K activations round ties to even (3ce7f84)

`quantize_q8k` in src/cpu_moe.rs used `.round()` (ties away from zero). llama.cpp's `nearest_int` and the GPU's
`__float2int_rn` round ties to even. Now `round_ties_even()`.

- `--gpu --llama-numerics --fixed-experts` (load 6.2): 632 -> 633/640 (98.91%), one token gained on prompt 6,
  no row lost.
- `--gpu --fixed-experts` default mode (load 4.3 / 19.5): 629 -> 629/640 (98.28%), unchanged.

### 2026-10-09 #63 — rejected: batched prefill for every prompt in llama mode

llama-server batches every prompt; Strata steps prompts under 32 tokens (`short`). Setting `short = 0` under
`--llama-numerics` (`--fixed-experts`, load 8.0): 633 -> 633/640. Prompt 9 (31 tokens) gained one token, prompt 7
(23 tokens) lost one. No net change, reverted.

### 2026-10-09 #64 — probe: expert placement vs parity; DeltaNet prefill already matches

- `--llama-numerics --fixed-experts --vram-margin 20000` (nearly every routed expert on the CPU kernels, load 3.5):
  626/640, vs 633 with the default fixed placement. Seven prompts change by one or two tokens each, in both
  directions.
- The CPU kernels match llama.cpp's AVX2 code (q8_K activations, ties-to-even since #62, the same int sums and
  `fmadd` + `hsum` order). The only non-integer expert types are in layer 40 (MTP, Q2_K/Q3_K), which scoring does
  not use. So the drop is float-level perturbation (SwiGLU `expf`, the order of the weighted sum), not a wrong
  algorithm.
- llama.cpp's CPU build runs DeltaNet as the fused sequential f32 recurrence in prefill too (`fused_gdn_ch`), and so
  does `deltanet_seq`. The "DeltaNet prefill algorithm" candidate is moot.
- Conclusion: at 633/640 every ulp-level change flips one or two prompts either way (#63, this probe). One more
  agreeing token would pass 99% (634), but there is no remaining algorithmic difference to fix. Parity work stops
  here unless a llama.cpp tensor dump (`GGML_SCHED` eval callback) shows a real divergence.

### 2026-10-09 #65 — MTP k = 2 measured again at 010f0e4: still loses (no code commit)

`bench --gpu --pp 512,4096 --tg 128 --reps 3`, 3 interleaved rounds (load 4.4-20.9; round 3 quietest at 7-9):

| decode tok/s | k = 1 | k = 2 |
|---|---|---|
| 512, r1 / r2 / r3 | 105.0 / 107.0 / 113.8 | 96.8 / 102.0 / 102.3 |
| 4096, r1 / r2 / r3 | 86.6 / 88.9 / 101.5 | 68.5 / 86.8 / 89.2 |
| drafts accepted | 297-300 of 468-471 (63-64%) | 384 of 780 (49.2%) |

- k = 2 gains 0.98 tokens per verify against 0.63 for k = 1, so it needs a step at most 1.22x as dear to win.
  It costs ~1.35x (the third verify token routes to a third expert set). A cheaper draft head cannot close that.
  k = 1 stays the default; the "retry k = 2" goal is answered unless verify cost per token drops.
- Side result: 4K prefill in the quieter rounds 2030.6 (2011-2046) and 2072.0 (2070-2073) tok/s, above the 2000
  target; 1865 in round 1 at load 4.4-5.6 with other loops starting.

## Next candidates

1. MTP cost: the draft head output GEMV (`gemv_q4_k`, ~0.9 ms per step) is the largest draft cost. The catch-up
   pass is now batched (#53). Vocab subsets are out (#24 id order, #33 frequency).
2. MMQ tile tuning toward 2000 tok/s (IQ2_XS on MMQ costs parity, #32).
4. Parity (`--llama-numerics`, 98.62% at #60; 633/640 with `--fixed-experts` at #62, the baseline for single
   changes): no algorithmic difference left (#64). Next step only with a llama.cpp per-layer tensor dump to find a
   real divergence.
3. VRAM for more resident experts: the margin floor is the driver reserve (#36). Remaining levers: smaller prefill
   scratch (470 MB at chunk 4096) or KV in fewer bytes.
5. MTP at long context: now wins at 32K (82.7 vs 78.8, see the 32K note under #59). An adaptive gate is not needed
   unless a later change makes MTP lose there again.

## Questions for owner

- 2026-10-08 10:20: the GPU lock deadlocked. Another loop ran `flock .gpu.lock rusting test --release --json`
  (PID 3491381), which started `flock .gpu.lock cargo test --release --features cuda --lib prompt` (PID 3495422).
  The inner flock waits for the lock its own parent holds, so it waits forever, and every other loop waits behind
  it. The GPU itself was idle. The Strata loop was not allowed to run without the lock or to kill the other loop's
  processes. Please kill PID 3495422 (or 3491381), and make the other loop not take the lock twice.

- 2026-10-08 21:40: the 99% parity gate is above llama.cpp's own consistency. llama.cpp scored against its own
  token-by-token reference, with the context run as one batch, gives 98.28% (#33). Strata gives 98.39% against
  the same reference, and 99.17% when a token counts if it matches either llama.cpp mode. Proposal: score the
  gate against both llama.cpp modes (or count misses whose llama.cpp top-2 gap is under 0.1 nats as ties, per the
  spec's logit-delta report). Which definition should v1 use?
  Update #34: that reference came from a CPU-only llama.cpp build. Against the CUDA build's reference Strata scores
  97.34%. Which llama.cpp backend should the gate use?
