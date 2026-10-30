# Improvement loop log

Each iteration: one verified item. Newest entry last. Loop instructions: `/home/vasylt/Code/rusting/loop-prompts/strata.md`.

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
- v1.0.0 (#84): gate passed. 4K prefill 2016, decode 100.7 / 96.0 after 512 / 4K, MTP 109.9; parity 99.32%
  either llama.cpp mode in parity mode, 98.85% default.
- Vulkan (branch `vulkan`, #121, 36f529c; `bench --mtp 0`, tg128 after pp512): default model decode ~99-100, pp512
  ~1000, pp4096 ~1650; 4B decode 99.3, pp512 2080, pp4096 ~1950. llama.cpp Vulkan (#110, #116): 4B 95.4 / 3120 /
  2849, default model 47.1 / 124 / 582. Greedy tokens identical to CUDA. 4B prefill gap: see Questions for owner.
  Without cooperative matrices (#123, `STRATA_VK_NO_COOP=1`): 4B pp4096 457, default model pp4096 786, parity
  632-634/640 (#124).
- At 52b364e (#165, `bench --mtp 0`, tg128 after pp512, load 0.6-2.9): CUDA decode 117.5 (4B 98.5), Vulkan 117.0 (4B 104.0).
- At c959c94 (#170, same bench, load 0.5-0.9): CUDA decode 118.7 (4B not re-measured).
- At 654a270 (#69): 4K prefill 2093 tok/s; decode 101.2 after 512, 95.6 after 4096; MTP k = 1 110.9 /
  99.0 (63% accepted); 32K prefill 1562, decode 77.4, MTP 80.8. Parity 629/640 default, 633/640 with
  `--llama-numerics --fixed-experts` (the baseline for single changes, #62). All local GGUF quant types load (#68).

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
- GPU held from 18:47 to ~19:55 by `flock .gpu.lock yolo detect train` (~/Code/games/cs2), then by RustingCS2 `train-bc` runs.

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
- Found: `~/Clones/llama.cpp/build` is a CPU-only build ("no usable GPU found, --gpu-layers option will be ignored").
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

### 2026-10-09 #66 — IQ3_S (type 21): 9B IQ3_M and 35B IQ4_XS now load (795a93b)

Both models failed before with `unsupported quant type id 21`. Added the 512-entry grid, the CPU dequant and the
GPU `deq8`/`QI` paths (types now 12).

- Every-type test on the 9B (`STRATA_MODEL`): IQ3_S dequant max diff 0, gemv rel err 1.13e-6.
- Greedy, "The capital of France is", 24 tokens: llama.cpp, Strata `--gpu` and Strata CPU give identical text on
  both models. 9B `--gpu` decodes at 48.8 tok/s. 35B IQ4_XS `--gpu` decodes at 14.4 tok/s: 5806/10240 experts fit,
  hit rate 56.8%, and CPU experts take 84.7% of decode, because the CPU int dot has no IQ3_S kernel and falls back
  to the f32 matvec.
- Main model unchanged: cuda 10 pass / 3 ignored, gpu_model 7 pass, llama_numerics 1 pass.
- Next: an AVX2 IQ3_S int dot for the CPU expert path; IQ2_XXS, IQ2_S, IQ1_S, IQ1_M for the 27B
  (Qwen3.8-27B-UD-IQ3_XXS, which stops at type 19).

### 2026-10-09 #67 — IQ3_S integer dot for CPU experts (b2578c9)

Qwen3.6-35B IQ4_XS keeps gate/up experts in IQ3_S, and only 5806 of 10240 experts fit in VRAM. Before this change
those CPU experts used the f32 dequant fallback. Added scalar and AVX2 ports of `ggml_vec_dot_iq3_s_q8_K`.

- Single-thread `kernel_throughput`, blk.0 gate IQ3_S: 1206.8 us (0.87 Gweights/s) before, 81.2 us (12.91) after.
  8 experts of one layer (`fast_experts_match_reference`): 3.09 / 2.93 ms before, 0.48 / 0.47 ms after.
- Decode, "The capital of France is", `-n 64 --gpu --fixed-experts`, 3 interleaved rounds, load 6.9-10.5:
  11.0-13.3 tok/s (CPU experts 86-89% of decode) before, 33.6 / 47.3 / 36.2 tok/s (51-64%) after.
- New unit test `int_dots_match_dequant` checks all four integer kernels (scalar and AVX2) against the dequantized
  f32 dot on random blocks. A one-character shift bug in the AVX2 IQ3_S index made it fail (23.2 vs 46.3).
- 64-token greedy text: llama.cpp and Strata match for the first ~30 tokens, then differ ("cuisine, fashion" vs
  "vibrant culture"). The 24-token check of #66 still matches. Mixed GPU/CPU expert numerics on this model; the
  main model has no IQ3_S tensors and is untouched (cpu_moe tests pass).
- Without `--fixed-experts` the adaptive placement learns the hot experts across runs: 30.8, 72.3, 83.1 tok/s on
  three successive runs of the old binary (CPU share 65%, 14%, 1%).
- Next on this model: layer 39 down is Q6_K and has no integer kernel (1390 us per expert matvec).

### 2026-10-09 #68 — IQ2_XXS, IQ2_S, IQ1_S, IQ1_M: Qwen3.8-27B-UD-IQ3_XXS runs (bd1f895, dd68d3e, 654a270)

The dense 27B (arch qwen35) stopped at `unsupported quant type id 19`. That error now names the type
(`unsupported quant type IQ1_S (ggml type id 19); supported: ...`, bd1f895).

- dd68d3e: IQ2_XXS (16) and IQ2_S (22) grids, CPU dequant, GPU `deq8`/`QI`. IQ2_S keeps the f16 prefill GEMM by
  default, like IQ2_XS (scale per 16, #31).
- 654a270: IQ1_S (19) and IQ1_M (29). The integer path folds the ±1/8 offset into the weight,
  dl * (g ± 1/8) = (dl / 8) * (8g ± 1), an exact int8 in {-9, -7, -1, 1, 7, 9}, so no min term is needed.
- Found on the way: `gemvn` with 3 or 4 inputs over the 27B's 17408-column `ffn_down` needs 59-77 KB of shared
  memory, above the default 48 KB, and the launch failed with `CUDA_ERROR_INVALID_VALUE`. `gemvn_*` now opts in to
  96 KB like `attn_prefill_*`.
- Proof on the 27B (`STRATA_MODEL`): CPU dequant equals gguf-py's `dequantize` exactly (row 3, first 1024 values) for
  all four types; GPU dequant max diff 0; gemv rel err 1.2-1.3e-6; gemm 1.2e-6 (int8), 3.1e-6 / 4.1e-6 (f16). All
  `cuda` tests pass on the 27B; the per-type tests now skip types the build does not support, and the MoE test skips
  dense models.
- Greedy, "The capital of France is", 24 tokens: llama.cpp, Strata `--gpu` and Strata CPU give identical text.
  `--gpu`: 19.3 tok/s decode, plan 10784 MB with KV for 4096 tokens.
- Main model, `benchpend.sh` k = 1, 3 rounds interleaved with the pre-change binary (load 4.9-8.0): 4K decode
  100.1 / 100.5 / 98.9 (HEAD) vs 100.0 / 99.7 / 96.2 tok/s; 4K prefill 2048.7 / 2049.7 / 1978.7 vs 2038.9 / 2037.1 /
  1816.9 (round 3 of the new binary had one 178 tok/s outlier rep). No change.
- Every GGUF type in the local Qwen3.5/3.6/3.8 files now loads. Not yet supported: F16, Q4_0/Q4_1/Q5_0/Q5_1,
  IQ4_NL, TQ*, MXFP4.

### 2026-10-09 #69 — README numbers refreshed (3f78cf6)

At 654a270, `bench --gpu --pp 512,4096 --tg 128 --reps 3`, 3 interleaved rounds of `--mtp 0` and `--mtp 1`
(load 4.4-6.7):

| | mtp 0, r1 / r2 / r3 | mtp 1, r1 / r2 / r3 |
|---|---|---|
| 512 prefill | 918.1 / 1016.0 / 1014.3 | 993.7 / 983.8 / 991.9 |
| 4K prefill | 2094.9 / 2095.5 / 2088.8 | 2076.4 / 2066.4 / 2056.4 |
| decode after 512 | 100.6 / 101.5 / 101.4 | 111.7 / 108.2 / 112.7 |
| decode after 4K | 95.8 / 95.7 / 95.4 | 100.4 / 98.2 / 98.4 (63.1% accepted) |

32K (`--pp 32768 --ctx 33280`, one round, load 4.0-4.3): prefill 1561.6 / 1548.5, decode 77.4 (mtp 0) / 80.8
(mtp 1, 62.0% accepted). README now has these, the agreement numbers from #60-#62, the other-model table from
#66-#68 and the supported type list. The old README said 82% MTP acceptance; it is 63% on this bench text.
Target check: 4K prefill (2000) and 32K decode (70) are met; 4K decode with MTP is 98-100, just under 100.

### 2026-10-09 #70 — Q6_K integer CPU dot (a992641)

The 35B IQ4_XS file has Q6_K `ffn_down_exps` in 3 of 40 layers, which fell back to f32 dequantization on the CPU
expert path. Added a scalar Q6_K x Q8_K dot (llama.cpp generic layout) to `src/cpu_moe.rs`; the compiler
vectorizes it well enough that no AVX2 version was needed. `int_dots_match_dequant` now covers Q6_K, and a
planted shift bug in the high bits makes it fail.

35B IQ4_XS, `generate -n 64 --gpu --fixed-experts`, 3 interleaved rounds (load 8.0-10.2, another loop busy):

| | r1 | r2 | r3 |
|---|---|---|---|
| 654a270 decode | 23.6 (cold page cache) | 48.8 | 48.9 |
| a992641 decode | 54.3 | 54.1 | 53.7 |
| CPU expert share, old / new | 75.1% / 43.7% | 49.4% / 43.9% | 49.2% / 44.2% |

Warm decode 48.9 -> 54.0 tok/s (+11%). The 24-token text is unchanged and still equals llama.cpp. The main model
has no Q6_K experts (one Q2_K/Q3_K layer, which still uses the fallback), so it is unaffected.

### 2026-10-09 #71 — 32 decode attention splits (144b43b)

Profile (nsys, `--cuda-graph-trace=node`, bench --pp 4096 --tg 128): decode attention grows from 21 us per layer
after 512 tokens to 69 us after 4096 (`attn_decode_gqa`, 8 MB of K/V, ~120 GB/s). The grid was 2 KV heads x 64
splits = 128 blocks of one block per SM (~230 registers), 64 positions each. `attn_decode_bandwidth` with
`ATT_SPLITS` 128 / 64 / 32 / 16 (us, 1 / 2 query tokens; includes attn_merge):

| Positions | 128 | 64 | 32 | 16 |
|---|---|---|---|---|
| 4096 | 91 / 157 | 64 / 107 | 58.5 / 81 | 62 / 79 |
| 16384 | 200 / 293 | 153 / 255 | 151 / 216 | 179 / 237 |
| 32768 | 308 / 474 | 257 / 425 | 272 / 386 | 331 / 440 |

Change: `ATT_SPLITS` 64 -> 32. A split count chosen per token count (64 for 1 token, 32 for 2) was faster still
for 1 token at 32K, but `attn_prefill_matches_decode` requires one batched launch to give the same bits as single
launches, which keeps greedy output equal with MTP on and off; the split count changes the summation order, so it
must be the same for every token count. Rejected for that reason.

Bench (`--pp 4096 --tg 128 --reps 3`, 3 interleaved rounds, load 5.4-14.3; 32K one round, `--reps 2`):

| | a992641 r1 / r2 / r3 | 144b43b r1 / r2 / r3 |
|---|---|---|
| decode after 4K, mtp 0 | 95.2 / 95.9 / 92.4 (mean 94.5) | 96.4 / 96.0 / 96.0 (mean 96.1) |
| decode after 4K, mtp 1 | 95.3 / 97.8 / 97.4 (mean 96.8) | 101.1 / 99.3 / 98.6 (mean 99.7) |
| 32K decode, mtp 0 / 1 | 70.1 / 77.1 (load 8.5-13.4) | 80.2 / 81.1 (load 3.6-4.4) |

MTP acceptance in this bench 58.0% -> 60.0-60.7% (the attention sum order changed). Parity (`--llama-numerics
--fixed-experts`, interleaved): 633/640 both (the parity prompts stay below the 2048 positions where the GQA
kernel starts). Full test suite passes.

### 2026-10-09 #72 — GQA attention threshold 2048 -> 1024 (rejected, no commit)

With 32 splits (#71) `attn_decode_bandwidth` puts the GQA kernel at or below the one-head kernel from ~768
positions (us, 1 / 2 tokens, g1 vs gqa: 768 27.7/49.2 vs 26.5/43.2; 1024 31.0/52.9 vs 29.6/45.3; 1536 41.5/75.9 vs
36.6/54.5; 512 21.2/35.3 vs 26.0/40.3). `ATT_GQA_MIN` 1024 passed the full suite, but `bench --pp 1280 --tg 128
--reps 3` (3 interleaved rounds, load 4.3-16.8) showed no gain: mtp 0 99.4 -> 98.9, mtp 1 102.9 -> 103.5 tok/s
(means, spread 0.5-2.3). Kept 2048. Redo only with a profile showing attention at 1-2K as a real cost.

### 2026-10-09 #73 — rms_norm with 512 threads (rejected for parity, no commit)

nsys (512 prompt, 256 decode tokens): `rms_norm` 4.47 us per call with 256 threads, 3.11 with 512, 2.98 with 1024
(~100 calls per token). With 512: full suite passes; bench (3 interleaved rounds, load 5.7-16.3) decode after 512
101.0-101.3 -> 102.3-102.7 tok/s (+1.3%), after 4K and at 32K inside the noise. Parity (`--llama-numerics
--fixed-experts`, interleaved, load 1.8-2.2): 633 -> 628/640. The block sum order changes with the block size, and
that moves 5 greedy tokens. Reverted per the parity rule. Redo only together with a change that restores parity, or
keep 256 threads for the sum and use the extra threads only for the scaling and quantization.

### 2026-10-09 #74 — rms_norm with 512 threads, same bits (90c1b9d); flaky prefill test fixed (abcfdaf)

Follow-up to #73: 512 threads per block, but only the first 256 accumulate the sum of squares with the 256-thread
stride, so the sum (and the block_sum tree, where the extra warps add zeros) is bit-identical; all 512 threads
scale. nsys: 3.06 us per call (256 threads: 4.47, plain 512: 3.11), so the gain was in the scaling phase.

Parity (`--llama-numerics --fixed-experts`, interleaved, load 8.1-14.3): 633/640 before and after. Bench (3
interleaved rounds, load 1.6-3.5, `--reps 3`):

| | ca47f5e r1 / r2 / r3 | 90c1b9d r1 / r2 / r3 |
|---|---|---|
| decode after 512, mtp 0 | 101.2 / 101.4 / 101.5 | 102.8 / 103.0 / 103.0 |
| decode after 4K, mtp 0 | 97.2 / 96.9 / 97.0 | 98.3 / 98.4 / 98.5 |
| decode after 512, mtp 1 | 113.4 / 113.0 / 113.1 | 115.7 / 114.3 / 115.1 |
| decode after 4K, mtp 1 | 105.1 / 105.7 / 105.8 | 105.3 / 106.8 / 107.6 (65.2% accepted) |
| 4K prefill | 2077 / 2076 / 2089 | 2067 / 2089 / 2090 |
| 32K decode mtp 0 / 1 | 79.3 / 85.7 | 81.6 / 84.9 |

Decode +1.5% after 512 and after 4K. With a quiet host, 4K decode with MTP is now 105-108, above the 100 target.

`prefill_matches_reference` failed at both ca47f5e and 90c1b9d with the same numbers (prefill error 1.62 above 1.5x
the decode error 0.82), although it passed at 144b43b. The test loads with a 512 MB expert budget and placed experts
from the routing profile in `~/.cache/rusting_strata`, which every bench run rewrites, so its decode yardstick
depended on what ran before. It now sets `STRATA_FIXED_EXPERTS` (layer-order placement); passes at both commits
with identical numbers.

### 2026-10-09 #75 — PTX cache pruning (5881050)

`~/.cache/rusting_strata` held 113 PTX files, 385 MB, one per kernels.cu edit, and nothing ever deleted them. A
cache hit now refreshes the file's mtime; a miss first deletes `.ptx` files not used for 7 days, then compiles.
Seven days keeps the PTX of every binary an interleaved bench alternates between, so no recompile thrash. Unit test
`prune_ptx_drops_only_stale_ptx` (old .ptx removed, recent .ptx and non-PTX files kept). Checked on the GPU: a hit
refreshed the mtime (09:34 -> 11:00), the run worked. No kernel or numeric change, so no parity or bench rerun.

### 2026-10-09 #76 — router_topk with map_hits fused: rejected

Folded the expert cache lookup (`map_hits`, one 32-thread launch per MoE layer) into `router_topk` and deleted
`map_hits`; extended `router_topk_matches_ops_top_k` to check slots and hit weights. Tests passed, parity 633/640
both. Bench vs 16ca272 (3 interleaved rounds, load 2.7-6.1, `--reps 3`):

| | 16ca272 r1 / r2 / r3 | fused r1 / r2 / r3 |
|---|---|---|
| decode after 512, mtp 0 | 103.4 / 103.2 / 102.4 | 103.4 / 102.6 / 101.6 |
| decode after 4K, mtp 0 | 98.6 / 98.3 / 98.4 | 98.5 / 98.1 / 97.8 |
| decode after 4K, mtp 1 | 105.8 / 105.0 / 106.2 | 105.2 / 105.6 / 103.1 |

No gain: inside a CUDA graph a dependent tiny launch costs well under 1 us, so removing 40 of them per token is
below the noise. Reverted (diff kept out of the tree). Fusing launch-bound kernels only pays when it also removes
memory traffic or a serial dependency with real work, so the rms_norm/router fusions in candidate 6 should be
judged by their kernel time, not by launch count.

### 2026-10-09 #77 — `--kv q8` KV cache option (d02ac75)

The spec's `--kv q8|f16`. With q8 a cache row holds hd int8 values, then hd/32 f16 scales (ggml q8_0, roundf);
`kv_store` quantizes one 32-value block per warp, and one loader (`ldkv`) gives the decode and tensor-core prefill
attention the same f16 words an f16 cache holds, so the attention math is unchanged. Refused together with
`--llama-numerics`. New test `tests/kv_q8.rs` (own process, since the module is compiled per KV format): q8 vs f16
attention rel err 1.6-2.1e-3, q8 prefill vs decode <= 7.7e-5, batched decode bit-identical; a planted scale-index
bug fails it (0.07).

Parity (load 3.6-4.8): default f16 with `--llama-numerics --fixed-experts` 633/640 (unchanged). Without
`--llama-numerics` (`--fixed-experts`): f16 629/640, q8 631/640, so q8 costs no measurable agreement.

| `--reps 3`, load 1.2-6.6 | f16 | q8 |
|---|---|---|
| KV at 4224 / 33280 tokens | 82 / 650 MB | 44 / 345 MB |
| resident experts at 32K, mtp 0 | 8428 | 8724 |
| 4K decode mtp 0 (r1/r2/r3) | 98.2 / 98.3 / 98.3 | 97.5 / 97.4 / 97.3 |
| 4K decode mtp 1 | 103.1 / 104.0 / 104.0 | 103.6 / 102.5 / 102.4 |
| 4K prefill | 2037 / 2068 / 2077 | 2022 / 2026 / 2034 |
| 32K decode mtp 0 (r1/r2) | 80.4 / 81.6 | 76.3 / 76.2 |
| 32K decode mtp 1 | 86.2 / 84.8 | 78.7 / 78.5 |
| 32K prefill | 1593 / 1608 | 1371 / 1372 |

q8 halves KV memory but is slower: the dequantize per value (byte extract, scale multiply, f16 pack) and the
separate 2-byte scale load cost more than the halved K/V bytes save, most visibly in the 32K prefill (-14%). So f16
stays the default, q8 is the low-VRAM option. Follow-up if q8 should win: dequantize K/V tiles once into shared
memory in prefill, and in decode convert q8 directly to f32 instead of round-tripping through f16 words.

### 2026-10-09 #78 — faster q8 KV dequant, KV format fixed per Gpu (e1d933c)

`ldkv` now makes f16 pairs from four int8 values with a byte permute under exponent 0x64, one f16x2 subtract (1152)
and one f16x2 multiply by the scale. int8 and the f16 scale are exact in f16, so the product rounds once and the words
equal the earlier `f2h((float)q * d)` (same test errors, q8 parity 631/640 both). `--kv q8`, interleaved, load
1.2-2.6:

| | d02ac75 r1 / r2 | e1d933c r1 / r2 |
|---|---|---|
| 32K decode mtp 0 | 75.9 / 76.2 | 78.3 / 78.2 |
| 32K prefill | 1381 / 1374 | 1406 / 1404 |
| 4K decode mtp 0 | 97.8 / 97.5 | 97.8 / 97.9 |

Still below f16 (81.6 / 1600 at 32K). Also from a commit security review: the KV format was read from the
environment separately for the kernel compile and for the cache size, so a library caller changing
`STRATA_KV_Q8` between the two could size an f16 cache as q8 and write out of bounds. `Gpu::new` now reads it once
into `Gpu::kv_q8`, sizes come from `Gpu::kv_words`, and `kv_store` refuses a q8 head_dim that is not a multiple of 32.

### 2026-10-09 #79 — rejected: 8-byte q8 KV loads

Tried `ldkv<PER, hd>` with head_dim at compile time and `uint2` loads where the row layout allows them. The q8 test
passed, but nsys at 32K (`--kv q8`, load 1.1-1.4) showed no gain, so it was reverted (no commit):

| per call | f16 | q8 (e1d933c) |
|---|---|---|
| attn_prefill_256 | 71.1 ms | 108.9 ms |
| attn_decode_gqa | 274 µs | 322 µs |

ptxas: q8 uses fewer registers than f16 (prefill 178 vs 232, decode 204 vs 227), 0 spills, so register pressure
is not why q8 is slower. Both prefill variants have a 528-byte stack frame (a local array, likely the `o` accumulator),
in f16 too; worth a look as a separate prefill item. Remaining q8 ideas: copy raw int8 tiles to shared memory and
dequantize there, or convert straight to f32 in decode.

### 2026-10-09 #80 — prefill attention accumulator back in registers (1fbf113)

ptxas showed `attn_prefill_256` with a 528-byte stack frame (272 at head_dim 128): the rescale and output loops
indexed `o[d]` without `#pragma unroll`, so the 512-byte accumulator lived in local memory. Unrolling them (and the
row loop of the output) gives 0 stack frame, 0 spills for both KV formats. nsys at 32K prefill, f16, alternating,
load 5-12: `attn_prefill_256` 69.35 / 69.77 ms per call vs 70.62 ms before (about -1.5%; the kernel is ~28% of
32K prefill, so end to end it is lost in noise). End-to-end benches this hour were too noisy to read (load 4-8 from
other work; the same binary gave 1576 and 1499 at 32K prefill).

Parity: 632/640 for both this commit and the previous binary (e1d933c) on rerun, where e1d933c gave 633 an hour
earlier, so the parity run itself moves by one token between runs; the `--llama-numerics` path does not use this
kernel. Tests all pass.

### 2026-10-09 #81 — parity drift traced to free VRAM (note, no code)

The 632 vs 633 in #80 is not a code change: two parity runs of one binary agree on the total but differ on five
prompts, and three `strata score` runs of one prompt give top-1 logprobs that differ by up to 0.1. The `plan:` line
shows why: resident experts were 3719, 3971 and (with `--vram-margin 3000`) 1170, because a game held ~4 GB of VRAM
during these runs. `--fixed-experts` fixes the order of placement but not the count, which comes from free VRAM, and
an expert on the CPU (integer q8_K dot, per-block float sum) does not round like the same expert in the GPU MMQ
kernel, so a different split gives slightly different logits. Expert types are not the cause: the only Q2_K/Q3_K
experts are in layer 40, the MTP layer, which `score` does not run. On an idle GPU the split is stable and parity
gave 633 every time this session. Parity and benchmark numbers taken while another program holds VRAM are not
comparable; `plan: ... resident experts` in the run log tells.

Recheck once VRAM use fell below 1.2 GB (load 4.0): 1fbf113 gave 633/640, so the gate holds. e1d933c run right after
gave 631 while other programs took VRAM again (2.0 GB used at the end, 7349 resident experts in a check run), which
fits the explanation above.

### 2026-10-09 #82 — input errors; MTP draft vocabulary cut judged not worth it (e5e27df, 4fa3297)

`strata score` took raw token ids without a range check: an id at or past n_vocab panicked in the embedding lookup
(`range start index 1407999998592 out of range`), and an id above u32::MAX was truncated silently. Both are now
errors (`token id 999999999 out of range (vocabulary has 248320 tokens)`, `bad id`). Context-full errors on the GPU
now say to raise `--ctx`. Probed and fine: missing file, non-GGUF file, truncated GGUF, empty prompt, `--ctx 16`,
`--ctx 40` with `--mtp 2` and with `--chunk 7`.

Also looked at a smaller draft LM head (only the first N vocabulary rows, like FR-Spec without a corpus): of the 640
llama.cpp reference tokens only 81.7% are below 32768, 86.6% below 65536 and 92.7% below 100000, since Qwen ids are
not ordered by frequency. Cutting to 64K rows would save ~0.65 ms per draft (~5% of an MTP step at 4K) but every
reference token past the cut is a sure miss, costing about as much acceptance (~62% now). Not built; a hot set
counted from generated tokens at run time could work, but needs its own measurement.

### 2026-10-09 #83 — serve request validation (9f4af2a)

Probed `strata serve --gpu` with malformed requests. Fine before: bad JSON, missing or non-array `messages`, a
prompt over `--ctx`, huge `max_tokens` (stops at the context). Fixed: an unknown role (`"wizard"`) was accepted and
written into the ChatML template verbatim, so a role string could inject template text; a user message without
`content` became an empty turn and the model made up a reply; `max_tokens` -5 or 2.5, `temperature` -1 or "x",
`top_k` -3 and `top_p` 5 were silently replaced by defaults; `max_tokens` 0 failed with "prompt of 13 tokens fills the
context". Each is now a 400 naming the field (`unknown role "wizard"`, `user message without content`,
`max_tokens must be an integer >= 1`, ...). Null still means "use the default", and an assistant turn may omit
content. tests/serve.rs covers each case; removing the role check fails it.

### 2026-10-10 #84 — v1.0.0: gate run, parity definition (5957785, release commit)

The owner asked to finish v1.0.0 and left the parity definition to the loop. Chosen: a token agrees when it matches
llama.cpp in either of its own run modes (token by token, the reference, or the context as one batch,
`target/llama-batch-argmax.json`). llama.cpp's two modes agree with each other on only 98.28% (#33), so the strict
99% gate was above llama.cpp's own consistency. `tools/parity.py` prints both numbers and gates on the second
(5957785); offline it reproduces #33 (631-637 of 640 on the #32 runs).

Gate run at 9f4af2a, all under the GPU lock (`docs/m6-results.md`):
- Strata, load 2.6-4.1: 4K prefill 2016.3 (2013.0-2019.8), decode after 512 100.7 (100.5-101.0), after 4K 96.0
  (95.0-96.6); MTP k = 1 109.9 / 96.7. 32K (load 21-24 from another loop's rustc): prefill 1540.8, decode 77.0,
  MTP 79.1.
- llama-bench (CUDA 5d8ccdf9d), load 4.3-7.5: pp512 1077.7, pp4096 1059.5, tg128 61.6; at d32768 pp512 808.6,
  tg128 50.4. Strata 1.90x prefill, 1.63x decode. Short-prompt prefill still 0.85x llama.cpp.
- Parity, either mode: `--llama-numerics --fixed-experts` 637 / 635 / 635 = **99.32%** (strict 634 / 633 / 632);
  default `--gpu` 633 / 634 / 631 = 98.85% (strict 628 / 630 / 625, load 4-11). Gate passed in parity mode.
- Found while testing the release: `tests/quant.rs` `rejects_unknown` still expected type 16 (IQ2_XXS) to fail,
  broken since IQ2_XXS support landed. Now checks IQ4_NL (20). All 47 tests pass.
- Release: version 1.0.0 in Cargo.toml, `strata --version`, README status and agreement row, local tag v1.0.0.

### 2026-10-15 #85 — backend seam (Part 1 of the Vulkan and architecture-table spec)

`src/backend.rs` puts a `Backend` trait between the model and CUDA: `Buf` (device address plus owner) replaces
`CudaSlice` in `gpu_model.rs`, `gpu_model/prefill.rs`, `expert_cache.rs` and `cpu_worker.rs`; the forward and MTP
graphs go through `Backend::graph`; the CPU mailbox comes from `Backend::host_mailbox`. cudarc is used only under
`src/cuda/`.
- Greedy output (64 tokens, MTP 0) byte-identical to v1.0.0. With `--mtp 1`, v1.0.0 itself is not reproducible run to
  run: across 3 runs (one with `STRATA_NO_GRAPH`) it gives two different texts after 185 bytes, and the new binary
  gives the same two texts. So MTP is compared as that set, not one run.
- Interleaved A/B against v1.0.0, 3 rounds, load 1.0-1.2: 4K prefill 2032.6 vs 2042.0 (v1 range 2017.8-2057.1),
  decode after 512 115.9 vs 115.4, after 4K 105.2 vs 106.1 tok/s (within 1%). `bench` now runs MTP by default, so
  decode is above the README ranges for both binaries.
- Parity `--gpu`: the reference caches were gone, so `target/parity-ref.json` was rebuilt with llama-server CUDA
  5d8ccdf9d (`--n-cpu-moe 20`). The batch-mode cache (`selfparity.py`, about 1 h) was not rebuilt, so only strict counts.
  On that reference, interleaved: v1.0.0 626 / 622, new 626 / 627 of 640.
- `cargo test --release`: 55 pass, 4 ignored, no warnings.
- `Backend::htod` syncs the stream after the copy. cudarc's `memcpy_htod` from a slice did not, and the final review
  suggested dropping the sync to match. Without it, greedy output degenerates into "The capital of France is Paris."
  repeated, and all tests still pass. So some caller relies on the upload having landed. The sync stays and its
  comment says why. The cause was not traced.

### 2026-10-16 #86 — architecture table (Part 2 of the Vulkan and architecture-table spec)

`src/config.rs` has one `match` on the architecture name. It sets `hybrid`, `q_gate`, `k_interleave` and tensor
`aliases`, and `ModelConfig::layers` lists a `LayerKind` (`Delta` or `Attn`) per block. Strata now loads `qwen3`,
`qwen3moe` and `qwen3next` next to `qwen35` / `qwen35moe`. Commits 38949be, 048f164, 414244d, b5fe15f, 4de35ba.
- Attention without the output gate: `qg == 0` in `attn_decode`, `attn_decode_gq` and `attn_prefill` means no gate.
  The CPU oracle, decode and prefill size the Q projection from `q_gate`.
- qwen3next: `Weights::new` splits the fused `ssm_ba` into `ssm_beta` / `ssm_alpha` rows at load time, and
  `deltanet` takes `interleave` (value head h reads key head h / (nv / nk), else h % nk).
- New quant type IQ4_NL (ggml id 20), CPU and CUDA. New pre-tokenizer `qwen2`.
- `tools/greedy24.py` (greedy 24 tokens against llama-server) and `parity.py --no-think` (see below).

Deviations from the spec, decided while planning:
1. `ssm_ba` is split at load time; no reorder kernel.
2. Legacy `ssm_in` files are not supported (unsloth ships `attn_qkv` + `attn_gate`).
3. No `qwen3next` alias for `post_attention_norm`; llama.cpp uses that name for both `qwen35` and `qwen3next`.
4. `router_topk` already handles 512 experts; only a config check was added.
5. New flag `k_interleave`.
6. New quant type IQ4_NL (the qwen3next shared-expert gate and up tensors use it).
7. New pre-tokenizer `qwen2`.

Gates. Reference: llama-server CUDA with `--n-cpu-moe 99 -c 4096 -np 1`. The auto-sized 52K context took 10 GB of
VRAM and strata ran out of memory next to it. Parity is strict (no batch-mode file). "Near-tie" counts prompts
whose first miss has a top-2 gap under 0.1 nats.

| Arch | File | Greedy 24 CPU / GPU | Parity `--gpu` (3 runs where listed) | Near-tie first misses |
|---|---|---|---|---|
| qwen3 | Qwen3-4B-Instruct-2507-Q4_K_M | pass / pass | 602/609 = 98.85% | 5 of 5 |
| qwen3moe | Qwen3-30B-A3B-Instruct-2507-UD-Q2_K_XL | fail (tie) / pass | `--no-think` 631, 631, 633 of 640 (98.59-98.91%) | 2, 1, 1 of 7 |
| qwen3next | Qwen3-Next-80B-A3B-Instruct-UD-IQ1_S | pass / pass | `--no-think` 636, 634, 635 of 640 (99.06-99.38%) | 1, 1, 3 of 4-6 |

- qwen3moe with the original prompts: 519/533 = 97.37% (11 prompts miss, 7 near-tie). `tools/prompts.json` uses the
  Qwen3.6 template with an empty `<think></think>` block. The Instruct-2507 models never saw it, so the first token
  after it is close to uniform. There, llama.cpp, the CPU oracle and the GPU differ by up to 2.5 nats, while in the
  middle of text the top-1 matches and alternatives agree within about 0.2 nats. `--no-think` drops the block.
  `--llama-numerics` made it worse (96.06%).
- qwen3moe CPU greedy: the oracle flips at token 5 on a 0.006 logprob tie (" Germany" against " the"). The oracle is
  f32; llama.cpp quantizes activations. No fix.
- qwen3moe `dequant_and_gemv_match_reference_for_every_type`: all 6 types in the file pass; only the final
  `seen.len() >= 10` coverage check fails (it is written for the default model). `real_file_config` likewise pins
  default-model values.
- The 80B parity run 2 first hit CUDA out of memory next to llama-server; it was rerun from the cached reference
  with the server stopped.

Default model: `cargo test --release` green after every task. Base (38949be) against Task 2, strict parity on a
fresh reference, 4 runs each: 628/629/627/627 against 628/627/626/627 of 640. Runs are not reproducible even with
`--fixed-experts`, so bit identity could not be checked.

Open items: README's supported-architecture list is not updated (the owner has uncommitted README edits). Legacy
`ssm_in` files are unsupported. The model tests that pin default-model values could skip on other architectures.
The GPU lock is now at `/home/vasylt/Code/rusting/.gpu.lock` (the old `/home/vasylt/Rusting` path is gone).

### 2026-10-16 #87 — validate DeltaNet metadata from the GGUF (65b43e7)

A security review of #86 found two places where a malformed file could crash the engine or the GPU:

- `ModelConfig::from_gguf` accepted `ssm.group_count = 0`. That gives a division by zero on the CPU, and the
  interleaved key head index `vh / (nv / nk)` divides by zero in the kernel when `nv < nk`. It now requires
  `group_count > 0`, `time_step_rank > 0`, `time_step_rank % group_count == 0` and
  `inner_size % time_step_rank == 0` whenever the model has DeltaNet layers.
- The `ssm_ba` split trusted the tensor shape. If it has a row count other than `2 * time_step_rank`, the beta and
  alpha tensors come out the wrong size, and the GPU kernels would read past them. `Weights::new` now checks the
  shape (and the group dims, because the split also runs for models without DeltaNet layers) before it splits.

Test: `rejects_bad_ssm_dims` (tests/gguf.rs) builds small qwen3next files in memory. It fails on the old code
(group_count 0 is accepted) and passes now. The full suite on the default model is green, at load 6.11. The
synthetic and split tests also pass on the qwen3moe and qwen3next files. `real_file_config` fails on those two
files, as documented in #86, because it pins the default model's values.

### 2026-10-16 #88 — tool calling in `serve` (f4d0ed5)

`/v1/chat/completions` now takes `tools` and `tool_choice` (`auto`, `required`, `none`, or a named function),
assistant messages with `tool_calls`, and `role: "tool"` messages. A reply that ends in tool calls comes back as
`message.tool_calls` (non-streamed) or as one `delta.tool_calls` chunk (streamed), with
`finish_reason: "tool_calls"`. New module `src/chat.rs` holds the templates; `src/serve.rs` only parses the request.

- Format: from the GGUF's `tokenizer.chat_template`. Qwen3.6 uses XML calls (`<function=f>` with
  `<parameter=k>` items). Qwen3-2507 4B/30B and Qwen3-Next use `<tool_call>{json}</tool_call>`; the 2507 templates
  also strip and re-wrap `<think>` in assistant history, the Next template does not. No Jinja engine (no new
  crates): each family is written out by hand and checked against llama.cpp.
- Byte identity: `tests/fixtures/chat/` holds the four local templates and llama.cpp's renderings
  (`llama-server --jinja --chat-template-file`, `POST /apply-template`, CPU build, made by
  `tools/gen_chat_fixtures.py`) of 21 requests. The requests cover tools with and without a system message, the
  full two-tool flow, a call with content, thinking on and in history, `reasoning_content`, argument value types,
  arguments given as an object, unicode and control characters, content parts, whitespace, two system messages,
  and a tool message first. `prompts_match_llama_cpp` renders all 84 and they match byte for byte. The one request
  llama.cpp rejects (arguments that are not JSON) is rejected too. Quirks matched: tools are rewritten to
  `{type, function: {name, description, parameters}}`, `tojson` uses Python separators and keeps key order,
  floats print like C `%g` (`1e-05`, `1.23457e+08`), XML parameter values print `True`/`False`/`None`.
- Parsing: calls must close the reply. If any block is malformed, the whole reply goes out as plain content with
  `finish_reason` `stop` or `length` (unit tests for both formats; never a 500). XML values are strings when
  the tool schema says `string`, else JSON if they parse, else strings. A stream sends text up to the
  `<tool_call>` tag as it comes and holds back a partial tag (`Hold`, tested at every split of a reply).
- E2E on the default model (`--gpu`, load 6.41): "weather and local time in Paris" with `get_weather` and
  `get_time` gives both calls (`{"city": "Paris"}` each, 51 tokens), streamed and not. With the two tool results
  added, the final answer is "The current weather in Paris is clear with a temperature of 21°C. The local time is
  14:05." (prompt 433 tokens, 333 cached: the history from the call turn is reused). With `tool_choice: "none"`
  the tools still go into the prompt (as llama.cpp renders them), and the model's call text comes back as content.
- Full suite green on the default model, load 2.20.

Not done: `tool_choice: "required"` and a named function do not force a call (no grammar-constrained decoding);
they act like `auto`. `parallel_tool_calls` is ignored. A literal `<tool_call>` inside the thinking part makes
the reply plain text. `chat` (the CLI) does not use tools. No speed or parity numbers changed (the prompt for
messages without tools is unchanged, `old_chat_prompt_unchanged`).

Follow-up (6874637): a security review found that the order-preserving JSON parser checked repeated keys with a
linear scan, so a body with many distinct keys parsed in quadratic time (a cheap denial of service under the body
size cap). A key index map makes it linear; `json_keeps_key_order_and_scales` parses 200k keys in 0.04 s. The old
code was not timed on that test. Full suite green, load 2.55.

### 2026-10-16 #89 — Vulkan plan; backend seam fixes for a second backend (7634b41, c865829)
- Branch `vulkan`, cut from `main` at 4b25af1. Plan: `docs/superpowers/plans/2026-10-16-vulkan.md` (V0 seam fixes,
  V1-V4 as in the spec, with the 48 KB shared-memory limit of the 3060 under Vulkan and CUDA-matched int8 numerics).
- c865829 (Task 0.1): `Backend: Send` (CudaGraph wrapped in a `Send` newtype), `Buf` fields private with `addr()` /
  `len()`, `GpuTensor::new`, op docs moved onto the trait, inherent `Gpu::zeros` renamed `zeros_slice`. Covers the
  four parked minors except the kernel tests through `dyn Backend` (Task 0.2).
- Gates: full suite green (load 7.7). `greedy24.py --extra=--gpu` identical to llama-server (load 7.9).
- Parity: `target/parity-ref.json` and `target/llama-batch-argmax.json` were gone, so the reference was rebuilt
  from llama-server CUDA (same flags as #84) and only the strict count is available. Interleaved control, load
  2.5-9.7: base 4b25af1 626 / 627 / 627, new 626 / 628 / 626 of 640 (one base run printed no result and was
  repeated). Same as the strict numbers of #84 (628 / 630 / 625). The 98.85% headline is "either llama.cpp mode",
  which needs the batch-argmax file (`tools/selfparity.py`, about 1 h); rebuild it before the V3 parity gate.

### 2026-10-16 #90 — Kernel tests on every backend; `--backend` / `--device` (e4db774, 4ba30c2)
- e4db774 (Task 0.2): `tests/cuda.rs` is now `tests/kernels.rs` and runs each kernel test for every backend in
  `common::backends()` through `dyn Backend`. `dequant`, `quant_xq` and `attn_decode_gq` moved onto the trait
  (`attn_decode` is a provided method). `tests/kv_q8.rs`, `tests/llama_numerics.rs` and `tests/gpu_model.rs` stay
  CUDA-only.
- 4ba30c2 (Task 0.3): `backend::open(kind, device)` and `Backend::name`; `Gpu::new(device)`. `--backend` and
  `--device` on the engine commands and `calibrate`. Calibration files are `strata-<model>-<backend>.json` for
  backends other than CUDA; CUDA keeps `strata-<model>.json`. `--backend vulkan` errors with "unknown backend" until
  V1.
- Gates: full suite green both times (load 4.7, 5.0). `generate --gpu`, 24 tokens on the default model: same output
  (md5) from the 7634b41 binary, the new binary, and the new binary with `--backend cuda --device 0` (load 5-15).
- Not verified: the ignored bandwidth kernel tests after the conversion; parity not re-run (no kernel or numerics
  change since the #89 control).

### 2026-10-16 #91 — Vulkan backend: device, memory, command flow, first kernels (2d5bed4, 36de601, 7bd9eeb)
- 2d5bed4 (Task 1.1): `src/vulkan` on `ash` 0.38 (Vulkan 1.3). One VkBuffer and allocation per `Buf` with a device
  address; host copies go through a 64 MB staging buffer in chunks; `mem_info` uses `VK_EXT_memory_budget` when the
  driver has it. `backend::open` falls back to Vulkan when CUDA does not load. Backend tests run on both backends,
  plus an 80 MB round trip through the staging chunks.
- 36de601 (Task 1.2): kernels are GLSL with `GL_EXT_buffer_reference` (pointers as push constants, one pipeline
  layout, no descriptor sets), compiled offline by `tools/gen_spirv.sh` and committed; `spirv_is_fresh` rebuilds
  them and compares. Pipelines need subgroup size 32 with full subgroups. Ops record into an open command buffer
  with a full barrier after each; `graph` records a reusable command buffer per key. `add` and `swiglu`.
- 7bd9eeb (Task 1.3): `rms_norm` (with the int8 copy), `quant_xq`, `argmax_rows`, `l2_norm`, `rope_neox`,
  `norm_gate`, `ssm_gates`, `conv_silu`. Reductions use the CUDA butterfly order. GLSL division and sqrt are a few
  ulp off on NVIDIA (the quant scales differed by 1 ulp from the CPU), so `fdiv`/`sqrt_` refine them with a
  `precise` fma (GLSL may split an fma that is not `precise`). Rope computes theta and sin/cos in f64 (Taylor after
  a pi/2 reduction; GLSL has no f64 pow/sin/cos), so the device needs `shaderFloat64`. New test
  `norm_rope_and_gate_ops_match_cpu` checks each op against `ops.rs` on both backends (quant bytes exactly).
- Gates: full suite green after each commit (load 4.2-21; another session was busy). CUDA greedy md5 unchanged
  after 2d5bed4 (the only commit touching shared code). Vulkan tests clean under `VK_LAYER_KHRONOS_validation`
  with `VK_LAYER_VALIDATE_SYNC=1` (no messages).
- Kernel tests not yet ported stay CUDA-only through `common::backends_except(&["vulkan"])`.
- Not verified: no on-disk VkPipelineCache (the NVIDIA driver caches shaders itself; first-run pipeline cost not
  measured). GLSL `exp`/`log` are not refined (Vulkan allows 3 + 2|x| ulp); revisit if the Task 1.9 greedy check
  differs from CUDA. RADV gates deferred: no AMD GPU on this machine.

### 2026-10-16 #92 — Vulkan dequant, GEMV and decode attention (2da2110, 12eedd2, a3b3d28)
- 2da2110 (Task 1.4): `deq.glsl` ports `QI<T>`/`deq8<T>` for all 17 weight types; the IQ grids and `kvalues_iq4nl`
  are generated into `tables.glsl` from `quant::tables` (`glsl_tables_are_fresh` checks it). `dequant` and the
  float GEMV (`gemv_f`, F32/BF16). Dequant is exact (max diff 0) for every type on the default and 80B models.
  SPIR-V with 16-bit loads needs `shaderInt16`, now required (validation reported it).
- 12eedd2 (Tasks 1.5, 1.6): integer GEMV (`gemv_q`) for every quantized type, x quantized by `quant_xq` into a
  4 MiB scratch buffer owned by `Vk` when the caller gives none; explicit `precise` fma order as CUDA. Vulkan GEMV
  output equals CUDA bit for bit ("0 of N differ") for all 17 types on the default and 80B models;
  `gemvn_matches_gemv` passes on Vulkan.
- a3b3d28 (Task 1.7): `kv_store` (f16, exact round-to-nearest-even conversion in integer code), `attn_decode`
  (one shader specialized per head_dim, group size and entry point; same splits, chunks and `warp_sum_scatter`
  order as CUDA) and `attn_merge`. `expf_` ports the PTX CUDA emits for `expf`; `silu`/`sigmoid` use it with IEEE
  division (changes conv_silu, norm_gate, ssm_gates and swiglu on Vulkan only). Decode attention is bit-identical
  to CUDA on all 5 test shapes through both kernels. `attn_prefill_matches_decode` runs on every backend: decode
  vs a CPU f64 softmax (rel err ~1e-7), `kv_store` vs `half` rounding, prefill skipped where missing. ATT_SPLITS_1
  moved to `backend.rs` (no CUDA behavior change, so no greedy/parity re-run).
- Gates: full suite green after each commit (load 3.4-11.5). Validation layers with sync validation: no messages.
- Not verified: `--llama-numerics` decode attention on Vulkan (returns not implemented); hd 512 shapes (no model
  uses them); the 4B and 30B models fail the GEMV tests' type-coverage asserts by design (also on CUDA, Goal 3
  item). RADV deferred: no AMD GPU.

### 2026-10-16 #93 — Vulkan V1: DeltaNet and dense decode end to end (197338b, eba809a)
- 197338b (Task 1.8): `deltanet` ports `deltanet_seq` (same lane layout, fma order, reductions; snapshots; both key
  head mappings). `deltanet_chunk_matches_steps` runs on every backend: output and state bit-identical to CUDA.
- eba809a: the first greedy check on a 2337-token prompt diverged from CUDA after 9 tokens. Cause: CUDA (LLVM fma
  contraction) fuses the per-chunk rescale into the first accumulate, `acc = fma(acc, cs, e * v)`; Vulkan did the
  multiply and the fma apart. Only splits with more than one 32-position chunk (contexts above 512) differed, and
  the test had none. The attention test gains nh 32 / nkv 8 (Qwen3-4B, group 4) and contexts of 1000 and 2100;
  all shapes now bit-identical to CUDA through both kernels.
- Task 1.9 gate, Qwen3-4B Q4_K_M, `generate --gpu --chunk 0 --mtp 0 -n 64`: identical output (md5) on CUDA and
  Vulkan for 5 prompts ("The capital of France is", a poem, `def fibonacci(n):`, a chat-template question, and a
  2337-token prompt that runs the GQA kernel). `GpuModel` needed no change.
- Decode speed, same model, 256 tokens, 3 interleaved runs (load 1.8-1.9): CUDA 86.4 tok/s (86.3-86.6), Vulkan
  47.6 tok/s (47.5-47.7). Speed is V4.
- Gates: full suite green (load 1.9-7.1). One earlier run had `prefill_cached_reuses_history` (CUDA only) fail
  with an error at `prefill_cached`; it passed alone and in two full reruns. Error text not captured; the commit
  under test touched only Vulkan code.
- Not verified: the default model and qwen3next on Vulkan (need MoE, V2); `--llama-numerics` attention on Vulkan;
  RADV deferred (no AMD GPU).

### 2026-10-16 #94 — Vulkan V2: MoE decode (44ce104, 46c8024, ab6908a, d5aa421)
- 44ce104 (Task 2.1): `router_topk`, `map_hits`, `moe_publish`, `moe_combine` and the CPU expert mailbox.
  `router_topk` is bit-identical to CUDA. The kernel side of the mailbox uses Vulkan memory model atomics (release
  store of the request flag, acquire loads of the done flag); a kernel spinning without them was never released,
  and after about 5 s the driver reports device lost, so `moe_combine` gives up after about 2.3 s and sets an
  error word that the host turns into an error.
- 46c8024 (Task 2.2): `moe_gemv_q`/`moe_gemv_f` (paired gate/up), bit-identical to CUDA. `moe_gemv_selects_experts`
  used expert id 200, past the 128 experts of Qwen3-30B (CUDA illegal address there); it now uses the last expert.
- d5aa421 (Task 2.3), three fixes found running the default model:
  - IQ tables as GLSL const arrays went to per-thread local memory on NVIDIA: IQ2_XS expert GEMVs took about 35 ms
    a call (about 200x CUDA), 0.7 tok/s. They now live in one device buffer read through buffer references, its
    address in spec constants 3 and 4; `tables.glsl` comes from `vulkan::iq_tables`. Then about 41 tok/s.
  - With many misses `moe_combine` timed out: the kernel saw the worker's done flag in host memory hundreds of ms
    late (stale L2 copy), even with an acquire load and with uncached host memory. The earlier mailbox test passed
    by timing. The mailbox is now device memory mapped through the BAR (ReBAR is on: memory type 5, 12 GB); the
    worker puts SeqCst fences around the done store (they also flush write-combining stores). Host reads of BAR
    memory are uncached, a V4 item.
  - `moe_publish` used `barrier()`, which orders only shared memory under the Vulkan memory model; a control
    barrier over buffer memory now precedes the request flag. No observed failure; formal fix.
  - `STRATA_EXPERT_MB` caps the expert slots. Placement depends on free VRAM, so without it two runs (or two
    backends) can place different experts and give different, equally valid, greedy text.
- ab6908a: with experts on the CPU, Vulkan greedy diverged from CUDA a few tokens in. Per-op checksums of layers
  0-3 found `ssm_gates`: Vulkan's softplus used `log(1 + x) * x / ((1 + x) - 1)`, a last-bit difference from CUDA's
  `log1pf`, which changes the DeltaNet decay from the second token on. `log1p_` now ports the PTX of `log1pf`. New
  test `ssm_gates_match_across_backends` (fails on the old shader). With all experts resident the difference had
  hidden behind matching text for 32 tokens.
- V2 gates (`--chunk 0 --mtp 0 --fixed-experts`, a frozen copy of the routing profile via `XDG_CACHE_HOME`, same
  `STRATA_EXPERT_MB`): greedy 64 tokens identical (md5) on CUDA and Vulkan for the default model (1 MB and 8000
  MB caps: 0 and 8189 resident experts), Qwen3-30B (1 and 5000 MB) and Qwen3-Next (1 and 5000 MB). Parity, default
  model, `--chunk 0`, cap 8000: Vulkan 632/640 = 98.75%, CUDA 632/640, identical per prompt (load 2.5-7.0).
- Decode, default model, 256 tokens, cap 8000 (79% hit), `--chunk 0 --mtp 0` (load 2.8-4.0): Vulkan 46.5 tok/s
  (45.9-46.6, 7 runs), CUDA 74.6 tok/s (73.5-75.0, 6 runs). In the first batch 3 of 6 runs printed no speed line;
  the exit status was not captured and 12 reruns were all clean. Not explained.
- CUDA unchanged (shared code: worker fences, the env cap): greedy md5 equal to the previous binary on 2 prompts
  x `--chunk 0`/`4096` with `--fixed-experts`. Parity `--gpu --backend cuda --fixed-experts`, 3 interleaved runs:
  previous 626/626/627, new 626/628/627 (load 2.1-4.6). That is below the 633 of #62 in both binaries, so the drop
  predates this work (routing profile and placement have moved since). `greedy24.py --extra="--gpu --mtp 0"`
  fails on CUDA in both binaries (3/3 each): " Paris." then a near-tie that placement decides.
- Full suite green after each commit (71 passed, load 7.7-8.8).
- Not verified: Vulkan prefill (`gemm`, `gemm_dense`, `conv_seq`, `attn_prefill`, `moe_combine_seq` are V3), so
  Vulkan parity runs only with `--chunk 0`; MTP on Vulkan; RADV deferred (no AMD GPU).

### 2026-10-16 #95 — Vulkan V3 Task 3.1: prefill sequence kernels (5b8d2a1)
- `conv_seq` + `conv_state`, `moe_combine_seq` and `attn_prefill` (head_dim 64, 128, 256) on Vulkan, all
  bit-identical to CUDA (the tests now require the same bits on every backend).
- `attn_prefill` uses `VK_KHR_cooperative_matrix` (16x16x16 f16 products, f32 sums, per subgroup). On the 3060 the
  driver's cooperative-matrix multiply gives the same bits as CUDA's `mma.sync.m16n8k16` when the k order is the
  same, so the prefill GEMMs (Tasks 3.2-3.4) can be bit-identical too. Two details needed for that: the softmax runs
  per lane in CUDA's fragment layout (the per-lane sums of l are reduced only at the end), and nvcc fuses `l *= c`
  into the first `l += p` of each block (checked in the PTX), so the shader writes that as an fma. The O
  accumulators are rescaled per row through two index matrices (row and column of each accumulator element),
  because the element layout of a cooperative matrix is the implementation's. Shared memory: 47.5 KB at head_dim
  256 (the Q tile is staged once in the K/V area and then kept in registers), under the 48 KB Vulkan limit.
- Without cooperative matrices (and with `--llama-numerics`) `attn_prefill` reports "not implemented"; the fallback
  waits for the RADV work.
- Load time (Task 2.3 asked for it; measured after #94): default model, cap 8000, warm page cache, load 3.6-4.6:
  Vulkan 2.49 s (2.45-2.55), CUDA 1.87 s (1.86-1.89).
- Full suite green (73 passed, load 2.3).
- Not verified: performance of `attn_prefill` (correctness only; head_dim 256 likely spills registers); end-to-end
  prefill still needs `gemm`/`gemm_dense` (Tasks 3.2-3.4).

### 2026-10-16 #96 — Vulkan V3 Tasks 3.2-3.4: prefill GEMM on cooperative matrices (9cb6597)
- `gemm` and `gemm_dense` on Vulkan. Types CUDA runs in f16 (`Gpu::mmq`: F32, BF16, IQ2_XS, IQ2_S) go through
  `gemm_f` (16x16x16 f16 cooperative matrices, port of `gemm_q`); the rest through `quant_q8` + `gemmi` (16x16x32
  int8 cooperative matrices, port of `gemmi_q`, exact int32 sums scaled per weight scale in f32).
- `gemm_matches_reference_for_every_type` now requires the same bits on every backend: Vulkan is bit-identical to
  CUDA for Q2_K..Q8_0, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, F32, BF16 (default
  model, Qwen3-Next 80B, Qwen3-30B, Qwen3-4B), also with `STRATA_LLAMA_NUMERICS` (there the reference rel-err assert
  fails on CUDA too, 4.8e-3, because the CPU reference does not model llama numerics; not a regression).
- Lesson: check float op order in the SASS, not the PTX. For the min types (q2_k, q4_k, q5_k) the PTX has
  `c*s` then `- sa*m` then `+ acc`, but ptxas contracts the first two into one FFMA
  (`ptxas -arch=sm_86` + `cuobjdump -sass`). Matching the PTX left 11k of 17k values off by an ulp.
- Throughput (`gemm_throughput`, 4096 positions, 3 runs, load 4.9-7.5), CUDA vs Vulkan ms:
  Q5_K 8192x2048 7.39-7.93 vs 21.01-21.17; Q5_K 4096x2048 3.76-3.92 vs 10.53-10.76; Q6_K 2048x4096 4.23-4.35 vs
  12.13-12.28; IQ2_XS 512x2048 (f16) 0.62-0.63 vs 2.00-2.01; IQ3_XXS 2048x512 0.46-0.59 vs 0.92-1.04. Vulkan ~2.7x
  slower: no double buffering, scalar epilogue from shared memory, byte-wide shared stores. For V4.
- The cooperative-matrix flag now needs both the f16 and the int8 shape.
- Full suite green (72 passed, 4 ignored, load 5.6; #95's "73" was a miscount).
- Not verified: `act` 1 and 2 and the dense path (`tiles == 0`) are not covered by a kernel test (Task 3.5 runs
  them end to end); fallbacks without matrix cores (Tasks 3.2/3.3 as written) wait for RADV.

### 2026-10-16 #97 — Vulkan V3 Task 3.5: prefill and MTP end to end (5469f79)
- No code change was needed: with the GEMMs of #96 the batched prefill, MTP verify and rollback run on Vulkan.
  `tests/gpu_model.rs` (7 tests: forward vs CPU, prefill vs reference, MTP draft, verify, rollback,
  `mtp_spec_matches_greedy` k = 1-3, cached prefill) now runs on every backend; all pass on Vulkan.
- Parity, default model, `--gpu --backend X`, 3 interleaved runs (load 1.1-2.8): Vulkan 628/629/630 (98.28%),
  CUDA 630/628/628 (98.23%). Both under the plan's 98.5%; the CUDA drop predates Vulkan (#94).
- Bit-for-bit check: with `--fixed-experts` the first comparison showed 2 prompts off. Cause: CUDA reads its
  calibration file (`short: 16`), Vulkan has none (default 32), so prompts of 27 and 31 tokens were batched on CUDA
  and stepped on Vulkan. With the same threshold (a temporary copy of the calibration for Vulkan, removed after)
  and `STRATA_EXPERT_MB=6000` for the same placement, argmax is equal on all 20 prompts, and `score` output is
  byte-identical on the two longest prompts (995 and 1367 tokens). Qwen3-4B (dense) and Qwen3-30B (MoE): `score`
  byte-identical with `--chunk 0` and `4096`.
- MTP: `generate -n 256 --fixed-experts STRATA_EXPERT_MB=6000`, 43-token prompt: Vulkan text equals CUDA (md5) for
  `--mtp 0`, `1`, `2`, with equal accepted counts (106/151, 134/248). `--mtp 0` vs `--mtp 1` text differs on both
  backends alike: loading the MTP layer changes the expert placement (also with `--fixed-experts`), so this plan
  gate is not a valid check; `mtp_spec_matches_greedy` (same instance) is.
- Speed, `bench --pp 512,4096 --tg 64 --reps 3 --mtp 0` (load 5.3-6.4): CUDA prefill 842 (751-890) / 1971
  (1952-1984), decode 96.0 / 92.3; Vulkan prefill 450 (433-461) / 704 (488-822), decode 58.6 / 55.2. V4 work.
- Full suite green (72 passed, 4 ignored, load 5.7-6.7).
- Not verified: Vulkan calibration (`strata calibrate --backend vulkan`) not run, so Vulkan defaults to
  `short: 32`, `mtp: 0`; RADV deferred. Another loop's process (RustingEngine-performance test binary, 281 MiB)
  held GPU memory outside the lock during these runs.

### 2026-10-16 #98 — Vulkan V4 Task 4.1: baseline against llama.cpp Vulkan (docs only)
- llama.cpp 5d8ccdf9d built with `-DGGML_VULKAN=ON` in `~/Clones/llama.cpp/build-vulkan`, no source edits. The
  machine has no Vulkan SDK: Vulkan-Headers v1.4.321 and SPIRV-Headers vulkan-sdk-1.4.321.0 were cloned into the
  session scratchpad and passed as `-DVulkan_INCLUDE_DIR`, `-DSPIRV-Headers_DIR` and `-DCMAKE_CXX_FLAGS=-I...`.
- `llama-bench -ngl 99 -fa 1 -r 3 -p 512,4096 -n 128` (`--n-cpu-moe 10` on the default model) vs
  `strata bench --gpu --backend vulkan --pp 512,4096 --tg 128 --reps 3 --mtp 0` at e3b0aea, one round each, load
  6.9-12.8:

  | model | | pp512 | pp4096 | tg128 (after 512 / 4096 for Strata) |
  |---|---|---|---|---|
  | Qwen3-4B Q4_K_M | llama.cpp Vulkan | 3149 ± 27 | 2825 ± 25 | 85.8 ± 2.8 |
  | | Strata Vulkan | 916 (897-927) | 774 (773-775) | 49.9 / 39.3 |
  | default (35B-A3B Q2_K_XL) | llama.cpp Vulkan | 357 ± 178 | 817 ± 45 | 48.5 ± 0.8 |
  | | Strata Vulkan | 454 (413-475) | 778 (673-831) | 59.7 / 56.3 |

- On the MoE default model Strata is ahead on decode and short prefill and even at 4K (its expert cache vs
  llama.cpp's 10 CPU layers). The dense 4B shows the kernel gap: 3.4x on prefill, 1.7x on decode. The 4B decode
  reads 2.3 GB per token, so 49.9 tok/s is ~115 GB/s of the 3060's 360: the GEMV is far from bandwidth.
- Not verified: llama.cpp tg is after an empty context while Strata's is after the prefill, so the decode columns
  are not the same test; one round only, at high load (another job ran during the 4B rows).

### 2026-10-16 #99 — Vulkan V4: weight decode with word loads (fc868d8)
- `qi8` (deq.glsl) read its 8 weight bytes with 8 byte loads. For Q8_0 and Q2_K-Q6_K it now loads two words
  (four 2-byte loads when the address is not 4-aligned) and unpacks bitwise as `QI<T>` in kernels.cu (a `vsub4`
  for Q3_K / Q6_K). Used by gemv_q, moe_gemv_q, gemmi and dequant. IQ types keep byte loads.
- Correctness: `dequant_and_gemv_match_reference_for_every_type` shows 0 bits differing from CUDA on all 11 types
  of the default model; `gemm_matches_reference_for_every_type` (bitwise assert) passes; full suite green (72
  passed, 4 ignored, load 2.3-4.6).
- A/B against f739a59, `bench --gpu --backend vulkan --pp 512 --tg 128 --reps 3 --mtp 0`, 3 interleaved rounds,
  load 2.1-8.7. Round 1 was disturbed (base decode min 14.9, and the new binary's default-model run printed no
  result: stderr was filtered out; 8 reruns all passed), so rounds 2-3 decide:

  | | base decode | new decode | base pp512 | new pp512 |
  |---|---|---|---|---|
  | Qwen3-4B | 46.3, 49.8 | 53.9, 54.3 (+12%) | 878, 904 | 945, 964 (+7%) |
  | default | 56.4, 59.0 | 61.8, 63.7 (+9%) | 453, 483 | 426, 498 |

- Not verified: the cause of the empty round-1 run (guess: another process took VRAM at load); parity not rerun
  (outputs bit-identical by the kernel tests). llama.cpp Vulkan 4B decode is still 85.8 (#98).

### 2026-10-16 #100 — Vulkan V4: integer GEMV on packed dot products (8e66d07)
- gemv_q and moe_gemv_q take the 8 weight values packed in two words (`qw` in deq.glsl) and sum the products with
  `dotPacked4x8EXT` (GL_EXT_integer_dot_product), as the CUDA kernels use `__dp4a`. `shaderIntegerDotProduct`
  (Vulkan 1.3 core) is enabled and required.
- `qw` is `qi8` + `pack4`: calling the word decoder `qw_k` directly makes glslc's optimizer fail ("does not
  dominate its use" / "SelectionMerge must appear in a block"; -O0 compiles). The driver folds the unpack/repack
  well enough for the gain below; a direct path is a later try.
- Correctness: every gemv and attention line of `tests/kernels.rs` shows 0 bits differing from CUDA; full suite
  green (72 passed, 4 ignored, load 1.2-2.9).
- A/B against 6c89560, `bench --gpu --backend vulkan --pp 512 --tg 128 --reps 3 --mtp 0`, 3 interleaved rounds,
  load 1.4-3.9: Qwen3-4B decode 53.7/54.2/53.5 -> 59.3/59.1/58.9 (+10%); default model 63.8/63.7/63.0 ->
  66.5/66.7/66.8 (+5%). pp512 unchanged (4B 949/964/950 vs 954/963/952; default 458/496/491 vs 498/492/492).
- Since #98 (Qwen3-4B, llama.cpp Vulkan tg 85.8): Strata Vulkan decode 49.9 -> 59.1. Not verified: parity
  (bit-identical kernels).

### 2026-10-16 #101 — tests skip default-model checks on other models (8882267)
- `real_file_config` (exact shapes of the default model) and the type-coverage asserts in
  `dequant_and_gemv_match_reference_for_every_type` / `gemvn_matches_gemv` now skip, with a SKIP note, unless
  STRATA_MODEL is the default model (`common::is_default_model`). The other checks in those tests still run.
- Verified: with Qwen3-4B, the three tests pass (load 3.9). Default-model suite: all targets green (load 6.4-8.7)
  except `gpu_model`, whose 7 tests failed with "allocate 2641920 bytes": Minecraft (java, 3 GB VRAM) was running
  outside the lock. That target is unaffected by this diff; rerun pending once the game exits.
- GPU A/B for the Q4_K scale word loads (k4_dm, saved as a patch) is paused for the same reason.

### 2026-10-16 #102 — serve: 400 for malformed requests (a58109b)
- `handle` answered a request that `read_request` rejects by closing the connection without a reply. It now sends
  400 with the reason (best effort) and still logs it. A bad Content-Length reads "bad Content-Length".
- Tests: every `read_request` error (empty, no path, cut-off headers, body too large, bad Content-Length, short
  body) in `rejects_bad_hosts_and_huge_headers`. Smoke test on the CPU path (Qwen3-4B, port 8197): a 9 MB
  Content-Length and `Content-Length: x` both got `400 Bad Request` JSON; `/v1/models` unchanged.
- Suite green except `gpu_model` (not run: Minecraft holds 3 GB VRAM, the default model OOMs; see #101). The
  change does not touch the GPU path.

### 2026-10-17 #103 — Vulkan V4: Q4_K scale word loads; fused x quant rejected (388e9a8)
- Tried first: gemv_q quantizing x itself in shared memory (no `quant_xq` dispatch or barrier) plus CUDA's rows
  per workgroup rule (rpw 2/8). Bits equal, but slower in every config of a 6-way sweep (load 1.3-3.2, rounds 2-3):
  base 4B 59.5 / default 67.3-67.6 tok/s decode; fused+rpw 54.0-56.3 / 65.6-65.8; fused only 57.3 / 66.2; rpw only
  ~57 / 62.4-65.6; rpw 4 49.5 / 56.6. Even both knobs off ran ~1-2% below base (larger shader). Reverted.
- Kept: `qw_k` reads the Q4_K d, dmin and 12 scale bytes as four words (k4_dm<true> in kernels.cu) instead of
  byte loads. 0 bits differ from CUDA in tests/kernels.rs; suite green (load 2.0).
- A/B against 0905725, `bench --gpu --backend vulkan --pp 512 --tg 128 --reps 3 --mtp 0`, 3 interleaved rounds,
  load 1.9-6.5: Qwen3-4B decode 59.0/58.9/58.8 -> 65.3/56.9*/65.2 (+11%; *one rep at 40.7, spread 40.7-65.2);
  default model 66.9/66.6/66.8 -> 67.7/67.3/67.7 (+1%). pp512 4B 949/954/950 -> 973/967/971 (+2%).
- `gemv_bandwidth` now runs on Vulkan too. On the default model Vulkan reaches about half of CUDA's GB/s on the
  dense K-quants (Q5_K attn_gate 143 vs 258, attn_qkv 167 vs 282) and ~70% on the MoE IQ types.
- Not verified: parity (bit-identical kernels).

### 2026-10-17 #104 — Vulkan V4: Q5_K scale word loads (dc524f2)
- Q5_K keeps d, dmin and the scales where Q4_K does and its 176-byte blocks are 4-aligned, so it takes the same
  word loads; `scale_min_k4` is gone. 0 bits differ; suite green (gpu_model OOMed once while another loop's
  process held 553 MiB; rerun 7/7, load 6.9).
- A/B against 388e9a8, same bench, load 3.3-8.2: default model decode 68.2/68.5/68.3 -> 69.7*/71.8/71.5 (+5%;
  *round 1 noisy, 67.8-70.9). Qwen3-4B (no Q5_K) 65.8/65.4/66.2 -> 43.9*/66.4/66.2, *round 1 hit by load (one
  rep 23.6). pp512 default 465/456/500 -> 444*/492/499.
- Not verified: parity (bit-identical kernels).

### 2026-10-17 #105 — Vulkan V4: gemv_q input count as a specialization constant (a766baf)
- gemv_q took nt (1-4 inputs) as a push constant, so `acc[4]` was indexed at run time. nt is now specialization
  constant 5 (a pipeline per count), as CUDA's `gemv_row_qn<T, SBB, NT>`. 0 bits differ; suite green (72 passed,
  load 1.5).
- A/B against dc524f2, same bench, 3 interleaved rounds, load 1.5-6.2: Qwen3-4B decode 66.4/65.8/65.7 ->
  78.6/78.6/78.7 (+19%); default model 71.3/71.3/71.3 -> 75.4/76.3/76.2 (+7%). pp512 unchanged.
- Tried first and dropped: `qw` calling `qw_k` directly for the word types (no unpack/repack): neutral (4B 66.2 vs
  66.2, default 71.5 vs 71.5). The glslc "does not dominate its use" failure behind #100's note is a glslang bug:
  `const bool WORDS = T == 8 || (T >= 10 && T <= 14)` on a spec constant becomes an OpPhi inside the first function
  using it, which a second function cannot use; a bit-mask form compiles.
- Qwen3-4B decode is now 78.7 against llama.cpp Vulkan 85.8 (#98). Not verified: parity (bit-identical kernels).

### 2026-10-17 #106 — Vulkan V4: gemv_f input count specialized too (1e27f0e)
- Same change for the F32/BF16 GEMV (router, ssm_alpha/beta on the default model). 0 bits differ; suite green
  (72 passed, load 2.8).
- A/B against a766baf, load 3.0-7.8: default model decode 76.3/76.5/76.4 -> 76.9/76.6/76.7 (+0.4%, every round);
  4B (no F32 GEMV) 79.1/79.2/79.0 vs 79.7/79.1/79.0. A first A/B was hit by another loop's process (CPU 900%):
  one 35B round at 40.7.

### 2026-10-17 #107 — Vulkan V4: no barriers between independent decode ops (850746e)
- Every Vulkan dispatch was followed by a full barrier. A timing-only hack with no barriers at all (wrong output)
  put the 4B decode ceiling at 78.6 -> 99.2 tok/s: barriers cost ~2.6 of 12.7 ms per token (~21 dispatches per
  layer). `Backend::batch(f)` now queues ops that do not read each other's output; Vulkan records them with no
  barrier between them and one after the batch (quant_xq keeps its barrier: the GEMV after it reads it). CUDA runs
  them as before. Decode batches q/k/v, q/k norm, q/k rope, dense gate/up, the four DeltaNet input GEMVs, and in
  MoE layers {router, shared gate_inp/gate/up}, {routed gate/up, shared swiglu}, {routed swiglu, shared down}.
- A/B against 1e27f0e, same bench, 3 interleaved rounds, load 2.5-10: Qwen3-4B decode 79.4/79.1/78.8 ->
  83.1/82.5/82.7 (+4.6%, llama.cpp Vulkan 85.8); default model 76.5/77.0/76.5 -> 83.9/83.6/83.2 (+9.2%). pp512
  unchanged. An attention/dense-only first version gave 4B +4% but the default model only +0.9%.
- `generate --gpu --mtp 0 -n 64` greedy text: CUDA and Vulkan byte-identical on both models (load 3.0). Suite
  green (72 passed). CUDA A/B on the default model (load 4.4-6.1): decode 101.8/101.7/101.9 vs 102.2/102.1/102.3,
  pp ~1005 both: no regression.
- Tried and dropped: quant_xq split into 2048-value workgroups (neutral, 4B 79.3 -> 79.6, default 76.8 -> 77.0);
  gemv_q super-block loop unrolled by 2 (slower, 4B 78.9 -> 73.7, default 76.2 -> 73.6).
- Not verified: parity (kernels are bit-identical, only the barrier placement changed).

### 2026-10-17 #108 — Vulkan V4: gemmi unrolled, scales loaded once per lane (3956997, 02a97b4)
- `STRATA_VK_PROF=1` (02a97b4) prints GPU time per kernel name (timestamps outside graphs). Qwen3-4B pp512: gemmi
  90%. Default model pp4096: gemmi 35.9%, gemm_f 35.8%, attn_prefill_256 13.5% (57.8 ms per call), deltanet 9.5%.
- The NVIDIA Vulkan driver did not unroll gemmi's small loops: `[[unroll]]` on all of them. When the int32
  accumulator has the mma.sync layout (checked in the shader with `subgroupAll`), each lane loads its 2 row and 4
  column scales once per 16/32-deep step instead of per element. Output bit-identical (`gemm_matches`, 0 bits).
- `gemm_throughput` (m=4096, 4B matrices; CUDA / Vulkan before / after): Q4_K attn_q 4.38 / 8.9 / 6.27 ms, Q4_K
  ffn_gate 9.96 / 21 / 14.7, Q6_K ffn_down 12.58 / 30.6 / ~17, Q6_K attn_v 1.44 / 3.5 / 1.96. Default model after:
  Q5_K attn_qkv 12.1 ms (11.3 TFLOP/s, CUDA 17.6), Q6_K ssm_out 6.4 (CUDA 4.4), IQ2_XS ffn_gate_exps on gemm_f
  1.98 ms (4.3 TFLOP/s, CUDA f16 path 0.66 ms / 13.1), IQ3_XXS ffn_down_exps 0.65 (CUDA 0.49).
- A/B against 850746e, 3 interleaved rounds, load 2.0-4.0: 4B pp512 903/977/978 -> 1251/1459/1458 (first round
  noisy), pp4096 783/812/813 -> 1099/1094/1094 (+35%); default model pp4096 802/904/846 -> 1008/1062/1038 (+20%),
  pp512 473/438/464 -> 507/504/516 (noisy). Decode unchanged (4B ~83, default ~83.5). llama.cpp Vulkan: 4B pp512
  3149, pp4096 2825; default pp4096 817 (Strata is ahead there), pp512 357 +-178.
- Breakdown on Q4_K attn_q: skipping the scale epilogue and all global loads leaves ~3.3 ms, so the epilogue still
  costs ~3 ms. The driver exposes int8 cooperative matrices only as 16x16x32 (and 16x8x32): KS=16 types (Q6_K, ...)
  multiply 16 zeros per 32. Neutral: i2f_small on its own (kept, as kernels.cu); `[[unroll]]` in gemm_f (IQ2_XS
  2.00 -> 1.98 ms, dropped).
- Suite green (72 passed). Not verified: parity (kernels bit-identical to before).

### 2026-10-17 #109 — Vulkan V4: wide loads and hardware f16 in the prefill GEMMs (0ac254c, fc52e3c)
- gemm_f (0ac254c): software `f2h` replaced by the hardware conversion (round to nearest even, as CUDA's: still bit
  for bit, checked on every type of the default model incl. IQ2_XS, F32, BF16), X read as vec4, smem stored as
  words. `gemm_throughput` IQ2_XS ffn_gate_exps 1.98 -> 0.53 ms (16 TFLOP/s; CUDA 0.66 ms).
- gemmi (fc52e3c): q8 X, scales and sums read as uvec4/uvec2 instead of single words (CUDA uses 16-byte cp.async).
  Q5_K attn_qkv 12.1 -> 9.3 ms (CUDA 7.8), Q6_K ssm_out 6.5 -> 5.1 (4.4), IQ3_XXS 0.65 -> 0.54 (0.49); 4B Q4_K
  attn_q 6.27 -> 5.15 (4.38), ffn_gate 14.7 -> 12.35 (9.96), Q6_K ffn_down ~17 -> 14.9 (12.6).
- Model A/B, `bench --pp 512,4096 --tg 32 --reps 3`, 3 interleaved rounds each. gemm_f vs fb5ce43 (load 2-7, r1
  disturbed by another GPU process): default model pp512 ~515 -> ~800, pp4096 ~1010 -> ~1220; 4B unchanged. gemmi
  vs 0ac254c (load 1.8-4.1): 4B pp512 1474 -> 1745 (+18%), pp4096 1111 -> 1240 (+12%); default pp512 908 -> 983
  (+8%), pp4096 ~1265 -> ~1375 (noisy). Decode unchanged. llama.cpp Vulkan: 4B pp512 3149, pp4096 2825; default
  pp4096 817.
- Default model greedy (`generate --mtp 0 -n 64`, 600-token prompt) CUDA and Vulkan byte-identical after gemm_f.
  Suite green (72) after each. Not verified: parity (kernels bit-identical).
- Tried and dropped: X for step kb + 1 loaded into registers during step kb's products in gemmi (neutral: 4B same,
  Q5_K 9.3 -> 9.8). Without the W decode gemmi takes 4.33 ms on Q4_K attn_q (vs 5.15) and 6.8 on Q5_K (vs 9.3).

### 2026-10-17 #110 — Vulkan V4: IQ grid signs per word, IQ4 word loads, llama.cpp comparison (05a310f, 774d565, 38db3ff)
- V4 comparison with llama.cpp Vulkan (build 5d8ccdf9d), RTX 3060, load 1.4-2.4, Strata at d60620e; llama-bench
  3 reps, Strata `bench --reps 3`:
  | model | llama.cpp pp512 / pp4096 / tg128 | Strata pp512 / pp4096 / tg128 |
  | Qwen3-4B Q4_K_M | 3120 +-57 / 2854 +-3 / 95.4 +-2.4 | 1722 (1687-1743) / 1218 / 82.7 |
  | default model (llama `--n-cpu-moe 10`) | 124 +-15 / 582 +-409 / 47.1 +-0.1 | 894 (869-907) / 923 (847-1028) / 78.5 |
  The goal (not slower than llama.cpp Vulkan on the same file) holds on the default model. On the 4B Strata is behind:
  prefill by ~1.8-2.3x (the int8 MMQ epilogue that keeps results bit-identical to CUDA, see #108), decode by ~13%,
  the same as Strata's CUDA decode on that file (82.0). iGPU numbers and the RADV gates: deferred, no AMD GPU here.
- 05a310f: `qw` for IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S negates the 4 grid bytes of a word with one bytewise subtract
  (`apply_signs`, as kernels.cu) instead of 8 sign selects and a pack; `iq_grid` is shared with `qi8`. gemv_bandwidth
  on the default model: IQ2_XS moe8 35.2 -> 23.9 us (CUDA 16.8), IQ3_XXS moe8 gate 27.6 -> 25.3 (19.4), down 31.2 ->
  29.2 (28.2). Decode A/B against d60620e, `bench --pp 512 --tg 128 --reps 3 --mtp 0`, 3 interleaved rounds, load
  1.0-2.7: default model 83.7/83.4/83.7 -> 85.4/86.2 (+2.5%; the first round had another session's GPU process
  running and is void), 4B unchanged (81.5/82.6/82.4 vs 83.2/82.5/80.7). Greedy `generate -n 64` CUDA == Vulkan on
  the 4B (mtp 0) and the default model (mtp 0 and 2). Kernel tests bit-identical on the default model, 30B, 80B, 4B.
- 774d565: IQ4_XS/IQ4_NL indices read with `ld8` instead of 8 byte loads: IQ4_XS moe8 40.5 -> 38.6 us interleaved
  (CUDA 31). Tried and dropped: `kvalues_iq4nl` picked from four constant words by selects instead of the table
  (47 us, slower).
- 38db3ff: `mtp_draft_matches_reference` failed (2 positions, 1 allowed) on every commit back to 7a65dfd. Its
  argument `512 << 20` is a VRAM margin, so the resident experts depend on free VRAM, and the Vulkan backend sees
  less while the CUDA context lives: the hidden states differed between backends. With no resident experts both
  backends match bit for bit, 9/9. Suite green (72 passed) after each commit.
- Rejected, not committed: residual add fused into `rms_norm_q` (`add_rms_norm_q`, bit-identical). Decode A/B 3
  rounds, load 2.2-4.3: 4B 77.3/81.6/82.2 vs 81.3/81.1/81.8, default 81.8/82.3/83.3 vs 82.6/81.2/83.1 (neutral);
  `STRATA_VK_PROF` 4B add+rms_norm 192.7 -> 206.3 ms over 256 tokens (the norm's loop stores and reloads x).
- Default model decode, CUDA 102.2 vs Vulkan ~84 (tg128 after pp512, all experts resident). Per call, Vulkan
  `STRATA_VK_PROF` vs CUDA nsys: gemv_q 21.7 us vs gemvn_q5_k 17.2, moe_gemv_q 34.5 vs moe_gemv_iq2_xs 28.7 /
  iq3_xxs 24.7, moe_combine 3.5 vs 10.1. gemv_bandwidth (us, CUDA / Vulkan): Q5_K attn_qkv 42.5 / 55.5, Q4_K output
  934 / 1115; Vulkan's small GEMVs carry ~4-5 us of quant_xq and dispatch overhead.
- Not verified: parity (kernels bit-identical to before).

### 2026-10-17 #111 — Vulkan V4: moe_gemv_q four super-blocks per step (4b32187)
- 4b32187: `moe_gemv_q` runs four super-blocks per loop step (`[[unroll]]` over a `sb_step` helper, as the
  `#pragma unroll 4` in kernels.cu), so their weight and x loads overlap; the sums keep their order (bit-identical).
  gemv_bandwidth on the default model, 3 interleaved rounds against 774d565, load 1.0-3.0 (us): IQ2_XS moe8
  24.4/24.0 -> 21.3/21.6/20.3, IQ3_XXS gate 25.3/25.5 -> 23.4/23.4/23.2, IQ3_XXS down 29.2/29.3 -> 30.5/30.5/30.0
  (+1 us), IQ4_XS 40.8/40.8 -> 41.8/41.8/40.4, Q3_K and Q2_K unchanged (the base's second round had another
  process on the GPU and is left out). Decode A/B, `bench --pp 512 --tg 128 --reps 3 --mtp 0`, default model, 3
  interleaved rounds, load 0.9-1.0: 84.8/86.3/85.6 -> 86.4/87.8/86.8 (+1.4%). Greedy `generate -n 64` CUDA ==
  Vulkan on the 4B (mtp 0) and the default model (mtp 0 and 2). Suite green (72 passed).
- Tried and dropped: the tail also unrolled (`if (s + j < nsb)` inside the unrolled step): IQ4_XS 41.8 vs 39.6 us,
  down still 30. The same unroll in the dense `gemv_q` (NT columns): Q8_0 eh_proj 33.3 -> 39.2 us, Q6_K attn_q 51 ->
  51-56, the rest and the 4B neutral (2 rounds, load 2.6-4.8).
- Not verified: parity (kernels bit-identical to before).

### 2026-10-17 #112 — Vulkan V4: weight loads without the alignment branch (e06cd8f, e6a4ecf)
- `ld8` (8 weight bytes per lane) tested the address for 4-alignment at run time and fell back to four 2-byte
  loads. e06cd8f: Q4_K, Q5_K and IQ4_XS (blocks of 144, 176, 136 bytes, every field 8-aligned) read the 8 bytes
  with one load. e6a4ecf: Q2_K (84, 4-aligned) with two word loads; the 2-aligned Q3_K, Q6_K, Q8_0 read the three
  words around the 8 bytes and shift them (ld8a2 in kernels.cu) without a branch (in Q8_0 the lanes of a subgroup
  differ in alignment). Results bit-identical (kernel tests on the default model, 30B, 80B, 4B).
- gemv_bandwidth, Vulkan, 3 interleaved rounds against the previous commit each (us; e06cd8f load 1.0-6.0,
  e6a4ecf load 1.2-9.4, one void row with another session's process on the GPU left out):
  | tensor | before | e06cd8f | e6a4ecf | CUDA (#110) |
  | Q5_K attn_qkv 8192x2048 | 55.5-56.5 | 42.0-42.4 | 42.3-42.5 | 42.5 |
  | Q5_K attn_output 2048x4096 | 33.7-34.8 | 25.5-26.5 | 25.5-25.9 | |
  | Q4_K output 248320x2048 | 1109-1139 | 879-895 | 875-885 | 934 |
  | IQ4_XS moe8 down | 39.2-40.2 | 36.6-37.0 | 36.9-37.2 | 31 |
  | Q6_K attn_q 8192x2048 | 51 | 51 | 47.9-48.3 | |
  | Q3_K moe8 down | 35.5 | 35.5 | 30.9-32.1 | |
  | Q2_K moe8 gate | 23.5 | 23.5 | 19.9 | |
  | 4B Q6_K output 151936x2560 | | 1069-1084 | 935-946 | |
  | 4B Q6_K ffn_down | | 78.2-78.5 | 72.6-72.8 | |
  Q8_0 unchanged (eh_proj 33.3).
- Decode A/B, `bench --pp 512 --tg 128 --reps 3 --mtp 0`, 3 interleaved rounds, load 0.8-1.7:
  e06cd8f: 4B 82.9/82.6/82.1 -> 90.3/89.9/89.5 (+9%), default 83.0/86.6/86.6 -> 93.4/93.9/94.6 (+10%).
  e6a4ecf: 4B 90.8/90.3/90.2 -> 92.9/92.0/91.7 (+2%), default 94.2/94.9/95.4 -> 95.2/95.7/95.2 (+0.6%).
  4B decode now 92 vs llama.cpp Vulkan 95.4 (#110), default model 95 vs 47.1; CUDA default model 102.
  Greedy `generate -n 64` CUDA == Vulkan on the 4B (mtp 0) and the default model (mtp 0 and 2) after each commit.
  Suite green (72 passed) after each.
- Dropped: `qw` taking `qw_k`'s packed words directly for the K types (skipping qi8's unpack and repack):
  spirv-opt fails ("does not dominate its use") on gemv_q and moe_gemv_q with any form tried.
- Not verified: parity (kernels bit-identical to before).

### 2026-10-17 #113 — Vulkan V4: gemv_q four super-blocks per step (9119ed2)
- gemv_q (decode GEMV, one subgroup per row) now runs four super-blocks per loop step through an `sb_step` helper,
  as `#pragma unroll 4` in CUDA's gemv_q and as moe_gemv_q in #111. Results bit-identical (kernel tests).
- gemv_bandwidth, Vulkan, 3 interleaved rounds against 02bf8f6 (us, load 1.0-8.0): 4B Q4_K attn_output 25.9 -> 24.6,
  attn_q 27.4 -> 25.4, ffn_gate 51.1 -> 48.7, Q4_K ffn_down 52.9 -> 51.5; default model Q4_K output 883 -> 848,
  Q5_K about -0.5; Q6_K and Q8_0 unchanged (an unroll before the #112 loads made Q8_0 33 -> 39).
- Decode A/B, `bench --pp 512 --tg 128 --reps 3 --mtp 0`, 3 interleaved rounds, load 1.7-2.6:
  4B 92.7/92.3/92.4 -> 94.6/94.6/94.4 (+2%), default model 93.8/95.7/96.1 -> 95.5/96.6/96.7 (+0.8%).
  4B decode 94.5 vs llama.cpp Vulkan 95.4 (#110).
- Greedy: the first greedy.sh run gave DIFF on the default model at mtp 2 (a near-tie, "mtp_head" vs "mtp"); six
  reruns of both binaries on both backends gave the same output, and a rerun of greedy.sh gave same on all three.
  Another session's process was on the GPU during the first run (expert residency depends on free VRAM, as #109's
  MTP test); greedy.sh now waits for an idle GPU before each generate. Suite green (72 passed).
- 4B Vulkan profile (pp16 tg256, `STRATA_VK_PROF=1 STRATA_NO_GRAPH=1`, load 4.6): gemv_q 81.5% (32.5 us/call),
  rms_norm 3.95, attn_decode 5.38, rope_neox 3.45, quant_xq 3.09, attn_merge 3.63, kv_store 2.44, swiglu 2.32,
  add 1.87 us/call.
- Not verified: parity (kernels bit-identical to before).

### 2026-10-17 #114 — Vulkan V4: quant_xq split, SwiGLU and RoPE fused (b191ca3, 0ed0067)
- b191ca3: quant_xq ran one 256-thread workgroup per token (one for 9728 values of the 4B's FFN); it now runs one
  subgroup per 256 values (the same lanes, scales and rounding; bit-identical). New `Backend::swiglu_gemv` (default
  swiglu then gemv, so CUDA is unchanged): Vulkan computes the dense FFN's SwiGLU inside that quant_xq.
- 0ed0067: new `Backend::rms_norm_rope` (default rms_norm then rope_neox): Vulkan's rms_norm kernel rotates the
  normed q and k heads itself, one stage fewer per attention layer. rope_neox's f64 sin / cos moved to `rope_cs` in
  common.glsl; a kernel test checks rms_norm_rope == rms_norm + rope_neox bit for bit.
- Decode A/B, `bench --pp 512 --tg 128 --reps 3 --mtp 0`, 3 interleaved rounds against the previous commit:
  SwiGLU fusion alone (on top of 9119ed2, quant_xq still one workgroup per token, load 1.0-6.5): 4B 94.0/93.6/93.6 ->
  93.0/92.8/93.8, no gain. With the quant_xq split, b191ca3 vs 9119ed2 (load 0.8-2.4): 4B 94.3/93.5/94.7 ->
  94.4/94.8/95.5 (+0.8%), default model 93.7/95.4/96.3 -> 95.1/96.0/97.0 (+0.9%). Fusion on vs off on top of the
  split (load 1.2-1.8): 4B 94.8/94.6 -> 95.1/94.7/94.9 (a third base round void: 83.5, spread 80.6-86.1), about +0.2%.
  0ed0067 vs b191ca3 (load 2.3-6.0): 4B 94.8/94.8/95.0 -> 96.0/96.1/96.6 (+1.4%), default model 94.7/95.6/95.7 ->
  95.8/96.1/96.3 (+0.7%).
- 4B decode now 96.2 vs llama.cpp Vulkan 95.4 (#110): the V4 target is met on both models (default model 96 vs 47.1).
  CUDA default model 102.
- Greedy `generate -n 64` CUDA == Vulkan on the 4B (mtp 0) and the default model (mtp 0 and 2) after each commit.
  Suite green (72 passed) after each.
- Not verified: parity (results bit-identical to before).

### 2026-10-17 #115 — Vulkan V4: gemmi weight decode overlapped (1034c40)

- llama.cpp Vulkan vs Strata Vulkan, 4B Q4_K_M, interleaved, 3 rounds of `-r 3` (load 3.2–4.0, before this
  change): tg128 llama.cpp 95.7/95.0/94.9, Strata 95.3/95.4/95.3, so decode is at parity. pp512 llama.cpp
  3059/3144/3128, Strata 1747/1746/1753: prefill is about 1.8x behind. Strata CUDA on the same file: pp512
  2109/2173, decode 81.1/81.5 (CUDA decode is now slower than Vulkan on 4B; not in V4 scope).
- 4B `gemm_throughput` before (ms, CUDA / Vulkan): Q4_K attn_q 4.2–4.5 / 5.15–5.24, Q4_K ffn_gate 10.1–10.4 /
  12.0, Q6_K ffn_down 12.7 / 14.9–15.0, Q6_K attn_v 1.45 / 1.71. The Vulkan gemmi decoded W after the X stores
  and then waited on it; CUDA decodes step kb + 1's weights in registers during step kb's MMA.
- Change: gemmi does the same. `ldw` decodes the 16 weights of the next step into a `uvec4` plus scale and min
  while the coopmat products run; they are stored after the products. After (Vulkan, load about 6): Q4_K attn_q
  4.82–5.06, Q4_K ffn_gate 11.6, Q6_K unchanged.
- Model A/B against 0ed0067, interleaved, 3 rounds of pp512 tg128 reps 3 (load 0.8–1.7): 4B prefill
  1792/1790/1790 -> 1802/1794/1778 (flat), 35B 909/956/996 -> 945/998/991 (+2%, base still warming). An
  earlier run (load 0.6–1.9) gave 4B 1761/1787/1789 -> 1823/1834/1836. Decode unchanged.
- Suite 72 passed; greedy CUDA vs Vulkan same on 4B mtp0, 35B mtp0 and mtp2 (a first greedy run reported DIFF on
  35B; manual reruns and a second full run were all same, so the first run was disturbed).
- Not verified: parity (the change is bit-identical by construction: same values, same order).

### 2026-10-17 #116 — Vulkan V4: attention prefill and decode loops unrolled (3f8c7d6, 20273b4)

- Profile (`STRATA_VK_PROF`, 4B, pp4096): gemmi 6.8 ms per call (53%), attn_prefill_128 37.7 ms per layer (42%). The
  CUDA kernel takes about 11 ms per layer (estimated from the CUDA pp512/pp4096 gap).
- 3f8c7d6: attn_prefill moves the Q, K and V tiles as uvec4 instead of one half per load (37.7 -> 32.6 ms). On the
  mma.sync accumulator layout (detected as in gemmi) the scores and rescales stay in registers as in CUDA instead of
  a round trip through shared memory. `[[unroll]]` on the fixed-bound loops took it to 11.9 ms: without it the
  score and output arrays were not kept in registers. A/B against 1034c40, interleaved, 3 rounds of
  `--pp 512,4096 --tg 32 --reps 3 --mtp 0` (load 1.1-3.7):

  | Model | pp512 before | pp512 after | pp4096 before | pp4096 after |
  |---|---|---|---|---|
  | 4B | 1590/1786/1815 | 1942/1928/1933 | 1268/1262/1273 | 1776/1801/1770 (+40%) |
  | Default model | 826/614/925 | 923/796/963 | 1310/1115/1301 | 1545/1471/1616 |

  (base rounds 1-2 of the default model had warm-up dips.)
- 20273b4: the same `[[unroll]]` in attn_decode, deltanet, deq.glsl, quant_x and gemv_q. A/B against 3f8c7d6
  (load 0.4-1.3): decode at 4096 tokens of context 4B 65.9/65.6/65.2 -> 82.1/82.5/82.2 (+25%), default model
  87.4/88.1/87.8 -> 92.5/92.2/92.7 (+5%). Decode at 512: 96.0 -> 97.0 (4B), 95.9 -> 96.3. Prefill unchanged.
- After both, against llama.cpp Vulkan (build 5d8ccdf9d) on the 4B (load 2.7-3.6): llama.cpp pp4096 2849 +-10,
  tg32 at depth 4096 79.6 +-1.1. Strata Vulkan pp4096 about 1790, tg32 at depth 4096 82-83. Strata CUDA: pp4096 2043,
  tg32 at depth 4096 69.5. Vulkan prefill is now 12% behind Strata CUDA. The gap to llama.cpp is in gemmi, whose
  int8 epilogue keeps the results bit-identical to CUDA.
- Tried and dropped:
  - gemmi X loads moved one step ahead into registers, as CUDA's cp.async double buffer: Q4_K attn_q 4.8 -> 6.6 ms
    and ffn_gate 11.3 -> 15.5 in `gemm_throughput` (register pressure).
  - `[[unroll]]` in gemm_f, gemv_f, quant_q8, router_topk and argmax_rows: A/B neutral on both models.
- Each commit: suite 72 passed; greedy `generate -n 64` CUDA == Vulkan on the 4B (mtp 0) and the default model
  (mtp 0 and 2); `attn_prefill_matches_decode` bit-identical across backends.
- Not verified: parity (results bit-identical to before).

### 2026-10-17 #117 — gemmi double-buffered X, Q6_K back under 128 registers (7be9129)

- New `STRATA_VK_STATS=1`: prints each pipeline's driver statistics. At 20273b4: gemmi Q4_K 128 registers, Q6_K 120,
  30.7 KB shared; attn_prefill_128 201 registers. CUDA (ptxas -v on the cached PTX): gemmi_q4_k 119, gemmi_q6_k 124,
  attn_prefill_128 128. So the dropped register X prefetch of #116 pushed gemmi past 128 registers, one workgroup per
  SM.
- 7be9129: X double-buffered in shared memory (the next step stored before the current products, like kernels.cu's
  cp.async). For KS = 16 types the zeros after each 16 values moved from the X rows to the W rows (written once), so
  X rows stay 20 words. Without that move Q6_K had 52 KB shared memory and the driver gave it 138 registers
  (ffn_down 14.8 -> 21 ms). After: Q4_K 128 registers / 35.9 KB, Q6_K 126 / 40 KB.
- `gemm_throughput` 4B, Vulkan (load 1.0-1.3): Q4_K attn_q 4.73-4.85 -> 4.67-4.68 ms, ffn_gate 11.32 -> 11.08,
  Q6_K ffn_down 14.8 -> 14.16-14.18, attn_v 1.72 -> 1.57. CUDA: 4.17-4.48, 9.85-10.09, 12.5-12.6, 1.44-1.46.
- A/B against 20273b4, 3 interleaved rounds (load 1.5-3.1), prefill tok/s:

  | | pp512 base | pp512 new | pp4096 base | pp4096 new |
  |---|---|---|---|---|
  | 4B | 1906/1956/1959 | 2035/2012/2017 | 1793/1798/1793 | 1857/1844/1841 (+3%) |
  | Default model | 746/1019/1023 | 875/1040/1044 | 1627/1641/1618 | 1863/1697/1695 |

  Decode unchanged (4B 97.2-98.2, default model 94.5-97.5 at 512).
- Tried and dropped:
  - gemm_f: subgroups whose 32 rows are all past `cnt` skip the products. Default model pp512 gemm_f 756 -> 748 ms in
    the profile, so the cost of small expert tiles is the weight decode, not the products.
  - gemmi: the row/column index matrices reloaded from shared memory where used instead of held through the loop:
    Q4_K stays at 128 registers, `gemm_throughput` a little slower.
- Suite 72 passed; greedy `generate -n 64` CUDA == Vulkan on the 4B (mtp 0) and the default model (mtp 0 and 2);
  `gemm_matches_reference_for_every_type` bit-identical across backends.
- Not verified: parity (results bit-identical to before).

### 2026-10-17 #118 — attn_prefill weights from registers, deltanet next-token prefetch (518e31c, dca523d)

- 518e31c: attn_prefill detects whether the f16 A matrix layout is mma.sync's (index matrices, as for the
  accumulator) and then writes the softmax weights into the A matrix elements directly, skipping the shared memory
  round trip, as kernels.cu. attn_prefill_128: 201 -> 168 registers (`STRATA_VK_STATS`), binary 202 -> 79 KB, so
  three workgroups per SM; 11.16 -> 10.17 ms per call at pp4096 (4B, profile).
  A/B against 7be9129, 3 interleaved rounds (load 1.1-5.4): 4B pp4096 1836/1851/1827 -> 1896/1918/1880 (+3%);
  4B pp512 and the default model within noise (one 4B pp512 round hit a load spike to 5.4).
- dca523d: deltanet loads k and q as vec4 and the next token's inputs during the current token, as deltanet_seq in
  kernels.cu. Default model pp512 profile: 912 -> 630 us per call (-31%, 164 -> 113 ms over 3 runs).
  A/B against 518e31c (load 2.2-4.4): default model pp4096 1657/1632/1616 -> 1675/1639/1666 (+1.5%); pp512
  798/980/987 -> 1001/876/1009 (noisy); 4B unchanged (no DeltaNet layers).
- Each commit: suite 72 passed; greedy `generate -n 64` CUDA == Vulkan on the 4B (mtp 0) and the default model
  (mtp 0 and 2); `attn_prefill_matches_decode` and `deltanet_chunk_matches_steps` bit-identical across backends.
- Not verified: parity (results bit-identical to before).

### 2026-10-17 #119 — gemmi W double buffer, attn_prefill hardware f16 (4a8ddf3, 21d44cb)

- 4a8ddf3: gemmi double-buffers W as well as X, so each step has one barrier instead of two (kernels.cu has two).
  The layout check's index matrices sit in X buffer 1 before its first use, so Q6_K stays at 48 KB of shared memory
  and two workgroups per SM. `gemm_throughput`: 4B Q4_K attn_q 4.76 -> 4.46 ms (CUDA 4.3-4.6), ffn_gate 11.0 ->
  10.77 (CUDA 10.0-10.3); default model Q5_K attn_qkv 8.09 -> 7.88; Q6_K unchanged (its KS = 16 products run 32 deep
  with 16 zeros, the only int8 shapes the driver offers). A/B against dca523d (load 1.2-1.7): 4B pp4096
  1850/1888/1882 -> 1896/1916/1900 (+1.5%); default model within noise.
- 21d44cb: attn_prefill converts Q and the register-fed softmax weights to f16 in hardware (bit for bit, also with q
  scaled by 24 in `attn_prefill_matches_decode`); the shared memory fallback keeps the software f2h, since the
  hardware conversion there raised the shader from 168 to 239 registers. 4B pp4096 profile: 9.32 -> 8.56 ms per
  call (CUDA attn_prefill_128 7.43, nsys). A/B against 4a8ddf3 (load 1.9-3.3): 4B pp4096 1956/1936/1929 ->
  1961/1939/1959; default model within noise (wide spreads, one base round disturbed).
- 4B pp4096 profile now: gemmi 1617 ms (CUDA gemmi_q4_k + gemmi_q6_k 1560), attn_prefill 308 ms (CUDA 267).
- Tried and dropped: gemm_f loading step kb + 1 into registers during step kb's products: 80 -> 123 registers
  (three workgroups per SM down to two), IQ2_XS ffn_gate_exps 0.54 -> 0.71 ms.
- ncu cannot read the performance counters here (ERR_NVGPUCTRPERM), so stall reasons are not measured.
- Each commit: suite 72 passed; greedy `generate -n 64` CUDA == Vulkan on the 4B (mtp 0) and the default model
  (mtp 0 and 2). Not verified: parity (results bit-identical to before).

### 2026-10-17 #120 — moe_gemv_q several rows per subgroup (3d145d9)

- Each subgroup of moe_gemv_q runs 2048 / cols rows, as CUDA's rows per warp, so the 512-column down experts get 4.
  CUDA's forced minimum of 2 is left out: 2 rows of 2048 columns measured 2.5% slower (IQ2_XS gate 21.2 -> 21.7 us).
- `gemv_bandwidth` A/B against 21d44cb, default model, clean rounds (load 0.4-1.9; two of six rounds ran 2.3x slower
  on both binaries, GPU busy, dropped): down IQ3_XXS 30.0 -> 27.4 us, IQ4_XS 37.1 -> 35.0, Q3_K 31.1 -> 29.9; gate
  experts unchanged.
- Decode A/B (`bench --pp 512 --tg 128 --mtp 0`, load 3.3-7.1): default model 96.3/97.0/97.2 -> 97.7/98.1/98.4
  (+1.3%); 4B 97.4/97.3/97.2 -> 97.1/97.3/96.9 (no MoE, noise).
- Suite 72 passed. Greedy: the first run under load gave a default-model mtp 0 DIFF; repeated twice each for CUDA,
  Vulkan and the base binary, all six outputs are identical. Expert placement moves with load (State: decode is not
  deterministic run to run), which matches the one-off. Not verified: parity (per-row results bit-identical).

### 2026-10-17 #121 — rms_norm and gemv_f keep four loads in flight (e0bd201, 36f529c)

- Default model decode profile (`STRATA_VK_PROF=1 STRATA_NO_GRAPH=1`, tg256): rms_norm 101 calls per token at 4.9 us,
  gemv_f 140 calls at 4.1 us. Both looped one load per iteration (rms_norm: 8 iterations of 256 for 2048 values).
- e0bd201: rms_norm's sum of squares and output loop take four elements per step (each lane's sum in the same order;
  the output values are loaded before any store, which may alias x). 4.9 -> 3.6 us per call, two interleaved runs.
- 36f529c: gemv_f runs four super-blocks per step, as gemv_q. 4.1 -> 3.1 us per call, two interleaved runs.
- Decode A/B against 3d145d9 (`bench --pp 512 --tg 128 --reps 3 --mtp 0`, load 3.3-6.3): 4B 97.7/97.9/98.3 ->
  99.5/99.3/99.0 (+1.3%); default model 80.0 (disturbed: 74.8-83.7)/97.7/98.7 -> 97.5/100.4/98.8.
- Tried and dropped (moe_gemv_q): weights read through `readonly` buffer references (gate IQ2_XS 21.0 -> 20.8 us,
  within noise); eight super-blocks per step (gate -1 to -2.5%, but IQ4_XS down +3.7%).
- moe_gemv_q in the decode profile is now 2.24 ms per token (was 2.42; CUDA ~2.1). `gemv_bandwidth` with x not
  pre-quantized adds a separate quant_xq dispatch on Vulkan only, so its moe numbers overstate the gap to CUDA.
- Each commit: suite 72 passed; greedy `generate -n 64` CUDA == Vulkan on the 4B (mtp 0) and the default model
  (mtp 0 and 2). Not verified: parity (results bit-identical to before).

### 2026-10-17 #122 — prefill without cooperative matrices runs token by token (b1d81a3)
- 4B prefill profile first (pp4096, Vulkan, load 3.8): gemmi 76.6%, attn_prefill 14.9%, rms_norm 3.1%, quant_q8
  3.1% (256 us per call, ~325 GB/s, already at bandwidth; sharing one quantized x among q/k/v and gate/up would save
  under 1%). No cheap lever there; the gap is gemmi (Questions for owner, #119).
- Spec gap: without `VK_KHR_cooperative_matrix` (RDNA2) `gemm`, `gemm_dense` and `attn_prefill` report "not
  implemented", so any prompt of 32+ tokens failed. New `Backend::batched_prefill` (Vulkan: the coop flag);
  without it `GpuModel::load` keeps no prefill chunk (as `--chunk 0`) and prints a note. `STRATA_VK_NO_COOP=1`
  forces it, for tests and for drivers with broken cooperative matrices.
- Check (`generate -n 64`, 80-line prompt, load 1.3-7.1): `STRATA_VK_NO_COOP=1` output equals Vulkan `--chunk 0`
  on 4B mtp 0 and default model mtp 0 and 1; Vulkan `--chunk 0` equals CUDA `--chunk 0` except one default-model
  mtp 0 run, and with `STRATA_EXPERT_MB=6000 --fixed-experts` CUDA x2, Vulkan x2 and no-coop give one md5.
- Full suite green (72 passed, load 1.6).
- Not verified: a real device without cooperative matrices (no AMD GPU here). The spec's dp4a tiled GEMM fallback
  is still not written; token-by-token prefill is slow (it is decode speed) but correct.

### 2026-10-17 #123 — prefill kernels without cooperative matrices (dbfcc55)
- Replaces #122's token-by-token fallback: `Backend::batched_prefill` is gone again. Without
  `VK_KHR_cooperative_matrix` (or with `STRATA_VK_NO_COOP=1`) Vulkan runs three new kernels:
  - `gemmi_dp`: gemmi with the int8 tile products done by `dotPacked4x8EXT` (GL_EXT_integer_dot_product). Int8
    sums are exact in int32 and the epilogue keeps its order, so output is bit-identical to coop gemmi and CUDA
    (kernel test, all MMQ types).
  - `gemm_f_dp`: gemm_f with `fma` over k in order. Rel err 0 against the CPU reference (IQ2_XS, F32, BF16); not
    bit-identical to mma.sync, whose f16 sum order is the hardware's.
  - `attn_prefill_dp`: one subgroup per query, keys one at a time with an online softmax, same f16 inputs and
    softmax weights as attn_prefill. Rel err against decode 2.2e-7 to 1.4e-6.
  - gemmi and gemm_f bodies moved to `.glsl`, with `.comp` wrappers setting `COOP` 1 or 0 (the build compiles only
    `*.comp`). The coop SPIR-V keeps its size and the coop kernel tests still match CUDA bit for bit.
- Kernel tests skip only the bit-identity checks for gemm_f and attn_prefill when `STRATA_VK_NO_COOP` is set;
  the error bounds still apply.
- Greedy (`generate -n 64`, `STRATA_EXPERT_MB=6000 --fixed-experts`, no-coop vs CUDA): 4B identical; default model
  differs at byte 69 (f16 sum order), so the gate there is parity.
- Parity (`tools/parity.py`, 640 tokens, interleaved, load 2.9-11.0): CUDA 628 / 629 / 629 / 627, Vulkan no-coop
  632 / 632 / 633 (98.75-98.91%). Two no-coop and one CUDA run died on a VRAM allocation while another project's
  test process held the GPU; those were rerun, not counted.
- No-coop speed on the 3060 (bench, 3 reps): 4B pp512 883.3 (873.9-888.0), pp4096 456.7 (456.0-457.6), tg 99.0 /
  84.3, load 4.7; default model pp512 311.1 (199.1-397.5, noisy), pp4096 785.7 (732.4-821.7), tg 88.3 / 79.4,
  load 2.8. Much slower than coop prefill (4B pp4096 ~1950); token by token would run at decode speed.
- Suite green (72 passed, load 5.1). One earlier run failed `rollback_matches_step` on an unwrap; it passed alone
  and in the rerun (not traced). Kernel tests with `STRATA_VK_NO_COOP=1`: 15 passed.
- Not verified: a real device without cooperative matrices (no AMD GPU here; RADV deferred). No tuning of the dp
  kernels (tile shape, attention keys per step).

### 2026-10-17 #124 — Vulkan `--llama-numerics` attention (12b5c4f)
- The last two `todo` paths on Vulkan are gone: `attn_decode_llama.comp` and `attn_prefill_llama.comp` port the
  CUDA kernels (one head per workgroup over `ATT_SPLITS_1` chunks of 6 positions, merged by `attn_merge`; prefill
  in f32 with chunks of 32 keys). `tests/llama_numerics.rs` now also runs them and asserts Vulkan == CUDA bit for
  bit (decode n_pos 7, 300, 511, 512, 1400 with gq_min 2048 and 0; prefill hd 64, 128, 256).
- The NVIDIA driver drops a float -> half -> float round trip: both `float(float16_t(x))` and
  `unpackHalf2x16(packHalf2x16(x))` compile to nothing, so the first port matched the f32 path (rel 3.33e-4 against
  CUDA). `common.glsl` gained `rh()`, rounding to f16 in integer ops; with it the kernels are bit-identical.
  (Coop attn_prefill gets its rounding from the f16 matrix types, which the driver keeps.)
- `attn_prefill_dp` (#123) had the same silent loss of its f16 rounding. Restoring it with `rh` lowered parity
  (629 / 632 / 631), so it now keeps Q and the softmax weights in f32 on purpose: 634 / 632 / 633 (load 2.3-8.3).
- Parity with `--llama-numerics`, `STRATA_EXPERT_MB=6000 --fixed-experts` (load 2.7-3.9): CUDA 628 / 628, Vulkan
  631 / 631, each reproducible run to run.
- Why CUDA and Vulkan scores differ on 6-7 of 20 prompts even with the same experts resident: the CUDA calibration
  file (`~/.cache/rusting_strata/strata-<model>.json`) sets `short: 16` and `mtp: 1`; there is no Vulkan one, so
  Vulkan uses 32 and 0. Prompts of 16-31 tokens therefore run batched on CUDA and token by token on Vulkan, and
  the MTP layer's experts change CUDA's resident set. With `--mtp 0 --chunk 0` on both, a 27-token prompt gives
  the same row-0 logprob (-0.8597) on CUDA, Vulkan batched and Vulkan token by token; CUDA batched gives -0.9346
  (a different top token). Not a bug; a fair comparison needs `--mtp 0` and equal `short` (or a Vulkan
  calibration file).
- Suite green (72 passed, load 1.6). Kernel tests with `STRATA_VK_NO_COOP=1`: 15 passed.
- Not verified: RADV / non-NVIDIA drivers (whether they also drop the f16 round trip; `rh` is correct either way).

### 2026-10-17 #125 — CUDA `rms_norm` keeps four loads in flight (eb3e696)

- Default-model decode profile at 729bfb2 (nsys with `--cuda-graph-trace node`): about 1062 kernels per token with
  about 0.85 us of gap each, so small launch-bound kernels are still the lever, not the big GEMVs.
- `rms_norm` unrolled four ways in both loops (sum of squares, scaled store): 2.96 to 2.42 us average per launch.
  Greedy output identical (4B and default model, CUDA, `--fixed-experts --mtp 0`), suite 72 passed (load 3.06).
  Decode change is within the bench spread; not measured on its own.
- Tried and reverted: SwiGLU computed inside the x quantization of the down GEMVs (CUDA `quant_x` with a `u` input;
  Vulkan `quant_xq` with `u` for the routed experts; MoE step unbatched so the fused ops stay ordered). Kernel tests
  bit-identical, greedy identical on both backends and models. Interleaved 3-run A/B against 729bfb2 (the base also
  lacks the rms_norm change; load 1.6-2.1): CUDA default 103.23 to 103.50, CUDA 4B 83.10 to 83.27 (fusion off above
  2048 columns: 4B's 9728-wide down GEMV was 1% slower fused, every block recomputes the SwiGLU), Vulkan default
  100.63 to 99.87 (-0.8%: the MoE step loses the overlap between the shared down GEMV and the routed SwiGLU),
  Vulkan 4B 98.70 to 99.10 (dense path unchanged, noise). Net gain at most 0.2% on CUDA, a loss on Vulkan.
- Parity (CUDA default, interleaved 3 runs, load 2.5-5.0): 729bfb2 626/627/626, eb3e696 627/627/627; two more
  eb3e696 runs 629, 627. No change from rms_norm. Plain default mode scored 629/640 at #69; 633/640 is the
  `--llama-numerics --fixed-experts` baseline, not run here.

### 2026-10-17 #126 — `router_topk` does the expert cache lookup (1a7efa3)

- `map_hits` (one tiny launch per MoE layer: slots = table[ids], hit weights) folded into the end of `router_topk`,
  CUDA and Vulkan; prefill passes no table. 40 launches fewer per decoded token on the default model.
- Kernel tests extended (`router_topk_matches_ops_top_k` checks slots and hit weights; the mailbox test uploads
  them), suite 72 passed. Greedy output identical to 1d89d02 (4B and default model, CUDA and Vulkan,
  `--fixed-experts --mtp 0`).
- Decode, default model, `bench --pp 512 --tg 128 --mtp 0`, interleaved 5 runs against 1d89d02 (load 4.4-7.5):
  CUDA 103.78 (102.7-104.1) to 104.08 (103.7-104.4), +0.3%; Vulkan 100.82 (100.7-100.9) to 101.60
  (101.5-101.7), +0.8%, every pair. A first 3-run A/B at load 6-10 was too noisy to read.
- Not verified: parity (routing arithmetic unchanged, greedy output identical).

### 2026-10-17 #127 — `moe_route`: decode routing and mailbox publish in one launch (3d75f5b)

- Decode now routes with `moe_route`: one block, warp i routes tokens i, i + 8, ... (top-k, then the cache lookup
  from #126), and the same block writes the CPU mailbox (`moe_publish`'s job, now removed). Prefill keeps
  `router_topk` (one warp per token, no lookup). Top-k body shared: CUDA `topk_warp`, GLSL `topk.glsl`. Another 40
  launches fewer per decoded token on the default model.
- `moe_mailbox_round_trip` now drives `moe_route` from router logits and checks ids, slots and hit weights. Kernel
  tests and suite 72 passed. Greedy output identical to a6ead70 (4B and default model, CUDA and Vulkan, `--mtp 0`),
  and with `--mtp 1` (default model, both backends; the two-token verify path).
- Decode, default model, interleaved 5 runs against a6ead70 (load 2.4-6.0): CUDA 104.34 (103.9-104.5) to 104.96
  (104.9-105.0), +0.6%, every pair; Vulkan 100.48 (98.8-101.5) to 101.88 (100.2-102.6), +1.4%, 4 of 5 pairs.
- Not verified: parity (routing arithmetic unchanged, greedy output identical); more than 8 tokens per decode step
  (the warp loop's second pass; decode steps have at most 4).

### 2026-10-17 #128 — `moe_route`: no system fence on layers with no miss, CUDA (49c2aba)

- nsys (fixed experts): `moe_route` 10.2-10.7 us per call, 6.3 us without `__threadfence_system`. The fence is
  ~4 us x 40 layers per token. Fencing from thread 0 only: no gain (10.9 us).
- When no routed expert missed, the CUDA kernel now writes NTOK (still read by `routing()`), raises a new skip flag
  (mailbox word 2) and returns: no h copy, no fence, no request flag. The CPU worker leaves a layer on its skip flag
  or on a newer seq. The default bench has 100% expert hits, so every layer takes this path.
- First version returned before writing NTOK; `routing()` then sized the read by a stale count and
  `mtp_spec_matches_greedy` panicked in `ExpertCache::record` (index out of range). Fixed before commit.
- Vulkan, same protocol in the shader: 101.83 vs 101.85 (8 pairs, load 2.4-3.0), so Vulkan keeps the request flag.
  One Vulkan bench of that build exited without a decode line (load 3.6); 14 more runs passed and it did not repeat.
  Cause unknown; the committed Vulkan path only adds the skip-flag poll in the worker.
- Decode, default model, interleaved against f57e215: CUDA 104.97 (104.9-105.0) to 106.87 (106.8-106.9), +1.8%,
  every pair, load 2.4-2.7 (5 pairs before the NTOK fix: 105.0 to 107.0). Vulkan 102.6 to 102.5 (3 pairs, noise).
- Greedy output identical to f57e215 (4B and default model, CUDA and Vulkan, `--mtp 0` with `STRATA_EXPERT_MB=6000`,
  so miss and no-miss layers mix) and with `--mtp 1` (both backends). Suite 72 passed. `moe_mailbox_round_trip`
  checks a no-miss round raises only the skip flag on CUDA.
- Not verified: parity (routing arithmetic unchanged, greedy identical).

### 2026-10-17 #129 — Decode: residual add in `rms_norm_q`, q/k l2 norms in `conv_silu` (0c8aa99)

- `rms_norm_q` takes an optional residual and adds it into x before the norm, so the mixer output add before the
  post-norm is gone (one launch per layer). `conv_silu` now runs 128-channel blocks and l2-normalizes the first
  `nn` of them: the DeltaNet q and k heads of a one-token step (one launch per DeltaNet layer; only when the heads
  are 128 wide; multi-token steps keep `conv_seq` + `l2_norm`). 70 fewer launches per token on the default model.
  Both on CUDA and Vulkan; new tests check both give the same bits as the separate kernels.
- Greedy output identical to c931cfa (4B and default model, CUDA and Vulkan, `--mtp 0` and `--mtp 1`). Suite 73
  passed.
- Decode, default model, interleaved 5 runs against c931cfa (load 0.7-1.6): CUDA 106.7 (106.7-106.7) to 107.0
  (106.9-107.1), +0.3%, pairs 2-5 (pair 1 was a cold start: 98.3 vs 101.1); Vulkan 102.42 (102.4-102.5) to 102.76
  (102.6-102.9), +0.3%, every pair. Smaller than the launch count suggested.
- A first A/B of the add fusion alone was void: `physics_bench` from RustingEngine held the GPU at 100% for about
  an hour under its own lock file (`/tmp/rusting-gpu.lock`), and both binaries ran at half speed (CUDA ~50.3,
  Vulkan ~39.6). The loop now waits for no other GPU compute process before an A/B.
- Not verified: parity (same bits as before by the new tests and greedy output); 4B decode speed.

### 2026-10-17 #130 — Decode: `ssm_gates` in `deltanet` (c31cd6f)

- `deltanet` takes `dt_bias` and `a`; when set, it applies the `ssm_gates` sigmoid and softplus to beta and g as it
  loads them. Decode drops the `ssm_gates` launch (30 per token on the default model); prefill keeps it. CUDA and
  Vulkan; new test `deltanet_gates_match_ssm_gates` checks the same bits as `ssm_gates` then `deltanet` on both.
- Greedy output identical to 9489929 (4B and default model, CUDA and Vulkan, `--mtp 0` and `--mtp 1`). Suite 74
  passed.
- Decode, default model, interleaved 5 runs against 9489929 (load 2.5-3.0): CUDA 107.12 (107.0-107.2) to 107.50
  (107.4-107.6), +0.35%, every pair; Vulkan 102.68 (102.4-102.9) to 102.84 (102.7-103.0), +0.15%, 4 of 5 pairs
  ahead or equal.
- Not verified: parity (same bits by test and greedy output); 4B decode speed (no DeltaNet layers there).

### 2026-10-17 #131 — Routed SwiGLU inside the down `moe_gemv`: slower, reverted (no code commit)

- Tried `moe_swiglu_gemv`: CUDA's per-block `quant_x` of the down `moe_gemv` read g and u and quantized
  silu(g) * u; Vulkan ran its `quant_xq` with u (as the dense `swiglu_gemv`). Saves the routed `swiglu` launch (40
  per token). Same bits as `swiglu` then `moe_gemv` (test) and same greedy output.
- Decode, default model, interleaved 5 runs against c31cd6f (load 3.1-4.5): CUDA 107.16 (106.4-107.5) to 106.48
  (106.2-106.6), -0.6%, every pair; Vulkan 102.30 (100.3-103.0) to 102.20 (101.7-102.4), 4 of 5 pairs behind.
  On CUDA every block of every slot recomputes the SiLU of its expert's whole input (`n_ff_exp` `expf`s), which
  costs more than the launch. Reverted, as #125 did for the dense down GEMV; patch kept out of the tree.

### 2026-10-17 #132 — One-token GEMVs on the `gemv` kernel; same-shape GEMV pairs in one launch (5a78b2c)

- `gemvq` with one token (every decode GEMV on the normed "h") now launches the `gemv` kernel, which takes the
  pre-quantized x, instead of `gemvn`. Same math (`gemv_row_q` vs `gemv_row_qn<1>`), but `gemvn` carries the
  nt = 2..4 branches and their registers. New `gemvq2`: two same-shape GEMVs on one x; CUDA runs both in one
  launch (grid.z = 2, weight and output byte offsets), so decode pairs DeltaNet beta/alpha, the shared expert
  gate/up and the dense gate/up. Same bits as two `gemvq` (test); greedy md5 unchanged on both models and both
  backends, MTP 1 too.
- First try paired through `moe_gemv` (one slot of a zero id at stride 0): CUDA 107.14 to 107.52 (+0.35%, every
  pair, load 3.4-4.0), 4B 82.8 to 85.1; Vulkan slower (default model 102.44 to 102.18, 4B 98.43 to 97.3, load
  2.2-3.2), so Vulkan keeps two launches (`gemvq2`'s default). Then paired as `gemvn` grid.z = 2: CUDA 107.48 to
  106.36 (every pair behind, load 4.8-5.8), 4B 82.97 to 83.8; that pointed at `gemvn` itself, hence the switch.
- Final, interleaved against c31cd6f, CUDA only (Vulkan runs the same launches as before): default model 106.83
  (106.1-107.5) to 107.73 (107.0-108.1), +0.8%, 5 of 6 pairs ahead, one tie; 4B 82.9 (82.5-83.1) to 84.07
  (84.0-84.1), +1.4%. Load 1.1 at the start, then 8-12 from another process (both binaries alike).
- Not verified: Vulkan bench after the final change (no Vulkan code path changed); MTP verify (nt = 2) speed, where
  pairs go through `gemvn` grid.z = 2; why the `moe_gemv` pairing gave more on the 4B (85.1 vs 84.1, different
  load).

### 2026-10-29 #133 — GEMV pairs of different row counts in one launch (65df19b)

- `gemv`/`gemvn` take a trailing `rows2`: with grid.z = 2 the second matrix may have fewer rows (the launch is sized
  for the larger one, `gemvq2` swaps the pair if needed); only type and columns must match. Decode now also pairs
  DeltaNet qkv/z, attention q/k and the router with the shared expert's gate input. Vulkan unchanged (default
  `gemvq2`, two launches). New test: a half-height matrix paired with a full one, same bits as two `gemvq`. Greedy
  md5 unchanged on both models and both backends, MTP 1 too; 74 tests pass.
- First version wrapped the offsets in `if (blockIdx.z) { ... }`: every gemv kernel got slower (nsys: gemv_q6_k
  14 to 18.4 us, LM head gemv_q4_k 909 to 987 us), default model 108.0 to 102.3. Unconditional
  `w += blockIdx.z * w2` plus `if (blockIdx.z) rows = rows2` costs nothing measurable.
- Interleaved against 5a78b2c, CUDA, load 0.9-1.1, GPU idle: default model 108.25 (108.2-108.3) to 110.52
  (110.5-110.6), +2.1%, 6 of 6 pairs ahead; 4B 84.03 (84.0-84.1) to 83.90 (83.9-83.9), -0.15%, 3 of 3 behind by
  0.1 (the q/k pair; small enough to keep).
- Not verified: Vulkan bench (no Vulkan path changed); MTP verify (nt = 2) speed.

### 2026-10-29 #134 — `rms_norm_rope` in one CUDA kernel (cececa4)

- CUDA ran `rms_norm_rope` as the trait default (`rms_norm` then `rope_neox`): two launches each for q and k in
  every attention layer. New `rms_norm_rope` kernel: one block per head normalizes it (same sum order as
  `rms_norm`) and rotates it (same f64 angle as `rope_neox`, now per head instead of per token, which is the same
  latency since blocks run side by side). The existing kernel test checks it against the two-kernel path bit for
  bit; greedy md5 unchanged on both models and both backends, MTP 1 too; 74 tests pass.
- Interleaved against 65df19b, CUDA, load 1.6-2.1, GPU idle: default model 110.37 (109.9-110.6) to 110.93
  (110.3-110.9), +0.5%, 6 of 6 pairs ahead (10 attention layers); 4B 83.87 (83.8-84.0) to 85.17 (85.1-85.3),
  +1.55%, 3 of 3 (36 attention layers).
- Not verified: Vulkan (unchanged, has its own fused kernel since #114); prefill (the same call, one launch for
  all tokens, not timed).

### 2026-10-29 #135 — `kv_store` inside the k head's `rms_norm_rope` (CUDA) (bb2410d)

- New `Backend::rms_norm_rope_kv`: CUDA's `rms_norm_rope` kernel optionally stores the rotated k head and its v row
  into the KV cache (`kv_put`, so f16 and q8 alike), one launch fewer per attention layer. Vulkan runs
  `rms_norm_rope` then `kv_store` with a barrier between them inside the q/k batch (same barrier count as
  before). Kernel test: output and both caches equal `rms_norm_rope` then `kv_store` bit for bit. Greedy md5
  unchanged on both models and both backends, MTP 1 too, and with `STRATA_KV_Q8`; 74 tests pass.
- Interleaved against cececa4, load 1.4-3.7, GPU idle: CUDA default model 110.85 (110.7-111.0) to 111.03
  (110.9-111.2), +0.16%, 6 of 6 pairs ahead; CUDA 4B 85.23 (85.2-85.3) to 85.30 (85.3), 3 of 3 ahead or equal.
  Vulkan (no kernel change) default model 102.43 to 102.57, 4B 98.80 to 98.77: noise.
- Small, but consistent on CUDA and it removes a kernel launch; kept.
- Not verified: prefill (it still calls `kv_store`).

### 2026-10-29 #136 — Dense FFN residual added by the next norm; unrolled norm residual path (b7a8f9e)

- A dense layer (4B) no longer launches `add(x, ye)`: `layer` returns the pending FFN output and the next layer's
  `attn_norm` (or the output norm) adds it through `rms_norm_q`'s `res` path (the MTP head adds it itself, if
  dense). Same sum order, same bits: greedy md5 unchanged on both models and both backends, MTP 1 and
  `STRATA_KV_Q8` too; 74 tests pass.
- First try alone, against bb2410d (load 1.2-3.3): 4B CUDA 85.33 to 85.0, Vulkan 98.87 to 98.5 (-0.4%, every
  pair). The `res` path loaded one value per step (the plain path four), costing more than the `add` launch.
- With that path unrolled to four loads per step on CUDA and Vulkan (it also serves the post-attention norm of
  every layer of both models), interleaved against bb2410d, load 1.1-1.7, GPU idle:
  CUDA default model 110.87 (110.8-111.0) to 111.18 (110.8-111.4), +0.3%, 5 of 6 pairs ahead, one tie;
  CUDA 4B 85.27 (85.2-85.3) to 85.50 (85.5), +0.3%, 3 of 3; Vulkan default model 102.57 (102.5-102.6) to 103.10
  (102.9-103.3), +0.5%, 3 of 3; Vulkan 4B 98.73 (98.7-98.8) to 99.70 (99.5-100.0), +1.0%, 3 of 3.
- Not verified: prefill (separate path, unchanged); which half of the change gives the default-model gain (it has
  no dense layer, so it is the unrolled post-norm).

### 2026-10-29 #137 — Vulkan: SwiGLU inside the down GEMV's x quantization (cfe0773)

- `Backend::swiglu_q` replaces `swiglu_gemv`. By default it is the plain `swiglu` and returns 0, so the GEMV
  quantizes x itself. Vulkan instead quantizes silu(g) * u in `quant_xq` and returns that xq. This now covers the
  routed and shared expert down GEMVs as well as the dense one. Vulkan drops one dispatch before each down GEMV.
- CUDA tried first and was slower, so CUDA keeps the old path:
  - `quant_xq` with a `u` argument, interleaved against b7a8f9e: 35B 111.08 → 110.97, 4B 85.53 → 83.77. The 4B
    loss came from one block per 9728-value row.
  - Rows split into 2048-value blocks: 35B 111.45 → 111.08, 4B 85.47 → 84.90.
  - The GEMV's per-block quantization from L2 is cheaper than a separate launch that writes xq, even with the
    swiglu launch removed.
- Final, interleaved against b7a8f9e (load 1.7–5.0):

  | Backend | Model | Before | After | Change |
  |---|---|---|---|---|
  | CUDA | 35B (×6) | 111.33 | 111.33 | 0 |
  | CUDA | 4B (×3) | 85.50 | 85.50 | 0 |
  | Vulkan | 35B (×3) | 103.03 (102.9–103.1) | 103.70 (103.6–103.8) | +0.65% |
  | Vulkan | 4B (×3) | 99.37 (99.2–99.5) | 99.63 (99.4–99.8) | +0.26% |

  The first Vulkan run (with the CUDA change) gave the same gain: +0.5% and +0.2%.
- Greedy md5 unchanged on both backends for 4B and 35B, mtp 0 and 1, and `STRATA_KV_Q8`.
- New test: `swiglu_q` matches `swiglu` (when it returns 0) or `swiglu` + `quant_xq` byte for byte (Vulkan). 74 tests pass.
- Not verified: prefill (it does not use this path), models with F32/BF16 down weights (they keep the plain swiglu).

### 2026-10-29 #138 — CUDA: q and k rms_norm_rope in one launch (200a5cc)

- `Backend::rms_norm_rope_qk` runs `rms_norm_rope` of the q heads and then `rms_norm_rope_kv` of the k heads. On CUDA
  this is one `rms_norm_rope` launch: the first `qrows` blocks take the q heads (from their own x, weights and output,
  with no cache store). Vulkan keeps the default (two calls in one batch). One launch fewer per attention layer.
- Interleaved against cfe0773 (load 2.1–2.4, GPU idle):

  | Backend | Model | Before | After | Change |
  |---|---|---|---|---|
  | CUDA | 35B (×6) | 111.20 (110.9–111.4) | 111.80 (111.4–112.0) | +0.54% |
  | CUDA | 4B (×3) | 85.50 (85.4–85.6) | 86.93 (86.8–87.0) | +1.67% |

  Every pair ahead. The 4B has attention in all 36 layers, the 35B only in every fourth.
- Greedy md5 unchanged on both backends for 4B and 35B, mtp 0 and 1, and `STRATA_KV_Q8`.
- New test: `rms_norm_rope_qk` matches `rms_norm_rope` + `rms_norm_rope_kv` bit for bit (outputs and caches). 74 tests pass.
- Not verified: Vulkan bench (path unchanged there), prefill speed (same launch, more blocks).

### 2026-10-29 #139 — Vulkan: kv_store inside the rms_norm_rope dispatch (2eae27b)

- The `rms_norm` shader takes `v`, `kc` and `vc`. With `kc` set it stores its rotated output and v into the caches
  after a workgroup barrier (`memoryBarrierBuffer` + `barrier`), as `kv_store` does. `rms_norm_rope_kv` drops a
  barrier and the `kv_store` dispatch, so there is one dispatch fewer per attention layer.
- Two interleaved runs against 7fc208e:

  | Model | Load | Before | After | Change |
  |---|---|---|---|---|
  | Vulkan 35B (×4) | 2.9–3.5 | 104.00 (103.7–104.2) | 104.07 (103.1–104.5) | +0.07% |
  | Vulkan 35B (×4) | 1.3–1.9 | 104.05 (103.9–104.2) | 104.15 (104.0–104.3) | +0.1% |
  | Vulkan 4B (×4) | 2.3–2.8 | 99.88 (99.4–100.1) | 100.02 (99.7–100.6) | +0.14% |
  | Vulkan 4B (×4) | 1.1–1.3 | 99.62 (99.2–99.9) | 100.12 (100.0–100.2) | +0.5% |

  The gain is small. The 4B is ahead in 7 of 8 pairs and the 35B in 6 of 8.
- Greedy md5 unchanged on Vulkan for 4B and 35B, mtp 0 and 1, and `STRATA_KV_Q8` (CUDA untouched).
- The existing `rms_norm_rope_kv` and `rms_norm_rope_qk` tests compare the caches with `kv_store` byte for byte. 74 tests pass.
- Not verified: CUDA bench (unchanged), prefill speed.

### 2026-10-29 #140 — CUDA: one row per warp for dense GEMVs of 2049–8191 rows (9e3ccef)

- Why the 4B is slower on CUDA (86.9) than on Vulkan (100.1): in nsys (256 tokens, us per token) the CUDA GEMVs take
  ~9.8 ms, against ~8.1 ms for Vulkan's `gemv_q` (`STRATA_VK_PROF`). By shape (median us, default rows per warp →
  `STRATA_RPW=1`): q+k pair (4096 rows) 28.2 → 26.6, o and Q4_K down (2560 rows) 30.2 → 27.1, Q6_K down (2560 rows,
  ~236 GB/s) 86.4 → 78.4, gate+up pair (9728 rows) 90.0 → 90.8. `STRATA_RPW=4` was no better.
- `rpw` now gives 2 rows per warp only for 2048 rows and from 8192 rows (it was from 2048). A first try with "from 8192"
  alone cost the default model 0.6% (111.75 → 111.10): its 2048-row matrices (Q6_K and Q5_K out projections, the
  shared expert down, Q8_0) were ~5% slower with 1. The default model has no dense matrix between 2048 and 8192 rows,
  so its launches are unchanged.
- Interleaved against 846f721 (load 1.6–1.9):

  | Backend | Model | Before | After | Change |
  |---|---|---|---|---|
  | CUDA | 4B (×3) | 86.93 (86.8–87.1) | 90.20 (90.1–90.3) | +3.8% |
  | CUDA | 35B (×6) | 111.90 (111.5–112.0) | 111.67 (110.6–112.0) | noise (4 ties, one 110.6 outlier; same launches) |

- Greedy md5 unchanged on CUDA for 4B and 35B, mtp 0 and 1, and `STRATA_KV_Q8`. 74 tests pass.
- Not verified: other GPUs (the best split depends on the SM count), prefill (separate GEMM path), MTP nt = 2 on the 4B.

### 2026-10-29 #141 — CUDA: attn_merge writes the o GEMV's quantized input (99faaa5)

- After #140 the CUDA 4B's o GEMV (2560 x 4096) ran at ~220 GB/s against ~311 for the gate+up pair. Its input `att`
  was the only dense GEMV input not pre-quantized, so each of its 320 blocks quantized 4096 floats (16 KB from L2,
  ~5 MB per GEMV next to 5.9 MB of weights).
- `attn_merge` holds a head's outputs one per thread, 32 to a warp, so it also writes them as `quant_xq` does into
  `attq` (warp max, same rounding, bit-identical to the per-block quantization). New `Backend::attn_decode_q`
  returns the xq to use; its default (Vulkan, and CUDA under `--llama-numerics`, whose scales span 256) returns 0
  and the o GEMV quantizes as before. The kernels test checks the xq against `quant_xq` of the output.
- Interleaved against fd6adcd (load 0.5–0.9):

  | Backend | Model | Before | After | Change |
  |---|---|---|---|---|
  | CUDA | 4B (×3) | 90.33 (90.3–90.4) | 91.60 (91.6–91.6) | +1.4% |
  | CUDA | 35B (×6) | 111.87 (111.7–112.1) | 112.10 (112.1–112.1) | +0.2% |

- Greedy md5 unchanged on both backends for 4B and 35B, mtp 0 and 1, and `STRATA_KV_Q8`. 74 tests pass.
- Not verified: the same for Vulkan's o GEMV (it already reads `gemv_q` inputs differently), nt > 1 speed (MTP verify).

### 2026-10-29 #142 — Vulkan: attn_merge writes the o GEMV's quantized input (0334f06)

- Same change as #141 on Vulkan: `attn_merge.comp` quantizes its output (`subgroupMax` over the 32 values of a scale,
  `roundEven` as `quant_x`) into `attq` through a new `U8 xq` push constant, so the o GEMV skips its `quant_xq`
  dispatch and the barrier before it. Off under `--llama-numerics`.
- Interleaved against fd6adcd (#141 is CUDA only; load 3.0–5.5, falling):

  | Backend | Model | Before | After | Change |
  |---|---|---|---|---|
  | Vulkan | 4B (×4) | 100.25 (100.1–100.4) | 100.65 (100.5–100.8) | +0.4% (4 of 4 pairs) |
  | Vulkan | 35B (×4) | 104.05 (103.9–104.2) | 104.20 (104.1–104.3) | +0.1% (3 of 4) |

- Greedy md5 unchanged on both backends for 4B and 35B, mtp 0 and 1, and `STRATA_KV_Q8`. 74 tests pass (the
  kernels test compares the xq with `quant_xq` on both backends).
- Not verified: other GPUs, subgroup sizes other than 32 (the backend requires 32).

### 2026-10-29 #143 — CUDA: one attention entry point per head size (210d2fb)

- Vulkan's per-dispatch profile of the 4B (`STRATA_VK_PROF`, gemv names tagged with their workgroup count, load
  4.6) put its GEMVs at ~211 us per layer against ~216 for CUDA's (nsys): the 4B gap is not in the GEMVs. CUDA's
  attention was: `attn_decode` 35.7 us plus 4.8 us for the `attn_decode_gqa` launch that returns early, against
  ~23 us for Vulkan's two dispatches.
- Both CUDA kernels chose the head size with a switch, so each got the registers of hd 512: 122 per thread for
  `attn_decode` (2 blocks of 256 per SM), 227 for `attn_decode_gqa`. One entry point per head size
  (`attn_decode_128`, `attn_decode_gqa_128`, ...): 40 and 122 registers for hd 128. Same code per size.
- Interleaved against 3b72202 (load 1.5–1.7):

  | Backend | Model | Before | After | Change |
  |---|---|---|---|---|
  | CUDA | 35B (×6) | 111.95 (111.7–112.0) | 112.37 (112.3–112.6) | +0.4% (6 of 6 pairs) |
  | CUDA | 4B (×3) | 91.37 (91.2–91.5) | 95.80 (95.7–95.9) | +4.8% |

- Greedy md5 unchanged on both backends for 4B and 35B, mtp 0 and 1, and `STRATA_KV_Q8`. 74 tests pass.
- Not verified: speed from 2048 positions (the `attn_decode_gqa` path; the bench decodes at 512–640), head sizes
  other than 128 and 256 on hardware.
- Note: a first nsys and md5 run overlapped a CS2 session (the check in the command named only dota2); its 35B md5s
  differed even for the base binary. Those numbers were dropped and everything above was rerun with no game running.

### 2026-10-29 #144 — CUDA batch on two streams (7a3347f)

`Backend::batch` on CUDA now sends every second launch to a side stream, joined to the main stream by events at
both ends, so the captured graph runs batched launches as parallel branches (Vulkan's batch already drops the
barriers). Every op batched on CUDA is one launch, so alternating launches never splits an op. Target: the 4B's v
GEMV (~7-10 us per layer) ran after the q+k pair instead of beside it.

nsys (4B): v now starts 24.6 us into the 27 us q+k launch (median) — q+k's 1024 blocks occupy every SM, so v only
fills its tail, and q+k slows 26.6 -> 27.4 us from the shared bandwidth.

| CUDA decode, tg128 after pp512, load 2.1-2.7 | 7a3347f^ | 7a3347f |
|---|---|---|
| default model (6 interleaved pairs) | 111.37 (111.2-111.5) | 111.73 (111.4-111.9) |
| 4B (3 pairs) | 95.07 (95.0-95.2) | 94.97 (94.3-95.4) |

Greedy md5s unchanged (4B, default model mtp 0/1, kvq8, both backends). 74 tests pass (a first run failed
`prefill_cached_reuses_history` with an error while CS2 started on the GPU mid-run; it passes alone and in the
full rerun on an idle GPU). Not verified: prefill (no batch there) and other models' batches.

### 2026-10-30 #145 — CUDA SwiGLU quantized in one launch for the down GEMV: slower, reverted (no code commit)

- Tried `swiglu_xq` (CUDA `swiglu_q`, as Vulkan's `quant_xq` with u): one warp per 256 values computes silu(g) * u
  and writes it in the quant_xq layout, so the down GEMV (dense and routed) reads that xq instead of every block
  quantizing x itself (the 4B's 320 down blocks each quantize 9728 values). Kernel test bit-identical; greedy md5s
  unchanged (4B, default model mtp 0/1, kvq8, both backends).
- Decode, interleaved against 94f590c (load 0.1-0.9): default model 113.02 (113.0-113.1) to 112.08 (112.0-112.1),
  4B 96.40 (96.3-96.5) to 95.73 (95.7-95.8): -0.8% and -0.7%, every pair. The extra launch (plus its graph-node gap)
  costs more than the per-block quantization it saves, which overlaps the blocks' weight loads. With #125 (SwiGLU
  inside every block's quantization) and #131, the down GEMV's input path is now tried three ways.
- Two earlier A/B runs were dropped: one overlapped a Dota 2 start, one ran with other GPU load (spread 93-110).

### 2026-10-30 #146 — Decode launches only the attention kernel that works (5561aae)

The decode attention launched both kernels per attention layer (one head per block below ATT_GQA_MIN = 2048
positions, GQA from it on) and the step picked the one that works; the other was an empty launch (graph node on
CUDA, dispatch plus barrier on Vulkan). The kernels pick the regime per token, so the host can fix it when all
tokens of a forward fall on one side: `gq_min(pos, nt)` in gpu_model.rs gives i32::MAX (all below), 0 (all from
2048 on) or ATT_GQA_MIN (a forward that straddles it; both launched as before). The decode and MTP graph keys
carry it, so the graph is re-recorded when the regime changes. `attn_decode_q` now takes `gq_min` like
`attn_decode_gq`.

A/B interleaved against 94f590c (`bench --mtp 0`, tg128 after pp512, load 1.3-2.4):

| | base | #146 |
|---|---|---|
| CUDA 35B (x6) | 112.9 (112.7-113.1) | 113.1 (112.5-113.4), +0.2% |
| CUDA 4B (x3) | 96.5 (96.4-96.6) | 97.2 (97.2-97.3), +0.8% |
| Vulkan 35B (x4) | 104.4 (104.2-104.8) | 104.8 (104.5-104.9), +0.3% |
| Vulkan 4B (x4) | 101.1 (100.6-101.4) | 101.7 (101.3-102.2), +0.5% |

Greedy md5 unchanged on both backends (4B 64bab3bb, 35B mtp0 06c92e18, mtp1 04cf91e9, kvq8 cuda 04cf91e9 /
vulkan 06c92e18). A prompt of 2023 tokens, so decode crosses 2048 (35B mtp0 and mtp1, MTP verify straddling it),
and a 1931-token prompt with 160 decoded on the 4B: md5 identical to base on CUDA and Vulkan. Tests 74 passed.
Not verified: decode after 2048 positions in `bench` (the GQA regime skips the one-head kernel; md5 above covers
its correctness, not its speed).

### 2026-10-30 #147 — CUDA MoE combine fused with the next layer's rms_norm: no gain, reverted (no code commit)

nsys of the 35B CUDA decode at 5561aae (pp512 + tg128): per token about 81 rms_norm launches (3.0 us median) and
40 moe_combine launches (3.1 us), so fusing each MoE combine with the following norm (next layer's attn_norm, or
the output norm) would drop 40 launches. The norm moved to the end of `layer` (it takes the next norm weight;
dense layers call `rms_norm_q` with the FFN output as residual there, identical to before), and a new
`moe_combine_norm` backend op (default: the two launches) got a CUDA kernel. Greedy md5 unchanged in all
variants (4B 64bab3bb, 35B mtp0 06c92e18, mtp1 04cf91e9, kvq8 04cf91e9; a 2023-token prompt crossing 2048
identical too).

CUDA 35B decode, interleaved A/B against 5561aae (`bench --mtp 0`, tg128 after pp512):

| variant | base | fused | load |
|---|---|---|---|
| one block of 512 per token (combine, then norm) | 113.3 (x6, flat) | 111.7 (111.4-111.8), -1.4% | 1.3-1.5 |
| one block of 1024 per token | 113.5 (113.4-113.5, x3) | 112.8 (112.7-112.8), -0.6% | 1.0-1.3 |
| combine on n/256 blocks, the last block to finish (atomic counter per token) runs the norm | 113.3 (113.2-113.5, x4) | 113.5 (113.3-114.0), +0.2% | 1.0-1.1 |

One block cannot stream the k expert outputs fast enough; the last-block variant is within noise and adds a counter
buffer and a fence protocol, so it was not kept (patch kept outside the repo). The launch gap between two graph
nodes this small is about 1 us, which caps such fusions at ~0.4% on the 35B.

### 2026-10-30 #148 — Greedy decode without MTP: argmax on the GPU (7475bd7)

An nsys profile of the 35B CUDA decode (5561aae) showed ~0.6 ms per token with the GPU idle between graphs: ~207 us
from the logits readback to the next upload, and ~400 us from that upload to the first kernel (`cuGraphLaunch`
median 397 us for a 692-node graph). The first part was plain greedy decode copying all 248320 logits (1 MB) back
and taking the argmax on the CPU. `GpuModel::step_greedy` runs `argmax_rows` on the GPU (as the MTP path does) and
reads back only the id; `decode` uses it for greedy GPU runs without MTP. Same tie rule (first index).

- A/B against 5561aae, interleaved, `bench --mtp 0`: CUDA 35B 113.5 -> 115.4 (6+6, spread 0.1/0.1, load 1.6-1.8),
  CUDA 4B 97.2 -> 98.0 (3+3, load 1.6); Vulkan 35B 104.4 -> 106.9 (4+4, spread 0.4/0.3, load 1.2-1.7),
  Vulkan 4B 101.4 -> 102.8 (4+4, spread 1.0/0.9, load 1.0-1.1).
- Greedy md5s unchanged: 4B both backends `64bab3bb`, 35B mtp0 `06c92e18`, mtp1 `04cf91e9`, kvq8 cuda `04cf91e9` /
  vulkan `06c92e18`, and the 2023-token prompt crossing 2048 (mtp1 `d8774b12`, mtp0 `d03a4d33`) on both backends.
- 74 tests pass. Not verified: sampling (temp > 0) is untouched and still reads the logits; `perplexity` and the
  `logits` command use `step` as before.
- Left: the ~400 us `cuGraphLaunch` before the GPU starts. Splitting the decode graph so the first part starts while
  the rest is launched is the next try.

### 2026-10-30 #149 — CUDA argmax over 64 blocks per row; decode graph cut in parts: no gain (9a265d9)

- Graph in parts: #148's ~400 us `cuGraphLaunch` was nsys overhead of `--cuda-graph-trace node`; with `graph`-level
  tracing the launch is ~97 us (still traced). A `graph_cut` that ended the capture at chosen layers and launched
  the parts in order (cuts at layers {1,4}, {1,2,4,..,32}, {1,3,10}) gave 115.0-115.1 against 115.0-115.4 uncut on
  the 35B (2 rounds, load 1.6-2.6). Reverted.
- The graph-level trace showed `argmax_rows` at 75 us per token on the 35B: one block of 1024 threads read the
  1 MB of logits. `argmax_part` (grid 64 x rows, 256 threads) reduces 64 slices of each row into a small partials
  buffer, and `argmax_rows` (one block of 64 per row) reduces those. First index on ties as before.
- A/B against 7475bd7, interleaved, `bench --mtp 0`: CUDA 35B 115.0 -> 115.9 (6+6, spread 0.5/0.5, both drift up
  together; load 0.9-1.2), CUDA 4B 98.2 -> 98.4 (3+3, load 1.2-1.3).
- Greedy md5s unchanged on CUDA: 4B `64bab3bb`, 35B mtp0 `06c92e18`, mtp1 `04cf91e9`, kvq8 `04cf91e9`, and the
  2023-token prompt (mtp1 `d8774b12`, mtp0 `d03a4d33`). 74 tests pass (`argmax_rows_matches_ops_argmax` included).
- Vulkan not changed: its `argmax_rows` takes 17 us per token on the 4B (STRATA_VK_PROF, 151936 tokens), ~28 us
  expected on the 35B, so at most ~0.2%.

### 2026-10-30 #150 — CUDA MoE GEMV: 16 rows per warp for 512-column experts (d2cdf5e)

The 35B MoE GEMVs reach well under peak bandwidth (profile at 5561aae: gate/up IQ2_XS pair 4.8 MB in 28.9 us,
~167 GB/s; down IQ3_XXS 3.2 MB in 24.4 us, ~131 GB/s). A sweep of rows per warp (temporary `STRATA_MOE_RPW=g,d`,
35B `bench --mtp 0`, 2-3 rounds, load 1.1-2.5): gate/up 1/2/3/4 -> 115.4/116.1/115.6/115.7 (2, the default, is
best); down 1/2/4/6/8/12/16/32/64 -> 111.1/114.5/116.1/116.1/116.4/114.1/116.6/110.8/99.2. Down at 16 rows per
warp is 128 blocks, all resident in one wave; 12 leaves a ragged last block per expert. `rpw` now gives 16 for
experts of at most 512 columns.

- A/B against 9a265d9, interleaved: 35B mtp0 116.0 -> 116.5 (6+6, spread 0.8/0.4, load 1.4-1.7), 35B mtp1
  130.9 -> 131.7 (4+4, base round 1 at 126.4 left out, load 1.4-1.6), 4B 98.4 = 98.4 (dense, untouched).
- Greedy md5s unchanged on CUDA (same list as #149, long prompt included). 74 tests pass.
- Not verified: other MoE models (only the 35B is at hand); experts of 512 columns with other quant types.
- Gate/up IQ2_XS stays at ~167 GB/s: each lane loads 2 bytes of indices per superblock plus two table lookups,
  so few bytes are in flight. A decode that reads more per lane is the next thing to try.

### 2026-10-30 #151 — MoE GEMV two rows per lane pass (CUDA); Vulkan MoE down rows per subgroup: no gain (no code commit)

- CUDA: `gemv_row_q` templated on NR rows that share each x load, `moe_gemv` stepping 16 rows with NR=2. 35B
  `bench --mtp 0`, 2 rounds, load 1.6: 116.2/116.3 -> 113.4/113.4, and greedy output changed (summation order
  per row differs). Reverted. More registers per lane cut occupancy more than the shared x loads saved.
- Vulkan: MoE down rows per subgroup swept on the 35B (`bench --mtp 0`, 2-3 rounds, load 1.0-1.8): 2/4/8/16 ->
  107.1/107.0/106.7/105.8. The current 4 is within noise of 2; left as is.
- Not verified: NR=2 for gate/up alone (the pair GEMV already shares x across gate and up).

### 2026-10-30 #152 — Vulkan deltanet state in vec4 loads and stores (f634692)

A per-kernel comparison of the 35B decode (`STRATA_VK_PROF=1 STRATA_NO_GRAPH=1` against nsys with
`STRATA_NO_GRAPH=1`, pp512 tg128, load 0.4-1.0) left deltanet as the clearest single gap: Vulkan 24.5 us per call,
CUDA `deltanet_seq` 18.2 us (30 layers, ~0.19 ms per token). The shader read and wrote each lane's 16 state floats
as scalars; they are now four vec4 chunks (the snapshots too). Decode deltanet: 24.5 -> 19.5 us.
(A pp16 profile shows `moe_combine` at 25 us: that run had 111 expert misses, so combine waited for the CPU. At
pp512 there are no misses and it is 3.5 us, as on CUDA.)

- A/B against d2cdf5e, interleaved, 4+4, load 0.9-1.3: Vulkan 35B decode 106.6 (106.4-106.8) -> 107.5
  (107.2-107.8), prefill pp512 1056.6 (1050.6-1060.5) -> 1087.9 (1083.0-1092.9); 4B 103.1 -> 102.9 (no deltanet,
  noise).
- Greedy md5s unchanged on both backends (4B, 35B mtp0/mtp1, kvq8, long prompt). 74 tests pass.
- Not verified: head dims other than 128 (the vec4 layout is the same for all of 32-256).
- Also tried on CUDA (no gain, reverted): `gemv_row_q` unroll 2 or 8 for all types 116.6 -> 111.0-111.4; unroll 8
  for IQ2_XS only 116.5 -> 115.9.

### 2026-10-30 #153 — CUDA IQ apply_signs as one add (35c937c)

- `apply_signs` (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S grids in `QI`) now computes `(g ^ m) + n` with
  n = 0x01 in each byte to negate and m = n * 0xFF, instead of `__vsub4(g ^ m, m)`. ~g + 1 never carries out
  of a byte because no grid byte is 0 (checked for all five tables in `src/quant/tables.rs`), so the result is
  bit-identical.
- Interleaved A/B against d2cdf5e (`bench --mtp 0`, tg128 after pp512, load 1.0-1.7): 35B CUDA decode 116.4
  (116.1-116.5) -> 117.0 (116.9-117.0), 6 runs each; 4B 98.4 -> 98.5 (3 runs, noise; 4B has no IQ grids).
- Greedy md5s unchanged on both backends (4B, 35B mtp0/mtp1, kvq8, 2023-token prompt). 74 tests pass.
- No gain, reverted (no code commit): `moe_gemv` handling 2 rows per warp pass (NR=2: 115.3 vs 116.5; NR=2 with
  rpw 8: within noise; NR=4: 111); the IQ2_XS grid (512 x u64) copied to shared memory per block: neutral
  (within +-0.2). Together with #151 this points at instruction count, not latency, in the MoE GEMVs.
- Not verified: Vulkan `apply_signs` in `deq.glsl` still uses the 7-op bytewise subtract; next item.

### 2026-10-30 #154 — Vulkan IQ apply_signs as one add (8aa5994)

- `apply_signs` in `deq.glsl` (IQ grid types in `qw`, used by `gemv_q` and `moe_gemv_q`) replaced its 7-op
  bytewise subtract with the one-add form from #153. Bit-identical for the same reason (no grid byte is 0).
- Interleaved A/B against f634692, 35B Vulkan decode (`bench --mtp 0`, tg128 after pp512):
  - first set, load 1.1-1.3: 107.6 (107.5-107.8) -> 108.3 (108.1-108.6), 3 runs each; one more loop run
    read 95.8 and is dropped as an outlier (load had been 2.7 a minute earlier);
  - second set, load 2.0-2.7: 107.1 (106.8-107.4) -> 108.1 (107.9-108.3), 4 runs each.
- 4B Vulkan 103.0 (102.6-103.4) -> 103.2 (102.8-103.4), noise; 4B has no IQ grids.
- Greedy md5s unchanged on both backends (4B, 35B mtp0/mtp1, kvq8, 2023-token prompt). 74 tests pass.
- Prefill is unaffected: only `gemv_q` and `moe_gemv_q` call `qw` (only their SPIR-V changed).

### 2026-10-30 #155 — Vulkan two-pass argmax (fd355a0)

- `argmax_rows` ran one 1024-wide workgroup per row over the 248320 logits (~27 us per token in the
  `STRATA_VK_PROF` trace, #149). Now 64 workgroups of 256 per row write (value, index) partials to the
  quantized-x scratch buffer, and a second dispatch of the same shader (spec constant 1 = final pass) reduces
  them, as `argmax_part` / `argmax_rows` on CUDA. First index wins on ties in both passes (test
  `argmax_rows_matches_ops_argmax` covers a tie, a last-element maximum and an all -inf row).
- Interleaved A/B against f0236ab, 35B Vulkan decode (`bench --mtp 0`, tg128 after pp512), load 0.8-0.9:
  - quick set: 107.9 (107.8-107.9) -> 108.2 (108.2-108.3), 3 runs each;
  - full set: 108.7 (107.8-109.0) -> 108.9 (108.3-109.2), 4 runs each.
  Seven pairs: +0.3 mean, none negative. The expected gain was ~0.25% (22 us of 9.2 ms).
- 4B Vulkan 103.1 (102.9-103.5) -> 103.0 (102.7-103.3), noise.
- Greedy md5s unchanged on both backends (4B, 35B mtp0/mtp1, kvq8, 2023-token prompt). 74 tests pass.
- Not verified: the per-kernel time of the new passes (not profiled).

### 2026-10-30 #156 — Vulkan norm_gate quantizes for the out GEMV (b128a7c)

- The Vulkan decode profile had 110 `quant_xq` dispatches per token: 80 are the SwiGLU quantizations, 30 quantized
  the DeltaNet `norm_gate` output before the `ssm_out` GEMV, each a dispatch plus a full barrier. `norm_gate` now
  takes an xq pointer: after the head is written, a workgroup barrier, then `quant_x` over the head (heads of a token
  are contiguous, so the rows are tokens as `quant_xq` lays them out). New `Backend::norm_gate_q`, default
  `norm_gate` and 0 (CUDA keeps its in-GEMV quantization); new 9/32 x d_inner buffer "oq". Llama numerics keep the
  separate quantization (their one scale per 256 values spans two 128-wide heads).
- Same values quantized by the same `quant_x`, so bit-identical.
- Interleaved A/B against 74184b5, 35B Vulkan decode (`bench --mtp 0`, tg128 after pp512):
  - quick set, load 1.5-1.7: 108.6 (108.1-108.9) -> 109.5 (109.4-109.8), 3 runs each;
  - full set, load 1.1-1.4: 109.1 (109.0-109.2) -> 109.7 (109.6-109.7), 4 runs each.
- 4B Vulkan 103.1 (102.4-103.7) -> 103.5 (103.3-103.7); 4B has no DeltaNet, so noise.
- Greedy md5s unchanged on both backends (4B, 35B mtp0/mtp1, kvq8, 2023-token prompt). 74 tests pass.
- Not verified: `--llama-numerics` on Vulkan (path unchanged: falls back to the old dispatches).

### 2026-10-30 #157 — Vulkan swiglu_q without a barrier in batches (180f59d)

- `run` adds a barrier after a `quant_xq` dispatch even inside `batch` (a GEMV that quantizes its own input reads
  it next). `swiglu_q` dispatched the same shader under that name, so in each MoE layer it split the batch of the
  routed SwiGLU and the shared expert's down GEMV (and added a second barrier at the end of the gate/up batch).
  Its output is only read after the batch. Now the shader is also registered as "swiglu_q" and run under that name.
- Interleaved A/B against 012ad7d, 35B Vulkan decode (`bench --mtp 0`, tg128 after pp512):
  - quick set, load 1.4-1.6: 109.7 (109.6-109.8) -> 110.8 (110.8-110.9), 3 runs each;
  - full set, load 1.0-1.2: 109.5 (109.4-109.6) -> 110.7 (110.6-110.9), 4 runs each.
- 4B Vulkan 103.2 (102.7-103.8) -> 103.2 (102.8-103.5): dense FFN, its SwiGLU is not in a batch.
- Greedy md5s unchanged on both backends (4B, 35B mtp0/mtp1, kvq8, 2023-token prompt). 74 tests pass.

### 2026-10-30 #158 — Vulkan down experts and combine in one dispatch (c2ee2db)

- New `moe_down_combine` backend op (default: `moe_gemv` then `moe_combine`, so CUDA is unchanged). Vulkan runs
  it as one shader: workgroup (block of rpw rows, token), subgroup w takes slots w, w + 8, .. and computes the
  down GEMV of its rows as `moe_gemv_q`, the sums go to shared memory, then one invocation per row adds them into x
  in `moe_combine`'s order (same miss wait on the CPU worker's flag). One dispatch and one barrier fewer per MoE
  layer, and no `ye` round trip. Falls back to the two ops when x is not pre-quantized (float experts).
- Interleaved A/B against d75ad25, 35B Vulkan decode (`bench --mtp 0`, tg128 after pp512):
  - quick set, load 1.4-1.5: 110.6 (110.2-110.9) -> 111.0 (111.0-111.0), 3 runs each;
  - full set, load 1.8-2.4: 110.1 (108.5-110.8) -> 110.6 (109.1-111.2), 4 runs each; every pair gained.
- Rows per workgroup: 4 (= `moe_gemv`'s rows per subgroup) kept. 8: 110.4 -> 110.8 (load 1.4-1.7, same as 4);
  2: 110.6 -> 108.3 (one run 104.3, others -0.3/-0.6; load 1.1-1.3).
- Greedy md5s unchanged on both backends (4B, 35B mtp0/mtp1, kvq8, 2023-token prompt; `STRATA_EXPERT_MB=6000`, so
  the CPU miss path runs). 74 tests pass. Not verified: 4B A/B beyond one pair (103.3 vs 103.4; dense, not touched).

### 2026-10-30 #159 — Vulkan post-mixer norm in the out GEMV's last workgroup: no gain (no code commit)

- Tried: the mixer's out GEMV (`gemv_q`) counts finished workgroups in a device word; the last one runs
  `rms_norm_q`'s steps (x += y, norm, quantize to hq) with the same operations in the same order, reading y through
  a `coherent` reference. One dispatch and one barrier fewer per layer; greedy md5s unchanged (35B mtp0/mtp1).
- Interleaved A/B against d679718, 35B Vulkan decode (`bench --mtp 0`, tg128 after pp512), 3 runs each:
  load 1.4-1.5: 110.8 (110.3-111.1) -> 110.8 (110.7-110.9); load 1.7-1.9: 110.7 (110.6-110.8) -> 110.2 (110.1-110.3).
  4B (dense, same path) 103.3 (103.1-103.5) -> 103.2 (103.0-103.3). Reverted.
- Reading: the norm still runs serially on one workgroup after the GEMV drains, so only the barrier's launch gap is
  saved, and it is smaller than for #156-#158; the `row < rows` guard and the tail may also cost the GEMV itself.
  Patch kept at the session scratchpad (gemv_norm.diff), not in the repo.

### 2026-10-30 #160 — Vulkan topk: subgroupMax/Min in place of the shuffle argmax (9c8892f)

Each of topk's k selection rounds (moe_route and router_topk) did a five-step shuffle butterfly on (value, index)
pairs. Now it does one `subgroupMax` of the value bits plus 1, with 0 meaning "none". The values are >= 0, so their
bits order like them. Then a `subgroupMin` of the index over the lanes that hold that max. The ties go to the lower
index as before, and the picked values are the same.

- Per-call `moe_route` (`STRATA_NO_GRAPH=1 STRATA_VK_PROF=1`, 35B, load 0.8-1.0): 5.38 / 5.90 us before, 5.15 /
  5.16 us after. Probes on the same build: without topk the call is ~3.3 us; the mailbox writes cost ~0.4 us and
  the release fence ~0.1 us. So topk is now ~1.4 us of the call.
- A 4-level tree over 64-bit (key, r) keys in place of each round's linear 16-value scan was slower (5.7-6.4 us),
  so it was dropped.
- 35B Vulkan decode, interleaved A/B against 757c888 (tg128 after pp512):
  - Quick set, load 1.1-1.3, 3 runs each: 110.70 before, 110.77 after.
  - Full set, load 1.3-1.6, 4 runs each: 110.80 before, 111.10 after. 3 pairs gained and 1 tied.
- 4B Vulkan decode, load 1.1-1.3, 4 runs each: 103.20 before, 103.35 after.
- Tests: 74 passed.
- Greedy md5s unchanged: 4B `64bab3bb`; 35B mtp0 `06c92e18`, mtp1 `04cf91e9`; kvq8; long prompt `d8774b12` /
  `d03a4d33`. This holds on both backends.
- Not verified: prefill routing time. router_topk runs once per prefill and is not timed.

### 2026-10-30 #161 — CUDA topk: __reduce_max_sync / __reduce_min_sync (0a54e9a)

This is #160 on CUDA. `topk_warp` (moe_route, router_topk) now finishes each of its k rounds with
`__reduce_max_sync` on the value bits plus 1, then `__reduce_min_sync` on the index among the lanes that hold that
max. That is one redux instruction each, in place of the five-step shuffle butterfly on (value, index). The kernels
already need sm_80 (cp.async, s8 mma), so there is no fallback.

- 35B CUDA decode, interleaved A/B against 907b33f (tg128 after pp512):
  - Quick set, load 2.2-2.6, 3 runs each: 116.77 before, 117.20 after.
  - Full set, load 2.0-2.9, 6 runs each: 116.53 (116.5-116.7) before, 116.93 (116.9-117.1) after.
  - All 9 pairs gained.
- 4B CUDA decode, load 2.1-2.3, 3 runs each: 98.37 before, 98.40 after (neutral).
- Tests: 74 passed.
- Greedy md5s unchanged on both backends: 4B `64bab3bb`; 35B mtp0 `06c92e18`, mtp1 `04cf91e9`; kvq8; long prompt
  `d8774b12` / `d03a4d33`.
- Not verified: the per-call kernel time (no nsys run this item).

### 2026-10-30 #162 — CUDA swiglu_q in registers; fused moe_down_combine: no gain (b0f7222)

- Change: `swiglu_q` kernel, one warp per 256 values of a row, computes silu(g) * u in registers and quantizes it
  with `quant8` (split out of `quant_x`) straight into the down GEMV's xq. Before: `swiglu` wrote g, `quant_xq`
  read it back. Used for the dense FFN, the shared expert and the routed experts (float down weights keep `swiglu`).
- Two earlier shapes lost: a 2048-value chunk per block (write g, sync, quant_x) gave 35B +0.6 but 4B -1.2 (n_ff
  9728 = 5 blocks); a 256 chunk that still wrote g and read it back lost on both (35B -1.6, 4B -0.7). nsys had
  that one at 4.46 us vs 1.6 us for swiglu.
- A/B vs a50e152, interleaved, load 1.2-1.3: 35B 117.42 -> 117.50 (6 pairs, none worse), 4B 98.40 -> 98.57 (3
  pairs). Quick run at load 1.7-1.8: 35B 117.30 -> 117.43, 4B 98.47 -> 98.67. Tests 74 passed; CUDA greedy md5s
  unchanged (4B, 35B mtp0/mtp1, kvq8, long prompt mtp0/mtp1).
- Step B, reverted (no code commit): CUDA port of the Vulkan `moe_down_combine` (#158). 35B decode 117.2 -> 111.3
  (3 pairs, load 1.7-1.9). nsys: the fused kernel took 37-40 us vs 22.7 (`moe_gemv_iq3_xxs` down) + 3.2
  (`moe_combine`) = 25.9 us. ptxas: 72 registers vs 48 for moe_gemv, so 3 blocks/SM instead of 5. With
  `__launch_bounds__(256, 5)` and warps splitting the rows of one expert (experts in turn): 27.0 us, still slower;
  `(256, 6)` 32-39 us. No misses in the bench (hit rate 100%), so the miss spin was not the cost. Under CUDA
  graphs the extra launch is cheap, unlike a Vulkan dispatch + barrier. Diff kept outside the repo.
- Not verified: Vulkan (untouched), prefill.

### 2026-10-30 #163 — CPU worker: bulk read of the routing from the BAR mailbox (47472c6)

- Cause, found by timing the host phases of a Vulkan decode token (temporary `Instant` prints, removed): the
  `after_forward` step took 541 us mean (max 944) on Vulkan vs 14.5 us on CUDA, with the GPU idle. It reads each
  CPU-cache layer's routing from the worker mailbox with per-word volatile reads, and on Vulkan the mailbox is BAR
  memory (#2.3 note above: host reads uncached). Other phases: `put` 57.6 us (CUDA 13.4), tokenizer decode 13.8.
- Change: `CpuWorker::routing` copies the layer's 3k words out of the mailbox with one `copy_nonoverlapping`, then
  builds the tuples from the local copy. after_forward 541 -> 117 us mean (single run).
- A/B vs 521462e, interleaved, load 0.6-0.9: Vulkan 35B 111.25 -> 116.25 (4 pairs, spread 115.9-116.5, all better),
  Vulkan 4B 103.18 -> 103.35 (4 pairs, no CPU worker, neutral). CUDA 35B 117.58 -> 117.57 (6 pairs), 4B 98.50 ->
  98.47 (3 pairs): neutral. Long prompt (2023 tokens, single run): Vulkan gen mtp0 53.8 -> 55.5, mtp1 49.7 -> 51.1.
- Tests 74 passed. Greedy md5s unchanged on both backends (4B, 35B mtp0/mtp1, kvq8, long prompt mtp0/mtp1).
- Not verified: what the remaining ~117 us is (still 8x CUDA); `put` (57.6 us) untouched; the worker thread's own
  reads of the BAR mailbox (activations of missed experts) untouched; no miss-heavy run (the bench hits 100%).

### 2026-10-30 #164 — CPU worker: streaming loads for the routing read (0b3dcba)

- Change: `routing` reads the layer's words with `_mm_stream_load_si128` (movntdqa, `read_wc`; plain copy without
  SSE4.1). On write-combining memory, which the Vulkan BAR mapping is, one movntdqa fetches the whole 64-byte line
  and later loads of that line come from the streaming buffer; plain loads each make an uncached PCIe read. The
  copy is rounded up to whole 16-byte chunks (still inside the layer's mailbox).
- Vulkan after_forward (temporary timer, removed; single run, load 2.5): 117 -> 70 us mean, median 55 (CUDA 14.5).
- A/B vs 6ed7690, interleaved, load 1.6-2.8: Vulkan 35B 116.13 -> 116.65 (8 pairs, 7 better, loop spread
  115.9-117.1); CUDA 35B 117.50 -> 117.48 (4 pairs, neutral).
- Tests 74 passed. Greedy md5s unchanged on both backends (4B, 35B mtp0/mtp1, kvq8, long prompt mtp0/mtp1).
- Not verified: 4B (no CPU worker, path not taken). The remaining ~55 us is about 40 layers x 3 uncached line
  reads (NTOK on line 0, the routing on lines 2-3); reading NTOK from `nt` or a contiguous routing area for all
  layers would cut it further. The worker thread's reads of the mailbox (spin on REQ, activations) are untouched.

### 2026-10-30 #165 — Vulkan htod: small copies inline in the command buffer (52b364e)

- Cause: `put` cost 57.6 us per Vulkan decode token (CUDA 13.4), GPU idle. Each `htod` copied into staging, then
  submitted and waited on the fence (`flush`), twice per token (embedding, step words).
- Change: copies up to 64 KiB (4-byte aligned, not while recording a graph) go into the open command buffer as
  `vkCmdUpdateBuffer`, with a barrier before and after. `submit` puts the open buffer ahead of the graph, so the
  next replay carries them; nothing waits. Larger copies keep the staging path.
- A/B vs da90fb0, interleaved, load 1.7-2.9: Vulkan 35B 116.68 -> 117.01 (8 pairs, 7 better, 1 equal), Vulkan 4B
  103.00 -> 104.03 (4 pairs, all better). CUDA 35B 117.50 -> 117.53 (3 pairs, code untouched).
- Tests 74 passed. Greedy md5s unchanged on both backends (4B, 35B mtp0/mtp1, kvq8, long prompt mtp0/mtp1).
- Not verified: the new host time of `put` (no timer this item); prefill (few large copies, path unchanged).

### 2026-10-30 #166 — Parity spot check after the shared CPU-worker change (no code commit)

- Why: #163/#164 changed `cpu_worker.rs`, shared by both backends. `tools/parity.py`, default model, 640 tokens,
  interleaved against 521462e (before #163), load 0.8-1.4.
- Vulkan: 628 / 628 / 626 before, 629 / 627 / 629 after (627.3 -> 628.3).
- CUDA: first 3 pairs 627 / 627 / 627 before, 626 / 625 / 626 after. Then 4 more pairs: 625 on all 8 runs, and the
  per-prompt result files of the two binaries are byte-identical (two outcome sets, each seen with both binaries).
  So the drop was run-to-run drift (MTP 1 from the CUDA calibration is nondeterministic), not the change: the
  routing read only copies the same words, and greedy md5s are unchanged.
- Both backends stay under the plan's 98.5% as before (#94/#98 numbers: 628-630); nothing new.
- Note: `--extra '--gpu'` fails in argparse (taken as an option); pass `--extra=--gpu`.

### 2026-10-30 #167 — Vulkan pp4096: one slow rep per process, GPU-side (no code commit)

- Symptom: `bench --pp 4096 --reps 4` on Vulkan gives per-rep prefill like 2139 / 1125 / 1940 / 2172 tok/s, so the
  mean (~1840) hides a steady state above CUDA (CUDA 2109-2120 on every rep). Seen on 4 of 4 runs, load 0.4-1.6.
- Pattern (temporary per-rep print): 10 reps give one slow rep (2nd), then 2150-2170 to the end. pp2048: the 3rd
  rep is slow (912, next 1523). So it starts about 2 s into the first prefill work and lasts about 2-3 s.
- Not the cause: VRAM pressure (`--vram-margin 2048` same), expert moves (`--fixed-experts` same), decode or graphs
  (`--tg 0` same), time since load (8 s sleep before the reps: still the 2nd rep), the recent host-side changes
  (521462e, before #163, same: 1025-2169 and 1128-2159), pipeline creation (8 ms total, #165 notes).
- What the GPU shows in that window (nvidia-smi at 100 ms): memory controller utilization jumps from ~36% to
  67-100%, power falls from the 170 W cap to 124-150 W, and the clock reason moves from SW power cap to
  "Reliability" at 1987 MHz. Every kernel is slower in it (STRATA_VK_PROF: attn_prefill +83%, deltanet +80%, gemm
  +24-30%), and there is no PCIe traffic. CUDA runs see brief "Reliability" events too, but memory utilization stays
  at 34-37%. So something adds GPU memory traffic for a few seconds under the Vulkan driver; our queue is single
  and nothing of ours runs then.
- No fix found. Vulkan prefill numbers should be read from the per-rep steady state (or the max), not the mean.

### 2026-10-30 #168 — CUDA moe_gemv: IQ lookup tables in shared memory (edd3bb7)

- Why: nsys (pp512 tg256, CUDA) puts the expert GEMVs at ~2050 us/tok and their bandwidth well below the dense
  GEMVs: gate/up IQ2_XS pair 28.4 us for 4.85 MB (~171 GB/s), down IQ3_XXS 22.2 us for 3.2 MB (~145 GB/s), against
  ~300 GB/s for q5_k. Each lane does random lookups into the `__device__` grid and sign tables.
- Change: `moe_gemv` for IQ2_XS/IQ3_XXS copies iq2xs_grid (4 KB) or iq3xxs_grid (1 KB) plus ksigns (128 B) into
  shared memory at block start and reads them there (`QS<T>` in kernels.cu). Down goes through `moe_gemv` on CUDA,
  so it is covered too. Dense gemv/gemvn, prefill gemm and deq8 keep the global tables. Vulkan is unchanged.
- A/B, 35B `bench --mtp 0` (tg128 after pp512), 4 interleaved rounds, load 1.7-2.1: base f20d1e4 114.1 (first,
  warm-up) / 117.5 / 117.5 / 117.5, new 117.9 / 118.0 / 118.0 / 118.0. +0.4%, small but steady.
- Greedy md5s unchanged on both backends (4B, 35B mtp 0/1, kvq8, long prompt mtp 0/1). Tests 74 passed.
- nsys after the change (pp512 tg128, kernel average): down IQ3_XXS 22.47 -> 22.11 us, gate/up IQ2_XS pair 28.44 ->
  28.59 us (the pair overlaps a second stream, so its duration says little). A variant with shared tables for IQ3_XXS
  only: 117.6 / 117.6 / 117.5 against 118.0 x3 (4 rounds, load 1.6-1.9), so IQ2_XS stays shared too.
- Not verified: 4B decode (no IQ types there), Vulkan (its shaders have their own table path; see #169).

### 2026-10-30 #169 — Vulkan expert GEMVs: IQ tables in shared memory, no gain (no code commit)

- Same idea as #168 on Vulkan: an `IQ_SHARED` block in deq.glsl that copies iq2xs_grid / iq3xxs_grid / ksigns to
  shared arrays and redirects the table macros to them, used by moe_gemv_q and moe_down_combine.
- A/B, 35B `bench --mtp 0`, 7 interleaved rounds, load 1.3-1.7: base f20d1e4 117.0 / 117.0 / 117.3 / 117.1 / 116.6 /
  116.5 / 116.6 (mean 116.87), new 99.6 (first round, outlier) / 117.3 / 117.2 / 117.3 / 117.0 / 116.5 / 117.1
  (mean 117.07 without the outlier). About +0.2%, inside the spread, so reverted. The Vulkan tables are already
  read through a buffer reference that the driver likely caches well; the CUDA `__device__` gathers gained more.
- Not verified: whether the 99.6 round was a one-off (driver shader compile on new SPIR-V, or the #167 window).

### 2026-10-30 #170 — CUDA norm_gate quantizes for the ssm_out GEMV (c959c94)

- nsys after #168: the DeltaNet out GEMV (q6_k, 2048 x 4096, 6.88 MB) takes ~28 us, ~246 GB/s, against ~297 GB/s
  for the q5_k qkv/gate pair. It has 128 blocks (rpw 2), and each quantized the whole 4096-wide x itself.
- Change: #156's `norm_gate_q` on CUDA. `norm_gate` takes an xq pointer and, after a block barrier, quantizes its
  head into quant_xq's layout. A 128-value head fills only 16 lanes of a warp, so `quant8` gets a shuffle mask
  (default full, so the other callers are unchanged) from a ballot. Llama numerics and heads not a multiple of 32
  keep the in-GEMV quantization.
- Same values through the same `quant8`, so bit-identical: greedy md5s unchanged on CUDA (4B, 35B mtp 0/1, kvq8,
  long prompt mtp 0/1). Tests 74 passed.
- A/B against 95f379a, 35B CUDA `bench --mtp 0` (tg128 after pp512), 4 interleaved rounds, load 0.5-0.9: 117.6 /
  118.2 / 118.1 / 118.1 -> 118.7 / 118.7 / 118.7 / 118.6 (+0.5%).
- Not verified: `--llama-numerics` on CUDA (path unchanged), 4B (no DeltaNet), per-kernel time after the change.

### 2026-10-30 #171 — CUDA deltanet_seq launch bounds: slower (no code commit)

- nsys: `deltanet_seq` uses 124 registers at 256 threads (2 blocks per SM, so 128 blocks in 3 waves). The kernel
  average (20.7 us) mixes prefill: decode launches take 11.3 us (2 MB state read + written, ~355 GB/s, at peak), the
  30 pp512 launches 1223 us each.
- `__launch_bounds__(256, 3)` (80 registers): decode 11.3 -> 21.2 us, prefill 1223 -> 1885 us (spills). Reverted.
- Decode DeltaNet is bandwidth-bound already. Prefill DeltaNet (~37 ms of a pp512) is the sequential scan that the
  kernel's ponytail note names; a chunked form is the way to speed it up.

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
6. Decode profile at 144b43b (nsys, 4K prompt, 512 tokens, us per token; nsys slows the CPU expert threads, so
   moe_combine's wait is not real): gemvn_q5_k 2514 (147 calls), moe_gemv_iq2_xs 1167, gemv_q6_k 1085,
   moe_gemv_iq3_xxs 933, LM head gemvn_q4_k 877 (bandwidth bound), attn_decode_gqa 646, gemvn_f32 626 (140 calls,
   router and SSM projections kept f32 like llama.cpp), rms_norm 495 (100 calls, 4.9 us each), moe_publish 249,
   router_topk 219, gemv_q5_k 205; about 1.5 ms of small launch-bound kernels in all. Fusing rms_norm into the
   following GEMV and the router GEMV with router_topk are the next levers (SwiGLU into the down GEMV: no gain, #125;
   MoE combine with the next norm: no gain, #147).

## Questions for owner

- (#119) V4 on the 4B prefill: llama.cpp Vulkan pp4096 2849, Strata Vulkan ~1950, Strata CUDA ~2040. Three
  quarters of the 4B prefill is gemmi, whose int8 epilogue (each exact int32 sum scaled into f32 with the roundings
  of kernels.cu) is what makes Vulkan greedy output identical to CUDA; llama.cpp dequantizes to f16 and uses NVIDIA's
  cooperative_matrix2. Under "greedy output identical to CUDA" the 4B prefill target looks out of reach (gemmi would
  need ~45% less time while doing the same rounded operations). Options: (a) accept the 4B prefill gap (decode and
  the default model meet V4); (b) allow a Vulkan-only f16 prefill path whose greedy output may differ from CUDA,
  gated by parity only; (c) change the prefill numerics of both backends (changes CUDA greedy output). Which?

- (#88) README: `serve` now does tool calling. Proposed addition after the sentence about `cached_tokens`:
  "`tools` and `tool_choice` work as in the OpenAI API (calls come back as `tool_calls` with
  `finish_reason: "tool_calls"`). The prompt follows the model's own chat template, rendered like llama.cpp's
  `--jinja`. `tool_choice: "required"` does not force a call." Also `src/serve.rs (HTTP and chat template)`
  should read `src/serve.rs (HTTP), src/chat.rs (chat templates and tool calls)`.

- (#81) GPU runs only check for Dota 2. CS2 (and any game) also takes VRAM and GPU time, which moved parity by one
  token and made benches unreadable. Should the loop also stop GPU work while `cs2` (or any process holding over
  1 GB of VRAM outside the lock) is running?
  Update #101: Minecraft (Java, 3 GB VRAM) made the default model OOM in `gpu_model` and in an A/B bench.
  Update 2026-10-16 23:54: Geometry Dash (Steam, gamescope) started during a Vulkan A/B and took about 38% of
  the GPU, so that A/B is void. The loop now waits for gamescope/java to exit and the GPU to idle before benching.
  Update 2026-10-17 (#129): RustingEngine's `physics_bench` ran about an hour at 100% GPU under
  `/tmp/rusting-gpu.lock`, a different lock file from `/home/vasylt/Code/rusting/.gpu.lock`, so the two loops do not
  exclude each other. Should all projects share one lock file?

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
