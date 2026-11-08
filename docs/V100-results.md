# V100 results

Machine: Tesla V100-SXM2-32GB, driver 580.178.04, Vulkan 1.4.312, CachyOS.
`strata` v1.2.0 + the `gemmi_dp` shared-memory fix (branch `v100`).

## 2026-11-06 — Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf, Vulkan

```
plan: weights+state+scratch 2580 MB (KV 642 MB for 32896 tokens, prefill 512 MB),
      expert slots 10040 MB, resident experts 10240/10240
pp    512: prefill  635.6 (361.6-774.2) tok/s | tg 128: decode 109.8 (82.7-123.4) tok/s
pp   4096: prefill 1423.9 (1422.0-1427.0) tok/s | tg 128: decode 108.6 (91.1-117.7) tok/s
pp  32768: prefill  523.5 (522.1-525.1) tok/s | tg 128: decode  95.7 (95.4-96.1) tok/s
expert hit rate 100.0% (0 misses), cache swaps 0
```

Note: the first repetition of `pp 512` includes Vulkan pipeline compilation;
the steady-state value is ~775 tok/s.

## 2026-11-06 — Qwen3.8-27B-UD-Q4_K_XL.gguf, Vulkan (dense, no tensor cores)

```
plan: weights+state+scratch 17118 MB (KV 512 MB for 8192 tokens, prefill 706 MB)
pp    512: prefill 317.9 (315.8-319.4) tok/s | tg 128: decode 27.6 (27.5-27.7) tok/s
pp   4096: prefill 291.1 (290.5-291.6) tok/s | tg 128: decode 26.6 (26.1-26.9) tok/s
```

## llama.cpp reference (same card)

Measured with `llama-server` (`/completion`, `cache_prompt: false`, temp 0):

| Model | prompt | prefill | decode |
| --- | --- | --- | --- |
| Qwen3.6-35B-A3B Q2_K_XL | 416 tok | 496 tok/s | 101.9 tok/s |
| Qwen3.6-35B-A3B Q2_K_XL | 1261 tok | 1243 tok/s | 101.9 tok/s |
| Qwen3.6-35B-A3B Q2_K_XL | 2941 tok | 1352 tok/s | 99.3 tok/s |
| Qwen3.8-27B Q4 (production logs) | 1092 / 4381 tok | 292 / 491 tok/s | ~49-53 tok/s |

## Notes

- Before the fix, any chunked prefill on Vulkan ended with
  `Xid 13 SKEDCHECK18_L1_CONFIG_TOO_SMALL` and a lost device.
- sm_80 `mma` and `cp.async` do not exist on sm_70: the CUDA backend requires
  the port described in `docs/ROADMAP.md`.

## 2026-11-07 — M0-T4 shared-memory preflight validation (Vulkan, V100)

```
strata check --backend vulkan: maxComputeSharedMemorySize 49152
  60 (kernel, spec) pairs checked, largest gemmi_dp = 49152 (at the limit, accepted)
  no cooperative-matrix kernels (gemmi, gemmi_s, gemm_f, gemm_fs) on this device
  cost 1.1-4.8 ms

STRATA_VK_SMEM_LIMIT=40000: exit 1
  vulkan: kernel gemmi_dp with specialization [[0, 13, 176, 176947200, 0]]
  needs 40960 bytes of shared memory; this device allows 40000
  (raised before any weight upload)

bench --pp 512  --tg 128 --ctx 32768: prefill 761.6, decode 121.4 tok/s
bench --pp 4096 --tg 128 --ctx 32896 (x3): prefill 1401-1404, decode 115.5-116.2 tok/s
same-window v100 baseline: prefill 1398-1407, decode 113.9-115.5 tok/s -> no regression
no new Xid; device usable after the forced-oversize run
```

Note: the 1423.9 / 108.6 figures earlier in this file come from a different
session; compare within the same window.

## 2026-11-07 — M0-T5 GGUF type matrix (Vulkan, V100, sm_70)

Commit: `1b19139` | driver 580.178.04 | Vulkan 1.4.312 | maxComputeSharedMemorySize 49152
Command: `STRATA_TEST_BACKENDS=vulkan cargo test --release --test kernels vulkan_type_matrix -- --nocapture --exact --test-threads=1`

The test builds one synthetic, finite block per `GgmlType` with the scale fields at the
offsets each dequantizer reads, uploads raw bytes (bypassing `Backend::upload`, so native
kernels run) and compares dequant, GEMV, GEMV-N, GEMV-Q (bit-equal to GEMV-N on the same
`quant_xq`), MoE GEMV (first expert, `-1` skipped slot, last expert), dense GEMM over 130
positions and the tiled `gemm` with 3 tiles against the CPU reference over the same bytes.
`strata check` and `Vk::max_shared_bytes` use the committed SPIR-V accounting.

| Type | Block elems | Upload path (loader) | dequant | gemv | gemvn | gemvq | moe_gemv | gemm_dense | gemm (tiled) | max shared (B) | Result | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F32 | 1 | native | pass | pass | pass | pass | pass | pass | pass | 27664 | pass | f16 GEMM inputs truncate (Volta `OpFConvert`); reference matches |
| F16 | 1 | native F32 (`upload_type`) | unsupported | unsupported | unsupported | unsupported | unsupported | unsupported | unsupported | 49152 | unsupported | no F16 shader path; loader uploads it as F32 |
| Q4_0 | 32 | native Q8_0 (`upload_type`) | pass | pass | pass | pass | pass | pass | pass | 40960 | pass | raw native kernels tested; loader uploads Q8_0 |
| Q4_1 | 32 | native | pass | pass | pass | pass | pass | pass | pass | 40960 | pass | |
| Q5_0 | 32 | native Q8_0 (`upload_type`) | pass | pass | pass | pass | pass | pass | pass | 40960 | pass | raw native kernels tested; loader uploads Q8_0 |
| Q5_1 | 32 | native | pass | pass | pass | pass | pass | pass | pass | 40960 | pass | |
| Q8_0 | 32 | native | pass | pass | pass | pass | pass | pass | pass | 40960 | pass | |
| Q2_K | 256 | native | pass | pass | pass | pass | pass | pass | pass | 49152 | pass | |
| Q3_K | 256 | native | pass | pass | pass | pass | pass | pass | pass | 49152 | pass | |
| Q4_K | 256 | native | pass | pass | pass | pass | pass | pass | pass | 40960 | pass | |
| Q5_K | 256 | native | pass | pass | pass | pass | pass | pass | pass | 40960 | pass | |
| Q6_K | 256 | native | pass | pass | pass | pass | pass | pass | pass | 49152 | pass | |
| IQ2_XXS | 256 | native | pass | pass | pass | pass | pass | pass | pass | 40960 | pass | |
| IQ2_XS | 256 | native | pass | pass | pass | pass | pass | pass | pass | 49152 | pass | f16 GEMM inputs truncate (Volta `OpFConvert`); reference matches |
| IQ3_XXS | 256 | native | pass | pass | pass | pass | pass | pass | pass | 40960 | pass | |
| IQ1_S | 256 | native | pass | pass | pass | pass | pass | pass | pass | 40960 | pass | |
| IQ4_NL | 32 | native | pass | pass | pass | pass | pass | pass | pass | 49152 | pass | |
| IQ3_S | 256 | native | pass | pass | pass | pass | pass | pass | pass | 40960 | pass | |
| IQ2_S | 256 | native | pass | pass | pass | pass | pass | pass | pass | 49152 | pass | f16 GEMM inputs truncate (Volta `OpFConvert`); reference matches |
| IQ4_XS | 256 | native | pass | pass | pass | pass | pass | pass | pass | 40960 | pass | |
| IQ1_M | 256 | native | pass | pass | pass | pass | pass | pass | pass | 49152 | pass | |
| BF16 | 1 | native | pass | pass | pass | pass | pass | pass | pass | 27664 | pass | f16 GEMM inputs truncate (Volta `OpFConvert`); reference matches |
| Q2_0 | 64 | native | pass | pass | pass | pass | pass | pass | pass | 40960 | pass | |

Summary: 22/23 types pass, 1 documented unsupported (F16, no shader path; the loader
uploads it as F32), 0 fail. Largest shared memory 49152 B (`gemmi_dp`, at the 49152 B
limit and accepted) — the `COOP = 0` table path of `53f823e` still fits. No Xid.

Note: on this driver the Vulkan f16 GEMM path (`packFloat2x16`, SPIR-V `OpFConvert`)
truncates its f16 inputs toward zero, while CUDA's `__float2half` rounds to nearest
even. The GEMM reference uses the conversion the backend performs, so the matrix measures
the kernel. The two pre-existing Vulkan tests that assumed round-to-nearest
(`gemv_partial_superblocks_and_q2_0`, `gemm_matches_reference_for_every_type`, plus
`f16_weight_uploads_as_f32`) needed the same backend-aware reference; no shader or
numerics changed. Residual risk: the Vulkan f16 GEMM differs from CUDA by up to ~6e-4
relative on these synthetic inputs.

Attention: type-independent (f32 after dequant). Covered by
`attn_prefill_matches_decode` and `strata generate` on both models.

### Checks

```
strata check --model Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf --backend vulkan
  60 (kernel, specialization) pairs checked, largest 49152 bytes (gemmi_dp) of 49152 allowed
  exit 0

strata check --model Qwen3.8-27B-UD-Q4_K_XL.gguf --backend vulkan
  66 (kernel, specialization) pairs checked, largest 49152 bytes (gemmi_dp) of 49152 allowed
  exit 0
```

### End-to-end

```
strata generate --model Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf --gpu --backend vulkan \
  --prompt "The river town woke before dawn." -n 32
  prefill 94.48 tok/s | gen 125.86 tok/s, exit 0
  "The first light of day was just beginning to touch the eastern horizon when the first
   sounds of activity could be heard. The clatter of hooves on cobble"

strata generate --model Qwen3.8-27B-UD-Q4_K_XL.gguf --gpu --backend vulkan \
  --prompt "The river town woke before dawn." -n 32
  prefill 27.10 tok/s | gen 27.97 tok/s, exit 0
  "Not with birdsong or the creak of shutters, but with the low, insistent hum of the
   *Königstein*—the old"
```

### Bench (informational, same window)

```
strata bench --pp 4096 --tg 128 --reps 1 --ctx 32896
  prefill 1402.2 tok/s | decode 116.3 tok/s
  adjacent M0-T4 window: prefill 1401-1404, decode 115.5-116.2 -> no regression
```

### Xid

`dmesg -T | grep -i Xid` before the window ended at `Fri Nov 6 20:40:46 2026`
(the M0-T4 forced-oversize run); after all M0-T5 runs the tail is unchanged.
No new Xid in the window.

## 2026-11-07 — M0-T7 Vulkan parity baseline (V100, sm_70)

Commit `d6c2beb` (v100) | driver 580.178.04 | Vulkan 1.4.312 | maxComputeSharedMemorySize 49152
Window: production (`llama.router.service`) stopped by the coordinator, V100 free (0 MiB used);
no other process on the card. Total window 13 min 43 s.

```
strata d6c2beb, --gpu --backend vulkan --llama-numerics --fixed-experts, tag vk-sm70
llama-server 0.9.0-dev build 12110 (cd959bae9), -ngl 99 -fa auto -c 4096, CUDA on the V100
20 prompts x 32 greedy tokens = 640 positions
strict vs llama.cpp token-by-token: 625/640 = 97.66%
either llama.cpp mode:              629/640 = 98.28%
llama.cpp self-agreement:           630/640 = 98.44%
verdict: FAIL (629/640, below 634/640 either-mode); documented deviation, see below
Xid: none in the window (dmesg tail unchanged since Fri Nov 6 20:40:46 2026)
```

Environment and reference build:

- `llama-server` `/srv/llama.cpp/build/bin/llama-server`, `version: 0.9.0-dev (build 12110,
  commit cd959bae9)`, `/props` `build_info: b12110-cd959bae9`, on `127.0.0.1:8199`.
- Flags: `-m Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf --host 127.0.0.1 --port 8199 -ngl 99 -fa auto
  -c 4096 --seed 1234`; `n_ctx_slot = 4096`, `n_slots = 4`, `kv_unified = true`, model on CUDA0,
  12185 MiB resident.
- `-fa auto` resolved to **disabled**: the server logged
  `resolve_fused_ops: Flash Attention over K/V rows not supported, set to disabled`
  (the 3060 gate ran with `-fa 1`; recorded here because the reference differs).
- Model load 4.4 s (page cache warm), ref cache + one-prompt smoke 32 s, `tools/selfparity.py`
  300 s (640 requests, 0 refused), gated run 145 s.

Per-prompt results for `vk-sm70` (parity mode):

| # | prompt | prompt tokens | agree/n | first_miss | top2_gap_at_miss |
| --- | --- | --- | --- | --- | --- |
| 1 | Write a Rust function that returns the n-th Fibona | 26 | 32/32 | None | None |
| 2 | In Python, how do I read a CSV file and compute th | 33 | 32/32 | None | None |
| 3 | Explain what this C code does: int f(int *a, int n | 54 | 32/32 | None | None |
| 4 | Write a SQL query that returns the top 5 customers | 42 | 32/32 | None | None |
| 5 | Write a regular expression that matches an IPv4 ad | 27 | 32/32 | None | None |
| 6 | Fix the bug: def avg(xs): return sum(xs) / len(xs) | 36 | 32/32 | None | None |
| 7 | Give a bash one-liner that finds the 10 largest fi | 31 | 31/32 | 9 | 0.0329 |
| 8 | Why is the sky blue? Answer in three sentences. | 23 | 32/32 | None | None |
| 9 | Розкажи коротко про історію Києва. | 26 | 32/32 | None | None |
| 10 | Объясни, что такое квантовая запутанность, простым | 31 | 28/32 | 3 | 0.5055 |
| 11 | 请用简单的语言解释什么是机器学习。 | 19 | 30/32 | 23 | 0.0837 |
| 12 | A train leaves at 9:15 and travels 180 km at 72 km | 44 | 32/32 | None | None |
| 13 | Define the word 'entropy' as used in information t | 24 | 32/32 | None | None |
| 14 | Give me a simple recipe for pancakes. | 20 | 31/32 | 11 | 0.1449 |
| 15 | Summarize this design document in five bullet poin | 995 | 32/32 | None | None |
| 16 | What license terms apply here? Summarize: | 444 | 30/32 | 0 | 0.0319 |
| 17 | Find the error in this log: | 1367 | 30/32 | 8 | 0.1615 |
| 18 | Which of these cities is the largest by population | 281 | 31/32 | 30 | 0.1016 |
| 19 | List the keys whose value is false: | 553 | 30/32 | 0 | 0.3904 |
| 20 | Which row has the highest score? | 541 | 32/32 | None | None |

Second block — same server, same window:

```
tag vk-sm70-r2 (repeat of parity mode): strict 625/640 = 97.66% | either 629/640 = 98.28%
tag vk-sm70-default (--gpu --backend vulkan only): strict 626/640 = 97.81% | either 630/640 = 98.44%
```

The repeat is position-identical to the first run (625/629), so the gap to the bar is not
sampling noise from expert placement (`--fixed-experts`) or scheduling.

Deviation note (below the 99% either-mode bar; issue #6 requirement "if below the bar, do not
write verdict PASS"):

- 629/640 (98.28%) is 5 positions short of the 634/640 gate; the 3060 parity-mode runs scored
  635-637/640 either-mode (99.32%), so this is 6-8 positions and ~1 point below that range.
  The default path is *better* here (630/640), the opposite of the 3060 (98.85% default vs
  99.32% parity mode).
- The 3060 bar was measured against a **CPU** llama.cpp build; this run used the **CUDA** build
  with flash attention disabled by the build itself (issue #34 already saw 97.34% strict against
  a CUDA-build reference). llama.cpp's own two modes agree on only 630/640 (98.44%) here, so the
  either-mode gate at 634/640 is stricter than llama.cpp's self-consistency on this card.
- The engine was not changed: no shader, SPIR-V or numerics edit is part of M0-T7. Diagnosing
  the 11 remaining positions (prompts 10, 16, 17, 19 concentrate 10 of them; prompt 10 misses at
  a 0.5055 top-2 gap, so it is not a near-tie) is a follow-up task.
- No new Xid, no device lost, no `strata score` failure; VRAM peak with server + score was
  ~29 GB of 32 GB and neither process was evicted.

## 2026-11-07 — Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf, CUDA sm_70 (M1-T7 bring-up)

First run of the CUDA backend on this card (issue #20, branch `task/20-cuda-sm70-bringup`).
Bring-up numbers, **not** the M2 benchmark protocol: one repetition, one window, no tuning,
`llama.router.service` stopped. Model, machine and driver as above; NVRTC 12.9.

```
cuda device 0: Tesla V100-SXM2-32GB, sm_70 (Volta fallback)
cuda shared memory: opt-in 98304 B per block, attn_decode_gqa*: 2112 B static, dynamic max 73728 B
cuda shared memory: opt-in 98304 B per block, attn_prefill_*: 0 B static, dynamic max 67584 B
cuda shared memory: opt-in 98304 B per block, gemvn_*: 0 B static, dynamic max 98304 B
plan: weights+state+scratch 1984 MB (KV 80 MB for 4096 tokens, prefill 512 MB),
      expert slots 10040 MB, resident experts 10240/10240
pp    512: prefill 836.5 (836.5-836.5) tok/s | tg 32: decode 116.2 (116.2-116.2) tok/s
expert hit rate 100.0% (0 misses), cache swaps 0
```

`generate --prompt "The capital of France is"`: `-n 8` gen 122.78 tok/s, `-n 32`
prompt 5 tok 84.03 tok/s | gen 32 tok 124.27 tok/s. Output coherent ("…Paris is the capital and
most populous city of France…"). The Vulkan baseline of 2026-11-06 on the same model was
`pp 512` 635.6 tok/s (steady state ~775) and `tg 128` 109.8 tok/s; the CUDA figures above are
higher but come from a shorter protocol, so they are indicative only.

Device facts measured here (they answer the M1-T7 open question): the driver reports a
dynamic shared memory opt-in of **98304 B (96 KiB) with no reserve**, `attn_decode_gqa_*` uses
**1920-2112 B of static shared memory** (1920 in `--llama-numerics`), `attn_prefill_*` and
`gemvn_*` use 0 B. `optin - static` is therefore 96192 B for `attn_decode_gqa_*`, below the
98304 B the sm_70 build used to request for every one of those functions.

Kernels that actually executed (`nsys profile --trace=cuda --cuda-graph-trace=node`):

- `generate -n 8`: `gemv_q4_k`, `gemv_q5_k`, `gemv_q6_k`, `gemv_q8_0`, `gemv_f32`,
  `moe_gemv_iq2_xs`, `moe_gemv_iq3_xxs`, `moe_gemv_iq4_xs`, `moe_route`, `moe_combine`,
  `rms_norm`, `rms_norm_rope`, `swiglu_q`, `deltanet_seq`, `conv_silu`, `norm_gate`,
  `rope_freq`, `attn_decode_256`, `attn_merge`, `argmax_part`, `argmax_rows`.
- `bench --pp 512 --tg 32` adds the prefill set: `attn_prefill_256` (10 launches, 2.04 s total,
  i.e. the 67584 B dynamic shared memory launch), `gemm_q5_k`, `gemm_q6_k`, `gemm_q8_0`,
  `gemm_iq2_xs`, `gemm_iq3_xxs`, `gemm_iq4_xs`, `gemms_q5_k`, `gemms_q6_k`, `gemms_iq2_xs`,
  `gemms_iq3_xxs`, `gemms_iq4_xs`, `gemms_f32`, plus `l2_norm`, `rope_neox`, `conv_seq`,
  `conv_state`, `router_topk`, `ssm_gates`, `moe_combine_seq`, `kv_store`.
- No `gemmi_*`/`gemmis_*` and no `quant_q8`: as designed, the sm_70 build has no int8 prefill
  GEMM (`Backend::mmq` is false there and `gemm_raw` takes the f16 `gemm_*` path).
- `gemvn_*` was **not** launched by either command (the lm head ran as `gemv_q4_k`, one input),
  so its 98304 B opt-in was set but the kernel itself is not yet exercised.

### Test suite (same window)

`STRATA_MODEL=$M cargo test --release --no-fail-fast`: 121 passed, 2 failed, 5 ignored,
`grep -c "SKIP: cuda"` = 0 (the CUDA backend opens in every GPU test binary). Failures:

- `backend::spirv_is_fresh` — the known pre-existing Vulkan/toolchain drift (40 of 47 committed
  `.spv` byte-differ with shaderc 2025.2-1); untouched by this task.
- `gpu_model::prefill_matches_reference` — fails on **Vulkan only**: at the step after a 73-token
  prefill the Vulkan argmax is 15019 where CPU and CUDA give 4938 (`max |dlogit|` 1.4263 prefill
  vs 1.4200 decode). Reproduced identically on the base commit `f295a6c` with
  `STRATA_TEST_BACKENDS=vulkan`, so it predates M1-T7 and is outside its scope (Vulkan numerics).

Cross-backend GEMM figures now recorded by `kernels::gemm_matches_reference_for_every_type`
(each backend matches its own 1e-4 reference; the cross-backend figure is kept as data):

- sm_70 CUDA f16 vs Vulkan int8, same tensor, different GEMM math: 3.17e-3 - 4.08e-3 relative
  (the int8 quantization of X, not summation order).
- f16 path with different input conversions (CUDA `__float2half` rounds, this driver's
  `OpFConvert` truncates): 5.41e-4 (IQ2_XS), 4.79e-5 (F32).
- CUDA vs CUDA-path references: 2.6e-7 - 8.2e-7 relative per backend.

No new `Xid`/`NVRM` line in `dmesg` before or after any run; the GPU was left at 0 MiB.

## 2026-11-07 — M1-T8 CUDA parity baseline (V100, sm_70)

Commit `ae95739` (v100, branch `task/22-cuda-parity`, issue #22) | driver 580.178.04 |
CUDA toolkit 12.9.2 (nvcc 12.9.86) | NVRTC 12.9.86
Window: production (`llama.router.service`) stopped by the coordinator, V100 free (0 MiB,
0 %); no other process on the card. Parity run 124 s, whole sequence (integrity check,
20 `strata score` invocations, artifact) 2 min 04 s wall.

```
strata ae95739, --gpu --backend cuda --llama-numerics --fixed-experts, tag cuda-sm70
reference: target/parity-ref.json (sha256 0c30267c…, llama.cpp 0.9.0-dev build 12110
           cd959bae9, FA disabled by the build) reused as-is; llama-server not started
           (0 cache misses, 0 "ref <n> prompt tokens" lines)
20 prompts x 32 greedy tokens = 640 positions
strict vs llama.cpp token-by-token: 630/640 = 98.44%
either llama.cpp mode:              636/640 = 99.38%
llama.cpp self-agreement:           630/640 = 98.44%
verdict: PASS (636/640, above the 634/640 either-mode bar)
Xid: none in the window (42 lines before and after, last Fri Nov 6 20:40:46 2026)
```

Reference integrity before the run: `sha256sum` matched both M0-T7 values
(`0c30267cb673…` / `ba5e28ac2368…`), 20 × 32 rows, `prompt_tokens` equal to
`docs/parity/vk-sm70.json` ([26,33,54,42,27,36,31,23,26,31,19,44,24,20,995,444,1367,281,
553,541]), llama.cpp self-agreement 630/640. `tools/parity.py` printed no cache-miss line,
so the engine was the only process that touched the model. First `strata score` paid the
cold NVRTC/cubin compile (sm_70, ~10 s); the remaining 19 reuse the cache.

Per-prompt results for `cuda-sm70` (parity mode, same reference as `vk-sm70`):

| # | prompt | prompt tokens | agree/n | first_miss | top2_gap_at_miss |
| --- | --- | --- | --- | --- | --- |
| 1 | Write a Rust function that returns the n-th Fibona | 26 | 32/32 | None | None |
| 2 | In Python, how do I read a CSV file and compute th | 33 | 32/32 | None | None |
| 3 | Explain what this C code does: int f(int *a, int n | 54 | 32/32 | None | None |
| 4 | Write a SQL query that returns the top 5 customers | 42 | 32/32 | None | None |
| 5 | Write a regular expression that matches an IPv4 ad | 27 | 32/32 | None | None |
| 6 | Fix the bug: def avg(xs): return sum(xs) / len(xs) | 36 | 32/32 | None | None |
| 7 | Give a bash one-liner that finds the 10 largest fi | 31 | 31/32 | 9 | 0.0329 |
| 8 | Why is the sky blue? Answer in three sentences. | 23 | 32/32 | None | None |
| 9 | Розкажи коротко про історію Києва. | 26 | 32/32 | None | None |
| 10 | Объясни, что такое квантовая запутанность, простым | 31 | 28/32 | 3 | 0.5055 |
| 11 | 请用简单的语言解释什么是机器学习。 | 19 | 30/32 | 23 | 0.0837 |
| 12 | A train leaves at 9:15 and travels 180 km at 72 km | 44 | 31/32 | 5 | 0.0564 |
| 13 | Define the word 'entropy' as used in information t | 24 | 32/32 | None | None |
| 14 | Give me a simple recipe for pancakes. | 20 | 31/32 | 11 | 0.1449 |
| 15 | Summarize this design document in five bullet poin | 995 | 32/32 | None | None |
| 16 | What license terms apply here? Summarize: | 444 | 32/32 | None | None |
| 17 | Find the error in this log: | 1367 | 31/32 | 13 | 0.2825 |
| 18 | Which of these cities is the largest by population | 281 | 32/32 | None | None |
| 19 | List the keys whose value is false: | 553 | 32/32 | None | None |
| 20 | Which row has the highest score? | 541 | 32/32 | None | None |

The 10 strict misses sit on 6 prompts (7, 10, 11, 12, 14, 17); six of them (prompt 10
positions 3 and 12, prompt 11 position 23, prompt 12 position 5, prompt 14 position 11,
prompt 17 position 13) are the position the second llama.cpp mode (context as one batch,
`target/llama-batch-argmax.json`) also produces, which is what carries the either-mode
count to 636/640. Prompt 10 (4 misses, 0.5055 top-2 gap at the first) is not a near-tie.

Comparison with the M0-T7 Vulkan run (same reference, same flags; `d6c2beb`, strict
625/640 = 97.66 %, either 629/640 = 98.28 %):

- CUDA strict is 630/640, i.e. exactly llama.cpp's own self-agreement (630/640), 5
  positions better than Vulkan; either-mode is +7 (636 vs 629).
- Token-level CUDA-vs-Vulkan divergence: 9/640 positions, on 5 prompts.
  - prompt 12 (pos 5): CUDA takes the second llama.cpp mode, Vulkan the reference.
  - prompt 16 (pos 0, 23), prompt 18 (pos 30), prompt 19 (pos 0, 3): CUDA matches the
    reference, Vulkan does not.
  - prompt 17 (pos 8, 13, 16): CUDA matches the reference at 8 and 16, Vulkan at 13;
    both miss one position of that 32-token continuation.
  - Prompts 7, 10, 11, 14 are position-identical in both backends, so the Vulkan
    deviation diagnosed by M3-T5 is not inherited by CUDA on this prompt set.
- This is the first numerical measurement of the CUDA `--llama-numerics` path
  (`attn_prefill_llama` / `attn_decode_llama` on sm_70); it completed with no NVRTC
  error, no launch failure, no NaN and no `strata score` failure (20/20 invocations,
  exit 0).

No new `Xid`/`NVRM` line in `dmesg` (`grep -ic Xid` = 42 before and after the window, the
tail still ends at `Fri Nov 6 20:40:46 2026`). No device lost, no engine crash; the GPU was
left at 0 MiB. Artifact: `docs/parity/cuda-sm70.json` (`gate: pass`; the generated
`target/parity-result-cuda-sm70.json` is not committed).

## 2026-11-07 — Context scaling, CUDA, KV q8 (V100, sm_70)

Model: `Qwen3.8-27B-UD-Q4_K_XL.gguf`. Build: `v100` at `b055eff` (M1 complete;
CUDA uses the sm_70 FMA fallback, no tensor cores until M2).

```
strata bench --gpu --backend cuda --kv q8 --ctx 262272 \
  --pp 32768,65536,131072,262144 --tg 128 --reps 1

plan: weights+state+scratch 25280 MB (KV 8708 MB for 262272 tokens, prefill 706 MB)
pp  32768: prefill 104.3 tok/s | tg 128: decode 25.0 tok/s
pp  65536: prefill  79.5 tok/s | tg 128: decode 22.0 tok/s
```

128k and 256k were not completed: the run was stopped at the production
window limit. Estimated on this build: 128k ~30-40 min, 256k ~80-100 min of
prefill. KV q8 at 256k is 8708 MB; the plan fits in 32 GB.

### Follow-up: 128k measured, 256k not completed

```
strata bench --gpu --backend cuda --kv q8 --ctx 262272 \
  --pp 131072,262144 --tg 128 --reps 1

pp 131072: prefill 53.9 tok/s | tg 128: decode 17.8 tok/s
```

256k did not complete: the prefill was still running when the production
window closed and the run was stopped. At this rate it needs roughly 100-120
minutes of prefill on this build (sm_70 FMA fallback).

## 2026-11-08 — Volta tensor cores in the prefill GEMM (M2-T1, V100, sm_70)

Model: `Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf`. Build: `task/24-cuda-tensor-core-gemm`
(PR #25), based on `v100` at `58fc084`. The f16 prefill GEMM (`gemm_q`,
`gemm_*`/`gemms_*`) now uses `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32`
on sm_70; the `m16n8k16` path is unchanged for sm_80+ and `STRATA_GEMM_FMA=1`
selects the old CUDA-core fallback.

### Fragment mapping

The per-lane mapping is the PTX ISA 8.5 §9.7.16.5.1 one (four computations
selected by `g = lane >> 2 & 3`): A row `t + 4*(g >= 4)` (row-major, k 0..3),
B column `t + 4*(g >= 4)` (column-major, k 0..3), and D register
`i = 4*p4 + 2*p2 + e` at row `(t & 1) + 2*p2 + 4*(g >= 4)`, column
`(t & 2) + 4*p4 + e`, with `t = lane & 3`. `SM == false` maps the four
computations to the four n8 tiles (`m8 = j`, `n8 = c = g & 3`, 16 ops per
16-deep k step); `SM == true` maps them to the four 8x8 quadrants of the 16x16
warp tile (`m8 = c >> 1`, `n8 = c & 1`, 4 ops). The standalone asm compiled by
NVRTC 12.9 for sm_70 lowers to `HMMA.884.F32.F32.STEP0..3` (`cuobjdump -sass`),
and the sm_80+ preprocessed source is byte-identical to `v100` (1861 lines).

### Correctness (device-free + V100)

```sh
STRATA_MODEL=$M STRATA_TEST_BACKENDS=cuda cargo test --release --test kernels \
  gemm_matches_reference_for_every_type -- --nocapture --test-threads=1

cuda device 0: Tesla V100-SXM2-32GB, sm_70 (Volta, m8n8k4)
Q5_K     blk.0.attn_gate.weight: gemm rel err 9.58e-7
IQ3_XXS  blk.0.ffn_down_exps.weight: gemm rel err 6.31e-7
Q6_K     blk.0.ffn_down_shexp.weight: gemm rel err 6.47e-7
IQ2_XS   blk.0.ffn_gate_exps.weight: gemm rel err 8.95e-7
F32      blk.0.ffn_gate_inp.weight: gemm rel err 4.87e-7
IQ4_XS   blk.1.ffn_down_exps.weight: gemm rel err 6.42e-7
Q8_0     blk.1.ffn_down_shexp.weight: gemm rel err 6.01e-7
Q4_K     output.weight: gemm rel err 9.13e-7
ok
```

All relative errors are `<= 1e-6`, against the 1e-4 assertion; both `gemm_*`
and `gemms_*` (including partial tiles) are covered. `strata generate
--prompt "The capital of France is" -n 32` produced coherent text (" Paris. ...
That is correct.") with no NaN. CPU side, `cargo test --release --lib cuda::`
compiles sm_70 with `GEMM_TC=1`, sm_70 with `GEMM_TC=0` and sm_80 with NVRTC
12.9 (14/14 tests green).

### Benchmarks (same window, median of 3 reps)

```sh
M=/mnt/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf
target/release/strata bench --model "$M" --gpu --backend cuda \
  --pp 512,4096 --tg 128 --reps 3 --ctx 32896            # GEMM_TC = 1
STRATA_GEMM_FMA=1 target/release/strata bench ...        # FMA fallback
```

| run | pp 512 (tok/s) | pp 4096 (tok/s) | tg 128 @512 (tok/s) | tg 128 @4096 (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| `GEMM_TC = 1` (m8n8k4) | 1880.4 (1767.1-1948.2) | 1976.0 (1974.1-1977.6) | 123.0 (121.8-123.7) | 119.6 (118.6-120.1) |
| `STRATA_GEMM_FMA=1` | 840.9 (839.5-842.2) | 1036.1 (1034.9-1037.3) | 123.3 (122.3-123.9) | 119.3 (119.0-119.6) |
| delta | **+123.6 %** | **+90.7 %** | -0.2 % | +0.3 % |

Prefill is 2.24x faster at 512 and 1.91x at 4096; decode is unchanged within
noise (99.8 % and 100.3 % of the FMA run, as expected: decode does not use
`gemm_q`). The absolute 512 value is also 2.25x the M1-T7 record of 836.5 tok/s
(`docs/V100-results.md`, M1-T7; same model, same protocol) because that record
was taken on the FMA fallback.

No new `Xid`/`NVRM` line in `dmesg` (`grep -ic Xid` = 42 before and after the
window), no device lost, no engine crash; the GPU was left at 0 MiB and
`llama.router.service` was restarted by the coordinator after the window.

## 2026-11-08 — Volta tensor cores in the prefill attention (M2-T2, V100, sm_70)

Model: `Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf`. Build: `task/26-cuda-attn-tc`
(PR #27), based on `v100` at `1aaf6af`. `attn_prefill_t` now computes both its
products (QKᵀ and PV) with `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32`
on sm_70, under the same `GEMM_TC` switch as the M2-T1 prefill GEMM; the
`m16n8k16` path stays for sm_80+ and `STRATA_GEMM_FMA=1` selects the
CUDA-core fallback for both.

### Fragment mapping

Constants as in `gemm_q` (`l = lane`, `t = l & 3`, `g = l >> 2`, `c = g & 3`,
`hi = g >> 2`). QKᵀ: A = Q rows `8*j + t + 4*hi` of the 16-row warp tile, B = K
row `8*(4*ng + c) + t + 4*hi`, eight `mma884` per 16-deep k step (2 `j` x 4
`ks4`), accumulators `s2[ng][j][4*p4 + 2*p2 + e]` =
`S[8*j + (t & 1) + 2*p2 + 4*hi][8*(4*ng + c) + (t & 2) + 4*p4 + e]`. Softmax in
that layout: rows `qp2[j][p2]`, keys `k0 + 8*(4*ng + c) + (t & 2) + 4*p4 + e`,
row max/sum across the eight lanes of a row with `__shfl_xor_sync(.., 2/4/8)`
(`xor 2` swaps `t & 2`, `xor 4`/`8` swap `c`), `mT`/`lT` of four entries each,
`lT` reduced once after the last tile. P is staged per warp into a
`FA_BQ x (FA_BK + 8)` half tile appended after `vs` in `fa_sm` (5120 B,
`__syncwarp()` before the reads); PV reads A as two `ld_s32` from it, B from
the transposed V tile as four `ld.shared.u16` (`vs + (4*kg + {0..3})*PQ + ...`),
64 `mma884` per tile at HD = 128, accumulators
`o2[j][grp][4*p4 + 2*p2 + e]`, and the store loop mirrors `gemm_q`'s sm_70 one.
SASS (`cuobjdump -sass`, NVRTC 12.9 sm_70 cubin): `attn_prefill_128` has 128
`HMMA.884.F32.F32` (64 QKᵀ + 64 PV, the same FLOPs as 32 `m16n8k16`),
`REG:159 LOCAL:0` (FMA build: `REG:248`). The sm_80+ preprocessed source is
byte-identical to `v100` (1861 lines) and the sm_70 `GEMM_TC=0` variant is
source-identical (1763 lines); `kernels.cu` is +187 lines and -0.

### Correctness (same window)

```sh
M=/mnt/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf
STRATA_MODEL=$M STRATA_TEST_BACKENDS=cuda cargo test --release --test kernels \
  attn_prefill_matches_decode -- --nocapture --test-threads=1
STRATA_GEMM_FMA=1 STRATA_MODEL=$M ... (same test)
STRATA_MODEL=$M cargo test --release --test kv_q8 -- --test-threads=1
```

`attn_prefill_{hd}` rel err vs decode, assertion `<= 5e-4`:

| case | `GEMM_TC = 1` | `STRATA_GEMM_FMA=1` |
| --- | ---: | ---: |
| hd 64, nh 4/nkv 2 | 6.42e-5 | 6.44e-5 |
| hd 128, nh 4/nkv 2 | 7.26e-5 | 7.39e-5 |
| hd 256, nh 4/nkv 2 | 6.37e-5 | 6.40e-5 |
| hd 256, nh 16/nkv 2 | 6.11e-5 | 6.11e-5 |
| hd 128, nh 8/nkv 8 | 6.13e-5 | 6.17e-5 |
| hd 128, nh 32/nkv 8 | 6.24e-5 | 6.26e-5 |
| hd 128, nh 32/nkv 8, pos 2100 | 2.98e-5 | 2.83e-5 |
| hd 256, nh 16/nkv 2, pos 1000 | 3.51e-5 | 3.36e-5 |
| hd 256, nh 24/nkv 2, pos 0 | 6.14e-5 | 6.18e-5 |

The decode reference itself is `8.7e-8 … 1.4e-7` against the CPU. The two
implementations agree with decode in the third significant digit of the error
metric, i.e. the difference between them is far below the f16 rounding error.
`kv_q8`: `q8 prefill vs decode` 6.16e-5 / 7.67e-5 / 6.16e-5 / 5.98e-5
(hd 64 / 128 / 256 / nh 8,nkv 8), `q8 vs f16 decode` 1.55e-3 … 2.06e-3
(cache quantisation, unchanged). The QSA mask path
(`qwen4exp_qsa_gpu_matches_cpu`) was **not run**: the qwen4exp GGUF is not
present on this host.

### Benchmarks (median of 3 reps, `--pp 512,4096,32768 --tg 128 --ctx 32896`)

| run | pp 512 | pp 4096 | pp 32768 | tg 128 @512 | @4096 | @32768 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `GEMM_TC = 1` (m8n8k4) | 2045.3 | 2795.3 | 2263.7 | 123.1 | 119.1 | 105.5 |
| `STRATA_GEMM_FMA=1` | 833.6 | 1030.7 | 489.2 | 121.9 | 118.4 | 105.3 |
| delta | **+145.3 %** | **+171.2 %** | **+362.7 %** | +1.0 % | +0.6 % | +0.2 % |

Against M2-T1 (same model and protocol, `docs/V100-results.md` block above):
prefill @512 1880.4 → **2045.3** (+8.8 %), @4096 1976.0 → **2795.3** (+41.5 %),
so the M2 exit target of 2000 tok/s at 4096 is met with the attention on the
tensor cores. Decode is unchanged within noise (decode does not use
`attn_prefill_t`). Reps ranges: 2034.8-2062.3 / 2786.7-2803.5 / 2263.1-2264.2.

`strata generate --prompt "The capital of France is" -n 32` (TC build) produced
"That is correct. Paris is the capital and most populous city of France. …"
at 123.38 tok/s, no NaN.

### Kernel time (`nsys profile --trace=cuda`, pp 4096, 10 prefill chunks)

```sh
nsys profile --trace=cuda -o nsys_tc target/release/strata bench --model "$M" \
  --gpu --backend cuda --pp 4096 --tg 1 --reps 1 --ctx 4200
nsys stats --report cuda_gpu_kern_sum nsys_tc.nsys-rep
```

| kernel | `GEMM_TC = 1` | `STRATA_GEMM_FMA=1` | delta |
| --- | ---: | ---: | ---: |
| `attn_prefill_256` self time per launch | 6.124 ms | 70.760 ms | **-91.3 %** |
| `attn_prefill_256` share of kernel time | 4.5 % | 18.3 % | |
| total kernel time (10 chunks) | 1.371 s | 3.869 s | -64.6 % |

The per-kernel requirement (at least 20 % lower self time per launch) is met by
a wide margin. The end-to-end prefill gain (+171 % vs the FMA build) also
includes the M2-T1 `gemm_q` win, since this build carries both.

No new `Xid`/`NVRM` line in `dmesg` (`grep -ic Xid` = 42 before and after both
windows), no device lost, no NaN in any of the 3 reps; the GPU was left at
0 MiB and `llama.router.service` was left stopped for the coordinator to
restart.
