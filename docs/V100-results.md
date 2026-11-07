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
