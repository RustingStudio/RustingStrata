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
