# PR: Vulkan Wave32 Kernel Parity, Native Rust EAGLE3 Safetensors Loader, and Hybrid N-gram + EAGLE3 Cascade (`feat/all-improvements`)

- **Base branch**: `route-c` (`e22aec5`)
- **Feature branch**: `feat/all-improvements`
- **Hardware verified**: `AMD Radeon 890M Graphics (RADV STRIX1)` (`gfx1150`, Mesa `26.2.6-arch3.1`)
- **Target model**: `Qwen3-30B-A3B-Instruct-2507-IQ4_XS.gguf` (48 MoE layers, `32x128` heads, `kv=4`, `n_vocab=151,936`)
- **EAGLE3 draft tested directly from `.safetensors`**: `angelslim/model.safetensors` + `config.json` (`277.2 MiB` converted GGUF, `extract_layers=[2, 24, 45]`, `heads=32x128`, `kv=4`, `ff=6144`, `draft_vocab=32000`)
- **Production isolation**: All server tests/benchmarks ran on `127.0.0.1:8013` while production `strata-route-c` (`:8011`) remained untouched and healthy (`{"slots":1,"status":"ok","waiting":0}`).

---

## Summary of Changes

### 1. Vulkan Wave32 Quantizer Rounding Fix (`tests/kernels.rs`: 26/26 Passing on RADV `gfx1150`)
- **Symptom**: On `AMD Radeon 890M Graphics (RADV STRIX1)` (`minSubgroupSize = 32`, `W64 = false`), `cargo test --release --test kernels` failed `gemv_partial_superblocks_and_q2_0` (`Q2_0 cols 640: gemm rel err 1.24e-4`) and all 18 `Gpu::mmq(ty)` dense GEMM checks in `vulkan_type_matrix` (`rel err 1.16e-4 .. 2.08e-4`).
- **Root cause**: `src/vulkan/shaders/common.glsl` (`rcp_d`) used single-NR `fdiv(a, b)` when `W64 == false` instead of `float(double(a) / double(b))`, perturbing `127.0 / mx` by 1 ULP on exact/near-tie `roundEven()` inputs (e.g. `mx = 127/128` -> `63.5` ties in rows `96, 97, 118` of `xm`). Meanwhile `src/vulkan/shaders/attn_merge.comp` called `fdiv` directly in its fused `xq` quantizer instead of `rcp_d`.
- **Fix**:
  - Updated [`src/vulkan/shaders/common.glsl`](../src/vulkan/shaders/common.glsl) so `rcp_d(a, b)` uses IEEE-754 `float(double(a) / double(b))` on both wave64 and wave32 devices (`shaderFloat64` is already mandatory in `Vk::unusable()`).
  - Updated [`src/vulkan/shaders/attn_merge.comp`](../src/vulkan/shaders/attn_merge.comp) to call `rcp_d` instead of `fdiv` for `xq` quantization so `attn_decode_q` matches `quant_xq` bit-for-bit.
  - Recompiled SPIR-V via `tools/gen_spirv.sh` (`attn_merge.spv`, `kv_store.spv`, `norm_gate.spv`, `quant_q8.spv`, `quant_xq.spv`, `rms_norm.spv`).

### 2. Native Rust EAGLE3 `.safetensors` + `config.json` Auto-Converter (Zero Python)
- **Added**: [`convert_safetensors_eagle3_to_gguf`](../src/gguf.rs) in `src/gguf.rs` to parse HuggingFace `.safetensors` headers + `config.json` directly in Rust (`memmap2` + `serde_json` + `half`), preserve 2D linear weights in native `BF16` (with `F32`/`F16` conversion support), convert 1D norms to `F32`, convert relative `d2t` (`I64`/`I32`) offsets to absolute `I32` target vocab IDs, set `eagle3.*` GGUF v3 metadata, and atomically write via `.gguf.tmp` rename.
- **Wired**: Updated [`open_mtp_gguf`](../src/main.rs) in `src/main.rs` to auto-convert `.safetensors` + `config.json` when `model.gguf` is absent (or `STRATA_EAGLE3_RECONVERT=1` is set).
- **Tested**: Added unit test `model::tests::eagle3_safetensors_auto_converts_and_loads` in [`src/model.rs`](../src/model.rs) and verified live conversion on `eagle3-drafts/angelslim/model.safetensors` (`277.2 MiB`).

### 3. Hybrid Speculative Decoding Cascade (`N-gram` + `EAGLE3/MTP`)
- **Added**: [`GpuModel::hybrid_spec_step`](../src/gpu_model.rs) in `src/gpu_model.rs` enabling simultaneous `--spec-type ngram-*` and `--mtp-model ... --mtp K`:
  - Attempts context N-gram / dynamic lookup bursts first (zero GPU draft cost; up to `draft_max.min(TMAX - 1)` on pure-attention models, capped to `k.min(SNAP_MAX)` on DeltaNet models).
  - Advances `MtpHead` over `[pend.., t]` via `mtp_run(..., false)` prior to `verify`/`verify_greedy` so `MtpHead` KV stays synchronized across N-gram bursts.
  - Falls back to `spec_step` (`EAGLE3/MTP`) whenever `NgramDrafter` returns empty, and feeds verified accepted tokens back into `NgramDrafter::accept`.
  - Expanded `ngram_spec_step` room calculation to honor `draft_max` cleanly alongside `k`.
- **Wired & Tested**: Removed legacy mutual-exclusion error in [`src/main.rs`](../src/main.rs), routed `mtps + ngram` through `hybrid_spec_step` in `decode`, and added `hybrid_spec_matches_greedy` in [`tests/gpu_model.rs`](../tests/gpu_model.rs).

---

## Verification

### Full Release Test Suite (`cargo test --release --lib --bins --tests`)

Executed after final code edits across all 17 test targets (`cargo build --release` + `cargo test --release --lib --bins --tests`): **194 passed, 0 failed, 5 ignored** (`#[ignore]` upstream throughput/benchmark tests in `tests/kernels.rs` [4] and `tests/model.rs` [1]).

| Test Target | Result |
|---|---|
| `src/lib.rs` (includes `eagle3_safetensors_auto_converts_and_loads`) | `61 passed, 0 failed` |
| `src/main.rs` | `9 passed, 0 failed` |
| `tests/kernels.rs` (on `RADV STRIX1 gfx1150`) | `26 passed, 0 failed, 4 ignored` |
| `tests/backend.rs` (`spirv_is_fresh`, `shared_memory_matches_the_committed_spirv`) | `10 passed, 0 failed` |
| `tests/serve.rs` | `23 passed, 0 failed` |
| `tests/quant.rs` + `tests/quant_ggml.rs` | `24 passed, 0 failed` |
| `tests/gpu_model.rs` + `tests/gguf.rs` + `tests/ops.rs` | `29 passed, 0 failed` |
| Remaining integration tests (`cpu_moe`, `deepseek_*`, `kv_q8`, `llama_numerics`, `model`, `tokenizer`) | `12 passed, 0 failed, 1 ignored` |

### Benchmarks & Greedy SHA-256 Parity (`127.0.0.1:8013`)

`Qwen3-30B-A3B-Instruct-2507-IQ4_XS.gguf`, `STRATA_EXPERT_MB=1500` (`626/6144` resident experts = `10.4%` hit rate alongside live `:8011`), `--ctx 4096 --kv q8 --fixed-experts`, `temperature=0.0, top_k=1, seed=42`:

| Arm | (a) New Text (180 tok) | (b) Code Rewrite (260 tok) | (c) P3 Anti-Loop (200 tok) | Overall Draft Accept | Greedy Parity vs Baseline |
|---|---:|---:|---:|---:|---|
| `baseline_mtp0` (`--mtp 0`) | `7.96 tok/s` (`22.61 s`) | `7.39 tok/s` (`35.21 s`) | `4.89 tok/s` (`40.90 s`) | — | baseline (`ab5e75a7a983` / `5f86857ebc8d` / `4e1c4a05a335`) |
| `eagle3_safetensors_direct` (`STRATA_EAGLE3_RECONVERT=1 --mtp-model .../angelslim/model.safetensors --mtp 1`) | `6.40 tok/s` (`0.80x`, `60.7%`) | `7.91 tok/s` (`1.07x`, `89.1%`) | `5.19 tok/s` (`1.06x`, `68.1%`) | `272/369` (`73.7%`) | **3/3 exact SHA-256** |
| `hybrid_angelslim_ngram_simple` (`+ --spec-type ngram-simple --spec-draft-n-max 15`) | `7.15 tok/s` (`0.90x`, `62.6%`) | **`11.93 tok/s` (`1.61x`, `99.2%`)** | `5.00 tok/s` (`1.02x`, `58.8%`) | `405/513` (`78.9%`) | **3/3 exact SHA-256** |
| `hybrid_angelslim_ngram_cache` (`+ --spec-type ngram-simple,ngram-cache --lookup-cache-dynamic ...`) | **`10.06 tok/s` (`1.26x`, `93.0%`)** | `11.06 tok/s` (`1.50x`, `95.2%`) | **`7.05 tok/s` (`1.44x`, `79.9%`)** | **`599/670` (`89.4%`)** | **3/3 exact SHA-256** |

Result: **12/12 runs (all 9 speculative runs across all modes) produced bit-identical greedy SHA-256 hashes**, `tests/kernels.rs` now passes **26/26** on RADV `gfx1150`, `--mtp-model` converts HuggingFace EAGLE3 `.safetensors` natively in Rust without Python, and hybrid N-gram + EAGLE3 raises overall draft acceptance from `73.7%` to **`89.4%`** (up to **`99.2%`** / **`1.61x`** on code rewrite and **`1.44x`** on P3 anti-loop).
