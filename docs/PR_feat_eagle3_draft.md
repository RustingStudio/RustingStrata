# PR: Native EAGLE3 Draft Support for `--mtp-model` (`feat/eagle3-draft`)

- **Base branch**: `fix/ngram-48layers`
- **Target branch**: `feat/eagle3-draft`
- **Hardware verified**: `AMD Radeon 890M Graphics (RADV STRIX1)` (`gfx1150`, Mesa `26.2.6-arch3.1`)
- **Target model**: `Qwen3-30B-A3B-Instruct-2507-IQ4_XS.gguf` (48 MoE layers, `32x128` heads, `kv=4`, `n_vocab=151,936`)
- **Drafts verified**:
  - `angelslim`: `heads=32x128`, `kv=4`, `ff=6144`, `draft_vocab=32000`, `extract_layers=[2, 24, 45]`
  - `tengyunw`: `heads=32x64`, `kv=4`, `ff=12288`, `draft_vocab=32000`, `extract_layers=[2, 24, 45]`
- **Production isolation**: All benchmarks ran on `127.0.0.1:8013` while production `strata-route-c` (`:8011`) remained untouched and healthy (`{"slots":1,"status":"ok","waiting":0}`).

---

## Blockers Resolved

1. **Three-layer hidden-state fusion (`extract_layers` + `fc`)**:
   EAGLE3 concatenates pre-layer hidden states from 3 target layers (`[2, 24, 45]`, `3 * 2048 = 6144`) and projects through `eagle3.fc.weight` (`6144 -> 2048`) instead of `nextn.enorm + nextn.hnorm` (`2 * 2048 -> 2048`).
2. **Independent draft attention geometry & dense FFN**:
   EAGLE3 checkpoints such as `tengyunw` use different attention head dimensions (`32 x 64`) and dense FFN intermediate sizes (`ff=12288`) than the MoE target (`32 x 128`, 128 routed experts). Draft attention, NeoX RoPE, and KV buffers now read geometry from `Eagle3Config` populated from draft weights rather than inheriting target `ModelConfig`.
3. **Draft-to-target vocabulary remap (`d2t`)**:
   EAGLE3 heads output logits over a compact `draft_vocab=32000`; `d2t` (`[32000]` `I32`/`I64`) maps draft argmax tokens back into target token IDs (`151,936`).
4. **CLI/Server `--mtp-model` wiring**:
   Fixed `serve_model` in `src/main.rs` which previously called `Model::load(&g)` and dropped `--mtp-model`, centralizing resolution of `.gguf`, model directories (`model.gguf`), or HF checkpoints with sibling `model.gguf` via `open_mtp_gguf`.

---

## Architecture & File Changes

- [`tools/convert_eagle3_to_gguf.py`](../tools/convert_eagle3_to_gguf.py): Converts HF EAGLE3 draft checkpoints (`.safetensors` or `.bin` + `config.json`) to GGUF v3 preserving native `BF16` linear weights, `F32` norms, NeoX Q/K layout (`eagle3.rope.neox = true`), `eagle3.target_layers`, and absolute `I32` `d2t` IDs (`idx + offset`).
- [`src/gguf.rs`](../src/gguf.rs): Added `Gguf::raw_bytes` to read raw `d2t` integer tensors (`I32` / `I64`).
- [`src/config.rs`](../src/config.rs): Added `Eagle3Config` and `ModelConfig::eagle3: Option<Eagle3Config>`.
- [`src/weights.rs`](../src/weights.rs): Added EAGLE3 auxiliary tensor remapping (`eagle3.fc.weight`, `eagle3.output.weight`, `nextn.shared_head_norm.weight`, `nextn.enorm.weight`, `nextn.hnorm.weight`, `blk.{n_layer}.*`) in `Weights::new_with_aux` and skipped `d2t`/`t2d` in `Weights::validate`.
- [`src/model.rs`](../src/model.rs): Populated `cfg.eagle3` (`n_nextn = 1`) in `Model::load_with_aux` from draft metadata/tensor shapes and loaded `d2t`; added unit test `model::tests::load_eagle3_aux_config`.
- [`src/gpu_model.rs`](../src/gpu_model.rs): Added `Eagle3Gpu` state (`cur_g`, `feat`, `fc`, `output`), dense draft FFN + draft-sized KV allocation, 3-layer capture into `eg.feat` + `fc` projection in `forward`, `d2t` remap in `mtp_run`, and single-layer EAGLE3 decode in `mtp_forward`.
- [`src/gpu_model/prefill.rs`](../src/gpu_model/prefill.rs): Added `Scratch::feat` (`[chunk, 3 * ne]`), pre-layer hidden-state capture + `fc` projection during prefill, EAGLE3 draft KV catch-up across `TMAX` chunks, and `eg.cur_g` handoff.
- [`src/main.rs`](../src/main.rs): Added `open_mtp_gguf` and wired `--mtp-model` across `serve_model` and CLI commands (`Run`, `Chat`, `Compare`, `Bench`, `Perplexity`).

---

## Verification

### Automated Checks

- `python3 -m py_compile tools/convert_eagle3_to_gguf.py`: exited `0`.
- Release test suite (`cargo test --release --lib --bins --test backend --test cpu_moe --test deepseek_forward --test deepseek_load --test gguf --test gpu_model --test kv_q8 --test llama_numerics --test model --test ops --test quant_ggml --test quant --test serve --test tokenizer`):
  - **166 passed, 0 failed, 1 ignored** (`tests::mtp_draft_acceptance` in `tests/model.rs`), including `model::tests::load_eagle3_aux_config`.
  - Note: `tests/kernels.rs` was omitted due to 2 pre-existing fp16 tolerance failures on `RADV STRIX1`.

---

## Before / After Benchmarks & Greedy Parity (`127.0.0.1:8013`)

All requests used `temperature=0.0, top_k=1, seed=42`, `--ctx 4096 --kv q8 --fixed-experts`.

### 1. Constrained Expert Cache (`STRATA_EXPERT_MB=1500`, `626/6144` resident experts, `10.4%` hit rate)

| Arm | (a) New Text (180 tok) | (b) Code Rewrite (260 tok) | (c) P3 Anti-Loop (200 tok) | Overall Draft Accept | Greedy Parity vs Baseline |
|---|---:|---:|---:|---:|---|
| `baseline_mtp0` (`--mtp 0`) | `4.35 tok/s` (`41.40 s`) | `4.43 tok/s` (`58.66 s`) | `3.20 tok/s` (`62.46 s`) | — | baseline (`a3f621a5ce5a` / `5f86857ebc8d` / `4e1c4a05a335`) |
| `angelslim_mtp1` (`--mtp 1`) | `5.77 tok/s` (`1.33x`, `51.3%`) | **`7.24 tok/s` (`1.63x`, `87.8%`)** | `4.67 tok/s` (`1.46x`, `68.1%`) | **`264/377` (`70.0%`)** | **3/3 exact SHA-256** |
| `angelslim_mtp2` (`--mtp 2`) | `4.57 tok/s` (`1.05x`, `28.3%`) | `5.32 tok/s` (`1.20x`, `57.4%`) | `3.63 tok/s` (`1.13x`, `42.6%`) | `296/688` (`43.0%`) | **3/3 exact SHA-256** |
| `tengyunw_mtp1` (`--mtp 1`, `32x64`) | `5.27 tok/s` (`1.21x`, `48.8%`) | `5.66 tok/s` (`1.28x`, `56.6%`) | **`4.68 tok/s` (`1.46x`, `65.3%`)** | `232/408` (`56.9%`) | **3/3 exact SHA-256** |

### 2. Expanded Expert Cache (`STRATA_EXPERT_MB=4000`, `1672/6144` resident experts, `27.3%` hit rate alongside live `:8011`)

| Arm | (a) New Text (180 tok) | (b) Code Rewrite (260 tok) | (c) P3 Anti-Loop (200 tok) | Overall Draft Accept | Greedy Parity vs Baseline |
|---|---:|---:|---:|---:|---|
| `4g_baseline_mtp0` (`--mtp 0`) | `3.30 tok/s` (`54.57 s`) | `2.94 tok/s` (`88.33 s`) | `2.56 tok/s` (`78.26 s`) | — | baseline (`b5b63ad73427` / `5f86857ebc8d` / `4e1c4a05a335`) |
| `4g_angelslim_mtp1` (`--mtp 1`) | **`4.53 tok/s` (`1.37x`, `56.0%`)** | **`5.05 tok/s` (`1.72x`, `89.1%`)** | **`3.95 tok/s` (`1.54x`, `68.1%`)** | **`269/373` (`72.1%`)** | **3/3 exact SHA-256** |
| `4g_tengyunw_mtp1` (`--mtp 1`, `32x64`) | `4.52 tok/s` (`1.37x`, `53.4%`) | `3.74 tok/s` (`1.27x`, `53.8%`) | `3.74 tok/s` (`1.46x`, `65.3%`) | `233/408` (`57.1%`) | **3/3 exact SHA-256** |

Result: **15/15 speculative benchmark runs achieved 100% SHA-256 bit-identical greedy parity** against `--mtp 0`, with `--mtp 1` achieving up to **`1.72x`** speedup (`89.1%` code acceptance, `70.0%–72.1%` overall acceptance on `angelslim`; `56.9%–57.1%` overall acceptance and up to `1.46x` speedup on `tengyunw`).
