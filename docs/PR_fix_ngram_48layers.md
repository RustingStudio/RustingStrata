# PR: Fix 48-Layer MoE N-gram Crash, Disable gfx11xx Prefetch, and Honor `STRATA_EXPERT_MB` (`route-c`)

- **Base branch**: `route-c`
- **Target branch**: `fix/ngram-48layers`
- **Hardware verified**: `AMD Radeon 890M Graphics (RADV STRIX1)` (`gfx1150`, Mesa `26.2.6-arch3.1`, unified LPDDR APU)
- **Production isolation**: All tests executed on `:8013`; production `strata-route-c` (`:8011`) untouched and verified healthy (`{"slots":1,"status":"ok","waiting":0}`).

---

## Summary of Root Causes & Fixes

### Bug #1 — Panic (`index out of range`) & `rollback` limit on 48-layer MoE N-gram speculation (`Qwen3-30B-A3B`)

- **Symptom**:
  Enabling any N-gram mode (`ngram-simple`, `ngram-mod`, `ngram-cache`) on `Qwen3-30B-A3B` (48 layers, `qwen3moe`, `n_vocab = 151,936`) caused an immediate Rust panic as soon as a request arrived:
  `range end index 266828544 out of range for slice of length 175030272` at `src/weights.rs:25:40` (`Tensor::row` called from `Model::embed`), e.g. token ID `231622 * 1152 = 266828544`.
- **Root Causes**:
  1. **Cross-vocabulary lookup cache pollution**: Shared lookup caches (`~/.local/share/rusting_strata/lookup_dynamic.bin`) populated by `Qwen3.8-35B-A3B` (`n_vocab = 248,320`) contain token IDs `>= 151,936`. `NgramDrafter::draft` falls back across `nc_dynamic`/`nc_static` caches, drafting out-of-vocab token IDs directly into `Model::embed` without filtering against `m.cfg.n_vocab`.
  2. **Short-history underflow in lookup helpers**: `ngram_cache_draft` performed `total_len - LLAMA_NGRAM_STATIC` even when `total_len < LLAMA_NGRAM_STATIC`, and `ngram_cache_update` only guarded `inp_size == 0` instead of `inp_size < 2`.
  3. **Pure-attention rollback artificially capped at `SNAP_MAX = 4`**: `GpuModel::rollback` and `rollback_tokens` checked `self.k.min(SNAP_MAX)` even for pure-attention models (no `Mixer::Delta` layers) that carry no recurrent DeltaNet state and only need `self.pos = start + keep`.
- **Fixes**:
  - `src/ngram.rs`:
    - Added `NgramConfig::n_vocab: usize`, `ngram_cache_filter_vocab(&mut cache, cfg.n_vocab)` invoked in `NgramDrafter::new` (pruning `9,483` out-of-vocab entries on `Qwen3-30B-A3B` with stderr diagnostic), and post-draft truncation in `NgramDrafter::draft` before returning drafted tokens.
    - Guarded `ngram_cache_update` (`inp_size < 2`) and `ngram_cache_draft` (`total_len >= LLAMA_NGRAM_STATIC`).
    - Added unit tests `test_ngram_cache_draft_short_history` and `test_ngram_cache_vocab_filter`.
  - `src/main.rs`:
    - Populated `ngram_cfg.n_vocab = m.cfg.n_vocab` in `Engine::runner` and initialized `n_vocab: 0` in `Engine::ngram_config` (plus removed 5 `unused_mut` warnings on `Model::load_with_aux`).
  - `src/model.rs`:
    - Added bounds clamp `(token as usize).min(self.cfg.n_vocab.saturating_sub(1))` in `Model::embed`.
  - `src/gpu_model.rs`:
    - Added `GpuModel::has_delta(&self) -> bool`; when `!self.has_delta()`, `rollback` supports `1..=TMAX` directly by rewinding `self.pos = start + keep`, and `rollback_tokens` delegates directly without `SNAP_MAX` snapshot replay.

---

### Bug #2 — `STRATA_MOE_PREFETCH=1` causes `-33%` decode regression on `gfx1150` / RDNA3.5 iGPU

- **Symptom**:
  Setting `STRATA_MOE_PREFETCH=1` on `AMD Radeon 890M Graphics (RADV STRIX1)` (`gfx1150`) dropped decode throughput from `17.2` to `12.1 tok/s` (`-33%`).
- **Root Cause**:
  `Vk::moe_prefetch_supported(&self)` unconditionally returned `true`. Pre-fetching next-layer MoE experts was tuned for large server L2 caches (e.g., V100 6 MB L2) and thrashes active weights out of RDNA3/3.5 / iGPU L2 caches.
- **Fixes**:
  - `src/vulkan/mod.rs`: Detect `igpu` (`vk::PhysicalDeviceType::INTEGRATED_GPU`) and `gfx11` (AMD `0x1002` with `igpu`, coop f16 without coop i8, or GFX11/STRIX/PHOENIX/KRACKAN/NAVI3/RADEON 7/8/RX 7 device names) in `Vk::new`, and return `!self.0.gfx11 && !self.0.igpu` from `Vk::moe_prefetch_supported`.
  - `src/gpu_model.rs`: Emit an explicit `note: STRATA_MOE_PREFETCH ignored on <device> (gfx11xx/iGPU L2 cache too small for MoE expert prefetch)` when requested on an unsupported GPU and leave prefetch disabled.

---

### Bug #3 — `STRATA_EXPERT_MB=9000` ignored / clamped at `7989 MB`

- **Symptom**:
  `STRATA_EXPERT_MB=9000` had no effect on `Qwen3-30B-A3B` (`--ctx 131072 --kv q8`); expert slots remained clamped at `7989 MB` (`3342/6144` resident experts).
- **Root Causes**:
  1. `GpuModel::load_part` computed `free1.saturating_sub(vram_margin).min(cap)`, which clamped `STRATA_EXPERT_MB` down whenever `STRATA_EXPERT_MB > free1 - vram_margin`.
  2. `ExpertCache::new` placed all identically sized layers into a single contiguous `VkBuffer`, exceeding RADV's per-heap/per-buffer allocation limit once pools grew past ~8 GiB and causing `.9` retry backoff.
  3. `Inner::buffer` only attempted `DEVICE` memory types for `DEVICE` allocations instead of spilling to `HOST` (GTT) memory on unified-LPDDR iGPUs when the `DEVICE_LOCAL` carveout heap filled.
- **Fixes**:
  - `src/gpu_model.rs`: When `STRATA_EXPERT_MB > 0`, use `mb << 20` directly as `budget` instead of `.min(...)` against `free1 - vram_margin`, logging `note: STRATA_EXPERT_MB={mb} override active (reported free device heap: <free1_mb> MB)`.
  - `src/expert_cache.rs`: Replaced single-pool-per-class mapping with `pool_specs` bounded by `MAX_POOL_FULL_BYTES = 8192 << 20`, splitting large MoE caches (such as 48 layers x 128 experts = ~15.4 GiB full) across multiple <= 8 GiB buffers.
  - `src/vulkan/mod.rs`: Added `DEVICE -> HOST` fallback iteration inside `Inner::buffer` when `self.igpu` is true and `allocate_memory` fails on device-local heaps.

---

## Before / After Benchmarks & Greedy Parity

All chat requests used `temperature=0.0, top_k=1, seed=42` on `127.0.0.1:8013` with `--fixed-experts` (deterministic GPU/CPU expert residency). Loop detection checked for any >=40-char substring repeated >=4 times (`loop=False` across every run).

### 1. Full-VRAM `Qwen3-30B-A3B-Instruct-2507-IQ4_XS.gguf` (`--ctx 131072 --kv q8`, 48 layers)

| Configuration | Expert Slots / Hit Rate | Prompt (a) New Text (180 tok) | Prompt (b) Code Rewrite (260 tok) | Prompt (c) P3 Anti-Loop (200 tok) | Overall Draft Accept | Greedy Parity vs Default |
|---|---|---|---|---|---|---|
| `--spec-type none` (baseline) | `7989 MB` (`3342/6144`, `54.3%`) | `9.89 tok/s` | `10.14 tok/s` | `10.30 tok/s` (`loop=False`) | — | baseline (`5af8560b211d` / `4e9030bbd453` / `527139f03e2a`) |
| `--spec-type none` + `STRATA_MOE_PREFETCH=1` (Bug #2) | `7989 MB` (`3342/6144`, `54.3%`) | `9.99 tok/s` | `9.97 tok/s` | `10.06 tok/s` (`loop=False`) | — (pref. ignored) | **3/3 exact SHA-256** |
| `--spec-type none` + `STRATA_EXPERT_MB=9000` (Bug #3) | `8998 MB` (`3764/6144`, `61.2%`) | `10.26 tok/s` (`+3.7%`) | `10.30 tok/s` (`+1.6%`) | `10.39 tok/s` (`+0.9%`) | — (`96,642` vs `113,928` misses) | *(different GPU/CPU split)* |
| `--spec-type ngram-simple --spec-draft-n-max 16` (Bug #1) | `7989 MB` (`3342/6144`, `54.3%`) | `9.95 tok/s` (`8/24`, `33.3%`) | **`25.50 tok/s` (`2.51x`)** (`245/256`, `95.7%`) | `10.37 tok/s` (`22/50`, `44.0%`, `loop=False`) | `275/330` (`83.3%`) | **3/3 exact SHA-256** |
| `--spec-type ngram-cache --spec-draft-n-max 16` (cross-vocab 248k bin) | `7989 MB` (`3342/6144`, `54.3%`) | **`15.20 tok/s` (`1.54x`)** (`129/171`, `75.4%`) | `17.82 tok/s` (`1.76x`) (`250/368`, `67.9%`) | **`19.78 tok/s` (`1.92x`)** (`183/250`, `73.2%`, `loop=False`) | `562/789` (`71.2%`, `9,483` OOV pruned) | **3/3 exact SHA-256** |

> Note: On `STRATA_EXPERT_MB=9000`, reported free device heap was `8503 MB`; pool splitting + iGPU GTT fallback allocated `8998 MB` (`+422` resident experts, `-15.2%` expert cache misses) with zero `.9` backoff.

### 2. Constrained-Slot Parity Matrix (`Qwen3-30B-A3B-Instruct-2507-IQ4_XS.gguf`, `STRATA_EXPERT_MB=1500 --ctx 4096 --kv q8`, `626/6144` resident experts)

| Arm | (a) New Text | (b) Code Rewrite | (c) P3 Anti-Loop | Total Drafts Accepted | Greedy Parity vs `baseline_none` |
|---|---|---|---|---|---|
| `baseline_none` (`--spec-type none`) | `4.17 tok/s` | `4.30 tok/s` | `3.19 tok/s` | — | baseline (`a3f621a5ce5a` / `5f86857ebc8d` / `4e1c4a05a335`) |
| `prefetch_ignored_gfx11` (`STRATA_MOE_PREFETCH=1`) | `3.70 tok/s` | `4.43 tok/s` | `3.30 tok/s` | — | **3/3 exact SHA-256** |
| `ngram_simple_16` | `4.39 tok/s` (`33.3%`) | **`10.19 tok/s` (`2.37x`, `95.7%`)** | `3.75 tok/s` (`44.0%`) | `275/330` (`83.3%`) | **3/3 exact SHA-256** |
| `ngram_mod_16` | `4.00 tok/s` (`6.2%`) | `8.77 tok/s` (`90.6%`) | `3.45 tok/s` (`0.0%`) | `233/272` (`85.7%`) | **3/3 exact SHA-256** |
| `ngram_cache_crossvocab` | `4.52 tok/s` (`33.3%`) | `7.19 tok/s` (`64.8%`) | `5.96 tok/s` (`75.3%`) | `515/876` (`58.8%`) | **3/3 exact SHA-256** |
| `ngram_combo_16` (`ngram-simple,ngram-map-k,ngram-cache`) | **`7.19 tok/s` (`65.2%`)** | `9.80 tok/s` (`90.8%`) | **`6.46 tok/s` (`81.4%`)** | `594/748` (`79.4%`) | **3/3 exact SHA-256** |

Result: **15/15 (100%) SHA-256 bit-identical greedy parity**, zero panics, zero loops.

### 3. Cross-Architecture Non-Regression (`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, `qwen35moe`, 40-layer hybrid DeltaNet + MoE, `--ctx 4096 --kv q8`)

| Arm | Resident Experts | (a) New Text (128 tok) | (b) Code Rewrite (260 tok) | Draft Accepted | Greedy Parity |
|---|---|---|---|---|---|
| `35b_default_4k` (`--spec-type none`) | `6766/10240` (`66.0%`, `12331 MB` slots, `1735 MB` scratch) | `11.35 tok/s` (`sha=6c23d12a1fd0`) | `12.52 tok/s` (`sha=593362741d4b`) | — | baseline |
| `35b_ngram_simple16_4k` (`--spec-type ngram-simple --spec-draft-n-max 16`) | `6582/10240` (`64.3%`, `11995 MB` slots, `2070 MB` scratch) | `11.42 tok/s` (`8.3%`, `sha=fbeb505f4e85`) | **`35.56 tok/s` (`2.84x`)** (`90.2%`, `sha=593362741d4b`) | `240/288` (`83.3%`) | `b_code_rewrite`: **exact SHA-256** (`a_new_text` shifted 1 word at char 673 due to `-184` resident experts from `+335 MB` spec scratch) |

---

## Improvement #4 Blueprint — Native EAGLE3 Support for `--mtp-model` (Follow-Up)

Left cleanly for a follow-up `feat/eagle3-draft` branch (HF drafts exist in `/mnt/storage/home-vault/models-local/eagle3-drafts/{angelslim,tengyunw}` as `safetensors` / `pytorch_model.bin`, and MoE bandwidth physics limits external draft gains under partial residency):

1. **Multi-layer hidden-state capture & `fc` projection**: Capture the 3 target layers specified by GGUF metadata (`eagle3.target_layers` / HF `aux_hidden_state_layer_ids`), concatenate (`3 * n_embd`, e.g. `6144 -> 2048`), and project through `fc.weight` instead of `nextn.enorm + nextn.hnorm` (`2 * n_embd`).
2. **Per-draft attention dimensions**: Read `head_count`, `head_count_kv`, and `key_length` (`head_dim`, e.g. `64` or `256`) from the draft GGUF metadata/tensor shapes rather than reusing the target model's `ModelConfig` (`32 x 128`).
3. **Draft-to-target vocabulary remap (`d2t`)**: Load the `[draft_vocab_size]` (`32000`) `d2t` offset tensor from the EAGLE3 draft and map `draft_tok -> draft_tok + d2t[draft_tok]` after draft argmax/sampling before feeding into target verification.
4. **Weight conversion**: Convert HF `eagle3-drafts/{angelslim,tengyunw}` `safetensors` weights into GGUF (or load `safetensors` directly) matching target vocab `151,936`.
