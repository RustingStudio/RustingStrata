# Pull Request: `feat: EAGLE3 GPU command-graph capture, adaptive speculative depth, and N-gram cache cascade`

## Summary

- **Symptom**:
  1. Target forward passes (`GpuModel::run()`) disabled Vulkan command-buffer graph capture whenever EAGLE3 was loaded (`h.eagle.is_none()`), and `GpuModel::mtp_run()` issued a redundant host `g.sync()` stall even when `head == false` with a 1-layer dense EAGLE3 head (`worker == None`).
  2. On partially-resident MoE models (`STRATA_EXPERT_MB=1500`), speculative drafting slowed down novel prose (`a_new_text`: `0.80x` at `60.7%` accept in PR #3; `--mtp 2` at `43.0%` accept in PR #2) because every rejected verification row triggers a larger CPU/GPU expert union, while high-acceptance code blocks (`89.1%`) could not use graphed target replay.
  3. `try_draft_primary()` in `src/ngram.rs` indexed `min_sample_size` / `min_percent` by slice position `i` instead of actual key token count (`off-by-1` when `nmin = 2`), and `SpecType::NgramCache` did not prioritize exact 6-token suffix matches or cap speculative burst length when only in-context 2/3-gram statistics were available.
- **Root Causes & Fixes**:
  1. **EAGLE3 Vulkan Graph Capture (`src/gpu_model.rs`)**:
     - `Vk::dtod()` records `cmd_copy_buffer` + transfer-to-compute memory barrier directly into the active graph command buffer (`st.rec`). Removed the `h.eagle.is_none()` guard in `GpuModel::run()` so `self.gpu.graph(...)` captures and replays the entire 48-layer forward including `eg.feat` extraction and `eg.fc` projection.
     - In `GpuModel::mtp_run()`, skipped the redundant host `g.sync()` when `head == false && is_eagle`, since Vulkan queue barriers order `mtp_forward`'s read of `eg.cur_g` before the next forward's `eg.fc` write.
  2. **Adaptive Speculative Depth Controller (`src/gpu_model.rs`)**:
     - Added a 12-step acceptance bitmask (`adapt_window`) and cooldown counter (`adapt_skip`) on `MtpHead` (reset on `load`, `reset`, and `restore_state`, updated by `spec_step` and `hybrid_spec_step`).
     - When `!cache.fully_resident()` (disable via `STRATA_MTP_ADAPT=0`):
       - Throttles multi-step `--mtp > 1` down to `k_eff = 1` whenever rolling 12-step acceptance falls below `11/12` (`91.7%`).
       - When acceptance drops (`hits12 <= 8 || hits6 <= 3`) on a rejection (`n == 0`), triggers `adapt_skip = 3`: advances `MtpHead` KV cache with `head = false` (no LM head GEMV / readback) and executes a graphed 1-token `step_greedy` / `step` pass, resuming EAGLE3 probes every 4th token and immediately clearing cooldown on N-gram hits.
  3. **N-gram Cache Length Indexing & Exact-Suffix Cascade (`src/ngram.rs`)**:
     - Fixed `try_draft_primary()` to compute `idx` from non-null tokens in `NgramKey`.
     - Updated `SpecType::NgramCache` in `NgramDrafter::draft()` to prefer exact 6-token context matches (`short_cooldown == 0`) before falling back to `ngram_cache_draft` (capped at 6 tokens when neither static nor dynamic lookup caches are populated).

## Verification

### Automated Test Suite
Ran `cargo build --release && cargo test --release --lib --bins --tests`:
- **195 passed, 0 failed, 5 ignored** across all 17 test targets (`src/lib.rs`: 62 passed including `ngram::tests::test_ngram_cache_try_draft_primary_indexes_by_ngram_len`; `src/main.rs`: 9 passed; `tests/gpu_model.rs`: 12 passed; `tests/kernels.rs`: 26 passed, 4 ignored; `tests/backend.rs`: 10 passed; `tests/serve.rs`: 23 passed; remaining integration crates: 53 passed, 1 ignored).

### Live Benchmark & Bit-Identical Greedy Parity (`127.0.0.1:8013`)
Target: `Qwen3-30B-A3B-Instruct-2507-IQ4_XS.gguf`, Vulkan (`RADV STRIX1 gfx1150`), `--ctx 4096 --kv q8 --fixed-experts`, `STRATA_EXPERT_MB=1500`, draft `angelslim/model.safetensors`, greedy decoding (`temperature=0.0, top_k=1, seed=42`), clean cold `lookup_dynamic_pr4.bin`:

| Arm | Prompt | Decode tok/s | Speedup vs same-run baseline | PR #3 Speedup (before) | Draft Accepted / Made | Greedy SHA-256 Match |
|---|---|---:|---:|---:|---:|---|
| `baseline_mtp0` | `a_new_text` | 4.03 | 1.00x | 1.00x | 0/0 | `ab5e75a7a983` (ref) |
| `baseline_mtp0` | `b_code_rewrite` | 4.37 | 1.00x | 1.00x | 0/0 | `5f86857ebc8d` (ref) |
| `baseline_mtp0` | `c_p3_antiloop` | 3.55 | 1.00x | 1.00x | 0/0 | `4e1c4a05a335` (ref) |
| `eagle3_graph_adapt_mtp1` | `a_new_text` | 5.67 | **1.41x** | 0.80x | 51/80 (63.8%) | **YES** (`ab5e75a7a983`) |
| `eagle3_graph_adapt_mtp1` | `b_code_rewrite` | **11.83** | **2.71x** | 1.07x | 123/138 (89.1%) | **YES** (`5f86857ebc8d`) |
| `eagle3_graph_adapt_mtp1` | `c_p3_antiloop` | 5.04 | **1.42x** | 1.06x | 59/90 (65.6%) | **YES** (`4e1c4a05a335`) |
| `eagle3_graph_adapt_mtp2` | `a_new_text` | 6.11 | **1.52x** | — | 51/90 (56.7%) | **YES** (`ab5e75a7a983`) |
| `eagle3_graph_adapt_mtp2` | `b_code_rewrite` | 7.52 | **1.72x** | — | 132/202 (65.3%) | **YES** (`5f86857ebc8d`) |
| `eagle3_graph_adapt_mtp2` | `c_p3_antiloop` | 4.60 | **1.30x** | — | 56/95 (58.9%) | **YES** (`4e1c4a05a335`) |
| `hybrid_graph_ngram_simple` | `a_new_text` | 6.20 | **1.54x** | 0.90x | 51/81 (63.0%) | **YES** (`ab5e75a7a983`) |
| `hybrid_graph_ngram_simple` | `b_code_rewrite` | 9.30 | **2.13x** | 1.61x | 243/245 (99.2%) | **YES** (`5f86857ebc8d`) |
| `hybrid_graph_ngram_simple` | `c_p3_antiloop` | 5.15 | **1.45x** | 1.02x | 75/130 (57.7%) | **YES** (`4e1c4a05a335`) |
| `hybrid_graph_ngram_cache` | `a_new_text` | **6.49** | **1.61x** | 1.26x | 55/106 (51.9%) | **YES** (`ab5e75a7a983`) |
| `hybrid_graph_ngram_cache` | `b_code_rewrite` | **11.55** | **2.64x** | 1.50x | 244/251 (97.2%) | **YES** (`5f86857ebc8d`) |
| `hybrid_graph_ngram_cache` | `c_p3_antiloop` | 4.23 | **1.19x** | 1.44x | 80/261 (30.7%) | **YES** (`4e1c4a05a335`) |

All **15/15** generations produced bit-identical SHA-256 greedy parity against `baseline_mtp0`, and every speculative configuration beat baseline across all 3 prompt workloads.
