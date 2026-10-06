# Improvement loop log

Each iteration: one verified item. Newest entry last. Loop instructions: `/home/vasylt/Rusting/loop-prompts/strata.md`.

## State

- Milestone: M2 done (parity 98.59%, gate 99%; see `docs/m2-results.md`). M3 expert cache partly in (commit 90db54a).
- Uncommitted, unverified: **chunked GPU prefill** (M4 first half). Files: `src/gpu_model/prefill.rs` (new),
  `src/cuda/kernels.cu` (deltanet_seq, conv_seq/conv_state, attn_prefill_{64,128,256}, moe_combine_seq, gemm with
  dense weights and act 0/1/2), `src/cuda/mod.rs`, `src/gpu_model.rs`, `src/main.rs` (`--chunk`, default 1024),
  `src/expert_cache.rs` (`slot`), `tests/gpu_model.rs` (`prefill_matches_decode`), `tests/cuda.rs`.
  Builds clean, CPU tests pass. Not yet run on the GPU. `DBG_LAYERS` env hack in `src/gpu_model.rs` is debug
  scaffolding: remove before commit.

## Iterations

### 2026-10-06 #1 — GPU busy, triage only
- GPU busy: `forever_bear` (1051 MiB) on the GPU. No GPU tests or benchmarks run.
- Found the prefill work above uncommitted. Release build OK (`CARGO_TARGET_DIR=target/loop`); CPU suites
  (lib, gguf, ops, quant, tokenizer, cpu_moe) pass. Kept the work: it cannot be verified or rejected without the GPU.
- No numbers moved. No commit of code.

## Tried and rejected

(none yet)

## Next candidates

No profile yet for prefill; ranking is by expected cost until measured.
1. Verify chunked prefill on the GPU: `cargo test --release --test gpu_model prefill_matches_decode` (also with
   `CHUNK=1024`), then parity with `--extra=--gpu`, then 3-run prefill benchmark at 4K vs 1017 tok/s. Remove
   `DBG_LAYERS`, commit.
2. Profile prefill (nsys): staged expert copies are synchronous pageable `memcpy_htod` (`ponytail:` note in
   `prefill.rs`); likely the top cost for cold layers. Fix: pinned double buffer on a copy stream.
3. Decode profile vs 67 tok/s gate; close parity gap 98.59% -> 99% (Q8 activation quantization like llama.cpp).
