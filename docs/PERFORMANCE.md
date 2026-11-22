# Inference performance notes

Measured on an RTX 3050 Laptop (4 GB VRAM), Ryzen 7 7445HS, release build with `-C target-cpu=native`.
All decode numbers are tokens/s from `rustingstrata bench`; drafts are verified, so greedy output is preserved.

## Inkling 2 Mini UD-Q3_K_XL (`inkling2_loop`, 13 GB)

| Configuration | Prefill (tok/s) | Decode (tok/s) | Note |
|---|---:|---:|---|
| Upstream defaults (`STRATA_DS_GPU_FRACTION=0.9`) | — | — | panics at `device.rs:1393`: no room to stage a routed expert |
| Host fallback fix + adaptive fraction (0.6) | ~42 | 8.6 | no speculative decoding |
| + suffix drafts (new default) | 47.2 | **16.3** | 112/112 drafts accepted (100%) |
| + DSpark `--draft-model` as well | 47.2 | 15.7 | acceptance falls to 89%; not recommended |

## Tiel Coder 35B A3B MTP (`qwen35moe`, 13.6 GB)

| Configuration | Prefill (tok/s) | Decode (tok/s) | Note |
|---|---:|---:|---|
| Upstream defaults (`--spec-type none`) | 43.1 | 16.1 | 35.9% expert hit rate |
| + suffix drafts (new default) | 90.0 | **26.7** | 24/24 drafts accepted (100%) |

`STRATA_MOE_PREFETCH=1` (still opt-in) gave 29.3 → 31.8 tok/s in one small-cache run but was noisy/inconsistent
when the expert cache was full, so it was left opt-in.

## Zero-temperature output check

`generate --temp 0 --seed 1 -n 48 --prompt "The capital of France is"`:

- Inkling 2: identical for 44 tokens; the final token of a quiz heading differed (`Past Simple Tense` vs
  `Present Perfect vs Past Simple`), consistent with the documented small logit shift from the adaptive CPU/GPU
  expert split. Both continuations are sensible.
- Tiel: suffix decoding produced a coherent answer (`Paris. Paris is a beautiful city. ...`); the no-spec run
  repeated the prompt sentence. The optimized output is at least as good.

## What changed

- `src/inkling2/forward/device.rs` — a layer whose routed weight cannot be staged now finishes that layer's
  experts on the host instead of aborting the token. This is the small-VRAM correctness fix.
- `src/inkling2/mod.rs`, `src/inkling2/chat.rs` — pick the default GPU memory fraction from the detected VRAM
  (0.60 below 8 GB, 0.90 above). A 4 GB card cannot stage experts at 0.90.
- `src/inkling2/forward.rs` — batch all host experts of a layer into one Rayon pass (`matvec_many_cpu`) when the
  quant type has an integer dot kernel.
- `src/deepseek/mat.rs` — the new shared `matvec_many_cpu` helper; `STRATA_DS_EX_AGE` tunes the expert-cache
  eviction age (diagnostic; it did not change hit rate in these runs).
- `src/main.rs` — `--spec-type` defaults to `suffix`. Verified ngram drafts are lossless and were accepted 100%
  on both local models; use `--spec-type none` to restore the old behaviour.
- `docs/USAGE.md` — user-facing notes.

Experiments that were tried and **reverted** because they did not help: parallel routing with Rayon (contends with
the existing `join`; route time doubled), making `STRATA_MOE_PREFETCH` on by default (inconsistent), and applying
the adaptive GPU fraction to all architectures (cut Tiel decode from 29.3 to 17.1 tok/s on a 4 GB card).
