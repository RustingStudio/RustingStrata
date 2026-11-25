# Calibration

`rustingstrata calibrate` measures the settings that make a model fastest on **this** machine and writes them to
`~/.cache/rusting_strata/strata-<model>[-backend][-dev…].json`. Later runs read that file unless a flag overrides the
value. One calibration per model, backend and device combination is enough; re-run after changing GPU, driver, quant,
KV format or a draft file.

Every architecture the binary supports is measured, including Inkling 2, Hy4, GLM, DeepSeek and Gemma 4, which the
old command skipped because they run on their own code path. Measurement is delegated to the existing `bench`
command (as a child process, reading its `strata.bench.v1` JSON), so there is a single timing implementation to trust.

## Quick start

```sh
M=path/to/model.gguf

# See the derived knob space and workload matrix, without loading weights:
rustingstrata calibrate --model $M --dry-run

# A normal calibration: 24 samples, prompt depths 128/512/2048:
rustingstrata calibrate --model $M

# A smaller, explicit search with two real corpora:
rustingstrata calibrate --model $M --chunk-values 512,1024,2048,4096 \
    --task prose --task coding --pp 128,512,2048 --tg 128 --samples 32 \
    --out ./calib-reports

# Every model in a folder, one after another:
rustingstrata calibrate --model ~/models
```

The command prints the workload matrix and knob space, then the baseline and the winner, then the path of the JSON
file. `--out DIR` additionally writes a full trace of every sample as `strata-<model>-calib-report.json`.

## What is optimized

Two signals are measured per workload: **prefill throughput** (prompt tokens / prefill time) and **decode throughput**
(generated tokens / greedy decode time). Both are divided by their own baseline and combined with a geometric mean:

```
score = sqrt((prefill / baseline_prefill) · (decode / baseline_decode))
```

An arithmetic mean would let a huge gain in one signal hide a regression in the other. Ratios are clamped to
`[1e-3, 1e3]` so a failed measurement cannot produce infinities. Several workloads are combined with a geometric mean,
so every prompt depth counts equally. Time to first token is *not* a separate signal: for greedy decoding the first
token's logits come out of the prefill, so TTFT would just reweight prefill.

The baseline is what a user gets today **without** a calibration file: the engine's own chunk — 4096 for the main
engine, 256 (Gemma 4), 512 (GLM), 1024 (Hy4) and 256 (Inkling 2, which clamps its GPU prefill to that size) for the
architectures with their own forward, and no chunk at all for DeepSeek, which prefills one token at a time. Where the
knobs apply, the baseline also uses short-prompt threshold 32, the physical core count and no drafts. The reported
score is relative to that, not to the first point of the search space.

Context length and KV format are deliberately not searched. They are memory choices the user makes; a noisy speed
win must not silently double someone's KV cache. `--ctx` and `--kv` only set the conditions of the measurement.

## Workloads and tasks

`--pp` lists prompt depths (default `128,512,2048`); `--tg` sets decoded tokens (default 128). All depths of one task
are measured in a single child process.

- `prose` (default): the built-in repeated paragraph, a stable machine-level baseline.
- `coding`: a built-in code snippet; real punctuation and identifiers change routing and draft acceptance.
- `user:PATH`: your own prompt file, for RAG documents, long system prompts, etc.

Repeat `--task` to mix corpora:

```sh
rustingstrata calibrate --model $M --task prose --task user:prompts.txt --pp 256,2048
```

## Search strategy

The search is deterministic for a given `--seed` and bounded by `--samples`:

1. **Baseline** — current defaults, measured on the full workload matrix (not counted in the budget). It is also a
   trial in the search, so a candidate that scores below it cannot win. If no candidate beats it, no calibration is
   written and the defaults stay in force. A failed baseline warns and scores against a neutral `1x` reference; such a
   run writes a file only when a point beats that reference.
2. **Coarse sweep** — up to `min(samples/3, 8)` points from a low-discrepancy (radical-inverse) sequence over the
   joint knob space. Their scores only pick starting points; they are never compared with full-matrix scores and
   cannot win.
3. **Multi-start coordinate descent** — each start is re-measured on the full matrix, then each knob is tried at
   every other value and the best move kept. The walk stops after two rounds without improvement. `stop_reason` is
   reset for every start, so all of them get a full descent.
4. **Final polish** — one more pass over the cheap knobs from the overall winner.

A candidate that fails (OOM, timeout, missing JSON fields) is recorded with its error and never wins; it still counts
against the budget. Every stage, including the final polish, draws from `--samples`; the baseline alone is outside the
budget. `--reps N` repetitions happen inside one sample and do not consume extra samples. `--timeout SECS` caps each
sample. `--warmup N` adds an extra discarded run per sample; it is off by default because it loads the model a second
time.

### Knobs

| knob | values | applies to |
| --- | --- | --- |
| `chunk` | `--chunk-values`, default `256,512,1024,2048,4096` | main engine and Gemma 4, GLM, Hy4, Inkling 2 (capped at 256); DeepSeek has no chunk knob |
| `mtp` | `0,1,2,4` | main-engine models with a draft head, otherwise `0` only |
| `short` | 32 plus one threshold just above each `--pp` depth (default `32,129,513,2049`) | main engine only |

Each `short` threshold flips exactly one prompt depth into the short-prompt path: a prompt uses token-at-a-time
prefill when it is shorter than the threshold. The old fixed `16/32/64/128` list was below every default workload
depth, so no candidate changed anything and the value written was noise.

Inkling 2, Hy4, GLM and Gemma 4 read their chunk from the environment at prefill time, so `bench_sample` sets
`STRATA_DS_CHUNK`, `STRATA_GLM_CHUNK` and `STRATA_G4_CHUNK` to the candidate value. Their engines apply a calibrated
chunk the same way at load time. In `serve`, those variables are restored before a different model is loaded, so a
model without a calibration never inherits the previous model's chunk. DeepSeek prefills one token at a time and
reads no chunk: its calibration records no chunk, MTP count or short threshold. The main engine reads `chunk`, `mtp`
and `short` from the file in the usual way.

Multi-GPU runs (pass `--device 0,1`) keep the engine's free-VRAM layer split; calibration does not measure or write a
layer split, and `--split` is ignored with a note.

## Flags

| flag | default | meaning |
| --- | --- | --- |
| `--samples N` | 24 | hard cap on benchmark samples |
| `--reps N` | 3 | repetitions per point (their mean is used) |
| `--warmup N` | 0 | extra discarded run before each sample |
| `--pp A,B,C` | `128,512,2048` | prompt depths |
| `--tg N` | 128 | decoded tokens |
| `--chunk-values A,B` | `256,…,4096` | prefill chunks to search |
| `--task LABEL` | `prose` | corpus (repeatable) |
| `--seed N` | 1 | deterministic search seed |
| `--timeout SECS` | 0 (none) | per-sample time limit |
| `--dry-run` | off | print the plan, take no measurements |
| `--out DIR` | — | also write the full sample trace |
| `--mtp-model` / `--draft-model` | — | measure with that draft file loaded |
| `--ctx`, `--kv` | 4096, f16 | conditions of the measurement, not searched |

`--model` also accepts a folder: every `.gguf` in it and its subfolders is calibrated with the same flags; a model
that fails is reported and the rest go on. `--mtp-model` and `--draft-model` need a single model file.

## The `spec` subcommand

`calibrate spec` sweeps speculative settings **per task**: which spec type and how many drafts are best for coding
versus prose on this model.

```sh
rustingstrata calibrate spec \
    --model ~/Models/Inkling2/Inkling-2-Mini-UD-Q3_K_XL.gguf \
    --draft-model ~/Models/Inkling2/dspark-Inkling-2-Mini-Q8_0.gguf \
    --tasks prose,coding --spec-types ngram-simple,ngram-mod,ngram-cache \
    --draft-max 4,8,16 --ngram-n 3,6,12 --samples 16
```

Knobs: `spec_type` (`none`, `suffix`, `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache`),
`draft_max` (drafts verified per step), `ngram_n` (lookup key length), `mtp` (MTP count, main engine only). The search
stores indices into those lists; the JSON report and the child `bench` command both receive the actual values. The
baseline is today's spec defaults (`none`, 16 drafts, n-gram 12, mtp 0).

The result is written under `spec` in the same calibration file, keyed by task, with decode rate and draft acceptance
percentage of the winner. A spec sweep **only adds that section**: engine knobs measured by a previous `calibrate`
run are left untouched.

## The calibration file

The file keeps the v1 keys the engines read (`chunk`, `mtp`, `short`, `ctx`, `gpu`, `bytes`, `measured`, `aux`), so
older binaries still read it. A stale v1 `split` is deliberately dropped: v2 does not measure a layer split and
nothing writes that key. v2 adds:

```json
{
  "schema": 2,
  "arch": "qwen35moe",
  "chunk": 2048, "mtp": 2, "short": 32,
  "ctx": 4096,
  "measured": {"prefill_tok_s": 1057.0, "decode_tok_s": 26.7},
  "best": {"chunk": 2048, "mtp": 2, "short": 32},
  "search": {"samples_used": 24, "score": 1.234, "stop_reason": "budget",
             "elapsed_ms": 123456, "seed": 1},
  "workloads": [{"pp": 128, "tg": 128, "task": "prose"}],
  "spec": {"prose": {"spec_type": "ngram-mod", "draft_max": 8, "ngram_n": 6,
                     "acceptance_pct": 72.5}}
}
```

`ctx` records the context the measurement ran at; it is never a recommendation applied to later runs. There is no
`kv` or `threads` key from v2 — those remain the user's choice. `--out DIR` writes a separate
`strata-<model>-calib-report.json` with all trials, their knob points, measurements, errors and timings, useful for
auditing a surprising result or comparing two seeds.

The filename encodes the backend that **actually opened** and the device order (`-vulkan`, `-dev0-1`), so a CUDA
to Vulkan fallback writes the file the runs will look for, and calibrations for different setups do not overwrite
each other. As before, `XDG_CACHE_HOME` moves the folder, and a file measured on another GPU or another file of the
same name (different size) is ignored with a warning.
