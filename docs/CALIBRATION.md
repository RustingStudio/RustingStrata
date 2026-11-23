# Calibration

`rustingstrata calibrate` measures the settings that make a model fastest on **this** machine and writes them to
`~/.cache/rusting_strata/strata-<model>[-backend][-dev…].json`. Later runs read that file unless a flag overrides the
value. One calibration per model, backend and device combination is enough; re-run after changing GPU, driver, quant,
KV format, or a draft file.

The v2 workflow is universal: every architecture the binary supports is calibrated, including Inkling 2, Hy4, GLM,
DeepSeek and Gemma 4, which used to be skipped because they run on their own code path. It does this by measuring
through the existing `bench` command, so there is only one timing implementation to trust.

## Quick start

```sh
M=path/to/model.gguf

# See what would be measured, without loading any weights:
rustingstrata calibrate --model $M --dry-run

# A normal calibration (24 samples, balanced objective):
rustingstrata calibrate --model $M

# An interactive model: favour time to first token, search a small space:
rustingstrata calibrate --model $M --profile interactive \
    --chunk-values 512,1024,2048,4096 --ctx-values 4096,8192 \
    --pp 128,512,2048 --tg 128 --samples 32

# Every model under a folder, one after another:
rustingstrata calibrate --model ~/models
```

The command prints the derived workload matrix and knob space first, then the baseline and the winning
configuration, and finally the path of the JSON file it wrote. `--out DIR` writes one line per sample to a report.

## What is optimized

Three signals are measured for every workload:

| signal | meaning | direction |
| --- | --- | --- |
| prefill throughput | prompt tokens / prompt prefill time | higher is better |
| decode throughput | generated tokens / greedy decode time | higher is better |
| TTFT | start of prefill → first generated token | lower is better |

Each is divided by the baseline (the first knob point, measured at the start), and combined with a **weighted
geometric mean**:

```
score = (prefill/baseline)^wp · (decode/baseline)^wd · (baseline_ttft/ttft)^wt
```

A weighted arithmetic mean would let a huge gain in one signal hide a regression in another; the geometric mean
does not. All three signal ratios are clamped to `[1e-3, 1e3]` so a failed measurement cannot produce infinities.

Presets set the weights:

| `--profile` | prefill | decode | TTFT | typical use |
| --- | --- | --- | --- | --- |
| `interactive` | 0.20 | 0.30 | 0.50 | chat, short prompts |
| `balanced` (default) | 0.25 | 0.35 | 0.40 | general use |
| `batch` | 0.50 | 0.40 | 0.10 | long prompts, throughput jobs |

Use `--objective prefill|decode|ttft` to optimize a single signal, or `--weights 0.3,0.3,0.4` for exact control
(the three numbers are normalized to sum to one).

## Workloads and tasks

`--pp` is the list of prompt depths (default `128,512,2048`); `--tg` is the number of decoded tokens (default 128).
Every depth of a task is benchmarked in one child process.

`--task` selects the corpus, repeatable:

- `prose` (default): the built-in repeated paragraph. It routes few experts and is a stable machine-level baseline.
- `coding`: a built-in code snippet — punctuation and identifiers change both prefill cost and n-gram acceptance.
- `user:PATH`: your own prompt file. Use it when real traffic matters (RAG documents, long system prompts, etc.).

For example:

```sh
rustingstrata calibrate --model $M --task prose --task coding --task user:$(pwd)/prompts.txt \
    --pp 256,2048 --tg 64
```

TTFT is the prefill time of each prompt: the prefill produces the logits of the first generated token, so loading the
model and compiling kernels are not counted (they are not part of a request's latency).

## Search strategy

The search is deterministic for a given `--seed` and bounded by `--samples`:

1. **Baseline** — the first value of every knob is measured on the full workload matrix (not counted in the budget).
2. **Coarse sweep** — up to `min(samples/3, 8)` points from a low-discrepancy (radical-inverse) sequence over the joint
   knob space. Unlike a fixed grid, this covers combinations, so a setting that only wins together with another one
   is represented.
3. **Multi-start coordinate descent** — the three best coarse points are refined: each knob is tried at every other
   value, the best move is kept, and the walk stops after `--patience`-equivalent stagnation (`2` full rounds by
   default). Multiple starts avoid the local minima of a single greedy walk.
4. **Final polish** — one more pass from the overall winner over the cheap knobs.

A candidate that fails (for example, OOM at a large chunk) is recorded with its error and never wins; it still counts
against the sample budget. `--timeout SECS` caps each sample so a bad point cannot hang the run.

### Knobs

| knob | values | applies to |
| --- | --- | --- |
| `ctx` | `--ctx-values` (default: `--ctx` and its double) | all |
| `kv` | `--kv-values` (default: `f16`) | all |
| `chunk` | `--chunk-values` (default `256,512,1024,2048,4096`) | all; the arch-specific engines read it from the environment |
| `mtp` | `0,1,2,4` | main engine only |
| `short` | `16,32,64,128` | main engine only |
| `threads` | half and all physical cores | main engine only |

Multi-GPU runs are supported (pass `--device 0,1`): measurements use the engine's free-VRAM layer split. The v1
layer-split sweep is not part of v2, and `--split` is ignored with a note. Every architecture is measured, including
Gemma 4 (its `bench` path is part of this change); for the models on their own code path, the measured chunk and
context are applied at load time through `STRATA_DS_CHUNK` / `STRATA_GLM_CHUNK` and the context limit.

Smaller spaces give more reliable answers for the same budget. If a large chunk OOMs on your card, pass
`--chunk-values 1024,2048,4096` rather than spending samples on failures. If you always run `q8`, pass
`--kv-values q8` so the measurement matches your workload.

## Flags

| flag | default | meaning |
| --- | --- | --- |
| `--samples N` | 24 | hard cap on benchmark samples |
| `--reps N` | 3 | repetitions per point (their mean is used) |
| `--warmup N` | 1 | extra discarded run before each sample |
| `--pp A,B,C` | `128,512,2048` | prompt depths |
| `--tg N` | 128 | decoded tokens |
| `--profile NAME` | `balanced` | `interactive`, `balanced`, `batch` |
| `--objective NAME` | — | optimize a single signal |
| `--weights P,D,T` | — | explicit objective weights |
| `--ctx-values A,B` | `--ctx`, double | context lengths to search |
| `--kv-values f16,q8` | `f16` | KV formats to search |
| `--chunk-values A,B` | `256,…,4096` | prefill chunks to search |
| `--task LABEL` | `prose` | corpus (repeatable) |
| `--seed N` | 1 | deterministic search seed |
| `--timeout SECS` | 0 (none) | per-sample time limit |
| `--dry-run` | off | print the plan, take no measurements |
| `--out DIR` | — | also write the full sample trace |
| `--mtp-model` / `--draft-model` | — | measure with that draft file loaded |

`--model` also accepts a folder: every `.gguf` in it (and its subfolders) is calibrated in turn with the same flags;
a model that fails is reported and the rest go on. `--mtp-model` and `--draft-model` need a single model file.

## The `spec` subcommand

`calibrate spec` sweeps speculative-decoding settings **per task**. It answers "which n-gram type and how many drafts
are best for coding vs. prose on this model", rather than tuning prefill knobs.

```sh
# Inkling 2, with its DSpark draft file, across two corpora:
rustingstrata calibrate spec \
    --model ~/Models/Inkling2/Inkling-2-Mini-UD-Q3_K_XL.gguf \
    --draft-model ~/Models/Inkling2/dspark-Inkling-2-Mini-Q8_0.gguf \
    --tasks prose,coding --spec-types ngram-simple,ngram-mod,ngram-cache \
    --draft-max 4,8,16 --ngram-n 3,6,12 --samples 16

# Look first, benchmark second:
rustingstrata calibrate spec --model $M --dry-run
```

The knobs are `spec_type` (`none`, `suffix`, `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`,
`ngram-cache`), `draft_max` (drafts verified per step), `ngram_n` (lookup key length) and `mtp` (MTP draft count, main
engine only). The result is written under `spec` in the same calibration file, keyed by task, with the decode rate,
TTFT and draft acceptance percentage of the winner. A spec sweep **only adds that section**: the engine knobs
measured by a previous `calibrate` run are left untouched. Acceptance is task-dependent, which is why the sweep is per task:
a lookup drafter can accept nearly every draft on repetitive text and far fewer on prose.

## The calibration file

The file keeps every v1 key (`chunk`, `mtp`, `short`, `ctx`, `gpu`, `bytes`, `measured`, `aux`, `split`), so older
binaries still read it. v2 adds:

```json
{
  "schema": 2,
  "arch": "qwen35moe",
  "chunk": 2048, "mtp": 2, "short": 32, "threads": 6,
  "ctx": 8192, "ctx_recommended": 8192, "kv": "f16",
  "measured": {"prefill_tok_s": 1057.0, "decode_tok_s": 26.7, "ttft_ms": 0.9},
  "best": {"chunk": 2048, "ctx": 8192, "kv": 0, "mtp": 2, "short": 32, "threads": 6},
  "search": {"samples_used": 24, "score": 1.234, "stop_reason": "budget",
             "elapsed_ms": 123456, "seed": 1, "objective": "balanced", "profile": "balanced"},
  "workloads": [{"pp": 128, "tg": 128, "task": "prose"}],
  "spec": {"prose": {"spec_type": "ngram-mod", "draft_max": 8, "ngram_n": 6,
                     "acceptance_pct": 72.5}}
}
```

`ctx_recommended` is advisory: it is applied only when `--ctx` is left at its default (4096); an explicit `--ctx`
always wins. `--out DIR` writes a separate `strata-<model>-calib-report.json` containing all trials, their knob
points, measurements, errors and timings — useful for auditing a surprising result or comparing two seeds.

The filename encodes backend and device order (`-vulkan`, `-dev0-1`), so calibrations for different setups do not
overwrite each other. As before, `XDG_CACHE_HOME` moves the folder, and a file measured on another GPU or another
file of the same name (different size) is ignored with a warning.
