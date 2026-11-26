# Models

Which GGUF files RustingStrata runs, how each family runs and what was checked. Speed against llama.cpp is in
[BENCHMARKS.md](BENCHMARKS.md); flags are in [USAGE.md](USAGE.md).

## Architectures

The engine picks the code path from `general.architecture` in the file.

| Model family | `general.architecture` | How it runs |
|---|---|---|
| Qwen3 | `qwen3`, `qwen3moe` | GPU forward with `--gpu` (CUDA or Vulkan), CPU reference without it |
| Qwen3-Next | `qwen3next` | same |
| Qwen3.5 / Qwen3.6 | `qwen35`, `qwen35moe` | same |
| Qwen3.8-Flash-Next | `qwen4exp` | same |
| SenseNova-Loop | `sensenova_loop` | same |
| Recursive Flint | `flint` | same |
| [Gemma 4](#gemma-4-26b-a4b-ud-q4_k_xl-17-gb) | `gemma4` | own forward, matvecs and expert cache on the card, attention on the CPU |
| [DeepSeek-V4.1-Flash](#deepseek-v41-flash-iq2_xxs-249-gb) | `deepseek41` | own forward, matvecs and expert cache on the CUDA card, greedy only |
| [GLM-5.3-Flash](#glm-53-flash-ud-q2_k_xl-119-gb) | `glm5-next` | own forward, matvecs and expert cache on the CUDA card, greedy only |
| [Hy4-flash](#hy4-flash-iq2_siq2_xxs-85-gb) | `hy_v4_flash` | own forward, matvecs and expert cache on CUDA or Vulkan, greedy only |
| [Kolibri 1.1](#kolibri-11-q4_k_m-434-gb) | `kolibri1` | own forward, matvecs, expert cache and attention on CUDA or Vulkan, greedy only |
| [Inkling 2 Mini](#inkling-2-mini-iq4_nl-176-gb) | `inkling2_loop`, `inkling2` | own forward, matvecs and expert cache on CUDA or Vulkan |
| Raon-1 mini | `raon1` | own forward (4-stream Raon-mHC, 18 RDA + 7 gated NoPE MLA, SiTU-GLU MoE, effort steering), matvecs on CUDA or Vulkan |

The models with their own forward open the GPU by themselves (no `--gpu` flag) and load from the chat page of one
server like the others. `rustingstrata calibrate` measures their prefill chunk too (DeepSeek has no chunk to tune);
they read it from the same calibration file at load time. Image input is in [VISION.md](VISION.md).

## Qwen models

Greedy "The capital of France is" with `--gpu` (CUDA / Vulkan, second run) on an RTX 3060 12 GB with a Ryzen 5 7600X
and 32 GB RAM:

| Model | Types | Decode |
|---|---|---|
| Qwen3.5-2B (fine-tune) Q4_K_M | Q4_K, Q6_K, ... | 208 / 208 tok/s |
| Qwen3-4B-Instruct-2507 Q4_K_M (dense) | Q4_K, Q6_K | 104 / 110 tok/s |
| Qwen3.5-9B Q5_K_S | Q5_K, Q6_K, Q8_0 | 51 / 53 tok/s |
| Qwen3-30B-A3B-Instruct-2507 UD-Q2_K_XL | Q2_K to Q6_K | 114 / 123 tok/s |
| Qwen3.8-27B UD-IQ3_XXS (dense, 10.8 GB on GPU) | IQ1_S to IQ4_XS, Q2_K to Q8_0 | 20.6 / 20.5 tok/s |
| Qwen3.6-35B-A3B UD-IQ4_XS (57% of experts fit in VRAM) | IQ3_S, IQ4_XS, Q6_K, Q8_0 | 94 / 96 tok/s |
| Qwen3-Next-80B-A3B-Instruct UD-IQ1_S (most experts on the CPU) | IQ1_S to IQ4_NL, Q4_K to Q8_0, BF16 | 92 / 90 tok/s |
| Qwen3.8-Flash-Next GSQ-RCO IQ2_XS (64 GB in two files, pass the `-00001-of-00002` one; 18% of experts fit in VRAM, and with 32 GB RAM the rest are partly read from disk) | Q2_0, IQ1_M to IQ4_XS, Q6_K, Q8_0, F16, BF16 | 22 / 20 tok/s |

## GGUF types

Supported GGUF types: F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_0, TQ1_0, TQ2_0, Q2_K-Q6_K, IQ1_S, IQ1_M,
IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL, NVFP4, and the trellis types TQ2_T, TQK6 and TQK7 of agentionai's
llama.cpp fork.

The Gyro files of agentionai load too. Their names say TQ1_0 and TQ2_0, but the experts are stored in the trellis
types and read inputs rotated by a Hadamard transform (`prism.hadamard.*`). `Gyro-S-TQ1_0` (TQK6, TQK7) is tested on
CPU, CUDA and Vulkan. `Gyro-M-TQ2_0` (TQ2_T) has the same layout and only supported types, but was not run here.
Gyro-L (TQP5 to TQP7, PTQ1_0) is not supported yet.

## Gemma 4 (26B-A4B UD-Q4_K_XL, 17 GB)

`chat`, `generate` and `serve` detect the `gemma4` architecture and run its own forward: the matrix products and the
routed experts that fit run on the card, attention and the experts that do not fit run on the CPU. The weights take
90% of the free VRAM (`STRATA_DS_GPU_FRACTION`; `STRATA_DS_GPU_EXPERTS=<GB>` sets an absolute budget). `--kv`,
`--chunk`, `--parallel` and `--cache-ram` have no effect on it. The prompt goes through the layers in chunks of 256
tokens (`STRATA_G4_CHUNK`; 1 is token by token). `STRATA_G4_PROF=1` prints the prefill and decode times.

Reference machine (RTX 3060 12 GB), a 595-token prompt and 64 reply tokens, warm page cache:

| | 1.3.0 | now |
|---|---|---|
| Prefill | about 30 tok/s | 109-114 tok/s |
| Decode | 45-49 ms/token (20-22 tok/s) | 34 ms/token (29 tok/s) |

The 1.3.0 prefill is the wall time less the load and the decode; it had no timer of its own. Decode still pays a
host round trip for each group of matrix products of each layer, and no llama.cpp run was made on this machine; the
report in issue #81 has llama.cpp well ahead on an RTX 4070 SUPER.

## Kolibri 1.1 (Q4_K_M, 43.4 GB)

`chat`, `generate`, `serve`, `score` and `bench` detect the `kolibri1` architecture
([Aleph-Alpha/Kolibri-1.1-GGUF](https://huggingface.co/Aleph-Alpha/Kolibri-1.1-GGUF)) and run its own forward: 50
layers, 384 experts a layer with 6 routed and one shared, a sliding window of 513 positions on four layers of every
five, and on every fifth layer an attention over the 2048 keys that an indexer picks (K and V rounded to NVFP4).

```bash
rustingstrata chat --model Kolibri-1.1-Q4_K_M.gguf
rustingstrata serve --model Kolibri-1.1-Q4_K_M.gguf --port 8080
```

The matrix products, both kinds of attention and the routed experts that fit run on the card (CUDA when an NVIDIA
card and driver are present, Vulkan otherwise; `--backend vulkan` or `--backend cuda` picks one); the other experts
run on the CPU at the same time. A prompt goes through the layers in chunks of 1024 tokens (`STRATA_DS_CHUNK`), and a
layer's experts on the card take one round trip for the whole chunk. `STRATA_KOL_CPU_ATTN=1` keeps the attention of a
chunk on the CPU. The chat template is ChatML with the model's "Reasoning effort" system section (`--think`).

Reference machine (RTX 3060 12 GB, 30 GB RAM), server to server, a streamed chat request with 128 reply tokens,
prefill / decode in tok/s (the table with ranges is in [BENCHMARKS.md](BENCHMARKS.md#kolibri-11-q4_k_m-434-gb)):

| Prompt | RustingStrata | llama.cpp `-ngl 4` | llama.cpp `-ngl 99 -cmoe` |
|---|---|---|---|
| 0.5k tokens | 181 / 33.4 | 145 / 15.5 | 99 / 27.0 |
| 3.8k | 149 / 23.0 | 41 / 5.9 | 28 / 10.6 |
| 16k | 139 / 21.0 | 48 / 3.8 | 30 / 8.7 |
| 32k | 159 / 21.1 | 34 / 3.3 | 33 / 9.3 |

Vulkan on the same card, in a later session of the same day with CUDA beside it (prefill / decode in tok/s): 129 /
26.8, 106 / 19.5, 103 / 19.1 and 101 / 19.2 at 0.5k, 3.8k, 16k and 32k; CUDA in that session: 209 / 33.7, 182 / 26.4,
155 / 20.2 and 163 / 21.6.

Quality: llama.cpp's greedy tokens fed through RustingStrata's forward give the same top token at 45, 46, 44 and 46
of 48 steps after prompts of 2k, 3.8k, 16k and 32k tokens. Two llama.cpp runs of the same file (CPU against GPU
layers) agree with each other at 40-45 of 48; the misses are near ties.

Caveats: the file is larger than the RAM and VRAM of the reference machine together, so the experts that are not on
the card are read from the drive through the page cache, and the first request after a start is slower. The Vulkan
path was run on the RTX 3060 only (no AMD or Intel card); its kernel for the attention over picked keys needs about
35 KB of workgroup shared memory, and on a card with a smaller limit those layers' attention runs on the CPU.
Decoding is greedy, there are no stop strings, and the model's thinking arrives inside `content`.

## DeepSeek-V4.1-Flash (IQ2_XXS, 249 GB)

`chat`, `generate` and `serve` detect the `deepseek41` architecture and run it on the CPU reference forward, with the
matvecs and a 9 GB expert cache on the CUDA card (no `--gpu` flag needed). Pass the first shard; the others are found
next to it.

```sh
M=DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf
rustingstrata chat     --model $M [--think]
rustingstrata generate --model $M --prompt "The capital of France is" -n 64
rustingstrata serve    --model $M --addr 127.0.0.1:8080 [--api-key K]
```

| DeepSeek-V4.1-Flash IQ2_XXS (249 GB), RTX 3060 12 GB, 30 GB RAM | RustingStrata | llama.cpp b12116 (`--cpu-moe`) |
|---|---|---|
| Decode | 1.8 tok/s at first, 2-3.6 tok/s now (VRAM expert cache with LRU) | 0.3-1.5 tok/s (cold) |
| Prefill | ~2 tok/s on short prompts (0.5 s/token), ~0.5 tok/s at 1400 tokens | 0.5-0.9 tok/s (cold) |
| Teacher-forced top-1 vs llama.cpp | 189/192 short, 21/23 on 1425 tokens (misses are near-ties) | - |
| Sampling, tool calls | no sampling; tool calls in `serve` | supported |

The llama.cpp numbers are from the first cold runs on this box and the RustingStrata numbers from the last runs
before the model was deleted; they were not re-measured side by side, and both move with the page cache.
Decode varies run to run with the page cache, because the experts that miss VRAM are read from the GGUF.
`STRATA_DS_GPU_EXPERTS=<GB>` sets the VRAM weight budget; unset, it is 80% of free VRAM (about 9 GB on a 12 GB
card). Prefill is one token at a time. Prompt plus reply is limited only by the model's trained context from the GGUF (1,048,576; `STRATA_DS_CTX` lowers it; checked against llama.cpp only up to a 1425-token prompt), decoding is greedy only, and sampling is not supported. Tool calls work in `serve`: the prompt matches the model template (jinja2 render, three cases byte for byte) and the reply's `<｜DSML｜ calls>` block is parsed, but this path has not been run against the model itself.

## GLM-5.3-Flash (UD-Q2_K_XL, 119 GB)

`chat`, `generate` and `serve` detect the `glm5-next` architecture (Kimi-delta attention, sparse MLA attention, four-stream
hyper-connections, sigmoid MoE) and run the CPU reference forward, with the dense matvecs and an expert cache on the
CUDA card (`STRATA_DS_GPU_EXPERTS=<GB>` sets the budget, which also counts the non-expert weights; unset, it is 90% of
free VRAM, about 10 GB on a 12 GB card). Pass the first shard.

```sh
M=GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf
rustingstrata chat     --model $M
rustingstrata generate --model $M --prompt "The capital of France is" -n 64
rustingstrata serve    --model $M --addr 127.0.0.1:8080 [--api-key K]
```

Greedy output equals llama.cpp b12119 (`--cpu-moe`) for 32 tokens on 3 fixed prompts on the CPU path; with the GPU matvecs
prompts 1 and 2 still match for 32 tokens and prompt 3 flips at the first token, a near-tie (logprob -2.65 against -2.73). Fed llama.cpp's own 64 tokens per prompt (teacher-forced), the GPU path picks the same top-1 on 188 of 192 steps; the four misses are llama.cpp near-ties (margins 0.03-0.53 logprob). Tokenizer ids
match llama.cpp on 29 test strings and the chat template matches `/apply-template` on 2 conversations.
On an RTX 3060 12 GB with 30 GB RAM decode took about 0.4-0.7 s per token with warm page cache (about 55% VRAM expert-cache
hits on a short text); after the 2026-11-12 batching and prompt-preload changes, `bench --pp 1 --tg 24` measured 2.3-2.6 tok/s
warm, 32-token runs fell to 1.5-1.8 tok/s as the page cache turned over, and a 128-token prompt with hot-expert preload
decoded at 1.6 tok/s. A paired run on 2026-11-13 (64 greedy tokens through `serve`, llama.cpp, ours, ours, llama.cpp; short prompt and `tools/prompts/bench-natural.txt`, 429 tokens) gave 1.35 tok/s decode against 0.62 on the long prompt and 1.57 against 0.57 on the short one, load until ready in 5 s against 66-73 s, and a first token after 84-93 s against 73-76 s on the long prompt; on the short prompt the first token came after 17-19 s with a warm page cache against 6-8 s. Everything moves with the page cache. A fresh teacher-forced check on 3 prompts x 32 tokens matched llama.cpp's top token on 96 of 96 steps (top-5 overlap 91.9%, mean |dlogprob| 0.230).
Caveats: prefill runs 512-token chunks layer by layer so each routed expert is read once per chunk (a 380-word, about 550-token prompt prefilled in 209 s with 512-token chunks against 325 s with 128, identical logprobs; short prompts cost more per token because the experts stream from SSD once per chunk, `STRATA_GLM_CHUNK` sets the size); prompt plus reply is limited only by the model's trained context from the GGUF (1,048,576; `STRATA_GLM_CTX` lowers it, and the KV cache grows with the text, so RAM is the real bound; the k-pool sparse-attention indexer takes over past 2048 rows and was checked against llama.cpp up to 3.6k tokens: same top-5 sets, same first token); the NextN/MTP layer is unused; decoding is greedy only; tool calls work in `serve` (the prompt matches llama.cpp's template byte for byte, checked on one tool call and one tool-result round trip); the reasoning
text comes back in `content` before `</think>`.

## Hy4-flash (IQ2_S/IQ2_XXS, 85 GB)

`chat`, `generate`, `serve` and `bench` detect the `hy_v4_flash` architecture (four-stream hyper-connections, gated MLA
attention with sinks, sliding-window and sparse layers, latent MoE, Engram) and run the forward in `src/hy4/`, with
the matvecs and an expert cache on the GPU (no `--gpu` flag needed). Pass the first shard.

```sh
M=Hy4-flash-AD-IQ2_S-IQ2_XXS-00001-of-00002.gguf
rustingstrata chat     --model $M
rustingstrata generate --model $M --prompt "The capital of France is" -n 64
rustingstrata serve    --model $M --addr 127.0.0.1:8080 [--api-key K]
```

The GPU is CUDA when an NVIDIA card and driver are present and Vulkan otherwise (AMD, Intel); `--backend vulkan` or
`--backend cuda` picks one, `--device 0,1` spreads the expert cache over several cards, and `STRATA_DS_GPU=0` runs
everything on the CPU. Experts that are not in VRAM are split between the CPU (straight from the mapped file) and the
GPU (during decoding, uploaded into the VRAM cache in place of the least recently used expert), and the split
follows the measured time of each side. `--fixed-experts` (or `STRATA_DS_CPU_SHARE=0`) sends every expert to the GPU,
which is slower here but gives the same text on every run.
The prompt runs layer by layer in chunks of `STRATA_DS_CHUNK` tokens (default 1024), so a routed expert is read once
for all the tokens of a chunk that picked it. When the files fit in RAM with 4 GiB to spare they are read into the
page cache in the background from the start (`STRATA_PRELOAD=0` turns that off).

On an RTX 3060 12 GB with 30 GB RAM and a Ryzen 5 7600X, `bench --pp 16,128,512 --tg 32 --reps 2` measured, with a
warm page cache:

| Backend | Prefill 16 / 128 / 512 tokens (tok/s) | Decode (tok/s) |
|---|---|---|
| CUDA | 3.0 / 8.7 / 18.1 | 4.3 / 5.1 / 4.5 |
| Vulkan (same card) | 1.3 / 8.5 / 13.5 | 2.0 / 3.6 / 3.7 |
| CPU only (`STRATA_DS_GPU=0`) | 2.5 / 3.0 / 4.1 | 2.4 / 2.2 / 1.7 |

A 20,000-token prompt of text that does not repeat (this repository's documentation followed by Rust source,
`bench --prompt-file F --pp 20000 --tg 64 --reps 1`, one run each) prefilled at 11.6 tok/s on CUDA and 10.6 tok/s on
Vulkan, and the 64 tokens after it decoded at 3.3 and 1.9 tok/s.

A token reads about 1.3 GB of expert weights and the model is larger than this box's RAM, so the numbers move with
the page cache; the expert cache policy knobs (`STRATA_DS_LFU`, `STRATA_DS_NOEVICT`) changed nothing measurable. A
CPU-only build of the llama.cpp pull request that adds the model (#32517) decoded at 0.28 tok/s on the same box (one
run, not side by side). Not measured: an AMD or Intel card (the Vulkan path was run on the RTX 3060 only), more than
one card, and a machine whose RAM holds the whole model.

Checked against that llama.cpp build with tensor dumps: the first greedy token matches, and the last layer's output
differs from llama.cpp's by 14-20%, no more than this engine's integer and f32 paths differ from each other (19%). With
this IQ2 quant that is enough for greedy text to leave llama.cpp's at flat positions. The k-pool indexer of the sparse layers, which takes over past 2048 tokens, was checked
on a 2619-token prompt for layers 0 and 1: its tensors are within 0.2% and at least 508 of the 511 kept groups are the
same at every position (the disputed groups score within 0.1% of the cut).
Caveats: decoding is greedy only, and prompt plus reply is limited to 32,768 tokens (`MAX_CTX` in
`src/hy4/forward.rs`; every layer keeps the key rows of all positions). The GPU and CPU kernels round differently, so
with the adaptive split and with chunked prefill the greedy text can differ from run to run at near-ties; chunked and
token-by-token prefill give the same logits within 1e-3 on the CPU path. `tests/hy4_forward.rs` runs when
`STRATA_HY4_MODEL` points at the first shard.

## Inkling 2 Mini (IQ4_NL, 17.6 GB)

`chat`, `generate`, `serve` and `bench` detect the `inkling2_loop` and `inkling2` architectures (Thinking Machines Inkling 2: 24
layers of which layers 6-17 run three times with message controllers between the passes, so a token is 48 layer runs;
sliding-window attention with a relative bias and sinks, full attention on every sixth layer; 128 experts, 8 used) and
run the forward in `src/inkling2/`, with the matvecs and an expert cache on the GPU (no `--gpu` flag needed).

```sh
M=Inkling-2-Mini-IQ4_NL.gguf
rustingstrata chat     --model $M
rustingstrata generate --model $M --prompt "The capital of France is" -n 64
rustingstrata serve    --model $M --addr 127.0.0.1:8080 [--api-key K]
```

Unsloth's quants load as well: the files in the `Mainline_Support` folder of `unsloth/Inkling-2-Mini-GGUF`
(architecture `inkling2_loop`), whose experts are NVFP4 with a per-expert `ffn_gate_up_exps.scale` and
`ffn_down_exps.scale`. If a file is named `.gguf_file`, rename it to `.gguf` so the chat page lists it. The files at
the top of that repo use the architecture name `inkling2` and the tensor layout of llama.cpp PR #32695 (one array
entry per layer run, the message queries in `loop_res_query` and `loop_kv_res_query`); they load as well.

The backend, device and expert options are those of Hy4-flash above: CUDA when an NVIDIA card and driver are present
and Vulkan otherwise, `--backend vulkan` or `--backend cuda` to pick one, `STRATA_DS_GPU=0` for the CPU only, and
`--fixed-experts` for the same text on every run. The prompt runs layer by layer in chunks of `STRATA_DS_CHUNK` tokens
(default 512). The weights take `STRATA_DS_GPU_FRACTION` of the free VRAM (default 0.9, and 0.6 on a card with less than 8 GB
free; a pair such as `0.6,0.9` sets both); the experts that do not fit run on the CPU.

On an RTX 3060 12 GB with 30 GB RAM and a Ryzen 5 7600X, `bench --pp 64,512,2048 --tg 64 --reps 2` measured (mean of
the two runs, and the two runs):

| Backend | Prefill 64 / 512 / 2048 tokens (tok/s) | Decode after them (tok/s) |
|---|---|---|
| CUDA | 120.7 (59.2, 182.2) / 259.2 (254.6, 263.9) / 220.2 (218.7, 221.8) | 34.5 / 31.2 / 31.2 |
| Vulkan (same card) | 82.7 (34.5, 130.9) / 226.9 (226.6, 227.1) / 182.6 (179.0, 186.2) | 27.3 / 27.5 / 24.5 |
| CPU only (`STRATA_DS_GPU=0`) | 47.5 / 49.8 / 48.2 | 11.8 / 11.6 / 10.9 |

The first run of the 64-token prompt is the first work of the process and is the slow one of the two. The decode
column was measured again after a decoded token moved onto the GPU (attention, controllers and mixes in VRAM, one
round trip per layer run for the CPU experts), with the GPU shared with a desktop; the prefill path did not change
(alternating runs within 1-4%). `STRATA_DS_DEVICE=0` runs a decoded token on the host like the prompt.
llama.cpp c622f62 built with CUDA, in the same session (`llama-bench -t 6 -r 2`; `-ngl 99` runs out of VRAM):
with `-ngl 18` prefill 124.7 / 383.9 / 322.6 tok/s and decode 8.7 tok/s, with `-ngl 0` prefill 122.3 / 377.5 tok/s
(64 / 512) and decode 10.5 tok/s. So on this box the prompt is read at 0.6-0.7 of llama.cpp's speed from 512 tokens
on, and the reply is written 3.0-4.0 times as fast. `llama-bench` decodes from an empty context and `bench` decodes
after the prompt. More in [BENCHMARKS.md](BENCHMARKS.md#inkling-2-mini-iq4_nl-176-gb).
Not measured: an AMD or Intel card (the Vulkan path was run on the RTX 3060 only) and more than one card.

Checked against llama.cpp c622f62 (CPU, temperature 0) by feeding its greedy tokens through this forward: every top
token is the same on a short prompt (24 of 24) and at most one differs on a prompt of several hundred tokens (48
tokens, a near tie), on the CPU, CUDA and Vulkan paths. `tests/inkling2_forward.rs` runs these when
`STRATA_INKLING2_MODEL` points at the file. With unsloth's `Inkling-2-Mini-UD-Q4_K_M.gguf` (NVFP4 experts), the
32-token greedy text of llama.cpp c622f62 (CUDA, `-ngl 18`) was compared on three prompts: the CPU path matched on two
and CUDA on two (Vulkan, run on the first prompt only, gave the CUDA text). Each difference starts at a token where llama.cpp's own top two
log-probabilities are 0.01 and 0.16 apart.
Caveats: image input through `--mmproj` was not checked on this machine (no projector file here), and prompt plus reply is
limited to `--ctx` (default: the context the model was trained on). The GPU and CPU kernels round differently, so with
the adaptive split and with chunked prefill the greedy text can differ from run to run at near-ties; chunked and
token-by-token prefill give the same logits within 1e-3 on the CPU path.
