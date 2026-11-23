# Benchmarks

Every model RustingStrata runs, measured against llama.cpp; some also against other engines. Raw logs and older runs are in the [benchmark history](README.md#benchmark-history).

Rules for every entry:

- Same machine, same GGUF file, same prompt length, greedy decoding, in one session. The machine is named per entry.
- llama.cpp is tuned for the file (`-ncmoe`, `-ub`, flash attention) and the setting is written next to the number.
- A number that moves with the page cache, thermals or the text (MTP) is given as a range and the caveat is written below it.
- Entries that were not measured side by side say so.

Reference machine unless stated: RTX 3060 12 GB, Ryzen 5 7600X, 32 GB RAM.

## Summary

| Model (file) | Decode, RustingStrata | Decode, llama.cpp | Other engine | Entry |
|---|---|---|---|---|
| Qwen3.6-35B-A3B UD-Q5_K_M, 4241-token prompt | 80.5 tok/s (75.1-82.0) | 64.4 (58.8-66.0) | - | [below](#qwen36-35b-a3b-ud-q5_k_m) |
| Qwen3.6-35B-A3B UD-Q2_K_XL | 118.2 tok/s (113.4 after 4K) | 68.4 (66.5) | - | [below](#qwen36-35b-a3b-ud-q2_k_xl) |
| Qwen3.6-35B-A3B UD-Q3_K_M, 6474-token prompt | 83.0 tok/s (108.0 with MTP) | 67.5 (101.1 with MTP) | PolyStrata 79.2 (113.3 with MTP) | [below](#qwen36-35b-a3b-ud-q3_k_m-vs-polystrata) |
| Qwen3.8-Flash-Next GSQ-RCO IQ2_XS, 1262-token prompt | 33.0 tok/s (29.3-35.8) | 25.6 (23.2-26.9) | - | [below](#qwen38-flash-next-gsq-rco-iq2_xs) |
| GLM-5.3-Flash UD-Q2_K_XL (119 GB) | 1.35-1.57 tok/s | 0.57-0.62 | - | [below](#glm-53-flash-ud-q2_k_xl-119-gb) |
| DeepSeek-V4.1-Flash IQ2_XXS (249 GB) | 2-3.6 tok/s | 0.3-1.5 | - | [below](#deepseek-v41-flash-iq2_xxs-249-gb) |
| Inkling 2 Mini IQ4_NL (17.6 GB), 512-token prompt | 31.2 tok/s (28.7-33.7) | 8.7-10.5 | - | [below](#inkling-2-mini-iq4_nl-176-gb) |
| Recursive Flint 39.8B UD-Q5_K_XL, 4102-token prompt | 64.6 tok/s (63.8-65.6) | 48.6 (47.5-49.3) | - | [below](#recursive-flint-398b-ud-q5_k_xl) |

## Qwen3.6-35B-A3B UD-Q5_K_M

`Qwen3.6-35B-A3B-UD-Q5_K_M.gguf` (25.2 GiB, does not fit in VRAM: a third of the experts are on the GPU). llama.cpp
is build 12191, tuned for the same file (`-cmoe --moe-cache-mib 7500 -ngl 99 -fa on -t 6 -ub 2048 -b 2048`, the best
decode of a 9-config sweep over `-ncmoe`, `--moe-cache-mib` and `--moe-cache-cpu-share`).

| | CUDA | llama.cpp |
|---|---|---|
| Decode after 4K prompt | **80.5 tok/s** (75.1-82.0) | 64.4 (58.8-66.0) |
| Prefill, 4K prompt | **1530 tok/s** (1451-1596) | 1118-1122 with `-ncmoe 27`, its best prefill flags (2 runs, one load); 793 (367-1063) with the decode flags above |

Measured at commit 5a27ed1 (1.2.5) with `rustingstrata serve --gpu --ctx 8192 --mtp 0` and `llama-server`: the same
4241-token prompt file and the same script for both, greedy, thinking off, 128 tokens, no drafts in either engine.
Two loads per engine in alternating order; in each load a warm-up and the first 4K request are thrown away. Decode is
the mean of runs 4-8 of each load (10 runs, range in brackets). Both engines climb for the first runs after a load;
over all 16 runs it is 78.8 (63.6-82.0) against 60.3 (31.3-66.0). Prefill is 4 new prompts per load (8 runs).

Not equal: the replies were 128 tokens (RustingStrata) and 125 (llama.cpp, ended by itself). About 20 GiB of RAM was
free for the 25.2 GiB file, so neither engine had the whole file in the page cache. Perplexity against llama.cpp was
not run on this file; for UD-Q3_K_M it is
[below](#quality-vs-llamacpp-qwen36-35b-a3b-ud-q3_k_m).

## Qwen3.6-35B-A3B UD-Q2_K_XL

`Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf`, 11.7 GiB, does not fit in VRAM. llama.cpp build 11876 (2026-10-28), `-ncmoe 12 -ub 2048 -b 2048`, flash attention (its 4K prefill is about 1550 with the default ubatch).

| | CUDA | Vulkan | llama.cpp |
|---|---|---|---|
| Prefill, 4K prompt | 2216 tok/s (2169-2252) | 2154 (2140-2162) | ~2236 (2183-2310) |
| Decode after 4K prompt | 113.4 tok/s (110.9-114.8) | 113.6 | 66.5 |
| Decode, short prompt | 118.2 tok/s (117.0-118.8) | 117.3 | 68.4 |
| Decode with MTP drafts (`--mtp 1`), short / after 4K | up to 160 / 152 tok/s | 158 / 154 | its MTP is slower |
| 32K: prefill / decode | 1800 / 91.9 tok/s | 1582 / 90.7 | - |
| Greedy top-1 agreement with llama.cpp (20 prompts x 32 tokens) | 98-99% | 98-99% | 98.3% (its batch vs token-by-token modes) |

Method: `rustingstrata bench --gpu --mtp 0 --pp 512,4096,32768 --tg 128 --reps 3 --ctx 32896`, mean of 3 rounds. llama.cpp from `llama-server` timings, 4047-token prompt plus 128 tokens, 3 runs. MTP gains depend on the text.

Agreement counts a token when it matches llama.cpp run token by token or with the context as one batch;
llama.cpp's own two modes agree on only 98.3%. The remaining misses are near-ties where
float rounding order picks a different token. The expert placement adapts between runs and the CPU and
GPU expert kernels round differently, so greedy output can change at such near-ties from one run to the next. MTP
on and off give identical greedy tokens.

With less VRAM, simulated on the 3060 with `--vram-margin` (not measured on real 8 GB or 6 GB cards): ~8 GB
keeps 116 / 110 tok/s decode (short / after 4K) and 2128 tok/s 4K prefill, ~6 GB 105 / 97 and 2056.

## Qwen3.6-35B-A3B UD-Q3_K_M vs PolyStrata

`Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` (Unsloth, MTP layer included, 15.9 GiB, does not fit in VRAM). One prompt file for every engine (about 3200 words of repo docs, 6474 tokens in RustingStrata and llama.cpp, 6476 in PolyStrata), chat template with thinking off, greedy, 128 generated tokens, sent through each engine's OpenAI-compatible server by the same script. llama.cpp is b12191 (has `--moe-cache-mib` and `--moe-cache-cpu-share`). Prefill is the first request (cold), decode is the mean of runs 2-4 on the warm server.

| | RustingStrata | PolyStrata | llama.cpp |
|---|---|---|---|
| Prefill | 2088 tok/s | 1780 tok/s | 1408-1742 tok/s (depends on `-ncmoe`) |
| Decode, no MTP | 83.0 tok/s | 79.2 tok/s (`--no-draft --lookup 0 --window 1`) | 67.5 tok/s (`-ncmoe 36 --moe-cache-mib 5000`); 50.4 with `-ncmoe 22`, no cache |
| Decode with MTP | 108.0 tok/s (90% of drafts accepted) | 113.3 tok/s (defaults; acceptance not read) | 101.1 tok/s (87.5% accepted) |

How each was run:

- RustingStrata: `serve --gpu --ctx 8192 --mtp 0|1`.
- PolyStrata: `serve/server.py --engine strata` with its Q3_K_M config (`--tail-skip 6`, expert profile). No MTP: `--no-draft --lookup 0 --window 1` added, because `--no-draft` alone leaves its n-gram guesses on. With MTP: defaults.
- llama.cpp: `llama-server -ngl 99 -fa on -ub 2048 -b 2048 -c 8192`, no-MTP rows with `-ncmoe 22`/`26`/`30`/`36` and `--moe-cache-mib` 0-5000, `--moe-cache-cpu-share` 0 or 0.5. The MTP row adds `--spec-type draft-mtp`. `-ncmoe 22` with a cache and MTP did not start.
- Sweep (decode tok/s, warm): `-ncmoe 22` no cache 50.4; 22 + 1500 MiB 61.9; 26 + 2500 MiB + share 0.5 65.4; 30 + 3000 + 0.5 63.4; 36 + 5000 + share 0 67.5. In `llama-bench` (`-p 5000`) the cache did not help (52.0 at `-ncmoe 22` against 42.9-47.5 with cache), the server runs are what is listed.

Not equal:

- Each engine's prefill after the first request is served from its prompt cache, so only the first request is a prefill number.
- PolyStrata's token count is 2 higher (different tokenizer files).
- PolyStrata offers three sizes of Qwen3.6 (UD-Q3_K_M, UD-Q2_K_XL, UD-Q4_K_M). RustingStrata and llama.cpp take any GGUF. The Q3_K_M file was used for all three.
- With MTP, PolyStrata is ahead of RustingStrata here (113.3 against 108.0). It was not checked why; PolyStrata's draft logic and acceptance were not read.
- PolyStrata's engine for this model is built on llama.cpp's CUDA kernels. What else differs was not checked.

## Qwen3.8-Flash-Next GSQ-RCO IQ2_XS

Two-shard GGUF: shard 1 is 39,225,954,592 B (SHA-256 `92cee27ae5bbadcd732416a0f7a7f0acc092399dbbe8f5a5efa707c2ec0a49d7`), shard 2 is 28,800,138,432 B (SHA-256 `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`). Measured 2026-11-10 on the reference machine, but with 30 GiB of RAM (below the model's comfort point), so experts stream from the NVMe.

The prompt is a fixed WikiText-2 raw test passage plus one instruction: 1,262 tokens raw, 1,273 after the chat template. Exact text and token IDs are in `.posts/data/2026-11-10-qwen38-prompt.txt` and `.posts/data/2026-11-10-qwen38-prompt.ids.txt`. Greedy, 128 generated tokens, same OpenAI-compatible request for both engines. One warm request, then three measured requests per session; two sessions per engine with alternating order, so six measured runs each. Decode is the mean of the measured runs; cold prefill is the first request of each session.

| Engine | Decode, no speculative | Cold prefill | Speculative/MTP |
|---|---:|---:|---|
| RustingStrata 1.2.1 | 33.0 tok/s (29.3-35.8) | 109.3 tok/s (99.9-118.8) | not available |
| llama.cpp b12191 | 25.6 tok/s (23.2-26.9) | 30.9 tok/s (29.2-32.6) | not available |

How each was run:

- RustingStrata: `serve --gpu --ctx 4096 --mtp 0`.
- llama.cpp b12191 (commit `e95426f16`): `-c 4096 -ngl 99 -fa on -b 512 -ub 512 --cpu-moe --moe-cache-mib 4000`. A small sweep (`-ncmoe` 32-48, MoE cache 1500-4000 MiB) put the best decode result at `--cpu-moe --moe-cache-mib 4000`.

No MTP row: RustingStrata prints `--mtp 1: not supported for this model (hyper-connections): decoding without drafts`; llama.cpp b12191 starts with `--spec-type draft-mtp` but fails with `context type MTP requested but model doesn't contain MTP layers`.

### Quality vs llama.cpp (Qwen3.8-Flash-Next IQ2_XS)

Same GGUF, 36 WikiText-2 raw test chunks of 512 tokens (9,180 scored), MTP disabled. RustingStrata wrote the KLD base:

```
rustingstrata perplexity --model M1 --file CORPUS --ctx-len 512 --chunks 36 --gpu --mtp 0 --kld-base rusting.bin
llama-perplexity -m M1 -f CORPUS -c 512 --chunks 36 --kl-divergence --kl-divergence-base rusting.bin \
  --cpu-moe --moe-cache-mib 4000 -ngl 99 -fa on -b 512 -ub 512
```

| Metric | Result |
|---|---:|
| PPL RustingStrata / llama.cpp | 2.9212 / 2.9156 |
| PPL ratio (llama.cpp / RustingStrata, per chunk) | 0.9987 +- 0.0019 |
| Mean KLD (median) | 0.0098 (0.0016) |
| Same top token | 96.78 +- 0.18% |
| RMS Δp | 3.60 +- 0.09% |
| Correlation of log-PPL | 99.58% |

Raw logs, server logs and commands: `.posts/data/qwen38-*.{log,txt,err}`, `speed_rs*.jsonl`, `speed_llama*.jsonl`.

## GLM-5.3-Flash UD-Q2_K_XL (119 GB)

Four shards, 119 GB, 30 GB RAM, so experts stream from SSD through the page cache. Measured 2026-11-13 on the reference machine; llama.cpp b12119 with `-ngl 99 --cpu-moe -c 2048`, RustingStrata `serve --gpu` with `STRATA_GLM_CTX=2048`. Both through `/v1/chat/completions` with streaming, 64 greedy tokens, run in the order llama.cpp, RustingStrata, RustingStrata, llama.cpp. The short prompt is "Explain in two sentences why the sky is blue." (23 tokens after the chat template); the long prompt is `tools/prompts/bench-natural.txt` (429 tokens). Each prompt was sent twice per session; decode is the mean of all runs, prefill is the first request of each session (llama.cpp's repeat hits its prompt cache).

| | RustingStrata | llama.cpp |
|---|---|---|
| Decode, short prompt | 1.57 tok/s (1.48-1.70) | 0.57 (0.53-0.61) |
| Decode, 429-token prompt | 1.35 tok/s (1.23-1.46) | 0.62 (0.56-0.67) |
| First token, short prompt, cold / warm cache | 55 s / 17-19 s | 35-38 s / 6-8 s |
| First token, 429-token prompt, cold | 84-93 s | 73-76 s |
| Load until ready | 5 s | 66-73 s |

Quality vs llama.cpp, same GGUF and prompts, teacher-forced: llama.cpp b12119's own greedy tokens (32 per prompt on the three `tools/glm_ref/p{1,2,3}` prompts) are fed through RustingStrata's GPU forward, and the top-5 distributions are compared at every step.

| Metric | Result |
|---|---|
| Same top token | 96 / 96 |
| Top-5 overlap | 91.9% |
| Mean \|dlogprob\| over shared top-5 ids | 0.230 |

How it was run:

- Speed: the raw JSON lines are in `.posts/data/2026-11-13-glm-bench.log` (local benchmark notes).
- Quality: `STRATA_GLM_FORCE=<ids> STRATA_GLM_TOP5=1 rustingstrata generate -n 32` against the `completion_probabilities` stored in `tools/glm_ref/p{1,2,3}_llama.json`; log in `.posts/data/2026-11-13-glm-quality.log`. `STRATA_GLM_PRELOAD=0` was set for the quality runs to keep memory pressure down.

Caveats: everything moves with the page cache, because the model is four times the size of RAM. llama.cpp's repeated-prompt TTFT (8-14 s on the long prompt) is its prompt cache, not prefill. One repeated 429-token RustingStrata request was OOM-killed under global memory pressure after the earlier runs; it is excluded. The four-shard file, build and prompts match the 2026-11-10 run below.

Previous paired run (2026-11-10, same machine): 1.25-1.40 tok/s against 0.47-0.56 on the short prompt, 1.25-1.33 against 0.53 on a 550-token prompt, first token after 17-19 s against 40-41 s, a 550-token first token after 206-209 s against 107-229 s, and 188/192 top-1 on 3 prompts x 64 tokens.

## DeepSeek-V4.1-Flash IQ2_XXS (249 GB)

| | RustingStrata | llama.cpp b12116 (`--cpu-moe`) |
|---|---|---|
| Decode | 2-3.6 tok/s | 0.3-1.5 tok/s |
| Prefill | ~2 tok/s short, ~0.5 tok/s at 1400 tokens | 0.5-0.9 tok/s |
| Top-1 agreement (teacher-forced) | 189/192 short, 21/23 on 1425 tokens | - |

Not measured side by side: the llama.cpp numbers are its first cold runs, the RustingStrata numbers are later runs. The model files were deleted afterwards, so this entry cannot be repeated until they are downloaded again.

On data-center cards (not this machine, RustingStrata only), with the experts in the page cache: end-to-end decode 4.5-4.6 tok/s at the default 9 GB VRAM expert budget on an A100 80 GB and an H100 NVL, and 6.6 (A100) and 8.9 (H100 NVL) tok/s at 60 GB, on one 30-token prompt and 33 decode tokens. See [datacenter-nvidia-results.md](datacenter-nvidia-results.md#2026-11-10--deepseek-v41-flash-1017-iq2_xxs).

## Recursive Flint 39.8B UD-Q5_K_XL

`flint-UD-Q5_K_XL.gguf`, 27.9 GiB, 39.79 B parameters, 49 layers (2 dense, 47 MoE), 256 experts, top-6. Measured
2026-11-14 on the reference machine. Prompt: `tools/prompts/bench-natural.txt` repeated ten times, 4102 tokens
(`.posts/data/2026-11-14-flint-prompt.txt`), chat template, greedy, 200 output tokens, the same OpenAI-compatible
streaming request for both engines. One 4K request thrown away, then three measured; RustingStrata ran first,
llama.cpp second, so llama.cpp got the warmer page cache.

| | RustingStrata 1.2.2 | llama.cpp b12191 |
|---|---:|---:|
| Decode, warm | 64.6 tok/s (63.8-65.6) | 48.6 (47.5-49.3) |
| Decode, first 4K request after load | 24.3 tok/s | 47.5 |
| Prefill, new 4K prompt, page cache warm | 1.14-1.30k tok/s | 0.85-1.02k |
| Expert hit rate | 54.7% | - |
| Worker busy | 12.98 ms/token | - |
| Resident experts | 3244 / 12032 (7157 MB VRAM) | - |
| Expert traffic | 622.1 MB/token | - |

How each was run:

- RustingStrata: `serve --model M --gpu --ctx 8192`; default placement. Raw runs in
  `.posts/data/2026-11-14-flint-rs-final-server.log` and `2026-11-14-flint-rs-final.jsonl`. The 64.6 warm run used
  the 1.2.1 build with the same placement code; 1.2.2 is that code with the unused n-gram/one-shot experiments
  removed.
- llama.cpp b12191 (commit `e95426f16`): `-ngl 99 -ncmoe 36 -fa on -t 6 -b 2048 -ub 2048 -c 8192`; raw runs in
  `.posts/data/2026-11-14-flint-llama-final-server.log` and `2026-11-14-flint-llama-final.jsonl`.
- Both: `.posts/data/bench.py PORT LABEL 3 PROMPT 200`, temperature 0, `enable_thinking: false`.
- Commands: `.posts/data/2026-11-14-flint-commands.txt`.

Not equal:

- RustingStrata ran before llama.cpp, so llama.cpp had the warmer page cache.
- The first 4K request depends on the page cache. A second RustingStrata load decoded 55.9 and 58.8 on its first two
  measured requests and was still climbing (`.posts/data/2026-11-14-flint-rs-repeat*.log`).
- Prefill: once the expert pages are cached, a new 4106-token prompt prefilled at 1.14-1.30k tok/s on RustingStrata
  and 0.85-1.02k on llama.cpp (mean 0.96k). The first prefill after a model load ranged 0.07-1.0k for both, and the
  order (RustingStrata then llama.cpp) changes which pages are cached; raw logs in
  `.posts/data/2026-11-14-flint-prefill-*.log`. After the first request both use their prompt cache.
- Decode follows the greedy text: a later RustingStrata run took the "However, they've pasted the same passage..."
  route and decoded at 41-46 tok/s (51.2% hit) on the current build; raw run in
  `.posts/data/2026-11-14-flint-rs-current*`. The 64.6 tok/s row is the fast route. A same-continuation comparison
  (teacher forcing) is still needed before publishing.
- The file has no MTP/draft head, so neither engine used speculative decoding.
- Quality (perplexity/KLD) is not measured yet; the post says so and is not ready to publish.

## Quality vs llama.cpp (Qwen3.6-35B-A3B UD-Q3_K_M)

Speed does not come from lost accuracy. Same GGUF, RTX 3060, llama.cpp b12191 `-ngl 99 -ncmoe 22`.
Corpus: repo docs prose, 36 chunks x 512 tokens (9180 scored), not wikitext.

| Metric | Result |
|---|---|
| PPL RustingStrata / llama.cpp | 9.0714 / 9.0587 (ratio 0.9991 +- 0.0013) |
| Mean KLD (median) | 0.0055 (0.0025) |
| Same top token | 96.3 +- 0.2% |
| RMS dp | 1.83% |
| Correlation of log-PPL | 99.88% |

Reproduce:
```
rustingstrata perplexity --model M --file F --ctx-len 512 --gpu --mtp 0 --kld-base base.bin
llama-perplexity -m M -f F -c 512 --kl-divergence --kl-divergence-base base.bin
```
llama.cpp batch vs token-by-token modes differ by about 1.7% in top-token agreement on their own.

## Inkling 2 Mini IQ4_NL (17.6 GB)

`Inkling-2-Mini-IQ4_NL.gguf` (architecture `inkling2_loop`, 26.07 B parameters), measured 2026-11-21 on the reference
machine (30 GB RAM), warm page cache, in one session. RustingStrata: `bench --pp 64,512,2048 --tg 64 --reps 2` with
`--backend cuda`, `--backend vulkan` and `STRATA_DS_GPU=0`; mean of the two runs, the two runs in brackets.
llama.cpp c622f62 built with CUDA: `llama-bench -t 6 -r 2 -p 64,512,2048 -n 64` with `-ngl 18` (`-ngl 99` runs out
of VRAM; 12, 16 and 20 layers measured within 3% of 18 at 64 and 512 tokens), and with `-ngl 0` at 64, 512 and 2048 tokens.

| | Prefill 64 tokens | Prefill 512 | Prefill 2048 | Decode (tok/s) |
|---|---|---|---|---|
| RustingStrata, CUDA | 120.7 (59.2, 182.2) | 259.2 (254.6, 263.9) | 220.2 (218.7, 221.8) | 34.5 / 31.2 / 31.2 after 64 / 512 / 2048 tokens |
| RustingStrata, Vulkan (same card) | 82.7 (34.5, 130.9) | 226.9 (226.6, 227.1) | 182.6 (179.0, 186.2) | 27.3 / 27.5 / 24.5 |
| RustingStrata, CPU only | 47.5 (47.1, 47.9) | 49.8 (49.7, 49.8) | 48.2 (48.2, 48.2) | 11.8 / 11.6 / 10.9 |
| llama.cpp, CUDA build, `-ngl 18` | 124.7 ± 3.6 | 383.9 ± 7.2 | 322.6 ± 1.0 | 8.7 ± 1.3 |
| llama.cpp, CUDA build, `-ngl 0` | 122.3 ± 5.0 | 377.5 ± 7.8 | 311.9 ± 4.4 | 10.5 ± 1.2 |

Quality vs llama.cpp, same GGUF, teacher-forced: llama.cpp c622f62's greedy tokens (CPU build, `/completion` at
temperature 0) are fed through RustingStrata's forward and the top token is compared at every step
(`tests/inkling2_forward.rs`). Short prompt: 24 of 24. Prompt of several hundred tokens: at least 47 of 48 (one near
tie). The same on the CPU, CUDA and Vulkan paths.

Caveats: llama.cpp reads the prompt faster (RustingStrata is at 0.6-0.7 of its speed from 512 tokens on): its CUDA
build runs the batch on the GPU even with no layer offloaded, and RustingStrata computes the attention scores of a
chunk on the CPU. `llama-bench` decodes 64 tokens from an empty context, `bench` decodes them after the prompt, so
the decode columns are not the same test. The first run of the 64-token prompt is the first work of the process and
is the slow one of the two. A CPU-only llama.cpp build was not benchmarked with `llama-bench`; its server reported
40.4 tok/s for the prompt and 11.0 tok/s for the reply earlier the same day (one run, not side by side). Vulkan was
run on the RTX 3060 only. Decoding is greedy in both engines.

The decode column was measured again later the same day, same command, after a decoded token moved onto the GPU
(it was 29.9 / 27.1 / 25.6 on CUDA, 21.5 / 20.9 / 19.8 on Vulkan and 12.0 / 11.6 / 11.3 on the CPU). A desktop app
used part of the GPU during that run. Alternating runs of the old and new build at a 512-token prompt, same load:
CUDA 25.4 to 33.5 tok/s, Vulkan 16.6 to 25.4 tok/s, prefill unchanged (within 1-4%). Most of the Vulkan gain is the
IQ4_NL expert matvec (105 to 270 GB/s on 8 experts of 2304x2304).

## Calibration v2 validation (RTX 3050 Laptop 4 GB)

The v2 `calibrate` command measures through `bench`, so its baseline and best trials are the benchmark. Two target
models, page cache warmed by the baseline, one child process per sample with `--reps 1`, greedy, built-in prose,
`--tg 16`, seed 1. Numbers are the mean of one sample (not a range); this is a search-validation run, not a
top-speed record. Full traces are in the JSON files under the cache directory and in the report written by `--out`.

### Tiel-Coder-35B-A3B-MTP-UD-IQ3_XXS (`qwen35moe`, 13.6 GB)

Command: `rustingstrata calibrate --model … --samples 10 --pp 128,512 --tg 16 --chunk-values 1024,2048,4096
--ctx-values 2048,4096 --seed 1`. Baseline knobs: chunk 1024, ctx 2048, mtp 0, short 16, threads 3. Best knobs:
**chunk 4096, ctx 2048, mtp 1, short 16, threads 3**, score 1.299 vs baseline (10 samples, 114.5 s, 0 failures).

| workload | prefill: baseline → best | decode: baseline → best | TTFT: baseline → best |
|---|---|---|---|
| 128-token prompt | 109.8 → **126.8** tok/s | 17.3 → **28.6** tok/s | 1165.8 → **1009.5** ms |
| 512-token prompt | 230.3 → **306.3** tok/s | 20.1 → **24.4** tok/s | 2223.2 → **1671.6** ms |

The gain is mostly the MTP draft head (accepted at 128 tokens) plus a larger chunk; the search found it jointly
rather than by optimizing drafts and prefill in separate sweeps. The old v1 calibration file for this model recorded
chunk 4096 and mtp 0.

### Inkling-2-Mini-UD-Q3_K_XL (`inkling2_loop`, 13.6 GB)

Command: `rustingstrata calibrate --model … --samples 8 --pp 128,512 --tg 16 --chunk-values 256,512,1024,2048
--ctx-values 2048,4096 --seed 1`. Baseline knobs: chunk 256, ctx 2048, threads 6. Best knobs: **chunk 256,
ctx 4096, threads 6**, score 1.166 vs baseline (8 samples, 314.2 s, 0 failures). This model used to be skipped by
`calibrate` entirely; it is now measured through its own `bench` path and its chunk/context are applied at load time.

| workload | prefill: baseline → best | decode: baseline → best | TTFT: baseline → best |
|---|---|---|---|
| 128-token prompt | 23.1 → **29.3** tok/s | 7.2 → **8.3** tok/s | 5541 → **4369** ms |
| 512-token prompt | 45.2 → **50.4** tok/s | 7.4 → 8.1 tok/s | 11327 → **10159** ms |

### `calibrate spec` on Inkling 2

Command: `rustingstrata calibrate spec --model … --tasks prose,coding --spec-types none,ngram-simple,ngram-mod,
ngram-cache --draft-max 4,8 --ngram-n 3,6 --mtp 0 --pp 128 --tg 16 --samples 8 --seed 1` (8 samples, score 1.077
vs baseline). Winner for both tasks: `ngram-cache`, 4 drafts, n-gram size 6. Decode: prose 8.5 tok/s, coding
6.5 tok/s. The sweep only added its `spec` section: the engine calibration above (chunk 256, ctx 4096) is still
in the file. Acceptance is reported as 0% because the short verbose prompt produced no drafts to accept in these
samples; the timing gain comes from the lookup path, not from accepted drafts.

## Not yet measured

- Qwen3-Next against llama.cpp in one table.
- Speculative/MTP for Qwen3.8-Flash-Next: neither RustingStrata nor llama.cpp has an MTP head for this file.
- Other GPUs: see [V100-results.md](V100-results.md) and [SUPPORTED_GPUS.md](SUPPORTED_GPUS.md).

## Adding an entry

One section per model and file. Give the machine, the file, the llama.cpp build and flags, the RustingStrata command, the number of runs, and what is not equal between the engines. Put raw logs next to the other results in `docs/` or link them.
