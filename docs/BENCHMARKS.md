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
| Qwen3.6-35B-A3B UD-Q2_K_XL | 118.2 tok/s (113.4 after 4K) | 68.4 (66.5) | - | [below](#qwen36-35b-a3b-ud-q2_k_xl) |
| Qwen3.6-35B-A3B UD-Q3_K_M, 6474-token prompt | 83.0 tok/s (108.0 with MTP) | 67.5 (101.1 with MTP) | PolyStrata 79.2 (113.3 with MTP) | [below](#qwen36-35b-a3b-ud-q3_k_m-vs-polystrata) |
| GLM-5.3-Flash UD-Q2_K_XL (119 GB) | 1.25-1.40 tok/s | 0.47-0.56 | - | [below](#glm-53-flash-ud-q2_k_xl-119-gb) |
| DeepSeek-V4.1-Flash IQ2_XXS (249 GB) | 2-3.6 tok/s | 0.3-1.5 | - | [below](#deepseek-v41-flash-iq2_xxs-249-gb) |

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

## GLM-5.3-Flash UD-Q2_K_XL (119 GB)

Four shards, 119 GB, 30 GB RAM, so experts stream from SSD through the page cache. llama.cpp b12119 with `-ngl 99 --cpu-moe -c 2048`. Both through `/v1/chat/completions` with streaming, 64 greedy tokens, run in the order llama.cpp, RustingStrata, RustingStrata, llama.cpp (ranges are the two runs).

| | RustingStrata | llama.cpp |
|---|---|---|
| Decode, short prompt | 1.40 tok/s | 0.47-0.56 |
| Decode, 550-token prompt | 1.25-1.33 tok/s | 0.53 |
| First token, 12-token prompt | 17-19 s | 40-41 s |
| First token, 550-token prompt | 206-209 s | 107-229 s |
| Load until ready | 4-6 s | 58-59 s |
| Peak resident memory | 27.5 GB | 26.5-27.2 GB |
| Top-1 agreement (teacher-forced, 3 prompts x 64 tokens) | 188/192 | - |

Caveats: long-prompt prefill is not a win (llama.cpp's second run had the page cache warm); everything moves with the page cache; each pair was run twice, not as a full suite. The four top-1 misses are near-ties in llama.cpp's own logprobs.

## DeepSeek-V4.1-Flash IQ2_XXS (249 GB)

| | RustingStrata | llama.cpp b12116 (`--cpu-moe`) |
|---|---|---|
| Decode | 2-3.6 tok/s | 0.3-1.5 tok/s |
| Prefill | ~2 tok/s short, ~0.5 tok/s at 1400 tokens | 0.5-0.9 tok/s |
| Top-1 agreement (teacher-forced) | 189/192 short, 21/23 on 1425 tokens | - |

Not measured side by side: the llama.cpp numbers are its first cold runs, the RustingStrata numbers are later runs. The model files were deleted afterwards, so this entry cannot be repeated until they are downloaded again.

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

## Not yet measured

- Original Strata and other engines on the same files.
- Qwen3.8-Flash-Next (`qwen4exp`) and Qwen3-Next against llama.cpp in one table.
- Other GPUs: see [V100-results.md](V100-results.md) and [SUPPORTED_GPUS.md](SUPPORTED_GPUS.md).

## Adding an entry

One section per model and file. Give the machine, the file, the llama.cpp build and flags, the RustingStrata command, the number of runs, and what is not equal between the engines. Put raw logs next to the other results in `docs/` or link them.
