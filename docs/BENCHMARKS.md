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
| Qwen3.6-35B-A3B UD-Q3_K_M, 5290-token prompt | 90.9 tok/s (128.4 with MTP) | 53.5 (77.7 with MTP) | PolyStrata 78.5 (97.5-100.4 with MTP) | [below](#qwen36-35b-a3b-ud-q3_k_m-vs-polystrata) |
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

`Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` (Unsloth, MTP layer included, 15.9 GiB, does not fit in VRAM). Prompt of 5290 tokens, 128 generated tokens, greedy. llama.cpp b12119.

| | RustingStrata | PolyStrata | llama.cpp |
|---|---|---|---|
| Prefill, 5290-token prompt | 2330 tok/s | 1458-1549 tok/s | 1692 tok/s (`-ncmoe 22`, `-ub 2048`) |
| Decode, no MTP | 90.9 tok/s | 78.5 tok/s (`--no-draft`, not a clean no-MTP number) | 53.5 tok/s |
| Decode with MTP drafts | 128.4 tok/s | 97.5-100.4 tok/s | 77.7 tok/s (`--spec-type draft-mtp`, 92 of 102 drafts accepted) |

How each was run:

- RustingStrata: `rustingstrata bench --gpu --mtp 0|1 --pp 5290 --tg 128 --reps 3`, mean of 3.
- PolyStrata: its own `polystrata.py benchmark --model qwen3.6 --prompt-tokens 4000 --max-tokens 128`, which came to 5290 tokens. Three runs; the first after setup was cold (decode 24.6 tok/s) and is left out. MTP drafts are on by default. For the no-MTP row the engine's `--no-draft` flag was set; it still offered a few drafts afterwards (6 of 16 and 15 of 36 accepted in the log). Prefill in those runs: 1602-1643 tok/s.
- llama.cpp: `llama-bench -ngl 99 -fa 1 -ub 2048 -b 2048 -p 5290 -n 128 -r 3`, `-ncmoe` swept 18 to 26; 22 was the fastest that fits in 12 GB. The MTP row is `llama-server --spec-type draft-mtp` with `-ncmoe 23` (22 does not fit with the MTP context and drops to 20 tok/s), a 5381-token prompt of repo docs, 3 runs.

Not equal:

- PolyStrata offers three sizes of Qwen3.6 (UD-Q3_K_M, UD-Q2_K_XL, UD-Q4_K_M, all from Unsloth's MTP repo). RustingStrata and llama.cpp take any GGUF. The Q3_K_M file was used for all three.
- PolyStrata's benchmark prompt is a page of its own docs, RustingStrata's is generated prose, so the two MTP rows are not an exact comparison. The no-MTP row is the clean one for RustingStrata against llama.cpp.
- PolyStrata's engine for this model is built on llama.cpp's CUDA kernels. What else differs was not checked.

Raw PolyStrata runs: `poly_run*.json`, `poly_nodraft*.json` (kept outside the repo).

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

## Not yet measured

- Original Strata and other engines on the same files.
- Qwen3.8-Flash-Next (`qwen4exp`) and Qwen3-Next against llama.cpp in one table.
- Other GPUs: see [V100-results.md](V100-results.md) and [SUPPORTED_GPUS.md](SUPPORTED_GPUS.md).

## Adding an entry

One section per model and file. Give the machine, the file, the llama.cpp build and flags, the RustingStrata command, the number of runs, and what is not equal between the engines. Put raw logs next to the other results in `docs/` or link them.
