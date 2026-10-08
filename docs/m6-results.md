# M6 results: v1 gate

Model: Qwen3.6-35B-A3B-UD-Q2_K_XL, RTX 3060 12 GB, Ryzen 5 7600X (6 cores / 12 threads). 2026-10-08, 21:38-21:46,
same session for both engines, nothing else on the GPU (899 MiB used before the run). Host load 6.0 at the start
(left over from a llama-server parity job), 1.4-2.4 during the llama-bench runs.

Strata: `strata bench --gpu --pp 512,4096 --tg 128 --reps 3` (and `--pp 32768 --ctx 33280`) at 671050e, mean
(min-max) of 3 reps. llama.cpp: `llama-bench -ngl 99 -fa 1 -r 3` (CUDA build 5d8ccdf9d), `--n-cpu-moe 10` at short
and 4K (best setting from `docs/baseline.md`), `--n-cpu-moe 14 -d 32768` at 32K, mean +- std.

| Test | Strata | Strata MTP k = 1 | llama.cpp |
|---|---|---|---|
| prefill, 512 tokens | 676.9 (359.6-836.2) | 853.6 (840.7-861.9) | 1046.4 +- 23.3 |
| prefill, 4096 tokens | **1715.6** (1618.4-1765.5) | 1759.2 (1754.2-1763.4) | 1048.1 +- 1.9 |
| decode 128 after 512 | **81.7** (80.0-83.3) | **88.9** (85.7-90.8) | 69.8 +- 0.2 |
| decode 128 after 4096 | 78.3 (76.4-79.4) | 78.7 (78.4-79.0) | — |
| prefill, 32768 tokens | 1413.0 (1394.8-1423.1) | 1418.1 (1411.4-1425.2) | 775.7 +- 59.1 (pp512 @ d32768) |
| decode 128 after 32768 | **67.8** (67.5-68.1) | 64.9 (63.7-65.6) | 57.9 +- 0.2 (tg128 @ d32768) |

- MTP drafts accepted: 64.1% at 512/4K and at 32K. Expert hit rate 97.1-99.2%.
- llama.cpp's 32K numbers are marginal rates at depth 32768 (a 512-token batch and 128 tokens after a filled
  cache); Strata's 32K prefill is the average over the whole 32768-token prompt.
- Short-prompt prefill is slower than llama.cpp: 512 tokens is one chunk, and the first rep includes cache warm-up
  (359.6 min).

## Gate

| Item | Gate | Result |
|---|---|---|
| Prefill at 4K | >= 1017 tok/s | 1715.6, passed (1.64x llama.cpp) |
| Decode | > 67 tok/s | 81.7 short / 78.3 after 4K, passed (1.17x llama.cpp); 88.9 with MTP |
| Parity vs llama.cpp | >= 99% | Open: 98.39% mean vs the llama.cpp CPU-build reference (#32), 97.34% vs the CUDA-build reference (#34); llama.cpp scores 98.28% against its own reference (#33) |

MTP pays at short context only: +7.2 tok/s after 512 tokens, +0.4 after 4K, -2.9 after 32K.
