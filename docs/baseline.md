# llama.cpp baseline (M0)

Reference machine: RTX 3060 12 GB (about 1.7 GB used by the desktop), Ryzen 5 7600X, 32 GB RAM, CachyOS.
llama.cpp commit `5d8ccdf9d`, CUDA build (sm_86), flash attention on, all layers on GPU (`-ngl 99`).
Model: `Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf` (11.70 GiB).

## `--n-cpu-moe` sweep (`tools/baseline.sh`, llama-bench, 2 repetitions)

| `--n-cpu-moe` | pp4096 tok/s | tg128 tok/s |
|---|---|---|
| 6, 8, 9 | does not fit (out of VRAM) | |
| **10** | **1017.5 ± 4.5** | **65.8 ± 1.7** |
| 12 | 1005.7 ± 4.7 | 63.9 ± 1.6 |
| 14 | 974.9 ± 4.1 | 63.4 ± 0.9 |
| 16 | 940.7 ± 2.1 | 60.5 ± 1.9 |
| 20 | 883.5 ± 0.8 | 58.4 ± 0.2 |

## Long context (32K)

`-d 32768`: n=10 and n=12 do not fit. n=14: pp512 814.0 tok/s, tg128 56.4 tok/s.

## MTP speculative decoding (`tools/mtp_bench.sh`, llama-server, 256 tokens, temperature 0)

| `--n-cpu-moe` | draft-n-max | decode tok/s | accepted / drafted |
|---|---|---|---|
| 10 | 0 (no MTP) | 67.0 | - |
| 10 | 1, 2 | crash (segfault, likely VRAM) | |
| 12 | 1, 2, 3 | crash | |
| 14 | 1 | 61.6 | 116 / 138 |
| 14 | 2 | 61.1 | 149 / 210 |
| 14 | 3 | crash | |

MTP does not help llama.cpp on this machine: the extra MTP layer forces more experts to the CPU,
and the loss outweighs the gain from accepted drafts.

## Gate for RustingStrata v1

- Decode: faster than **67 tok/s** (llama.cpp best; its MTP is slower than plain decode here).
- Prefill: at least **1017 tok/s** at 4K.
- Top-1 agreement with llama.cpp: at least 99%.
