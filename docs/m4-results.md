# M4 results: chunked GPU prefill

Model: Qwen3.6-35B-A3B-UD-Q2_K_XL, RTX 3060 12 GB, Ryzen 5 7600X. Prompt: 3970 tokens (spec + plans text),
`strata generate --gpu --ctx 8192 -n 128`, default `--chunk 1024`. Three runs each, GPU otherwise idle.

| Commit | Prefill tok/s | Decode tok/s | Parity (3 runs) |
|---|---|---|---|
| 62ebb6f chunked prefill | ~444 (428-470, noisy) | 43-47 | 98.28% |
| 76a57e2 GEMM/DeltaNet registers | **1001** (961-1022) | **49.6** (49.1-49.9) | 98.18% |

Gate: prefill >= 1017 tok/s at 4K, decode > 67 tok/s, parity >= 99%.
