# M4 results: chunked GPU prefill

Model: Qwen3.6-35B-A3B-UD-Q2_K_XL, RTX 3060 12 GB, Ryzen 5 7600X. Prompt: 3970 tokens (spec + plans text),
`strata generate --gpu --ctx 8192 -n 128`, `--chunk` as listed (1024 until 8d0c3c4, then 4096). Three runs each, GPU otherwise idle.

| Commit | Prefill tok/s | Decode tok/s | Parity (3 runs) |
|---|---|---|---|
| 62ebb6f chunked prefill | ~444 (428-470, noisy) | 43-47 | 98.28% |
| 76a57e2 GEMM/DeltaNet registers | 1001 (961-1022) | 49.6 (49.1-49.9) | 98.18% |
| 8d0c3c4 chunk 4096 default | **1366** (1363-1369) | 49.6 (49.2-50.0) | 98.09% (6 runs, 624-632) |
| 1866f5e split decode attention | 1360 (warm runs 1316-1360) | **60.9** (59.8-62.4) | 98.28% (628-630) |

Short-prompt decode (8-token prompt, -n 128): 85.8 tok/s at 8d0c3c4, 83.3 at 1866f5e (5 clean runs each).

Gate: prefill >= 1017 tok/s at 4K, decode > 67 tok/s, parity >= 99%.
