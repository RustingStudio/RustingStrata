# V100 results

Macchina: Tesla V100-SXM2-32GB, driver 580.178.04, Vulkan 1.4.312, CachyOS,
Ryzen/host come da `nvidia-smi`; `strata` v1.2.0 + fix `gemmi_dp` (ramo `v100`).

## 2026-11-06 — Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf, Vulkan

```
plan: weights+state+scratch 2580 MB (KV 642 MB for 32896 tokens, prefill 512 MB),
      expert slots 10040 MB, resident experts 10240/10240
pp    512: prefill  635.6 (361.6-774.2) tok/s | tg 128: decode 109.8 (82.7-123.4) tok/s
pp   4096: prefill 1423.9 (1422.0-1427.0) tok/s | tg 128: decode 108.6 (91.1-117.7) tok/s
pp  32768: prefill  523.5 (522.1-525.1) tok/s | tg 128: decode  95.7 (95.4-96.1) tok/s
expert hit rate 100.0% (0 misses), cache swaps 0
```

Nota: la prima ripetizione di `pp 512` include la compilazione delle pipeline
Vulkan; il valore a regime è ~775 tok/s.

## 2026-11-06 — Qwen3.8-27B-UD-Q4_K_XL.gguf, Vulkan (dense, senza tensor core)

```
plan: weights+state+scratch 17118 MB (KV 512 MB for 8192 tokens, prefill 706 MB)
pp    512: prefill 317.9 (315.8-319.4) tok/s | tg 128: decode 27.6 (27.5-27.7) tok/s
pp   4096: prefill 291.1 (290.5-291.6) tok/s | tg 128: decode 26.6 (26.1-26.9) tok/s
```

## Riferimento llama.cpp (stessa scheda)

Misurato con `llama-server` (`/completion`, `cache_prompt: false`, temp 0):

| Modello | prompt | prefill | decode |
| --- | --- | --- | --- |
| Qwen3.6-35B-A3B Q2_K_XL | 416 tok | 496 tok/s | 101.9 tok/s |
| Qwen3.6-35B-A3B Q2_K_XL | 1261 tok | 1243 tok/s | 101.9 tok/s |
| Qwen3.6-35B-A3B Q2_K_XL | 2941 tok | 1352 tok/s | 99.3 tok/s |
| Qwen3.8-27B Q4 (produzione, dai log) | 1092 / 4381 tok | 292 / 491 tok/s | ~49-53 tok/s |

## Note

- Prima del fix, qualsiasi prefill a chunk su Vulkan terminava con
  `Xid 13 SKEDCHECK18_L1_CONFIG_TOO_SMALL` e device lost.
- `mma` sm_80 e `cp.async` non esistono su sm_70: il backend CUDA richiede il
  port descritto in `docs/V100.md`.
