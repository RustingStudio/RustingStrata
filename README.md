# RustingStrata

A pure-Rust inference engine for the Qwen3.5/3.6 hybrid model family (`qwen35moe`, `qwen35`) in GGUF. CUDA kernels
are compiled at run time with NVRTC through cudarc: no C++ build step, no ggml. Experts that do not fit in VRAM run
on the CPU at the same time as the GPU work, and the expert placement adapts to the routing it sees.

Status: v1 milestones M0-M5 are done, M6 (server, chat, calibration, bench) is mostly done.
`docs/LOOP_LOG.md` has every change with its measurements.

## Numbers

Reference machine: RTX 3060 12 GB, Ryzen 5 7600X, 32 GB RAM. Model: `Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf` (11.7 GiB,
does not fit in VRAM). llama.cpp is tuned for the same file (`--n-cpu-moe 10`, flash attention, see
`docs/baseline.md`).

| | RustingStrata | llama.cpp |
|---|---|---|
| Prefill, 4K prompt | **1581 tok/s** (1538-1610) | 1017 |
| Decode after 4K prompt | **81.0 tok/s** (76.8-81.8) | 65.8 |
| Decode, short prompt (like tg128) | **88.1 tok/s** | 65.8 |
| Decode with MTP drafts (k = 1, greedy) | **~87-92 tok/s**, 82% of drafts accepted | 61.6 (its MTP is slower) |
| 32K: prefill / decode | ~1290 / ~62 tok/s | 814 (pp512 at depth) / 56.4 |
| Greedy top-1 agreement with llama.cpp (20 prompts x 32 tokens) | 98.3% | - |

Agreement is below the 99% v1 target: the remaining misses are mostly near-ties where llama.cpp's int8 activation
rounding and this engine's kernels pick different tokens. MTP on and off give identical greedy tokens.

## Build

Needs Rust (2024 edition), an NVIDIA driver and the CUDA 13 NVRTC library (loaded at run time).

```sh
cargo build --release
```

## Use

```sh
M=path/to/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf

# Measure the best prefill chunk, short-prompt threshold and MTP drafts on this machine once.
# Later runs read ~/.cache/rusting_strata/strata-<model>.json unless flags override it.
target/release/strata calibrate --model $M

target/release/strata generate --model $M --gpu --prompt "Write a haiku about rivers." -n 128
target/release/strata chat --model $M --gpu                 # --think for thinking mode, --temp 0.7 to sample
target/release/strata bench --model $M --gpu --pp 512,4096 --tg 128
STRATA_API_KEY=secret target/release/strata serve --model $M --gpu --addr 127.0.0.1:8080
```

`serve` speaks the OpenAI chat completions API (`POST /v1/chat/completions`, streamed or not, `GET /v1/models`).
It runs one request at a time. `temperature`, `top_k`, `top_p` and `seed` are honored, and decoding is greedy when
they are left out. When a request repeats the previous request's messages and adds new ones, only the new part is
prefilled (`usage.prompt_tokens_details.cached_tokens`).
The server has one key and treats all clients as one user: the cached-token count and the response time show
whether the previous request started with the same messages, so do not share one server between people who must
not learn each other's prompts.

Useful flags: `--ctx` (KV cache size, default 4096), `--mtp [N]` (speculative decoding with the model's draft
head; only greedy decoding uses it), `--chunk`, `--threads`, `--vram-margin`. On a GPU that drives no display,
`--vram-margin 256` keeps more experts resident (+1-2% prefill); 128 runs out of memory. Without `--gpu`
everything runs on the CPU reference, which is slow and meant for checking.

## Tests

```sh
cargo test --release                      # GPU tests need the model: STRATA_MODEL=path/to/model.gguf
python -I tools/parity.py $M --extra "--gpu" --tag gpu   # agreement with llama.cpp (needs llama-server once)
```

## Layout

`src/gguf.rs` (file reader), `src/quant/` (dequantization), `src/model.rs` (CPU reference forward),
`src/cuda/` (GPU kernels), `src/gpu_model.rs` (GPU forward, MTP, verify),
`src/gpu_model/prefill.rs` (chunked prefill), `src/expert_cache.rs` (expert placement), `src/cpu_moe.rs` and `src/cpu_worker.rs`
(CPU expert compute), `src/serve.rs` (HTTP and chat template), `src/main.rs` (CLI).
Design: `docs/superpowers/specs/2026-10-06-rusting-strata-v1-design.md`.
