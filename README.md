# RustingStrata

A pure-Rust inference engine for the Qwen3.5/3.6 hybrid model family (`qwen35moe`, `qwen35`) in GGUF. CUDA kernels
are compiled at run time with NVRTC through cudarc: no C++ build step, no ggml. Experts that do not fit in VRAM run
on the CPU at the same time as the GPU work, and the expert placement adapts to the routing it sees.

Status: v1.0.0. Milestones M0-M6 are done (M6: server, chat, calibration, bench); the v1 gate run is in
`docs/m6-results.md`.
`docs/LOOP_LOG.md` has every change with its measurements.

## Numbers

Reference machine: RTX 3060 12 GB, Ryzen 5 7600X, 32 GB RAM. Model: `Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf` (11.7 GiB,
does not fit in VRAM). llama.cpp is tuned for the same file (`--n-cpu-moe 10`, flash attention, see
`docs/baseline.md`).

| | RustingStrata | llama.cpp |
|---|---|---|
| Prefill, 4K prompt | **2093 tok/s** (2089-2096) | 1017 |
| Decode after 4K prompt | **95.6 tok/s** (95.4-95.8) | 65.8 |
| Decode, short prompt (like tg128) | **101.2 tok/s** (100.6-101.5) | 65.8 |
| Decode with MTP drafts (`--mtp 1`, greedy), short / after 4K | **110.9 / 99.0 tok/s**, 63% of drafts accepted | 61.6 (its MTP is slower) |
| 32K: prefill / decode / decode with MTP | **1562 / 77.4 / 80.8 tok/s** | 814 (pp512 at depth) / 56.4 |
| Greedy top-1 agreement with llama.cpp (20 prompts x 32 tokens), either llama.cpp mode | 98.9%; **99.3%** with `--llama-numerics --fixed-experts` | 98.3% (its batch vs token-by-token modes) |

Measured 2026-10-09 at 654a270: `strata bench --gpu --pp 512,4096 --tg 128 --reps 3`, mean of 3 rounds (range of
the round means in brackets), CPU load 4-7 from other jobs; 32K is one round of 3 reps with `--ctx 33280`.
Agreement counts a token when it matches llama.cpp run token by token or with the context as one batch;
llama.cpp's own two modes agree on only 98.3% (`docs/LOOP_LOG.md` #33). The remaining misses are near-ties where
float rounding order picks a different token (#60-#64). MTP on and off give identical greedy tokens.

Other models, greedy "The capital of France is" with `--gpu` (output identical to llama.cpp for 24 tokens):

| Model | Types | Decode |
|---|---|---|
| Qwen3.5-2B Q4_K_M | Q4_K, Q6_K, ... | 182 tok/s |
| Qwen3.5-9B IQ3_M | IQ3_S, Q4_K, BF16 | 48.8 tok/s |
| Qwen3.8-27B UD-IQ3_XXS (dense, 10.8 GB on GPU) | IQ1_S to IQ4_XS, Q2_K to Q8_0 | 19.3 tok/s |
| Qwen3.6-35B-A3B UD-IQ4_XS (57% of experts fit in VRAM) | IQ3_S, IQ4_XS, Q6_K, Q8_0 | 34-47 tok/s cold, 83 tok/s once the placement has adapted |

Supported GGUF types: F32, BF16, Q8_0, Q2_K-Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS.

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
`--vram-margin 256` keeps more experts resident (+1-2% prefill); 128 runs out of memory. `--kv q8` halves the KV
cache (345 instead of 650 MB at 32K) at about 6% slower 32K decode and 14% slower 32K prefill. Without `--gpu`
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

## License

RustingStrata is released under the [RustingStrata License 1.0](LICENSE.md).
You may use, modify and distribute it, including for commercial services, as
long as you keep the license and credit Vasyl Trefilov as the original author.
You may not sell RustingStrata itself.
