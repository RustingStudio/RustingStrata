# RustingStrata

A pure-Rust inference engine for the Qwen3.5/3.6 hybrid model family (`qwen35moe`, `qwen35`) in GGUF; Qwen3-Next
(`qwen3next`) and Qwen3 (`qwen3`, `qwen3moe`) also run. CUDA kernels are compiled at run time with NVRTC through
cudarc (cached as a cubin), and the same kernels also run on Vulkan (`--backend vulkan`, GLSL compiled to SPIR-V and
committed): no C++ build step, no ggml. Experts that do not fit in VRAM run on the CPU at the same time as the GPU
work, and the expert placement adapts to the routing it sees.

Status: v1.1.0. Milestones M0-M6 are done (M6: server, chat, calibration, bench); the v1 gate run is in
`docs/m6-results.md`.

## Numbers

Reference machine: RTX 3060 12 GB, Ryzen 5 7600X, 32 GB RAM. Model: `Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf` (11.7 GiB,
does not fit in VRAM). llama.cpp is the 2026-10-28 build 11876, tuned for the same file (`-ncmoe 12 -ub 2048
-b 2048`, flash attention; with its default ubatch its 4K prefill is ~1550).

| | CUDA | Vulkan | llama.cpp |
|---|---|---|---|
| Prefill, 4K prompt | **2203 tok/s** (2199-2205) | 2101 (2091-2109) | ~2236 (2183-2310) |
| Decode after 4K prompt | **111.5 tok/s** (110.4-112.6) | 112.8 | 66.5 |
| Decode, short prompt (like tg128) | **119.0 tok/s** (118.7-119.3) | 116.8 | 68.4 |
| Decode with MTP drafts (`--mtp 1`, greedy), short / after 4K | **130-134 / 117-119 tok/s** | 119 / 111 | (its MTP is slower) |
| 32K: prefill / decode | **1675 / 88.0 tok/s** | 1436 / 87.3 | |
| Greedy top-1 agreement with llama.cpp (20 prompts x 32 tokens), either llama.cpp mode | 98-99% | 98-99% | 98.3% (its batch vs token-by-token modes) |

Measured 2026-10-30: `strata bench --gpu --pp 512,4096,32768 --tg 128 --reps 3 --ctx 32896`, mean of 3 rounds
(range in brackets); llama.cpp from `llama-server` timings, 4047-token prompt + 128 tokens, 3 runs.
MTP gains depend on the text.
Agreement counts a token when it matches llama.cpp run token by token or with the context as one batch;
llama.cpp's own two modes agree on only 98.3%. The remaining misses are near-ties where
float rounding order picks a different token. The expert placement adapts between runs and the CPU and
GPU expert kernels round differently, so greedy output can change at such near-ties from one run to the next. MTP
on and off give identical greedy tokens.

Other models, greedy "The capital of France is" with `--gpu` (2026-10-30, CUDA / Vulkan, second run):

| Model | Types | Decode |
|---|---|---|
| Qwen3.5-2B (fine-tune) Q4_K_M | Q4_K, Q6_K, ... | 208 / 208 tok/s |
| Qwen3-4B-Instruct-2507 Q4_K_M (dense) | Q4_K, Q6_K | 104 / 110 tok/s |
| Qwen3.5-9B Q5_K_S | Q5_K, Q6_K, Q8_0 | 51 / 53 tok/s |
| Qwen3-30B-A3B-Instruct-2507 UD-Q2_K_XL | Q2_K to Q6_K | 94 / 114 tok/s |
| Qwen3.8-27B UD-IQ3_XXS (dense, 10.8 GB on GPU) | IQ1_S to IQ4_XS, Q2_K to Q8_0 | 20.6 / 20.5 tok/s |
| Qwen3.6-35B-A3B UD-IQ4_XS (57% of experts fit in VRAM) | IQ3_S, IQ4_XS, Q6_K, Q8_0 | 94 / 96 tok/s |
| Qwen3-Next-80B-A3B-Instruct UD-IQ1_S (most experts on the CPU) | IQ1_S to IQ4_NL, Q4_K to Q8_0, BF16 | 24 / 30 tok/s |

Supported GGUF types: F32, BF16, Q8_0, Q2_K-Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS,
IQ4_NL.

## Build

Needs Rust (2024 edition) and either an NVIDIA driver with the CUDA 13 NVRTC library (loaded at run time) or a
Vulkan 1.3 driver with shaderFloat64, int8/16/64 and subgroup size 32 (tested only on an RTX 3060; Intel Arc lacks
shaderFloat64, pre-RDNA AMD lacks subgroup size 32). `--gpu` picks CUDA when it loads, else Vulkan; `--backend`
chooses. The first CUDA run compiles the kernels (~15 s) and caches them.

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
`src/cuda/` and `src/vulkan/` (GPU kernels and shaders, behind `src/backend.rs`), `src/gpu_model.rs` (GPU forward,
MTP, verify), `src/gpu_model/prefill.rs` (chunked prefill), `src/expert_cache.rs` (expert placement), `src/cpu_moe.rs`
and `src/cpu_worker.rs` (CPU expert compute), `src/serve.rs` (HTTP and chat template), `src/main.rs` (CLI).

## License

RustingStrata is released under the [RustingStrata License 1.0](LICENSE.md).
You may use, modify and distribute it, including for commercial services, as
long as you keep the license and credit Vasyl Trefilov as the original author.
You may not sell RustingStrata itself.
