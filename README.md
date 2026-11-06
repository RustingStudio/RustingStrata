# RustingStrata

A pure-Rust inference engine for the Qwen3.5/3.6 hybrid model family (`qwen35moe`, `qwen35`) in GGUF; Qwen3-Next
(`qwen3next`), Qwen3.8-Flash-Next (`qwen4exp`) and Qwen3 (`qwen3`, `qwen3moe`) also run. CUDA kernels are compiled at run time with NVRTC through
cudarc (cached as a cubin), and the same kernels also run on Vulkan (`--backend vulkan`, GLSL compiled to SPIR-V and
committed): no C++ build step, no ggml. Experts that do not fit in VRAM run on the CPU at the same time as the GPU
work, and the expert placement adapts to the routing it sees.

Status: v1.2.0. Milestones M0-M6 are done (M6: server, chat, calibration, bench); the v1 gate run is in
`docs/m6-results.md`.

## Numbers

Reference machine: RTX 3060 12 GB, Ryzen 5 7600X, 32 GB RAM. Model: `Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf` (11.7 GiB,
does not fit in VRAM). llama.cpp is the 2026-10-28 build 11876, tuned for the same file (`-ncmoe 12 -ub 2048
-b 2048`, flash attention; with its default ubatch its 4K prefill is ~1550).

| | CUDA | Vulkan | llama.cpp |
|---|---|---|---|
| Prefill, 4K prompt | **2216 tok/s** (2169-2252) | 2154 (2140-2162) | ~2236 (2183-2310) |
| Decode after 4K prompt | **113.4 tok/s** (110.9-114.8) | 113.6 | 66.5 |
| Decode, short prompt (like tg128) | **118.2 tok/s** (117.0-118.8) | 117.3 | 68.4 |
| Decode with MTP drafts (`--mtp 1`, greedy), short / after 4K | **up to 160 / 152 tok/s** | 158 / 154 | (its MTP is slower) |
| 32K: prefill / decode | **1800 / 91.9 tok/s** | 1582 / 90.7 | |
| Greedy top-1 agreement with llama.cpp (20 prompts x 32 tokens), either llama.cpp mode | 98-99% | 98-99% | 98.3% (its batch vs token-by-token modes) |

Measured with `strata bench --gpu --mtp 0 --pp 512,4096,32768 --tg 128 --reps 3 --ctx 32896`, mean of 3 rounds
(range in brackets); llama.cpp from `llama-server` timings, 4047-token prompt + 128 tokens, 3 runs.
MTP gains depend on the text.
Agreement counts a token when it matches llama.cpp run token by token or with the context as one batch;
llama.cpp's own two modes agree on only 98.3%. The remaining misses are near-ties where
float rounding order picks a different token. The expert placement adapts between runs and the CPU and
GPU expert kernels round differently, so greedy output can change at such near-ties from one run to the next. MTP
on and off give identical greedy tokens.

With less VRAM, simulated on the 3060 with `--vram-margin` (not measured on real 8 GB or 6 GB cards): ~8 GB
keeps 116 / 110 tok/s decode (short / after 4K) and 2128 tok/s 4K prefill, ~6 GB 105 / 97 and 2056.

Other models, greedy "The capital of France is" with `--gpu` (CUDA / Vulkan, second run):

| Model | Types | Decode |
|---|---|---|
| Qwen3.5-2B (fine-tune) Q4_K_M | Q4_K, Q6_K, ... | 208 / 208 tok/s |
| Qwen3-4B-Instruct-2507 Q4_K_M (dense) | Q4_K, Q6_K | 104 / 110 tok/s |
| Qwen3.5-9B Q5_K_S | Q5_K, Q6_K, Q8_0 | 51 / 53 tok/s |
| Qwen3-30B-A3B-Instruct-2507 UD-Q2_K_XL | Q2_K to Q6_K | 114 / 123 tok/s |
| Qwen3.8-27B UD-IQ3_XXS (dense, 10.8 GB on GPU) | IQ1_S to IQ4_XS, Q2_K to Q8_0 | 20.6 / 20.5 tok/s |
| Qwen3.6-35B-A3B UD-IQ4_XS (57% of experts fit in VRAM) | IQ3_S, IQ4_XS, Q6_K, Q8_0 | 94 / 96 tok/s |
| Qwen3-Next-80B-A3B-Instruct UD-IQ1_S (most experts on the CPU) | IQ1_S to IQ4_NL, Q4_K to Q8_0, BF16 | 92 / 90 tok/s |
| Qwen3.8-Flash-Next GSQ-RCO IQ2_XS (64 GB in two files, pass the `-00001-of-00002` one; 18% of experts fit in VRAM, and with 32 GB RAM the rest are partly read from disk) | Q2_0, IQ1_M to IQ4_XS, Q6_K, Q8_0, F16, BF16 | 22 / 20 tok/s |

Supported GGUF types: F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_0, Q2_K-Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S,
IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL.

## Build

Needs Rust (2024 edition) and either an NVIDIA driver with the CUDA 13 NVRTC library (loaded at run time) or a
Vulkan 1.3 driver with shaderFloat64, int8/16/64 and subgroup size 32 (tested only on an RTX 3060; Intel Arc lacks
shaderFloat64, pre-RDNA AMD lacks subgroup size 32). `--gpu` picks CUDA when it loads, else Vulkan; `--backend`
chooses. The first CUDA run compiles the kernels (~15 s) and caches them. AMD GPUs: see [docs/AMD.md](docs/AMD.md)
for known issues and what to send in a report.

```sh
cargo build --release
```

Windows: install the Visual Studio C++ Build Tools first (one dependency compiles C code).

## Use

Without a terminal: run `strata` (double-click `strata.exe` on Windows; release binaries are on the GitHub releases
page). It starts the server on http://127.0.0.1:8080 and opens a chat page in the browser. The page keeps chats in
the browser, downloads models from Hugging Face (Models and settings; `HF_TOKEN` for gated repos), switches between
the models in the models folder (`%LOCALAPPDATA%\rusting_strata\models` on Windows,
`~/.local/share/rusting_strata/models` elsewhere, or `serve --models-dir`), and shows the context use, speed, totals and CPU, memory and graphics memory use
of the server. Other apps that take an OpenAI-compatible address (Open WebUI, Jan, SillyTavern, code editors) can
use `http://127.0.0.1:8080/v1` too.

From a terminal:

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
It runs one request at a time; `--parallel N` (one unsplit GPU) lets N replies take turns, each at about 1/N the
speed of a single one, with MTP off. `temperature`, `top_k`, `top_p` and `seed` are honored, and decoding is greedy when
they are left out. When a request repeats the previous request's messages and adds new ones, only the new part is
prefilled (`usage.prompt_tokens_details.cached_tokens`). `tools` and `tool_choice` work as in the OpenAI API (calls
come back as `tool_calls` with `finish_reason: "tool_calls"`). The prompt follows the model's own chat template,
rendered like llama.cpp's `--jinja`. `tool_choice: "required"` does not force a call.
`POST /v1/messages` speaks the Anthropic Messages API (text, `tool_use` and `tool_result` blocks, streamed or not,
the key in `x-api-key`), so Claude Code can use the server:
`ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_API_KEY=secret ANTHROPIC_MODEL=local ANTHROPIC_SMALL_FAST_MODEL=local claude`.
OpenCode and other agents use the OpenAI address. Qwen3.6's template leaves the think block out of replies before the
last user message, so each new user message changes the text of the earlier turns and the whole conversation is
prefilled again. Strata keeps those blocks by default, so a long agent session prefills only what was added;
`--preserve-thinking false` (or `chat_template_kwargs.preserve_thinking` in a request) follows the template.
A request with another history (an agent's side request for a title or a summary) parks the conversation's state in
host memory (`--cache-ram`, 4096 MB by default), and the next request that continues it takes it back instead of
prefilling it again.
The server has one key and treats all clients as one user: the cached-token count and the response time show
whether the previous request started with the same messages, so do not share one server between people who must
not learn each other's prompts.

Useful flags: `--ctx` (KV cache size, default 4096), `--mtp [N]` (speculative decoding with the model's draft
head; greedy tokens stay the same, sampling keeps its distribution), `--chunk`, `--threads`, `--vram-margin`. On a
GPU that drives no display, `--vram-margin 256` keeps more experts resident (+1-2% prefill); 128 runs out of
memory. `--kv q8` halves the KV cache (345 instead of 650 MB at 32K) at about 6% slower 32K decode and 14% slower
32K prefill. Without `--gpu` everything runs on the CPU reference, which is slow and meant for checking.

Several GPUs: `--device 0,1` splits the layers over the cards in proportion to their free VRAM, so more experts stay
resident. The cards run one after the other, so one request gets more memory, not twice the speed of one card. MTP
drafting is off on a split model. AMD cards clock down in the gaps while the other card works; `echo high >
/sys/class/drm/cardN/device/power_dpm_force_performance_level` (as root, per card) took a two-card split from 44 to 57
tok/s in one report.

The experts that do not fit in VRAM are read from the GGUF through the page cache. A file system that keeps file data
out of the page cache (ZFS with `primarycache=metadata`) makes every run read them from disk again; load the model
from a file system that caches data, or from a tmpfs copy.

## Tests

```sh
cargo test --release                      # GPU tests need the model: STRATA_MODEL=path/to/model.gguf
python -I tools/parity.py $M --extra "--gpu" --tag gpu   # agreement with llama.cpp (needs llama-server once)
```

## Layout

`src/gguf.rs` (file reader), `src/quant/` (dequantization), `src/model.rs` (CPU reference forward),
`src/cuda/` and `src/vulkan/` (GPU kernels and shaders, behind `src/backend.rs`), `src/gpu_model.rs` (GPU forward,
MTP, verify), `src/gpu_model/prefill.rs` (chunked prefill), `src/expert_cache.rs` (expert placement), `src/cpu_moe.rs`
and `src/cpu_worker.rs` (CPU expert compute), `src/serve.rs` (HTTP), `src/chat.rs` (chat templates and tool calls),
`src/main.rs` (CLI).

## License

RustingStrata is released under the [RustingStrata License 1.0](LICENSE.md).
You may use, modify and distribute it, including for commercial services, as
long as you keep the license and credit Vasyl Trefilov as the original author.
You may not sell RustingStrata itself.
