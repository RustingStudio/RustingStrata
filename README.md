# RustingStrata

[![Reviewed by Aletheore](https://img.shields.io/badge/Reviewed%20by-Aletheore-16140F)](https://aletheore.com)

A pure-Rust inference engine for GGUF models, built for MoE models that do not fit in VRAM. The CUDA kernels are
compiled at run time with NVRTC, and the same kernels run on Vulkan: no C++ build step, no ggml. Experts that do not
fit in VRAM run on the CPU at the same time as the GPU work, and the expert placement adapts to the routing it sees.

Supported models: Qwen3, Qwen3-Next, Qwen3.5/3.6, Qwen3.8-Flash-Next, Recursive Flint, Gemma 4, DeepSeek-V4.1-Flash,
GLM-5.3-Flash, Hy4-flash, Inkling 2 Mini and Kolibri 1.1. The architectures, the GGUF types and the notes for each model are in
[docs/MODELS.md](docs/MODELS.md).

Version 1.3.5.

## Numbers

Qwen3.6-35B-A3B on an RTX 3060 12 GB (Ryzen 5 7600X, 32 GB RAM), CUDA, greedy, against llama.cpp tuned for the same
file. Neither file fits in VRAM.

| | RustingStrata | llama.cpp |
|---|---|---|
| UD-Q5_K_M (25.2 GiB), decode after a 4K prompt | **80.5 tok/s** | 64.4 |
| UD-Q5_K_M, prefill of a 4K prompt | **1530 tok/s** | 1118-1122 |
| UD-Q2_K_XL (11.7 GiB), decode after a 4K prompt | **113.4 tok/s** | 66.5 |
| UD-Q2_K_XL, prefill of a 4K prompt | 2216 tok/s | ~2236 |

### Many users at once

`serve --parallel N` decodes the requests together in one batched forward (CUDA). Same machine, same load script
(`tools/loadtest.py`: N streamed chats at once, 128 reply tokens each, greedy), total decode tok/s:

| Clients | Qwen3-4B, RustingStrata (Q4_K_M) | Qwen3-4B, vLLM 0.33 (bf16) | Qwen3.6-35B-A3B UD-Q5_K_M, RustingStrata |
|---:|---:|---:|---:|
| 1 | 91.3 | 36.7 | 49.0 |
| 4 | 228.1 | 139.9 | 88.9 |
| 16 | 799.0 | 520.1 | 100.7 |
| 64 | 1698 | 1688-1765 | 129.3 |

vLLM runs bf16 weights and RustingStrata Q4_K_M, so the dense columns are not like for like. The 27 GB MoE file does
not fit on the 12 GB card, so vLLM was not run on it. RustingStrata reads each expert once for all requests of a step,
runs the experts that are not in VRAM on the CPU, and keeps slot state that does not fit in VRAM in RAM. The MoE
column moves by about 30% between runs on this machine (the 27 GB file is read through a 30 GB page cache).

The method, the ranges, Vulkan, the other models and the quality checks are in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Install

Windows (PowerShell; installs to `%LOCALAPPDATA%\Programs\RustingStrata`, adds it to PATH and makes a Start menu
entry that opens the chat page):

```powershell
irm https://raw.githubusercontent.com/RustingStudio/RustingStrata/main/install.ps1 | iex
```

Linux x86_64 (installs to `~/.local/bin`):

```sh
curl -fsSL https://raw.githubusercontent.com/RustingStudio/RustingStrata/main/install.sh | sh
```

Arch Linux: `yay -S rustingstrata-bin` (AUR). Or download a binary from the
[releases page](https://github.com/RustingStudio/RustingStrata/releases), or build it (Rust 2024 edition; on Windows
install the Visual Studio C++ Build Tools first):

```sh
cargo build --release    # the binary is target/release/rustingstrata
```

Then `rustingstrata serve --gpu --open` opens the chat page, which downloads models, and `rustingstrata interface`
asks a few questions in the terminal. `rustingstrata update` replaces the binary with the newest release (`--check`
only says whether one is out); `serve` prints a line at start when a newer release is out. A binary installed by a
package manager (the AUR package) is updated with that package manager.

A GPU run needs one of:

- an NVIDIA driver with the CUDA NVRTC library (loaded at run time). Volta (sm_70), Turing (sm_75) and Ampere or newer
  (sm_80+) work; CUDA 13 dropped Volta, so a V100 needs the 12.x toolkit. Pascal and older use Vulkan.
  The library is found by name (`libnvrtc.so`; on Windows `nvrtc64_130_0.dll` or `nvrtc64_120_0.dll`, on `PATH` or
  under `%CUDA_PATH%\bin`); `STRATA_NVRTC` names another file. Without it the run falls back to Vulkan.
- a Vulkan 1.3 driver with shaderFloat64, int8/16/64 and subgroup size 32 or a fixed 64 (tested on RTX 3060, Radeon VII
  and V100; Intel Arc lacks shaderFloat64).
- for `--backend hip`: a ROCm runtime with hiprtc (`libamdhip64.so`, `libhiprtc.so`), also loaded at run time; the
  kernels are compiled for the card's `gfx` target on first use. Experimental — status, numbers and limits are in
  [docs/HIP.md](docs/HIP.md).

`--gpu` picks CUDA when it loads, else Vulkan; `--backend` chooses (`cuda`, `vulkan` or `hip`). The first CUDA run compiles the kernels (~15 s) and
caches them. Every GPU tried so far is in [docs/SUPPORTED_GPUS.md](docs/SUPPORTED_GPUS.md).

## Quick start

Run `rustingstrata` with no arguments (double-click `rustingstrata.exe` on Windows). It starts a server on
http://127.0.0.1:8080 and opens a chat page that downloads models from Hugging Face and switches between the ones in
your models folder. Any app that takes an OpenAI-compatible address (Open WebUI, Jan, SillyTavern, code editors) can
use `http://127.0.0.1:8080/v1`.

`rustingstrata interface` asks for the model, the mode (terminal chat, chat page, API server, speed test), GPU or CPU,
the context length and the address, prints the command these make and runs it.

With flags:

```sh
M=path/to/model.gguf

rustingstrata calibrate --model $M --dry-run  # preview the workload matrix and knob space
rustingstrata calibrate --model $M     # once per model and machine: best chunk, MTP drafts, short threshold
rustingstrata chat --model $M --gpu    # --think for thinking mode, --temp 0.7 to sample
rustingstrata generate --model $M --gpu --prompt "Write a haiku about rivers." -n 128
rustingstrata serve --model $M --gpu --addr 127.0.0.1:8080
rustingstrata bench --model $M --gpu --pp 512,4096 --tg 128
```

Without `--gpu` the model runs on the CPU reference, which is slow and meant for checking. DeepSeek, GLM, Hy4 and
Inkling 2 open the GPU by themselves.

## Docs

- [USAGE.md](docs/USAGE.md): flags, the chat page and MCP tools, the OpenAI and Anthropic APIs, prompt caching,
  several GPUs.
- [CALIBRATION.md](docs/CALIBRATION.md): the budgeted search, task corpora, the `spec` sweep and the JSON file.
- [MODELS.md](docs/MODELS.md): supported architectures and GGUF types, notes and caveats for each model.
- [BENCHMARKS.md](docs/BENCHMARKS.md): every model against llama.cpp; older runs are in the
  [benchmark history](docs/README.md#benchmark-history).
- [HOSTING.md](docs/HOSTING.md): running a server for other people.
- [SUPPORTED_GPUS.md](docs/SUPPORTED_GPUS.md) and [AMD.md](docs/AMD.md): the GPUs tried so far and AMD notes.
- [VISION.md](docs/VISION.md): image input. [LORA.md](docs/LORA.md): LoRA adapters.

## Development

```sh
cargo test --release                      # GPU tests need the model: STRATA_MODEL=path/to/model.gguf
python -I tools/parity.py $M --extra "--gpu" --tag gpu   # agreement with llama.cpp (needs llama-server once)
```

CI only builds, so run the tests before a pull request.

Source map: `src/gguf.rs` (file reader), `src/quant/` (dequantization), `src/model.rs` (CPU reference forward),
`src/cuda/` and `src/vulkan/` (GPU kernels, behind `src/backend.rs`), `src/gpu_model.rs` (GPU forward),
`src/expert_cache.rs` (expert placement), `src/deepseek/`, `src/glm/`, `src/hy4/`, `src/inkling2/` and `src/gemma4/`
(the models with their own forward), `src/serve.rs` (HTTP), `src/chat.rs` (chat templates and tool calls),
`src/main.rs` (CLI).

## License

RustingStrata is released under the [RustingStrata License 1.0](LICENSE.md). You may use, modify and distribute it,
including for commercial services, as long as you keep the license and credit Vasyl Trefilov as the original author.
You may not sell RustingStrata itself.
