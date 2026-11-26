# Roadmap

> **Note:** everything in this roadmap is planned, but new things such as new architectures and new models have
> higher priority than the roadmap. When a new model comes out, we support it first and then come back here.

The roadmap is not only the owner's decision: it actively takes ideas from the community. Many items below came
from issues, Reddit and GitHub comments. Open an issue or comment on one to suggest an item or to move one up.

Every change keeps single-user greedy output unchanged by default and must match llama.cpp in `tools/parity.py`
before it ships.

## Performance

- Batched decode for many requests at once (vLLM-style serving), MoE and DeltaNet included; single-request speed must
  not drop (#98).
- Batched prefill across requests and faster GEMM at 16 to 128 rows.
- Inkling 2: close the speed and VRAM gaps against llama.cpp (#101).
- Layer-major prefill for models larger than VRAM: each expert goes to the card once per prompt, not once per chunk.
- Faster CPU expert kernels (AVX-512, IQ2 / Q5_K / Q6_K).
- Resident expert set ranked by measured use (#53).
- Multi-GPU: tensor split (#70) and pipelined prefill across split cards.
- Vulkan decode speed when all experts are in RAM.
- `calibrate` without a model reload per sample.

## Fixes

- Inkling 2: CPU fallback for the q8 KV cache when VRAM is short (#97).
- HIP does not detect the current device (#102).
- A dense model on Windows shows as both in RAM and in VRAM (#100).
- A split model keeps no cached conversation, so a side request makes the next turn read the whole prompt again.
- Low-VRAM load: pick a smaller prefill chunk when the default does not fit, and name `--chunk` in the error.
- Vulkan can pick an iGPU as device 0.

## Features

- More flexible KV cache allocation and more cache quantization types (#105).
- Inkling 2: speculative drafting on the loop path (#66).
- MTP on a multi-GPU split and on Flash-Next (#38).
- A weaker second GPU or iGPU that only computes expert misses (#67).
- `router.ini` like llama.cpp: many models behind one port (#86).
- `response_format` JSON mode.
- `--ple` to load a separate n-gram table for Flash-Next.

## Models

- Inkling 2 GPU vision encoder.
- Vision input for more models.
- New architectures on request (LFM, Ling, Spark), each checked for scope first.

## Hardware support

- AMD on Windows (#109).
- CUDA on Turing (RTX 20xx, GTX 16xx) and Pascal (GTX 10xx).
- Mixed NVIDIA + AMD multi-GPU.

## UI/UX

- Web page: accent color follows the loaded model, with an option to pick it by hand.
- MCP tools in the chat page: test with more servers and on Windows.
- Clearer first-run messages: what to calibrate and which flags matter for the card.

## Docs

- One page per model family with tested quants, flags and speeds.
- A troubleshooting page for the common load and speed reports.
