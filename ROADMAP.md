# Roadmap

> **Note:** everything in this roadmap is planned, but new things such as new architectures and new models have
> higher priority than the roadmap. When a new model comes out, we support it first and then come back here.

The roadmap is not only the owner's decision: it actively takes ideas from the community. Open an issue or comment
on one to suggest an item or to move one up.

Every change keeps single-user greedy output unchanged by default and must match llama.cpp in `tools/parity.py`
before it ships.

## Serving

- **vLLM-style serving for MoE models on a lot of RAM and little VRAM.** vLLM and SGLang expect the whole model in
  VRAM; nobody serves many users well from a machine with a big RAM pool and a small card. We want to be the first:
  continuous batching where the experts of a batch step are read once for all requests, a resident expert set that
  follows what the batch uses, and the misses split between CPU threads and PCIe staging. Single-request speed must
  not drop.
- **Automatic shared prefix caching.** Requests that start with the same tokens (a system prompt, a tool list, a long
  document) reuse one KV prefix instead of each reading it again.
- **Tensor parallelism and multi-node serving.** Split one model across GPUs by tensor, not only by layer, and run a
  model across more than one machine.

## Performance

- **Prefill on low-end hardware.** On a small card with the experts in RAM, prefill is the slowest part and the main
  thing users feel. Layer-major prefill (each expert goes to the card once per prompt, not once per chunk), faster CPU
  expert kernels and better overlap of PCIe copies with compute.
- **Inkling 2.** Large optimization pass on prefill, decode and VRAM use.
- **Vulkan on par with CUDA.** Vulkan is still clearly slower than CUDA on the same card. The two backends should be
  close to equal for every supported model.
- **Cold start from disk.** When a model is larger than both VRAM and RAM, the first prompt is limited by page-cache
  and disk reads. Read ahead in the order the forward needs, and keep the hot experts in memory.

## Backends

- **Apple Silicon (Metal).** A Metal backend for M-series Macs with unified memory.

## Features

- **Better MCP support** in the chat page and the server.
- **Embedding and rerank pipelines** with OpenAI-compatible endpoints.
- **Non-LLM modalities** beyond the current image input.

## Model formats

- **More than GGUF.** Today the main models load only from GGUF. Load Hugging Face Safetensors directly, and the
  common quantized formats: FP8, AWQ, GPTQ and EXL2.
