# LoRA adapters

`--lora` applies a GGUF LoRA adapter when the model loads. The base GGUF is not changed and nothing is requantized,
so one base model can run with or without an adapter, or with a different strength, from the same file.

The main use is abliteration: an adapter that removes the refusal direction makes any supported base model
uncensored without a separate abliterated download. Other LoRA adapters work too, within the limits below.

Contributed by [@jazir555](https://github.com/jazir555) in
[#15](https://github.com/RustingStudio/RustingStrata/pull/15).

## Usage

```
rustingstrata generate --model MODEL.gguf --lora ADAPTER.gguf --prompt "..." -n 50
rustingstrata chat     --model MODEL.gguf --lora ADAPTER.gguf:0.8
rustingstrata serve    --model MODEL.gguf --lora ADAPTER.gguf:1.0 --ctx 32768
```

- `--lora PATH` or `--lora PATH:SCALE`. The scale is 1.0 when it is left out. Windows paths with a drive letter
  (`C:\adapters\a.gguf`) work with and without a scale.
- Repeat the flag for several adapters; they add up.
- It works on the CPU and on the GPU (CUDA and Vulkan), in `generate`, `chat`, `serve`, `bench`, `perplexity` and
  `score`.
- `calibrate` accepts the flag but ignores it: calibration always measures the base model.

## Adapter format

A GGUF adapter as written by llama.cpp's `convert_lora_to_gguf.py`: for each target tensor `NAME` it holds
`NAME.lora_a` and `NAME.lora_b`, and optionally `adapter.lora.alpha` in the metadata. The applied weight is

```
W' = W + m * B * A      m = scale * alpha / rank, or scale when the adapter has no alpha
```

The adapter must be made for the same base model: a target with a different shape is skipped with a warning.

## What is applied

- **Routed expert down projections** (`blk.N.ffn_down_exps.weight`): stay quantized. The correction
  `m * B * (A * activation)` is added for each selected expert while the model runs, on the CPU and on the GPU. An
  adapter with one expert slice is applied to every expert.
- **Every other target** (`attn_output`, `ssm_out`, `ple_value`, `ffn_down_shexp`, ...): the tensor is dequantized
  to F32 once at load, the adapter is added, and the CPU and GPU both read the result.

These messages on stderr mean a target was left out:

```
lora: target NAME not found, skipping
lora: target NAME is RxC, adapter is RxC; skipping
lora: target NAME has N experts, expected 1; skipping
```

The last one is for per-expert tensors other than `ffn_down_exps` (the expert gate and up projections), which are
not supported.

## Cost and limits

- Prompts are processed token by token instead of in batched chunks, so prefill is slower with an adapter.
- Recorded GPU graphs are off while an adapter is loaded, and the expert correction is computed on the host for each
  token, so decode is slower too. MTP drafts still work.
- Each dequantized target is held as F32 in host RAM and counts toward `--max-ram`.
- Only models that run through the Qwen-family forward take adapters (`qwen3`, `qwen3moe`, `qwen3next`, `qwen35`,
  `qwen35moe`, `qwen4exp`). `deepseek41`, `glm5-next` and `gemma4` do not apply `--lora`.
- Output is not compared against llama.cpp with the same adapter; the LoRA arithmetic itself is plain F32.
