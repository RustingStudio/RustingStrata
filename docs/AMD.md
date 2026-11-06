# AMD GPUs

All benchmarks in the README were run on an NVIDIA RTX 3060. AMD GPUs run the Vulkan backend
(`--backend vulkan`), and everything known about them comes from user reports (RX 9070 XT on Linux with RADV, and on
Windows with AMD's driver). Expect problems; this page lists the known ones and what to send when you hit a new one.

Needs a Vulkan 1.3 driver and subgroup size 32 or a fixed subgroup size 64. The wave64 path has also been tested on Radeon VII (Vega 20) with RADV Mesa 26.0.8. Qwen3 dense decode on wave64 uses unquantized activations for accuracy and can be slower than the int8 path. Batched prefill still uses quantized activations; `--chunk 0` uses token-by-token prefill when closer agreement with the CPU reference matters more than prompt processing speed.

## Wave64 Qwen3 speed and accuracy

On fixed wave64 GPUs, dense Qwen3 decode uses unquantized activations by default. Set
`STRATA_VK_QWEN3_F32_LAYERS` before starting Strata to use unquantized activations only in the final N layers;
earlier layers use int8 activations. The output head remains unquantized. For example:

```sh
STRATA_VK_QWEN3_F32_LAYERS=8 strata generate --model model.gguf --gpu --backend vulkan --chunk 0 --prompt "Hello" -n 128
```

This is an opt-in speed/accuracy tradeoff; results depend on the model and prompt. On one Radeon VII with
Qwen3-0.6B Q8_0, N=8 raised short-context decode from 118.8 to 177.1 tokens/s (median of three runs), while
teacher-forced CPU top-1 agreement on 14 prompts fell from 224/224 to 221/224 tokens. For Qwen3-4B
Instruct Q4_K_M the same setting raised decode from 21.6 to 44.8 tokens/s (median of three), while agreement
on five prompts fell from 40/40 to 39/40. These samples are not guarantees of accuracy or performance. The usual batched
prefill introduces its own numerical differences; `--chunk 0` uses token-by-token prefill for closer CPU agreement.
Unset the variable to retain the accuracy-oriented default. This switch has no effect on other model architectures
or devices using 32-lane subgroups.

## First line of the output

Every run on Vulkan starts with a line like:

```
vulkan: AMD Radeon RX 9070 XT (radv Mesa 26.2.5), device-coherent memory yes, cooperative matrix f16 yes int8 no, conditional rendering yes
```

- `device-coherent memory` must be `yes` for models whose experts do not all fit in VRAM (see below).
- `cooperative matrix` speeds up prefill. `int8 no` is normal on RDNA.
- `conditional rendering` lets decode recover when the CPU expert worker is slow. With `no`, a slow worker is an error.

## Known issues

**Slow decode with experts on the CPU, and "GPU waits for the CPU expert worker that timed out and resumed: N".**
While the CPU computes the experts that are not in VRAM, the GPU polls a small mailbox buffer for the answer. On AMD the
GPU caches that buffer unless it comes from device-coherent memory (`VK_AMD_device_coherent_memory`), and then it misses
the answer and waits until it gives up. RADV hides the extension on RDNA4 (RX 9000) unless the driconf option
`radv_device_coherent_memory` is set. Strata sets it before it opens Vulkan (unless you set it yourself) and prints a
warning when an AMD GPU still has no device-coherent memory. If the first line says `device-coherent memory no`, send
the output of `vulkaninfo | grep -i coherent`. A few resumed waits on a busy CPU are normal; hundreds are not.

**The GPU resets during a long prefill.** amdgpu resets a submission that runs for more than 10 s. Prefill submits
each layer separately to stay under that. If it still happens, try a smaller `--chunk`.

**"the forward pass produced NaN or infinite logits".** Seen once on Qwen3-Next 80B with a 15K prompt. Rerun with
`STRATA_NAN_TRACE=1`, which prints per-layer magnitudes and the first layer with a non-finite value, and attach the
output.

**A model split over two cards (`--device 0,1`) is slower than expected.** Each card waits while the other works, and
amdgpu clocks it down in those gaps. `echo high > /sys/class/drm/cardN/device/power_dpm_force_performance_level` (as
root, for each card; `auto` restores the default) keeps the clocks up: 44 to 57 tok/s decode in one two-card report.
The per-device times that `generate` prints on a split show whether a card is slow.

**Load fails for lack of VRAM.** The desktop on the same card takes memory, and the driver's free-memory estimate can
be more than it will allocate. Lower `--ctx`, use `--kv q8`, or a smaller `--chunk`.

## Switches for testing

Environment variables; set any of them to turn the feature off:

- `STRATA_VK_NO_COOP`: no cooperative matrices.
- `STRATA_VK_NO_COND`: no conditional rendering.
- `STRATA_VK_NO_TQ`: no separate transfer queue for expert uploads.
- `radv_device_coherent_memory=false`: undo the RADV option strata sets.

If one of these makes a problem go away, say which one in the report.

## What to send in an issue

- GPU model, OS, driver and version (Mesa version on Linux).
- The output of `vulkaninfo --summary`.
- The full output of the failing command, including the first `vulkan:` line.
- If decode is slow: `strata bench --model <file> --gpu --backend vulkan --pp 1024 --tg 128` with the same model.
