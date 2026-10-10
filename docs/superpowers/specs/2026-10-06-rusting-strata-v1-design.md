# RustingStrata v1 — Design

Date: 2026-10-06
Status: approved in brainstorming, pending written-spec review

## Goal

A pure-Rust inference engine for the Qwen hybrid model family (`qwen35`, `qwen35moe` in GGUF), in the spirit of
[Strata](https://github.com/Niko1221/Strata) (C++/CUDA, Qwen3.8-Flash-Next only), but generic over the family and
built on cudarc + NVRTC like RustingBrain. No C++ build step, no ggml dependency.

v1 runs **Qwen3.6-35B-A3B-UD-Q2_K_XL** (Unsloth, MTP variant, 12.5 GB) on the target PC and must beat a tuned
llama.cpp on the same file.

## Target hardware (v1 reference machine)

- NVIDIA RTX 3060 12 GB, PCIe 4.0 x16
- AMD Ryzen 5 7600X (Zen 4, AVX-512), 6 cores / 12 threads
- 32 GB RAM
- Linux (CachyOS), CUDA 13.x

## The model family

Read from the local GGUFs:

| Key | Qwen3.6-35B-A3B (`qwen35moe`) | Qwen3.8-27B (`qwen35`) |
|---|---|---|
| block_count | 40 (+1 `nextn` MTP layer) | 64 (+1 `nextn`) |
| embedding_length | 2048 | 5120 |
| full_attention_interval | 4 (3 Gated DeltaNet : 1 gated attention) | 4 |
| attention heads / kv heads / head dim | 16 / 2 / 256 | 24 / 4 / 256 |
| rope | MRoPE, sections [11, 11, 10, 0], dim 64, base 1e7 | same |
| ssm (DeltaNet) | inner 4096, state 128, groups 16, dt rank 32, conv 4 | inner 6144, dt rank 48 |
| FFN | 256 experts, top-8, expert ff 512, 1 shared expert ff 512 (gated by `ffn_gate_inp_shexp`) | dense ff 17408 |
| vocab | 248,320 | 248,320 |

Per-layer tensors (MoE): `attn_norm`, `post_attention_norm`; DeltaNet layers: `attn_qkv`, `attn_gate`, `ssm_a`,
`ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_dt.bias`, `ssm_norm`, `ssm_out`; attention layers: `attn_q` (query +
gate, 2× width), `attn_k`, `attn_v`, `attn_q_norm`, `attn_k_norm`, `attn_output`; MoE: `ffn_gate_inp`,
`ffn_{gate,up,down}_exps`, `ffn_{gate,up,down}_shexp`, `ffn_gate_inp_shexp`. MTP: `nextn.eh_proj`, `nextn.enorm`,
`nextn.hnorm`, `nextn.shared_head_norm` plus a full attention+MoE block.

The exact op semantics (gating, norms, DeltaNet update rule, MRoPE position layout) are taken from llama.cpp's
`qwen35`/`qwen35moe` graph and verified numerically against it (see Testing), not re-derived.

### v1 quant types

Exactly the types present in the target file: **IQ2_XS, IQ3_XXS, IQ4_XS, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, BF16,
F32**. Any other type in a file is a load-time error naming the tensor and type.

## Scope

In v1:

- GGUF loader, generic `ModelConfig` from metadata
- `qwen35moe` forward (the `qwen35` dense FFN path is included because it is small, but 27B is not a v1 gate)
- VRAM expert cache + concurrent CPU expert compute (adaptive)
- Chunked prefill with expert streaming
- MTP speculative decoding
- OpenAI-compatible server, terminal chat, calibration, bench

Not in v1 (later sub-projects, each with its own spec):

1. Qwen3.8-27B as a gated target + remaining IQ types (IQ1_S, IQ1_M, IQ2_XXS, IQ2_S, IQ3_S)
2. Vision (mmproj)
3. Multi-GPU
4. Windows, installer, web UI, model download

## Success criteria

On the reference machine, same GGUF file:

1. **Correctness:** greedy top-1 token agreement ≥ 99% vs llama.cpp on a fixed 20-prompt set (code, chat, long
   document). MTP on and off produce identical greedy output.
2. **Decode speed:** tokens/s higher than llama.cpp's best tuned configuration (`--n-cpu-moe` / `-ot` sweep, plus its
   MTP if it supports `nextn` for this architecture) at short context and at 32K context.
3. **Prefill speed:** ≥ llama.cpp at 4K and 32K-token prompts.

## Architecture

One crate, `rusting_strata`, at `~/Code/rusting/RustingStrata`. Standalone: no dependency on RustingBrain (pieces may be
ported). Dependencies kept to: `cudarc` (nvrtc, runtime, cublas only if a kernel loses to it), `memmap2`, `rayon`
(or a hand-rolled pinned worker pool if rayon's latency is the bottleneck), `half`, `tokenizers`, `minijinja`,
`serde`/`serde_json`, `clap`, an HTTP server crate (`axum` or `tiny_http`, chosen in the plan), `anyhow`.

### Modules

| Module | Responsibility | Depends on |
|---|---|---|
| `gguf` | mmap file, parse header/metadata/tensor index, typed `ModelConfig` from `<arch>.*` keys | memmap2 |
| `quant` | Block layouts for the v1 types, IQ grid/sign tables, scalar reference dequant (the oracle) | — |
| `cpu` | AVX-512 quantized GEMV per type vs Q8_K-quantized activations, expert FFN (gate, up, SwiGLU, down), worker pool | quant |
| `cuda` | Context, NVRTC compile with on-disk cache, quantized GEMV/GEMM per type, RMSNorm, MRoPE, gated attention (flash decode + flash prefill), Gated DeltaNet (recurrent + chunked), conv1d state, router top-k, sampling | quant |
| `model/qwen35` | Layer schedule from config, dense or MoE FFN, MTP head; owns KV cache and DeltaNet/conv state | cuda, cpu |
| `experts` | VRAM slot pool, hit statistics, routing profile persist/seed, miss dispatch to CPU, prefill streaming | cuda, cpu |
| `spec` | MTP drafting, batched verify, acceptance, state rollback | model |
| `tokenizer` | GPT-2 BPE from GGUF vocab/merges, chat template via minijinja | gguf |
| `server` | `/v1/chat/completions`, `/v1/models`, SSE streaming, single active request with queue, prefix reuse | all |

Binary `strata` with subcommands `serve`, `chat`, `bench`, `calibrate`.

Universality rule: nothing in `model/qwen35` hardcodes sizes; everything derives from `ModelConfig`. The one
abstraction is the FFN variant (dense vs MoE), which has two real implementations from day one. A future block type
is a new module, not a flag.

## Data flow

### Decode (one token)

1. Embedding row looked up on CPU (`token_embd` stays in RAM, mmap), dequantized, uploaded (2048 floats).
2. For each layer on GPU:
   - RMSNorm → mixer. DeltaNet layers: qkv projection, conv1d with rolling state, recurrent gated delta update,
     norm, gate, output projection. Attention layers (every 4th): q/k/v, q/k norm, MRoPE, flash-decode over KV,
     output gate, output projection.
   - Residual, `post_attention_norm`, router GEMV, softmax, top-8.
   - Expert split: resident experts (cache hits) and the shared expert run on GPU. Miss ids, routing weights and the
     normed activation are written to host-mapped pinned memory and a flag is raised; a CPU worker spin-waits on the
     flag, computes misses with AVX-512, writes results back to mapped memory and raises a done flag; the GPU kernel
     that sums expert outputs waits on that flag. CPU work overlaps GPU hits + shared expert. A layer with no misses
     never waits on the CPU.
3. Final norm, output head (Q4_K GEMV over 248K vocab), sampling (greedy / temperature / top-k / top-p) on GPU.

Kernel sequences between CPU sync points are captured as CUDA graphs.

### Prefill (chunks up to 4096 tokens, size calibrated)

- DeltaNet: chunked parallel form. Attention: flash-attention prefill with causal mask.
- MoE: tokens sorted by expert; non-resident experts of layer L+1 stream over PCIe into a double-buffered staging area
  while layer L computes; grouped GEMM per expert dequantizes tiles to fp16 in shared memory and uses tensor-core MMA.
- Prompts shorter than a calibrated threshold (order of 32 tokens) use the decode path (CPU misses) instead.

### MTP speculative decoding

- Draft: `nextn` head on `[enorm(embed(t)), hnorm(h_last)]` → `eh_proj` → its attention+MoE block →
  `shared_head_norm` → output head. Chain of k drafts, k ∈ 1..3, adapted from the running acceptance rate.
- Verify: main model runs on `[t, d1..dk]` as a batch; MoE computes the union of needed experts.
- Accept: longest matching prefix + one bonus token. Sampling mode uses standard speculative-sampling acceptance.
- Rollback: KV cache truncates by length. DeltaNet recurrent state and conv state are snapshotted per verify position
  (≈60 MB per position for 30 DeltaNet layers in fp32) and the accepted position's snapshot is restored.

### Expert cache adaptation

- Per-expert hit counter with exponential decay.
- Every N decoded tokens and between requests, a background copy on a separate stream replaces the coldest resident
  experts with the hottest missed ones.
- Routing profile is saved next to the model config on exit and loaded at start (warm start). Without a profile, the
  initial fill is by layer order.
- Milestone M2 uses a fixed placement (no adaptation) as the first working baseline.

## Memory planning

At startup a deterministic planner reads free VRAM and RAM, then reserves in order:

1. Non-expert weights (≈2 GB for the target file; embeddings stay in RAM)
2. KV cache for the requested context (q8 default, f16 option; ≈20 KB/token at f16 for the target model)
3. DeltaNet/conv state + MTP snapshots
4. Prefill staging buffers, scratch
5. Safety margin (default 512 MB, `--vram-margin`)

The remainder becomes expert slots. The planner prints every reservation. If items 1–5 do not fit, it exits with a
message naming what to lower (context, chunk size, margin).

RAM: experts stay in the mmap and are pinned with `cudaHostRegister` in per-layer slices; if pinning fails, fall back
to staged copies with a warning. Warn at start if RAM < file size + 4 GB.

Rough budget for the target file on 12 GB: 6–8 GB of expert slots, i.e. 60–75% of the 10.3 GB of experts; the hit
rate should be well above that because routing is skewed.

## Error handling

- Unknown architecture or unsupported quant type → refuse at load, name tensor and type.
- NVRTC compile failure → print kernel name and compiler log, exit.
- CUDA error during a request → request fails (server: HTTP 500), session state reset.
- CPU pool watchdog: no progress for 60 s → error with every worker's state dumped.
- Server: malformed request → 400; concurrent requests queue (one active).

## Configuration

CLI flags: `--model`, `--ctx`, `--kv q8|f16`, `--chunk`, `--mtp-k`, `--threads`, `--vram-margin`, `--host`, `--port`,
`--api-key`. `strata calibrate` measures threads, chunk size, MTP k and short-prompt threshold and writes
`strata-<model>.json` beside the routing profile. CLI flags override the file. No other config system.

## Testing

- **Quant oracle:** a one-time script using `~/Clones/llama.cpp/gguf-py` dumps dequantized fixture blocks for every v1 type
  from the target file; Rust scalar dequant must match exactly.
- **Kernel parity:** each CPU and CUDA kernel vs the Rust scalar reference, with stated tolerances (GEMV per type,
  DeltaNet recurrent vs chunked, flash decode/prefill, router top-k, MRoPE, sampling).
- **Model parity:** fixed 20-prompt set, greedy, vs `llama-server` with `n_probs`; top-1 agreement ≥ 99% and a
  logit-delta report.
- **Spec parity:** MTP on vs off, greedy, identical tokens.
- **Speed gate:** `strata bench` vs `llama-bench`/`llama-server` on the same file, llama.cpp tuned; decode at short and
  32K context, prefill at 4K and 32K.

## Milestones

Each milestone ends runnable and verified.

- **M0** Baseline: tuned llama.cpp numbers on the reference machine, recorded in `docs/baseline.md`.
- **M1** `gguf` + `quant` + pure-CPU reference forward; correct tokens vs llama.cpp. Becomes the golden oracle.
- **M2** CUDA kernels + fixed expert placement; correct; first speed numbers.
- **M3** Adaptive expert cache + concurrent CPU misses.
- **M4** Chunked prefill with expert streaming.
- **M5** MTP speculative decoding.
- **M6** `serve`, `chat`, `calibrate`; run the speed gate.

## Risks

- **Per-layer CPU↔GPU sync latency** (40 sync points per token). Mitigation: mapped-memory flags with spin-waits, not
  stream syncs; measure in M3 early.
- **IQ kernel performance in pure Rust/NVRTC** vs ggml's tuned kernels. Mitigation: kernel-level microbenchmarks vs
  llama.cpp in M2; the expert cache reduces how much CPU IQ work is on the critical path.
- **DeltaNet chunked prefill correctness.** Mitigation: parity against the recurrent form, which is simple.
- **llama.cpp may already be fast on this file** (12.5 GB is close to fitting). M0 measures this first; if the gap is
  unreachable, the speed gate is revisited with the user before M2.
