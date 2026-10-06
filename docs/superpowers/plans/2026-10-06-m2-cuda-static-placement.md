# M2: CUDA Forward + Static Expert Placement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the whole `qwen35moe` decode step on the GPU with a fixed set of expert layers resident in VRAM and the
rest computed on the CPU, matching the CPU reference, and record the first speed numbers.

**Architecture:** One NVRTC source (`src/cuda/kernels.cu`) with a per-type 8-value dequant device function shared by
dequant, GEMV and expert-GEMV kernels. `src/cuda/mod.rs` owns the context, PTX disk cache and kernel handles.
`src/gpu_model.rs` uploads non-expert weights, plans expert residency by whole layers in layer order, and runs one
token at a time; CPU-resident layers round-trip the normed activation through the host and reuse the reference
routed-expert code.

**Tech Stack:** Rust 2024, cudarc 0.19 (`driver`, `nvrtc`, `dynamic-loading`, `cuda-13010`), CUDA 13.4, sm_86.

**Spec:** `docs/superpowers/specs/2026-10-06-rusting-strata-v1-design.md`

## Global Constraints

- No size is hardcoded; everything derives from `ModelConfig`.
- Quantized weights stay quantized in VRAM; kernels dequantize on the fly.
- Fixed placement only (adaptation is M3). Fill by layer order (spec "Expert cache adaptation").
- Activations f32; KV cache f16 (q8 option is later).
- Unsupported type or NVRTC failure: error naming the kernel/tensor, with compiler log.
- VRAM safety margin default 512 MB.
- Model-level parity vs the CPU reference: same greedy top-1 on the parity set (≥ 99% vs llama.cpp through `tools/parity.py`).

## Ceilings accepted in M2 (fixed later)

- CPU-resident layers use the scalar reference `Tensor::matvec` (M3: AVX-512 kernels, concurrency with GPU, per-expert cache).
- One host sync per CPU-resident layer.
- No CUDA graphs, no GPU argmax (logits are copied to host, 1 MB/token).
- Prompt is processed token by token (chunked prefill is M4).

---

### Task 1: CUDA context, NVRTC with PTX disk cache, dequant + GEMV kernels

**Files:**
- Modify: `Cargo.toml` (add `cudarc`)
- Create: `src/cuda/mod.rs`, `src/cuda/kernels.cu`
- Modify: `src/lib.rs`
- Test: `tests/cuda.rs`

**Interfaces:**
- Produces: `cuda::Gpu::new() -> anyhow::Result<Gpu>` (context, stream, module);
  `Gpu::upload(&Tensor) -> GpuTensor { ty, rows, cols, buf: CudaSlice<u8> }`;
  `Gpu::dequant(&GpuTensor, rows: usize) -> Vec<f32>` (test helper);
  `Gpu::gemv(&GpuTensor, x: &CudaSlice<f32>, y: &mut CudaSlice<f32>)`;
  `Gpu::moe_gemv(w: &GpuTensor, n_expert, ids: &CudaSlice<i32>, k, x, x_stride, y)`.
- Kernel contract: `deq8<T>(const uint8_t* sb, int sub, float* o)` writes elements `[8*sub, 8*sub+8)` of a 256-element
  super-block for every v1 type. F32/BF16/Q8_0 are viewed as 256-element super-blocks (1024/512/272 bytes).
  GEMV: one warp per row, lane = sub index, warp-shuffle reduce. All row lengths in the target file are multiples of 256;
  `upload` refuses others.

- [ ] **Step 1:** Test: for every quant type present in the model, take the first tensor of that type, dequantize its
  first 4 rows on the GPU and compare with `quant::dequant_row` (max abs diff ≤ 1e-5 × max |value|).
- [ ] **Step 2:** Test: for the same tensors, GPU `gemv` vs `Tensor::matvec` on a fixed pseudo-random `x`
  (relative L2 error ≤ 1e-5), plus `moe_gemv` on `ffn_gate_exps` of layer 0 with ids [3, 200] vs `Tensor::expert().matvec`.
- [ ] **Step 3:** Run `cargo test --release --test cuda`; expect compile failure (no `cuda` module).
- [ ] **Step 4:** Implement `kernels.cu` (port each dequant from `src/quant/mod.rs` to 8-value form) and `mod.rs`
  (PTX cache at `$XDG_CACHE_HOME/rusting_strata/<hash>.ptx`, `arch=compute_86` from device capability).
- [ ] **Step 5:** Tests pass. Commit `feat(cuda): context, PTX cache, dequant and GEMV kernels`.

### Task 2: Layer kernels

**Files:** Modify `src/cuda/kernels.cu`, `src/cuda/mod.rs`.

Kernels (all f32): `rms_norm(x, x_stride, w, out, n, eps)` (grid = rows; same weight per row),
`l2_norm_rows`, `rope_neox(x, row_stride, n_rot, pos, base)`, `add`, `swiglu`, `ssm_gates` (beta = sigmoid,
g = softplus(alpha + dt_bias) · a), `conv_silu` (rolling state update), `deltanet_step` (block per value head,
thread per value row, state in global memory), `norm_gate` (per-head RMSNorm × silu(z)),
`kv_store` (f32 → f16 at `pos`), `attn_decode` (warp per position stride, online softmax, sigmoid output gate),
`router_topk` (one block: softmax, top-k with lower-index ties, renormalize, optional scale),
`moe_combine` (Σ w_k · y_k + sigmoid(s) · shared + cpu part, added into residual).

These are checked at model level in Task 3 (every one is on the path of every token); a mismatch is debugged by
dumping the residual stream per layer from both paths.

- [ ] **Step 1:** Implement kernels and their launch wrappers.
- [ ] **Step 2:** `cargo build --release` clean. Commit with Task 3.

### Task 3: GpuModel with static placement

**Files:** Create `src/gpu_model.rs`; modify `src/model.rs` (expose `routed_experts(il, h)`), `src/lib.rs`.
Test: `tests/gpu_model.rs`.

**Interfaces:**
- `GpuModel::load(&Model, gpu: Gpu, ctx: usize, vram_margin: usize) -> anyhow::Result<GpuModel>`; prints the plan
  (non-expert MB, KV MB, state MB, resident expert layers N/40).
- `GpuModel::step(&mut self, token: u32) -> anyhow::Result<Vec<f32>>` (logits), `reset()`.
- `Model::routed_experts(&self, il, h: &[f32]) -> Vec<f32>` (router + top-k + weighted sum, no shared expert);
  `Model::moe` = routed + shared, behaviour unchanged.

- [ ] **Step 1:** Test: feed the 6 tokens of "The capital of France is" to both `Model::step` and `GpuModel::step`;
  at every position argmax equal and max |Δlogit| ≤ 0.05 × (max logit − min logit); final token " Paris".
  Run once with forced `resident_layers = 0`-equivalent (huge margin) and once with default margin, so both expert paths are covered.
- [ ] **Step 2:** Implement; tests pass.
- [ ] **Step 3:** Commit `feat: GPU forward with static expert placement`.

### Task 4: CLI, parity gate on GPU, first speed numbers

**Files:** Modify `src/main.rs`, `tools/parity.py`. Create `docs/m2-results.md`.

- [ ] **Step 1:** `score` and `generate` take `--gpu` (plus `--ctx`, `--vram-margin`); `generate` prints prompt and
  decode tok/s.
- [ ] **Step 2:** `tools/parity.py` passes extra args after the model path to `strata score` (e.g. `--gpu`); result
  files get a suffix so the CPU run is not overwritten.
- [ ] **Step 3:** Run parity with `--gpu`; must be ≥ 99%.
- [ ] **Step 4:** Measure `generate --gpu -n 128` decode tok/s; record with placement and the per-token split
  (GPU vs CPU-layer time) in `docs/m2-results.md`.
- [ ] **Step 5:** Commit `feat: --gpu for score/generate; M2 parity and speed`.
