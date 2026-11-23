# TurboQuant work log — RustingStrata

Status as of 2026-11-24. This file records what was tried, what worked, what failed, and what is still blocking a defensible PR.

## 1. Goal and scope

Port TheTom's `llama-cpp-turboquant` KV-cache algorithm into RustingStrata natively (Rust orchestration + runtime NVRTC CUDA kernels, no C++/ggml fork), then extend it to all model formats, not just Inkling 2. Also add asymmetric K/V types (e.g. K=4-bit, V=3-bit) and expose them through the CLI.

The intended final state was:

- correct TurboQuant KV for Inkling 2 and traditional MoE/dense models (Tiel Coder, Holo, GLM, DeepSeek),
- `--kv turbo4`, `turbo4-3`, `turbo4-2`, `turbo3`, `turbo2`, `turbo3-2`,
- `--k` / `--v` overrides for arbitrary combinations,
- measured memory reduction, quality retention, and speed vs f16/q8/q4/q2,
- a pushed PR with detailed benchmarks and `Co-authored-by: Hy4-Flash`.

Only part of this is proven. See section 7.

## 2. Branch and environment state

- Current branch after the requested switch: `main-upstream` at `25c402c`.
- The implementation work was done on `turboquant/cuda`, based on `main-upstream` (ancestry verified).
- Commits on that branch:
  - `9fe46d1` — "CUDA TurboQuant KV for Inkling 2" (the original PR #84 commit; it contained a broken store path).
  - `af24c0c` — "TurboQuant: all KV modes, mixed K/V widths and --k/--v flags", with `Co-authored-by: Hy4-Flash <hy4-flash@local>`.
- Local hardware: RTX 3050 Laptop 4 GB, 14 GB RAM, CUDA 13.5 / NVRTC 13.5.
- Local models used: Inkling-2-Mini UD-Q3_K_XL, Tiel-Coder-35B-A3B-MTP UD-IQ3_XXS, Holo-3.1-4B Q4_K_M.
- `main-upstream` is 12 commits ahead of `myfork/main-upstream`, but identical to `origin/main` at research time.

## 3. Reference algorithm findings (TheTom's fork)

Source: `/home/sayam/Llama/llama-cpp-turboquant`, branch `feature/turboquant-kv-cache`, pulled to `f7c3638e8`.

Verified details:

- Quantization is **unit-L2**, not RMS: `grp_norm = sqrt(sum x_i^2)`, then `x_i /= grp_norm`.
- Stored norm is reconstruction-corrected: `norm = grp_norm / sqrt(sum centroid_idx^2)`, stored as fp16.
- Dequantization is `centroid[idx] * norm`; **no** inverse WHT on K/V at load.
- WHT sign arrays are seed-42 packed bits:
  - `S1 = {0xE46A0359, 0x42F85949, 0xA4C63BBB, 0x49F250A9}`
  - `S2 = {0x16ACEE90, 0x3F628FDC, 0xB7A5357A, 0xBEA56562}`
- Forward WHT: `S1 → butterfly → ×1/sqrt(128) → S2`.
- Inverse WHT: `S2 → butterfly → ×1/sqrt(128) → S1` (swapped sign order).
- Q is forward-rotated in the attention graph; K/V are rotated when written to cache; only the attention output is inverse-rotated, once, before the gate.
- Lloyd-Max centroids for `N(0, 1/128)`:
  - 4-bit max `±0.241529`, midpoints 15 entries including `0.0`;
  - 3-bit `±0.190207`, 7 midpoints;
  - 2-bit `±0.133462`, 3 midpoints.
- TheTom supports separate cache types for K and V (`-ctk`, `-ctv`), including turbo2–turbo6.
- Open PRs checked: #387 (Vulkan turbo3 codebook fix), #420 (multi-GPU per-device InnerQ scale), #421 (turbo5/6 at head_dim 512). None explained our bugs.

A read-only comparison by a researcher agent confirmed our centroids, midpoints, WHT signs, norm correction, and graph placement match the reference.

## 4. Approaches tried

### 4.1 Original buggy `kv-turbo-v2` branch

- Added `--kv turbo4/turbo3/turboquant/rotorquant/...`.
- Implementation used a Givens "rotor" and local inverse-on-load rather than TheTom's Q-rotation/output-inverse design.
- Measured on 4 GB: prefill improved ~6–10 tok/s but decode got worse.
- **Verdict**: abandoned; not faithful to the reference.

### 4.2 Faithful Rust core (`src/kv_turbo.rs`)

- Added seeded WHT-128 s1/s2 diagonals, Lloyd-Max 2/3/4-bit codebooks, QJL projection/reconstruction, and unit tests.
- Standalone CUDA round-trip test (`/tmp/opencode/tq_test.cu`) on random vectors: 4-bit RMSE 0.104, 3-bit RMSE 0.179.
- **Verdict**: core math is correct; failures were elsewhere.

### 4.3 CUDA integration for Inkling 2

Attempts and fixes:

- Added `ink_clip` store path: normalize → WHT → quantize → packed nibble + corrected norm.
- Added Q rotation in `ink_attn` / `ink_attn_prefill_t` and output inverse WHT.
- Fixed misaligned `float4` pointer casts (element-wise loads).
- Fixed prefill Q/K/V element↔sign mapping to contiguous `4*lane + j`; the old strided `lane + 32*j` mapping corrupted every score.
- Fixed inverse sign order (swapped signs).
- Fixed `tq_wht2` sign extraction and avoided an hd=64 barrier deadlock by restricting the early path to head_dim 128.
- **Verdict**: compiled, tests passed, but GPU output was still wrong for reasons later traced to a missing store dispatch and, ultimately, broken GPU inference in this environment.

### 4.4 General-engine expansion (Tiel/Holo/etc.)

- Added `kv_store_turbo` for the generic KV store path.
- Added Q rotation to `attn_decode_t` and `attn_prefill_t`.
- Added output inverse WHT to `attn_merge` and `attn_prefill_t`.
- Fixed `attn_merge` to apply the output gate **after** inverse WHT; prefill already did this.
- Fixed `attn_prefill_t` to use `tro[2][128]` so two query rows are not overwritten before inverse rotation.
- Added modes 4–9 and `TQ_KBITS` / `TQ_VBITS` injection, with 2/3/4-bit templates.
- Added `--k` / `--v` flags mapping turbo pairs to modes 4–9.
- **Verdict**: builds and all library tests pass, but cannot be validated on GPU here (see 6–7).

### 4.5 Debugging experiments that did NOT fix the observed garbage

- RMS vs L2 scale variants; corrected norm on/off.
- Uniform 4-bit quantizer instead of Lloyd-Max: still wrong.
- With rotation disabled (f16 KV in turbo path): output sensible — proved rotation path was at least self-consistent.
- With Q rotation but no output inverse: wrong.
- With inverse-on-load instead of output inverse: wrong.
- Separate centroid arrays for K and V: no effect.
- Extra `__syncthreads()` between K and V stores: no effect.
- Distinct `IS_V` template instantiations for store and load: no effect.
- Replacing template `tq_nearest<4>` with a plain function: no effect.
- Changing all `#if KV_MODE == 4` to `>= 4`: necessary for mixed modes, but did not fix mode 4 alone.
- Temporarily disabling Q rotation/output inverse: still garbage with quantized KV.
- Compute-sanitizer memcheck on Inkling turbo4: **0 memory errors**.

## 5. Root causes found

1. **Store/load domain mismatch (primary PR bug).** The committed PR branch had no real `kv_store_turbo` dispatch. `kv_store()` wrote plain f16 rows while attention read them as 4-bit rotated TurboQuant codes. Fixed by adding the kernel and dispatching to it for `kv_mode >= 4`.

2. **Mixed modes compiled as f16.** All TurboQuant blocks were gated `#if KV_MODE == 4`; modes 5–9 therefore fell through to f16. Fixed by switching to `>= 4`.

3. **Eager kernel loader.** `kernel_names()` loads every name in `KERNELS` from the cubin. Once `kv_store_turbo` was added to the list, non-turbo modes failed with `CUDA_ERROR_NOT_FOUND` because the symbol did not exist. Fixed by defining `kv_store_turbo` unconditionally with a no-op body when `KV_MODE < 4`.

4. **Decode gate ordering.** `attn_merge` applied the gate before inverse WHT, unlike prefill. Fixed.

5. **Prefill query-row overwrite.** The inverse-WHT staging buffer held one row while two query rows were written into it. Fixed with `tro[2][128]`.

6. **False-positive correctness tests.** Several earlier "turbo4 works" results were from runs **without `--gpu`**. `generate`/`bench` default to the CPU reference, which ignores CUDA KV modes. Only `--gpu` exercises this code.

## 6. What works now

- `cargo build --release` succeeds.
- `cargo test --release --lib`: **105 passed, 0 failed** (after updating the expected kernel-name count from 370/324 to 371/325).
- NVRTC compilation tests pass.
- `--kv turbo4|turbo4-3|turbo4-2|turbo3|turbo2|turbo3-2` parse and map to modes 4–9.
- `--k turbo4 --v turbo3` etc. parse and map correctly.
- No CUDA symbol error for f16/q4/q2 after the unconditional kernel definition.
- Memory target: 72 bytes per 128 KV values ≈ **28% of f16** for all turbo modes (2/3-bit modes currently use the same 4-bit container; their benefit is codebook/quality, not extra compression).

## 7. Blocking problem: GPU inference is broken independently of TurboQuant

On the current machine, **upstream baseline `main-upstream` also produces garbage with `--gpu`** for the local models:

- Inkling 2, f16, GPU: `in progress 0 Thu Hương 5 years...`; CPU reference: correct `391`.
- Tiel Coder, f16, GPU: `uffic uffic ... fkk fkk ...`; CPU reference: correct `391`.
- Holo 4B GPU needs a lower `--vram-margin`, but after loading it is not a reliable reference here.
- `--fixed-experts` does not change the garbage output.

Therefore no trustworthy GPU benchmark or quality comparison for TurboQuant can be produced in this environment. This is why PR #84 was **not** pushed/updated with benchmarks.

## 8. Commands and flags

```bash
# TurboQuant modes (CUDA only)
rustingstrata generate --model MODEL --prompt "..." --gpu --kv turbo4
rustingstrata bench --model MODEL --pp 128 --tg 32 --gpu --kv turbo4-3
rustingstrata serve --model MODEL --gpu --kv turbo3

# Explicit K/V widths (turbo pairs)
rustingstrata generate --model MODEL --prompt "..." --gpu --k turbo4 --v turbo2
```

Non-turbo mixed pairs such as `K=f16, V=q4` are not yet implemented; `--k`/`--v` currently support matching legacy types or turbo pairs.

## 9. Recommendations for continuing

1. Diagnose baseline GPU garbage first. Compare a known-good model/quant and try a clean rebuild, different `--ctx`/`--chunk`, `--vram-margin 0`, and a minimal dense model. Without correct f16 GPU output, no TurboQuant result is meaningful.
2. Once GPU f16 is correct, rerun:
   - temp=0 coding/math/prose on Inkling 2 and Tiel,
   - interleaved A/B benchmarks vs f16/q8/q4/q2 with `--gpu`,
   - memory reduction report.
3. Then push `turboquant/cuda` and rewrite PR #84 with real numbers.
4. Implement true arbitrary K/V layout support (`K=f16/q8 + V=q4/turbo`) by splitting `kv_words` into K and V variants and introducing separate `K_MODE`/`V_MODE` store/load macros in `kernels.cu`. This is a larger refactor and should only be done after step 2.
5. Keep the CPU reference harness as a correctness oracle, but remember it does not exercise CUDA KV modes.
