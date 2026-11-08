# Roadmap — RustingStrata on the Tesla V100 (sm_70)

Living milestone and task plan for the `TheBrutalX/RustingStrata` fork. It
supersedes the earlier plan and issue #1 (informal roadmap) as the source of
truth for planning; the earlier Italian document was removed and this plan is
English-only. `docs/V100-results.md` remains the append-only record of
measured numbers.

Every task below becomes exactly one GitHub issue (workflow step 1, analyst)
and exactly one pull request (step 2, developer) against `v100`.

---

## 1. Purpose and update protocol

**Purpose.** Deliver and maintain a fork of RustingStrata that:

1. runs correctly on a Tesla V100-SXM2-32GB (Volta, sm_70) on both backends;
2. matches upstream numerics within the project parity bar;
3. is tuned for this card (32 GB HBM2, ~900 GB/s, 80 SMs, 6 MB L2, 96 KB
   shared memory per SM);
4. stays rebasable onto `upstream/main`.

**Who updates this document.** The **coordinator** only. The planner writes
the initial decomposition and revises it when the coordinator asks for a new
request to be broken down; it never edits statuses, issue numbers or PR
numbers.

**When it is updated.**

- after every **verified merge** (reviewer APPROVE → PR squash-merged → issue
  closed): the task status becomes `done`, and the issue/PR columns are
  filled in;
- when a task enters development or review (status `in progress` /
  `in review`), when a task is blocked, and when a task is re-scoped after a
  failed review round;
- when a new measurement is appended to `docs/V100-results.md` and an
  acceptance threshold in this document has to be restated;
- when a milestone exits (all tasks `done` and exit criteria met).

**How status changes.** Status mirrors, but is not driven by, the `step:*`
labels of the workflow:

| Roadmap status | Meaning | Corresponding issue state |
| --- | --- | --- |
| `todo` | not started, no issue yet | no `step:*` label |
| `in progress` | issue open with `step:dev`, developer working | `step:dev` |
| `in review` | PR open, reviewer verifying | `step:review` |
| `blocked` | needs a human decision | `step:blocked` |
| `done` | reviewer approved, PR merged, issue closed | closed |

The coordinator updates the task row (status, issue, PR) after the reviewer's
final comment. A task whose PR was merged without an issue (pre-workflow work)
is marked `done` with the merge commit in the status cell.

**Measurement policy.** Only numbers present in `docs/V100-results.md` (or in
the documents it references) are quoted here. Targets are explicitly labelled
as targets; anything not yet measured is marked **to be measured**. Nobody
edits old measurement entries: new runs append.

---

## 2. Goals and non-goals

### Goals

- **Correct on sm_70.** Both backends must run the reference models without
  `Xid` errors or lost devices, from short prompts to the full 32K context.
- **Numerical parity.** CUDA sm_70 and Vulkan sm_70 must meet the project
  parity bar against llama.cpp (see §3.4) and must not silently diverge from
  the CPU reference.
- **Performance for this card.** On
  `Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf` (MoE, 12.3 GB, all experts resident):
  prefill 4096 ≥ 2000 tok/s and decode ≥ 130 tok/s on CUDA (targets from the
  earlier planning work), with no Vulkan regression below the measured
  baseline (`docs/V100-results.md`: 1423.9 tok/s prefill @4K, 109.8 tok/s
  decode).
- **Upstreamable.** All Volta-specific code stays behind `#if __CUDA_ARCH__`
  or equivalent helpers; the sm_80+ path and its numerics stay unchanged.
  Offering fixes upstream is explicitly out of scope for now (see non-goals).
- **Repeatable.** CI (non-GPU) and a scheduled GPU suite on the V100 keep the
  state from rotting; releases carry both backends with sm_70 notes.

### Non-goals

- Porting to other Volta SKUs (V100-PCIe, Titan V) other than incidentally.
- Supporting sm_70 with CUDA 13 or any driver newer than the 580 branch: none
  exists.
- Beating llama.cpp on every model: the dense 27B gap is a known consequence
  of Vulkan lacking cooperative matrices; CUDA is the intended fix.
- Offering fixes upstream to `RustingStudio/RustingStrata`: deferred, not
  planned in this roadmap (the fork stays rebasable, nothing more).
- Changing upstream numerics or refactoring the sm_80+ kernels for style.
- Multimodal/other architectures beyond what upstream supports.
- Selling the engine or re-licensing the fork (see §3.3).

---

## 3. Hard constraints

### 3.1 Hardware and toolchain

| Constraint | Value | Consequence |
| --- | --- | --- |
| GPU | Tesla V100-SXM2-32GB, sm_70, 80 SMs, 6 MB L2, ~900 GB/s | 5120 FP32 cores; FP16 tensor cores only at `mma.m8n8k4` |
| Driver | 580.178.04 (last branch for Volta) | No fallback if it breaks; pin the version in docs |
| CUDA toolkit / NVRTC | 12.9 (`/usr/local/cuda`); CUDA 13 dropped sm_70 | `src/cuda/mod.rs:118` must not just say "CUDA 13"; runtime compile must target `sm_70` |
| Vulkan | 1.4.312, **no** `VK_KHR_cooperative_matrix` on sm_70 | Only the `COOP = 0` (`gemmi_dp`, `gemm_f_dp`) path exists |
| Vulkan shared memory | 48 KB per workgroup (49152 B) | `gemmi` KS = 16 pipelines need 49168 B → fixed by reading IQ tables from the buffer (`src/vulkan/shaders/gemmi.glsl:17-25`) |
| CUDA shared memory | 96 KB per SM; max dynamic per block 96 KB | `attn_prefill*`, `attn_decode_gqa*`, `gemvn*` opt in to 96 KB (`src/cuda/mod.rs:196-198`) |
| CUDA registers | 64K per SM | Occupancy vs tile size trade-off; measure, do not guess |
| CUDA int8 | `dp4a` only, **no** int8 tensor cores | `mma.m16n8k16.s8` / `m16n8k32.s8` cannot exist on sm_70 |
| Rust | stable (rustup), edition 2024 | — |

### 3.2 Production GPU

The V100 also hosts `llama.router.service` (verified active). GPU tests run
only in a window the coordinator schedules; the coordinator stops the service
for the test and restarts it immediately after. No test may break production.
Any `Xid`/device-lost output means stop and record the command.

### 3.3 License

RustingStrata License 1.0 (`LICENSE.md`): forks and modifications allowed,
including commercial services, provided the license and credit to the original
author are kept; the engine itself may not be sold. Every derived file keeps
the license and attribution.

### 3.4 Numerical parity bar

The project bar is **≥ 99% top-1 agreement** with llama.cpp using the v1 gate
in `tools/parity.py`: a token counts when it matches llama.cpp in either of its
two modes, because llama.cpp's token-by-token and single-batch modes agree with
each other on only 98.28% of positions (`docs/m6-results.md`). Reference
measured values: 99.32% in parity mode, 98.85% on the default path
(`docs/m6-results.md`). For sm_70 a bit-exact match with the sm_80 CUDA path is
**not** required; parity is measured per backend and any deviation is recorded
in `docs/V100-results.md` and signed off by the coordinator.

### 3.5 Upstream compatibility and repository rules

- `main` mirrors `upstream/main`; all work lands on `v100` or `task/*`.
- Volta code behind `#if __CUDA_ARCH__` (or a helper with the same effect).
  No behaviour change for sm_80+.
- Vulkan shader changes regenerate SPIR-V with `tools/gen_spirv.sh`;
  `spirv_is_fresh` and `glsl_tables_are_fresh` in `tests/backend.rs` enforce
  it.
- No models, binaries or secrets in the repository; models live in
  `/mnt/models`, builds in `target/`.
- Every issue, comment, document, commit and PR is in English.

---

## 4. Status legend

| Status | Meaning |
| --- | --- |
| `todo` | planned, no issue opened yet |
| `in progress` | issue open (`step:dev`), implementation underway |
| `in review` | PR open against `v100` (`step:review`) |
| `blocked` | needs a human decision (`step:blocked`) |
| `done` | merged and closed; `docs/V100-results.md` updated if it changed numbers |

Task sizes: **S** ≤ 1 day, **M** 1–3 days. Anything larger is split before an
issue is opened.

---

## 5. Current state (verified on `v100`)

- Fork exists with `origin` (fork) and `upstream` (RustingStudio); `v100` is
  the fork's default branch.
- Vulkan runs on the V100 after `53f823e` (`gemmi_dp` shared memory 49168 B >
  49152 B; IQ tables read from the buffer when `COOP = 0`).
- CUDA does **not** run on sm_70. `src/cuda/kernels.cu` contains sm_80-only
  constructs with no arch guard:
  - `ldmatrix` in `ldsm4` / `ldsm4t` (`:821-829`), used only by
    `attn_prefill_t` (`:2135`, `:2138`, `:2183`);
  - `mma.sync.aligned.m16n8k16...f16.f16.f32` in `mma16816` (`:831-834`),
    used by `gemm_q` (`:915`) and `attn_prefill_t` (`:2139-2140`,
    `:2184-2185`);
  - int8 `mma` (`:993-1009`) and the whole `gemmi_q` path (`:1027-1170`);
  - `cp.async` helpers (`:1014-1023`), used by `gemmi_q` (`:1044-1081`),
    `attn_decode_t` (`:1733-1798`) and `attn_prefill_t` (`:2094-2191`);
  - `__reduce_max_sync` / `__reduce_min_sync` in `topk_warp` (`:1958-1959`).
  `gemm_q` already loads its fragments with plain `ld_s32` (`:912-916`), not
  `ldmatrix`.
- `src/cuda/mod.rs:183-185` already derives `sm_70` from the compute
  capability, but `:118` reports a missing NVRTC as "CUDA 13", the function
  list at `:188-192` loads `gemmi*`/`gemmis*` unconditionally, and
  `Gpu::mmq` (`:255-258`) still selects the int8 path.
- Measured Vulkan baseline (`docs/V100-results.md`): MoE Q2 1423.9 tok/s
  prefill @4K, 109.8 tok/s decode, 100% expert hit rate; dense 27B Q4 291.1
  tok/s prefill @4K, 27.6 tok/s decode.
- `tests/common/mod.rs` defaults `STRATA_MODEL` to a `/mnt/ollama/...` path
  that does not exist on this machine; the reference model is
  `/mnt/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf`.
  Tests must be run with `STRATA_MODEL` set (fix tracked by M4-T1).

---

## 6. Milestones

### M0 — Baseline, process and Vulkan stability

**Objective.** Freeze what works, make the 48 KB limit impossible to trip
silently, prove type and flag coverage on this card, and record a Vulkan
parity baseline.

**Exit criteria.**

- A preflight path fails *before* loading model weights with a message naming
  the kernel, specialization and shared bytes when a pipeline exceeds
  `maxComputeSharedMemorySize`.
- Every supported `GgmlType` is exercised on the V100 on Vulkan
  (dequant/GEMV/GEMM/attention) with a pass/fail matrix in
  `docs/V100-results.md`; no `Xid`.
- `--chunk`, `--kv q8`, `--parallel`, `--mtp`, `STRATA_VK_NO_COOP`,
  `STRATA_NO_GRAPH` are each run once and their supported/unsupported status
  is documented.
- Vulkan parity number recorded with `tools/parity.py`.
- `tools/bench_v100.sh` can emit a machine-readable summary.

**Dependencies.** None. M0 can run in parallel with M1.

**Risks.** Preflight needs a pipeline-creation path that does not allocate
model memory; creating *all* pipelines eagerly could add minutes of startup
time. Mitigation: preflight only the kernels and specializations the loaded
model's tensor types need, exposed as an explicit check command too.

| ID | Title | Scope | Files / areas | Size | Deps | Acceptance | Status | Issue | PR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0-T1 | Workflow and agent scaffolding | AGENTS.md, the three-step workflow, repo contract, planner role, default coordinator agent | `AGENTS.md`, `docs/agents/WORKFLOW.md`, `.opencode/`, `.github/` | S | — | Documents exist and are internally consistent | done (`4315ebc`, `c0bb3bc`, `f8cb5b3`) | — | — |
| M0-T2 | Vulkan 48 KB shared-memory fix | `gemmi_dp` reads the IQ tables from the buffer when `COOP = 0`; shared memory drops from 49168 B to ≤ 49152 B | `src/vulkan/shaders/gemmi.glsl`, `deq.glsl`, `src/vulkan/spv/*` | S | — | Chunked prefill on V100 no longer produces `Xid 13 SKEDCHECK18_L1_CONFIG_TOO_SMALL` | done (`53f823e`) | — | — |
| M0-T3 | Benchmark script and baseline numbers | `tools/bench_v100.sh` appends repeatable runs to `docs/V100-results.md`; MoE and dense baselines recorded | `tools/bench_v100.sh`, `docs/V100-results.md` | S | M0-T2 | Script runs the documented command and appends a timestamped block; numbers present | done (`9992a09`) | — | — |
| M0-T4 | Vulkan shared-memory preflight | Query `maxComputeSharedMemorySize`, expose pipeline creation/shared-byte accounting, and check the specializations a model needs before uploading weights; clear error on overflow | `src/vulkan/mod.rs` (`Vk::new`, `pipeline`), `src/main.rs` (check command), `tests/backend.rs` | M | M0-T2 | On V100, an intentionally oversized specialization is refused with kernel/spec/bytes; a normal load emits no error and no `Xid` | done (`cf30f29`) | #2 | #3 |
| M0-T5 | Vulkan GGUF type matrix on sm_70 | Exercise every supported type, especially the KS = 16 / `PW = 36` set (Q2_K, Q3_K, Q6_K, IQ1_M, IQ2_XS, IQ2_S, IQ4_NL) and the table-backed set (IQ3_S, IQ4_XS), plus Q4_K, Q5_K, Q8_0, BF16; record pass/fail | `tests/kernels.rs`, `tests/backend.rs`, `docs/V100-results.md`, `src/vulkan/shaders/` | M | M0-T4 | Matrix table appended to `docs/V100-results.md`; every type passes or is documented as unsupported; no `Xid` | done (`673f215`) | #4 | #5 |
| M0-T6 | Vulkan flag/knob matrix | Run `--chunk` (512/4096/32768), `--kv q8`, `--parallel 2/4`, `--mtp`, `STRATA_VK_NO_COOP`, `STRATA_NO_GRAPH`; note interaction `--kv q8` × `--llama-numerics` (rejected by design) | `src/main.rs`, `src/vulkan/mod.rs`, `docs/V100-results.md` | M | M0-T4 | Table of combination → result/numbers in the docs; crashes or wrong output become follow-up tasks | todo | | |
| M0-T7 | Vulkan parity baseline on V100 | Run `tools/parity.py` against llama-server with `--tag vk-sm70`; compare against the 99% either-mode bar and record the JSON/result | `tools/parity.py`, `docs/V100-results.md`, `docs/baseline.md` | M | M0-T4 | Parity ≥ 99% either-mode, or a documented, coordinator-approved deviation; result recorded. Measured 629/640 either-mode (98.28%): deviation approved, diagnosis in M3-T5 | done (`7627641`) | #6 | #7 |
| M0-T8 | Machine-readable benchmark capture | Add JSON output (or a JSONL sidecar) to `strata bench` and an aggregation step in `tools/bench_v100.sh`; keep the human-readable block | `src/main.rs` (`Cmd::Bench`), `tools/bench_v100.sh` | S | M0-T3 | One run produces both the existing text block and a parseable JSON record with backend, model, pp, tg, reps and per-run values | todo | | |

---

### M1 — CUDA sm_70 bring-up (correctness first)

**Objective.** Make `--gpu --backend cuda` run on sm_70 with no sm_80-only
instruction, no int8 tensor cores, and correct numerics, before any tuning.
**Status: complete (2026-11-07).**

**Exit criteria.**

- NVRTC 12.9 compiles `kernels.cu` for `sm_70` and the cubin loads.
- `cargo test --release` with `STRATA_MODEL` set runs the kernel/backend tests
  on CUDA with no `SKIP: cuda` and no panic.
- `strata generate` and `strata bench` complete on the reference model
  without `Xid`; first CUDA numbers appended to `docs/V100-results.md`.
- Parity measured and recorded; ≥ 99% either-mode or an explicit
  coordinator-approved deviation.
- `grep` shows every removed construct still present for sm_80+ behind an
  arch guard (no behaviour change there).

**Dependencies.** None on M0; recommended after M0-T4 because the preflight
tooling makes CUDA failures easier to read. M1-T7 depends on M1-T1..T6;
M1-T8 depends on M1-T7.

**Risks.** The `mma.m16n8k16` → FMA fallback is fragment-layout-sensitive and
can produce plausible-but-wrong numbers; the `cp.async` replacement changes
synchronization, not just instruction selection. Both are covered by the
per-op tests in `tests/kernels.rs` and cross-checked against the Vulkan and
CPU reference paths.

| ID | Title | Scope | Files / areas | Size | Deps | Acceptance | Status | Issue | PR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1-T1 | Arch and NVRTC bring-up | Keep `sm_70` selection from the compute capability, detect the NVRTC major version, replace the "CUDA 13" message with an accurate one, log `sm_70 (Volta fallback)`, include the toolkit version in the cubin cache key | `src/cuda/mod.rs:104-144`, `:176-185` | S | — | On V100 the log shows `sm_70`; with a hypothetical CUDA 13 NVRTC the error names the required 12.x; cached cubins are not reused across toolkit versions | done (`435a584`) | #8 | #9 |
| M1-T2 | Warp reduce without `__reduce_*_sync` | Replace `__reduce_max_sync`/`__reduce_min_sync` in `topk_warp` with `__shfl_xor_sync` reductions under `#if __CUDA_ARCH__ < 800`; keep the sm_80 path | `src/cuda/kernels.cu:1935-1969` | S | M1-T1 | `router_topk` matches the CPU reference bit for bit on sm_70 and still matches the old sm_80 result | done (`fe3eb1a`) | #10 | #11 |
| M1-T3 | `ldmatrix` fallback for attention prefill | Load the four 8×8 fragments with per-lane `ld.shared.b32` (same addresses the layout comment documents), both normal and `.trans`, behind `#if __CUDA_ARCH__ < 750` | `src/cuda/kernels.cu:821-829`, uses at `:2135`, `:2138`, `:2183` | S | M1-T1 | `attn_prefill` on sm_70 matches the CUDA sm_80/CPU reference within the documented tolerance; no layout drift in the softmax input | done (`dc26339`) | #12 | #13 |
| M1-T4 | `mma.m16n8k16` FMA fallback | Implement the FP16 fragment layout of `mma.sync.m16n8k16` with half multiplies and f32 accumulation on CUDA cores for `#if __CUDA_ARCH__ < 800`; keep `mma16816` for sm_80+ | `src/cuda/kernels.cu:831-834`, uses at `:915`, `:2139-2140`, `:2184-2185` | M | M1-T1 | Per-op GEMM and attention-prefill tests on sm_70 match the CPU reference; the sm_80 code path is unchanged (diff shows only guarded blocks) | done (`f4b4d80`) | #18 | #19 |
| M1-T5 | Quarantine the int8 tensor-core path | Guard `mma_s8_*` and `gemmi_q`/`gemmis_*` behind `#if __CUDA_ARCH__ >= 800`; make `Gpu::mmq` false on sm_70 and stop loading `gemmi*` function names there, so NVRTC and `load_function` never see them | `src/cuda/kernels.cu:993-1009`, `:1027-1170`; `src/cuda/mod.rs:186-200`, `:255-258`, `:260-298` | M | M1-T1 | NVRTC sm_70 compile succeeds without the int8 kernels; prefill uses `gemm_*`/`gemms_*` and matches the reference; sm_80 still uses int8 | done (`ea39c4a`) | #14 | #15 |
| M1-T6 | Replace `cp.async` with synchronous double buffering | Provide `cp_async16/8/commit/wait` fallbacks using `ld.global` → registers → `st.shared` with zero-fill on the out-of-range flag; audit every wait site for the barrier it needs and add `__syncthreads` where the async version relied on the wait | `src/cuda/kernels.cu:1014-1023`, uses at `:1044-1081`, `:1733-1798`, `:2094-2191` | M | M1-T1 | `gemmi_q` (later unused), `attn_decode_t` and `attn_prefill_t` outputs match the reference on sm_70; no missing-barrier race under repeated runs; `STRATA_NO_GRAPH` runs also pass | done (`bd5a930`) | #16 | #17 |
| M1-T7 | First green sm_70 build and kernel-load gate | Drive the full NVRTC compile/load and `cargo test --release` on CUDA; fix remaining compile fallout (function list, 96 KB attribute, launch bounds) and record which kernels execute | `src/cuda/mod.rs`, `src/cuda/kernels.cu`, `tests/kernels.rs`, `tests/backend.rs` | M | M1-T1..T6 | `cargo test --release` with `STRATA_MODEL` shows no `SKIP: cuda` and no failure; `strata generate` completes on the reference model without `Xid` | done (`a945aee`) | #20 | #21 |
| M1-T8 | CUDA correctness baseline and parity | Run `tools/parity.py --tag cuda-sm70`, compare CUDA/Vulkan/CPU outputs per op, append first CUDA numbers and divergence notes | `tools/parity.py`, `docs/V100-results.md` | M | M1-T7 | Parity ≥ 99% either-mode or a coordinator-approved documented deviation; first prefill/decode numbers recorded with the same harness as M0-T3. Measured 636/640 either-mode (99.38%): bar passed | done (`68f4f50`) | #22 | #23 |

---

### M2 — CUDA performance on the V100

**Objective.** Use Volta's FP16 tensor cores and tune tiling, occupancy and
MoE residency for 80 SMs / 32 GB, to meet the MoE targets.

**Exit criteria.**

- `Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf` on CUDA: prefill 4096 ≥ 2000 tok/s and
  decode ≥ 130 tok/s (targets; measured with `tools/bench_v100.sh`,
  `--reps 3`), no `Xid`, parity not below M1-T8.
- `sm_80+` numerics and performance unaffected: the arch-guarded diff is
  reviewed, and (if a non-Volta GPU is available) the existing
  `tests/llama_numerics.rs` CUDA/Vulkan comparison still passes. If no such
  GPU is available this is stated explicitly as a review-only check.
- `git rebase upstream/main` applies without semantic conflicts in the
  touched kernels.

**Dependencies.** M1-T8. M2-T1/M2-T2 also require M1-T3 and M1-T4 (the
fallback must be correct before it is replaced by tensor cores).

**Risks.** Volta tensor cores change fragment layouts and the code that
consumes the accumulators (softmax in `attn_prefill_t`, stores in `gemm_q`);
a wrong mapping is silent. Mitigation: keep the FMA fallback selectable at
compile time, add a per-op numerical test for the tensor-core variant, and
land `gemm_q` before attention.

| ID | Title | Scope | Files / areas | Size | Deps | Acceptance | Status | Issue | PR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2-T1 | Volta tensor cores in the prefill GEMM | Map `m16n8k16` onto `mma.sync.m8n8k4` (2 row tiles × 4 K steps) in `gemm_q`, keeping the fragment loads from shared memory and the store mapping; guard by arch and keep the FMA fallback for non-tensor builds | `src/cuda/kernels.cu:844-951` | M | M1-T8 | GEMM test on sm_70 matches the reference; prefill 4K on the reference model improves over M1-T8; sm_80 unchanged. Measured +123.6% @512 / +90.7% @4096 | done (`3d1aa82`) | #24 | #25 |
| M2-T2 | Volta tensor cores in prefill attention | Same mapping for the QKᵀ and PV products in `attn_prefill_t`, including the transposed V fragment and the softmax index mapping; keep the FMA fallback | `src/cuda/kernels.cu:2069-2210` | M | M2-T1 | Attention-prefill test matches the reference on sm_70; end-to-end prefill improves; no NaNs; sm_80 unchanged | todo | | |
| M2-T3 | Shared-memory carveout and tile sizing | Set the preferred shared-memory carveout, validate the 96 KB dynamic limit on sm_70, and re-check the `GEMM_BM`/`GEMM_BK` and attention tile choices against 64K registers/SM | `src/cuda/mod.rs:193-200`, `src/cuda/kernels.cu` (`GEMM_BM`, `FA_BQ`, `FA_BK`) | S | M2-T1 | Occupancy/register report captured; no regression; chosen constants documented with the measured reason | todo | | |
| M2-T4 | MoE expert prefetch and residency | All 10240 experts fit in 32 GB (measured 100% hit rate on Vulkan); make the CUDA path prefetch the next layer's experts and keep the CPU worker cost at zero; review the L2 (6 MB) reuse policy for hot weights | `src/gpu_model.rs`, `src/expert_cache.rs`, `src/cuda/` | M | M1-T8 | Cache swaps and worker stalls observed as 0 in a 3-rep 4K/32K run; decode not worse than M1-T8; numbers recorded | todo | | |
| M2-T5 | Decode GEMV bandwidth | Measure the achieved decode bandwidth against the ~900 GB/s roofline; tune rows per block, vector loads and launch shape until ≥ 85% of the roofline or a documented ceiling is reached | `src/cuda/kernels.cu:675-812` (`gemv_row_qn`, `DEF_*GEMV*`), `src/cuda/mod.rs` | M | M1-T8 | Achieved GB/s and % of roofline appended to `docs/V100-results.md` (to be measured); decode on the reference model improves or the ceiling is explained | todo | | |
| M2-T6 | CUDA flags: KV q8, parallel, MTP | Make `--kv q8`, `--serve --parallel N` and `--mtp` work on sm_70 CUDA; fix combos that fail; measure each on the reference model | `src/cuda/`, `src/main.rs`, `src/gpu_model.rs`, `docs/V100-results.md` | M | M1-T8 | Each flag runs without error; unsupported combinations are rejected with a clear message; numbers recorded | todo | | |
| M2-T7 | Cross-backend numerical consistency report | Compare CPU reference, CUDA sm_70 and Vulkan sm_70 per op and end-to-end (parity + KLD or top-1 where available); document where the architectures legitimately differ | `tools/parity.py`, `tests/llama_numerics.rs`, `docs/V100-results.md` | M | M2-T2 | Report appended: per-op max/relative deviation and end-to-end agreement; no unexplained divergence | todo | | |
| M2-T8 | CUDA performance gate | Run the full benchmark protocol on the reference model (`pp 512,4096,32768`, `tg 128`, `reps 3`), append results, and compare against targets and against the Vulkan baseline | `tools/bench_v100.sh`, `docs/V100-results.md` | S | M2-T1..T7 | prefill @4K ≥ 2000 tok/s and decode ≥ 130 tok/s, or a coordinator-approved revised target with evidence; no `Xid`; no Vulkan regression | todo | | |

---

### M3 — Vulkan fallback hardening and tuning

**Objective.** Keep Vulkan as a trustworthy fallback: profile it, tune within
the 48 KB ceiling, guard the measured baseline, and diagnose the parity gap
recorded by M0-T7 (629/640 either-mode vs the 99% bar).

**Exit criteria.**

- A profile identifies the dominant kernels at 4K prefill and decode.
- At least one shader/tile change is measured; the result is either a gain or
  a documented "no headroom" conclusion, with the SPIR-V regenerated.
- Final Vulkan numbers are ≥ the baseline in `docs/V100-results.md`
  (1423.9 tok/s prefill @4K, 109.8 tok/s decode) and parity is unchanged.

**Dependencies.** M0-T4 and M0-T7 (preflight and parity baseline). Can run in
parallel with M1/M2.

**Risks.** 48 KB per workgroup leaves little room; any tile increase can
re-trigger the `Xid` class of failure. Mitigation: every shared-memory change
goes through the preflight from M0-T4 and is measured.

| ID | Title | Scope | Files / areas | Size | Deps | Acceptance | Status | Issue | PR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M3-T1 | Vulkan profile of prefill and decode | Use `STRATA_VK_STATS`/`STRATA_VK_PROF` (and pipeline statistics) to attribute time to `gemmi_dp`, `gemm_f_dp`, `attn_prefill`, `attn_decode`; record the breakdown | `src/vulkan/mod.rs`, `docs/V100-results.md` | S | M0-T7 | Breakdown appended; the next tuning task has a named target | todo | | |
| M3-T2 | Tune the non-cooperative GEMM shaders | Try local size, subgroup count, `BM`/`BK` and staged variants of `gemmi_dp`/`gemm_f_dp` within 48 KB; keep the bit-exact int32 path | `src/vulkan/shaders/gemmi.glsl`, `gemm_f.glsl`, `src/vulkan/spv/*` | M | M3-T1 | Measured change recorded; SPIR-V regenerated; preflight passes; no parity change | todo | | |
| M3-T3 | Tune prefill attention on Volta | Examine `attn_prefill` shared usage and workgroup shape; test a variant only if M3-T1 shows it matters | `src/vulkan/shaders/attn_prefill.glsl`, `src/vulkan/spv/*` | M | M3-T1 | Gain measured or the conclusion documented; SPIR-V fresh; no regression | todo | | |
| M3-T4 | Vulkan regression gate | Re-run the standard benchmark and parity after M1/M2/M3 changes; fail the milestone if below baseline | `tools/bench_v100.sh`, `tools/parity.py`, `docs/V100-results.md` | S | M3-T2, M3-T3 | Numbers ≥ baseline and parity unchanged; otherwise a follow-up task is opened before the milestone exits | todo | | |
| M3-T5 | Diagnose the Vulkan parity gap | Explain the 629/640 either-mode baseline from M0-T7 (concentration on prompts 10/16/17/19, real top-2 gap on #10) and either close it or state the accepted ceiling with evidence | `src/vulkan/`, `src/gpu_model.rs`, `tools/parity.py`, `docs/parity/`, `docs/V100-results.md` | M | M0-T7 | Root cause identified; fix merged with a new parity run, or a documented, coordinator-approved ceiling with the evidence | todo | | |

---

### M4 — CI, parity automation and release

**Objective.** Make the V100 state reproducible (non-GPU CI + scheduled GPU
suite), automate the parity gate, and ship a release with both backends.
Upstreaming is deferred and not part of this milestone (see non-goals).

**Exit criteria.**

- A pull-request workflow builds and runs the CPU test suite plus the SPIR-V
  and table freshness checks, with no GPU.
- A scheduled workflow on a self-hosted V100 runner acquires an exclusive
  lock, stops `llama.router.service`, runs bench + parity, restarts the
  service, and archives the artifacts; a failure is visible without touching
  production outside the window.
- Parity for both backends is automated with the 99% threshold and a
  regression comparison against the previous run.
- A release tag carries Linux/Windows binaries with sm_70 notes and updated
  English documentation.

**Dependencies.** M0-T7/M0-T8 (parity and JSON capture) for the automation;
M2-T8 and M3-T4 for the release gate. M4-T1 can start immediately.

**Risks.** A self-hosted runner on the production host can compete with
`llama.router.service`; the lock and the service stop/restart must be part of
the workflow, not a human habit.

| ID | Title | Scope | Files / areas | Size | Deps | Acceptance | Status | Issue | PR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M4-T1 | Non-GPU CI workflow | PR job: `cargo build --release`, `cargo test --release` (GPU tests skip), `spirv_is_fresh` with `glslc`, table freshness; fix the `STRATA_MODEL` default so a missing model skips cleanly and document it | `.github/workflows/`, `tests/common/mod.rs`, `docs/` | M | — | Workflow green on a PR without a GPU; a stale SPIR-V or table fails with an actionable message | todo | | |
| M4-T2 | Self-hosted V100 GPU suite | Scheduled workflow on a V100 runner: exclusive lock, stop/restart `llama.router.service`, run bench + parity for CUDA and Vulkan, upload artifacts, never leave the service stopped | `.github/workflows/`, `tools/`, `docs/agents/WORKFLOW.md` | M | M4-T1 | One scheduled run completes and restarts production; a deliberately failed step still restarts it (tested with a dry-run job) | todo | | |
| M4-T3 | Automated parity gate for both backends | Wrap `tools/parity.py` for CUDA and Vulkan, apply the 99% either-mode threshold, compare with the previous stored result, publish a summary | `tools/parity.py`, `tools/`, `.github/workflows/` | M | M0-T7, M0-T8 | Gate fails on a parity drop; results are archived per run and tagged by backend | todo | | |
| M4-T4 | Documentation consolidation | English V100 documentation covering verified state, flags, constraints and benchmarks; update `README`/`AGENTS.md` links and the production-GPU procedure | `docs/`, `README.md`, `AGENTS.md` | S | M2-T8, M3-T4 | All planning docs and commands are English and work verbatim; no stale links | todo | | |
| M4-T5 | Release with both backends | Tag a fork release, verify binaries build and load both backends on the V100, publish notes covering sm_70 limits, driver/toolkit pins and license/credit | `.github/workflows/release.yml`, `docs/` | S | M4-T4, M4-T3 | Release artifacts exist; notes verified against this document; license and attribution present | todo | | |

---

## 7. Critical path and recommended order

**Critical path** (each arrow is a hard dependency):

```
M1-T1 ──▶ M1-T2 ─┐
        ──▶ M1-T3 ─┤
        ──▶ M1-T4 ─┼──▶ M1-T7 ──▶ M1-T8 ──▶ M2-T1 ──▶ M2-T2 ──▶ M2-T7 ──▶ M2-T8
        ──▶ M1-T5 ─┤                              └──▶ M2-T3/M2-T4/M2-T5/M2-T6
        ──▶ M1-T6 ─┘
M0-T4 ──▶ M0-T5/M0-T6/M0-T7 ──▶ M3-T1 ──▶ M3-T2/T3 ──▶ M3-T4
M0-T7/M0-T8 ──▶ M4-T3;  M4-T1 ──▶ M4-T2
M2-T8 + M3-T4 ──▶ M4-T4 ──▶ M4-T5
```

The longest chain is **M1-T1 → M1-T4 → M1-T7 → M1-T8 → M2-T1 → M2-T2 →
M2-T8**. M2-T2 (attention tensor cores) is the riskiest item on it.

**Recommended execution order.**

1. **M0-T4, M0-T5, M0-T7** — cheap, removes the class of bug that already cost
   a device reset, and produces the parity baseline every later task is
   compared against. Run in parallel with the first CUDA tasks.
2. **M1-T1 → M1-T2 → M1-T3 → M1-T5 → M1-T6 → M1-T4 → M1-T7.** Landing the
   inexpensive guards first means the first sm_70 build reaches the `mma`
   fallback instead of failing on six things at once. M1-T5 before M1-T6
   because removing the int8 kernels shrinks the compile surface.
3. **M1-T8**, then **M2-T1 → M2-T2** (tensor cores), then the tuning tasks
   M2-T3..M2-T6 and the M2-T8 gate.
4. **M0-T6/T8 and M3** in parallel with M2, since they use the same GPU
   windows and do not block the CUDA chain.
5. **M4-T1** early (no GPU), **M4-T2/M4-T3** after M2-T8, **M4-T4/M4-T5** at the
   end.

**GPU-window batching.** Tasks that need the V100 should be scheduled in the
same window: M0-T5, M0-T6, M0-T7, M1-T7, M1-T8, M2-T8, M3-T4. CPU-only tasks
(S: M1-T1, M1-T2, M4-T1, M4-T4) can proceed without a window.

---

## 8. Top risks and mitigations

| # | Risk | Impact | Mitigation |
| --- | --- | --- | --- |
| 1 | Volta tensor-core fragment mapping in `attn_prefill_t` is wrong but plausible | High (silent numerical corruption; weeks) | Keep the FMA fallback compiled and selectable; land `gemm_q` (M2-T1) first; per-op tests against CPU reference; M2-T7 cross-backend report; coordinator sign-off before tuning |
| 2 | `cp.async` removal changes synchronization, not just instructions | High (races that appear only at long context) | Audit every wait site in M1-T6 (`:1044-1081`, `:1733-1798`, `:2094-2191`), add explicit `__syncthreads`, test with `STRATA_NO_GRAPH`, repeated runs at 32K, and compare prefill logits with Vulkan |
| 3 | Upstream drift makes `v100` expensive to rebase | Medium (schedule) | All Volta code behind `__CUDA_ARCH__`; one logical commit per task; no reformatting of sm_80 code; a manual rebase check before large changes |
| 4 | 48 KB Vulkan limit re-triggered by a new type or tile | Medium (device reset on production GPU) | M0-T4 preflight plus M0-T5 type matrix; every shader change measured and regenerated |
| 5 | GPU contention or turbo/noise makes numbers non-comparable | Medium (wrong conclusions) | Same harness (`tools/bench_v100.sh`, `reps 3`), same window policy, record min/mean/max, never compare across different `--ctx`/`--chunk` |
| 6 | NVRTC/driver regression on the last Volta branch | Low but fatal | Pin driver 580.178.04 and toolkit 12.9 in docs; `sm_70` compile is a CI smoke test on the runner (M4-T2) |
| 7 | Self-hosted runner leaves `llama.router.service` stopped | Medium (production outage) | Workflow-level trap/`if: always()` restart; lock; dry-run test in M4-T2 |
| 8 | Fork drifts from the license/attribution rules | Low | Reviewer checklist in `docs/agents/WORKFLOW.md` already enforces license and attribution per PR |

---

## 9. Traceability

- **Task → issue.** Each row is one issue created by the analyst with the
  contract from `AGENTS.md` (context, goal, requirements, acceptance criteria,
  files, out of scope, test plan, risks, estimate). The issue title should
  carry the task ID, e.g. `[cuda] Replace ldmatrix on sm_70 (M1-T3)`.
- **Task → PR.** The developer opens one PR towards `v100` per issue,
  `Closes #<issue>`; the reviewer merges and closes. The coordinator then
  fills the `issue` and `PR` columns and sets the status.
- **Measurements.** Every task that changes performance or numerics appends
  to `docs/V100-results.md` (append-only) and links the entry from its issue.
- **Superseded artifacts.** Issue #1 remains for history (closed); this
  document is the plan of record and is maintained in English only.
### 9.1 GitHub Project and milestones

- **Project:** [RustingStrata V100](https://github.com/users/TheBrutalX/projects/5)
  (project 5), linked to `TheBrutalX/RustingStrata`.
- **Views:** `Roadmap` (table: Task ID, Milestone group, Status, Size, Depends
  on) and `Board` (board by Status).
- **Fields:** built-in Status (`Todo`, `In progress`, `In review`, `Blocked`,
  `Done`), `Milestone group` (M0–M4), `Task ID`, `Size`, `Depends on`.
- **Items:** one draft item per task (36). The three completed M0 tasks are
  `Done`; the rest are `Todo`.
- **Repository milestones:** `M0 — Baseline, process and Vulkan stability` (#1),
  `M1 — CUDA sm_70 bring-up` (#2), `M2 — CUDA performance on the V100` (#3),
  `M3 — Vulkan fallback hardening` (#4), `M4 — CI, parity and release` (#5).

**Sync protocol (coordinator).** When the analyst opens the issue for a task:

1. add the issue to the project:
   `gh project item-add 5 --owner TheBrutalX --url <issue-url>`;
2. set the repository milestone that matches the task's `Milestone group`;
3. copy `Task ID`, `Size` and `Depends on` to the issue item (the draft item
   keeps the long scope/acceptance text);
4. archive the draft item once the issue item carries the same fields
   (`gh project item-archive`), so the board has one row per task;
5. move `Status` as the workflow advances (`step:dev` → `In progress`,
   `step:review` → `In review`, merged → `Done`, blocked → `Blocked`).

The roadmap and the project must agree at all times: the document is the
source of truth, the project is the board view of it.

- **References.** `docs/V100-results.md` (numbers), `docs/baseline.md`,
  `docs/m2-results.md`, `docs/m4-results.md`, `docs/m6-results.md` (parity
  gate history), `docs/agents/WORKFLOW.md` (process),
  `src/cuda/kernels.cu`, `src/cuda/mod.rs`, `src/vulkan/mod.rs`,
  `src/vulkan/shaders/`, `tools/bench_v100.sh`, `tools/parity.py`,
  `tools/gen_spirv.sh`.

---

## 10. Open verification notes

Items the planner could not confirm from the repository or the machine and
that the analyst must verify while opening the first issue of the affected
task:

- The exact NVRTC compile diagnostics for `sm_70` were not produced during
  planning (the GPU is shared and compilation was not run). The blocker list
  in §5 is from code inspection of the unguarded asm.
- Whether `mma.sync.m8n8k4` mapping can keep the existing shared-memory
  fragment loads without changing the tile geometry is unproven; M2-T1 must
  prototype it before M2-T2 is estimated.
- The `attn_decode_t` synchronous `cp.async` fallback may need extra barriers
  (`asm volatile("" ::: "memory")` does not synchronize threads); M1-T6 must
  audit it.
- Deleting or keeping `gemmi_q` on sm_70 does not change performance if
  `Gpu::mmq` is false, but the compile-time guard and the runtime function
  list must agree; M1-T5 covers both sides.
- No non-Volta NVIDIA GPU was available to re-run the sm_80 CUDA tests; the
  "sm_80 unchanged" acceptance criterion is review-by-diff unless the
  coordinator can provide one.
