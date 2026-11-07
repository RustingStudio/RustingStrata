# AGENTS.md — RustingStrata V100 fork

This repository is the `TheBrutalX/RustingStrata` fork, dedicated to running and
optimizing RustingStrata on the NVIDIA Tesla V100 (Volta, **sm_70**).

Upstream: `RustingStudio/RustingStrata` (remote `upstream`), license
**RustingStrata License 1.0** (see `LICENSE.md`): forks and modifications are
allowed, including commercial services, provided the license and credit to the
original author are kept; selling the engine itself is not allowed. Every
derived file keeps the license and attribution.

## Branches

- `main` = mirror of `upstream/main` (never edited by hand).
- `v100` = integration branch and **default branch of the fork**; the fork's
  fixes and documentation live here.
- `task/<issue>-<slug>` = one branch per task, PR towards `v100`.
- Sync: `git fetch upstream && git merge upstream/main` on `v100`.

## Non-negotiable rules

1. **The coordinator does not write code.** Its job is to orchestrate the
   steps, read issues/PRs and decide the next action. Only the `planner`,
   `analyst`, `developer` and `reviewer` agents modify the repository.
2. **One issue = one task.** The issue is the single source of truth: analysis,
   instructions, progress and verdict live there (and in the linked PR).
3. **Everything is written in English**: issues, comments, documents, commits,
   PRs, code comments. No exceptions.
4. **Production gets stopped only inside a scheduled window.** The V100 also
   hosts `llama.router.service`. Before stopping it, drain in-flight and queued
   requests; stop it only for the GPU tests that require it; restart it
   immediately after **every** window (never leave it stopped between tests)
   and verify it is active and serving the queue again.
5. **Upstream compatibility.** Branches must stay rebasable onto
   `upstream/main`. Volta-specific CUDA changes go behind
   `#if __CUDA_ARCH__` (or equivalent helpers) without changing sm_80+
   behaviour. Vulkan fixes regenerate the SPIR-V with `tools/gen_spirv.sh`.
6. **No secrets, models or binaries in the repository.** Models live in
   `/mnt/models`, builds in `target/`.
7. **Published history is not rewritten.** No force-push to `main`, `v100` or
   any branch already in review.

## Workflow (mandatory)

Full protocol: [`docs/agents/WORKFLOW.md`](docs/agents/WORKFLOW.md).
Living plan: [`docs/ROADMAP.md`](docs/ROADMAP.md).

```
request ──▶ 0. PLANNER ──▶ docs/ROADMAP.md (milestones + tasks)
                                │
task ─────▶ 1. ANALYST ──▶ GitHub issue (step:dev)
                                │
          2. DEVELOPER ──▶ commit + PR (step:review)
                                │
          3. REVIEWER ──▶ APPROVE: merge + close
                          CHANGES: back to step 1 (step:analysis)
```

- **Step 0 — Planner** (`planner`): decomposes the request into milestones and
  tasks and writes `docs/ROADMAP.md`. It does not write code, commit or open
  issues. The coordinator keeps the document updated.
- **Step 1 — Analyst** (`analyst`): reads the task from the roadmap, analyses
  the code, opens the **issue** with the full contract (context, requirements,
  acceptance criteria, files, out of scope, test plan, risks, estimate) and
  sets `step:dev`.
- **Step 2 — Developer** (`developer`): works **only** on the assigned issue,
  creates `task/<issue>-<slug>`, commits, opens the PR towards `v100`, posts
  the dev report and sets `step:review`.
- **Step 3 — Reviewer** (`reviewer`): verifies the work **against the issue**,
  not against intentions. If complete and correct: squash merge and close the
  issue. Otherwise: comments the gaps (`file:line`), sets `step:analysis` and
  the task goes back to step 1.
- After **2 failed review rounds** the task goes to `step:blocked` and the
  coordinator asks the human.

## OpenCode roles

Defined in `.opencode/agents/`:

| Agent | Mode | Can | Cannot |
| --- | --- | --- | --- |
| `coordinator` | primary | read, orchestrate, `gh`, read-only `git` | edit files, commit |
| `planner` | subagent | read, `gh` read, write `docs/**` | edit code, commit, open issues |
| `analyst` | subagent | read, search, `gh issue`, read-only tests | edit code, commit |
| `developer` | subagent | edit, commit, push, `gh pr`, build/test | spawn subagents |
| `reviewer` | subagent | read diffs, build/test, `gh` (comment/merge/close) | edit code |

The project sets `default_agent: coordinator` in `.opencode/opencode.json`.
Select another agent from the session agent picker when needed.

## Build and test

```sh
cargo build --release

# CPU tests (always available)
cargo test --release

# GPU Vulkan (V100) — requires the GPU to be free
M=/mnt/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf
target/release/strata bench --model "$M" --gpu --backend vulkan \
  --pp 512,4096,32768 --tg 128 --reps 3 --ctx 32896

# repeatable benchmark (appends to docs/V100-results.md)
tools/bench_v100.sh "$M" vulkan

# parity (after the CUDA port; requires llama-server)
python3 -I tools/parity.py "$M" --extra "--gpu --backend cuda" --tag sm70
```

## Reference environment

| Item | Value |
| --- | --- |
| GPU | Tesla V100-SXM2-32GB, sm_70, driver 580.178.04 (last branch for Volta) |
| CUDA | toolkit/NVRTC 12.9 (`/usr/local/cuda`), CUDA 13 does **not** support sm_70 |
| Vulkan | 1.4.312, **without** `VK_KHR_cooperative_matrix` |
| Rust | stable (rustup), edition 2024 |
| Reference model | `Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf` (12.3 GB) |
| Production | `llama.router.service` on the same GPU |

## Git conventions

- Branches: `task/<issue-number>-<slug>` (e.g. `task/12-cuda-sm70-warp-reduce`).
- Commits: Conventional Commits — `feat(cuda): ...`, `fix(vulkan): ...`,
  `test: ...`, `docs: ...`, `chore: ...`. One commit per logical step.
- PR: title `[#<issue>] <task title>`, body with `Closes #<issue>`, target
  `v100`, squash merge.
- Before review: `git fetch upstream && git rebase upstream/main` (or `v100`),
  build and tests green.

## GitHub

`gh` is authenticated and the default repository is `TheBrutalX/RustingStrata`
(`gh repo set-default`). Agents still pass
`--repo TheBrutalX/RustingStrata` explicitly to avoid ambiguity with the
`upstream` remote. The fork's issues are separate from upstream's.

## Project map

- `docs/ROADMAP.md` — living milestone and task plan (updated by the coordinator).
- [GitHub Project 5](https://github.com/users/TheBrutalX/projects/5) — board view of the roadmap (draft items + repo milestones M0–M4); the coordinator keeps it in sync.
- `docs/V100-results.md` — measured numbers (append-only).
- `docs/agents/WORKFLOW.md` — agent protocol.
- `src/cuda/` — CUDA backend (`kernels.cu`, `mod.rs`).
- `src/vulkan/` — Vulkan backend (`mod.rs`, `shaders/`, `spv/`).
- `.github/ISSUE_TEMPLATE/task.yml` — issue contract.
