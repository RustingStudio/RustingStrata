# Agent workflow — RustingStrata V100

Operating protocol for the fork. Three delivery steps (analysis, development,
review) plus a planning step that produces the living roadmap. A coordinator
orchestrates them. The GitHub issue is the single source of truth for every
task.

Applies to every task: fix, port, optimization, test, docs. One task = one
issue = one branch = one PR. **Everything is written in English.**

---

## 0. State and labels

The `step:*` label on the issue says where the task is. The coordinator never
works on an issue that has no `step:*` label or more than one.

| Label | Meaning |
| --- | --- |
| `step:analysis` | with the analyst (or waiting for re-analysis after a failed review) |
| `step:dev` | analysis accepted, in development |
| `step:review` | development ready, in review |
| `step:blocked` | a human decision is needed |

Area labels: `area:cuda`, `area:vulkan`, `area:docs`, `area:ci`.
Priority labels: `priority:p0` (blocking), `priority:p1`, `priority:p2`.

Allowed transitions:

```
(new) --analyst--> step:dev --developer--> step:review --reviewer--> closed
                        ^                       |
                        └──── step:analysis <───┘ (changes requested)
any step --coordinator--> step:blocked --human--> step:*
```

The **roadmap** (`docs/ROADMAP.md`) is a separate artifact: it lists
milestones and tasks with their status and points to the issues. The
coordinator keeps it updated; it is not driven by labels.

---

## 0-bis. Planner — milestones and tasks

**Input from the coordinator:** the request (or the next milestone to break
down) and the constraints.

**What it does:**

1. Reads `AGENTS.md`, this protocol, `docs/V100-results.md`, the open issues
   and the relevant code.
2. Decomposes the request into milestones with verifiable exit criteria, and
   each milestone into tasks small enough for one issue and one PR (S or M; L
   must be split).
3. Writes `docs/ROADMAP.md` with: goals/non-goals, hard constraints, status
   legend, milestones (objective, exit criteria, dependencies, risks), task
   tables (ID, title, scope, files, size, dependencies, acceptance, status,
   issue, PR), critical path.

**Does not:** modify code, commit, open issues, invent measurements.

---

## 1. Analyst — analysis and issue creation

**Coordinator input:** the task taken from `docs/ROADMAP.md` (or a human
request), with any constraints.

**What it does:**

1. Reads the code and the relevant documents (`docs/ROADMAP.md`,
   `docs/V100-results.md`, `AGENTS.md`, files touched by the task).
2. Checks assumptions against reality: for CUDA/Vulkan, look at the APIs used,
   versions and hardware limits (sm_70: no `cp.async`, `ldmatrix`,
   `mma.m16n8k16`, no int8 tensor cores; Vulkan: 48 KB shared memory, no
   cooperative matrix). Never invent: mark unverifiable items as assumptions.
3. Splits the work into verifiable steps and estimates effort.
4. **Opens the issue** with the contract below and sets `step:dev`.

**Does not:** modify files, commit, implement.

### Issue contract

Title: `[<area>] <verb> <object>` — e.g. `[cuda] Replace ldmatrix on sm_70`.

```md
## Context
Why this is needed, current state, what has already been done (links).

## Goal
One verifiable sentence.

## Requirements
- [ ] specific requirement 1
- [ ] specific requirement 2

## Acceptance criteria
- [ ] exact command + expected output (build, test, bench, parity)
- [ ] no regression: <what to measure and the threshold>

## Files and areas
Paths and functions touched (e.g. `src/cuda/kernels.cu:ldsm4`).

## Out of scope
What must NOT be done in this task (avoid scope creep).

## Test plan
Commands, models, backend, GPU free yes/no, duration.

## Risks and assumptions
- risk -> mitigation
- assumption -> how to verify it

## Estimate
S / M / L (max 1 day / 1-3 days / >3 days). If L, propose the split.

## References
Links to code, upstream docs, measurements in `docs/V100-results.md`.
```

If the task is L, the analyst does **not** open a single issue: it opens an
umbrella issue with the split into sub-tasks and the coordinator executes them
one at a time.

---

## 2. Developer — implementation

**Coordinator input:** issue number and base branch (`v100`).

**What it does:**

1. Reads the whole issue (contract + comments) and the code.
2. Creates the branch: `git switch -c task/<issue>-<slug> v100`.
3. Implements **only** the issue scope. If the contract turns out to be wrong,
   it stops and comments on the issue asking to go back to step 1: it does not
   silently widen the task.
4. Runs the tests in the plan (build, CPU tests, GPU tests if planned and if
   the coordinator has freed the GPU).
5. Commits with Conventional Commits, opens the PR towards `v100`
   (`gh pr create --draft`, body with `Closes #<issue>`), marks it ready and
   posts the report.
6. Sets the issue to `step:review` (removing `step:dev`).

**Does not:** merge, close issues, touch the production GPU without the
coordinator's approval, modify files outside scope.

### Dev report (issue comment)

```md
## Dev report
- Branch: `task/...`
- PR: #...
- Commits: <sha> <message> (one per line)

### What changed
Concrete points, with file:line.

### How it was verified
Commands run and relevant output (build, test, bench, parity).

### Deviations from the contract
None, or what and why (requires explicit review).

### Open questions
Questions for the reviewer/analyst.
```

---

## 3. Reviewer — verification and closure

**Coordinator input:** issue number + PR.

**What it does:**

1. Re-reads the issue (contract) and the PR. Verifies **requirement by
   requirement**, not by impression.
2. Rebuilds and re-runs the declared tests (or samples them if the GPU is
   busy; in that case it says so).
3. Checks: scope respected, no out-of-scope files, conventions
   (`__CUDA_ARCH__` for Volta code, regenerated SPIR-V, clean commits),
   license/attribution, no secrets/binaries.
4. Delivers a verdict.

### Review (issue comment)

```md
## Review
Verdict: APPROVE | CHANGES REQUESTED

### Checklist
- [ ] all issue requirements
- [ ] acceptance criteria verified (output attached)
- [ ] no measured regression
- [ ] scope respected, no extra files
- [ ] build/tests green
- [ ] repository conventions (AGENTS.md)

### Findings
- [blocking] file:line — what is wrong and what is needed
- [minor] ...
- [note] ...

### Verification performed
Commands + output.
```

**APPROVE:** squash-merge the PR into `v100`, close the issue, write the merge
commit in the final comment. The coordinator updates the roadmap and moves to
the next task.

**CHANGES REQUESTED:** leaves the PR open, comments the specific gaps, sets
`step:analysis` (removing `step:review`). The task goes back to step 1: the
analyst updates the contract with what emerged and the new instructions, then
`step:dev`.

After **2** failed rounds: `step:blocked` and a question to the human.

---

## 4. Coordinator — orchestration

Does not write code. For every task:

1. Takes the next task from `docs/ROADMAP.md`, respecting priority and
   dependencies. One task at a time.
2. Launches `analyst` with: objective, constraints, links to the roadmap and
   measurements.
3. When the issue is ready with `step:dev`, launches `developer` with the
   number.
4. When it is `step:review`, launches `reviewer` with issue + PR.
5. On APPROVE: verifies the issue is closed and the PR merged, updates
   `docs/ROADMAP.md` (task status, issue/PR links), moves to the next task.
6. On CHANGES REQUESTED: relaunches `analyst` with the review comment; the
   cycle restarts. Counts the rounds.
7. On `step:blocked` or after 2 rounds: stops and asks the human.

The coordinator may always: read files/issues/PRs, `git log/diff/status`,
`gh issue/pr view`, and stop/restart the production GPU **only** for the tests
planned in the issue.

All GitHub commands pass `--repo TheBrutalX/RustingStrata` (the fork has its own
tracker, separate from upstream; `gh repo set-default` is already configured).

### Launch prompts (reference)

- Planner: *"Decompose `<request>` into milestones and tasks and write
  `docs/ROADMAP.md` per the contract in AGENTS.md. English only."*
- Analyst: *"Analyse task `<title>`. Context: `<...>`. Open the issue per the
  contract in AGENTS.md. Do not modify code."*
- Developer: *"Implement issue #`<n>`. Branch from `v100`. Follow the contract
  exactly; if it is wrong, stop and comment. Open the PR and update the
  labels."*
- Reviewer: *"Verify PR #`<p>` against issue #`<n>`. Approve and merge only if
  every criterion is verified; otherwise list the gaps and send it back to
  `step:analysis`."*

---

## 5. Definition of ready / done

**Ready (can move to step 2):** issue with context, verifiable requirements,
acceptance criteria with exact commands, files touched, out of scope, test
plan, estimate <= M.

**Done (can be closed):** all requirements and acceptance criteria verified by
the reviewer; build and tests green; no measured regression; scope respected;
docs updated when behaviour changes; no secrets or artifacts committed.

---

## 6. Safety and environment rules

- The V100 is shared with `llama.router.service` (production). GPU tests run
  only after the coordinator's approval; the coordinator stops and restarts
  the service.
- `CUDA 13` does not support sm_70: use NVRTC 12.x. `cargo`/`rustup` live in
  `~/.cargo`.
- Test models live in `/mnt/models` and are never committed.
- If a test produces `Xid`/`device lost`, record the command and stop: that is
  a bug, not noise.
