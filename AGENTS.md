# AGENTS.md — RustingStrata V100 fork

Questo repository è il fork `TheBrutalX/RustingStrata`, dedicato a far girare
RustingStrata su NVIDIA Tesla V100 (Volta, **sm_70**) e a ottimizzarlo.

Origine: `RustingStudio/RustingStrata` (remote `upstream`), licenza
**RustingStrata License 1.0** (vedi `LICENSE.md`): fork e modifiche sono
consentiti, anche per servizi commerciali, con licenza e credito all'autore
originale; è vietato vendere l'engine stesso. Ogni file derivato mantiene
licenza e attribution.

## Rami

- `main` = specchio di `upstream/main` (non si modifica a mano).
- `v100` = ramo di integrazione e **default branch del fork**; qui vivono i
  fix e la documentazione del fork.
- `task/<issue>-<slug>` = un ramo per task, PR verso `v100`.
- Sync: `git fetch upstream && git merge upstream/main` su `v100`.

## Regole non negoziabili

1. **Il coordinatore non scrive codice.** Il suo unico compito è orchestrare
   gli step, leggere issue/PR e decidere il prossimo passo. Le modifiche le
   fanno solo gli agenti `analyst`, `developer`, `reviewer`.
2. **Una issue = un task.** La issue è l'unica fonte di verità: analisi,
   istruzioni, avanzamento e verdetto vivono lì (e nella PR collegata).
3. **Nessun test che fermi la produzione senza via libera del coordinatore.**
   La V100 ospita `llama.router.service`: va fermato solo per i test GPU che
   lo richiedono, e riavviato subito dopo.
4. **Compatibilità upstream.** I rami restano allineabili a `upstream/main`.
   Le modifiche CUDA specifiche di Volta vanno dietro `#if __CUDA_ARCH__`
   (o helper equivalenti) senza cambiare il comportamento su sm_80+.
   I fix Vulkan rigenerano gli SPIR-V con `tools/gen_spirv.sh`.
5. **Niente segreti, modelli o binari nel repo.** I modelli stanno in
   `/mnt/models`, le build in `target/`.
6. **La storia pubblicata non si riscrive.** Niente force-push su `main`,
   `v100` o rami già in review.

## Workflow (obbligatorio)

Dettaglio completo: [`docs/agents/WORKFLOW.md`](docs/agents/WORKFLOW.md).

```
        ┌──────────────┐   issue + label step:dev
task ──▶│ 1. ANALYST   │──────────────────────────┐
        └──────────────┘                          ▼
                                          ┌──────────────┐
                                          │ 2. DEVELOPER │ commit + PR, label step:review
                                          └──────┬───────┘
                                                 ▼
                                          ┌──────────────┐
                                          │ 3. REVIEWER  │
                                          └──┬────────┬──┘
                               CHANGES ──────┘        └────── APPROVE
                                    │                        │
                                    ▼                        ▼
                        torna a STEP 1 (issue      merge + close issue
                        aggiornata con i gap)      → prossimo task
```

- **Step 1 — Analyst** (`analyst`): legge il task dal coordinatore, analizza il
  codice, produce il report e **apre la issue** con il contratto completo
  (contesto, requisiti, criteri di accettazione, file, fuori scope, test,
  rischi, stima). Etichetta `step:dev`.
- **Step 2 — Developer** (`developer`): lavora **solo** sulla issue assegnata,
  crea il ramo `task/<issue>-<slug>`, committa, apre la PR verso `v100`,
  commenta il report di sviluppo. Etichetta `step:review`.
- **Step 3 — Reviewer** (`reviewer`): verifica il lavoro **contro la issue**,
  non contro le intenzioni. Se è corretto e completo: merge squash, chiude la
  issue. Altrimenti: commenta i gap puntuali (file:riga), etichetta
  `step:analysis` e il task torna allo Step 1.
- Dopo **2 giri di review falliti** il task va in `step:blocked` e il
  coordinatore chiede istruzioni all'umano.

## Ruoli OpenCode

Definiti in `.opencode/agents/`:

| Agente | Modalità | Può | Non può |
| --- | --- | --- | --- |
| `coordinator` | primary | leggere, orchestrare, `gh`, `git` in lettura | editare file, fare commit |
| `analyst` | subagent | leggere, cercare, `gh issue`, test in lettura | editare, committare |
| `developer` | subagent | editare, committare, push, `gh pr`, build/test | lanciare subagent |
| `reviewer` | subagent | leggere diff, build/test, `gh` (comment/merge/close) | editare codice |

Per usare il coordinatore come agente primario: selezionarlo nella sessione
(`/agent` o selettore agent) oppure impostare `"default_agent": "coordinator"`
in `.opencode/opencode.json`.

## Costruire e testare

```sh
cargo build --release

# test CPU (sempre eseguibili)
cargo test --release

# GPU Vulkan (V100) — richiede la GPU libera
M=/mnt/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf
target/release/strata bench --model "$M" --gpu --backend vulkan \
  --pp 512,4096,32768 --tg 128 --reps 3 --ctx 32896

# bench ripetibile (scrive in docs/V100-results.md)
tools/bench_v100.sh "$M" vulkan

# parità (dopo il port CUDA; richiede llama-server)
python3 -I tools/parity.py "$M" --extra "--gpu --backend cuda" --tag sm70
```

## Ambiente di riferimento

| Voce | Valore |
| --- | --- |
| GPU | Tesla V100-SXM2-32GB, sm_70, driver 580.178.04 (ultimo per Volta) |
| CUDA | toolkit/NVRTC 12.9 (`/usr/local/cuda`), CUDA 13 **non** supporta sm_70 |
| Vulkan | 1.4.312, **senza** `VK_KHR_cooperative_matrix` |
| Rust | stable (rustup), edition 2024 |
| Modello di riferimento | `Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf` (12.3 GB) |
| Produzione | `llama.router.service` sulla stessa GPU |

## Convenzioni git

- Rami: `task/<numero-issue>-<slug>` (es. `task/12-cuda-sm70-warp-reduce`).
- Commit: Conventional Commits — `feat(cuda): ...`, `fix(vulkan): ...`,
  `test: ...`, `docs: ...`, `chore: ...`. Un commit = un passo logico.
- PR: titolo `[#<issue>] <titolo task>`, body con `Closes #<issue>`, verso
  `v100`, squash merge.
- Prima della review: `git fetch upstream && git rebase upstream/main`
  (o `v100`), build e test verdi.

## Mappa del progetto

- `docs/V100.md` — piano per fasi.
- `docs/V100-results.md` — numeri misurati (append-only).
- `docs/agents/WORKFLOW.md` — protocollo operativo degli agenti.
- `src/cuda/` — backend CUDA (`kernels.cu`, `mod.rs`).
- `src/vulkan/` — backend Vulkan (`mod.rs`, `shaders/`, `spv/`).
- `.github/ISSUE_TEMPLATE/task.yml` — contratto della issue.
