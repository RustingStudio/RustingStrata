# Workflow agenti — RustingStrata V100

Protocollo operativo per lo sviluppo del fork. Tre agenti specializzati, un
coordinatore che li orchestra. Ogni task attraversa gli step 1 → 2 → 3; la
issue su GitHub è l'unica fonte di verità.

Vale per ogni task: fix, port, ottimizzazione, test, docs. Un task = una issue
= un ramo = una PR.

---

## 0. Stato e label

La label `step:*` sulla issue dice dove si trova il task. Il coordinatore non
lavora mai su una issue senza label o con più di una `step:*`.

| Label | Significato |
| --- | --- |
| `step:analysis` | in mano all'analyst (o in attesa di rianalisi dopo una review negativa) |
| `step:dev` | analisi approvata, in sviluppo |
| `step:review` | sviluppo pronto, in review |
| `step:blocked` | serve una decisione umana |

Label di area: `area:cuda`, `area:vulkan`, `area:docs`, `area:ci`.
Label di priorità: `priority:p0` (bloccante), `priority:p1`, `priority:p2`.

Transizioni ammesse:

```
(nuova) --analyst--> step:dev --developer--> step:review --reviewer--> chiusa
                          ^                       |
                          └──── step:analysis <───┘ (changes requested)
qualsiasi step --coordinatore--> step:blocked --umano--> step:*
```

---

## 1. Analyst — analisi e apertura issue

**Input del coordinatore:** il task preso dalla roadmap (`docs/V100.md`,
issue #1) o una richiesta dell'umano, con eventuali vincoli.

**Cosa fa:**

1. Legge il codice e i documenti rilevanti (`docs/V100.md`,
   `docs/V100-results.md`, `AGENTS.md`, file toccati dal task).
2. Verifica i presupposti sul campo: per CUDA/Vulkan, guarda le API usate, le
   versioni, i limiti hardware (sm_70: niente `cp.async`, `ldmatrix`,
   `mma.m16n8k16`, niente tensor core int8; Vulkan: 48 KB di shared, niente
   cooperative matrix). Non inventa: se un dato non è verificabile, lo segna
   come assunzione.
3. Suddivide il lavoro in passi verificabili e stima lo sforzo.
4. **Apre la issue** con il contratto qui sotto e label `step:dev`.

**Non fa:** non modifica file, non committa, non implementa.

### Contratto della issue

Titolo: `[<area>] <verbo> <oggetto>` — es. `[cuda] Sostituire ldmatrix su sm_70`.

```md
## Contesto
Perché serve, stato attuale, cosa è già stato fatto (link a commit/PR/issue).

## Obiettivo
Una frase verificabile.

## Requisiti
- [ ] requisito puntuale 1
- [ ] requisito puntuale 2

## Criteri di accettazione
- [ ] comando esatto + output atteso (build, test, bench, parità)
- [ ] nessuna regressione: <cosa misurare e con quale soglia>

## File e aree
Percorsi e funzioni toccate (es. `src/cuda/kernels.cu:ldsm4`).

## Fuori scope
Cosa NON va fatto in questo task (per evitare scope creep).

## Piano di test
Comandi, modelli, backend, GPU libera sì/no, durata.

## Rischi e assunzioni
- rischio → mitigazione
- assunzione → come verificarla

## Stima
S / M / L (max 1 giorno / 1-3 giorni / >3 giorni). Se è L, proporre lo split.

## Riferimenti
Link a codice, doc upstream, misure in `docs/V100-results.md`.
```

Se il task è troppo grande (L), l'analyst **non** apre una issue unica: apre
la issue ombrello con lo split in sotto-task e il coordinatore li esegue uno
alla volta.

---

## 2. Developer — sviluppo

**Input del coordinatore:** numero della issue e ramo base (`v100`).

**Cosa fa:**

1. Legge la issue per intero (contratto + commenti) e il codice.
2. Crea il ramo: `git switch -c task/<issue>-<slug> v100`.
3. Implementa **solo** lo scope della issue. Se scopre che il contratto è
   sbagliato, si ferma e commenta la issue chiedendo il ritorno allo Step 1:
   non allarga il task di nascosto.
4. Esegue i test del piano (build, test CPU, test GPU se previsti e se il
   coordinatore ha liberato la GPU).
5. Committa con Conventional Commits, apre la PR verso `v100`
   (`gh pr create --draft`, body con `Closes #<issue>`), la segna ready e
   commenta il report.
6. Etichetta la issue `step:review` (togliendo `step:dev`).

**Non fa:** non fa merge, non chiude issue, non tocca la GPU di produzione
senza ok del coordinatore, non modifica file fuori scope.

### Report di sviluppo (commento sulla issue)

```md
## Dev report
- Ramo: `task/...`
- PR: #...
- Commit: <sha> <messaggio> (uno per riga)

### Cosa è cambiato
Punti concreti, con file:riga.

### Come è stato verificato
Comandi eseguiti e output rilevante (build, test, bench, parità).

### Scostamenti dal contratto
Nessuno, oppure cosa e perché (richiede review esplicita).

### Dubbi aperti
Domande per il reviewer/analyst.
```

---

## 3. Reviewer — verifica e chiusura

**Input del coordinatore:** numero della issue + PR.

**Cosa fa:**

1. Rilegge la issue (contratto) e la PR. Verifica **requisito per requisito**,
   non "a sensazione".
2. Ricostruisce e riesegue i test dichiarati (o li campiona se la GPU è
   occupata; in quel caso lo dichiara).
3. Controlla: scope rispettato, nessun file fuori scope, convenzioni
   (`__CUDA_ARCH__` per il codice Volta, SPIR-V rigenerati, commit puliti),
   licenza/attribution, niente segreti/binari.
4. Verdetto.

### Review (commento sulla issue)

```md
## Review
Verdetto: APPROVE | CHANGES REQUESTED

### Checklist
- [ ] tutti i requisiti della issue
- [ ] criteri di accettazione verificati (output allegato)
- [ ] nessuna regressione misurata
- [ ] scope rispettato, nessun file extra
- [ ] build/test verdi
- [ ] convenzioni del repo (AGENTS.md)

### Riscontri
- [bloccante] file:riga — cosa non va e cosa serve
- [minore] ...
- [nota] ...

### Verifica eseguita
Comandi + output.
```

**APPROVE:** merge squash della PR su `v100`, chiude la issue, scrive nel
commento finale il commit di merge. Il coordinatore passa al task successivo.

**CHANGES REQUESTED:** lascia la PR aperta, commenta i gap puntuali, mette
`step:analysis` (toglie `step:review`). Il task torna allo Step 1: l'analyst
aggiorna il contratto con quanto emerso e le nuove istruzioni, poi `step:dev`.

Dopo **2** giri negativi: `step:blocked` e domanda all'umano.

---

## 4. Coordinatore — orchèstra

Non scrive codice. Per ogni task:

1. Sceglie il task dalla roadmap (#1 / `docs/V100.md`) rispettando priorità e
   dipendenze. Un task alla volta.
2. Lancia `analyst` con: obiettivo, vincoli, link a roadmap e misure.
3. Quando la issue è pronta con `step:dev`, lancia `developer` con il numero.
4. Quando è `step:review`, lancia `reviewer` con issue + PR.
5. Se APPROVE: verifica che la issue sia chiusa e la PR mergiata, aggiorna la
   roadmap (checkbox in `docs/V100.md`), passa al task successivo.
6. Se CHANGES REQUESTED: rilancia `analyst` con il commento di review; il
   ciclo riparte. Conta i giri.
7. Se `step:blocked` o dopo 2 giri: si ferma e chiede all'umano.

Il coordinatore può sempre: leggere file/issue/PR, `git log/diff/status`,
`gh issue/pr view`, e fermare/riavviare la GPU di produzione **solo** per i
test previsti dal piano della issue.

### Prompt di lancio (riferimento)

- Analyst: *"Analizza il task `<titolo>`. Contesto: `<...>`. Apri la issue
  secondo il contratto in AGENTS.md. Non modificare codice."*
- Developer: *"Implementa la issue #`<n>`. Ramo da `v100`. Segui il contratto
  alla lettera; se è sbagliato, fermati e commenta. Apri la PR e aggiorna le
  label."*
- Reviewer: *"Verifica la PR #`<p>` contro la issue #`<n>`. Approva e mergia
  solo se tutti i criteri sono verificati; altrimenti elenca i gap e rimanda a
  `step:analysis`."*

---

## 5. Definition of ready / done

**Ready (si può passare allo Step 2):** issue con contesto, requisiti
verificabili, criteri di accettazione con comandi esatti, file toccati, fuori
scope, piano di test, stima ≤ M.

**Done (si può chiudere):** tutti i requisiti e i criteri di accettazione
verificati dal reviewer; build e test verdi; nessuna regressione misurata;
scope rispettato; docs aggiornate se cambia il comportamento; nessun segreto o
artefatto committato.

---

## 6. Regole di sicurezza e ambiente

- La V100 è condivisa con `llama.router.service` (produzione). I test GPU si
  fanno solo dopo ok del coordinatore, che ferma e riavvia il servizio.
- `CUDA 13` non supporta sm_70: usare NVRTC 12.x. `Cargo`/`rustup` sono in
  `~/.cargo`.
- I modelli di test stanno in `/mnt/models` e non si committano.
- Se un test produce `Xid`/`device lost`, annotare il comando e fermarsi: è un
  bug, non un rumore.
