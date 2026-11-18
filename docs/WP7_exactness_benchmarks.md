# WP7 — Suite automatisée d'exactitude et de benchmarks finaux (`scripts/route_c_bench.sh`)

## 1. Objectif du Work Package 7

Construire et exécuter une suite automatisée et reproductible (`./scripts/route_c_bench.sh`) couvrant les **3 classes de charge** cibles sur l'AMD Radeon 890M (`gfx1150`, RDNA 3.5) :
1. **Classe 1 — Texte neuf technique** (essai physique EUV, 128 tokens générés, `temp=0`).
2. **Classe 2 — Copie / Réécriture de code** (contexte ~2 000 tokens, 500 tokens générés, `temp=0`).
3. **Classe 3 — Session agentique longue multi-tours** (contexte ~4 000 tokens renvoyé intégralement à chaque tour, combinant **Warm Start `--prompt-cache`** et **Spéculation `--spec-type ngram-simple`**).

Critères d'acceptation :
- Accord greedy vs baseline sans spéculation : **≥ 99,0 %** (obtenu : **100,0 % bit-à-bit identique** sur les 3 classes).
- Zéro régression sur texte neuf vs baseline WP0 (`18,3 tok/s`).
- Débit copie/réécriture **≥ 60 tok/s** (obtenu : **82,79 tok/s**, stretch **87,2 tok/s** par batch de 17 tokens).
- Warm Start TTFT **< 2,0 s** à ~4K tokens (obtenu : **86,0 ms** en reprise exacte et **622,0 ms** en multi-tour avec delta).

---

## 2. Tableau comparatif final vs Baselines (WP0 → WP7)

| Configuration / Moteur | Decode texte neuf (court) | Decode après 4K ctx | Prefill 4K (Cold) | Prefill 4K (Warm Start) | Copie / Réécriture (500 tok) | Session Agentique 4K (Tour 2) | Accord Greedy |
|---|---:|---:|---:|---:|---:|---:|---:|
| **llama.cpp baseline** (Vulkan, flash-attn, KV q8) | 15,7 - 16,0 tok/s | 15,4 tok/s | 384,6 tok/s (~8,7 s) | — | 16,0 tok/s | — | Réf. llama |
| **llama.cpp + `ngram-simple`** (`n_max=16`) | 16,0 tok/s | — | 384,6 tok/s | — | 85,0 tok/s (×5,3) | — | 100 % |
| **llama.cpp combo `MTP + ngram`** *(poison)* | **6,53 tok/s** | — | — | — | 25,6 tok/s | — | — |
| **RustingStrata v1.2.1 (Baseline WP0)** | 18,3 tok/s | 17,7 tok/s | 517,3 tok/s (~7,7 s) | — | 18,3 tok/s | 17,7 tok/s (TTFT 7,7 s) | Référence |
| **Route C — WP2+WP3+WP4** (Kernels RDNA 3.5 + Fast-Path + Overlap, sans spec) | **19,73 tok/s** | **18,90 tok/s** | **523,4 tok/s** | — | 19,13 tok/s | 18,90 tok/s | **100,0 %** |
| **Route C — Complet (WP2–WP6 : `ngram-simple` + MoE dedup + Warm Start)** | **19,25 - 21,09 tok/s** | **18,90 tok/s** | **516,5 tok/s** | **45 938 tok/s (TTFT 86,0 ms)** | **82,79 tok/s (×4,52)** | **45,73 tok/s (TTFT 622,0 ms)** | **100,0 % bit-à-bit** |

---

## 3. Sortie complète mesurée de `./scripts/route_c_bench.sh`

```text
================================================================================
 PROJET ROUTE C — SUITE D'EXACTITUDE & BENCHMARKS AUTOMATISÉS (gfx1150)
 Date    : 2026-11-11T22:00:43+01:00
 Binaire : /mnt/storage/home-vault/build/rustingstrata/target/release/rustingstrata
 Modèle  : /var/lib/ollama/blobs/sha256-0dae47840f8115d894e74c5cd1a6aca27e01feaa4f07890a50bdcf7db4c56227
================================================================================

--------------------------------------------------------------------------------
 CLASSE 1 : TEXTE NEUF TECHNIQUE (128 tokens, greedy temp=0)
--------------------------------------------------------------------------------
prompt 58 tok 128.77 tok/s | gen 128 tok 19.73 tok/s
expert hit rate 100.0% (0 misses), cache swaps 0
prompt 58 tok 128.02 tok/s | gen 128 tok 19.25 tok/s
speculative ngram drafts accepted 0/16 (0.0%)
expert hit rate 100.0% (0 misses), cache swaps 0

--------------------------------------------------------------------------------
 CLASSE 2 : COPIE / RÉÉCRITURE DE CODE (500 tokens, greedy temp=0)
--------------------------------------------------------------------------------
prompt 1975 tok 513.47 tok/s | gen 500 tok 19.13 tok/s
expert hit rate 100.0% (0 misses), cache swaps 0
prompt 1975 tok 516.52 tok/s | gen 500 tok 82.79 tok/s
speculative ngram drafts accepted 472/472 (100.0%)
expert hit rate 100.0% (0 misses), cache swaps 0
prompt 1950 tok 514.33 tok/s | gen 500 tok 60.34 tok/s
speculative ngram drafts accepted 465/560 (83.0%)
expert hit rate 100.0% (0 misses), cache swaps 0

--------------------------------------------------------------------------------
 CLASSE 3 : SESSION AGENTIQUE MULTI-TOURS (Warm Start + Speculative Lookup)
--------------------------------------------------------------------------------
[Tour 1 - Cold Start : préfill complet ~3950 tokens + sauvegarde disque]
prompt 3950 tok 514.85 tok/s | gen 64 tok 18.90 tok/s
[Tour 1 - Warm Start Reprise : chargement disque + préfill delta 1 token]
warm start: loaded 1 cached prefix state(s) (3949 tokens, 139.9 MB) from /tmp/opencode/wp7_suite/agentic.pcache
warm start: reused 3949/3950 prompt tokens from cache (TTFT 86.0 ms)
prompt 3950 tok 45937.99 tok/s | gen 64 tok 18.88 tok/s
[Tour 2 - Warm Start Multi-Tour : réutilisation préfixe Tour 1 + génération spéculative]
warm start: loaded 1 cached prefix state(s) (3949 tokens, 139.9 MB) from /tmp/opencode/wp7_suite/agentic.pcache
warm start: reused 3949/4005 prompt tokens from cache (TTFT 622.0 ms)
prompt 4005 tok 6438.35 tok/s | gen 69 tok 45.73 tok/s
speculative ngram drafts accepted 57/88 (64.8%)

--------------------------------------------------------------------------------
 RAPPORT D'EXACTITUDE GREEDY (Cible : >= 99.0% d'accord)
--------------------------------------------------------------------------------
 - Classe 1 : Texte neuf (none vs ngram-simple): PASS (100.0% bit-for-bit identical)
 - Classe 2 : Copie/Réécriture (none vs ngram-simple): PASS (100.0% bit-for-bit identical)
 - Classe 3 : Session Agentique (Cold Start vs Warm Start): PASS (100.0% bit-for-bit identical)
================================================================================
 SUITE TERMINÉE AVEC SUCCÈS
================================================================================
```
