# WP5 — Décodage spéculatif par lookup de contexte (`ngram` & cache statique/dynamique)

## 1. Objectif du Work Package 5 (LE WP CRITIQUE)

Porter la logique de décodage spéculatif sans modèle de draft (`ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache` avec `--lookup-cache-static` et `--lookup-cache-dynamic`) depuis `llama.cpp` vers `RustingStrata` sur le chemin Vulkan (`Radeon 890M`, `gfx1150`), avec :
1. **Zéro surcharge mémoire GPU pour le drafter** : recherche N-gram sur CPU en quelques microsecondes à partir du prompt courant, de la génération en cours et des caches de lookup statique/dynamique sur disque.
2. **Vérification GPU en un seul passage (`nt ≤ 17`, `1` token courant + jusqu'à `16` tokens spéculatifs)** avec déduplication des experts routés inter-tokens dans `moe_gemv_q.comp` et rollback exact des états récurrents hybrides (`DeltaNet` + `conv1d` + KV cache).
3. **Exclusion mutuelle stricte avec `--mtp`** pour empêcher structurellement la régression observée sur `llama.cpp` (`combo MTP + ngram = 6,53 tok/s`).
4. **Cibles** :
   - Copie / réécriture de code : **≥ 60 tok/s** (stretch **85+ tok/s**), soit **×3+** vs baseline.
   - Texte neuf : **zéro régression** vs baseline (`18,3 tok/s`), accord greedy **100 % bit-à-bit**.

---

## 2. Architecture et optimisations implémentées

### 2.1. Module `src/ngram.rs` (Portage complet des 5 algorithmes de `llama.cpp`)
- **`ngram-simple`** (`ngram_simple_draft_eos`) :
  - Recherche arrière d'un N-gram de taille `N` (défaut `12`, avec repli adaptatif `12 → 6 → 4` pour franchir immédiatement la frontière `<|im_start|>assistant\n` des prompts formatés en chat et cooldown automatique en cas de rejet sur texte neuf).
  - Évitement intelligent des fenêtres traversant un token de fin de tour (`<|im_end|>`) lorsqu'une occurrence antérieure propre existe dans le contexte.
- **`ngram-map-k` et `ngram-map-k4v`** (`NgramMap`) :
  - Table de hachage LCG (`262 144` entrées) indexant les clés de longueur `N` et jusqu'à `4` continuations distinctes de longueur `M` (`--spec-ngram-min-hits`).
  - Adaptation dynamique de la longueur de draft (`n_accepted`) après chaque étape de vérification.
- **`ngram-mod`** (`NgramMod`) :
  - Table légère à adressage direct (`4 MiB`) avec fenêtre glissante et désactivation temporaire si le taux d'acceptation passe sous `n_min`.
- **`ngram-cache`** (`NgramCacheMap`, `--lookup-cache-static`, `--lookup-cache-dynamic`) :
  - Hiérarchie à 3 niveaux (`nc_context`, `nc_dynamic`, `nc_static`) utilisant les seuils de confiance `LAX` et `STRICT` de `llama.cpp` sur les 1..4-grams.
  - Chargement dual : supporte nativement le format binaire `llama.cpp` **et** les fichiers texte UTF-8 de scaffolds (JSON de tool-calls, chemins, patterns d'outils), tokenisés à la volée au chargement.
  - Sauvegarde atomique du cache dynamique (`save_dynamic`) après chaque génération CLI et chaque requête HTTP `serve`.

### 2.2. Vérification batchée `nt = 1..17` et déduplication des experts MoE (`moe_gemv_q.comp`)
Lors d'une étape de vérification spéculative de `nt = 17` tokens (`1 + 16` drafts) :
- Les poids denses (`1 267 Mo`) sont lus **une seule fois** par `gemv_q.comp` (avec déroulement super-bloc adapté à `NT ≤ 17`).
- **Problème initial identifié au profilage** : sans déduplication des experts routés, `moe_gemv_q.comp` et `moe_down_combine.comp` exécutaient `nt × 8 = 136` slots indépendamment, relisant les mêmes experts `Q3_K` / `IQ3_XXS` jusqu'à 17 fois depuis la LPDDR5X (`285,6 ms` par batch de 17 tokens, plafonnant à `54,2 tok/s`).
- **Solution implémentée (`src/vulkan/shaders/moe_gemv_q.comp` + `src/vulkan/mod.rs`)** :
  - Lorsque `nslots > 8` (`nt > 1`), chaque workgroup inspecte les `nslots` entrées de `slots[]`.
  - Si un slot précédent `p < slot` dans le même batch de vérification cible déjà le même expert `e`, le workgroup se termine immédiatement (`return;`).
  - Le premier workgroup associé à l'expert `e` collecte la liste des `nm` slots partageant cet expert (`s_slots[0..nm]`), charge chaque super-bloc `Q3_K` / `IQ3_XXS` de l'expert `e` **une seule fois depuis la LPDDR5X**, et accumule les produits scalaires `dotPacked4x8EXT` pour tous les tokens correspondants en registre.
  - De même, pour `nt > 1`, `moe_down_combine` emprunte le chemin `moe_gemv` dédupliqué + `moe_combine` au lieu de relire la matrice `down` par slot.
  - **Gain mesuré** : le temps d'un batch de vérification de 17 tokens chute de **`285,6 ms` à `195,0 ms` (`-31,7 %`)**, soit **`87,2 tok/s` en régime établi** et **`81,94 tok/s` de bout en bout** sur 500 tokens !

### 2.3. Rollback exact de l'état hybride (`DeltaNet` + `conv1d` + KV cache)
- `qwen35moe` combine 10 couches d'Attention (`KV cache`) et 30 couches récurrentes `Gated DeltaNet` (`conv` + matrice d'état `s` de `2,09 Mo` par couche).
- Pendant `verify(nt)`, le kernel `gated_deltanet.comp` sauvegarde des snapshots intermédiaires `(conv, s)` aux positions `0..SNAP_MAX` (`SNAP_MAX = 8`, soit `525 Mo` au total pour les 30 couches DeltaNet, laissant `10240/10240` experts résidents en VRAM).
- Dans `GpuModel::rollback_tokens(start, keep, toks)` :
  - Si `keep <= SNAP_MAX` (`1..=8` tokens gardés) : restauration instantanée par copie GPU-à-GPU (`dtod`) depuis le snapshot `keep - 1` (`< 0,3 ms`).
  - Si `keep > SNAP_MAX` (`9..=16` tokens gardés) : restauration du snapshot `SNAP_MAX - 1` suivie d'un unique passage forward batché sur les `keep - SNAP_MAX` tokens acceptés restants.
- Les 4 pipelines Vulkan supplémentaires utilisés par la vérification et le replay (`nt = 17` et `nt < 8`) sont pré-compilés lors du `preflight` au chargement du modèle (`src/vulkan/preflight.rs`), éliminant les `112 ms` de compilation JIT au premier pas spéculatif.

---

## 3. Résultats mesurés sur Radeon 890M (`gfx1150`)

Tous les tests sont exécutés de manière séquentielle (`qwen-ngram.service` arrêté, `10240/10240` experts résidents en VRAM).

### 3.1. Tableau récapitulatif des performances

| Scénario / Charge | Configuration | Prefill (tok/s) | Decode (tok/s) | Drafts acceptés | Speedup vs Baseline | Exactitude Greedy (`diff -u`) |
|---|---|---:|---:|---:|---:|---|
| **Baseline Strata (WP0)** | `v1.2.1` sans spec | 517,3 | 18,30 | — | 1,00× | Référence |
| **Strata optimisé (WP4)** | `--spec-type none` (copie/réécriture) | 506,9 | 19,30 | — | 1,05× | Référence (0 diff) |
| **Texte neuf (essai EUV)** | `--spec-type none` | 123,6 | 19,73 | — | 1,08× | Référence |
| **Texte neuf (essai EUV)** | `--spec-type ngram-simple --spec-draft-n-max 16` | 132,7 | **19,15 - 21,09** | 0/16 à 16/40 | **1,05× - 1,15× (zéro perte)** | **100 % bit-à-bit identique** |
| **Copie / Réécriture (chat)** | `--spec-type ngram-simple --spec-draft-n-max 16` | 506,0 | **81,70** | 466/480 (97,1 %) | **×4,46 vs WP0 (×4,23 vs WP4)** | **100 % bit-à-bit identique** |
| **Copie / Réécriture (raw)** | `--spec-type ngram-simple --spec-draft-n-max 16` | 514,1 | **81,94** | 480/480 (100,0 %) | **×4,48 vs WP0** | **100 % bit-à-bit identique** |
| **Copie + Cache statique/dyn** | `--spec-type ngram-map-k` + `--lookup-cache-static/dynamic` | 502,7 | **81,53** | 480/480 (100,0 %) | **×4,46 vs WP0** | **100 % bit-à-bit identique** |

### 3.2. Vérification de l'exclusion mutuelle `--mtp` / `--spec-type`
```text
$ ./target/release/rustingstrata generate --model $BLOB --gpu --backend vulkan --spec-type ngram-simple --mtp 1 --prompt "Test"
Error: --spec-type ngram-simple and --mtp 1 are mutually exclusive (combo causes severe regression; disable --mtp or set --spec-type none)
```

### 3.3. Tests unitaires
```text
$ cargo test --lib ngram
running 4 tests
test ngram::tests::ngram_cache_roundtrip_and_draft ... ok
test ngram::tests::ngram_map_key_and_k4v_draft ... ok
test ngram::tests::ngram_mod_drafts_and_resets_on_low_acceptance ... ok
test ngram::tests::ngram_simple_matches_repeated_sequence ... ok
test result: ok. 4 passed; 0 failed
```
