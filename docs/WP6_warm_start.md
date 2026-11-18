# WP6 — Warm Start (Persistance disque du KV cache et des états récurrents hybrides)

## 1. Objectif du Work Package 6

Les clients agentiques (comme OpenCode) renvoient l'intégralité de l'historique de conversation à chaque tour. En mémoire vive, `RustingStrata` dispose déjà d'un mécanisme `prefill_cached` / `Parked`, mais tout redémarrage du serveur ou exécution CLI repartait de zéro ( ~6 à 8 secondes de préfill pour 3K–4K tokens).

L'objectif de WP6 :
1. **Persister sur disque (`--prompt-cache <PATH>`)** à la fois le cache KV des 10 couches d'Attention (`kc`, `vc`) et les états récurrents hybrides des 30 couches `Gated DeltaNet` (`conv1d` + matrice d'état `s`).
2. **Ne re-préfiller que le delta** lors de la reprise d'une session ou du tour suivant, même après redémarrage complet du processus.
3. **Cible** : **TTFT < 2,0 s** sur session reprise (contre ~6–8 s à froid) avec **100 % d'accord greedy**.

---

## 2. Architecture de la persistance hybride (`STRATAPC` v1)

### 2.1. Taille compacte de l'état hybride `qwen35moe`
Grâce à l'architecture hybride `30 DeltaNet + 10 Attention` de `qwen35moe` :
- **30 couches `Mixer::Delta`** : chacune maintient un buffer `conv` (`98 304` octets) et une matrice récurrente `s` (`2 097 152` octets), soit **`65,86 Mo` fixes** quelle que soit la longueur du contexte !
- **10 couches `Mixer::Attn`** : chacune stocke `kc` et `vc` (`2 × 2 × 256` éléments par position), soit seulement **`20 Ko / token` en `f16`** (`62,3 Mo` pour 3 113 tokens, `80 Mo` pour 4 096 tokens) et **`10,6 Ko / token` en `q8`** (`42,5 Mo` pour 4 096 tokens).
- **Total sur disque** : **`123,6 Mo`** pour une session de 3 113 tokens (`~146 Mo` à 4 096 tokens).

### 2.2. Implémentation (`src/gpu_model.rs`, `src/gpu_model/prefill.rs`, `src/main.rs`)
- **Format binaire auto-vérifié (`STRATAPC` v1)** :
  - En-tête magique `b"STRATAPC"`, version `1`, et empreinte de configuration (`prompt_cache_fingerprint`) hachant la géométrie du modèle (`n_layer`, `n_embd`, `n_vocab`, `n_head`, `n_head_kv`, `head_dim`, `ssm_d_inner`, `ssm_d_state`) ainsi que le pas mémoire par token du KV cache (`b.len() / ctx`, distinguant automatiquement `--kv f16` et `--kv q8`).
  - Écriture atomique via fichier temporaire `.tmp` + `rename()` dans `GpuModel::save_prompt_cache` (`BufWriter` de 8 Mio).
  - Chargement au démarrage dans `GpuModel::load_prompt_cache` (`BufReader` de 8 Mio), peuplant directement la file `self.parked`.
- **Intégration transparente dans `serve` et `generate`** :
  - Dans `prefill_cached_with` (utilisé par `serve` et `chat`) : chaque avancée de l'état sauvegardé (`keep > from`) synchronise l'état GPU vers `self.parked` et le persiste sur disque (`sync_saved_to_prompt_cache`).
  - Dans `prefill` (utilisé par `generate`) : lorsque `--prompt-cache` est actif, `prefill` délègue automatiquement à `prefill_cached(tokens, 0, tokens.len() - 1)`, permettant aussi aux appels CLI successifs de réutiliser le préfixe persisté sur disque.

---

## 3. Résultats mesurés sur Radeon 890M (`gfx1150`)

Protocole : session agentique de `3 114` tokens au Tour 1, suivie d'une reprise exacte (processus neuf) puis d'un Tour 2 ajoutant la réponse de l'assistant et une nouvelle consigne utilisateur (`3 165` tokens au total, soit `3 113` tokens réutilisés + `52` tokens delta) dans un troisième processus neuf.

| Scénario | Tokens prompt | Tokens réutilisés | TTFT mesuré | Débit préfill effectif | Speedup TTFT | Accord Greedy (`diff -u`) |
|---|---:|---:|---:|---:|---:|---|
| **Tour 1 — Cold Start** (création du cache `123,6 Mo`) | 3 114 | 0 | **5 971,7 ms (5,97 s)** | 521,5 tok/s | 1,0× | Référence |
| **Tour 1 — Warm Start** (reprise exacte, processus neuf) | 3 114 | 3 113 / 3 114 | **84,4 ms (0,084 s)** | **36 892,8 tok/s** | **×70,8** | **100 % bit-à-bit identique** |
| **Tour 2 — Warm Start Multi-Tour** (delta 52 tok, processus neuf) | 3 165 | 3 113 / 3 165 | **605,6 ms (0,61 s)** | **5 226,4 tok/s** | **×9,9** | **100 % exact** |

La cible **TTFT < 2,0 s** est largement battue : **`0,084 s`** sur reprise directe et **`0,606 s`** sur tour suivant avec delta de conversation.
