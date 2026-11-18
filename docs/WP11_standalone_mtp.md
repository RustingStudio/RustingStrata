# WP11 — Rapport d'Implémentation et Mesures de la Tête MTP Autonome (Radeon 890M / gfx1150)

## Objectif & Architecture

Le modèle principal Qwen3.8-35B-A3B (`qwen35moe`) ne dispose pas de tête MTP embarquée dans son fichier GGUF principal (`n_nextn == 0`). Une tête autonome MTP Q8_0 de 1.99 Go (`~/models-local/mtp-Q8_0.gguf`) était dormante sur la machine.
L'objectif de WP11 était de permettre le chargement dynamique de cette tête externe sans modifier le blob GGUF d'origine, via le drapeau `--mtp-model <PATH>`.

### Modifications logicielles apportées :
1. **`src/weights.rs`** : Ajout du support multi-conteneurs `Weights::new_with_aux(g, aux_g)` pour interroger de manière transparente les tenseurs manquants dans le GGUF compagnon.
2. **`src/model.rs`** : Méthode `Model::load_with_aux`. Si la tête MTP autonome contient les 23 tenseurs de la couche 40 (`blk.40.*`), le modèle active automatiquement `cfg.n_nextn = 1`.
3. **`src/main.rs`** : Ajout du flag CLI `--mtp-model <PATH>`.
4. **VRAM Allocation** : Les 256 experts de la couche 40 (Q8_0, 2048x512) et ses projections d'attention sont intégrés directement dans le pool résident de l'iGPU (10496 experts résidents sur 10496, 0 swap, hit-rate 100%).

---

## Résultats Mesurés (Banc d'Essai Rigoureux)

Matériel : AMD Radeon 890M (iGPU Strix Point, gfx1150, RADV Mesa 26.2.6).
Binaire : `rustingstrata` (branch `route-c`, backend Vulkan).
Modèle : `Qwen3.8-35B-A3B` (13.65 GiB) + MTP standalone (1.99 GiB).

### 1. Texte Neuf Technique (128 tokens, essai physique EUV, greedy temp=0)
| Configuration | Débit mesuré | Taux d'acceptation | Gain vs Base |
|---|---|---|---|
| Baseline (`none`) | 19.64 tok/s | 0/0 (0%) | 1.00× |
| N-gram simple ($N=12, M=48$) | 19.04 tok/s | 0/16 (0.0%) | 0.97× (-3.0%) |
| **MTP seul ($k=1$)** | **23.71 tok/s** | **57/72 (79.2%)** | **1.21× (+20.7%)** |
| **MTP seul ($k=2$)** | **24.68 tok/s** | **75/106 (70.8%)** | **1.26× (+25.7%)** |

*Analyse* : Sur texte neuf technique, la tête MTP autonome réalise une percée en franchissant la barre des **24.7 tok/s** avec un taux d'acceptation remarquable de 70.8% à 79.2%. Le surcoût d'évaluation de la couche 40 est largement amorti par la validation de 2 tokens par forward des 40 couches MoE principales.

---

### 2. Copie / Réécriture de Code (400 tokens, greedy temp=0)
| Configuration | Débit mesuré | Taux d'acceptation | Gain vs Base |
|---|---|---|---|
| Baseline (`none`) | 19.28 tok/s | 0/0 (0%) | 1.00× |
| MTP seul ($k=1$) | 25.82 tok/s | 200/200 (100%) | 1.34× (+33.9%) |
| MTP seul ($k=2$) | 29.20 tok/s | 268/268 (100%) | 1.51× (+51.5%) |
| **N-gram simple ($d_{max}=16$)** | **82.38 tok/s** | **376/376 (100.0%)** | **4.27× (+327%)** |

---

## Conclusion & Stratégie de Déploiement

1. **Complémentarité parfaite des deux moteurs** :
   - Pour la **génération de texte neuf et le raisonnement agentique pur**, la tête MTP (`--mtp-model ~/models-local/mtp-Q8_0.gguf --mtp 2`) est reine avec **24.7 tok/s**.
   - Pour la **reprise de contexte, le scaffolding de fichiers et la réécriture de code**, le drafter N-gram (`--spec-type ngram-simple --spec-draft-n-max 16`) domine outrageusement avec **82.4 tok/s**.
2. **Protection Anti-Poison** :
   L'exclusion mutuelle stricte dans `Engine::runner` empêche le combo désastreux `MTP + Ngram` identifié lors de la phase 1 (qui chutait à 6.5 tok/s sous llama.cpp).
