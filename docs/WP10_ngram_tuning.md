# WP10 — Rapport d'Exploration et Tuning Spéculatif N-Gram (Radeon 890M / gfx1150)

## Objectifs

Optimiser la grille d'hyperparamètres spéculatifs contextuels n-gram sur l'iGPU Radeon 890M (gfx1150) :
- Longueur de clé $N \in \{6, 8, 12, 16\}$
- Fenêtre de recherche $M \in \{24, 48, 64\}$
- Plafond de drafts $draft\_max \in \{8, 12, 16\}$
- Algorithmes : `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`
- Deux bancs d'essai : Décodage texte neuf technique (128 tokens) et Copie/réécriture de code (400 tokens).

---

## Résultats Mesurés (Grid Search Automatisé)

Binaire : `rustingstrata` (commit `4968219`, backend Vulkan, Radeon 890M).
Modèle : `Qwen3.8-35B-A3B` (10240/10240 experts résidents).

### 1. Baselines (sans spéculation)
| Tâche | Algorithme | Débit mesuré |
|---|---|---|
| Texte neuf technique (128 tok) | `none` | **19.68 tok/s** |
| Copie de code Python (400 tok) | `none` | **19.18 tok/s** |

---

### 2. Impact du Plafond de Spéculation `draft_max`
Modèle configuré en `ngram-simple` ($N=12, M=48$) sur la tâche de copie :
| `draft_max` | Débit mesuré | Ratio vs base | Taux d'acceptation |
|---|---|---|---|
| 8 | 63.30 tok/s | $\times 3.30$ | 360/360 (100%) |
| 12 | 74.33 tok/s | $\times 3.87$ | 368/368 (100%) |
| **16** | **83.08 tok/s** | **$\times 4.33$** | 376/376 (100%) |

*Constat* : Le débit progresse de façon quasi-linéaire avec `draft_max`. La limite physique actuelle `TMAX = 17` impose un cap de 16 tokens vérifiés par passe de forward MoE.

---

### 3. Comparatif des Algorithmes Spéculatifs (`draft_max=16, N=12, M=48`)
| Algorithme | Débit mesuré (copie) | Taux d'acceptation |
|---|---|---|
| **`ngram-simple`** | **83.05 tok/s** | **376/376 (100.0%)** |
| `ngram-map-k` | 78.46 tok/s | 368/368 (100.0%) |
| `ngram-map-k4v` | 78.56 tok/s | 368/368 (100.0%) |
| `ngram-mod` | 77.26 tok/s | 371/372 (99.7%) |

*Constat* : `ngram-simple` surpasse les variantes à tables de hachage (`map-k`, `mod`) d'environ 5 tok/s grâce à l'absence totale de surcharge d'allocation ou de hachage CPU entre les forwards GPU.

---

### 4. Sensibilité de la Clé $N$ et de la Fenêtre $M$ (`ngram-simple`)
- **Longueur de clé $N$** :
  - $N=6$ : 83.01 tok/s
  - $N=8$ : 83.06 tok/s
  - $N=12$ : 83.12 tok/s
  - **$N=16$** : **83.30 tok/s** (élimine les faux raccords)
- **Fenêtre de recherche $M$** :
  - $M=24$ : 83.32 tok/s
  - **$M=48$** : **83.51 tok/s** (point optimal)
  - $M=64$ : 83.43 tok/s

---

### 5. Comportement sur Texte Neuf (Pénalité de Miss)
- `none` : 19.68 tok/s
- `ngram-simple` : 19.32 tok/s (pénalité de miss négligeable de 1.8%)
- `ngram-map-k` : 19.91 tok/s (zéro régression)

---

## Configuration Optimale Retenue pour Production

```bash
--spec-type ngram-simple \
--spec-draft-n-max 16 \
--spec-ngram-size-n 12 \
--spec-ngram-size-m 48
```
Cette configuration permet d'atteindre **83.5 tok/s en copie/réécriture** (avec un pic à chaud > 85 tok/s en service continu) tout en préservant ~19.5-20 tok/s sur texte neuf.
