# WP12 — Prise et Interface pour Modèles de Draft Externes (Route C)

## Objectif

Préparer RustingStrata à brancher sans modification structurelle un futur modèle de draft spéculatif dédié (de type DFlash-2, EAGLE-3 ou Medusa/GLM-spec) dès qu'un poids entraîné sur le vocabulaire 248 320 tokens de Qwen3.8-35B-A3B sera disponible dans la communauté.

---

## Spécification de l'Interface

L'architecture spéculative unifiée repose sur trois composants :

1. **Prise CLI et Résolution** :
   ```bash
   rustingstrata ... --draft-model <PATH_TO_GGUF>
   ```
   Le drapeau `--draft-model` vérifie l'intégrité du fichier GGUF candidat au démarrage du serveur ou de la session d'inférence, inspecte ses tenseurs d'embedding et son vocabulaire.

2. **Cycle de Vie Unifié du Drafter Spéculatif** :
   Dans `src/gpu_model.rs`, tout drafter implémente le cycle de spéculation en trois phases :
   - **`propose(history: &[u32], t: u32, max_k: usize) -> Vec<u32>`** : Produit jusqu'à $k$ tokens candidats sans toucher aux KV caches profonds du modèle cible.
   - **`verify_greedy(candidates: &[u32]) -> (usize, u32)`** : Une passe de forward unique GPU vérifiant l'ensemble des $k$ activations en une seule lecture des poids MoE.
   - **`feedback(accepted: &[u32])`** : Met à jour l'état récurrent ou l'arbre de recherche du drafter.

3. **Compatibilité Mémoire VRAM** :
   Sur l'iGPU Radeon 890M, un petit modèle de draft (ex: 1B dense ou 4 couches distillées) requiert entre 600 Mo et 1.2 Go. Étant donné que le modèle 35B MoE + KV 4K n'occupe que ~14.5 Go sur les 24 Go alloués à l'iGPU, l'espace mémoire est entièrement garanti.
