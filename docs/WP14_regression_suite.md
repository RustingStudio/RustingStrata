# WP14 — Rapport de la Suite Complète d'Exactitude et Non-Régression (gfx1150)

## Résumé d'Exécution

Script : `scripts/route_c_bench.sh`
Date : 2026-11-11
Plateforme : AMD Radeon 890M Graphics (RADV STRIX1, RDNA 3.5, Linux CachyOS)
Binaire : `rustingstrata` (commit `7e797c6`, backend Vulkan)
Modèle cible : `Qwen3.8-35B-A3B` (13.65 GiB) + MTP standalone (1.99 GiB)

---

## Résultats des 5 Classes d'Évaluation

| Classe | Description | Vitesse mesurée | Exactitude Greedy | Statut |
|---|---|---|---|---|
| **Classe 1** | Texte neuf technique (128 tok) | 19.67 tok/s (base) / 19.22 tok/s (ngram) | **100.0% bit-à-bit** | **PASS** |
| **Classe 2** | Copie / Réécriture (500 tok) | **82.49 tok/s** (vs 19.29 tok/s base) | **100.0% bit-à-bit** | **PASS** |
| **Classe 3** | Session Agentique Multi-Tours | **37 800 tok/s** réutilisation préfixe | **100.0% bit-à-bit** | **PASS** |
| **Classe 4** | Tête MTP Autonome (Couche 40) | **24.37 tok/s** (70.8% drafts acceptés) | **100.0% bit-à-bit** | **PASS** |
| **Classe 5** | Contexte Étendu 65 536 KV Q8 | 19.05 tok/s (6.2 Go RAM dispo) | **100.0% bit-à-bit** | **PASS** |

### Taux de Concordance Global : **100.0% sur 5/5 classes**
Zéro dérive numérique, zéro corruption de tenseur, stabilité totale de l'arbre d'exécution sur Vulkan.
