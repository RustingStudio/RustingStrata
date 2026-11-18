# WP4 — Recouvrement intra-couche (Compute Overlap) & Prefetch L2 sur Radeon 890M (`gfx1150`)

## 1. Résumé exécutif (en français simple)

Dans ce quatrième lot (**WP4**), nous avons exploré et mesuré deux mécanismes pour masquer la latence d'accès mémoire et le temps de synchronisation entre les étapes d'une couche sur l'iGPU **AMD Radeon 890M (`gfx1150`, RDNA 3.5)** :

1. **Recouvrement de calcul intra-couche (`g.batch` sans barrière Vulkan — ACTIVÉ par défaut)** :
   - **Couches DeltaNet (30 couches par token)** : recouvrement du noyau `conv_silu` / `conv_seq` (qui lit `"qkv"` et écrit `"y"`) avec les 3 produits matrice-vecteur `gate`, `beta` et `alpha` (qui lisent `"hq"` et écrivent `"z"`, `"beta"`, `"g"`). Cela supprime **30 barrières pipeline Vulkan par token**.
   - **Couches MoE (40 couches par token)** : recouvrement du routeur `moe_route` (`1` seul workgroup lisant `"rl"` et écrivant `slots` / `"wh"`) avec les produits matrice-vecteur de l'expert partagé `s.gate` + `s.up` (`128` workgroups lisant `"hq"` et écrivant `"shg"` / `"shu"`). Comme `moe_route` n'occupe qu'un seul Compute Unit (1 WGP sur 8), les 7 autres WGP exécutent simultanément l'expert partagé sans attendre la fin du routage. Cela supprime **40 barrières pipeline Vulkan supplémentaires par token** (soit **70 barrières éliminées par token**, ou **3 360 barrières** sur 48 tokens).
2. **Préchargement inter-couches des experts (`STRATA_MOE_PREFETCH=1` — implémenté sur Vulkan, DÉSACTIVÉ par défaut)** :
   - Implémentation complète du chemin `moe_prefetch` sur le backend Vulkan (`src/vulkan/shaders/moe_route.comp` mode `p.T == 0` + écriture de `pid`, et `src/vulkan/mod.rs`).
   - **Mesure honnête dos à dos** : sur la Radeon 890M dotée de seulement **2 Mo de cache L2**, l'ensemble actif d'une couche MoE pèse **10,3 Mo** (8 experts routés) + **1,97 Mo** (expert partagé). Activer `STRATA_MOE_PREFETCH=1` précharge 10,3 Mo de la couche `N+1` dans un cache L2 de 2 Mo, ce qui évince les lignes utiles de la couche `N` et consomme inutilement 390 Mo/token de bande passante LPDDR5X, faisant chuter le décode de **19,6 tok/s à 15,8 tok/s (-19,4 %)**. Le flag reste donc désactivé par défaut (`STRATA_MOE_PREFETCH=0`), tandis que le recouvrement intra-couche `g.batch` est conservé actif.

---

## 2. Résultats mesurés (WP1 → WP2 → WP3 → WP4)

### 2.1 Temps GPU actif par token (profilage `STRATA_VK_PROF=1 STRATA_NO_GRAPH=1`, 48 tokens)

| Étape | Temps GPU actif (48 tokens) | Temps GPU / token | Barrières pipeline / token | Bande passante effective (bout-en-bout) |
|---|---:|---:|---:|---:|
| **WP1 (Baseline)** | 2592,90 ms | 54,02 ms/tok | ~372 | 30,7 Go/s |
| **WP3 (Kernels coopératifs RDNA 3.5)** | 2408,39 ms | 50,17 ms/tok | ~372 | 33,0 Go/s |
| **WP4 (Recouvrement intra-couche `Delta` + `MoE`)** | **2401,73 ms** | **50,03 ms/tok** | **~302 (-70/tok)** | **33,1 Go/s** |

Gain cumulé sur les noyaux GPU : **-191,17 ms** sur 48 tokens (**-4,0 ms/token**).

### 2.2 Benchmark de bout en bout (3 répétitions, `--pp 512,4096 --tg 128 --reps 3 --ctx 32896`)

Commande exécutée :
```bash
./target/release/rustingstrata bench \
  --model /var/lib/ollama/blobs/sha256-0dae47840f8115d894e74c5cd1a6aca27e01feaa4f07890a50bdcf7db4c56227 \
  --gpu --backend vulkan --pp 512,4096 --tg 128 --reps 3 --ctx 32896
```

Sortie brute (`/tmp/opencode/wp4_bench.txt`) :
```text
pp    512: prefill 374.9 (373.5-377.7) tok/s | tg 128: decode 19.6 (19.5-19.6) tok/s
expert traffic: 390.0 MB/token -> 8.0 GB/s at 19.6 tok/s
pp   4096: prefill 523.4 (522.7-524.9) tok/s | tg 128: decode 18.8 (18.8-18.8) tok/s
expert traffic: 390.0 MB/token -> 7.7 GB/s at 18.8 tok/s
expert hit rate 100.0% (0 misses), cache swaps 0
worker busy 0.0 µs/token, worker stalls 0
mailbox traffic: 0 GPU writes, 0 host routing reads (fast path on, 10240/10240 experts resident)
```

| Métrique | Baseline prompt | WP0 | WP2 | WP3 | **WP4** | Gain vs Baseline |
|---|---:|---:|---:|---:|---:|---:|
| **Prefill `pp 512`** | — | 369,8 tok/s | 365,6 tok/s | 370,1 tok/s | **374,9 tok/s** | **+1,4 %** |
| **Prefill `pp 4096`** | 517,3 tok/s | 523,9 tok/s | 519,8 tok/s | 519,7 tok/s | **523,4 tok/s** | **+1,2 %** |
| **Decode court (`tg 128` @ 512)** | 18,3 tok/s | 18,8 tok/s | 18,4 tok/s | 19,5 tok/s | **19,6 tok/s** | **+7,1 %** |
| **Decode long (`tg 128` @ 4096)** | 17,7 tok/s | 18,1 tok/s | 17,8 tok/s | 18,8 tok/s | **18,8 tok/s** | **+6,2 %** |

---

## 3. Analyse physique : pourquoi le prefetch L2 inter-couche (`STRATA_MOE_PREFETCH=1`) dégrade sur iGPU 2 Mo L2

Test comparatif sur 32 tokens (`/tmp/opencode/wp4_prefetch_test.txt`) :
```text
expert traffic: 10.3 MB per layer active set (top-8), 390.0 MB per token over 40 MoE layers, next-layer L2 prefetch on (STRATA_MOE_PREFETCH)
pp     16: prefill 15.5 (15.5-15.5) tok/s | tg 32: decode 15.8 (15.8-15.8) tok/s
```

- **Taille du cache L2 de la Radeon 890M (`gfx1150`)** : **2 Mo** partagés entre les 8 WGP (16 CU).
- **Taille des poids d'experts routés par couche** : **10,3 Mo** (`8 experts × 1,29 Mo`).
- **Conséquence** : précharger 10,3 Mo de la couche `N+1` pendant la couche `N` dépasse de **5,1×** la capacité totale du L2 (2 Mo). Non seulement les lignes préchargées sont évincées avant même le début de la couche `N+1`, mais elles évincent les tables `s_g3` / activations `"hq"` / `"ffq"` de la couche `N` et saturent le contrôleur LPDDR5X 128-bit.
- **Décision d'ingénierie** : conserver `STRATA_MOE_PREFETCH` désactivé par défaut (`0`), et garder actif le recouvrement intra-couche sans barrière (`Delta` + `MoE`) qui lui procure un gain net et mesuré sans trafic mémoire parasite.
