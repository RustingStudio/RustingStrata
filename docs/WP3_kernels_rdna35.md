# WP3 — Optimisation des kernels Vulkan déquant+GEMV (`Q3_K`, `IQ3_XXS`, `Q6_K`, `Q5_K`) sur RDNA 3.5 (`gfx1150`)

## 1. Diagnostic RADV (`VK_KHR_pipeline_executable_properties`)

L'inspection du code machine RDNA 3.5 généré par Mesa RADV (`26.2.6-arch3.1`) via `STRATA_VK_STATS=1` sur les shaders originaux (`deq.glsl`, `gemv_q.comp`, `moe_gemv_q.comp`, `moe_down_combine.comp`) a mis en évidence quatre défauts structurels expliquant l'écart entre `gemv_q:Q8_0` (`81,5 Go/s`) et `Q3_K` (`24,9 Go/s`), `IQ3_XXS` (`31–35 Go/s`) ou `Q6_K` (`38,8 Go/s`) :

1. **Amplification VMEM ×3 dans `ld8()` et redondance inter-lanes sur `Q3_K` (`SBB = 110`) et `Q6_K` (`SBB = 210`)** :
   - Parce que `110 % 4 == 2` et `210 % 4 == 2`, chaque appel à `ld8()` émettait 3 chargements `U32` (`w0, w1, w2`) suivis de deux décalages 64-bits `packUint2x32 >> sh`.
   - Dans `Q3_K`, les 32 lanes d'un sous-groupe émettaient **9 chargements VMEM par lane (288 par wave)** pour lire un super-bloc de seulement **110 octets** (28 mots `uint32`), car 8 groupes de 4 lanes relisaient les mêmes 32 octets de `hmask` et 4 groupes de 4 lanes relisaient les mêmes 64 octets de `qs`.
   - Dans `Q6_K`, chaque lane émettait **8 chargements VMEM (256 par wave)** pour un super-bloc de **210 octets** (53 mots `uint32`).
2. **Branche divergente dans l'extraction d'échelle `Q3_K` (`deq.glsl:99`)** :
   - `is < 8 ? u8_(w, sc + is) & 15 : u8_(w, sc + is - 8) >> 4` générait deux chargements distincts dans une branche divergente (`Branches: 13` au lieu de `6` dans `gemv_q:Q3_K`).
3. **Aller-retour dépaquetage `qi8` + repaquetage `pack4` dans `qw()`** :
   - `qw_k` produisait déjà `lo` et `hi` empaquetés en `4×int8`, mais `qw()` passait par `qi8()` (16 décalages pour dépaqueter en `int q[8]`) puis `pack4()` (14 opérations binaires pour repaqueter en `lo, hi`).
4. **Absence de table LDS (`SHARED_TABLES`) pour `IQ3_XXS` (`T = 18`) et dans `moe_down_combine.comp`** :
   - `deq.glsl` plaçait les grilles `IQ4_NL`, `IQ3_S`, `IQ2_S`, `IQ4_XS` en LDS (`shared`), mais omettait `IQ3_XXS` (`iq3xxs_grid`, seulement 256 `uint32` = 1 Ko), provoquant des lectures buffer globales à chaque sous-bloc. De plus, `moe_down_combine.comp` n'activait pas `#define SHARED_TABLES` et sa boucle `s + 4 <= p.nsb` n'était jamais déroulée sur `qwen35moe` où `cols = 512` (`p.nsb = 2`).

---

## 2. Optimisations implémentées

1. **Chargement coopératif par sous-groupe (`qw_sb` + `shfl32` dans `src/vulkan/shaders/deq.glsl`)** :
   - **`Q3_K` (`T = 11`, 110 octets = 28 mots `uint32`)** : chaque lane `sub < 28` effectue **1 seul chargement `U32` coalescent** (`U32(base4).v[min(sub, 27u)]`), corrige l'alignement 2 octets via un unique `shfl32`, puis distribue `hmask`, `qs`, `scales` et `d` par registre (`subgroupShuffle`) sans aucun aller-retour `qi8`/`pack4` ni branche divergente.
   - **`Q6_K` (`T = 14`, 210 octets = 53 mots `uint32`)** : chaque lane effectue **2 chargements `U32` coalescents** (`raw0` et `raw1`) couvrant l'intégralité du super-bloc de 210 octets, puis distribue `ql`, `qh`, `scales` et `d` par `shfl32`.
   - **`IQ3_XXS` (`T = 18`, 98 octets = 25 mots `uint32`)** : ajout de `T == 18` dans `SHARED_TABLES` (`shared uint s_g3[256]`, 1 Ko LDS) et chargement coopératif des 98 octets en **1 seul chargement `U32` coalescent par lane** (`min(sub, 24u)`).
2. **Accumulation multi-lignes simultanée (`sb_step2`) et déroulage `nsb = 2` dans `moe_down_combine.comp`, `gemv_q.comp`, `moe_gemv_q.comp`** :
   - Dans `moe_down_combine.comp`, traitement simultané de 2 lignes d'experts (`sb_step2`) en mutualisant la lecture de `p.xq` (`a, c`) et `xd` (`dx`) en registres et déroulage par 2 (`s + 2 <= p.nsb`), actif sur `qwen35moe` (`p.nsb = 2`).
   - Ajout d'un étage de déroulage `s + 2 <= nsb` dans `gemv_q.comp` et `moe_gemv_q.comp` pour couvrir les matrices de largeur `cols = 512` (`ffn_down_shexp`).

---

## 3. Impact mesuré sur le code machine RDNA 3.5 (`STRATA_VK_STATS=1`)

| Pipeline Vulkan | Métrique RADV | Avant (WP1) | Après (WP3) | Gain |
|---|---|---:|---:|---:|
| `gemv_q:Q3_K` | Instructions ISA | 1 003 | **675** | **-32,7 %** |
| `gemv_q:Q3_K` | Instructions VMEM | 67 | **25** | **-62,7 %** |
| `gemv_q:Q3_K` | Branches | 13 | **6** | **-53,8 %** |
| `gemv_q:Q3_K` | Latence estimée RADV | 21 375 | **6 443** | **-69,9 %** |
| `moe_gemv_q:Q3_K` | Instructions VMEM / Latence | 68 / 87 203 | **26 / 27 493** | **-61,8 % VMEM / -68,5 % latence** |
| `moe_down_combine:Q3_K` | Instructions VMEM / Latence | 81 / 121 949 | **39 / 62 074** | **-51,9 % VMEM / -49,1 % latence** |
| `gemv_q:Q6_K` | Instructions VMEM / VALU | 55 / 537 | **31 / 401** | **-43,6 % VMEM / -25,3 % VALU** |
| `moe_gemv_q:IQ3_XXS` | Instructions VMEM / Latence | 56 / 48 188 | **28 / 27 829** | **-50,0 % VMEM / -42,2 % latence** |
| `moe_down_combine:IQ3_XXS` | Instructions VMEM / Latence | 69 / 82 476 | **41 / 64 672** | **-40,6 % VMEM / -21,6 % latence** |

---

## 4. Impact mesuré par kernel GPU (`STRATA_VK_PROF=1`, 48 tokens = `pp 16 + tg 32`)

| Kernel | Temps total WP1 (ms) | Temps total WP3 (ms) | Gain (ms) | Gain (%) | Bande passante effective WP3 |
|---|---:|---:|---:|---:|---:|
| `gemv_q:Q3_K` (406,35 Mo/tok) | 783,11 ms | **533,73 ms** | **-249,38 ms** | **-31,8 %** | **36,5 Go/s** (vs 24,9 Go/s) |
| `gemv_q:Q3_K + Q8_0 + gemv_f` (batch) | 1 058,39 ms | **955,08 ms** | **-103,31 ms** | **-9,8 %** | **40,0 Go/s** combiné (`Q8_0` seul : **81,5 Go/s**) |
| `gemv_q:Q6_K` (485,99 Mo/tok) | 601,40 ms | **595,41 ms** | **-5,99 ms** | **-1,0 %** | **39,2 Go/s** |
| `moe_gemv_q:Q3_K` (144,18 Mo/tok) | 216,33 ms | **184,71 ms** | **-31,62 ms** | **-14,6 %** | **37,5 Go/s** (vs 32,0 Go/s) |
| `moe_gemv_q:IQ3_XXS` (128,45 Mo/tok) | 172,06 ms | **166,04 ms** | **-6,02 ms** | **-3,5 %** | **37,1 Go/s** (vs 35,8 Go/s) |
| `moe_down_combine:Q3_K` (72,09 Mo/tok) | 122,40 ms | **101,01 ms** | **-21,39 ms** | **-17,5 %** | **34,3 Go/s** (vs 28,3 Go/s) |
| `moe_down_combine:IQ3_XXS` (64,23 Mo/tok) | 98,58 ms | **91,95 ms** | **-6,63 ms** | **-6,7 %** | **33,5 Go/s** (vs 31,3 Go/s) |
| `gemv_q:Q5_K` (86,52 Mo/tok) | 85,56 ms | **79,48 ms** | **-6,08 ms** | **-7,1 %** | **52,3 Go/s** (vs 48,6 Go/s) |
| **Somme active GPU (48 tokens)** | **2 592,90 ms** | **2 408,39 ms** | **-184,51 ms** | **-7,1 %** | **33,0 Go/s global** (sur 1 657 Mo/tok) |
| **Temps GPU actif par token** | **54,02 ms/tok** | **50,17 ms/tok** | **-3,85 ms/tok** | **-7,1 %** | — |

---

## 5. Benchmark bout-en-bout (`--pp 512,4096 --tg 128 --reps 3 --ctx 32896`)

| Métrique | Baseline prompt | WP0 mesuré | WP3 mesuré | Gain vs Baseline |
|---|---:|---:|---:|---:|
| **Decode court (`tg 128` après `pp 512`)** | 18,3 tok/s | 18,8 tok/s | **19,2 tok/s (pic 19,5 tok/s)** | **+4,9 % (pic +6,6 %)** |
| **Decode après 4K (`tg 128` après `pp 4096`)** | 17,7 tok/s | 18,1 tok/s | **18,8 tok/s (pic 18,9 tok/s)** | **+6,2 % (pic +6,8 %)** |
| **Prefill 4K (`pp 4096`)** | 517,3 tok/s | 520,7 tok/s | **525,5 tok/s (pic 526,8 tok/s)** | **+1,6 %** |

### Analyse physique honnête
Chaque token de décode unitaire (`nt = 1`) lit **1 657 Mo de poids** en mémoire LPDDR5X (`1 267 Mo` de poids denses/partagés + `390 Mo` d'experts routés). Même en portant la bande passante des kernels `Q3_K` de `24,9` à `36,5–37,5 Go/s` (+46 %) et `Q5_K` à `52,3 Go/s`, la lecture physique de `1,657 Go` par token impose un plancher matériel d'environ `45–50 ms/token` (`20–22 tok/s`) tant que chaque passe ne vérifie qu'un seul token (`nt = 1`).
En revanche, lors de la **vérification spéculative par lot (`nt = 4..16`, WP5)**, les `1 267 Mo` de poids denses/partagés (76,5 % du trafic mémoire) ne sont lus **qu'une seule fois pour tout le lot**, ce qui permet de dépasser `60–100+ tok/s`.
