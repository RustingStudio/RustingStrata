# WP1 — Profiling du Décode RustingStrata sur Radeon 890M (gfx1150)

Date : 2026-11-11
Modèle : `qwen35moe` (34,66 B, 40 couches = 30 DeltaNet + 10 Attention, 100 % résident en VRAM : 10240/10240 experts)
Instrumentation : `STRATA_VK_PROF=1` avec mise à l'échelle exacte par `VkPhysicalDeviceLimits::timestampPeriod = 10.02 ns`, labellisation par type de quantification (`spec[1]`), mesure des gaps inter-dispatches (`_barrier_gap`) et chronométrage CPU/hôte dans `step()` / `verify_greedy()`.

---

## 1. Profil GPU complet par Kernel et Type de Quantification (48 tokens : `pp 16 + tg 32`, `2592,90 ms` soit `54,02 ms/token` = `18,5 tok/s`)

| Kernel (`type`) | Temps total (48 tok) | Temps / token | Part (%) | Appels / token | Temps / appel | Volume lu / token | Bande passante effective |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`gemv_q:Q3_K`** (dense `attn_qkv`, `ssm_out`, `attn_q/k/v`, `ssm_beta`) | 782,78 ms | **16,31 ms** | **30,2 %** | 120 | 135,90 µs | 406,35 Mo | **24,9 Go/s** ⚠️ |
| **`gemv_q:Q6_K`** (`output.weight` 417,2 Mo + `attn_output` 68,8 Mo) | 601,51 ms | **12,53 ms** | **23,2 %** | 11 | 1139,22 µs | 485,99 Mo | **38,8 Go/s** ⚠️ |
| **`moe_gemv_q:Q3_K`** (`ffn_gate/up_exps`, 20 couches Q3_K, top-8) | 216,54 ms | **4,51 ms** | **8,4 %** | 20 | 225,56 µs | 144,18 Mo | **32,0 Go/s** |
| **`moe_gemv_q:IQ3_XXS`** (`ffn_gate/up_exps`, 20 couches IQ3_XXS, top-8) | 172,02 ms | **3,58 ms** | **6,6 %** | 20 | 179,18 µs | 128,45 Mo | **35,8 Go/s** |
| **`deltanet`** (récurrence d'état SSM/DeltaNet, 30 couches) | 160,70 ms | **3,35 ms** | **6,2 %** | 30 | 111,60 µs | État en registres/VRAM | — |
| **`gemv_q:Q8_0`** (`blk.L.attn_gate.weight`, 30 couches) | 157,67 ms | **3,28 ms** | **6,1 %** | 30 | 109,49 µs | 267,39 Mo | **81,5 Go/s** ✅ |
| **`moe_down_combine:Q3_K`** (`ffn_down_exps`, 20 couches Q3_K, top-8) | 122,28 ms | **2,55 ms** | **4,7 %** | 20 | 127,37 µs | 72,09 Mo | **28,3 Go/s** |
| **`gemv_f`** (`ffn_gate_inp` routeur F32 83,9 Mo + `ssm_alpha` 7,9 Mo) | 117,45 ms | **2,45 ms** | **4,5 %** | 110 | 22,24 µs | 91,75 Mo | **37,4 Go/s** |
| **`moe_down_combine:IQ3_XXS`** (`ffn_down_exps`, 20 couches IQ3_XXS, top-8) | 98,56 ms | **2,05 ms** | **3,8 %** | 20 | 102,67 µs | 64,23 Mo | **31,3 Go/s** |
| **`gemv_q:Q5_K`** (expert partagé `gate/up/down_shexp`, 40 couches) | 85,61 ms | **1,78 ms** | **3,3 %** | 120 | 14,86 µs | 86,52 Mo | **48,6 Go/s** |
| **`swiglu_q`** + **`conv_silu`** + **`rms_norm`** + **`moe_route`** + **`norm_gate`** | 66,52 ms | **1,39 ms** | **2,6 %** | 280 | 4,95 µs | Activations | — |
| **`_barrier_gap`** (barrières inter-dispatches GPU) | 10,10 ms | **0,21 ms** | **0,4 %** | 802 | 0,26 µs | — | — |
| **`attn_decode`** + **`attn_merge`** + **`argmax_rows`** | 11,25 ms | **0,23 ms** | **0,4 %** | 21 | 11,10 µs | KV cache | — |
| **TOTAL / TOKEN** | **2592,90 ms** | **54,02 ms** | **100,0 %** | **802** | — | **1657,0 Mo/tok** | **30,7 Go/s moy.** |

---

## 2. Profil Hôte / CPU avec Graphe Vulkan actif (`STRATA_VK_PROF=1` sans `STRATA_NO_GRAPH`)

Sur un step de décode en régime permanent (`~53,8 ms/token` = `18,6 tok/s`) :
- **`embed_htod`** (`put_embed` + `step_buf` via `cmd_update_buffer`) : `56–76 µs` en régime normal, mais **pics périodiques de `2000–3200 µs` (`2–3,2 ms`)** lorsque l'allocateur/command buffer ou le worker CPU interagit avec la mémoire.
- **`pass_submit`** (soumission #1 du graphe pré-enregistré `[cur_cb, graph_cb]`) : `58–85 µs`.
- **`gpu_wait_dtoh`** (soumission #2 `dtoh` + `vkWaitForFences`) : `52,6–53,8 ms` (temps d'exécution GPU).
- **`after_experts`** (lecture synchrone de 40 mailboxes `BAR_AMD` non-cachées sur le thread principal + `cache.adapt()` tous les 8 tokens) : `20–64 µs` par token normal, **pic à `935 µs` (`0,94 ms`) tous les 8 tokens** lors de `cache.adapt()` sur les 10 240 entrées, alors que 100 % des experts sont résidents !
- **Écart Graph vs No-Graph** : `18,3–18,8 tok/s` avec graphe contre `16,4–17,3 tok/s` sans graphe (`-8 à -10 %`), confirmant que le command-buffer replay (`Vk::graph`) économise déjà `~4 ms/token` de soumission CPU, mais laisse encore 2 `vkQueueSubmit`/token, 40 barrières `conditional_rendering` par token et le trafic mailbox `BAR_AMD` actif.

---

## 3. Top 3 des Pertes Chiffrées (Cibles WP2, WP3, WP4)

### Perte #1 : Déquantification `Q3_K` et `Q6_K` dans `deq.glsl` (`28,84 ms/token` + `7,06 ms/token` MoE = **66,5 % du temps total**)
- **Preuve expérimentale irréfutable** : dans le même step de décode, le kernel `gemv_q:Q8_0` (`blk.L.attn_gate.weight`, 267,39 Mo) atteint **81,5 Go/s** (`3,28 ms/token`), prouvant que le contrôleur mémoire LPDDR5X de la Radeon 890M délivre plus de 80 Go/s en GEMV Vulkan.
- À l'inverse, `gemv_q:Q3_K` (406,35 Mo) plafonne à **24,9 Go/s** (`16,31 ms/token`) et `gemv_q:Q6_K` (485,99 Mo) à **38,8 Go/s** (`12,53 ms/token`).
- **Causes dans `src/vulkan/shaders/deq.glsl`** :
  1. **Aller-retour `unpack` + `pack4` inutile dans `qw()` (`deq.glsl:200–204, 252–254`)** : pour tous les types `WORDS` (`Q3_K`, `Q6_K`, `Q5_K`, `Q8_0`), `qw_k()` calcule déjà `lo` et `hi` packés en 4×int8 (`vsub4`), puis `qi8()` les dépacke en `int q[8]` (16 shifts) et `qw()` les repacke aussitôt via `pack4()` (14 opérations bit-à-bit).
  2. **Amplification massive des lectures dans `ld8()` (`deq.glsl:61–67`) et redondance inter-lanes** : pour `Q3_K` (`T=11`, aligné sur 2 octets) et `Q6_K` (`T=14`, aligné sur 2 octets), `A8` et `A4` valent `false`. Chaque appel à `ld8()` exécute **3 chargements 32 bits (`w0, w1, w2`) + deux shifts 64 bits (`packUint2x32 >> sh`)**, et chaque lane appelle `ld8()` deux fois (`h` et `v`), alors que 4 à 8 lanes du même wave32 lisent exactement les mêmes octets (`i0 = (sub & 3) * 8`) !

### Perte #2 : `IQ3_XXS` (`moe_gemv_q` + `moe_down_combine` = `5,63 ms/token`, 10,4 % du temps) sans table LDS (`SHARED_TABLES`)
- Dans `deq.glsl:9–16, 68–75, 181`, `SHARED_TABLES` charge les tables en mémoire partagée (LDS) pour `T = 20, 21, 22, 23`, mais **oublie `T = 18` (`IQ3_XXS`)**, alors que `iq3xxs_grid` ne fait que 256 `uint` (1 Ko) ! De plus, `moe_down_combine.comp` omet `#define SHARED_TABLES`. Chaque thread effectue donc 2 lectures globales 64 bits divergentes par super-bloc dans `Tab32` au lieu d'une lecture LDS en 1 cycle.

### Perte #3 : `fast_path` MoE désactivé sur Vulkan + double `vkQueueSubmit` + 40 barrières `conditional_rendering` (`~2,5–4 ms/token` de temps mort)
- `Vk` n'implémente pas `fn moe_mailbox_optional(&self) -> bool { true }` car `moe_route.comp` ne teste pas `uint64_t(p.mb) != 0`. Résultat : malgré 10 240/10 240 experts résidents, chaque token effectue 40 écritures GPU non-cachées (`BAR_AMD`) + barrières device-scope, 40 `skip_rest()` (`cmd_end/begin_conditional_rendering_ext` + barrière pipeline), réveille le thread CPU `strata-cpu-experts`, et lit 40 fois la mailbox non-cachée sur le thread principal + `cache.adapt()` (`0,94 ms` tous les 8 tokens).
