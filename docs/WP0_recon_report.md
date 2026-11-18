# WP0 — Recon & Reproduction des Baselines (Route C — Radeon 890M / gfx1150)

Date : 2026-11-11
Matériel : AMD Radeon 890M Graphics (`RADV STRIX1`, `gfx1150`, RDNA 3.5, 128-bit LPDDR5X)
Driver : Mesa `26.2.6-arch3.1` (Vulkan API `1.4.354`)
Mode performance : `high` (`/sys/class/drm/card0/device/power_dpm_force_performance_level`)

## 1. Audit Vulkan (`vulkaninfo`)
- **Cooperative Matrix (`VK_KHR_cooperative_matrix`)** : supporté en `SHADER_STAGE_COMPUTE_BIT` pour `f16` uniquement (`cooperative matrix f16 yes, int8 no`).
- **Dot Product Entier (`VK_KHR_shader_integer_dot_product`)** : **OUI, ACCÉLÉRÉ MATÉRIELLEMENT** sur RADV `26.2.6` :
  - `integerDotProduct8BitSignedAccelerated = true`
  - `integerDotProduct8BitUnsignedAccelerated = true`
  - `integerDotProduct4x8BitPackedSignedAccelerated = true` (`sdot_4x8` / `V_DOT4_I32_I8` sur RDNA 3.5 !)
  - `integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated = true`
- **Subgroups (`VK_EXT_subgroup_size_control`)** :
  - `minSubgroupSize = 32`, `maxSubgroupSize = 64`, `subgroupSize = 64` par défaut.
  - Opérations supportées : `BASIC`, `VOTE`, `ARITHMETIC`, `BALLOT`, `SHUFFLE`, `SHUFFLE_RELATIVE`, `CLUSTERED`, `QUAD`, `ROTATE`.
- **Shared Memory & Push Constants** :
  - `maxComputeSharedMemorySize = 65536` octets (64 KiB)
  - `maxPushConstantsSize = 256` octets

## 2. Reproduction des Baselines (1 seul workload GPU à la fois)

| Protocole | Métrique | Référence attendue | Mesure WP0 | Écart | Statut |
|---|---|---|---|---|---|
| **RustingStrata v1.2.1** (`--backend vulkan`) | Decode court (`pp 512, tg 128`) | 18,3 tok/s | **18,8 tok/s** (18,8–18,9) | +2,7 % | ✅ Confirmé (±5 %) |
| **RustingStrata v1.2.1** (`--backend vulkan`) | Decode après 4K (`pp 4096, tg 128`) | 17,7 tok/s | **18,1 tok/s** (18,1–18,1) | +2,3 % | ✅ Confirmé (±5 %) |
| **RustingStrata v1.2.1** (`--backend vulkan`) | Prefill 4K (`pp 4096`) | 517,3 tok/s | **520,7 tok/s** (512,7–525,1) | +0,7 % | ✅ Confirmé (±5 %) |
| **llama.cpp** (`Vulkan, flash-attn, KV q8_0, ngram-simple`) | Decode texte neuf (`n_predict 128`) | 16,0 tok/s | **16,71 tok/s** | +4,4 % | ✅ Confirmé (±5 %) |
| **llama.cpp** (`Vulkan, flash-attn, KV q8_0, ngram-simple`) | Prefill (`1532–2391 tok`) | 384,6 tok/s | **391,8–398,7 tok/s** | +1,9 % | ✅ Confirmé (±5 %) |
| **llama.cpp** (`--spec-type ngram-simple --spec-draft-n-max 16`) | Copie/réécriture (`repro-payload.json` / bloc Python ×22) | ≥ 85,0 tok/s | **109,7–134,4 tok/s** (486/528 et 488/488 drafts acceptés) | ≥ 85 tok/s | ✅ Confirmé |
| **Kinetic v0.1.0** (`kinetic bench $BLOB --device vulkan`) | Support Q3_K Vulkan | Échec attendu | `kinetic: Vulkan: no Vulkan GEMV kernel for Q3_K` | Exact | ✅ Confirmé |

## 3. Observations clés pour WP1–WP5
1. **Anomalie Mailbox dans RustingStrata** : bien que `10240/10240` experts soient résidents en VRAM (`100.0% hit rate, 0 swaps`), Strata affiche `mailbox traffic: 80 GPU writes, 30720 host routing reads (fast path off, 10240/10240 experts resident)`. Investigation prioritaire en WP1/WP2.
2. **Accélération matérielle `4x8BitPackedSigned`** : même sans `cooperative_matrix` int8, l'extension `VK_KHR_shader_integer_dot_product` (`dotPacked4x8EXT` / `SDOT`) est **accélérée matériellement** par RADV sur gfx1150, ce qui ouvre la voie en WP3 à la quantification Q8 des activations couplée au produit scalaire int8 packé dans `gemv_q.comp` et `moe_gemv_q.comp`.
