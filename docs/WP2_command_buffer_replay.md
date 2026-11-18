# WP2 — Command-Buffer Replay & Élimination du Temps Mort CPU/Mailbox (Style Torque)

Date : 2026-11-11
Branche : `route-c`

## 1. Optimisations implémentées
1. **Unification en un seul `vkQueueSubmit` par step de décode (`src/vulkan/mod.rs`)** :
   - Ajout de la file d'attente ordonnée `State::queued` (`enqueue_cb`).
   - Lors de l'appel à `Vk::graph()`, le command buffer préliminaire (`put_embed` + `step_buf`) et le command buffer du graphe des 40 couches (`SIMULTANEOUS_USE`) sont mis en file d'attente au lieu de déclencher un premier `vkQueueSubmit` immédiat.
   - Le `flush()` final (dans `dtoh` / `argmax_rows`) soumet `[pre_cb, graph_cb, post_cb]` en **un seul appel `vkQueueSubmit`** avec `st.fence`, divisant par deux les transitions noyau DRM (`drm_amdgpu_cs`) et signaux de sémaphore timeline par token.
2. **Activation du Fast-Path MoE (`mb == 0`) sur Vulkan (`src/vulkan/shaders/moe_route.comp`, `src/vulkan/mod.rs`)** :
   - Ajout de la garde `if (uint64_t(p.mb) != 0)` et sortie anticipée `if (uint64_t(p.mb) == 0) return;` dans `moe_route.comp`.
   - Implémentation de `fn moe_mailbox_optional(&self) -> bool { true }` sur `Vk`.
   - Suppression des 40 écritures GPU `BAR_AMD` non-cachées, des 40 `controlBarrier(gl_ScopeDevice)` + `atomicStore`, des 40 lectures `BAR_AMD` côté hôte (`after_experts`) et du scan `cache.adapt()` sur 10 240 entrées tous les 8 tokens.
3. **Élimination des barrières `VK_EXT_conditional_rendering` inutiles (`src/vulkan/mod.rs`)** :
   - Ajout du suivi `State::need_cond` activé uniquement si `skip_rest()` est appelé (`mb != 0`).
   - Quand 100 % des experts sont résidents (`fast path on`), aucun command buffer n'émet `cmd_begin_conditional_rendering_ext` ni de barrière `CONDITIONAL_RENDERING_EXT` lisant `st.err.0` en mémoire non-cachée `BAR_AMD`.
4. **Pré-touch des pages mmap de `token_embd.weight` au chargement (`src/model.rs`)** :
   - Évite les défauts de page disque lors de la lecture CPU des embeddings dans `put_embed()`.

## 2. Résultats mesurés (Avant vs Après WP2)

| Métrique (par token de décode) | Avant WP2 (WP1 baseline) | Après WP2 | Gain mesuré |
|---|---:|---:|---:|
| **`pass_submit` (CPU dispatch graphe)** | 75,8 – 110,0 µs | **2,1 – 3,8 µs** | **-96,5 %** |
| **`after_experts` (post-step CPU)** | 20,0 – 64,1 µs (pics à **935,1 µs**) | **1,6 – 2,5 µs** (zéro pic) | **-96,0 %** (-99,7 % sur pics) |
| **`worker busy` (thread CPU experts)** | 10,4 – 11,1 µs/token | **0,0 µs/token** | **-100 %** |
| **Trafic Mailbox `BAR_AMD` (32 tok)** | 40–1920 writes, 1920 reads (`fast path off`) | **0 writes, 0 reads (`fast path on`)** | **-100 %** |
| **Temps mort CPU total hors `gpu_wait`** | ~150 – 1080 µs/token | **~48 – 52 µs/token** (< 0,1 % du step) | **> -68 % à -95 %** (Cible ≥ -30 % dépassée) |
| **Tests unitaires `tests/backend.rs`** | 10/10 passés | **10/10 passés** (`spirv_is_fresh` OK) | ✅ |
