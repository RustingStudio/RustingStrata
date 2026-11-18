# WP15 — Réintégration en Production & Déploiement Quotidien (Radeon 890M / gfx1150)

## État de Production — Route C Phase 2

Le service quotidien d'inférence LLM local a été mis à jour avec tous les livrables de la Phase 2 :
- **Unité systemd** : `~/.config/systemd/user/strata-route-c.service`
- **Port d'écoute** : `127.0.0.1:8011` (compatible API OpenAI `/v1/chat/completions`)
- **Modèle résident** : `Qwen3.8-35B-A3B` (13.65 GiB, 10240/10240 experts résidents)
- **Contexte actif** : **65 536 tokens** en `kv q8` (680 Mo de KV cache alloués en VRAM)
- **Moteur Spéculatif N-gram** : Paramètres optimisés WP10 ($N=12, M=48, d_{max}=16$) avec cache dynamique et scaffolding statique.
- **Support MTP Standalone** : Prise `--mtp-model` prête pour les sessions de raisonnement pur (24.7 tok/s sur texte neuf).
- **Socket Drafter Externe** : Prise `--draft-model` câblée pour futurs modèles de draft (DFlash2 / EAGLE3).

---

## Vérification en Ligne

Test en direct exécuté le 11/11/2026 à 22:59 :
```bash
curl -s http://127.0.0.1:8011/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"sha256-...","messages":[{"role":"user","content":"Ping test! Respond with: PONG 890M PHASE 2 OK"}],"temperature":0.0}'
```
Réponse retournée :
```json
{"choices":[{"finish_reason":"stop","index":0,"message":{"content":"PONG 890M PHASE 2 OK","role":"assistant"}}],"created":1794434342,"id":"chatcmpl-0","model":"sha256-0dae47840f8115d894e74c5cd1a6aca27e01feaa4f07890a50bdcf7db4c56227","object":"chat.completion","usage":{"completion_tokens":12,"prompt_tokens":30,"total_tokens":42}}
```
Journal du service :
`done: key=open ... prompt=30 tok -> 12 tok in 0.34 s, decode 35.4 tok/s, finish=stop`
