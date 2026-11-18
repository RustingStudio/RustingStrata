# WP8 — Intégration quotidienne (Service systemd sur port 8011 & switch OpenCode)

## 1. Objectif du Work Package 8

Déployer le moteur `RustingStrata` optimisé (branche `route-c`, WP0–WP7) comme serveur quotidien OpenAI-compatible sur `127.0.0.1:8011`, géré par `systemd --user` (`strata-route-c.service`), avec :
1. **Décodage spéculatif multi-stratégie** : `--spec-type ngram-simple,ngram-map-k,ngram-cache --spec-draft-n-max 16`.
2. **Cache de scaffolds statique et dynamique** :
   - `--lookup-cache-static ~/.local/share/rusting_strata/scaffolds_static.txt` (amorcé avec les appels d'outils JSON d'OpenCode, chemins fréquents et blocs idiomatiques).
   - `--lookup-cache-dynamic ~/.local/share/rusting_strata/lookup_dynamic.bin` (enrichi et sauvegardé automatiquement à chaque requête).
3. **Warm Start persistant sur disque** :
   - `--prompt-cache ~/.local/share/rusting_strata/warm_start.pcache` (persistance disque des 30 états récurrents DeltaNet et des 10 caches KV d'Attention).
4. **Exclusivité GPU garantie** : directive `Conflicts=qwen-ngram.service` dans l'unité systemd pour éviter toute cohabitation accidentelle en VRAM avec le serveur `llama-server` du port 8010.

---

## 2. Fichiers déployés

### 2.1. Unité systemd : `~/.config/systemd/user/strata-route-c.service`
```ini
[Unit]
Description=Qwen lane Route C (RustingStrata Vulkan gfx1150, ngram-spec x4.5 + Warm Start)
After=default.target
Conflicts=qwen-ngram.service

[Service]
ExecStart=/home/cvsbd/build/rustingstrata/target/release/rustingstrata serve --model /var/lib/ollama/blobs/sha256-0dae47840f8115d894e74c5cd1a6aca27e01feaa4f07890a50bdcf7db4c56227 --gpu --backend vulkan --addr 127.0.0.1:8011 --ctx 32768 --kv q8 --spec-type ngram-simple,ngram-map-k,ngram-cache --spec-draft-n-max 16 --lookup-cache-static /home/cvsbd/.local/share/rusting_strata/scaffolds_static.txt --lookup-cache-dynamic /home/cvsbd/.local/share/rusting_strata/lookup_dynamic.bin --prompt-cache /home/cvsbd/.local/share/rusting_strata/warm_start.pcache
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

### 2.2. Commandes de gestion du service
```bash
# Recharger les unités systemd user
systemctl --user daemon-reload

# Démarrer Route C (arrête automatiquement qwen-ngram grâce à Conflicts=)
systemctl --user start strata-route-c

# Vérifier l'état et les journaux en direct (tok/s prefill, decode et cache hits)
systemctl --user status strata-route-c
journalctl --user -u strata-route-c -f

# Activer au démarrage de la session utilisateur (optionnel)
systemctl --user enable strata-route-c

# Revenir à l'ancienne lane llama.cpp (port 8010) à tout moment
systemctl --user stop strata-route-c && systemctl --user start qwen-ngram
```

---

## 3. Instructions de switch du provider OpenCode (`~/.config/opencode/opencode.json`)

### Option A — Basculer le provider existant `lane-ngram` du port `8010` au port `8011`
Dans `~/.config/opencode/opencode.json`, remplacer `"baseURL": "http://127.0.0.1:8010/v1"` par `"baseURL": "http://127.0.0.1:8011/v1"` :
```json
    "lane-ngram": {
      "name": "Lane Route C (RustingStrata Vulkan gfx1150, ngram + Warm Start)",
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": {
        "baseURL": "http://127.0.0.1:8011/v1",
        "apiKey": "local",
        "headerTimeout": 1800000,
        "chunkTimeout": false
      },
```

### Option B — Ajouter un provider dédié `lane-route-c` dans `~/.config/opencode/opencode.json`
Vous pouvez conserver `lane-ngram` (port 8010) intact et ajouter un bloc `lane-route-c` (port 8011), puis définir `"model": "lane-route-c/qwen-uncensored"` en haut du fichier :
```json
    "lane-route-c": {
      "name": "Lane Route C (RustingStrata Vulkan, 19.7 tok/s neuf | 81.9 tok/s copie | Warm Start 84 ms)",
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": {
        "baseURL": "http://127.0.0.1:8011/v1",
        "apiKey": "local",
        "headerTimeout": 1800000,
        "chunkTimeout": false
      },
      "models": {
        "qwen-uncensored": {
          "name": "Qwen-Uncensored Route C (Strata Vulkan + ngram-spec + Warm Start)",
          "limit": {
            "context": 32768,
            "output": 16384
          },
          "modalities": {
            "input": ["text"],
            "output": ["text"]
          },
          "body": {
            "temperature": 0.5,
            "top_p": 0.95,
            "top_k": 20
          }
        }
      }
    }
```

---

## 4. Routes API supportées par `rustingstrata serve` (port 8011)

- `GET  http://127.0.0.1:8011/v1/models` : liste des modèles (compatible OpenAI).
- `POST http://127.0.0.1:8011/v1/chat/completions` : chat OpenAI-compatible (streaming SSE `stream: true` et non-streaming `stream: false`, `tools` / `tool_calls`, `usage.prompt_tokens_details.cached_tokens`).
- `POST http://127.0.0.1:8011/v1/messages` : API Anthropic Messages (compatible Claude Code).
- `GET  http://127.0.0.1:8011/api/stats` : télémétrie temps réel (VRAM utilisée, hit rate des 10 240 experts, drafts spéculatifs acceptés, tokens en cache).
