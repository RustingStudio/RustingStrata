# WP13 — Rapport d'Extension de Contexte à 65 536 Tokens & Sécurité RAM (Radeon 890M / gfx1150)

## Objectif

Étendre la fenêtre de contexte de RustingStrata de 32 768 à **65 536 tokens** natifs avec quantification KV `q8` sur l'iGPU AMD Radeon 890M (27 Go de RAM partagée), en garantissant :
1. Une marge de sécurité RAM résiduelle $\ge$ 3.5 Go libres (protection anti-OOM).
2. La conservation intégrale du hit-rate expert (10240/10240 experts résidents).
3. Le fonctionnement transparent du Warm Start persistant (`--prompt-cache`).

---

## Mesures et Validation Mémoire

Commande exécutée :
```bash
rustingstrata generate \
  --model /var/lib/ollama/blobs/sha256-0dae47840f8115d894e74c5cd1a6aca27e01feaa4f07890a50bdcf7db4c56227 \
  --gpu --backend vulkan --ctx 65536 --kv q8 \
  --spec-type ngram-simple --spec-draft-n-max 16 \
  --prompt-cache /tmp/opencode/test_65k.pcache
```

### 1. Dimensionnement VRAM / GPU Pool
- **Empreinte Poids + State + Scratch** : 2 613 Mo (dont KV Cache : 680 Mo pour 65 536 tokens en `q8`).
- **Expert Slots** : 12 480 Mo réservés.
- **Experts résidents** : **10240 / 10240 (100.0%)** — Zéro expert déchargé sur CPU ou SSD.
- **Trafic Mailbox** : 0 écritures GPU, 0 lectures de routage host (fast path intégralement actif).

### 2. Monitoring RAM Système Continu
- **RAM disponible avant démarrage** : 19 077 Mo.
- **RAM minimale disponible au pic d'inférence** : **6 249 Mo libres** (très supérieur au seuil de sécurité exigé de 3 500 Mo).
- **RAM après fermeture** : 19 036 Mo (zéro fuite mémoire résiduelle).

### 3. Validation Warm Start à 65k Contexte
- **Préfill initial (Cold Start)** : 398.7 tok/s.
- **Reprise Warm Start (`.pcache`)** :
  - Tokens réutilisés : 654 / 655 tokens (99.85%).
  - **TTFT (Time To First Token) : 73.2 ms** ($\le 100\text{ ms}$ requis).
  - Vitesse de préfill effectif perçu : **8 947.6 tok/s**.
  - Exactitude : 100% bit-à-bit.
