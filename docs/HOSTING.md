# Hosting Strata for other people

This page is for running `strata serve` as a service that other people or programs use over the network, for
example to sell API access to a model. It covers keys, TLS through a reverse proxy, a systemd unit, health checks,
the usage log for billing, and how far one server goes.

The [license](../LICENSE.md) allows commercial services built on Strata: you can charge for access to a model that
Strata serves. Keep the license and the credit to the original author. You may not sell Strata itself. The model has
its own license (most Qwen models are Apache 2.0; check the model card).

## How much one server does

`strata serve` answers one chat request at a time. Others wait in a queue (`--max-waiting`, 16 by default) and run in
arrival order; more than that get `503 Service Unavailable` with `Retry-After: 2`. One request decodes at the speed in
the README (about 113 tok/s for Qwen3.6-35B-A3B Q2_K_XL on an RTX 3060), so that is also the total for all users of
one server. Long prompts block the queue for their prefill time (about 2 s for 4K tokens on the 3060).

For more users, run one server per GPU and put a load balancer in front (see [Several GPUs](#several-gpus)).

## Quick start

On a Linux machine with an NVIDIA driver (or a Vulkan 1.3 driver):

```sh
# 1. The binary: a release binary, or build it on this machine (.cargo/config.toml builds for the local CPU).
cargo build --release
sudo install -m 755 target/release/strata /usr/local/bin/strata

# 2. A model, and the calibration for this machine (once per model and machine).
M=/srv/models/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf
strata calibrate --model $M

# 3. Keys: one admin key, and a file of customer keys.
openssl rand -hex 24                            # run once per key
sudo mkdir -p /etc/strata
sudo tee /etc/strata/keys.txt >/dev/null <<'EOF'
# name     key
acme       3f9c0d6e5b...
beta-corp  a81e44f2c7...
EOF
sudo chmod 600 /etc/strata/keys.txt

# 4. Run it on the loopback address; the reverse proxy below is the only thing that faces the network.
STRATA_API_KEY=<admin key> strata serve --model $M --gpu --ctx 32768 --addr 127.0.0.1:8080 \
    --api-keys /etc/strata/keys.txt --usage-log /var/lib/strata/usage.jsonl --max-tokens 4096
```

Clients then use the OpenAI address `https://api.example.com/v1` with their key, or the Anthropic address
`https://api.example.com` (`POST /v1/messages`).

## Keys

- **Admin key** (`--api-key`, or `STRATA_API_KEY` to keep it out of the process list). It can use every route: the
  chat APIs, the chat page's stats, loading another model (`/api/load`) and downloading models (`/api/download`).
  Keep it for yourself.
- **Customer keys** (`--api-keys FILE`). One `name key` pair a line, `#` starts a comment. Keys must be at least 16
  characters, and names and keys must not repeat. A customer key reaches only `POST /v1/chat/completions`,
  `POST /v1/messages` and `GET /v1/models`; other routes answer `403`. The name goes into the usage log. The file is
  read at start, so restart the server after you change it.
- With customer keys and no admin key, nobody can use the admin routes.
- With no key at all the server is open to anyone who reaches it. That is meant for a local machine only.

Clients send the key as `Authorization: Bearer <key>` (OpenAI) or `x-api-key: <key>` (Anthropic).

**Privacy between keys.** Strata keeps recent conversation states in GPU and host memory (`--cache-ram`) so that a
request that continues a conversation prefills only the new part. Each key only reuses states that its own requests
made, so one customer cannot tell from `cached_tokens` or from the response time what another customer sent. All
requests under one key count as one user: if a customer serves many end users through one key, the customer's end
users can learn this about each other. Strata does not write prompts or replies to disk; the usage log holds only
counts and times.

## Reverse proxy and TLS

Strata speaks plain HTTP. Put a reverse proxy in front of it for TLS, and bind Strata to `127.0.0.1`. When keys are
set, Strata accepts any `Host` header, so the proxy can pass the public name through.

Streaming replies (`"stream": true`) must reach the client as they are written, so turn off response buffering, and
give the proxy a read timeout that is longer than your longest reply.

**Caddy** (gets and renews the certificate itself):

```
api.example.com {
    reverse_proxy 127.0.0.1:8080 {
        flush_interval -1
        transport http {
            read_timeout 15m
        }
    }
}
```

**nginx:**

```nginx
server {
    listen 443 ssl;
    server_name api.example.com;
    ssl_certificate     /etc/letsencrypt/live/api.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.example.com/privkey.pem;

    client_max_body_size 8m;        # Strata refuses bodies over 8 MiB
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_buffering off;        # stream tokens as they come
        proxy_read_timeout 900s;    # a long reply, or a long wait in the queue
        proxy_send_timeout 900s;
    }
}
```

To keep the chat page and the admin routes off the public address, serve only `/v1/` and `/health` there:

```nginx
    location /v1/    { proxy_pass http://127.0.0.1:8080; proxy_http_version 1.1; proxy_buffering off; proxy_read_timeout 900s; }
    location /health { proxy_pass http://127.0.0.1:8080; }
    location /       { return 404; }
```

Then reach the chat page through an SSH tunnel: `ssh -L 8080:127.0.0.1:8080 server`, and open
http://127.0.0.1:8080 with the admin key.

Rate limits per customer (requests per minute) belong in the proxy too, for example nginx `limit_req` keyed on the
`Authorization` header.

## systemd

`/etc/strata/env` (mode 600):

```
STRATA_API_KEY=<admin key>
```

`/etc/systemd/system/strata.service`:

```ini
[Unit]
Description=Strata inference server
After=network-online.target
Wants=network-online.target

[Service]
User=strata
EnvironmentFile=/etc/strata/env
ExecStart=/usr/local/bin/strata serve --model /srv/models/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf --gpu --ctx 32768 \
    --addr 127.0.0.1:8080 --api-keys /etc/strata/keys.txt --usage-log /var/lib/strata/usage.jsonl --max-tokens 4096
StateDirectory=strata
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```sh
sudo useradd --system --create-home --home-dir /var/lib/strata strata
sudo chown strata /etc/strata/keys.txt
sudo usermod -aG video,render strata      # GPU device access on most distributions
sudo -u strata strata calibrate --model /srv/models/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf
sudo systemctl daemon-reload && sudo systemctl enable --now strata
journalctl -u strata -f
```

Run `calibrate` as the service user, because the calibration file and the CUDA kernel cache live in that user's
`~/.cache/rusting_strata`. The first start after an update compiles the CUDA kernels (about 15 s); `/health` answers
`503` until the model has loaded.

## Health checks

`GET /health` needs no key and is answered at once, also while a reply is generated or a model loads:

- `200 {"status": "ok", "waiting": N}` once a model is loaded. `waiting` counts the requests read or queued besides
  this one.
- `503 {"status": "loading"}` or `503 {"status": "no model"}` otherwise.

Point the load balancer or your monitoring at it.

## Usage log and billing

With `--usage-log FILE`, every answered chat request adds one JSON line:

```json
{"at":1793917331,"key":"acme","model":"Qwen3.6-35B-A3B-UD-Q2_K_XL","prompt_tokens":1200,"cached_tokens":1100,"system_tokens":300,"completion_tokens":250,"prefill_ms":95.1,"decode_ms":2210.4,"finish":"stop"}
```

- `at`: Unix time in seconds when the request started.
- `key`: the key's name from the key file (`admin` for the admin key, empty on a server without keys).
- `prompt_tokens`: all prompt tokens. `cached_tokens` of them came from the conversation cache and were not prefilled
  again; many providers bill them at a lower price.
- `completion_tokens`: reply tokens. `finish` is `stop`, `length` (hit `max_tokens` or the context) or `tool_calls`.

Requests that fail, and replies that stop because the client went away, are not logged. Tokens per key and day:

```sh
jq -r '[(.at | strftime("%Y-%m-%d")), .key, .prompt_tokens - .cached_tokens, .cached_tokens, .completion_tokens] | @tsv' usage.jsonl \
  | awk '{k=$1" "$2; p[k]+=$3; c[k]+=$4; o[k]+=$5} END {for (k in p) print k, p[k], c[k], o[k]}' | sort
```

The file only grows; rotate it with logrotate (`copytruncate`, since Strata keeps it open).

## Limits per request

- `--ctx` is the context length for prompt plus reply (4096 by default). Requests whose prompt fills it get `400`.
  KV memory grows with it: about 650 MB at 32K for the 35B model, half that with `--kv q8`.
- `--max-tokens` caps the reply length a request may ask for, and is the length when it asks for none. Without it a
  request may decode until the context is full, which holds the server for minutes.
- Bodies over 8 MiB are refused, and a client has 30 s to send its request.
- A client that disconnects stops its reply within a few tokens, so the next request in the queue starts.

## Several GPUs

**One server per GPU, more users.** Start one server per card on its own port and balance between them:

```sh
strata serve --model $M --gpu --device 0 --addr 127.0.0.1:8081 --threads 6 ...
strata serve --model $M --gpu --device 1 --addr 127.0.0.1:8082 --threads 6 ...
```

```nginx
upstream strata { least_conn; server 127.0.0.1:8081; server 127.0.0.1:8082; }
```

Give each server half of the CPU cores (`--threads`), because each one computes the experts that do not fit in its
VRAM on the CPU. Conversation caches are per server, so a conversation that moves to the other server is prefilled
again; hashing on the key (`hash $http_authorization consistent;` instead of `least_conn`) keeps each customer on one
server.

**One server over several GPUs, a bigger model.** `--device 0,1` splits the layers over the cards, so more experts
fit in VRAM. It does not serve two requests at once.

## Checklist

- [ ] Strata listens on `127.0.0.1` only; the firewall opens 443 (and 80 for the certificate), not 8080.
- [ ] An admin key, and customer keys in a file with mode 600.
- [ ] The proxy streams without buffering and has a long read timeout.
- [ ] `--max-tokens` and `--ctx` fit your price list and your GPU memory.
- [ ] `--usage-log` is on and rotated.
- [ ] `/health` is monitored, and systemd restarts the server on failure.
- [ ] `strata calibrate` ran on this machine as the service user.
