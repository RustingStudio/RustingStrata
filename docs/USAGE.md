# Usage

Everything here applies to `rustingstrata chat`, `generate`, `bench` and `serve` unless noted. See the
[README](../README.md) for install and a quick start, and [HOSTING.md](HOSTING.md) for running a server for other people.

## Terminal

`rustingstrata interface` is a short set of questions in the terminal: the model (a numbered list of the models
folder and of the Hugging Face and llama.cpp caches; its last row takes a path to one file, or to a folder whose
models it then lists), the mode (chat in the terminal,
chat page in the browser, API server, speed test), GPU or CPU, the context length and, for a server, the address.
Enter takes the default in brackets. It prints the command the answers make and runs it, so the command can be
copied for the next time. `--models-dir DIR` lists another folder.

On a terminal the engine's notes on stderr are dim, `warning` and `error` are coloured, and the terminal chats show
the prompt in colour and the reasoning dim. Output that goes to a pipe or a file has no escape codes; `NO_COLOR=1`
turns them off on a terminal too. A failed command prints one `error: ...` line and exits with status 1.

## Chat page

`rustingstrata` with no arguments starts the server on http://127.0.0.1:8080 and opens a chat page. The page keeps chats in
the browser, downloads models from Hugging Face (settings; `HF_TOKEN` for gated repos), switches between the
models in the models folder (`%LOCALAPPDATA%\rusting_strata\models` on Windows, `~/.local/share/rusting_strata/models`
elsewhere, or `serve --models-dir`) and the GGUF files that the Hugging Face cache (`~/.cache/huggingface/hub`,
`HF_HUB_CACHE`, `HF_HOME`; `hf download` and llama.cpp `-hf`) and llama.cpp's cache (`~/.cache/llama.cpp`, `LLAMA_CACHE`)
already hold, loads a `.gguf` file from any other folder by its full path (settings, models;
`POST /api/load` with `"path"`, admin key only; the path is one the server sees, so in a container it is a path inside
the container), and shows context use, speed, totals, and CPU, memory and graphics memory use of
the server (status bar at the bottom; click it for details).

Messages can be edited and resent, deleted, regenerated, and a reply that stopped at the token limit or was stopped by
hand can be continued. Settings hold the sampler values (`temperature`, `top_p`, `top_k`, `seed`, stop strings, max
tokens; empty fields leave the server's flags in effect), named presets, instructions for one chat, and export of chats
as Markdown or JSON and import of a JSON export. The models tab also holds load options (`--ctx`, `--kv`, `--cache-ram`,
`--vram-margin`, `--mtp`, `--chunk`): "apply and reload" loads the model again with them, and they stay until the server
restarts (`POST /api/load` takes them as `options`, admin key only). For a GPU model with no calibration file the
page says so and offers "calibrate", which measures the model and loads it with the result (`POST /api/load` with
`"calibrate": true`). The accent color follows the loaded model's
family (Qwen purple, DeepSeek blue, Gemma multicolor, Flint and GLM gray, anything else green); "color theme" in the
connection and data tab picks one color by hand instead. Open the page with `#selftest` at the end of the address to run its
built-in checks.

### Tools (MCP)

The chat page can call the tools of Model Context Protocol servers. List the servers in a JSON file, in the format
other MCP clients use, and pass it with `--mcp <file>` (or `STRATA_MCP`; the default is `mcp.json` in the
`rusting_strata` folder that holds the default models folder):

    {"mcpServers": {
      "files": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/me/notes"]},
      "docs":  {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer ..."}}
    }}

- Servers start when the page first asks for them. The tools tab in settings shows each server, its tools and why a
  server did not start; untick a server or a tool to keep it from the model. The `tools` switch beside `think`
  turns tools on for the replies.
- Each call waits for your answer unless "run tools without asking" is on. The same tab sets how many tool rounds one
  reply may take and how long a tool result may be.
- Only tools with text results. Resources, prompts, sampling and OAuth sign-in are not supported.
- Servers come only from the file: the page cannot add a command. The routes (`GET /api/mcp`, `POST /api/mcp/reload`,
  `POST /api/mcp/call`) need the admin key when keys are set; customer keys cannot use them. Without keys they answer
  only requests from the same computer that name it `localhost`, `127.0.0.1` or `[::1]` and carry no proxy header
  (`Forwarded`, `X-Forwarded-*`, `X-Real-IP`). Set `--api-key` to use tools from another machine, and always set it
  behind a reverse proxy or a tunnel: a proxy that adds no such header looks like a local browser.
- Tool definitions are part of the prompt. Many tools cost context and prefill time; switch off what a chat does not
  need.
- The model needs a chat template with tool calls (the Qwen families have one).
- API clients are not affected: `/v1/chat/completions` and `/v1/messages` take the client's own `tools` as before.
- `STRATA_MCP_TIMEOUT` (seconds, default 120) is how long one answer from a server may take.
- On Windows a command starts through `cmd /c` (for the `npx`/`uvx` shims); this path is not tested yet.

## Flags

- `--backend NAME` GPU backend: `cuda`, `vulkan` or `hip` (AMD cards through ROCm, one device per run, see [HIP.md](HIP.md)). Without it CUDA is used when it loads, else Vulkan.
- `--ctx` KV cache size in tokens (default 4096). For Inkling 2, Hy4, GLM, DeepSeek and Gemma 4 it is the prompt plus reply limit instead, and the default is the context the model was trained on; their caches grow with the text.
- `--mtp [N]` speculative decoding with the model's draft head. Greedy tokens stay the same; sampling keeps its
  distribution. Needs a GGUF with an MTP head (for example the Qwen3.6-35B-A3B MTP files).
  Qwen3.8-Flash-Next (`qwen4exp`) takes its head from a companion file: `--mtp-model mtp-...-shared-Q8_0.gguf`.
  Drafts only pay when most experts are in VRAM: on the 12 GB reference machine with the IQ2_XS file (CPU experts
  on most tokens) `--mtp 1` decodes slower than no drafts. With a `--device 0,1` split the head runs on the
  last device (EAGLE3 heads excepted: those decode without drafts on a split).
- `--draft-model PATH` speculative decoding with a separate draft model (needs `--gpu` and one stream, so not with
  `--parallel`). The target checks every drafted token, so a draft never decides a token; sampling keeps its
  distribution. As with `--mtp`, the batched check rounds differently from one-token decoding, so greedy output can
  differ from a run without drafts at a near tie. Two kinds of file work:
  - a DFlash or DFlash2 draft: a small block-diffusion model trained for one target model. It reads hidden states
    of the target and predicts a whole block of tokens in one pass; the target then checks the block in one
    batched pass and keeps the tokens that match. `--mtp N` caps the drafted tokens per step (default: the
    draft's block size minus one). The draft must be made for this target: its embedding width, its target
    layers and its tensor shapes are checked at load, and a file that does not fit is refused. When the target
    GGUF has its own MTP head, the DFlash draft is used instead of that head.
  - an ordinary smaller model with the same vocabulary as the target (default `--mtp 4`).

  `PATH` is a `.gguf` file, a Hugging Face `.safetensors` file with its `config.json` next to it, or a
  directory holding either. Safetensors DFlash checkpoints are converted once and written next to the source as
  `model-dflash-q4_0.gguf` (`STRATA_DFLASH_Q8=1` writes and reads `model-dflash-q8_0.gguf`;
  `STRATA_DFLASH_RECONVERT=1` converts again). `STRATA_DFLASH_BLOCK=N` fixes the block size instead of adapting
  it to how many drafts are accepted.
- `--spec-type LIST` drafts from text that is already in the context, with no draft model: a comma-separated
  list of `suffix`, `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache`. It pays on
  output that repeats the prompt (code edits, quoting, structured data). It combines with `--mtp`,
  `--mtp-model` and `--draft-model`: the lookup drafts when it finds a match (next to a DFlash draft, only on a
  long match) and the model drafts otherwise.
- Inkling 2 drafts with `--draft-model PATH` (a DSpark draft file) and with `--spec-type`; it has no MTP head, so
  `--mtp` and `--mtp-model` are ignored with a warning. Drafts need the GPU path, and one pass checks at most 16
  rows. In a layer whose experts are not all in VRAM the rows of a check share few experts, so on a card that does
  not hold the whole model a draft model gains little. RTX 3060 12 GB, `Inkling-2-Mini-IQ4_NL.gguf` (17.6 GB, 15
  of 24 expert layers in VRAM on CUDA, 16 on Vulkan), `bench --pp 512 --tg 128`, two runs of each: CUDA 31.1 and
  35.2 tok/s without drafts, 48.5 and 52.4 with `--spec-type ngram-simple` (the bench text repeats, so nearly
  every draft is accepted), 35.1 and 35.6 with the DSpark file (285 of 294 drafts accepted); Vulkan 22.8 and 27.0
  without drafts, 51.1 and 56.2 with the lookup, 29.2 and 30.4 with the DSpark file. The batched check rounds
  differently from one-token decoding, so greedy output can differ at a near tie.
- `--chunk`, `--threads` prefill chunk size and CPU expert threads. On a GPU `--threads` defaults to the physical
  core count (SMT siblings only add contention; on the 6-core/12-thread reference machine 6 beats 12 by ~20%).
  `rustingstrata calibrate` measures the best chunk and later runs read it from
  `~/.cache/rusting_strata/strata-<model>.json` unless flags override.
  `XDG_CACHE_HOME` moves that folder to `$XDG_CACHE_HOME/rusting_strata` (on Windows it is
  `%LOCALAPPDATA%\rusting_strata`). In a container, set it to a mounted path: `docker run --rm` otherwise discards
  the file on exit, and the next container runs uncalibrated. The file names the GPU it was measured on, and a file
  measured on another GPU is ignored with a warning. It also holds the size of the GGUF file: another file with the
  same name (from another repo or quantizer) does not use it, and a warning asks to calibrate again. Vulkan adds
  `-vulkan` to the name, and a run on several cards adds the devices in `--device` order (`-dev0-1`), so a two-card
  calibration does not replace the one-card one.
- `STRATA_AVX512=1` uses the AVX-512 kernels for experts that run on the CPU, in a build for a CPU that has
  AVX-512 VNNI and VBMI (a `target-cpu=native` build on such a machine; the release binaries do not have them). Their
  results differ from the default kernels in the last bits, so greedy output can change at a near tie.
- `--vram-margin` MB of VRAM left free. One value applies to every device; `512,256` gives one per device in
  `--device` order, and `0:512,1:256` gives one per GPU index. On a GPU that drives no display, `--vram-margin 256` keeps more experts
  resident (+1-2% prefill); 128 runs out of memory.
- `--kv q8` halves the KV cache (345 instead of 650 MB at 32K) at about 6% slower 32K decode and 14% slower 32K prefill.
  `--kv q4` and `--kv q2` store it in 4-bit and 2-bit form (about 28% and 19% of the f16 size) for very long
  contexts. Attention then reads rounded keys and values, so the output can differ from f16; q2 differs most.
  The head size must be a multiple of 32, and `--llama-numerics` needs the f16 cache. `calibrate` takes the
  same `--kv` value as the runs that will use it.
- `calibrate --mtp-model PATH` or `--draft-model PATH` measures with that draft file loaded, so the chunk and the
  expert placement see the VRAM it takes, and picks the draft count. Runs with the same `--mtp-model` or
  `--draft-model` file use that count unless `--mtp` is given; runs with another draft file or none do not.
- `calibrate --model ~/models` calibrates every model in a folder (and its subfolders, as the chat page lists them),
  one after another, with the other flags applied to each. A model that fails is reported and the rest go on; Inkling
  2, Hy4, GLM, DeepSeek and Gemma 4 models are skipped: `calibrate` measures the main engine only, and these models
  run on their own code path, which reads no calibration file. `--mtp-model` and `--draft-model` need a single file.
- `--thinking-budget N` (`chat` and `serve`) ends the thinking block after `N` tokens by inserting `</think>`,
  so the model goes on to the answer (0, the default, is no limit). A request overrides it with
  `thinking_budget` or `chat_template_kwargs.thinking_budget` (Anthropic: `thinking.budget_tokens`).
- `--short N` prompts shorter than `N` tokens skip the batched prefill and go token by token (default: from
  `rustingstrata calibrate`, else 32).
- `--allowed-hosts ai-server.lan` (or `STRATA_ALLOWED_HOSTS`) allows a LAN name when `serve` is bound to `0.0.0.0`.
- `--auto-unload 600` (`serve`) frees the model after 600 seconds without a chat request: the VRAM and the memory of
  the process go back to the system, so an idle home server does not hold the GPU. The next chat request loads the
  model again with the same options and is answered after the load; the model stays in `/v1/models` meanwhile, and
  `/health` answers `200 {"status": "ok", "unloaded": true}`. Only chat requests count as use and load the model:
  the chat page, the model list, `/health` and the stats do not. The file stays in the page cache (`buff/cache`,
  which the system takes back when it needs the memory), so the load after an unload is faster than the first one.
  The saved conversation (prompt cache) goes with the model: the first request after a load prefills its whole
  prompt.
- `--alias prism` (`serve`) is the model name in `/v1/models` and in the replies, whatever file is loaded (default: the
  file name), so clients set up for `prism` keep working when another model is loaded. Calibration and the expert
  profile stay per file.
- Inkling 2, Hy4, GLM, DeepSeek and Gemma 4 run on their own code path. The server loads them like any other model
  (`serve --model <file>`, or from the chat page of a running server, in either direction), with these limits: one
  request at a time, greedy (Inkling 2 also samples: `temperature`, `top_k`, `top_p`, `seed`; a seed repeats its
  text only with `STRATA_FIXED_EXPERTS=1`, because the adaptive expert split moves the logits a little),
  `/v1/chat/completions` only (no `/v1/messages`), and no calibration (`calibrate` does
  not cover them). The context limit is `--ctx` (or the page's context setting) when given, else the trained context.
- `--lora PATH:SCALE` applies a LoRA adapter at runtime, see [LORA.md](LORA.md).
- `--mcp FILE` (`serve` only, or `STRATA_MCP`) lists the MCP servers whose tools the chat page may call, see
  [Tools (MCP)](#tools-mcp).

## Server API

`serve` speaks the OpenAI chat completions API (`POST /v1/chat/completions`, streamed or not, `GET /v1/models`).

- `GET /v1/models` lists the loaded model first, then every other model the chat page offers (the models folder, the
  Hugging Face and llama.cpp caches), each by its file name without `.gguf`. A chat request whose `model` is one of
  these names makes the server load that model in place of the loaded one, and is answered after the load; so a
  server started without `--model` serves whichever model the client picks. The replies in progress end first. A
  `model` that no listed file has is answered by the loaded model. If the load fails, the request gets `503` with
  the reason and the next chat request loads the model that was there before. With API keys only the admin key sees
  the other models and switches; a customer key sees and gets the loaded model.
- It runs one request at a time; `--parallel N` (GPU) decodes up to N together (4 requests: 164 tok/s in total on the
  RTX 3060, 41 each; see [HOSTING.md](HOSTING.md)). From 8 requests decoding at once (`STRATA_BATCH_MIN` sets the
  count), a dense model without MTP, LoRA or recurrent layers on CUDA runs them in one batched forward, and the
  prompts of requests that arrive together are prefilled together. That forward
  rounds differently from the single-request one, so greedy output under load can differ at near-ties from the same
  request run alone.
- `temperature`, `top_k`, `top_p` and `seed` are honored; decoding is greedy when they are left out.
- `presence_penalty` and `frequency_penalty` (OpenAI, -2 to 2) and `dry_multiplier`, `dry_base`, `dry_allowed_length`
  (the DRY sampler, llama.cpp server names) act against repetition loops. They count only the tokens of the reply,
  not the prompt, and do not apply inside a tool call. `--presence-penalty`, `--frequency-penalty`,
  `--dry-multiplier`, `--dry-base` and `--dry-allowed-length` set the values for requests that leave them out. All
  are off by default, and they need a temperature above 0: greedy decoding does not use the sampler.
- `repeat_penalty` and `penalty_last_n` (llama.cpp server names; `--repeat-penalty`, `--penalty-last-n`) follow the
  same rules. A repeat penalty above 1 (1.1 is usual; 1 is off, the default) lowers every token among the last
  `penalty_last_n` tokens of the reply (default 64; 0 turns it off, -1 takes the whole reply). The window is for
  the repeat penalty only: the presence and frequency penalties count the whole reply.
- `max_tokens` (or `max_completion_tokens`) and `stop` (up to 16 strings) work as in the OpenAI API; `n` must be 1.
- A request whose last message has role `assistant` (and no tool calls) continues that message: the reply holds only
  the new text. The chat page's continue button uses this.
- Thinking is on by default (`--think false` turns it off). A request overrides it with `chat_template_kwargs.enable_thinking`
  (Anthropic: `thinking: {"type": "enabled"}`; `enable_thinking: false` turns it off for that request).
- `tools` and `tool_choice` work as in the OpenAI API: calls come back as `tool_calls` with `finish_reason:
  "tool_calls"`. `tool_choice: "required"` does not force a call.
- The prompt follows the model's own chat template, rendered like llama.cpp's `--jinja`.

`POST /v1/messages` speaks the Anthropic Messages API (text, `tool_use` and `tool_result` blocks, streamed or not, the
key in `x-api-key`, and `POST /v1/messages/count_tokens`), so Claude Code can use the server:

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_API_KEY=secret ANTHROPIC_MODEL=local ANTHROPIC_SMALL_FAST_MODEL=local claude
```

OpenCode and other agents use the OpenAI address.

## Prompt caching

- When a request repeats the previous request's messages and adds new ones, only the new part is prefilled
  (`usage.prompt_tokens_details.cached_tokens`).
- Qwen3.6's template leaves the think block out of replies before the last user message, so each new user message would
  change the earlier turns and force a full prefill. RustingStrata keeps those blocks by default;
  `--preserve-thinking false` (or `chat_template_kwargs.preserve_thinking` in a request) follows the template.
- A request with another history (an agent's side request for a title or a summary) parks the conversation's state in
  host memory (`--cache-ram`, 4096 MB by default), and the next request that continues it takes it back.
- With customer keys, each key reuses only its own cached conversations; see [HOSTING.md](HOSTING.md).

## Several GPUs

`--device 0,1` splits the layers over the cards in proportion to their free VRAM, so more experts stay resident. The
cards run one after the other, so one request gets more memory, not twice the speed of one card. MTP drafting is off on
a split model. The HIP backend runs one device, so `--device 0,1` is refused with `--backend hip`.

An index is the same card on both backends. NVIDIA cards are counted as CUDA counts them (by default the fastest card
first, which is not the order of `nvidia-smi`), on Vulkan too; the other Vulkan devices follow, dedicated cards before
integrated ones. The load prints the card of every index (`cuda device 1: ...` or `vulkan device 1: ...`).

The split by free VRAM does not know which card is faster or which sits on a slow PCIe link (x1 riser, x4 slot).
`rustingstrata calibrate --model M.gguf --device 0,1` measures it: it times prefill and decode with the split by free
VRAM, then with layers moved from one card to the other in steps of 15% of the layers, and follows the faster
direction for up to three rounds. A split that is more than 3% faster (prefill tok/s times decode tok/s; each split line prints this product as
a share of the best one so far) is written to the calibration file as layers per card, and later runs with the same
cards in the same `--device` order load with it.
A card with fewer layers keeps more of their experts in VRAM, so fewer are uploaded to it during prefill, and the other
card then keeps fewer of its own: the timings decide. The order of the cards is not searched: to compare, calibrate
`--device 1,0` too. Each order has its own file (`strata-<model>-dev0-1.json`, `-dev1-0.json`), and a run uses the
file of the order it is given.

`--split 30,70` (or `STRATA_SPLIT=30,70`) sets the share of the layers per card by hand, in `--device` order, and
takes precedence over the calibrated split; `calibrate --split 30,70` measures chunk and drafts with that split and
stores it. The load prints each card's layers and its measured host-to-VRAM copy rate. AMD cards clock down in the gaps while the other card works; `echo high >
/sys/class/drm/cardN/device/power_dpm_force_performance_level` (as root, per card) took a two-card split from 44 to 57
tok/s in one report.

## Page cache

Experts that do not fit in VRAM are read from the GGUF through the page cache. A file system that keeps file data out
of the page cache (ZFS with `primarycache=metadata`) makes every run read them from disk again; load the model from a
file system that caches data, or from a tmpfs copy.

When the model files fit in RAM with 4 GiB to spare, they are read into the page cache in the background from the
start and nothing is dropped, so experts that run on the CPU do not wait for the drive on their first use.
`STRATA_PRELOAD=1` forces that, `STRATA_PRELOAD=0` turns it off.

Otherwise bytes copied to VRAM are dropped from the page cache (`MADV_DONTNEED`), so on a model close to the machine's
RAM the experts that missed VRAM keep their pages; `STRATA_COLD_KEEP=1` only marks them reclaimable instead. `serve` learns
the placement from the reply (16 swaps every 16 tokens after the prompt); `STRATA_ADAPT_EVERY` and
`STRATA_ADAPT_SWAPS` override that, and `STRATA_RESET_COUNTS=1` turns on the reply-only placement for other commands.
