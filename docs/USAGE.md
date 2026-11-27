# Usage

Everything here applies to `rustingstrata chat`, `generate`, `bench` and `serve` unless noted. See the
[README](../README.md) for install and a quick start, and [HOSTING.md](HOSTING.md) for running a server for other people.

## Terminal

`rustingstrata interface` is a short set of questions in the terminal: the model (a numbered list of the last 10
models picked here, the last one first, then the models folder and the Hugging Face and llama.cpp caches; its last
row takes a path to one file, or to a folder whose models it then lists), the mode (chat in the terminal,
chat page in the browser, API server, speed test), GPU or CPU, the context length and, for a server, the address.
Enter takes the default in brackets. It prints the command the answers make and runs it, so the command can be
copied for the next time. `--models-dir DIR` lists another folder.

On a terminal the engine's notes on stderr are dim, `warning` and `error` are coloured, and the terminal chats show
the prompt in colour and the reasoning dim. Output that goes to a pipe or a file has no escape codes; `NO_COLOR=1`
turns them off on a terminal too. A failed command prints one `error: ...` line and exits with status 1.

`rustingstrata update` downloads the newest release from GitHub for Linux or Windows x86_64 and puts it in place of
the running binary; the old binary stays next to it as `rustingstrata.old` until the next update. `update --check`
only says whether a newer release is out. At start `serve` (and a double-clicked binary) asks GitHub once, with a
5 s limit, and prints one line if a newer release is out; `STRATA_NO_UPDATE_CHECK=1` turns that off. A binary built
from source is replaced by the release build too. A binary under `/usr/` or `/opt/` (the AUR package
`rustingstrata-bin`) or in winget's or Scoop's folders is left alone: `update` says to use the package manager.

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
- `--ctx` KV cache size in tokens (default 4096). For Inkling 2, Hy4, GLM, DeepSeek, Kolibri and Gemma 4 it is the prompt plus reply limit instead, and the default is the context the model was trained on; their caches grow with the text.
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
  Without `--chunk`, a GPU load that runs VRAM out with the default (4096) or calibrated chunk halves it down to
  512 and loads again, saying which chunk it took; a `--chunk` given on the command line is kept and fails as before.
  `rustingstrata calibrate` measures the best chunk, draft count and short-prompt threshold and later runs read them
  from `~/.cache/rusting_strata/strata-<model>[-backend][-dev…].json` unless flags override. The suffix uses the
  backend that actually opened (so a CUDA to Vulkan fallback writes the Vulkan file). The models with their own
  forward are calibrated too; see [CALIBRATION.md](CALIBRATION.md) for the budgeted search and the `calibrate spec`
  sweep.
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
- `STRATA_TQ_F32=1` runs the CPU experts of trellis files (Gyro TQ2_T, TQK6, TQK7) with the f32 kernel. By default
  they use an int16 codebook against int8 activations, as the original Strata does. On a Ryzen 5 7600X with the
  RTX 3060 at `--vram-margin 4096` (about 8 GB free) and `--fixed-experts`, Gyro-S decodes at 12.3-12.6 tok/s with the
  integer kernel against 8.3-8.7 with the f32 one. The two do not give the same bits, so greedy output differs.
- `--vram-margin` MB of VRAM left free. One value applies to every device; `512,256` gives one per device in
  `--device` order, and `0:512,1:256` gives one per GPU index. On a GPU that drives no display, `--vram-margin 256` keeps more experts
  resident (+1-2% prefill); 128 runs out of memory.
- `--kv q8` halves the KV cache (345 instead of 650 MB at 32K) at about 6% slower 32K decode and 14% slower 32K prefill.
  `--kv q4` and `--kv q2` store it in 4-bit and 2-bit form (about 28% and 19% of the f16 size) for very long
  contexts. Attention then reads rounded keys and values, so the output can differ from f16; q2 differs most.
  The head size must be a multiple of 32, and `--llama-numerics` needs the f16 cache. `calibrate` takes the
  same `--kv` value as the runs that will use it.
- `--kv turbo4` (CUDA) uses TurboQuant: each 128-value K/V block is signed-Walsh rotated, quantized with
  4-bit Lloyd-Max centroids, and stored in about 28% of the f16 cache size. Q is rotated by the same transform
  when the K cache is TurboQuant, and the attention output is de-rotated when the V cache is TurboQuant.
  `--kv turbo4-3`, `turbo4-2`, `turbo3`, `turbo2` and `turbo3-2` select narrower K/V centroids.
- `--k TYPE` and `--v TYPE` set the two cache formats independently (CUDA). Either half may be `f16`, `q8`,
  `q4`, `q2`, `turbo4`, `turbo3` or `turbo2`; for example `--k f16 --v q4` or `--k turbo4 --v turbo3`.
  A mixed layout allocates each half with its own word width. The Vulkan backend only supports symmetric
  legacy formats. Text can differ from f16 because attention reads rounded keys and values.
- `calibrate --mtp-model PATH` or `--draft-model PATH` measures with that draft file loaded, so the chunk and the
  expert placement see the VRAM it takes, and picks the draft count. Runs with the same `--mtp-model` or
  `--draft-model` file use that count unless `--mtp` is given; runs with another draft file or none do not.
- `calibrate --model ~/models` calibrates every model in a folder (and its subfolders, as the chat page lists them),
  one after another, with the other flags applied to each. A model that fails is reported and the rest go on. All
  architectures are included: Inkling 2, Hy4, GLM, DeepSeek and Gemma 4 are measured through their own `bench` path.
  `--mtp-model` and `--draft-model` need a single model file.
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
- Inkling 2, Hy4, GLM, DeepSeek, Kolibri and Gemma 4 run on their own code path. The server loads them like any other model
  (`serve --model <file>`, or from the chat page of a running server, in either direction), with these limits: one
  request at a time, greedy (`logit_bias` and `grammar` work; Inkling 2 also samples: `temperature`, `top_k`, `top_p`, `min_p`, `seed`; a seed repeats its
  text only with `STRATA_FIXED_EXPERTS=1`, because the adaptive expert split moves the logits a little),
  `/v1/chat/completions` only (no `/v1/messages`). `calibrate` measures their prefill chunk only
  ([CALIBRATION.md](CALIBRATION.md)). The context limit is `--ctx` (or the page's context setting) when given, else the trained context.
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
- A streamed reply sends its response head when the model takes the request, before the prompt runs. While the prompt
  runs, and while the text of a tool call is held back, the stream sends the SSE comment `: keep-alive` every 15 s, so
  a client that stops after some minutes without a byte keeps reading. If the prompt fails after the head, the stream
  ends early.
  A model with its own forward (Inkling 2, Kolibri, Gemma 4, DeepSeek, GLM, Hy4) answers one request at a time; a
  streamed chat that waits behind another gets its head after 10 s in the queue and then the same comment every 15 s,
  so a client with a time limit for the head (OpenCode) waits for its turn. A refusal after that head is one
  `data: {"error": ...}` line.
- It runs one request at a time; `--parallel N` (GPU) decodes up to N together (4 requests: 164 tok/s in total on the
  RTX 3060, 41 each; see [HOSTING.md](HOSTING.md)). From 8 requests decoding at once (16 for a model with experts;
  `STRATA_BATCH_MIN` sets the count), a model without MTP, LoRA, Hadamard rotation or hyper-connections (CUDA and
  Vulkan) runs them in one batched forward, and the prompts of requests that arrive together are prefilled together. For a MoE
  model each expert is read once for all requests of the step, and the CPU runs the experts that are not in VRAM.
  Slot state that does not fit in VRAM is kept in RAM, and `--parallel` is lowered to the slots that fit. For a model
  without experts each slot's KV cache starts at 512 tokens and doubles when a request needs more, up to `--ctx`, so
  short requests do not each hold a full context (Qwen3-4B Q4_K_M on the RTX 3060 at the default `--ctx 4096`: all 64
  slots fit and 64 clients get 1725 tok/s in total, against 12 slots and 528 tok/s before); a request whose cache
  cannot grow for lack of VRAM fails alone. That forward
  rounds differently from the single-request one, so greedy output under load can differ at near-ties from the same
  request run alone. An image request (`--mmproj`) takes a slot too: its prompt runs a token at a time between the
  other slots' steps, and its reply decodes beside the batched forward, one token a step. With `--gpu` the image encoder runs on the GPU too
  (`STRATA_VISION_CPU=1` keeps it on the CPU; see docs/VISION.md).
- `temperature`, `top_k`, `top_p`, `min_p` and `seed` are honored; decoding is greedy when they are left out.
  `min_p` (0 to 1, `--min-p`, default 0 = off) drops the tokens whose probability is below `min_p` times the
  best token's, as in llama.cpp.
- `logit_bias` is added to the logits before the token is picked, greedy or sampled, on every model (also the
  models with their own forward). Both forms work: the OpenAI object `{"15339": -100}` and llama.cpp's list
  `[[15339, -100], ["hello", 5], ["hello", false]]`. A text applies to each of its tokens. The bias is -100 to 100;
  -100 and `false` ban the token. A token id that is not in the vocabulary is a `400`.
- `grammar` (llama.cpp's GBNF, as a string) limits the reply to the text the grammar allows, for example
  `root ::= "yes" | "no"` or llama.cpp's `grammars/json.gbnf`. The reply ends only once the grammar is complete.
  Thinking is off for the request unless `chat_template_kwargs.enable_thinking` turns it on, because the grammar
  holds for the thinking text too. A grammar that does not parse is a `400` with the reason. With a grammar or a
  bias the GPU drafts (`--mtp`, `--spec-type`) are off for that request, and each token checks the whole vocabulary
  against the grammar (about 10 ms per token for a 150K vocabulary). `/v1/messages` takes neither field.
- `response_format` (OpenAI) makes the reply JSON through the same grammar check: `{"type": "json_object"}` takes
  any JSON object, and `{"type": "json_schema", "json_schema": {"schema": {...}}}` (or llama.cpp's
  `{"type": "json_object", "schema": {...}}`) takes JSON that fits the schema. The schema may use `type` (one or a
  list), `properties` and `required` (the properties come in the schema's order, the ones that are not required may
  be left out), `items`, `enum`, `const`, `anyOf`/`oneOf`, `allOf` of one schema and local `$ref`s (`#/$defs/...`).
  Lengths, counts, `pattern`, `format` and number bounds are not checked. `{"type": "text"}` is the default.
  A request with both `grammar` and `response_format` is a `400`.
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
  Without `max_tokens` (and without `--max-tokens` on the server) a reply runs until the model stops or the context
  is full.
- A request whose last message has role `assistant` (and no tool calls) continues that message: the reply holds only
  the new text. The chat page's continue button uses this.
- Thinking is on by default (`--think false` turns it off). A request overrides it with `chat_template_kwargs.enable_thinking`
  (Anthropic: `thinking: {"type": "enabled"}`; `enable_thinking: false` turns it off for that request).
- `tools` and `tool_choice` work as in the OpenAI API: calls come back as `tool_calls` with `finish_reason:
  "tool_calls"`. `tool_choice: "required"` does not force a call. A streamed chat on `/v1/chat/completions` sends each
  call's name as soon as the model writes it, and for a model with `<function=..>` calls (Qwen3.5, Qwen3.6, Qwen3
  Coder) a string argument as it is written, so a client can show a long file write while it runs. Other call formats
  send a call's arguments when the call is complete. This is the main engine; the Messages API and the separate
  forwards (Inkling 2, DeepSeek, GLM and the others) still send calls at the end.
- Each reply carries `usage` and llama-server's `timings` block (`prompt_n`, `prompt_ms`, `prompt_per_second`,
  `predicted_n`, `predicted_ms`, `predicted_per_second`, `cache_n`), in the body or in the last chunk of a stream, on
  every model, so llama-swap shows the prefill and decode speed.
- The server adds each answered chat to `usage.json` next to the default models folder
  (`~/.local/share/rusting_strata/usage.json` on Linux). `rustingstrata stats` prints the totals over all runs, by model
  and by API key name (`open` without `--api-keys`): requests, prompt, cached and reply tokens, prefill and decode
  speed, and the first and last day. Delete the file to start again. `--usage-log` still writes one line per request.
- The prompt follows the model's own chat template, rendered like llama.cpp's `--jinja`.
- `POST /tokenize` and `POST /detokenize` work as in llama.cpp's server: `{"content": "text", "add_special": false,
  "with_pieces": false}` gives `{"tokens": [...]}` (with pieces each entry is `{"id", "piece"}`, and a piece that is
  not whole UTF-8 is its list of bytes), and `{"tokens": [...]}` gives `{"content": "text"}`. `add_special` puts the
  BOS token in front when the model asks for one.
- Each status line on stderr starts with the local time (`[23:41:07]`). While a prompt runs or a reply is
  generated, a line about every 2 s gives the speed: `prefill: X tok/s, N tok` (the prompt tokens so far over their
  time), then `decode: X tok/s, N tok` (over the last 2 s). The terminal chat prints only the prefill line, so the
  reply text is not broken up. Nothing is printed while the server is idle.

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
`--split 30,70` (or `STRATA_SPLIT=30,70`) sets the share of the layers per card by hand, in `--device` order; the load
prints each card's layers and its measured host-to-VRAM copy rate. `rustingstrata calibrate` does not measure or write
a split: it times the chunk, drafts and short-prompt threshold with the split the run will use, and `--split` is
accepted for symmetry but ignored with a note. To compare card orders, calibrate `--device 1,0` too: each order has
its own file (`strata-<model>-dev0-1.json`, `-dev1-0.json`), and a run uses the file of the order it is given.
AMD cards clock down in the gaps while the other card works; `echo high >
/sys/class/drm/cardN/device/power_dpm_force_performance_level` (as root, per card) took a two-card split from 44 to 57
tok/s in one report.

### Inkling 2 on several GPUs

Inkling 2 is the model this work targets, and `--device 0,1` is opt-in: the default is always one card, even on a
machine with two. With two cards the 24 expert layers are placed as whole slabs (the load line says how many and
where), a layer's FFN runs on the card that holds it, and the attention and tail stay on GPU 0.

`STRATA_DS_FAST_KERNELS` selects the kernels:

- `auto` (default): the tiled controllers, `ink_attn_128` and the 16-query prefill (`kp16`) only when the run has more
  than one device. One card keeps the `main` kernels, so the greedy output does not change.
- `1`: force them on one card (the one-card kernel numbers in the PR).
- `0`: force the old kernels on two cards (the A/B).

Per-kernel overrides: `STRATA_CTRL_TILE=0` (per-token controllers), `STRATA_DS_ATTN_OLD=1` (scalar decode and
one-query-a-warp prefill), `STRATA_DS_ATTN_KP4=1` (four-query prefill).

The two-card plan uses these knobs:

| variable | default | effect |
|---|---|---|
| `STRATA_DS_FAST_KERNELS` | `auto` | as above; one card keeps the old kernels unless forced |
| `STRATA_DS_PLAIN_FIRST` | unset; `chat`/`serve`/`bench` set `1` | keep the plain weights on GPU 0, where the decode state lives |
| `STRATA_DS_PF_TOKENS` | 2048 one card, 512 two | tokens of one prompt chunk (at most 2048) |
| `STRATA_DS_KV_TOKENS` | `--ctx` | KV rows GPU 0 must still be able to take before it gives up a slab |
| `STRATA_DS_PROF` | unset | print `inkling2: preloaded N/24 ...` and the per-chunk profile |
| `STRATA_DS_NO_AFFINITY` | unset | pick the GPU per expert instead of per stacked tensor |
| `STRATA_DS_NO_SEG_GRAPH` | unset | do not capture the segmented decode graph |
| `STRATA_CTRL_TILE` | follows `FAST_KERNELS` | `0` forces the per-token controllers |
| `STRATA_DS_ATTN_OLD` | off | `1` keeps the scalar decode and one-query-a-warp prefill |
| `STRATA_DS_ATTN_KP4` | off | `1` selects the four-query 128-key prefill for the A/B |

`STRATA_INKLING2_MODEL` (path to the GGUF) only selects the model for `tests/inkling2_forward.rs`.

## Page cache

Experts that do not fit in VRAM are read from the GGUF through the page cache. A file system that keeps file data out
of the page cache (ZFS with `primarycache=metadata`) makes every run read them from disk again; load the model from a
file system that caches data, or from a tmpfs copy.

When the model files fit in RAM with 4 GiB to spare, they are read into the page cache in the background from the
start and nothing is dropped, so experts that run on the CPU do not wait for the drive on their first use.
`STRATA_PRELOAD=1` forces that, `STRATA_PRELOAD=0` turns it off.
A file on a network file system (SMB, NFS, a FUSE mount) is not read ahead, because that asks the server for the whole
file at once: its pages are read when they are used (`STRATA_PRELOAD=1` reads it ahead all the same).

Otherwise bytes copied to VRAM are dropped from the page cache (`MADV_DONTNEED`), so on a model close to the machine's
RAM the experts that missed VRAM keep their pages; `STRATA_COLD_KEEP=1` only marks them reclaimable instead. `serve` learns
the placement from the reply (16 swaps every 16 tokens after the prompt); `STRATA_ADAPT_EVERY` and
`STRATA_ADAPT_SWAPS` override that, and `STRATA_RESET_COUNTS=1` turns on the reply-only placement for other commands.
