use clap::{Args, Parser, Subcommand};
use anyhow::Context;
use rusting_strata::memory;
use rusting_strata::{say, term};
use rusting_strata::model::{Model, State};
use rusting_strata::{backend, gpu_model::GpuModel};
use rusting_strata::{config::ModelConfig, gguf::Gguf, hf, ops, ops::Sampler, quant::GgmlType, serve, tokenizer::Tokenizer};
use rusting_strata::chat::{Hold, Msg, Template, Think};
use rusting_strata::vision::{self, VisionProjector};
use serde_json::json;
use std::io::{BufRead, BufReader, Read, Write};
use std::{collections::BTreeMap, net::{TcpListener, TcpStream}, sync::atomic::Ordering::Relaxed, path::{Path, PathBuf}, time::{Duration, Instant, SystemTime}};

#[derive(Parser)]
#[command(name = "rustingstrata", version, about = "Rust inference engine for Qwen3.5/3.6-family GGUF models")]
struct Cli {
    /// Without a command, strata serves the chat page on 127.0.0.1:8080 on the GPU (else the CPU) and opens it in
    /// the browser.
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Args, Clone)]
struct Engine {
    /// Run on the GPU (experts that do not fit in VRAM run on the CPU) instead of the CPU reference.
    #[arg(long)]
    gpu: bool,
    /// GPU context length (KV cache size) in tokens. Inkling 2, Hy4, GLM, DeepSeek and Gemma 4: the prompt plus reply
    /// limit, the model's trained context when not given (their caches grow with the text).
    #[arg(long, default_value_t = 4096)]
    ctx: usize,
    /// Prefill chunk in tokens (GPU; 0 = token by token) [default: calibrated, else 4096].
    #[arg(long)]
    chunk: Option<usize>,
    /// VRAM left free after placing expert layers, in MB. One value applies to every device; `512,256` gives one
    /// per device in `--device` order, and `0:512,1:256` gives one per GPU index.
    #[arg(long, default_value = "512", value_delimiter = ',')]
    vram_margin: Vec<String>,
    /// Speculative decoding with the model's MTP draft head, N drafts per step (GPU; greedy output unchanged, sampling keeps its distribution)
    /// [default: calibrated, else 0].
    #[arg(long, value_name = "N", default_missing_value = "1", num_args = 0..=1)]
    mtp: Option<usize>,
    /// Standalone MTP draft head GGUF (e.g. ~/models-local/mtp-Q8_0.gguf).
    #[arg(long)]
    mtp_model: Option<PathBuf>,
    /// External speculative draft model GGUF socket (e.g. future DFlash/EAGLE draft).
    #[arg(long)]
    draft_model: Option<PathBuf>,
    /// Speculative context-lookup drafter type(s), comma-separated: none, suffix, ngram-simple, ngram-map-k,
    /// ngram-map-k4v, ngram-mod, ngram-cache (combines with --mtp/--mtp-model/--draft-model in hybrid cascade).
    #[arg(long, default_value = "none")]
    spec_type: String,
    /// Maximum number of speculative draft tokens verified per step (1..=16).
    #[arg(long, default_value_t = 16)]
    spec_draft_n_max: usize,
    /// N-gram lookup key length in tokens (for ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod).
    #[arg(long, default_value_t = 12)]
    spec_ngram_size_n: usize,
    /// N-gram draft window length in tokens (for ngram-simple, ngram-map-k, ngram-map-k4v).
    #[arg(long, default_value_t = 48)]
    spec_ngram_size_m: usize,
    /// Minimum hit count before drafting (for ngram-map-k, ngram-map-k4v).
    #[arg(long, default_value_t = 1)]
    spec_ngram_min_hits: u16,
    /// Static lookup cache file (llama.cpp binary ngram cache or UTF-8 text scaffold corpus).
    #[arg(long)]
    lookup_cache_static: Option<PathBuf>,
    /// Dynamic lookup cache file loaded at startup and updated/saved across requests.
    #[arg(long)]
    lookup_cache_dynamic: Option<PathBuf>,
    /// Persistent disk cache file for KV and recurrent prefix states (Warm Start across requests and restarts).
    #[arg(long)]
    prompt_cache: Option<PathBuf>,
    /// Sampling temperature; 0 is greedy.
    #[arg(long, default_value_t = 0.0)]
    temp: f32,
    /// Sample only from the K most likely tokens (0 = all).
    #[arg(long, default_value_t = 0)]
    top_k: usize,
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    /// Default presence penalty (OpenAI `presence_penalty`); needs --temp > 0. Qwen recommends 1.5 against loops.
    #[arg(long, default_value_t = 0.0)]
    presence_penalty: f32,
    /// Default frequency penalty (OpenAI `frequency_penalty`); needs --temp > 0.
    #[arg(long, default_value_t = 0.0)]
    frequency_penalty: f32,
    /// Repeat penalty (llama.cpp `repeat_penalty`; 1 = off, 1.1 is a usual value) on the tokens among the last
    /// --penalty-last-n of the reply. Needs --temp > 0.
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,
    /// How many of the reply's last tokens --repeat-penalty looks at (0 = off, -1 = the whole reply).
    #[arg(long, default_value_t = 64, allow_negative_numbers = true)]
    penalty_last_n: i32,
    /// DRY multiplier (0 = off; 0.8 is the usual value): penalizes a token that would repeat a sequence the reply
    /// already wrote, more the longer the repeat. Stops paragraph loops that a presence penalty does not. Needs --temp > 0.
    #[arg(long, default_value_t = 0.0)]
    dry_multiplier: f32,
    #[arg(long, default_value_t = 1.75)]
    dry_base: f32,
    /// Repeats up to this many tokens long are free.
    #[arg(long, default_value_t = 2)]
    dry_allowed_length: usize,
    /// Random seed for sampling [default: from the clock].
    #[arg(long)]
    seed: Option<u64>,
    /// CPU threads for expert work [default: calibrated, else physical cores on the GPU, else all logical cores].
    #[arg(long)]
    threads: Option<usize>,
    /// Match llama.cpp's CPU numerics on the GPU (q8_K activations, f32 prefill attention, f16 decode attention
    /// accumulator) for parity runs; attention gets much slower.
    #[arg(long)]
    llama_numerics: bool,
    /// Place experts in layer order and never move them, ignoring and keeping the routing profile, so parity runs
    /// repeat exactly.
    #[arg(long)]
    fixed_experts: bool,
    /// Host RAM left free for the OS, in MB. The cap is `total RAM - margin` and bounds memory the process owns
    /// (baked tensors, LoRA factors, `--cache-ram`); the mmapped model is evictable page cache and is not counted.
    #[arg(long, default_value_t = 512)]
    ram_margin: usize,
    /// Absolute host RAM cap in MB; overrides `--ram-margin`. 0 uses the margin.
    #[arg(long, default_value_t = 0)]
    max_ram: usize,
    /// Host memory in MB for conversation states that a request with another history replaced (GPU): a later request
    /// that continues one of them takes it back instead of prefilling it again. 0 keeps none.
    #[arg(long, default_value_t = 4096)]
    cache_ram: usize,
    /// GGUF LoRA adapter(s) to apply at runtime as "path:scale" (repeatable; CPU and GPU).
    #[arg(long, value_name = "PATH:SCALE", action = clap::ArgAction::Append)]
    lora: Vec<String>,
    /// Keep the think block of earlier replies in the prompt (Qwen3.5/3.6 templates' `preserve_thinking`; a request
    /// can set it in `chat_template_kwargs`). The template drops it from replies before the last user message, so
    /// each new user message changes the earlier text and the whole conversation is prefilled again; with this, a
    /// coding agent's long history stays cached across user messages. `--preserve-thinking false` follows the
    /// template's default.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    preserve_thinking: bool,
    /// Let the model think before answering (chat and `serve`); `--think false` turns it off. A request's `chat_template_kwargs.enable_thinking` overrides it in `serve`.
    #[arg(long = "think", default_value_t = true, action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    think_default: bool,
    /// Cap tokens inside `<think>` in `chat` and `serve`, then inject `</think>\n\n` so the model answers immediately (0 = unlimited). A request's `thinking_budget` overrides it.
    #[arg(long, default_value_t = 0)]
    thinking_budget: usize,
    /// KV cache format (GPU): f16, q8 (q8_0, half KV memory), q4 (q4_0, quarter KV memory), or q2 (2-bit affine,
    /// 5.3x smaller KV memory for 512k-1M ctx; numerics differ from f16).
    #[arg(long, default_value = "f16", value_parser = ["f16", "q8", "q4", "q2"])]
    kv: String,
    /// GPU backend: cuda or vulkan [default: cuda if it loads, else vulkan].
    #[arg(long)]
    backend: Option<String>,
    /// GPU index, or several separated by commas (such as 0,1) to split the layers over those GPUs in proportion to
    /// their free VRAM.
    #[arg(long, default_value = "0", value_delimiter = ',')]
    device: Vec<usize>,
    /// Share of the layers per device of a `--device 0,1` split, in `--device` order (such as 30,70 to give a card
    /// on a slow PCIe link fewer layers) [default: in proportion to free VRAM].
    #[arg(long)]
    split: Option<String>,
    /// Prompts shorter than this skip the batched prefill [default: from `strata calibrate`, else 32].
    #[arg(long)]
    short: Option<usize>,
    /// Run on the CPU when no GPU backend opens (the bare `strata` launch).
    #[arg(skip)]
    cpu_fallback: bool,
    /// Sequences in GPU memory at once (`serve --parallel`).
    #[arg(skip)]
    parallel: usize,
    /// Multimodal projector (mmproj) GGUF used by `--image`.
    #[arg(long)]
    mmproj: Option<PathBuf>,
}

fn set_kv_env(kv: &str) {
    match kv {
        "q8" | "q4" | "q2" => {
            unsafe { std::env::set_var("STRATA_KV", kv) };
            if kv == "q8" { unsafe { std::env::set_var("STRATA_KV_Q8", "1") }; }
            else { unsafe { std::env::remove_var("STRATA_KV_Q8") }; }
        }
        _ => {
            unsafe { std::env::remove_var("STRATA_KV") };
            unsafe { std::env::remove_var("STRATA_KV_Q8") };
        }
    }
}

/// `serve` options for a server that other people use (docs/HOSTING.md).
#[derive(clap::Args, Clone, Debug)]
struct Hosting {
    /// File of customer keys, one `name key` pair a line (`#` comments). These keys reach only the chat APIs and
    /// the model list, and each key's cached conversations are kept apart from the others'. `--api-key` stays
    /// the admin key for every route. A changed file is read again before the next request.
    #[arg(long)]
    api_keys: Option<PathBuf>,
    /// Append one JSON line per chat request (time, key name, token counts, timings, how it finished) to this file.
    #[arg(long)]
    usage_log: Option<PathBuf>,
    /// Open connections (being read, generated or queued, health checks included); more get `503` with `Retry-After`
    /// [default: no limit, requests wait in the queue].
    #[arg(long)]
    max_waiting: Option<usize>,
    /// Requests one customer key may have read, generated or queued at once; more get `429` with `Retry-After`
    /// [default: no limit besides `--max-waiting`].
    #[arg(long)]
    max_per_key: Option<usize>,
    /// Extra `Host` header names accepted by the chat page, model list and stats, comma-separated. Needed when the
    /// server is reached through a LAN name while bound to `0.0.0.0`. [default: $STRATA_ALLOWED_HOSTS]
    #[arg(long, value_delimiter = ',')]
    allowed_hosts: Vec<String>,
    /// Most reply tokens a request may ask for (and the default when it asks for none) [default: what the context
    /// leaves].
    #[arg(long)]
    max_tokens: Option<usize>,
    /// Requests answered at once (GPU): each gets its own slot of `--ctx` tokens of KV cache, and the
    /// slots that decode share each forward, up to 4 at a time. Between the chunks of a prompt's prefill the others
    /// decode for as long as the chunk took (a smaller `--chunk` gives shorter pauses). MTP drafting is off.
    #[arg(long, default_value_t = 1)]
    parallel: usize,
    /// Free the model (its VRAM and its memory in this process) after this many seconds without a chat request. The
    /// next chat request loads it again, with the same options, and is answered after the load.
    #[arg(long, value_name = "SECONDS")]
    auto_unload: Option<u64>,
    /// The model name in `/v1/models` and in the replies, whatever file is loaded [default: the file name]. Clients
    /// configured for this name keep working when another model is loaded.
    #[arg(long)]
    alias: Option<String>,
}

/// `Own`: a model outside the main engine (`config::OWN_ARCHS`). Its chat answers a request whole, so the token
/// methods below do not apply to it.
enum Runner<'a> { Cpu(&'a Model<'a>, State), Gpu(Box<GpuModel<'a>>), Slot(Client), Own(serve::Own<'a>) }

impl Runner<'_> {
    fn step(&mut self, t: u32) -> anyhow::Result<Vec<f32>> {
        match self { Runner::Cpu(m, st) => Ok(m.step(st, t)), Runner::Gpu(g) => g.step(t), Runner::Slot(c) => Ok(c.step(t, true)?.1), Runner::Own(_) => anyhow::bail!("step on a model outside the main engine") }
    }
    /// Feed a prompt; returns the logits after its last token.
    fn prefill(&mut self, ts: &[u32]) -> anyhow::Result<Vec<f32>> {
        match self {
            Runner::Cpu(m, st) => Ok(ts.iter().map(|&t| m.step(st, t)).last().unwrap_or_default()),
            Runner::Gpu(g) => g.prefill(ts),
            Runner::Slot(_) | Runner::Own(_) => anyhow::bail!("prefill on a slot or a model outside the main engine"),
        }
    }
    /// Feed a chat prompt as a fresh sequence, reusing the earlier request's state for the shared message history
    /// (GPU; `history` is the prompt without the assistant header, `system` its system and tools block, kept for
    /// later requests that rewrite the history). Returns the logits and the number of tokens reused.
    fn chat(&mut self, tok: &Tokenizer, prompt: &str, history: &str, system: &str) -> anyhow::Result<(Vec<u32>, Vec<f32>, usize)> {
        let ids = tok.encode(prompt)?;
        let h = tok.encode(history)?;
        let keep = if ids.starts_with(&h) { h.len() } else { 0 };
        let s = if system.is_empty() { vec![] } else { tok.encode(system)? };
        let sys = if ids.starts_with(&s) && s.len() <= keep { s.len() } else { 0 };
        let (logits, reused) = match self {
            Runner::Gpu(g) => g.prefill_cached(&ids, sys, keep)?,
            Runner::Slot(c) => c.chat(&ids, sys, keep)?,
            _ => { self.reset()?; (self.prefill(&ids)?, 0) }
        };
        Ok((ids, logits, reused))
    }
    /// Feed a prompt that contains image embeddings as a fresh sequence.
    fn chat_image(&mut self, p: &vision::Prompt, embs: &[Vec<Vec<f32>>]) -> anyhow::Result<Vec<f32>> {
        match self {
            Runner::Cpu(m, st) => { *st = State::new(&m.cfg); vision::feed_prompt(m, st, p, embs) }
            Runner::Gpu(g) => g.prefill_image(p, embs),
            Runner::Slot(_) | Runner::Own(_) => anyhow::bail!("image input is not supported on a serve slot"),
        }
    }
    /// Forget the sequence so far.
    fn reset(&mut self) -> anyhow::Result<()> {
        match self { Runner::Cpu(m, st) => { *st = State::new(&m.cfg); Ok(()) } Runner::Gpu(g) => g.reset(), Runner::Slot(_) | Runner::Own(_) => Ok(()) }
    }
    /// Seconds spent waiting on CPU experts so far (GPU only).
    fn cpu_secs(&self) -> f64 { if let Runner::Gpu(g) = self { g.parts().map(|p| p.cpu_time.as_secs_f64()).sum() } else { 0.0 } }
    /// Bytes of routed-expert weights one decode token reads from the cache (0 without an expert cache).
    fn expert_bytes_per_token(&self) -> usize {
        if let Runner::Gpu(g) = self { g.parts().map(|p| p.expert_bytes_per_token()).sum() } else { 0 }
    }
    /// Split model: seconds per part but the last (forward until its GPU is done, copy to the next part).
    fn split_secs(&self) -> Vec<(f64, f64)> {
        let Runner::Gpu(g) = self else { return vec![] };
        let mut v: Vec<_> = g.parts().map(|p| (p.split_time.0.as_secs_f64(), p.split_time.1.as_secs_f64())).collect();
        v.pop();
        v
    }
    /// Backend name of the first part (`"cpu"` on the CPU, `"slot"` for a serve slot).
    fn backend_name(&self) -> &'static str {
        match self {
            Runner::Cpu(..) => "cpu",
            Runner::Gpu(g) => g.parts().next().map_or("cpu", |p| p.gpu.name()),
            Runner::Slot(_) => "slot",
            Runner::Own(_) => "own",
        }
    }
    /// Expert-cache counters summed over the parts, for `finish()` and the benchmark JSON (`None` off the GPU).
    fn expert_stats(&self) -> Option<ExpertStats> {
        let Runner::Gpu(g) = self else { return None };
        let sum = |f: &dyn Fn(&GpuModel) -> usize| g.parts().map(f).sum::<usize>();
        let parts: Vec<_> = g.parts().collect();
        let cached: Vec<_> = parts.iter().filter(|p| p.cache.is_some()).collect();
        Some(ExpertStats {
            hits: sum(&|p| p.hits),
            misses: sum(&|p| p.misses),
            swaps: sum(&|p| p.cache.as_ref().map_or(0, |c| c.swaps)),
            resident: sum(&|p| p.cache.as_ref().map_or(0, |c| c.resident_count())),
            total: sum(&|p| p.cache.as_ref().map_or(0, |c| c.total_experts())),
            bytes_per_token: sum(&|p| p.expert_bytes_per_token()),
            cached: !cached.is_empty(),
            all_fast: cached.iter().all(|p| p.expert_fast_path()),
        })
    }
    /// The same counters, but only when an expert cache exists (the benchmark record's top-level `expert`).
    fn expert_summary(&self) -> Option<ExpertStats> {
        self.expert_stats().filter(|s| s.cached)
    }
    fn finish(&self) -> anyhow::Result<()> {
        if let Runner::Gpu(g) = self {
            let sum = |f: &dyn Fn(&GpuModel) -> usize| g.parts().map(f).sum::<usize>();
            let st = self.expert_stats().expect("Runner::Gpu has expert stats");
            if g.drafts.0 > 0 {
                let kind = if g.has_dflash() && g.ngram.is_some() {
                    "FlashCascade (DFlash + suffix/ngram)"
                } else if g.has_dflash() {
                    "DFlash"
                } else if g.has_ext_draft() && g.ngram.is_some() {
                    "FlashCascade (external draft + suffix/ngram)"
                } else if g.has_ext_draft() {
                    "external draft"
                } else if g.ngram.is_some() {
                    "speculative ngram"
                } else {
                    "MTP"
                };
                say!("{kind} drafts accepted {}/{} ({:.1}%)", g.drafts.1, g.drafts.0, 100.0 * g.drafts.1 as f64 / g.drafts.0 as f64);
            }
            say!("expert hit rate {:.1}% ({} misses), cache swaps {}", 100.0 * st.hits as f64 / (st.hits + st.misses).max(1) as f64, st.misses, st.swaps);
            // Issue #34 R1: printed on every run, so a fully resident run reads as a measured 0 in the log instead of
            // an absent line. `tokens` counts the token forwards the worker could have been asked about.
            let tokens = sum(&|p| p.tokens);
            let busy: f64 = g.parts().map(|p| p.cpu_time.as_secs_f64()).sum();
            let stalls = sum(&|p| p.resumed.get());
            say!("worker busy {:.1} µs/token, worker stalls {stalls}", if tokens > 0 { 1e6 * busy / tokens as f64 } else { 0.0 });
            let mode = if !st.cached { "no expert cache".to_string() }
                else if st.all_fast { format!("fast path on, {}/{} experts resident", st.resident, st.total) }
                else { format!("fast path off, {}/{} experts resident", st.resident, st.total) };
            say!("mailbox traffic: {} GPU writes, {} host routing reads ({mode})", sum(&|p| p.mailbox_writes.get()), sum(&|p| p.mailbox_reads.get()));
            for p in g.parts() { if let Some(c) = &p.cache { c.save_profile()?; } }
        }
        Ok(())
    }
}

impl Engine {
    /// Hand `--backend`, `--device`, and `--kv` to the DeepSeek, GLM, Gemma 4 and Inkling 2 paths.
    fn ds_env(&self) {
        // SAFETY: called at the start of a command, before any thread reads these variables.
        if let Some(b) = &self.backend { unsafe { std::env::set_var("STRATA_DS_BACKEND", b) } }
        if self.device != [0] { unsafe { std::env::set_var("STRATA_DS_DEVICES", self.device.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",")) } }
        if self.fixed_experts { unsafe { std::env::set_var("STRATA_FIXED_EXPERTS", "1") } }  // Hy4: every expert on the GPU
        if self.kv != "f16" && std::env::var_os("STRATA_KV").is_none() { set_kv_env(&self.kv); }
    }

    /// Fills settings not given on the command line from the model's calibration file, else built-in defaults.
    fn ram_cap(&self) -> Option<u64> { memory::ram_cap((self.ram_margin as u64) << 20, (self.max_ram as u64) << 20) }
    /// A copy with the load options of an `/api/load` body applied (`ctx`, `kv`, `cache_ram`, `vram_margin`, `mtp`,
    /// `chunk`); a field that is absent or null keeps its value.
    fn with(&self, v: &serde_json::Value) -> anyhow::Result<Engine> {
        let num = |k: &str, lo: u64, hi: u64| -> anyhow::Result<Option<usize>> {
            if v[k].is_null() { return Ok(None); }
            let n = v[k].as_u64().filter(|n| (lo..=hi).contains(n)).with_context(|| format!("{k} must be a whole number from {lo} to {hi}"))?;
            Ok(Some(n as usize))
        };
        let mut e = self.clone();
        if let Some(n) = num("ctx", 256, 1 << 20)? { e.ctx = n; }
        if let Some(n) = num("cache_ram", 0, 1 << 20)? { e.cache_ram = n; }
        if let Some(n) = num("vram_margin", 0, 1 << 20)? { e.vram_margin = vec![n.to_string()]; }
        if let Some(n) = num("mtp", 0, 8)? { e.mtp = Some(n); }
        if let Some(n) = num("chunk", 0, 1 << 20)? { e.chunk = Some(n); }
        if !v["kv"].is_null() { e.kv = v["kv"].as_str().filter(|k| ["f16", "q8", "q4", "q2"].contains(k)).context("kv must be f16, q8, q4 or q2")?.into(); }
        Ok(e)
    }

    fn runner<'a>(&mut self, m: &'a Model<'a>, model: &Path) -> anyhow::Result<Runner<'a>> {
        if let Some(cap) = self.ram_cap() {
            let used = m.ram_estimate() as u64;
            anyhow::ensure!(used <= cap, "model needs ~{} MB of host RAM (baked tensors, LoRA), over the {} MB cap; lower --ram-margin or raise --max-ram", used >> 20, cap >> 20);
            let want = (self.cache_ram as u64) << 20;
            let room = cap - used;
            if want > room {
                say!("RAM cap: clamping --cache-ram from {} MB to {} MB", self.cache_ram, room >> 20);
                self.cache_ram = (room >> 20) as usize;
            }
        }
        if self.llama_numerics { unsafe { std::env::set_var("STRATA_LLAMA_NUMERICS", "1") } }  // read by Gpu::new
        set_kv_env(&self.kv);  // read by Gpu::new and Vk::new
        let open = || self.device.iter().map(|&d| backend::open(self.backend.as_deref(), d)).collect::<anyhow::Result<Vec<_>>>();
        let bk = match self.gpu.then(open).transpose() {
            Err(e) if self.cpu_fallback => { say!("no GPU backend ({e:#}), running on the CPU"); None }
            r => r?,
        };
        let path = calibration_path(model, bk.as_ref().map_or("cuda", |b| b[0].name()), &self.device);
        let mut v: serde_json::Value = std::fs::read(&path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
        if let Some(gpu) = bk.as_ref().map(|b| b[0].device()).filter(|gpu| !calibration_fits(&v, gpu)) {
            say!("warning: {} was measured on {}, not on {gpu}: ignored", path.display(), v["gpu"]);
            v = serde_json::Value::Null;
        }
        if let Some(n) = v["bytes"].as_u64().filter(|&n| std::fs::metadata(model).is_ok_and(|m| m.len() != n)) {
            say!("warning: {} was measured on another file with this name ({n} bytes): ignored", path.display());
            v = serde_json::Value::Null;
        }
        if !v.is_null() { say!("calibration: {}", path.display()); }
        else if bk.is_some() { say!("warning: not calibrated on this machine, so the prefill chunk, MTP drafts and the short-prompt threshold are defaults: run `rustingstrata calibrate --model {}` (a few minutes, once per model)", model.display()); }
        let get = |k: &str| v[k].as_u64().map(|x| x as usize);
        if let Some(dp) = &self.draft_model {
            anyhow::ensure!(dp.exists(), "draft model path does not exist: {}", dp.display());
            say!("draft model socket: registered candidate {}", dp.display());
        }
        let mut ngram_cfg = self.ngram_config()?;
        ngram_cfg.n_vocab = m.cfg.n_vocab;
        self.chunk.get_or_insert(get("chunk").unwrap_or(4096));
        // The draft count was measured with this draft file (`calibrate --mtp-model/--draft-model`), or with none.
        let aux = self.aux_model_path().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned());
        let cal_mtp = get("mtp").filter(|_| v["aux"].as_str() == aux.as_deref());
        if aux.is_some() && self.mtp.is_none() { self.mtp = cal_mtp; }
        if let Some(df) = &m.cfg.dflash {
            if self.mtp.is_none() {
                self.mtp = Some(df.block_size.saturating_sub(1).max(1));
            }
        } else if m.draft_model.is_some() && self.mtp.is_none() {
            self.mtp = Some(4);
        } else if (self.mtp_model.is_some() || self.draft_model.is_some()) && self.mtp.is_none() {
            self.mtp = Some(1);
        } else if ngram_cfg.is_enabled() && self.mtp.is_none() {
            self.mtp = Some(0);
        } else {
            self.mtp.get_or_insert(cal_mtp.unwrap_or(0));
        }
        let short = self.short.unwrap_or_else(|| get("short").unwrap_or(32));
        // The global pool can be built once; a model loaded later by `serve` keeps the first one's threads.
        // GPU expert work is memory-bound, so SMT siblings only add contention (6 threads beat 12 on the reference
        // 6-core/12-thread Ryzen). CPU-only inference keeps every logical core.
        let gpu_threads = bk.is_some().then(physical_cores).flatten();
        if let Some(n) = self.threads.or(get("threads")).or(gpu_threads) {
            let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
        }
        let aux_heads = m.cfg.n_nextn.max(usize::from(m.cfg.dflash.is_some() || m.draft_model.is_some()));
        let Some(bk) = bk else {
            let (notes, _, mtp) = resolve_flags(false, self.parallel, self.mtp(), aux_heads)?;
            for n in notes { say!("{n}"); }
            self.mtp = Some(mtp);
            return Ok(Runner::Cpu(m, State::new(&m.cfg)));
        };
        let (notes, parallel, mtp) = resolve_flags(true, self.parallel, self.mtp(), aux_heads)?;
        for n in notes { say!("{n}"); }
        (self.parallel, self.mtp) = (parallel, Some(mtp));
        if self.fixed_experts { unsafe { std::env::set_var("STRATA_FIXED_EXPERTS", "1") } }  // read by ExpertCache::new
        // Before the first weight is uploaded: fail if a pipeline these weight types need does not fit the device's
        // shared memory (Vulkan; a no-op elsewhere). No pipeline or model buffer exists yet.
        let mut types = rusting_strata::gpu_model::gpu_weight_types(&m.w.g);
        if let Some(dm) = &m.draft_model {
            for t in rusting_strata::gpu_model::gpu_weight_types(&dm.w.g) {
                if !types.contains(&t) {
                    types.push(t);
                }
            }
        }
        for b in &bk { b.preflight(&types, false)?; }
        let margins = vram_margins(&self.vram_margin, &self.device)?.into_iter().map(|mb| mb << 20).collect::<Vec<_>>();
        if (ngram_cfg.is_enabled() || m.cfg.dflash.is_some() || m.draft_model.is_some()) && self.parallel <= 1 {
            unsafe { std::env::set_var("STRATA_SPEC_SNAP", "1") };
        } else {
            unsafe { std::env::remove_var("STRATA_SPEC_SNAP") };
        }
        // The layer split: `--split`, else the one `calibrate` measured as fastest for these devices, else by free VRAM.
        let split = match split_arg(self.split.as_deref(), self.device.len())? {
            Some(s) => Some(s),
            None => calibrated_split(&v, &self.device, &bk.iter().map(|b| b.device()).collect::<Vec<_>>()),
        };
        let mut g = if self.parallel > 1 {
            GpuModel::load_split_margins(m, bk, self.ctx, self.chunk(), &margins, split.as_deref(), 0, self.parallel).context(LOAD_HINT)?
        } else {
            GpuModel::load_split_margins(m, bk, self.ctx, self.chunk(), &margins, split.as_deref(), mtp, 1).context(LOAD_HINT)?
        };
        if ngram_cfg.is_enabled() && self.parallel <= 1 {
            let tok = Tokenizer::from_gguf(&m.w.g).ok();
            g.ngram = Some(rusting_strata::ngram::NgramDrafter::new(ngram_cfg, tok.as_ref())?);
        }
        g.short = short;
        g.cache_ram = self.cache_ram << 20;
        if let Some(p) = &self.prompt_cache {
            if g.cache_ram == 0 {
                g.cache_ram = 2048 << 20;
            }
            if let Err(e) = g.load_prompt_cache(p) {
                say!("warning: could not load prompt cache {}: {e:#}", p.display());
            }
        }
        Ok(Runner::Gpu(Box::new(g)))
    }
    fn ngram_config(&self) -> anyhow::Result<rusting_strata::ngram::NgramConfig> {
        let types = rusting_strata::ngram::SpecType::parse_list(&self.spec_type)?;
        anyhow::ensure!((1..=rusting_strata::cpu_worker::TMAX - 1).contains(&self.spec_draft_n_max),
            "--spec-draft-n-max {}: must be between 1 and {}", self.spec_draft_n_max, rusting_strata::cpu_worker::TMAX - 1);
        Ok(rusting_strata::ngram::NgramConfig {
            types,
            draft_max: self.spec_draft_n_max,
            size_n: self.spec_ngram_size_n.max(1),
            size_m: self.spec_ngram_size_m.max(1),
            min_hits: self.spec_ngram_min_hits.max(1),
            mod_n_match: self.spec_ngram_size_n.max(1),
            mod_n_min: 2,
            lookup_cache_static: self.lookup_cache_static.clone(),
            lookup_cache_dynamic: self.lookup_cache_dynamic.clone(),
            n_vocab: 0,
        })
    }
    fn sampler(&self) -> Sampler {
        Sampler { temp: self.temp, top_k: self.top_k, top_p: self.top_p, rng: seed(self.seed),
            presence: self.presence_penalty, frequency: self.frequency_penalty, repeat: self.repeat_penalty,
            last_n: self.penalty_last_n, dry_mult: self.dry_multiplier,
            dry_base: self.dry_base, dry_allowed: self.dry_allowed_length, ..Sampler::greedy() }
    }
    fn chunk(&self) -> usize { self.chunk.unwrap_or(4096) }
    fn mtp(&self) -> usize { self.mtp.unwrap_or(0) }
    fn aux_model_path(&self) -> Option<&Path> { self.draft_model.as_deref().or(self.mtp_model.as_deref()) }

    /// Says which draft options do nothing for a model outside the main engine (`config::OWN_ARCHS`).
    fn own_warn(&self, g: &Gguf) {
        let Some(arch) = g.get("general.architecture").ok().and_then(|v| v.as_str()) else { return };
        if !rusting_strata::config::OWN_ARCHS.contains(&arch) { return; }
        let mtp = self.mtp.is_some_and(|n| n > 0) || self.mtp_model.is_some();
        // Inkling 2 drafts with a DSpark file (`--draft-model`) and with n-grams (`--spec-type`); it has no MTP head.
        if rusting_strata::inkling2::is_arch(arch) {
            if mtp { say!("warning: '{arch}' models have no MTP drafts: --mtp and --mtp-model are ignored"); }
        } else if mtp || self.draft_model.is_some() || self.spec_type != "none" {
            say!("warning: '{arch}' models decode without drafts: --mtp, --mtp-model, --draft-model and --spec-type are ignored");
        }
    }
}

/// Per-device VRAM margins in MB from `--vram-margin`: one value for every device, one per device like `512,256`,
/// or one per GPU index like `0:512,1:256`. The result is in the order of `devices`.
fn vram_margins(spec: &[String], devices: &[usize]) -> anyhow::Result<Vec<usize>> {
    let parse = |s: &str| -> anyhow::Result<(Option<usize>, usize)> {
        let (idx, mb) = match s.split_once(':') {
            Some((i, m)) => (Some(i.trim().parse().with_context(|| format!("--vram-margin {s}: `{}` is not a GPU index", i.trim()))?), m),
            None => (None, s),
        };
        let mb = mb.trim().parse().with_context(|| format!("--vram-margin {s}: `{}` is not a value in MB", mb.trim()))?;
        Ok((idx, mb))
    };
    let vals = spec.iter().map(|s| parse(s)).collect::<anyhow::Result<Vec<_>>>()?;
    anyhow::ensure!(!vals.is_empty(), "--vram-margin needs at least one value");
    if vals.len() == 1 && vals[0].0.is_none() {
        return Ok(vec![vals[0].1; devices.len()]);
    }
    if vals.iter().all(|v| v.0.is_none()) {
        anyhow::ensure!(vals.len() == devices.len(), "--vram-margin {}: {} value(s) for {} device(s) (give one value, \
            or one per device in --device order)", spec.join(","), vals.len(), devices.len());
        return Ok(vals.into_iter().map(|v| v.1).collect());
    }
    anyhow::ensure!(vals.iter().all(|v| v.0.is_some()), "--vram-margin {}: mixing `IDX:MB` and plain MB values", spec.join(","));
    let mut out = vec![0; devices.len()];
    let mut seen = vec![false; devices.len()];
    for (idx, mb) in vals {
        let idx = idx.unwrap();
        let pos = devices.iter().position(|&d| d == idx)
            .with_context(|| format!("--vram-margin: GPU {idx} is not in --device"))?;
        anyhow::ensure!(!seen[pos], "--vram-margin: GPU {idx} is given twice");
        seen[pos] = true;
        out[pos] = mb;
    }
    if let Some(pos) = seen.iter().position(|&s| !s) {
        anyhow::bail!("--vram-margin: no value for GPU {} (give every device one)", devices[pos]);
    }
    Ok(out)
}

/// A projector opened once and the images encoded with it.
struct ImageInput {
    grids: Vec<(usize, usize)>,
    embeddings: Vec<Vec<Vec<f32>>>,
}

/// Physical cores, not SMT siblings: the CPU expert workers are memory-bound, and oversubscribing them with the
/// hyperthreads slows the CPU/GPU token loop. Linux reads the CPU topology; other platforms keep every logical CPU.
fn physical_cores() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        let groups = cpu_topology_groups(std::path::Path::new("/sys/devices/system/cpu"));
        let avail = std::thread::available_parallelism().map_or(0, |n| n.get());
        if groups > 0 {
            return Some(if avail > 0 { groups.min(avail) } else { groups });
        }
    }
    None
}

/// Distinct `thread_siblings_list` values under `dir` (one per physical core).
#[cfg(target_os = "linux")]
fn cpu_topology_groups(dir: &std::path::Path) -> usize {
    let groups: std::collections::BTreeSet<String> = std::fs::read_dir(dir).into_iter().flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name();
            let n = name.to_str()?.strip_prefix("cpu")?;
            if n.is_empty() || !n.bytes().all(|b| b.is_ascii_digit()) { return None; }
            std::fs::read_to_string(e.path().join("topology/thread_siblings_list")).ok().map(|s| s.trim().to_string())
        })
        .collect();
    groups.len()
}

/// Decode and encode every `--image` with the projector at `mmproj`.
fn load_images(mmproj: &Path, images: &[String]) -> anyhow::Result<ImageInput> {
    let projector = VisionProjector::open(mmproj)?;
    let mut grids = Vec::with_capacity(images.len());
    let mut embeddings = Vec::with_capacity(images.len());
    for src in images {
        let img = vision::load_image(src)?;
        let t0 = Instant::now();
        let (feats, nx, ny) = vision::encode_image(&projector, &img)?;
        say!("image: {nx}x{ny} tokens encoded in {:.1} s", t0.elapsed().as_secs_f64());
        grids.push((nx, ny));
        embeddings.push(feats);
    }
    Ok(ImageInput { grids, embeddings })
}

/// Expand `<image>` placeholders into the projector's vision tokens and assign M-RoPE positions.
fn build_image_prompt(tok: &Tokenizer, input: &ImageInput, images: &[String], prompt: &str) -> anyhow::Result<vision::Prompt> {
    let (start, end, pad) = (
        tok.token_id("<|vision_start|>").context("tokenizer has no <vision_start|>")?,
        tok.token_id("<|vision_end|>").context("tokenizer has no <vision_end|>")?,
        tok.token_id("<|image_pad|>").context("tokenizer has no <image_pad|>")?,
    );
    let tokens = "<|vision_start|><|image_pad|><|vision_end|>";
    let n_mark = prompt.matches("<image>").count();
    let text = if n_mark == 0 {
        format!("{}{}", tokens.repeat(images.len()), prompt)
    } else {
        anyhow::ensure!(n_mark == images.len(), "prompt has {n_mark} <image> placeholder(s) for {} --image(s)", images.len());
        prompt.replace("<image>", tokens)
    };
    let ids = tok.encode(&text)?;
    vision::expand_images(&ids, start, end, pad, &input.grids)
}

/// The negative log-likelihood of `tok` under `logits`, and in `row` the log-probabilities as llama-perplexity stores
/// them: scale and minimum as two f32, then each logit as a u16 step over the 16 nats below the largest (0 below
/// that), padded to an even count. Same arithmetic as its `log_softmax`.
fn log_probs(logits: &[f32], tok: u32, row: &mut Vec<u16>) -> f64 {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let min = logits.iter().fold(f32::INFINITY, |a, &b| a.min(b)).max(max - 16.0);
    let lse = logits.iter().map(|&x| (x - max).exp() as f64).sum::<f64>().ln() as f32;
    let scale = (max - min) / 65535.0;
    row.clear();
    for f in [scale, min - max - lse] { let b = f.to_bits(); row.extend([b as u16, (b >> 16) as u16]); }
    row.extend(logits.iter().map(|&x| if scale != 0.0 && x > min { ((x - min) * (1.0 / scale)).round_ties_even() as u16 } else { 0 }));
    if logits.len() % 2 == 1 { row.push(0); }
    (max + lse - logits[tok as usize]) as f64
}

/// A nonzero xorshift state from `s`, or from the clock.
fn seed(s: Option<u64>) -> u64 {
    s.unwrap_or_else(|| SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64))
        .wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1
}

const LOAD_HINT: &str = "loading the model onto the GPU failed (if VRAM ran out: lower --ctx, add --kv q8, or close programs using VRAM)";

/// Settle the flag combinations that cannot all take effect: an invalid request (`--parallel` outside 1..=255) is an
/// error; a feature that cannot run is turned off with one message naming the flag and the reason. `parallel` 0 means
/// "not set" (commands other than `serve`), treated as one slot. Returns the notices, the slots in effect and the MTP
/// drafts in effect.
fn resolve_flags(gpu: bool, parallel: usize, mtp: usize, n_nextn: usize) -> anyhow::Result<(Vec<String>, usize, usize)> {
    let p = parallel.max(1);
    anyhow::ensure!((1..=255).contains(&p), "--parallel {parallel}: 1 to 255 slots");
    let mut notes = vec![];
    if !gpu {
        if p > 1 { notes.push(format!("--parallel {p}: requires --gpu (the CPU answers one request at a time)")); }
        if mtp > 0 { notes.push(format!("--mtp {mtp}: requires --gpu: decoding without drafts")); }
        return Ok((notes, p, 0));
    }
    if p > 1 && mtp > 0 {
        notes.push(format!("--parallel {p} and --mtp {mtp}: not supported together, decoding without MTP drafts"));
        return Ok((notes, p, 0));
    }
    let why = if mtp == 0 { None }
        else if n_nextn == 0 { Some("this model has no MTP head") }
        else { None };
    let mut k = mtp;
    if let Some(why) = why { notes.push(format!("--mtp {mtp}: {why}: decoding without drafts")); k = 0; }
    Ok((notes, p, k))
}

/// Calibration file for `model` on `backend` and `devices`; CUDA keeps the original name, and a run on several
/// devices has its own file (`-dev0-1`), so it does not replace the one-GPU calibration.
fn calibration_path(model: &Path, backend: &str, devices: &[usize]) -> PathBuf {
    let stem = model.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    let mut sfx = if backend == "cuda" { String::new() } else { format!("-{backend}") };
    if devices.len() > 1 { sfx += &format!("-dev{}", devices.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("-")); }
    rusting_strata::expert_cache::cache_dir().join(format!("strata-{stem}{sfx}.json"))
}

/// The layer shares of `--split` or, without the flag, of `STRATA_SPLIT`, for `n` devices (`None`: not given, or one
/// device).
fn split_arg(flag: Option<&str>, n: usize) -> anyhow::Result<Option<Vec<f64>>> {
    let Some(s) = flag.map(String::from).or_else(|| std::env::var("STRATA_SPLIT").ok()).filter(|s| !s.is_empty() && n > 1) else { return Ok(None) };
    rusting_strata::gpu_model::split_shares(&s, n).map(Some)
}

/// The layers per device that `calibrate` chose, if file `v` has them for these devices: the same GPU indices and
/// names (`Backend::device`) in the same `--device` order.
fn calibrated_split(v: &serde_json::Value, devices: &[usize], gpus: &[String]) -> Option<Vec<f64>> {
    let s = &v["split"];
    if s["devices"] != json!(devices) || s["gpus"] != json!(gpus) { return None; }
    let l: Vec<f64> = s["layers"].as_array()?.iter().filter_map(|x| x.as_f64()).filter(|&x| x > 0.0).collect();
    (l.len() == devices.len()).then_some(l)
}

/// The splits one move away from `cur` (layers per device): `step` layers moved between two neighbouring devices,
/// each device keeping at least one layer.
fn split_moves(cur: &[usize], step: usize) -> Vec<Vec<usize>> {
    let mut v = vec![];
    for i in 0..cur.len().saturating_sub(1) {
        for (a, b) in [(i, i + 1), (i + 1, i)] {
            if cur[a] > step {
                let mut c = cur.to_vec();
                (c[a], c[b]) = (c[a] - step, c[b] + step);
                v.push(c);
            }
        }
    }
    v
}

/// Whether calibration file `v` may be used on `gpu` (`Backend::device`). A file names the GPU it was measured on;
/// one without a name (written before that was recorded, or the driver gave none) is accepted.
fn calibration_fits(v: &serde_json::Value, gpu: &str) -> bool {
    v["gpu"].as_str().is_none_or(|g| g.is_empty() || gpu.is_empty() || g == gpu)
}

impl Engine {
    pub fn lora_specs(&self) -> Vec<(String, f32)> {
        self.lora.iter().map(|s| {
            if let Some(idx) = s.rfind(':') {
                if let Ok(scale) = s[idx+1..].parse::<f32>() {
                    return (s[..idx].to_string(), scale);
                }
            }
            (s.clone(), 1.0)
        }).collect()
    }
}

/// Expert-cache counters summed over the GPU parts (`Runner::finish` and the benchmark JSON share them).
#[derive(Debug, Clone, Copy, PartialEq)]
struct ExpertStats {
    hits: usize,
    misses: usize,
    swaps: usize,
    resident: usize,
    total: usize,
    bytes_per_token: usize,
    /// At least one part has an expert cache.
    cached: bool,
    /// Every part with a cache takes the fully-resident fast path.
    all_fast: bool,
}

/// Round to the one decimal the text block prints, so the JSON carries exactly those digits (no second, different
/// rounding of the raw float).
fn round1(x: f64) -> f64 {
    format!("{x:.1}").parse().unwrap_or(x)
}

/// Mean/min/max of a benchmark series. The text block and the JSON record both read these fields, so the numbers
/// cannot drift apart.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Stat {
    mean: f64,
    min: f64,
    max: f64,
}

impl Stat {
    fn of(v: &[f64]) -> Self {
        if v.is_empty() { return Self { mean: 0.0, min: 0.0, max: 0.0 }; }
        // The same fold as the text block always used, so formatting cannot change any printed digit.
        let (lo, hi) = v.iter().fold((f64::INFINITY, 0f64), |(a, b), &x| (a.min(x), b.max(x)));
        Self { mean: round1(v.iter().sum::<f64>() / v.len() as f64), min: round1(lo), max: round1(hi) }
    }
    /// The `MEAN (MIN-MAX)` field of the `pp …` line.
    fn text(&self) -> String { format!("{:.1} ({:.1}-{:.1})", self.mean, self.min, self.max) }
}

/// The run-wide values of a `strata bench` invocation (`strata.bench.v1`).
struct BenchMeta<'a> {
    backend: &'a str,
    backend_arg: Option<&'a str>,
    device: &'a [usize],
    model: &'a Path,
    ctx: usize,
    chunk: usize,
    kv: &'a str,
    mtp: usize,
    tg: usize,
    reps: usize,
}

/// One `--pp` length: the two `Stat`s the text prints and what the expert record needs.
struct BenchResult {
    pp: usize,
    prefill: Stat,
    decode: Stat,
    /// Bytes of expert weights one decode token reads (0 without an expert cache: no `expert` object).
    bytes_per_token: usize,
    /// The same median decode rate the `expert traffic:` line prints.
    decode_median: f64,
}

/// The `strata.bench.v1` record. Pure so it can be tested without a model or a GPU.
fn bench_record(meta: &BenchMeta, results: &[BenchResult], expert: Option<&ExpertStats>, unix_time: u64) -> serde_json::Value {
    let mut v = json!({
        "schema": "strata.bench.v1",
        "unix_time": unix_time,
        "backend": meta.backend,
        "device": meta.device,
        "model": {
            "path": meta.model.display().to_string(),
            "name": meta.model.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
        },
        "ctx": meta.ctx,
        "chunk": meta.chunk,
        "kv": meta.kv,
        "mtp": meta.mtp,
        "tg": meta.tg,
        "reps": meta.reps,
    });
    if let Some(b) = meta.backend_arg { v["backend_arg"] = json!(b); }
    let results: Vec<_> = results.iter().map(|r| {
        let mut o = json!({
            "pp": r.pp,
            "prefill_tok_s": {"mean": r.prefill.mean, "min": r.prefill.min, "max": r.prefill.max},
            "decode_tok_s": {"mean": r.decode.mean, "min": r.decode.min, "max": r.decode.max},
        });
        if r.bytes_per_token > 0 {
            o["expert"] = json!({
                "bytes_per_token": r.bytes_per_token,
                "mb_per_token": round1(r.bytes_per_token as f64 / 1048576.0),
                "gb_per_s": round1(r.bytes_per_token as f64 * r.decode_median / 1e9),
                "at_decode_tok_s": round1(r.decode_median),
            });
        }
        o
    }).collect();
    v["results"] = json!(results);
    if let Some(e) = expert {
        v["expert"] = json!({
            "hit_rate_pct": round1(100.0 * e.hits as f64 / (e.hits + e.misses).max(1) as f64),
            "misses": e.misses,
            "swaps": e.swaps,
            "resident": e.resident,
            "total": e.total,
            "bytes_per_token": e.bytes_per_token,
            "mb_per_token": round1(e.bytes_per_token as f64 / 1048576.0),
        });
    }
    v
}

/// `--json-out` values: a file path, never `-` (stdout keeps the text block).
fn json_out_path(s: &str) -> Result<PathBuf, String> {
    if s == "-" {
        Err("`-` is not a file: stdout carries the text block, give a path".into())
    } else {
        Ok(PathBuf::from(s))
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Print model config and tensor type histogram.
    Info {
        #[arg(long)]
        model: PathBuf,
    },
    /// Teacher-force token ids; print argmax and top-5 logprobs for each continuation position.
    /// Input JSON: {"prompt": [ids], "continuation": [ids]} ("-" reads stdin).
    Score {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: String,
        #[command(flatten)]
        engine: Engine,
    },
    /// Interactive chat on stdin (greedy unless --temp; on the GPU each turn prefills only what follows the previous turn's messages).
    Chat {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        system: Option<String>,
        /// Max tokens per reply.
        #[arg(short, default_value_t = 2048)]
        n: usize,
        /// Image path, data URL or http(s) URL (repeatable), attached to the conversation. Needs --mmproj (or a
        /// sibling mmproj next to the model).
        #[arg(long, action = clap::ArgAction::Append)]
        image: Vec<String>,
        #[command(flatten)]
        engine: Engine,
    },
    /// OpenAI-compatible HTTP server: POST /v1/chat/completions (streaming or not), GET /v1/models.
    /// One request at a time; temperature, top_k, top_p and seed are honored, greedy when left out. A request that
    /// extends the previous one's messages prefills only the new part (usage.prompt_tokens_details.cached_tokens);
    /// one that keeps only the system prompt and tools (a compacted conversation) starts after them.
    /// GET / is a chat page that can also switch between the models in `--models-dir` (POST /api/load), so the
    /// server can start without `--model`.
    Serve {
        #[arg(long)]
        model: Option<PathBuf>,
        /// Folder of .gguf models the chat page offers [default: $XDG_DATA_HOME/rusting_strata/models,
        /// %LOCALAPPDATA%\rusting_strata\models on Windows, else ~/.local/share/rusting_strata/models].
        #[arg(long)]
        models_dir: Option<PathBuf>,
        #[arg(long, default_value = "127.0.0.1:8080")]
        addr: String,
        /// Open the chat page in the browser (if the address is taken, open it and exit: strata already runs there).
        #[arg(long)]
        open: bool,
        /// Require `Authorization: Bearer <key>` on every request [default: $STRATA_API_KEY, which keeps the key out
        /// of the process list].
        #[arg(long)]
        api_key: Option<String>,
        /// MCP servers whose tools the chat page may call: a JSON file `{"mcpServers": {"name": {"command": ..,
        /// "args": [..], "env": {..}} | {"url": .., "headers": {..}}}}` [default: $STRATA_MCP, else mcp.json beside the
        /// default models folder]. The servers start when the page first asks for them.
        #[arg(long)]
        mcp: Option<PathBuf>,
        #[command(flatten)]
        host: Hosting,
        #[command(flatten)]
        engine: Engine,
    },
    /// Measure prefill chunk, MTP drafts per step and the short-prompt threshold on this machine (GPU) and write
    /// them to the model's calibration file, which later runs read unless flags override.
    Calibrate {
        /// The GGUF to calibrate, or a folder: every model in it (and its subfolders, as the chat page lists them)
        /// is calibrated in turn.
        #[arg(long)]
        model: PathBuf,
        /// GPU context length used while measuring and assumed for the results.
        #[arg(long, default_value_t = 4096)]
        ctx: usize,
        /// VRAM left free per device, as `--vram-margin`.
        #[arg(long, default_value = "512", value_delimiter = ',')]
        vram_margin: Vec<String>,
        #[arg(long)]
        backend: Option<String>,
        /// GPU index, or several separated by commas (such as 0,1) to calibrate the layer split over those GPUs.
        #[arg(long, default_value = "0", value_delimiter = ',')]
        device: Vec<usize>,
        /// Share of the layers per device, as `--split` of the other commands: calibrate with this split and do not
        /// measure others.
        #[arg(long)]
        split: Option<String>,
        /// Accepted for consistency with the other commands; calibration always uses the base model.
        #[arg(long, value_name = "PATH:SCALE", action = clap::ArgAction::Append)]
        lora: Vec<String>,
        /// KV cache format to calibrate with (q8/q4/q2 leave more VRAM for experts), as `--kv` of the runs that use it.
        #[arg(long, default_value = "f16", value_parser = ["f16", "q8", "q4", "q2"])]
        kv: String,
        /// Accepted for symmetry with the other commands: calibration always runs on the GPU.
        #[arg(long, hide = true)]
        gpu: bool,
        /// Standalone MTP head GGUF, as `--mtp-model` of the runs: calibrate the drafts with it.
        #[arg(long)]
        mtp_model: Option<PathBuf>,
        /// Draft model (DFlash, EAGLE3 or a small model of the same vocabulary), as `--draft-model` of the runs:
        /// calibrate the drafts with it.
        #[arg(long)]
        draft_model: Option<PathBuf>,
    },
    /// Check that every Vulkan pipeline this model needs fits the device's `maxComputeSharedMemorySize`, naming the
    /// kernel, specialization, bytes needed and limit when one does not; exits non-zero on overflow. No model is
    /// loaded, no weight uploaded and no pipeline created: it is SPIR-V plus spec-constant arithmetic only.
    Check {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        backend: Option<String>,
        #[arg(long, default_value_t = 0)]
        device: usize,
    },
    /// Speed test: for each prompt length, prefill that many tokens of prose, then greedily decode `tg` more at that
    /// depth. Prints prefill and decode tok/s as mean (min-max) over `reps` runs. `--ctx` grows to fit.
    /// `--json-out PATH` also writes the same numbers as one `strata.bench.v1` JSON object plus a newline (`-` is
    /// rejected: stdout keeps the text block).
    Bench {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "512,4096")]
        pp: Vec<usize>,
        #[arg(long, default_value_t = 128)]
        tg: usize,
        #[arg(long, default_value_t = 3)]
        reps: usize,
        #[arg(long, value_parser = json_out_path)]
        json_out: Option<PathBuf>,
        /// Prefill this text file instead of the built-in paragraph (repeated when shorter than `--pp`). The built-in
        /// prose routes few experts; a real prompt shows what a first request costs.
        #[arg(long)]
        prompt_file: Option<PathBuf>,
        #[command(flatten)]
        engine: Engine,
    },
    /// Perplexity over a text file as llama-perplexity computes it: chunks of `ctx-len` tokens, each from an empty
    /// context, scored over their second half (the first half prefilled, the rest decoded token by token).
    /// `--kld-base` writes the log-probabilities in llama-perplexity's format, so
    /// `llama-perplexity -m MODEL --kl-divergence --kl-divergence-base FILE` compares llama.cpp's logits with these
    /// (KLD, top-token agreement and both perplexities) on the same tokens.
    Perplexity {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        file: PathBuf,
        #[arg(long, default_value_t = 512)]
        ctx_len: usize,
        /// Score only the first N chunks.
        #[arg(long)]
        chunks: Option<usize>,
        #[arg(long)]
        kld_base: Option<PathBuf>,
        #[command(flatten)]
        engine: Engine,
    },
    /// Ask for the model, the mode and the main settings in the terminal, then run the command they make.
    Interface {
        /// Folder of .gguf models to list, as `--models-dir` of `serve`.
        #[arg(long)]
        models_dir: Option<PathBuf>,
    },
    /// Text generation (greedy unless --temp).
    Generate {
        #[arg(long)]
        model: PathBuf,
        /// Prompt text ("-" reads stdin; long prompts exceed the 128 KiB argument limit).
        #[arg(long)]
        prompt: String,
        #[arg(short, default_value_t = 32)]
        n: usize,
        /// Image path, data URL or http(s) URL (repeatable). Each is inserted where the prompt has `<image>`;
        /// without placeholders all images are prepended. Needs --mmproj (or a sibling mmproj next to the model).
        #[arg(long, action = clap::ArgAction::Append)]
        image: Vec<String>,
        #[command(flatten)]
        engine: Engine,
    },
}

fn main() {
    if let Err(e) = start() {
        say!("error: {e:#}");
        std::process::exit(1);
    }
}

fn start() -> anyhow::Result<()> {
    let m = <Cli as clap::CommandFactory>::command().get_matches();
    if let Some((_, sm)) = m.subcommand() && let Ok(Some(&n)) = sm.try_get_one::<usize>("ctx")
        && sm.value_source("ctx") == Some(clap::parser::ValueSource::CommandLine) {
        rusting_strata::CTX.store(n, std::sync::atomic::Ordering::Relaxed);
    }
    let Some(cmd) = <Cli as clap::FromArgMatches>::from_arg_matches(&m).unwrap_or_else(|e| e.exit()).cmd else {
        // Started without arguments, as by a double click: keep the window open on an error so it can be read.
        let Some(Cmd::Serve { model, models_dir, addr, api_key, mcp, open, host, mut engine }) = Cli::parse_from(["rustingstrata", "serve", "--gpu", "--open"]).cmd else { unreachable!() };
        engine.cpu_fallback = true;
        let r = run(Cmd::Serve { model, models_dir, addr, api_key, mcp, open, host, engine });
        if let Err(e) = &r {
            say!("error: {e:#}\npress Enter to close");
            let _ = std::io::stdin().read_line(&mut String::new());
            std::process::exit(1);
        }
        return r;
    };
    run(cmd)
}

fn run(cmd: Cmd) -> anyhow::Result<()> {
    // The CPU kernels are chosen at compile time. A build without AVX2 (RUSTFLAGS in the environment replaces
    // .cargo/config.toml's target-cpu=native) runs MoE experts that miss the VRAM cache about 10x slower (#44).
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    if std::arch::is_x86_feature_detected!("avx2") {
        say!("warning: this binary was built without AVX2 but this CPU has it: CPU work is about 10x slower. Rebuild with \
                   RUSTFLAGS=\"-C target-cpu=native\" (or x86-64-v3), or use a release binary.");
    }
    match cmd {
        Cmd::Info { model } => {
            let g = Gguf::open(&model)?;
            println!("{:#?}", ModelConfig::from_gguf(&g)?);
            let mut hist = BTreeMap::<String, usize>::new();
            for t in g.tensors.values() {
                let name = GgmlType::from_u32(t.ty).map(|ty| ty.name().to_string()).unwrap_or(format!("type{}", t.ty));
                *hist.entry(name).or_default() += 1;
            }
            for (ty, n) in hist { println!("{n:5} {ty}"); }
        }
        Cmd::Score { model, input, mut engine } => {
            let text = if input == "-" {
                let mut s = String::new();
                std::io::stdin().read_to_string(&mut s)?;
                s
            } else {
                std::fs::read_to_string(&input)?
            };
            let v: serde_json::Value = serde_json::from_str(&text)?;
            let ids = |k: &str| -> anyhow::Result<Vec<u32>> {
                v[k].as_array().with_context(|| format!("missing {k}"))?
                    .iter().map(|x| x.as_u64().and_then(|x| u32::try_from(x).ok()).context("bad id")).collect()
            };
            let (prompt, cont) = (ids("prompt")?, ids("continuation")?);
            anyhow::ensure!(!prompt.is_empty() && !cont.is_empty(), "empty prompt or continuation");
            let g = Gguf::open(&model)?;
            let aux_g = open_mtp_gguf(engine.aux_model_path())?;
            let m = Model::load_with_aux(&g, aux_g.as_ref(), &engine.lora_specs())?;
            if let Some(&t) = prompt.iter().chain(&cont).find(|&&t| t as usize >= m.cfg.n_vocab) {
                anyhow::bail!("token id {t} out of range (vocabulary has {} tokens)", m.cfg.n_vocab);
            }
            let mut r = engine.runner(&m, &model)?;
            let feed: Vec<u32> = prompt.iter().chain(&cont[..cont.len() - 1]).copied().collect();
            let (mut argmax, mut top5) = (Vec::new(), Vec::new());
            let mut logits = r.prefill(&prompt)?;
            for i in prompt.len() - 1..feed.len() {
                if i >= prompt.len() { logits = r.step(feed[i])?; }
                let lse = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let lse = lse + logits.iter().map(|x| (x - lse).exp()).sum::<f32>().ln();
                argmax.push(ops::argmax(&logits));
                top5.push(ops::top_k(&logits, 5).into_iter().map(|j| (j, logits[j] - lse)).collect::<Vec<_>>());
            }
            r.finish()?;
            println!("{}", serde_json::json!({"argmax": argmax, "top5": top5}));
        }
        Cmd::Generate { model, prompt, n, image, mut engine } => {
            engine.ds_env();
            let g = Gguf::open(&model)?;
            engine.own_warn(&g);
            if g.get("general.architecture").ok().and_then(|v| v.as_str()) == Some("gemma4") {
                let prompt = if prompt == "-" { let mut s = String::new(); std::io::stdin().read_to_string(&mut s)?; s } else { prompt };
                if image.is_empty() { return rusting_strata::gemma4::generate(&g, &prompt, n); }
                let mmproj = engine.mmproj.clone().or_else(|| vision::find_sibling(&model));
                let mmproj = mmproj.as_deref().context("--image needs --mmproj")?;
                let input = load_images(mmproj, &image)?;
                return rusting_strata::gemma4::generate_image(&g, &prompt, &input.grids, &input.embeddings, n);
            }
            if g.get("general.architecture").ok().and_then(|v| v.as_str()) == Some("glm5-next") {
                anyhow::ensure!(image.is_empty(), "--image is not supported for glm5-next");
                let prompt = if prompt == "-" { let mut s = String::new(); std::io::stdin().read_to_string(&mut s)?; s } else { prompt };
                return rusting_strata::glm::generate(&g, &prompt, n);
            }
            if g.get("general.architecture").ok().and_then(|v| v.as_str()) == Some("deepseek41") {
                anyhow::ensure!(image.is_empty(), "--image is not supported for deepseek41");
                let prompt = if prompt == "-" { let mut s = String::new(); std::io::stdin().read_to_string(&mut s)?; s } else { prompt };
                return rusting_strata::deepseek::chat::generate(&g, &prompt, n);
            }
            if g.get("general.architecture").ok().and_then(|v| v.as_str()) == Some(rusting_strata::hy4::ARCH) {
                anyhow::ensure!(image.is_empty(), "--image is not supported for hy_v4_flash");
                let prompt = if prompt == "-" { let mut s = String::new(); std::io::stdin().read_to_string(&mut s)?; s } else { prompt };
                return rusting_strata::hy4::chat::generate(&g, &prompt, n);
            }
            if g.get("general.architecture").ok().and_then(|v| v.as_str()) .is_some_and(rusting_strata::inkling2::is_arch) {
                let prompt = if prompt == "-" { let mut s = String::new(); std::io::stdin().read_to_string(&mut s)?; s } else { prompt };
                let max_ctx = 0;
                let mmproj = engine.mmproj.clone().or_else(|| vision::find_sibling(&model));
                let opts = rusting_strata::inkling2::chat::Inkling2Opts {
                    max_ctx,
                    sampler: engine.sampler(),
                    thinking_budget: engine.thinking_budget,
                    think_default: engine.think_default,
                    mmproj: mmproj.clone(),
                    draft_model: engine.draft_model.clone(),
                    ngram_cfg: engine.ngram_config()?,
                };
                if image.is_empty() {
                    return rusting_strata::inkling2::chat::generate_opts(&g, &prompt, n, opts);
                }
                let mmproj = mmproj.as_deref().context("--image needs --mmproj")?;
                let input = load_images(mmproj, &image)?;
                return rusting_strata::inkling2::chat::generate_image_opts(&g, &prompt, Some((&input.grids, &input.embeddings)), n, opts);
            }
            let aux_g = open_mtp_gguf(engine.aux_model_path())?;
            let m = Model::load_with_aux(&g, aux_g.as_ref(), &engine.lora_specs())?;
            let tok = Tokenizer::from_gguf(&g)?;
            let prompt = if prompt == "-" {
                let mut s = String::new();
                std::io::stdin().read_to_string(&mut s)?;
                s
            } else { prompt };
            let vision_prompt = if image.is_empty() {
                None
            } else {
                let mmproj = engine.mmproj.clone().or_else(|| vision::find_sibling(&model));
                let mmproj = mmproj.as_deref().context("--image needs --mmproj")?;
                let input = load_images(mmproj, &image)?;
                Some((build_image_prompt(&tok, &input, &image, &prompt)?, input))
            };
            let ids = tok.encode(&prompt)?;
            anyhow::ensure!(!ids.is_empty() || vision_prompt.is_some(), "empty prompt");
            let mut r = engine.runner(&m, &model)?;
            let t0 = Instant::now();
            let (logits, n_prompt) = match vision_prompt {
                None => (r.prefill(&ids)?, ids.len()),
                Some((p, input)) => (r.chat_image(&p, &input.embeddings)?, p.ids.len()),
            };
            let t1 = Instant::now();
            let (cpu0, split0) = (r.cpu_secs(), r.split_secs());
            let cap = room(&engine, n_prompt, n);
            let think_cap = Some((prompt.trim_end().ends_with("<think>"), engine.thinking_budget));
            let (out, _) = decode(&mut r, &tok, engine.mtp() > 0, &mut engine.sampler(), logits, cap, think_cap, &mut |t| { print!("{t}"); std::io::stdout().flush()?; Ok(false) })?;
            let (pp, tg) = ((t1 - t0).as_secs_f64(), t1.elapsed().as_secs_f64());
            if out.len() == cap && cap < n { eprint!("\n(stopped: the context is full at {} tokens; raise --ctx)", engine.ctx); }
            say!("\nprompt {} tok {:.2} tok/s | gen {} tok {:.2} tok/s", n_prompt, n_prompt as f64 / pp, out.len(), out.len() as f64 / tg);
            if engine.gpu { say!("CPU expert compute, share of gen time: {:.1}%", 100.0 * (r.cpu_secs() - cpu0) / tg); }
            let ms = |s: f64| 1e3 * s / out.len().max(1) as f64;
            for (i, ((a, b), (a0, b0))) in r.split_secs().into_iter().zip(split0).enumerate() {
                say!("device {i}: {:.2} ms/token forward, {:.2} ms/token copy to the next device", ms(a - a0), ms(b - b0));
            }
            if !r.split_secs().is_empty() { say!("gen total {:.2} ms/token", ms(tg)); }
            r.finish()?;
        }
        Cmd::Perplexity { model, file, ctx_len: n, chunks, kld_base, mut engine } => {
            let g = Gguf::open(&model)?;
            let aux_g = open_mtp_gguf(engine.aux_model_path())?;
            let m = Model::load_with_aux(&g, aux_g.as_ref(), &engine.lora_specs())?;
            let tok = Tokenizer::from_gguf(&g)?;
            // ponytail: no BOS at chunk starts (llama-perplexity adds one where the vocabulary asks for it; Qwen does not)
            let ids = tok.encode(&std::fs::read_to_string(&file)?)?;
            let n_chunk = (ids.len() / n).min(chunks.unwrap_or(usize::MAX));
            anyhow::ensure!(n >= 4 && n_chunk > 0, "{} tokens: need at least one chunk of --ctx-len {n} (>= 4)", ids.len());
            engine.ctx = engine.ctx.max(n);
            let mut r = engine.runner(&m, &model)?;
            let nv = m.cfg.n_vocab;
            let mut out = kld_base.map(|p| std::fs::File::create(p).map(std::io::BufWriter::new)).transpose()?;
            if let Some(o) = &mut out {
                o.write_all(b"_logits_")?;
                for v in [n, nv, n_chunk] { o.write_all(&(v as u32).to_le_bytes())?; }
                for &t in &ids[..n_chunk * n] { o.write_all(&t.to_le_bytes())?; }
            }
            let (mut nll, mut count, mut row) = (0f64, 0usize, Vec::new());
            let first = n / 2;
            for c in ids.chunks_exact(n).take(n_chunk) {
                r.reset()?;
                let mut logits = r.prefill(&c[..=first])?;
                for i in first..n - 1 {
                    if i > first { logits = r.step(c[i])?; }
                    anyhow::ensure!(logits.len() == nv, "{} logits, vocabulary {nv}", logits.len());
                    nll += log_probs(&logits, c[i + 1], &mut row);
                    if let Some(o) = &mut out { for v in &row { o.write_all(&v.to_le_bytes())?; } }
                    count += 1;
                }
                eprint!("[{}]{:.4},", count / (n - 1 - first), (nll / count as f64).exp());
            }
            if let Some(o) = &mut out { o.flush()?; }
            println!("\nPPL = {:.4} over {n_chunk} chunks of {n} tokens ({count} scored)", (nll / count as f64).exp());
            r.finish()?;
        }
        Cmd::Bench { model, pp, tg, reps, json_out, prompt_file, mut engine } => {
            if let Some(f) = &prompt_file {
                let text = std::fs::read_to_string(f).with_context(|| format!("--prompt-file {}", f.display()))?;
                anyhow::ensure!(!text.trim().is_empty(), "--prompt-file {} is empty", f.display());
                let _ = BENCH_TEXT.set(text);
            }
            let g = Gguf::open(&model)?;
            engine.own_warn(&g);
            engine.ctx = engine.ctx.max(pp.iter().max().unwrap_or(&0) + tg);
            if let Some(arch) = g.get("general.architecture").ok().and_then(|v| v.as_str()) {
                if arch == "deepseek41" || arch == "glm5-next" || arch == rusting_strata::hy4::ARCH || rusting_strata::inkling2::is_arch(arch) {
                    engine.ds_env();
                    // A --ctx below the test grows to fit it, as on the main engine.
                    if rusting_strata::CTX.load(std::sync::atomic::Ordering::Relaxed) != 0 { rusting_strata::CTX.store(engine.ctx, std::sync::atomic::Ordering::Relaxed); }
                    return if arch == "deepseek41" {
                        bench_deepseek(&g, &model, &pp, tg, reps, json_out, &mut engine)
                    } else if arch == rusting_strata::hy4::ARCH {
                        bench_hy4(&g, &model, &pp, tg, reps, json_out, &mut engine)
                    } else if rusting_strata::inkling2::is_arch(arch) {
                        bench_inkling2(&g, &model, &pp, tg, reps, json_out, &mut engine)
                    } else {
                        bench_glm(&g, &model, &pp, tg, reps, json_out, &mut engine)
                    };
                }
            }
            let aux_g = open_mtp_gguf(engine.aux_model_path())?;
            let m = Model::load_with_aux(&g, aux_g.as_ref(), &engine.lora_specs())?;
            let tok = Tokenizer::from_gguf(&g)?;
            let mut r = engine.runner(&m, &model)?;
            let mut results = Vec::with_capacity(pp.len());
            for n in pp {
                let ids = prose(&tok, n)?;
                let (mut p, mut d) = (Vec::new(), Vec::new());
                for _ in 0..reps {
                    r.reset()?;
                    let t = Instant::now();
                    let l = r.prefill(&ids)?;
                    p.push(n as f64 / t.elapsed().as_secs_f64());
                    let t = Instant::now();
                    let k = decode(&mut r, &tok, engine.mtp() > 0, &mut Sampler::greedy(), l, tg, None, &mut |_| Ok(false))?.0.len();
                    d.push(k as f64 / t.elapsed().as_secs_f64());
                }
                let (prefill, decode_stat) = (Stat::of(&p), Stat::of(&d));
                println!("pp {n:6}: prefill {} tok/s | tg {tg}: decode {} tok/s", prefill.text(), decode_stat.text());
                // R3: the attained expert-read bandwidth (bytes per token x tokens per second), against the ~900 GB/s
                // of the V100, for the prefetch decision the window records.
                let eb = r.expert_bytes_per_token();
                let mid = if eb > 0 {
                    let mid = { let mut v = d.clone(); v.sort_by(f64::total_cmp); v[v.len() / 2] };
                    say!("expert traffic: {:.1} MB/token -> {:.1} GB/s at {mid:.1} tok/s", eb as f64 / 1048576.0, eb as f64 * mid / 1e9);
                    mid
                } else { 0.0 };
                results.push(BenchResult { pp: n, prefill, decode: decode_stat, bytes_per_token: eb, decode_median: mid });
            }
            r.finish()?;
            // After `finish()`: the expert counters are final, and the record cannot reorder `save_profile()`.
            if let Some(path) = json_out {
                let device: &[usize] = if r.backend_name() == "cpu" { &[] } else { &engine.device };
                let meta = BenchMeta {
                    backend: r.backend_name(),
                    backend_arg: engine.backend.as_deref(),
                    device,
                    model: &model,
                    ctx: engine.ctx,
                    chunk: engine.chunk(),
                    kv: &engine.kv,
                    mtp: engine.mtp(),
                    tg,
                    reps,
                };
                let unix_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0, |d| d.as_secs());
                let record = bench_record(&meta, &results, r.expert_summary().as_ref(), unix_time);
                let mut line = serde_json::to_string(&record)?;
                line.push('\n');
                std::fs::write(&path, line).with_context(|| format!("writing the JSON record to {}", path.display()))?;
            }
        }
        Cmd::Calibrate { model, ctx, vram_margin, backend, device, split, kv, lora, gpu: _, mtp_model, draft_model } => {
            set_kv_env(&kv);  // read by Gpu::new and Vk::new
            if !lora.is_empty() { say!("note: --lora is ignored by calibrate; calibration uses the base model"); }
            let margins = vram_margins(&vram_margin, &device)?.into_iter().map(|mb| mb << 20).collect::<Vec<_>>();
            let split = split_arg(split.as_deref(), device.len())?;
            // The architecture of a file that runs on its own code path, where nothing reads a calibration file.
            let own_arch = |p: &std::path::Path| {
                let arch = Gguf::open(p).ok()?.get("general.architecture").ok()?.as_str()?.to_string();
                rusting_strata::config::OWN_ARCHS.contains(&arch.as_str()).then_some(arch)
            };
            if !model.is_dir() {
                if let Some(arch) = own_arch(&model) {
                    say!("nothing to calibrate: '{arch}' models run on their own code path, which reads no calibration file. Run chat, generate, bench or serve with this model as it is.");
                    return Ok(());
                }
                return calibrate(&model, draft_model.as_deref().or(mtp_model.as_deref()), ctx, &margins, backend.as_deref(), &device, split);
            }
            anyhow::ensure!(draft_model.is_none() && mtp_model.is_none(), "--mtp-model and --draft-model belong to one model: give --model a file");
            let list = rusting_strata::serve::list_models(&model);
            anyhow::ensure!(!list.is_empty(), "no .gguf models in {}", model.display());
            let mut failed = vec![];
            for (k, (id, path, _)) in list.iter().enumerate() {
                say!("calibrate {}/{}: {id}", k + 1, list.len());
                if let Some(arch) = own_arch(path) {
                    say!("calibrate {id}: skipped, calibrate measures the main engine only and '{arch}' models run on their own code path");
                    continue;
                }
                // One model that fails (too big, not a model) does not stop the others.
                if let Err(e) = calibrate(path, None, ctx, &margins, backend.as_deref(), &device, split.clone()) {
                    say!("calibrate {id}: failed: {e:#}");
                    failed.push(id.as_str());
                }
            }
            anyhow::ensure!(failed.is_empty(), "calibration failed for {}", failed.join(", "));
        }
        Cmd::Check { model, backend, device } => {
            let g = Gguf::open(&model)?;
            if g.get("general.architecture").ok().and_then(|v| v.as_str()) .is_some_and(rusting_strata::inkling2::is_arch) {
                let m = rusting_strata::inkling2::forward::Inkling2Model::new(&g)?;
                println!("check: inkling2 ok: {} layers ({} layer runs a token), {} experts ({} used), {} tensors, every tensor of the forward is present", m.cfg.n_layer, m.cfg.n_slot(), m.cfg.n_expert, m.cfg.n_used, g.tensors.len());
                return Ok(());
            }
            if g.get("general.architecture").ok().and_then(|v| v.as_str()) == Some(rusting_strata::hy4::ARCH) {
                let m = rusting_strata::hy4::forward::Hy4Model::new(&g)?;
                println!("check: hy_v4_flash ok: {} layers, {} experts ({} used), {} tensors, every tensor of the forward is present", m.cfg.n_layer, m.cfg.n_expert, m.cfg.n_used, g.tensors.len());
                return Ok(());
            }
            if g.get("general.architecture").ok().and_then(|v| v.as_str()) == Some("deepseek41") {
                let c = rusting_strata::deepseek::DsConfig::from_gguf(&g)?;
                c.validate(&g)?;
                println!("check: deepseek41 ok: {} layers, {} experts ({} used), {} tensors, tensor table valid (CPU reference only; no GPU path yet)", c.n_layer, c.n_expert, c.n_used, g.tensors.len());
                return Ok(());
            }
            if g.get("general.architecture").ok().and_then(|v| v.as_str()) == Some("glm5-next") {
                let c = rusting_strata::glm::GlmConfig::from_gguf(&g)?;
                c.validate(&g)?;
                println!("check: glm5-next ok: {} layers ({} KDA, {} DSA) + {} NextN, {} experts ({} used), {} tensors, tensor table valid", c.n_layer, c.n_kda(), c.n_dsa(), c.n_nextn, c.n_expert, c.n_used, g.tensors.len());
                return Ok(());
            }
            if g.get("general.architecture").ok().and_then(|v| v.as_str()) == Some("flint") {
                let c = ModelConfig::from_gguf(&g)?;
                let n_moe = c.layers.iter().enumerate().filter(|&(il, _)| c.is_moe_layer(il)).count();
                println!("check: flint ok: {} layers ({} dense, {} MoE), {} experts ({} used), {}/{} sliding-window / no-RoPE layers, {} tensors", c.n_layer, c.n_dense, n_moe, c.n_expert, c.n_expert_used, c.swa.iter().filter(|&&s| s).count(), c.rope_layer.iter().filter(|&&r| !r).count(), g.tensors.len());
            }
            if g.get("general.architecture").ok().and_then(|v| v.as_str()) == Some("sensenova_loop") {
                let c = ModelConfig::from_gguf(&g)?;
                let phys_layers = c.phys_layer.iter().copied().collect::<std::collections::BTreeSet<_>>().len();
                let n_delta = c.layers.iter().filter(|&&k| k == rusting_strata::config::LayerKind::Delta).count();
                let n_attn = c.layers.len() - n_delta;
                println!("check: sensenova_loop ok: {} virtual layers ({phys_layers} physical, mhc_streams={}, {} DeltaNet, {} Attn), {} tensors", c.n_layer, c.mhc_streams, n_delta, n_attn, g.tensors.len());
            }
            let types = rusting_strata::gpu_model::gpu_weight_types(&g);
            let b = backend::open(backend.as_deref(), device)?;
            if b.name() != "vulkan" {
                say!("check: the shared-memory preflight is Vulkan-only; nothing to check on {}", b.name());
                return Ok(());
            }
            b.preflight(&types, true)?;
        }
        Cmd::Interface { models_dir } => {
            println!("{}", term::paint(term::out(), "1", concat!("RustingStrata ", env!("CARGO_PKG_VERSION"))));
            let mut models = serve::list_models(&models_dir.clone().unwrap_or_else(models_dir_default));
            for m in serve::cached_models() { if !models.iter().any(|x| x.1 == m.1) { models.push(m); } }
            let mut args = interface_args(&mut std::io::stdin().lock(), models)?;
            if let Some(d) = models_dir.filter(|_| args[0] == "serve") { args.extend(["--models-dir".into(), d.to_string_lossy().into()]); }
            let shown: Vec<_> = args.iter().map(|a| if a.contains(' ') { format!("\"{a}\"") } else { a.clone() }).collect();
            println!("\n{} rustingstrata {}\n", term::paint(term::out(), "2", "running:"), shown.join(" "));
            // The same process image runs the command, so it behaves as if it were typed.
            let mut cmd = std::process::Command::new(std::env::current_exe()?);
            cmd.args(&args);
            #[cfg(unix)]
            return Err(std::os::unix::process::CommandExt::exec(&mut cmd).into());
            #[cfg(not(unix))]
            std::process::exit(cmd.status()?.code().unwrap_or(1));
        }
        Cmd::Chat { model, system, n, image, mut engine } => {
            let think = engine.think_default;
            engine.ds_env();
            let g = Gguf::open(&model)?;
            engine.own_warn(&g);
            if g.get("general.architecture").ok().and_then(|v| v.as_str()) == Some("gemma4") {
                let input = if image.is_empty() { None } else {
                    let mmproj = engine.mmproj.clone().or_else(|| vision::find_sibling(&model));
                    let mmproj = mmproj.as_deref().context("--image needs --mmproj")?;
                    Some(load_images(mmproj, &image)?)
                };
                return match &input {
                    Some(i) => rusting_strata::gemma4::chat(&g, system, n, think, Some((&i.grids, &i.embeddings))),
                    None => rusting_strata::gemma4::chat(&g, system, n, think, None),
                };
            }
            if g.get("general.architecture").ok().and_then(|v| v.as_str()) == Some("glm5-next") {
                anyhow::ensure!(image.is_empty(), "--image is not supported for glm5-next");
                return rusting_strata::glm::chat(&g, system, n);
            }
            if g.get("general.architecture").ok().and_then(|v| v.as_str()) == Some("deepseek41") {
                anyhow::ensure!(image.is_empty(), "--image is not supported for deepseek41");
                return rusting_strata::deepseek::chat::chat(&g, system, n, think);
            }
            if g.get("general.architecture").ok().and_then(|v| v.as_str()) == Some(rusting_strata::hy4::ARCH) {
                anyhow::ensure!(image.is_empty(), "--image is not supported for hy_v4_flash");
                return rusting_strata::hy4::chat::chat(&g, system, n, think);
            }
            if g.get("general.architecture").ok().and_then(|v| v.as_str()) .is_some_and(rusting_strata::inkling2::is_arch) {
                let max_ctx = 0;
                let mmproj = engine.mmproj.clone().or_else(|| vision::find_sibling(&model));
                let opts = rusting_strata::inkling2::chat::Inkling2Opts {
                    max_ctx,
                    sampler: engine.sampler(),
                    thinking_budget: engine.thinking_budget,
                    think_default: think,
                    mmproj: mmproj.clone(),
                    draft_model: engine.draft_model.clone(),
                    ngram_cfg: engine.ngram_config()?,
                };
                let input = if image.is_empty() {
                    None
                } else {
                    let mmproj = mmproj.as_deref().context("--image needs --mmproj")?;
                    Some(load_images(mmproj, &image)?)
                };
                return match &input {
                    Some(i) => rusting_strata::inkling2::chat::chat_image_opts(&g, system, n, Some((&i.grids, &i.embeddings)), opts),
                    None => rusting_strata::inkling2::chat::chat_opts(&g, system, n, opts),
                };
            }
            let aux_g = open_mtp_gguf(engine.aux_model_path())?;
            let m = Model::load_with_aux(&g, aux_g.as_ref(), &engine.lora_specs())?;
            let tok = Tokenizer::from_gguf(&g)?;
            let image_input = if image.is_empty() { None } else {
                let mmproj = engine.mmproj.clone().or_else(|| vision::find_sibling(&model));
                let mmproj = mmproj.as_deref().context("--image needs --mmproj")?;
                Some(load_images(mmproj, &image)?)
            };
            let mut r = engine.runner(&m, &model)?;
            let mut smp = engine.sampler();
            let tpl = template(&g);
            let mut msgs: Vec<Msg> = system.iter().map(|s| Msg::new("system", s)).collect();
            let stdin = std::io::stdin();
            loop {
                print!("{}", rusting_strata::term::prompt());
                std::io::stdout().flush()?;
                let mut line = String::new();
                if stdin.lock().read_line(&mut line)? == 0 { break; }
                if line.trim().is_empty() { continue; }
                msgs.push(Msg::new("user", line.trim_end()));
                let prompt = tpl.render(&msgs, &[], think, engine.preserve_thinking, true);
                let (ids, logits) = match &image_input {
                    Some(input) => {
                        let p = build_image_prompt(&tok, input, &image, &prompt)?;
                        let ids = p.ids.clone();
                        let logits = r.chat_image(&p, &input.embeddings)?;
                        (ids, logits)
                    }
                    None => {
                        let (ids, logits, _) = r.chat(&tok, &prompt, &tpl.render(&msgs, &[], think, engine.preserve_thinking, false), "")?;
                        (ids, logits)
                    }
                };
                let think_cap = Some((prompt.trim_end().ends_with("<think>"), engine.thinking_budget));
                (smp.seen.clear(), smp.hist.clear());  // the penalties count this reply only, not the earlier turns
                // The reasoning is dim on a terminal; it starts in the prompt when the template opens the block itself.
                let dim = term::out();
                if dim && think_cap.is_some_and(|c| c.0) { print!("\x1b[2m"); }
                let (out, _) = decode(&mut r, &tok, engine.mtp() > 0, &mut smp, logits, room(&engine, ids.len(), n), think_cap, &mut |t| {
                    if dim { print!("{}", t.replace("<think>", "\x1b[2m<think>").replace("</think>", "</think>\x1b[0m")); } else { print!("{t}"); }
                    std::io::stdout().flush()?;
                    Ok(false)
                })?;
                println!("{}", if dim { "\x1b[0m" } else { "" });
                msgs.push(Msg::new("assistant", &answer(&tok.decode(&out)?)));
            }
            r.finish()?;
        }
        Cmd::Serve { model, models_dir, addr, api_key, mcp, open, host, mut engine } => {
            engine.ds_env();
            anyhow::ensure!(host.auto_unload != Some(0), "--auto-unload must be at least 1 second");
            engine.parallel = host.parallel;
            // Before the model is loaded: a slot count outside 1..=255 is an invalid request, not a downgrade.
            anyhow::ensure!((1..=255).contains(&engine.parallel), "--parallel {}: 1 to 255 slots", engine.parallel);
            let api_key = api_key.or_else(|| std::env::var("STRATA_API_KEY").ok());
            anyhow::ensure!(api_key.as_ref().is_none_or(|k| !k.is_empty()), "empty API key");
            let mut allowed_hosts = host.allowed_hosts.clone();
            if allowed_hosts.is_empty() && let Ok(v) = std::env::var("STRATA_ALLOWED_HOSTS") {
                allowed_hosts.extend(v.split(',').map(str::trim).filter(|h| !h.is_empty()).map(str::to_owned));
            }
            let mut keys_time = host.api_keys.as_ref().and_then(|f| std::fs::metadata(f).and_then(|m| m.modified()).ok());
            let users = match &host.api_keys {
                Some(f) => serve::Keys::parse(&std::fs::read_to_string(f).with_context(|| format!("reading {}", f.display()))?)
                    .with_context(|| format!("in {}", f.display()))?,
                None => vec![],
            };
            anyhow::ensure!(!users.iter().any(|u| api_key.as_ref() == Some(&u.1)), "the admin key is also a customer key");
            // An empty key file would leave the server open to anyone.
            anyhow::ensure!(host.api_keys.is_none() || !users.is_empty() || api_key.is_some(), "the key file has no keys");
            let keys = std::sync::Arc::new(std::sync::RwLock::new(serve::Keys { admin: api_key, users }));
            // Beside the default models folder (its parent), whatever --models-dir says.
            let hub = std::sync::Arc::new(rusting_strata::mcp::Hub::new(mcp.or_else(|| std::env::var_os("STRATA_MCP").map(PathBuf::from))
                .unwrap_or_else(|| models_dir_default().with_file_name("mcp.json"))));
            let keys_file = host.api_keys.clone();
            let usage = match &host.usage_log {
                Some(f) => Some(std::sync::Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(f)
                    .with_context(|| format!("opening {}", f.display()))?)),
                None => None,
            };
            let max_waiting = host.max_waiting.map_or(usize::MAX, |n| n.max(1));
            anyhow::ensure!(host.max_per_key != Some(0), "--max-per-key must be at least 1");
            let url = format!("http://{addr}/");
            let listener = match TcpListener::bind(&addr) {
                Err(e) if open && e.kind() == std::io::ErrorKind::AddrInUse => { open_browser(&url); return Ok(()); }
                l => l?,
            };
            // Connections are read on their own threads, so a client that is slow to send its request does not hold
            // up the others; complete requests run one at a time, in arrival order.
            let (tx, rx) = std::sync::mpsc::channel();
            let open_conns = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let waiting = open_conns.clone();
            let stats = std::sync::Arc::new(std::sync::Mutex::new(serve::Stats::default()));
            let (st, a, k, allowed, t0) = (stats.clone(), addr.clone(), keys.clone(), allowed_hosts.clone(), Instant::now());
            let (dir, extra) = (models_dir.unwrap_or_else(models_dir_default), std::sync::Arc::<std::sync::Mutex<Vec<PathBuf>>>::default());
            let offer = std::sync::Arc::new((dir.clone(), model.clone(), extra.clone()));
            let (per_key, held) = (host.max_per_key, serve::Open::default());
            let stop = on_sigterm();
            // SIGTERM (systemctl stop or restart): new requests get 503, the requests accepted before it finish, then
            // the server exits. The model thread does that itself, after saving the expert profile (`Serving::stopped`);
            // this thread covers a server with no model or one loading. Two looks 0.2 s apart cover a connection counted
            // just before the signal, and the model thread gets 3 s to save first.
            let oc = open_conns.clone();
            std::thread::spawn(move || loop {
                std::thread::sleep(Duration::from_millis(200));
                if !stop.load(Relaxed) { continue; }
                let n = oc.load(Relaxed);
                if n == 0 { std::thread::sleep(Duration::from_millis(200)); if oc.load(Relaxed) == 0 { std::thread::sleep(Duration::from_secs(3)); say!("stopped"); std::process::exit(0); } }
                else { say!("stopping: {n} requests to finish"); std::thread::sleep(Duration::from_secs(5)); }
            });
            std::thread::spawn(move || for (i, conn) in listener.incoming().enumerate() {
                let mut c = match conn { Ok(c) => c, Err(e) => { say!("request {i}: {e:#}"); continue } };
                if stop.load(Relaxed) {
                    let b = json!({"error": {"message": "server is stopping, retry later", "type": "overloaded_error"}}).to_string();
                    let _ = c.set_write_timeout(Some(Duration::from_secs(5)));
                    let _ = write!(c, "HTTP/1.1 503 Service Unavailable\r\nRetry-After: 5\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len());
                    let _ = c.shutdown(std::net::Shutdown::Write);
                    continue;
                }
                // ponytail: one cap on connections read or queued (each may hold an 8 MB body), not one per key.
                if open_conns.fetch_add(1, Relaxed) >= max_waiting {
                    open_conns.fetch_sub(1, Relaxed);
                    say!("request {i}: {max_waiting} requests already waiting, answered 503");
                    let b = json!({"error": {"message": "server busy, retry later", "type": "overloaded_error"}}).to_string();
                    let _ = c.set_write_timeout(Some(Duration::from_secs(5)));
                    let _ = write!(c, "HTTP/1.1 503 Service Unavailable\r\nRetry-After: 2\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len());
                    let _ = c.shutdown(std::net::Shutdown::Write);
                    continue;
                }
                // A changed key file takes effect from the next request; one that does not parse keeps the old keys.
                if let Some(f) = &keys_file && let Ok(t) = std::fs::metadata(f).and_then(|m| m.modified()) && Some(t) != keys_time {
                    keys_time = Some(t);
                    let mut kw = k.write().unwrap();
                    match std::fs::read_to_string(f).map_err(Into::into).and_then(|t| serve::Keys::parse(&t)) {
                        Ok(u) if u.iter().any(|u| kw.admin.as_ref() == Some(&u.1)) => say!("{}: the admin key is also a customer key, keeping the old keys", f.display()),
                        // An empty file (or one caught mid-write) would leave the server open.
                        Ok(u) if u.is_empty() && kw.admin.is_none() => say!("{}: no keys, keeping the old keys", f.display()),
                        Ok(u) => { say!("{}: {} keys", f.display(), u.len()); kw.users = u; }
                        Err(e) => say!("{}: {e:#}, keeping the old keys", f.display()),
                    }
                }
                let (tx, oc, st, a, k, allowed, held, hub, offer) = (tx.clone(), open_conns.clone(), st.clone(), a.clone(), k.clone(), allowed.clone(), held.clone(), hub.clone(), offer.clone());
                std::thread::spawn(move || {
                    let q = c.set_read_timeout(Some(Duration::from_secs(30))).and_then(|_| c.set_write_timeout(Some(Duration::from_secs(30))))
                        .map_err(Into::into).and_then(|_| serve::read_request(&mut BufReader::new(&c)));
                    let reply = |v: serde_json::Value, status: &str| {
                        let b = v.to_string();
                        let _ = write!(&c, "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nCache-Control: no-store\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len());
                        oc.fetch_sub(1, Relaxed);
                    };
                    // For load balancers: no key, no Host check, and an answer while a reply is generated.
                    if let Ok(r) = &q && r.method == "GET" && r.path == "/health" {
                        let (model, loading, slots, unloaded) = { let g = st.lock().unwrap(); (g.model.is_some() && !g.loading, g.loading, g.slots, g.unloaded) };
                        let waiting = oc.load(Relaxed) - 1;
                        // A model that `--auto-unload` freed still answers chats (after its load), so it is "ok".
                        return if unloaded { reply(json!({"status": "ok", "unloaded": true, "waiting": waiting}), "200 OK") }
                            else if model { reply(json!({"status": "ok", "waiting": waiting, "slots": slots}), "200 OK") }
                            else { reply(json!({"status": if loading { "loading" } else { "no model" }}), "503 Service Unavailable") };
                    }
                    // The page's side panel is answered here, so it stays current while a reply is generated or a
                    // model loads. A bad Host or key falls through to `handle`, which refuses it.
                    if let Ok(r) = &q && r.method == "GET" && matches!(r.path.as_str(), "/api/stats" | "/metrics") && let k = k.read().unwrap().clone()
                        && (k.any() || serve::host_ok(&r.host, &a, &allowed)) && k.who(&r.auth).is_some_and(|w| w.admin) {
                        if r.path == "/metrics" {
                            let b = st.lock().unwrap().prometheus(oc.load(Relaxed) - 1, t0.elapsed().as_secs());
                            let _ = write!(&c, "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\nCache-Control: no-store\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len());
                            oc.fetch_sub(1, Relaxed);
                            return;
                        }
                        let mut v = json!(*st.lock().unwrap());
                        (v["system"], v["uptime_s"]) = (serve::system(), t0.elapsed().as_secs().into());
                        return reply(v, "200 OK");
                    }
                    // MCP tools for the page. Answered here: a tool can run for minutes and must not hold the model's
                    // queue. The checks are those of `handle`: Host (or a key), admin, and JSON for a POST (a form post
                    // from another site needs no CORS preflight, a JSON one does and gets no answer). A request that
                    // fails one falls through to `handle`, which refuses it.
                    // Without keys only this computer may use them: on `--addr 0.0.0.0` a tool call from the network
                    // would otherwise need nothing but a matching Host header. Behind a reverse proxy or a tunnel on
                    // this computer every peer is loopback, so a proxy's header or a Host that is not a loopback
                    // name is refused too. A proxy that sends neither cannot be told apart from a local browser:
                    // docs/USAGE.md says to set a key there.
                    if let Ok(r) = &q && let route = r.path.split('?').next().unwrap_or_default() && matches!((r.method.as_str(), route), ("GET", "/api/mcp") | ("POST", "/api/mcp/reload" | "/api/mcp/call"))
                        && !k.read().unwrap().any() && !(c.peer_addr().is_ok_and(|p| p.ip().to_canonical().is_loopback()) && !r.proxied && serve::host_ok(&r.host, "", &[])) {
                        return reply(json!({"error": {"message": "MCP tools need an API key (--api-key) for requests from another computer or through a proxy", "type": "forbidden"}}), "403 Forbidden");
                    }
                    if let Ok(r) = &q && let route = r.path.split('?').next().unwrap_or_default() && matches!((r.method.as_str(), route), ("GET", "/api/mcp") | ("POST", "/api/mcp/reload" | "/api/mcp/call"))
                        && let k = k.read().unwrap().clone() && (k.any() || serve::host_ok(&r.host, &a, &allowed)) && k.who(&r.auth).is_some_and(|w| w.admin)
                        && (r.method == "GET" || r.json) {
                        if route != "/api/mcp/call" { return reply(hub.list(r.method == "POST"), "200 OK"); }
                        let v: serde_json::Value = serde_json::from_slice(&r.body).unwrap_or_default();
                        let (server, tool) = (v["server"].as_str().unwrap_or_default(), v["tool"].as_str().unwrap_or_default());
                        say!("request {i}: mcp {server} {tool}");
                        return match hub.call(server, tool, if v["arguments"].is_object() { v["arguments"].clone() } else { json!({}) }) {
                            Ok(v) => reply(v, "200 OK"),
                            Err(e) => { say!("request {i}: mcp {server} {tool}: {e:#}"); reply(json!({"error": {"message": format!("{e:#}"), "type": "server_error"}}), "502 Bad Gateway") }
                        };
                    }
                    // A malformed request, a wrong key, the page and the model list are answered here, not in the queue,
                    // where without `--parallel` they would wait for the reply being generated (`handle` checks the key
                    // again: the keys may change meanwhile). The page holds no secret, so it loads without the key (it
                    // asks for it); without keys the Host check stops DNS rebinding.
                    let who = q.as_ref().ok().and_then(|r| k.read().unwrap().who(&r.auth));
                    match &q {
                        Err(e) => {
                            say!("request {i}: {e:#}");
                            return reply(json!({"error": {"message": format!("{e:#}"), "type": "invalid_request_error"}}), "400 Bad Request");
                        }
                        Ok(r) if r.method == "GET" && r.path == "/" && (k.read().unwrap().any() || serve::host_ok(&r.host, &a, &allowed)) => {
                            let _ = write!(&c, "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nX-Frame-Options: DENY\r\nContent-Security-Policy: frame-ancestors 'none'\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{PAGE}", PAGE.len());
                            oc.fetch_sub(1, Relaxed);
                            return;
                        }
                        Ok(_) if who.is_none() =>
                            return reply(json!({"error": {"message": "missing or wrong API key", "type": "invalid_api_key"}}), "401 Unauthorized"),
                        // Clients list the models often; a model still loading is listed, as its chats wait for the load.
                        Ok(r) if r.method == "GET" && r.path.split('?').next() == Some("/v1/models") && (k.read().unwrap().any() || serve::host_ok(&r.host, &a, &allowed)) => {
                            let created = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0, |d| d.as_secs());
                            let (m, file, n_ctx, n_ctx_train) = {
                                let state = st.lock().unwrap();
                                (state.model.clone(), state.file.clone(), state.ctx, state.ctx_train)
                            };
                            let mut data: Vec<_> = m.iter().map(|id| json!({
                                "id": id, "object": "model", "created": created, "owned_by": "strata",
                                "meta": {"n_ctx": n_ctx, "n_ctx_train": n_ctx_train}
                            })).collect();
                            // An admin key also sees the other models the server offers, by file name: a chat request
                            // that names one loads it (`Serving::other`). Customer keys may not load models.
                            // ponytail: the folders are read for every list; cache the names if a client polls hard.
                            if who.as_ref().is_some_and(|w| w.admin) {
                                let mut seen: Vec<String> = m.into_iter().chain([file]).collect();
                                for (_, p, _) in offered(&offer.0, offer.1.as_ref(), &offer.2.lock().unwrap()) {
                                    let id = stem(&p);
                                    if !seen.contains(&id) { data.push(json!({"id": id, "object": "model", "created": created, "owned_by": "strata"})); seen.push(id); }
                                }
                            }
                            return reply(json!({"object": "list", "data": data}), "200 OK");
                        }
                        _ => {}
                    }
                    // A customer key over `--max-per-key` is refused here, before it takes a place in the queue.
                    let hold = match (who, per_key) {
                        (Some(w), Some(cap)) if !w.admin => match serve::Hold::take(&held, w.id, cap) {
                            Some(h) => Some(h),
                            None => {
                                say!("request {i}: {} already has {cap} requests open, answered 429", w.name);
                                let b = json!({"error": {"message": format!("this key already has {cap} requests open, retry later"), "type": "rate_limit_error"}}).to_string();
                                let _ = write!(&c, "HTTP/1.1 429 Too Many Requests\r\nRetry-After: 1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len());
                                oc.fetch_sub(1, Relaxed);
                                return;
                            }
                        },
                        _ => None,
                    };
                    let _ = tx.send((c, q, i, hold));
                });
            });
            let s = Serving { rx: rx.into(), waiting, stop, addr, allowed_hosts, keys, usage, max_tokens: host.max_tokens.unwrap_or(usize::MAX), dir, model: model.clone(), extra,
                dl: Default::default(), stats, opts: Default::default(), idle: host.auto_unload.map(Duration::from_secs), asleep: Default::default(), parked: Default::default(), alias: host.alias };
            // The model to load next and the /api/load request waiting for the outcome (none for --model).
            let mut next = model.map(|p| (p, None));
            say!("listening on {} (models folder: {})", term::hl(&url), s.dir.display());
            // Chats wait while a model loads; the page shows at once.
            if open { open_browser(&url); }
            loop {
                let (p, mut w) = match next.take() {
                    Some(x) => x,
                    None => match s.serve(None, &engine) {
                        // Another model replaces the one `--auto-unload` freed.
                        Next::Load(p, w) => { *s.asleep.lock().unwrap() = None; (p, Some(w)) }
                        Next::Wake(p) => (p, None),
                        _ => return Ok(()),
                    },
                };
                // The page's load options stay for later loads; a load that fails takes the old ones back.
                let (old, old_ctx) = (engine.clone(), rusting_strata::CTX.load(Relaxed));
                let o = s.opts.lock().unwrap().take();
                if !o.is_null() {
                    engine = engine.with(&o)?;
                    // The models outside the main engine read their limit here (`ctx_limit`).
                    if !o["ctx"].is_null() { rusting_strata::CTX.store(engine.ctx, Relaxed); }
                    // `runner` also sets it, and calibrate runs before `runner`.
                    set_kv_env(&engine.kv);
                }
                // The page's calibrate button: measure with no model loaded, then load with the measured chunk and MTP
                // (they replace `--chunk`/`--mtp` and the page's earlier values).
                let cal = if o["calibrate"] == true && engine.gpu {
                    (engine.chunk, engine.mtp) = (None, None);
                    set_kv_env(&engine.kv);
                    s.stats.lock().unwrap().unload(Some(s.name(&p)));
                    vram_margins(&engine.vram_margin, &engine.device)
                        .and_then(|mb| calibrate(&p, engine.aux_model_path(), engine.ctx, &mb.into_iter().map(|mb| mb << 20).collect::<Vec<_>>(), engine.backend.as_deref(), &engine.device,
                            split_arg(engine.split.as_deref(), engine.device.len())?))
                } else { Ok(()) };
                match cal.and_then(|()| serve_model(&p, engine.clone(), &s, &mut w)) {
                    Ok(Next::Load(p, w)) => next = Some((p, Some(w))),
                    // A chat request named another model. This one comes back with the next chat if that load fails.
                    Ok(Next::Wake(q)) => { *s.asleep.lock().unwrap() = Some(p); next = Some((q, None)); }
                    // `--auto-unload`: `serve_model` has freed the model. Its name stays in the model list, as a chat
                    // loads it again.
                    Ok(Next::Idle) => {
                        say!("unloaded {} after {} s without a chat request", p.display(), s.idle.map_or(0, |d| d.as_secs()));
                        s.sleep(p, engine.ctx);
                    }
                    Ok(_) => return Ok(()),
                    // The load for a chat request failed (after `--auto-unload` another program may hold the VRAM now):
                    // the chat that asked for it gets the error. The next chat loads the model that was there before
                    // (the one `--auto-unload` freed, or the one the request replaced); with none, no model is loaded.
                    Err(e) if w.is_none() && s.parked.lock().unwrap().is_some() => {
                        say!("loading {}: {e:#}", p.display());
                        let (mut c, ..) = s.parked.lock().unwrap().take().unwrap();
                        s.waiting.fetch_sub(1, Relaxed);
                        let b = json!({"error": {"message": format!("loading {}: {e:#}", stem(&p)), "type": "server_error"}}).to_string();
                        let _ = write!(c, "HTTP/1.1 503 Service Unavailable\r\nRetry-After: 10\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len());
                        let back = s.asleep.lock().unwrap().take();
                        match back { Some(a) => s.sleep(a, engine.ctx), None => s.stats.lock().unwrap().unload(None) }
                    }
                    Err(e) if w.is_none() => return Err(e),
                    Err(e) => {
                        engine = old;
                        rusting_strata::CTX.store(old_ctx, Relaxed);
                        set_kv_env(&engine.kv);
                        s.stats.lock().unwrap().unload(None);
                        say!("loading {}: {e:#}", p.display());
                        let b = json!({"error": {"message": format!("{e:#}"), "type": "server_error"}}).to_string();
                        let _ = write!(w.unwrap(), "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len());
                    }
                }
            }
        }
    }
    Ok(())
}

/// `n` tokens of varied prose, repeated: the prefill prompt for `bench` and `calibrate`.
/// `bench --prompt-file`: the text `prose` repeats instead of its paragraph.
static BENCH_TEXT: std::sync::OnceLock<String> = std::sync::OnceLock::new();

fn prose(tok: &Tokenizer, n: usize) -> anyhow::Result<Vec<u32>> {
    if let Some(text) = BENCH_TEXT.get() {
        let mut ids = tok.encode(text)?;
        anyhow::ensure!(!ids.is_empty(), "--prompt-file gives no tokens");
        while ids.len() < n { ids.extend_from_within(..); }
        ids.truncate(n);
        return Ok(ids);
    }
    let text = "The river town woke before dawn. Fishermen checked their nets, bakers lit ovens, and a clockmaker \
        argued with his apprentice about gears, tides and the price of brass. In 1874 the bridge washed away; \
        the council voted 7 to 5 to rebuild it in stone. ";
    let ids: Vec<u32> = tok.encode(&text.repeat(n / 20 + 1))?.into_iter().take(n).collect();
    anyhow::ensure!(ids.len() == n, "prose prompt too short");
    Ok(ids)
}

/// Expert-cache counters for a DeepSeek/GLM bench run. `tensors_per_expert` is 3 (gate, up, down).
fn big_expert_stats(layers: usize, n_expert: usize) -> Option<ExpertStats> {
    let (_backend, resident, _bytes) = rusting_strata::deepseek::mat::dev_info()?;
    let l = &rusting_strata::deepseek::mat::EX_LOOKUPS;
    Some(ExpertStats {
        hits: l[0].swap(0, Relaxed) as usize,
        misses: l[1].swap(0, Relaxed) as usize,
        swaps: 0,
        resident,
        total: 3 * layers * n_expert,
        bytes_per_token: 0,
        cached: true,
        all_fast: false,
    })
}

fn write_bench_json(path: Option<PathBuf>, meta: &BenchMeta, results: &[BenchResult], expert: Option<&ExpertStats>) -> anyhow::Result<()> {
    let Some(path) = path else { return Ok(()) };
    let unix_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let record = bench_record(meta, results, expert, unix_time);
    let mut line = serde_json::to_string(&record)?;
    line.push('\n');
    std::fs::write(&path, line).with_context(|| format!("writing the JSON record to {}", path.display()))?;
    Ok(())
}

fn decode_greedy_big(m: &mut dyn FnMut(u32) -> Vec<f32>, logits: Vec<f32>, eos: &[u32], tg: usize) -> usize {
    let mut logits = logits;
    let mut n = 0;
    for i in 0..tg {
        let next = ops::argmax(&logits) as u32;
        if eos.contains(&next) { break; }
        n += 1;
        if i + 1 < tg { logits = m(next); }
    }
    n
}

/// `strata bench` for DeepSeek-V4.1 (token-at-a-time reference forward, GPU matvecs when enabled).
fn bench_deepseek(g: &Gguf, model: &Path, pp: &[usize], tg: usize, reps: usize, json_out: Option<PathBuf>, engine: &mut Engine) -> anyhow::Result<()> {
    rusting_strata::deepseek::chat::gpu_defaults();
    let tok = Tokenizer::from_gguf(g)?;
    let mut m = rusting_strata::deepseek::forward::DsModel::new(g)?;
    let eos = tok.eos_ids();
    let mut results = Vec::with_capacity(pp.len());
    for &n in pp {
        let ids = prose(&tok, n)?;
        let (mut p, mut d) = (Vec::new(), Vec::new());
        for _ in 0..reps {
            m.reset();
            let t = Instant::now();
            let logits = m.prefill(&ids);
            p.push(n as f64 / t.elapsed().as_secs_f64());
            let t = Instant::now();
            let k = decode_greedy_big(&mut |t| m.step(t), logits, &eos, tg);
            d.push(k as f64 / t.elapsed().as_secs_f64());
        }
        let (prefill, decode_stat) = (Stat::of(&p), Stat::of(&d));
        println!("pp {n:6}: prefill {} tok/s | tg {tg}: decode {} tok/s", prefill.text(), decode_stat.text());
        results.push(BenchResult { pp: n, prefill, decode: decode_stat, bytes_per_token: 0, decode_median: 0.0 });
    }
    let expert = big_expert_stats(m.cfg.n_layer, m.cfg.n_expert);
    let backend = rusting_strata::deepseek::mat::dev_info().map_or("cpu", |i| i.0);
    let meta = BenchMeta {
        backend,
        backend_arg: engine.backend.as_deref(),
        device: if backend == "cpu" { &[] } else { &engine.device },
        model,
        ctx: engine.ctx,
        chunk: 1,
        kv: &engine.kv,
        mtp: 0,
        tg,
        reps,
    };
    write_bench_json(json_out, &meta, &results, expert.as_ref())
}

/// `strata bench` for a token-by-token forward (GPU matvecs when enabled).
macro_rules! bench_tokenwise {
    ($name:ident, $model:ty) => {
fn $name(g: &Gguf, model: &Path, pp: &[usize], tg: usize, reps: usize, json_out: Option<PathBuf>, engine: &mut Engine) -> anyhow::Result<()> {
    rusting_strata::deepseek::chat::gpu_defaults();
    let tok = Tokenizer::from_gguf(g)?;
    let mut m = <$model>::new(g)?;
    let eos = tok.eos_ids();
    let mut results = Vec::with_capacity(pp.len());
    for &n in pp {
        let ids = prose(&tok, n)?;
        let (mut p, mut d) = (Vec::new(), Vec::new());
        for _ in 0..reps {
            m.reset();
            let t = Instant::now();
            let logits = m.prefill(&ids);
            p.push(n as f64 / t.elapsed().as_secs_f64());
            let t = Instant::now();
            let k = decode_greedy_big(&mut |t| m.step(t), logits, &eos, tg);
            d.push(k as f64 / t.elapsed().as_secs_f64());
        }
        let (prefill, decode_stat) = (Stat::of(&p), Stat::of(&d));
        println!("pp {n:6}: prefill {} tok/s | tg {tg}: decode {} tok/s", prefill.text(), decode_stat.text());
        results.push(BenchResult { pp: n, prefill, decode: decode_stat, bytes_per_token: 0, decode_median: 0.0 });
    }
    let expert = big_expert_stats(m.cfg.n_layer, m.cfg.n_expert);
    let backend = rusting_strata::deepseek::mat::dev_info().map_or("cpu", |i| i.0);
    let meta = BenchMeta {
        backend,
        backend_arg: engine.backend.as_deref(),
        device: if backend == "cpu" { &[] } else { &engine.device },
        model,
        ctx: engine.ctx,
        chunk: 1,
        kv: &engine.kv,
        mtp: 0,
        tg,
        reps,
    };
    write_bench_json(json_out, &meta, &results, expert.as_ref())
}
    };
}
bench_tokenwise!(bench_hy4, rusting_strata::hy4::forward::Hy4Model);

/// `strata bench` for Inkling 2 (uses GPU argmax decode, plus `--draft-model` DSpark / `--spec-type` context lookup).
fn bench_inkling2(g: &Gguf, model: &Path, pp: &[usize], tg: usize, reps: usize, json_out: Option<PathBuf>, engine: &mut Engine) -> anyhow::Result<()> {
    // The same defaults `chat`, `generate` and `serve` set: without them the plain weights spread over both cards,
    // `Device::new` refuses the second one and the whole benchmark runs the host path (about a fifth of the speed).
    rusting_strata::inkling2::chat::gpu_defaults();
    let tok = Tokenizer::from_gguf(g)?;
    let max_ctx = engine.ctx.max(pp.iter().copied().max().unwrap_or(0) + tg + 64);
    let opts = rusting_strata::inkling2::chat::Inkling2Opts {
        max_ctx,
        draft_model: engine.draft_model.clone(),
        ngram_cfg: engine.ngram_config()?,
        ..Default::default()
    };
    let mut m = opts.build_model(g, &tok)?;
    let eos = tok.eos_ids();
    let mut results = Vec::with_capacity(pp.len());
    for &n in pp {
        let ids = prose(&tok, n)?;
        let (mut p, mut d) = (Vec::new(), Vec::new());
        for _ in 0..reps {
            m.reset();
            let t = Instant::now();
            let logits = m.prefill(&ids);
            p.push(n as f64 / t.elapsed().as_secs_f64());
            let t = Instant::now();
            let mut next = ops::argmax(&logits) as u32;
            let mut pending = std::collections::VecDeque::<u32>::new();
            let mut k = 0;
            for i in 0..tg {
                if eos.contains(&next) { break; }
                k += 1;
                if i + 1 < tg {
                    if let Some(t) = pending.pop_front() {
                        next = t;
                    } else if m.has_spec() {
                        let rem = tg - (i + 1);
                        pending.extend(m.spec_step_argmax(next, rem, &eos));
                        next = pending.pop_front().unwrap();
                    } else {
                        next = m.step_argmax(next);
                    }
                }
            }
            d.push(k as f64 / t.elapsed().as_secs_f64());
        }
        let (prefill, decode_stat) = (Stat::of(&p), Stat::of(&d));
        println!("pp {n:6}: prefill {} tok/s | tg {tg}: decode {} tok/s", prefill.text(), decode_stat.text());
        results.push(BenchResult { pp: n, prefill, decode: decode_stat, bytes_per_token: 0, decode_median: 0.0 });
    }
    if m.drafts.0 > 0 {
        say!("speculative drafts accepted {}/{} ({:.1}%)", m.drafts.1, m.drafts.0, 100.0 * m.drafts.1 as f64 / m.drafts.0 as f64);
    }
    let expert = big_expert_stats(m.cfg.n_layer, m.cfg.n_expert);
    let backend = rusting_strata::deepseek::mat::dev_info().map_or("cpu", |i| i.0);
    let meta = BenchMeta {
        backend,
        backend_arg: engine.backend.as_deref(),
        device: if backend == "cpu" { &[] } else { &engine.device },
        model,
        ctx: engine.ctx,
        chunk: 1,
        kv: &engine.kv,
        mtp: 0,
        tg,
        reps,
    };
    write_bench_json(json_out, &meta, &results, expert.as_ref())
}

/// `strata bench` for GLM-5.3-Flash (chunked reference forward, GPU matvecs when enabled).
fn bench_glm(g: &Gguf, model: &Path, pp: &[usize], tg: usize, reps: usize, json_out: Option<PathBuf>, engine: &mut Engine) -> anyhow::Result<()> {
    rusting_strata::glm::gpu_defaults();
    let tok = Tokenizer::from_gguf(g)?;
    let mut m = rusting_strata::glm::forward::GlmModel::new(g)?;
    let eos = tok.eos_ids();
    let chunk = std::env::var("STRATA_GLM_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
    let mut results = Vec::with_capacity(pp.len());
    for &n in pp {
        let ids = prose(&tok, n)?;
        let (mut p, mut d) = (Vec::new(), Vec::new());
        for _ in 0..reps {
            m.reset();
            let t = Instant::now();
            let logits = m.prefill(&ids);
            p.push(n as f64 / t.elapsed().as_secs_f64());
            let t = Instant::now();
            let k = decode_greedy_big(&mut |t| m.step(t), logits, &eos, tg);
            d.push(k as f64 / t.elapsed().as_secs_f64());
        }
        let (prefill, decode_stat) = (Stat::of(&p), Stat::of(&d));
        println!("pp {n:6}: prefill {} tok/s | tg {tg}: decode {} tok/s", prefill.text(), decode_stat.text());
        results.push(BenchResult { pp: n, prefill, decode: decode_stat, bytes_per_token: 0, decode_median: 0.0 });
    }
    let expert = big_expert_stats(m.cfg.n_layer.saturating_sub(m.cfg.n_dense), m.cfg.n_expert);
    let backend = rusting_strata::deepseek::mat::dev_info().map_or("cpu", |i| i.0);
    let meta = BenchMeta {
        backend,
        backend_arg: engine.backend.as_deref(),
        device: if backend == "cpu" { &[] } else { &engine.device },
        model,
        ctx: engine.ctx,
        chunk,
        kv: &engine.kv,
        mtp: 0,
        tg,
        reps,
    };
    write_bench_json(json_out, &meta, &results, expert.as_ref())
}

/// Best (lowest) of three timings of `f`, in seconds.
fn best(f: &mut dyn FnMut() -> anyhow::Result<()>) -> anyhow::Result<f64> {
    let mut b = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        f()?;
        b = b.min(t.elapsed().as_secs_f64());
    }
    Ok(b)
}

/// With `aux` (`--mtp-model`/`--draft-model`) the drafts are measured with that file, and runs given the same file use
/// the measured draft count.
fn calibrate(model: &Path, aux: Option<&Path>, ctx: usize, margin: &[usize], kind: Option<&str>, device: &[usize], mut split: Option<Vec<f64>>) -> anyhow::Result<()> {
    let g = Gguf::open(model)?;
    let aux_g = open_mtp_gguf(aux)?;
    let m = Model::load_with_aux(&g, aux_g.as_ref(), &[])?;
    // Most drafts per step to measure: a DFlash block, else as many as the runs default to.
    let kload = m.cfg.dflash.as_ref().map_or(if m.draft_model.is_some() { 4 } else { 3 }, |d| d.block_size.saturating_sub(1).max(1));
    let tok = Tokenizer::from_gguf(&g)?;
    let n_pf = 4096.min(ctx.saturating_sub(256));
    anyhow::ensure!(n_pf >= 512, "--ctx too small to calibrate");
    let pf = prose(&tok, n_pf)?;
    let load = |chunk: usize, split: Option<&[f64]>| -> anyhow::Result<GpuModel> {
        let bk = device.iter().map(|&d| backend::open(kind, d)).collect::<anyhow::Result<Vec<_>>>()?;
        let mut gm = GpuModel::load_split_margins(&m, bk, ctx, chunk, margin, split, kload, 1).context(LOAD_HINT)?;
        gm.adapt_every = 0; // fixed placement: steadier timings
        Ok(gm)
    };

    let mut chunk = (0, 0.0);
    for c in [512, 1024, 2048, 4096].into_iter().filter(|&c| c <= ctx) {
        // A larger chunk's prefill scratch may not fit beside a long --ctx: keep the chunks measured so far.
        let mut gm = match load(c, split.as_deref()) { Ok(gm) => gm, Err(e) if chunk.0 > 0 => { say!("chunk {c:5}: {e:#}"); break } Err(e) => return Err(e) };
        let t = best(&mut || { gm.reset()?; gm.prefill(&pf).map(drop) })?;
        let tps = n_pf as f64 / t;
        say!("chunk {c:5}: prefill {n_pf} tok {tps:.0} tok/s");
        if tps > chunk.1 { chunk = (c, tps); }
    }

    let dec = tok.encode("Write a long story about a dragon.")?;
    // Several devices and no split given by hand: measure other layer splits and keep the fastest. The split by free
    // VRAM does not know a card's speed or its PCIe link; the timings do. A card with fewer layers keeps more of
    // their experts in VRAM (fewer uploads over its link in prefill), and the other cards then keep fewer.
    if device.len() > 1 && split.is_none() {
        // Prefill of the long prompt and decode after the short one, in tok/s, with the layers per device.
        let measure = |s: Option<&[f64]>| -> anyhow::Result<(Vec<usize>, f64, f64)> {
            let mut gm = load(chunk.0, s)?;
            gm.k = 1;
            let layers = gm.part_layers();
            let mut r = Runner::Gpu(Box::new(gm));
            let tp = best(&mut || { r.reset()?; r.prefill(&pf).map(drop) })?;
            let mut toks = 0;
            let td = best(&mut || {
                r.reset()?;
                let l = r.prefill(&dec)?;
                toks = decode(&mut r, &tok, false, &mut Sampler::greedy(), l, 64, None, &mut |_| Ok(false))?.0.len();
                Ok(())
            })?;
            Ok((layers, n_pf as f64 / tp, toks as f64 / td))
        };
        let (base, mut bp, mut bd) = measure(None)?;
        say!("split {base:?} (by free VRAM): prefill {bp:.0} tok/s, decode {bd:.1} tok/s");
        // ponytail: a fixed step of 15% of the layers and three rounds; a finer search if a report shows the best
        // split lies between two steps.
        let step = (m.cfg.n_layer * 3 / 20).max(1);
        let (mut cur, mut seen) = (base.clone(), vec![base.clone()]);
        for _ in 0..3 {
            let mut better = None;
            for c in split_moves(&cur, step) {
                if seen.contains(&c) { continue; }
                seen.push(c.clone());
                match measure(Some(&c.iter().map(|&x| x as f64).collect::<Vec<_>>())) {
                    Ok((_, p, d)) => {
                        say!("split {c:?}: prefill {p:.0} tok/s, decode {d:.1} tok/s, prefill x decode {:.3} of the best so far", p * d / (bp * bd));
                        // Prefill and decode count the same; 3% over the best so far, above the run-to-run spread.
                        if p * d > 1.03 * bp * bd { (bp, bd, better) = (p, d, Some(c)); }
                    }
                    Err(e) => say!("split {c:?}: {e:#}"),
                }
            }
            match better { Some(c) => cur = c, None => break }
        }
        if cur != base {
            chunk.1 = bp;
            split = Some(cur.iter().map(|&x| x as f64).collect());
        }
    }

    let mut gm = load(chunk.0, split.as_deref())?;
    let mut short = usize::MAX;
    for n in [8, 16, 32, 64, 128] {
        let p = &pf[..n];
        gm.short = usize::MAX;
        let ts = best(&mut || { gm.reset()?; gm.prefill(p).map(drop) })?;
        gm.short = 0;
        let tb = best(&mut || { gm.reset()?; gm.prefill(p).map(drop) })?;
        say!("prompt {n:4}: steps {:.1} ms, batched {:.1} ms", ts * 1e3, tb * 1e3);
        if tb < ts && short == usize::MAX { short = n; }
    }
    if short == usize::MAX { short = 256; }

    let drafts = m.cfg.n_nextn > 0 || m.cfg.dflash.is_some() || m.cfg.eagle3.is_some() || m.draft_model.is_some();
    let kmax = if drafts { gm.k } else { 0 };
    let mut r = Runner::Gpu(Box::new(gm));
    let mut mtp = (0, 0.0);
    for k in (0..=kmax).filter(|&k| k <= 4 || k % 2 == 0 || k == kmax) {
        if let Runner::Gpu(g) = &mut r { g.k = k.max(1); }
        let mut toks = 0;
        let t = best(&mut || {
            r.reset()?;
            let l = r.prefill(&dec)?;
            toks = decode(&mut r, &tok, k > 0, &mut Sampler::greedy(), l, 128, None, &mut |_| Ok(false))?.0.len();
            Ok(())
        })?;
        // `best` times prefill too; it is the same for every k, so the ranking holds.
        let tps = toks as f64 / t;
        say!("mtp {k}: decode {toks} tok {tps:.1} tok/s");
        if tps > mtp.1 { mtp = (k, tps); }
    }

    let Runner::Gpu(g) = &r else { unreachable!() };
    let path = calibration_path(model, g.gpu.name(), device);
    std::fs::create_dir_all(path.parent().unwrap())?;
    // The file size tells another file of the same name (another repo or quant source) from this one.
    let mut v = json!({"chunk": chunk.0, "mtp": mtp.0, "short": short, "ctx": ctx, "gpu": g.gpu.device(), "bytes": std::fs::metadata(model)?.len(),
        "measured": {"prefill_tok_s": chunk.1.round(), "decode_tok_s": (mtp.1 * 10.0).round() / 10.0}});
    if let Some(a) = aux.and_then(|a| a.file_name()) { v["aux"] = a.to_string_lossy().into(); }
    // Only a split that was measured as faster, or given by hand: without one, later runs split by the free VRAM
    // they find.
    if split.is_some() {
        v["split"] = json!({"devices": device, "gpus": g.parts().map(|p| p.gpu.device()).collect::<Vec<_>>(), "layers": g.part_layers()});
    }
    std::fs::write(&path, serde_json::to_string_pretty(&v)?)?;
    println!("{v}\nwritten to {}", path.display());
    Ok(())
}

/// Which tokens end a DRY match: those whose text has a newline, colon, double quote or asterisk (llama.cpp's
/// default breakers). Built once per model, as decoding the whole vocabulary takes a moment.
fn dry_breakers(tok: &Tokenizer, model: &str) -> std::sync::Arc<Vec<bool>> {
    static CACHE: std::sync::Mutex<Option<(String, std::sync::Arc<Vec<bool>>)>> = std::sync::Mutex::new(None);
    let mut c = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((m, b)) = c.as_ref() { if m == model { return b.clone(); } }
    let b: Vec<bool> = (0..tok.vocab_size() as u32)
        .map(|i| tok.decode(&[i]).is_ok_and(|s| s.contains(['\n', ':', '"', '*']))).collect();
    let b = std::sync::Arc::new(b);
    *c = Some((model.to_owned(), b.clone()));
    b
}

/// Decoding of up to `n` tokens after `logits`; `emit` gets each new complete piece of text and returns true to stop.
/// When `think_cap` is `Some((start_open, budget))` with `budget > 0`, forces `</think>\n\n` once `out.len() >= budget`
/// while still inside a `<think>` block so the model proceeds directly to the answer.
/// Returns the tokens and whether an end-of-turn token or `emit` stopped it.
fn decode(r: &mut Runner, tok: &Tokenizer, mtp: bool, smp: &mut Sampler, mut logits: Vec<f32>, n: usize,
          think_cap: Option<(bool, usize)>,
          emit: &mut dyn FnMut(&str) -> anyhow::Result<bool>) -> anyhow::Result<(Vec<u32>, bool)> {
    let eos = tok.eos_ids();
    // `ahead`: the next token when `spec_step` or `step_greedy` already picked it.
    // `start`: the first token whose text is not emitted yet (the text before it ends on a whole character).
    let (mut out, mut start, mut row, mut ahead, mut stopped) = (Vec::new(), 0, 0, None, false);
    let (mut in_think, budget) = think_cap.unwrap_or((false, 0));
    let (mut saw_close, mut tail) = (false, String::new());
    'outer: while out.len() < n {
        let next = match ahead.take() { Some(t) => t, None => smp.pick(&logits)? as u32 };
        if eos.contains(&next) { stopped = true; break; }
        let toks = match r {
            Runner::Gpu(g) if mtp && g.ngram.is_some() => {
                let (toks, nx, rw) = g.hybrid_spec_step(next, row, smp)?;
                (ahead, row) = (Some(nx), rw);
                toks
            }
            Runner::Gpu(g) if mtp => {
                let (toks, nx, rw) = g.spec_step(next, row, smp)?;
                (ahead, row) = (Some(nx), rw);
                toks
            }
            Runner::Gpu(g) if g.ngram.is_some() => {
                let (toks, nx) = g.ngram_spec_step(next, smp)?;
                ahead = Some(nx);
                toks
            }
            // Greedy without MTP: the argmax on the GPU, so only the token id is copied back, not the logits.
            Runner::Gpu(g) if smp.temp <= 0.0 => { ahead = Some(g.step_greedy(next)?); vec![next] }
            Runner::Slot(c) if smp.temp <= 0.0 => { ahead = Some(c.step(next, false)?.0); vec![next] }
            _ => { logits = r.step(next)?; vec![next] }
        };
        for t in toks {
            if eos.contains(&t) { stopped = true; break 'outer; }
            if out.len() == n { break; }
            out.push(t);
            let s = tok.decode(&out[start..])?;
            if !s.ends_with('\u{fffd}') {
                if budget > 0 && !saw_close {
                    tail.push_str(&s);
                    while tail.len() > 64 {
                        let c = tail.char_indices().nth(1).map_or(tail.len(), |(i, _)| i);
                        tail.drain(..c);
                    }
                    if !in_think && tail.contains("<think>") { in_think = true; }
                    if in_think && tail.contains("</think>") { in_think = false; saw_close = true; }
                }
                if emit(&s)? { stopped = true; break 'outer; }
                start = out.len();
            }
        }
        if budget > 0 && in_think && !saw_close && out.len() >= budget && out.len() < n {
            let close_str = if tail.ends_with('\n') { "</think>\n\n" } else { "\n</think>\n\n" };
            let mut close_ids = tok.encode(close_str)?;
            close_ids.truncate(n - out.len());
            if !close_ids.is_empty() {
                ahead = None;
                match r {
                    Runner::Gpu(g) => { logits = g.force_tokens(&close_ids, &mut row)?; }
                    Runner::Cpu(..) | Runner::Slot(..) | Runner::Own(..) => {
                        for &ct in &close_ids { logits = r.step(ct)?; }
                    }
                }
                for ct in close_ids {
                    out.push(ct);
                    let s = tok.decode(&out[start..])?;
                    if !s.ends_with('\u{fffd}') {
                        if emit(&s)? { stopped = true; break 'outer; }
                        start = out.len();
                    }
                }
            }
            in_think = false;
            saw_close = true;
        }
    }
    if let Runner::Gpu(g) = r && let Some(ng) = g.ngram.as_mut() {
        let _ = ng.save_dynamic();
    }
    Ok((out, stopped))
}

/// The answers `interface` asks for, as the arguments of the command to run: the model (a number from `models`, or
/// the last row and then a path to a file or to a folder to list), the mode, GPU or CPU, the context length and, for a server, the
/// address. An empty answer takes the default in brackets.
fn interface_args(input: &mut impl BufRead, mut models: Vec<(String, PathBuf, u64)>) -> anyhow::Result<Vec<String>> {
    let on = term::out();
    let mut ask = |q: &str, default: &str| -> anyhow::Result<String> {
        if default.is_empty() { print!("{q}: "); } else { print!("{q} {}: ", term::paint(on, "2", &format!("[{default}]"))); }
        std::io::stdout().flush()?;
        let mut line = String::new();
        anyhow::ensure!(input.read_line(&mut line)? > 0, "no answer (end of input)");
        Ok(match line.trim() { "" => default.to_string(), a => a.to_string() })
    };
    let head = |s: &str| println!("\n{}", term::paint(on, "1", s));
    let row = |k: usize, name: &str, note: &str| println!("  {}  {name}{}", term::paint(on, "36", &format!("{k:2}")), if note.is_empty() { String::new() } else { format!("  {}", term::paint(on, "2", note)) });

    let model = loop {
        head("Model");
        let w = models.iter().map(|m| m.0.len()).max().unwrap_or(0);
        for (k, (name, _, bytes)) in models.iter().enumerate() { row(k + 1, &format!("{name:w$}"), &format!("{:5.1} GB", *bytes as f64 / 1e9)); }
        // The last row asks for a path; a path typed in place of a number is taken as well.
        let other = models.len() + 1;
        if !models.is_empty() { row(other, "another folder or .gguf file", ""); }
        let mut a = if models.is_empty() { String::new() } else { ask("number", "1")? };
        if let Ok(k) = a.parse::<usize>() {
            if k != other {
                match models.get(k.wrapping_sub(1)) { Some(m) => break m.1.clone(), None => println!("no model {k}") }
                continue;
            }
            a.clear();
        }
        if a.is_empty() { a = ask("path to a folder of .gguf files or to one file", "")?; }
        if a.is_empty() { continue; }
        // A path dragged into a Windows console arrives in quotes; a typed `~/` is not expanded by a shell here.
        let a = a.trim_matches(['"', '\'']);
        let p = match (a.strip_prefix("~/"), std::env::var_os("HOME")) { (Some(rest), Some(h)) => PathBuf::from(h).join(rest), _ => PathBuf::from(a) };
        if p.is_file() { break p; }
        if !p.is_dir() { println!("{a}: not a file or a folder"); continue; }
        models = serve::list_models(&p);
        if models.is_empty() { println!("{a}: no .gguf files"); }
    };

    head("Mode");
    for (k, m) in ["chat in this terminal", "chat page in the browser", "API server (OpenAI-compatible)", "speed test"].iter().enumerate() { row(k + 1, m, ""); }
    let mode = loop { match ask("number", "1")?.parse::<usize>() { Ok(k @ 1..=4) => break k, _ => println!("a number from 1 to 4") } };

    head("Settings");
    let gpu = !ask("run on the GPU (y/n)", "y")?.to_lowercase().starts_with('n');
    let ctx = loop {
        let a = ask("context length in tokens", "default")?;
        if a == "default" || a.parse::<usize>().is_ok_and(|n| n > 0) { break a; }
        println!("a number of tokens, such as 8192");
    };
    let addr = if matches!(mode, 2 | 3) { ask("address", "127.0.0.1:8080")? } else { String::new() };

    let mut args: Vec<String> = vec![["chat", "serve", "serve", "bench"][mode - 1].into(), "--model".into(), model.to_string_lossy().into()];
    if gpu { args.push("--gpu".into()); }
    if ctx != "default" { args.extend(["--ctx".into(), ctx]); }
    if !addr.is_empty() { args.extend(["--addr".into(), addr]); }
    if mode == 2 { args.push("--open".into()); }
    Ok(args)
}

/// Tokens left for a reply: `n`, capped by the context after a `used`-token prompt.
fn room(e: &Engine, used: usize, n: usize) -> usize { n.min(e.ctx.saturating_sub(used)) }

/// The chat format from the GGUF's `tokenizer.chat_template`.
fn template(g: &Gguf) -> Template {
    let t = Template::detect(g.meta.get("tokenizer.chat_template").and_then(|v| v.as_str()));
    say!("chat template: {t:?}");
    t
}

/// Reply text without the thinking part (the template drops it from history too).
/// Streams one (reasoning, answer) pair as `reasoning_content` and `content` deltas.
fn send_split(w: &mut impl Write, chunk: &impl Fn(serde_json::Value, Option<&str>) -> serde_json::Value, (r, a): (String, String)) -> anyhow::Result<()> {
    if !r.is_empty() { write!(w, "data: {}\n\n", chunk(json!({"reasoning_content": r}), None))?; }
    if !a.is_empty() { write!(w, "data: {}\n\n", chunk(json!({"content": a}), None))?; }
    Ok(())
}

fn answer(s: &str) -> String { s.rsplit_once("</think>").map_or(s, |x| x.1).trim().to_string() }

/// A model `serve` has loaded.
struct Loaded<'a> { r: Runner<'a>, tok: &'a Tokenizer, tpl: Template, name: String, path: PathBuf }

/// What `serve` keeps across model loads.
struct Serving {
    /// Complete requests, with their key's place when `--max-per-key` counts it (freed once answered).
    rx: std::sync::Mutex<std::sync::mpsc::Receiver<(TcpStream, anyhow::Result<serve::Request>, usize, Option<serve::Hold>)>>,
    waiting: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    /// Set by SIGTERM.
    stop: &'static std::sync::atomic::AtomicBool,
    addr: String,
    allowed_hosts: Vec<String>,
    keys: std::sync::Arc<std::sync::RwLock<serve::Keys>>,
    usage: Option<std::sync::Mutex<std::fs::File>>,
    max_tokens: usize,
    dir: PathBuf,
    /// The --model, offered beside the models folder.
    /// `--alias`.
    alias: Option<String>,
    model: Option<PathBuf>,
    /// GGUF files outside the models folder that `/api/load` was given as `path`, offered until the server stops.
    extra: std::sync::Arc<std::sync::Mutex<Vec<PathBuf>>>,
    /// The Hugging Face download running on its own thread, or the last one.
    dl: std::sync::Arc<std::sync::Mutex<hf::Progress>>,
    stats: std::sync::Arc<std::sync::Mutex<serve::Stats>>,
    /// Load options of the last /api/load (`Engine::with`), applied before the next model loads.
    opts: std::sync::Mutex<serde_json::Value>,
    /// `--auto-unload`: the model is freed after this long without a chat request.
    idle: Option<Duration>,
    /// The model `--auto-unload` freed (while another loads for a chat request: the model to go back to if that load
    /// fails), and the chat request that waits for a load (answered once the model has loaded).
    asleep: std::sync::Mutex<Option<PathBuf>>,
    parked: std::sync::Mutex<Option<Queued>>,
}

/// A complete request from the queue, with its key's place when `--max-per-key` counts it (freed once answered).
type Queued = (TcpStream, anyhow::Result<serve::Request>, usize, Option<serve::Hold>);

/// Why `Serving::serve` returned.
enum Next {
    /// POST /api/load asked for this model; its connection waits for the outcome.
    Load(PathBuf, TcpStream),
    /// `--auto-unload`: no chat request for that long. The caller frees the model.
    Idle,
    /// A chat request came for the model that `--auto-unload` freed, or named another model the server offers; it
    /// waits in `Serving::parked`.
    Wake(PathBuf),
    /// The listener is gone, or SIGTERM came and the accepted requests are answered.
    Stop,
}

/// A flag that SIGTERM sets (instead of ending the process). Never set on Windows.
fn on_sigterm() -> &'static std::sync::atomic::AtomicBool {
    static STOP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    #[cfg(unix)]
    {
        extern "C" fn h(_: i32) { STOP.store(true, Relaxed); }
        unsafe extern "C" { fn signal(sig: i32, handler: extern "C" fn(i32)) -> usize; }
        // SAFETY: the handler only stores to an atomic, which is async-signal-safe. 15 is SIGTERM on Linux and macOS.
        unsafe { signal(15, h); }
    }
    &STOP
}

impl Serving {
    /// Answers requests until one asks for a model to be loaded, or until the model is to be freed or loaded again
    /// for `--auto-unload`.
    fn serve(&self, mut l: Option<&mut Loaded>, e: &Engine) -> Next {
        if l.is_some() { *self.asleep.lock().unwrap() = None; }
        if let Some(l) = l.as_deref_mut() && matches!(&l.r, Runner::Gpu(g) if g.slots() > 1) { return self.serve_slots(l, e); }
        let rx = self.rx.lock().unwrap();
        let mut last = Instant::now();
        loop {
            // The chat that asked for the freed model comes first.
            let parked = self.parked.lock().unwrap().take();
            let (c, q, i, hold) = match parked.map_or_else(|| rx.recv_timeout(Duration::from_millis(200)), Ok) {
                Ok(x) => x,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) if self.stopped() => { say!("stopped"); return Next::Stop }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) if l.is_some() && self.idle.is_some_and(|d| last.elapsed() >= d) => return Next::Idle,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return Next::Stop,
            };
            let chat = self.wants_model(&q);
            // The model the request names, else the one `--auto-unload` freed.
            if chat && let Some(p) = self.other(&q, l.as_deref().map(|l| l.path.as_path())).or_else(|| self.asleep.lock().unwrap().clone().filter(|_| l.is_none())) {
                *self.parked.lock().unwrap() = Some((c, q, i, hold));
                return Next::Wake(p);
            }
            let mut next = None;
            let res = handle(c, q, l.as_deref_mut(), e, self, i, &mut next);
            self.waiting.fetch_sub(1, Relaxed);
            if chat { last = Instant::now(); }
            if let Err(e) = res { say!("request {i}: {e:#}"); serve::exit_if_gpu_dead(&e); }
            if let Some((p, w)) = next { return Next::Load(p, w); }
        }
    }
    /// `--auto-unload` freed `model` (or its load failed after that). It stays in the stats and so in the model list,
    /// as a chat request loads it again.
    fn sleep(&self, model: PathBuf, ctx: usize) {
        let mut st = self.stats.lock().unwrap();
        st.unload(None);
        (st.model, st.file, st.unloaded, st.ctx) = (Some(self.name(&model)), stem(&model), true, ctx);
        *self.asleep.lock().unwrap() = Some(model);
    }
    /// A request for the model that `handle` does not refuse for its Host, key or content type. Only these count as
    /// use for `--auto-unload` and load the freed model again: the page, the stats and the model list do not.
    fn wants_model(&self, q: &anyhow::Result<serve::Request>) -> bool {
        let Ok(r) = q else { return false };
        let keys = self.keys.read().unwrap();
        r.method == "POST" && r.json && matches!(r.path.split('?').next(), Some("/v1/chat/completions" | "/v1/messages" | "/v1/messages/count_tokens"))
            && (keys.any() || serve::host_ok(&r.host, &self.addr, &self.allowed_hosts)) && keys.who(&r.auth).is_some()
    }
    /// SIGTERM came and the requests accepted before it are answered: `serve` returns None, so the expert profile
    /// is saved and the server exits.
    fn stopped(&self) -> bool { self.stop.load(Relaxed) && self.waiting.load(Relaxed) == 0 }
    /// The model's name in `/v1/models` and the replies: `--alias`, else the file name without `.gguf`.
    fn name(&self, p: &Path) -> String { self.alias.clone().unwrap_or_else(|| stem(p)) }
    fn models(&self) -> Vec<(String, PathBuf, u64)> { offered(&self.dir, self.model.as_ref(), &self.extra.lock().unwrap()) }
    /// The model a chat request names when it is not `loaded` and the server offers it (by file name, as
    /// `/v1/models` lists it, or by its name in the page's list): the server loads it for the request, as the page's
    /// model switch does. Only an admin key may (customer keys may not load models), and a name that no offered
    /// model has is answered by the loaded model.
    fn other(&self, q: &anyhow::Result<serve::Request>, loaded: Option<&Path>) -> Option<PathBuf> {
        #[derive(serde::Deserialize)]
        struct Named { model: Option<String> }
        let Ok(r) = q else { return None };
        if !self.wants_model(q) || !self.keys.read().unwrap().who(&r.auth).is_some_and(|w| w.admin) { return None; }
        let want = serde_json::from_slice::<Named>(&r.body).ok()?.model?;
        // Before the folders are read: most requests name the loaded model.
        if loaded.is_some_and(|p| want == self.name(p) || want == stem(p)) { return None; }
        // ponytail: two files with one name in different folders: the first of the list is meant.
        self.models().into_iter().find(|(id, p, _)| want == stem(p) || want == *id).map(|m| m.1).filter(|p| Some(p.as_path()) != loaded)
    }

    /// Adds a GGUF file from anywhere on this computer to the list (`/api/load` with `path`). Only the file's
    /// first four bytes are read, so the route tells a caller no more than whether a path is a GGUF file.
    fn add(&self, p: &Path) -> anyhow::Result<()> {
        anyhow::ensure!(p.is_absolute(), "{}: give the full path of the file", p.display());
        let mut magic = [0; 4];
        std::fs::File::open(p).and_then(|mut f| f.read_exact(&mut magic)).with_context(|| p.display().to_string())?;
        anyhow::ensure!(&magic == b"GGUF", "{}: not a GGUF file", p.display());
        let mut x = self.extra.lock().unwrap();
        if !x.iter().any(|q| q == p) { x.push(p.to_path_buf()); }
        Ok(())
    }
}

/// What a request thread asks of the model's thread under `--parallel` (`Serving::serve_slots`).
enum Job {
    /// `prefill_cached` in a free slot; answers the logits, the tokens reused, the slot and when the prefill began.
    /// `gone` is set when the client leaves while the chat waits for a slot; the chat is then dropped.
    Chat { ids: Vec<u32>, sys: usize, keep: usize, owner: u64, gone: std::sync::Arc<std::sync::atomic::AtomicBool>, back: std::sync::mpsc::Sender<anyhow::Result<(Vec<f32>, usize, usize, Instant)>> },
    /// One decode step in a slot the request holds; answers the greedy next token and the logits if `logits`.
    Step { slot: usize, token: u32, logits: bool, back: std::sync::mpsc::Sender<anyhow::Result<(u32, Vec<f32>)>> },
    /// The request is done with its slot.
    Done(usize),
}

/// A request's handle on the model under `--parallel`; dropping it frees the slot.
/// `conn` is the request's connection, watched while the chat waits for a slot; `started` is when the model took
/// the chat, after that wait.
struct Client { tx: std::sync::mpsc::Sender<Job>, slot: Option<usize>, owner: u64, conn: Option<TcpStream>, started: Option<Instant> }

impl Client {
    fn chat(&mut self, ids: &[u32], sys: usize, keep: usize) -> anyhow::Result<(Vec<f32>, usize)> {
        if let Some(s) = self.slot.take() { let _ = self.tx.send(Job::Done(s)); }
        let (back, r) = std::sync::mpsc::channel();
        let gone = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        self.tx.send(Job::Chat { ids: ids.to_vec(), sys, keep, owner: self.owner, gone: gone.clone(), back }).map_err(|_| anyhow::anyhow!("model gone"))?;
        let (logits, reused, s, started) = loop {
            match r.recv_timeout(Duration::from_millis(250)) {
                Ok(x) => break x?,
                // A client that left is flagged, and the wait goes on: the model drops the chat if it is still queued
                // (the channel closes), or sends its slot if the prefill already started, which `handle` then frees.
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => if !gone.load(Relaxed) && self.conn.as_ref().is_some_and(serve::gone) {
                    gone.store(true, Relaxed);
                },
                Err(_) if gone.load(Relaxed) => anyhow::bail!("client closed the connection"),
                Err(_) => anyhow::bail!("model gone"),
            }
        };
        (self.slot, self.started) = (Some(s), Some(started));
        Ok((logits, reused))
    }
    fn step(&mut self, token: u32, logits: bool) -> anyhow::Result<(u32, Vec<f32>)> {
        let slot = self.slot.context("step before chat")?;
        let (back, r) = std::sync::mpsc::channel();
        self.tx.send(Job::Step { slot, token, logits, back }).map_err(|_| anyhow::anyhow!("model gone"))?;
        r.recv()?
    }
}

impl Drop for Client {
    fn drop(&mut self) { if let Some(s) = self.slot { let _ = self.tx.send(Job::Done(s)); } }
}

/// The model thread's view of the slots under `--parallel` (`Serving::serve_slots`).
#[derive(Default)]
struct Sched {
    /// Slots held by a request, and when each was last freed (the least recently used goes first).
    held: Vec<bool>,
    freed: Vec<u64>,
    tick: u64,
    /// The key that holds each slot (`Client::owner`).
    who: Vec<u64>,
    /// Chat jobs waiting for a free slot, and decode steps waiting for a forward (since the first came).
    chats: std::collections::VecDeque<Job>,
    steps: Vec<(usize, u32, bool, std::sync::mpsc::Sender<anyhow::Result<(u32, Vec<f32>)>>)>,
    since: Option<Instant>,
}

impl Sched {
    /// How long a decode step waits for the other decoding slots' steps to share its forward.
    const GATHER: Duration = Duration::from_millis(2);

    fn decoding(&self) -> usize { self.held.iter().filter(|&&h| h).count() }

    /// The queued chat to start next: the one whose key holds the fewest slots, the oldest among those, so that one
    /// customer's burst of requests does not hold back the others'. With one key (or none) it is the oldest.
    fn next_chat(&self) -> Option<usize> {
        let holds = |o: u64| self.held.iter().zip(&self.who).filter(|&(&h, &w)| h && w == o).count();
        (0..self.chats.len()).min_by_key(|&j| (if let Job::Chat { owner, .. } = &self.chats[j] { holds(*owner) } else { usize::MAX }, j))
    }

    /// How long to wait for jobs: until the steps of all decoding slots are in, at most `GATHER` after the first.
    fn wait(&self) -> Duration {
        match self.since {
            Some(t) if self.steps.len() < self.decoding() => Self::GATHER.saturating_sub(t.elapsed()),
            Some(_) => Duration::ZERO,
            None => Duration::from_millis(5),
        }
    }

    /// Take the jobs that come within `wait`.
    fn take(&mut self, jobs: &std::sync::mpsc::Receiver<Job>, wait: Duration) {
        let mut got: Vec<Job> = jobs.recv_timeout(wait).into_iter().collect();
        got.extend(jobs.try_iter());
        for j in got {
            match j {
                Job::Done(s) => { self.held[s] = false; self.tick += 1; self.freed[s] = self.tick; }
                Job::Step { slot, token, logits, back } => { self.since.get_or_insert_with(Instant::now); self.steps.push((slot, token, logits, back)); }
                c => self.chats.push_back(c),
            }
        }
    }

    fn ready(&self) -> bool { self.since.is_some_and(|t| self.steps.len() >= self.decoding() || t.elapsed() >= Self::GATHER) }

    /// One forward over the oldest steps (with more than `STEP_ROWS` slots decoding, each waits its turn), in slot order
    /// (fewer graphs to capture).
    fn run(&mut self, g: &mut GpuModel) {
        // ponytail: fixed row count from which the GEMM forward (`step_batch`) replaces the matvec one; `calibrate`
        // should measure it per model and machine. STRATA_BATCH_MIN overrides.
        let min = std::env::var("STRATA_BATCH_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
        let big = g.batch_cap() > 0 && self.steps.len() >= min;
        let cap = if big { g.batch_cap() } else { rusting_strata::gpu_model::STEP_ROWS };
        let mut batch: Vec<_> = self.steps.drain(..self.steps.len().min(cap)).collect();
        batch.sort_by_key(|s| s.0);
        self.since = if self.steps.is_empty() { None } else { Some(Instant::now() - Self::GATHER) };
        let rows: Vec<_> = batch.iter().map(|b| (b.0, b.1, b.2)).collect();
        match if big { g.step_batch(&rows) } else { g.step_rows(&rows) } {
            Ok(rows) => for (b, l) in batch.into_iter().zip(rows) { let _ = b.3.send(Ok(l)); },
            Err(err) => for b in batch { let _ = b.3.send(Err(serve::copy_err(&err))); },
        }
    }
}

impl Serving {
    /// `serve` with several slots (`--parallel`): each request runs `handle` on its own thread and sends its model
    /// work here, where prefills run one at a time (the decoding slots run between its chunks) and the decode steps of
    /// up to `TMAX` slots share a forward.
    fn serve_slots(&self, l: &mut Loaded, e: &Engine) -> Next {
        use std::sync::mpsc::{TryRecvError, channel};
        let Loaded { r, tok, tpl, name, path } = l;
        let Runner::Gpu(g) = r else { unreachable!() };
        let (tx, jobs) = channel::<Job>();
        let (next, live) = (std::sync::Mutex::new(None), std::sync::atomic::AtomicUsize::new(0));
        // `--auto-unload`: when the last chat request ended, and whether the loop ended for it.
        let (last, mut idle, mut wake) = (std::sync::Mutex::new(Instant::now()), false, None);
        let rx = self.rx.lock().unwrap();
        let n = g.slots();
        std::thread::scope(|sc| {
            // Owned here, so that a panic drops the queued jobs' reply channels and the request threads end (the
            // scope then joins them and the panic goes on).
            let (jobs, mut s) = (jobs, Sched { held: vec![false; n], freed: vec![0; n], who: vec![0; n], ..Default::default() });
            let mut open = true;
            loop {
                while open && wake.is_none() && next.lock().unwrap().is_none() {
                    let parked = self.parked.lock().unwrap().take();
                    match parked.map_or_else(|| rx.try_recv(), Ok) {
                        // A chat request names another model: it waits while the replies in progress end.
                        Ok(x) if let Some(p) = self.other(&x.1, Some(path)) => { *self.parked.lock().unwrap() = Some(x); wake = Some(p); }
                        Ok((c, q, i, hold)) => {
                            live.fetch_add(1, Relaxed);
                            let (tx, tok, tpl, name, path, next, live, last) = (tx.clone(), &**tok, *tpl, name.clone(), path.clone(), &next, &live, &last);
                            sc.spawn(move || {
                                let chat = self.wants_model(&q);
                                let mut l = Loaded { r: Runner::Slot(Client { tx, slot: None, owner: 0, conn: None, started: None }), tok, tpl, name, path };
                                let mut nx = None;
                                // A panic still gives back the counts below, so the model thread can switch models.
                                let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handle(c, q, Some(&mut l), e, self, i, &mut nx)))
                                    .unwrap_or_else(|_| Err(anyhow::anyhow!("panicked")));
                                drop((l, hold));
                                self.waiting.fetch_sub(1, Relaxed);
                                if let Err(e) = res { say!("request {i}: {e:#}"); serve::exit_if_gpu_dead(&e); }
                                if nx.is_some() { *next.lock().unwrap() = nx; }
                                if chat { *last.lock().unwrap() = Instant::now(); }
                                live.fetch_sub(1, Relaxed);
                            });
                        }
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => open = false,
                    }
                }
                if open && self.stopped() { say!("stopped"); open = false; }
                if (!open || wake.is_some() || next.lock().unwrap().is_some()) && live.load(Relaxed) == 0 { break; }
                // A request that comes now stays in the queue and loads the model again.
                if live.load(Relaxed) == 0 && self.idle.is_some_and(|d| last.lock().unwrap().elapsed() >= d) { idle = true; break; }
                s.take(&jobs, s.wait());
                // Prefill between decode steps, in the free slot whose saved state the prompt extends the furthest, else
                // the least recently used one.
                s.chats.retain(|j| !matches!(j, Job::Chat { gone, .. } if gone.load(Relaxed)));
                // Models with a batched forward prefill the queued prompts together, one free slot each.
                let lim = g.batch_cap().max(1);
                let fits = |c: &Job| matches!(c, Job::Chat { ids, .. } if ids.len() <= g.ctx());
                let mut adm: Vec<(usize, Job)> = vec![];
                while adm.len() < lim {
                    let Some(j) = s.next_chat() else { break };
                    if !adm.is_empty() && !(fits(&s.chats[j]) && fits(&adm[0].1)) { break; }
                    let Job::Chat { ids, keep, owner, .. } = &s.chats[j] else { break };
                    let Some(sl) = (0..n).filter(|&i| !s.held[i]).max_by_key(|&i| {
                        let (sv, o) = g.slot_saved(i);
                        (if o == *owner && ids[..*keep].starts_with(sv) { sv.len() } else { 0 }, std::cmp::Reverse(s.freed[i]))
                    }) else { break };
                    // Held while the rest are chosen, so that `next_chat` counts it against its key.
                    (s.held[sl], s.who[sl]) = (true, *owner);
                    adm.push((sl, s.chats.remove(j).unwrap()));
                }
                for (sl, _) in &adm { s.held[*sl] = false; }
                if !adm.is_empty() {
                    // Between two prefill forwards the decoding slots run for as long as the forward before took, so
                    // long prompts share the GPU with them evenly instead of stalling them.
                    // ponytail: an even share; weigh it (or cap the prompt's chunk) if first-token time matters more.
                    let (started, mut t0) = (Instant::now(), Instant::now());
                    let res: Vec<anyhow::Result<(Vec<f32>, usize)>> = {
                        let mut between = |g: &mut GpuModel| {
                            let end = Instant::now() + t0.elapsed();
                            while s.decoding() > 0 && Instant::now() < end {
                                s.take(&jobs, s.wait().min(end.saturating_duration_since(Instant::now())));
                                if s.ready() { s.run(g); }
                            }
                            t0 = Instant::now();
                            Ok(())
                        };
                        if adm.len() > 1 {
                            let list: Vec<_> = adm.iter().map(|(sl, c)| {
                                let Job::Chat { ids, sys, keep, owner, .. } = c else { unreachable!() };
                                (*sl, *owner, &ids[..], *sys, *keep)
                            }).collect();
                            match g.prefill_slots(&list, &mut between) {
                                Ok(v) => v.into_iter().map(Ok).collect(),
                                Err(e) => adm.iter().map(|_| Err(serve::copy_err(&e))).collect(),
                            }
                        } else {
                            let (sl, Job::Chat { ids, sys, keep, owner, .. }) = &adm[0] else { unreachable!() };
                            g.use_slot(*sl);
                            g.owner = *owner;
                            vec![g.prefill_cached_with(ids, *sys, *keep, &mut between)]
                        }
                    };
                    for ((sl, c), r) in adm.into_iter().zip(res) {
                        let Job::Chat { owner, back, .. } = c else { unreachable!() };
                        (s.held[sl], s.who[sl]) = (r.is_ok(), owner);
                        // A client that left during the prefill takes no slot.
                        if back.send(r.map(|(l, u)| (l, u, sl, started))).is_err() && s.held[sl] {
                            s.held[sl] = false;
                            s.tick += 1;
                            s.freed[sl] = s.tick;
                        }
                    }
                    gpu_model_stats(g, &mut self.stats.lock().unwrap());
                }
                if s.ready() { s.run(g); }
            }
        });
        match (next.into_inner().unwrap(), wake) { (Some((p, w)), _) => Next::Load(p, w), (None, Some(p)) => Next::Wake(p), (None, None) if idle => Next::Idle, _ => Next::Stop }
    }
}

/// Resolve `--mtp-model` / `--draft-model` (`.gguf` file, HF `.safetensors` file, or directory containing either
/// `model.gguf`/`model-dflash-q8_0.gguf` or `model.safetensors` + `config.json`) and open it as a companion GGUF.
/// If `.gguf` is missing (or `STRATA_EAGLE3_RECONVERT` / `STRATA_DFLASH_RECONVERT` is set) and `.safetensors` +
/// `config.json` are present, converts natively in Rust.
fn open_mtp_gguf(path: Option<&Path>) -> anyhow::Result<Option<Gguf>> {
    let Some(p) = path else { return Ok(None) };
    let reconvert = std::env::var_os("STRATA_DFLASH_RECONVERT").is_some()
        || std::env::var_os("STRATA_EAGLE3_RECONVERT").is_some();
    let cfg_probe = if p.is_dir() {
        p.join("config.json")
    } else if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
        p.with_file_name("config.json")
    } else {
        PathBuf::new()
    };
    let is_dflash = cfg_probe.exists() && rusting_strata::gguf::is_dflash_config(&cfg_probe);
    let use_q8 = std::env::var_os("STRATA_DFLASH_Q8").is_some_and(|v| v != "0");
    let gguf_name = if is_dflash {
        if use_q8 { "model-dflash-q8_0.gguf" } else { "model-dflash-q4_0.gguf" }
    } else {
        "model.gguf"
    };
    let (mut gguf, st_cfg) = if p.is_dir() {
        (p.join(gguf_name), Some((p.join("model.safetensors"), p.join("config.json"))))
    } else if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
        let out = if is_dflash { p.with_file_name(gguf_name) } else { p.with_extension("gguf") };
        (out, Some((p.to_path_buf(), p.with_file_name("config.json"))))
    } else if p.extension().and_then(|e| e.to_str()) == Some("bin") {
        (p.with_file_name("model.gguf"), None)
    } else {
        (p.to_path_buf(), None)
    };
    if p.is_dir() && !gguf.exists() && !p.join("model.safetensors").exists() {
        if let Ok(rd) = std::fs::read_dir(p) {
            let mut found: Vec<PathBuf> = rd
                .flatten()
                .map(|e| e.path())
                .filter(|f| f.extension().and_then(|e| e.to_str()) == Some("gguf"))
                .collect();
            found.sort();
            if let Some(first) = found.into_iter().next() {
                gguf = first;
            }
        }
    }
    if let Some((st, cfg)) = st_cfg
        && (!gguf.exists() || reconvert)
        && st.exists()
        && cfg.exists()
    {
        if is_dflash {
            say!("note: native Rust DFlash conversion {} -> {}", st.display(), gguf.display());
            rusting_strata::gguf::convert_safetensors_dflash_to_gguf(&st, &cfg, &gguf, 40)?;
        } else {
            say!("note: native Rust EAGLE3 conversion {} -> {}", st.display(), gguf.display());
            rusting_strata::gguf::convert_safetensors_eagle3_to_gguf(&st, &cfg, &gguf, 48)?;
        }
    }
    anyhow::ensure!(
        gguf.exists(),
        "{}: not found (for PyTorch .bin EAGLE3 checkpoints, convert first via tools/convert_eagle3_to_gguf.py)",
        gguf.display()
    );
    Ok(Some(Gguf::open(&gguf)?))
}

/// Loads `model`, answers `w` (the /api/load request that asked for it) and serves until another model is asked
/// for or `--auto-unload` ends it. The model and its GPU memory are freed on return.
/// A model file's name without `.gguf`: its name in `/v1/models` and in the replies unless `--alias` gives one.
fn stem(p: &Path) -> String { p.file_stem().and_then(|n| n.to_str()).unwrap_or("strata").to_string() }

/// The models the server offers: the models folder, the Hugging Face and llama.cpp caches, then `--model` and the
/// files `/api/load` was given by path, if they are elsewhere.
fn offered(dir: &Path, model: Option<&PathBuf>, extra: &[PathBuf]) -> Vec<(String, PathBuf, u64)> {
    let mut v = serve::list_models(dir);
    for m in serve::cached_models() { if !v.iter().any(|x| x.1 == m.1) { v.push(m); } }
    for p in model.into_iter().chain(extra) {
        if !v.iter().any(|x| &x.1 == p) { v.push((p.display().to_string(), p.clone(), std::fs::metadata(p).map_or(0, |m| m.len()))); }
    }
    v
}

fn serve_model(model: &Path, mut engine: Engine, s: &Serving, w: &mut Option<TcpStream>) -> anyhow::Result<Next> {
    say!("loading {}", model.display());
    let name = s.name(model);
    {
        let mut st = s.stats.lock().unwrap();
        st.unload(Some(name.clone()));
        st.file = stem(model);
        st.ctx = engine.ctx;
        st.ram_cap = engine.ram_cap();
    }
    let g = Gguf::open(model)?;
    engine.own_warn(&g);
    let arch = g.get("general.architecture")?.as_str().context("general.architecture")?;
    s.stats.lock().unwrap().ctx_train = g.meta.get(&format!("{arch}.context_length")).and_then(|v| v.as_u64());
    let tok = Tokenizer::from_gguf(&g)?;
    // Declared before the model, so it is dropped after it.
    let _own = rusting_strata::config::OWN_ARCHS.contains(&arch).then(OwnState::new);
    let (aux_g, m);
    let mut r = if _own.is_some() {
        let mmproj = engine.mmproj.clone().or_else(|| vision::find_sibling(model));
        Runner::Own(match arch {
            "gemma4" => rusting_strata::gemma4::serve(&g, &tok, &name, mmproj.as_deref())?,
            "glm5-next" => rusting_strata::glm::serve(&g, &tok, &name)?,
            "deepseek41" => rusting_strata::deepseek::chat::serve(&g, &tok, &name)?,
            rusting_strata::hy4::ARCH => rusting_strata::hy4::chat::serve(&g, &tok, &name)?,
            _ => rusting_strata::inkling2::chat::serve(&g, &tok, &name, rusting_strata::inkling2::chat::Inkling2Opts {
                max_ctx: 0, sampler: engine.sampler(), thinking_budget: engine.thinking_budget, think_default: engine.think_default,
                mmproj, draft_model: engine.draft_model.clone(), ngram_cfg: engine.ngram_config()? })?,
        })
    } else {
        aux_g = open_mtp_gguf(engine.aux_model_path())?;
        m = { let mut m = Model::load_with_aux(&g, aux_g.as_ref(), &engine.lora_specs())?; if !engine.lora.is_empty() { m.apply_lora(&engine.lora_specs())?; } m };
        engine.runner(&m, model)?
    };
    // Serving decodes one reply after a (usually cached) prompt: follow the reply's routing, not the prompt's.
    if let Runner::Gpu(g) = &mut r { g.set_decode_only(true); }
    {
        let mut st = s.stats.lock().unwrap();
        st.loading = false;
        (st.backend, st.device, st.ctx, st.slots) = match &r {
            Runner::Gpu(g) => (g.gpu.name().into(), g.gpu.device(), engine.ctx, g.slots()),
            Runner::Own(_) => { let d = rusting_strata::deepseek::mat::dev_stats(); (d.as_ref().map_or("cpu", |d| d.0).into(), d.map(|d| d.1).unwrap_or_default(), rusting_strata::ctx_limit(st.ctx_train.unwrap_or(0) as usize), 1) }
            _ => ("cpu".into(), String::new(), engine.ctx, 1),
        };
        gpu_stats(&r, &mut st);
    }
    let mut l = Loaded { r, tok: &tok, tpl: if _own.is_some() { Template::Xml } else { template(&g) }, name: name.clone(), path: model.to_path_buf() };
    say!("loaded {}", l.name);
    if let Some(mut w) = w.take() {
        let b = json!({"loaded": l.name}).to_string();
        let _ = write!(w, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len());
    }
    let next = s.serve(Some(&mut l), &engine);
    s.stats.lock().unwrap().unload(None);
    // A failed profile save must not stop the server between two models.
    if let Err(e) = l.r.finish() { say!("{e:#}"); }
    Ok(next)
}

/// The side panel's device memory, expert cache and MTP counts.
fn gpu_stats(r: &Runner, s: &mut serve::Stats) {
    match r {
        Runner::Gpu(g) => gpu_model_stats(g, s),
        Runner::Own(_) => s.vram = rusting_strata::deepseek::mat::dev_stats().and_then(|d| d.2),
        _ => {}
    }
}

/// What a model outside the main engine leaves in the process: the GPUs of `deepseek::mat` and the `STRATA_*`
/// defaults its architecture set. Dropping it frees the GPUs and puts the variables back, so the next model (of any
/// architecture) starts clean.
struct OwnState(Vec<(&'static str, Option<std::ffi::OsString>)>);

impl OwnState {
    fn new() -> OwnState {
        OwnState(["STRATA_DS_GPU", "STRATA_DS_GPU_FRACTION", "STRATA_DS_GPU_EXPERTS", "STRATA_DS_NOEVICT", "STRATA_GLM_PREFETCH"].map(|k| (k, std::env::var_os(k))).into())
    }
}

impl Drop for OwnState {
    fn drop(&mut self) {
        rusting_strata::deepseek::mat::dev_free();
        for (k, v) in &self.0 {
            unsafe { match v { Some(v) => std::env::set_var(k, v), None => std::env::remove_var(k) } }
        }
    }
}

fn gpu_model_stats(g: &GpuModel, s: &mut serve::Stats) {
    s.vram = g.gpu.mem_info().ok().map(|(f, t)| ((t - f) as u64, t as u64));
    s.experts = g.cache.as_ref().map(|_| (g.hits as u64, g.misses as u64));
    s.drafts = (g.drafts.0 > 0).then_some((g.drafts.0 as u64, g.drafts.1 as u64));
}

/// Default models folder: `$XDG_DATA_HOME`, `%LOCALAPPDATA%` on Windows, else `~/.local/share`.
fn models_dir_default() -> PathBuf {
    std::env::var_os("XDG_DATA_HOME").or_else(|| std::env::var_os("LOCALAPPDATA")).map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".local/share"))
        .join("rusting_strata").join("models")
}

fn open_browser(url: &str) {
    #[cfg(windows)]
    let r = std::process::Command::new("cmd").args(["/c", "start", "", url]).spawn();
    #[cfg(target_os = "macos")]
    let r = std::process::Command::new("open").arg(url).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let r = std::process::Command::new("xdg-open").arg(url).spawn();
    if let Err(e) = r { say!("could not open a browser ({e}); open {url}"); }
}

/// One Messages API stream event.
fn sse(w: &mut impl Write, v: serde_json::Value) -> anyhow::Result<()> { Ok(write!(w, "event: {}\ndata: {v}\n\n", v["type"].as_str().unwrap_or_default())?) }

/// Streamed reply text as a Messages API text block (index 0), started by the first piece.
fn text_delta(w: &mut impl Write, opened: &mut bool, t: &str) -> anyhow::Result<()> {
    if !std::mem::replace(opened, true) { sse(w, json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}))?; }
    sse(w, json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": t}}))
}

/// The `timings` block llama-server puts in its responses. llama-swap and other
/// OpenAI-compatible clients read it for prefill/decode rates. `prompt_n` is the
/// uncached part of the prompt and `cache_n` the part served from cache, as in
/// llama-server.
fn llama_timings(l: &serve::Last) -> serde_json::Value {
    let prompt_n = l.prompt_tokens.saturating_sub(l.cached_tokens);
    let per_token_ms = |ms: f64, n: usize| if n == 0 { 0.0 } else { ms / n as f64 };
    let per_second = |ms: f64, n: usize| if ms > 0.0 { n as f64 * 1e3 / ms } else { 0.0 };
    json!({
        "prompt_n": prompt_n,
        "prompt_ms": l.prefill_ms,
        "prompt_per_token_ms": per_token_ms(l.prefill_ms, prompt_n),
        "prompt_per_second": per_second(l.prefill_ms, prompt_n),
        "predicted_n": l.completion_tokens,
        "predicted_ms": l.decode_ms,
        "predicted_per_token_ms": per_token_ms(l.decode_ms, l.completion_tokens),
        "predicted_per_second": per_second(l.decode_ms, l.completion_tokens),
        "cache_n": l.cached_tokens,
    })
}

/// The chat page, served by the connection threads of `strata serve`.
use rusting_strata::serve::PAGE;

/// The Messages API needs the main engine: for another model the request gets a 400, and true is returned.
fn anthropic_refused(w: &mut TcpStream, r: &Runner) -> bool {
    let own = matches!(r, Runner::Own(_));
    if own { serve::send(w, "400 Bad Request", "application/json", r#"{"error":{"message":"the Messages API is not available for this model; use /v1/chat/completions","type":"invalid_request_error"}}"#); }
    own
}

fn handle(mut w: TcpStream, q: anyhow::Result<serve::Request>, l: Option<&mut Loaded>, e: &Engine, s: &Serving, i: usize,
    next: &mut Option<(PathBuf, TcpStream)>) -> anyhow::Result<()> {
    let send = |w: &mut TcpStream, status: &str, v: serde_json::Value| -> anyhow::Result<()> {
        let b = v.to_string();
        Ok(write!(w, "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len())?)
    };
    // A malformed request gets a 400 (best effort: the client may be gone) and is still logged.
    let serve::Request { method, path, host, auth, json, body, .. } = match q {
        Ok(q) => q,
        Err(err) => {
            let _ = send(&mut w, "400 Bad Request", json!({"error": {"message": format!("{err:#}"), "type": "invalid_request_error"}}));
            return Err(err);
        }
    };
    let created = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_secs();
    // Browsers can reach a loopback server: the Host check stops DNS rebinding, and requiring a JSON content type
    // makes a cross-site POST need a CORS preflight, which is never answered.
    // With a key the check is not needed (a rebinding page has no key) and would refuse a reverse proxy's Host.
    let keys = s.keys.read().unwrap().clone();
    if !keys.any() && !serve::host_ok(&host, &s.addr, &s.allowed_hosts) { return send(&mut w, "403 Forbidden", json!({"error": {"message": "bad Host header: name this host with --allowed-hosts, or set --api-key", "type": "forbidden"}})); }
    let Some(who) = keys.who(&auth) else { return send(&mut w, "401 Unauthorized", json!({"error": {"message": "missing or wrong API key", "type": "invalid_api_key"}})) };
    if method == "POST" && !json { return send(&mut w, "415 Unsupported Media Type", json!({"error": {"message": "Content-Type must be application/json", "type": "invalid_request_error"}})); }
    // Claude Code adds `?beta=true`.
    let route = path.split('?').next().unwrap_or_default();
    // Customer keys may not load models, download files or read the server's stats.
    if !who.admin && !matches!(route, "/v1/chat/completions" | "/v1/messages" | "/v1/messages/count_tokens" | "/v1/models") {
        return send(&mut w, "403 Forbidden", json!({"error": {"message": "this key may use only /v1/chat/completions, /v1/messages (and count_tokens) and /v1/models", "type": "permission_error"}}));
    }
    match (method.as_str(), route) {
        ("GET", "/api/models") => {
            // The model `--auto-unload` freed counts as loaded: a chat loads it again.
            let asleep = s.asleep.lock().unwrap().clone();
            let models: Vec<_> = s.models().into_iter().map(|(id, p, n)| json!({"id": id, "bytes": n, "image": e.mmproj.is_some() || vision::find_sibling(&p).is_some(), "loaded": l.as_ref().is_some_and(|l| l.path == p) || asleep.as_ref() == Some(&p)})).collect();
            // The values in effect: with a model loaded, `chunk` and `mtp` include its calibration.
            let options = json!({"ctx": e.ctx, "kv": e.kv, "cache_ram": e.cache_ram, "vram_margin": e.vram_margin.join(","), "mtp": e.mtp, "chunk": e.chunk,
                "gpu": e.gpu, "ctx_train": s.stats.lock().unwrap().ctx_train,
                // false: a GPU model with no calibration file (the page offers to calibrate it).
                "calibrated": l.as_ref().filter(|l| e.gpu && !matches!(l.r, Runner::Own(_))).map(|l| calibration_path(&l.path, &s.stats.lock().unwrap().backend, &e.device).exists())});
            send(&mut w, "200 OK", json!({"dir": s.dir, "models": models, "options": options}))
        }
        // Answered once the model has loaded (or failed to).
        ("POST", "/api/load") => {
            let v = serde_json::from_slice::<serde_json::Value>(&body).unwrap_or_default();
            let id = v["model"].as_str().unwrap_or_default();
            if let Err(err) = e.with(&v["options"]) { return send(&mut w, "400 Bad Request", json!({"error": {"message": format!("{err:#}"), "type": "invalid_request_error"}})); }
            // `path` names a file outside the models folder; it joins the list.
            let path = v["path"].as_str().map(PathBuf::from);
            if let Some(p) = &path && let Err(err) = s.add(p) { return send(&mut w, "400 Bad Request", json!({"error": {"message": format!("{err:#}"), "type": "invalid_request_error"}})); }
            match s.models().into_iter().find(|m| path.as_ref().map_or(m.0 == id, |p| &m.1 == p)) {
                Some((_, p, _)) => {
                    // `calibrate: true` measures the model first; it travels with the options to the load loop.
                    let mut o = v["options"].clone();
                    if v["calibrate"] == true { if !o.is_object() { o = json!({}); } o["calibrate"] = true.into(); }
                    *s.opts.lock().unwrap() = o; *next = Some((p, w)); Ok(())
                }
                None => send(&mut w, "404 Not Found", json!({"error": {"message": format!("no model {id:?} in {}", s.dir.display()), "type": "not_found"}})),
            }
        }
        ("POST", "/api/hf/files") => {
            let repo = serde_json::from_slice::<serde_json::Value>(&body).ok().and_then(|v| v["repo"].as_str().map(String::from)).unwrap_or_default();
            match hf::list(&repo) {
                Ok(f) => send(&mut w, "200 OK", json!({"files": f.into_iter().map(|(f, n)| json!({"file": f, "bytes": n})).collect::<Vec<_>>()})),
                Err(err) => send(&mut w, "502 Bad Gateway", json!({"error": {"message": format!("{err:#}"), "type": "server_error"}})),
            }
        }
        ("GET", "/api/download") => send(&mut w, "200 OK", json!(*s.dl.lock().unwrap())),
        ("POST", "/api/download/cancel") => { s.dl.lock().unwrap().cancel = true; send(&mut w, "200 OK", json!(*s.dl.lock().unwrap())) }
        // Runs on its own thread so chats go on meanwhile; the page polls GET /api/download.
        ("POST", "/api/download") => {
            let v: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
            let (repo, file) = (v["repo"].as_str().unwrap_or_default().to_string(), v["file"].as_str().unwrap_or_default().to_string());
            let mut g = s.dl.lock().unwrap();
            if g.running { drop(g); return send(&mut w, "409 Conflict", json!({"error": {"message": "a download is already running", "type": "conflict"}})); }
            *g = hf::Progress { repo: repo.clone(), file: file.clone(), running: true, ..Default::default() };
            drop(g);
            let (dl, dir) = (s.dl.clone(), s.dir.clone());
            std::thread::spawn(move || {
                let r = hf::download(&repo, &file, &dir, &dl);
                if let Err(e) = &r { say!("download {repo} {file}: {e:#}"); } else { say!("downloaded {repo} {file}"); }
                let mut g = dl.lock().unwrap();
                (g.running, g.error) = (false, r.err().map(|e| format!("{e:#}")));
            });
            send(&mut w, "202 Accepted", json!(*s.dl.lock().unwrap()))
        }
        // The prompt's tokens as the model would see them, template and tools included.
        ("POST", "/v1/messages/count_tokens") => {
            let Some(l) = l else { return send(&mut w, "503 Service Unavailable", json!({"error": {"message": "no model loaded", "type": "server_error"}})) };
            if anthropic_refused(&mut w, &l.r) { return Ok(()); }
            match serve::messages_body(&body).and_then(|b| serve::chat_request(&b)) {
                Ok(req) => {
                    let prompt = l.tpl.render(&req.messages, &req.tools, req.think.unwrap_or(e.think_default), req.preserve_thinking.unwrap_or(e.preserve_thinking), true);
                    let images: Vec<String> = req.messages.iter().flat_map(|m| m.images.iter().cloned()).collect();
                    let n = if images.is_empty() { l.tok.encode(&prompt)?.len() } else {
                        let mmproj = e.mmproj.clone().or_else(|| vision::find_sibling(&l.path));
                        let mmproj = mmproj.as_deref().context("image input needs --mmproj")?;
                        let input = load_images(mmproj, &images)?;
                        build_image_prompt(l.tok, &input, &images, &prompt)?.ids.len()
                    };
                    send(&mut w, "200 OK", json!({"input_tokens": n}))
                }
                Err(err) => send(&mut w, "400 Bad Request", json!({"error": {"message": format!("{err:#}"), "type": "invalid_request_error"}})),
            }
        }
        // The Anthropic Messages API (Claude Code) is translated to a chat request, and the reply back.
        ("POST", p @ ("/v1/chat/completions" | "/v1/messages")) => {
            let anth = p == "/v1/messages";
            let Some(l) = l else { return send(&mut w, "503 Service Unavailable", json!({"error": {"message": "no model loaded", "type": "server_error"}})) };
            if anth && anthropic_refused(&mut w, &l.r) { return Ok(()); }
            let (r, tok, tpl, name) = (&mut l.r, &l.tok, l.tpl, l.name.as_str());
            match r { Runner::Gpu(g) => g.owner = who.id, Runner::Slot(c) => (c.owner, c.conn) = (who.id, w.try_clone().ok()), _ => {} }
            let mut req = match if anth { serve::messages_body(&body) } else { Ok(body.clone()) }.and_then(|b| serve::chat_request(&b)) {
                Ok(q) => q,
                Err(err) => return send(&mut w, "400 Bad Request", json!({"error": {"message": format!("{err:#}"), "type": "invalid_request_error"}})),
            };
            let key_name = if who.name.is_empty() { "open" } else { who.name.as_str() };
            // The `done:` line of a chat request, and its line in the usage log.
            let log = |last: &serve::Last| {
                let decode_s = last.decode_ms / 1e3;
                let prefill_s = last.prefill_ms / 1e3;
                // Cached prompt tokens were not prefilled again, so they do not count toward the prefill speed.
                let new_prompt = last.prompt_tokens.saturating_sub(last.cached_tokens);
                let decode_tps = if decode_s > 0.0 { last.completion_tokens as f64 / decode_s } else { 0.0 };
                let prefill_tps = if prefill_s > 0.0 { new_prompt as f64 / prefill_s } else { 0.0 };
                say!("done: key={key_name} model={} prompt={} tok ({} cached) -> {} tok in {decode_s:.2} s, \
                    decode {decode_tps:.1} tok/s, prefill {prefill_tps:.1} tok/s, finish={}",
                    last.model, last.prompt_tokens, last.cached_tokens, last.completion_tokens, last.finish);
                if let Some(f) = &s.usage {
                    let mut v = json!(last);
                    v["key"] = who.name.clone().into();
                    // A failed write must not fail the reply; it is reported.
                    if let Err(err) = writeln!(f.lock().unwrap(), "{v}") { say!("usage log: {err}"); }
                }
            };
            // A model outside the main engine answers the request itself (greedy; it fails before it writes).
            if let Runner::Own(chat) = r {
                if s.max_tokens < usize::MAX { req.max_tokens = Some(req.max_tokens.unwrap_or(s.max_tokens).min(s.max_tokens)); }
                say!("request: {method} {p} key={key_name} model={name} messages={} stream={} tools={}", req.messages.len(), req.stream, req.tools.len());
                s.stats.lock().unwrap().busy += 1;
                let t = Instant::now();
                let res = chat(&mut w, &req);
                let total_ms = t.elapsed().as_secs_f64() * 1e3;
                let mut st = s.stats.lock().unwrap();
                st.busy -= 1;
                gpu_stats(r, &mut st);
                return match res {
                    Ok(d) => {
                        let last = serve::Last { at: created, model: name.into(), system_tokens: 0, cached_tokens: d.cached, prompt_tokens: d.prompt,
                            completion_tokens: d.completion, prefill_ms: d.prefill_s * 1e3, decode_ms: (total_ms - d.prefill_s * 1e3).max(0.0), finish: d.finish.into() };
                        log(&last);
                        st.record(last);
                        Ok(())
                    }
                    Err(err) => {
                        say!("request: {err:#}");
                        send(&mut w, "400 Bad Request", json!({"error": {"message": format!("{err:#}"), "type": "invalid_request_error"}}))
                    }
                };
            }
            // Room is checked on the full prompt before any state changes.
            let preserve = req.preserve_thinking.unwrap_or(e.preserve_thinking);
            // An assistant message at the end is continued, not answered: the prompt stops inside it.
            let open = tpl.render_open(&req.messages, &req.tools, req.think.unwrap_or(e.think_default), preserve);
            let n_hist = req.messages.len() - open.is_some() as usize;
            let prompt = open.unwrap_or_else(|| tpl.render(&req.messages, &req.tools, req.think.unwrap_or(e.think_default), preserve, true));
            // The system and tools block, which compacted or rewritten conversations keep.
            let n_sys = req.messages.iter().take_while(|m| m.role == "system").count();
            let system = tpl.render(&req.messages[..n_sys], &req.tools, req.think.unwrap_or(e.think_default), preserve, false);
            let images: Vec<String> = req.messages.iter().flat_map(|m| m.images.iter().cloned()).collect();
            let image_input = if images.is_empty() { None } else {
                let mmproj = e.mmproj.clone().or_else(|| vision::find_sibling(&l.path));
                let mmproj = mmproj.as_deref().context("image input needs --mmproj")?;
                Some(load_images(mmproj, &images)?)
            };
            let vision_prompt = match &image_input {
                Some(input) => Some(build_image_prompt(tok, input, &images, &prompt)?),
                None => None,
            };
            let ids = match &vision_prompt { Some(p) => p.ids.clone(), None => tok.encode(&prompt)? };
            // One line per chat request, and one when it finishes (see `log` above).
            say!("request: {method} {p} key={key_name} model={name} prompt={} tok stream={} tools={}",
                ids.len(), req.stream, req.tools.len());
            let mut smp = e.sampler();
            smp.temp = req.temperature.unwrap_or(smp.temp);
            smp.top_k = req.top_k.unwrap_or(smp.top_k);
            smp.top_p = req.top_p.unwrap_or(smp.top_p);
            smp.presence = req.presence_penalty.unwrap_or(smp.presence);
            smp.frequency = req.frequency_penalty.unwrap_or(smp.frequency);
            smp.repeat = req.repeat_penalty.unwrap_or(smp.repeat);
            smp.last_n = req.penalty_last_n.unwrap_or(smp.last_n);
            smp.dry_mult = req.dry_multiplier.unwrap_or(smp.dry_mult);
            smp.dry_base = req.dry_base.unwrap_or(smp.dry_base);
            smp.dry_allowed = req.dry_allowed_length.unwrap_or(smp.dry_allowed);
            if smp.dry_mult > 0.0 { smp.dry_breakers = dry_breakers(tok, name); }
            smp.tool_tags = tok.token_id("<tool_call>").zip(tok.token_id("</tool_call>")).map(|(a, b)| (a as usize, b as usize));
            say!("sampling: temp={} top_k={} top_p={} presence={} frequency={} repeat={}/{} dry={}/{}/{}", smp.temp, smp.top_k, smp.top_p,
                smp.presence, smp.frequency, smp.repeat, smp.last_n, smp.dry_mult, smp.dry_base, smp.dry_allowed);
            if req.seed.is_some() { smp.rng = seed(req.seed); }
            let n = room(e, ids.len(), req.max_tokens.unwrap_or(usize::MAX).min(s.max_tokens));
            if n == 0 { return send(&mut w, "400 Bad Request", json!({"error": {"message": format!("prompt of {} tokens fills the context", ids.len()), "type": "invalid_request_error"}})); }
            let id = format!("chatcmpl-{i}");
            let chunk = |delta: serde_json::Value, finish: Option<&str>| json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
            // With tools, a stream holds back the text from a `<tool_call>` tag on until the reply is parsed.
            let tools = req.tool_calls && !req.tools.is_empty();
            let mut cached = 0;
            let mut k = 0;  // tokens decoded
            let mut opened = false;  // Messages API stream: the text block has started
            let mut hold = Hold::new(tools, req.stop);
            // A prompt that ends in an open think block makes the reply start with reasoning, up to `</think>`.
            let think_open = prompt.trim_end().ends_with("<think>");
            let mut think = Think::new(think_open);
            let n_system = tok.encode(&system)?.len();
            let (mut t0, mut t1) = (Instant::now(), Instant::now());
            // Under `--parallel` a request is busy (and its prefill timed) from when the model takes it, not while it
            // waits for a slot.
            let mut busy = false;
            let start = |busy: &mut bool| {
                let mut st = s.stats.lock().unwrap();
                if st.busy == 0 { st.live_tokens = 0; }
                st.busy += 1;
                *busy = true;
            };
            if !matches!(r, Runner::Slot(_)) { start(&mut busy); }
            // Whether the prompt reached the model: a client that left while it waited in the queue is not logged.
            let mut prefilled = false;
            let res = (|| -> anyhow::Result<(Vec<u32>, bool)> {
                let logits;
                if serve::gone(&w) { anyhow::bail!("client closed the connection"); }
                let (chat_ids, chat_logits, chat_cached) = match &vision_prompt {
                    Some(p) => (p.ids.clone(), r.chat_image(p, &image_input.as_ref().unwrap().embeddings)?, 0),
                    None => r.chat(tok, &prompt, &tpl.render(&req.messages[..n_hist], &req.tools, req.think.unwrap_or(e.think_default), preserve, false), &system)?,
                };
                (_, logits, cached) = (chat_ids, chat_logits, chat_cached);
                if let Runner::Slot(c) = &*r && let Some(t) = c.started { t0 = t; start(&mut busy); }
                prefilled = true;
                t1 = Instant::now();
                if serve::gone(&w) { anyhow::bail!("client closed the connection"); }
                if req.stream {
                    write!(w, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")?;
                    if anth {
                        sse(&mut w, json!({"type": "message_start", "message": {"id": format!("msg_{i}"), "type": "message", "role": "assistant", "model": name,
                            "content": [], "stop_reason": null, "stop_sequence": null,
                            "usage": {"input_tokens": ids.len() - cached, "cache_read_input_tokens": cached, "output_tokens": 0}}}))?;
                    } else {
                        write!(w, "data: {}\n\n", chunk(json!({"role": "assistant", "content": ""}), None))?;
                    }
                }
                let mut last_stream_activity = Instant::now();
                let think_cap = Some((think_open, req.thinking_budget.unwrap_or(e.thinking_budget)));
                decode(r, tok, e.mtp() > 0, &mut smp, logits, n, think_cap, &mut |t| {
                    s.stats.lock().unwrap().live_tokens += 1;
                    // A client that left stops the reply (a stream notices on the next write; held text writes nothing).
                    k += 1;
                    if k % 16 == 0 && serve::gone(&w) { anyhow::bail!("client closed the connection"); }
                    let t = hold.push(t);
                    if req.stream {
                        if !t.is_empty() {
                            if !anth { send_split(&mut w, &chunk, think.feed(&t))?; } else { text_delta(&mut w, &mut opened, &t)?; }
                            last_stream_activity = Instant::now();
                        } else if last_stream_activity.elapsed() >= Duration::from_secs(15) {
                            w.write_all(b": keep-alive\n\n")?;
                            last_stream_activity = Instant::now();
                        }
                    }
                    Ok(hold.stopped)
                })
            })();
            {
                let mut st = s.stats.lock().unwrap();
                if busy { st.busy -= 1; }
                gpu_stats(r, &mut st);
            }
            let decoded = t1.elapsed();
            let ms = |d: Duration| d.as_secs_f64() * 1e3;
            let mut last = serve::Last { at: created, model: name.into(), system_tokens: n_system, cached_tokens: cached,
                prompt_tokens: ids.len(), completion_tokens: k, prefill_ms: ms(t1 - t0), decode_ms: ms(decoded), finish: String::new() };
            // The next request starts from the saved history state or from scratch. A stream that already started is just cut off.
            let (out, stop) = match res {
                Ok(x) => x,
                Err(err) => {
                    // Logged with the tokens decoded so far, which a client that left may have received.
                    let gone = serve::gone(&w);
                    last.finish = if gone { "cut" } else { "error" }.into();
                    // A request that failed before its prefill ended measured no prefill speed (t1 is still t0).
                    if !prefilled { last.prefill_ms = 0.0; }
                    if prefilled || !gone { log(&last); }
                    if !req.stream && !gone {
                        let sent = send(&mut w, "500 Internal Server Error", json!({"error": {"message": format!("{err:#}"), "type": "server_error"}}));
                        serve::exit_if_gpu_dead(&err);
                        return sent;
                    }
                    return Err(err);
                }
            };
            // A malformed call goes out as plain text.
            let calls = if tools { tpl.parse_calls(&hold.text, &req.tools) } else { None };
            last.completion_tokens = out.len();
            last.finish = if calls.is_some() { "tool_calls" } else if stop { "stop" } else { "length" }.into();
            let timings = llama_timings(&last);
            log(&last);
            s.stats.lock().unwrap().record(last);
            let calls_json = |(content, calls): &(String, Vec<(String, String)>), stream: bool| {
                let c: Vec<_> = calls.iter().enumerate().map(|(k, (f, a))| {
                    let mut c = json!({"id": format!("call_{i}_{k}"), "type": "function", "function": {"name": f, "arguments": a}});
                    if stream { c["index"] = k.into(); }
                    c
                }).collect();
                if stream { json!({"tool_calls": c}) }
                else { json!({"role": "assistant", "content": if content.is_empty() { None } else { Some(content) }, "tool_calls": c}) }
            };
            if anth {
                let reason = if calls.is_some() { "tool_use" } else if stop { "end_turn" } else { "max_tokens" };
                let usage = json!({"input_tokens": ids.len() - cached, "cache_read_input_tokens": cached, "cache_creation_input_tokens": 0, "output_tokens": out.len()});
                let uses: Vec<_> = calls.iter().flat_map(|c| &c.1).enumerate().map(|(k, (f, a))| json!({"type": "tool_use", "id": format!("toolu_{i}_{k}"),
                    "name": f, "input": serde_json::from_str::<serde_json::Value>(a).unwrap_or(json!({}))})).collect();
                if !req.stream {
                    let text = calls.as_ref().map_or(&hold.text, |c| &c.0);
                    let content: Vec<_> = (!text.is_empty()).then(|| json!({"type": "text", "text": text})).into_iter().chain(uses).collect();
                    return send(&mut w, "200 OK", json!({"id": format!("msg_{i}"), "type": "message", "role": "assistant", "model": name, "content": content,
                        "stop_reason": reason, "stop_sequence": null, "usage": usage, "timings": timings}));
                }
                if calls.is_none() && !hold.rest().is_empty() { text_delta(&mut w, &mut opened, hold.rest())?; }
                if opened { sse(&mut w, json!({"type": "content_block_stop", "index": 0}))?; }
                for (k, u) in uses.iter().enumerate() {
                    let (index, input) = (k + opened as usize, u["input"].to_string());
                    let mut start = u.clone();
                    start["input"] = json!({});
                    sse(&mut w, json!({"type": "content_block_start", "index": index, "content_block": start}))?;
                    sse(&mut w, json!({"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": input}}))?;
                    sse(&mut w, json!({"type": "content_block_stop", "index": index}))?;
                }
                sse(&mut w, json!({"type": "message_delta", "delta": {"stop_reason": reason, "stop_sequence": null}, "usage": {"output_tokens": out.len()}, "timings": timings}))?;
                return sse(&mut w, json!({"type": "message_stop"}));
            }
            let finish = if calls.is_some() { "tool_calls" } else if stop { "stop" } else { "length" };
            let usage = json!({"prompt_tokens": ids.len(), "completion_tokens": out.len(), "total_tokens": ids.len() + out.len(),
                "prompt_tokens_details": {"cached_tokens": cached}});
            if req.stream {
                match &calls {
                    Some(c) => write!(w, "data: {}\n\n", chunk(calls_json(c, true), None))?,
                    None => {
                        let (r, a) = think.feed(hold.rest());
                        send_split(&mut w, &chunk, (r, a))?;
                        send_split(&mut w, &chunk, think.finish())?;
                    }
                }
                let mut last = chunk(json!({}), Some(finish));
                last["usage"] = usage;
                last["timings"] = timings;
                Ok(write!(w, "data: {last}\n\ndata: [DONE]\n\n")?)
            } else {
                let message = match &calls { Some(c) => calls_json(c, false), None => {
                    let mut th = Think::new(think_open);
                    let (mut r, mut a) = th.feed(&hold.text);
                    let (r2, a2) = th.finish();
                    r += &r2; a += &a2;
                    if r.is_empty() { json!({"role": "assistant", "content": a}) } else { json!({"role": "assistant", "content": a, "reasoning_content": r}) }
                } };
                send(&mut w, "200 OK", json!({"id": id, "object": "chat.completion", "created": created, "model": name,
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage, "timings": timings}))
            }
        }
        _ => send(&mut w, "404 Not Found", json!({"error": {"message": format!("no route {method} {path}"), "type": "not_found"}})),
    }
}

#[cfg(test)]
mod interface_tests {
    #[test]
    fn answers_become_the_arguments_of_the_command() {
        let models = vec![("a.gguf".to_string(), "/m/a.gguf".into(), 1), ("b.gguf".to_string(), "/m/b.gguf".into(), 2)];
        // Model 9 does not exist, row 3 asks for a path and /no/such is not one, and mode 7 does not exist: each is
        // asked again. Then model 2, API server, GPU, 8192, default address.
        let args = super::interface_args(&mut &b"9\n3\n/no/such\n2\n7\n3\n\nmany\n8192\n\n"[..], models.clone()).unwrap();
        assert_eq!(args, ["serve", "--model", "/m/b.gguf", "--gpu", "--ctx", "8192", "--addr", "127.0.0.1:8080"]);
        // Every default, on the CPU: chat with the first model.
        let args = super::interface_args(&mut &b"\n\nn\n\n"[..], models.clone()).unwrap();
        assert_eq!(args, ["chat", "--model", "/m/a.gguf"]);
        assert!(super::interface_args(&mut &b"1\n"[..], models).is_err(), "the input ended before the last answer");
    }
}

#[cfg(test)]
mod flag_tests {
    use super::{calibrated_split, resolve_flags, split_moves, vram_margins};

    #[test]
    fn calibrated_split_needs_the_same_devices_in_the_same_order() {
        let v = serde_json::json!({"split": {"devices": [0, 1], "gpus": ["A", "B"], "layers": [12, 28]}});
        let names = |n: &[&str]| n.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(calibrated_split(&v, &[0, 1], &names(&["A", "B"])), Some(vec![12.0, 28.0]));
        assert_eq!(calibrated_split(&v, &[1, 0], &names(&["A", "B"])), None);
        assert_eq!(calibrated_split(&v, &[0, 1], &names(&["A", "C"])), None);
        assert_eq!(calibrated_split(&v, &[0], &names(&["A"])), None);
        assert_eq!(calibrated_split(&serde_json::json!({"chunk": 4096}), &[0, 1], &names(&["A", "B"])), None);
    }

    #[test]
    fn split_moves_shift_layers_between_neighbours() {
        assert_eq!(split_moves(&[20, 20], 6), [vec![14, 26], vec![26, 14]]);
        assert_eq!(split_moves(&[4, 36], 6), [vec![10, 30]]);
        assert_eq!(split_moves(&[15, 15, 10], 6), [vec![9, 21, 10], vec![21, 9, 10], vec![15, 9, 16], vec![15, 21, 4]]);
        assert!(split_moves(&[40], 6).is_empty());
    }

    #[test]
    fn calibration_from_another_gpu_is_refused() {
        let v = serde_json::json!({"chunk": 4096, "gpu": "NVIDIA GeForce RTX 5080"});
        assert!(super::calibration_fits(&v, "NVIDIA GeForce RTX 5080"));
        assert!(!super::calibration_fits(&v, "NVIDIA GeForce RTX 3060"));
        assert!(super::calibration_fits(&serde_json::json!({"chunk": 4096}), "NVIDIA GeForce RTX 3060"));  // older file
        assert!(super::calibration_fits(&v, ""));  // the driver gave no name
    }

    #[test]
    fn calibration_file_per_backend_and_device_set() {
        let name = |b, d: &[usize]| super::calibration_path(std::path::Path::new("/m/Q-8B.gguf"), b, d).file_name().unwrap().to_string_lossy().into_owned();
        assert_eq!(name("cuda", &[0]), "strata-Q-8B.json");
        assert_eq!(name("cuda", &[1]), "strata-Q-8B.json");
        assert_eq!(name("vulkan", &[0]), "strata-Q-8B-vulkan.json");
        assert_eq!(name("vulkan", &[0, 1]), "strata-Q-8B-vulkan-dev0-1.json");
        assert_eq!(name("cuda", &[1, 0]), "strata-Q-8B-dev1-0.json");
    }
    #[cfg(target_os = "linux")]
    use super::cpu_topology_groups;

    #[test]
    fn serve_takes_an_mcp_file() {
        use clap::Parser;
        let Some(super::Cmd::Serve { mcp, .. }) = super::Cli::try_parse_from(["rustingstrata", "serve", "--mcp", "/tmp/m.json"]).unwrap().cmd else { panic!() };
        assert_eq!(mcp, Some(std::path::PathBuf::from("/tmp/m.json")));
    }

    #[test]
    fn load_options_apply_and_validate() {
        #[derive(clap::Parser)]
        struct T { #[command(flatten)] e: super::Engine }
        let e = <T as clap::Parser>::parse_from(["t"]).e;
        let n = e.with(&serde_json::json!({"ctx": 8192, "kv": "q8", "mtp": 2, "vram_margin": 256})).unwrap();
        assert_eq!((n.ctx, n.kv.as_str(), n.mtp, n.vram_margin, n.cache_ram), (8192, "q8", Some(2), vec!["256".to_string()], e.cache_ram));
        assert_eq!(e.with(&serde_json::json!({"kv": "q4"})).unwrap().kv, "q4");
        assert_eq!(e.with(&serde_json::json!({"kv": "q2"})).unwrap().kv, "q2");
        assert_eq!(e.with(&serde_json::Value::Null).unwrap().ctx, e.ctx);
        for bad in [serde_json::json!({"ctx": 8}), serde_json::json!({"ctx": "big"}), serde_json::json!({"kv": "q1"}), serde_json::json!({"mtp": -1})] {
            assert!(e.with(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn vram_margins_are_per_device() {
        let spec = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(vram_margins(&spec(&["512"]), &[0, 1]).unwrap(), [512, 512]);
        assert_eq!(vram_margins(&spec(&["512", "256"]), &[1, 0]).unwrap(), [512, 256]);
        assert_eq!(vram_margins(&spec(&["0:512", "1:256"]), &[1, 0]).unwrap(), [256, 512]);
        assert_eq!(vram_margins(&spec(&["0:512", "1:256"]), &[0, 1]).unwrap(), [512, 256]);
    }

    #[test]
    fn vram_margins_reject_bad_specs() {
        let spec = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        for bad in [&["512", "256"][..], &["0:512", "0:256"][..], &["0:512", "256"][..]] {
            let e = vram_margins(&spec(bad), &[0]).unwrap_err().to_string();
            assert!(e.starts_with("--vram-margin"), "{bad:?}: {e}");
        }
        assert!(vram_margins(&spec(&["2:512"]), &[0, 1]).unwrap_err().to_string().contains("not in --device"));
        assert!(vram_margins(&spec(&["x:512"]), &[0]).unwrap_err().to_string().contains("not a GPU index"));
    }

    /// SMT siblings share one `thread_siblings_list`: 4 logical CPUs, 2 physical cores.
    #[cfg(target_os = "linux")]
    #[test]
    fn physical_cores_count_distinct_sibling_groups() {
        let d = std::env::temp_dir().join(format!("strata_cpu_topology_{}", std::process::id()));
        for (cpu, siblings) in [(0, "0,2"), (2, "0,2"), (1, "1,3"), (3, "1,3")] {
            let p = d.join(format!("cpu{cpu}/topology"));
            std::fs::create_dir_all(&p).unwrap();
            std::fs::write(p.join("thread_siblings_list"), siblings).unwrap();
        }
        std::fs::create_dir_all(d.join("cpufreq")).unwrap();
        assert_eq!(cpu_topology_groups(&d), 2);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn parallel_mtp_are_explained() {
        // `--parallel` with MTP: MTP off, one message naming both flags.
        let (n, p, k) = resolve_flags(true, 4, 1, 1).unwrap();
        assert_eq!((p, k), (4, 0));
        assert_eq!(n, ["--parallel 4 and --mtp 1: not supported together, decoding without MTP drafts"]);
    }

    #[test]
    fn mtp_without_a_head_runs_without_drafts() {
        let (n, _, k) = resolve_flags(true, 1, 3, 0).unwrap();
        assert_eq!(k, 0);
        assert_eq!(n, ["--mtp 3: this model has no MTP head: decoding without drafts"]);
        // A model with a head keeps its drafts.
        let (n, _, k) = resolve_flags(true, 1, 3, 1).unwrap();
        assert!(n.is_empty() && k == 3);
    }

    #[test]
    fn cpu_requires_the_gpu_for_both() {
        let (n, _, k) = resolve_flags(false, 2, 1, 1).unwrap();
        assert_eq!(k, 0);
        assert_eq!(n, ["--parallel 2: requires --gpu (the CPU answers one request at a time)",
                       "--mtp 1: requires --gpu: decoding without drafts"]);
        // No flags set (0 means unset) prints nothing.
        assert_eq!(resolve_flags(false, 0, 0, 0).unwrap(), (vec![], 1, 0));
    }

    #[test]
    fn parallel_range_is_rejected() {
        for n in [256, 1000] {
            let e = resolve_flags(true, n, 0, 0).unwrap_err().to_string();
            assert_eq!(e, format!("--parallel {n}: 1 to 255 slots"));
        }
        assert!(resolve_flags(true, 255, 0, 0).is_ok());
        assert_eq!(resolve_flags(true, 1, 0, 0).unwrap(), (vec![], 1, 0));
    }
}

#[cfg(test)]
mod timing_tests {
    use super::{llama_timings, serve::Last};

    #[test]
    fn timings_match_llama_server() {
        let last = Last { prompt_tokens: 100, cached_tokens: 40, completion_tokens: 20,
            prefill_ms: 60.0, decode_ms: 400.0, ..Default::default() };
        let t = llama_timings(&last);
        // llama-server counts the uncached prompt in prompt_n and the reused part in cache_n.
        assert_eq!(t["prompt_n"], 60);
        assert_eq!(t["cache_n"], 40);
        assert_eq!(t["prompt_ms"], 60.0);
        assert_eq!(t["prompt_per_second"], 1000.0);
        assert_eq!(t["prompt_per_token_ms"], 1.0);
        assert_eq!(t["predicted_n"], 20);
        assert_eq!(t["predicted_ms"], 400.0);
        assert_eq!(t["predicted_per_second"], 50.0);
        assert_eq!(t["predicted_per_token_ms"], 20.0);
    }
}

#[cfg(test)]
mod bench_tests {
    use super::{bench_record, round1, BenchMeta, BenchResult, Cli, ExpertStats, Stat};
    use clap::Parser;
    use serde_json::Value;
    use std::path::Path;

    fn meta() -> BenchMeta<'static> {
        BenchMeta {
            backend: "cpu",
            backend_arg: None,
            device: &[],
            model: Path::new("/mnt/models/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf"),
            ctx: 32896,
            chunk: 4096,
            kv: "f16",
            mtp: 0,
            tg: 128,
            reps: 3,
        }
    }

    fn result(pp: usize, prefill: Stat, decode: Stat) -> BenchResult {
        BenchResult { pp, prefill, decode, bytes_per_token: 0, decode_median: decode.mean }
    }

    fn parsed(record: &Value) -> Value {
        // One compact object plus a newline is what reaches the file: it must parse back unchanged.
        let mut line = serde_json::to_string(record).unwrap();
        line.push('\n');
        assert!(line.ends_with("}\n") && line.matches('\n').count() == 1);
        serde_json::from_slice::<Value>(line.as_bytes()).unwrap()
    }

    #[test]
    fn bench_record_schema_and_one_result_per_pp_in_order() {
        let st = |x: f64| Stat { mean: round1(x), min: round1(x), max: round1(x) };
        let results = [result(512, st(2135.7), st(129.5)), result(4096, st(1900.0), st(120.0)), result(32768, st(1500.0), st(80.0))];
        let v = parsed(&bench_record(&meta(), &results, None, 1_762_617_600));
        assert_eq!(v["schema"], "strata.bench.v1");
        assert_eq!(v["unix_time"], 1_762_617_600u64);
        assert_eq!(v["backend"], "cpu");
        assert!(v.get("backend_arg").is_none());
        assert_eq!(v["device"], serde_json::json!([]));
        assert_eq!(v["model"]["name"], "Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf");
        assert_eq!(v["model"]["path"], "/mnt/models/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf");
        assert_eq!((v["ctx"].as_u64(), v["chunk"].as_u64(), v["mtp"].as_u64()), (Some(32896), Some(4096), Some(0)));
        assert_eq!((v["tg"].as_u64(), v["reps"].as_u64()), (Some(128), Some(3)));
        let rs = v["results"].as_array().unwrap();
        assert_eq!(rs.iter().map(|r| r["pp"].as_u64().unwrap()).collect::<Vec<_>>(), [512, 4096, 32768]);
        assert_eq!(rs[0]["prefill_tok_s"]["mean"], 2135.7);
    }

    #[test]
    fn bench_record_rates_match_the_text_rendering() {
        let raw_p = [2144.9, 2129.6, 2132.6];
        let raw_d = [128.9, 129.8, 129.8];
        let (p, d) = (Stat::of(&raw_p), Stat::of(&raw_d));
        let text = format!("pp {:6}: prefill {} tok/s | tg {}: decode {} tok/s", 512, p.text(), 32, d.text());
        let v = parsed(&bench_record(&meta(), &[result(512, p, d)], None, 0));
        for (key, stat) in [("prefill_tok_s", p), ("decode_tok_s", d)] {
            let o = &v["results"][0][key];
            assert_eq!(o["mean"], stat.mean);
            assert_eq!(o["min"], stat.min);
            assert_eq!(o["max"], stat.max);
            // The same decimal strings the text line carries.
            assert!(text.contains(&format!("{:.1}", o["mean"].as_f64().unwrap())));
        }
        assert_eq!(format!("{:.1}", p.mean), "2135.7");
        assert_eq!(format!("{:.1}", p.min), "2129.6");
        assert_eq!(format!("{:.1}", p.max), "2144.9");
        assert!(text.starts_with("pp    512: prefill 2135.7 (2129.6-2144.9) tok/s | tg 32: decode 129.5 (128.9-129.8) tok/s"));
    }

    #[test]
    fn bench_record_omits_expert_without_a_cache() {
        let v = parsed(&bench_record(&meta(), &[result(512, Stat { mean: 1.0, min: 1.0, max: 1.0 }, Stat { mean: 2.0, min: 2.0, max: 2.0 })], None, 0));
        assert!(v.get("expert").is_none());
        assert!(v["results"][0].get("expert").is_none());
    }

    #[test]
    fn bench_record_expert_object_uses_the_printed_numbers() {
        let stats = ExpertStats {
            hits: 1024, misses: 0, swaps: 3, resident: 10240, total: 10240,
            bytes_per_token: 329_006_694, cached: true, all_fast: true,
        };
        let mut r = result(512, Stat { mean: 2135.7, min: 2129.6, max: 2144.9 }, Stat { mean: 129.5, min: 128.9, max: 129.8 });
        r.bytes_per_token = 329_006_694;
        r.decode_median = 129.7;
        let mut m = meta();
        m.backend = "vulkan";
        m.backend_arg = Some("vulkan");
        m.device = &[0];
        let v = parsed(&bench_record(&m, &[r], Some(&stats), 0));
        assert_eq!(v["backend"], "vulkan");
        assert_eq!(v["backend_arg"], "vulkan");
        assert_eq!(v["device"], serde_json::json!([0]));
        let e = &v["expert"];
        assert_eq!(e["hit_rate_pct"], 100.0);
        assert_eq!((e["misses"].as_u64(), e["swaps"].as_u64()), (Some(0), Some(3)));
        assert_eq!((e["resident"].as_u64(), e["total"].as_u64()), (Some(10240), Some(10240)));
        assert_eq!(e["bytes_per_token"], 329_006_694u64);
        assert_eq!(e["mb_per_token"], 313.8);
        let pe = &v["results"][0]["expert"];
        assert_eq!(pe["bytes_per_token"], 329_006_694u64);
        assert_eq!(pe["mb_per_token"], 313.8);
        assert_eq!(pe["gb_per_s"], round1(329_006_694.0 * 129.7 / 1e9));
        assert_eq!(pe["at_decode_tok_s"], 129.7);
    }

    /// A model outside the main engine leaves no `STRATA_*` default behind for the next model.
    #[test]
    fn own_state_restores_the_environment() {
        // Safety: no other test reads this variable.
        unsafe { std::env::remove_var("STRATA_GLM_PREFETCH") };
        let st = super::OwnState::new();
        unsafe { std::env::set_var("STRATA_GLM_PREFETCH", "2") };
        drop(st);
        assert!(std::env::var_os("STRATA_GLM_PREFETCH").is_none());
    }

    #[test]
    fn bench_record_rejects_json_out_dash() {
        let e = Cli::try_parse_from(["rustingstrata", "bench", "--model", "m.gguf", "--json-out", "-"]).err().unwrap();
        assert!(e.to_string().contains("stdout carries the text block"), "{e}");
        assert!(Cli::try_parse_from(["rustingstrata", "bench", "--model", "m.gguf", "--json-out", "/tmp/x.json"]).is_ok());
    }
}
