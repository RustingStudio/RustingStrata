use clap::{Args, Parser, Subcommand};
use anyhow::Context;
use rusting_strata::model::{Model, State};
use rusting_strata::{backend, gpu_model::GpuModel};
use rusting_strata::{config::ModelConfig, gguf::Gguf, hf, ops, ops::Sampler, quant::GgmlType, serve, tokenizer::Tokenizer};
use rusting_strata::chat::{Hold, Msg, Template};
use serde_json::json;
use std::io::{BufRead, BufReader, Read, Write};
use std::{collections::BTreeMap, net::{TcpListener, TcpStream}, sync::atomic::Ordering::Relaxed, path::{Path, PathBuf}, time::{Duration, Instant, SystemTime}};

#[derive(Parser)]
#[command(name = "strata", version, about = "Rust inference engine for Qwen3.5/3.6-family GGUF models")]
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
    /// GPU context length (KV cache size) in tokens.
    #[arg(long, default_value_t = 4096)]
    ctx: usize,
    /// Prefill chunk in tokens (GPU; 0 = token by token) [default: calibrated, else 4096].
    #[arg(long)]
    chunk: Option<usize>,
    /// VRAM left free after placing expert layers, in MB.
    #[arg(long, default_value_t = 512)]
    vram_margin: usize,
    /// Speculative decoding with the model's MTP draft head, N drafts per step (GPU; greedy output unchanged, sampling keeps its distribution)
    /// [default: calibrated, else 0].
    #[arg(long, value_name = "N", default_missing_value = "1", num_args = 0..=1)]
    mtp: Option<usize>,
    /// Sampling temperature; 0 is greedy.
    #[arg(long, default_value_t = 0.0)]
    temp: f32,
    /// Sample only from the K most likely tokens (0 = all).
    #[arg(long, default_value_t = 0)]
    top_k: usize,
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    /// Random seed for sampling [default: from the clock].
    #[arg(long)]
    seed: Option<u64>,
    /// CPU threads for expert work [default: calibrated, else all logical cores].
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
    /// Host memory in MB for conversation states that a request with another history replaced (GPU): a later request
    /// that continues one of them takes it back instead of prefilling it again. 0 keeps none.
    #[arg(long, default_value_t = 4096)]
    cache_ram: usize,
    /// Keep the think block of earlier replies in the prompt (Qwen3.5/3.6 templates' `preserve_thinking`; a request
    /// can set it in `chat_template_kwargs`). The template drops it from replies before the last user message, so
    /// each new user message changes the earlier text and the whole conversation is prefilled again; with this, a
    /// coding agent's long history stays cached across user messages. `--preserve-thinking false` follows the
    /// template's default.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    preserve_thinking: bool,
    /// KV cache format (GPU): f16, or q8 (q8_0 blocks, half the KV memory; numerics differ slightly from
    /// llama.cpp's default f16 cache).
    #[arg(long, default_value = "f16", value_parser = ["f16", "q8"])]
    kv: String,
    /// GPU backend: cuda or vulkan [default: cuda if it loads, else vulkan].
    #[arg(long)]
    backend: Option<String>,
    /// GPU index, or several separated by commas (such as 0,1) to split the layers over those GPUs in proportion to
    /// their free VRAM.
    #[arg(long, default_value = "0", value_delimiter = ',')]
    device: Vec<usize>,
    /// Prompts shorter than this skip the batched prefill (from `strata calibrate`, else 32).
    #[arg(skip)]
    short: usize,
    /// Run on the CPU when no GPU backend opens (the bare `strata` launch).
    #[arg(skip)]
    cpu_fallback: bool,
    /// Sequences in GPU memory at once (`serve --parallel`).
    #[arg(skip)]
    parallel: usize,
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
    /// Requests read or waiting at once; more get `503` with `Retry-After`.
    #[arg(long, default_value_t = 16)]
    max_waiting: usize,
    /// Most reply tokens a request may ask for (and the default when it asks for none) [default: what the context
    /// leaves].
    #[arg(long)]
    max_tokens: Option<usize>,
    /// Requests answered at once (GPU): each gets its own slot of `--ctx` tokens of KV cache, and the
    /// slots that decode share each forward, up to 4 at a time. Between the chunks of a prompt's prefill the others
    /// decode for as long as the chunk took (a smaller `--chunk` gives shorter pauses). MTP drafting is off.
    #[arg(long, default_value_t = 1)]
    parallel: usize,
}

enum Runner<'a> { Cpu(&'a Model<'a>, State), Gpu(Box<GpuModel<'a>>), Slot(Client) }

impl Runner<'_> {
    fn step(&mut self, t: u32) -> anyhow::Result<Vec<f32>> {
        match self { Runner::Cpu(m, st) => Ok(m.step(st, t)), Runner::Gpu(g) => g.step(t), Runner::Slot(c) => Ok(c.step(t, true)?.1) }
    }
    /// Feed a prompt; returns the logits after its last token.
    fn prefill(&mut self, ts: &[u32]) -> anyhow::Result<Vec<f32>> {
        match self {
            Runner::Cpu(m, st) => Ok(ts.iter().map(|&t| m.step(st, t)).last().unwrap_or_default()),
            Runner::Gpu(g) => g.prefill(ts),
            Runner::Slot(_) => anyhow::bail!("prefill on a slot"),
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
    /// Forget the sequence so far.
    fn reset(&mut self) -> anyhow::Result<()> {
        match self { Runner::Cpu(m, st) => { *st = State::new(&m.cfg); Ok(()) } Runner::Gpu(g) => g.reset(), Runner::Slot(_) => Ok(()) }
    }
    /// Seconds spent waiting on CPU experts so far (GPU only).
    fn cpu_secs(&self) -> f64 { if let Runner::Gpu(g) = self { g.parts().map(|p| p.cpu_time.as_secs_f64()).sum() } else { 0.0 } }
    /// Split model: seconds per part but the last (forward until its GPU is done, copy to the next part).
    fn split_secs(&self) -> Vec<(f64, f64)> {
        let Runner::Gpu(g) = self else { return vec![] };
        let mut v: Vec<_> = g.parts().map(|p| (p.split_time.0.as_secs_f64(), p.split_time.1.as_secs_f64())).collect();
        v.pop();
        v
    }
    fn finish(&self) -> anyhow::Result<()> {
        if let Runner::Gpu(g) = self {
            let sum = |f: &dyn Fn(&GpuModel) -> usize| g.parts().map(f).sum::<usize>();
            let (hits, misses) = (sum(&|p| p.hits), sum(&|p| p.misses));
            if g.drafts.0 > 0 { eprintln!("MTP drafts accepted {}/{} ({:.1}%)", g.drafts.1, g.drafts.0, 100.0 * g.drafts.1 as f64 / g.drafts.0 as f64); }
            eprintln!("expert hit rate {:.1}% ({} misses), cache swaps {}", 100.0 * hits as f64 / (hits + misses).max(1) as f64, misses,
                sum(&|p| p.cache.as_ref().map_or(0, |c| c.swaps)));
            let resumed = sum(&|p| p.resumed.get());
            if resumed > 0 { eprintln!("GPU waits for the CPU expert worker that timed out and resumed: {resumed}"); }
            for p in g.parts() { if let Some(c) = &p.cache { c.save_profile()?; } }
        }
        Ok(())
    }
}

impl Engine {
    /// Fills settings not given on the command line from the model's calibration file, else built-in defaults.
    fn runner<'a>(&mut self, m: &'a Model<'a>, model: &Path) -> anyhow::Result<Runner<'a>> {
        if self.llama_numerics { unsafe { std::env::set_var("STRATA_LLAMA_NUMERICS", "1") } }  // read by Gpu::new
        if self.kv == "q8" { unsafe { std::env::set_var("STRATA_KV_Q8", "1") } }  // read by Gpu::new and Vk::new
        let open = || self.device.iter().map(|&d| backend::open(self.backend.as_deref(), d)).collect::<anyhow::Result<Vec<_>>>();
        let bk = match self.gpu.then(open).transpose() {
            Err(e) if self.cpu_fallback => { eprintln!("no GPU backend ({e:#}), running on the CPU"); None }
            r => r?,
        };
        let path = calibration_path(model, bk.as_ref().map_or("cuda", |b| b[0].name()));
        let v: serde_json::Value = std::fs::read(&path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
        if !v.is_null() { eprintln!("calibration: {}", path.display()); }
        let get = |k: &str| v[k].as_u64().map(|x| x as usize);
        self.chunk.get_or_insert(get("chunk").unwrap_or(4096));
        self.mtp.get_or_insert(get("mtp").unwrap_or(0));
        self.short = get("short").unwrap_or(32);
        // The global pool can be built once; a model loaded later by `serve` keeps the first one's threads.
        if let Some(n) = self.threads.or(get("threads")) { let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global(); }
        let Some(bk) = bk else {
            if self.parallel > 1 { eprintln!("--parallel: the CPU runs one request at a time"); }
            return Ok(Runner::Cpu(m, State::new(&m.cfg)));
        };
        if self.fixed_experts { unsafe { std::env::set_var("STRATA_FIXED_EXPERTS", "1") } }  // read by ExpertCache::new
        let mut g = if self.parallel > 1 {
            if self.mtp() > 0 { eprintln!("--parallel: decoding without MTP drafts"); }
            self.mtp = Some(0);
            GpuModel::load_split(m, bk, self.ctx, self.chunk(), self.vram_margin << 20, 0, self.parallel).context(LOAD_HINT)?
        } else {
            GpuModel::load_split(m, bk, self.ctx, self.chunk(), self.vram_margin << 20, self.mtp(), 1).context(LOAD_HINT)?
        };
        g.short = self.short;
        g.cache_ram = self.cache_ram << 20;
        if self.mtp() > 0 && g.k == 0 {
            eprintln!("--mtp: this model has no MTP draft head (or it is split over GPUs), decoding without it");
            self.mtp = Some(0);
        }
        Ok(Runner::Gpu(Box::new(g)))
    }
    fn sampler(&self) -> Sampler {
        Sampler { temp: self.temp, top_k: self.top_k, top_p: self.top_p, rng: seed(self.seed) }
    }
    fn chunk(&self) -> usize { self.chunk.unwrap_or(4096) }
    fn mtp(&self) -> usize { self.mtp.unwrap_or(0) }
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

/// Calibration file for `model` on `backend`; CUDA keeps the original name.
fn calibration_path(model: &Path, backend: &str) -> PathBuf {
    let stem = model.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    let sfx = if backend == "cuda" { String::new() } else { format!("-{backend}") };
    rusting_strata::expert_cache::cache_dir().join(format!("strata-{stem}{sfx}.json"))
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
        /// Let the model think before answering (template default; off here).
        #[arg(long)]
        think: bool,
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
        #[command(flatten)]
        host: Hosting,
        #[command(flatten)]
        engine: Engine,
    },
    /// Measure prefill chunk, MTP drafts per step and the short-prompt threshold on this machine (GPU) and write
    /// them to the model's calibration file, which later runs read unless flags override.
    Calibrate {
        #[arg(long)]
        model: PathBuf,
        /// GPU context length used while measuring and assumed for the results.
        #[arg(long, default_value_t = 4096)]
        ctx: usize,
        #[arg(long, default_value_t = 512)]
        vram_margin: usize,
        #[arg(long)]
        backend: Option<String>,
        #[arg(long, default_value_t = 0)]
        device: usize,
        /// KV cache format to calibrate with (q8 leaves more VRAM for experts), as `--kv` of the runs that use it.
        #[arg(long, default_value = "f16", value_parser = ["f16", "q8"])]
        kv: String,
        /// Accepted for symmetry with the other commands: calibration always runs on the GPU.
        #[arg(long, hide = true)]
        gpu: bool,
    },
    /// Speed test: for each prompt length, prefill that many tokens of prose, then greedily decode `tg` more at that
    /// depth. Prints prefill and decode tok/s as mean (min-max) over `reps` runs. `--ctx` grows to fit.
    Bench {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "512,4096")]
        pp: Vec<usize>,
        #[arg(long, default_value_t = 128)]
        tg: usize,
        #[arg(long, default_value_t = 3)]
        reps: usize,
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
    /// Text generation (greedy unless --temp).
    Generate {
        #[arg(long)]
        model: PathBuf,
        /// Prompt text ("-" reads stdin; long prompts exceed the 128 KiB argument limit).
        #[arg(long)]
        prompt: String,
        #[arg(short, default_value_t = 32)]
        n: usize,
        #[command(flatten)]
        engine: Engine,
    },
}

fn main() -> anyhow::Result<()> {
    let Some(cmd) = Cli::parse().cmd else {
        // Started without arguments, as by a double click: keep the window open on an error so it can be read.
        let Some(Cmd::Serve { model, models_dir, addr, api_key, open, host, mut engine }) = Cli::parse_from(["strata", "serve", "--gpu", "--open"]).cmd else { unreachable!() };
        engine.cpu_fallback = true;
        let r = run(Cmd::Serve { model, models_dir, addr, api_key, open, host, engine });
        if let Err(e) = &r {
            eprintln!("error: {e:#}\npress Enter to close");
            let _ = std::io::stdin().read_line(&mut String::new());
        }
        return r;
    };
    run(cmd)
}

fn run(cmd: Cmd) -> anyhow::Result<()> {
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
            let m = Model::load(&g)?;
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
        Cmd::Generate { model, prompt, n, mut engine } => {
            let g = Gguf::open(&model)?;
            let m = Model::load(&g)?;
            let tok = Tokenizer::from_gguf(&g)?;
            let prompt = if prompt == "-" {
                let mut s = String::new();
                std::io::stdin().read_to_string(&mut s)?;
                s
            } else { prompt };
            let ids = tok.encode(&prompt)?;
            anyhow::ensure!(!ids.is_empty(), "empty prompt");
            let mut r = engine.runner(&m, &model)?;
            let t0 = Instant::now();
            let logits = r.prefill(&ids)?;
            let t1 = Instant::now();
            let (cpu0, split0) = (r.cpu_secs(), r.split_secs());
            let cap = room(&engine, ids.len(), n);
            let (out, _) = decode(&mut r, &tok, engine.mtp() > 0, &mut engine.sampler(), logits, cap, &mut |t| { print!("{t}"); std::io::stdout().flush()?; Ok(false) })?;
            let (pp, tg) = ((t1 - t0).as_secs_f64(), t1.elapsed().as_secs_f64());
            if out.len() == cap && cap < n { eprint!("\n(stopped: the context is full at {} tokens; raise --ctx)", engine.ctx); }
            eprintln!("\nprompt {} tok {:.2} tok/s | gen {} tok {:.2} tok/s", ids.len(), ids.len() as f64 / pp, out.len(), out.len() as f64 / tg);
            if engine.gpu { eprintln!("CPU expert compute, share of gen time: {:.1}%", 100.0 * (r.cpu_secs() - cpu0) / tg); }
            let ms = |s: f64| 1e3 * s / out.len().max(1) as f64;
            for (i, ((a, b), (a0, b0))) in r.split_secs().into_iter().zip(split0).enumerate() {
                eprintln!("device {i}: {:.2} ms/token forward, {:.2} ms/token copy to the next device", ms(a - a0), ms(b - b0));
            }
            if !r.split_secs().is_empty() { eprintln!("gen total {:.2} ms/token", ms(tg)); }
            r.finish()?;
        }
        Cmd::Perplexity { model, file, ctx_len: n, chunks, kld_base, mut engine } => {
            let g = Gguf::open(&model)?;
            let m = Model::load(&g)?;
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
        Cmd::Bench { model, pp, tg, reps, mut engine } => {
            let g = Gguf::open(&model)?;
            let m = Model::load(&g)?;
            let tok = Tokenizer::from_gguf(&g)?;
            engine.ctx = engine.ctx.max(pp.iter().max().unwrap_or(&0) + tg);
            let mut r = engine.runner(&m, &model)?;
            for n in pp {
                let ids = prose(&tok, n)?;
                let (mut p, mut d) = (Vec::new(), Vec::new());
                for _ in 0..reps {
                    r.reset()?;
                    let t = Instant::now();
                    let l = r.prefill(&ids)?;
                    p.push(n as f64 / t.elapsed().as_secs_f64());
                    let t = Instant::now();
                    let k = decode(&mut r, &tok, engine.mtp() > 0, &mut Sampler::greedy(), l, tg, &mut |_| Ok(false))?.0.len();
                    d.push(k as f64 / t.elapsed().as_secs_f64());
                }
                let stat = |v: &[f64]| {
                    let (lo, hi) = v.iter().fold((f64::INFINITY, 0f64), |(a, b), &x| (a.min(x), b.max(x)));
                    format!("{:.1} ({lo:.1}-{hi:.1})", v.iter().sum::<f64>() / v.len() as f64)
                };
                println!("pp {n:6}: prefill {} tok/s | tg {tg}: decode {} tok/s", stat(&p), stat(&d));
            }
            r.finish()?;
        }
        Cmd::Calibrate { model, ctx, vram_margin, backend, device, kv, gpu: _ } => {
            if kv == "q8" { unsafe { std::env::set_var("STRATA_KV_Q8", "1") } }  // read by Gpu::new and Vk::new
            calibrate(&model, ctx, vram_margin << 20, backend.as_deref(), device)?
        }
        Cmd::Chat { model, system, n, think, mut engine } => {
            let g = Gguf::open(&model)?;
            let m = Model::load(&g)?;
            let tok = Tokenizer::from_gguf(&g)?;
            let mut r = engine.runner(&m, &model)?;
            let mut smp = engine.sampler();
            let tpl = template(&g);
            let mut msgs: Vec<Msg> = system.iter().map(|s| Msg::new("system", s)).collect();
            let stdin = std::io::stdin();
            loop {
                print!("> ");
                std::io::stdout().flush()?;
                let mut line = String::new();
                if stdin.lock().read_line(&mut line)? == 0 { break; }
                if line.trim().is_empty() { continue; }
                msgs.push(Msg::new("user", line.trim_end()));
                let (ids, logits, _) = r.chat(&tok, &tpl.render(&msgs, &[], think, engine.preserve_thinking, true), &tpl.render(&msgs, &[], think, engine.preserve_thinking, false), "")?;
                let (out, _) = decode(&mut r, &tok, engine.mtp() > 0, &mut smp, logits, room(&engine, ids.len(), n), &mut |t| { print!("{t}"); std::io::stdout().flush()?; Ok(false) })?;
                println!();
                msgs.push(Msg::new("assistant", &answer(&tok.decode(&out)?)));
            }
            r.finish()?;
        }
        Cmd::Serve { model, models_dir, addr, api_key, open, host, mut engine } => {
            engine.parallel = host.parallel;
            let api_key = api_key.or_else(|| std::env::var("STRATA_API_KEY").ok());
            anyhow::ensure!(api_key.as_ref().is_none_or(|k| !k.is_empty()), "empty API key");
            let users = match &host.api_keys {
                Some(f) => serve::Keys::parse(&std::fs::read_to_string(f).with_context(|| format!("reading {}", f.display()))?)
                    .with_context(|| format!("in {}", f.display()))?,
                None => vec![],
            };
            anyhow::ensure!(!users.iter().any(|u| api_key.as_ref() == Some(&u.1)), "the admin key is also a customer key");
            let keys = std::sync::Arc::new(std::sync::RwLock::new(serve::Keys { admin: api_key, users }));
            let (keys_file, mut keys_time) = (host.api_keys.clone(), host.api_keys.as_ref().and_then(|f| std::fs::metadata(f).and_then(|m| m.modified()).ok()));
            let usage = match &host.usage_log {
                Some(f) => Some(std::sync::Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(f)
                    .with_context(|| format!("opening {}", f.display()))?)),
                None => None,
            };
            let max_waiting = host.max_waiting.max(1);
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
            let (st, a, k, t0) = (stats.clone(), addr.clone(), keys.clone(), Instant::now());
            let stop = on_sigterm();
            // SIGTERM (systemctl stop or restart): new requests get 503, the replies being generated finish, then
            // the server exits. Two looks 0.2 s apart cover a connection counted just before the signal.
            let oc = open_conns.clone();
            std::thread::spawn(move || loop {
                std::thread::sleep(Duration::from_millis(200));
                if !stop.load(Relaxed) { continue; }
                let n = oc.load(Relaxed);
                if n == 0 { std::thread::sleep(Duration::from_millis(200)); if oc.load(Relaxed) == 0 { eprintln!("stopped"); std::process::exit(0); } }
                else { eprintln!("stopping: {n} requests to finish"); std::thread::sleep(Duration::from_secs(5)); }
            });
            std::thread::spawn(move || for (i, conn) in listener.incoming().enumerate() {
                let mut c = match conn { Ok(c) => c, Err(e) => { eprintln!("request {i}: {e:#}"); continue } };
                if stop.load(Relaxed) {
                    let b = json!({"error": {"message": "server is stopping, retry later", "type": "overloaded_error"}}).to_string();
                    let _ = c.set_write_timeout(Some(Duration::from_secs(5)));
                    let _ = write!(c, "HTTP/1.1 503 Service Unavailable\r\nRetry-After: 5\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len());
                    let _ = c.shutdown(std::net::Shutdown::Write);
                    continue;
                }
                // A changed key file takes effect from the next request; one that does not parse keeps the old keys.
                if let Some(f) = &keys_file && let Ok(t) = std::fs::metadata(f).and_then(|m| m.modified()) && Some(t) != keys_time {
                    keys_time = Some(t);
                    let mut kw = k.write().unwrap();
                    match std::fs::read_to_string(f).map_err(Into::into).and_then(|t| serve::Keys::parse(&t)) {
                        Ok(u) if u.iter().any(|u| kw.admin.as_ref() == Some(&u.1)) => eprintln!("{}: the admin key is also a customer key, keeping the old keys", f.display()),
                        // An empty file (or one caught mid-write) would leave the server open.
                        Ok(u) if u.is_empty() && kw.admin.is_none() => eprintln!("{}: no keys, keeping the old keys", f.display()),
                        Ok(u) => { eprintln!("{}: {} keys", f.display(), u.len()); kw.users = u; }
                        Err(e) => eprintln!("{}: {e:#}, keeping the old keys", f.display()),
                    }
                }
                // ponytail: one cap on connections read or queued (each may hold an 8 MB body), not one per key.
                if open_conns.fetch_add(1, Relaxed) >= max_waiting {
                    open_conns.fetch_sub(1, Relaxed);
                    eprintln!("request {i}: {max_waiting} requests already waiting, answered 503");
                    let b = json!({"error": {"message": "server busy, retry later", "type": "overloaded_error"}}).to_string();
                    let _ = c.set_write_timeout(Some(Duration::from_secs(5)));
                    let _ = write!(c, "HTTP/1.1 503 Service Unavailable\r\nRetry-After: 2\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len());
                    let _ = c.shutdown(std::net::Shutdown::Write);
                    continue;
                }
                let (tx, oc, st, a, k) = (tx.clone(), open_conns.clone(), st.clone(), a.clone(), k.clone());
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
                        let (model, loading, slots) = { let g = st.lock().unwrap(); (g.model.is_some() && !g.loading, g.loading, g.slots) };
                        let waiting = oc.load(Relaxed) - 1;
                        return if model { reply(json!({"status": "ok", "waiting": waiting, "slots": slots}), "200 OK") }
                            else { reply(json!({"status": if loading { "loading" } else { "no model" }}), "503 Service Unavailable") };
                    }
                    // The page's side panel is answered here, so it stays current while a reply is generated or a
                    // model loads. A bad Host or key falls through to `handle`, which refuses it.
                    if let Ok(r) = &q && r.method == "GET" && r.path == "/api/stats" && let k = k.read().unwrap().clone()
                        && (k.any() || serve::host_ok(&r.host, &a)) && k.who(&r.auth).is_some_and(|w| w.admin) {
                        let mut v = json!(*st.lock().unwrap());
                        (v["system"], v["uptime_s"]) = (serve::system(), t0.elapsed().as_secs().into());
                        return reply(v, "200 OK");
                    }
                    let _ = tx.send((c, q, i));
                });
            });
            let s = Serving { rx: rx.into(), waiting, addr, keys, usage, max_tokens: host.max_tokens.unwrap_or(usize::MAX), dir: models_dir.unwrap_or_else(models_dir_default), model: model.clone(),
                dl: Default::default(), stats };
            // The model to load next and the /api/load request waiting for the outcome (none for --model).
            let mut next = model.map(|p| (p, None));
            eprintln!("listening on {url} (models folder: {})", s.dir.display());
            // Requests wait while a model loads, so the page may take that long to show.
            if open { open_browser(&url); }
            loop {
                let (p, mut w) = match next.take() {
                    Some(x) => x,
                    None => match s.serve(None, &engine) { Some((p, w)) => (p, Some(w)), None => return Ok(()) },
                };
                match serve_model(&p, engine.clone(), &s, &mut w) {
                    Ok(Some((p, w))) => next = Some((p, Some(w))),
                    Ok(None) => return Ok(()),
                    Err(e) if w.is_none() => return Err(e),
                    Err(e) => {
                        s.stats.lock().unwrap().unload(None);
                        eprintln!("loading {}: {e:#}", p.display());
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
fn prose(tok: &Tokenizer, n: usize) -> anyhow::Result<Vec<u32>> {
    let text = "The river town woke before dawn. Fishermen checked their nets, bakers lit ovens, and a clockmaker \
        argued with his apprentice about gears, tides and the price of brass. In 1874 the bridge washed away; \
        the council voted 7 to 5 to rebuild it in stone. ";
    let ids: Vec<u32> = tok.encode(&text.repeat(n / 20 + 1))?.into_iter().take(n).collect();
    anyhow::ensure!(ids.len() == n, "prose prompt too short");
    Ok(ids)
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

fn calibrate(model: &Path, ctx: usize, margin: usize, kind: Option<&str>, device: usize) -> anyhow::Result<()> {
    let g = Gguf::open(model)?;
    let m = Model::load(&g)?;
    let tok = Tokenizer::from_gguf(&g)?;
    let n_pf = 4096.min(ctx.saturating_sub(256));
    anyhow::ensure!(n_pf >= 512, "--ctx too small to calibrate");
    let pf = prose(&tok, n_pf)?;
    let load = |chunk: usize| -> anyhow::Result<GpuModel> {
        let mut gm = GpuModel::load(&m, backend::open(kind, device)?, ctx, chunk, margin, 3).context(LOAD_HINT)?;
        gm.adapt_every = 0; // fixed placement: steadier timings
        Ok(gm)
    };

    let mut chunk = (0, 0.0);
    for c in [512, 1024, 2048, 4096].into_iter().filter(|&c| c <= ctx) {
        // A larger chunk's prefill scratch may not fit beside a long --ctx: keep the chunks measured so far.
        let mut gm = match load(c) { Ok(gm) => gm, Err(e) if chunk.0 > 0 => { eprintln!("chunk {c:5}: {e:#}"); break } Err(e) => return Err(e) };
        let t = best(&mut || { gm.reset()?; gm.prefill(&pf).map(drop) })?;
        let tps = n_pf as f64 / t;
        eprintln!("chunk {c:5}: prefill {n_pf} tok {tps:.0} tok/s");
        if tps > chunk.1 { chunk = (c, tps); }
    }

    let mut gm = load(chunk.0)?;
    let mut short = usize::MAX;
    for n in [8, 16, 32, 64, 128] {
        let p = &pf[..n];
        gm.short = usize::MAX;
        let ts = best(&mut || { gm.reset()?; gm.prefill(p).map(drop) })?;
        gm.short = 0;
        let tb = best(&mut || { gm.reset()?; gm.prefill(p).map(drop) })?;
        eprintln!("prompt {n:4}: steps {:.1} ms, batched {:.1} ms", ts * 1e3, tb * 1e3);
        if tb < ts && short == usize::MAX { short = n; }
    }
    if short == usize::MAX { short = 256; }

    let dec = tok.encode("Write a long story about a dragon.")?;
    let kmax = gm.k;
    let mut r = Runner::Gpu(Box::new(gm));
    let mut mtp = (0, 0.0);
    for k in 0..=kmax {
        if let Runner::Gpu(g) = &mut r { g.k = k.max(1); }
        let mut toks = 0;
        let t = best(&mut || {
            r.reset()?;
            let l = r.prefill(&dec)?;
            toks = decode(&mut r, &tok, k > 0, &mut Sampler::greedy(), l, 128, &mut |_| Ok(false))?.0.len();
            Ok(())
        })?;
        // `best` times prefill too; it is the same for every k, so the ranking holds.
        let tps = toks as f64 / t;
        eprintln!("mtp {k}: decode {toks} tok {tps:.1} tok/s");
        if tps > mtp.1 { mtp = (k, tps); }
    }

    let Runner::Gpu(g) = &r else { unreachable!() };
    let path = calibration_path(model, g.gpu.name());
    std::fs::create_dir_all(path.parent().unwrap())?;
    let v = json!({"chunk": chunk.0, "mtp": mtp.0, "short": short, "ctx": ctx,
        "measured": {"prefill_tok_s": chunk.1.round(), "decode_tok_s": (mtp.1 * 10.0).round() / 10.0}});
    std::fs::write(&path, serde_json::to_string_pretty(&v)?)?;
    println!("{v}\nwritten to {}", path.display());
    Ok(())
}

/// Decoding of up to `n` tokens after `logits`; `emit` gets each new complete piece of text and returns true to stop.
/// Returns the tokens and whether an end-of-turn token or `emit` stopped it.
fn decode(r: &mut Runner, tok: &Tokenizer, mtp: bool, smp: &mut Sampler, mut logits: Vec<f32>, n: usize,
          emit: &mut dyn FnMut(&str) -> anyhow::Result<bool>) -> anyhow::Result<(Vec<u32>, bool)> {
    let eos = tok.eos_ids();
    // `ahead`: the next token when `spec_step` or `step_greedy` already picked it.
    // `start`: the first token whose text is not emitted yet (the text before it ends on a whole character).
    let (mut out, mut start, mut row, mut ahead) = (Vec::new(), 0, 0, None);
    while out.len() < n {
        let next = match ahead.take() { Some(t) => t, None => smp.pick(&logits)? as u32 };
        if eos.contains(&next) { return Ok((out, true)); }
        let toks = match r {
            Runner::Gpu(g) if mtp => {
                let (toks, nx, rw) = g.spec_step(next, row, smp)?;
                (ahead, row) = (Some(nx), rw);
                toks
            }
            // Greedy without MTP: the argmax on the GPU, so only the token id is copied back, not the logits.
            Runner::Gpu(g) if smp.temp <= 0.0 => { ahead = Some(g.step_greedy(next)?); vec![next] }
            Runner::Slot(c) if smp.temp <= 0.0 => { ahead = Some(c.step(next, false)?.0); vec![next] }
            _ => { logits = r.step(next)?; vec![next] }
        };
        for t in toks {
            if eos.contains(&t) { return Ok((out, true)); }
            if out.len() == n { break; }
            out.push(t);
            let s = tok.decode(&out[start..])?;
            if !s.ends_with('\u{fffd}') {
                if emit(&s)? { return Ok((out, true)); }
                start = out.len();
            }
        }
    }
    Ok((out, false))
}

/// Tokens left for a reply: `n`, capped by the context after a `used`-token prompt.
fn room(e: &Engine, used: usize, n: usize) -> usize { n.min(e.ctx.saturating_sub(used)) }

/// The chat format from the GGUF's `tokenizer.chat_template`.
fn template(g: &Gguf) -> Template {
    let t = Template::detect(g.meta.get("tokenizer.chat_template").and_then(|v| v.as_str()));
    eprintln!("chat template: {t:?}");
    t
}

/// Reply text without the thinking part (the template drops it from history too).
fn answer(s: &str) -> String { s.rsplit_once("</think>").map_or(s, |x| x.1).trim().to_string() }

/// A model `serve` has loaded.
struct Loaded<'a> { r: Runner<'a>, tok: &'a Tokenizer, tpl: Template, name: String, path: PathBuf }

/// What `serve` keeps across model loads.
struct Serving {
    rx: std::sync::Mutex<std::sync::mpsc::Receiver<(TcpStream, anyhow::Result<serve::Request>, usize)>>,
    waiting: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    addr: String,
    keys: std::sync::Arc<std::sync::RwLock<serve::Keys>>,
    usage: Option<std::sync::Mutex<std::fs::File>>,
    max_tokens: usize,
    dir: PathBuf,
    /// The --model, offered beside the models folder.
    model: Option<PathBuf>,
    /// The Hugging Face download running on its own thread, or the last one.
    dl: std::sync::Arc<std::sync::Mutex<hf::Progress>>,
    stats: std::sync::Arc<std::sync::Mutex<serve::Stats>>,
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
    /// Answers requests until one asks for a model to be loaded; returns its path and connection (None once the
    /// listener is gone).
    fn serve(&self, mut l: Option<&mut Loaded>, e: &Engine) -> Option<(PathBuf, TcpStream)> {
        if let Some(l) = l.as_deref_mut() && matches!(&l.r, Runner::Gpu(g) if g.slots() > 1) { return self.serve_slots(l, e); }
        for (c, q, i) in self.rx.lock().unwrap().iter() {
            let mut next = None;
            let res = handle(c, q, l.as_deref_mut(), e, self, i, &mut next);
            self.waiting.fetch_sub(1, Relaxed);
            if let Err(e) = res { eprintln!("request {i}: {e:#}"); }
            if next.is_some() { return next; }
        }
        None
    }
    /// The models the page offers: the models folder, then --model if it is elsewhere.
    fn models(&self) -> Vec<(String, PathBuf, u64)> {
        let mut v = serve::list_models(&self.dir);
        if let Some(p) = self.model.as_ref().filter(|p| !v.iter().any(|x| &x.1 == *p)) {
            v.push((p.display().to_string(), p.clone(), std::fs::metadata(p).map_or(0, |m| m.len())));
        }
        v
    }
}

/// What a request thread asks of the model's thread under `--parallel` (`Serving::serve_slots`).
enum Job {
    /// `prefill_cached` in a free slot; answers the logits, the tokens reused and the slot.
    Chat { ids: Vec<u32>, sys: usize, keep: usize, owner: u64, back: std::sync::mpsc::Sender<anyhow::Result<(Vec<f32>, usize, usize)>> },
    /// One decode step in a slot the request holds; answers the greedy next token and the logits if `logits`.
    Step { slot: usize, token: u32, logits: bool, back: std::sync::mpsc::Sender<anyhow::Result<(u32, Vec<f32>)>> },
    /// The request is done with its slot.
    Done(usize),
}

/// A request's handle on the model under `--parallel`; dropping it frees the slot.
struct Client { tx: std::sync::mpsc::Sender<Job>, slot: Option<usize>, owner: u64 }

impl Client {
    fn chat(&mut self, ids: &[u32], sys: usize, keep: usize) -> anyhow::Result<(Vec<f32>, usize)> {
        if let Some(s) = self.slot.take() { let _ = self.tx.send(Job::Done(s)); }
        let (back, r) = std::sync::mpsc::channel();
        self.tx.send(Job::Chat { ids: ids.to_vec(), sys, keep, owner: self.owner, back }).map_err(|_| anyhow::anyhow!("model gone"))?;
        let (logits, reused, s) = r.recv()??;
        self.slot = Some(s);
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
    /// Chat jobs waiting for a free slot, and decode steps waiting for a forward (since the first came).
    chats: std::collections::VecDeque<Job>,
    steps: Vec<(usize, u32, bool, std::sync::mpsc::Sender<anyhow::Result<(u32, Vec<f32>)>>)>,
    since: Option<Instant>,
}

impl Sched {
    /// How long a decode step waits for the other decoding slots' steps to share its forward.
    const GATHER: Duration = Duration::from_millis(2);

    fn decoding(&self) -> usize { self.held.iter().filter(|&&h| h).count() }

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

    /// One forward over the oldest steps (with more than `TMAX` slots decoding, each waits its turn), in slot order
    /// (fewer graphs to capture).
    fn run(&mut self, g: &mut GpuModel) {
        let mut batch: Vec<_> = self.steps.drain(..self.steps.len().min(rusting_strata::cpu_worker::TMAX)).collect();
        batch.sort_by_key(|s| s.0);
        self.since = if self.steps.is_empty() { None } else { Some(Instant::now() - Self::GATHER) };
        match g.step_rows(&batch.iter().map(|b| (b.0, b.1, b.2)).collect::<Vec<_>>()) {
            Ok(rows) => for (b, l) in batch.into_iter().zip(rows) { let _ = b.3.send(Ok(l)); },
            Err(err) => for b in batch { let _ = b.3.send(Err(anyhow::anyhow!("{err:#}"))); },
        }
    }
}

impl Serving {
    /// `serve` with several slots (`--parallel`): each request runs `handle` on its own thread and sends its model
    /// work here, where prefills run one at a time (the decoding slots run between its chunks) and the decode steps of
    /// up to `TMAX` slots share a forward.
    fn serve_slots(&self, l: &mut Loaded, e: &Engine) -> Option<(PathBuf, TcpStream)> {
        use std::sync::mpsc::{TryRecvError, channel};
        let Loaded { r, tok, tpl, name, path } = l;
        let Runner::Gpu(g) = r else { unreachable!() };
        let (tx, jobs) = channel::<Job>();
        let (next, live) = (std::sync::Mutex::new(None), std::sync::atomic::AtomicUsize::new(0));
        let rx = self.rx.lock().unwrap();
        let n = g.slots();
        std::thread::scope(|sc| {
            // Owned here, so that a panic drops the queued jobs' reply channels and the request threads end (the
            // scope then joins them and the panic goes on).
            let (jobs, mut s) = (jobs, Sched { held: vec![false; n], freed: vec![0; n], ..Default::default() });
            let mut open = true;
            loop {
                while open && next.lock().unwrap().is_none() {
                    match rx.try_recv() {
                        Ok((c, q, i)) => {
                            live.fetch_add(1, Relaxed);
                            let (tx, tok, tpl, name, path, next, live) = (tx.clone(), &**tok, *tpl, name.clone(), path.clone(), &next, &live);
                            sc.spawn(move || {
                                let mut l = Loaded { r: Runner::Slot(Client { tx, slot: None, owner: 0 }), tok, tpl, name, path };
                                let mut nx = None;
                                let res = handle(c, q, Some(&mut l), e, self, i, &mut nx);
                                drop(l);
                                self.waiting.fetch_sub(1, Relaxed);
                                if let Err(e) = res { eprintln!("request {i}: {e:#}"); }
                                if nx.is_some() { *next.lock().unwrap() = nx; }
                                live.fetch_sub(1, Relaxed);
                            });
                        }
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => open = false,
                    }
                }
                if (!open || next.lock().unwrap().is_some()) && live.load(Relaxed) == 0 { break; }
                s.take(&jobs, s.wait());
                // One prefill between decode steps, in the free slot whose saved state this prompt extends the
                // furthest, else the least recently used one.
                if let Some(sl) = (0..n).filter(|&i| !s.held[i]).max_by_key(|&i| {
                    let reuse = match s.chats.front() {
                        Some(Job::Chat { ids, keep, owner, .. }) => { let (sv, o) = g.slot_saved(i); if o == *owner && ids[..*keep].starts_with(sv) { sv.len() } else { 0 } }
                        _ => 0,
                    };
                    (reuse, std::cmp::Reverse(s.freed[i]))
                }) && let Some(Job::Chat { ids, sys, keep, owner, back }) = s.chats.pop_front() {
                    g.use_slot(sl);
                    g.owner = owner;
                    // Between two prefill chunks the decoding slots run for as long as the chunk before took, so a long
                    // prompt shares the GPU with them evenly instead of stalling them.
                    // ponytail: an even share; weigh it (or cap the prompt's chunk) if first-token time matters more.
                    let mut t0 = Instant::now();
                    let r = g.prefill_cached_with(&ids, sys, keep, &mut |g| {
                        let end = Instant::now() + t0.elapsed();
                        while s.decoding() > 0 && Instant::now() < end {
                            s.take(&jobs, s.wait().min(end.saturating_duration_since(Instant::now())));
                            if s.ready() { s.run(g); }
                        }
                        t0 = Instant::now();
                        Ok(())
                    });
                    s.held[sl] = r.is_ok();
                    let _ = back.send(r.map(|(l, u)| (l, u, sl)));
                    gpu_model_stats(g, &mut self.stats.lock().unwrap());
                }
                if s.ready() { s.run(g); }
            }
        });
        next.into_inner().unwrap()
    }
}

/// Loads `model`, answers `w` (the /api/load request that asked for it) and serves until another model is asked
/// for. The model and its GPU memory are freed on return.
fn serve_model(model: &Path, mut engine: Engine, s: &Serving, w: &mut Option<TcpStream>) -> anyhow::Result<Option<(PathBuf, TcpStream)>> {
    eprintln!("loading {}", model.display());
    let name = model.file_stem().and_then(|s| s.to_str()).unwrap_or("strata").to_string();
    s.stats.lock().unwrap().unload(Some(name.clone()));
    let g = Gguf::open(model)?;
    let m = Model::load(&g)?;
    let tok = Tokenizer::from_gguf(&g)?;
    let r = engine.runner(&m, model)?;
    {
        let mut st = s.stats.lock().unwrap();
        st.loading = false;
        (st.backend, st.device, st.ctx, st.slots) = match &r { Runner::Gpu(g) => (g.gpu.name().into(), g.gpu.device(), engine.ctx, g.slots()), _ => ("cpu".into(), String::new(), engine.ctx, 1) };
        gpu_stats(&r, &mut st);
    }
    let mut l = Loaded { r, tok: &tok, tpl: template(&g), name, path: model.to_path_buf() };
    eprintln!("loaded {}", l.name);
    if let Some(mut w) = w.take() {
        let b = json!({"loaded": l.name}).to_string();
        let _ = write!(w, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len());
    }
    let next = s.serve(Some(&mut l), &engine);
    s.stats.lock().unwrap().unload(None);
    // A failed profile save must not stop the server between two models.
    if let Err(e) = l.r.finish() { eprintln!("{e:#}"); }
    Ok(next)
}

/// The side panel's device memory, expert cache and MTP counts.
fn gpu_stats(r: &Runner, s: &mut serve::Stats) {
    if let Runner::Gpu(g) = r { gpu_model_stats(g, s) }
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
    if let Err(e) = r { eprintln!("could not open a browser ({e}); open {url}"); }
}

/// One Messages API stream event.
fn sse(w: &mut impl Write, v: serde_json::Value) -> anyhow::Result<()> { Ok(write!(w, "event: {}\ndata: {v}\n\n", v["type"].as_str().unwrap_or_default())?) }

/// Streamed reply text as a Messages API text block (index 0), started by the first piece.
fn text_delta(w: &mut impl Write, opened: &mut bool, t: &str) -> anyhow::Result<()> {
    if !std::mem::replace(opened, true) { sse(w, json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}))?; }
    sse(w, json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": t}}))
}

fn handle(mut w: TcpStream, q: anyhow::Result<serve::Request>, l: Option<&mut Loaded>, e: &Engine, s: &Serving, i: usize,
    next: &mut Option<(PathBuf, TcpStream)>) -> anyhow::Result<()> {
    let send = |w: &mut TcpStream, status: &str, v: serde_json::Value| -> anyhow::Result<()> {
        let b = v.to_string();
        Ok(write!(w, "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len())?)
    };
    // A malformed request gets a 400 (best effort: the client may be gone) and is still logged.
    let serve::Request { method, path, host, auth, json, body } = match q {
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
    if !keys.any() && !serve::host_ok(&host, &s.addr) { return send(&mut w, "403 Forbidden", json!({"error": {"message": "bad Host header", "type": "forbidden"}})); }
    // The chat page holds no secret, so it loads without the key (the page asks for it).
    if method == "GET" && path == "/" {
        const PAGE: &str = include_str!("web/index.html");
        return Ok(write!(w, "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nX-Frame-Options: DENY\r\nContent-Security-Policy: frame-ancestors 'none'\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{PAGE}", PAGE.len())?);
    }
    let Some(who) = keys.who(&auth) else { return send(&mut w, "401 Unauthorized", json!({"error": {"message": "missing or wrong API key", "type": "invalid_api_key"}})) };
    if method == "POST" && !json { return send(&mut w, "415 Unsupported Media Type", json!({"error": {"message": "Content-Type must be application/json", "type": "invalid_request_error"}})); }
    // Claude Code adds `?beta=true`.
    let route = path.split('?').next().unwrap_or_default();
    // Customer keys may not load models, download files or read the server's stats.
    if !who.admin && !matches!(route, "/v1/chat/completions" | "/v1/messages" | "/v1/messages/count_tokens" | "/v1/models") {
        return send(&mut w, "403 Forbidden", json!({"error": {"message": "this key may use only /v1/chat/completions, /v1/messages (and count_tokens) and /v1/models", "type": "permission_error"}}));
    }
    match (method.as_str(), route) {
        ("GET", "/v1/models") => send(&mut w, "200 OK", json!({"object": "list", "data": l.map(|l| json!([{"id": l.name, "object": "model", "created": created, "owned_by": "strata"}])).unwrap_or(json!([]))})),
        ("GET", "/api/models") => {
            let models: Vec<_> = s.models().into_iter().map(|(id, p, n)| json!({"id": id, "bytes": n, "loaded": l.as_ref().is_some_and(|l| l.path == p)})).collect();
            send(&mut w, "200 OK", json!({"dir": s.dir, "models": models}))
        }
        // Answered once the model has loaded (or failed to).
        ("POST", "/api/load") => {
            let id = serde_json::from_slice::<serde_json::Value>(&body).ok().and_then(|v| v["model"].as_str().map(String::from)).unwrap_or_default();
            match s.models().into_iter().find(|m| m.0 == id) {
                Some((_, p, _)) => { *next = Some((p, w)); Ok(()) }
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
                if let Err(e) = &r { eprintln!("download {repo} {file}: {e:#}"); } else { eprintln!("downloaded {repo} {file}"); }
                let mut g = dl.lock().unwrap();
                (g.running, g.error) = (false, r.err().map(|e| format!("{e:#}")));
            });
            send(&mut w, "202 Accepted", json!(*s.dl.lock().unwrap()))
        }
        // The prompt's tokens as the model would see them, template and tools included.
        ("POST", "/v1/messages/count_tokens") => {
            let Some(l) = l else { return send(&mut w, "503 Service Unavailable", json!({"error": {"message": "no model loaded", "type": "server_error"}})) };
            match serve::messages_body(&body).and_then(|b| serve::chat_request(&b)) {
                Ok(req) => {
                    let prompt = l.tpl.render(&req.messages, &req.tools, req.think, req.preserve_thinking.unwrap_or(e.preserve_thinking), true);
                    send(&mut w, "200 OK", json!({"input_tokens": l.tok.encode(&prompt)?.len()}))
                }
                Err(err) => send(&mut w, "400 Bad Request", json!({"error": {"message": format!("{err:#}"), "type": "invalid_request_error"}})),
            }
        }
        // The Anthropic Messages API (Claude Code) is translated to a chat request, and the reply back.
        ("POST", p @ ("/v1/chat/completions" | "/v1/messages")) => {
            let anth = p == "/v1/messages";
            let Some(l) = l else { return send(&mut w, "503 Service Unavailable", json!({"error": {"message": "no model loaded", "type": "server_error"}})) };
            let (r, tok, tpl, name) = (&mut l.r, &l.tok, l.tpl, l.name.as_str());
            match r { Runner::Gpu(g) => g.owner = who.id, Runner::Slot(c) => c.owner = who.id, _ => {} }
            let req = match if anth { serve::messages_body(&body) } else { Ok(body.clone()) }.and_then(|b| serve::chat_request(&b)) {
                Ok(q) => q,
                Err(err) => return send(&mut w, "400 Bad Request", json!({"error": {"message": format!("{err:#}"), "type": "invalid_request_error"}})),
            };
            // Room is checked on the full prompt before any state changes.
            let preserve = req.preserve_thinking.unwrap_or(e.preserve_thinking);
            let prompt = tpl.render(&req.messages, &req.tools, req.think, preserve, true);
            // The system and tools block, which compacted or rewritten conversations keep.
            let n_sys = req.messages.iter().take_while(|m| m.role == "system").count();
            let system = tpl.render(&req.messages[..n_sys], &req.tools, req.think, preserve, false);
            let ids = tok.encode(&prompt)?;
            let mut smp = e.sampler();
            smp.temp = req.temperature.unwrap_or(smp.temp);
            smp.top_k = req.top_k.unwrap_or(smp.top_k);
            smp.top_p = req.top_p.unwrap_or(smp.top_p);
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
            let n_system = tok.encode(&system)?.len();
            let (t0, mut t1) = (Instant::now(), Instant::now());
            {
                let mut st = s.stats.lock().unwrap();
                if st.busy == 0 { st.live_tokens = 0; }
                st.busy += 1;
            }
            let res = (|| -> anyhow::Result<(Vec<u32>, bool)> {
                let logits;
                (_, logits, cached) = r.chat(tok, &prompt, &tpl.render(&req.messages, &req.tools, req.think, preserve, false), &system)?;
                t1 = Instant::now();
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
                decode(r, tok, e.mtp() > 0, &mut smp, logits, n, &mut |t| {
                    s.stats.lock().unwrap().live_tokens += 1;
                    // A client that left stops the reply (a stream notices on the next write; held text writes nothing).
                    k += 1;
                    if k % 16 == 0 && serve::gone(&w) { anyhow::bail!("client closed the connection"); }
                    let t = hold.push(t);
                    if req.stream && !t.is_empty() {
                        if !anth { write!(w, "data: {}\n\n", chunk(json!({"content": t}), None))?; } else { text_delta(&mut w, &mut opened, &t)?; }
                    }
                    Ok(hold.stopped)
                })
            })();
            {
                let mut st = s.stats.lock().unwrap();
                st.busy -= 1;
                gpu_stats(r, &mut st);
            }
            let decoded = t1.elapsed();
            let ms = |d: Duration| d.as_secs_f64() * 1e3;
            let mut last = serve::Last { at: created, model: name.into(), system_tokens: n_system, cached_tokens: cached,
                prompt_tokens: ids.len(), completion_tokens: k, prefill_ms: ms(t1 - t0), decode_ms: ms(decoded), finish: String::new() };
            let log = |last: &serve::Last| if let Some(f) = &s.usage {
                let mut v = json!(last);
                v["key"] = who.name.clone().into();
                // A failed write must not fail the reply; it is reported.
                if let Err(err) = writeln!(f.lock().unwrap(), "{v}") { eprintln!("usage log: {err}"); }
            };
            // The next request starts from the saved history state or from scratch. A stream that already started is just cut off.
            let (out, stop) = match res {
                Ok(x) => x,
                Err(err) => {
                    // Logged with the tokens decoded so far, which a client that left may have received.
                    let gone = serve::gone(&w);
                    last.finish = if gone { "cut" } else { "error" }.into();
                    log(&last);
                    if !req.stream && !gone { return send(&mut w, "500 Internal Server Error", json!({"error": {"message": format!("{err:#}"), "type": "server_error"}})); }
                    return Err(err);
                }
            };
            // A malformed call goes out as plain text.
            let calls = if tools { tpl.parse_calls(&hold.text, &req.tools) } else { None };
            last.completion_tokens = out.len();
            last.finish = if calls.is_some() { "tool_calls" } else if stop { "stop" } else { "length" }.into();
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
                        "stop_reason": reason, "stop_sequence": null, "usage": usage}));
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
                sse(&mut w, json!({"type": "message_delta", "delta": {"stop_reason": reason, "stop_sequence": null}, "usage": {"output_tokens": out.len()}}))?;
                return sse(&mut w, json!({"type": "message_stop"}));
            }
            let finish = if calls.is_some() { "tool_calls" } else if stop { "stop" } else { "length" };
            let usage = json!({"prompt_tokens": ids.len(), "completion_tokens": out.len(), "total_tokens": ids.len() + out.len(),
                "prompt_tokens_details": {"cached_tokens": cached}});
            if req.stream {
                match &calls {
                    Some(c) => write!(w, "data: {}\n\n", chunk(calls_json(c, true), None))?,
                    None if !hold.rest().is_empty() => write!(w, "data: {}\n\n", chunk(json!({"content": hold.rest()}), None))?,
                    None => {}
                }
                let mut last = chunk(json!({}), Some(finish));
                last["usage"] = usage;
                Ok(write!(w, "data: {last}\n\ndata: [DONE]\n\n")?)
            } else {
                let message = match &calls { Some(c) => calls_json(c, false), None => json!({"role": "assistant", "content": hold.text}) };
                send(&mut w, "200 OK", json!({"id": id, "object": "chat.completion", "created": created, "model": name,
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage}))
            }
        }
        _ => send(&mut w, "404 Not Found", json!({"error": {"message": format!("no route {method} {path}"), "type": "not_found"}})),
    }
}
