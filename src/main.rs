use clap::{Args, Parser, Subcommand};
use anyhow::Context;
use rusting_strata::model::{Model, State};
use rusting_strata::{cuda::Gpu, gpu_model::GpuModel};
use rusting_strata::{config::ModelConfig, gguf::Gguf, ops, ops::Sampler, quant::GgmlType, serve, tokenizer::Tokenizer};
use serde_json::json;
use std::io::{BufRead, BufReader, Read, Write};
use std::{collections::BTreeMap, net::{TcpListener, TcpStream}, path::{Path, PathBuf}, time::{Duration, Instant, SystemTime}};

#[derive(Parser)]
#[command(name = "strata", about = "Rust inference engine for Qwen3.5/3.6-family GGUF models")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args)]
struct Engine {
    /// Run on the GPU (static expert placement) instead of the CPU reference.
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
    /// Speculative decoding with the model's MTP draft head, N drafts per step (GPU, greedy output unchanged)
    /// [default: calibrated, else 0].
    #[arg(long, value_name = "N", default_missing_value = "1", num_args = 0..=1)]
    mtp: Option<usize>,
    /// Sampling temperature; 0 is greedy (and the only mode MTP speeds up).
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
    /// Prompts shorter than this skip the batched prefill (from `strata calibrate`, else 32).
    #[arg(skip)]
    short: usize,
}

enum Runner<'a> { Cpu(&'a Model<'a>, State), Gpu(Box<GpuModel<'a>>) }

impl Runner<'_> {
    fn step(&mut self, t: u32) -> anyhow::Result<Vec<f32>> {
        match self { Runner::Cpu(m, st) => Ok(m.step(st, t)), Runner::Gpu(g) => g.step(t) }
    }
    /// Feed a prompt; returns the logits after its last token.
    fn prefill(&mut self, ts: &[u32]) -> anyhow::Result<Vec<f32>> {
        match self {
            Runner::Cpu(m, st) => Ok(ts.iter().map(|&t| m.step(st, t)).last().unwrap_or_default()),
            Runner::Gpu(g) => g.prefill(ts),
        }
    }
    /// Feed a chat prompt as a fresh sequence, reusing the earlier request's state for the shared message history
    /// (GPU). Returns the logits and the number of tokens reused.
    fn chat(&mut self, tok: &Tokenizer, msgs: &[(String, String)], think: bool) -> anyhow::Result<(Vec<u32>, Vec<f32>, usize)> {
        let ids = tok.encode(&serve::chat_prompt(msgs, think))?;
        let h = tok.encode(&serve::chat_history(msgs))?;
        let keep = if ids.starts_with(&h) { h.len() } else { 0 };
        let (logits, reused) = match self {
            Runner::Gpu(g) => g.prefill_cached(&ids, keep)?,
            _ => { self.reset()?; (self.prefill(&ids)?, 0) }
        };
        Ok((ids, logits, reused))
    }
    /// Forget the sequence so far.
    fn reset(&mut self) -> anyhow::Result<()> {
        match self { Runner::Cpu(m, st) => { *st = State::new(&m.cfg); Ok(()) } Runner::Gpu(g) => g.reset() }
    }
    /// Seconds spent waiting on CPU experts so far (GPU only).
    fn cpu_secs(&self) -> f64 { if let Runner::Gpu(g) = self { g.cpu_time.as_secs_f64() } else { 0.0 } }
    fn finish(&self) -> anyhow::Result<()> {
        if let Runner::Gpu(g) = self {
            let n = (g.hits + g.misses).max(1);
            if g.drafts.0 > 0 { eprintln!("MTP drafts accepted {}/{} ({:.1}%)", g.drafts.1, g.drafts.0, 100.0 * g.drafts.1 as f64 / g.drafts.0 as f64); }
            eprintln!("expert hit rate {:.1}% ({} misses), cache swaps {}", 100.0 * g.hits as f64 / n as f64, g.misses,
                g.cache.as_ref().map_or(0, |c| c.swaps));
            if let Some(c) = &g.cache { c.save_profile()?; }
        }
        Ok(())
    }
}

impl Engine {
    /// Fills settings not given on the command line from the model's calibration file, else built-in defaults.
    fn runner<'a>(&mut self, m: &'a Model<'a>, model: &Path) -> anyhow::Result<Runner<'a>> {
        let path = calibration_path(model);
        let v: serde_json::Value = std::fs::read(&path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
        if !v.is_null() { eprintln!("calibration: {}", path.display()); }
        let get = |k: &str| v[k].as_u64().map(|x| x as usize);
        self.chunk.get_or_insert(get("chunk").unwrap_or(4096));
        self.mtp.get_or_insert(get("mtp").unwrap_or(0));
        self.short = get("short").unwrap_or(32);
        if let Some(n) = self.threads.or(get("threads")) { rayon::ThreadPoolBuilder::new().num_threads(n).build_global()?; }
        if !self.gpu { return Ok(Runner::Cpu(m, State::new(&m.cfg))); }
        let mut g = GpuModel::load(m, Gpu::new()?, self.ctx, self.chunk(), self.vram_margin << 20, self.mtp())?;
        g.short = self.short;
        Ok(Runner::Gpu(Box::new(g)))
    }
    fn sampler(&self) -> Sampler {
        Sampler { temp: self.temp, top_k: self.top_k, top_p: self.top_p, rng: seed(self.seed) }
    }
    fn chunk(&self) -> usize { self.chunk.unwrap_or(4096) }
    fn mtp(&self) -> usize { self.mtp.unwrap_or(0) }
}

/// A nonzero xorshift state from `s`, or from the clock.
fn seed(s: Option<u64>) -> u64 {
    s.unwrap_or_else(|| SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64))
        .wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1
}

fn calibration_path(model: &Path) -> PathBuf {
    let stem = model.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    rusting_strata::expert_cache::cache_dir().join(format!("strata-{stem}.json"))
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
    /// extends the previous one's messages prefills only the new part (usage.prompt_tokens_details.cached_tokens).
    Serve {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value = "127.0.0.1:8080")]
        addr: String,
        /// Require `Authorization: Bearer <key>` on every request [default: $STRATA_API_KEY, which keeps the key out
        /// of the process list].
        #[arg(long)]
        api_key: Option<String>,
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
    match Cli::parse().cmd {
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
                    .iter().map(|x| x.as_u64().map(|x| x as u32).context("bad id")).collect()
            };
            let (prompt, cont) = (ids("prompt")?, ids("continuation")?);
            anyhow::ensure!(!prompt.is_empty() && !cont.is_empty(), "empty prompt or continuation");
            let g = Gguf::open(&model)?;
            let m = Model::load(&g)?;
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
            let mut r = engine.runner(&m, &model)?;
            let prompt = if prompt == "-" {
                let mut s = String::new();
                std::io::stdin().read_to_string(&mut s)?;
                s
            } else { prompt };
            let ids = tok.encode(&prompt)?;
            let t0 = Instant::now();
            let logits = r.prefill(&ids)?;
            let t1 = Instant::now();
            let cpu0 = r.cpu_secs();
            let (out, _) = decode(&mut r, &tok, engine.mtp() > 0, &mut engine.sampler(), logits, n, &mut |t| { print!("{t}"); Ok(std::io::stdout().flush()?) })?;
            let (pp, tg) = ((t1 - t0).as_secs_f64(), t1.elapsed().as_secs_f64());
            eprintln!("\nprompt {} tok {:.2} tok/s | gen {} tok {:.2} tok/s", ids.len(), ids.len() as f64 / pp, out.len(), out.len() as f64 / tg);
            if engine.gpu { eprintln!("CPU expert compute, share of gen time: {:.1}%", 100.0 * (r.cpu_secs() - cpu0) / tg); }
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
                    let k = decode(&mut r, &tok, engine.mtp() > 0, &mut Sampler::greedy(), l, tg, &mut |_| Ok(()))?.0.len();
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
        Cmd::Calibrate { model, ctx, vram_margin } => calibrate(&model, ctx, vram_margin << 20)?,
        Cmd::Chat { model, system, n, think, mut engine } => {
            let g = Gguf::open(&model)?;
            let m = Model::load(&g)?;
            let tok = Tokenizer::from_gguf(&g)?;
            let mut r = engine.runner(&m, &model)?;
            let mut smp = engine.sampler();
            let mut msgs: Vec<(String, String)> = system.into_iter().map(|s| ("system".into(), s)).collect();
            let stdin = std::io::stdin();
            loop {
                print!("> ");
                std::io::stdout().flush()?;
                let mut line = String::new();
                if stdin.lock().read_line(&mut line)? == 0 { break; }
                if line.trim().is_empty() { continue; }
                msgs.push(("user".into(), line.trim_end().into()));
                let (ids, logits, _) = r.chat(&tok, &msgs, think)?;
                let (out, _) = decode(&mut r, &tok, engine.mtp() > 0, &mut smp, logits, room(&engine, ids.len(), n), &mut |t| { print!("{t}"); Ok(std::io::stdout().flush()?) })?;
                println!();
                msgs.push(("assistant".into(), answer(&tok.decode(&out)?)));
            }
            r.finish()?;
        }
        Cmd::Serve { model, addr, api_key, mut engine } => {
            let api_key = api_key.or_else(|| std::env::var("STRATA_API_KEY").ok());
            anyhow::ensure!(api_key.as_ref().is_none_or(|k| !k.is_empty()), "empty API key");
            let g = Gguf::open(&model)?;
            let m = Model::load(&g)?;
            let tok = Tokenizer::from_gguf(&g)?;
            let mut r = engine.runner(&m, &model)?;
            let name = model.file_stem().and_then(|s| s.to_str()).unwrap_or("strata").to_string();
            let listener = TcpListener::bind(&addr)?;
            eprintln!("listening on http://{addr}");
            for (i, conn) in listener.incoming().enumerate() {
                let res = conn.map_err(Into::into).and_then(|c| handle(c, &mut r, &tok, &engine, &name, &addr, api_key.as_deref(), i));
                if let Err(e) = res { eprintln!("request {i}: {e:#}"); }
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

fn calibrate(model: &Path, ctx: usize, margin: usize) -> anyhow::Result<()> {
    let g = Gguf::open(model)?;
    let m = Model::load(&g)?;
    let tok = Tokenizer::from_gguf(&g)?;
    let n_pf = 4096.min(ctx.saturating_sub(256));
    anyhow::ensure!(n_pf >= 512, "--ctx too small to calibrate");
    let pf = prose(&tok, n_pf)?;
    let load = |chunk: usize| -> anyhow::Result<GpuModel> {
        let mut gm = GpuModel::load(&m, Gpu::new()?, ctx, chunk, margin, 3)?;
        gm.adapt_every = 0; // fixed placement: steadier timings
        Ok(gm)
    };

    let mut chunk = (0, 0.0);
    for c in [512, 1024, 2048, 4096].into_iter().filter(|&c| c <= ctx) {
        let mut gm = load(c)?;
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
            toks = decode(&mut r, &tok, k > 0, &mut Sampler::greedy(), l, 128, &mut |_| Ok(()))?.0.len();
            Ok(())
        })?;
        // `best` times prefill too; it is the same for every k, so the ranking holds.
        let tps = toks as f64 / t;
        eprintln!("mtp {k}: decode {toks} tok {tps:.1} tok/s");
        if tps > mtp.1 { mtp = (k, tps); }
    }

    let path = calibration_path(model);
    std::fs::create_dir_all(path.parent().unwrap())?;
    let v = json!({"chunk": chunk.0, "mtp": mtp.0, "short": short, "ctx": ctx,
        "measured": {"prefill_tok_s": chunk.1.round(), "decode_tok_s": (mtp.1 * 10.0).round() / 10.0}});
    std::fs::write(&path, serde_json::to_string_pretty(&v)?)?;
    println!("{v}\nwritten to {}", path.display());
    Ok(())
}

/// Decoding of up to `n` tokens after `logits`; `emit` gets each new complete piece of text.
/// Returns the tokens and whether an end-of-turn token stopped it.
fn decode(r: &mut Runner, tok: &Tokenizer, mtp: bool, smp: &mut Sampler, mut logits: Vec<f32>, n: usize,
          emit: &mut dyn FnMut(&str) -> anyhow::Result<()>) -> anyhow::Result<(Vec<u32>, bool)> {
    let eos = tok.eos_ids();
    let (mut out, mut printed, mut row) = (Vec::new(), 0, 0);
    while out.len() < n {
        let next = smp.pick(&logits) as u32;
        if eos.contains(&next) { return Ok((out, true)); }
        let toks = match r {
            // ponytail: MTP drafts are checked greedily, so sampling decodes one token per step
            Runner::Gpu(g) if mtp && smp.temp <= 0.0 => {
                let (toks, l, rw) = g.spec_step(next, row)?;
                (logits, row) = (l, rw);
                toks
            }
            _ => { logits = r.step(next)?; vec![next] }
        };
        for t in toks {
            if eos.contains(&t) { return Ok((out, true)); }
            if out.len() == n { break; }
            out.push(t);
            let s = tok.decode(&out)?;
            if !s.ends_with('\u{fffd}') {
                emit(&s[printed..])?;
                printed = s.len();
            }
        }
    }
    Ok((out, false))
}

/// Tokens left for a reply: `n`, capped by the context after a `used`-token prompt.
fn room(e: &Engine, used: usize, n: usize) -> usize { n.min(e.ctx.saturating_sub(used)) }

/// Reply text without the thinking part (the template drops it from history too).
fn answer(s: &str) -> String { s.rsplit_once("</think>").map_or(s, |x| x.1).trim().to_string() }

fn handle(c: TcpStream, r: &mut Runner, tok: &Tokenizer, e: &Engine, name: &str, addr: &str, key: Option<&str>, i: usize) -> anyhow::Result<()> {
    // A stalled client must not hold the single request slot.
    c.set_read_timeout(Some(Duration::from_secs(30)))?;
    c.set_write_timeout(Some(Duration::from_secs(30)))?;
    let mut w = c.try_clone()?;
    let serve::Request { method, path, host, auth, json, body } = serve::read_request(&mut BufReader::new(c))?;
    let send = |w: &mut TcpStream, status: &str, v: serde_json::Value| -> anyhow::Result<()> {
        let b = v.to_string();
        Ok(write!(w, "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len())?)
    };
    let created = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_secs();
    // Browsers can reach a loopback server: the Host check stops DNS rebinding, and requiring a JSON content type
    // makes a cross-site POST need a CORS preflight, which is never answered.
    if !serve::host_ok(&host, addr) { return send(&mut w, "403 Forbidden", json!({"error": {"message": "bad Host header", "type": "forbidden"}})); }
    if key.is_some_and(|k| !serve::key_ok(&auth, k)) { return send(&mut w, "401 Unauthorized", json!({"error": {"message": "missing or wrong API key", "type": "invalid_api_key"}})); }
    if method == "POST" && !json { return send(&mut w, "415 Unsupported Media Type", json!({"error": {"message": "Content-Type must be application/json", "type": "invalid_request_error"}})); }
    match (method.as_str(), path.as_str()) {
        ("GET", "/v1/models") => send(&mut w, "200 OK", json!({"object": "list", "data": [{"id": name, "object": "model", "created": created, "owned_by": "strata"}]})),
        ("POST", "/v1/chat/completions") => {
            let req = match serve::chat_request(&body) {
                Ok(q) => q,
                Err(err) => return send(&mut w, "400 Bad Request", json!({"error": {"message": format!("{err:#}"), "type": "invalid_request_error"}})),
            };
            // Room is checked on the full prompt before any state changes.
            let ids = tok.encode(&serve::chat_prompt(&req.messages, req.think))?;
            let mut smp = e.sampler();
            smp.temp = req.temperature.unwrap_or(smp.temp);
            smp.top_k = req.top_k.unwrap_or(smp.top_k);
            smp.top_p = req.top_p.unwrap_or(smp.top_p);
            if req.seed.is_some() { smp.rng = seed(req.seed); }
            let n = room(e, ids.len(), req.max_tokens.unwrap_or(usize::MAX));
            if n == 0 { return send(&mut w, "400 Bad Request", json!({"error": {"message": format!("prompt of {} tokens fills the context", ids.len()), "type": "invalid_request_error"}})); }
            let id = format!("chatcmpl-{i}");
            let chunk = |delta: serde_json::Value, finish: Option<&str>| json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
            let mut cached = 0;
            let res = (|| -> anyhow::Result<(Vec<u32>, bool, String)> {
                let logits;
                (_, logits, cached) = r.chat(tok, &req.messages, req.think)?;
                if req.stream {
                    write!(w, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")?;
                    write!(w, "data: {}\n\n", chunk(json!({"role": "assistant", "content": ""}), None))?;
                    let (out, stop) = decode(r, tok, e.mtp() > 0, &mut smp, logits, n, &mut |t| Ok(write!(w, "data: {}\n\n", chunk(json!({"content": t}), None))?))?;
                    Ok((out, stop, String::new()))
                } else {
                    let mut text = String::new();
                    let (out, stop) = decode(r, tok, e.mtp() > 0, &mut smp, logits, n, &mut |t| { text += t; Ok(()) })?;
                    Ok((out, stop, text))
                }
            })();
            // The next request starts from the saved history state or from scratch. A stream that already started is just cut off.
            let (out, stop, text) = match res {
                Ok(x) => x,
                Err(err) if !req.stream => return send(&mut w, "500 Internal Server Error", json!({"error": {"message": format!("{err:#}"), "type": "server_error"}})),
                Err(err) => return Err(err),
            };
            let finish = if stop { "stop" } else { "length" };
            let usage = json!({"prompt_tokens": ids.len(), "completion_tokens": out.len(), "total_tokens": ids.len() + out.len(),
                "prompt_tokens_details": {"cached_tokens": cached}});
            if req.stream {
                let mut last = chunk(json!({}), Some(finish));
                last["usage"] = usage;
                Ok(write!(w, "data: {last}\n\ndata: [DONE]\n\n")?)
            } else {
                send(&mut w, "200 OK", json!({"id": id, "object": "chat.completion", "created": created, "model": name,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}], "usage": usage}))
            }
        }
        _ => send(&mut w, "404 Not Found", json!({"error": {"message": format!("no route {method} {path}"), "type": "not_found"}})),
    }
}
