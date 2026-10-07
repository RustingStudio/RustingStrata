use clap::{Args, Parser, Subcommand};
use anyhow::Context;
use rusting_strata::model::{Model, State};
use rusting_strata::{cuda::Gpu, gpu_model::GpuModel};
use rusting_strata::{config::ModelConfig, gguf::Gguf, ops, quant::GgmlType, serve, tokenizer::Tokenizer};
use serde_json::json;
use std::io::{BufRead, BufReader, Read, Write};
use std::{collections::BTreeMap, net::{TcpListener, TcpStream}, path::PathBuf, time::{Duration, Instant, SystemTime}};

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
    /// Prefill chunk in tokens (GPU; 0 = token by token).
    #[arg(long, default_value_t = 4096)]
    chunk: usize,
    /// VRAM left free after placing expert layers, in MB.
    #[arg(long, default_value_t = 512)]
    vram_margin: usize,
    /// Speculative decoding with the model's MTP draft head, N drafts per step (GPU, greedy output unchanged).
    #[arg(long, value_name = "N", default_value_t = 0, default_missing_value = "1", num_args = 0..=1)]
    mtp: usize,
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
    fn runner<'a>(&self, m: &'a Model<'a>) -> anyhow::Result<Runner<'a>> {
        if !self.gpu { return Ok(Runner::Cpu(m, State::new(&m.cfg))); }
        Ok(Runner::Gpu(Box::new(GpuModel::load(m, Gpu::new()?, self.ctx, self.chunk, self.vram_margin << 20, self.mtp)?)))
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
    /// Interactive chat on stdin (greedy; the whole conversation is prefilled again each turn).
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
    /// One request at a time; greedy decoding.
    Serve {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value = "127.0.0.1:8080")]
        addr: String,
        #[command(flatten)]
        engine: Engine,
    },
    /// Greedy generation.
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
        Cmd::Score { model, input, engine } => {
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
            let mut r = engine.runner(&m)?;
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
        Cmd::Generate { model, prompt, n, engine } => {
            let g = Gguf::open(&model)?;
            let m = Model::load(&g)?;
            let tok = Tokenizer::from_gguf(&g)?;
            let mut r = engine.runner(&m)?;
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
            let (out, _) = decode(&mut r, &tok, engine.mtp > 0, logits, n, &mut |t| { print!("{t}"); Ok(std::io::stdout().flush()?) })?;
            let (pp, tg) = ((t1 - t0).as_secs_f64(), t1.elapsed().as_secs_f64());
            eprintln!("\nprompt {} tok {:.2} tok/s | gen {} tok {:.2} tok/s", ids.len(), ids.len() as f64 / pp, out.len(), out.len() as f64 / tg);
            if engine.gpu { eprintln!("CPU expert compute, share of gen time: {:.1}%", 100.0 * (r.cpu_secs() - cpu0) / tg); }
            r.finish()?;
        }
        Cmd::Chat { model, system, n, think, engine } => {
            let g = Gguf::open(&model)?;
            let m = Model::load(&g)?;
            let tok = Tokenizer::from_gguf(&g)?;
            let mut r = engine.runner(&m)?;
            let mut msgs: Vec<(String, String)> = system.into_iter().map(|s| ("system".into(), s)).collect();
            let stdin = std::io::stdin();
            loop {
                print!("> ");
                std::io::stdout().flush()?;
                let mut line = String::new();
                if stdin.lock().read_line(&mut line)? == 0 { break; }
                if line.trim().is_empty() { continue; }
                msgs.push(("user".into(), line.trim_end().into()));
                let ids = tok.encode(&serve::chat_prompt(&msgs, think))?;
                r.reset()?;
                let logits = r.prefill(&ids)?;
                let (out, _) = decode(&mut r, &tok, engine.mtp > 0, logits, room(&engine, ids.len(), n), &mut |t| { print!("{t}"); Ok(std::io::stdout().flush()?) })?;
                println!();
                msgs.push(("assistant".into(), answer(&tok.decode(&out)?)));
            }
            r.finish()?;
        }
        Cmd::Serve { model, addr, engine } => {
            let g = Gguf::open(&model)?;
            let m = Model::load(&g)?;
            let tok = Tokenizer::from_gguf(&g)?;
            let mut r = engine.runner(&m)?;
            let name = model.file_stem().and_then(|s| s.to_str()).unwrap_or("strata").to_string();
            let listener = TcpListener::bind(&addr)?;
            eprintln!("listening on http://{addr}");
            for (i, conn) in listener.incoming().enumerate() {
                let res = conn.map_err(Into::into).and_then(|c| handle(c, &mut r, &tok, &engine, &name, &addr, i));
                if let Err(e) = res { eprintln!("request {i}: {e:#}"); }
            }
        }
    }
    Ok(())
}

/// Greedy decoding of up to `n` tokens after `logits`; `emit` gets each new complete piece of text.
/// Returns the tokens and whether an end-of-turn token stopped it.
fn decode(r: &mut Runner, tok: &Tokenizer, mtp: bool, mut logits: Vec<f32>, n: usize,
          emit: &mut dyn FnMut(&str) -> anyhow::Result<()>) -> anyhow::Result<(Vec<u32>, bool)> {
    let eos = tok.eos_ids();
    let (mut out, mut printed, mut row) = (Vec::new(), 0, 0);
    while out.len() < n {
        let next = ops::argmax(&logits) as u32;
        if eos.contains(&next) { return Ok((out, true)); }
        let toks = match r {
            Runner::Gpu(g) if mtp => {
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

fn handle(c: TcpStream, r: &mut Runner, tok: &Tokenizer, e: &Engine, name: &str, addr: &str, i: usize) -> anyhow::Result<()> {
    // A stalled client must not hold the single request slot.
    c.set_read_timeout(Some(Duration::from_secs(30)))?;
    c.set_write_timeout(Some(Duration::from_secs(30)))?;
    let mut w = c.try_clone()?;
    let serve::Request { method, path, host, json, body } = serve::read_request(&mut BufReader::new(c))?;
    let send = |w: &mut TcpStream, status: &str, v: serde_json::Value| -> anyhow::Result<()> {
        let b = v.to_string();
        Ok(write!(w, "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len())?)
    };
    let created = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_secs();
    // Browsers can reach a loopback server: the Host check stops DNS rebinding, and requiring a JSON content type
    // makes a cross-site POST need a CORS preflight, which is never answered.
    if !serve::host_ok(&host, addr) { return send(&mut w, "403 Forbidden", json!({"error": {"message": "bad Host header", "type": "forbidden"}})); }
    if method == "POST" && !json { return send(&mut w, "415 Unsupported Media Type", json!({"error": {"message": "Content-Type must be application/json", "type": "invalid_request_error"}})); }
    match (method.as_str(), path.as_str()) {
        ("GET", "/v1/models") => send(&mut w, "200 OK", json!({"object": "list", "data": [{"id": name, "object": "model", "created": created, "owned_by": "strata"}]})),
        ("POST", "/v1/chat/completions") => {
            let req = match serve::chat_request(&body) {
                Ok(q) => q,
                Err(err) => return send(&mut w, "400 Bad Request", json!({"error": {"message": format!("{err:#}"), "type": "invalid_request_error"}})),
            };
            let ids = tok.encode(&serve::chat_prompt(&req.messages, req.think))?;
            let n = room(e, ids.len(), req.max_tokens.unwrap_or(usize::MAX));
            if n == 0 { return send(&mut w, "400 Bad Request", json!({"error": {"message": format!("prompt of {} tokens fills the context", ids.len()), "type": "invalid_request_error"}})); }
            r.reset()?;
            let logits = r.prefill(&ids)?;
            let id = format!("chatcmpl-{i}");
            let chunk = |delta: serde_json::Value, finish: Option<&str>| json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
            let (out, stop, text) = if req.stream {
                write!(w, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")?;
                write!(w, "data: {}\n\n", chunk(json!({"role": "assistant", "content": ""}), None))?;
                let (out, stop) = decode(r, tok, e.mtp > 0, logits, n, &mut |t| Ok(write!(w, "data: {}\n\n", chunk(json!({"content": t}), None))?))?;
                (out, stop, String::new())
            } else {
                let mut text = String::new();
                let (out, stop) = decode(r, tok, e.mtp > 0, logits, n, &mut |t| { text += t; Ok(()) })?;
                (out, stop, text)
            };
            let finish = if stop { "stop" } else { "length" };
            let usage = json!({"prompt_tokens": ids.len(), "completion_tokens": out.len(), "total_tokens": ids.len() + out.len()});
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
