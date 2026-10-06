use clap::{Args, Parser, Subcommand};
use anyhow::Context;
use rusting_strata::model::{Model, State};
use rusting_strata::{cuda::Gpu, gpu_model::GpuModel};
use rusting_strata::{config::ModelConfig, gguf::Gguf, ops, quant::GgmlType, tokenizer::Tokenizer};
use std::io::{Read, Write};
use std::{collections::BTreeMap, path::PathBuf, time::Instant};

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
    #[arg(long, default_value_t = 1024)]
    chunk: usize,
    /// VRAM left free after placing expert layers, in MB.
    #[arg(long, default_value_t = 512)]
    vram_margin: usize,
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
    /// Seconds spent waiting on CPU experts so far (GPU only).
    fn cpu_secs(&self) -> f64 { if let Runner::Gpu(g) = self { g.cpu_time.as_secs_f64() } else { 0.0 } }
    fn finish(&self) -> anyhow::Result<()> {
        if let Runner::Gpu(g) = self {
            let n = (g.hits + g.misses).max(1);
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
        Ok(Runner::Gpu(Box::new(GpuModel::load(m, Gpu::new()?, self.ctx, self.chunk, self.vram_margin << 20)?)))
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
    /// Greedy generation.
    Generate {
        #[arg(long)]
        model: PathBuf,
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
            let eos = tok.eos_ids();
            let mut r = engine.runner(&m)?;
            let ids = tok.encode(&prompt)?;
            let t0 = Instant::now();
            let mut logits = r.prefill(&ids)?;
            let t1 = Instant::now();
            let cpu0 = r.cpu_secs();
            let mut out = Vec::new();
            let mut printed = 0;
            for _ in 0..n {
                let next = ops::argmax(&logits) as u32;
                if eos.contains(&next) { break; }
                out.push(next);
                let s = tok.decode(&out)?;
                if !s.ends_with('\u{fffd}') {
                    print!("{}", &s[printed..]);
                    std::io::stdout().flush()?;
                    printed = s.len();
                }
                logits = r.step(next)?;
            }
            let (pp, tg) = ((t1 - t0).as_secs_f64(), t1.elapsed().as_secs_f64());
            eprintln!("\nprompt {} tok {:.2} tok/s | gen {} tok {:.2} tok/s", ids.len(), ids.len() as f64 / pp, out.len(), out.len() as f64 / tg);
            if engine.gpu { eprintln!("CPU expert compute, share of gen time: {:.1}%", 100.0 * (r.cpu_secs() - cpu0) / tg); }
            r.finish()?;
        }
    }
    Ok(())
}
