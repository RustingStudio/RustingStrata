use clap::{Parser, Subcommand};
use anyhow::Context;
use rusting_strata::model::{Model, State};
use rusting_strata::{config::ModelConfig, gguf::Gguf, ops, quant::GgmlType, tokenizer::Tokenizer};
use std::io::{Read, Write};
use std::{collections::BTreeMap, path::PathBuf, time::Instant};

#[derive(Parser)]
#[command(name = "strata", about = "Rust inference engine for Qwen3.5/3.6-family GGUF models")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
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
    },
    /// Greedy generation with the CPU reference forward.
    Generate {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(short, default_value_t = 32)]
        n: usize,
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
        Cmd::Score { model, input } => {
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
            let mut st = State::new(&m.cfg);
            let feed: Vec<u32> = prompt.iter().chain(&cont[..cont.len() - 1]).copied().collect();
            let (mut argmax, mut top5) = (Vec::new(), Vec::new());
            for (i, &t) in feed.iter().enumerate() {
                let logits = m.step(&mut st, t);
                if i + 1 < prompt.len() { continue; }
                let lse = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let lse = lse + logits.iter().map(|x| (x - lse).exp()).sum::<f32>().ln();
                argmax.push(ops::argmax(&logits));
                top5.push(ops::top_k(&logits, 5).into_iter().map(|j| (j, logits[j] - lse)).collect::<Vec<_>>());
            }
            println!("{}", serde_json::json!({"argmax": argmax, "top5": top5}));
        }
        Cmd::Generate { model, prompt, n } => {
            let g = Gguf::open(&model)?;
            let m = Model::load(&g)?;
            let tok = Tokenizer::from_gguf(&g)?;
            let eos = tok.eos_ids();
            let mut st = State::new(&m.cfg);
            let ids = tok.encode(&prompt)?;
            let t0 = Instant::now();
            let mut logits = Vec::new();
            for &t in &ids { logits = m.step(&mut st, t); }
            let t1 = Instant::now();
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
                logits = m.step(&mut st, next);
            }
            let (pp, tg) = ((t1 - t0).as_secs_f64(), t1.elapsed().as_secs_f64());
            eprintln!("\nprompt {} tok {:.2} tok/s | gen {} tok {:.2} tok/s", ids.len(), ids.len() as f64 / pp, out.len(), out.len() as f64 / tg);
        }
    }
    Ok(())
}
