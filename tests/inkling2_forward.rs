//! Gated by STRATA_INKLING2_MODEL (path to Inkling-2-Mini-IQ4_NL.gguf). Skips when unset.
//! `STRATA_DS_GPU=1` (with `STRATA_DS_BACKEND=cuda` or `vulkan`) runs the same tests through the GPU matvecs.
//! The reference tokens are llama.cpp's greedy continuations (commit c622f62, CPU, `/completion` at temperature 0).
mod common;
use rusting_strata::{gguf::Gguf, inkling2::forward::Inkling2Model, tokenizer::Tokenizer};

const HEAD: &str = "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>";
const SHORT: [u32; 24] = [200001, 200008, 371, 150409, 171960, 149878, 150744, 151231, 1355, 150377, 25, 151451, 149879, 151194, 13, 170289, 150893, 169411, 13, 151194, 6, 82, 151451, 149884];
const LONG: [u32; 48] = [200001, 200008, 371, 150409, 171960, 25, 196980, 31488, 928, 149876, 158107, 150749, 149881, 152703, 152199, 167608, 13, 68014, 31488, 928, 149876, 150311, 150749, 149881, 150396, 167608, 13, 1, 150939, 149914, 150103, 155985, 279, 157823, 25, 152703, 152199, 167608, 149880, 150396, 167608, 13, 170289, 154573, 13, 150197, 150201, 150893];

fn open() -> Option<Gguf> {
    let Ok(p) = std::env::var("STRATA_INKLING2_MODEL") else { eprintln!("STRATA_INKLING2_MODEL unset, skipping"); return None };
    Some(Gguf::open(std::path::Path::new(&p)).unwrap())
}

/// Feed llama.cpp's tokens after the prompt and count the steps whose top token is llama.cpp's.
fn agree(g: &Gguf, prompt: &str, want: &[u32]) -> usize {
    let tok = Tokenizer::from_gguf(g).unwrap();
    let mut m = Inkling2Model::new(g).unwrap();
    let mut logits = m.prefill(&tok.encode(prompt).unwrap());
    let mut ok = 0;
    for (i, &w) in want.iter().enumerate() {
        let top = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        if top == w as usize { ok += 1; } else { eprintln!("step {i}: top {top} ({:.3}), llama.cpp {w} ({:.3})", logits[top], logits[w as usize]); }
        if i + 1 < want.len() { logits = m.step(w); }
    }
    ok
}

#[test]
fn short_prompt_follows_llama_cpp() {
    let Some(g) = open() else { return };
    let _lock = common::gpu_lock();
    let ok = agree(&g, &format!("{HEAD}What is the capital of France?<|end_message|>"), &SHORT);
    assert_eq!(ok, SHORT.len(), "top token differs from llama.cpp");
}

/// A prompt of several hundred tokens (chunked prefill, every sliding window partly full). Its continuation has
/// near ties, so one step may differ.
#[test]
fn long_prompt_follows_llama_cpp() {
    let Some(g) = open() else { return };
    let _lock = common::gpu_lock();
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tools/prompts/bench-natural.txt")).unwrap();
    let ok = agree(&g, &format!("{HEAD}{}\n\nSummarize the text above in three sentences.<|end_message|>", text.trim()), &LONG);
    assert!(ok + 1 >= LONG.len(), "{ok} of {} top tokens are llama.cpp's", LONG.len());
}

/// On the CPU a chunk runs the same kernels as single tokens, so the logits agree closely. On the GPU a chunk runs
/// through the tensor-core GEMM and one token through the f32 GEMV, and an expert runs on the GPU or on the CPU
/// (integer dots) as the cache decides, so two runs of the same tokens already differ by about 1 in the worst logit.
#[test]
fn chunked_prefill_matches_token_by_token() {
    let Some(g) = open() else { return };
    let _lock = common::gpu_lock();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let toks = tok.encode(&"The capital of France is Paris. ".repeat(12)).unwrap();
    let (mut a, mut b) = (Inkling2Model::new(&g).unwrap(), Inkling2Model::new(&g).unwrap());
    let chunked = a.prefill(&toks);
    let mut stepped = vec![];
    for &t in &toks { stepped = b.step(t); }
    let worst = chunked.iter().zip(&stepped).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let limit = if std::env::var("STRATA_DS_GPU").as_deref() == Ok("1") { 3.0 } else { 1e-3 };
    assert!(worst < limit, "chunked prefill logits differ by {worst}");
}
