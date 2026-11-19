//! Gated by STRATA_HY4_MODEL (path to shard 1 of Hy4-flash). Skips when unset.
use rusting_strata::{gguf::Gguf, hy4::forward::Hy4Model, tokenizer::Tokenizer};

#[test]
fn first_token_after_france_prompt_is_paris() {
    let Ok(p) = std::env::var("STRATA_HY4_MODEL") else { eprintln!("STRATA_HY4_MODEL unset, skipping"); return };
    let g = Gguf::open(std::path::Path::new(&p)).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let mut m = Hy4Model::new(&g).unwrap();
    let logits = m.prefill(&tok.encode("The capital of France is").unwrap());
    let top = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
    assert_eq!(tok.decode(&[top]).unwrap(), " Paris", "llama.cpp greedy picks ' Paris'");
}

/// The CPU path (STRATA_DS_GPU unset) runs the same kernels either way, so the logits agree closely.
#[test]
fn chunked_prefill_matches_token_by_token() {
    let Ok(p) = std::env::var("STRATA_HY4_MODEL") else { eprintln!("STRATA_HY4_MODEL unset, skipping"); return };
    let g = Gguf::open(std::path::Path::new(&p)).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    // More tokens than the widest layer has heads, so a chunk's per-head projections run too.
    let toks = tok.encode(&"The capital of France is Paris. ".repeat(12)).unwrap();
    assert!(toks.len() > 65, "{} tokens", toks.len());
    let (mut a, mut b) = (Hy4Model::new(&g).unwrap(), Hy4Model::new(&g).unwrap());
    let chunked = a.prefill(&toks);
    let mut stepped = vec![];
    for &t in &toks { stepped = b.step(t); }
    let worst = chunked.iter().zip(&stepped).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    assert!(worst < 1e-3, "chunked prefill logits differ by {worst}");
}
