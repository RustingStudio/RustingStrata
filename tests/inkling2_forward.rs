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

/// Both exports of the model describe the same 24 blocks whose core 12 run three times: the legacy file
/// (`inkling2`) puts the 48 layer runs in `block_count`, the current one (`inkling2_loop`) the 24 blocks.
#[test]
fn config_counts_physical_blocks() {
    let Some(g) = open() else { return };
    let cfg = rusting_strata::inkling2::Inkling2Config::from_gguf(&g).unwrap();
    assert_eq!(cfg.n_layer, 24, "physical blocks");
    assert_eq!(cfg.n_slot(), 48, "layer runs a token");
    assert_eq!(cfg.swa.len(), 24);
    assert_eq!((cfg.loop_start, cfg.loop_end, cfg.loop_count), (6, 18, 3));
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

/// A draft run verifies its rows through the GEMM and the baseline steps through the GEMV, and the expert cache is in
/// another state, so the logits differ a little (see `chunked_prefill_matches_token_by_token`). The tokens must be
/// the baseline's up to the first step where the baseline's two candidates are closer than that noise. A rollback
/// that leaves a rejected draft in the state moves the logits by 10 or more.
fn same_up_to_tie(what: &str, got: &[u32], want: &[u32], logits: &[Vec<f32>]) {
    let Some(i) = (0..want.len().min(got.len())).find(|&i| got[i] != want[i]) else {
        assert_eq!(got.len(), want.len(), "{what}: the run stops at another step than the baseline");
        return;
    };
    let gap = logits[i][want[i] as usize] - logits[i][got[i] as usize];
    eprintln!("{what}: token {i} differs from the baseline at a logit gap of {gap:.4}");
    assert!(gap < 1.0, "{what}: token {i} differs from the baseline at a logit gap of {gap}");
}

/// Greedy decode of `n_gen` tokens through `spec_step_argmax`.
fn spec_run(m: &mut Inkling2Model, ids: &[u32], n_gen: usize, eos: &[u32]) -> Vec<u32> {
    let mut next = rusting_strata::ops::argmax(&m.prefill(ids)) as u32;
    let (mut got, mut pending) = (vec![], std::collections::VecDeque::<u32>::new());
    for i in 0..n_gen {
        if eos.contains(&next) { break; }
        got.push(next);
        if i + 1 < n_gen {
            if pending.is_empty() { pending.extend(m.spec_step_argmax(next, n_gen - (i + 1), eos)); }
            next = pending.pop_front().unwrap();
        }
    }
    got
}

/// Drafts need the GPU path (a draft batch is verified in one pass there), so this runs with `STRATA_DS_GPU=1`.
/// `STRATA_INKLING2_DSPARK` (path to a DSpark draft file) adds the draft-model part.
#[test]
fn speculative_ngram_and_dspark_match_greedy() {
    let Some(g) = open() else { return };
    if std::env::var("STRATA_DS_GPU").as_deref() != Ok("1") { eprintln!("STRATA_DS_GPU is not 1, skipping"); return; }
    let _lock = common::gpu_lock();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let eos = tok.eos_ids();
    let ids = tok.encode(&format!("{HEAD}Repeat this exact sentence three times: The quick brown fox jumps over the lazy dog.<|end_message|>")).unwrap();
    let n_gen = 32usize;

    let mut base = Inkling2Model::new_with_ctx(&g, 2048).unwrap();
    let (mut want, mut logits) = (vec![], vec![base.prefill(&ids)]);
    for i in 0..n_gen {
        let next = rusting_strata::ops::argmax(&logits[i]) as u32;
        if eos.contains(&next) { break; }
        want.push(next);
        if i + 1 < n_gen { logits.push(base.step(next)); }
    }
    drop(base);

    // n-grams of 3 tokens: the repeated sentence gives drafts, and most are rejected, so the rollback runs.
    let mut ng_cfg = rusting_strata::ngram::NgramConfig::default();
    ng_cfg.types = vec![rusting_strata::ngram::SpecType::NgramSimple];
    ng_cfg.size_n = 3;
    ng_cfg.draft_max = 8;
    ng_cfg.n_vocab = 200058;
    let mut m = Inkling2Model::new_with_ctx(&g, 2048).unwrap();
    m.ngram = Some(rusting_strata::ngram::NgramDrafter::new(ng_cfg, Some(&tok)).unwrap());
    let got = spec_run(&mut m, &ids, n_gen, &eos);
    eprintln!("ngram-simple drafts: accepted {}/{}", m.drafts.1, m.drafts.0);
    assert!(m.drafts.0 > 0, "ngram-simple proposed no drafts on a repeated sentence");
    same_up_to_tie("ngram-simple", &got, &want, &logits);
    drop(m);

    let Ok(dspark) = std::env::var("STRATA_INKLING2_DSPARK") else { return };
    let opts = rusting_strata::inkling2::chat::Inkling2Opts { max_ctx: 2048, draft_model: Some(dspark.into()), ..Default::default() };
    let mut m = opts.build_model(&g, &tok).unwrap();
    let got = spec_run(&mut m, &ids, n_gen, &eos);
    eprintln!("DSpark drafts: accepted {}/{}", m.drafts.1, m.drafts.0);
    assert!(m.drafts.0 > 0, "DSpark proposed no drafts");
    same_up_to_tie("DSpark", &got, &want, &logits);
}
