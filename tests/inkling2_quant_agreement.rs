//! How far a smaller file of the model drifts from a bigger one: the same prompt decoded greedily by both, on the CPU
//! reference path (deterministic, unlike the GPU one where two runs of one binary already differ).
//!
//! `STRATA_INKLING2_MODEL` is the file under test and `STRATA_INKLING2_REF` the reference (the bigger quant). The
//! llama.cpp parity tokens in `inkling2_forward` are IQ4_NL's, so they say nothing about another quant; this does.
mod common;
use rusting_strata::{gguf::Gguf, inkling2::forward::Inkling2Model, ops::argmax, tokenizer::Tokenizer};

const HEAD: &str = "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>";

fn open(var: &str) -> Option<(Gguf, String)> {
    let p = std::env::var(var).unwrap_or_else(|_| { eprintln!("{var} unset, skipping"); String::new() });
    if p.is_empty() { return None; }
    Some((Gguf::open(std::path::Path::new(&p)).unwrap(), p))
}

/// KL(reference || test) over the vocabulary, in nats.
fn kl(ref_l: &[f32], test_l: &[f32]) -> f64 {
    let sm = |v: &[f32]| -> Vec<f64> {
        let m = v.iter().fold(f32::MIN, |a, &b| a.max(b)) as f64;
        let e: Vec<f64> = v.iter().map(|&x| ((x as f64) - m).exp()).collect();
        let z: f64 = e.iter().sum();
        e.iter().map(|x| x / z).collect()
    };
    let (p, q) = (sm(ref_l), sm(test_l));
    p.iter().zip(&q).filter(|(a, _)| **a > 1e-9).map(|(a, b)| a * (a / b.max(1e-12)).ln()).sum()
}

/// Greedy continuation of `n` tokens plus the logits each step chose from.
fn greedy(m: &mut Inkling2Model, ids: &[u32], n: usize) -> (Vec<u32>, Vec<Vec<f32>>) {
    let mut logits = m.prefill(ids);
    let (mut toks, mut all) = (vec![], vec![]);
    for i in 0..n {
        let t = argmax(&logits) as u32;
        all.push(logits.clone());
        toks.push(t);
        if i + 1 < n { logits = m.step(t); }
    }
    (toks, all)
}

#[test]
fn smaller_quant_follows_the_reference() {
    let Some((g_test, name_test)) = open("STRATA_INKLING2_MODEL") else { return };
    let Some((g_ref, name_ref)) = open("STRATA_INKLING2_REF") else { return };
    let _lock = common::gpu_lock();
    let tok = Tokenizer::from_gguf(&g_test).unwrap();
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tools/prompts/bench-natural.txt")).unwrap();
    // A question, a summary and a rewrite: the shapes a smaller quant usually loses first.
    let prompts = [
        format!("{HEAD}What is the capital of France?<|end_message|>"),
        format!("{HEAD}{}\n\nSummarize the text above in three sentences.<|end_message|>", text.trim()),
        format!("{HEAD}Rewrite this sentence in the passive voice: The committee approved the proposal.<|end_message|>"),
    ];
    let n = 24usize;
    let (mut same, mut total, mut worst_kl, mut kl_sum, mut kl_n) = (0usize, 0usize, 0f64, 0f64, 0usize);
    for (pi, p) in prompts.iter().enumerate() {
        let ids = tok.encode(p).unwrap();
        let (want, want_l) = greedy(&mut Inkling2Model::new(&g_ref).unwrap(), &ids, n);
        let (got, got_l) = greedy(&mut Inkling2Model::new(&g_test).unwrap(), &ids, n);
        let agree = got.iter().zip(&want).filter(|(a, b)| a == b).count();
        // KL is only meaningful while both models stand on the same context: once a token differs, every later step
        // compares distributions over different text. So the KL stops at the first mismatch.
        let keep = (0..n).take_while(|&i| got[i] == want[i]).count();
        let k: Vec<f64> = (0..keep).map(|i| kl(&want_l[i], &got_l[i])).collect();
        let mean = if k.is_empty() { 0.0 } else { k.iter().sum::<f64>() / k.len() as f64 };
        eprintln!("prompt {pi}: {agree}/{n} tokens match, mean KL {mean:.4} nats over the {keep} steps before the first \
                   mismatch, worst {:.4}", k.iter().fold(0f64, |a, &b| a.max(b)));
        same += agree;
        total += n;
        kl_sum += k.iter().sum::<f64>();
        kl_n += k.len();
        worst_kl = worst_kl.max(k.iter().fold(0f64, |a, &b| a.max(b)));
    }
    let mean_kl = if kl_n == 0 { 0.0 } else { kl_sum / kl_n as f64 };
    eprintln!("{name_test} against {name_ref}: {same}/{total} tokens match ({:.1}%), mean KL {mean_kl:.4}, worst {worst_kl:.4}",
        100.0 * same as f64 / total as f64);
    // Unsloth's own table puts UD-Q2_K_XL at 91.6% of the top tokens of UD-Q8_K_XL (0.0595 KLD). Against IQ4_XS the
    // gap is smaller. A broken kernel or a misread quant shows up far below this.
    assert!(same as f64 / total as f64 >= 0.85, "only {same}/{total} tokens follow {name_ref}");
    assert!(mean_kl < 0.10, "mean KL {mean_kl:.4} nats against {name_ref}");
}
