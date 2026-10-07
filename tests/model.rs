mod common;
use rusting_strata::{gguf::Gguf, model::{Model, State}, ops, tokenizer::Tokenizer};

#[test]
fn greedy_continuation_is_sane() {
    let Some(p) = common::model_path() else { return };
    let g = Gguf::open(&p).unwrap();
    let m = Model::load(&g).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let ids = tok.encode("The capital of France is").unwrap();
    let mut st = State::new(&m.cfg);
    let mut logits = vec![];
    let t0 = std::time::Instant::now();
    for &t in &ids { logits = m.step(&mut st, t); }
    let next = ops::argmax(&logits) as u32;
    eprintln!("{} tokens in {:?}; next = {next} {:?}", ids.len(), t0.elapsed(), tok.decode(&[next]).unwrap());
    assert_eq!(tok.decode(&[next]).unwrap().trim(), "Paris");
}

/// MTP draft acceptance on the CPU reference: the nextn block's greedy draft for position p + 2 against the main
/// model's greedy token over 24 generated tokens. (#19: 24/25; feeding the post-output_norm state instead gave the same.)
#[test]
#[ignore]
fn mtp_draft_acceptance() {
    use rusting_strata::model::Mtp;
    let Some(p) = common::model_path() else { return };
    let g = Gguf::open(&p).unwrap();
    let m = Model::load(&g).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let mut ids = tok.encode("Write a short story about a dragon.").unwrap();
    let n_prompt = ids.len();
    let (mut st, mut mtp, mut drafts) = (State::new(&m.cfg), Mtp::default(), vec![]);
    for i in 0..n_prompt + 24 {
        let (logits, x) = m.step_h(&mut st, ids[i]);
        if i + 1 == ids.len() { ids.push(ops::argmax(&logits) as u32); }
        drafts.push(ops::argmax(&m.mtp_step(&mut mtp, &x, ids[i + 1]).0) as u32);
    }
    // drafts[p] predicts ids[p + 2]; score generated tokens only.
    let hits: Vec<bool> = (n_prompt - 2..drafts.len() - 1).map(|p| drafts[p] == ids[p + 2]).collect();
    let n_ok = hits.iter().filter(|&&b| b).count();
    eprintln!("MTP acceptance {n_ok}/{} on {:?}", hits.len(), tok.decode(&ids[n_prompt..]).unwrap());
    assert!(n_ok * 10 >= hits.len() * 6, "MTP acceptance {n_ok}/{}", hits.len());
}
