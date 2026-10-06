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
