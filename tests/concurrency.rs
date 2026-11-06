//! Four sequences alternate on one model: prefill each, park it, then run one token per sequence per round and
//! check the tokens match generating them one at a time. This is what the server's concurrent mode does (see
//! `GpuModel::park_current` / `adopt`), so it also times the state switches.
mod common;
use rusting_strata::{gguf::Gguf, gpu_model::GpuModel, model::Model, ops, tokenizer::Tokenizer};
use std::time::Instant;

const N: usize = 16;

fn argmax_t(logits: &[f32]) -> u32 { ops::argmax(logits) as u32 }

#[test]
fn four_sequences_interleave() {
    let _gpu = common::gpu_lock();
    // Same expert placement in both runs, so near-ties cannot pick a different token.
    unsafe { std::env::set_var("STRATA_FIXED_EXPERTS", "1") }
    let Some(p) = common::model_path() else { return };
    let g = Gguf::open(&p).unwrap();
    let m = Model::load(&g).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let prompts = [
        "The capital of France is",
        "Write a haiku about rivers:",
        "In Rust, a Vec is",
        "The quick brown fox jumps over",
    ];
    let ids: Vec<Vec<u32>> = prompts.iter().map(|s| tok.encode(s).unwrap()).collect();
    let open = || {
        let mut g = GpuModel::load(&m, Box::new(rusting_strata::cuda::Gpu::new(0).unwrap()), 2048, 512, 512 << 20, 0).unwrap();
        g.cache_ram = 1 << 30;
        g
    };

    // Yardstick: each sequence alone, greedy.
    let mut want = Vec::new();
    let mut gm = open();
    let mut one = 0.0;
    for id in &ids {
        gm.reset().unwrap();
        let mut logits = gm.prefill(id).unwrap();
        let mut out = Vec::new();
        let t0 = Instant::now();
        for _ in 0..N {
            let t = argmax_t(&logits);
            out.push(t);
            logits = gm.step(t).unwrap();
        }
        one += t0.elapsed().as_secs_f64();
        want.push(out);
    }
    drop(gm);

    // Interleaved: prefill and park every sequence, then `q` tokens each per round.
    for q in [1usize, 4, 8] {
        let mut gm = open();
        let (mut logits, mut hist, mut got) = (Vec::new(), ids.clone(), vec![Vec::new(); ids.len()]);
        for id in &ids {
            gm.reset().unwrap();
            logits.push(gm.prefill(id).unwrap());
            gm.park_current(id).unwrap();
        }
        let mut swaps = 0;
        let t1 = Instant::now();
        while got[0].len() < N {
            for i in 0..ids.len() {
                gm.adopt(&hist[i]).unwrap();
                for _ in 0..q {
                    if got[i].len() == N { break; }
                    let t = argmax_t(&logits[i]);
                    got[i].push(t);
                    hist[i].push(t);
                    logits[i] = gm.step(t).unwrap();
                }
                gm.park_current(&hist[i]).unwrap();
                swaps += 2;
            }
        }
        let inter = t1.elapsed().as_secs_f64();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g, w, "sequence {i}, q {q}");
        }
        let toks = (ids.len() * N) as f64;
        eprintln!("q {q:>2}: {:.2} tok/s total, {:.2} tok/s per sequence ({swaps} swaps, {:.2} ms each)",
            toks / inter, toks / inter / ids.len() as f64, 1e3 * inter / swaps as f64);
    }
    eprintln!("one at a time: {:.2} tok/s", (ids.len() * N) as f64 / one);
}
