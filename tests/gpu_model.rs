mod common;
use rusting_strata::{cuda::Gpu, gguf::Gguf, gpu_model::GpuModel, model::{Model, State}, ops, tokenizer::Tokenizer};

/// GPU forward matches the CPU reference per position, with all routed experts on the CPU and with the default placement.
#[test]
fn gpu_forward_matches_cpu_reference() {
    let Some(p) = common::model_path() else { return };
    let g = Gguf::open(&p).unwrap();
    let m = Model::load(&g).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let ids = tok.encode("The capital of France is").unwrap();
    let mut st = State::new(&m.cfg);
    let want: Vec<Vec<f32>> = ids.iter().map(|&t| m.step(&mut st, t)).collect();

    for margin in [usize::MAX / 2, 512 << 20] {
        let gpu = match Gpu::new() {
            Ok(g) => g,
            Err(e) if format!("{e}") == "CUDA init" => { eprintln!("SKIP: no GPU: {e:#}"); return }
            Err(e) => panic!("{e:#}"),
        };
        let mut gm = GpuModel::load(&m, gpu, 256, 0, margin).unwrap();
        let t0 = std::time::Instant::now();
        for (i, (&t, w)) in ids.iter().zip(&want).enumerate() {
            let got = gm.step(t).unwrap();
            let range = w.iter().fold(f32::MIN, |a, &b| a.max(b)) - w.iter().fold(f32::MAX, |a, &b| a.min(b));
            let diff = got.iter().zip(w).fold(0f32, |a, (x, y)| a.max((x - y).abs()));
            eprintln!("pos {i}: argmax {} vs {}, max |dlogit| {diff:.4} (range {range:.2})", ops::argmax(&got), ops::argmax(w));
            assert_eq!(ops::argmax(&got), ops::argmax(w), "pos {i}");
            assert!(diff <= 0.05 * range, "pos {i}: {diff} vs range {range}");
        }
        eprintln!("{} tokens in {:?} (CPU {:?}), expert hits {} misses {}", ids.len(), t0.elapsed(), gm.cpu_time, gm.hits, gm.misses);
    }
    assert_eq!(tok.decode(&[ops::argmax(want.last().unwrap()) as u32]).unwrap().trim(), "Paris");
}

/// Chunked prefill (several chunks, tail chunk shorter) matches the CPU f32 reference at the last prompt token and at
/// the decode step after it. Token-by-token GPU decode is the yardstick for the error it may have: its int8
/// activations drift further from f32 over a long prompt than the f16 prefill GEMMs do.
#[test]
fn prefill_matches_reference() {
    let Some(p) = common::model_path() else { return };
    let g = Gguf::open(&p).unwrap();
    let m = Model::load(&g).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let text = "Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, type safety, \
                and concurrency. It enforces memory safety, meaning that all references point to valid memory, without \
                a garbage collector. To simultaneously enforce memory safety and prevent data races, its borrow checker \
                tracks the object lifetime of all references in a program during compilation. Rust was influenced by";
    let ids = tok.encode(text).unwrap();
    assert!(ids.len() > 64, "{} tokens", ids.len());
    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => { eprintln!("SKIP: no GPU: {e:#}"); return }
    };
    let mut st = State::new(&m.cfg);
    let mut cpu = Vec::new();
    for &t in &ids { cpu = m.step(&mut st, t); }
    let next = ops::argmax(&cpu) as u32;
    let cpu2 = m.step(&mut st, next);
    let chunk = std::env::var("CHUNK").map_or(48, |v| v.parse().unwrap());
    let mut gm = GpuModel::load(&m, gpu, 512, chunk, 512 << 20).unwrap();
    let t0 = std::time::Instant::now();
    let pre = gm.prefill(&ids).unwrap();
    eprintln!("prefill {} tokens in {:?}", ids.len(), t0.elapsed());
    let pre2 = gm.step(next).unwrap();
    gm.reset().unwrap();
    let mut dec = Vec::new();
    for &t in &ids { dec = gm.step(t).unwrap(); }
    let dec2 = gm.step(next).unwrap();
    let diff = |a: &[f32], b: &[f32]| a.iter().zip(b).fold(0f32, |x, (p, q)| x.max((p - q).abs()));
    for (name, pf, dc, want) in [("last prompt token", &pre, &dec, &cpu), ("next step", &pre2, &dec2, &cpu2)] {
        let (dp, dd) = (diff(pf, want), diff(dc, want));
        eprintln!("{name}: argmax prefill {} decode {} CPU {}, max |dlogit| prefill {dp:.4} decode {dd:.4}",
                  ops::argmax(pf), ops::argmax(dc), ops::argmax(want));
        assert_eq!(ops::argmax(pf), ops::argmax(want), "{name}");
        assert!(dp <= dd.max(0.5), "{name}: prefill error {dp} above decode error {dd}");
    }
}
