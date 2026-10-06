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
        let mut gm = GpuModel::load(&m, gpu, 256, margin).unwrap();
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
