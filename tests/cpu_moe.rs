mod common;
use rusting_strata::{cpu_moe, gguf::Gguf, model::Model};

/// Integer Q8_K expert path matches the f32 reference within activation-quantization error, and is timed.
#[test]
fn fast_experts_match_reference() {
    let Some(p) = common::model_path() else { return };
    let g = Gguf::open(&p).unwrap();
    let m = Model::load(&g).unwrap();
    if m.cfg.n_expert == 0 { return eprintln!("SKIP: dense model"); }
    let h: Vec<f32> = (0..m.cfg.n_embd).map(|i| ((i as f32 * 0.7548777).fract() - 0.5) * 2.0).collect();
    for il in [0, 17, m.cfg.n_layer - 1] {
        let sel = m.route(il, &h);
        let want = m.routed_experts(il, &h);
        let mut got = cpu_moe::experts(&m, il, &h, &sel);
        let t0 = std::time::Instant::now();
        for _ in 0..10 { got = cpu_moe::experts(&m, il, &h, &sel); }
        let dt = t0.elapsed() / 10;
        let num: f32 = got.iter().zip(&want).map(|(a, b)| (a - b) * (a - b)).sum();
        let den: f32 = want.iter().map(|b| b * b).sum();
        let e = (num / den).sqrt();
        eprintln!("layer {il}: rel err {e:.2e}, {:?} for {} experts", dt, sel.len());
        assert!(e < 2e-2, "layer {il}: rel err {e}");
    }
}

/// A verify batch gives every token the bits of its own `experts` call, for tokens that share experts and not.
#[test]
fn experts_multi_matches_experts() {
    let Some(p) = common::model_path() else { return };
    let g = Gguf::open(&p).unwrap();
    let m = Model::load(&g).unwrap();
    if m.cfg.n_expert == 0 { return eprintln!("SKIP: dense model"); }
    let hs: Vec<Vec<f32>> = (0..3).map(|t| (0..m.cfg.n_embd).map(|i| ((i * 13 + t * 7) % 101) as f32 / 50.0 - 1.0).collect()).collect();
    for il in [0, m.cfg.n_layer - 1] {
        // Token 2 routes like token 0; one token with nothing missed.
        let sels = vec![m.route(il, &hs[0]), m.route(il, &hs[1]), m.route(il, &hs[0])];
        let want: Vec<Vec<f32>> = hs.iter().zip(&sels).map(|(h, s)| cpu_moe::experts(&m, il, h, s)).collect();
        assert_eq!(cpu_moe::experts_multi(&m, il, &hs, &sels), want, "layer {il}");
        let none = vec![sels[0].clone(), vec![], sels[1].clone()];
        let got = cpu_moe::experts_multi(&m, il, &hs, &none);
        assert_eq!(got[0], want[0], "layer {il}");
        assert!(got[1].iter().all(|&v| v == 0.0), "layer {il}");
        assert_eq!(got[2], cpu_moe::experts(&m, il, &hs[2], &sels[1]), "layer {il}");
    }
}

/// Single-thread kernel throughput (prints only).
#[test]
fn kernel_throughput() {
    let Some(p) = common::model_path() else { return };
    let g = Gguf::open(&p).unwrap();
    let m = Model::load(&g).unwrap();
    if m.cfg.n_expert == 0 { return eprintln!("SKIP: dense model"); }
    let last = format!("blk.{}.ffn_down_exps.weight", m.cfg.n_layer - 1);
    for name in ["blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight", &last] {
        let t = m.w.get(name).unwrap().expert(5, m.cfg.n_expert);
        let x: Vec<f32> = (0..t.cols).map(|i| ((i as f32 * 0.7548777).fract() - 0.5) * 2.0).collect();
        let mut y = vec![0f32; t.rows];
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        pool.install(|| for _ in 0..3 { cpu_moe::matvec(&t, &x, &mut y) });
        let t0 = std::time::Instant::now();
        pool.install(|| for _ in 0..50 { cpu_moe::matvec(&t, &x, &mut y) });
        let s = t0.elapsed().as_secs_f64() / 50.0;
        eprintln!("{name} {}: {:.1} us, {:.2} Gweights/s", t.ty.name(), s * 1e6, (t.rows * t.cols) as f64 / s / 1e9);
    }
}
