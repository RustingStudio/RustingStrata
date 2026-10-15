mod common;
use rusting_strata::{cuda::Gpu, gguf::Gguf, gpu_model::GpuModel, model::{Model, State}, ops, tokenizer::Tokenizer};

/// GPU forward matches the CPU reference per position, with all routed experts on the CPU and with the default placement.
#[test]
fn gpu_forward_matches_cpu_reference() {
    let _gpu = common::gpu_lock();
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
        let mut gm = GpuModel::load(&m, Box::new(gpu), 256, 0, margin, 0).unwrap();
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
    let _gpu = common::gpu_lock();
    // Place experts in layer order, not from the routing profile that every earlier run rewrites (with it the
    // decode yardstick, and so this test's outcome, depended on what ran before).
    unsafe { std::env::set_var("STRATA_FIXED_EXPERTS", "1") }
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
    let mut gm = GpuModel::load(&m, Box::new(gpu), 512, chunk, 512 << 20, 0).unwrap();
    // Fixed expert placement: adaptive moves change the decode yardstick from run to run (it differed inside the
    // full suite and failed once).
    gm.adapt_every = 0;
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
        // int8 MMQ prefill GEMMs (3754e65) quantize activations, decode does not: allow 1.5x (seen 1.17x).
        assert!(dp <= (1.5 * dd).max(0.5), "{name}: prefill error {dp} above 1.5x decode error {dd}");
    }
}

/// GPU MTP draft head matches `Model::mtp_step` per position (prompt plus a few greedy tokens). Both get the GPU's
/// hidden state, so the check covers the head alone, not the main model's drift. One position may miss: a near-tie in
/// the head's expert routing sometimes picks a different expert on the two sides (|dlogit| ~1.1, seen on 0991789 too).
#[test]
fn mtp_draft_matches_reference() {
    use rusting_strata::model::Mtp;
    let _gpu = common::gpu_lock();
    let Some(p) = common::model_path() else { return };
    let g = Gguf::open(&p).unwrap();
    let m = Model::load(&g).unwrap();
    if m.cfg.n_nextn == 0 { eprintln!("SKIP: no nextn layer"); return }
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => { eprintln!("SKIP: no GPU: {e:#}"); return }
    };
    let mut gm = GpuModel::load(&m, Box::new(gpu), 256, 0, 512 << 20, 1).unwrap();
    let mut ids = tok.encode("The capital of France is").unwrap();
    let (mut st, mut mtp) = (State::new(&m.cfg), Mtp::default());
    let mut bad = 0;
    for i in 0..ids.len() + 4 {
        let logits = m.step(&mut st, ids[i]);
        gm.step(ids[i]).unwrap();
        if i + 1 == ids.len() { ids.push(ops::argmax(&logits) as u32); }
        let want = m.mtp_step(&mut mtp, &gm.hidden().unwrap(), ids[i + 1]).0;
        let got = gm.mtp_draft(ids[i + 1], 0).unwrap();
        let range = want.iter().fold(f32::MIN, |a, &b| a.max(b)) - want.iter().fold(f32::MAX, |a, &b| a.min(b));
        let diff = got.iter().zip(&want).fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        eprintln!("pos {i}: draft {} vs {}, max |dlogit| {diff:.4} (range {range:.2})", ops::argmax(&got), ops::argmax(&want));
        if ops::argmax(&got) != ops::argmax(&want) || diff > 0.05 * range { bad += 1; }
    }
    assert!(bad <= 1, "{bad} positions differ");
}

/// `verify` on several tokens matches the same tokens stepped one by one after `restore_state`, and the restored
/// state replays bit for bit (rollback for rejected drafts).
#[test]
fn verify_matches_steps() {
    let _gpu = common::gpu_lock();
    let Some(p) = common::model_path() else { return };
    let g = Gguf::open(&p).unwrap();
    let m = Model::load(&g).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => { eprintln!("SKIP: no GPU: {e:#}"); return }
    };
    let mut gm = GpuModel::load(&m, Box::new(gpu), 256, 0, 512 << 20, 0).unwrap();
    let ids = tok.encode("The capital of France is Paris, and the capital of Spain is").unwrap();
    let (pre, rest) = ids.split_at(ids.len() - 4);
    for &t in pre { gm.step(t).unwrap(); }
    gm.save_state().unwrap();
    let mut seq = vec![];
    for &t in rest { seq.push(gm.step(t).unwrap()); }
    gm.restore_state().unwrap();
    assert_eq!(gm.step(rest[0]).unwrap(), seq[0], "replay after restore");
    for nt in [4, 2] {
        gm.restore_state().unwrap();
        let got = gm.verify(&rest[..nt]).unwrap();
        for (t, (g, w)) in got.iter().zip(&seq).enumerate() {
            let range = w.iter().fold(f32::MIN, |a, &b| a.max(b)) - w.iter().fold(f32::MAX, |a, &b| a.min(b));
            let diff = g.iter().zip(w).fold(0f32, |a, (x, y)| a.max((x - y).abs()));
            eprintln!("nt {nt} token {t}: argmax {} vs {}, max |dlogit| {diff:.4} (range {range:.2})", ops::argmax(g), ops::argmax(w));
            assert_eq!(ops::argmax(g), ops::argmax(w), "nt {nt} token {t}");
            assert!(diff <= 0.01 * range, "nt {nt} token {t}: {diff} vs range {range}");
        }
        assert_eq!(gm.pos, pre.len() + nt);
        // State after verify continues like the stepped one.
        let next = gm.step(rest.get(nt).copied().unwrap_or(rest[0])).unwrap();
        if nt < rest.len() {
            let diff = next.iter().zip(&seq[nt]).fold(0f32, |a, (x, y)| a.max((x - y).abs()));
            eprintln!("nt {nt} next step max |dlogit| {diff:.4}");
            assert_eq!(ops::argmax(&next), ops::argmax(&seq[nt]));
        }
    }
}

/// Greedy decoding with MTP speculation (`spec_step`, k = 1 to 3) gives the same tokens as plain steps. Same instance and fixed
/// expert placement, so both runs see the same numerics.
#[test]
fn mtp_spec_matches_greedy() {
    let _gpu = common::gpu_lock();
    let Some(p) = common::model_path() else { return };
    let g = Gguf::open(&p).unwrap();
    let m = Model::load(&g).unwrap();
    if m.cfg.n_nextn == 0 { eprintln!("SKIP: no nextn layer"); return }
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => { eprintln!("SKIP: no GPU: {e:#}"); return }
    };
    let mut gm = GpuModel::load(&m, Box::new(gpu), 512, 0, 512 << 20, 3).unwrap();
    assert_eq!(gm.k, 3);
    gm.adapt_every = 0;
    let ids = tok.encode("Write a short story about a dragon.").unwrap();
    let n = 64;
    let mut run = |k: usize| -> (Vec<u32>, (usize, usize)) {
        (gm.k, gm.drafts) = (k.max(1), (0, 0));
        gm.reset().unwrap();
        let mut t = ops::argmax(&gm.prefill(&ids).unwrap()) as u32;
        let (mut out, mut row) = (vec![], 0);
        while out.len() < n {
            if k > 0 {
                let (toks, nx, r) = gm.spec_step(t, row).unwrap();
                (t, row) = (nx, r);
                out.extend(toks);
            } else {
                out.push(t);
                t = ops::argmax(&gm.step(t).unwrap()) as u32;
            }
        }
        out.truncate(n);
        (out, gm.drafts)
    };
    let plain = run(0).0;
    for k in [1, 2, 3] {
        let (spec, (made, ok)) = run(k);
        eprintln!("k {k}: drafts accepted {ok}/{made}");
        assert_eq!(spec, plain, "k {k}");
        if k == 1 { assert!(2 * ok >= made, "acceptance {ok}/{made}"); }
    }
}

/// `rollback` to the first token of a 2-token verify leaves the same state as stepping that token alone.
#[test]
fn rollback_matches_step() {
    let _gpu = common::gpu_lock();
    let Some(p) = common::model_path() else { return };
    let g = Gguf::open(&p).unwrap();
    let m = Model::load(&g).unwrap();
    if m.cfg.n_nextn == 0 { eprintln!("SKIP: no nextn layer"); return }
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => { eprintln!("SKIP: no GPU: {e:#}"); return }
    };
    let mut gm = GpuModel::load(&m, Box::new(gpu), 256, 0, 512 << 20, 1).unwrap();
    gm.adapt_every = 0;
    let ids = tok.encode("The capital of France is Paris, and the capital of Spain is").unwrap();
    let (pre, rest) = ids.split_at(ids.len() - 3);
    for &t in pre { gm.step(t).unwrap(); }
    gm.save_state().unwrap();
    gm.step(rest[0]).unwrap();
    let want = gm.step(rest[2]).unwrap();
    gm.restore_state().unwrap();
    let start = gm.pos;
    gm.verify(&rest[..2]).unwrap();
    gm.rollback(start, 1).unwrap();
    assert_eq!(gm.pos, start + 1);
    assert_eq!(gm.step(rest[2]).unwrap(), want, "step after rollback");
}

/// `prefill_cached` restarts from the saved history state: the logits equal a fresh run split at the same points,
/// with decode steps in between that must not leak into the next prompt.
#[test]
fn prefill_cached_reuses_history() {
    let _gpu = common::gpu_lock();
    let Some(p) = common::model_path() else { return };
    let g = Gguf::open(&p).unwrap();
    let m = Model::load(&g).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => { eprintln!("SKIP: no GPU: {e:#}"); return }
    };
    let mut gm = GpuModel::load(&m, Box::new(gpu), 512, 128, 512 << 20, 0).unwrap();
    gm.adapt_every = 0;
    let enc = |s: &str| tok.encode(s).unwrap();
    let h1 = enc("<|im_start|>user\nTell me about rivers, the ways they carve valleys over thousands of years, and why towns grow on their banks.<|im_end|>\n");
    let h2 = [h1.clone(), enc("<|im_start|>assistant\nRivers move water and sediment downhill.<|im_end|>\n<|im_start|>user\nAnd lakes?<|im_end|>\n")].concat();
    let tail = enc("<|im_start|>assistant\n");
    let fresh = |gm: &mut GpuModel, parts: &[&[u32]]| {
        gm.reset().unwrap();
        parts.iter().map(|p| gm.prefill(p).unwrap()).last().unwrap()
    };

    let p1 = [&h1[..], &tail].concat();
    let want1 = fresh(&mut gm, &[&h1, &tail]);
    let (got1, r1) = gm.prefill_cached(&p1, h1.len()).unwrap();
    assert_eq!((r1, &got1), (0, &want1));
    for t in [11u32, 22, 33] { gm.step(t).unwrap(); } // a reply that the next prompt does not contain

    let p2 = [&h2[..], &tail].concat();
    let (got2, r2) = gm.prefill_cached(&p2, h2.len()).unwrap();
    assert_eq!(r2, h1.len());
    // A prompt that does not extend the saved history (now h2) starts over.
    let (got3, r3) = gm.prefill_cached(&p1, h1.len()).unwrap();
    assert_eq!((r3, &got3), (0, &want1));
    let want2 = fresh(&mut gm, &[&h1, &h2[h1.len()..], &tail]);
    assert_eq!(got2, want2, "reused prefix");
}
