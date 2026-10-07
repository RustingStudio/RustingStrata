mod common;
use rusting_strata::{cuda::Gpu, gguf::Gguf, quant::GgmlType, weights::Weights};

/// Skip only when there is no usable CUDA device; kernel compile errors fail the test.
fn gpu() -> Option<Gpu> {
    match Gpu::new() {
        Ok(g) => Some(g),
        Err(e) if format!("{e}") == "CUDA init" => { eprintln!("SKIP: no GPU: {e:#}"); None }
        Err(e) => panic!("{e:#}"),
    }
}

fn xs(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 * 0.7548777).fract() - 0.5) * 2.0).collect()
}

/// x as the integer GEMV path sees it: int8 per 32 elements with scale max|x| / 127 (float weight types read x as is).
fn as_seen(ty: GgmlType, x: &[f32]) -> Vec<f32> {
    if matches!(ty, GgmlType::F32 | GgmlType::BF16) { return x.to_vec(); }
    x.chunks(32).flat_map(|b| {
        let mx = b.iter().fold(0f32, |m, v| m.max(v.abs()));
        let id = if mx > 0.0 { 127.0 / mx } else { 0.0 };
        b.iter().map(move |v| (v * id).round_ties_even() * (mx / 127.0))
    }).collect()
}

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f32 = b.iter().map(|y| y * y).sum();
    (num / den.max(1e-30)).sqrt()
}

#[test]
fn dequant_and_gemv_match_reference_for_every_type() {
    let (Some(p), Some(gpu)) = (common::model_path(), gpu()) else { return };
    let g = Gguf::open(&p).unwrap();
    let w = Weights { g: &g };
    let mut seen = std::collections::BTreeSet::new();
    for (name, info) in &g.tensors {
        let ty = GgmlType::from_u32(info.ty).unwrap();
        if info.dims.len() < 2 || info.dims[0] % 256 != 0 || !seen.insert(ty.name()) { continue; }
        let t = w.get(name).unwrap();
        let dt = gpu.upload(&t).unwrap();
        let rows = 4.min(t.rows);
        let got = gpu.dequant(&dt, rows).unwrap();
        let mut want = vec![0f32; rows * t.cols];
        for (r, out) in want.chunks_exact_mut(t.cols).enumerate() { t.row(r, out); }
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        let diff = got.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(diff <= 1e-5 * scale, "{} dequant ({name}): max diff {diff} vs scale {scale}", ty.name());

        let x = xs(t.cols);
        let dx = gpu.stream.clone_htod(&x).unwrap();
        let dy = gpu.zeros::<f32>(t.rows).unwrap();
        gpu.gemv(&dt, gpu.ptr(&dx), gpu.ptr(&dy)).unwrap();
        let got = gpu.stream.clone_dtoh(&dy).unwrap();
        let mut want = vec![0f32; t.rows];
        t.matvec(&as_seen(ty, &x), &mut want);
        let e = rel_l2(&got, &want);
        eprintln!("{:8} {name}: dequant max diff {diff:.2e}, gemv rel err {e:.2e}", ty.name());
        assert!(e <= 1e-4, "{} gemv ({name}): rel err {e}", ty.name());
    }
    assert!(seen.len() >= 10, "types covered: {seen:?}");
}

#[test]
fn moe_gemv_selects_experts() {
    let (Some(p), Some(gpu)) = (common::model_path(), gpu()) else { return };
    let g = Gguf::open(&p).unwrap();
    let w = Weights { g: &g };
    let t = w.get("blk.0.ffn_gate_exps.weight").unwrap();
    let n_expert = g.tensors["blk.0.ffn_gate_exps.weight"].dims[2] as usize;
    let dt = gpu.upload(&t).unwrap();
    let ids = [3i32, -1, 200];
    let x = xs(t.cols);
    let (di, dx) = (gpu.stream.clone_htod(&ids).unwrap(), gpu.stream.clone_htod(&x).unwrap());
    let rows = t.rows / n_expert;
    let dy = gpu.zeros::<f32>(3 * rows).unwrap();
    gpu.moe_gemv(t.ty, rows, t.cols, dt.ptr, dt.bytes() / n_expert, gpu.ptr(&di), 3, gpu.ptr(&dx), 0, gpu.ptr(&dy), None).unwrap();
    let got = gpu.stream.clone_dtoh(&dy).unwrap();
    for (k, &e) in ids.iter().enumerate() {
        let mut want = vec![0f32; rows];
        if e >= 0 { t.expert(e as usize, n_expert).matvec(&as_seen(t.ty, &x), &mut want); }
        else { assert!(got[k * rows..(k + 1) * rows].iter().all(|&v| v == 0.0), "skipped slot written"); continue; }
        let err = rel_l2(&got[k * rows..(k + 1) * rows], &want);
        assert!(err <= 1e-4, "expert {e}: rel err {err}");
    }
    // The paired launch (gate and up in one) matches a separate launch of the up matrices bit for bit.
    let u = gpu.upload(&w.get("blk.0.ffn_up_exps.weight").unwrap()).unwrap();
    let (dg, du) = (gpu.zeros::<f32>(3 * rows).unwrap(), gpu.zeros::<f32>(3 * rows).unwrap());
    let pair = (u.ptr as i64 - dt.ptr as i64, gpu.ptr(&du) as i64 - gpu.ptr(&dg) as i64);
    gpu.moe_gemv(t.ty, rows, t.cols, dt.ptr, dt.bytes() / n_expert, gpu.ptr(&di), 3, gpu.ptr(&dx), 0, gpu.ptr(&dg), Some(pair)).unwrap();
    gpu.moe_gemv(u.ty, rows, u.cols, u.ptr, u.bytes() / n_expert, gpu.ptr(&di), 3, gpu.ptr(&dx), 0, gpu.ptr(&dy), None).unwrap();
    assert_eq!(gpu.stream.clone_dtoh(&dg).unwrap(), got, "paired gate");
    assert_eq!(gpu.stream.clone_dtoh(&du).unwrap(), gpu.stream.clone_dtoh(&dy).unwrap(), "paired up");
}

/// Achieved GEMV bandwidth per type (one dense matrix per type, plus 8-expert MoE GEMVs). Run with --ignored.
/// gemvn over 1 to 4 inputs equals gemv per input, bit for bit, for every integer type.
#[test]
fn gemvn_matches_gemv() {
    let (Some(p), Some(gpu)) = (common::model_path(), gpu()) else { return };
    let g = Gguf::open(&p).unwrap();
    let w = Weights { g: &g };
    let mut seen = std::collections::BTreeSet::new();
    for (name, info) in &g.tensors {
        let ty = GgmlType::from_u32(info.ty).unwrap();
        if info.dims.len() != 2 || info.dims[0] % 256 != 0 || !seen.insert(ty.name()) { continue; }
        let t = w.get(name).unwrap();
        let dt = gpu.upload(&t).unwrap();
        let x: Vec<f32> = (0..4).flat_map(|i| xs(t.cols + i).into_iter().skip(i)).collect();
        let dx = gpu.stream.clone_htod(&x).unwrap();
        let mut want = vec![];
        for i in 0..4u64 {
            let dy = gpu.zeros::<f32>(t.rows).unwrap();
            gpu.gemv(&dt, gpu.ptr(&dx) + 4 * i * t.cols as u64, gpu.ptr(&dy)).unwrap();
            want.extend(gpu.stream.clone_dtoh(&dy).unwrap());
        }
        for nt in 1..=4 {
            let dy = gpu.zeros::<f32>(nt * t.rows).unwrap();
            gpu.gemvn(&dt, gpu.ptr(&dx), t.cols, gpu.ptr(&dy), t.rows, nt).unwrap();
            assert_eq!(gpu.stream.clone_dtoh(&dy).unwrap(), want[..nt * t.rows], "{} {name} nt {nt}", ty.name());
        }
    }
    assert!(seen.len() >= 5, "types covered: {seen:?}");
}

#[test]
#[ignore]
fn gemv_bandwidth() {
    let (Some(p), Some(gpu)) = (common::model_path(), gpu()) else { return };
    let g = Gguf::open(&p).unwrap();
    let w = Weights { g: &g };
    let time = |f: &dyn Fn()| {
        // Warm up long enough for the clocks to ramp.
        let t0 = std::time::Instant::now();
        while t0.elapsed().as_millis() < 300 { f(); gpu.stream.synchronize().unwrap(); }
        let t0 = std::time::Instant::now();
        for _ in 0..200 { f(); }
        gpu.stream.synchronize().unwrap();
        t0.elapsed().as_secs_f64() / 200.0
    };
    let mut seen = std::collections::BTreeSet::new();
    for (name, info) in &g.tensors {
        let ty = GgmlType::from_u32(info.ty).unwrap();
        let moe = info.dims.len() == 3;
        if info.dims.len() < 2 || info.dims[0] % 256 != 0 || !seen.insert((ty.name(), moe)) { continue; }
        let t = w.get(name).unwrap();
        let dt = gpu.upload(&t).unwrap();
        let x = gpu.stream.clone_htod(&xs(t.cols)).unwrap();
        let y = gpu.zeros::<f32>(t.rows).unwrap();
        let run = || if moe {
            let n = info.dims[2] as usize;
            let ids = gpu.stream.clone_htod(&[3i32, 17, 40, 77, 101, 150, 200, 255]).unwrap();
            let rows = t.rows / n;
            (time(&|| gpu.moe_gemv(t.ty, rows, t.cols, dt.ptr, dt.bytes() / n, gpu.ptr(&ids), 8, gpu.ptr(&x), 0, gpu.ptr(&y), None).unwrap()),
             8 * dt.bytes() / info.dims[2] as usize)
        } else {
            (time(&|| gpu.gemv(&dt, gpu.ptr(&x), gpu.ptr(&y)).unwrap()), dt.bytes())
        };
        if !moe {
            let (x4, y4) = (gpu.stream.clone_htod(&xs(4 * t.cols)).unwrap(), gpu.zeros::<f32>(4 * t.rows).unwrap());
            let tn: Vec<String> = (2..=4).map(|nt| format!("{:.1}", 1e6 * time(&|| gpu.gemvn(&dt, gpu.ptr(&x4), t.cols, gpu.ptr(&y4), t.rows, nt).unwrap()))).collect();
            eprintln!("  gemvn 2/3/4 inputs: {} us", tn.join(" / "));
        }
        let (s, bytes) = run();
        eprintln!("{:8} {:5} {name:32} {:6}x{:5}  {:7.1} us  {:6.1} GB/s", ty.name(), if moe { "moe8" } else { "" },
            t.rows, t.cols, s * 1e6, bytes as f64 / s / 1e9);
    }
}

fn f16r(v: f32) -> f32 { half::f16::from_f32(v).to_f32() }

/// Tensor-core GEMM over a tile list matches per-row dot products of fp16-rounded inputs, for every type, with a
/// gathered X and partial tiles.
#[test]
fn gemm_matches_reference_for_every_type() {
    let (Some(p), Some(gpu)) = (common::model_path(), gpu()) else { return };
    let g = Gguf::open(&p).unwrap();
    let w = Weights { g: &g };
    let mut seen = std::collections::BTreeSet::new();
    for (name, info) in &g.tensors {
        let ty = GgmlType::from_u32(info.ty).unwrap();
        if info.dims.len() < 2 || info.dims[0] % 256 != 0 || !seen.insert(ty.name()) { continue; }
        let t = w.get(name).unwrap();
        let dt = gpu.upload(&t).unwrap();
        let (rows, n_x, m) = (t.rows.min(100), 50, 170);
        let x = xs(n_x * t.cols);
        let xrow: Vec<i32> = (0..m).map(|i| ((i * 7) % n_x) as i32).collect();
        // Tiles: positions 0..128, 128..144 and 144..170 (all with the same matrix).
        let tiles: Vec<u64> = [(0u64, 128u64), (128, 16), (144, 26)].iter().flat_map(|&(m0, c)| [dt.ptr, m0 | c << 32]).collect();
        let (dtl, dx, dr) = (gpu.stream.clone_htod(&tiles).unwrap(), gpu.stream.clone_htod(&x).unwrap(), gpu.stream.clone_htod(&xrow).unwrap());
        let dy = gpu.zeros::<f32>(m * rows).unwrap();
        gpu.gemm(t.ty, rows, t.cols, gpu.ptr(&dtl), 3, gpu.ptr(&dx), t.cols, gpu.ptr(&dr), gpu.ptr(&dy), rows, 0).unwrap();
        let got = gpu.stream.clone_dtoh(&dy).unwrap();
        let mut wr = vec![0f32; t.cols];
        let mut want = vec![0f32; m * rows];
        for r in 0..rows {
            t.row(r, &mut wr);
            for (i, &xr) in xrow.iter().enumerate() {
                let xv = &x[xr as usize * t.cols..][..t.cols];
                want[i * rows + r] = wr.iter().zip(xv).map(|(a, b)| f16r(*a) * f16r(*b)).sum();
            }
        }
        let e = rel_l2(&got, &want);
        eprintln!("{:8} {name}: gemm rel err {e:.2e}", ty.name());
        assert!(e <= 1e-4, "{} gemm ({name}): rel err {e}", ty.name());
    }
}

/// Prefill GEMM throughput on 4096 positions for a few layer matrices. Run with --ignored.
#[test]
#[ignore]
fn gemm_throughput() {
    let (Some(p), Some(gpu)) = (common::model_path(), gpu()) else { return };
    let g = Gguf::open(&p).unwrap();
    let w = Weights { g: &g };
    let m = 4096usize;
    for name in ["blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_out.weight", "blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight"] {
        let t = w.get(name).unwrap();
        let rows = if g.tensors[name].dims.len() == 3 { t.rows / g.tensors[name].dims[2] as usize } else { t.rows };
        let dt = gpu.upload(&t).unwrap();
        let bm = rusting_strata::cuda::GEMM_BM;
        let tiles: Vec<u64> = (0..m / bm).flat_map(|i| [dt.ptr, (bm * i) as u64 | (bm as u64) << 32]).collect();
        let (dtl, dx) = (gpu.stream.clone_htod(&tiles).unwrap(), gpu.stream.clone_htod(&xs(m * t.cols)).unwrap());
        let dy = gpu.zeros::<f32>(m * rows).unwrap();
        let run = || gpu.gemm(t.ty, rows, t.cols, gpu.ptr(&dtl), m / bm, gpu.ptr(&dx), t.cols, 0, gpu.ptr(&dy), rows, 0).unwrap();
        for _ in 0..3 { run(); }
        gpu.stream.synchronize().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..10 { run(); }
        gpu.stream.synchronize().unwrap();
        let s = t0.elapsed().as_secs_f64() / 10.0;
        eprintln!("{:8} {name:28} {rows:6}x{:5} x {m}: {:8.2} ms  {:6.2} TFLOP/s", t.ty.name(), t.cols, s * 1e3, 2.0 * (m * rows * t.cols) as f64 / s / 1e12);
    }
}

/// Prefill attention (tensor cores, f16 Q and P, 64-query tiles) matches decode attention at every position, with
/// keys cached before the chunk, a chunk that does not fill its last tile, and early positions that leave decode
/// splits empty.
#[test]
fn attn_prefill_matches_decode() {
    let Some(gpu) = gpu() else { return };
    let (pos0, t) = (5, 100);
    let h16 = |v: Vec<f32>| -> Vec<u16> { v.iter().map(|x| half::f16::from_f32(*x).to_bits()).collect() };
    for (nh, nkv, hd) in [(4, 2, 64), (4, 2, 128), (4, 2, 256), (16, 2, 256), (8, 8, 128)] {
        let n = (pos0 + t) * nkv * hd;
        let (kc, vc) = (h16(xs(n)), h16(xs(n + 5)[5..].to_vec()));
        let q: Vec<f32> = xs(t * nh * hd + 3)[3..].iter().map(|v| 4.0 * v).collect();
        let qg = xs(t * nh * 2 * hd);
        let (dk, dv, dq, dg) = (gpu.stream.clone_htod(&kc).unwrap(), gpu.stream.clone_htod(&vc).unwrap(),
                                gpu.stream.clone_htod(&q).unwrap(), gpu.stream.clone_htod(&qg).unwrap());
        let scale = 1.0 / (hd as f32).sqrt();
        let step = gpu.stream.clone_htod(&[pos0 as i32]).unwrap();
        let dout = gpu.zeros::<f32>(t * nh * hd).unwrap();
        gpu.attn_prefill(gpu.ptr(&dq), gpu.ptr(&dk), gpu.ptr(&dv), gpu.ptr(&dg), gpu.ptr(&dout), gpu.ptr(&step), t, nh, nkv, hd, scale).unwrap();
        let got = gpu.stream.clone_dtoh(&dout).unwrap();
        let ddec = gpu.zeros::<f32>(t * nh * hd).unwrap();
        let part = gpu.zeros::<f32>(nh * rusting_strata::cuda::ATT_SPLITS * (hd + 2)).unwrap();
        for gq_min in [rusting_strata::cuda::ATT_GQA_MIN, 0] {
            for i in 0..t {
                let s = gpu.stream.clone_htod(&[(pos0 + i) as i32]).unwrap();
                let off = |p: u64, per: usize| p + (i * per * 4) as u64;
                gpu.attn_decode_gq(off(gpu.ptr(&dq), nh * hd), gpu.ptr(&dk), gpu.ptr(&dv), off(gpu.ptr(&dg), nh * 2 * hd),
                                   off(gpu.ptr(&ddec), nh * hd), gpu.ptr(&part), gpu.ptr(&s), nh, nkv, hd, scale, gq_min).unwrap();
            }
            let want = gpu.stream.clone_dtoh(&ddec).unwrap();
            let e = rel_l2(&got, &want);
            eprintln!("attn_prefill_{hd} nh {nh} nkv {nkv} gq_min {gq_min}: rel err vs decode {e:.2e}");
            assert!(e <= 5e-4, "attn_prefill_{hd} nh {nh} nkv {nkv} gq_min {gq_min}: rel err {e}");
        }
    }
}

/// DeltaNet scan over a chunk of T tokens equals T single-token scans and a CPU scan (output and final state).
#[test]
fn deltanet_chunk_matches_steps() {
    let Some(gpu) = gpu() else { return };
    let (nk, nv, dk, dv, t) = (2, 4, 128, 128, 37);
    let ych = 2 * nk * dk + nv * dv;
    let y: Vec<f32> = xs(t * ych).iter().map(|v| 0.1 * v).collect();
    let beta: Vec<f32> = xs(t * nv).iter().map(|v| 0.5 + 0.4 * v).collect();
    let g: Vec<f32> = xs(t * nv + 1)[1..].iter().map(|v| -0.2 * v.abs()).collect();
    let s0: Vec<f32> = xs(nv * dv * dk + 2)[2..].iter().map(|v| 0.1 * v).collect();
    let (dy, db, dg) = (gpu.stream.clone_htod(&y).unwrap(), gpu.stream.clone_htod(&beta).unwrap(), gpu.stream.clone_htod(&g).unwrap());
    let run = |chunk: bool| {
        let ds = gpu.stream.clone_htod(&s0).unwrap();
        let o = gpu.zeros::<f32>(t * nv * dv).unwrap();
        let p = |d: u64, per: usize, i: usize| d + (i * per * 4) as u64;
        let steps: Vec<(usize, usize)> = if chunk { vec![(0, t)] } else { (0..t).map(|i| (i, 1)).collect() };
        for (i, n) in steps {
            gpu.deltanet(p(gpu.ptr(&dy), ych, i), gpu.ptr(&ds), p(gpu.ptr(&db), nv, i), p(gpu.ptr(&dg), nv, i),
                         p(gpu.ptr(&o), nv * dv, i), nk, nv, dk, dv, 0.5, n, ych).unwrap();
        }
        (gpu.stream.clone_dtoh(&o).unwrap(), gpu.stream.clone_dtoh(&ds).unwrap())
    };
    let ((o1, s1), (o2, s2)) = (run(true), run(false));
    let (eo, es) = (rel_l2(&o1, &o2), rel_l2(&s1, &s2));
    eprintln!("deltanet chunk vs steps: out {eo:.2e}, state {es:.2e}");
    assert!(eo <= 1e-5 && es <= 1e-5, "out {eo}, state {es}");
    // CPU: S_j = decay * S_j; d = beta (v_j - S_j . k); S_j += d k; o_j = scale * S_j . q
    let (mut sc, mut oc) = (s0.clone(), vec![0.0f32; t * nv * dv]);
    for tok in 0..t {
        let yt = &y[tok * ych..];
        for vh in 0..nv {
            let kh = vh % nk;
            let (q, k) = (&yt[kh * dk..][..dk], &yt[nk * dk + kh * dk..][..dk]);
            let (decay, b) = (g[tok * nv + vh].exp(), beta[tok * nv + vh]);
            for j in 0..dv {
                let row = &mut sc[(vh * dv + j) * dk..][..dk];
                row.iter_mut().for_each(|x| *x *= decay);
                let d = b * (yt[2 * nk * dk + vh * dv + j] - row.iter().zip(k).map(|(a, b)| a * b).sum::<f32>());
                row.iter_mut().zip(k).for_each(|(x, k)| *x += d * k);
                oc[tok * nv * dv + vh * dv + j] = 0.5 * row.iter().zip(q).map(|(a, b)| a * b).sum::<f32>();
            }
        }
    }
    let (eo, es) = (rel_l2(&o1, &oc), rel_l2(&s1, &sc));
    eprintln!("deltanet vs CPU: out {eo:.2e}, state {es:.2e}");
    assert!(eo <= 1e-5 && es <= 1e-5, "vs CPU: out {eo}, state {es}");
}

/// Chunked causal conv (conv_seq + conv_state) equals T single-token conv_silu steps, also for T < kernel width - 1.
#[test]
fn conv_seq_matches_steps() {
    let Some(gpu) = gpu() else { return };
    let (ch, kw) = (96, 4);
    let w = xs(ch * kw);
    let st0 = xs((kw - 1) * ch + 9)[9..].to_vec();
    for t in [2, 10] {
        let x = xs(t * ch + 4)[4..].to_vec();
        let (dw, dx) = (gpu.stream.clone_htod(&w).unwrap(), gpu.stream.clone_htod(&x).unwrap());
        let (s1, s2) = (gpu.stream.clone_htod(&st0).unwrap(), gpu.stream.clone_htod(&st0).unwrap());
        let (y1, y2) = (gpu.zeros::<f32>(t * ch).unwrap(), gpu.zeros::<f32>(t * ch).unwrap());
        gpu.conv_seq(gpu.ptr(&dx), gpu.ptr(&s1), gpu.ptr(&dw), gpu.ptr(&y1), ch, kw, t).unwrap();
        for i in 0..t {
            let off = (i * ch * 4) as u64;
            gpu.conv_silu(gpu.ptr(&dx) + off, gpu.ptr(&s2), gpu.ptr(&dw), gpu.ptr(&y2) + off, ch, kw).unwrap();
        }
        let d = |a: &cudarc::driver::CudaSlice<f32>| gpu.stream.clone_dtoh(a).unwrap();
        let (ey, es) = (rel_l2(&d(&y1), &d(&y2)), rel_l2(&d(&s1), &d(&s2)));
        assert!(ey <= 1e-6 && es == 0.0, "T {t}: y {ey}, state {es}");
    }
}
