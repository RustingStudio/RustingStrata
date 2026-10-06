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
    gpu.moe_gemv(t.ty, rows, t.cols, dt.ptr, dt.bytes() / n_expert, gpu.ptr(&di), 3, gpu.ptr(&dx), 0, gpu.ptr(&dy)).unwrap();
    let got = gpu.stream.clone_dtoh(&dy).unwrap();
    for (k, &e) in ids.iter().enumerate() {
        let mut want = vec![0f32; rows];
        if e >= 0 { t.expert(e as usize, n_expert).matvec(&as_seen(t.ty, &x), &mut want); }
        else { assert!(got[k * rows..(k + 1) * rows].iter().all(|&v| v == 0.0), "skipped slot written"); continue; }
        let err = rel_l2(&got[k * rows..(k + 1) * rows], &want);
        assert!(err <= 1e-4, "expert {e}: rel err {err}");
    }
}

/// Achieved GEMV bandwidth per type (one dense matrix per type, plus 8-expert MoE GEMVs). Run with --ignored.
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
            (time(&|| gpu.moe_gemv(t.ty, rows, t.cols, dt.ptr, dt.bytes() / n, gpu.ptr(&ids), 8, gpu.ptr(&x), 0, gpu.ptr(&y)).unwrap()),
             8 * dt.bytes() / info.dims[2] as usize)
        } else {
            (time(&|| gpu.gemv(&dt, gpu.ptr(&x), gpu.ptr(&y)).unwrap()), dt.bytes())
        };
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
