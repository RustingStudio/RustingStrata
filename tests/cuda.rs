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
        t.matvec(&x, &mut want);
        let e = rel_l2(&got, &want);
        eprintln!("{:8} {name}: dequant max diff {diff:.2e}, gemv rel err {e:.2e}", ty.name());
        assert!(e <= 1e-5, "{} gemv ({name}): rel err {e}", ty.name());
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
        if e >= 0 { t.expert(e as usize, n_expert).matvec(&x, &mut want); }
        else { assert!(got[k * rows..(k + 1) * rows].iter().all(|&v| v == 0.0), "skipped slot written"); continue; }
        let err = rel_l2(&got[k * rows..(k + 1) * rows], &want);
        assert!(err <= 1e-5, "expert {e}: rel err {err}");
    }
}
