//! Kernel tests, run on every backend that opens (`common::backends`).
mod common;
use common::{down, up};
use rusting_strata::backend::{ATT_GQA_MIN, ATT_SPLITS, GEMM_BM};
use rusting_strata::{cuda::Gpu, gguf::Gguf, quant::GgmlType, weights::Weights};

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
    let Some(p) = common::model_path() else { return };
    for (bk, gpu) in common::backends_except(&["vulkan"]) {
        let gpu = &*gpu;
        eprintln!("backend {bk}");
        let g = Gguf::open(&p).unwrap();
        let w = Weights::new(&g).unwrap();
        let mut seen = std::collections::BTreeSet::new();
        for (name, info) in &g.tensors {
            let Ok(ty) = GgmlType::from_u32(info.ty) else { continue };
            if info.dims.len() < 2 || info.dims[0] % 256 != 0 || !seen.insert(ty.name()) { continue; }
            let t = w.get(name).unwrap();
            let dt = gpu.upload(&t).unwrap();
            let rows = 4.min(t.rows);
            let dq = gpu.zeros::<f32>(rows * t.cols).unwrap();
            gpu.dequant(&dt, rows, dq.addr()).unwrap();
            let got = down::<f32>(gpu, &dq);
            let mut want = vec![0f32; rows * t.cols];
            for (r, out) in want.chunks_exact_mut(t.cols).enumerate() { t.row(r, out); }
            let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
            let diff = got.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(diff <= 1e-5 * scale, "{} dequant ({name}): max diff {diff} vs scale {scale}", ty.name());

            let x = xs(t.cols);
            let dx = up(gpu, &x);
            let dy = gpu.zeros::<f32>(t.rows).unwrap();
            gpu.gemv(&dt, dx.addr(), dy.addr()).unwrap();
            let got = down::<f32>(gpu, &dy);
            let mut want = vec![0f32; t.rows];
            t.matvec(&as_seen(ty, &x), &mut want);
            let e = rel_l2(&got, &want);
            eprintln!("{:8} {name}: dequant max diff {diff:.2e}, gemv rel err {e:.2e}", ty.name());
            assert!(e <= 1e-4, "{} gemv ({name}): rel err {e}", ty.name());
        }
        assert!(seen.len() >= 10, "types covered: {seen:?}");
    }
}

#[test]
fn rms_norm_q_matches_quant_xq() {
    for (bk, gpu) in common::backends() {
        let gpu = &*gpu;
        eprintln!("backend {bk}");
        let (n, rows) = (2048, 3);
        let (x, w) = (up(gpu, &xs(rows * n)), up(gpu, &xs(n + 7)[7..]));
        let (out, q1, q2) = (gpu.zeros::<f32>(rows * n).unwrap(), gpu.zeros::<u8>(rows * n * 9 / 8).unwrap(), gpu.zeros::<u8>(rows * n * 9 / 8).unwrap());
        gpu.rms_norm_q(x.addr(), n, w.addr(), out.addr(), n, n, rows, 1e-6, q1.addr()).unwrap();
        gpu.quant_xq(out.addr(), n, n, rows, q2.addr()).unwrap();
        assert_eq!(down::<u8>(gpu, &q1), down::<u8>(gpu, &q2));
    }
}

#[test]
fn moe_gemv_selects_experts() {
    let Some(p) = common::model_path() else { return };
    for (bk, gpu) in common::backends_except(&["vulkan"]) {
        let gpu = &*gpu;
        eprintln!("backend {bk}");
        let g = Gguf::open(&p).unwrap();
        let w = Weights::new(&g).unwrap();
        let Some(info) = g.tensors.get("blk.0.ffn_gate_exps.weight") else { return };  // dense model
        let t = w.get("blk.0.ffn_gate_exps.weight").unwrap();
        let n_expert = info.dims[2] as usize;
        let dt = gpu.upload(&t).unwrap();
        let ids = [3i32, -1, 200];
        let x = xs(t.cols);
        let (di, dx) = (up(gpu, &ids), up(gpu, &x));
        let rows = t.rows / n_expert;
        let dy = gpu.zeros::<f32>(3 * rows).unwrap();
        gpu.moe_gemv(t.ty, rows, t.cols, dt.ptr, dt.bytes() / n_expert, di.addr(), 3, dx.addr(), 0, 1, dy.addr(), None, 0).unwrap();
        let got = down::<f32>(gpu, &dy);
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
        let pair = (u.ptr as i64 - dt.ptr as i64, du.addr() as i64 - dg.addr() as i64);
        gpu.moe_gemv(t.ty, rows, t.cols, dt.ptr, dt.bytes() / n_expert, di.addr(), 3, dx.addr(), 0, 1, dg.addr(), Some(pair), 0).unwrap();
        gpu.moe_gemv(u.ty, rows, u.cols, u.ptr, u.bytes() / n_expert, di.addr(), 3, dx.addr(), 0, 1, dy.addr(), None, 0).unwrap();
        assert_eq!(down::<f32>(gpu, &dg), got, "paired gate");
        assert_eq!(down::<f32>(gpu, &du), down::<f32>(gpu, &dy), "paired up");
        // x quantized once by quant_xq (one token for the 3 experts) gives the same result as per-block quantization.
        let xq = gpu.zeros::<u8>(t.cols * 9 / 8).unwrap();
        gpu.quant_xq(dx.addr(), 0, t.cols, 1, xq.addr()).unwrap();
        gpu.moe_gemv(t.ty, rows, t.cols, dt.ptr, dt.bytes() / n_expert, di.addr(), 3, 0, 0, 3, dg.addr(), Some(pair), xq.addr()).unwrap();
        assert_eq!(down::<f32>(gpu, &dg), got, "quant_xq gate");
        assert_eq!(down::<f32>(gpu, &du), down::<f32>(gpu, &dy), "quant_xq up");
    }
}

/// Achieved GEMV bandwidth per type (one dense matrix per type, plus 8-expert MoE GEMVs). Run with --ignored.
/// gemvn over 1 to 4 inputs equals gemv per input, bit for bit, for every type.
#[test]
fn gemvn_matches_gemv() {
    let Some(p) = common::model_path() else { return };
    for (bk, gpu) in common::backends_except(&["vulkan"]) {
        let gpu = &*gpu;
        eprintln!("backend {bk}");
        let g = Gguf::open(&p).unwrap();
        let w = Weights::new(&g).unwrap();
        let mut seen = std::collections::BTreeSet::new();
        for (name, info) in &g.tensors {
            let Ok(ty) = GgmlType::from_u32(info.ty) else { continue };
            if info.dims.len() != 2 || info.dims[0] % 256 != 0 || !seen.insert(ty.name()) { continue; }
            let t = w.get(name).unwrap();
            let dt = gpu.upload(&t).unwrap();
            let x: Vec<f32> = (0..4).flat_map(|i| xs(t.cols + i).into_iter().skip(i)).collect();
            let dx = up(gpu, &x);
            let mut want = vec![];
            for i in 0..4u64 {
                let dy = gpu.zeros::<f32>(t.rows).unwrap();
                gpu.gemv(&dt, dx.addr() + 4 * i * t.cols as u64, dy.addr()).unwrap();
                want.extend(down::<f32>(gpu, &dy));
            }
            for nt in 1..=4 {
                let dy = gpu.zeros::<f32>(nt * t.rows).unwrap();
                gpu.gemvn(&dt, dx.addr(), t.cols, dy.addr(), t.rows, nt).unwrap();
                assert_eq!(down::<f32>(gpu, &dy), want[..nt * t.rows], "{} {name} nt {nt}", ty.name());
                let xq = gpu.zeros::<u8>(nt * t.cols * 9 / 8).unwrap();
                gpu.quant_xq(dx.addr(), t.cols, t.cols, nt, xq.addr()).unwrap();
                let dy = gpu.zeros::<f32>(nt * t.rows).unwrap();
                gpu.gemvq(&dt, dx.addr(), t.cols, xq.addr(), dy.addr(), t.rows, nt).unwrap();
                assert_eq!(down::<f32>(gpu, &dy), want[..nt * t.rows], "{} {name} quant_xq nt {nt}", ty.name());
            }
        }
        assert!(seen.len() >= 5, "types covered: {seen:?}");
    }
}

#[test]
#[ignore]
fn gemv_bandwidth() {
    let Some(p) = common::model_path() else { return };
    for (bk, gpu) in common::backends_except(&["vulkan"]) {
        let gpu = &*gpu;
        eprintln!("backend {bk}");
        let g = Gguf::open(&p).unwrap();
        let w = Weights::new(&g).unwrap();
        let time = |f: &dyn Fn()| {
            // Warm up long enough for the clocks to ramp.
            let t0 = std::time::Instant::now();
            while t0.elapsed().as_millis() < 300 { f(); gpu.sync().unwrap(); }
            let t0 = std::time::Instant::now();
            for _ in 0..200 { f(); }
            gpu.sync().unwrap();
            t0.elapsed().as_secs_f64() / 200.0
        };
        let mut seen = std::collections::BTreeSet::new();
        for (name, info) in &g.tensors {
            let Ok(ty) = GgmlType::from_u32(info.ty) else { continue };
            let moe = info.dims.len() == 3;
            if info.dims.len() < 2 || info.dims[0] % 256 != 0 || !seen.insert((ty.name(), info.dims.clone())) { continue; }
            let t = w.get(name).unwrap();
            let dt = gpu.upload(&t).unwrap();
            let x = up(gpu, &xs(t.cols));
            let y = gpu.zeros::<f32>(t.rows).unwrap();
            let run = || if moe {
                let n = info.dims[2] as usize;
                let ids = up(gpu, &[3i32, 17, 40, 77, 101, 150, 200, 255]);
                let rows = t.rows / n;
                (time(&|| gpu.moe_gemv(t.ty, rows, t.cols, dt.ptr, dt.bytes() / n, ids.addr(), 8, x.addr(), 0, 1, y.addr(), None, 0).unwrap()),
                 8 * dt.bytes() / info.dims[2] as usize)
            } else {
                (time(&|| gpu.gemv(&dt, x.addr(), y.addr()).unwrap()), dt.bytes())
            };
            if !moe {
                let (x4, y4) = (up(gpu, &xs(4 * t.cols)), gpu.zeros::<f32>(4 * t.rows).unwrap());
                let tn: Vec<String> = (2..=4).map(|nt| format!("{:.1}", 1e6 * time(&|| gpu.gemvn(&dt, x4.addr(), t.cols, y4.addr(), t.rows, nt).unwrap()))).collect();
                eprintln!("  gemvn 2/3/4 inputs: {} us", tn.join(" / "));
            }
            let (s, bytes) = run();
            eprintln!("{:8} {:5} {name:32} {:6}x{:5}  {:7.1} us  {:6.1} GB/s", ty.name(), if moe { "moe8" } else { "" },
                t.rows, t.cols, s * 1e6, bytes as f64 / s / 1e9);
        }
    }
}

fn f16r(v: f32) -> f32 { half::f16::from_f32(v).to_f32() }

/// Tensor-core GEMM over a tile list matches per-row dot products, for every type, with a gathered X and partial
/// tiles. Types on the int8 path (`Gpu::mmq`) see X as int8 per 32 (as the GEMV path does), the others fp16 inputs.
#[test]
fn gemm_matches_reference_for_every_type() {
    let Some(p) = common::model_path() else { return };
    for (bk, gpu) in common::backends_except(&["vulkan"]) {
        let gpu = &*gpu;
        eprintln!("backend {bk}");
        let g = Gguf::open(&p).unwrap();
        let w = Weights::new(&g).unwrap();
        let mut seen = std::collections::BTreeSet::new();
        for (name, info) in &g.tensors {
            let Ok(ty) = GgmlType::from_u32(info.ty) else { continue };
            if info.dims.len() < 2 || info.dims[0] % 256 != 0 || !seen.insert(ty.name()) { continue; }
            let t = w.get(name).unwrap();
            let dt = gpu.upload(&t).unwrap();
            let (rows, n_x, m) = (t.rows.min(100), 50, 170);
            let x = xs(n_x * t.cols);
            let xrow: Vec<i32> = (0..m).map(|i| ((i * 7) % n_x) as i32).collect();
            // Tiles: positions 0..128, 128..144 and 144..170 (all with the same matrix).
            let tiles: Vec<u64> = [(0u64, 128u64), (128, 16), (144, 26)].iter().flat_map(|&(m0, c)| [dt.ptr, m0 | c << 32]).collect();
            let (dtl, dx, dr) = (up(gpu, &tiles), up(gpu, &x), up(gpu, &xrow));
            let dy = gpu.zeros::<f32>(m * rows).unwrap();
            gpu.gemm(t.ty, rows, t.cols, dtl.addr(), 3, dx.addr(), t.cols, n_x, dr.addr(), dy.addr(), rows, 0).unwrap();
            let got = down::<f32>(gpu, &dy);
            let mut wr = vec![0f32; t.cols];
            let mut want = vec![0f32; m * rows];
            for r in 0..rows {
                t.row(r, &mut wr);
                for (i, &xr) in xrow.iter().enumerate() {
                    let xv = &x[xr as usize * t.cols..][..t.cols];
                    want[i * rows + r] = if Gpu::mmq(t.ty) {
                        wr.iter().zip(as_seen(t.ty, xv)).map(|(a, b)| a * b).sum()
                    } else {
                        wr.iter().zip(xv).map(|(a, b)| f16r(*a) * f16r(*b)).sum()
                    };
                }
            }
            let e = rel_l2(&got, &want);
            eprintln!("{:8} {name}: gemm rel err {e:.2e}", ty.name());
            assert!(e <= 1e-4, "{} gemm ({name}): rel err {e}", ty.name());
        }
    }
}

/// Prefill GEMM throughput on 4096 positions for a few layer matrices. Run with --ignored.
#[test]
#[ignore]
fn gemm_throughput() {
    let Some(p) = common::model_path() else { return };
    for (bk, gpu) in common::backends_except(&["vulkan"]) {
        let gpu = &*gpu;
        eprintln!("backend {bk}");
        let g = Gguf::open(&p).unwrap();
        let w = Weights::new(&g).unwrap();
        let m = 4096usize;
        for name in ["blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_out.weight", "blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight"] {
            let t = w.get(name).unwrap();
            let rows = if g.tensors[name].dims.len() == 3 { t.rows / g.tensors[name].dims[2] as usize } else { t.rows };
            let dt = gpu.upload(&t).unwrap();
            let bm = GEMM_BM;
            let tiles: Vec<u64> = (0..m / bm).flat_map(|i| [dt.ptr, (bm * i) as u64 | (bm as u64) << 32]).collect();
            let (dtl, dx) = (up(gpu, &tiles), up(gpu, &xs(m * t.cols)));
            let dy = gpu.zeros::<f32>(m * rows).unwrap();
            let run = || gpu.gemm(t.ty, rows, t.cols, dtl.addr(), m / bm, dx.addr(), t.cols, m, 0, dy.addr(), rows, 0).unwrap();
            for _ in 0..3 { run(); }
            gpu.sync().unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..10 { run(); }
            gpu.sync().unwrap();
            let s = t0.elapsed().as_secs_f64() / 10.0;
            eprintln!("{:8} {name:28} {rows:6}x{:5} x {m}: {:8.2} ms  {:6.2} TFLOP/s", t.ty.name(), t.cols, s * 1e3, 2.0 * (m * rows * t.cols) as f64 / s / 1e12);
        }
    }
}

/// Prefill attention (tensor cores, f16 Q and P, 64-query tiles) matches decode attention at every position, with
/// keys cached before the chunk, a chunk that does not fill its last tile, and early positions that leave decode
/// splits empty.
#[test]
fn attn_prefill_matches_decode() {
    for (bk, gpu) in common::backends_except(&["vulkan"]) {
        let gpu = &*gpu;
        eprintln!("backend {bk}");
        let (pos0, t) = (5, 100);
        let h16 = |v: Vec<f32>| -> Vec<u16> { v.iter().map(|x| half::f16::from_f32(*x).to_bits()).collect() };
        for (nh, nkv, hd) in [(4, 2, 64), (4, 2, 128), (4, 2, 256), (16, 2, 256), (8, 8, 128)] {
            let n = (pos0 + t) * nkv * hd;
            let (kc, vc) = (h16(xs(n)), h16(xs(n + 5)[5..].to_vec()));
            let q: Vec<f32> = xs(t * nh * hd + 3)[3..].iter().map(|v| 4.0 * v).collect();
            let qg = xs(t * nh * 2 * hd);
            let (dk, dv, dq, dg) = (up(gpu, &kc), up(gpu, &vc),
                                    up(gpu, &q), up(gpu, &qg));
            let scale = 1.0 / (hd as f32).sqrt();
            let step = up(gpu, &[pos0 as i32]);
            let dout = gpu.zeros::<f32>(t * nh * hd).unwrap();
            gpu.attn_prefill(dq.addr(), dk.addr(), dv.addr(), dg.addr(), dout.addr(), step.addr(), t, nh, nkv, hd, scale).unwrap();
            let got = down::<f32>(gpu, &dout);
            // qg = 0: no output gate. The gated output is the ungated output times sigmoid(gate).
            let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
            let gated = |plain: &[f32]| -> Vec<f32> {
                plain.iter().enumerate().map(|(i, o)| o * sig(qg[(i / hd) * 2 * hd + hd + i % hd])).collect()
            };
            gpu.attn_prefill(dq.addr(), dk.addr(), dv.addr(), 0, dout.addr(), step.addr(), t, nh, nkv, hd, scale).unwrap();
            let e = rel_l2(&gated(&down::<f32>(gpu, &dout)), &got);
            assert!(e <= 1e-6, "attn_prefill without gate nh {nh} hd {hd}: {e}");
            let ddec = gpu.zeros::<f32>(t * nh * hd).unwrap();
            let part = gpu.zeros::<f32>(nh * ATT_SPLITS * (hd + 2)).unwrap();
            for gq_min in [ATT_GQA_MIN, 0] {
                for i in 0..t {
                    let s = up(gpu, &[(pos0 + i) as i32]);
                    let off = |p: u64, per: usize| p + (i * per * 4) as u64;
                    gpu.attn_decode_gq(off(dq.addr(), nh * hd), dk.addr(), dv.addr(), off(dg.addr(), nh * 2 * hd),
                                       off(ddec.addr(), nh * hd), part.addr(), s.addr(), nh, nkv, hd, scale, gq_min, 1).unwrap();
                }
                let want = down::<f32>(gpu, &ddec);
                for i in 0..t {
                    let s = up(gpu, &[(pos0 + i) as i32]);
                    let off = |p: u64, per: usize| p + (i * per * 4) as u64;
                    gpu.attn_decode_gq(off(dq.addr(), nh * hd), dk.addr(), dv.addr(), 0,
                                       off(ddec.addr(), nh * hd), part.addr(), s.addr(), nh, nkv, hd, scale, gq_min, 1).unwrap();
                }
                let e = rel_l2(&gated(&down::<f32>(gpu, &ddec)), &want);
                assert!(e <= 1e-6, "attn_decode without gate nh {nh} hd {hd} gq_min {gq_min}: {e}");
                // One launch for all t tokens (the MTP verify path) gives the same bits as t launches.
                let steps: Vec<i32> = (0..t).flat_map(|i| [(pos0 + i) as i32, 0]).collect();
                let ds = up(gpu, &steps);
                let (dbat, pbat) = (gpu.zeros::<f32>(t * nh * hd).unwrap(), gpu.zeros::<f32>(t * nh * ATT_SPLITS * (hd + 2)).unwrap());
                gpu.attn_decode_gq(dq.addr(), dk.addr(), dv.addr(), dg.addr(), dbat.addr(), pbat.addr(), ds.addr(),
                                   nh, nkv, hd, scale, gq_min, t).unwrap();
                assert_eq!(down::<f32>(gpu, &dbat), want, "attn_decode batched nh {nh} nkv {nkv} hd {hd} gq_min {gq_min}");
                let e = rel_l2(&got, &want);
                eprintln!("attn_prefill_{hd} nh {nh} nkv {nkv} gq_min {gq_min}: rel err vs decode {e:.2e}");
                assert!(e <= 5e-4, "attn_prefill_{hd} nh {nh} nkv {nkv} gq_min {gq_min}: rel err {e}");
            }
        }
    }
}

/// DeltaNet scan over a chunk of T tokens equals T single-token scans and a CPU scan (output and final state), and
/// its snapshots equal the states after the first single-token scans.
#[test]
fn deltanet_chunk_matches_steps() {
    for (bk, gpu) in common::backends_except(&["vulkan"]) {
        let gpu = &*gpu;
        eprintln!("backend {bk}");
        let (nk, nv, dk, dv, t) = (2, 4, 128, 128, 37);
        let ych = 2 * nk * dk + nv * dv;
        let y: Vec<f32> = xs(t * ych).iter().map(|v| 0.1 * v).collect();
        let beta: Vec<f32> = xs(t * nv).iter().map(|v| 0.5 + 0.4 * v).collect();
        let g: Vec<f32> = xs(t * nv + 1)[1..].iter().map(|v| -0.2 * v.abs()).collect();
        let s0: Vec<f32> = xs(nv * dv * dk + 2)[2..].iter().map(|v| 0.1 * v).collect();
        let (dy, db, dg) = (up(gpu, &y), up(gpu, &beta), up(gpu, &g));
        for ilv in [false, true] {
            let (ns, len) = (3, nv * dv * dk);
            let run = |chunk: bool| {
                let ds = up(gpu, &s0);
                let o = gpu.zeros::<f32>(t * nv * dv).unwrap();
                let snap = gpu.zeros::<f32>(ns * len).unwrap();
                let p = |d: u64, per: usize, i: usize| d + (i * per * 4) as u64;
                let steps: Vec<(usize, usize)> = if chunk { vec![(0, t)] } else { (0..t).map(|i| (i, 1)).collect() };
                let mut snaps = vec![];
                for (i, n) in steps {
                    gpu.deltanet(p(dy.addr(), ych, i), ds.addr(), p(db.addr(), nv, i), p(dg.addr(), nv, i),
                                 p(o.addr(), nv * dv, i), nk, nv, ilv, dk, dv, 0.5, n, ych, snap.addr(), ns).unwrap();
                    if !chunk && i < ns { snaps.extend(down::<f32>(gpu, &ds)); }
                }
                if chunk { snaps = down::<f32>(gpu, &snap); }
                (down::<f32>(gpu, &o), down::<f32>(gpu, &ds), snaps)
            };
            let ((o1, s1, n1), (o2, s2, n2)) = (run(true), run(false));
            let (eo, es, en) = (rel_l2(&o1, &o2), rel_l2(&s1, &s2), rel_l2(&n1, &n2));
            eprintln!("deltanet ilv {ilv} chunk vs steps: out {eo:.2e}, state {es:.2e}, snapshots {en:.2e}");
            assert!(eo <= 1e-5 && es <= 1e-5 && en <= 1e-5, "ilv {ilv}: out {eo}, state {es}, snapshots {en}");
            // CPU: S_j = decay * S_j; d = beta (v_j - S_j . k); S_j += d k; o_j = scale * S_j . q
            let (mut sc, mut oc) = (s0.clone(), vec![0.0f32; t * nv * dv]);
            for tok in 0..t {
                let yt = &y[tok * ych..];
                for vh in 0..nv {
                    let kh = if ilv { vh / (nv / nk) } else { vh % nk };
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
            eprintln!("deltanet ilv {ilv} vs CPU: out {eo:.2e}, state {es:.2e}");
            assert!(eo <= 1e-5 && es <= 1e-5, "ilv {ilv} vs CPU: out {eo}, state {es}");
        }
    }
}

/// Chunked causal conv (conv_seq + conv_state) equals T single-token conv_silu steps, also for T < kernel width - 1.
#[test]
fn conv_seq_matches_steps() {
    for (bk, gpu) in common::backends_except(&["vulkan"]) {
        let gpu = &*gpu;
        eprintln!("backend {bk}");
        let (ch, kw) = (96, 4);
        let w = xs(ch * kw);
        let st0 = xs((kw - 1) * ch + 9)[9..].to_vec();
        for t in [2, 10] {
            let x = xs(t * ch + 4)[4..].to_vec();
            let (dw, dx) = (up(gpu, &w), up(gpu, &x));
            let (s1, s2) = (up(gpu, &st0), up(gpu, &st0));
            let (y1, y2) = (gpu.zeros::<f32>(t * ch).unwrap(), gpu.zeros::<f32>(t * ch).unwrap());
            let ns = 3;
            let snap = gpu.zeros::<f32>(ns * (kw - 1) * ch).unwrap();
            gpu.conv_seq(dx.addr(), s1.addr(), dw.addr(), y1.addr(), ch, kw, t, snap.addr(), ns).unwrap();
            let d = |a: &rusting_strata::backend::Buf| down::<f32>(gpu, a);
            let mut snaps = vec![];
            for i in 0..t {
                let off = (i * ch * 4) as u64;
                gpu.conv_silu(dx.addr() + off, s2.addr(), dw.addr(), y2.addr() + off, ch, kw).unwrap();
                if i + 1 < t && i < ns { snaps.extend(d(&s2)); }
            }
            let (ey, es) = (rel_l2(&d(&y1), &d(&y2)), rel_l2(&d(&s1), &d(&s2)));
            assert!(ey <= 1e-6 && es == 0.0, "T {t}: y {ey}, state {es}");
            assert_eq!(d(&snap)[..snaps.len()], snaps[..], "T {t}: snapshots");
        }
    }
}

#[test]
fn elementwise_ops_match_cpu() {
    for (bk, gpu) in common::backends() {
        let gpu = &*gpu;
        eprintln!("backend {bk}");
        let n = 1000; // not a multiple of the block size
        let (g, u) = (xs(n).iter().map(|v| v * 8.0).collect::<Vec<f32>>(), xs(n + 1)[1..].to_vec());
        let (dg, du) = (up(gpu, &g), up(gpu, &u));
        gpu.swiglu(dg.addr(), du.addr(), n).unwrap();
        let want: Vec<f32> = g.iter().zip(&u).map(|(g, u)| g / (1.0 + (-g).exp()) * u).collect();
        assert!(rel_l2(&down::<f32>(gpu, &dg), &want) < 1e-6);
        gpu.add(dg.addr(), du.addr(), n).unwrap();
        let want: Vec<f32> = want.iter().zip(&u).map(|(a, b)| a + b).collect();
        assert!(rel_l2(&down::<f32>(gpu, &dg), &want) < 1e-6);
    }
}

#[test]
fn norm_rope_and_gate_ops_match_cpu() {
    use rusting_strata::ops;
    for (bk, gpu) in common::backends() {
        let gpu = &*gpu;
        eprintln!("backend {bk}");
        let close = |got: &[f32], want: &[f32], what: &str| assert!(rel_l2(got, want) < 1e-6, "{what}: rel err {}", rel_l2(got, want));
        let (n, rows) = (2048, 3);
        let (x, w) = (xs(rows * n).iter().map(|v| v * 3.0).collect::<Vec<f32>>(), xs(n + 7)[7..].to_vec());
        let (dx, dw, dout) = (up(gpu, &x), up(gpu, &w), gpu.zeros::<f32>(rows * n).unwrap());
        gpu.rms_norm(dx.addr(), n, dw.addr(), dout.addr(), n, n, rows, 1e-6).unwrap();
        let mut want = vec![0f32; rows * n];
        for (o, r) in want.chunks_mut(n).zip(x.chunks(n)) { ops::rms_norm(r, &w, 1e-6, o) }
        close(&down::<f32>(gpu, &dout), &want, "rms_norm");

        // quant_xq: int8 rows, then scales, as the CPU rounds them.
        let dq = gpu.zeros::<u8>(rows * n * 9 / 8).unwrap();
        gpu.quant_xq(dx.addr(), n, n, rows, dq.addr()).unwrap();
        let q = down::<u8>(gpu, &dq);
        let (mut wq, mut wd) = (vec![], vec![]);
        for b in x.chunks(32) {
            let mx = b.iter().fold(0f32, |m, v| m.max(v.abs()));
            let id = if mx > 0.0 { 127.0 / mx } else { 0.0 };
            wq.extend(b.iter().map(|v| (v * id).round_ties_even() as i8 as u8));
            wd.extend((mx / 127.0).to_le_bytes());
        }
        assert_eq!(q[..rows * n], wq[..], "quant_xq values");
        assert_eq!(q[rows * n..], wd[..], "quant_xq scales");

        // l2_norm on vectors of 128, 2 per token, tokens 300 floats apart.
        let (hn, per, stride) = (128, 2, 300);
        let dx = up(gpu, &x[..3 * stride]);
        gpu.l2_norm(dx.addr(), hn, per, stride, 3 * per, 1e-6).unwrap();
        let mut want = x[..3 * stride].to_vec();
        for t in 0..3 { for h in 0..per { ops::l2_norm(&mut want[t * stride + h * hn..][..hn], 1e-6) } }
        close(&down::<f32>(gpu, &dx), &want, "l2_norm");

        // rope_neox: 3 tokens from position 30000, 2 heads each, 64 of 128 dims rotated.
        let (hd, n_rot, pos) = (128, 64, 30_000);
        let dx = up(gpu, &x[..6 * hd]);
        let step = up(gpu, &[pos as i32, 0]);
        gpu.rope_neox(dx.addr(), hd, 6, 2, n_rot, step.addr(), 1e7).unwrap();
        let mut want = x[..6 * hd].to_vec();
        for (h, v) in want.chunks_mut(hd).enumerate() { ops::rope_neox(v, n_rot, pos + h / 2, 1e7) }
        close(&down::<f32>(gpu, &dx), &want, "rope_neox");

        // norm_gate over 4 heads of 128.
        let z = xs(4 * hn + 3)[3..].to_vec();
        let (dx, dz) = (up(gpu, &x[..4 * hn]), up(gpu, &z));
        gpu.norm_gate(dx.addr(), dw.addr(), dz.addr(), hn, 4, 1e-6).unwrap();
        let mut want = vec![0f32; 4 * hn];
        for h in 0..4 {
            ops::rms_norm(&x[h * hn..][..hn], &w[..hn], 1e-6, &mut want[h * hn..][..hn]);
            for i in h * hn..(h + 1) * hn { want[i] *= ops::silu(z[i]) }
        }
        close(&down::<f32>(gpu, &dx), &want, "norm_gate");

        // ssm_gates on 3 tokens of 5 heads, with g reaching the softplus cutoff.
        let (nv, nt) = (5, 3);
        let (beta, g): (Vec<f32>, Vec<f32>) = (x[..nv * nt].to_vec(), x[100..100 + nv * nt].iter().map(|v| v * 30.0).collect());
        let (dt, a) = (xs(nv + 1)[1..].to_vec(), xs(nv + 2)[2..].to_vec());
        let (db, dg, ddt, da) = (up(gpu, &beta), up(gpu, &g), up(gpu, &dt), up(gpu, &a));
        gpu.ssm_gates(db.addr(), dg.addr(), ddt.addr(), da.addr(), nv * nt, nv).unwrap();
        close(&down::<f32>(gpu, &db), &beta.iter().map(|&b| ops::sigmoid(b)).collect::<Vec<_>>(), "ssm_gates beta");
        let want: Vec<f32> = g.iter().enumerate().map(|(i, &g)| ops::softplus(g + dt[i % nv]) * a[i % nv]).collect();
        close(&down::<f32>(gpu, &dg), &want, "ssm_gates g");

        // conv_silu: 300 channels, kernel 4, two steps so the state shifts.
        let (ch, kw) = (300, 4);
        let (cw, st0) = (xs(ch * kw + 5)[5..].to_vec(), xs((kw - 1) * ch + 9)[9..].to_vec());
        let (dcw, dst, dy) = (up(gpu, &cw), up(gpu, &st0), gpu.zeros::<f32>(ch).unwrap());
        let mut st = st0.clone();
        for s in 0..2 {
            let qkv = x[s * ch..][..ch].to_vec();
            let dq = up(gpu, &qkv);
            gpu.conv_silu(dq.addr(), dst.addr(), dcw.addr(), dy.addr(), ch, kw).unwrap();
            let want: Vec<f32> = (0..ch).map(|c| {
                let acc = (0..kw - 1).map(|k| st[k * ch + c] * cw[c * kw + k]).sum::<f32>() + qkv[c] * cw[c * kw + kw - 1];
                ops::silu(acc)
            }).collect();
            st.drain(..ch);
            st.extend(&qkv);
            close(&down::<f32>(gpu, &dy), &want, "conv_silu y");
            close(&down::<f32>(gpu, &dst), &st, "conv_silu state");
        }
    }
}

#[test]
fn argmax_rows_matches_ops_argmax() {
    for (bk, gpu) in common::backends() {
        let gpu = &*gpu;
        eprintln!("backend {bk}");
        let n = 248_320;
        // Row 0 generic, row 1 a tie (first index wins), row 2 the maximum in the last element, row 3 all -inf but one.
        let mut x: Vec<f32> = (0..4).flat_map(|r| xs(n + r).into_iter().skip(r)).collect();
        x[n + 1000] = 9.0; x[n + 200_000] = 9.0;
        x[3 * n - 1] = 5.0;
        x[3 * n..].fill(f32::NEG_INFINITY); x[3 * n + 77] = -1.0;
        let dx = up(gpu, &x);
        let dout = gpu.zeros::<i32>(4).unwrap();
        gpu.argmax_rows(dx.addr(), n, 4, dout.addr()).unwrap();
        let got = down::<i32>(gpu, &dout);
        let want: Vec<i32> = x.chunks(n).map(|r| rusting_strata::ops::argmax(r) as i32).collect();
        assert_eq!(got, want);
        assert_eq!(want[1..], [1000, n as i32 - 1, 77]);
    }
}

#[test]
fn router_topk_matches_ops_top_k() {
    for (bk, gpu) in common::backends_except(&["vulkan"]) {
        let gpu = &*gpu;
        eprintln!("backend {bk}");
        let (n, k, t) = (256, 8, 3);
        // Token 1 has a tie between experts 7 and 200 (lower index first), token 2 a tie at the k-th place.
        let mut x: Vec<f32> = xs(n * t).iter().map(|v| v * 4.0).collect();
        x[n + 7] = 9.0; x[n + 200] = 9.0;
        x[2 * n + 30] = 5.0; x[2 * n + 31] = 5.0; x[2 * n + 255] = 5.0;
        let dx = up(gpu, &x);
        let (dids, dw) = (gpu.zeros::<i32>(k * t).unwrap(), gpu.zeros::<f32>(k * t).unwrap());
        gpu.router_topk(dx.addr(), n, k, 2.5, dids.addr(), dw.addr(), t).unwrap();
        let (ids, w) = (down::<i32>(gpu, &dids), down::<f32>(gpu, &dw));
        for r in 0..t {
            let mut p = x[r * n..(r + 1) * n].to_vec();
            rusting_strata::ops::softmax(&mut p);
            let want = rusting_strata::ops::top_k(&p, k);
            assert_eq!(ids[r * k..(r + 1) * k].iter().map(|&i| i as usize).collect::<Vec<_>>(), want, "token {r}");
            let s: f32 = want.iter().map(|&i| p[i]).sum();
            for (j, &i) in want.iter().enumerate() { assert!((w[r * k + j] - p[i] / s * 2.5).abs() < 1e-5, "token {r} {j}"); }
        }
        assert_eq!(ids[k..k + 2], [7, 200]);
    }
}

/// Decode attention time per call for the model's shape (16 query heads, 2 KV heads, head_dim 256), both kernels.
#[test]
#[ignore]
fn attn_decode_bandwidth() {
    for (bk, gpu) in common::backends_except(&["vulkan"]) {
        let gpu = &*gpu;
        eprintln!("backend {bk}");
        let (nh, nkv, hd) = (16, 2, 256);
        for n_pos in [1024, 2048, 4096, 16384, 32768] {
            let n = n_pos * nkv * hd;
            let h16: Vec<u16> = xs(n).iter().map(|x| half::f16::from_f32(*x).to_bits()).collect();
            let (dk, dv) = (up(gpu, &h16), up(gpu, &h16));
            for nt in [1, 2] {
                let (dq, dg) = (up(gpu, &xs(nt * nh * hd)), up(gpu, &xs(nt * nh * 2 * hd)));
                let steps: Vec<i32> = (0..nt).flat_map(|i| [(n_pos - nt + i) as i32, 0]).collect();
                let ds = up(gpu, &steps);
                let (dout, part) = (gpu.zeros::<f32>(nt * nh * hd).unwrap(), gpu.zeros::<f32>(nt * nh * ATT_SPLITS * (hd + 2)).unwrap());
                for gq_min in [i32::MAX, 0] {
                    let run = || gpu.attn_decode_gq(dq.addr(), dk.addr(), dv.addr(), dg.addr(), dout.addr(), part.addr(), ds.addr(),
                                                    nh, nkv, hd, 0.0625, gq_min, nt).unwrap();
                    let t0 = std::time::Instant::now();  // warm up long enough for the clocks to rise
                    while t0.elapsed().as_millis() < 300 { run(); gpu.sync().unwrap(); }
                    let t0 = std::time::Instant::now();
                    for _ in 0..200 { run(); }
                    gpu.sync().unwrap();
                    let s = t0.elapsed().as_secs_f64() / 200.0;
                    eprintln!("pos {n_pos:6} nt {nt} {}: {:7.1} us  {:5.0} GB/s", if gq_min == 0 { "gqa" } else { "g1 " }, s * 1e6, (2 * n * 2) as f64 / s / 1e9);
                }
            }
        }
    }
}
