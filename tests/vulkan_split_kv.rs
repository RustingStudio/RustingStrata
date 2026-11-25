//! Vulkan split-K/V cache formats. Skips when Vulkan or CUDA is unavailable.
//! Symmetric legacy modes stay byte-identical to CUDA; mixed legacy halves keep their own layouts; TurboQuant
//! prefill agrees with decode with one or both turbo halves.
mod common;
use common::{down, up};
use rusting_strata::backend::Backend;
use rusting_strata::cuda::{ATT_GQA_MIN, ATT_SPLITS, Gpu};
use rusting_strata::vulkan::Vk;

fn xs(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 * 0.7548777).fract() - 0.5) * 2.0).collect()
}
fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f32 = b.iter().map(|y| y * y).sum();
    (num / den.max(1e-30)).sqrt()
}
fn set(k: &str, v: &str) {
    unsafe {
        std::env::set_var("STRATA_KV_K", k);
        std::env::set_var("STRATA_KV_V", v);
        std::env::remove_var("STRATA_KV");
        std::env::remove_var("STRATA_KV_Q8");
    }
}
fn open_vk() -> Option<Vk> {
    for i in 0..8 {
        match Vk::new(i) {
            Ok(v) => return Some(v),
            Err(e) if format!("{e}") == "Vulkan init" => { eprintln!("vulkan device {i}: {e:#}"); }
            Err(e) => panic!("{e:#}"),
        }
    }
    eprintln!("SKIP: no usable Vulkan device");
    None
}
fn open_cu() -> Option<Gpu> {
    match Gpu::new(0) {
        Ok(g) => Some(g),
        Err(e) if format!("{e}") == "CUDA init" => { eprintln!("SKIP: CUDA: {e:#}"); None }
        Err(e) => panic!("{e:#}"),
    }
}

type Shape = (usize, usize, usize, usize, usize); // nh, nkv, hd, nt, pos0
fn workload(s: Shape) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, f32) {
    let (nh, nkv, hd, t, pos0) = s;
    let n = (pos0 + t) * nkv * hd;
    let k = xs(n);
    let v = xs(n + 5)[5..].iter().map(|x| 3.0 * x).collect::<Vec<_>>();
    let q: Vec<f32> = xs(t * nh * hd + 3)[3..].iter().map(|v| 4.0 * v).collect();
    let qg = xs(t * nh * 2 * hd);
    (k, v, q, qg, 1.0 / (hd as f32).sqrt())
}

fn run(vk: &Vk, s: Shape) -> (Vec<u16>, Vec<u16>, Vec<f32>, Vec<f32>) {
    let (nh, nkv, hd, t, pos0) = s;
    let (k, v, q, qg, scale) = workload(s);
    let n = (pos0 + t) * nkv * hd;
    let g: &dyn Backend = vk;
    let (kc, vc) = (g.zeros::<u16>(g.kv_words(n)).unwrap(), g.zeros::<u16>(g.kv_words_v(n)).unwrap());
    let zero = up(g, &[0i32]);
    g.kv_store(up(g, &k).addr(), up(g, &v).addr(), kc.addr(), vc.addr(), nkv * hd, pos0 + t, zero.addr(), hd).unwrap();
    let (dq, dg) = (up(g, &q), up(g, &qg));
    let dout = g.zeros::<f32>(t * nh * hd).unwrap();
    g.attn_prefill(dq.addr(), kc.addr(), vc.addr(), dg.addr(), dout.addr(), up(g, &[pos0 as i32]).addr(),
                   t, nh, nkv, hd, scale).unwrap();
    let steps: Vec<i32> = (0..t).flat_map(|i| [(pos0 + i) as i32, 0]).collect();
    let (dd, part) = (g.zeros::<f32>(t * nh * hd).unwrap(), g.zeros::<f32>(t * nh * ATT_SPLITS * (hd + 2)).unwrap());
    g.attn_decode_gq(dq.addr(), kc.addr(), vc.addr(), dg.addr(), dd.addr(), part.addr(), up(g, &steps).addr(),
                     nh, nkv, hd, scale, ATT_GQA_MIN, t, 0, 0).unwrap();
    (down::<u16>(g, &kc), down::<u16>(g, &vc), down::<f32>(g, &dout), down::<f32>(g, &dd))
}

fn cu_caches(k: &str, v: &str, s: Shape) -> (Vec<u16>, Vec<u16>) {
    set(k, v);
    let Some(g) = open_cu() else { return (vec![], vec![]) };
    let (kv, vv, _, _, _) = workload(s);
    let n = (s.4 + s.3) * s.1 * s.2;
    let g: &dyn Backend = &g;
    let (kc, vc) = (g.zeros::<u16>(g.kv_words(n)).unwrap(), g.zeros::<u16>(g.kv_words_v(n)).unwrap());
    let zero = up(g, &[0i32]);
    g.kv_store(up(g, &kv).addr(), up(g, &vv).addr(), kc.addr(), vc.addr(), s.1 * s.2, s.4 + s.3, zero.addr(), s.2).unwrap();
    (down::<u16>(g, &kc), down::<u16>(g, &vc))
}

#[test]
fn vulkan_split_legacy_and_turbo() {
    let _lock = common::gpu_lock();
    let shape: Shape = (4, 2, 128, 16, 3);
    let n = (shape.4 + shape.3) * shape.1 * shape.2;
    // Symmetric legacy formats through the split K/V environment stay byte-identical to CUDA.
    for mode in ["q8", "q4", "q2"] {
        set(mode, mode);
        let Some(vk) = open_vk() else { continue };
        let (kc, vc, _, _) = run(&vk, shape);
        let (ckc, cvc) = cu_caches(mode, mode, shape);
        assert_eq!((kc, vc), (ckc, cvc), "vulkan {mode}/{mode} cache bytes differ from CUDA");
    }
    // Mixed legacy halves get their own layouts.
    set("q4", "q2");
    let Some(vk) = open_vk() else { return };
    assert_eq!((vk.kv_words(n), vk.kv_words_v(n)), (n * 9 / 32, n * 3 / 16));
    let (kc, vc, _, _) = run(&vk, shape);
    let (ckc, cvc) = cu_caches("q4", "q2", shape);
    assert_eq!((kc, vc), (ckc, cvc), "vulkan K=q4 V=q2 cache bytes differ from CUDA");
    // Turbo4: same cache size as q4, prefill and decode stay consistent.
    set("turbo4", "turbo4");
    let Some(vk) = open_vk() else { return };
    assert_eq!(vk.kv_words(n), n * 9 / 32);
    let (_, _, pf, dec) = run(&vk, shape);
    eprintln!("vulkan turbo4 prefill vs decode {:.2e}", rel_l2(&pf, &dec));
    assert!(rel_l2(&pf, &dec) < 5e-4, "vulkan turbo4 prefill vs decode");
    // Mixed turbo K / f16 V: the f16 half stays byte-identical and both halves stay consistent.
    set("turbo4", "f16");
    let Some(vk) = open_vk() else { return };
    let (_, vc, pf, dec) = run(&vk, shape);
    let (_, cvc) = cu_caches("f16", "f16", shape);
    assert_eq!(vc, cvc, "vulkan V=f16 half differs from CUDA with a turbo K half");
    eprintln!("vulkan mixed turbo4/f16 prefill vs decode {:.2e}", rel_l2(&pf, &dec));
    assert!(rel_l2(&pf, &dec) < 5e-4, "vulkan mixed prefill vs decode");
    // Mixed f16 K / turbo V: the K half stays byte-identical and both halves stay consistent.
    set("f16", "turbo4");
    let Some(vk) = open_vk() else { return };
    let (kc, _, pf, dec) = run(&vk, shape);
    let (ckc, _) = cu_caches("f16", "f16", shape);
    assert_eq!(kc, ckc, "vulkan K=f16 half differs from CUDA with a turbo V half");
    eprintln!("vulkan mixed f16/turbo4 prefill vs decode {:.2e}", rel_l2(&pf, &dec));
    assert!(rel_l2(&pf, &dec) < 5e-4, "vulkan mixed prefill vs decode");
}

/// Inkling 2 uses head_dim 64, where TurboQuant needs WHT-64 on both store and attention.
#[test]
fn vulkan_inkling_turbo_hd64() {
    let _lock = common::gpu_lock();
    set("turbo4", "turbo4");
    let Some(vk) = open_vk() else { return };
    let (nh, nkv, hd, nt, pos0, ring) = (4usize, 2usize, 64usize, 6usize, 3usize, 16usize);
    let (nr, ext) = (16usize, 8usize);
    let g: &dyn Backend = &vk;
    let q = xs(nt * nh * hd);
    let r = xs(nt * nh * nr);
    let sink = xs(nt * nh);
    let gate = xs(nt * nh * hd);
    let bias = xs(ext * nr);
    let k_in = xs(nt * nkv * hd);
    let v_in = xs(nt * nkv * hd + 7)[7..].iter().map(|x| 0.5 * x).collect::<Vec<_>>();
    let kc = g.zeros::<u16>(g.kv_words(nkv * hd * ring)).unwrap();
    let vc = g.zeros::<u16>(g.kv_words_v(nkv * hd * ring)).unwrap();
    g.ink_clip_n(up(g, &k_in).addr(), 0, hd, nkv, nt, 8.0, 0.0, 1.0, kc.addr(), pos0, ring, 0, 0.0, 1.0).unwrap();
    g.ink_clip_n_v(up(g, &v_in).addr(), 0, hd, nkv, nt, 0.0, 1e-6, 1.0, vc.addr(), pos0, ring, 0, 0.0, 1.0).unwrap();
    let (dq, dr, ds, dg, db) = (up(g, &q), up(g, &r), up(g, &sink), up(g, &gate), up(g, &bias));
    let dout_pre = g.zeros::<f32>(nt * nh * hd).unwrap();
    g.ink_attn_prefill(dq.addr(), dr.addr(), ds.addr(), dg.addr(), kc.addr(), vc.addr(), 0, 0, db.addr(),
                       dout_pre.addr(), pos0, nt, nh, nkv, hd, nr, ext, ring, 0).unwrap();
    let dsc = g.zeros::<f32>(nh * ring.div_ceil(64) * (hd + 2)).unwrap();
    let mut want = Vec::with_capacity(nt * nh * hd);
    for tok in 0..nt {
        let at = pos0 + tok;
        let dout = g.zeros::<f32>(nh * hd).unwrap();
        g.ink_attn(dq.addr() + (tok * nh * hd * 4) as u64, dr.addr() + (tok * nh * nr * 4) as u64,
                   ds.addr() + (tok * nh * 4) as u64, dg.addr() + (tok * nh * hd * 4) as u64,
                   kc.addr(), vc.addr(), db.addr(), dsc.addr(), dout.addr(), at + 1, at, nh, nkv, hd, nr, ext).unwrap();
        want.extend_from_slice(&down::<f32>(g, &dout));
    }
    let got = down::<f32>(g, &dout_pre);
    eprintln!("vulkan turbo hd64 prefill vs decode {:.2e}", rel_l2(&got, &want));
    assert!(rel_l2(&got, &want) < 5e-4, "vulkan turbo hd64 prefill vs decode");
}
