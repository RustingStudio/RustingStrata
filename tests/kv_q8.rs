//! `--kv q8`: the KV cache stores q8_0 blocks. Its own process, since the kernel module is compiled per KV format.
mod common;
use common::{down, up};
use rusting_strata::backend::Backend;
use rusting_strata::cuda::{ATT_GQA_MIN, ATT_SPLITS, Gpu};

fn xs(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 * 0.7548777).fract() - 0.5) * 2.0).collect()
}

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f32 = b.iter().map(|y| y * y).sum();
    (num / den.max(1e-30)).sqrt()
}

/// Prefill and decode attention over a cache filled by kv_store: q8 vs f16 within q8 rounding, q8 prefill vs q8
/// decode as close as with f16, q8 batched decode bit-identical to single-token decode, and Vulkan q8 bit-identical
/// to CUDA q8 (cache bytes, decode, and prefill with cooperative matrices).
#[test]
fn q8_kv_attention_matches_f16() {
    let mk = |q8: bool| {
        if q8 { unsafe { std::env::set_var("STRATA_KV_Q8", "1") } } else { unsafe { std::env::remove_var("STRATA_KV_Q8") } }
        match Gpu::new(0) {
            Ok(g) => Some(g),
            Err(e) if format!("{e}") == "CUDA init" => { eprintln!("SKIP: no GPU: {e:#}"); None }
            Err(e) => panic!("{e:#}"),
        }
    };
    let Some(g8) = mk(true) else { return };
    let vk8 = match rusting_strata::vulkan::Vk::new(0) {
        Ok(v) => Some(v),
        Err(e) if format!("{e}") == "Vulkan init" => { eprintln!("SKIP: vulkan: {e:#}"); None }
        Err(e) => panic!("{e:#}"),
    };
    let gf = mk(false).unwrap();
    assert_eq!((g8.kv_words(64), gf.kv_words(64)), (34, 64));
    let (pos0, t) = (5, 100);
    for (nh, nkv, hd) in [(4, 2, 64), (4, 2, 128), (16, 2, 256), (8, 8, 128)] {
        let n = (pos0 + t) * nkv * hd;
        let (k, v) = (xs(n), xs(n + 5)[5..].iter().map(|x| 3.0 * x).collect::<Vec<_>>());
        let q: Vec<f32> = xs(t * nh * hd + 3)[3..].iter().map(|v| 4.0 * v).collect();
        let qg = xs(t * nh * 2 * hd);
        let scale = 1.0 / (hd as f32).sqrt();
        // (prefill, decode) outputs of all t query tokens, and the K cache.
        let run = |g: &dyn Backend| {
            let cache = g.kv_words(n);
            let (dk, dv) = (up(g, &k), up(g, &v));
            let (kc, vc) = (g.zeros::<u16>(cache).unwrap(), g.zeros::<u16>(cache).unwrap());
            let zero = up(g, &[0i32]);
            g.kv_store(dk.addr(), dv.addr(), kc.addr(), vc.addr(), nkv * hd, pos0 + t, zero.addr(), hd).unwrap();
            let (dq, dg) = (up(g, &q), up(g, &qg));
            let step = up(g, &[pos0 as i32]);
            let dout = g.zeros::<f32>(t * nh * hd).unwrap();
            g.attn_prefill(dq.addr(), kc.addr(), vc.addr(), dg.addr(), dout.addr(), step.addr(), t, nh, nkv, hd, scale).unwrap();
            let mut dec = vec![];
            for gq_min in [ATT_GQA_MIN, 0] {
                let steps: Vec<i32> = (0..t).flat_map(|i| [(pos0 + i) as i32, 0]).collect();
                let ds = up(g, &steps);
                let (dd, part) = (g.zeros::<f32>(t * nh * hd).unwrap(), g.zeros::<f32>(t * nh * ATT_SPLITS * (hd + 2)).unwrap());
                g.attn_decode_gq(dq.addr(), kc.addr(), vc.addr(), dg.addr(), dd.addr(), part.addr(), ds.addr(),
                                 nh, nkv, hd, scale, gq_min, t).unwrap();
                let batched = down::<f32>(g, &dd);
                // Last token alone gives the same bits as in the batch.
                let i = t - 1;
                let s = up(g, &[(pos0 + i) as i32, 0]);
                let off = |p: u64, per: usize| p + (i * per * 4) as u64;
                g.attn_decode_gq(off(dq.addr(), nh * hd), kc.addr(), vc.addr(), off(dg.addr(), nh * 2 * hd),
                                 off(dd.addr(), nh * hd), part.addr(), s.addr(), nh, nkv, hd, scale, gq_min, 1).unwrap();
                assert_eq!(down::<f32>(g, &dd), batched, "batched decode nh {nh} nkv {nkv} hd {hd} gq_min {gq_min}");
                dec.push(batched);
            }
            (down::<f32>(g, &dout), dec, down::<u16>(g, &kc))
        };
        let (pf, df, _) = run(&gf);
        let (p8, d8, c8) = run(&g8);
        for (j, (a, b)) in d8.iter().zip(&df).enumerate() {
            let e = rel_l2(a, b);
            eprintln!("nh {nh} nkv {nkv} hd {hd} kernel {j}: q8 vs f16 decode {e:.2e}, q8 prefill vs decode {:.2e}", rel_l2(&p8, a));
            assert!(e < 1e-2, "q8 vs f16 decode nh {nh} nkv {nkv} hd {hd}: {e}");
            assert!(rel_l2(&p8, a) <= 5e-4, "q8 prefill vs decode nh {nh} nkv {nkv} hd {hd}");
        }
        assert!(rel_l2(&p8, &pf) < 1e-2);
        if let Some(vk) = &vk8 {
            let (pv, dv, cv) = run(vk);
            assert!(cv == c8, "vulkan q8 cache nh {nh} nkv {nkv} hd {hd}");
            // attn_prefill_dp (no cooperative matrices) is not bit-identical to CUDA's prefill, f16 cache or not.
            assert!(pv == p8 || rel_l2(&pv, &dv[0]) <= 5e-4, "vulkan q8 prefill nh {nh} nkv {nkv} hd {hd}");
            assert!(dv == d8, "vulkan q8 decode nh {nh} nkv {nkv} hd {hd}");
        }
    }
}
