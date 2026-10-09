//! `--kv q8`: the KV cache stores q8_0 blocks. Its own process, since the kernel module is compiled per KV format.
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
/// decode as close as with f16, and q8 batched decode bit-identical to single-token decode.
#[test]
fn q8_kv_attention_matches_f16() {
    let mk = |q8: bool| {
        if q8 { unsafe { std::env::set_var("STRATA_KV_Q8", "1") } } else { unsafe { std::env::remove_var("STRATA_KV_Q8") } }
        match Gpu::new() {
            Ok(g) => Some(g),
            Err(e) if format!("{e}") == "CUDA init" => { eprintln!("SKIP: no GPU: {e:#}"); None }
            Err(e) => panic!("{e:#}"),
        }
    };
    let Some(g8) = mk(true) else { return };
    let words = rusting_strata::cuda::kv_words(64);
    let gf = mk(false).unwrap();
    assert_eq!(words, 34);
    let (pos0, t) = (5, 100);
    for (nh, nkv, hd) in [(4, 2, 64), (4, 2, 128), (16, 2, 256), (8, 8, 128)] {
        let n = (pos0 + t) * nkv * hd;
        let (k, v) = (xs(n), xs(n + 5)[5..].iter().map(|x| 3.0 * x).collect::<Vec<_>>());
        let q: Vec<f32> = xs(t * nh * hd + 3)[3..].iter().map(|v| 4.0 * v).collect();
        let qg = xs(t * nh * 2 * hd);
        let scale = 1.0 / (hd as f32).sqrt();
        // (prefill, decode) outputs of all t query tokens.
        let run = |g: &Gpu, cache: usize| {
            let (dk, dv) = (g.stream.clone_htod(&k).unwrap(), g.stream.clone_htod(&v).unwrap());
            let (kc, vc) = (g.zeros::<u16>(cache).unwrap(), g.zeros::<u16>(cache).unwrap());
            let zero = g.stream.clone_htod(&[0i32]).unwrap();
            g.kv_store(g.ptr(&dk), g.ptr(&dv), g.ptr(&kc), g.ptr(&vc), nkv * hd, pos0 + t, g.ptr(&zero), hd).unwrap();
            let (dq, dg) = (g.stream.clone_htod(&q).unwrap(), g.stream.clone_htod(&qg).unwrap());
            let step = g.stream.clone_htod(&[pos0 as i32]).unwrap();
            let dout = g.zeros::<f32>(t * nh * hd).unwrap();
            g.attn_prefill(g.ptr(&dq), g.ptr(&kc), g.ptr(&vc), g.ptr(&dg), g.ptr(&dout), g.ptr(&step), t, nh, nkv, hd, scale).unwrap();
            let mut dec = vec![];
            for gq_min in [ATT_GQA_MIN, 0] {
                let steps: Vec<i32> = (0..t).flat_map(|i| [(pos0 + i) as i32, 0]).collect();
                let ds = g.stream.clone_htod(&steps).unwrap();
                let (dd, part) = (g.zeros::<f32>(t * nh * hd).unwrap(), g.zeros::<f32>(t * nh * ATT_SPLITS * (hd + 2)).unwrap());
                g.attn_decode_gq(g.ptr(&dq), g.ptr(&kc), g.ptr(&vc), g.ptr(&dg), g.ptr(&dd), g.ptr(&part), g.ptr(&ds),
                                 nh, nkv, hd, scale, gq_min, t).unwrap();
                let batched = g.stream.clone_dtoh(&dd).unwrap();
                // Last token alone gives the same bits as in the batch.
                let i = t - 1;
                let s = g.stream.clone_htod(&[(pos0 + i) as i32, 0]).unwrap();
                let off = |p: u64, per: usize| p + (i * per * 4) as u64;
                g.attn_decode_gq(off(g.ptr(&dq), nh * hd), g.ptr(&kc), g.ptr(&vc), off(g.ptr(&dg), nh * 2 * hd),
                                 off(g.ptr(&dd), nh * hd), g.ptr(&part), g.ptr(&s), nh, nkv, hd, scale, gq_min, 1).unwrap();
                assert_eq!(g.stream.clone_dtoh(&dd).unwrap(), batched, "batched decode nh {nh} nkv {nkv} hd {hd} gq_min {gq_min}");
                dec.push(batched);
            }
            (g.stream.clone_dtoh(&dout).unwrap(), dec)
        };
        let (pf, df) = run(&gf, n);
        let (p8, d8) = run(&g8, n * 17 / 32);
        for (j, (a, b)) in d8.iter().zip(&df).enumerate() {
            let e = rel_l2(a, b);
            eprintln!("nh {nh} nkv {nkv} hd {hd} kernel {j}: q8 vs f16 decode {e:.2e}, q8 prefill vs decode {:.2e}", rel_l2(&p8, a));
            assert!(e < 1e-2, "q8 vs f16 decode nh {nh} nkv {nkv} hd {hd}: {e}");
            assert!(rel_l2(&p8, a) <= 5e-4, "q8 prefill vs decode nh {nh} nkv {nkv} hd {hd}");
        }
        assert!(rel_l2(&p8, &pf) < 1e-2);
    }
}
