//! TurboQuant KV on CUDA. Its own process: the kernel module is compiled per KV format.
//!
//! The first test pins the legacy symmetric modes to the exact cache bytes, prefill and decode
//! produced by main at bbd57c2 (FNV-1a over little-endian u16/f32). Every legacy path touched by
//! this PR must stay bit-identical; the TurboQuant and mixed K/V paths are separate opt-in modes.
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
fn hash32(v: &[f32]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for x in v { for b in x.to_bits().to_le_bytes() { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); } }
    h
}
fn hash16(v: &[u16]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for x in v { for b in x.to_le_bytes() { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); } }
    h
}
fn set_kv(k: &str, v: &str) {
    unsafe {
        std::env::set_var("STRATA_KV_K", k);
        std::env::set_var("STRATA_KV_V", v);
        std::env::set_var("STRATA_KV", if k == v { k } else { "q4" });
        if k == "q8" || v == "q8" { std::env::set_var("STRATA_KV_Q8", "1"); }
        else { std::env::remove_var("STRATA_KV_Q8"); }
    }
}

/// Cache bytes, prefill output and decode output of f16/q8/q4/q2 for four (nh, nkv, hd) shapes.
/// Each line is `mode nh nkv hd k_hash v_hash prefill_hash decode_hash` from main @ bbd57c2.
const MAIN_REFERENCE: &str = "
f16 4 2 64 78caa2a80cfa10a7 3bcfc21757d78a23 a57ee46c4c639bd3 d4f04a7b0c6df80e
f16 4 2 128 dfe2171c148e789d 72fae506d01ba632 033f81eeb95d8145 86d7f667d304a145
f16 16 2 256 68d7d4e7121fbe38 b3e45897fef7ff50 7911ff6714a6dab2 a6f0fb3a2221aeb1
f16 8 8 128 67e29b12aaed3c94 b9f0dc213da4b775 93f3d16615408ac7 10be884b38eb6e74
q8 4 2 64 07c54db5ed94d112 c0a2ed71a66247cc bae90aa5312fc525 49849c15dfae5f13
q8 4 2 128 80fdcbf78047e4ec df3442477362b960 2e97b83efe2ffa6f f43403b2a477cb12
q8 16 2 256 ff33d70bf833ee97 edb6be75ad286544 03bf49248f78af9c d2eb1f691c815c94
q8 8 8 128 15a5ccf186cb82d9 5c11745ba63b4da5 55d3c2630b1a09ed 7e04804514989de6
q4 4 2 64 447ab262ea711901 febb164977c324d8 25e651f5f048a76d 2d4375e1dc829176
q4 4 2 128 a6b6fcf0e15d57a8 ba27be594dc54f3e 5978942beda784e1 6d2c4003a55f8a07
q4 16 2 256 3dbc191cad296f5f 600b6004f27b4e30 c3f89deea836e994 bf88f3f3afdd6624
q4 8 8 128 9571d3e72332fff4 ee12b4710fa007d3 81309dc3d750f1f6 df16ce6d0771100a
q2 4 2 64 bd0f840e53ce64de c8e5ccef19e480df 417fe5dd5658e90d c2c8598377d823c2
q2 4 2 128 03eb270888f1d205 bcd33f0e028c0824 867f0f6b1e0e8ab6 3ce515d6cf41f37a
q2 16 2 256 36a62ff1eb5ea0bb 2035cf12e1db2f42 e6f84b45cd7effa1 7520251b958cdb93
q2 8 8 128 f1030ecd713b642f 6e9a5fca2f6bd630 88fed7b50798c9ee 8be7f5e03672026e
";

fn open(k: &str, v: &str) -> Option<Gpu> {
    set_kv(k, v);
    match Gpu::new(0) {
        Ok(g) => Some(g),
        Err(e) if format!("{e}") == "CUDA init" => { eprintln!("SKIP: no CUDA device: {e:#}"); None }
        Err(e) => panic!("{e:#}"),
    }
}

/// One deterministic workload shared by the bit-identity and the TurboQuant checks.
fn run(g: &Gpu, nh: usize, nkv: usize, hd: usize, t: usize, pos0: usize) -> (Vec<u16>, Vec<u16>, Vec<f32>, Vec<f32>) {
    let n = (pos0 + t) * nkv * hd;
    let k = xs(n);
    let v = xs(n + 5)[5..].iter().map(|x| 3.0 * x).collect::<Vec<_>>();
    let q: Vec<f32> = xs(t * nh * hd + 3)[3..].iter().map(|x| 4.0 * x).collect();
    let qg = xs(t * nh * 2 * hd);
    let scale = 1.0 / (hd as f32).sqrt();
    let g: &dyn Backend = g;
    let (kc_n, vc_n) = (g.kv_words(n), g.kv_words_v(n));
    let (dk, dv) = (up(g, &k), up(g, &v));
    let (kc, vc) = (g.zeros::<u16>(kc_n).unwrap(), g.zeros::<u16>(vc_n).unwrap());
    let zero = up(g, &[0i32]);
    g.kv_store(dk.addr(), dv.addr(), kc.addr(), vc.addr(), nkv * hd, pos0 + t, zero.addr(), hd).unwrap();
    let (dq, dg) = (up(g, &q), up(g, &qg));
    let step = up(g, &[pos0 as i32]);
    let dout = g.zeros::<f32>(t * nh * hd).unwrap();
    g.attn_prefill(dq.addr(), kc.addr(), vc.addr(), dg.addr(), dout.addr(), step.addr(), t, nh, nkv, hd, scale).unwrap();
    let steps: Vec<i32> = (0..t).flat_map(|i| [(pos0 + i) as i32, 0]).collect();
    let ds = up(g, &steps);
    let (dd, part) = (g.zeros::<f32>(t * nh * hd).unwrap(), g.zeros::<f32>(t * nh * ATT_SPLITS * (hd + 2)).unwrap());
    g.attn_decode_gq(dq.addr(), kc.addr(), vc.addr(), dg.addr(), dd.addr(), part.addr(), ds.addr(),
                     nh, nkv, hd, scale, ATT_GQA_MIN, t, 0, 0).unwrap();
    (down::<u16>(g, &kc), down::<u16>(g, &vc), down::<f32>(g, &dout), down::<f32>(g, &dd))
}

#[test]
fn turboquant_kv_modes() {
    let _gpu = common::gpu_lock();
    let shapes = [(4, 2, 64), (4, 2, 128), (16, 2, 256), (8, 8, 128)];
    let (t, pos0) = (32, 5);
    // 1. Legacy modes: byte-for-byte the same as main. A change to any shared prefill/decode/store path fails here.
    for kv in ["f16", "q8", "q4", "q2"] {
        let Some(g) = open(kv, kv) else { continue };
        let rows: Vec<_> = MAIN_REFERENCE.lines().filter(|l| l.starts_with(kv)).collect();
        for (line, (nh, nkv, hd)) in rows.iter().zip(shapes) {
            let f: Vec<_> = line.split_whitespace().collect();
            assert_eq!((f[1].parse::<usize>().unwrap(), f[2].parse::<usize>().unwrap(), f[3].parse::<usize>().unwrap()), (nh, nkv, hd));
            let (kc, vc, pf, dec) = run(&g, nh, nkv, hd, t, pos0);
            let got = [hash16(&kc), hash16(&vc), hash32(&pf), hash32(&dec)];
            for (j, want) in f[4..].iter().enumerate() {
                assert_eq!(format!("{:016x}", got[j]), *want, "{kv} nh {nh} nkv {nkv} hd {hd} field {j} differs from main");
            }
        }
    }
    // 2. Turbo4 occupies the same cache as q4 and stays within its own rounding error of f16.
    let Some(g16) = open("f16", "f16") else { return };
    let Some(gq4) = open("q4", "q4") else { return };
    let Some(gt4) = open("turbo4", "turbo4") else { return };
    for &(nh, nkv, hd) in shapes.iter().filter(|&&(_, _, hd)| hd % 128 == 0) {
        let n = (pos0 + t) * nkv * hd;
        assert_eq!(gt4.kv_words(n), gq4.kv_words(n), "turbo4 and q4 must use the same cache size");
        let (_, _, pf16, dec16) = run(&g16, nh, nkv, hd, t, pos0);
        let (_, _, _, dec4) = run(&gq4, nh, nkv, hd, t, pos0);
        let (_, _, pft, dect) = run(&gt4, nh, nkv, hd, t, pos0);
        eprintln!("nh {nh} nkv {nkv} hd {hd}: turbo4 vs f16 decode {:.2e}, prefill {:.2e}, prefill-vs-decode {:.2e}; f16 pf-vs-dec {:.2e}; q4 vs f16 decode {:.2e}",
                  rel_l2(&dect, &dec16), rel_l2(&pft, &pf16), rel_l2(&pft, &dect), rel_l2(&pf16, &dec16), rel_l2(&dec4, &dec16));
        assert!(rel_l2(&pft, &dect) < 5e-4, "turbo4 prefill vs decode nh {nh} nkv {nkv} hd {hd}");
        assert!(rel_l2(&dect, &dec16) < 0.9, "turbo4 decode nh {nh} nkv {nkv} hd {hd}");
        assert!(rel_l2(&pft, &dec16) < 0.9, "turbo4 prefill vs decode nh {nh} nkv {nkv} hd {hd}");
    }
    // 3. Mixed halves are sized independently: only the K half changes when --k changes.
    let Some(gmix) = open("turbo4", "f16") else { return };
    let n = 4096;
    assert_eq!((gmix.kv_words(n), gmix.kv_words_v(n)), (gt4.kv_words(n), g16.kv_words(n)));
}
