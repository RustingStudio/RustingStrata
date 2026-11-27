//! The fast prefill path (`STRATA_DS_FAST_KERNELS=1`): `kp16` must cover every 16-token tile, including the
//! fallback for a relative-bias width it cannot read as float4. Separate test binary so the env is isolated.
mod common;
use common::{down, up};
use rusting_strata::backend::Backend;

fn xs(n: usize) -> Vec<f32> { (0..n).map(|i| ((i as f32 * 0.7548777).fract() - 0.5) * 2.0).collect() }

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f32 = b.iter().map(|y| y * y).sum();
    (num / den.max(1e-30)).sqrt()
}

/// The 16-query prefill entry point is launched with a 16-token grid. When the relative-bias width does not fit its
/// four-query fallback (`nr > 32` or `nr % 4 != 0`), the wrapper must still cover all 16 tokens: one call to the
/// four-query kernel leaves tokens 4..15 of every tile unwritten. `nr=28` is the fast path, `nr=30` the tile fallback
/// and `nr=40` the wide-bias fallback (which also makes decode use the scalar kernel).
#[test]
fn ink_attn_prefill_kp16_fallback_covers_every_query() {
    unsafe { std::env::set_var("STRATA_DS_FAST_KERNELS", "1") };
    for (bk, gpu) in common::backends() {
        let gpu = &*gpu;
        // kp16 is the CUDA f16-cache path; Vulkan has its own prefill, and a quantized KV or the measurement knobs
        // select another kernel.
        if bk != "cuda"
            || std::env::var_os("STRATA_KV").is_some()
            || std::env::var_os("STRATA_DS_ATTN_OLD").is_some()
            || std::env::var_os("STRATA_DS_ATTN_KP4").is_some() { continue; }
        for nr in [28usize, 30, 40] {
            let (nh, nkv, hd, ext, nt) = (2usize, 1usize, 128usize, 8usize, 20usize);
            let q = xs(nt * nh * hd + 3)[3..].to_vec();
            let r = xs(nt * nh * nr + 5)[5..].to_vec();
            let sink = xs(nt * nh + 7)[7..].to_vec();
            let gate = xs(nt * nh * hd + 11)[11..].to_vec();
            let k = xs(nt * nkv * hd + 13)[13..].to_vec();
            let v = xs(nt * nkv * hd + 17)[17..].to_vec();
            let bias = xs(ext * nr + 19)[19..].to_vec();
            let kc: Vec<u16> = k.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
            let vc: Vec<u16> = v.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
            let (dq, dr, dsink, dgate) = (up(gpu, &q), up(gpu, &r), up(gpu, &sink), up(gpu, &gate));
            let (dkc, dvc, dbias) = (up(gpu, &kc), up(gpu, &vc), up(gpu, &bias));
            let dout = gpu.zeros::<f32>(nt * nh * hd).unwrap();
            gpu.ink_attn_prefill(
                dq.addr(), dr.addr(), dsink.addr(), dgate.addr(), dkc.addr(), dvc.addr(), 0, 0, dbias.addr(), dout.addr(),
                0, nt, nh, nkv, hd, nr, ext, nt, 0,
            ).unwrap();
            let dsc = gpu.zeros::<f32>(nh * nt.div_ceil(64) * (hd + 2)).unwrap();
            let mut want = Vec::with_capacity(nt * nh * hd);
            for tok in 0..nt {
                let dout1 = gpu.zeros::<f32>(nh * hd).unwrap();
                gpu.ink_attn(
                    dq.addr() + (tok * nh * hd * 4) as u64,
                    dr.addr() + (tok * nh * nr * 4) as u64,
                    dsink.addr() + (tok * nh * 4) as u64,
                    dgate.addr() + (tok * nh * hd * 4) as u64,
                    dkc.addr(), dvc.addr(), dbias.addr(), dsc.addr(), dout1.addr(),
                    tok + 1, tok, nh, nkv, hd, nr, ext,
                ).unwrap();
                want.extend_from_slice(&down::<f32>(gpu, &dout1));
            }
            let got = down::<f32>(gpu, &dout);
            assert!(rel_l2(&got, &want) < 1e-6, "{bk} ink_attn_prefill kp16 fallback nr={nr} nt={nt}");
        }
    }
}
