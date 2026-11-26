//! The tiled controller path (`STRATA_DS_FAST_KERNELS=1`, `nt >= 64`): `ink_ctrl_n` must match the per-token
//! `ink_ctrl` for every token. Separate test binary so the env is isolated.
mod common;
use common::{down, up};
use rusting_strata::backend::Backend;

fn xs(n: usize) -> Vec<f32> { (0..n).map(|i| ((i as f32 * 0.7548777).fract() - 0.5) * 2.0).collect() }

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f32 = b.iter().map(|y| y * y).sum();
    (num / den.max(1e-30)).sqrt()
}

#[test]
fn tiled_controllers_match_per_token() {
    unsafe { std::env::set_var("STRATA_DS_FAST_KERNELS", "1") };
    for (bk, gpu) in common::backends() {
        let gpu = &*gpu;
        if bk != "cuda" { continue; }
        let (rank, count, n, nt) = (4usize, 3usize, 128usize, 64usize);
        let down_w: Vec<u16> = xs(count * rank * 2 * n + 7)[7..].iter().map(|&v| half::f16::from_f32(v * 0.25).to_bits()).collect();
        let gain_w: Vec<u16> = xs(count * rank * n + 13)[13..].iter().map(|&v| half::f16::from_f32(v * 0.25).to_bits()).collect();
        let forget_w: Vec<u16> = xs(count * rank * n + 19)[19..].iter().map(|&v| half::f16::from_f32(v * 0.25).to_bits()).collect();
        let msg = xs(count * nt * n + 23)[23..].to_vec();
        for nins in [1usize, count] {
            let cur_c = xs(nins * nt * n + 31)[31..].to_vec();
            let (dd, dg, df) = (up(gpu, &down_w), up(gpu, &gain_w), up(gpu, &forget_w));
            let mut want = vec![0.0f32; count * nt * n];
            let dgate1 = gpu.zeros::<f32>(count * rank).unwrap();
            for tok in 0..nt {
                let mut c1 = Vec::with_capacity(nins * n);
                for j in 0..nins { c1.extend_from_slice(&cur_c[(j * nt + tok) * n..(j * nt + tok + 1) * n]); }
                let mut m1 = Vec::with_capacity(count * n);
                for j in 0..count { m1.extend_from_slice(&msg[(j * nt + tok) * n..(j * nt + tok + 1) * n]); }
                let (dc1, dm1, do1) = (up(gpu, &c1), up(gpu, &m1), gpu.zeros::<f32>(count * n).unwrap());
                gpu.ink_ctrl(dd.addr(), dg.addr(), df.addr(), dc1.addr(), nins, dm1.addr(), dgate1.addr(), do1.addr(), n, rank, count).unwrap();
                let o1 = down::<f32>(gpu, &do1);
                for j in 0..count {
                    want[(j * nt + tok) * n..(j * nt + tok + 1) * n].copy_from_slice(&o1[j * n..(j + 1) * n]);
                }
            }
            let (dcc, dmm, dgg, doo) = (up(gpu, &cur_c), up(gpu, &msg),
                gpu.zeros::<f32>(nt * count * rank).unwrap(), gpu.zeros::<f32>(count * nt * n).unwrap());
            gpu.ink_ctrl_n(dd.addr(), dg.addr(), df.addr(), dcc.addr(), nins, dmm.addr(), dgg.addr(), doo.addr(), n, rank, count, nt).unwrap();
            let got = down::<f32>(gpu, &doo);
            assert!(rel_l2(&got, &want) < 1e-6, "{bk} tiled ink_ctrl_n nins={nins} nt={nt}");
        }
    }
}
