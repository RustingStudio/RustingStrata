//! cargo run --release --example host_attn_bench -- [positions]: one CPU decode-attention call over a q4 cache.
use rusting_strata::host_attn::{HostAttn, run};
fn main() {
    let n: usize = std::env::args().nth(1).map_or(52_000, |s| s.parse().unwrap());
    let (nh, nkv, hd) = (24, 4, 256);
    let rb = hd / 2 + hd / 16;
    let mut seed = 1u32;
    let mut rnd = || { seed = seed.wrapping_mul(1664525).wrapping_add(1013904223); seed };
    let mut kc = vec![0u8; n * nkv * rb];
    let mut vc = vec![0u8; n * nkv * rb];
    for (k, v) in kc.iter_mut().zip(vc.iter_mut()) { *k = rnd() as u8; *v = rnd() as u8; }
    for r in 0..n * nkv { for b in 0..hd / 32 { let o = r * rb + hd / 2 + 2 * b; let d = half::f16::from_f32(0.05).to_le_bytes(); kc[o..o + 2].copy_from_slice(&d); vc[o..o + 2].copy_from_slice(&d); } }
    let nt: usize = std::env::args().nth(2).map_or(1, |s| s.parse().unwrap());
    let q1: Vec<f32> = (0..nh * hd).map(|_| (rnd() % 1000) as f32 / 1000.0 - 0.5).collect();
    let q: Vec<f32> = (0..nt).flat_map(|_| q1.clone()).collect();
    let step: Vec<i32> = (0..nt).flat_map(|t| [(n - nt + t) as i32, 0]).collect();
    let mut part = vec![0f32; nt * nh * (hd + 2)];
    let a = HostAttn { q: q.as_ptr() as usize, step: step.as_ptr() as usize, part: part.as_mut_ptr() as usize, kc: kc.as_ptr() as usize, vc: vc.as_ptr() as usize, nh, nkv, hd, nt, scale: 1.0 / 16.0, mode: 2 };
    run(&a);
    let t = std::time::Instant::now();
    let gap: u64 = std::env::var("GAP_US").ok().map_or(0, |s| s.parse().unwrap());
    let mut busy = 0.0;
    for _ in 0..20 { if gap > 0 { std::thread::sleep(std::time::Duration::from_micros(gap)); } let t1 = std::time::Instant::now(); run(&a); busy += t1.elapsed().as_secs_f64(); }
    let _ = t; let ms = busy * 1e3 / 20.0;
    let gb = (2 * n * nkv * rb) as f64 / 1e9;
    println!("{n} positions: {ms:.2} ms/call, {:.1} GB/s, x16 layers = {:.1} ms/token; check {:.6} {:.6}", gb / ms * 1e3, ms * 16.0, part[(nt - 1) * nh * (hd + 2)], part[(nt - 1) * nh * (hd + 2) + hd + 1]);
}
