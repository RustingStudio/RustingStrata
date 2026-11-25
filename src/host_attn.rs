//! Decode attention on the CPU for KV caches in host RAM (`STRATA_KV_HOST`). The GPU copies the queries and
//! positions into mapped pinned memory, a stream host function runs this over the cache rows (the CPU reads RAM
//! far faster than the GPU does over PCIe), and `attn_merge` turns the one split per head into the output.
use rayon::prelude::*;

/// Pointers are host addresses of mapped pinned memory (the same as their device addresses under UVA).
pub struct HostAttn {
    pub q: usize, pub step: usize, pub part: usize, pub kc: usize, pub vc: usize,
    pub nh: usize, pub nkv: usize, pub hd: usize, pub nt: usize, pub scale: f32, pub mode: u32,
}

const CHUNK: usize = 1024;

fn row_bytes(mode: u32, hd: usize) -> usize {
    match mode { 0 => 2 * hd, 1 => hd + hd / 16, _ => hd / 2 + hd / 16 }
}

/// Dequantize a cache row. q4 comes out permuted inside each 32-value block (the 16 low nibbles, then the 16
/// high ones: values 0 2 4 .. 30, then 1 3 .. 31), which keeps the unpacking contiguous; `perm` maps a value index
/// to its place, and the queries and outputs use the same order.
#[inline(always)]
fn deq(mode: u32, r: &[u8], hd: usize, out: &mut [f32]) {
    match mode {
        0 => for (o, c) in out[..hd].iter_mut().zip(r.chunks_exact(2)) { *o = half::f16::from_le_bytes([c[0], c[1]]).to_f32(); },
        1 => for b in 0..hd / 32 {
            let d = half::f16::from_le_bytes([r[hd + 2 * b], r[hd + 2 * b + 1]]).to_f32();
            let src: &[u8; 32] = r[b * 32..b * 32 + 32].try_into().unwrap();
            let dst: &mut [f32; 32] = (&mut out[b * 32..b * 32 + 32]).try_into().unwrap();
            for i in 0..32 { dst[i] = (src[i] as i8) as f32 * d; }
        },
        _ => for b in 0..hd / 32 {
            let d = half::f16::from_le_bytes([r[hd / 2 + 2 * b], r[hd / 2 + 2 * b + 1]]).to_f32();
            let src: &[u8; 16] = r[b * 16..b * 16 + 16].try_into().unwrap();
            let dst: &mut [f32; 32] = (&mut out[b * 32..b * 32 + 32]).try_into().unwrap();
            for i in 0..16 {
                dst[i] = ((src[i] & 15) as f32 - 8.0) * d;
                dst[16 + i] = ((src[i] >> 4) as f32 - 8.0) * d;
            }
        },
    }
}

#[inline(always)]
fn perm(mode: u32, i: usize) -> usize {
    if mode == 2 { let (b, w) = (i / 32, i % 32); b * 32 + if w % 2 == 0 { w / 2 } else { 16 + w / 2 } } else { i }
}

#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    // 64 lanes: four independent AVX-512 accumulators, so the FMAs do not wait on each other.
    let mut s = [0f32; 16];
    for (x, y) in a.chunks_exact(16).zip(b.chunks_exact(16)) { for i in 0..16 { s[i] += x[i] * y[i]; } }
    s.iter().sum()
}

/// e^x for x <= 0 (softmax weights), ~1e-7 relative error, branch-free so loops over it vectorize.
#[inline(always)]
fn fexp(x: f32) -> f32 {
    let x = x.max(-87.0);
    let t = x * std::f32::consts::LOG2_E;
    let fi = t.floor();
    let f = t - fi;
    let p = 1.0 + f * (0.693_147_2 + f * (0.240_226_5 + f * (0.055_504_11 + f * (0.009_618_129 + f * (0.001_333_355_8 + f * 0.000_154_035_3)))));
    f32::from_bits(((fi as i32 + 127) << 23) as u32) * p
}

/// (max, sum, acc[nh * hd] in `perm` order) of positions `p0..p1` for one token; `q` is in `perm` order.
/// All `nt` tokens at once (each cache row is read once for all of them): `q` is [token][head][hd], token t sees
/// positions up to `lim[t]`; results are [token][head].
fn chunk(a: &HostAttn, q: &[f32], lim: &[usize], p0: usize, p1: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    #[cfg(target_arch = "x86_64")]
    if a.mode == 2 && a.hd % 32 == 0 && std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
        let grp = a.nh / a.nkv;
        macro_rules! go { ($($g:literal),*; $($t:literal),*) => { go!(@g [$($g),*] [$($t),*]) };
            (@g [$($g:literal),*] $ts:tt) => { match grp { $( $g => go!(@t $g $ts), )* _ => None } };
            (@t $g:literal [$($t:literal),*]) => { match lim.len() { $( $t => Some(unsafe { avx512::chunk_q4::<$g, $t>(a, q, lim, p0, p1) }), )* _ => None } };
        }
        if let Some(r) = go!(1, 2, 4, 6, 8; 1, 2, 3, 4) { return r; }
    }
    chunk_generic(a, q, lim, p0, p1)
}

fn softmax_rows(sc: &mut [f32], nh: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut m = vec![f32::NEG_INFINITY; nh];
    let mut l = vec![0f32; nh];
    for h in 0..nh {
        let s = &mut sc[h * n..(h + 1) * n];
        m[h] = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mh = m[h];
        if mh == f32::NEG_INFINITY { s.fill(0.0); continue; }
        let mut sum = 0.0;
        for x in s.iter_mut() { *x = fexp(*x - mh); sum += *x; }
        l[h] = sum;
    }
    (m, l)
}

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::*;
    use std::arch::x86_64::*;

    /// The 32 values of q4 block `b` of row `r`, scaled: (low nibbles = even values, high = odd), as `perm` orders them.
    #[inline(always)]
    unsafe fn blk(r: *const u8, hd: usize, b: usize) -> (__m512, __m512) {
        unsafe {
            let d = _mm512_set1_ps(half::f16::from_bits((r.add(hd / 2 + 2 * b) as *const u16).read_unaligned()).to_f32());
            let x = _mm512_cvtepu8_epi32(_mm_loadu_si128(r.add(16 * b) as *const __m128i));
            let e = _mm512_set1_ps(8.0);
            let lo = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_and_si512(x, _mm512_set1_epi32(15))), e);
            let hi = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_srli_epi32::<4>(x)), e);
            (_mm512_mul_ps(lo, d), _mm512_mul_ps(hi, d))
        }
    }

    /// K scores with int8 dot products (VNNI `vpdpbusd`, 64 products an instruction): the queries quantized to int8
    /// per 32-value block (as llama.cpp's q8_0 activations), the cache nibbles as unsigned bytes, and the -8 offset
    /// folded in through each block's query sum. head_dim 256 (8 blocks: 4 vectors of 2 blocks, 8 f16 scales).
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,f16c,avx2")]
    unsafe fn kscores_vnni<const G: usize, const T: usize>(a: &HostAttn, q: &[f32], lim: &[usize], p0: usize, p1: usize, sc: &mut [f32]) {
        let (nh, nkv, hd) = (a.nh, a.nkv, a.hd);
        let (rb, n, rows) = (hd / 2 + hd / 16, p1 - p0, T * nh);
        let mut qq = vec![0i8; rows * hd];
        let mut qs = vec![0f32; rows * 64];  // per row, 4 vectors of 16 lanes: the block scale of each lane
        let mut cl = vec![0f32; rows * 64];  // the scale times the block's int8 sum
        for r in 0..rows {
            for b in 0..8 {
                let x = &q[r * hd + 32 * b..r * hd + 32 * b + 32];
                let mx = x.iter().fold(0f32, |m, v| m.max(v.abs()));
                let d = mx / 127.0;
                let id = if d > 0.0 { 1.0 / d } else { 0.0 };
                let mut sum = 0i32;
                for i in 0..32 { let v = (x[i] * id).round() as i32; qq[r * hd + 32 * b + i] = v as i8; sum += v; }
                for k in 0..8 { qs[r * 64 + 8 * b + k] = d; cl[r * 64 + 8 * b + k] = d * sum as f32; }
            }
        }
        unsafe {
            let m4 = _mm256_set1_epi8(15);
            let order = _mm512_setr_epi64(0, 1, 4, 5, 2, 3, 6, 7);
            let sidx: [__m512i; 4] = std::array::from_fn(|j| _mm512_setr_epi32(
                2 * j as i32, 2 * j as i32, 2 * j as i32, 2 * j as i32, 2 * j as i32, 2 * j as i32, 2 * j as i32, 2 * j as i32,
                2 * j as i32 + 1, 2 * j as i32 + 1, 2 * j as i32 + 1, 2 * j as i32 + 1, 2 * j as i32 + 1, 2 * j as i32 + 1, 2 * j as i32 + 1, 2 * j as i32 + 1));
            let kc = a.kc as *const u8;
            let (qqp, qsp, clp) = (qq.as_ptr(), qs.as_ptr(), cl.as_ptr());
            for p in p0..p1 {
                for g in 0..nkv {
                    let r = kc.add((p * nkv + g) * rb);
                    let d8 = _mm512_castps256_ps512(_mm256_cvtph_ps(_mm_loadu_si128(r.add(hd / 2) as *const __m128i)));
                    let mut z = [_mm512_setzero_si512(); 4];
                    let mut dk = [_mm512_setzero_ps(); 4];
                    for j in 0..4 {
                        let x = _mm256_loadu_si256(r.add(32 * j) as *const __m256i);
                        let lo = _mm256_and_si256(x, m4);
                        let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(x), m4);
                        z[j] = _mm512_permutexvar_epi64(order, _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi));
                        dk[j] = _mm512_permutexvar_ps(sidx[j], d8);
                    }
                    for t in 0..T {
                        if p > lim[t] { continue; }
                        for h in 0..G {
                            let row = t * nh + g * G + h;
                            let mut acc = _mm512_setzero_ps();
                            for j in 0..4 {
                                let dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), z[j], _mm512_loadu_si512(qqp.add(row * hd + 64 * j) as *const _));
                                let v = _mm512_fmsub_ps(_mm512_cvtepi32_ps(dot), _mm512_loadu_ps(qsp.add(row * 64 + 16 * j)), _mm512_loadu_ps(clp.add(row * 64 + 16 * j)));
                                acc = _mm512_fmadd_ps(v, dk[j], acc);
                            }
                            *sc.get_unchecked_mut(row * n + p - p0) = _mm512_reduce_add_ps(acc) * a.scale;
                        }
                    }
                }
            }
        }
    }

    #[target_feature(enable = "avx512f,avx512bw,f16c")]
    pub unsafe fn chunk_q4<const G: usize, const T: usize>(a: &HostAttn, q: &[f32], lim: &[usize], p0: usize, p1: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (nh, nkv, hd, nt) = (a.nh, a.nkv, a.hd, T);
        let (rb, nb, n) = (hd / 2 + hd / 16, hd / 32, p1 - p0);
        let (kc, vc) = (a.kc as *const u8, a.vc as *const u8);
        let rows = nt * nh;
        let mut sc = vec![f32::NEG_INFINITY; rows * n];
        let qp = q.as_ptr();
        unsafe {
            if nb == 8 && std::is_x86_feature_detected!("avx512vnni") {
                kscores_vnni::<G, T>(a, q, lim, p0, p1, &mut sc);
            } else {
                for p in p0..p1 {
                    for g in 0..nkv {
                        let r = kc.add((p * nkv + g) * rb);
                        for t in 0..T {
                            if p > lim[t] { continue; }
                            let mut acc = [_mm512_setzero_ps(); G];
                            for b in 0..nb {
                                let (lo, hi) = blk(r, hd, b);
                                for h in 0..G {
                                    let qh = qp.add((t * nh + g * G + h) * hd + 32 * b);
                                    acc[h] = _mm512_fmadd_ps(lo, _mm512_loadu_ps(qh), acc[h]);
                                    acc[h] = _mm512_fmadd_ps(hi, _mm512_loadu_ps(qh.add(16)), acc[h]);
                                }
                            }
                            for h in 0..G { sc[(t * nh + g * G + h) * n + p - p0] = _mm512_reduce_add_ps(acc[h]) * a.scale; }
                        }
                    }
                }
            }
            let (m, l) = softmax_rows(&mut sc, rows, n);
            let mut out = vec![0f32; rows * hd];
            // V by 32-value block: the accumulators of the G heads of each token stay in registers (or L1) over all positions.
            for g in 0..nkv {
                for b in 0..nb {
                    // One token at a time: 2 G accumulators stay in registers (all T tokens' would spill).
                    for t in 0..T {
                        let mut al = [_mm512_setzero_ps(); G];
                        let mut ah = [_mm512_setzero_ps(); G];
                        let pe = p1.min(lim[t] + 1);
                        for p in p0..pe {
                            let (lo, hi) = blk(vc.add((p * nkv + g) * rb), hd, b);
                            for h in 0..G {
                                let w = _mm512_set1_ps(*sc.get_unchecked((t * nh + g * G + h) * n + p - p0));
                                al[h] = _mm512_fmadd_ps(w, lo, al[h]);
                                ah[h] = _mm512_fmadd_ps(w, hi, ah[h]);
                            }
                        }
                        for h in 0..G {
                            let o = out.as_mut_ptr().add((t * nh + g * G + h) * hd + 32 * b);
                            _mm512_storeu_ps(o, al[h]);
                            _mm512_storeu_ps(o.add(16), ah[h]);
                        }
                    }
                }
            }
            (m, l, out)
        }
    }
}

fn chunk_generic(a: &HostAttn, q: &[f32], lim: &[usize], p0: usize, p1: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let w = a.nh * a.hd;
    let (mut m, mut l, mut acc) = (vec![], vec![], vec![]);
    for (t, &lt) in lim.iter().enumerate() {
        let e = (lt + 1).min(p1).max(p0);
        let (mt, lt, at) = if e > p0 { chunk1(a, &q[t * w..(t + 1) * w], p0, e) } else { (vec![f32::NEG_INFINITY; a.nh], vec![0.0; a.nh], vec![0.0; w]) };
        m.extend(mt); l.extend(lt); acc.extend(at);
    }
    (m, l, acc)
}

fn chunk1(a: &HostAttn, q: &[f32], p0: usize, p1: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (nh, nkv, hd) = (a.nh, a.nkv, a.hd);
    let grp = nh / nkv;
    let rb = row_bytes(a.mode, hd);
    let kc = unsafe { std::slice::from_raw_parts(a.kc as *const u8, p1 * nkv * rb) };
    let vc = unsafe { std::slice::from_raw_parts(a.vc as *const u8, p1 * nkv * rb) };
    let n = p1 - p0;
    let mut sc = vec![0f32; nh * n];
    let mut row = vec![0f32; hd];
    for p in p0..p1 {
        for g in 0..nkv {
            let r = (p * nkv + g) * rb;
            deq(a.mode, &kc[r..r + rb], hd, &mut row);
            for h in g * grp..(g + 1) * grp { sc[h * n + p - p0] = dot(&q[h * hd..(h + 1) * hd], &row) * a.scale; }
        }
    }
    let mut m = vec![f32::NEG_INFINITY; nh];
    let mut l = vec![0f32; nh];
    for h in 0..nh {
        let s = &mut sc[h * n..(h + 1) * n];
        m[h] = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mh = m[h];
        if mh == f32::NEG_INFINITY { s.fill(0.0); continue; }
        let mut sum = 0.0;
        for x in s.iter_mut() { *x = fexp(*x - mh); sum += *x; }
        l[h] = sum;
    }
    let mut acc = vec![0f32; nh * hd];
    for p in p0..p1 {
        for g in 0..nkv {
            let r = (p * nkv + g) * rb;
            deq(a.mode, &vc[r..r + rb], hd, &mut row);
            for h in g * grp..(g + 1) * grp {
                let w = sc[h * n + p - p0];
                for (o, v) in acc[h * hd..(h + 1) * hd].iter_mut().zip(&row) { *o += w * v; }
            }
        }
    }
    (m, l, acc)
}

static PROF: std::sync::Mutex<(u32, f64, usize, f64)> = std::sync::Mutex::new((0, 0.0, 0, 0.0));

pub fn run(a: &HostAttn) {
    let t0 = std::time::Instant::now();
    run_inner(a);
    if std::env::var_os("STRATA_HA_PROF").is_some() {
        let mut p = PROF.lock().unwrap();
        p.0 += 1; p.1 += t0.elapsed().as_secs_f64(); p.2 += a.nt;
        if p.0 % 320 == 0 {
            let pos = unsafe { *(a.step as *const i32) };
            eprintln!("host_attn window: {:.2} ms/call, {:.2} tokens/call, pos {pos}", 1e3 * p.1 / p.0 as f64, p.2 as f64 / p.0 as f64);
            *p = (0, 0.0, 0, 0.0);
        }
    }
}

/// Its own pool on every logical core (`STRATA_HA_THREADS` overrides): the global one has one thread a physical
/// core for the memory-bound GPU expert work, while this int8/f32 math gains from SMT (2.8 to 2.2 ms a layer at 52K).
fn pool() -> &'static rayon::ThreadPool {
    static P: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    P.get_or_init(|| {
        let n = std::env::var("STRATA_HA_THREADS").ok().and_then(|v| v.parse().ok())
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(8, |n| n.get()));
        rayon::ThreadPoolBuilder::new().num_threads(n).thread_name(|i| format!("host-attn-{i}")).build().unwrap()
    })
}

fn run_inner(a: &HostAttn) {
    let (nh, hd, nt) = (a.nh, a.hd, a.nt);
    let lim: Vec<usize> = (0..nt).map(|t| unsafe { *(a.step as *const i32).add(2 * t) } as usize).collect();
    let q0 = unsafe { std::slice::from_raw_parts(a.q as *const f32, nt * nh * hd) };
    let mut q = vec![0f32; nt * nh * hd];
    for r in 0..nt * nh { for i in 0..hd { q[r * hd + perm(a.mode, i)] = q0[r * hd + i]; } }
    let n = lim.iter().max().unwrap() + 1;
    let parts: Vec<_> = pool().install(|| (0..n.div_ceil(CHUNK)).into_par_iter().map(|c| chunk(a, &q, &lim, c * CHUNK, ((c + 1) * CHUNK).min(n))).collect());
    let part = unsafe { std::slice::from_raw_parts_mut(a.part as *mut f32, nt * nh * (hd + 2)) };
    for r in 0..nt * nh {
        let mm = parts.iter().map(|p| p.0[r]).fold(f32::NEG_INFINITY, f32::max);
        let o = &mut part[r * (hd + 2)..(r + 1) * (hd + 2)];
        o.fill(0.0);
        let mut l = 0.0;
        for p in &parts {
            if p.0[r] == f32::NEG_INFINITY { continue; }
            let w = (p.0[r] - mm).exp();
            l += p.1[r] * w;
            for i in 0..hd { o[i] += p.2[r * hd + perm(a.mode, i)] * w; }
        }
        o[hd] = mm;
        o[hd + 1] = l;
    }
}

pub unsafe extern "C" fn callback(user: *mut std::ffi::c_void) {
    run(unsafe { &*(user as *const HostAttn) });
}

/// Whether block `il`'s KV cache lives in host RAM (`STRATA_KV_HOST=N`: blocks N and up, `all`: every one).
pub fn on_host(il: usize) -> bool {
    match std::env::var("STRATA_KV_HOST").ok().as_deref() {
        None | Some("") => false,
        Some("all") => true,
        Some(v) => v.parse::<usize>().is_ok_and(|n| il >= n),
    }
}

/// The CPU side of the `ha_post`/`ha_wait` mailbox at host address `mb` (u32 words: request at 0, context pointer
/// at 4-5, done at 16). Spins while requests come, naps after 2 ms without one.
pub fn start_worker(mb: usize) {
    std::thread::Builder::new().name("host-attn".into()).spawn(move || {
        let w = mb as *mut u32;
        let mut last = 0u32;
        let mut idle = std::time::Instant::now();
        loop {
            let r = unsafe { std::ptr::read_volatile(w) };
            if r != last {
                std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
                let ctx = unsafe { std::ptr::read_volatile(w.add(4) as *const u64) } as usize;
                run(unsafe { &*(ctx as *const HostAttn) });
                std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
                unsafe { std::ptr::write_volatile(w.add(16), r) };
                last = r;
                idle = std::time::Instant::now();
            } else if idle.elapsed() > std::time::Duration::from_millis(2) {
                std::thread::sleep(std::time::Duration::from_micros(50));
            } else {
                std::hint::spin_loop();
            }
        }
    }).expect("host-attn thread");
}
