//! Fast CPU routed-expert path: activations quantized to Q8_K blocks, integer dot products per weight type
//! (ported from llama.cpp `ggml_vec_dot_*_q8_K_generic`, MIT License, ggml authors). Types without an integer
//! kernel fall back to f32 dequantization.

use crate::model::Model;
use crate::ops;
use crate::quant::{GgmlType, tables::*};
use crate::weights::Tensor;
use half::f16;
use rayon::prelude::*;

pub struct Q8K { d: f32, qs: [i8; 256] }

/// Signs of IQ2/IQ3 sub-blocks as ±1 bytes, indexed by the 7-bit sign field.
static SIGNS: [[i8; 8]; 128] = {
    let mut t = [[1i8; 8]; 128];
    let mut i = 0;
    while i < 128 {
        let mut j = 0;
        while j < 8 { if KSIGNS_IQ2XS[i] & (1 << j) != 0 { t[i][j] = -1; } j += 1; }
        i += 1;
    }
    t
};

pub fn quantize_q8k(x: &[f32]) -> Vec<Q8K> {
    x.chunks_exact(256).map(|b| {
        let max = b.iter().fold(0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
        if max == 0.0 { return Q8K { d: 0.0, qs: [0; 256] }; }
        let is = -127.0 / max;
        Q8K { d: 1.0 / is, qs: std::array::from_fn(|j| ((is * b[j]).round_ties_even() as i32).min(127) as i8) }
    }).collect()
}

fn d16(b: &[u8]) -> f32 { f16::from_le_bytes([b[0], b[1]]).to_f32() }

#[inline]
fn dot8(g: [u8; 8], s: &[i8; 8], q: &[i8]) -> i32 {
    (0..8).map(|j| g[j] as i32 * s[j] as i32 * q[j] as i32).sum()
}

fn dot_iq2xs(row: &[u8], xq: &[Q8K]) -> f32 {
    let mut sumf = 0.0;
    for (b, y) in row.chunks_exact(74).zip(xq) {
        let sc = &b[66..74];
        let mut bsum = 0i32;
        for ib in 0..8 {
            let mut s = [0i32; 2];
            for l in 0..4 {
                let o = 2 + 2 * (4 * ib + l);
                let q = u16::from_le_bytes([b[o], b[o + 1]]);
                s[l / 2] += dot8(IQ2XS_GRID[(q & 511) as usize].to_le_bytes(), &SIGNS[(q >> 9) as usize], &y.qs[32 * ib + 8 * l..]);
            }
            bsum += s[0] * (2 * (sc[ib] & 0xF) as i32 + 1) + s[1] * (2 * (sc[ib] >> 4) as i32 + 1);
        }
        sumf += d16(b) * y.d * bsum as f32;
    }
    0.125 * sumf
}

fn dot_iq3xxs(row: &[u8], xq: &[Q8K]) -> f32 {
    let mut sumf = 0.0;
    for (b, y) in row.chunks_exact(98).zip(xq) {
        let (qs, gas) = (&b[2..66], &b[66..98]);
        let mut bsum = 0i32;
        for ib in 0..8 {
            let aux = u32::from_le_bytes(gas[4 * ib..4 * ib + 4].try_into().unwrap());
            let mut s = 0i32;
            for l in 0..4 {
                let g1 = IQ3XXS_GRID[qs[8 * ib + 2 * l] as usize].to_le_bytes();
                let g2 = IQ3XXS_GRID[qs[8 * ib + 2 * l + 1] as usize].to_le_bytes();
                let g = [g1[0], g1[1], g1[2], g1[3], g2[0], g2[1], g2[2], g2[3]];
                s += dot8(g, &SIGNS[((aux >> (7 * l)) & 127) as usize], &y.qs[32 * ib + 8 * l..]);
            }
            bsum += s * (2 * (aux >> 28) as i32 + 1);
        }
        sumf += d16(b) * y.d * bsum as f32;
    }
    0.25 * sumf
}

fn dot_iq4xs(row: &[u8], xq: &[Q8K]) -> f32 {
    let mut sumf = 0.0;
    for (b, y) in row.chunks_exact(136).zip(xq) {
        let dd = d16(b) * y.d;
        let sh = u16::from_le_bytes([b[2], b[3]]);
        let (sl, qs) = (&b[4..8], &b[8..136]);
        for ib in 0..8 {
            let ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) as i32 | ((((sh >> (2 * ib)) & 3) as i32) << 4);
            let q8 = &y.qs[32 * ib..32 * ib + 32];
            let s: i32 = (0..16).map(|j| {
                let q = qs[16 * ib + j];
                q8[j] as i32 * KVALUES_IQ4NL[(q & 0xF) as usize] as i32 + q8[j + 16] as i32 * KVALUES_IQ4NL[(q >> 4) as usize] as i32
            }).sum();
            sumf += dd * (ls - 32) as f32 * s as f32;
        }
    }
    sumf
}

/// AVX2 versions of the kernels above (llama.cpp `arch/x86/quants.c`).
#[cfg(target_feature = "avx2")]
mod avx2 {
    use super::{Q8K, SIGNS, d16};
    use crate::quant::tables::*;
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn hsum(v: __m256) -> f32 { unsafe {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)))
    } }
    #[inline(always)]
    fn sg(i: usize) -> i64 { unsafe { std::mem::transmute::<[i8; 8], i64>(SIGNS[i]) } }

    pub fn dot_iq2xs(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let mut acc = _mm256_setzero_ps();
            for (b, y) in row.chunks_exact(74).zip(xq) {
                let q = |i: usize| u16::from_le_bytes([b[2 + 2 * i], b[3 + 2 * i]]);
                let mut sumi = _mm256_setzero_si256();
                for ib in 0..8 {
                    let (q0, q1, q2, q3) = (q(4 * ib), q(4 * ib + 1), q(4 * ib + 2), q(4 * ib + 3));
                    let g = |q: u16| IQ2XS_GRID[(q & 511) as usize] as i64;
                    let grid = _mm256_set_epi64x(g(q3), g(q2), g(q1), g(q0));
                    let signs = _mm256_set_epi64x(sg((q3 >> 9) as usize), sg((q2 >> 9) as usize), sg((q1 >> 9) as usize), sg((q0 >> 9) as usize));
                    let q8 = _mm256_loadu_si256(y.qs.as_ptr().add(32 * ib) as *const __m256i);
                    let dot = _mm256_maddubs_epi16(grid, _mm256_sign_epi8(q8, signs));
                    let sc = b[66 + ib];
                    let ls = _mm256_set_m128i(_mm_set1_epi16(2 * (sc >> 4) as i16 + 1), _mm_set1_epi16(2 * (sc & 0xF) as i16 + 1));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(dot, ls));
                }
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d16(b) * y.d), _mm256_cvtepi32_ps(sumi), acc);
            }
            0.125 * hsum(acc)
        }
    }

    pub fn dot_iq3xxs(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let mut acc = _mm256_setzero_ps();
            for (b, y) in row.chunks_exact(98).zip(xq) {
                let (qs, gas) = (&b[2..66], &b[66..98]);
                let mut sumi = _mm256_setzero_si256();
                for ib in 0..8 {
                    let g = |i: usize| IQ3XXS_GRID[qs[8 * ib + i] as usize] as i32;
                    let grid = _mm256_set_epi32(g(7), g(6), g(5), g(4), g(3), g(2), g(1), g(0));
                    let aux = u32::from_le_bytes(gas[4 * ib..4 * ib + 4].try_into().unwrap());
                    let s = |l: u32| sg(((aux >> (7 * l)) & 127) as usize);
                    let signs = _mm256_set_epi64x(s(3), s(2), s(1), s(0));
                    let q8 = _mm256_loadu_si256(y.qs.as_ptr().add(32 * ib) as *const __m256i);
                    let dot = _mm256_maddubs_epi16(grid, _mm256_sign_epi8(q8, signs));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(dot, _mm256_set1_epi16(2 * (aux >> 28) as i16 + 1)));
                }
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d16(b) * y.d), _mm256_cvtepi32_ps(sumi), acc);
            }
            0.25 * hsum(acc)
        }
    }

    pub fn dot_iq4xs(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let values = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
            let m4 = _mm_set1_epi8(0xF);
            let mut acc = _mm256_setzero_ps();
            for (b, y) in row.chunks_exact(136).zip(xq) {
                let sh = u16::from_le_bytes([b[2], b[3]]);
                let mut sumi = _mm256_setzero_si256();
                for ib in 0..8 {
                    let bits = _mm_loadu_si128(b.as_ptr().add(8 + 16 * ib) as *const __m128i);
                    let w = _mm256_set_m128i(_mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(bits, 4), m4)),
                                             _mm_shuffle_epi8(values, _mm_and_si128(bits, m4)));
                    let q8 = _mm256_loadu_si256(y.qs.as_ptr().add(32 * ib) as *const __m256i);
                    let p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(q8, w));
                    let ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) as i16 | ((((sh >> (2 * ib)) & 3) as i16) << 4);
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(p16, _mm256_set1_epi16(ls - 32)));
                }
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d16(b) * y.d), _mm256_cvtepi32_ps(sumi), acc);
            }
            hsum(acc)
        }
    }
}

fn int_dot(ty: GgmlType) -> Option<fn(&[u8], &[Q8K]) -> f32> {
    #[cfg(target_feature = "avx2")]
    if std::env::var_os("STRATA_SCALAR").is_none() {
        return match ty { GgmlType::IQ2XS => Some(avx2::dot_iq2xs), GgmlType::IQ3XXS => Some(avx2::dot_iq3xxs), GgmlType::IQ4XS => Some(avx2::dot_iq4xs), _ => None };
    }
    match ty { GgmlType::IQ2XS => Some(dot_iq2xs), GgmlType::IQ3XXS => Some(dot_iq3xxs), GgmlType::IQ4XS => Some(dot_iq4xs), _ => None }
}

/// y = t · x for every row, using the integer kernel when the type has one.
pub fn matvec(t: &Tensor, x: &[f32], y: &mut [f32]) {
    let Some(dot) = int_dot(t.ty) else { return t.matvec(x, y) };
    let xq = quantize_q8k(x);
    let rb = t.data.len() / t.rows;
    y.par_chunks_mut(32).enumerate().for_each(|(c, ys)| {
        for (k, v) in ys.iter_mut().enumerate() {
            let r = c * 32 + k;
            *v = dot(&t.data[r * rb..(r + 1) * rb], &xq);
        }
    });
}

/// Rows `out.len()` of a batch of matrices: out[i] = mats[i / rows] row (i % rows) · xs[i / rows].
fn batch_rows(mats: &[Tensor], xs: &[&[f32]], out: &mut [f32]) {
    let rows = out.len() / mats.len();
    let dot = int_dot(mats[0].ty);
    let xq: Vec<Vec<Q8K>> = if dot.is_some() { xs.iter().map(|x| quantize_q8k(x)).collect() } else { vec![] };
    out.par_chunks_mut(16).enumerate().for_each_init(|| vec![0f32; mats[0].cols], |buf, (c, ys)| {
        for (k, v) in ys.iter_mut().enumerate() {
            let i = c * 16 + k;
            let (t, r) = (&mats[i / rows], i % rows);
            let rb = t.data.len() / t.rows;
            *v = match dot {
                Some(d) => d(&t.data[r * rb..(r + 1) * rb], &xq[i / rows]),
                None => { t.row(r, buf); buf.iter().zip(xs[i / rows]).map(|(a, b)| a * b).sum() }
            };
        }
    });
}

/// Weighted sum of the given routed experts of layer `il` applied to `h` (no router, no shared expert).
pub fn experts(m: &Model, il: usize, h: &[f32], sel: &[(usize, f32)]) -> Vec<f32> {
    let c = &m.cfg;
    let mut y = vec![0f32; c.n_embd];
    if sel.is_empty() { return y; }
    let t = |s: &str| m.w.layer(il, s).unwrap_or_else(|e| panic!("{e:#}"));
    let (gate, up, down) = (t("ffn_gate_exps.weight"), t("ffn_up_exps.weight"), t("ffn_down_exps.weight"));
    let (n, ff, k) = (c.n_expert, c.n_ff_exp, sel.len());
    // Gate and up of every expert in one parallel pass (gate and up share a type in all supported files).
    let gu: Vec<Tensor> = sel.iter().flat_map(|&(e, _)| [gate.expert(e, n), up.expert(e, n)]).collect();
    let mut a = vec![0f32; 2 * k * ff];
    if gate.ty == up.ty {
        batch_rows(&gu, &vec![h; 2 * k], &mut a);
    } else {
        for (i, t) in gu.iter().enumerate() { matvec(t, h, &mut a[i * ff..(i + 1) * ff]); }
    }
    let acts: Vec<Vec<f32>> = a.chunks_exact(2 * ff).map(|g| (0..ff).map(|i| ops::silu(g[i]) * g[ff + i]).collect()).collect();
    let ds: Vec<Tensor> = sel.iter().map(|&(e, _)| down.expert(e, n)).collect();
    let mut o = vec![0f32; k * c.n_embd];
    batch_rows(&ds, &acts.iter().map(|v| v.as_slice()).collect::<Vec<_>>(), &mut o);
    for (oj, &(_, w)) in o.chunks_exact(c.n_embd).zip(sel) {
        for (yy, oo) in y.iter_mut().zip(oj) { *yy += w * oo; }
    }
    y
}
