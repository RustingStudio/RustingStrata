//! Fast CPU routed-expert path: activations quantized to Q8_K blocks, integer dot products per weight type
//! (ported from llama.cpp `ggml_vec_dot_*_q8_K_generic`, MIT License, ggml authors). Types without an integer
//! kernel fall back to f32 dequantization.

use crate::model::Model;
use crate::ops;
use crate::quant::{GgmlType, tables::*};
use crate::weights::Tensor;
use half::f16;
use rayon::prelude::*;

/// `bs` holds the sums of the 16 groups of 16 `qs`, for the min terms of Q2_K and Q4_K.
pub struct Q8K { d: f32, qs: [i8; 256], bs: [i16; 16] }

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
        // The first value of largest magnitude (a max over |v| first, which vectorizes).
        let amax = b.iter().fold(0f32, |m, &v| m.max(v.abs()));
        if amax == 0.0 { return Q8K { d: 0.0, qs: [0; 256], bs: [0; 16] }; }
        let max = *b.iter().find(|v| v.abs() == amax).unwrap();
        let is = -127.0 / max;
        let mut qs = [0i8; 256];
        #[cfg(target_feature = "avx2")]
        avx2::quant_block(is, b, &mut qs);
        #[cfg(not(target_feature = "avx2"))]
        for (q, &v) in qs.iter_mut().zip(b) { *q = ((is * v).round_ties_even() as i32).min(127) as i8; }
        let mut bs = [0i16; 16];
        for (s, c) in bs.iter_mut().zip(qs.chunks_exact(16)) { *s = c.iter().map(|&v| v as i16).sum(); }
        Q8K { d: 1.0 / is, qs, bs }
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

fn dot_iq2xxs(row: &[u8], xq: &[Q8K]) -> f32 {
    let mut sumf = 0.0;
    for (b, y) in row.chunks_exact(66).zip(xq) {
        let mut bsum = 0i32;
        for ib in 0..8 {
            let aux = u32::from_le_bytes(b[6 + 8 * ib..10 + 8 * ib].try_into().unwrap());
            let s: i32 = (0..4).map(|l| dot8(IQ2XXS_GRID[b[2 + 8 * ib + l] as usize].to_le_bytes(), &SIGNS[((aux >> (7 * l)) & 127) as usize], &y.qs[32 * ib + 8 * l..])).sum();
            bsum += s * (2 * (aux >> 28) as i32 + 1);
        }
        sumf += d16(b) * y.d * bsum as f32;
    }
    0.125 * sumf
}

fn dot_iq2s(row: &[u8], xq: &[Q8K]) -> f32 {
    let mut sumf = 0.0;
    for (b, y) in row.chunks_exact(82).zip(xq) {
        let (qs, signs, qh, sc) = (&b[2..34], &b[34..66], &b[66..74], &b[74..82]);
        let mut bsum = 0i32;
        for ib in 0..8 {
            let s: [i32; 2] = std::array::from_fn(|h| (2 * h..2 * h + 2).map(|l| {
                let g = IQ2S_GRID[qs[4 * ib + l] as usize | ((qh[ib] as usize) << (8 - 2 * l) & 0x300)].to_le_bytes();
                let sg: [i8; 8] = std::array::from_fn(|j| 1 - 2 * (signs[4 * ib + l] >> j & 1) as i8);
                dot8(g, &sg, &y.qs[32 * ib + 8 * l..])
            }).sum());
            bsum += s[0] * (2 * (sc[ib] & 0xF) as i32 + 1) + s[1] * (2 * (sc[ib] >> 4) as i32 + 1);
        }
        sumf += d16(b) * y.d * bsum as f32;
    }
    0.125 * sumf
}

/// Dot of 8 IQ1 grid values (-1, 0, 1) with `q`.
fn dot1(g: u64, q: &[i8]) -> i32 { g.to_le_bytes().iter().zip(&q[..8]).map(|(&a, &b)| a as i8 as i32 * b as i32).sum() }

fn dot_iq1s(row: &[u8], xq: &[Q8K]) -> f32 {
    let mut sumf = 0.0;
    for (b, y) in row.chunks_exact(50).zip(xq) {
        let (mut s, mut s1) = (0i32, 0i32);
        for ib in 0..8 {
            let qh = u16::from_le_bytes([b[34 + 2 * ib], b[35 + 2 * ib]]) as usize;
            let ls = 2 * ((qh >> 12) & 7) as i32 + 1;
            s += ls * (0..4).map(|l| dot1(IQ1S_GRID[b[2 + 4 * ib + l] as usize | ((qh >> (3 * l)) & 7) << 8], &y.qs[32 * ib + 8 * l..])).sum::<i32>();
            s1 += ls * if qh & 0x8000 != 0 { -1 } else { 1 } * (y.bs[2 * ib] + y.bs[2 * ib + 1]) as i32;
        }
        sumf += d16(b) * y.d * (s as f32 + 0.125 * s1 as f32);
    }
    sumf
}

/// IQ1_M block scale: the top 4 bits of the four scale words.
fn iq1m_d(b: &[u8]) -> f32 {
    let sc = |i: usize| u16::from_le_bytes([b[48 + 2 * i], b[49 + 2 * i]]);
    f16::from_bits((sc(0) >> 12) | ((sc(1) >> 8) & 0xF0) | ((sc(2) >> 4) & 0xF00) | (sc(3) & 0xF000)).to_f32()
}

/// Scale of values 16 * (2 * ib + h)..+16 of an IQ1_M block.
fn iq1m_ls(b: &[u8], ib: usize, h: usize) -> i32 {
    2 * ((u16::from_le_bytes([b[48 + ib / 2 * 2], b[49 + ib / 2 * 2]]) >> (6 * (ib % 2) + 3 * h)) & 7) as i32 + 1
}

fn dot_iq1m(row: &[u8], xq: &[Q8K]) -> f32 {
    let mut sumf = 0.0;
    for (b, y) in row.chunks_exact(56).zip(xq) {
        let (mut s1, mut s2) = (0i32, 0i32);
        for ib in 0..8 {
            for l in 0..4 {
                let qh = b[32 + 2 * ib + l / 2] as usize >> (4 * (l % 2));
                let q = &y.qs[32 * ib + 8 * l..32 * ib + 8 * l + 8];
                let ls = iq1m_ls(b, ib, l / 2);
                s1 += ls * dot1(IQ1S_GRID[b[4 * ib + l] as usize | (qh & 7) << 8], q);
                s2 += ls * if qh & 8 != 0 { -1 } else { 1 } * q.iter().map(|&v| v as i32).sum::<i32>();
            }
        }
        sumf += iq1m_d(b) * y.d * (s1 as f32 + 0.125 * s2 as f32);
    }
    sumf
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

fn dot_iq3s(row: &[u8], xq: &[Q8K]) -> f32 {
    let mut sumf = 0.0;
    for (b, y) in row.chunks_exact(110).zip(xq) {
        let (qs, qh, signs, sc) = (&b[2..66], &b[66..74], &b[74..106], &b[106..110]);
        let mut bsum = 0i32;
        for ib in 0..8 {
            let mut s = 0i32;
            for l in 0..4 {
                let g = |k: usize, sh: usize| IQ3S_GRID[(qs[8 * ib + 2 * l + k] as usize) | ((qh[ib] as usize) << sh & 256)].to_le_bytes();
                let (g1, g2) = (g(0, 8 - 2 * l), g(1, 7 - 2 * l));
                let g = [g1[0], g1[1], g1[2], g1[3], g2[0], g2[1], g2[2], g2[3]];
                let sg: [i8; 8] = std::array::from_fn(|j| if signs[4 * ib + l] >> j & 1 != 0 { -1 } else { 1 });
                s += dot8(g, &sg, &y.qs[32 * ib + 8 * l..]);
            }
            bsum += s * (2 * (sc[ib / 2] >> (4 * (ib % 2)) & 0xF) as i32 + 1);
        }
        sumf += d16(b) * y.d * bsum as f32;
    }
    sumf
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

/// Sum over the 16 groups of 16 of `sc(group) * (a · q)`, written so the compiler vectorizes it.
#[inline(always)]
fn grouped(a: &[i8; 256], q: &[i8; 256], sc: impl Fn(usize) -> i32) -> i32 {
    (0..16).map(|j| sc(j) * (16 * j..16 * j + 16).map(|i| a[i] as i32 * q[i] as i32).sum::<i32>()).sum()
}

fn dot_q2k(row: &[u8], xq: &[Q8K]) -> f32 {
    let mut sumf = 0.0;
    for (b, y) in row.chunks_exact(84).zip(xq) {
        let sc = &b[0..16];
        let mut a = [0i8; 256];
        for i in 0..256 { a[i] = (b[16 + 32 * (i / 128) + i % 32] >> (2 * (i / 32 % 4)) & 3) as i8; }
        let s = grouped(&a, &y.qs, |j| (sc[j] & 0xF) as i32);
        let m: i32 = (0..16).map(|j| (sc[j] >> 4) as i32 * y.bs[j] as i32).sum();
        sumf += y.d * d16(&b[80..]) * s as f32 - y.d * d16(&b[82..]) * m as f32;
    }
    sumf
}

fn dot_q3k(row: &[u8], xq: &[Q8K]) -> f32 {
    let mut sumf = 0.0;
    for (b, y) in row.chunks_exact(110).zip(xq) {
        let (hm, sc) = (&b[0..32], crate::quant::q3k_scales(b));
        let mut a = [0i8; 256];
        for i in 0..256 {
            let k = i / 32 % 4 + 4 * (i / 128);
            a[i] = (b[32 + 32 * (i / 128) + i % 32] >> (2 * (k % 4)) & 3) as i8 - if hm[i % 32] >> k & 1 != 0 { 0 } else { 4 };
        }
        sumf += d16(&b[108..]) * y.d * grouped(&a, &y.qs, |j| sc[j] as i32 - 32) as f32;
    }
    sumf
}

fn dot_q4k(row: &[u8], xq: &[Q8K]) -> f32 {
    let mut sumf = 0.0;
    for (b, y) in row.chunks_exact(144).zip(xq) {
        let sm: [(u8, u8); 8] = std::array::from_fn(|j| crate::quant::scale_min_k4(j, &b[4..16]));
        let mut a = [0i8; 256];
        for i in 0..256 { a[i] = (b[16 + 32 * (i / 64) + i % 32] >> (4 * (i / 32 % 2)) & 0xF) as i8; }
        let s = grouped(&a, &y.qs, |j| sm[j / 2].0 as i32);
        let m: i32 = (0..8).map(|j| sm[j].1 as i32 * (y.bs[2 * j] + y.bs[2 * j + 1]) as i32).sum();
        sumf += d16(b) * y.d * s as f32 - d16(&b[2..]) * y.d * m as f32;
    }
    sumf
}

fn dot_q6k(row: &[u8], xq: &[Q8K]) -> f32 {
    let mut sumf = 0.0;
    for (b, y) in row.chunks_exact(210).zip(xq) {
        let mut a = [0i8; 256];
        for n in 0..2 {
            let (ql, qh, a) = (&b[64 * n..], &b[128 + 32 * n..], &mut a[128 * n..]);
            for l in 0..32 {
                a[l] = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
                a[l + 32] = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                a[l + 64] = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                a[l + 96] = ((ql[l + 32] >> 4) | ((qh[l] >> 6) << 4)) as i8 - 32;
            }
        }
        let s: i32 = (0..16).map(|j| b[192 + j] as i8 as i32 * (16 * j..16 * j + 16).map(|i| a[i] as i32 * y.qs[i] as i32).sum::<i32>()).sum();
        sumf += d16(&b[208..]) * y.d * s as f32;
    }
    sumf
}

/// AVX2 versions of the kernels above (llama.cpp `arch/x86/quants.c`).
#[cfg(target_feature = "avx2")]
mod avx2 {
    use super::{Q8K, SIGNS, d16, iq1m_d};
    use crate::quant::tables::*;
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn hsum(v: __m256) -> f32 { unsafe {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)))
    } }
    /// qs = round(is * b), as the scalar path (|is * b| <= 127, so the saturating packs equal its min(127)).
    pub fn quant_block(is: f32, b: &[f32], qs: &mut [i8; 256]) { unsafe {
        let s = _mm256_set1_ps(is);
        let q = |i: usize| _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(s, _mm256_loadu_ps(b[i..i + 8].as_ptr())), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        for j in (0..256).step_by(32) {
            let p = _mm256_packs_epi16(_mm256_packs_epi32(q(j), q(j + 8)), _mm256_packs_epi32(q(j + 16), q(j + 24)));
            _mm256_storeu_si256(qs[j..j + 32].as_mut_ptr() as *mut __m256i, _mm256_permutevar8x32_epi32(p, perm));
        }
    } }
    #[inline(always)]
    fn sg(i: usize) -> i64 { unsafe { std::mem::transmute::<[i8; 8], i64>(SIGNS[i]) } }

    pub fn dot_iq2xs(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let mut acc = _mm256_setzero_ps();
            for (b, y) in row.chunks_exact(74).zip(xq) {
                let mut sumi = _mm256_setzero_si256();
                for ib in 0..8 {
                    let v = u64::from_le_bytes(b[2 + 8 * ib..10 + 8 * ib].try_into().unwrap());
                    let (q0, q1, q2, q3) = (v as u16, (v >> 16) as u16, (v >> 32) as u16, (v >> 48) as u16);
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
                    let v = u64::from_le_bytes(qs[8 * ib..8 * ib + 8].try_into().unwrap());
                    let g = |i: usize| IQ3XXS_GRID[(v >> (8 * i)) as u8 as usize] as i32;
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

    pub fn dot_iq3s(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            // Byte k of a lane picks sign byte k / 8, then bit k % 8 (llama.cpp k_mask1 / k_mask2).
            let m1 = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0);
            let m2 = _mm256_set1_epi64x(0x8040201008040201u64 as i64);
            let mut acc = _mm256_setzero_ps();
            for (b, y) in row.chunks_exact(110).zip(xq) {
                let (qs, qh) = (&b[2..66], &b[66..74]);
                let mut sumi = _mm256_setzero_si256();
                for ib in 0..8 {
                    let g = |i: usize| IQ3S_GRID[qs[8 * ib + i] as usize | ((qh[ib] as usize) << (8 - i) & 256)] as i32;
                    let grid = _mm256_set_epi32(g(7), g(6), g(5), g(4), g(3), g(2), g(1), g(0));
                    let sb = i32::from_le_bytes(b[74 + 4 * ib..78 + 4 * ib].try_into().unwrap());
                    let s = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(sb), m1), m2), m2);
                    let q8 = _mm256_loadu_si256(y.qs.as_ptr().add(32 * ib) as *const __m256i);
                    let dot = _mm256_maddubs_epi16(grid, _mm256_sub_epi8(_mm256_xor_si256(s, q8), s));
                    let ls = 2 * (b[106 + ib / 2] >> (4 * (ib % 2)) & 0xF) as i16 + 1;
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(dot, _mm256_set1_epi16(ls)));
                }
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d16(b) * y.d), _mm256_cvtepi32_ps(sumi), acc);
            }
            hsum(acc)
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

    /// Pair sums `p` (16 i16) times `lo` (first 8, values 0..16) and `hi` (last 8), summed into 8 i32.
    #[inline(always)]
    unsafe fn udot_scale(p: __m256i, lo: i32, hi: i32) -> __m256i { unsafe {
        _mm256_madd_epi16(p, _mm256_set_m128i(_mm_set1_epi16(hi as i16), _mm_set1_epi16(lo as i16)))
    } }
    /// `(w · q8)` per 16 values times `lo` (first 16) and `hi` (last 16), for 32 weights in 0..=127.
    #[inline(always)]
    unsafe fn udot32(w: __m256i, q8: __m256i, lo: i32, hi: i32) -> __m256i { unsafe { udot_scale(_mm256_maddubs_epi16(w, q8), lo, hi) } }
    #[inline(always)]
    unsafe fn q8(y: &Q8K, i: usize) -> __m256i { unsafe { _mm256_loadu_si256(y.qs.as_ptr().add(i) as *const __m256i) } }
    #[inline(always)]
    unsafe fn ld(b: &[u8], i: usize) -> __m256i { unsafe { _mm256_loadu_si256(b[i..i + 32].as_ptr() as *const __m256i) } }
    /// Crumb `k` (bits 2k, 2k+1) of every byte.
    #[inline(always)]
    unsafe fn crumb(v: __m256i, k: usize) -> __m256i { unsafe {
        _mm256_and_si256(_mm256_srl_epi16(v, _mm_cvtsi32_si128(2 * k as i32)), _mm256_set1_epi8(3))
    } }

    pub fn dot_iq2xxs(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let mut acc = _mm256_setzero_ps();
            for (b, y) in row.chunks_exact(66).zip(xq) {
                let mut sumi = _mm256_setzero_si256();
                for ib in 0..8 {
                    let g = |l: usize| IQ2XXS_GRID[b[2 + 8 * ib + l] as usize] as i64;
                    let aux = u32::from_le_bytes(b[6 + 8 * ib..10 + 8 * ib].try_into().unwrap());
                    let s = |l: u32| sg(((aux >> (7 * l)) & 127) as usize);
                    let grid = _mm256_set_epi64x(g(3), g(2), g(1), g(0));
                    let dot = _mm256_maddubs_epi16(grid, _mm256_sign_epi8(q8(y, 32 * ib), _mm256_set_epi64x(s(3), s(2), s(1), s(0))));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(dot, _mm256_set1_epi16(2 * (aux >> 28) as i16 + 1)));
                }
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d16(b) * y.d), _mm256_cvtepi32_ps(sumi), acc);
            }
            0.125 * hsum(acc)
        }
    }

    pub fn dot_iq2s(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            // As in dot_iq3s: byte k of a lane picks sign byte k / 8, then bit k % 8.
            let m1 = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0);
            let m2 = _mm256_set1_epi64x(0x8040201008040201u64 as i64);
            let mut acc = _mm256_setzero_ps();
            for (b, y) in row.chunks_exact(82).zip(xq) {
                let (qs, qh, sc) = (&b[2..34], &b[66..74], &b[74..82]);
                let mut sumi = _mm256_setzero_si256();
                for ib in 0..8 {
                    let g = |l: usize| IQ2S_GRID[qs[4 * ib + l] as usize | ((qh[ib] as usize) << (8 - 2 * l) & 0x300)] as i64;
                    let grid = _mm256_set_epi64x(g(3), g(2), g(1), g(0));
                    let sb = i32::from_le_bytes(b[34 + 4 * ib..38 + 4 * ib].try_into().unwrap());
                    let s = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(sb), m1), m2), m2);
                    let q = q8(y, 32 * ib);
                    let dot = _mm256_maddubs_epi16(grid, _mm256_sub_epi8(_mm256_xor_si256(s, q), s));
                    sumi = _mm256_add_epi32(sumi, udot_scale(dot, 2 * (sc[ib] & 0xF) as i32 + 1, 2 * (sc[ib] >> 4) as i32 + 1));
                }
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d16(b) * y.d), _mm256_cvtepi32_ps(sumi), acc);
            }
            0.125 * hsum(acc)
        }
    }

    /// Products of 32 IQ1 grid values (-1, 0, 1) with q8, in pairs (maddubs needs an unsigned first operand).
    #[inline(always)]
    unsafe fn dot1(g: __m256i, q: __m256i) -> __m256i { unsafe { _mm256_maddubs_epi16(_mm256_sign_epi8(g, g), _mm256_sign_epi8(q, g)) } }

    pub fn dot_iq1s(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let mut acc = _mm256_setzero_ps();
            let mut sumf = 0.0;
            for (b, y) in row.chunks_exact(50).zip(xq) {
                let mut sumi = _mm256_setzero_si256();
                let mut s1 = 0i32;
                for ib in 0..8 {
                    let qh = u16::from_le_bytes([b[34 + 2 * ib], b[35 + 2 * ib]]) as usize;
                    let g = |l: usize| IQ1S_GRID[b[2 + 4 * ib + l] as usize | ((qh >> (3 * l)) & 7) << 8] as i64;
                    let ls = 2 * ((qh >> 12) & 7) as i32 + 1;
                    let p = dot1(_mm256_set_epi64x(g(3), g(2), g(1), g(0)), q8(y, 32 * ib));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(p, _mm256_set1_epi16(ls as i16)));
                    s1 += ls * if qh & 0x8000 != 0 { -1 } else { 1 } * (y.bs[2 * ib] + y.bs[2 * ib + 1]) as i32;
                }
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d16(b) * y.d), _mm256_cvtepi32_ps(sumi), acc);
                sumf += d16(b) * y.d * 0.125 * s1 as f32;
            }
            hsum(acc) + sumf
        }
    }

    pub fn dot_iq1m(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            // Group l of 8 values takes its delta sign from bit 3 (l even) or 7 (l odd) of qh byte l / 2.
            let pick = _mm256_set_m128i(_mm_set1_epi8(1), _mm_setzero_si128());
            let bit = _mm256_set_epi64x(0x8080808080808080u64 as i64, 0x0808080808080808, 0x8080808080808080u64 as i64, 0x0808080808080808);
            let one = _mm256_set1_epi8(1);
            let mut acc = _mm256_setzero_ps();
            for (b, y) in row.chunks_exact(56).zip(xq) {
                // Weights 8 * grid + delta (-9..=9), so one 0.125 at the end gives grid + delta / 8.
                let mut sumi = _mm256_setzero_si256();
                let sc = u64::from_le_bytes(b[48..56].try_into().unwrap());
                for ib in 0..8 {
                    let ls = |h: usize| 2 * ((sc >> (16 * (ib / 2) + 6 * (ib % 2) + 3 * h)) & 7) as i32 + 1;
                    let qh = |l: usize| (b[32 + 2 * ib + l / 2] as usize) >> (4 * (l % 2));
                    let g = |l: usize| IQ1S_GRID[b[4 * ib + l] as usize | (qh(l) & 7) << 8] as i64;
                    let g = _mm256_set_epi64x(g(3), g(2), g(1), g(0));
                    let (g2, h) = (_mm256_add_epi8(g, g), _mm256_set1_epi16(u16::from_le_bytes([b[32 + 2 * ib], b[33 + 2 * ib]]) as i16));
                    let neg = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(h, pick), bit), bit);
                    let g8 = _mm256_add_epi8(_mm256_add_epi8(g2, g2), _mm256_add_epi8(g2, g2));
                    let p = dot1(_mm256_add_epi8(g8, _mm256_or_si256(neg, one)), q8(y, 32 * ib));
                    sumi = _mm256_add_epi32(sumi, udot_scale(p, ls(0), ls(1)));
                }
                acc = _mm256_fmadd_ps(_mm256_set1_ps(iq1m_d(b) * y.d), _mm256_cvtepi32_ps(sumi), acc);
            }
            0.125 * hsum(acc)
        }
    }

    pub fn dot_q2k(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let mut acc = _mm256_setzero_ps();
            for (b, y) in row.chunks_exact(84).zip(xq) {
                let sc = &b[0..16];
                let mut sumi = _mm256_setzero_si256();
                for n in 0..2 {
                    let bits = ld(b, 16 + 32 * n);
                    for k in 0..4 {
                        let g = 8 * n + 2 * k;
                        sumi = _mm256_add_epi32(sumi, udot32(crumb(bits, k), q8(y, 32 * (4 * n + k)), (sc[g] & 0xF) as i32, (sc[g + 1] & 0xF) as i32));
                    }
                }
                let mins = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(_mm_loadu_si128(sc.as_ptr() as *const __m128i), 4), _mm_set1_epi8(0xF)));
                let m = _mm256_madd_epi16(mins, _mm256_loadu_si256(y.bs.as_ptr() as *const __m256i));
                acc = _mm256_fmadd_ps(_mm256_set1_ps(y.d * d16(&b[80..])), _mm256_cvtepi32_ps(sumi), acc);
                acc = _mm256_fmadd_ps(_mm256_set1_ps(-y.d * d16(&b[82..])), _mm256_cvtepi32_ps(m), acc);
            }
            hsum(acc)
        }
    }

    pub fn dot_q3k(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let mut acc = _mm256_setzero_ps();
            for (b, y) in row.chunks_exact(110).zip(xq) {
                let (hm, sc) = (ld(b, 0), crate::quant::q3k_scales(b));
                let mut sumi = _mm256_setzero_si256();
                for n in 0..2 {
                    let bits = ld(b, 32 + 32 * n);
                    for k in 0..4 {
                        let kk = 4 * n + k;
                        // Weight = crumb - 4 where the high bit is clear; signs moved onto q8 for maddubs.
                        let mask = _mm256_set1_epi8(1 << kk);
                        let low = _mm256_cmpeq_epi8(_mm256_and_si256(hm, mask), _mm256_setzero_si256());
                        let w = _mm256_sub_epi8(crumb(bits, k), _mm256_and_si256(low, _mm256_set1_epi8(4)));
                        let p = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(q8(y, 32 * kk), w));
                        let s = _mm256_set_m128i(_mm_set1_epi16(sc[2 * kk + 1] as i16 - 32), _mm_set1_epi16(sc[2 * kk] as i16 - 32));
                        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(p, s));
                    }
                }
                acc = _mm256_fmadd_ps(_mm256_set1_ps(y.d * d16(&b[108..])), _mm256_cvtepi32_ps(sumi), acc);
            }
            hsum(acc)
        }
    }

    pub fn dot_q4k(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let m4 = _mm256_set1_epi8(0xF);
            let mut acc = _mm256_setzero_ps();
            let mut sumf = 0.0;
            for (b, y) in row.chunks_exact(144).zip(xq) {
                let sm: [(u8, u8); 8] = std::array::from_fn(|j| crate::quant::scale_min_k4(j, &b[4..16]));
                let mut sumi = _mm256_setzero_si256();
                for j in 0..4 {
                    let bits = ld(b, 16 + 32 * j);
                    let (s0, s1) = (sm[2 * j].0 as i32, sm[2 * j + 1].0 as i32);
                    sumi = _mm256_add_epi32(sumi, udot32(_mm256_and_si256(bits, m4), q8(y, 64 * j), s0, s0));
                    sumi = _mm256_add_epi32(sumi, udot32(_mm256_and_si256(_mm256_srli_epi16(bits, 4), m4), q8(y, 64 * j + 32), s1, s1));
                }
                let m: i32 = (0..8).map(|j| sm[j].1 as i32 * (y.bs[2 * j] + y.bs[2 * j + 1]) as i32).sum();
                acc = _mm256_fmadd_ps(_mm256_set1_ps(y.d * d16(b)), _mm256_cvtepi32_ps(sumi), acc);
                sumf -= y.d * d16(&b[2..]) * m as f32;
            }
            hsum(acc) + sumf
        }
    }
}

fn int_dot(ty: GgmlType) -> Option<fn(&[u8], &[Q8K]) -> f32> {
    #[cfg(target_feature = "avx2")]
    if std::env::var_os("STRATA_SCALAR").is_none() {
        return match ty { GgmlType::IQ2XS => Some(avx2::dot_iq2xs), GgmlType::IQ3XXS => Some(avx2::dot_iq3xxs), GgmlType::IQ4XS => Some(avx2::dot_iq4xs), GgmlType::IQ3S => Some(avx2::dot_iq3s), GgmlType::Q6K => Some(dot_q6k), GgmlType::IQ2XXS => Some(avx2::dot_iq2xxs), GgmlType::IQ2S => Some(avx2::dot_iq2s), GgmlType::IQ1S => Some(avx2::dot_iq1s), GgmlType::IQ1M => Some(avx2::dot_iq1m), GgmlType::Q2K => Some(avx2::dot_q2k), GgmlType::Q3K => Some(avx2::dot_q3k), GgmlType::Q4K => Some(avx2::dot_q4k), _ => None };
    }
    match ty { GgmlType::IQ2XS => Some(dot_iq2xs), GgmlType::IQ3XXS => Some(dot_iq3xxs), GgmlType::IQ4XS => Some(dot_iq4xs), GgmlType::IQ3S => Some(dot_iq3s), GgmlType::Q6K => Some(dot_q6k), GgmlType::IQ2XXS => Some(dot_iq2xxs), GgmlType::IQ2S => Some(dot_iq2s), GgmlType::IQ1S => Some(dot_iq1s), GgmlType::IQ1M => Some(dot_iq1m), GgmlType::Q2K => Some(dot_q2k), GgmlType::Q3K => Some(dot_q3k), GgmlType::Q4K => Some(dot_q4k), _ => None }
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

/// Rows `out.len()` of a batch of matrices: out[i] = mats[i / rows] row (i % rows) · x of that matrix, where
/// matrix j uses xs[j * xs.len() / mats.len()] (one x for all, or one each), so a shared x is quantized once.
fn batch_rows(mats: &[Tensor], xs: &[&[f32]], out: &mut [f32]) {
    let rows = out.len() / mats.len();
    let dot = int_dot(mats[0].ty);
    let xq: Vec<Vec<Q8K>> = if dot.is_some() { xs.iter().map(|x| quantize_q8k(x)).collect() } else { vec![] };
    out.par_chunks_mut(16).enumerate().for_each_init(|| vec![0f32; mats[0].cols], |buf, (c, ys)| {
        for (k, v) in ys.iter_mut().enumerate() {
            let i = c * 16 + k;
            let (t, r, x) = (&mats[i / rows], i % rows, i / rows * xs.len() / mats.len());
            let rb = t.data.len() / t.rows;
            *v = match dot {
                Some(d) => d(&t.data[r * rb..(r + 1) * rb], &xq[x]),
                None => { t.row(r, buf); buf.iter().zip(xs[x]).map(|(a, b)| a * b).sum() }
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
        batch_rows(&gu, &[h], &mut a);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Every integer kernel (scalar and, when built, AVX2) equals the f32 dequantized dot on random blocks.
    #[test]
    fn int_dots_match_dequant() {
        let mut r = 0x9E3779B97F4A7C15u64;
        let mut rnd = || { r ^= r << 13; r ^= r >> 7; r ^= r << 17; r as u8 };
        let x: Vec<f32> = (0..1024).map(|i| ((i as f32 * 0.7548777).fract() - 0.5) * 2.0).collect();
        let xq = quantize_q8k(&x);
        for ty in [GgmlType::IQ2XS, GgmlType::IQ3XXS, GgmlType::IQ3S, GgmlType::IQ4XS, GgmlType::Q6K, GgmlType::Q2K, GgmlType::Q3K, GgmlType::Q4K, GgmlType::IQ2XXS, GgmlType::IQ2S, GgmlType::IQ1S, GgmlType::IQ1M] {
            let mut row: Vec<u8> = (0..4 * ty.block_bytes()).map(|_| rnd()).collect();
            // Offsets of the f16 scales (d, and dmin where the type has one).
            let fs: &[usize] = match ty { GgmlType::Q6K => &[208], GgmlType::Q2K => &[80, 82], GgmlType::Q3K => &[108], GgmlType::Q4K => &[0, 2], GgmlType::IQ1M => &[], _ => &[0] };
            for b in row.chunks_exact_mut(ty.block_bytes()) { for &o in fs { b[o..o + 2].copy_from_slice(&f16::from_f32(0.01).to_le_bytes()); } }
            // IQ1_M keeps d in the top nibbles of its four scale words.
            if ty == GgmlType::IQ1M {
                let d = f16::from_f32(0.01).to_bits();
                for b in row.chunks_exact_mut(56) { for i in 0..4 { b[49 + 2 * i] = b[49 + 2 * i] & 0xF | ((d >> (4 * i)) as u8 & 0xF) << 4; } }
            }
            let mut w = vec![0f32; 1024];
            crate::quant::dequant_row(ty, &row, &mut w);
            let want: f64 = w.iter().enumerate().map(|(i, &v)| v as f64 * (xq[i / 256].d * xq[i / 256].qs[i % 256] as f32) as f64).sum();
            let scalar = match ty { GgmlType::IQ2XS => dot_iq2xs, GgmlType::IQ3XXS => dot_iq3xxs, GgmlType::IQ3S => dot_iq3s, GgmlType::Q6K => dot_q6k, GgmlType::Q2K => dot_q2k, GgmlType::Q3K => dot_q3k, GgmlType::Q4K => dot_q4k, GgmlType::IQ2XXS => dot_iq2xxs, GgmlType::IQ2S => dot_iq2s, GgmlType::IQ1S => dot_iq1s, GgmlType::IQ1M => dot_iq1m, _ => dot_iq4xs };
            for got in [scalar(&row, &xq), int_dot(ty).unwrap()(&row, &xq)] {
                assert!(((got as f64 - want) / want.abs().max(1e-3)).abs() < 1e-5, "{}: {got} vs {want}", ty.name());
            }
        }
    }
}
