//! Fast CPU routed-expert path: activations quantized to Q8_K blocks, integer dot products per weight type
//! (ported from llama.cpp `ggml_vec_dot_*_q8_K_generic`, MIT License, ggml authors). Types without an integer
//! kernel fall back to f32 dequantization.

use crate::model::Model;
use crate::ops;
use crate::quant::{GgmlType, KVALUES_FP4, tables::*};
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

/// Quantizes `x` in blocks of 256; a short last block is zero-padded (Q2_0 rows can hold 64 * k values).
pub fn quantize_q8k(x: &[f32]) -> Vec<Q8K> {
    x.chunks(256).map(|b| {
        let mut pad = [0f32; 256];
        let b = if b.len() == 256 { b } else { pad[..b.len()].copy_from_slice(b); &pad[..] };
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

/// IQ4_NL: 32 values d * KVALUES_IQ4NL[q] per block, eight blocks per Q8_K block.
fn dot_iq4nl(row: &[u8], xq: &[Q8K]) -> f32 {
    row.chunks_exact(18).enumerate().map(|(i, b)| {
        let (y, o) = (&xq[i / 8], i % 8 * 32);
        let s: i32 = (0..16).map(|j| {
            let q = b[2 + j];
            KVALUES_IQ4NL[(q & 0xF) as usize] as i32 * y.qs[o + j] as i32 + KVALUES_IQ4NL[(q >> 4) as usize] as i32 * y.qs[o + j + 16] as i32
        }).sum();
        d16(b) * y.d * s as f32
    }).sum()
}

/// NVFP4: 64 values per block in four sub-blocks of 16, each with its own UE4M3 scale; four blocks per Q8_K block.
fn dot_nvfp4(row: &[u8], xq: &[Q8K]) -> f32 {
    row.chunks_exact(36).enumerate().map(|(i, b)| {
        let (y, o) = (&xq[i / 4], i % 4 * 64);
        (0..4).map(|s| {
            let q8 = &y.qs[o + 16 * s..];
            let v: i32 = (0..8).map(|j| {
                let q = b[4 + 8 * s + j];
                KVALUES_FP4[(q & 0xF) as usize] as i32 * q8[j] as i32 + KVALUES_FP4[(q >> 4) as usize] as i32 * q8[j + 8] as i32
            }).sum();
            crate::quant::ue4m3(b[s]) * y.d * v as f32
        }).sum::<f32>()
    }).sum()
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

/// Q2_0: 64 values (q - 1) * d with q in 0..4, four to a byte; four blocks per Q8_K block.
fn dot_q2_0(row: &[u8], xq: &[Q8K]) -> f32 {
    row.chunks_exact(18).enumerate().map(|(i, b)| {
        let (y, o) = (&xq[i / 4], i % 4 * 64);
        let s: i32 = (0..64).map(|j| (b[2 + j / 4] >> (2 * (j % 4)) & 3) as i32 * y.qs[o + j] as i32).sum();
        let m: i32 = y.bs[o / 16..o / 16 + 4].iter().map(|&v| v as i32).sum();
        d16(b) * y.d * (s - m) as f32
    }).sum()
}

/// TQ1_0: five base-3 trits per byte in qs, four in qh; values -1, 0, 1 times d.
fn dot_tq1_0(row: &[u8], xq: &[Q8K]) -> f32 {
    let mut sumf = 0.0;
    for (b, y) in row.chunks_exact(54).zip(xq) {
        let trit = |v: u8, n: usize| { const POW3: [u8; 5] = [1, 3, 9, 27, 81]; ((v.wrapping_mul(POW3[n]) as u16 * 3) >> 8) as i32 - 1 };
        let mut s = 0i32;
        for n in 0..5 { for m in 0..32 { s += trit(b[m], n) * y.qs[32 * n + m] as i32; } }
        for n in 0..5 { for m in 0..16 { s += trit(b[32 + m], n) * y.qs[160 + 16 * n + m] as i32; } }
        for n in 0..4 { for j in 0..4 { s += trit(b[48 + j], n) * y.qs[240 + 4 * n + j] as i32; } }
        sumf += d16(&b[52..]) * y.d * s as f32;
    }
    sumf
}

/// TQ2_0: 64 bytes of 2-bit codes (code - 1 in -1..2), then f16 d; 256 values a block.
fn dot_tq2_0(row: &[u8], xq: &[Q8K]) -> f32 {
    let mut sumf = 0.0;
    for (b, y) in row.chunks_exact(66).zip(xq) {
        let mut s = 0i32;
        for j in 0..2 { for l in 0..4 { for m in 0..32 {
            let code = (b[32 * j + m] >> (2 * l)) & 3;
            s += (code as i32 - 1) * y.qs[128 * j + 32 * l + m] as i32;
        } } }
        sumf += d16(&b[64..]) * y.d * s as f32;
    }
    sumf
}

/// Q4_0 (`T` = 2, 18 bytes), Q4_1 (3, 20), Q5_0 (6, 22), Q5_1 (7, 24): block bytes, offset of the 32 high bits
/// (0 = none) and of the 16 bytes of nibbles.
const fn q4_layout(t: u8) -> (usize, usize, usize) { match t { 2 => (18, 0, 2), 3 => (20, 0, 4), 6 => (22, 2, 6), _ => (24, 4, 8) } }
/// The block's m in value = d * q + m (q the unsigned 4- or 5-bit code): its own for Q4_1 / Q5_1, -8 d or -16 d for
/// Q4_0 / Q5_0.
fn q4_min(t: u8, b: &[u8]) -> f32 { match t { 3 | 7 => d16(&b[2..]), 2 => -8.0 * d16(b), _ => -16.0 * d16(b) } }

/// Q4_0, Q4_1, Q5_0, Q5_1 (`q4_layout`): 32 values d * q + m per block, eight blocks per Q8_K block.
fn dot_q4<const T: u8>(row: &[u8], xq: &[Q8K]) -> f32 {
    let (bb, oh, oq) = q4_layout(T);
    row.chunks_exact(bb).enumerate().map(|(i, b)| {
        let (y, o) = (&xq[i / 8], i % 8 * 32);
        let (qh, qs) = if oh > 0 { (u32::from_le_bytes(b[oh..oh + 4].try_into().unwrap()), &b[oq..oq + 16]) } else { (0, &b[oq..oq + 16]) };
        let s: i32 = (0..16).map(|j| {
            let (l, h) = ((qs[j] & 0xF) as i32 | ((qh >> j) & 1) as i32 * 16, (qs[j] >> 4) as i32 | ((qh >> (j + 16)) & 1) as i32 * 16);
            l * y.qs[o + j] as i32 + h * y.qs[o + j + 16] as i32
        }).sum();
        y.d * (d16(b) * s as f32 + q4_min(T, b) * (y.bs[o / 16] as i32 + y.bs[o / 16 + 1] as i32) as f32)
    }).sum()
}

/// Q8_0: 32 values d * q per block, eight blocks per Q8_K block.
fn dot_q8_0(row: &[u8], xq: &[Q8K]) -> f32 {
    row.chunks_exact(34).enumerate().map(|(i, b)| {
        let (y, o) = (&xq[i / 8], i % 8 * 32);
        let s: i32 = (0..32).map(|j| b[2 + j] as i8 as i32 * y.qs[o + j] as i32).sum();
        d16(b) * y.d * s as f32
    }).sum()
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

/// Q5_K: Q4_K plus a fifth bit per value from `qh` (bit 2 * (i / 64) + i / 32 % 2 of byte i % 32).
fn dot_q5k(row: &[u8], xq: &[Q8K]) -> f32 {
    let mut sumf = 0.0;
    for (b, y) in row.chunks_exact(176).zip(xq) {
        let sm: [(u8, u8); 8] = std::array::from_fn(|j| crate::quant::scale_min_k4(j, &b[4..16]));
        let mut a = [0i8; 256];
        for j in 0..4 {
            for l in 0..32 {
                let (q, h) = (b[48 + 32 * j + l], b[16 + l] >> (2 * j));
                a[64 * j + l] = (q & 0xF | (h & 1) << 4) as i8;
                a[64 * j + 32 + l] = (q >> 4 | (h & 2) << 3) as i8;
            }
        }
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
                acc = _mm256_fmadd_ps(ds(b, y.d), _mm256_cvtepi32_ps(sumi), acc);
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
                acc = _mm256_fmadd_ps(ds(b, y.d), _mm256_cvtepi32_ps(sumi), acc);
            }
            0.25 * hsum(acc)
        }
    }

    pub fn dot_iq3s(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            // Byte k of a lane picks sign byte k / 8, then bit k % 8 (llama.cpp k_mask1 / k_mask2).
            let m1 = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0);
            let m2 = _mm256_set1_epi64x(0x8040201008040201u64 as i64);
            let shl = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
            let mut acc = _mm256_setzero_ps();
            for (b, y) in row.chunks_exact(110).zip(xq) {
                let (qs, qh) = (&b[2..66], &b[66..74]);
                let mut sumi = _mm256_setzero_si256();
                for ib in 0..8 {
                    let lo = _mm256_cvtepu8_epi32(_mm_loadl_epi64(qs[8 * ib..].as_ptr() as *const __m128i));
                    let hi = _mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32(qh[ib] as i32), shl), _mm256_set1_epi32(256));
                    let grid = _mm256_i32gather_epi32(IQ3S_GRID.as_ptr() as *const i32, _mm256_or_si256(lo, hi), 4);
                    let sb = i32::from_le_bytes(b[74 + 4 * ib..78 + 4 * ib].try_into().unwrap());
                    let s = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(sb), m1), m2), m2);
                    let q8 = _mm256_loadu_si256(y.qs.as_ptr().add(32 * ib) as *const __m256i);
                    let dot = _mm256_maddubs_epi16(grid, _mm256_sub_epi8(_mm256_xor_si256(s, q8), s));
                    let ls = 2 * (b[106 + ib / 2] >> (4 * (ib % 2)) & 0xF) as i16 + 1;
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(dot, _mm256_set1_epi16(ls)));
                }
                acc = _mm256_fmadd_ps(ds(b, y.d), _mm256_cvtepi32_ps(sumi), acc);
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
                acc = _mm256_fmadd_ps(ds(b, y.d), _mm256_cvtepi32_ps(sumi), acc);
            }
            hsum(acc)
        }
    }

    /// Ternary `TQ1_0` (54 bytes = 256 trits in `{-1, 0, +1}`, f16 `d`): unpack `u_n = trit + 1` in `{0, 1, 2}`
    /// via `(3 * (v * 3^n mod 256)) >> 8` (`> 85` and `> 170`), dot against signed `Q8K` with `_mm256_maddubs_epi16`,
    /// and subtract `y.bs` once per block to account for the `-1` offset.
    pub fn dot_tq1_0(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let flip = _mm256_set1_epi8(-128i8);
            let th1 = _mm256_set1_epi8(-43i8);
            let th2 = _mm256_set1_epi8(42i8);
            let zero = _mm256_setzero_si256();
            let ones16 = _mm256_set1_epi16(1);
            let trits_u = |q: __m256i| -> __m256i {
                let qb = _mm256_xor_si256(q, flip);
                _mm256_sub_epi8(zero, _mm256_add_epi8(_mm256_cmpgt_epi8(qb, th1), _mm256_cmpgt_epi8(qb, th2)))
            };
            let mul3 = |q: __m256i| _mm256_add_epi8(_mm256_add_epi8(q, q), q);
            let mul3_128 = |q: __m128i| _mm_add_epi8(_mm_add_epi8(q, q), q);
            let mut acc = _mm256_setzero_ps();
            for (b, y) in row.chunks_exact(54).zip(xq) {
                let mut sum16 = zero;
                let mut q0 = ld(b, 0);
                for n in 0..5 {
                    sum16 = _mm256_add_epi16(sum16, _mm256_maddubs_epi16(trits_u(q0), q8(y, 32 * n)));
                    if n < 4 { q0 = mul3(q0); }
                }
                let q1_0 = _mm_loadu_si128(b.as_ptr().add(32) as *const __m128i);
                let p01 = _mm256_set_m128i(mul3_128(q1_0), q1_0);
                sum16 = _mm256_add_epi16(sum16, _mm256_maddubs_epi16(trits_u(p01), q8(y, 160)));
                let p23 = mul3(mul3(p01));
                sum16 = _mm256_add_epi16(sum16, _mm256_maddubs_epi16(trits_u(p23), q8(y, 192)));
                let q1_4 = mul3_128(_mm256_extracti128_si256::<1>(p23));
                let bm = |m: u8| u32::from_le_bytes([b[48].wrapping_mul(m), b[49].wrapping_mul(m), b[50].wrapping_mul(m), b[51].wrapping_mul(m)]) as i32;
                let qh = _mm_set_epi32(bm(27), bm(9), bm(3), bm(1));
                let p4h = _mm256_set_m128i(qh, q1_4);
                sum16 = _mm256_add_epi16(sum16, _mm256_maddubs_epi16(trits_u(p4h), q8(y, 224)));
                let bs = _mm256_loadu_si256(y.bs.as_ptr() as *const __m256i);
                let sum32 = _mm256_madd_epi16(_mm256_sub_epi16(sum16, bs), ones16);
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d16(&b[52..]) * y.d), _mm256_cvtepi32_ps(sum32), acc);
            }
            hsum(acc)
        }
    }

    /// NVFP4 (see `super::dot_nvfp4`): 64 values per 36-byte block across 4 sub-blocks of 16, processed two sub-blocks
    /// (32 values) per AVX2 step with `_mm_shuffle_epi8` + `_mm256_maddubs_epi16`.
    pub fn dot_nvfp4(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let values = _mm_loadu_si128(crate::quant::KVALUES_FP4.as_ptr() as *const __m128i);
            let (m4, ones) = (_mm_set1_epi8(0xF), _mm256_set1_epi16(1));
            let mut acc = _mm256_setzero_ps();
            for (i, b) in row.chunks_exact(36).enumerate() {
                let (y, o) = (&xq[i / 4], (i % 4) * 64);
                for h in 0..2 {
                    let s0 = 2 * h;
                    let bits0 = _mm_loadl_epi64(b.as_ptr().add(4 + 8 * s0) as *const __m128i);
                    let bits1 = _mm_loadl_epi64(b.as_ptr().add(12 + 8 * s0) as *const __m128i);
                    let w0 = _mm_unpacklo_epi64(
                        _mm_shuffle_epi8(values, _mm_and_si128(bits0, m4)),
                        _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(bits0, 4), m4)),
                    );
                    let w1 = _mm_unpacklo_epi64(
                        _mm_shuffle_epi8(values, _mm_and_si128(bits1, m4)),
                        _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(bits1, 4), m4)),
                    );
                    let w = _mm256_set_m128i(w1, w0);
                    let p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(q8(y, o + 32 * h), w));
                    let p32 = _mm256_madd_epi16(p16, ones);
                    let d = _mm256_set_m128(
                        _mm_set1_ps(crate::quant::ue4m3(b[s0 + 1]) * y.d),
                        _mm_set1_ps(crate::quant::ue4m3(b[s0]) * y.d),
                    );
                    acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(p32), acc);
                }
            }
            hsum(acc)
        }
    }

    /// IQ4_NL (see `super::dot_iq4nl`): two 32-value blocks per step so the two lookups and FMAs overlap.
    pub fn dot_iq4nl(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let values = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
            let (m4, ones) = (_mm_set1_epi8(0xF), _mm256_set1_epi16(1));
            let (mut acc0, mut acc1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
            let blk = |b: &[u8], y: &Q8K, o: usize| {
                let bits = _mm_loadu_si128(b.as_ptr().add(2) as *const __m128i);
                let w = _mm256_set_m128i(_mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(bits, 4), m4)),
                                         _mm_shuffle_epi8(values, _mm_and_si128(bits, m4)));
                let p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(q8(y, o), w));
                (_mm256_cvtepi32_ps(_mm256_madd_epi16(p16, ones)), ds(b, y.d))
            };
            let mut it = row.chunks_exact(36).enumerate();
            for (i, b2) in &mut it {
                let (y, o) = (&xq[2 * i / 8], 2 * i % 8 * 32);
                let (s0, d0) = blk(&b2[..18], y, o);
                let (s1, d1) = blk(&b2[18..], y, o + 32);
                acc0 = _mm256_fmadd_ps(d0, s0, acc0);
                acc1 = _mm256_fmadd_ps(d1, s1, acc1);
            }
            let n = row.len() / 18;
            if n % 2 == 1 {
                let i = n - 1;
                let (s, d) = blk(&row[18 * i..18 * i + 18], &xq[i / 8], i % 8 * 32);
                acc0 = _mm256_fmadd_ps(d, s, acc0);
            }
            hsum(_mm256_add_ps(acc0, acc1))
        }
    }

    /// `dot_iq4nl` of one row with four activations (see `super::int_dot4`).
    pub fn dot4_iq4nl(row: &[u8], xq: [&[Q8K]; 4]) -> [f32; 4] {
        unsafe {
            let values = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
            let (m4, ones) = (_mm_set1_epi8(0xF), _mm256_set1_epi16(1));
            // Sums of the even and of the odd blocks, as the two accumulators of `dot_iq4nl`.
            let (mut even, mut odd) = ([_mm256_setzero_ps(); 4], [_mm256_setzero_ps(); 4]);
            let blk = |b: &[u8], i: usize, acc: &mut [__m256; 4]| {
                let bits = _mm_loadu_si128(b.as_ptr().add(2) as *const __m128i);
                let w = _mm256_set_m128i(_mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(bits, 4), m4)),
                                         _mm_shuffle_epi8(values, _mm_and_si128(bits, m4)));
                let aw = _mm256_sign_epi8(w, w);
                for k in 0..4 {
                    let y = &xq[k][i / 8];
                    let p16 = _mm256_maddubs_epi16(aw, _mm256_sign_epi8(q8(y, i % 8 * 32), w));
                    acc[k] = _mm256_fmadd_ps(ds(b, y.d), _mm256_cvtepi32_ps(_mm256_madd_epi16(p16, ones)), acc[k]);
                }
            };
            let mut it = row.chunks_exact(36).enumerate();
            for (i, b2) in &mut it {
                blk(&b2[..18], 2 * i, &mut even);
                blk(&b2[18..], 2 * i + 1, &mut odd);
            }
            let n = row.len() / 18;
            if n % 2 == 1 { blk(&row[18 * (n - 1)..18 * n], n - 1, &mut even); }
            std::array::from_fn(|k| hsum(_mm256_add_ps(even[k], odd[k])))
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
    /// Eight copies of `d16(b) * d`, the same bits. A scalar f16 load compiles to `vpinsrw` into a register that the
    /// accumulator may hold, and then each block waits for the block before it; a broadcast has no such input.
    #[inline(always)]
    unsafe fn ds(b: &[u8], d: f32) -> __m256 { unsafe {
        #[cfg(target_feature = "f16c")]
        return _mm256_mul_ps(_mm256_cvtph_ps(_mm_set1_epi16(i16::from_le_bytes([b[0], b[1]]))), _mm256_set1_ps(d));
        #[cfg(not(target_feature = "f16c"))]
        _mm256_set1_ps(d16(b) * d)
    } }
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
                acc = _mm256_fmadd_ps(ds(b, y.d), _mm256_cvtepi32_ps(sumi), acc);
            }
            0.125 * hsum(acc)
        }
    }

    pub fn dot_iq2s(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            // As in dot_iq3s: byte k of a lane picks sign byte k / 8, then bit k % 8.
            let m1 = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0);
            let m2 = _mm256_set1_epi64x(0x8040201008040201u64 as i64);
            // Grid index high bits: word p of the 32 indices takes qh[p / 4] << (8 - 2 * (p % 4)) & 0x300.
            let hq0 = _mm256_set_epi64x(0x8003800380038003u64 as i64, 0x8002800280028002u64 as i64, 0x8001800180018001u64 as i64, 0x8000800080008000u64 as i64);
            let hq1 = _mm256_add_epi16(hq0, _mm256_set1_epi16(4));
            let hmul = _mm256_set1_epi64x(0x0004_0010_0040_0100);
            let h3 = _mm256_set1_epi16(0x300);
            // Scale words: the low lane takes byte 2 ib of the 16 block scales, the high lane byte 2 ib + 1.
            let sm0 = _mm256_set_epi64x(0x8001800180018001u64 as i64, 0x8001800180018001u64 as i64, 0x8000800080008000u64 as i64, 0x8000800080008000u64 as i64);
            let (sstep, n15) = (_mm256_set1_epi16(2), _mm_set1_epi8(15));
            let mut acc = _mm256_setzero_ps();
            for (b, y) in row.chunks_exact(82).zip(xq) {
                let qh = _mm256_broadcastsi128_si256(_mm_loadl_epi64(b[66..74].as_ptr() as *const __m128i));
                let mut idx = [0u16; 32];
                for (h, hq) in [hq0, hq1].into_iter().enumerate() {
                    let lo = _mm256_cvtepu8_epi16(_mm_loadu_si128(b[2 + 16 * h..18 + 16 * h].as_ptr() as *const __m128i));
                    let hi = _mm256_and_si256(_mm256_mullo_epi16(_mm256_shuffle_epi8(qh, hq), hmul), h3);
                    _mm256_storeu_si256(idx[16 * h..].as_mut_ptr() as *mut __m256i, _mm256_or_si256(lo, hi));
                }
                let sc = _mm_loadl_epi64(b[74..82].as_ptr() as *const __m128i);
                let sc = _mm_unpacklo_epi8(_mm_and_si128(sc, n15), _mm_and_si128(_mm_srli_epi16(sc, 4), n15));
                let sc = _mm256_broadcastsi128_si256(_mm_add_epi8(_mm_add_epi8(sc, sc), _mm_set1_epi8(1)));
                let (mut sm, mut sumi) = (sm0, _mm256_setzero_si256());
                for ib in 0..8 {
                    let g = |l: usize| IQ2S_GRID[idx[4 * ib + l] as usize & 1023] as i64;
                    let grid = _mm256_set_epi64x(g(3), g(2), g(1), g(0));
                    let sb = i32::from_le_bytes(b[34 + 4 * ib..38 + 4 * ib].try_into().unwrap());
                    let s = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(sb), m1), m2), m2);
                    let dot = _mm256_maddubs_epi16(grid, _mm256_sub_epi8(_mm256_xor_si256(s, q8(y, 32 * ib)), s));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(dot, _mm256_shuffle_epi8(sc, sm)));
                    sm = _mm256_add_epi16(sm, sstep);
                }
                acc = _mm256_fmadd_ps(ds(b, y.d), _mm256_cvtepi32_ps(sumi), acc);
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
                for ib in 0..8 {
                    let qh = u16::from_le_bytes([b[34 + 2 * ib], b[35 + 2 * ib]]) as usize;
                    let g = |l: usize| IQ1S_GRID[b[2 + 4 * ib + l] as usize | ((qh >> (3 * l)) & 7) << 8] as i64;
                    let ls = 2 * ((qh >> 12) & 7) as i32 + 1;
                    let p = dot1(_mm256_set_epi64x(g(3), g(2), g(1), g(0)), q8(y, 32 * ib));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(p, _mm256_set1_epi16(ls as i16)));
                }
                // Delta term: sum of ±scale (sign in bit 15 of qh) times each 32 values' q8 sum.
                let qh = _mm_loadu_si128(b[34..50].as_ptr() as *const __m128i);
                let ls = _mm_or_si128(_mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh, 12), _mm_set1_epi16(7)), 1), _mm_set1_epi16(1));
                let ls = _mm_sign_epi16(ls, _mm_or_si128(qh, _mm_set1_epi16(1)));
                let bs = |i: usize| _mm_loadu_si128(y.bs[8 * i..].as_ptr() as *const __m128i);
                let s = _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(ls, ls), bs(0)), _mm_madd_epi16(_mm_unpackhi_epi16(ls, ls), bs(1)));
                let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
                let s1 = _mm_cvtsi128_si32(_mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01)));
                acc = _mm256_fmadd_ps(ds(b, y.d), _mm256_cvtepi32_ps(sumi), acc);
                sumf += d16(b) * y.d * 0.125 * s1 as f32;
            }
            hsum(acc) + sumf
        }
    }

    /// IQ1_M weights plus 9 (8 * grid + 8, plus 2 when the delta is +1), indexed by the 8 grid bits, then 3 high grid
    /// bits and the delta sign bit (a qh nibble).
    static IQ1M_W: std::sync::LazyLock<Box<[u64; 4096]>> = std::sync::LazyLock::new(|| {
        let mut t = Box::new([0u64; 4096]);
        for (i, w) in t.iter_mut().enumerate() {
            let d = if i & 2048 != 0 { 8 } else { 10 };
            *w = u64::from_le_bytes(IQ1S_GRID[i & 2047].to_le_bytes().map(|g| (8 * g as i8 + d) as u8));
        }
        t
    });

    pub fn dot_iq1m(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let (t, nine) = (&**IQ1M_W, _mm256_set1_epi8(9));
            let mut acc = _mm256_setzero_ps();
            for (b, y) in row.chunks_exact(56).zip(xq) {
                // Weights 8 * grid + delta (-9..=9), so one 0.125 at the end gives grid + delta / 8. The table holds
                // them plus 9 (unsigned for maddubs), taken off the pair sums again with 9 * q8.
                let sc = u64::from_le_bytes(b[48..56].try_into().unwrap());
                let ls: [i16; 16] = std::array::from_fn(|i| 2 * ((sc >> (16 * (i / 4) + 3 * (i % 4))) & 7) as i16 + 1);
                let mut sumi = _mm256_setzero_si256();
                for ib in 0..8 {
                    let ls = |h: usize| ls[2 * ib + h] as i32;
                    let qh = u16::from_le_bytes([b[32 + 2 * ib], b[33 + 2 * ib]]) as usize;
                    let g = |l: usize| t[b[4 * ib + l] as usize | ((qh >> (4 * l)) & 15) << 8] as i64;
                    let q = q8(y, 32 * ib);
                    let p = _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_set_epi64x(g(3), g(2), g(1), g(0)), q), _mm256_maddubs_epi16(nine, q));
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

    /// The 64 crumbs of a Q2_0 block in element order, as two vectors of 32 bytes.
    #[cfg(all(target_feature = "avx512vbmi", target_feature = "avx512vl"))]
    #[inline(always)]
    unsafe fn q2_0_crumbs(b: &[u8]) -> (__m256i, __m256i) { unsafe {
        // Output byte i of qword t takes bits 16 t + 2 i of the source qword: element 8 t + i of 32.
        let ctl = _mm256_set_epi64x(0x3e3c3a3836343230, 0x2e2c2a2826242220, 0x1e1c1a1816141210, 0x0e0c0a0806040200);
        let h = |o: usize| _mm256_and_si256(_mm256_multishift_epi64_epi8(ctl, _mm256_set1_epi64x(i64::from_le_bytes(b[o..o + 8].try_into().unwrap()))), _mm256_set1_epi8(3));
        (h(2), h(10))
    } }
    #[cfg(not(all(target_feature = "avx512vbmi", target_feature = "avx512vl")))]
    #[inline(always)]
    unsafe fn q2_0_crumbs(b: &[u8]) -> (__m256i, __m256i) { unsafe {
        let v = _mm_loadu_si128(b[2..18].as_ptr() as *const __m128i);
        let c = |k: i32| _mm_and_si128(_mm_srl_epi16(v, _mm_cvtsi32_si128(2 * k)), _mm_set1_epi8(3));
        let (c0, c1, c2, c3) = (c(0), c(1), c(2), c(3));
        // Re-interleave the crumbs into element order: byte k holds elements 4k..4k+3.
        let (lo01, lo23) = (_mm_unpacklo_epi8(c0, c1), _mm_unpacklo_epi8(c2, c3));
        let (hi01, hi23) = (_mm_unpackhi_epi8(c0, c1), _mm_unpackhi_epi8(c2, c3));
        (_mm256_set_m128i(_mm_unpackhi_epi16(lo01, lo23), _mm_unpacklo_epi16(lo01, lo23)),
         _mm256_set_m128i(_mm_unpackhi_epi16(hi01, hi23), _mm_unpacklo_epi16(hi01, hi23)))
    } }
    pub fn dot_q2_0(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let (mut acc, mut sumf) = (_mm256_setzero_ps(), 0f32);
            let ones = _mm256_set1_epi16(1);
            for (sb, y) in row.chunks(72).zip(xq) {
                // The sums of y over each 64 values, in lanes 0, 1, 4 and 5.
                let bs = _mm256_madd_epi16(_mm256_loadu_si256(y.bs.as_ptr() as *const __m256i), ones);
                let mut m = [0f32; 8];
                _mm256_storeu_ps(m.as_mut_ptr(), _mm256_cvtepi32_ps(_mm256_hadd_epi32(bs, bs)));
                for (j, b) in sb.chunks_exact(18).enumerate() {
                    let (w0, w1) = q2_0_crumbs(b);
                    let p = _mm256_add_epi16(_mm256_maddubs_epi16(w0, q8(y, 64 * j)), _mm256_maddubs_epi16(w1, q8(y, 64 * j + 32)));
                    let d = d16(b) * y.d;
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_madd_epi16(p, ones)), acc);
                    sumf -= d * m[j + j / 2 * 2];
                }
            }
            hsum(acc) + sumf
        }
    }

    /// Q4_0, Q4_1, Q5_0, Q5_1 (scalar `dot_q4`).
    pub fn dot_q4<const T: u8>(row: &[u8], xq: &[Q8K]) -> f32 {
        let (bb, oh, oq) = super::q4_layout(T);
        unsafe {
            let (m4, ones, mut acc, mut sumf) = (_mm_set1_epi8(0xF), _mm256_set1_epi16(1), _mm256_setzero_ps(), 0f32);
            for (i, b) in row.chunks_exact(bb).enumerate() {
                let (y, o) = (&xq[i / 8], i % 8 * 32);
                let q = _mm_loadu_si128(b[oq..oq + 16].as_ptr() as *const __m128i);
                let mut w = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q, 4), m4), _mm_and_si128(q, m4));
                if oh > 0 {  // bit j of qh as 16 in byte j (llama.cpp bytes_from_bits_32)
                    let qh = i32::from_le_bytes(b[oh..oh + 4].try_into().unwrap());
                    let sh = _mm256_shuffle_epi8(_mm256_set1_epi32(qh), _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0));
                    let set = _mm256_cmpeq_epi8(_mm256_or_si256(sh, _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe)), _mm256_set1_epi64x(-1));
                    w = _mm256_or_si256(w, _mm256_and_si256(set, _mm256_set1_epi8(16)));
                }
                let s = _mm256_madd_epi16(_mm256_maddubs_epi16(w, q8(y, o)), ones);
                acc = _mm256_fmadd_ps(ds(b, y.d), _mm256_cvtepi32_ps(s), acc);
                sumf += super::q4_min(T, b) * y.d * (y.bs[o / 16] as i32 + y.bs[o / 16 + 1] as i32) as f32;
            }
            hsum(acc) + sumf
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

    /// Q8_0 (scalar `dot_q8_0`): |w| times q8 with w's sign, as maddubs needs an unsigned first operand.
    pub fn dot_q8_0(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let (ones, mut acc) = (_mm256_set1_epi16(1), _mm256_setzero_ps());
            for (i, b) in row.chunks_exact(34).enumerate() {
                let (y, o) = (&xq[i / 8], i % 8 * 32);
                let w = _mm256_loadu_si256(b[2..34].as_ptr() as *const __m256i);
                let p = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(q8(y, o), w));
                acc = _mm256_fmadd_ps(ds(b, y.d), _mm256_cvtepi32_ps(_mm256_madd_epi16(p, ones)), acc);
            }
            hsum(acc)
        }
    }

    /// `dot_q8_0` of one row with four activations (see `super::int_dot4`).
    pub fn dot4_q8_0(row: &[u8], xq: [&[Q8K]; 4]) -> [f32; 4] {
        unsafe {
            let (ones, mut acc) = (_mm256_set1_epi16(1), [_mm256_setzero_ps(); 4]);
            for (i, b) in row.chunks_exact(34).enumerate() {
                let w = _mm256_loadu_si256(b[2..34].as_ptr() as *const __m256i);
                let aw = _mm256_sign_epi8(w, w);
                for k in 0..4 {
                    let y = &xq[k][i / 8];
                    let p = _mm256_maddubs_epi16(aw, _mm256_sign_epi8(q8(y, i % 8 * 32), w));
                    acc[k] = _mm256_fmadd_ps(ds(b, y.d), _mm256_cvtepi32_ps(_mm256_madd_epi16(p, ones)), acc[k]);
                }
            }
            acc.map(|a| hsum(a))
        }
    }

    pub fn dot_q5k(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let (m4, one) = (_mm256_set1_epi8(0xF), _mm256_set1_epi8(1));
            // Bit k of every qh byte as 16.
            let hb = |qh: __m256i, k: i32| _mm256_slli_epi16(_mm256_and_si256(_mm256_srl_epi16(qh, _mm_cvtsi32_si128(k)), one), 4);
            let mut acc = _mm256_setzero_ps();
            let mut sumf = 0.0;
            for (b, y) in row.chunks_exact(176).zip(xq) {
                let sm: [(u8, u8); 8] = std::array::from_fn(|j| crate::quant::scale_min_k4(j, &b[4..16]));
                let qh = ld(b, 16);
                let mut sumi = _mm256_setzero_si256();
                for j in 0..4 {
                    let bits = ld(b, 48 + 32 * j);
                    let (s0, s1) = (sm[2 * j].0 as i32, sm[2 * j + 1].0 as i32);
                    let w0 = _mm256_or_si256(_mm256_and_si256(bits, m4), hb(qh, 2 * j as i32));
                    let w1 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(bits, 4), m4), hb(qh, 2 * j as i32 + 1));
                    sumi = _mm256_add_epi32(sumi, udot32(w0, q8(y, 64 * j), s0, s0));
                    sumi = _mm256_add_epi32(sumi, udot32(w1, q8(y, 64 * j + 32), s1, s1));
                }
                let m: i32 = (0..8).map(|j| sm[j].1 as i32 * (y.bs[2 * j] + y.bs[2 * j + 1]) as i32).sum();
                acc = _mm256_fmadd_ps(_mm256_set1_ps(y.d * d16(b)), _mm256_cvtepi32_ps(sumi), acc);
                sumf -= y.d * d16(&b[2..]) * m as f32;
            }
            hsum(acc) + sumf
        }
    }

    /// Q6_K with unsigned 6-bit codes (maddubs needs an unsigned first operand); the -32 offset comes off through
    /// the Q8_K group sums.
    pub fn dot_q6k(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let m4 = _mm256_set1_epi8(0xF);
            let mut acc = _mm256_setzero_ps();
            let mut sumf = 0.0;
            for (b, y) in row.chunks_exact(210).zip(xq) {
                let sc = |g: usize| b[192 + g] as i8 as i32;
                let mut sumi = _mm256_setzero_si256();
                for n in 0..2 {
                    let (la, lb, qh) = (ld(b, 64 * n), ld(b, 64 * n + 32), ld(b, 128 + 32 * n));
                    let hi = |k: usize| _mm256_slli_epi16(crumb(qh, k), 4);
                    let w = [
                        _mm256_or_si256(_mm256_and_si256(la, m4), hi(0)),
                        _mm256_or_si256(_mm256_and_si256(lb, m4), hi(1)),
                        _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(la, 4), m4), hi(2)),
                        _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lb, 4), m4), hi(3)),
                    ];
                    for (k, w) in w.into_iter().enumerate() {
                        let g = 8 * n + 2 * k;
                        sumi = _mm256_add_epi32(sumi, udot32(w, q8(y, 128 * n + 32 * k), sc(g), sc(g + 1)));
                    }
                }
                let m: i32 = (0..16).map(|g| sc(g) * y.bs[g] as i32).sum();
                let d = y.d * d16(&b[208..]);
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
                sumf -= 32.0 * d * m as f32;
            }
            hsum(acc) + sumf
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw", target_feature = "avx512vnni", target_feature = "avx512vbmi"))]
mod avx512 {
    use super::{Q8K, d16};
    use crate::quant::tables::*;
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn q8_512(y: &Q8K, i: usize) -> __m512i { unsafe { _mm512_loadu_si512(y.qs.as_ptr().add(i) as *const __m512i) } }
    #[inline(always)]
    unsafe fn mul3_512(v: __m512i) -> __m512i { unsafe { _mm512_add_epi8(_mm512_add_epi8(v, v), v) } }
    #[inline(always)]
    unsafe fn mul3_256(v: __m256i) -> __m256i { unsafe { _mm256_add_epi8(_mm256_add_epi8(v, v), v) } }
    #[inline(always)]
    unsafe fn mul3_128(v: __m128i) -> __m128i { unsafe { _mm_add_epi8(_mm_add_epi8(v, v), v) } }

    /// Q8_0 (2 blocks = 64 values per 512-bit `vpdpbusd`).
    pub fn dot_q8_0(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let zero = _mm512_setzero_si512();
            let mut acc = _mm512_setzero_ps();
            for (j, b2) in row.chunks_exact(68).enumerate() {
                let (y, o) = (&xq[j / 4], (j % 4) * 64);
                let w0 = _mm256_loadu_si256(b2[2..34].as_ptr() as *const __m256i);
                let w1 = _mm256_loadu_si256(b2[36..68].as_ptr() as *const __m256i);
                let w = _mm512_inserti64x4(_mm512_castsi256_si512(w0), w1, 1);
                let q = q8_512(y, o);
                let neg = _mm512_movepi8_mask(w);
                let p32 = _mm512_dpbusd_epi32(zero, _mm512_abs_epi8(w), _mm512_mask_sub_epi8(q, neg, zero, q));
                let dv = _mm512_mask_blend_ps(0xFF00, _mm512_set1_ps(d16(b2) * y.d), _mm512_set1_ps(d16(&b2[34..]) * y.d));
                acc = _mm512_fmadd_ps(dv, _mm512_cvtepi32_ps(p32), acc);
            }
            let mut sum = _mm512_reduce_add_ps(acc);
            let n = row.len() / 34;
            if n % 2 == 1 {
                let i = n - 1;
                let b = &row[34 * i..34 * i + 34];
                let (y, o) = (&xq[i / 8], (i % 8) * 32);
                let s: i32 = b[2..34].iter().zip(&y.qs[o..o + 32]).map(|(&w, &q)| w as i8 as i32 * q as i32).sum();
                sum += d16(b) * y.d * s as f32;
            }
            sum
        }
    }

    /// Q4_0, Q4_1, Q5_0, Q5_1: two 32-value blocks per 512-bit `vpdpbusd` + 64-bit mask high-bit expansion.
    pub fn dot_q4<const T: u8>(row: &[u8], xq: &[Q8K]) -> f32 {
        let (bb, oh, oq) = super::q4_layout(T);
        unsafe {
            let (m4, sixteen, zero) = (_mm_set1_epi8(0xF), 16i8, _mm512_setzero_si512());
            let mut acc = _mm512_setzero_ps();
            let mut sumf = 0.0f32;
            for (j, b2) in row.chunks_exact(2 * bb).enumerate() {
                let (b0, b1) = (&b2[..bb], &b2[bb..]);
                let (y, o) = (&xq[j / 4], (j % 4) * 64);
                let q0 = _mm_loadu_si128(b0[oq..oq + 16].as_ptr() as *const __m128i);
                let q1 = _mm_loadu_si128(b1[oq..oq + 16].as_ptr() as *const __m128i);
                let w0 = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q0, 4), m4), _mm_and_si128(q0, m4));
                let w1 = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q1, 4), m4), _mm_and_si128(q1, m4));
                let mut w = _mm512_inserti64x4(_mm512_castsi256_si512(w0), w1, 1);
                if oh > 0 {
                    let qh0 = u32::from_le_bytes(b0[oh..oh + 4].try_into().unwrap()) as u64;
                    let qh1 = u32::from_le_bytes(b1[oh..oh + 4].try_into().unwrap()) as u64;
                    w = _mm512_or_si512(w, _mm512_maskz_set1_epi8(qh0 | (qh1 << 32), sixteen));
                }
                let p32 = _mm512_dpbusd_epi32(zero, w, q8_512(y, o));
                let dv = _mm512_mask_blend_ps(0xFF00, _mm512_set1_ps(d16(b0) * y.d), _mm512_set1_ps(d16(b1) * y.d));
                acc = _mm512_fmadd_ps(dv, _mm512_cvtepi32_ps(p32), acc);
                let bs0 = (y.bs[o / 16] as i32 + y.bs[o / 16 + 1] as i32) as f32;
                let bs1 = (y.bs[o / 16 + 2] as i32 + y.bs[o / 16 + 3] as i32) as f32;
                sumf += y.d * (super::q4_min(T, b0) * bs0 + super::q4_min(T, b1) * bs1);
            }
            let mut sum = _mm512_reduce_add_ps(acc) + sumf;
            let n = row.len() / bb;
            if n % 2 == 1 {
                let i = n - 1;
                let b = &row[bb * i..bb * i + bb];
                let (y, o) = (&xq[i / 8], (i % 8) * 32);
                let mut s = 0i32;
                let qh = if oh > 0 { u32::from_le_bytes(b[oh..oh + 4].try_into().unwrap()) } else { 0 };
                for l in 0..16 {
                    s += ((b[oq + l] & 0xF) as i32 | ((qh as i32 >> l) & 1) << 4) * y.qs[o + l] as i32;
                    s += ((b[oq + l] >> 4) as i32 | ((qh as i32 >> (l + 16)) & 1) << 4) * y.qs[o + 16 + l] as i32;
                }
                sum += d16(b) * y.d * s as f32 + super::q4_min(T, b) * y.d * (y.bs[o / 16] as i32 + y.bs[o / 16 + 1] as i32) as f32;
            }
            sum
        }
    }

    /// IQ4_NL: two 32-value blocks via broadcasted 512-bit `vpshufb` + `vpdpbusd`.
    pub fn dot_iq4nl(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let values = _mm512_broadcast_i32x4(_mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i));
            let (m4, zero) = (_mm_set1_epi8(0xF), _mm512_setzero_si512());
            let mut acc = _mm512_setzero_ps();
            for (j, b2) in row.chunks_exact(36).enumerate() {
                let (b0, b1) = (&b2[..18], &b2[18..]);
                let (y, o) = (&xq[j / 4], (j % 4) * 64);
                let bits0 = _mm_loadu_si128(b0[2..].as_ptr() as *const __m128i);
                let bits1 = _mm_loadu_si128(b1[2..].as_ptr() as *const __m128i);
                let idx0 = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(bits0, 4), m4), _mm_and_si128(bits0, m4));
                let idx1 = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(bits1, 4), m4), _mm_and_si128(bits1, m4));
                let w = _mm512_shuffle_epi8(values, _mm512_inserti64x4(_mm512_castsi256_si512(idx0), idx1, 1));
                let q = q8_512(y, o);
                let neg = _mm512_movepi8_mask(w);
                let p32 = _mm512_dpbusd_epi32(zero, _mm512_abs_epi8(w), _mm512_mask_sub_epi8(q, neg, zero, q));
                let dv = _mm512_mask_blend_ps(0xFF00, _mm512_set1_ps(d16(b0) * y.d), _mm512_set1_ps(d16(b1) * y.d));
                acc = _mm512_fmadd_ps(dv, _mm512_cvtepi32_ps(p32), acc);
            }
            let mut sum = _mm512_reduce_add_ps(acc);
            let n = row.len() / 18;
            if n % 2 == 1 {
                let i = n - 1;
                let b = &row[18 * i..18 * i + 18];
                let (y, o) = (&xq[i / 8], (i % 8) * 32);
                let mut s = 0i32;
                for l in 0..16 {
                    s += KVALUES_IQ4NL[(b[2 + l] & 0xF) as usize] as i32 * y.qs[o + l] as i32;
                    s += KVALUES_IQ4NL[(b[2 + l] >> 4) as usize] as i32 * y.qs[o + 16 + l] as i32;
                }
                sum += d16(b) * y.d * s as f32;
            }
            sum
        }
    }

    /// IQ4_XS: 4 pairs of 32-value sub-blocks per super-block via 512-bit `vpshufb` + `vpdpbusd`.
    pub fn dot_iq4xs(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let values = _mm512_broadcast_i32x4(_mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i));
            let (m4, zero) = (_mm_set1_epi8(0xF), _mm512_setzero_si512());
            let mut sumf = 0.0f32;
            for (b, y) in row.chunks_exact(136).zip(xq) {
                let sh = u16::from_le_bytes([b[2], b[3]]);
                let mut sumi = zero;
                for p in 0..4 {
                    let ib0 = 2 * p;
                    let ib1 = ib0 + 1;
                    let bits0 = _mm_loadu_si128(b.as_ptr().add(8 + 16 * ib0) as *const __m128i);
                    let bits1 = _mm_loadu_si128(b.as_ptr().add(8 + 16 * ib1) as *const __m128i);
                    let idx0 = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(bits0, 4), m4), _mm_and_si128(bits0, m4));
                    let idx1 = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(bits1, 4), m4), _mm_and_si128(bits1, m4));
                    let w = _mm512_shuffle_epi8(values, _mm512_inserti64x4(_mm512_castsi256_si512(idx0), idx1, 1));
                    let q = q8_512(y, 64 * p);
                    let neg = _mm512_movepi8_mask(w);
                    let d32 = _mm512_dpbusd_epi32(zero, _mm512_abs_epi8(w), _mm512_mask_sub_epi8(q, neg, zero, q));
                    let sc = |ib: usize| (((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) as i32 | ((((sh >> (2 * ib)) & 3) as i32) << 4)) - 32;
                    let sv = _mm512_mask_set1_epi32(_mm512_set1_epi32(sc(ib0)), 0xFF00, sc(ib1));
                    sumi = _mm512_add_epi32(sumi, _mm512_mullo_epi32(d32, sv));
                }
                sumf += d16(b) * y.d * _mm512_reduce_add_epi32(sumi) as f32;
            }
            sumf
        }
    }

    /// Q4_K: 4 x 64 weights per super-block using 512-bit `vpdpbusd`.
    pub fn dot_q4k(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let (m4, zero) = (_mm512_set1_epi8(0xF), _mm512_setzero_si512());
            let mut sumf = 0.0f32;
            for (b, y) in row.chunks_exact(144).zip(xq) {
                let sm: [(u8, u8); 8] = std::array::from_fn(|j| crate::quant::scale_min_k4(j, &b[4..16]));
                let mut sumi = zero;
                for j in 0..4 {
                    let bits256 = _mm256_loadu_si256(b[16 + 32 * j..].as_ptr() as *const __m256i);
                    let bits = _mm512_inserti64x4(_mm512_castsi256_si512(bits256), bits256, 1);
                    let hi = _mm512_srli_epi16(bits, 4);
                    let w = _mm512_and_si512(_mm512_mask_blend_epi32(0xFF00, bits, hi), m4);
                    let d32 = _mm512_dpbusd_epi32(zero, w, q8_512(y, 64 * j));
                    let sv = _mm512_mask_set1_epi32(_mm512_set1_epi32(sm[2 * j].0 as i32), 0xFF00, sm[2 * j + 1].0 as i32);
                    sumi = _mm512_add_epi32(sumi, _mm512_mullo_epi32(d32, sv));
                }
                let m: i32 = (0..8).map(|j| sm[j].1 as i32 * (y.bs[2 * j] + y.bs[2 * j + 1]) as i32).sum();
                sumf += y.d * d16(b) * _mm512_reduce_add_epi32(sumi) as f32 - y.d * d16(&b[2..]) * m as f32;
            }
            sumf
        }
    }

    /// Q6_K: 4 x 64 weights per super-block using 512-bit `vpdpbusd`.
    pub fn dot_q6k(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let (m4, m2, zero) = (_mm512_set1_epi8(0xF), _mm512_set1_epi8(3), _mm512_setzero_si512());
            let mut sumf = 0.0f32;
            for (b, y) in row.chunks_exact(210).zip(xq) {
                let sc = |g: usize| b[192 + g] as i8 as i32;
                let mut sumi = zero;
                for n in 0..2 {
                    let ql = _mm512_loadu_si512(b[64 * n..].as_ptr() as *const __m512i);
                    let qh256 = _mm256_loadu_si256(b[128 + 32 * n..].as_ptr() as *const __m256i);
                    let qh = _mm512_inserti64x4(_mm512_castsi256_si512(qh256), qh256, 1);
                    let lo_h = _mm512_and_si512(ql, m4);
                    let hi_h = _mm512_and_si512(_mm512_srli_epi16(ql, 4), m4);
                    let h0 = _mm512_slli_epi16(_mm512_and_si512(_mm512_mask_blend_epi32(0xFF00, qh, _mm512_srli_epi16(qh, 2)), m2), 4);
                    let h1 = _mm512_slli_epi16(_mm512_and_si512(_mm512_mask_blend_epi32(0xFF00, _mm512_srli_epi16(qh, 4), _mm512_srli_epi16(qh, 6)), m2), 4);
                    let g = 8 * n;
                    let s0 = _mm512_set_epi32(sc(g + 3), sc(g + 3), sc(g + 3), sc(g + 3), sc(g + 2), sc(g + 2), sc(g + 2), sc(g + 2),
                                              sc(g + 1), sc(g + 1), sc(g + 1), sc(g + 1), sc(g), sc(g), sc(g), sc(g));
                    let s1 = _mm512_set_epi32(sc(g + 7), sc(g + 7), sc(g + 7), sc(g + 7), sc(g + 6), sc(g + 6), sc(g + 6), sc(g + 6),
                                              sc(g + 5), sc(g + 5), sc(g + 5), sc(g + 5), sc(g + 4), sc(g + 4), sc(g + 4), sc(g + 4));
                    sumi = _mm512_add_epi32(sumi, _mm512_mullo_epi32(_mm512_dpbusd_epi32(zero, _mm512_or_si512(lo_h, h0), q8_512(y, 128 * n)), s0));
                    sumi = _mm512_add_epi32(sumi, _mm512_mullo_epi32(_mm512_dpbusd_epi32(zero, _mm512_or_si512(hi_h, h1), q8_512(y, 128 * n + 64)), s1));
                }
                let m: i32 = (0..16).map(|g| sc(g) * y.bs[g] as i32).sum();
                let d = y.d * d16(&b[208..]);
                sumf += d * (_mm512_reduce_add_epi32(sumi) as f32 - 32.0 * m as f32);
            }
            sumf
        }
    }

    /// TQ1_0: all 256 trits per block evaluated in 4 512-bit `vpdpbusd` instructions.
    pub fn dot_tq1_0(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let th85 = _mm512_set1_epi8(85u8 as i8);
            let th170 = _mm512_set1_epi8(170u8 as i8);
            let (one8, zero) = (_mm512_set1_epi8(1), _mm512_setzero_si512());
            let trits_u = |q: __m512i| -> __m512i {
                let m1 = _mm512_cmpgt_epu8_mask(q, th85);
                let m2 = _mm512_cmpgt_epu8_mask(q, th170);
                let u = _mm512_maskz_set1_epi8(m1, 1);
                _mm512_mask_add_epi8(u, m2, u, one8)
            };
            let mut sumf = 0.0f32;
            for (b, y) in row.chunks_exact(54).zip(xq) {
                let b0 = _mm256_loadu_si256(b.as_ptr() as *const __m256i);
                let raw0 = _mm512_inserti64x4(_mm512_castsi256_si512(b0), mul3_256(b0), 1);
                let raw1 = mul3_512(mul3_512(raw0));
                let b0_4 = mul3_256(_mm512_extracti64x4_epi64::<1>(raw1));
                let b1_0 = _mm_loadu_si128(b.as_ptr().add(32) as *const __m128i);
                let b1_01 = _mm256_set_m128i(mul3_128(b1_0), b1_0);
                let raw2 = _mm512_inserti64x4(_mm512_castsi256_si512(b0_4), b1_01, 1);
                let b1_23 = mul3_256(mul3_256(b1_01));
                let b1_4 = mul3_128(_mm256_extracti128_si256::<1>(b1_23));
                let bm = |m: u8| u32::from_le_bytes([b[48].wrapping_mul(m), b[49].wrapping_mul(m), b[50].wrapping_mul(m), b[51].wrapping_mul(m)]) as i32;
                let qh = _mm_set_epi32(bm(27), bm(9), bm(3), bm(1));
                let b14_qh = _mm256_set_m128i(qh, b1_4);
                let raw3 = _mm512_inserti64x4(_mm512_castsi256_si512(b1_23), b14_qh, 1);

                let mut s32 = zero;
                s32 = _mm512_dpbusd_epi32(s32, trits_u(raw0), q8_512(y, 0));
                s32 = _mm512_dpbusd_epi32(s32, trits_u(raw1), q8_512(y, 64));
                s32 = _mm512_dpbusd_epi32(s32, trits_u(raw2), q8_512(y, 128));
                s32 = _mm512_dpbusd_epi32(s32, trits_u(raw3), q8_512(y, 192));
                let bs: i32 = y.bs.iter().map(|&v| v as i32).sum();
                sumf += d16(&b[52..]) * y.d * (_mm512_reduce_add_epi32(s32) - bs) as f32;
            }
            sumf
        }
    }

    /// IQ1_S: two 32-value blocks per 512-bit `vpdpbusd` + delta bs term.
    pub fn dot_iq1s(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let zero = _mm512_setzero_si512();
            let (m7, ones, lo1) = (_mm_set1_epi16(7), _mm_set1_epi16(1), _mm_set1_epi16(1));
            let mut sumf = 0.0f32;
            for (b, y) in row.chunks_exact(50).zip(xq) {
                let mut sumi = zero;
                for p in 0..4 {
                    let (ib0, ib1) = (2 * p, 2 * p + 1);
                    let qh0 = u16::from_le_bytes([b[34 + 2 * ib0], b[35 + 2 * ib0]]) as usize;
                    let qh1 = u16::from_le_bytes([b[34 + 2 * ib1], b[35 + 2 * ib1]]) as usize;
                    let g64 = |ib: usize, qh: usize, l: usize| IQ1S_GRID[b[2 + 4 * ib + l] as usize | ((qh >> (3 * l)) & 7) << 8] as i64;
                    let g0 = _mm256_set_epi64x(g64(ib0, qh0, 3), g64(ib0, qh0, 2), g64(ib0, qh0, 1), g64(ib0, qh0, 0));
                    let g1 = _mm256_set_epi64x(g64(ib1, qh1, 3), g64(ib1, qh1, 2), g64(ib1, qh1, 1), g64(ib1, qh1, 0));
                    let g = _mm512_inserti64x4(_mm512_castsi256_si512(g0), g1, 1);
                    let q = q8_512(y, 64 * p);
                    let neg = _mm512_movepi8_mask(g);
                    let p32 = _mm512_dpbusd_epi32(zero, _mm512_abs_epi8(g), _mm512_mask_sub_epi8(q, neg, zero, q));
                    let ls0 = 2 * ((qh0 >> 12) & 7) as i32 + 1;
                    let ls1 = 2 * ((qh1 >> 12) & 7) as i32 + 1;
                    let sv = _mm512_mask_set1_epi32(_mm512_set1_epi32(ls0), 0xFF00, ls1);
                    sumi = _mm512_add_epi32(sumi, _mm512_mullo_epi32(p32, sv));
                }
                let qh = _mm_loadu_si128(b[34..50].as_ptr() as *const __m128i);
                let ls = _mm_or_si128(_mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh, 12), m7), 1), ones);
                let ls = _mm_sign_epi16(ls, _mm_or_si128(qh, lo1));
                let bs = |i: usize| _mm_loadu_si128(y.bs[8 * i..].as_ptr() as *const __m128i);
                let s = _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(ls, ls), bs(0)), _mm_madd_epi16(_mm_unpackhi_epi16(ls, ls), bs(1)));
                let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
                let s1 = _mm_cvtsi128_si32(_mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01)));
                let dy = d16(b) * y.d;
                sumf += dy * _mm512_reduce_add_epi32(sumi) as f32 + dy * 0.125 * s1 as f32;
            }
            sumf
        }
    }

    /// IQ2_XXS: two 32-value blocks per 512-bit `vpdpbusd` using 64-bit sign masks from `KSIGNS_IQ2XS`.
    pub fn dot_iq2xxs(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let zero = _mm512_setzero_si512();
            let ks = |aux: u32| -> u64 {
                (KSIGNS_IQ2XS[(aux & 127) as usize] as u64)
                    | ((KSIGNS_IQ2XS[((aux >> 7) & 127) as usize] as u64) << 8)
                    | ((KSIGNS_IQ2XS[((aux >> 14) & 127) as usize] as u64) << 16)
                    | ((KSIGNS_IQ2XS[((aux >> 21) & 127) as usize] as u64) << 24)
            };
            let mut sumf = 0.0f32;
            for (b, y) in row.chunks_exact(66).zip(xq) {
                let mut sumi = zero;
                for p in 0..4 {
                    let (ib0, ib1) = (2 * p, 2 * p + 1);
                    let g = |ib: usize, l: usize| IQ2XXS_GRID[b[2 + 8 * ib + l] as usize] as i64;
                    let g0 = _mm256_set_epi64x(g(ib0, 3), g(ib0, 2), g(ib0, 1), g(ib0, 0));
                    let g1 = _mm256_set_epi64x(g(ib1, 3), g(ib1, 2), g(ib1, 1), g(ib1, 0));
                    let grid = _mm512_inserti64x4(_mm512_castsi256_si512(g0), g1, 1);
                    let aux0 = u32::from_le_bytes(b[6 + 8 * ib0..10 + 8 * ib0].try_into().unwrap());
                    let aux1 = u32::from_le_bytes(b[6 + 8 * ib1..10 + 8 * ib1].try_into().unwrap());
                    let neg = ks(aux0) | (ks(aux1) << 32);
                    let q = q8_512(y, 64 * p);
                    let p32 = _mm512_dpbusd_epi32(zero, grid, _mm512_mask_sub_epi8(q, neg, zero, q));
                    let (ls0, ls1) = (2 * (aux0 >> 28) as i32 + 1, 2 * (aux1 >> 28) as i32 + 1);
                    let sv = _mm512_mask_set1_epi32(_mm512_set1_epi32(ls0), 0xFF00, ls1);
                    sumi = _mm512_add_epi32(sumi, _mm512_mullo_epi32(p32, sv));
                }
                sumf += 0.125 * d16(b) * y.d * _mm512_reduce_add_epi32(sumi) as f32;
            }
            sumf
        }
    }

    /// IQ3_S: two 32-value blocks per 512-bit `vpdpbusd` with zero-cost 64-bit sign mask.
    pub fn dot_iq3s(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let zero = _mm512_setzero_si512();
            let shl = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
            let h256 = _mm256_set1_epi32(256);
            let mut sumf = 0.0f32;
            for (b, y) in row.chunks_exact(110).zip(xq) {
                let (qs, qh) = (&b[2..66], &b[66..74]);
                let g256 = |ib: usize| -> __m256i {
                    let lo = _mm256_cvtepu8_epi32(_mm_loadl_epi64(qs[8 * ib..].as_ptr() as *const __m128i));
                    let hi = _mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32(qh[ib] as i32), shl), h256);
                    _mm256_i32gather_epi32(IQ3S_GRID.as_ptr() as *const i32, _mm256_or_si256(lo, hi), 4)
                };
                let mut sumi = zero;
                for p in 0..4 {
                    let (ib0, ib1) = (2 * p, 2 * p + 1);
                    let grid = _mm512_inserti64x4(_mm512_castsi256_si512(g256(ib0)), g256(ib1), 1);
                    let sb0 = u32::from_le_bytes(b[74 + 4 * ib0..78 + 4 * ib0].try_into().unwrap()) as u64;
                    let sb1 = u32::from_le_bytes(b[74 + 4 * ib1..78 + 4 * ib1].try_into().unwrap()) as u64;
                    let neg = sb0 | (sb1 << 32);
                    let q = q8_512(y, 64 * p);
                    let p32 = _mm512_dpbusd_epi32(zero, grid, _mm512_mask_sub_epi8(q, neg, zero, q));
                    let ls0 = 2 * ((b[106 + ib0 / 2] >> (4 * (ib0 % 2))) & 0xF) as i32 + 1;
                    let ls1 = 2 * ((b[106 + ib1 / 2] >> (4 * (ib1 % 2))) & 0xF) as i32 + 1;
                    let sv = _mm512_mask_set1_epi32(_mm512_set1_epi32(ls0), 0xFF00, ls1);
                    sumi = _mm512_add_epi32(sumi, _mm512_mullo_epi32(p32, sv));
                }
                sumf += d16(b) * y.d * _mm512_reduce_add_epi32(sumi) as f32;
            }
            sumf
        }
    }
}

/// A trellis row (blocks of B bytes for 128 values, `quant::tq_points`) times x, in f32: the codebook is not on an
/// int8 grid, so these types have no Q8_K kernel. Eight partial sums per block, where `Tensor::matvec` adds in order.
fn dot_tq<const B: usize>(row: &[u8], x: &[f32]) -> f32 {
    use crate::quant::{TQ_LUT as L, tq_points};
    let mut sum = 0f32;
    for (b, xb) in row.chunks_exact(B).zip(x.chunks_exact(128)) {
        let (qs, mut acc) = (&b[2..B], [0f32; 8]);
        for (t, xs) in xb.chunks_exact(8).enumerate() {
            let ((p0, p1), (p2, p3)) = (tq_points(qs, 2 * t), tq_points(qs, 2 * t + 1));
            let w = [L[p0][0], L[p0][1], L[p1][0], L[p1][1], L[p2][0], L[p2][1], L[p3][0], L[p3][1]];
            for i in 0..8 { acc[i] += w[i] * xs[i]; }
        }
        sum += d16(b) * acc.iter().sum::<f32>();
    }
    sum
}

/// Signed OCP FP8 E4M3 row times x via the 256-entry `F8_E4M3_LUT` (no intermediate dequant buffer).
fn dot_f8_e4m3(row: &[u8], x: &[f32]) -> f32 {
    let lut = &crate::quant::F8_E4M3_LUT;
    let mut acc = [0.0f32; 8];
    let mut r = row.chunks_exact(8);
    let mut xv = x.chunks_exact(8);
    for (rb, xb) in r.by_ref().zip(xv.by_ref()) {
        for i in 0..8 { acc[i] += lut[rb[i] as usize] * xb[i]; }
    }
    let mut sum: f32 = acc.iter().sum();
    for (&b, &v) in r.remainder().iter().zip(xv.remainder()) {
        sum += lut[b as usize] * v;
    }
    sum
}

/// The f32 kernel of a type that has no integer one but is too slow through `Tensor::row`.
pub fn float_dot(ty: GgmlType) -> Option<fn(&[u8], &[f32]) -> f32> {
    match ty {
        GgmlType::F8E4M3 => Some(dot_f8_e4m3),
        GgmlType::TQ2T => Some(dot_tq::<34>),
        GgmlType::TQK6 => Some(dot_tq::<26>),
        GgmlType::TQK7 => Some(dot_tq::<30>),
        _ => None,
    }
}

pub fn int_dot(ty: GgmlType) -> Option<fn(&[u8], &[Q8K]) -> f32> {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw", target_feature = "avx512vnni", target_feature = "avx512vbmi"))]
    // Opt-in: these kernels add the block sums in a different order than the AVX2 ones, so their results differ in the
    // last bits, and that is enough to make a drafted run and a plain greedy run pick different tokens at a near tie.
    if std::env::var_os("STRATA_SCALAR").is_none() && std::env::var_os("STRATA_AVX512").is_some() {
        match ty {
            GgmlType::Q8_0 => return Some(avx512::dot_q8_0),
            GgmlType::Q4_0 => return Some(avx512::dot_q4::<2>),
            GgmlType::Q4_1 => return Some(avx512::dot_q4::<3>),
            GgmlType::Q5_0 => return Some(avx512::dot_q4::<6>),
            GgmlType::Q5_1 => return Some(avx512::dot_q4::<7>),
            GgmlType::IQ4NL => return Some(avx512::dot_iq4nl),
            GgmlType::IQ4XS => return Some(avx512::dot_iq4xs),
            GgmlType::Q4K => return Some(avx512::dot_q4k),
            GgmlType::Q6K => return Some(avx512::dot_q6k),
            GgmlType::TQ1_0 => return Some(avx512::dot_tq1_0),
            GgmlType::IQ1S => return Some(avx512::dot_iq1s),
            GgmlType::IQ2XXS => return Some(avx512::dot_iq2xxs),
            GgmlType::IQ3S => return Some(avx512::dot_iq3s),
            _ => {}
        }
    }
    #[cfg(target_feature = "avx2")]
    if std::env::var_os("STRATA_SCALAR").is_none() {
        return match ty { GgmlType::IQ2XS => Some(avx2::dot_iq2xs), GgmlType::IQ3XXS => Some(avx2::dot_iq3xxs), GgmlType::IQ4XS => Some(avx2::dot_iq4xs), GgmlType::IQ4NL => Some(avx2::dot_iq4nl), GgmlType::IQ3S => Some(avx2::dot_iq3s), GgmlType::Q6K => Some(avx2::dot_q6k), GgmlType::Q5K => Some(avx2::dot_q5k), GgmlType::Q8_0 => Some(avx2::dot_q8_0), GgmlType::IQ2XXS => Some(avx2::dot_iq2xxs), GgmlType::IQ2S => Some(avx2::dot_iq2s), GgmlType::IQ1S => Some(avx2::dot_iq1s), GgmlType::IQ1M => Some(avx2::dot_iq1m), GgmlType::Q2K => Some(avx2::dot_q2k), GgmlType::Q3K => Some(avx2::dot_q3k), GgmlType::Q4K => Some(avx2::dot_q4k), GgmlType::Q2_0 => Some(avx2::dot_q2_0), GgmlType::TQ1_0 => Some(avx2::dot_tq1_0), GgmlType::TQ2_0 => Some(dot_tq2_0), GgmlType::NVFP4 => Some(avx2::dot_nvfp4), GgmlType::Q4_0 => Some(avx2::dot_q4::<2>), GgmlType::Q4_1 => Some(avx2::dot_q4::<3>), GgmlType::Q5_0 => Some(avx2::dot_q4::<6>), GgmlType::Q5_1 => Some(avx2::dot_q4::<7>), _ => None };
    }
    match ty { GgmlType::IQ2XS => Some(dot_iq2xs), GgmlType::IQ3XXS => Some(dot_iq3xxs), GgmlType::IQ4XS => Some(dot_iq4xs), GgmlType::IQ4NL => Some(dot_iq4nl), GgmlType::IQ3S => Some(dot_iq3s), GgmlType::Q6K => Some(dot_q6k), GgmlType::Q5K => Some(dot_q5k), GgmlType::Q8_0 => Some(dot_q8_0), GgmlType::IQ2XXS => Some(dot_iq2xxs), GgmlType::IQ2S => Some(dot_iq2s), GgmlType::IQ1S => Some(dot_iq1s), GgmlType::IQ1M => Some(dot_iq1m), GgmlType::Q2K => Some(dot_q2k), GgmlType::Q3K => Some(dot_q3k), GgmlType::Q4K => Some(dot_q4k), GgmlType::Q2_0 => Some(dot_q2_0), GgmlType::TQ1_0 => Some(dot_tq1_0), GgmlType::TQ2_0 => Some(dot_tq2_0), GgmlType::NVFP4 => Some(dot_nvfp4), GgmlType::Q4_0 => Some(dot_q4::<2>), GgmlType::Q4_1 => Some(dot_q4::<3>), GgmlType::Q5_0 => Some(dot_q4::<6>), GgmlType::Q5_1 => Some(dot_q4::<7>), _ => None }
}

/// Four dots of one row at once, with the bits of `int_dot`'s kernel: the row is read and decoded once, and the
/// four sums do not wait for each other. None where no such kernel exists; the caller then uses `int_dot`.
pub fn int_dot4(ty: GgmlType) -> Option<fn(&[u8], [&[Q8K]; 4]) -> [f32; 4]> {
    #[cfg(target_feature = "avx2")]
    if std::env::var_os("STRATA_SCALAR").is_none() && std::env::var_os("STRATA_AVX512").is_none() {
        return match ty { GgmlType::Q8_0 => Some(avx2::dot4_q8_0), GgmlType::IQ4NL => Some(avx2::dot4_iq4nl), _ => None };
    }
    let _ = ty;
    None
}

/// y = t · x for every row, using the integer kernel when the type has one.
pub fn matvec(t: &Tensor, x: &[f32], y: &mut [f32]) {
    let rb = t.data.len() / t.rows;
    if let Some(dot) = float_dot(t.ty) {
        return y.par_chunks_mut(32).enumerate().for_each(|(c, ys)| {
            for (k, v) in ys.iter_mut().enumerate() { let r = c * 32 + k; *v = dot(&t.data[r * rb..(r + 1) * rb], x) * t.scale; }
        });
    }
    let Some(dot) = int_dot(t.ty) else { return t.matvec(x, y) };
    let xq = quantize_q8k(x);
    y.par_chunks_mut(32).enumerate().for_each(|(c, ys)| {
        for (k, v) in ys.iter_mut().enumerate() {
            let r = c * 32 + k;
            *v = dot(&t.data[r * rb..(r + 1) * rb], &xq) * t.scale;
        }
    });
}

/// Rows `out.len()` of a batch of matrices: out[i] = mats[i / rows] row (i % rows) · x of that matrix, where
/// matrix j uses xs[j * xs.len() / mats.len()] (one x for all, or one each), so a shared x is quantized once.
fn batch_rows(mats: &[Tensor], xs: &[&[f32]], out: &mut [f32]) {
    let rows = out.len() / mats.len();
    let (dot, fdot) = (int_dot(mats[0].ty), float_dot(mats[0].ty));
    let xq: Vec<Vec<Q8K>> = if dot.is_some() { xs.iter().map(|x| quantize_q8k(x)).collect() } else { vec![] };
    // Q2_0 experts benefit from fewer Rayon tasks on AVX2. Keep enough chunks to occupy the pool,
    // including when only one expert misses; other formats keep their existing granularity.
    let batch = if mats[0].ty == GgmlType::Q2_0 && cfg!(target_feature = "avx2")
        && out.len() / 64 >= rayon::current_num_threads() { 64 } else { 16 };
    out.par_chunks_mut(batch).enumerate().for_each_init(|| if dot.is_none() { vec![0f32; mats[0].cols] } else { Vec::new() }, |buf, (c, ys)| {
        for (k, v) in ys.iter_mut().enumerate() {
            let i = c * batch + k;
            let (t, r, x) = (&mats[i / rows], i % rows, i / rows * xs.len() / mats.len());
            let rb = t.data.len() / t.rows;
            *v = match (dot, fdot) {
                (Some(d), _) => d(&t.data[r * rb..(r + 1) * rb], &xq[x]) * t.scale,
                (_, Some(d)) => d(&t.data[r * rb..(r + 1) * rb], xs[x]) * t.scale,
                _ => { t.row(r, buf); buf.iter().zip(xs[x]).map(|(a, b)| a * b).sum() }
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
    // Gyro files: the routed experts read rotated inputs (`config::Hadamard`); the LoRA delta below reads `acts`.
    let rot = c.hadamard.as_ref();
    let hr = rot.map(|r| r.rotated(h));
    let h = hr.as_deref().unwrap_or(h);
    if gate.ty == up.ty {
        batch_rows(&gu, &[h], &mut a);
    } else {
        for (i, t) in gu.iter().enumerate() { matvec(t, h, &mut a[i * ff..(i + 1) * ff]); }
    }
    let acts: Vec<Vec<f32>> = a.chunks_exact(2 * ff).map(|g| (0..ff).map(|i| ops::silu(g[i]) * g[ff + i]).collect()).collect();
    let racts: Option<Vec<Vec<f32>>> = rot.map(|r| acts.iter().map(|a| r.rotated(a)).collect());
    let ds: Vec<Tensor> = sel.iter().map(|&(e, _)| down.expert(e, n)).collect();
    let mut o = vec![0f32; k * c.n_embd];
    batch_rows(&ds, &racts.as_ref().unwrap_or(&acts).iter().map(|v| v.as_slice()).collect::<Vec<_>>(), &mut o);
    for (j, &(e, _)) in sel.iter().enumerate() {
        m.lora_delta(&format!("blk.{il}.ffn_down_exps.weight"), &acts[j], &mut o[j * c.n_embd..(j + 1) * c.n_embd], Some(e));
    }
    for (oj, &(_, w)) in o.chunks_exact(c.n_embd).zip(sel) {
        for (yy, oo) in y.iter_mut().zip(oj) { *yy += w * oo; }
    }
    y
}

/// `experts` for several tokens of one layer (a verify batch): ys[t] = `experts(m, il, &hs[t], &sels[t])`, bit for bit.
/// All tokens share the two parallel passes of `experts` (gate and up, then down), so a batch costs no more passes
/// than one token: a pass per expert costs more in thread wake-ups than the experts two tokens share save.
pub fn experts_multi(m: &Model, il: usize, hs: &[Vec<f32>], sels: &[Vec<(usize, f32)>]) -> Vec<Vec<f32>> {
    let c = &m.cfg;
    let mut ys = vec![vec![0f32; c.n_embd]; hs.len()];
    // (token, expert, weight) of every routed expert, in token and router order.
    let pairs: Vec<(usize, usize, f32)> = sels.iter().enumerate().flat_map(|(t, s)| s.iter().map(move |&(e, w)| (t, e, w))).collect();
    let o = expert_rows(m, il, hs, &pairs);
    for (oj, &(t, _, w)) in o.chunks_exact(c.n_embd).zip(&pairs) {
        for (yy, oo) in ys[t].iter_mut().zip(oj.iter()) { *yy += w * oo; }
    }
    ys
}

/// The output of each (token, expert, _) pair of layer `il`, unweighted, as rows of n_embd in pair order: the experts
/// of every pair in the two parallel passes of `experts_multi`.
pub fn expert_rows(m: &Model, il: usize, hs: &[Vec<f32>], pairs: &[(usize, usize, f32)]) -> Vec<f32> {
    let c = &m.cfg;
    if pairs.is_empty() { return vec![]; }
    let t = |s: &str| m.w.layer(il, s).unwrap_or_else(|e| panic!("{e:#}"));
    let (gate, up, down) = (t("ffn_gate_exps.weight"), t("ffn_up_exps.weight"), t("ffn_down_exps.weight"));
    let (n, ff, k) = (c.n_expert, c.n_ff_exp, pairs.len());
    let gu: Vec<Tensor> = pairs.iter().flat_map(|&(_, e, _)| [gate.expert(e, n), up.expert(e, n)]).collect();
    // Gyro files: the routed experts read rotated inputs (`config::Hadamard`); the LoRA delta below reads `acts`.
    let rot = c.hadamard.as_ref();
    let hr: Option<Vec<Vec<f32>>> = rot.map(|r| hs.iter().map(|h| r.rotated(h)).collect());
    let hs = hr.as_deref().unwrap_or(hs);
    let xs: Vec<&[f32]> = pairs.iter().map(|&(t, ..)| hs[t].as_slice()).collect();
    let mut a = vec![0f32; 2 * k * ff];
    if gate.ty == up.ty {
        batch_rows(&gu, &xs, &mut a);
    } else {
        for (i, t) in gu.iter().enumerate() { matvec(t, xs[i / 2], &mut a[i * ff..(i + 1) * ff]); }
    }
    let acts: Vec<Vec<f32>> = a.chunks_exact(2 * ff).map(|g| (0..ff).map(|i| ops::silu(g[i]) * g[ff + i]).collect()).collect();
    let racts: Option<Vec<Vec<f32>>> = rot.map(|r| acts.iter().map(|a| r.rotated(a)).collect());
    let ds: Vec<Tensor> = pairs.iter().map(|&(_, e, _)| down.expert(e, n)).collect();
    let mut o = vec![0f32; k * c.n_embd];
    batch_rows(&ds, &racts.as_ref().unwrap_or(&acts).iter().map(|v| v.as_slice()).collect::<Vec<_>>(), &mut o);
    for (j, (oj, &(_, e, _))) in o.chunks_exact_mut(c.n_embd).zip(pairs).enumerate() {
        m.lora_delta(&format!("blk.{il}.ffn_down_exps.weight"), &acts[j], oj, Some(e));
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 4-activation kernel must give the bits of the single kernel (72 blocks, and 9: an odd count).
    #[test]
    fn dot4_matches_four_dots() {
        for ty in [GgmlType::Q8_0, GgmlType::IQ4NL] {
            let (Some(dot), Some(dot4)) = (int_dot(ty), int_dot4(ty)) else { continue };
            for cols in [2304usize, 288] {
                let rb = ty.row_bytes(cols as u64);
                let mut row: Vec<u8> = (0..rb).map(|i| (i * 37 % 251) as u8).collect();
                // A valid f16 scale in each block (1.0, 1.5, ...).
                for (i, b) in row.chunks_exact_mut(rb / (cols / 32)).enumerate() { b[..2].copy_from_slice(&[0, 0x3C + (i % 4) as u8]); }
                let xq: Vec<_> = (0..4).map(|t| quantize_q8k(&(0..cols).map(|i| ((i * 13 + t * 29) % 97) as f32 * 0.37 - 17.0).collect::<Vec<_>>())).collect();
                let got = dot4(&row, [&xq[0], &xq[1], &xq[2], &xq[3]]);
                for (g, x) in got.iter().zip(&xq) { assert_eq!(g.to_bits(), dot(&row, x).to_bits(), "{ty:?}, {cols} values"); }
            }
        }
    }

    /// Batch boundaries can cross matrices; both Q2_0 and the f32 fallback must use the right input row.
    #[test]
    fn batch_rows_match_individual_matvecs() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let (cols, rows, count) = (640, 257, 3);
        let inputs: Vec<Vec<f32>> = (0..count).map(|m| (0..cols).map(|i| ((i * 13 + m * 7) % 101) as f32 / 50.0 - 1.0).collect()).collect();
        for ty in [GgmlType::Q2_0, GgmlType::F32] {
            let rb = ty.row_bytes(cols as u64);
            let data: Vec<Vec<u8>> = (0..count).map(|m| {
                let mut v = vec![0u8; rows * rb];
                if ty == GgmlType::Q2_0 {
                    for (i, b) in v.chunks_exact_mut(18).enumerate() {
                        b[..2].copy_from_slice(&f16::from_f32(0.01).to_le_bytes());
                        for (j, q) in b[2..].iter_mut().enumerate() { *q = ((i * 17 + j * 11 + m) % 256) as u8; }
                    }
                } else {
                    for (i, b) in v.chunks_exact_mut(4).enumerate() { b.copy_from_slice(&(((i + m * 3) % 113) as f32 / 57.0 - 1.0).to_le_bytes()); }
                }
                v
            }).collect();
            let mats: Vec<Tensor> = data.iter().map(|d| Tensor { ty, cols, rows, data: d, scale: 1.0 }).collect();
            let xs: Vec<&[f32]> = inputs.iter().map(Vec::as_slice).collect();
            let mut got = vec![0.0; rows * count];
            pool.install(|| batch_rows(&mats, &xs, &mut got));
            for (i, t) in mats.iter().enumerate() {
                let xq = quantize_q8k(xs[i]);
                let dot = int_dot(ty);
                let mut decoded = vec![0.0; cols];
                for r in 0..rows {
                    let want = if let Some(d) = dot { d(&t.data[r * rb..(r + 1) * rb], &xq) }
                        else { t.row(r, &mut decoded); decoded.iter().zip(xs[i]).map(|(a, b)| a * b).sum() };
                    assert_eq!(got[i * rows + r], want, "{} matrix {i} row {r}", ty.name());
                }
            }
        }
    }

    /// The trellis kernels equal the f32 dequantized dot on random blocks.
    #[test]
    fn tq_dots_match_dequant() {
        let mut r = 0x9E3779B97F4A7C15u64;
        let mut rnd = || { r ^= r << 13; r ^= r >> 7; r ^= r << 17; r as u8 };
        let x: Vec<f32> = (0..640).map(|i| ((i as f32 * 0.7548777).fract() - 0.5) * 2.0).collect();
        for ty in [GgmlType::TQ2T, GgmlType::TQK6, GgmlType::TQK7] {
            let mut row: Vec<u8> = (0..5 * ty.block_bytes()).map(|_| rnd()).collect();
            for b in row.chunks_exact_mut(ty.block_bytes()) { b[..2].copy_from_slice(&f16::from_f32(0.01).to_le_bytes()); }
            let mut w = vec![0f32; 640];
            crate::quant::dequant_row(ty, &row, &mut w);
            let want: f64 = w.iter().zip(&x).map(|(&a, &b)| a as f64 * b as f64).sum();
            let got = float_dot(ty).unwrap()(&row, &x);
            assert!(((got as f64 - want) / want.abs().max(1e-3)).abs() < 1e-5, "{}: {got} vs {want}", ty.name());
        }
    }

    /// Every integer kernel (scalar and, when built, AVX2) equals the f32 dequantized dot on random blocks.
    #[test]
    fn int_dots_match_dequant() {
        let mut r = 0x9E3779B97F4A7C15u64;
        let mut rnd = || { r ^= r << 13; r ^= r >> 7; r ^= r << 17; r as u8 };
        let x: Vec<f32> = (0..1024).map(|i| ((i as f32 * 0.7548777).fract() - 0.5) * 2.0).collect();
        for ty in [GgmlType::Q2_0, GgmlType::TQ1_0, GgmlType::Q4_0, GgmlType::Q4_1, GgmlType::Q5_0, GgmlType::Q5_1, GgmlType::IQ2XS, GgmlType::IQ3XXS, GgmlType::IQ3S, GgmlType::IQ4XS, GgmlType::Q6K, GgmlType::Q2K, GgmlType::Q3K, GgmlType::Q4K, GgmlType::Q5K, GgmlType::Q8_0, GgmlType::IQ2XXS, GgmlType::IQ2S, GgmlType::IQ1S, GgmlType::IQ1M, GgmlType::IQ4NL, GgmlType::NVFP4] {
            // Q2_0 at 640 values exercises the zero-padded last Q8_K block.
            let n = match ty { GgmlType::Q2_0 => 640, GgmlType::IQ4NL => 672, GgmlType::NVFP4 => 704, _ => 1024 };
            let xq = quantize_q8k(&x[..n]);
            let mut row: Vec<u8> = (0..n / ty.block_elems() * ty.block_bytes()).map(|_| rnd()).collect();
            // Offsets of the f16 scales (d, and dmin where the type has one).
            let fs: &[usize] = match ty { GgmlType::TQ1_0 => &[52], GgmlType::Q6K => &[208], GgmlType::Q2K => &[80, 82], GgmlType::Q3K => &[108], GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q4_1 | GgmlType::Q5_1 => &[0, 2], GgmlType::IQ1M | GgmlType::NVFP4 => &[], _ => &[0] };
            for b in row.chunks_exact_mut(ty.block_bytes()) { for &o in fs { b[o..o + 2].copy_from_slice(&f16::from_f32(0.01).to_le_bytes()); } }
            if ty == GgmlType::NVFP4 {
                for b in row.chunks_exact_mut(36) { for s in 0..4 { b[s] = 0x20 + (b[s] & 0x1F); } }
            }
            // IQ1_M keeps d in the top nibbles of its four scale words.
            if ty == GgmlType::IQ1M {
                let d = f16::from_f32(0.01).to_bits();
                for b in row.chunks_exact_mut(56) { for i in 0..4 { b[49 + 2 * i] = b[49 + 2 * i] & 0xF | ((d >> (4 * i)) as u8 & 0xF) << 4; } }
            }
            let mut w = vec![0f32; n];
            crate::quant::dequant_row(ty, &row, &mut w);
            let want: f64 = w.iter().enumerate().map(|(i, &v)| v as f64 * (xq[i / 256].d * xq[i / 256].qs[i % 256] as f32) as f64).sum();
            let scalar = match ty { GgmlType::TQ1_0 => dot_tq1_0, GgmlType::IQ2XS => dot_iq2xs, GgmlType::IQ3XXS => dot_iq3xxs, GgmlType::IQ3S => dot_iq3s, GgmlType::Q6K => dot_q6k, GgmlType::Q2K => dot_q2k, GgmlType::Q3K => dot_q3k, GgmlType::Q4K => dot_q4k, GgmlType::Q5K => dot_q5k, GgmlType::Q8_0 => dot_q8_0, GgmlType::IQ2XXS => dot_iq2xxs, GgmlType::IQ2S => dot_iq2s, GgmlType::IQ1S => dot_iq1s, GgmlType::IQ1M => dot_iq1m, GgmlType::Q2_0 => dot_q2_0, GgmlType::Q4_0 => dot_q4::<2>, GgmlType::Q4_1 => dot_q4::<3>, GgmlType::Q5_0 => dot_q4::<6>, GgmlType::Q5_1 => dot_q4::<7>, GgmlType::IQ4NL => dot_iq4nl, GgmlType::NVFP4 => dot_nvfp4, _ => dot_iq4xs };
            for got in [scalar(&row, &xq), int_dot(ty).unwrap()(&row, &xq)] {
                assert!(((got as f64 - want) / want.abs().max(1e-3)).abs() < 1e-5, "{}: {got} vs {want}", ty.name());
            }
        }
    }

    #[test]
    fn f8_e4m3_dot_and_scale_match_dequant() {
        let mut r = 0x517CC1B727220A95u64;
        let mut rnd = || { r ^= r << 13; r ^= r >> 7; r ^= r << 17; r as u8 };
        let (rows, cols) = (5usize, 131usize);
        let data: Vec<u8> = (0..rows * cols).map(|_| rnd()).collect();
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32 * 0.7548777).fract() - 0.5) * 2.0).collect();
        let scale = 1.75f32;
        let t = Tensor { ty: GgmlType::F8E4M3, cols, rows, data: &data, scale };
        let (mut got, mut want) = (vec![0.0f32; rows], vec![0.0f32; rows]);
        matvec(&t, &x, &mut got);
        t.matvec(&x, &mut want);
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            assert!(((g - w) / w.abs().max(1e-3)).abs() < 1e-5, "row {i}: {g} vs {w}");
        }
    }

    /// AVX-512 VNNI+VBMI kernels match AVX2 and scalar dequantized dot, including odd-block tails.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw", target_feature = "avx512vnni", target_feature = "avx512vbmi"))]
    #[test]
    fn avx512_dots_match_avx2_and_dequant() {
        let mut r = 0xD1B54A32D192ED03u64;
        let mut rnd = || { r ^= r << 13; r ^= r >> 7; r ^= r << 17; r as u8 };
        let x: Vec<f32> = (0..1024).map(|i| ((i as f32 * 0.7548777).fract() - 0.5) * 2.0).collect();
        for (ty, n) in [
            (GgmlType::Q8_0, 672),
            (GgmlType::Q8_0, 1024),
            (GgmlType::Q4_0, 672),
            (GgmlType::Q4_1, 672),
            (GgmlType::Q5_0, 672),
            (GgmlType::Q5_1, 672),
            (GgmlType::IQ4NL, 672),
            (GgmlType::IQ4XS, 1024),
            (GgmlType::Q4K, 1024),
            (GgmlType::Q6K, 1024),
            (GgmlType::TQ1_0, 1024),
            (GgmlType::IQ1S, 1024),
            (GgmlType::IQ2XXS, 1024),
            (GgmlType::IQ3S, 1024),
        ] {
            let xq = quantize_q8k(&x[..n]);
            let mut row: Vec<u8> = (0..n / ty.block_elems() * ty.block_bytes()).map(|_| rnd()).collect();
            let fs: &[usize] = match ty {
                GgmlType::TQ1_0 => &[52],
                GgmlType::Q6K => &[208],
                GgmlType::Q4K | GgmlType::Q4_1 | GgmlType::Q5_1 => &[0, 2],
                _ => &[0],
            };
            for b in row.chunks_exact_mut(ty.block_bytes()) {
                for &o in fs { b[o..o + 2].copy_from_slice(&f16::from_f32(0.01).to_le_bytes()); }
            }
            let mut w = vec![0f32; n];
            crate::quant::dequant_row(ty, &row, &mut w);
            let want: f64 = w.iter().enumerate().map(|(i, &v)| v as f64 * (xq[i / 256].d * xq[i / 256].qs[i % 256] as f32) as f64).sum();
            let got512 = match ty {
                GgmlType::Q8_0 => avx512::dot_q8_0(&row, &xq),
                GgmlType::Q4_0 => avx512::dot_q4::<2>(&row, &xq),
                GgmlType::Q4_1 => avx512::dot_q4::<3>(&row, &xq),
                GgmlType::Q5_0 => avx512::dot_q4::<6>(&row, &xq),
                GgmlType::Q5_1 => avx512::dot_q4::<7>(&row, &xq),
                GgmlType::IQ4NL => avx512::dot_iq4nl(&row, &xq),
                GgmlType::IQ4XS => avx512::dot_iq4xs(&row, &xq),
                GgmlType::Q4K => avx512::dot_q4k(&row, &xq),
                GgmlType::Q6K => avx512::dot_q6k(&row, &xq),
                GgmlType::TQ1_0 => avx512::dot_tq1_0(&row, &xq),
                GgmlType::IQ1S => avx512::dot_iq1s(&row, &xq),
                GgmlType::IQ2XXS => avx512::dot_iq2xxs(&row, &xq),
                GgmlType::IQ3S => avx512::dot_iq3s(&row, &xq),
                _ => unreachable!(),
            };
            let got256 = match ty {
                GgmlType::Q8_0 => avx2::dot_q8_0(&row, &xq),
                GgmlType::Q4_0 => avx2::dot_q4::<2>(&row, &xq),
                GgmlType::Q4_1 => avx2::dot_q4::<3>(&row, &xq),
                GgmlType::Q5_0 => avx2::dot_q4::<6>(&row, &xq),
                GgmlType::Q5_1 => avx2::dot_q4::<7>(&row, &xq),
                GgmlType::IQ4NL => avx2::dot_iq4nl(&row, &xq),
                GgmlType::IQ4XS => avx2::dot_iq4xs(&row, &xq),
                GgmlType::Q4K => avx2::dot_q4k(&row, &xq),
                GgmlType::Q6K => avx2::dot_q6k(&row, &xq),
                GgmlType::TQ1_0 => avx2::dot_tq1_0(&row, &xq),
                GgmlType::IQ1S => avx2::dot_iq1s(&row, &xq),
                GgmlType::IQ2XXS => avx2::dot_iq2xxs(&row, &xq),
                GgmlType::IQ3S => avx2::dot_iq3s(&row, &xq),
                _ => unreachable!(),
            };
            assert!((got512 - got256).abs() <= 1e-5 * got256.abs().max(1.0), "{} n={n}: avx512={got512} vs avx2={got256}", ty.name());
            for got in [got512, got256] {
                assert!(((got as f64 - want) / want.abs().max(0.05)).abs() < 1e-5, "{} n={n}: {got} vs {want} (avx512={got512}, avx2={got256})", ty.name());
            }
        }
    }
}
