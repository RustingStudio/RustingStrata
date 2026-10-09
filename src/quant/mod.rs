//! GGML block layouts and scalar reference dequantization.
//! Ported from llama.cpp `ggml-quants.c` `dequantize_row_*` (MIT License, ggml authors).

pub mod tables;

use half::{bf16, f16};
use tables::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType { F32, BF16, Q8_0, Q2K, Q3K, Q4K, Q5K, Q6K, IQ2XS, IQ3XXS, IQ3S, IQ4XS }
use GgmlType::*;

impl GgmlType {
    pub fn from_u32(id: u32) -> anyhow::Result<Self> {
        Ok(match id {
            0 => F32, 30 => BF16, 8 => Q8_0, 10 => Q2K, 11 => Q3K, 12 => Q4K, 13 => Q5K, 14 => Q6K,
            17 => IQ2XS, 18 => IQ3XXS, 21 => IQ3S, 23 => IQ4XS,
            _ => {
                const NAMES: [&str; 40] = ["F32", "F16", "Q4_0", "Q4_1", "", "", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K",
                    "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL", "IQ3_S",
                    "IQ2_S", "IQ4_XS", "I8", "I16", "I32", "I64", "F64", "IQ1_M", "BF16", "", "", "", "TQ1_0", "TQ2_0",
                    "", "", "", "MXFP4"];
                let name = NAMES.get(id as usize).filter(|n| !n.is_empty()).unwrap_or(&"unknown");
                anyhow::bail!("unsupported quant type {name} (ggml type id {id}); supported: F32 BF16 Q8_0 Q2_K-Q6_K IQ2_XS IQ3_XXS IQ3_S IQ4_XS")
            }
        })
    }
    pub fn block_elems(self) -> usize { match self { F32 | BF16 => 1, Q8_0 => 32, _ => 256 } }
    pub fn block_bytes(self) -> usize {
        match self {
            F32 => 4, BF16 => 2, Q8_0 => 34, Q2K => 84, Q3K => 110, Q4K => 144, Q5K => 176, Q6K => 210,
            IQ2XS => 74, IQ3XXS => 98, IQ3S => 110, IQ4XS => 136,
        }
    }
    pub fn row_bytes(self, n: u64) -> usize { n as usize / self.block_elems() * self.block_bytes() }
    pub fn name(self) -> &'static str {
        match self {
            F32 => "F32", BF16 => "BF16", Q8_0 => "Q8_0", Q2K => "Q2_K", Q3K => "Q3_K", Q4K => "Q4_K",
            Q5K => "Q5_K", Q6K => "Q6_K", IQ2XS => "IQ2_XS", IQ3XXS => "IQ3_XXS", IQ3S => "IQ3_S", IQ4XS => "IQ4_XS",
        }
    }
}

#[inline]
fn h(b: &[u8], o: usize) -> f32 { f16::from_le_bytes([b[o], b[o + 1]]).to_f32() }

fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 { (q[j] & 63, q[j + 4] & 63) }
    else { ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)) }
}

/// Dequantize `dst.len()` elements; `src` must hold exactly `ty.row_bytes(dst.len())` bytes.
pub fn dequant_row(ty: GgmlType, src: &[u8], dst: &mut [f32]) {
    let (be, bb) = (ty.block_elems(), ty.block_bytes());
    debug_assert_eq!(src.len(), ty.row_bytes(dst.len() as u64));
    for (blk, y) in src.chunks_exact(bb).zip(dst.chunks_exact_mut(be)) {
        match ty {
            F32 => y[0] = f32::from_le_bytes(blk.try_into().unwrap()),
            BF16 => y[0] = bf16::from_le_bytes([blk[0], blk[1]]).to_f32(),
            Q8_0 => { let d = h(blk, 0); for j in 0..32 { y[j] = blk[2 + j] as i8 as f32 * d; } }
            Q2K => q2k(blk, y), Q3K => q3k(blk, y), Q4K => q4k(blk, y), Q5K => q5k(blk, y), Q6K => q6k(blk, y),
            IQ2XS => iq2xs(blk, y), IQ3XXS => iq3xxs(blk, y), IQ3S => iq3s(blk, y), IQ4XS => iq4xs(blk, y),
        }
    }
}

fn q2k(b: &[u8], y: &mut [f32]) {
    let (sc, d, min) = (&b[0..16], h(b, 80), h(b, 82));
    let (mut q, mut is, mut o) = (16, 0, 0);
    for _ in 0..2 {
        for shift in [0, 2, 4, 6] {
            for half in 0..2 {
                let s = sc[is];
                is += 1;
                let (dl, ml) = (d * (s & 0xF) as f32, min * (s >> 4) as f32);
                for l in 0..16 { y[o] = dl * ((b[q + l + 16 * half] >> shift) & 3) as f32 - ml; o += 1; }
            }
        }
        q += 32;
    }
}

fn q3k(b: &[u8], y: &mut [f32]) {
    let (hm, d_all) = (&b[0..32], h(b, 108));
    let mut aux = [0u32; 4];
    for i in 0..3 { aux[i] = u32::from_le_bytes(b[96 + 4 * i..100 + 4 * i].try_into().unwrap()); }
    let (k1, k2, tmp) = (0x03030303u32, 0x0f0f0f0fu32, aux[2]);
    aux[2] = ((aux[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4);
    aux[3] = ((aux[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4);
    aux[0] = (aux[0] & k2) | ((tmp & k1) << 4);
    aux[1] = (aux[1] & k2) | (((tmp >> 2) & k1) << 4);
    let mut scales = [0i8; 16];
    for (i, a) in aux.iter().enumerate() {
        for (j, x) in a.to_le_bytes().into_iter().enumerate() { scales[4 * i + j] = x as i8; }
    }
    let (mut q, mut is, mut o, mut m) = (32usize, 0, 0, 1u8);
    for _ in 0..2 {
        for shift in [0, 2, 4, 6] {
            for half in 0..2 {
                let dl = d_all * (scales[is] as i32 - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let i = l + 16 * half;
                    let v = ((b[q + i] >> shift) & 3) as i32 - if hm[i] & m != 0 { 0 } else { 4 };
                    y[o] = dl * v as f32;
                    o += 1;
                }
            }
            m <<= 1;
        }
        q += 32;
    }
}

fn q4k(b: &[u8], y: &mut [f32]) {
    let (d, min, sc) = (h(b, 0), h(b, 2), &b[4..16]);
    let (mut q, mut o) = (16, 0);
    for j in 0..4 {
        let (s1, m1) = scale_min_k4(2 * j, sc);
        let (s2, m2) = scale_min_k4(2 * j + 1, sc);
        let (d1, m1, d2, m2) = (d * s1 as f32, min * m1 as f32, d * s2 as f32, min * m2 as f32);
        for l in 0..32 { y[o + l] = d1 * (b[q + l] & 0xF) as f32 - m1; }
        for l in 0..32 { y[o + 32 + l] = d2 * (b[q + l] >> 4) as f32 - m2; }
        q += 32;
        o += 64;
    }
}

fn q5k(b: &[u8], y: &mut [f32]) {
    let (d, min, sc, qh) = (h(b, 0), h(b, 2), &b[4..16], &b[16..48]);
    let (mut q, mut o, mut u1, mut u2) = (48, 0, 1u8, 2u8);
    for j in 0..4 {
        let (s1, m1) = scale_min_k4(2 * j, sc);
        let (s2, m2) = scale_min_k4(2 * j + 1, sc);
        let (d1, m1, d2, m2) = (d * s1 as f32, min * m1 as f32, d * s2 as f32, min * m2 as f32);
        for l in 0..32 { y[o + l] = d1 * ((b[q + l] & 0xF) as f32 + if qh[l] & u1 != 0 { 16.0 } else { 0.0 }) - m1; }
        for l in 0..32 { y[o + 32 + l] = d2 * ((b[q + l] >> 4) as f32 + if qh[l] & u2 != 0 { 16.0 } else { 0.0 }) - m2; }
        q += 32;
        o += 64;
        u1 <<= 2;
        u2 <<= 2;
    }
}

fn q6k(b: &[u8], y: &mut [f32]) {
    let d = h(b, 208);
    for n in 0..2 {
        let (ql, qh, sc) = (&b[64 * n..], &b[128 + 32 * n..], &b[192 + 8 * n..]);
        let yo = 128 * n;
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
            let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
            y[yo + l] = d * sc[is] as i8 as f32 * q1 as f32;
            y[yo + l + 32] = d * sc[is + 2] as i8 as f32 * q2 as f32;
            y[yo + l + 64] = d * sc[is + 4] as i8 as f32 * q3 as f32;
            y[yo + l + 96] = d * sc[is + 6] as i8 as f32 * q4 as f32;
        }
    }
}

fn sign(signs: u8, j: usize) -> f32 { if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 } }

fn iq2xs(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    let scales = &b[66..74];
    for ib in 0..8 {
        let db = [d * (0.5 + (scales[ib] & 0xF) as f32) * 0.25, d * (0.5 + (scales[ib] >> 4) as f32) * 0.25];
        for l in 0..4 {
            let o = 2 + 2 * (4 * ib + l);
            let q = u16::from_le_bytes([b[o], b[o + 1]]);
            let grid = IQ2XS_GRID[(q & 511) as usize].to_le_bytes();
            let signs = KSIGNS_IQ2XS[(q >> 9) as usize];
            for j in 0..8 { y[32 * ib + 8 * l + j] = db[l / 2] * grid[j] as f32 * sign(signs, j); }
        }
    }
}

fn iq3xxs(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    let (qs, ss) = (&b[2..66], &b[66..98]);
    for ib in 0..8 {
        let aux = u32::from_le_bytes(ss[4 * ib..4 * ib + 4].try_into().unwrap());
        let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
        for l in 0..4 {
            let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
            let g1 = IQ3XXS_GRID[qs[8 * ib + 2 * l] as usize].to_le_bytes();
            let g2 = IQ3XXS_GRID[qs[8 * ib + 2 * l + 1] as usize].to_le_bytes();
            let o = 32 * ib + 8 * l;
            for j in 0..4 {
                y[o + j] = db * g1[j] as f32 * sign(signs, j);
                y[o + j + 4] = db * g2[j] as f32 * sign(signs, j + 4);
            }
        }
    }
}

fn iq3s(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    let (qs, qh, signs, sc) = (&b[2..66], &b[66..74], &b[74..106], &b[106..110]);
    for ib in 0..8 {
        let db = d * (1 + 2 * ((sc[ib / 2] >> (4 * (ib % 2))) & 0xF) as i32) as f32;
        for l in 0..4 {
            let hb = qh[ib] as usize;
            let g1 = IQ3S_GRID[qs[8 * ib + 2 * l] as usize | ((hb << (8 - 2 * l)) & 256)].to_le_bytes();
            let g2 = IQ3S_GRID[qs[8 * ib + 2 * l + 1] as usize | ((hb << (7 - 2 * l)) & 256)].to_le_bytes();
            let (sg, o) = (signs[4 * ib + l], 32 * ib + 8 * l);
            for j in 0..4 {
                y[o + j] = db * g1[j] as f32 * sign(sg, j);
                y[o + j + 4] = db * g2[j] as f32 * sign(sg, j + 4);
            }
        }
    }
}

fn iq4xs(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    let sh = u16::from_le_bytes([b[2], b[3]]);
    let (sl, qs) = (&b[4..8], &b[8..136]);
    for ib in 0..8 {
        let ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) as i32 | ((((sh >> (2 * ib)) & 3) as i32) << 4);
        let dl = d * (ls - 32) as f32;
        for j in 0..16 {
            let q = qs[16 * ib + j];
            y[32 * ib + j] = dl * KVALUES_IQ4NL[(q & 0xF) as usize] as f32;
            y[32 * ib + j + 16] = dl * KVALUES_IQ4NL[(q >> 4) as usize] as f32;
        }
    }
}
