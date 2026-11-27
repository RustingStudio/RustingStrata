//! GGML block layouts and scalar reference dequantization.
//! Ported from llama.cpp `ggml-quants.c` `dequantize_row_*` (MIT License, ggml authors).

pub mod tables;
pub mod tq_lut;

use half::{bf16, f16};
use tables::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType { F32, F16, BF16, F8E4M3, Q2_0, TQ1_0, TQ2_0, TQ2T, TQK6, TQK7, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2K, Q3K, Q4K, Q5K, Q6K, IQ1S, IQ1M, IQ2XXS, IQ2XS, IQ2S, IQ3XXS, IQ3S, IQ4XS, IQ4NL, NVFP4 }
use GgmlType::*;

impl GgmlType {
    pub fn from_u32(id: u32) -> anyhow::Result<Self> {
        Ok(match id {
            0 => F32, 1 => F16, 30 => BF16, 43 => F8E4M3, 42 => Q2_0, 34 => TQ1_0, 35 => TQ2_0, 144 => TQ2T, 145 => TQK6, 146 => TQK7, 2 => Q4_0, 3 => Q4_1, 6 => Q5_0, 7 => Q5_1, 8 => Q8_0, 10 => Q2K, 11 => Q3K, 12 => Q4K, 13 => Q5K, 14 => Q6K,
            16 => IQ2XXS, 17 => IQ2XS, 19 => IQ1S, 29 => IQ1M, 18 => IQ3XXS, 21 => IQ3S, 22 => IQ2S, 23 => IQ4XS, 20 => IQ4NL, 40 => NVFP4,
            _ => {
                const NAMES: [&str; 44] = ["F32", "F16", "Q4_0", "Q4_1", "", "", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K",
                    "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL", "IQ3_S",
                    "IQ2_S", "IQ4_XS", "I8", "I16", "I32", "I64", "F64", "IQ1_M", "BF16", "", "", "", "TQ1_0", "TQ2_0",
                    "", "", "", "MXFP4", "NVFP4", "Q1_0", "Q2_0", "F8_E4M3"];
                // agentionai's llama.cpp fork: the Gyro-L file holds the pair trellis types, which are not ported.
                const GYRO: [(u32, &str); 4] = [(143, "PTQ1_0"), (147, "TQP5"), (148, "TQP6"), (149, "TQP7")];
                if let Some((_, name)) = GYRO.iter().find(|g| g.0 == id) {
                    anyhow::bail!("unsupported quant type {name} (ggml type id {id}): of the trellis types of agentionai's llama.cpp fork only TQ2_T, TQK6 and TQK7 (the Gyro-S and Gyro-M files) are supported, not the pair trellis of Gyro-L");
                }
                let name = NAMES.get(id as usize).filter(|n| !n.is_empty()).unwrap_or(&"unknown");
                anyhow::bail!("unsupported quant type {name} (ggml type id {id}); supported: F32 F16 BF16 F8_E4M3 Q2_0 TQ1_0 TQ2_0 TQ2_T TQK6 TQK7 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q2_K-Q6_K IQ1_S IQ1_M IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S IQ4_NL IQ4_XS NVFP4")
            }
        })
    }
    /// A GPU matvec of this type reads its input as floats, not as int8: the float types (F16 is uploaded as F32)
    /// and the trellis types, whose codebook is not on an int8 grid (as int8 its values are 0.8% off).
    pub fn float_x(self) -> bool { matches!(self, F32 | BF16 | F8E4M3 | TQ2T | TQK6 | TQK7) }
    pub fn block_elems(self) -> usize { match self { F32 | F16 | BF16 | F8E4M3 => 1, Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | IQ4NL => 32, Q2_0 | NVFP4 => 64, TQ2T | TQK6 | TQK7 => 128, _ => 256 } }
    pub fn block_bytes(self) -> usize {
        match self {
            F32 => 4, F16 => 2, BF16 => 2, F8E4M3 => 1, Q2_0 => 18, TQ1_0 => 54, TQ2_0 => 66, TQ2T => 34, TQK6 => 26, TQK7 => 30, Q4_0 => 18, Q4_1 => 20, Q5_0 => 22, Q5_1 => 24, Q8_0 => 34, Q2K => 84, Q3K => 110, Q4K => 144, Q5K => 176, Q6K => 210,
            IQ1S => 50, IQ1M => 56, IQ2XXS => 66, IQ2XS => 74, IQ2S => 82, IQ3XXS => 98, IQ3S => 110, IQ4XS => 136, IQ4NL => 18, NVFP4 => 36,
        }
    }
    /// Backend `backend` ("cuda" or "vulkan") has kernels for this type. F16 has none: `Backend::upload` converts it to
    /// F32, which holds its values exactly.
    pub fn gpu(self, _backend: &str) -> bool { self != F16 }
    pub fn row_bytes(self, n: u64) -> usize { n as usize / self.block_elems() * self.block_bytes() }
    pub fn name(self) -> &'static str {
        match self {
            F32 => "F32", F16 => "F16", BF16 => "BF16", F8E4M3 => "F8_E4M3", Q2_0 => "Q2_0", TQ1_0 => "TQ1_0", TQ2_0 => "TQ2_0", TQ2T => "TQ2_T", TQK6 => "TQK6", TQK7 => "TQK7", Q4_0 => "Q4_0", Q4_1 => "Q4_1", Q5_0 => "Q5_0", Q5_1 => "Q5_1", Q8_0 => "Q8_0", Q2K => "Q2_K", Q3K => "Q3_K", Q4K => "Q4_K",
            Q5K => "Q5_K", Q6K => "Q6_K", IQ1S => "IQ1_S", IQ1M => "IQ1_M", IQ2XXS => "IQ2_XXS", IQ2XS => "IQ2_XS", IQ2S => "IQ2_S", IQ3XXS => "IQ3_XXS", IQ3S => "IQ3_S", IQ4XS => "IQ4_XS", IQ4NL => "IQ4_NL", NVFP4 => "NVFP4",
        }
    }
}

/// Compile-time lookup table for signed OCP FP8 E4M3 (`0x7F`/`0xFF` NaN -> `0.0`).
pub static F8_E4M3_LUT: [f32; 256] = {
    let mut t = [0.0f32; 256];
    let mut i = 0usize;
    while i < 256 {
        let ax = (i & 0x7F) as u32;
        if ax != 0 && ax != 0x7F {
            let exp = ax >> 3;
            let man = ax & 7;
            let mag = if exp == 0 {
                man as f32 * (1.0 / 512.0)
            } else {
                f32::from_bits(((exp + 120) << 23) | (man << 20))
            };
            t[i] = if i & 0x80 != 0 { -mag } else { mag };
        }
        i += 1;
    }
    t
};

/// Convert signed OCP FP8 E4M3 (`0x7F`/`0xFF` NaN -> `0.0`) to `f32`.
#[inline]
pub fn f8_e4m3_to_f32(x: u8) -> f32 { F8_E4M3_LUT[x as usize] }

#[inline]
fn h(b: &[u8], o: usize) -> f32 { f16::from_le_bytes([b[o], b[o + 1]]).to_f32() }

pub(crate) fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
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
            F16 => y[0] = h(blk, 0),
            BF16 => y[0] = bf16::from_le_bytes([blk[0], blk[1]]).to_f32(),
            F8E4M3 => y[0] = f8_e4m3_to_f32(blk[0]),
            // Q2_0 (Strata/ggml type 42): f16 d, then 2-bit codes, value = (code - 1) * d.
            Q2_0 => { let d = h(blk, 0); for j in 0..64 { y[j] = (((blk[2 + j / 4] >> (2 * (j % 4))) & 3) as f32 - 1.0) * d; } }
            TQ1_0 => tq1_0(blk, y), TQ2_0 => tq2_0(blk, y),
            TQ2T | TQK6 | TQK7 => tq(blk, y),
            Q4_0 | Q5_0 => { let (d, q) = (h(blk, 0), q5_codes(ty, blk)); for j in 0..32 { y[j] = q[j] as f32 * d; } }
            // Q4_1 / Q5_1: f16 d, f16 m, (Q5_1: u32 high bits,) nibbles as in Q4_0; value = d * code + m.
            Q4_1 | Q5_1 => {
                let (d, m, qh, qs) = if ty == Q5_1 { (h(blk, 0), h(blk, 2), u32::from_le_bytes(blk[4..8].try_into().unwrap()), &blk[8..24]) } else { (h(blk, 0), h(blk, 2), 0, &blk[4..20]) };
                for j in 0..16 {
                    y[j] = d * ((qs[j] & 0xF) as u32 | ((qh >> j) & 1) << 4) as f32 + m;
                    y[j + 16] = d * ((qs[j] >> 4) as u32 | ((qh >> (j + 16)) & 1) << 4) as f32 + m;
                }
            }
            Q8_0 => { let d = h(blk, 0); for j in 0..32 { y[j] = blk[2 + j] as i8 as f32 * d; } }
            Q2K => q2k(blk, y), Q3K => q3k(blk, y), Q4K => q4k(blk, y), Q5K => q5k(blk, y), Q6K => q6k(blk, y),
            IQ1S => iq1s(blk, y), IQ1M => iq1m(blk, y), IQ2XXS => iq2xxs(blk, y), IQ2XS => iq2xs(blk, y), IQ2S => iq2s(blk, y), IQ3XXS => iq3xxs(blk, y), IQ3S => iq3s(blk, y), IQ4XS => iq4xs(blk, y), IQ4NL => iq4nl(blk, y), NVFP4 => nvfp4(blk, y),
        }
    }
}

/// Value of trit `n` (0 = most significant) packed in a TQ1_0 byte, as llama.cpp's `dequantize_row_tq1_0`:
/// the byte times 3^n in u8, then (3 q) >> 8. The result is -1, 0 or 1.
fn tq1_trit(v: u8, n: usize) -> f32 {
    const POW3: [u8; 5] = [1, 3, 9, 27, 81];
    let q = v.wrapping_mul(POW3[n]);
    (((q as u16) * 3) >> 8) as f32 - 1.0
}

/// TQ1_0 (54 bytes): 48 bytes of five trits, 4 bytes of four trits, then f16 d. Values 0..160 come from the first 32
/// bytes (trit n of byte m is element 32 n + m), 160..240 from the next 16 (element 160 + 16 n + m), 240..256 from qh
/// (element 240 + 4 n + j).
fn tq1_0(b: &[u8], y: &mut [f32]) {
    let d = h(b, 52);
    let mut o = 0;
    for n in 0..5 { for m in 0..32 { y[o] = d * tq1_trit(b[m], n); o += 1; } }
    for n in 0..5 { for m in 0..16 { y[o] = d * tq1_trit(b[32 + m], n); o += 1; } }
    for n in 0..4 { for j in 0..4 { y[o] = d * tq1_trit(b[48 + j], n); o += 1; } }
}

/// TQ2_0 (66 bytes): 64 bytes of 2-bit codes, then f16 d. Element 128 j + 32 l + m is codes byte 32 j + m bits 2 l,
/// value (code - 1) d.
fn tq2_0(b: &[u8], y: &mut [f32]) {
    let d = h(b, 64);
    for j in 0..2 { for l in 0..4 { for m in 0..32 {
        y[32 * (4 * j + l) + m] = d * (((b[32 * j + m] >> (2 * l)) & 3) as f32 - 1.0);
    } } }
}

/// The 2048 codebook points of the trellis types (TQ2_T, TQK6, TQK7 of agentionai's llama.cpp fork), two values each.
pub static TQ_LUT: [[f32; 2]; 2048] = {
    let mut t = [[0f32; 2]; 2048];
    let mut i = 0;
    while i < 4096 { t[i / 2][i % 2] = f16::from_bits(tq_lut::TQ_LUT_F16[i]).to_f32_const(); i += 1; }
    t
};

/// The two codebook points of trellis step `t` (0..32) of a block's `qs` (4 K bytes: K = 8 for TQ2_T, 7 for TQK7,
/// 6 for TQK6). The step's state is 15 bits of the stream: for TQ2_T bytes `t - 1` and `t` (circular), for TQK the bits
/// from bit `(31 - t) K` on, least significant first, circular over the 32 K bits. The points are two 11-bit fields
/// of the state times 0x9e3779b1.
#[inline]
pub fn tq_points(qs: &[u8], t: usize) -> (usize, usize) { state_points(tq_state(qs, t)) }

/// The 15-bit state of trellis step `t` (see `tq_points`).
#[inline]
pub fn tq_state(qs: &[u8], t: usize) -> usize {
    let n = qs.len();
    let s = if n == 32 { (qs[(t + 31) & 31] as u32) << 8 | qs[t] as u32 } else {
        let off = (31 - t) * (n / 4);
        let b = off >> 3;
        let w = |i: usize| qs[if i < n { i } else { i - n }] as u32;  // i < 2 n; a division here costs a third of a dot
        (qs[b] as u32 | w(b + 1) << 8 | w(b + 2) << 16) >> (off & 7)
    };
    (s & 0x7fff) as usize
}

/// A TQK stream with its first 4 bytes repeated after it, so `tq_state_in` reads a state with one 4-byte load.
#[inline(always)]
pub fn tq_stream(qs: &[u8]) -> [u8; 36] {
    let (n, mut b) = (qs.len(), [0u8; 36]);
    if n < 32 {
        b[..n].copy_from_slice(qs);
        b[n..n + 4].copy_from_slice(&qs[..4]);
    }
    b
}

/// `tq_state(qs, t)`, with `buf = tq_stream(qs)`.
#[inline(always)]
pub fn tq_state_in(buf: &[u8; 36], qs: &[u8], t: usize) -> usize {
    let n = qs.len();
    if n == 32 { return tq_state(qs, t) }
    let off = (31 - t) * (n / 4);
    (u32::from_le_bytes(buf[off >> 3..][..4].try_into().unwrap()) >> (off & 7) & 0x7fff) as usize
}

#[inline]
fn state_points(s: usize) -> (usize, usize) {
    let x = (s as u32).wrapping_mul(0x9e3779b1);
    ((x >> 21) as usize, ((x >> 10) & 2047) as usize)
}

/// The four values of each trellis state (its two codebook points), 512 KB: a step is one load instead of the hash
/// and two loads, and the values are the same f32s.
pub static TQ_STATES: std::sync::LazyLock<Box<[[f32; 4]]>> = std::sync::LazyLock::new(|| (0..1 << 15).map(|s| {
    let (a, b) = state_points(s);
    [TQ_LUT[a][0], TQ_LUT[a][1], TQ_LUT[b][0], TQ_LUT[b][1]]
}).collect());

/// `TQ_STATES` times 8192, rounded to int16 (|values| < 4), for the integer CPU kernel.
pub static TQ_STATES_I16: std::sync::LazyLock<Box<[[i16; 4]]>> =
    std::sync::LazyLock::new(|| TQ_STATES.iter().map(|v| v.map(|x| (x * 8192.0).round() as i16)).collect());

/// TQ2_T (34 bytes), TQK7 (30), TQK6 (26): f16 d, then 32 trellis steps; step t gives values 4 t .. 4 t + 4, the two
/// codebook points of `tq_points` times d.
fn tq(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    for (t, y) in y.chunks_exact_mut(4).enumerate() {
        let (pa, pb) = tq_points(&b[2..], t);
        y[0] = TQ_LUT[pa][0] * d; y[1] = TQ_LUT[pa][1] * d; y[2] = TQ_LUT[pb][0] * d; y[3] = TQ_LUT[pb][1] * d;
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

/// The 16 6-bit scales of a Q3_K block (bias 32 not removed).
pub(crate) fn q3k_scales(b: &[u8]) -> [i8; 16] {
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
    scales
}

fn q3k(b: &[u8], y: &mut [f32]) {
    let (hm, d_all, scales) = (&b[0..32], h(b, 108), q3k_scales(b));
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

fn iq1s(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    for ib in 0..8 {
        let qh = u16::from_le_bytes([b[34 + 2 * ib], b[35 + 2 * ib]]) as usize;
        let dl = d * (2 * ((qh >> 12) & 7) + 1) as f32;
        let delta = if qh & 0x8000 != 0 { -0.125 } else { 0.125 };
        for l in 0..4 {
            let grid = IQ1S_GRID[b[2 + 4 * ib + l] as usize | (((qh >> (3 * l)) & 7) << 8)].to_le_bytes();
            for j in 0..8 { y[32 * ib + 8 * l + j] = dl * (grid[j] as i8 as f32 + delta); }
        }
    }
}

fn iq1m(b: &[u8], y: &mut [f32]) {
    let sc: [usize; 4] = std::array::from_fn(|i| u16::from_le_bytes([b[48 + 2 * i], b[49 + 2 * i]]) as usize);
    let d = f16::from_bits(((sc[0] >> 12) | ((sc[1] >> 8) & 0xF0) | ((sc[2] >> 4) & 0xF00) | (sc[3] & 0xF000)) as u16).to_f32();
    for ib in 0..8 {
        for l in 0..4 {
            let dl = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3 * (l / 2))) & 7) + 1) as f32;
            let qh = b[32 + 2 * ib + l / 2] as usize >> (4 * (l % 2));
            let grid = IQ1S_GRID[b[4 * ib + l] as usize | ((qh & 7) << 8)].to_le_bytes();
            let delta = if qh & 8 != 0 { -0.125 } else { 0.125 };
            for j in 0..8 { y[32 * ib + 8 * l + j] = dl * (grid[j] as i8 as f32 + delta); }
        }
    }
}

fn iq2xxs(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    for ib in 0..8 {
        let (qs, aux) = (&b[2 + 8 * ib..6 + 8 * ib], u32::from_le_bytes(b[6 + 8 * ib..10 + 8 * ib].try_into().unwrap()));
        let db = d * (0.5 + (aux >> 28) as f32) * 0.25;
        for l in 0..4 {
            let (grid, signs) = (IQ2XXS_GRID[qs[l] as usize].to_le_bytes(), KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize]);
            for j in 0..8 { y[32 * ib + 8 * l + j] = db * grid[j] as f32 * sign(signs, j); }
        }
    }
}

fn iq2s(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    let (qs, signs, qh, scales) = (&b[2..34], &b[34..66], &b[66..74], &b[74..82]);
    for ib in 0..8 {
        let db = [d * (0.5 + (scales[ib] & 0xF) as f32) * 0.25, d * (0.5 + (scales[ib] >> 4) as f32) * 0.25];
        for l in 0..4 {
            let grid = IQ2S_GRID[qs[4 * ib + l] as usize | ((qh[ib] as usize) << (8 - 2 * l) & 0x300)].to_le_bytes();
            for j in 0..8 { y[32 * ib + 8 * l + j] = db[l / 2] * grid[j] as f32 * sign(signs[4 * ib + l], j); }
        }
    }
}

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

/// Signed codes of a Q4_0 block (f16 d, 16 bytes of nibbles; value = (q - 8) d) or a Q5_0 block (f16 d, 32 high bits,
/// 16 bytes of nibbles; value = (q - 16) d). Element j < 16 is the low nibble of byte j, j + 16 its high nibble.
fn q5_codes(ty: GgmlType, b: &[u8]) -> [i8; 32] {
    let mut q = [0i8; 32];
    let (qh, qs) = if ty == Q5_0 { (u32::from_le_bytes(b[2..6].try_into().unwrap()), &b[6..22]) } else { (0, &b[2..18]) };
    let off = if ty == Q5_0 { 16 } else { 8 };
    for j in 0..16 {
        q[j] = ((qs[j] & 0xF) as i32 | (((qh >> j) & 1) << 4) as i32) as i8 - off;
        q[j + 16] = ((qs[j] >> 4) as i32 | (((qh >> (j + 16)) & 1) << 4) as i32) as i8 - off;
    }
    q
}

fn iq4nl(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    for j in 0..16 {
        y[j] = d * KVALUES_IQ4NL[(b[2 + j] & 0xF) as usize] as f32;
        y[j + 16] = d * KVALUES_IQ4NL[(b[2 + j] >> 4) as usize] as f32;
    }
}

/// E2M1 values times 2 (llama.cpp `kvalues_mxfp4`); `ue4m3` halves the scale to match.
pub const KVALUES_FP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// Compile-time lookup table for unsigned E4M3 scales, halved to match `KVALUES_FP4` (`0x7F` NaN -> `0.0`).
static UE4M3_LUT: [f32; 256] = {
    let mut t = [0.0f32; 256];
    let mut i = 0usize;
    while i < 256 {
        if i != 0 && i != 0x7F {
            let exp = ((i >> 3) & 15) as u32;
            let man = (i & 7) as u32;
            t[i] = if exp == 0 {
                man as f32 * (1.0 / 1024.0)
            } else {
                f32::from_bits(((exp + 119) << 23) | (man << 20))
            };
        }
        i += 1;
    }
    t
};

/// An unsigned E4M3 scale byte, halved (llama.cpp `ggml_ue4m3_to_fp32`); 0x7F (NaN) reads as 0.
#[inline]
pub fn ue4m3(x: u8) -> f32 { UE4M3_LUT[x as usize] }

/// NVFP4: 64 values as four sub-blocks of 16, each with a UE4M3 scale `d[s]`, then 32 bytes of E2M1 codes.
fn nvfp4(b: &[u8], y: &mut [f32]) {
    for s in 0..4 {
        let d = ue4m3(b[s]);
        for j in 0..8 {
            let q = b[4 + 8 * s + j];
            y[16 * s + j] = d * KVALUES_FP4[(q & 0xF) as usize] as f32;
            y[16 * s + j + 8] = d * KVALUES_FP4[(q >> 4) as usize] as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn tq_state_in_matches() {
        let mut r = 0x12345678u32;
        for n in [24, 28, 32] {
            for _ in 0..200 {
                let qs: Vec<u8> = (0..n).map(|_| { r ^= r << 13; r ^= r >> 17; r ^= r << 5; r as u8 }).collect();
                let buf = super::tq_stream(&qs);
                for t in 0..32 { assert_eq!(super::tq_state_in(&buf, &qs, t), super::tq_state(&qs, t)); }
            }
        }
    }

    use super::*;

    #[test]
    fn nvfp4_dequant() {
        // Scale bytes: 0x38 = e 7, m 0 (1.0, halved 0.5); 0x01 = subnormal 2^-9 (halved 2^-10); 0x7F = NaN (0); 0x41 = e 8, m 1 (2.25, halved 1.125).
        assert_eq!(ue4m3(0x38), 0.5);
        assert_eq!(ue4m3(0x01), 2f32.powi(-10));
        assert_eq!(ue4m3(0x7F), 0.0);
        assert_eq!(ue4m3(0x41), 1.125);
        let mut b = [0u8; 36];
        b[..4].copy_from_slice(&[0x38, 0x41, 0x7F, 0x38]);
        for (i, q) in b[4..].iter_mut().enumerate() { *q = (i as u8 * 7) ^ 0x5A; }
        let mut y = [0f32; 64];
        dequant_row(NVFP4, &b, &mut y);
        let d = [0.5, 1.125, 0.0, 0.5];
        for i in 0..64 {
            let (s, j) = (i / 16, i % 16);
            let q = b[4 + 8 * s + j % 8];
            let c = if j < 8 { q & 15 } else { q >> 4 };
            assert_eq!(y[i], d[s] * KVALUES_FP4[c as usize] as f32, "value {i}");
        }
    }
}
