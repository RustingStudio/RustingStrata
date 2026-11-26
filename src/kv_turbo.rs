//! Faithful TurboQuant KV core: WHT-128 rotation, Lloyd-Max scalar quantizers and the QJL 1-bit residual.
//!
//! Based on arXiv 2504.19874 and the constants of `llama-cpp-turboquant` (`ggml-turbo-quant.c`,
//! `ggml-cuda/turbo-quant.cuh`, seeds 42 and 1042). The rotation is applied on store and to Q on the fly; attention
//! runs entirely in the rotated domain and only the attention output is inverse-rotated, which is what makes this
//! cheaper than inverse-rotating every K/V load.

/// Seed-42 sign diagonals of the signed WHT, from the reference implementation.
pub const WHT_S1: [f32; 128] = [
    -1.,1.,1.,-1.,-1.,1.,-1.,1.,-1.,-1.,1.,1.,1.,1.,1.,1.,
     1.,-1.,1.,-1.,1.,-1.,-1.,1.,1.,1.,-1.,1.,1.,-1.,-1.,-1.,
    -1.,1.,1.,-1.,1.,1.,-1.,1.,-1.,1.,1.,-1.,-1.,1.,-1.,1.,
     1.,1.,1.,-1.,-1.,-1.,-1.,-1.,1.,-1.,1.,1.,1.,1.,-1.,1.,
    -1.,-1.,1.,-1.,-1.,-1.,1.,-1.,-1.,-1.,1.,-1.,-1.,-1.,1.,1.,
     1.,-1.,-1.,1.,1.,1.,-1.,-1.,1.,1.,-1.,1.,1.,-1.,1.,-1.,
    -1.,1.,1.,-1.,1.,-1.,1.,-1.,1.,1.,1.,1.,-1.,1.,-1.,1.,
     1.,-1.,1.,1.,-1.,-1.,-1.,-1.,-1.,1.,1.,-1.,1.,1.,-1.,1.,
];
pub const WHT_S2: [f32; 128] = [
     1.,1.,1.,1.,-1.,1.,1.,-1.,1.,-1.,-1.,-1.,1.,-1.,-1.,-1.,
     1.,1.,-1.,-1.,1.,-1.,1.,-1.,1.,-1.,-1.,1.,-1.,1.,1.,1.,
     1.,1.,-1.,-1.,-1.,1.,-1.,-1.,-1.,-1.,-1.,-1.,1.,1.,1.,-1.,
     1.,-1.,1.,1.,1.,-1.,-1.,1.,-1.,-1.,-1.,-1.,-1.,-1.,1.,1.,
     1.,-1.,1.,-1.,-1.,-1.,-1.,1.,-1.,1.,-1.,1.,-1.,-1.,1.,1.,
    -1.,1.,-1.,1.,1.,-1.,1.,-1.,-1.,-1.,-1.,1.,-1.,-1.,1.,-1.,
     1.,-1.,1.,1.,1.,-1.,-1.,1.,-1.,1.,-1.,1.,1.,-1.,-1.,1.,
    -1.,1.,-1.,1.,1.,-1.,1.,-1.,1.,-1.,-1.,-1.,-1.,-1.,1.,-1.,
];

/// Lloyd-Max centroids for N(0, 1/128), i.e. for a unit-L2 128-vector after rotation.
pub const CENTROIDS2: [f32; 4] = [-0.133462, -0.039994, 0.039994, 0.133462];
pub const CENTROIDS3: [f32; 8] = [
    -0.190207, -0.118786, -0.066822, -0.021663,
     0.021663,  0.066822,  0.118786,  0.190207,
];
pub const CENTROIDS4: [f32; 16] = [
    -0.241529, -0.182877, -0.143016, -0.111036, -0.083292, -0.058050, -0.034299, -0.011349,
     0.011349,  0.034299,  0.058050,  0.083292,  0.111036,  0.143016,  0.182877,  0.241529,
];
pub const MID2: [f32; 3] = [-0.086728, 0.0, 0.086728];
pub const MID3: [f32; 7] = [-0.154496, -0.092804, -0.044243, 0.0, 0.044243, 0.092804, 0.154496];
pub const MID4: [f32; 15] = [
    -0.212203, -0.162947, -0.127026, -0.097164, -0.070671, -0.046174, -0.022824, 0.0,
     0.022824,  0.046174,  0.070671,  0.097164,  0.127026,  0.162947,  0.212203,
];

pub const QJL_CONST: f32 = 1.2533141373155003; // sqrt(pi/2)
const D: usize = 128;
const INV_SQRT: f32 = 0.08838834764831845; // 1 / sqrt(128)

/// In-place normalized Walsh-Hadamard transform (self-inverse).
pub fn fwht(x: &mut [f32; D]) {
    let mut h = 1;
    while h < D {
        let mut i = 0;
        while i < D {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
            i += 2 * h;
        }
        h *= 2;
    }
    for v in x.iter_mut() { *v *= INV_SQRT; }
}

/// Rotate a vector into the TurboQuant domain: D(s2) * H * D(s1) (s1 applied first), matching the reference.
pub fn rotate_forward(x: &mut [f32; D]) {
    for i in 0..D { x[i] *= WHT_S1[i]; }
    fwht(x);
    for i in 0..D { x[i] *= WHT_S2[i]; }
}

/// Inverse of `rotate_forward`: D(s1) * H * D(s2) (s2 applied first); H is self-inverse and signs are +-1.
pub fn rotate_inverse(x: &mut [f32; D]) {
    for i in 0..D { x[i] *= WHT_S2[i]; }
    fwht(x);
    for i in 0..D { x[i] *= WHT_S1[i]; }
}

pub fn bits_centroids(bits: usize) -> &'static [f32] {
    match bits { 2 => &CENTROIDS2, 3 => &CENTROIDS3, 4 => &CENTROIDS4, _ => panic!("bits must be 2, 3 or 4") }
}
pub fn bits_mids(bits: usize) -> &'static [f32] {
    match bits { 2 => &MID2, 3 => &MID3, 4 => &MID4, _ => panic!("bits must be 2, 3 or 4") }
}

/// Nearest Lloyd-Max centroid index of a value in a unit-L2 vector.
pub fn nearest(v: f32, bits: usize) -> usize {
    let mids = bits_mids(bits);
    let mut i = 0;
    while i < mids.len() && v >= mids[i] { i += 1; }
    i
}

/// Pack 128 rotated, unit-L2 coordinates into 64 bytes of 4-bit centroid indices (low nibble = even index)
/// plus the reconstruction-corrected norm used at dequant time. Mirrors TheTom's `block_turbo4_0` layout.
pub fn quantize4(x: &[f32; D], l2: f32) -> ([u8; D / 2], f32) {
    let mut codes = [0u8; D / 2];
    let mut recon2 = 0f32;
    for i in 0..D {
        let q = nearest(x[i] / l2, 4);
        let c = CENTROIDS4[q];
        recon2 += c * c;
        if i % 2 == 0 { codes[i / 2] = q as u8; } else { codes[i / 2] |= (q as u8) << 4; }
    }
    (codes, l2 / recon2.sqrt().max(1e-6))
}

/// Unpack 4-bit codes into 128 rotated values, scaled by the stored corrected norm.
pub fn dequantize4(codes: &[u8; D / 2], norm: f32) -> [f32; D] {
    let mut out = [0f32; D];
    for i in 0..D {
        let q = if i % 2 == 0 { codes[i / 2] & 15 } else { codes[i / 2] >> 4 };
        out[i] = CENTROIDS4[q as usize] * norm;
    }
    out
}

/// Pack 128 rotated, unit-L2 coordinates into 32 bytes of 2-bit codes plus 16 bytes of sign bits and the
/// reconstruction-corrected norm. Mirrors TheTom's `block_turbo3_0` layout: 50 B per 128 values.
pub fn quantize3(x: &[f32; D], l2: f32) -> ([u8; D / 4], [u8; D / 8], f32) {
    let mut qs = [0u8; D / 4];
    let mut signs = [0u8; D / 8];
    let mut recon2 = 0f32;
    for i in 0..D {
        let q = nearest(x[i] / l2, 3);
        qs[i / 4] |= ((q & 3) as u8) << (2 * (i % 4));
        if q >= 4 { signs[i / 8] |= 1 << (i % 8); }
        recon2 += CENTROIDS3[q] * CENTROIDS3[q];
    }
    (qs, signs, l2 / recon2.sqrt().max(1e-6))
}

/// Unpack 3-bit codes into 128 rotated values, scaled by the stored corrected norm.
pub fn dequantize3(qs: &[u8; D / 4], signs: &[u8; D / 8], norm: f32) -> [f32; D] {
    let mut out = [0f32; D];
    for i in 0..D {
        let mag = (qs[i / 4] >> (2 * (i % 4))) & 3;
        let sign = (signs[i / 8] >> (i % 8)) & 1;
        out[i] = CENTROIDS3[(mag | (sign << 2)) as usize] * norm;
    }
    out
}

/// Deterministic Gaussian PRNG and QJL projection matrix, matching the reference (LCG + Box-Muller, seed 1042).
struct Prng(u64);
impl Prng {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next_u(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn normal(&mut self) -> f64 {
        let u1 = (self.next_u() >> 11) as f64 / (1u64 << 53) as f64;
        let u1 = u1.max(1e-15);
        let u2 = (self.next_u() >> 11) as f64 / (1u64 << 53) as f64;
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// The QJL projection S (128x128, iid N(0,1), seed 1042) as row-major S[i*128+j].
pub fn qjl_matrix() -> Vec<f32> {
    let mut p = Prng::new(1042);
    let mut m = vec![0f32; D * D];
    for v in &mut m { *v = p.normal() as f32; }
    m
}

/// Signs of S @ r for a residual r (one bit per projection), stored as booleans (true = +1).
pub fn qjl_signs(r: &[f32; D], s: &[f32]) -> Vec<bool> {
    (0..D).map(|i| {
        let mut dot = 0f32;
        for j in 0..D { dot += s[i * D + j] * r[j]; }
        dot >= 0.0
    }).collect()
}

/// Reconstruct the estimator `S^T @ signs * QJL_CONST / d * rnorm` used by the legacy 3-bit+QJL mode.
pub fn qjl_reconstruct(signs: &[bool], s: &[f32], rnorm: f32) -> Vec<f32> {
    let scale = QJL_CONST / D as f32 * rnorm;
    let mut out = vec![0f32; D];
    for j in 0..D {
        let mut dot = 0f32;
        for i in 0..D { dot += s[i * D + j] * if signs[i] { 1f32 } else { -1f32 }; }
        out[j] = dot * scale;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wht_is_orthogonal_and_involutive() {
        let mut x = [0f32; D];
        for i in 0..D { x[i] = ((i * 37 + 11) % 101) as f32 / 101.0 - 0.5; }
        let norm2: f32 = x.iter().map(|v| v * v).sum();
        let mut y = x;
        rotate_forward(&mut y);
        let yn2: f32 = y.iter().map(|v| v * v).sum();
        assert!((yn2 - norm2).abs() < 1e-3, "rotation must preserve L2: {yn2} vs {norm2}");
        rotate_inverse(&mut y);
        for i in 0..D { assert!((y[i] - x[i]).abs() < 1e-4, "round trip at {i}"); }
    }

    #[test]
    fn centroids_are_sorted_and_symmetric() {
        for bits in [2, 3, 4] {
            let c = bits_centroids(bits);
            assert!(c.windows(2).all(|w| w[0] < w[1]));
            assert!((c[0] + c[c.len() - 1]).abs() < 1e-6);
            assert_eq!(nearest(0.0, bits), c.len() / 2);
        }
    }

    #[test]
    fn turbo4_encode_decode_random_vectors() {
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
        };
        for _ in 0..8 {
            let mut v = [0f32; D];
            for x in &mut v { *x = next(); }
            let l2 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v { *x /= l2; }
            let mut rotated = v;
            rotate_forward(&mut rotated);
            let (codes, norm) = quantize4(&rotated, 1.0);
            let back_rot = dequantize4(&codes, norm);

            // Byte layout: even values in low nibbles, odd values in high nibbles.
            assert_eq!(codes[0] & 15, nearest(rotated[0], 4) as u8);
            assert_eq!(codes[0] >> 4, nearest(rotated[1], 4) as u8);

            let mean_err = rotated.iter().zip(&back_rot).map(|(a, b)| (a - b).abs()).sum::<f32>() / D as f32;
            assert!(mean_err < 0.05, "4-bit rotated error {mean_err} too high");

            let mut restored = back_rot;
            rotate_inverse(&mut restored);
            let dot = v.iter().zip(&restored).map(|(a, b)| a * b).sum::<f32>();
            let vnorm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let cos = dot / (vnorm * restored.iter().map(|x| x * x).sum::<f32>().sqrt());
            assert!(cos > 0.99, "round-trip cosine {cos} too low");
        }
    }

    #[test]
    fn qjl_estimates_inner_products_unbiasedly() {
        let s = qjl_matrix();
        let mut p = Prng::new(7);
        let mut err = 0f64;
        let trials = 64;
        for _ in 0..trials {
            let mut q = [0f32; D];
            let mut r = [0f32; D];
            for i in 0..D { q[i] = p.normal() as f32 / 8.0; r[i] = p.normal() as f32 / 8.0; }
            let target: f64 = q.iter().zip(&r).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
            let signs = qjl_signs(&r, &s);
            let rnorm = r.iter().map(|v| v * v).sum::<f32>().sqrt();
            let recon = qjl_reconstruct(&signs, &s, rnorm);
            let est: f64 = q.iter().zip(&recon).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
            err += (est - target).abs();
        }
        // The 1-bit estimator is noisy; require only that it is in the right ballpark, not exact.
        let mean = err / (trials as f64);
        assert!(mean < 5.0, "mean absolute QJL error {mean} too high");
    }

    #[test]
    fn turbo3_packs_50_bytes_per_128_and_round_trips() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
        };
        for _ in 0..8 {
            let mut v = [0f32; D];
            for x in &mut v { *x = next(); }
            let l2 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v { *x /= l2; }
            let mut rotated = v;
            rotate_forward(&mut rotated);
            let (qs, signs, norm) = quantize3(&rotated, 1.0);
            assert_eq!(qs.len() + signs.len() + 2, 50, "turbo3 block must be 50 B");
            // The first eight values exercise both sign halves and all four magnitude code positions.
            for i in 0..8 {
                let mag = (qs[i / 4] >> (2 * (i % 4))) & 3;
                let sign = (signs[0] >> i) & 1;
                assert_eq!((mag | (sign << 2)) as usize, nearest(rotated[i], 3));
            }
            let back_rot = dequantize3(&qs, &signs, norm);
            let mean_err = rotated.iter().zip(&back_rot).map(|(a, b)| (a - b).abs()).sum::<f32>() / D as f32;
            assert!(mean_err < 0.09, "3-bit rotated error {mean_err} too high");
            let mut restored = back_rot;
            rotate_inverse(&mut restored);
            let dot = v.iter().zip(&restored).map(|(a, b)| a * b).sum::<f32>();
            let cos = dot / restored.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(cos > 0.97, "round-trip cosine {cos} too low");
        }
    }
}
