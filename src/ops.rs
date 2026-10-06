//! Scalar f32 reference math.

pub fn rms_norm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let s = 1.0 / (ms + eps).sqrt();
    for ((o, v), w) in out.iter_mut().zip(x).zip(w) { *o = v * s * w; }
}

pub fn l2_norm(x: &mut [f32], eps: f32) {
    let s = 1.0 / x.iter().map(|v| v * v).sum::<f32>().sqrt().max(eps);
    for v in x { *v *= s; }
}

pub fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
pub fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
pub fn softplus(x: f32) -> f32 { if x > 20.0 { x } else { x.exp().ln_1p() } }

pub fn softmax(x: &mut [f32]) {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0;
    for v in x.iter_mut() { *v = (*v - m).exp(); s += *v; }
    for v in x { *v /= s; }
}

/// NEOX-style rotary embedding on the first `n_rot` dims: pairs (i, i + n_rot/2).
pub fn rope_neox(x: &mut [f32], n_rot: usize, pos: usize, base: f32) {
    let half = n_rot / 2;
    for i in 0..half {
        let theta = pos as f64 * (base as f64).powf(-2.0 * i as f64 / n_rot as f64);
        let (s, c) = (theta.sin() as f32, theta.cos() as f32);
        let (a, b) = (x[i], x[i + half]);
        x[i] = a * c - b * s;
        x[i + half] = a * s + b * c;
    }
}

/// Indices of the `k` largest values, descending; ties go to the lower index.
pub fn top_k(p: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..p.len()).collect();
    idx.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
    idx.truncate(k);
    idx
}

pub fn argmax(x: &[f32]) -> usize { top_k(x, 1)[0] }
