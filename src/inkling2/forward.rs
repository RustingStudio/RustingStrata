//! Inkling 2 forward, mirroring llama.cpp `src/models/inkling2.cpp` and `inkling2-loop.cpp` (trace names are its
//! callback names, so `STRATA_DS_DUMP` output lines up with `llama-eval-callback`).
//! A slot is one layer run of a token: the core layers run once per pass with the same weights, and every pass keeps
//! its own attention rows and convolution state.
//! The prompt runs layer by layer over a chunk of tokens: one GPU GEMM per dense matrix and per routed expert.
//! ponytail: the attention scores run on the CPU. Move them to the GPU when a long prompt waits on them.
use super::Inkling2Config;
use crate::deepseek::mat::{Mat, matvec_many, matvec_many_opt, matvec_rows, matvec_rows_many, matvec_rows_sets, situ_chunk, situ_experts};
use crate::deepseek::ops::{rms_norm, rms_norm_w};
use crate::gguf::Gguf;
use crate::quant::GgmlType;
use crate::weights::dot;
use anyhow::{Context, Result, ensure};
use half::{f16, slice::HalfFloatSliceExt};
use rayon::prelude::*;

mod device;

/// Epsilon of the norms without a weight (depth mix, loop controllers).
const MIX_EPS: f32 = 1e-6;

struct Layer<'a> {
    swa: bool,
    /// Depth-mix queries of the attention inputs: one (sliding window) or three (Q, K, V). Layer 0 has none.
    res_query: Option<Vec<f32>>,
    /// Input norms: one (sliding window) or three (Q, K, V).
    norms: Vec<Vec<f32>>,
    wq: Mat<'a>,
    wk: Mat<'a>,
    wv: Mat<'a>,
    wo: Mat<'a>,
    gate: Mat<'a>,
    rel: Mat<'a>,
    sink: Mat<'a>,
    /// Full-attention layers: projections to the input-dependent taps of the K and V convolutions.
    conv_proj: Option<[Mat<'a>; 2]>,
    q_clip: Vec<f32>,
    /// The relative bias table as `[channel][rel_extent - 1 - distance]`: the rows a token sees, oldest first, are
    /// then a run of each channel.
    rel_bias: Vec<f32>,
    rel_clip: Vec<f32>,
    /// Convolution weights `[channel][lag]` of K, V, the attention output and the FFN output.
    convs: [Vec<f32>; 4],
    ffn_query: Vec<f32>,
    ffn_norm: Vec<f32>,
    gate_inp: Vec<f32>,
    bias: Vec<f32>,
    gate_up: Mat<'a>,
    down: Mat<'a>,
    /// Per-expert scales of the gate_up and down outputs (`ffn_*_exps.scale`, e.g. NVFP4 files); ones without them.
    gu_scale: Vec<f32>,
    down_scale: Vec<f32>,
}

/// Cross-pass controllers of one core layer, in the order Q, K, V, FFN.
struct Loop {
    msg_query: Vec<f32>,
    /// The controller matrices of Q, K, V and the feed-forward input, as f16 (`dot_h`). `gain` and `forget` are
    /// `[rank][channel]`, the transpose of the file: a column of them is one `axpy_h`.
    down: Vec<Vec<f16>>,
    gain: Vec<Vec<f16>>,
    forget: Vec<Vec<f16>>,
}

#[derive(Default)]
struct Slot {
    /// Key and value rows of the visible positions per key head, the newest last. f16, as llama.cpp keeps them: a
    /// token reads all of them, so the scores run at the speed of the memory.
    k: Vec<Vec<f16>>,
    v: Vec<Vec<f16>>,
    /// The last `n_conv - 1` inputs of the four convolutions, the oldest first.
    hist: [Vec<f32>; 4],
}

pub struct Inkling2Model<'a> {
    pub cfg: Inkling2Config,
    tok_embd: Mat<'a>,
    tok_norm: Vec<f32>,
    layers: Vec<Layer<'a>>,
    loops: Vec<Loop>,
    out_query: Vec<f32>,
    out_norm: Vec<f32>,
    output: Mat<'a>,
    slots: Vec<Slot>,
    /// Share of a layer's VRAM-cache misses the CPU computes; it follows the two sides' times.
    cpu_share: std::cell::Cell<f32>,
    /// The same share for a chunk of tokens, where an expert runs once for all its tokens.
    chunk_share: std::cell::Cell<f32>,
    pub toks: Vec<u32>,
    pub trace: bool,
    /// Seconds per part of a token, printed with `STRATA_DS_PROF`; see `PROF`.
    prof: [std::cell::Cell<f64>; PROF.len()],
    /// The state of `step` in VRAM, made by the first `step`. `no_device`: it cannot be made, `run` does every token.
    device: Option<Box<device::Device>>,
    no_device: bool,
}

/// A token of `device` counts all the device work before the router as "mix": the queue runs while the host waits
/// for the router's input.
const PROF: [&str; 8] = ["mix", "proj", "scores", "out-proj", "route", "experts-gpu", "experts-cpu", "logits"];

/// A tensor read from the mapped file only: never uploaded.
fn host_mat<'a>(g: &'a Gguf, name: &str) -> Result<Mat<'a>> {
    let t = g.tensors.get(name).with_context(|| format!("missing tensor {name}"))?;
    let rows = t.dims[1..].iter().product::<u64>() as usize;
    Ok(Mat { data: g.bytes(t), ty: GgmlType::from_u32(t.ty)?, rows, cols: t.dims[0] as usize, cpu: false, gpu: None })
}

/// A small tensor as f32, row-major (`dims[0]` is the row width).
fn f32s(g: &Gguf, name: &str) -> Result<Vec<f32>> {
    let m = host_mat(g, name)?;
    let mut v = vec![0f32; m.rows * m.cols];
    for (r, row) in v.chunks_exact_mut(m.cols).enumerate() { m.row_into(r, row); }
    Ok(v)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}

/// e^x to 2 ulp (cephes `expf`) with no branch and no table, so a loop of it compiles to vector code. libm's `exp`
/// and `tanh` are calls: about 4 ms per token in the activation and the loop controllers.
#[inline(always)]
fn exp(x: f32) -> f32 {
    let x = x.clamp(-87.0, 88.0);
    let k = (x * std::f32::consts::LOG2_E).round_ties_even();
    let r = k.mul_add(2.121_944_4e-4, k.mul_add(-0.693_359_375, x));
    let mut p = 1.987_569_1e-4f32;
    for c in [1.398_2e-3, 8.333_452e-3, 4.166_579_6e-2, 1.666_666_5e-1, 0.5] { p = p.mul_add(r, c); }
    ((p * r).mul_add(r, r) + 1.0) * f32::from_bits(((k as i32 + 127) as u32) << 23)
}

/// tanh from `exp`: an absolute error of 1.2e-7.
#[inline(always)]
fn tanh(x: f32) -> f32 {
    let t = exp(2.0 * x.clamp(-10.0, 10.0));
    (t - 1.0) / (t + 1.0)
}

/// Scale each row of width `w` down to an L2 norm of at most `thr`, after the optional per-channel scale.
fn clip(x: &mut [f32], scale: Option<&[f32]>, thr: f32, w: usize) {
    if let Some(s) = scale {
        for (x, s) in x.iter_mut().zip(s) { *x *= s; }
    }
    for row in x.chunks_exact_mut(w) {
        let f = (thr / (dot(row, row).sqrt() + 1e-6)).clamp(0.0, 1.0);
        for v in row { *v *= f; }
    }
}

/// Attention over the depth: for each query row, the softmax-weighted sum of the normalised states. The states are
/// the earlier block outputs (the newest first, as llama.cpp stacks them) and the current stream.
fn mix(blocks: &[Vec<f32>], cur: &[f32], query: &[f32]) -> Vec<f32> {
    let n = cur.len();
    let states: Vec<Vec<f32>> = blocks.iter().rev().map(|b| b.as_slice()).chain([cur]).map(|s| {
        let mut o = vec![0f32; n];
        rms_norm(s, MIX_EPS, &mut o);
        o
    }).collect();
    let scale = 1.0 / (n as f32).sqrt();
    let mut out = vec![0f32; query.len()];
    for (q, o) in query.chunks_exact(n).zip(out.chunks_exact_mut(n)) {
        let mut sc: Vec<f32> = states.iter().map(|s| dot(s, q) * scale).collect();
        let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for s in sc.iter_mut() { *s = (*s - mx).exp(); }
        let z: f32 = sc.iter().sum();
        for (s, p) in states.iter().zip(&sc) {
            for (o, s) in o.iter_mut().zip(s) { *o += p / z * s; }
        }
    }
    out
}

/// Causal short convolution with a residual. `hist` holds the last `n_conv - 1` inputs and is advanced by `cur`.
/// `taps` (`[head][lag]`, lag 0 is the current input) adds input-dependent taps shared by the channels of a head.
fn convolve(cur: &[f32], w: &[f32], taps: Option<&[f32]>, hist: &mut Vec<f32>, n_conv: usize) -> Vec<f32> {
    let n = cur.len();
    hist.resize((n_conv - 1) * n, 0.0);
    let x = |k: usize, c: usize| if k + 1 < n_conv { hist[k * n + c] } else { cur[c] };
    let per = taps.map_or(n, |t| n / (t.len() / n_conv));
    let s = 1.0 / ((if taps.is_some() { 2 * n_conv + 1 } else { n_conv + 1 }) as f32).sqrt();
    let out = (0..n).map(|c| {
        let mut o: f32 = (0..n_conv).map(|k| x(k, c) * w[c * n_conv + k]).sum();
        if let Some(t) = taps {
            for lag in 0..n_conv { o += x(n_conv - 1 - lag, c) * t[c / per * n_conv + lag]; }
        }
        (cur[c] + o) * s
    }).collect();
    hist.copy_within(n.., 0);
    hist[(n_conv - 2) * n..].copy_from_slice(cur);
    out
}

/// `convolve` of a chunk of tokens at once: token t reads the inputs of the tokens before it, those of the chunk
/// and (in `hist`) those before the chunk.
fn convolve_rows(rows: &[&[f32]], w: &[f32], taps: Option<&[&[f32]]>, hist: &mut Vec<f32>, n_conv: usize) -> Vec<Vec<f32>> {
    if let [row] = rows { return vec![convolve(row, w, taps.map(|t| t[0]), hist, n_conv)]; }
    let n = rows[0].len();
    hist.resize((n_conv - 1) * n, 0.0);
    let mut all = std::mem::take(hist);
    for r in rows { all.extend_from_slice(r); }
    let out = (0..rows.len()).into_par_iter().map(|t| {
        convolve(rows[t], w, taps.map(|x| x[t]), &mut all[t * n..(t + n_conv - 1) * n].to_vec(), n_conv)
    }).collect();
    *hist = all[rows.len() * n..].to_vec();
    out
}

/// Sum of the eight values, as a tree.
#[cfg(all(target_feature = "f16c", target_feature = "fma"))]
#[inline(always)]
fn hsum(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    // SAFETY: the features are those of the build.
    unsafe {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps(s, s, 1)))
    }
}

/// `p[j * n + row]` += `scale` times the dot of f16 row `row` of `w` (`n` rows) with row j of `x`.
/// One to three rows of `x` take the same sums in the same order, so a token gets the same bits alone or in a chunk.
fn dots_h(p: &mut [f32], scale: f32, w: &[f16], x: &[f32], n: usize) {
    let hd = w.len() / n;
    #[cfg(all(target_feature = "f16c", target_feature = "fma"))]
    if hd % 32 == 0 && x.len() % hd == 0 && p.len() == x.len() / hd * n {
        match x.len() / hd {
            1 => return dots_k::<1>(p, scale, w, x, n),
            2 => return dots_k::<2>(p, scale, w, x, n),
            3 => return dots_k::<3>(p, scale, w, x, n),
            _ => {}
        }
    }
    for (a, w) in w.chunks_exact(hd).enumerate() {
        for (j, x) in x.chunks_exact(hd).enumerate() { p[j * n + a] += scale * dot_h(w, x); }
    }
}

/// `dots_h` for `K` rows of `x`: a weight is widened once for `K` products, and the sums stay in registers.
#[cfg(all(target_feature = "f16c", target_feature = "fma"))]
fn dots_k<const K: usize>(p: &mut [f32], scale: f32, w: &[f16], x: &[f32], n: usize) {
    use std::arch::x86_64::*;
    let hd = w.len() / n;
    assert!(x.len() == K * hd && p.len() == K * n && hd % 32 == 0);
    for a in 0..n {
        unsafe {
            let mut acc = [[_mm256_setzero_ps(); 4]; K];
            for c in (0..hd).step_by(32) {
                for b in 0..4 {
                    let k = _mm256_cvtph_ps(_mm_loadu_si128(w.as_ptr().add(a * hd + c + 8 * b).cast()));
                    for j in 0..K { acc[j][b] = _mm256_fmadd_ps(k, _mm256_loadu_ps(x.as_ptr().add(j * hd + c + 8 * b)), acc[j][b]); }
                }
            }
            for j in 0..K { p[j * n + a] += scale * hsum(_mm256_add_ps(_mm256_add_ps(acc[j][0], acc[j][1]), _mm256_add_ps(acc[j][2], acc[j][3]))); }
        }
    }
}

/// Dot of f16 weights and f32 values. The loop controllers are 113 MB of F32 in the file and every token reads them
/// twice, at the speed of the memory: as f16 they take half the time (their values stay within 5e-4 relative).
fn dot_h(w: &[f16], x: &[f32]) -> f32 {
    #[cfg(all(target_feature = "f16c", target_feature = "fma"))]
    {
        use std::arch::x86_64::*;
        let (cw, cx) = (w.chunks_exact(32), x.chunks_exact(32));
        let tail: f32 = cw.remainder().iter().zip(cx.remainder()).map(|(w, x)| w.to_f32() * x).sum();
        // Four sums: one would wait on its own multiply-add.
        let mut acc = [unsafe { _mm256_setzero_ps() }; 4];
        for (w, x) in cw.zip(cx) {
            for i in 0..4 {
                // SAFETY: both chunks hold 32 values, and the features are those of the build.
                acc[i] = unsafe { _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(w.as_ptr().add(8 * i).cast())), _mm256_loadu_ps(x.as_ptr().add(8 * i)), acc[i]) };
            }
        }
        let s: [f32; 8] = unsafe { std::mem::transmute(_mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]))) };
        s.iter().sum::<f32>() + tail
    }
    #[cfg(not(all(target_feature = "f16c", target_feature = "fma")))]
    w.iter().zip(x).map(|(w, x)| w.to_f32() * x).sum()
}

/// o += p * w, for f16 w.
fn axpy_h(o: &mut [f32], p: f32, w: &[f16]) {
    #[cfg(all(target_feature = "f16c", target_feature = "fma"))]
    {
        use std::arch::x86_64::*;
        let (co, cw) = (o.chunks_exact_mut(8), w.chunks_exact(8));
        let tail = cw.remainder();
        for (o, w) in co.zip(cw) {
            // SAFETY: both chunks hold 8 values, and the features are those of the build.
            unsafe { _mm256_storeu_ps(o.as_mut_ptr(), _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(w.as_ptr().cast())), _mm256_set1_ps(p), _mm256_loadu_ps(o.as_ptr()))) };
        }
        let at = o.len() - tail.len();
        for (o, w) in o[at..].iter_mut().zip(tail) { *o += p * w.to_f32(); }
    }
    #[cfg(not(all(target_feature = "f16c", target_feature = "fma")))]
    for (o, w) in o.iter_mut().zip(w) { *o += p * w.to_f32(); }
}

/// Row j of `o` += the sum over the `n` f16 rows of `w` of `p[j * n + row]` times that row. One to three rows of
/// `o` take the same sums in the same order.
fn mix_h(o: &mut [f32], p: &[f32], w: &[f16], n: usize) {
    let hd = w.len() / n;
    #[cfg(all(target_feature = "f16c", target_feature = "fma"))]
    if hd % 32 == 0 && o.len() % hd == 0 && p.len() == o.len() / hd * n {
        match o.len() / hd {
            1 => return mix_k::<1>(o, p, w, n),
            2 => return mix_k::<2>(o, p, w, n),
            3 => return mix_k::<3>(o, p, w, n),
            _ => {}
        }
    }
    for (a, w) in w.chunks_exact(hd).enumerate() {
        for (j, o) in o.chunks_exact_mut(hd).enumerate() { axpy_h(o, p[j * n + a], w); }
    }
}

/// `mix_h` for `K` rows of `o`, 32 columns at a time in registers.
#[cfg(all(target_feature = "f16c", target_feature = "fma"))]
fn mix_k<const K: usize>(o: &mut [f32], p: &[f32], w: &[f16], n: usize) {
    use std::arch::x86_64::*;
    let hd = w.len() / n;
    assert!(o.len() == K * hd && p.len() == K * n && hd % 32 == 0);
    for c in (0..hd).step_by(32) {
        unsafe {
            let mut acc = [[_mm256_setzero_ps(); 4]; K];
            for j in 0..K { for b in 0..4 { acc[j][b] = _mm256_loadu_ps(o.as_ptr().add(j * hd + c + 8 * b)); } }
            for a in 0..n {
                let pj: [__m256; K] = std::array::from_fn(|j| _mm256_broadcast_ss(&p[j * n + a]));
                for b in 0..4 {
                    let v = _mm256_cvtph_ps(_mm_loadu_si128(w.as_ptr().add(a * hd + c + 8 * b).cast()));
                    for j in 0..K { acc[j][b] = _mm256_fmadd_ps(v, pj[j], acc[j][b]); }
                }
            }
            for j in 0..K { for b in 0..4 { _mm256_storeu_ps(o.as_mut_ptr().add(j * hd + c + 8 * b), acc[j][b]); } }
        }
    }
}

/// Mix a cross-pass message into a looped layer input: `(1 - forget)*cur + gain*message`, both gated on the two.
/// Controller `c` on a run of tokens, `jobs` = (state, message). Three tokens share a pass over the matrices
/// (`dots_h`, `mix_h`): a weight is widened from f16 once for three products.
fn control(jobs: &[(&[f32], &[f32])], lp: &Loop, c: usize) -> Vec<Vec<f32>> {
    let n = jobs[0].0.len();
    let rank = lp.down[c].len() / (2 * n);
    jobs.par_chunks(3).flat_map_iter(|js| {
        let mut cat = vec![0f32; js.len() * 2 * n];
        for (cat, &(cur, msg)) in cat.chunks_exact_mut(2 * n).zip(js) {
            rms_norm(cur, MIX_EPS, &mut cat[..n]);
            rms_norm(msg, MIX_EPS, &mut cat[n..]);
        }
        let (mut gate, mut g, mut f) = (vec![0f32; js.len() * rank], vec![0f32; js.len() * n], vec![0f32; js.len() * n]);
        dots_h(&mut gate, 1.0, &lp.down[c], &cat, rank);
        mix_h(&mut g, &gate, &lp.gain[c], rank);
        mix_h(&mut f, &gate, &lp.forget[c], rank);
        js.iter().zip(g.chunks_exact(n).zip(f.chunks_exact(n))).map(|(&(cur, msg), (g, f))| {
            (0..n).map(|i| cur[i] - tanh(f[i]) * cur[i] + tanh(g[i]) * msg[i]).collect::<Vec<f32>>()
        }).collect::<Vec<_>>()
    }).collect()
}

/// situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb) on one output of the fused gate_up matrix
/// (the gate rows first), both times the expert's gate_up scale `s`.
fn situ(y: &[f32], s: f32, cfg: &Inkling2Config) -> Vec<f32> {
    let (nf, beta, lb) = (cfg.n_ff_exp, cfg.situ_beta, cfg.situ_linear_beta);
    let (g, u, mut o) = (&y[..nf], &y[nf..2 * nf], vec![0f32; nf]);
    for i in 0..nf {
        let (g, u) = (g[i] * s, u[i] * s);
        let up = if lb > 0.0 { lb * tanh(u / lb) } else { u };
        o[i] = beta * tanh(g / beta) * sigmoid(g) * up;
    }
    o
}

/// The experts of one token and their weights: sigmoid router, the selection bias only picks the experts. The
/// weight carries the expert's down scale.
fn route(l: &Layer, x: &[f32], cfg: &Inkling2Config) -> Vec<(usize, f32)> {
    let probs: Vec<f32> = l.gate_inp.chunks_exact(x.len()).map(|w| sigmoid(dot(w, x))).collect();
    let mut sel: Vec<usize> = (0..cfg.n_expert).collect();
    sel.sort_by(|&a, &b| (probs[b] + l.bias[b]).partial_cmp(&(probs[a] + l.bias[a])).unwrap().then(a.cmp(&b)));
    sel.truncate(cfg.n_used);
    let sum = sel.iter().map(|&e| probs[e]).sum::<f32>() + 1e-20;
    sel.into_iter().map(|e| (e, probs[e] / sum * cfg.exp_scale * l.down_scale[e])).collect()
}

/// Weighted sum of the listed experts' outputs, `(gate_up, down, weight, gate_up scale)` each. Experts that share a GPU run as two
/// batched round trips; otherwise each matvec runs where its matrix lives.
fn run_experts(list: &[(Mat, Mat, f32, f32)], x: &[f32], cfg: &Inkling2Config) -> Vec<f32> {
    let (nf, nl) = (cfg.n_ff_exp, x.len());
    let mut mix = vec![0f32; nl];
    let batched = (!list.is_empty() && list.iter().all(|e| e.0.gpu.is_some())).then(|| {
        let (gus, downs): (Vec<Mat>, Vec<Mat>) = list.iter().map(|e| (e.0, e.1)).unzip();
        let gs: Vec<f32> = list.iter().map(|e| e.3).collect();
        // One round trip with the activation on the device; two, with the activation here, on more than one GPU.
        situ_experts(&gus, &downs, &gs, x, cfg.situ_beta, cfg.situ_linear_beta).or_else(|| {
            let y = matvec_many_opt(&gus, x, 0)?;
            let a: Vec<f32> = y.chunks_exact(2 * nf).zip(&gs).flat_map(|(y, &s)| situ(y, s, cfg)).collect();
            matvec_many_opt(&downs, &a, nf)
        })
    }).flatten();
    match batched {
        Some(dy) => {
            for (e, dy) in list.iter().zip(dy.chunks_exact(nl)) {
                for (m, d) in mix.iter_mut().zip(dy) { *m += e.2 * d; }
            }
        }
        None => {
            let (mut y, mut dy) = (vec![0f32; 2 * nf], vec![0f32; nl]);
            for (gu, down, w, s) in list {
                gu.matvec(x, &mut y);
                down.matvec(&situ(&y, *s, cfg), &mut dy);
                for (m, d) in mix.iter_mut().zip(&dy) { *m += w * d; }
            }
        }
    }
    mix
}

impl<'a> Inkling2Model<'a> {
    pub fn new(g: &'a Gguf) -> Result<Self> {
        let cfg = Inkling2Config::from_gguf(g)?;
        let has = |n: &str| g.tensors.contains_key(n);
        let mut layers = vec![];
        for il in 0..cfg.n_layer {
            let p = |s: &str| format!("blk.{il}.{s}");
            let m = |s: &str| Mat::load(g, &p(s));
            let v = |s: &str| f32s(g, &p(s));
            let swa = cfg.swa[il];
            layers.push(Layer {
                swa,
                res_query: if has(&p("attn_res_query.weight")) { Some(v("attn_res_query.weight")?) } else { None },
                norms: if swa {
                    vec![v("attn_norm.weight")?]
                } else {
                    vec![v("attn_q_inp_norm.weight")?, v("attn_k_inp_norm.weight")?, v("attn_v_inp_norm.weight")?]
                },
                wq: m("attn_q.weight")?,
                wk: m("attn_k.weight")?,
                wv: m("attn_v.weight")?,
                wo: m("attn_output.weight")?,
                gate: m("attn_gate.weight")?,
                // Few rows: `Mat` would take the f32 reference dots on the CPU, which cost more than all of Q's
                // integer dots. `cpu` selects the integer dots; with a VRAM copy the flag changes nothing.
                rel: Mat { cpu: true, ..m("attn_rel.weight")? },
                sink: Mat { cpu: true, ..m("attn_sink_proj.weight")? },
                conv_proj: if swa { None } else { Some([m("attn_k_conv_proj.weight")?, m("attn_v_conv_proj.weight")?]) },
                q_clip: v("attn_q_clip.weight")?,
                rel_bias: {
                    let (b, nr, ext) = (v("attn_rel_bias.weight")?, cfg.n_rel, cfg.rel_extent);
                    ensure!(b.len() == nr * ext, "attn_rel_bias: {} values for {ext} distances of {nr}", b.len());
                    (0..nr * ext).map(|i| b[(ext - 1 - i % ext) * nr + i / ext]).collect()
                },
                rel_clip: v("attn_rel_clip.weight")?,
                convs: [v("attn_k_conv.weight")?, v("attn_v_conv.weight")?, v("attn_post_conv.weight")?, v("ffn_post_conv.weight")?],
                ffn_query: v("ffn_res_query.weight")?,
                ffn_norm: v("ffn_norm.weight")?,
                gate_inp: v("ffn_gate_inp.weight")?,
                bias: if has(&p("exp_probs_b.bias")) { v("exp_probs_b.bias")? } else { vec![0.0; cfg.n_expert] },
                gate_up: m("ffn_gate_up_exps.weight")?,
                down: m("ffn_down_exps.weight")?,
                gu_scale: if has(&p("ffn_gate_up_exps.scale")) { v("ffn_gate_up_exps.scale")? } else { vec![1.0; cfg.n_expert] },
                down_scale: if has(&p("ffn_down_exps.scale")) { v("ffn_down_exps.scale")? } else { vec![1.0; cfg.n_expert] },
            });
        }
        let mut loops = vec![];
        for il in cfg.loop_start..cfg.loop_end {
            let v = |s: &str| f32s(g, &format!("blk.{il}.loop_{s}.weight"));
            let ctrl = |kind: &str| ["q", "k", "v", "ffn"].iter().map(|c| {
                let mut w = v(&format!("{c}_ctrl_{kind}"))?;
                if kind != "down" {
                    let (n, rank) = (cfg.n_embd, w.len() / cfg.n_embd);
                    w = (0..w.len()).map(|i| w[i % n * rank + i / n]).collect();
                }
                let mut h = vec![f16::ZERO; w.len()];
                h.convert_from_f32_slice(&w);
                Ok(h)
            }).collect::<Result<Vec<_>>>();
            loops.push(Loop { msg_query: v("msg_query")?, down: ctrl("down")?, gain: ctrl("gain")?, forget: ctrl("forget")? });
        }
        Ok(Self {
            tok_embd: host_mat(g, "token_embd.weight")?,
            tok_norm: f32s(g, "token_embd_norm.weight")?,
            layers,
            loops,
            out_query: f32s(g, "output_res_query.weight")?,
            out_norm: f32s(g, "output_norm.weight")?,
            output: Mat::load(g, "output.weight")?,
            slots: (0..cfg.n_slot()).map(|_| Slot::default()).collect(),
            cpu_share: std::cell::Cell::new(0.5),
            chunk_share: std::cell::Cell::new(0.3),
            toks: vec![],
            trace: std::env::var_os("STRATA_DS_TRACE").is_some(),
            prof: Default::default(),
            device: None,
            no_device: false,
            cfg,
        })
    }

    pub fn reset(&mut self) {
        for s in &mut self.slots { *s = Slot::default(); }
        self.toks.clear();
        if let Some(d) = &mut self.device { d.forget(); }
    }

    /// Add the time since `t` to part `i` of the profile; returns the time now.
    fn lap(&self, i: usize, t: std::time::Instant) -> std::time::Instant {
        self.prof[i].set(self.prof[i].get() + t.elapsed().as_secs_f64());
        std::time::Instant::now()
    }

    fn tr(&self, name: &str, il: usize, v: &[f32]) {
        if let Ok(dir) = std::env::var("STRATA_DS_DUMP") {
            use std::io::Write;
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::OpenOptions::new().create(true).append(true).open(format!("{dir}/{name}-{il}.f32")).unwrap().write_all(&bytes).unwrap();
        }
        if self.trace {
            let n = v.len();
            let f = |i: usize| format!("{:12.4}", v[i]);
            eprintln!("TRACE {name}-{il} [{} {} {} ... {} {} {}] sum={:.6}", f(0), f(1), f(2), f(n - 3), f(n - 2), f(n - 1), v.iter().sum::<f32>());
        }
    }

    /// Attention of a chunk of tokens (the first at `pos`) in `slot` with the weights of layer `ip`. `ins` are the
    /// normed inputs of each token: one shared, or one each for Q, K and V.
    fn attention(&self, ip: usize, slot: &mut Slot, ins: &[Vec<Vec<f32>>], pos: usize) -> Vec<Vec<f32>> {
        let (cfg, l, nt) = (&self.cfg, &self.layers[ip], ins.len());
        let (hd, nh, nkv, nr, nc) = (cfg.head_dim, cfg.n_head, cfg.n_head_kv, cfg.n_rel, cfg.n_conv);
        let kw = hd * nkv;
        // Q, the relative-position code, the sink and the output gate read the Q input, K and V (and their tap
        // projections) their own. One token: every projection in one round trip. A chunk: one GEMM per matrix.
        let t = std::time::Instant::now();
        let mut ms = vec![(l.wq, 0), (l.rel, 0), (l.sink, 0), (l.gate, 0), (l.wk, 1), (l.wv, 2)];
        if let Some([kp, vp]) = l.conv_proj { ms.extend([(kp, 1), (vp, 2)]); }
        let inp = |t: usize, j: usize| &ins[t][j.min(ins[t].len() - 1)][..];
        let ys = if nt == 1 {
            let mats: Vec<Mat> = ms.iter().map(|m| m.0).collect();
            if ins[0].len() == 1 {
                matvec_many(&mats, inp(0, 0), 0)
            } else {
                matvec_many(&mats, &ms.iter().flat_map(|m| inp(0, m.1).iter().copied()).collect::<Vec<_>>(), cfg.n_embd)
            }
        } else {
            // The inputs go up once: one set when the three are the same.
            let sets = ins[0].len().min(3);
            let mut x = Vec::with_capacity(sets * nt * cfg.n_embd);
            for j in 0..sets { for t in 0..nt { x.extend_from_slice(inp(t, j)); } }
            matvec_rows_sets(&ms.iter().map(|&(m, j)| (m, j.min(sets - 1))).collect::<Vec<_>>(), &x, nt)
        };
        // Matrix after matrix, each a row a token.
        let at: Vec<usize> = ms.iter().scan(0, |a, m| { *a += nt * m.0.rows; Some(*a - nt * m.0.rows) }).collect();
        let row = |m: usize, t: usize| { let w = ms[m].0.rows; &ys[at[m] + t * w..at[m] + (t + 1) * w] };
        let t0 = self.lap(1, t);
        // A sliding-window layer sees the last `n_swa` positions; drop the older rows now and then.
        slot.k.resize(nkv, vec![]);
        slot.v.resize(nkv, vec![]);
        if l.swa && slot.k[0].len() >= 2 * cfg.n_swa * hd {
            let cut = slot.k[0].len() - cfg.n_swa * hd;
            for h in slot.k.iter_mut().chain(&mut slot.v) { h.drain(..cut); }
        }
        let base = slot.k[0].len() / hd;
        let mut half = vec![f16::ZERO; kw];
        let (mut qs, mut rs) = (Vec::with_capacity(nt * nh * hd), Vec::with_capacity(nt * nh * nr));
        let rows_of = |m: usize| -> Vec<&[f32]> { (0..nt).map(|t| row(m, t)).collect() };
        let taps = l.conv_proj.map(|_| (rows_of(6), rows_of(7)));
        let ks = convolve_rows(&rows_of(4), &l.convs[0], taps.as_ref().map(|t| &t.0[..]), &mut slot.hist[0], nc);
        let vs = convolve_rows(&rows_of(5), &l.convs[1], taps.as_ref().map(|t| &t.1[..]), &mut slot.hist[1], nc);
        let prep: Vec<[Vec<f32>; 4]> = ks.into_par_iter().zip(vs).enumerate().map(|(t, (mut k, mut v))| {
            let (mut q, mut r) = (row(0, t).to_vec(), row(1, t).to_vec());
            clip(&mut q, Some(&l.q_clip), cfg.qk_clip, hd);
            clip(&mut k, None, cfg.qk_clip, hd);
            for h in v.chunks_exact_mut(hd) {
                let t = h.to_vec();
                rms_norm(&t, cfg.eps, h);
            }
            clip(&mut r, Some(&l.rel_clip), cfg.rel_clip, nr);
            if !l.swa {
                // Log-scaled positions sharpen the full-attention layers in a long context.
                let tau = 1.0 + cfg.log_alpha * ((pos + t + 1) as f32 / cfg.log_floor).max(1.0).ln();
                for x in q.iter_mut().chain(r.iter_mut()) { *x *= tau; }
            }
            [q, r, k, v]
        }).collect();
        for [q, r, k, v] in &prep {
            if nt == 1 {
                self.tr("Qcur", ip, q);
                self.tr("Kcur", ip, k);
                self.tr("Vcur", ip, v);
            }
            qs.extend_from_slice(q);
            rs.extend_from_slice(r);
            for (rows, x) in [(&mut slot.k, k), (&mut slot.v, v)] {
                half.convert_from_f32_slice(x);
                for (rows, h) in rows.iter_mut().zip(half.chunks_exact(hd)) { rows.extend_from_slice(h); }
            }
        }
        let (scale, group) = (1.0 / (hd as f32).sqrt(), nh / nkv);
        let (keys, vals) = (&slot.k, &slot.v);
        let mut out = vec![0f32; nt * nh * hd];
        // One task per key head: the query heads that share it read its rows once.
        let head = |(i, o): (usize, &mut [f32])| {
            let (t, kh) = (i / nkv, i % nkv);
            // Token t sees the rows up to its own.
            let rows = base + t + 1;
            let first = if l.swa { rows.saturating_sub(cfg.n_swa) } else { 0 };
            let (keys, vals, n) = (&keys[kh], &vals[kh], rows - first);
            // Logit of a row at distance d: q.k/sqrt(width) + bias[d].r - sink. Rows past the bias table have no
            // bias. The sink is one more logit of zero that takes probability mass and has no value.
            let mut sc = vec![0f32; group * n];
            let (h0, ext) = (t * nh + kh * group, cfg.rel_extent);
            // The bias of the last `m` rows, a channel at a time: a dot per row costs as much as the row's q.k.
            let m = n.min(ext);
            for (j, sc) in sc.chunks_exact_mut(n).enumerate() {
                sc.fill(-row(2, t)[kh * group + j]);
                for (c, &r) in rs[(h0 + j) * nr..(h0 + j + 1) * nr].iter().enumerate() {
                    for (s, b) in sc[n - m..].iter_mut().zip(&l.rel_bias[(c + 1) * ext - m..(c + 1) * ext]) { *s += r * b; }
                }
            }
            dots_h(&mut sc, scale, &keys[first * hd..rows * hd], &qs[h0 * hd..(h0 + group) * hd], n);
            for sc in sc.chunks_exact_mut(n) {
                let mx = sc.iter().cloned().fold(0f32, f32::max);
                let mut z = (-mx).exp();
                // Two loops: the running sum keeps the first from vector code.
                for s in sc.iter_mut() { *s = exp(*s - mx); }
                for s in sc.iter() { z += *s; }
                for s in sc.iter_mut() { *s /= z; }
            }
            mix_h(o, &sc, &vals[first * hd..rows * hd], n);
        };
        out.par_chunks_mut(group * hd).enumerate().for_each(head);
        for (t, o) in out.chunks_exact_mut(nh * hd).enumerate() {
            for (o, g) in o.iter_mut().zip(row(3, t)) { *o *= sigmoid(*g); }
        }
        let t0 = self.lap(2, t0);
        let proj = if nt == 1 {
            let mut proj = vec![0f32; l.wo.rows];
            l.wo.matvec(&out, &mut proj);
            proj
        } else {
            matvec_rows(&l.wo, &out.chunks_exact(nh * hd).collect::<Vec<_>>())
        };
        self.lap(3, t0);
        convolve_rows(&proj.chunks_exact(l.wo.rows).collect::<Vec<_>>(), &l.convs[2], None, &mut slot.hist[2], nc)
    }

    /// The CPU's share of a layer's VRAM-cache misses, and whether it is fixed (no GPU, or set by the user).
    fn share(&self, share: &std::cell::Cell<f32>) -> (f32, bool) {
        static FIXED: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
        let fixed = *FIXED.get_or_init(|| {
            let share = std::env::var("STRATA_DS_CPU_SHARE").ok().and_then(|v| v.parse().ok());
            share.or(std::env::var_os("STRATA_FIXED_EXPERTS").map(|_| 0.0))
        });
        if crate::deepseek::mat::dev_info().is_none() { return (1.0, true); }
        (fixed.unwrap_or(share.get()).clamp(0.0, 1.0), fixed.is_some())
    }

    /// The routed experts of one token in layer `ip`: those of the GPU as `(gate_up, expert, weight)`, those of the
    /// CPU as `(gate_up, down, weight, gate_up scale)`, and whether the CPU share follows the two times.
    #[allow(clippy::type_complexity)]
    fn plan(&self, ip: usize, x: &[f32]) -> (Vec<(Mat<'a>, usize, f32)>, Vec<(Mat<'a>, Mat<'a>, f32, f32)>, bool) {
        let (cfg, l) = (&self.cfg, &self.layers[ip]);
        let ne = cfg.n_expert;
        let t = std::time::Instant::now();
        let sel = route(l, x, cfg);
        if std::env::var_os("STRATA_DS_NOPREFETCH").is_none() {
            for &(e, _) in &sel {
                for m in [&l.gate_up, &l.down] { m.prefetch_expert(e, ne); }
            }
        }
        // Experts resident in VRAM run on the GPU. Of the others, the CPU computes `cpu_share` from the mapped file
        // while the GPU uploads and runs the rest, both at once (see `hy4::forward`). `STRATA_DS_CPU_SHARE` fixes the
        // share; 0 (also `--fixed-experts`) runs every expert on the GPU: the same numerics on every run.
        self.lap(4, t);
        let (share, fixed) = self.share(&self.cpu_share);
        let gus: Vec<Mat> = sel.iter().map(|s| l.gate_up.expert_opt(s.0, ne, share == 0.0)).collect();
        let n_miss = gus.iter().filter(|m| m.gpu.is_none()).count();
        let mut left = (share * n_miss as f32).round() as usize;
        let (mut on_gpu, mut on_cpu) = (vec![], vec![]);
        for (gu, &(e, w)) in gus.into_iter().zip(&sel) {
            if gu.gpu.is_none() && left > 0 {
                left -= 1;
                on_cpu.push((gu, l.down.expert_host(e, ne), w, l.gu_scale[e]));
            } else {
                on_gpu.push((gu, e, w));
            }
        }
        (on_gpu, on_cpu, !fixed && n_miss > 0)
    }

    /// The GPU's experts of `plan` with both matrices in VRAM: in a cache slot, or copied up for this use.
    fn resident(l: &Layer<'a>, ne: usize, on_gpu: &[(Mat<'a>, usize, f32)]) -> Vec<(Mat<'a>, Mat<'a>, f32, f32)> {
        let on = |m: &Mat<'a>, e: usize| { let x = m.expert_opt(e, ne, true); if x.gpu.is_some() { x } else { x.staged() } };
        on_gpu.iter().map(|&(gu, e, w)| (if gu.gpu.is_some() { gu } else { on(&l.gate_up, e) }, on(&l.down, e), w, l.gu_scale[e])).collect()
    }

    /// MoE of one token with the weights of layer `ip`.
    fn ffn(&self, ip: usize, x: &[f32]) -> Vec<f32> {
        let (cfg, l) = (&self.cfg, &self.layers[ip]);
        let ne = cfg.n_expert;
        let (on_gpu, on_cpu, adapt) = self.plan(ip, x);
        let gpu_side = || {
            let t = std::time::Instant::now();
            (run_experts(&Self::resident(l, ne, &on_gpu), x, cfg), t.elapsed().as_secs_f32())
        };
        let cpu_side = || {
            let t = std::time::Instant::now();
            (run_experts(&on_cpu, x, cfg), t.elapsed().as_secs_f32())
        };
        let ((a, tg), (b, tc)) = rayon::join(gpu_side, cpu_side);
        self.balance(&self.cpu_share, adapt, tg, tc);
        a.iter().zip(&b).map(|(a, b)| a + b).collect()
    }

    /// MoE of a chunk of tokens: each routed expert runs once, on all the tokens that picked it (see
    /// `hy4::forward`). The split of the VRAM-cache misses between the GPU and the CPU is that of `ffn`.
    fn ffn_batch(&self, ip: usize, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let (cfg, l) = (&self.cfg, &self.layers[ip]);
        let (ne, nf, nl) = (cfg.n_expert, cfg.n_ff_exp, cfg.n_embd);
        let t = std::time::Instant::now();
        let sels: Vec<Vec<(usize, f32)>> = xs.par_iter().map(|x| route(l, x, cfg)).collect();
        // Per expert: the (token, rank in the token's selection) pairs that use it.
        let mut by_e: std::collections::BTreeMap<usize, Vec<(usize, usize)>> = Default::default();
        for (t, sel) in sels.iter().enumerate() {
            for (k, &(e, _)) in sel.iter().enumerate() { by_e.entry(e).or_default().push((t, k)); }
        }
        if std::env::var_os("STRATA_DS_NOPREFETCH").is_none() {
            for &e in by_e.keys() {
                for m in [&l.gate_up, &l.down] { m.prefetch_expert(e, ne); }
            }
        }
        self.lap(4, t);
        let (share, fixed) = self.share(&self.chunk_share);
        let on = |m: &Mat<'a>, e: usize| { let x = m.expert_opt(e, ne, share == 0.0); if x.gpu.is_some() { x } else { x.staged() } };
        let gus: Vec<(usize, Mat)> = by_e.keys().map(|&e| (e, l.gate_up.expert_opt(e, ne, share == 0.0))).collect();
        let n_miss = gus.iter().filter(|g| g.1.gpu.is_none()).count();
        let mut left = (share * n_miss as f32).round() as usize;
        let (mut on_gpu, mut on_cpu) = (vec![], vec![]);
        for (e, gu) in gus {
            if gu.gpu.is_none() && left > 0 {
                left -= 1;
                on_cpu.push((e, gu, l.down.expert_host(e, ne)));
            } else {
                on_gpu.push((e, if gu.gpu.is_some() { gu } else { gu.staged() }, on(&l.down, e)));
            }
        }
        let rows_of = |e: usize| -> Vec<&[f32]> { by_e[&e].iter().map(|&(t, _)| &xs[t][..]).collect() };
        let gpu_side = || {
            let t = std::time::Instant::now();
            // All the experts of the layer in one round trip, with a row a token up and down. Without room on the
            // device (or on more than one GPU): two round trips with a row a (token, expert) pair, the activation here.
            let (gus, downs): (Vec<Mat>, Vec<Mat>) = on_gpu.iter().map(|(_, gu, d)| (*gu, *d)).unzip();
            let pairs: Vec<Vec<(usize, usize, f32)>> = on_gpu.iter().map(|(e, ..)| by_e[e].iter().map(|&(t, k)| (t, k, sels[t][k].1)).collect()).collect();
            let gs: Vec<f32> = on_gpu.iter().map(|(e, ..)| l.gu_scale[*e]).collect();
            if let Some(sum) = situ_chunk(&gus, &downs, &gs, &pairs, &xs.concat(), cfg.n_used, cfg.situ_beta, cfg.situ_linear_beta) {
                return ((Some(sum), vec![]), t.elapsed().as_secs_f32());
            }
            let mut x = Vec::with_capacity(on_gpu.iter().map(|(e, ..)| by_e[e].len()).sum::<usize>() * nl);
            for (e, ..) in &on_gpu {
                for &(t, _) in &by_e[e] { x.extend_from_slice(&xs[t]); }
            }
            let jobs = |down: bool| -> Vec<(Mat, usize)> { on_gpu.iter().map(|(e, gu, d)| (if down { *d } else { *gu }, by_e[e].len())).collect() };
            let rs: Vec<f32> = on_gpu.iter().flat_map(|(e, ..)| std::iter::repeat_n(l.gu_scale[*e], by_e[e].len())).collect();
            let a: Vec<f32> = matvec_rows_many(&jobs(false), &x).par_chunks_exact(2 * nf).zip(&rs).flat_map_iter(|(y, &s)| situ(y, s, cfg)).collect();
            let dy = matvec_rows_many(&jobs(true), &a);
            let mut at = 0;
            let out: Vec<(usize, Vec<f32>)> = on_gpu.iter().map(|(e, ..)| {
                at += by_e[e].len() * nl;
                (*e, dy[at - by_e[e].len() * nl..at].to_vec())
            }).collect();
            ((None, out), t.elapsed().as_secs_f32())
        };
        let cpu_side = || {
            let t = std::time::Instant::now();
            let out: Vec<(usize, Vec<f32>)> = on_cpu.par_iter().map(|(e, gu, d)| {
                let a: Vec<f32> = matvec_rows(gu, &rows_of(*e)).chunks_exact(2 * nf).flat_map(|y| situ(y, l.gu_scale[*e], cfg)).collect();
                (*e, matvec_rows(d, &a.chunks_exact(nf).collect::<Vec<_>>()))
            }).collect();
            (out, t.elapsed().as_secs_f32())
        };
        let (((sum, a), tg), (b, tc)) = rayon::join(gpu_side, cpu_side);
        // A layer with few misses says nothing of the split: the GPU side is then the experts it holds.
        self.balance(&self.chunk_share, !fixed && 2 * n_miss >= by_e.len(), tg, tc);
        // Each token sums its experts in its own selection order, as `ffn` does (after the sum made on the device).
        let mut ds: Vec<Vec<&[f32]>> = sels.iter().map(|s| vec![&[][..]; s.len()]).collect();
        for (e, dy) in a.iter().chain(&b) {
            for (&(t, k), d) in by_e[e].iter().zip(dy.chunks_exact(nl)) { ds[t][k] = d; }
        }
        ds.par_iter().zip(&sels).enumerate().map(|(t, (ds, sel))| {
            let mut m = sum.as_ref().map_or_else(|| vec![0f32; nl], |s| s[t * nl..(t + 1) * nl].to_vec());
            for (d, &(_, w)) in ds.iter().zip(sel) {
                for (m, d) in m.iter_mut().zip(*d) { *m += w * d; }
            }
            m
        }).collect()
    }

    /// Count the two sides' times and move the CPU share toward the side that finished first.
    fn balance(&self, share: &std::cell::Cell<f32>, adapt: bool, tg: f32, tc: f32) {
        self.prof[5].set(self.prof[5].get() + tg as f64);
        self.prof[6].set(self.prof[6].get() + tc as f64);
        if adapt && tg + tc > 0.0 { share.set((share.get() + 0.1 * (tg - tc) / (tg + tc)).clamp(0.0, 1.0)); }
    }

    /// One layer run of a chunk of tokens (the first at `pos`): slot `il` (state) with the weights of layer `ip`.
    /// `cur`, `blocks` and `msgs` (the four cross-pass messages of a repeated core layer) have one entry a token.
    fn layer(&mut self, il: usize, ip: usize, cur: Vec<Vec<f32>>, blocks: &mut [Vec<Vec<f32>>], msgs: Option<&[Vec<f32>]>, pos: usize) -> Vec<Vec<f32>> {
        let mut slot = std::mem::take(&mut self.slots[il]);
        let (cfg, l, nt) = (&self.cfg, &self.layers[ip], cur.len());
        let n = cfg.n_embd;
        let lp = &self.loops[ip.clamp(cfg.loop_start, cfg.loop_end - 1) - cfg.loop_start];
        let t = std::time::Instant::now();
        let bl = &*blocks;
        // Attention inputs: a sliding-window layer mixes one, a full-attention layer one each for Q, K and V.
        let mut ins: Vec<Vec<Vec<f32>>> = (0..nt).into_par_iter().map(|t| match &l.res_query {
            Some(q) => mix(&bl[t], &cur[t], q).chunks_exact(n).map(|c| c.to_vec()).collect(),
            None => vec![cur[t].clone()],
        }).collect();
        if let Some(m) = msgs {
            // The messages make the Q, K and V inputs of a sliding-window layer differ.
            let mut o: Vec<Vec<Vec<f32>>> = (0..3usize).into_par_iter().map(|j| {
                control(&(0..nt).map(|t| (&ins[t][j.min(ins[t].len() - 1)][..], &m[t][j * n..(j + 1) * n])).collect::<Vec<_>>(), lp, j)
            }).collect();
            ins = (0..nt).map(|t| (0..3).map(|j| std::mem::take(&mut o[j][t])).collect()).collect();
        }
        let ins: Vec<Vec<Vec<f32>>> = ins.into_par_iter().map(|ins| {
            (0..ins.len().max(l.norms.len())).map(|j| {
                let mut o = vec![0f32; n];
                rms_norm_w(&ins[j.min(ins.len() - 1)], &l.norms[j.min(l.norms.len() - 1)], cfg.eps, &mut o);
                o
            }).collect()
        }).collect();
        self.lap(0, t);
        let mut x = self.attention(ip, &mut slot, &ins, pos);
        if nt == 1 { self.tr("attn_out", il, &x[0]); }
        // The first layer of each block starts a new residual stream; the earlier one joins the mixed blocks.
        for ((x, cur), blocks) in x.iter_mut().zip(cur).zip(blocks.iter_mut()) {
            if ip % cfg.block_size == 0 {
                blocks.push(cur);
            } else {
                for (x, c) in x.iter_mut().zip(&cur) { *x += c; }
            }
        }
        if nt == 1 { self.tr("ffn_inp", il, &x[0]); }
        let t = std::time::Instant::now();
        let bl = &*blocks;
        let mut h: Vec<Vec<f32>> = (0..nt).into_par_iter().map(|t| mix(&bl[t], &x[t], &l.ffn_query)).collect();
        if let Some(m) = msgs { h = control(&(0..nt).map(|t| (&h[t][..], &m[t][3 * n..])).collect::<Vec<_>>(), lp, 3); }
        let hn: Vec<Vec<f32>> = h.into_par_iter().map(|h| {
            let mut hn = vec![0f32; n];
            rms_norm_w(&h, &l.ffn_norm, cfg.eps, &mut hn);
            hn
        }).collect();
        self.lap(0, t);
        let ys = if nt == 1 { vec![self.ffn(ip, &hn[0])] } else { self.ffn_batch(ip, &hn) };
        let ys = convolve_rows(&ys.iter().map(|y| &y[..]).collect::<Vec<_>>(), &l.convs[3], None, &mut slot.hist[3], cfg.n_conv);
        for (x, y) in x.iter_mut().zip(&ys) {
            if nt == 1 { self.tr("ffn_out", il, y); }
            for (x, y) in x.iter_mut().zip(y) { *x += y; }
        }
        if nt == 1 { self.tr("l_out", il, &x[0]); }
        self.slots[il] = slot;
        x
    }

    /// Run one token. Returns the logits of the next one.
    pub fn step(&mut self, tok: u32) -> Vec<f32> {
        let (pos, t0) = (self.toks.len(), std::time::Instant::now());
        if self.device.is_none() && !self.no_device {
            self.device = device::Device::new(self).map(Box::new);
            self.no_device = self.device.is_none();
        }
        if let Some(mut d) = self.device.take() {
            match d.prepare(self, pos) {
                Ok(()) => {
                    let logits = d.step(self, tok);
                    self.device = Some(d);
                    self.toks.push(tok);
                    self.report(pos, 1, t0);
                    return logits;
                }
                Err(e) => {
                    eprintln!("inkling2: no room in VRAM for the decode state ({e:#}); the tokens run as prompt chunks do");
                    d.to_host(&self.cfg, &self.layers, &mut self.slots, pos);
                    self.no_device = true;
                }
            }
        }
        self.run(&[tok], true)
    }

    /// Run the prompt tokens in chunks of `STRATA_DS_CHUNK` (default 512; 1 or a tensor dump: token by token).
    /// Returns the logits after the last one.
    pub fn prefill(&mut self, toks: &[u32]) -> Vec<f32> {
        let mut chunk: usize = std::env::var("STRATA_DS_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
        if self.trace || std::env::var_os("STRATA_DS_DUMP").is_some() { chunk = 1; }
        let mut logits = vec![];
        for (i, c) in toks.chunks(chunk.max(1)).enumerate() { logits = self.run(c, (i + 1) * chunk.max(1) >= toks.len()); }
        logits
    }

    /// Run a chunk of tokens layer by layer. Returns the logits after the last one when `want`.
    fn run(&mut self, toks: &[u32], want: bool) -> Vec<f32> {
        let cfg = self.cfg.clone();
        let (pos, nt) = (self.toks.len(), toks.len());
        let t0 = std::time::Instant::now();
        if let Some(d) = &mut self.device { d.to_host(&cfg, &self.layers, &mut self.slots, pos); }
        assert!(pos + nt <= self.cfg.n_ctx, "context is limited to {} tokens", self.cfg.n_ctx);
        let n = cfg.n_embd;
        let mut cur: Vec<Vec<f32>> = toks.iter().map(|&tok| {
            assert!((tok as usize) < self.tok_embd.rows, "token {tok} is outside the vocabulary");
            let (mut emb, mut cur) = (vec![0f32; n], vec![0f32; n]);
            self.tok_embd.row_into(tok as usize, &mut emb);
            rms_norm_w(&emb, &self.tok_norm, cfg.eps, &mut cur);
            cur
        }).collect();
        let mut blocks: Vec<Vec<Vec<f32>>> = vec![vec![]; nt];
        for il in 0..cfg.loop_start { cur = self.layer(il, il, cur, &mut blocks, None, pos); }
        // Every pass of the core starts from the prelude output.
        let (prelude, n_pre, core) = (cur.clone(), blocks[0].len(), cfg.loop_end - cfg.loop_start);
        for pass in 0..cfg.loop_count {
            // The bank of the previous pass, per token: its blocks after the first core block, and its output.
            let bank: Option<Vec<(Vec<Vec<f32>>, Vec<f32>)>> = (pass > 0).then(|| {
                let outs = std::mem::replace(&mut cur, prelude.clone());
                blocks.iter_mut().zip(outs).map(|(b, out)| (b.split_off(n_pre).into_iter().skip(1).collect(), out)).collect()
            });
            for ip in cfg.loop_start..cfg.loop_end {
                let il = if pass == 0 { ip } else { cfg.n_layer + (pass - 1) * core + ip - cfg.loop_start };
                let query = &self.loops[ip - cfg.loop_start].msg_query;
                let msgs: Option<Vec<Vec<f32>>> = bank.as_ref().map(|bk| bk.par_iter().map(|(b, out)| mix(b, out, query)).collect());
                cur = self.layer(il, ip, cur, &mut blocks, msgs.as_deref(), pos);
            }
        }
        for il in cfg.loop_end..cfg.n_layer { cur = self.layer(il, il, cur, &mut blocks, None, pos); }
        self.toks.extend_from_slice(toks);
        let t = std::time::Instant::now();
        let mut logits = vec![];
        if want {
            let x = mix(&blocks[nt - 1], &cur[nt - 1], &self.out_query);
            let mut h = vec![0f32; n];
            rms_norm_w(&x, &self.out_norm, cfg.eps, &mut h);
            self.tr("result_norm", 0, &h);
            logits = vec![0f32; self.output.rows];
            self.output.matvec(&h, &mut logits);
            logits.truncate(cfg.n_vocab);
            if cfg.softcap != 0.0 { logits.par_iter_mut().for_each(|v| *v = cfg.softcap * (*v / cfg.softcap).tanh()); }
        }
        self.lap(7, t);
        self.report(pos, nt, t0);
        logits
    }

    /// The `STRATA_DS_PROF` line of `nt` tokens run from `pos` on.
    fn report(&self, pos: usize, nt: usize, t0: std::time::Instant) {
        if std::env::var_os("STRATA_DS_PROF").is_some() {
            use std::sync::atomic::Ordering::Relaxed;
            let ex = &crate::deepseek::mat::EX_LOOKUPS;
            let parts: Vec<String> = PROF.iter().zip(&self.prof).map(|(n, c)| format!("{n} {:.1}", c.replace(0.0) * 1e3)).collect();
            eprintln!("PROF pos {pos}, {nt} tokens: {:.1} ms ({}), expert cache hit {} miss {}, cpu share {:.2} chunk {:.2}",
                t0.elapsed().as_secs_f64() * 1e3, parts.join(", "), ex[0].swap(0, Relaxed), ex[1].swap(0, Relaxed), self.cpu_share.get(), self.chunk_share.get());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_dot_and_axpy_follow_the_f16_values() {
        let x: Vec<f32> = (0..77).map(|i| (i as f32 * 0.37).sin()).collect();
        let w: Vec<f16> = (0..77).map(|i| f16::from_f32((i as f32 * 0.11).cos())).collect();
        let want: f32 = w.iter().zip(&x).map(|(w, x)| w.to_f32() * x).sum();
        assert!((dot_h(&w, &x) - want).abs() < 1e-5);
        let mut o = x.clone();
        axpy_h(&mut o, 0.5, &w);
        for i in 0..77 { assert!((o[i] - (x[i] + 0.5 * w[i].to_f32())).abs() < 1e-6); }
    }

    #[test]
    fn rows_in_one_pass_follow_the_row_kernels_and_do_not_depend_on_the_row_count() {
        let (n, hd) = (5, 64);
        let w: Vec<f16> = (0..n * hd).map(|i| f16::from_f32((i as f32 * 0.11).cos())).collect();
        let x: Vec<f32> = (0..3 * hd).map(|i| (i as f32 * 0.37).sin()).collect();
        let (mut p, mut o) = (vec![0f32; 3 * n], vec![0.25f32; 3 * hd]);
        dots_h(&mut p, 0.5, &w, &x, n);
        mix_h(&mut o, &p, &w, n);
        for j in 0..3 {
            for a in 0..n { assert!((p[j * n + a] - 0.5 * dot_h(&w[a * hd..(a + 1) * hd], &x[j * hd..(j + 1) * hd])).abs() < 1e-5); }
            let mut want = vec![0.25f32; hd];
            for a in 0..n { axpy_h(&mut want, p[j * n + a], &w[a * hd..(a + 1) * hd]); }
            for i in 0..hd { assert!((o[j * hd + i] - want[i]).abs() < 1e-4); }
            // A row alone: the same bits as in the pass of three.
            let (mut p1, mut o1) = (vec![0f32; n], vec![0.25f32; hd]);
            dots_h(&mut p1, 0.5, &w, &x[j * hd..(j + 1) * hd], n);
            mix_h(&mut o1, &p1, &w, n);
            assert!(p1 == p[j * n..(j + 1) * n] && o1 == o[j * hd..(j + 1) * hd]);
        }
    }

    #[test]
    fn exp_and_tanh_follow_libm() {
        let mut x = -87.0f32;
        while x < 87.0 {
            assert!((exp(x) / x.exp() - 1.0).abs() < 3e-7, "exp({x})");
            assert!((tanh(x) - x.tanh()).abs() < 3e-7, "tanh({x})");
            x += 0.0137;
        }
        assert_eq!((tanh(-50.0), tanh(50.0), exp(-1e9)), (-1.0, 1.0, exp(-87.0)));
    }

    #[test]
    fn convolve_uses_the_last_inputs() {
        // Two channels, three lags. Weight of lag k is 10^k for channel 0 and zero for channel 1.
        let w = [1.0, 10.0, 100.0, 0.0, 0.0, 0.0];
        let mut hist = vec![];
        let s = 0.5; // 1/sqrt(n_conv + 1)
        assert_eq!(convolve(&[1.0, 5.0], &w, None, &mut hist, 3), vec![(1.0 + 100.0) * s, 5.0 * s]);
        assert_eq!(convolve(&[2.0, 5.0], &w, None, &mut hist, 3), vec![(2.0 + 10.0 + 200.0) * s, 5.0 * s]);
        assert_eq!(convolve(&[3.0, 5.0], &w, None, &mut hist, 3), vec![(3.0 + 1.0 + 20.0 + 300.0) * s, 5.0 * s]);
        // Taps: one head over both channels; lag 0 is the current input, lag 2 the oldest.
        let got = convolve(&[4.0, 5.0], &[0.0; 6], Some(&[1.0, 0.0, 10.0]), &mut hist, 3);
        let s = 1.0 / 7f32.sqrt();
        assert_eq!(got, vec![(4.0 + 4.0 + 20.0) * s, (5.0 + 5.0 + 50.0) * s]);
    }

    #[test]
    fn clip_caps_the_row_norm() {
        let mut x = [3.0, 4.0, 0.3, 0.4];
        clip(&mut x, None, 1.0, 2);
        assert!((x[0] - 0.6).abs() < 1e-5 && (x[1] - 0.8).abs() < 1e-5);
        assert_eq!(&x[2..], &[0.3, 0.4]);
    }

    #[test]
    fn mix_of_equal_scores_is_the_mean_of_the_normed_states() {
        let got = mix(&[vec![2.0, 0.0]], &[0.0, 3.0], &[0.0, 0.0]);
        let r = 2f32.sqrt() / 2.0;
        assert!((got[0] - r).abs() < 1e-4 && (got[1] - r).abs() < 1e-4, "{got:?}");
    }
}
