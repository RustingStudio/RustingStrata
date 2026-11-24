//! KRAFTON Raon-1 mini forward pass: 4-stream Raon-mHC, 18 RDA (Raon Delta Attention) layers, 7 gated NoPE MLA
//! layers, SiTU-GLU dense/MoE FFN + shared expert, and reasoning effort steering.
//! Matvecs and routed experts reuse `crate::deepseek::mat` (CPU AVX2/AVX-512 + CUDA/Vulkan/HIP GPU).
use super::RaonConfig;
use crate::deepseek::mat::{Mat, matvec_many, matvec_many_opt, matvec_rows, vec_f32};
use crate::deepseek::ops::rms_norm_w;
use crate::gguf::Gguf;
use crate::quant::GgmlType;
use crate::weights::dot;
use anyhow::{Context, Result};
use rayon::prelude::*;

struct Hcw<'a> {
    f: Mat<'a>,
    base: Vec<f32>,
    scale: Vec<f32>,
}

struct RdaWeights<'a> {
    wq: Mat<'a>,
    wk: Mat<'a>,
    wv: Mat<'a>,
    q_conv: Vec<f32>,
    k_conv: Vec<f32>,
    v_conv: Vec<f32>,
    f_a: Mat<'a>,
    f_b: Mat<'a>,
    beta: Mat<'a>,
    rda: Mat<'a>,
    ssm_a: Vec<f32>,
    dt_b: Vec<f32>,
    g: Mat<'a>,
    o_norm: Vec<f32>,
    wo: Mat<'a>,
}

struct MlaWeights<'a> {
    q_a: Mat<'a>,
    q_a_norm: Vec<f32>,
    q_b: Mat<'a>,
    kv_a: Mat<'a>,
    kv_a_norm: Vec<f32>,
    k_b: Mat<'a>,
    v_b: Mat<'a>,
    k_groups: Vec<Mat<'a>>,
    v_groups: Vec<Mat<'a>>,
    gate: Mat<'a>,
    wo: Mat<'a>,
}

enum Attn<'a> {
    Rda(RdaWeights<'a>),
    Mla(MlaWeights<'a>),
}

enum Ffn<'a> {
    Dense { gate: Mat<'a>, up: Mat<'a>, down: Mat<'a> },
    Moe {
        gate_inp: Mat<'a>,
        bias: Vec<f32>,
        gexps: Mat<'a>,
        uexps: Mat<'a>,
        dexps: Mat<'a>,
        shared: Option<(Mat<'a>, Mat<'a>, Mat<'a>)>,
        effort_steer: Option<Vec<f32>>,
    },
}

struct Layer<'a> {
    hc_attn: Hcw<'a>,
    hc_ffn: Hcw<'a>,
    attn_norm: Vec<f32>,
    attn: Attn<'a>,
    ffn_norm: Vec<f32>,
    ffn: Ffn<'a>,
}

/// Recurrent state for one RDA layer: rolling 3-token Q/K/V conv windows + `[n_head][head_dim][head_dim]` V-first state.
struct RdaState {
    q_hist: Vec<f32>,
    k_hist: Vec<f32>,
    v_hist: Vec<f32>,
    state: Vec<f32>,
}

impl RdaState {
    fn new(n_head: usize, hd: usize, d_conv: usize) -> Self {
        let d_inner = n_head * hd;
        let hlen = (d_conv - 1) * d_inner;
        Self {
            q_hist: vec![0.0; hlen],
            k_hist: vec![0.0; hlen],
            v_hist: vec![0.0; hlen],
            state: vec![0.0; n_head * hd * hd],
        }
    }

    fn clear(&mut self) {
        self.q_hist.fill(0.0);
        self.k_hist.fill(0.0);
        self.v_hist.fill(0.0);
        self.state.fill(0.0);
    }
}

pub struct RaonModel<'a> {
    pub cfg: RaonConfig,
    tok_embd: Mat<'a>,
    out_norm: Vec<f32>,
    out_w: Mat<'a>,
    layers: Vec<Layer<'a>>,
    helmert: Vec<f32>,
    chart_base: Vec<f32>,
    /// One `RdaState` per layer (`Some` for RDA layers, `None` for MLA layers).
    rda_states: Vec<Option<RdaState>>,
    /// One KV cache per layer (`[pos][kv_lora_rank + qk_rope_hd]` for MLA layers, empty for RDA layers).
    mla_keys: Vec<Vec<f32>>,
    /// Active reasoning effort steering alpha (`0.0` disables steering).
    pub effort_alpha: f32,
    pub toks: Vec<u32>,
    pub trace: bool,
}

fn host_mat<'a>(g: &'a Gguf, name: &str) -> Result<Mat<'a>> {
    let t = g.tensors.get(name).with_context(|| format!("missing tensor {name}"))?;
    Ok(Mat { data: g.bytes(t), ty: GgmlType::from_u32(t.ty)?, rows: t.dims[1] as usize, cols: t.dims[0] as usize, cpu: false, gpu: None, scale: 1.0 })
}

#[inline]
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

#[inline]
fn silu(x: f32) -> f32 { x * sigmoid(x) }

#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else if x < -20.0 { x.exp() } else { (1.0 + x.exp()).ln() }
}

/// SiTU-GLU: `(gate_beta * tanh(g / gate_beta) * sigmoid(g)) * (value_beta * tanh(v / value_beta))`.
#[inline]
pub fn situ_glu(g: f32, v: f32, gate_beta: f32, value_beta: f32) -> f32 {
    (gate_beta * (g / gate_beta).tanh() * sigmoid(g)) * (value_beta * (v / value_beta).tanh())
}

/// Orthonormal Helmert basis `Q` in `R^{n x (n-1)}` (row-major `[n][n-1]`).
pub fn helmert_basis(n: usize) -> Vec<f32> {
    let nm = n - 1;
    let mut q = vec![0.0f32; n * nm];
    for k in 1..n {
        let inv = 1.0 / ((k * (k + 1)) as f32).sqrt();
        for i in 0..k {
            q[i * nm + (k - 1)] = inv;
        }
        q[k * nm + (k - 1)] = -(k as f32) * inv;
    }
    q
}

/// Chart base `B = 0.5 * (I + 11^T / n)` (row-major `[n][n]`).
fn chart_base_mat(n: usize) -> Vec<f32> {
    let mut b = vec![0.5 / n as f32; n * n];
    for i in 0..n {
        b[i * n + i] += 0.5;
    }
    b
}

/// Compute Raon-mHC `(h_pre [hc], h_post [hc], h_res [hc*hc])` from `mixes = (fn @ flat) / rms`.
pub fn raon_mhc_coeffs(
    mixes: &[f32], base: &[f32], scale: &[f32], hc: usize, hc_mag: f32, helmert: &[f32], chart_base: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let nm = hc - 1;
    let mut pre = vec![0.0f32; hc];
    let mut post = vec![0.0f32; hc];
    for i in 0..hc {
        pre[i] = sigmoid(mixes[i] * scale[0] + base[i]);
        post[i] = hc_mag * sigmoid(mixes[hc + i] * scale[1] + base[hc + i]);
    }
    let mut z = vec![0.0f32; nm * nm];
    for i in 0..nm * nm {
        z[i] = mixes[2 * hc + i] * scale[2] + base[2 * hc + i];
    }
    // tmp = Q @ Z ([hc, nm]), e = tmp @ Q^T ([hc, hc])
    let mut tmp = vec![0.0f32; hc * nm];
    for i in 0..hc {
        for j in 0..nm {
            let mut s = 0.0f32;
            for m in 0..nm { s += helmert[i * nm + m] * z[m * nm + j]; }
            tmp[i * nm + j] = s;
        }
    }
    let mut e = vec![0.0f32; hc * hc];
    let mut max_ratio = 0.0f32;
    for i in 0..hc {
        for j in 0..hc {
            let mut s = 0.0f32;
            for m in 0..nm { s += tmp[i * nm + m] * helmert[j * nm + m]; }
            e[i * hc + j] = s;
            let r = -s / chart_base[i * hc + j];
            if r > max_ratio { max_ratio = r; }
        }
    }
    let inv_rho = 1.0 / (1.0 + max_ratio);
    let gamma = sigmoid(scale[3]);
    let mut h_res = vec![0.0f32; hc * hc];
    for i in 0..hc {
        for j in 0..hc {
            let phi = chart_base[i * hc + j] + e[i * hc + j] * inv_rho;
            let eye = if i == j { 1.0 } else { 0.0 };
            h_res[i * hc + j] = eye + gamma * (phi - eye);
        }
    }
    (pre, post, h_res)
}

/// Depthwise causal conv1d (kernel 4) + SiLU for one token, advancing the 3-token rolling history `[3, d_inner]`.
/// `w` is stored channel-major `[d_inner, 4]` (GGUF `{4, 1, d_inner, 1}`).
fn conv1d_step_silu(x: &[f32], hist: &mut [f32], w: &[f32], d_inner: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; d_inner];
    for c in 0..d_inner {
        let wc = &w[c * 4..c * 4 + 4];
        let h0 = hist[c];
        let h1 = hist[d_inner + c];
        let h2 = hist[2 * d_inner + c];
        let x0 = x[c];
        let y = h0 * wc[0] + h1 * wc[1] + h2 * wc[2] + x0 * wc[3];
        hist[c] = h1;
        hist[d_inner + c] = h2;
        hist[2 * d_inner + c] = x0;
        out[c] = silu(y);
    }
    out
}

/// Single-token V-first RDA recurrence across all heads (`state`: `[n_head][hd][hd]`, row-major `[v][k]`).
pub fn rda_recurrent_step(
    q: &[f32], k: &[f32], v: &[f32], log_alpha: &[f32], beta: &[f32],
    state: &mut [f32], n_head: usize, hd: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n_head * hd];
    out.par_chunks_mut(hd)
        .zip(state.par_chunks_mut(hd * hd))
        .enumerate()
        .for_each(|(h, (oh, sh))| {
            let qh = &q[h * hd..(h + 1) * hd];
            let kh = &k[h * hd..(h + 1) * hd];
            let vh = &v[h * hd..(h + 1) * hd];
            let lh = &log_alpha[h * hd..(h + 1) * hd];
            let bh = beta[h];
            let mut dec = [0.0f32; 128];
            for j in 0..hd { dec[j] = lh[j].exp(); }
            for vi in 0..hd {
                let row = &mut sh[vi * hd..(vi + 1) * hd];
                let mut pred = 0.0f32;
                for ki in 0..hd {
                    let s = row[ki] * dec[ki];
                    row[ki] = s;
                    pred += s * kh[ki];
                }
                let err = bh * (vh[vi] - pred);
                let mut o = 0.0f32;
                for ki in 0..hd {
                    let s = row[ki] + err * kh[ki];
                    row[ki] = s;
                    o += s * qh[ki];
                }
                oh[vi] = o;
            }
        });
    out
}

/// Sigmoid router with selection-only bias and optional renormalization.
pub fn route(logits: &[f32], bias: &[f32], top_k: usize, renorm: bool, scale: f32) -> Vec<(usize, f32)> {
    let probs: Vec<f32> = logits.iter().map(|&v| sigmoid(v)).collect();
    let mut sel: Vec<usize> = (0..probs.len()).collect();
    sel.sort_by(|&a, &b| (probs[b] + bias[b]).total_cmp(&(probs[a] + bias[a])));
    sel.truncate(top_k);
    let mut out: Vec<(usize, f32)> = sel.into_iter().map(|e| (e, probs[e])).collect();
    if renorm {
        let sum: f32 = out.iter().map(|(_, w)| *w).sum::<f32>() + 1e-20;
        for (_, w) in &mut out { *w /= sum; }
    }
    if scale != 1.0 {
        for (_, w) in &mut out { *w *= scale; }
    }
    out
}

fn run_experts(list: &[(Mat, Mat, Mat, f32)], x: &[f32], gate_beta: f32, value_beta: f32, n: usize) -> Vec<f32> {
    let mut mix = vec![0.0f32; n];
    let Some(first) = list.first() else { return mix };
    let (k, nf) = (list.len(), first.0.rows);
    let batched = list.iter().all(|e| e.0.gpu.is_some()).then(|| {
        let gu: Vec<Mat> = list.iter().map(|e| e.0).chain(list.iter().map(|e| e.1)).collect();
        let y = matvec_many_opt(&gu, x, 0)?;
        let a: Vec<f32> = (0..k * nf).map(|i| situ_glu(y[i], y[k * nf + i], gate_beta, value_beta)).collect();
        matvec_many_opt(&list.iter().map(|e| e.2).collect::<Vec<_>>(), &a, nf)
    }).flatten();
    match batched {
        Some(dy) => {
            for (i, e) in list.iter().enumerate() {
                for j in 0..n { mix[j] += e.3 * dy[i * n + j]; }
            }
        }
        None => {
            let (mut gy, mut uy, mut dy) = (vec![0.0f32; nf], vec![0.0f32; nf], vec![0.0f32; n]);
            for (g, u, d, w) in list {
                g.matvec(x, &mut gy);
                u.matvec(x, &mut uy);
                for (gv, uv) in gy.iter_mut().zip(&uy) { *gv = situ_glu(*gv, *uv, gate_beta, value_beta); }
                d.matvec(&gy, &mut dy);
                for j in 0..n { mix[j] += w * dy[j]; }
            }
        }
    }
    mix
}

impl<'a> RaonModel<'a> {
    pub fn new(g: &'a Gguf) -> Result<Self> {
        let cfg = RaonConfig::from_gguf(g)?;
        let hcw = |p: &str| -> Result<Hcw<'a>> {
            Ok(Hcw {
                f: Mat::load(g, &format!("{p}_fn.weight"))?,
                base: vec_f32(g, &format!("{p}_base.weight"))?,
                scale: vec_f32(g, &format!("{p}_scale.weight"))?,
            })
        };
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let m = |s: &str| Mat::load(g, &format!("blk.{il}.{s}.weight"));
            let v = |s: &str| vec_f32(g, &format!("blk.{il}.{s}.weight"));
            let attn = if cfg.is_rda[il] {
                Attn::Rda(RdaWeights {
                    wq: m("attn_q")?,
                    wk: m("attn_k")?,
                    wv: m("attn_v")?,
                    q_conv: v("ssm_conv1d_q")?,
                    k_conv: v("ssm_conv1d_k")?,
                    v_conv: v("ssm_conv1d_v")?,
                    f_a: m("ssm_f_a")?,
                    f_b: m("ssm_f_b")?,
                    beta: m("ssm_beta")?,
                    rda: m("ssm_rda")?,
                    ssm_a: vec_f32(g, &format!("blk.{il}.ssm_a"))?,
                    dt_b: vec_f32(g, &format!("blk.{il}.ssm_dt.bias"))?,
                    g: m("ssm_g")?,
                    o_norm: v("ssm_norm")?,
                    wo: m("attn_output")?,
                })
            } else {
                let k_b = Mat::load_groups(g, &format!("blk.{il}.attn_k_b.weight"), cfg.n_head)?;
                let v_b = Mat::load_groups(g, &format!("blk.{il}.attn_v_b.weight"), cfg.n_head)?;
                let k_groups = (0..cfg.n_head).map(|h| k_b.group(h, cfg.n_head)).collect();
                let v_groups = (0..cfg.n_head).map(|h| v_b.group(h, cfg.n_head)).collect();
                Attn::Mla(MlaWeights {
                    q_a: m("attn_q_a")?,
                    q_a_norm: v("attn_q_a_norm")?,
                    q_b: m("attn_q_b")?,
                    kv_a: m("attn_kv_a_mqa")?,
                    kv_a_norm: v("attn_kv_a_norm")?,
                    k_b,
                    v_b,
                    k_groups,
                    v_groups,
                    gate: m("attn_gate")?,
                    wo: m("attn_output")?,
                })
            };
            let ffn = if il < cfg.n_dense {
                Ffn::Dense { gate: m("ffn_gate")?, up: m("ffn_up")?, down: m("ffn_down")? }
            } else {
                let shared = if cfg.n_shared > 0 {
                    Some((m("ffn_gate_shexp")?, m("ffn_up_shexp")?, m("ffn_down_shexp")?))
                } else {
                    None
                };
                let effort_steer = if cfg.effort_layers[il] {
                    Some(v("ffn_effort_steer")?)
                } else {
                    None
                };
                Ffn::Moe {
                    gate_inp: m("ffn_gate_inp")?,
                    bias: vec_f32(g, &format!("blk.{il}.exp_probs_b.bias"))?,
                    gexps: m("ffn_gate_exps")?,
                    uexps: m("ffn_up_exps")?,
                    dexps: m("ffn_down_exps")?,
                    shared,
                    effort_steer,
                }
            };
            layers.push(Layer {
                hc_attn: hcw(&format!("blk.{il}.hc_attn"))?,
                hc_ffn: hcw(&format!("blk.{il}.hc_ffn"))?,
                attn_norm: v("attn_norm")?,
                attn,
                ffn_norm: v("ffn_norm")?,
                ffn,
            });
        }
        let rda_states = (0..cfg.n_layer)
            .map(|il| cfg.is_rda[il].then(|| RdaState::new(cfg.n_head, cfg.rda_hd, cfg.d_conv)))
            .collect();
        let effort_alpha = cfg.effort_alpha_for(None, true);
        let (helmert, chart_base) = (helmert_basis(cfg.hc), chart_base_mat(cfg.hc));
        Ok(RaonModel {
            tok_embd: host_mat(g, "token_embd.weight")?,
            out_norm: vec_f32(g, "output_norm.weight")?,
            out_w: Mat::load(g, "output.weight")?,
            mla_keys: vec![Vec::new(); cfg.n_layer],
            rda_states,
            layers,
            helmert,
            chart_base,
            effort_alpha,
            toks: Vec::new(),
            trace: std::env::var_os("STRATA_DS_TRACE").is_some(),
            cfg,
        })
    }

    pub fn reset(&mut self) {
        for s in self.rda_states.iter_mut().flatten() { s.clear(); }
        for k in &mut self.mla_keys { k.clear(); }
        self.toks.clear();
    }

    pub fn set_effort(&mut self, effort: Option<&str>, think: bool) {
        let next = self.cfg.effort_alpha_for(effort, think);
        if (next - self.effort_alpha).abs() > 1e-6 && !self.toks.is_empty() {
            // Prefix KV/RDA states were built under a different steering alpha; invalidate cache.
            self.reset();
        }
        self.effort_alpha = next;
    }

    fn hc_pre(&self, flat: &[f32], hc_w: &Hcw) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (hc, n) = (self.cfg.hc, self.cfg.n_embd);
        let inv = 1.0 / (flat.iter().map(|&v| v * v).sum::<f32>() / flat.len() as f32 + self.cfg.hc_eps).sqrt();
        let mut mixes = vec![0.0f32; hc_w.f.rows];
        hc_w.f.matvec(flat, &mut mixes);
        for v in &mut mixes { *v *= inv; }
        let (pre, post, h_res) = raon_mhc_coeffs(&mixes, &hc_w.base, &hc_w.scale, hc, self.cfg.hc_mag, &self.helmert, &self.chart_base);
        let mut cur = vec![0.0f32; n];
        for s in 0..hc {
            let w = pre[s];
            for j in 0..n { cur[j] += w * flat[s * n + j]; }
        }
        (cur, post, h_res)
    }

    fn hc_post(&self, flat: &mut [f32], branch: &[f32], post: &[f32], h_res: &[f32]) {
        let (hc, n) = (self.cfg.hc, self.cfg.n_embd);
        if hc <= 8 {
            let mut col = [0.0f32; 8];
            for j in 0..n {
                for src in 0..hc { col[src] = flat[src * n + j]; }
                let bj = branch[j];
                for dst in 0..hc {
                    let mut s = post[dst] * bj;
                    let row = &h_res[dst * hc..(dst + 1) * hc];
                    for src in 0..hc { s += row[src] * col[src]; }
                    flat[dst * n + j] = s;
                }
            }
        } else {
            let old = flat.to_vec();
            for dst in 0..hc {
                let out = &mut flat[dst * n..(dst + 1) * n];
                let p = post[dst];
                for j in 0..n {
                    let mut s = p * branch[j];
                    for src in 0..hc { s += h_res[dst * hc + src] * old[src * n + j]; }
                    out[j] = s;
                }
            }
        }
    }

    fn rda_step(&mut self, il: usize, cur: &[f32]) -> Vec<f32> {
        let Attn::Rda(r) = &self.layers[il].attn else { unreachable!() };
        let (nh, hd) = (self.cfg.n_head, self.cfg.rda_hd);
        let d_inner = nh * hd;
        let q_proj = matvec_rows(&r.wq, &[cur]);
        let k_proj = matvec_rows(&r.wk, &[cur]);
        let v_proj = matvec_rows(&r.wv, &[cur]);
        let gate = matvec_rows(&r.g, &[cur]);
        let beta_raw = matvec_rows(&r.beta, &[cur]);
        let rda_delta = matvec_rows(&r.rda, &[cur]);
        let f_a = matvec_rows(&r.f_a, &[cur]);
        let mut decay = matvec_rows(&r.f_b, &[&f_a]);
        for (d, &b) in decay.iter_mut().zip(&r.dt_b) { *d += b; }

        let st = self.rda_states[il].as_mut().unwrap();
        let mut q = conv1d_step_silu(&q_proj, &mut st.q_hist, &r.q_conv, d_inner);
        let mut k = conv1d_step_silu(&k_proj, &mut st.k_hist, &r.k_conv, d_inner);
        let mut v = conv1d_step_silu(&v_proj, &mut st.v_hist, &r.v_conv, d_inner);

        let q_scale = 1.0 / (hd as f32).sqrt();
        let mut log_alpha = vec![0.0f32; d_inner];
        let mut beta = vec![0.0f32; nh];
        for h in 0..nh {
            let qs = &mut q[h * hd..(h + 1) * hd];
            let qn = dot(qs, qs).sqrt().max(1e-6);
            let qf = q_scale / qn;
            for x in qs.iter_mut() { *x *= qf; }

            let ks = &mut k[h * hd..(h + 1) * hd];
            let kn = dot(ks, ks).sqrt().max(1e-6);
            let kf = 1.0 / kn;
            for x in ks.iter_mut() { *x *= kf; }

            // ssm_a holds -exp(A_log); negate to recover exp(A_log)
            let exp_a = -r.ssm_a[h];
            for j in 0..hd {
                let scaled = exp_a * decay[h * hd + j];
                log_alpha[h * hd + j] = match self.cfg.gate_lower_bound {
                    Some(lb) => lb * sigmoid(scaled),
                    None => -softplus(scaled),
                };
            }
            let br = beta_raw[h];
            let dr = rda_delta[h];
            let ratio = (softplus(-br) - softplus(-(br + dr))).exp();
            for x in &mut v[h * hd..(h + 1) * hd] { *x *= ratio; }
            beta[h] = sigmoid(br);
        }

        let out = rda_recurrent_step(&q, &k, &v, &log_alpha, &beta, &mut st.state, nh, hd);
        let mut gated = vec![0.0f32; d_inner];
        for h in 0..nh {
            let s = h * hd..(h + 1) * hd;
            rms_norm_w(&out[s.clone()], &r.o_norm, self.cfg.eps, &mut gated[s.clone()]);
            for j in s { gated[j] *= sigmoid(gate[j]); }
        }
        let mut y = vec![0.0f32; self.cfg.n_embd];
        r.wo.matvec(&gated, &mut y);
        y
    }

    fn mla_step(&mut self, il: usize, cur: &[f32]) -> Vec<f32> {
        let Attn::Mla(a) = &self.layers[il].attn else { unreachable!() };
        let (nh, kr, nr, kl, vl) = (self.cfg.n_head, self.cfg.kv_lora_rank, self.cfg.qk_rope_hd, self.cfg.mla_key_len, self.cfg.mla_val_len);
        let nope = kl - nr;
        let kw = kr + nr;

        let mut qr_raw = vec![0.0f32; self.cfg.q_lora_rank];
        a.q_a.matvec(cur, &mut qr_raw);
        let mut qr = vec![0.0f32; self.cfg.q_lora_rank];
        rms_norm_w(&qr_raw, &a.q_a_norm, self.cfg.latent_eps, &mut qr);
        let mut q = vec![0.0f32; nh * kl];
        a.q_b.matvec(&qr, &mut q);

        let mut kv = vec![0.0f32; kw];
        a.kv_a.matvec(cur, &mut kv);
        let mut krow = vec![0.0f32; kw];
        rms_norm_w(&kv[..kr], &a.kv_a_norm, self.cfg.latent_eps, &mut krow[..kr]);
        krow[kr..].copy_from_slice(&kv[kr..]);
        self.mla_keys[il].extend_from_slice(&krow);

        let qn = matvec_many(&a.k_groups, &q, kl);
        let scale = 1.0 / (kl as f32).sqrt();
        let all_k = &self.mla_keys[il];
        let n_pos = all_k.len() / kw;

        let mut att = vec![0.0f32; nh * kr];
        att.par_chunks_mut(kr).enumerate().for_each(|(h, oh)| {
            let qnh = &qn[h * kr..(h + 1) * kr];
            let qeh = &q[h * kl + nope..(h + 1) * kl];
            let mut scores = Vec::with_capacity(n_pos);
            let mut mx = f32::NEG_INFINITY;
            for p in 0..n_pos {
                let row = &all_k[p * kw..(p + 1) * kw];
                let s = (dot(qnh, &row[..kr]) + dot(qeh, &row[kr..])) * scale;
                if s > mx { mx = s; }
                scores.push(s);
            }
            let mut sum = 0.0f32;
            for s in &mut scores { *s = (*s - mx).exp(); sum += *s; }
            let inv = 1.0 / sum;
            for (p, &s) in scores.iter().enumerate() {
                let w = s * inv;
                let lat = &all_k[p * kw..p * kw + kr];
                for j in 0..kr { oh[j] += w * lat[j]; }
            }
        });

        let mut out = matvec_many(&a.v_groups, &att, kr);
        let mut g = vec![0.0f32; nh * vl];
        a.gate.matvec(cur, &mut g);
        for (o, gv) in out.iter_mut().zip(&g) { *o *= sigmoid(*gv); }
        let mut y = vec![0.0f32; self.cfg.n_embd];
        a.wo.matvec(&out, &mut y);
        y
    }

    fn ffn_step(&self, il: usize, cur: &[f32]) -> Vec<f32> {
        let (n, gb, vb) = (self.cfg.n_embd, self.cfg.situ_beta, self.cfg.situ_linear_beta);
        match &self.layers[il].ffn {
            Ffn::Dense { gate, up, down } => {
                let nf = gate.rows;
                let (mut gy, mut uy, mut y) = (vec![0.0f32; nf], vec![0.0f32; nf], vec![0.0f32; n]);
                gate.matvec(cur, &mut gy);
                up.matvec(cur, &mut uy);
                for (g, u) in gy.iter_mut().zip(&uy) { *g = situ_glu(*g, *u, gb, vb); }
                down.matvec(&gy, &mut y);
                y
            }
            Ffn::Moe { gate_inp, bias, gexps, uexps, dexps, shared, effort_steer } => {
                let mut logits = vec![0.0f32; gate_inp.rows];
                gate_inp.matvec(cur, &mut logits);
                let sel = route(&logits, bias, self.cfg.n_used, self.cfg.exp_norm, self.cfg.exp_scale);
                let list: Vec<(Mat, Mat, Mat, f32)> = sel.into_iter()
                    .map(|(e, w)| (
                        gexps.expert_opt(e, self.cfg.n_expert, true),
                        uexps.expert_opt(e, self.cfg.n_expert, true),
                        dexps.expert_opt(e, self.cfg.n_expert, true),
                        w,
                    ))
                    .collect();
                let mut y = run_experts(&list, cur, gb, vb, n);
                if let Some((sg, su, sd)) = shared {
                    let nf = sg.rows;
                    let (mut gy, mut uy, mut sy) = (vec![0.0f32; nf], vec![0.0f32; nf], vec![0.0f32; n]);
                    sg.matvec(cur, &mut gy);
                    su.matvec(cur, &mut uy);
                    for (g, u) in gy.iter_mut().zip(&uy) { *g = situ_glu(*g, *u, gb, vb); }
                    sd.matvec(&gy, &mut sy);
                    for j in 0..n { y[j] += sy[j]; }
                }
                if self.effort_alpha != 0.0
                    && let Some(steer) = effort_steer
                {
                    let a = self.effort_alpha;
                    for j in 0..n { y[j] += a * steer[j]; }
                }
                y
            }
        }
    }

    fn step_opt(&mut self, tok: u32, want_logits: bool) -> Vec<f32> {
        let (n, hc) = (self.cfg.n_embd, self.cfg.hc);
        self.toks.push(tok);
        let mut emb = vec![0.0f32; n];
        self.tok_embd.row_into(tok as usize, &mut emb);
        let mut flat = emb.repeat(hc);
        let mut normed = vec![0.0f32; n];

        for il in 0..self.cfg.n_layer {
            let (cur, post, h_res) = self.hc_pre(&flat, &self.layers[il].hc_attn);
            rms_norm_w(&cur, &self.layers[il].attn_norm, self.cfg.eps, &mut normed);
            let attn_out = if self.cfg.is_rda[il] {
                self.rda_step(il, &normed)
            } else {
                self.mla_step(il, &normed)
            };
            self.hc_post(&mut flat, &attn_out, &post, &h_res);

            let (cur, post, h_res) = self.hc_pre(&flat, &self.layers[il].hc_ffn);
            rms_norm_w(&cur, &self.layers[il].ffn_norm, self.cfg.eps, &mut normed);
            let ffn_out = self.ffn_step(il, &normed);
            self.hc_post(&mut flat, &ffn_out, &post, &h_res);
        }

        if !want_logits {
            return Vec::new();
        }
        let inv_hc = 1.0 / hc as f32;
        let mut cur = vec![0.0f32; n];
        for s in 0..hc {
            for j in 0..n { cur[j] += flat[s * n + j] * inv_hc; }
        }
        rms_norm_w(&cur, &self.out_norm, self.cfg.eps, &mut normed);
        let mut logits = vec![0.0f32; self.out_w.rows];
        self.out_w.matvec(&normed, &mut logits);
        logits
    }

    pub fn step(&mut self, tok: u32) -> Vec<f32> {
        self.step_opt(tok, true)
    }

    fn prefill_chunk(&mut self, toks: &[u32]) {
        let (nt, n, hc) = (toks.len(), self.cfg.n_embd, self.cfg.hc);
        if nt == 0 { return; }
        let p0 = self.toks.len();
        self.toks.extend_from_slice(toks);

        let mut flats: Vec<Vec<f32>> = toks.iter().map(|&t| {
            let mut emb = vec![0.0f32; n];
            self.tok_embd.row_into(t as usize, &mut emb);
            emb.repeat(hc)
        }).collect();

        let grouped = |m: &Mat<'a>, x: &[f32], nh: usize, w: usize| -> Vec<f32> {
            let gs: Vec<Mat> = (0..nh).map(|h| m.group(h, nh)).collect();
            if nt < nh {
                return x.chunks_exact(nh * w).flat_map(|xt| matvec_many(&gs, xt, w)).collect();
            }
            let r = gs[0].rows;
            let mut y = vec![0.0f32; nt * nh * r];
            for (h, g) in gs.iter().enumerate() {
                let rows_h: Vec<&[f32]> = x.chunks_exact(nh * w).map(|xt| &xt[h * w..][..g.cols]).collect();
                let yh = matvec_rows(g, &rows_h);
                for (t, yt) in yh.chunks_exact(r).enumerate() {
                    y[(t * nh + h) * r..][..r].copy_from_slice(yt);
                }
            }
            y
        };

        for il in 0..self.cfg.n_layer {
            // 1. Batched hc_attn pre + attn_norm
            let flat_refs: Vec<&[f32]> = flats.iter().map(Vec::as_slice).collect();
            let hc_mix_all = matvec_rows(&self.layers[il].hc_attn.f, &flat_refs);
            let mix_w = self.layers[il].hc_attn.f.rows;
            let mut posts = Vec::with_capacity(nt);
            let mut h_ress = Vec::with_capacity(nt);
            let mut normed_rows: Vec<Vec<f32>> = Vec::with_capacity(nt);
            for t in 0..nt {
                let flat = &flats[t];
                let inv = 1.0 / (flat.iter().map(|&v| v * v).sum::<f32>() / flat.len() as f32 + self.cfg.hc_eps).sqrt();
                let mixes: Vec<f32> = hc_mix_all[t * mix_w..(t + 1) * mix_w].iter().map(|&v| v * inv).collect();
                let (pre, post, h_res) = raon_mhc_coeffs(
                    &mixes, &self.layers[il].hc_attn.base, &self.layers[il].hc_attn.scale,
                    hc, self.cfg.hc_mag, &self.helmert, &self.chart_base,
                );
                let mut cur = vec![0.0f32; n];
                for s in 0..hc {
                    let w = pre[s];
                    for j in 0..n { cur[j] += w * flat[s * n + j]; }
                }
                let mut nr = vec![0.0f32; n];
                rms_norm_w(&cur, &self.layers[il].attn_norm, self.cfg.eps, &mut nr);
                posts.push(post);
                h_ress.push(h_res);
                normed_rows.push(nr);
            }

            // 2. Attention (batched RDA or batched MLA)
            let nrefs: Vec<&[f32]> = normed_rows.iter().map(Vec::as_slice).collect();
            let attn_out_all = match &self.layers[il].attn {
                Attn::Rda(r) => {
                    let (nh, hd) = (self.cfg.n_head, self.cfg.rda_hd);
                    let d_inner = nh * hd;
                    let q_all = matvec_rows(&r.wq, &nrefs);
                    let k_all = matvec_rows(&r.wk, &nrefs);
                    let v_all = matvec_rows(&r.wv, &nrefs);
                    let g_all = matvec_rows(&r.g, &nrefs);
                    let b_all = matvec_rows(&r.beta, &nrefs);
                    let d_all = matvec_rows(&r.rda, &nrefs);
                    let fa_all = matvec_rows(&r.f_a, &nrefs);
                    let fa_refs: Vec<&[f32]> = fa_all.chunks_exact(r.f_a.rows).collect();
                    let fb_all = matvec_rows(&r.f_b, &fa_refs);

                    let q_scale = 1.0 / (hd as f32).sqrt();
                    let mut gated_all = vec![0.0f32; nt * d_inner];
                    let st = self.rda_states[il].as_mut().unwrap();
                    for t in 0..nt {
                        let mut q = conv1d_step_silu(&q_all[t * d_inner..(t + 1) * d_inner], &mut st.q_hist, &r.q_conv, d_inner);
                        let mut k = conv1d_step_silu(&k_all[t * d_inner..(t + 1) * d_inner], &mut st.k_hist, &r.k_conv, d_inner);
                        let mut v = conv1d_step_silu(&v_all[t * d_inner..(t + 1) * d_inner], &mut st.v_hist, &r.v_conv, d_inner);
                        let decay_t = &fb_all[t * d_inner..(t + 1) * d_inner];
                        let br_t = &b_all[t * nh..(t + 1) * nh];
                        let dr_t = &d_all[t * nh..(t + 1) * nh];
                        let mut log_alpha = vec![0.0f32; d_inner];
                        let mut beta = vec![0.0f32; nh];
                        for h in 0..nh {
                            let qs = &mut q[h * hd..(h + 1) * hd];
                            let qf = q_scale / dot(qs, qs).sqrt().max(1e-6);
                            for x in qs.iter_mut() { *x *= qf; }
                            let ks = &mut k[h * hd..(h + 1) * hd];
                            let kf = 1.0 / dot(ks, ks).sqrt().max(1e-6);
                            for x in ks.iter_mut() { *x *= kf; }
                            let exp_a = -r.ssm_a[h];
                            for j in 0..hd {
                                let scaled = exp_a * (decay_t[h * hd + j] + r.dt_b[h * hd + j]);
                                log_alpha[h * hd + j] = match self.cfg.gate_lower_bound {
                                    Some(lb) => lb * sigmoid(scaled),
                                    None => -softplus(scaled),
                                };
                            }
                            let (br, dr) = (br_t[h], dr_t[h]);
                            let ratio = (softplus(-br) - softplus(-(br + dr))).exp();
                            for x in &mut v[h * hd..(h + 1) * hd] { *x *= ratio; }
                            beta[h] = sigmoid(br);
                        }
                        let out = rda_recurrent_step(&q, &k, &v, &log_alpha, &beta, &mut st.state, nh, hd);
                        let gt = &g_all[t * d_inner..(t + 1) * d_inner];
                        let dst = &mut gated_all[t * d_inner..(t + 1) * d_inner];
                        for h in 0..nh {
                            let s = h * hd..(h + 1) * hd;
                            rms_norm_w(&out[s.clone()], &r.o_norm, self.cfg.eps, &mut dst[s.clone()]);
                            for j in s { dst[j] *= sigmoid(gt[j]); }
                        }
                    }
                    let grefs: Vec<&[f32]> = gated_all.chunks_exact(d_inner).collect();
                    matvec_rows(&r.wo, &grefs)
                }
                Attn::Mla(a) => {
                    let (nh, kr, nr, kl, vl) = (self.cfg.n_head, self.cfg.kv_lora_rank, self.cfg.qk_rope_hd, self.cfg.mla_key_len, self.cfg.mla_val_len);
                    let (nope, kw) = (kl - nr, kr + nr);
                    let qr_raw = matvec_rows(&a.q_a, &nrefs);
                    let mut qr = vec![0.0f32; qr_raw.len()];
                    for (src, dst) in qr_raw.chunks_exact(self.cfg.q_lora_rank).zip(qr.chunks_exact_mut(self.cfg.q_lora_rank)) {
                        rms_norm_w(src, &a.q_a_norm, self.cfg.latent_eps, dst);
                    }
                    let q_refs: Vec<&[f32]> = qr.chunks_exact(self.cfg.q_lora_rank).collect();
                    let q = matvec_rows(&a.q_b, &q_refs);

                    let kv_all = matvec_rows(&a.kv_a, &nrefs);
                    for kv in kv_all.chunks_exact(kw) {
                        let mut krow = vec![0.0f32; kw];
                        rms_norm_w(&kv[..kr], &a.kv_a_norm, self.cfg.latent_eps, &mut krow[..kr]);
                        krow[kr..].copy_from_slice(&kv[kr..]);
                        self.mla_keys[il].extend_from_slice(&krow);
                    }

                    let qn = grouped(&a.k_b, &q, nh, kl);
                    let scale = 1.0 / (kl as f32).sqrt();
                    let all_k = &self.mla_keys[il];
                    let mut att = vec![0.0f32; nt * nh * kr];
                    att.par_chunks_mut(kr).enumerate().for_each(|(idx, oh)| {
                        let t = idx / nh;
                        let n_pos = p0 + t + 1;
                        let qnh = &qn[idx * kr..(idx + 1) * kr];
                        let qeh = &q[idx * kl + nope..(idx + 1) * kl];
                        let mut scores = Vec::with_capacity(n_pos);
                        let mut mx = f32::NEG_INFINITY;
                        for p in 0..n_pos {
                            let row = &all_k[p * kw..(p + 1) * kw];
                            let s = (dot(qnh, &row[..kr]) + dot(qeh, &row[kr..])) * scale;
                            if s > mx { mx = s; }
                            scores.push(s);
                        }
                        let mut sum = 0.0f32;
                        for s in &mut scores { *s = (*s - mx).exp(); sum += *s; }
                        let inv = 1.0 / sum;
                        for (p, &s) in scores.iter().enumerate() {
                            let w = s * inv;
                            let lat = &all_k[p * kw..p * kw + kr];
                            for j in 0..kr { oh[j] += w * lat[j]; }
                        }
                    });

                    let mut out = grouped(&a.v_b, &att, nh, kr);
                    let g_all = matvec_rows(&a.gate, &nrefs);
                    for (o, &gv) in out.iter_mut().zip(&g_all) { *o *= sigmoid(gv); }
                    let orefs: Vec<&[f32]> = out.chunks_exact(nh * vl).collect();
                    matvec_rows(&a.wo, &orefs)
                }
            };

            for t in 0..nt {
                self.hc_post(&mut flats[t], &attn_out_all[t * n..(t + 1) * n], &posts[t], &h_ress[t]);
            }

            // 3. Batched hc_ffn pre + ffn_norm
            let flat_refs: Vec<&[f32]> = flats.iter().map(Vec::as_slice).collect();
            let hc_mix_all = matvec_rows(&self.layers[il].hc_ffn.f, &flat_refs);
            let mix_w = self.layers[il].hc_ffn.f.rows;
            posts.clear();
            h_ress.clear();
            normed_rows.clear();
            for t in 0..nt {
                let flat = &flats[t];
                let inv = 1.0 / (flat.iter().map(|&v| v * v).sum::<f32>() / flat.len() as f32 + self.cfg.hc_eps).sqrt();
                let mixes: Vec<f32> = hc_mix_all[t * mix_w..(t + 1) * mix_w].iter().map(|&v| v * inv).collect();
                let (pre, post, h_res) = raon_mhc_coeffs(
                    &mixes, &self.layers[il].hc_ffn.base, &self.layers[il].hc_ffn.scale,
                    hc, self.cfg.hc_mag, &self.helmert, &self.chart_base,
                );
                let mut cur = vec![0.0f32; n];
                for s in 0..hc {
                    let w = pre[s];
                    for j in 0..n { cur[j] += w * flat[s * n + j]; }
                }
                let mut nr = vec![0.0f32; n];
                rms_norm_w(&cur, &self.layers[il].ffn_norm, self.cfg.eps, &mut nr);
                posts.push(post);
                h_ress.push(h_res);
                normed_rows.push(nr);
            }

            // 4. Batched FFN (dense or expert-deduplicated MoE + shared + effort steering)
            let nrefs: Vec<&[f32]> = normed_rows.iter().map(Vec::as_slice).collect();
            let (gb, vb) = (self.cfg.situ_beta, self.cfg.situ_linear_beta);
            let ffn_out_all = match &self.layers[il].ffn {
                Ffn::Dense { gate, up, down } => {
                    let mut gy = matvec_rows(gate, &nrefs);
                    let uy = matvec_rows(up, &nrefs);
                    for (g, u) in gy.iter_mut().zip(&uy) { *g = situ_glu(*g, *u, gb, vb); }
                    let grefs: Vec<&[f32]> = gy.chunks_exact(gate.rows).collect();
                    matvec_rows(down, &grefs)
                }
                Ffn::Moe { gate_inp, bias, gexps, uexps, dexps, shared, effort_steer } => {
                    let logits_all = matvec_rows(gate_inp, &nrefs);
                    let mut by_e: std::collections::BTreeMap<usize, Vec<(usize, f32)>> = std::collections::BTreeMap::new();
                    for t in 0..nt {
                        let lg = &logits_all[t * gate_inp.rows..(t + 1) * gate_inp.rows];
                        for (e, w) in route(lg, bias, self.cfg.n_used, self.cfg.exp_norm, self.cfg.exp_scale) {
                            by_e.entry(e).or_default().push((t, w));
                        }
                    }
                    let mut y_all = vec![0.0f32; nt * n];
                    for (e, picks) in by_e {
                        let ge = gexps.expert_opt(e, self.cfg.n_expert, true);
                        let ue = uexps.expert_opt(e, self.cfg.n_expert, true);
                        let de = dexps.expert_opt(e, self.cfg.n_expert, true);
                        let x_e: Vec<&[f32]> = picks.iter().map(|&(t, _)| normed_rows[t].as_slice()).collect();
                        let mut gy = matvec_rows(&ge, &x_e);
                        let uy = matvec_rows(&ue, &x_e);
                        for (g, u) in gy.iter_mut().zip(&uy) { *g = situ_glu(*g, *u, gb, vb); }
                        let grefs: Vec<&[f32]> = gy.chunks_exact(ge.rows).collect();
                        let dy = matvec_rows(&de, &grefs);
                        for (pi, &(t, w)) in picks.iter().enumerate() {
                            let dst = &mut y_all[t * n..(t + 1) * n];
                            let src = &dy[pi * n..(pi + 1) * n];
                            for j in 0..n { dst[j] += w * src[j]; }
                        }
                    }
                    if let Some((sg, su, sd)) = shared {
                        let mut gy = matvec_rows(sg, &nrefs);
                        let uy = matvec_rows(su, &nrefs);
                        for (g, u) in gy.iter_mut().zip(&uy) { *g = situ_glu(*g, *u, gb, vb); }
                        let grefs: Vec<&[f32]> = gy.chunks_exact(sg.rows).collect();
                        let sy = matvec_rows(sd, &grefs);
                        for (a, b) in y_all.iter_mut().zip(&sy) { *a += b; }
                    }
                    if self.effort_alpha != 0.0
                        && let Some(steer) = effort_steer
                    {
                        let a = self.effort_alpha;
                        for row in y_all.chunks_exact_mut(n) {
                            for j in 0..n { row[j] += a * steer[j]; }
                        }
                    }
                    y_all
                }
            };

            for t in 0..nt {
                self.hc_post(&mut flats[t], &ffn_out_all[t * n..(t + 1) * n], &posts[t], &h_ress[t]);
            }
        }
    }

    pub fn prefill(&mut self, toks: &[u32]) -> Vec<f32> {
        let Some((&last, head)) = toks.split_last() else { return Vec::new() };
        let chunk: usize = std::env::var("STRATA_DS_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
        if chunk > 1 && !self.trace {
            for c in head.chunks(chunk) {
                self.prefill_chunk(c);
            }
        } else {
            for &t in head {
                self.step_opt(t, false);
            }
        }
        self.step_opt(last, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn situ_glu_matches_definition() {
        let y = situ_glu(1.5, -2.0, 4.0, 25.0);
        let want = (4.0 * (1.5f32 / 4.0).tanh() * sigmoid(1.5)) * (25.0 * (-2.0f32 / 25.0).tanh());
        assert!((y - want).abs() < 1e-6);
        assert!(situ_glu(0.0, 10.0, 4.0, 25.0).abs() < 1e-7);
    }

    #[test]
    fn raon_mhc_is_doubly_stochastic() {
        let hc = 4;
        let h = helmert_basis(hc);
        let b = chart_base_mat(hc);
        let mixes: Vec<f32> = (0..2 * hc + (hc - 1) * (hc - 1)).map(|i| (i as f32 * 0.7).sin() * 2.0).collect();
        let base = vec![0.1f32; mixes.len()];
        let scale = vec![0.5, 0.5, 1.2, 0.3];
        let (pre, post, h_res) = raon_mhc_coeffs(&mixes, &base, &scale, hc, 2.0, &h, &b);
        assert!(pre.iter().all(|&v| v > 0.0 && v < 1.0));
        assert!(post.iter().all(|&v| v > 0.0 && v < 2.0));
        for i in 0..hc {
            let rsum: f32 = (0..hc).map(|j| h_res[i * hc + j]).sum();
            let csum: f32 = (0..hc).map(|j| h_res[j * hc + i]).sum();
            assert!((rsum - 1.0).abs() < 1e-5, "row {i} sum {rsum}");
            assert!((csum - 1.0).abs() < 1e-5, "col {i} sum {csum}");
            for j in 0..hc {
                assert!(h_res[i * hc + j] >= -1e-6, "negative entry ({i},{j})={}", h_res[i * hc + j]);
            }
        }
    }

    #[test]
    fn rda_recurrent_matches_scalar_reference() {
        let (nh, hd) = (2, 4);
        let q: Vec<f32> = (0..nh * hd).map(|i| (i as f32 * 0.3 + 0.1).sin()).collect();
        let k: Vec<f32> = (0..nh * hd).map(|i| (i as f32 * 0.5 + 0.2).cos()).collect();
        let v: Vec<f32> = (0..nh * hd).map(|i| (i as f32 * 0.2 - 0.4).sin()).collect();
        let la: Vec<f32> = (0..nh * hd).map(|i| -0.1 * (i as f32 + 1.0)).collect();
        let beta = [0.6f32, 0.8f32];
        let mut st = vec![0.0f32; nh * hd * hd];
        let out = rda_recurrent_step(&q, &k, &v, &la, &beta, &mut st, nh, hd);
        // Step 0 from zero state: S = beta * v * k^T, out = beta * (k . q) * v
        for h in 0..nh {
            let kq = dot(&k[h * hd..(h + 1) * hd], &q[h * hd..(h + 1) * hd]);
            for vi in 0..hd {
                let want = beta[h] * kq * v[h * hd + vi];
                assert!((out[h * hd + vi] - want).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn router_uses_bias_for_selection_only() {
        let logits = [0.0f32, 2.0, -2.0, 0.1];
        let bias = [5.0f32, 0.0, 0.0, 0.0];
        let sel = route(&logits, &bias, 2, true, 1.0);
        assert_eq!(sel[0].0, 0);
        assert_eq!(sel[1].0, 1);
        let s0 = sigmoid(0.0);
        let s1 = sigmoid(2.0);
        assert!((sel[0].1 - s0 / (s0 + s1)).abs() < 1e-6);
        assert!((sel[1].1 - s1 / (s0 + s1)).abs() < 1e-6);
    }
}
