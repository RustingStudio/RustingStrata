//! GigaChat 4 forward, mirroring llama.cpp `src/models/gigachat4.cpp` (trace names are its callback names, so
//! `STRATA_DS_DUMP` output lines up with `llama-eval-callback`). Decode is one token at a time; prefill runs a chunk
//! of prompt tokens layer by layer, so a routed expert is read once for all the tokens that picked it.
//! ponytail: MLA scores, the indexer and the DeltaNet recurrence run on the CPU; GPU kernels are the next step for
//! prompt speed.
use super::GigaChat4Config;
use crate::deepseek::mat::{Mat, matvec_many, matvec_rows, matvec_rows_many, vec_f32};
use crate::deepseek::ops::{Rope, rms_norm, rms_norm_w, swiglu_clamp};
use crate::gguf::Gguf;
use crate::glm::forward::route_sigmoid;
use crate::hy4::forward::{host_mat, run_experts, sigmoid, silu};
use crate::inkling2::forward::{dots_h, mix_h};
use anyhow::Result;
use half::f16;
use rayon::prelude::*;

/// RMS norm with a learned low-rank sigmoid gate (llama.cpp `build_gated_norm`): w * rms(x) * sigmoid(down silu(up
/// rms)) * 2.
struct GNorm {
    w: Vec<f32>,
    /// [rank][n] and [n][rank]
    up: Vec<f32>,
    down: Vec<f32>,
}

impl GNorm {
    fn load(g: &Gguf, name: &str) -> Result<Self> {
        Ok(GNorm { w: vec_f32(g, &format!("{name}.weight"))?, up: vec_f32(g, &format!("{name}_gate_up.weight"))?, down: vec_f32(g, &format!("{name}_gate_down.weight"))? })
    }

    fn apply(&self, x: &[f32], eps: f32) -> Vec<f32> {
        let n = x.len();
        let mut cur = vec![0f32; n];
        rms_norm_w(x, &self.w, eps, &mut cur);
        let h: Vec<f32> = self.up.chunks_exact(n).map(|r| silu(r.iter().zip(&cur).map(|(a, b)| a * b).sum())).collect();
        for (c, r) in cur.iter_mut().zip(self.down.chunks_exact(h.len())) {
            *c *= 2.0 * sigmoid(r.iter().zip(&h).map(|(a, b)| a * b).sum());
        }
        cur
    }
}

struct Gdn<'a> {
    qkv: Mat<'a>,
    z: Mat<'a>,
    alpha: Mat<'a>,
    beta: Mat<'a>,
    a: Vec<f32>,
    dt_bias: Vec<f32>,
    conv: Vec<f32>,
    norm: Vec<f32>,
    out: Mat<'a>,
}

struct Indexer<'a> {
    q_b: Mat<'a>,
    k: Mat<'a>,
    k_norm: Vec<f32>,
    proj: Mat<'a>,
}

struct Mla<'a> {
    gate: Mat<'a>,
    q_a: Mat<'a>,
    q_a_norm: GNorm,
    q_b: Mat<'a>,
    kv_a: Mat<'a>,
    kv_a_norm: GNorm,
    k_b: Vec<Mat<'a>>,
    v_b: Vec<Mat<'a>>,
    out: Mat<'a>,
    ix: Option<Indexer<'a>>,
}

enum Mixer<'a> {
    Gdn(Gdn<'a>),
    Mla(Mla<'a>),
}

enum Ffn<'a> {
    Dense { gate: Mat<'a>, up: Mat<'a>, down: Mat<'a> },
    Moe { gate_inp: Mat<'a>, bias: Vec<f32>, gexps: Mat<'a>, uexps: Mat<'a>, dexps: Mat<'a>, gate_sh: Mat<'a>, up_sh: Mat<'a>, down_sh: Mat<'a> },
}

struct Layer<'a> {
    attn_norm: GNorm,
    post_attn_norm: GNorm,
    ffn_norm: GNorm,
    post_ffw_norm: GNorm,
    mixer: Mixer<'a>,
    ffn: Ffn<'a>,
}

/// Per-layer cache: DeltaNet conv rows and state, or the MLA latent rows [kv_cmpr, k_pe] (f16, as llama.cpp keeps
/// them) and the indexer keys.
#[derive(Clone, Default)]
struct State {
    conv: Vec<f32>,
    s: Vec<f32>,
    kv: Vec<f16>,
    ik: Vec<f32>,
}

pub struct GigaChat4Model<'a> {
    pub cfg: GigaChat4Config,
    tok_embd: Mat<'a>,
    out_norm: GNorm,
    out_w: Mat<'a>,
    layers: Vec<Layer<'a>>,
    state: Vec<State>,
    rope: Rope,
    /// Share of a layer's VRAM-cache misses that the CPU computes while the GPU runs the rest (see `ffn`).
    cpu_share: std::cell::Cell<f32>,
    pub toks: Vec<u32>,
    pub trace: bool,
}

/// SwiGLU, clamped like ggml `swiglu_oai`-free `swiglu_clamp` when `limit` > 0.
fn act(g: f32, u: f32, limit: f32) -> f32 {
    if limit > 1e-6 { swiglu_clamp(g, u, limit) } else { silu(g) * u }
}

/// `x / sqrt(sum x² + eps)` (llama.cpp `build_gdn_l2_norm`).
fn l2_norm(x: &mut [f32], eps: f32) {
    let s = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() + eps).sqrt();
    for v in x { *v *= s; }
}

/// NeoX RoPE of the first `2 * rope.n_dims / 2` dims of `x` (pairs i, i + n_dims/2), the rest unrotated.
fn rope_neox(rope: &Rope, x: &mut [f32], pos: usize) {
    let h = rope.n_dims / 2;
    for i in 0..h {
        let (c, s) = rope.cs(pos as f32, i);
        let (a, b) = (x[i], x[i + h]);
        x[i] = a * c - b * s;
        x[i + h] = a * s + b * c;
    }
}

/// The `n` best of `scored` (score, index): highest score first, the lower index on a tie. Returns the indices.
fn top_n(mut scored: Vec<(f32, u32)>, n: usize) -> Vec<u32> {
    if scored.len() > n {
        scored.select_nth_unstable_by(n - 1, |a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        scored.truncate(n);
    }
    scored.into_iter().map(|s| s.1).collect()
}

/// `m` on each row of `xs`, row-major: one GPU round trip for a single token.
fn rows(ms: &[Mat], xs: &[&[f32]]) -> Vec<Vec<f32>> {
    if xs.len() == 1 {
        let y = matvec_many(ms, xs[0], 0);
        let mut at = 0;
        return ms.iter().map(|m| { at += m.rows; y[at - m.rows..at].to_vec() }).collect();
    }
    ms.iter().map(|m| matvec_rows(m, xs)).collect()
}

impl<'a> GigaChat4Model<'a> {
    pub fn new(g: &'a Gguf) -> Result<Self> {
        let cfg = GigaChat4Config::from_gguf(g)?;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let name = |n: &str| format!("blk.{il}.{n}");
            let m = |n: &str| Mat::load(g, &name(&format!("{n}.weight")));
            let v = |n: &str| vec_f32(g, &name(n));
            let gn = |n: &str| GNorm::load(g, &name(n));
            let mixer = if cfg.recurrent[il] {
                Mixer::Gdn(Gdn {
                    qkv: m("attn_qkv")?, z: m("attn_gate")?, alpha: m("ssm_alpha")?, beta: m("ssm_beta")?,
                    a: v("ssm_a")?, dt_bias: v("ssm_dt.bias")?, conv: v("ssm_conv1d.weight")?, norm: v("ssm_norm.weight")?, out: m("ssm_out")?,
                })
            } else {
                let (k_b, v_b) = (m("attn_k_b")?, m("attn_v_b")?);
                let nh = cfg.n_head;
                Mixer::Mla(Mla {
                    gate: m("attn_gate")?, q_a: m("attn_q_a")?, q_a_norm: gn("attn_q_a_norm")?, q_b: m("attn_q_b")?,
                    kv_a: m("attn_kv_a_mqa")?, kv_a_norm: gn("attn_kv_a_norm")?,
                    k_b: (0..nh).map(|h| k_b.group(h, nh)).collect(), v_b: (0..nh).map(|h| v_b.group(h, nh)).collect(),
                    out: m("attn_output")?,
                    ix: match g.tensors.contains_key(&name("indexer.attn_q_b.weight")) {
                        true => Some(Indexer { q_b: m("indexer.attn_q_b")?, k: m("indexer.attn_k")?, k_norm: v("indexer.k_norm.weight")?, proj: m("indexer.proj")? }),
                        false => None,
                    },
                })
            };
            let ffn = if il < cfg.n_dense {
                Ffn::Dense { gate: m("ffn_gate")?, up: m("ffn_up")?, down: m("ffn_down")? }
            } else {
                Ffn::Moe {
                    gate_inp: m("ffn_gate_inp")?, bias: v("exp_probs_b.bias")?,
                    gexps: m("ffn_gate_exps")?, uexps: m("ffn_up_exps")?, dexps: m("ffn_down_exps")?,
                    gate_sh: m("ffn_gate_shexp")?, up_sh: m("ffn_up_shexp")?, down_sh: m("ffn_down_shexp")?,
                }
            };
            layers.push(Layer { attn_norm: gn("attn_norm")?, post_attn_norm: gn("post_attention_norm")?, ffn_norm: gn("ffn_norm")?, post_ffw_norm: gn("post_ffw_norm")?, mixer, ffn });
        }
        let ext = if cfg.rope_scale > 1.0 { 1.0 } else { 0.0 };
        let rope = Rope::new(cfg.rope_base, 1.0 / cfg.rope_scale, ext, cfg.n_rot, cfg.rope_ctx_orig, cfg.beta_fast, cfg.beta_slow);
        let mut m = GigaChat4Model {
            tok_embd: host_mat(g, "token_embd.weight")?,
            out_norm: GNorm::load(g, "output_norm")?,
            out_w: Mat::load(g, "output.weight")?,
            state: vec![State::default(); cfg.n_layer],
            rope,
            cpu_share: std::cell::Cell::new(0.5),
            layers,
            cfg,
            toks: vec![],
            trace: std::env::var("STRATA_DS_TRACE").is_ok(),
        };
        m.reset();
        Ok(m)
    }

    pub fn reset(&mut self) {
        let c = &self.cfg;
        let channels = c.ssm_d_inner + 2 * c.ssm_n_group * c.ssm_d_state;
        for (il, st) in self.state.iter_mut().enumerate() {
            *st = State::default();
            if c.recurrent[il] {
                st.conv = vec![0.0; (c.ssm_conv - 1) * channels];
                st.s = vec![0.0; c.ssm_d_inner * c.ssm_d_state];
            }
        }
        self.toks.clear();
    }

    fn tr(&self, name: &str, il: i32, v: &[f32]) {
        if let Ok(dir) = std::env::var("STRATA_DS_DUMP") {
            use std::io::Write;
            let fname = if il < 0 { format!("{dir}/{name}.f32") } else { format!("{dir}/{name}-{il}.f32") };
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::OpenOptions::new().create(true).append(true).open(fname).unwrap().write_all(&bytes).unwrap();
        }
        if self.trace {
            let n = v.len();
            let f = |i: usize| format!("{:12.4}", v[i]);
            eprintln!("TRACE {name}-{il} [{} {} {} ... {} {} {}] sum={:.6}", f(0), f(1), f(2), f(n - 3), f(n - 2), f(n - 1), v.iter().sum::<f32>());
        }
    }

    /// Gated DeltaNet of the consecutive tokens `hs` (llama.cpp `build_layer_attn_linear`), before `ssm_out`'s norm.
    fn gdn(&mut self, il: usize, hs: &[&[f32]]) -> Vec<f32> {
        let c = &self.cfg;
        let Mixer::Gdn(w) = &self.layers[il].mixer else { unreachable!() };
        let (dk, nk, nv, kw, eps) = (c.ssm_d_state, c.ssm_n_group, c.ssm_dt_rank, c.ssm_conv, c.eps);
        let dv = c.ssm_d_inner / nv;
        let channels = c.ssm_d_inner + 2 * nk * dk;
        let p = rows(&[w.qkv, w.z, w.beta, w.alpha], hs);
        let st = &mut self.state[il];
        let mut out = vec![0f32; hs.len() * nv * dv];
        let scale = 1.0 / (dk as f32).sqrt();
        for t in 0..hs.len() {
            let qkv = &p[0][t * channels..(t + 1) * channels];
            let z = &p[1][t * nv * dv..(t + 1) * nv * dv];
            let beta: Vec<f32> = p[2][t * nv..(t + 1) * nv].iter().map(|&b| sigmoid(b)).collect();
            let g: Vec<f32> = p[3][t * nv..(t + 1) * nv].iter().enumerate().map(|(i, &al)| crate::ops::softplus(al + w.dt_bias[i]) * w.a[i]).collect();
            // Causal depthwise conv over [state..., current], then SiLU.
            let mut y = vec![0f32; channels];
            for ch in 0..channels {
                let mut acc = qkv[ch] * w.conv[ch * kw + kw - 1];
                for k in 0..kw - 1 { acc += st.conv[k * channels + ch] * w.conv[ch * kw + k]; }
                y[ch] = silu(acc);
            }
            st.conv.copy_within(channels.., 0);
            st.conv[(kw - 2) * channels..].copy_from_slice(qkv);
            let (q, rest) = y.split_at_mut(nk * dk);
            let (k, v) = rest.split_at_mut(nk * dk);
            for h in q.chunks_exact_mut(dk).chain(k.chunks_exact_mut(dk)) { l2_norm(h, eps); }
            let (q, k, v) = (&*q, &*k, &*v);
            let o = &mut out[t * nv * dv..(t + 1) * nv * dv];
            st.s.par_chunks_exact_mut(dv * dk).zip(o.par_chunks_exact_mut(dv)).enumerate().for_each(|(vh, (sm, oh))| {
                // v-head vh reads k-head vh / (nv / nk) (ggml repeats each k-head for consecutive v-heads)
                let kh = vh / (nv / nk);
                let (qh, khv, vv) = (&q[kh * dk..(kh + 1) * dk], &k[kh * dk..(kh + 1) * dk], &v[vh * dv..(vh + 1) * dv]);
                let decay = g[vh].exp();
                let mut o = vec![0f32; dv];
                for j in 0..dv {
                    let row = &mut sm[j * dk..(j + 1) * dk];
                    for x in row.iter_mut() { *x *= decay; }
                    let sk: f32 = row.iter().zip(khv).map(|(a, b)| a * b).sum();
                    let d = beta[vh] * (vv[j] - sk);
                    for (r, kk) in row.iter_mut().zip(khv) { *r += d * kk; }
                    o[j] = row.iter().zip(qh).map(|(a, b)| a * b).sum::<f32>() * scale;
                }
                rms_norm_w(&o, &w.norm, eps, oh);
                for (x, zz) in oh.iter_mut().zip(&z[vh * dv..(vh + 1) * dv]) { *x *= 2.0 * sigmoid(*zz); }
            });
        }
        matvec_rows(&w.out, &out.chunks_exact(nv * dv).collect::<Vec<_>>())
    }

    /// The positions a query at `pos` attends by the lightning indexer, or None for all of them.
    fn select(&self, il: usize, iq: &[f32], iw: &[f32], pos: usize) -> Option<Vec<u32>> {
        let c = &self.cfg;
        if pos < c.idx_top_k { return None; }
        let (hd, ik) = (c.idx_head_dim, &self.state[il].ik);
        let scored: Vec<(f32, u32)> = (0..=pos as u32).into_par_iter().map(|j| {
            let kj = &ik[j as usize * hd..][..hd];
            (iq.chunks_exact(hd).zip(iw).map(|(q, w)| q.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>().max(0.0) * w).sum(), j)
        }).collect();
        let mut sel = top_n(scored, c.idx_top_k);
        sel.sort_unstable();
        Some(sel)
    }

    /// MLA attention of the consecutive tokens `hs` from position `p0` (llama.cpp `build_layer_attn_mla`).
    fn mla(&mut self, il: usize, hs: &[&[f32]], p0: usize) -> Vec<f32> {
        let c = &self.cfg;
        let Mixer::Mla(w) = &self.layers[il].mixer else { unreachable!() };
        let (nt, nh, kd, kl, nr, eps) = (hs.len(), c.n_head, c.k_mla, c.kv_lora, c.n_rot, c.eps);
        let (nope, row, vd) = (kd - nr, kl + nr, c.v_mla);
        let mut ms = vec![w.gate, w.q_a, w.kv_a];
        if let Some(ix) = &w.ix { ms.extend([ix.k, ix.proj]); }
        let p = rows(&ms, hs);
        let alpha_q = (c.n_embd as f32 / c.q_lora as f32).sqrt();
        let alpha_kv = (c.n_embd as f32 / kl as f32).sqrt();
        let qr: Vec<f32> = p[1].chunks_exact(c.q_lora).flat_map(|x| w.q_a_norm.apply(x, eps).into_iter().map(|v| v * alpha_q)).collect();
        let qrs: Vec<&[f32]> = qr.chunks_exact(c.q_lora).collect();
        let mut q = matvec_rows(&w.q_b, &qrs);
        if nt == 1 { self.tr("q", il as i32, &q); }
        for t in 0..nt {
            let kv = &p[2][t * row..(t + 1) * row];
            let mut k_pe = kv[kl..].to_vec();
            self.rope.apply(&mut k_pe, p0 + t, false);
            let lat: Vec<f32> = w.kv_a_norm.apply(&kv[..kl], eps).into_iter().map(|v| v * alpha_kv).collect();
            self.state[il].kv.extend(lat.iter().chain(&k_pe).map(|&x| f16::from_f32(x)));
            for h in q[t * nh * kd..(t + 1) * nh * kd].chunks_exact_mut(kd) { self.rope.apply(h, p0 + t, false); }
        }
        // indexer keys of these tokens, then each query's selection
        let mut sels: Vec<Option<Vec<u32>>> = vec![None; nt];
        if let Some(ix) = &w.ix {
            let hd = c.idx_head_dim;
            for t in 0..nt {
                let mut ik = vec![0f32; hd];
                rms_norm(&p[3][t * hd..(t + 1) * hd], 1e-6, &mut ik);
                for (x, n) in ik.iter_mut().zip(&ix.k_norm) { *x *= n; }
                rope_neox(&self.rope, &mut ik, p0 + t);
                self.state[il].ik.extend_from_slice(&ik);
            }
            let long: Vec<usize> = (0..nt).filter(|&t| p0 + t >= c.idx_top_k).collect();
            if !long.is_empty() {
                let iq = matvec_rows(&ix.q_b, &long.iter().map(|&t| qrs[t]).collect::<Vec<_>>());
                let sc = 1.0 / ((hd * c.idx_n_head) as f32).sqrt();
                for (i, &t) in long.iter().enumerate() {
                    let mut iq = iq[i * c.idx_n_head * hd..(i + 1) * c.idx_n_head * hd].to_vec();
                    for h in iq.chunks_exact_mut(hd) { rope_neox(&self.rope, h, p0 + t); }
                    let iw: Vec<f32> = p[4][t * c.idx_n_head..(t + 1) * c.idx_n_head].iter().map(|v| v * sc).collect();
                    sels[t] = self.select(il, &iq, &iw, p0 + t);
                }
            }
        }
        // Absorbed query per head: [k_b q_nope (kl), q_pe (nr)]
        let mut qc = vec![0f32; nt * nh * row];
        for h in 0..nh {
            let xs: Vec<&[f32]> = (0..nt).map(|t| &q[(t * nh + h) * kd..][..nope]).collect();
            let y = matvec_rows(&w.k_b[h], &xs);
            for t in 0..nt {
                let dst = &mut qc[(t * nh + h) * row..][..row];
                dst[..kl].copy_from_slice(&y[t * kl..(t + 1) * kl]);
                dst[kl..].copy_from_slice(&q[(t * nh + h) * kd + nope..][..nr]);
            }
        }
        if nt == 1 { self.tr("Qcur", il as i32, &qc); }
        let mscale = c.mscale();
        let scale = mscale * mscale / (kd as f32).sqrt();
        let cache = &self.state[il].kv;
        // One task per token and group of 4 heads; V is the first kl dims of each row.
        const G: usize = 4;
        let mut att = vec![0f32; nt * nh * row];
        att.par_chunks_mut(G * row).enumerate().for_each(|(i, o)| {
            let t = i * G / nh;
            let pos = p0 + t;
            let gathered: Vec<f16>;
            let k: &[f16] = match &sels[t] {
                Some(sel) => { gathered = sel.iter().flat_map(|&p| &cache[p as usize * row..][..row]).copied().collect(); &gathered }
                None => &cache[..(pos + 1) * row],
            };
            let nk = k.len() / row;
            let qh = &qc[i * G * row..][..o.len()];
            let mut p = vec![0f32; o.len() / row * nk];
            dots_h(&mut p, scale, k, qh, nk);
            for p in p.chunks_exact_mut(nk) {
                let mx = p.iter().cloned().fold(f32::MIN, f32::max);
                let mut z = 0f32;
                for s in p.iter_mut() { *s = (*s - mx).exp(); z += *s; }
                for s in p.iter_mut() { *s /= z; }
            }
            mix_h(o, &p, k, nk);
        });
        let mut heads = vec![0f32; nt * nh * vd];
        for h in 0..nh {
            let xs: Vec<&[f32]> = (0..nt).map(|t| &att[(t * nh + h) * row..][..kl]).collect();
            let y = matvec_rows(&w.v_b[h], &xs);
            for t in 0..nt { heads[(t * nh + h) * vd..][..vd].copy_from_slice(&y[t * vd..(t + 1) * vd]); }
        }
        if nt == 1 { self.tr("attn_pregate", il as i32, &heads); }
        for (x, g) in heads.iter_mut().zip(&p[0]) { *x *= sigmoid(*g); }
        matvec_rows(&w.out, &heads.chunks_exact(nh * vd).collect::<Vec<_>>())
    }

    /// The (gate, up) clamps of layer `il`: routed experts, shared expert / dense FFN.
    fn clamps(&self, il: usize) -> (f32, f32) {
        (self.cfg.clamp_exp.get(il).copied().unwrap_or(0.0), self.cfg.clamp_shexp.get(il).copied().unwrap_or(0.0))
    }

    fn ffn(&self, il: usize, cur: &[f32]) -> Vec<f32> {
        let cfg = &self.cfg;
        let (lim, lim_sh) = self.clamps(il);
        match &self.layers[il].ffn {
            Ffn::Dense { gate, up, down } => {
                let y = matvec_many(&[*gate, *up], cur, 0);
                let a: Vec<f32> = (0..gate.rows).map(|i| act(y[i], y[gate.rows + i], lim_sh)).collect();
                let mut out = vec![0f32; down.rows];
                down.matvec(&a, &mut out);
                out
            }
            Ffn::Moe { gate_inp, bias, gexps, uexps, dexps, gate_sh, up_sh, down_sh } => {
                let ne = cfg.n_expert;
                // The shared expert rides with the routed ones on the GPU (weight 1) when both have the same clamp; a
                // tensor dump keeps it apart.
                let apart = self.trace || std::env::var_os("STRATA_DS_DUMP").is_some() || crate::deepseek::mat::dev_info().is_none() || lim != lim_sh;
                let y = if apart { matvec_many(&[*gate_inp, *gate_sh, *up_sh], cur, 0) } else { matvec_many(&[*gate_inp], cur, 0) };
                let (logits, sh) = y.split_at(ne);
                self.tr("ffn_moe_logits", il as i32, logits);
                let sel = route_sigmoid(logits, bias, cfg.n_used, cfg.expert_norm, cfg.expert_scale);
                if std::env::var_os("STRATA_DS_NOPREFETCH").is_none() {
                    for &(e, _) in &sel {
                        for m in [gexps, uexps, dexps] { m.prefetch_expert(e, ne); }
                    }
                }
                let nl = dexps.rows / ne;
                // Experts resident in VRAM run on the GPU; of the others the CPU computes `cpu_share` while the GPU
                // uploads and runs the rest (as in `kolibri::forward`).
                let (share, fixed) = self.share();
                let on = |m: &Mat<'a>, e: usize| { let x = m.expert_opt(e, ne, true); if x.gpu.is_some() { x } else { x.staged() } };
                let gates: Vec<Mat> = sel.iter().map(|s| gexps.expert_opt(s.0, ne, share == 0.0)).collect();
                let n_miss = gates.iter().filter(|m| m.gpu.is_none()).count();
                let mut left = (share * n_miss as f32).round() as usize;
                let (mut on_gpu, mut on_cpu) = (vec![], vec![]);
                for (g, &(e, w)) in gates.into_iter().zip(&sel) {
                    if g.gpu.is_none() && left > 0 {
                        left -= 1;
                        on_cpu.push((g, uexps.expert_host(e, ne), dexps.expert_host(e, ne), w));
                    } else {
                        on_gpu.push((g, e, w));
                    }
                }
                let cpu_side = || { let t = std::time::Instant::now(); (run_experts(&on_cpu, cur, lim, nl), t.elapsed().as_secs_f32()) };
                let gpu_side = || {
                    let t = std::time::Instant::now();
                    let mut list: Vec<_> = on_gpu.iter().map(|&(g, e, w)| (if g.gpu.is_some() { g } else { on(gexps, e) }, on(uexps, e), on(dexps, e), w)).collect();
                    if !apart { list.push((*gate_sh, *up_sh, *down_sh, 1.0)); }
                    (run_experts(&list, cur, lim, nl), t.elapsed().as_secs_f32())
                };
                let ((a, tg), (b, tc)) = rayon::join(gpu_side, cpu_side);
                if !fixed && n_miss > 0 { self.balance(tg, tc); }
                let mix: Vec<f32> = a.iter().zip(&b).map(|(a, b)| a + b).collect();
                if !apart { return mix; }
                self.tr("ffn_moe_out", il as i32, &mix);
                let a: Vec<f32> = (0..gate_sh.rows).map(|i| act(sh[i], sh[gate_sh.rows + i], lim_sh)).collect();
                let mut s = vec![0f32; down_sh.rows];
                down_sh.matvec(&a, &mut s);
                self.tr("ffn_shared", il as i32, &s);
                mix.iter().zip(&s).map(|(a, b)| a + b).collect()
            }
        }
    }

    /// The CPU's share of a layer's VRAM-cache misses, and whether it is fixed (no GPU, or set by the user).
    fn share(&self) -> (f32, bool) {
        static FIXED: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
        let fixed = *FIXED.get_or_init(|| {
            let share = std::env::var("STRATA_DS_CPU_SHARE").ok().and_then(|v| v.parse().ok());
            share.or(std::env::var_os("STRATA_FIXED_EXPERTS").map(|_| 0.0))
        });
        if crate::deepseek::mat::dev_info().is_none() { return (1.0, true); }
        (fixed.unwrap_or(self.cpu_share.get()).clamp(0.0, 1.0), fixed.is_some())
    }

    fn balance(&self, tg: f32, tc: f32) {
        if tg + tc > 0.0 { self.cpu_share.set((self.cpu_share.get() + 0.1 * (tg - tc) / (tg + tc)).clamp(0.0, 1.0)); }
    }

    /// `ffn` for the tokens of a prefill chunk, token-major. A routed expert is read once and applied to every token
    /// that picked it; the experts are split between the GPU and the CPU like the misses in `ffn`.
    /// ponytail: no `swiglu_chunk` (its kernel has no clamp); the experts on the GPU go in two batched passes.
    fn ffn_batch(&self, il: usize, curs: &[Vec<f32>]) -> Vec<f32> {
        let cfg = &self.cfg;
        let (lim, lim_sh) = self.clamps(il);
        let xs: Vec<&[f32]> = curs.iter().map(|c| c.as_slice()).collect();
        let swiglu = |gate: &Mat, up: &Mat, down: &Mat, x: &[&[f32]], lim: f32| -> Vec<f32> {
            let (gy, uy) = (matvec_rows(gate, x), matvec_rows(up, x));
            let a: Vec<f32> = gy.iter().zip(&uy).map(|(g, u)| act(*g, *u, lim)).collect();
            matvec_rows(down, &a.chunks_exact(gate.rows).collect::<Vec<_>>())
        };
        match &self.layers[il].ffn {
            Ffn::Dense { gate, up, down } => swiglu(gate, up, down, &xs, lim_sh),
            Ffn::Moe { gate_inp, bias, gexps, uexps, dexps, gate_sh, up_sh, down_sh } => {
                let ne = cfg.n_expert;
                let (nf, nl) = (gexps.rows / ne, dexps.rows / ne);
                let logits = matvec_rows(gate_inp, &xs);
                let sels: Vec<Vec<(usize, f32)>> = logits.par_chunks_exact(ne).map(|l| route_sigmoid(l, bias, cfg.n_used, cfg.expert_norm, cfg.expert_scale)).collect();
                let mut by_e: std::collections::BTreeMap<usize, Vec<(usize, usize)>> = Default::default();
                for (t, sel) in sels.iter().enumerate() {
                    for (k, &(e, _)) in sel.iter().enumerate() { by_e.entry(e).or_default().push((t, k)); }
                }
                if std::env::var_os("STRATA_DS_NOPREFETCH").is_none() {
                    by_e.keys().collect::<Vec<_>>().par_iter().for_each(|&&e| {
                        for m in [gexps, uexps, dexps] { m.prefetch_expert(e, ne); }
                    });
                }
                let (share, fixed) = self.share();
                let on = |m: &Mat<'a>, e: usize| { let x = m.expert_opt(e, ne, share == 0.0); if x.gpu.is_some() { x } else { x.staged() } };
                let gates: Vec<(usize, Mat)> = by_e.keys().map(|&e| (e, gexps.expert_opt(e, ne, share == 0.0))).collect();
                let n_miss = gates.iter().filter(|g| g.1.gpu.is_none()).count();
                let mut left = (share * n_miss as f32).round() as usize;
                let (mut on_gpu, mut on_cpu) = (vec![], vec![]);
                for (e, g) in gates {
                    if g.gpu.is_none() && left > 0 {
                        left -= 1;
                        on_cpu.push((e, g, uexps.expert_host(e, ne), dexps.expert_host(e, ne)));
                    } else {
                        on_gpu.push((e, if g.gpu.is_some() { g } else { g.staged() }, on(uexps, e), on(dexps, e)));
                    }
                }
                let rows_of = |e: usize| -> Vec<&[f32]> { by_e[&e].iter().map(|&(t, _)| xs[t]).collect() };
                let gpu = || {
                    let t = std::time::Instant::now();
                    let n_of = |e: &usize| by_e[e].len();
                    let mut x = Vec::with_capacity(2 * on_gpu.iter().map(|(e, ..)| n_of(e)).sum::<usize>() * cfg.n_embd);
                    for (e, ..) in &on_gpu {
                        for _ in 0..2 { for &(t, _) in &by_e[e] { x.extend_from_slice(xs[t]); } }
                    }
                    let y = matvec_rows_many(&on_gpu.iter().flat_map(|(e, g, u, _)| [(*g, n_of(e)), (*u, n_of(e))]).collect::<Vec<_>>(), &x);
                    let mut a = Vec::with_capacity(y.len() / 2);
                    let mut at = 0;
                    for (e, ..) in &on_gpu {
                        let n = n_of(e) * nf;
                        a.extend(y[at..at + n].iter().zip(&y[at + n..at + 2 * n]).map(|(g, u)| act(*g, *u, lim)));
                        at += 2 * n;
                    }
                    let dy = matvec_rows_many(&on_gpu.iter().map(|(e, _, _, d)| (*d, n_of(e))).collect::<Vec<_>>(), &a);
                    let mut at = 0;
                    let out: Vec<(usize, Vec<f32>)> = on_gpu.iter().map(|(e, ..)| { at += n_of(e) * nl; (*e, dy[at - n_of(e) * nl..at].to_vec()) }).collect();
                    (out, t.elapsed().as_secs_f32())
                };
                let cpu = || {
                    let t = std::time::Instant::now();
                    let out: Vec<(usize, Vec<f32>)> = on_cpu.par_iter().map(|(e, g, u, d)| (*e, swiglu(g, u, d, &rows_of(*e), lim))).collect();
                    (out, t.elapsed().as_secs_f32())
                };
                let ((a, tg), (b, tc)) = rayon::join(gpu, cpu);
                if !fixed && n_miss > 0 { self.balance(tg, tc); }
                // Each token sums its experts in its own selection order, as `ffn` does; the shared expert is added last.
                let mut ds: Vec<Vec<&[f32]>> = sels.iter().map(|s| vec![&[][..]; s.len()]).collect();
                for (e, dy) in a.iter().chain(&b) {
                    for (&(t, k), d) in by_e[e].iter().zip(dy.chunks_exact(nl)) { ds[t][k] = d; }
                }
                let mut mix = swiglu(gate_sh, up_sh, down_sh, &xs, lim_sh);
                for (t, m) in mix.chunks_exact_mut(nl).enumerate() {
                    let mut sum = vec![0f32; nl];
                    for (d, &(_, w)) in ds[t].iter().zip(&sels[t]) {
                        for j in 0..nl { sum[j] += w * d[j]; }
                    }
                    for j in 0..nl { m[j] += sum[j]; }
                }
                mix
            }
        }
    }

    pub fn step(&mut self, tok: u32) -> Vec<f32> {
        self.forward(&[tok], true)
    }

    /// Run `toks` through the model; logits only for the last one. All but the last go in chunks of
    /// `STRATA_DS_CHUNK` tokens (default 1024, or 1 with a tensor dump).
    pub fn prefill(&mut self, toks: &[u32]) -> Vec<f32> {
        let Some((&last, head)) = toks.split_last() else { return vec![] };
        let dump = self.trace || std::env::var_os("STRATA_DS_DUMP").is_some();
        let chunk: usize = std::env::var("STRATA_DS_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(if dump { 1 } else { 1024 });
        for c in head.chunks(chunk.max(1)) { self.forward(c, false); crate::term::prefilled(c.len()); }
        self.step(last)
    }

    /// Layer-major pass over consecutive tokens; the logits of the last one when `want`.
    fn forward(&mut self, toks: &[u32], want: bool) -> Vec<f32> {
        let t0 = std::time::Instant::now();
        let (n, nt, eps) = (self.cfg.n_embd, toks.len(), self.cfg.eps);
        let p0 = self.toks.len();
        self.toks.extend_from_slice(toks);
        let mut xs: Vec<Vec<f32>> = toks.iter().map(|&t| {
            let mut emb = vec![0f32; n];
            self.tok_embd.row_into(t as usize, &mut emb);
            emb
        }).collect();
        let (mut t_attn, mut t_ffn) = (0f64, 0f64);
        for il in 0..self.cfg.n_layer {
            let t = std::time::Instant::now();
            let hs: Vec<Vec<f32>> = xs.par_iter().map(|x| self.layers[il].attn_norm.apply(x, eps)).collect();
            if nt == 1 { self.tr("attn_norm", il as i32, &hs[0]); }
            let hr: Vec<&[f32]> = hs.iter().map(|h| h.as_slice()).collect();
            let att = if self.cfg.recurrent[il] { self.gdn(il, &hr) } else { self.mla(il, &hr, p0) };
            if nt == 1 { self.tr("attn_out", il as i32, &att); }
            let post: Vec<Vec<f32>> = att.par_chunks_exact(n).map(|a| self.layers[il].post_attn_norm.apply(a, eps)).collect();
            for (x, a) in xs.iter_mut().zip(&post) { for j in 0..n { x[j] += a[j]; } }
            if nt == 1 { self.tr("attn_residual", il as i32, &xs[0]); }
            t_attn += t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let hs: Vec<Vec<f32>> = xs.par_iter().map(|x| self.layers[il].ffn_norm.apply(x, eps)).collect();
            if nt == 1 { self.tr("ffn_norm", il as i32, &hs[0]); }
            let f = if nt == 1 { self.ffn(il, &hs[0]) } else { self.ffn_batch(il, &hs) };
            if nt == 1 { self.tr("ffn_out", il as i32, &f); }
            let post: Vec<Vec<f32>> = f.par_chunks_exact(n).map(|a| self.layers[il].post_ffw_norm.apply(a, eps)).collect();
            for (x, a) in xs.iter_mut().zip(&post) { for j in 0..n { x[j] += a[j]; } }
            if nt == 1 { self.tr("l_out", il as i32, &xs[0]); }
            t_ffn += t.elapsed().as_secs_f64();
        }
        let mut logits = vec![];
        if want {
            let cur = self.out_norm.apply(&xs[nt - 1], eps);
            self.tr("result_norm", -1, &cur);
            logits = vec![0f32; self.out_w.rows];
            self.out_w.matvec(&cur, &mut logits);
            let cap = self.cfg.softcap;
            if cap > 0.0 { for l in logits.iter_mut() { *l = (*l / cap).tanh() * cap; } }
            self.tr("result_output", -1, &logits);
        }
        if std::env::var_os("STRATA_DS_PROF").is_some() {
            use std::sync::atomic::Ordering::Relaxed;
            let ex = &crate::deepseek::mat::EX_LOOKUPS;
            eprintln!("PROF {nt} at {p0}: {:.0} ms (attn {:.0}, ffn {:.0}), expert cache hit {} miss {}, cpu share {:.2}",
                t0.elapsed().as_secs_f64() * 1e3, t_attn * 1e3, t_ffn * 1e3, ex[0].swap(0, Relaxed), ex[1].swap(0, Relaxed), self.cpu_share.get());
        }
        logits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gated_norm_and_l2() {
        // rank 1: up = [1, 0], down = [[0], [1]]; x = [3, 4] → rms = sqrt(12.5)
        let g = GNorm { w: vec![1.0, 1.0], up: vec![1.0, 0.0], down: vec![0.0, 1.0] };
        let y = g.apply(&[3.0, 4.0], 0.0);
        let r = 12.5f32.sqrt();
        let (c0, c1) = (3.0 / r, 4.0 / r);
        let h = silu(c0);
        assert!((y[0] - c0 * 2.0 * 0.5).abs() < 1e-6 && (y[1] - c1 * 2.0 * sigmoid(h)).abs() < 1e-6, "{y:?}");
        let mut v = [3.0, 4.0];
        l2_norm(&mut v, 0.0);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn neox_rope_rotates_the_first_dims() {
        let rope = Rope::new(10000.0, 1.0, 0.0, 4, 4096, 32.0, 1.0);
        let mut x = [1.0, 2.0, 0.0, 0.0, 7.0, 8.0];
        rope_neox(&rope, &mut x, 1);
        assert!((x[0] - 1f32.cos()).abs() < 1e-6 && (x[2] - 1f32.sin()).abs() < 1e-6);
        assert!((x[1] - 2.0 * 0.01f32.cos()).abs() < 1e-6 && (x[3] - 2.0 * 0.01f32.sin()).abs() < 1e-6);
        assert_eq!(x[4..], [7.0, 8.0]);
    }

    #[test]
    fn top_n_ties_to_the_lower_index() {
        assert_eq!(top_n(vec![(1.0, 0), (2.0, 1), (2.0, 2), (0.5, 3)], 2), [1, 2]);
        assert_eq!(top_n(vec![(1.0, 0), (1.0, 1), (1.0, 2)], 2), [0, 1]);
    }

    #[test]
    fn routing_normalises_then_scales() {
        let r = route_sigmoid(&[0.0, 0.0, 10.0], &[1.0, 0.0, 0.0], 2, true, 2.5);
        assert_eq!(r.iter().map(|x| x.0).collect::<Vec<_>>(), [0, 2]);
        assert!((r[0].1 + r[1].1 - 2.5).abs() < 1e-6);
        assert!((act(20.0, 20.0, 10.0) - swiglu_clamp(10.0, 10.0, 10.0)).abs() < 1e-6);
    }
}
