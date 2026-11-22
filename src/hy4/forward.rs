//! Hy4-flash forward, mirroring llama.cpp `src/models/hy-v4-flash.cpp` (trace names are its callback names, so
//! `STRATA_DS_DUMP` output lines up with `llama-eval-callback`). Decode is one token at a time; prefill runs a chunk
//! of prompt tokens layer by layer, so a routed expert is read once for all the tokens that picked it.
//! ponytail: the attention scores and the Engram projection of a chunk still run on the CPU one position at a time;
//! a GPU kernel for the scores is the next step when prompt speed matters more.
use super::Hy4Config;
use crate::deepseek::mat::{Mat, matvec_many, matvec_many_opt, matvec_rows, vec_f32};
use crate::deepseek::ops::{Rope, hc_collapse, rms_norm, rms_norm_w, swiglu_clamp};
use crate::gguf::Gguf;
use crate::quant::GgmlType;
use crate::weights::dot;
use anyhow::{Context, Result};
use rayon::prelude::*;

/// Epsilon of every Engram norm (FLT_EPSILON).
const ENGRAM_EPS: f32 = 1.1920929e-7;

struct Hcw<'a> {
    f: Mat<'a>,
    base: Vec<f32>,
    scale: Vec<f32>,
}

struct Engram<'a> {
    gate: Vec<f32>,
    hc_scale: Vec<f32>,
    kv: Mat<'a>,
    kv_b: Vec<f32>,
    lookup_norm: Vec<f32>,
    pre_gamma: Vec<f32>,
}

enum Ffn<'a> {
    Dense { gate: Mat<'a>, up: Mat<'a>, down: Mat<'a> },
    Moe {
        gate_inp: Mat<'a>,
        bias: Vec<f32>,
        gexps: Mat<'a>,
        uexps: Mat<'a>,
        dexps: Mat<'a>,
        routed_down: Mat<'a>,
        routed_up: Mat<'a>,
        gate_sh: Mat<'a>,
        up_sh: Mat<'a>,
        down_sh: Mat<'a>,
    },
}

/// kpool indexer of a sparse layer: picks the groups of earlier positions the layer attends.
struct Indexer<'a> {
    k: Mat<'a>,
    k_norm: Vec<f32>,
    value: Mat<'a>,
    gate: Mat<'a>,
    /// [kpool][key width]
    ape: Vec<f32>,
    comp_norm: Vec<f32>,
    q_b: Mat<'a>,
    proj: Mat<'a>,
}

struct Layer<'a> {
    idx: Option<Indexer<'a>>,
    hc_attn: Hcw<'a>,
    hc_ffn: Hcw<'a>,
    attn_norm: Vec<f32>,
    q_a: Mat<'a>,
    q_a_norm: Vec<f32>,
    q_b: Mat<'a>,
    kv_a: Mat<'a>,
    kv_a_norm: Vec<f32>,
    /// Per-head [kv_rank x nope] and [val_len x kv_rank] (row groups, see `Mat::load_groups`).
    k_b: Mat<'a>,
    v_b: Mat<'a>,
    sinks: Vec<f32>,
    gate: Mat<'a>,
    out: Mat<'a>,
    post_attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    post_ffw_norm: Vec<f32>,
    ffn: Ffn<'a>,
    engram: Option<Engram<'a>>,
    rope: Rope,
    nh: usize,
    swa: bool,
}

pub struct Hy4Model<'a> {
    pub cfg: Hy4Config,
    tok_embd: Mat<'a>,
    out_hc: Hcw<'a>,
    out_norm: Vec<f32>,
    out_w: Mat<'a>,
    engram_tab: Option<Mat<'a>>,
    layers: Vec<Layer<'a>>,
    /// Per layer: one row of kv_rank + n_rot floats per position (normed compressed KV, then the rotated key part).
    keys: Vec<Vec<f32>>,
    /// Per indexer layer: value|gate rows of the open group, and the pooled keys of the complete groups.
    idx_open: Vec<Vec<f32>>,
    idx_pools: Vec<Vec<f32>>,
    /// Share of a layer's VRAM-cache misses that the CPU computes while the GPU runs the rest (see `ffn`).
    cpu_share: std::cell::Cell<f32>,
    pub toks: Vec<u32>,
    pub trace: bool,
}

/// A table that is only read row by row: never uploaded (the Engram table is 36 GB).
fn host_mat<'a>(g: &'a Gguf, name: &str) -> Result<Mat<'a>> {
    let t = g.tensors.get(name).with_context(|| format!("missing tensor {name}"))?;
    Ok(Mat { data: g.bytes(t), ty: GgmlType::from_u32(t.ty)?, rows: t.dims[1] as usize, cols: t.dims[0] as usize, cpu: false, gpu: None, scale: 1.0, expert_scales: None })
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

fn act(gate: f32, up: f32, limit: f32) -> f32 {
    if limit > 1e-6 { swiglu_clamp(gate, up, limit) } else { silu(gate) * up }
}

/// Weighted sum of the listed experts' outputs, `(gate, up, down, weight)` each, for the latent input `x`. Experts
/// that share a GPU run as two batched round trips; otherwise each matvec runs where its matrix lives.
fn run_experts(list: &[(Mat, Mat, Mat, f32)], x: &[f32], clamp: f32, nl: usize) -> Vec<f32> {
    let mut mix = vec![0f32; nl];
    let Some(first) = list.first() else { return mix };
    let (k, nf) = (list.len(), first.0.rows);
    let batched = list.iter().all(|e| e.0.gpu.is_some()).then(|| {
        let gu: Vec<Mat> = list.iter().map(|e| e.0).chain(list.iter().map(|e| e.1)).collect();
        let y = matvec_many_opt(&gu, x, 0)?;
        let a: Vec<f32> = (0..k * nf).map(|i| act(y[i], y[k * nf + i], clamp)).collect();
        matvec_many_opt(&list.iter().map(|e| e.2).collect::<Vec<_>>(), &a, nf)
    }).flatten();
    match batched {
        Some(dy) => {
            for (i, e) in list.iter().enumerate() {
                for j in 0..nl { mix[j] += e.3 * dy[i * nl + j]; }
            }
        }
        None => {
            let (mut gy, mut uy, mut dy) = (vec![0f32; nf], vec![0f32; nf], vec![0f32; nl]);
            for (g, u, d, w) in list {
                g.matvec(x, &mut gy);
                u.matvec(x, &mut uy);
                for (g, u) in gy.iter_mut().zip(&uy) { *g = act(*g, *u, clamp); }
                d.matvec(&gy, &mut dy);
                for j in 0..nl { mix[j] += w * dy[j]; }
            }
        }
    }
    mix
}

/// The routed experts of one token and their weights, best first.
fn route(logits: &[f32], bias: &[f32], cfg: &Hy4Config) -> Vec<(usize, f32)> {
    let probs: Vec<f32> = logits.iter().map(|&v| sigmoid(v)).collect();
    let mut sel: Vec<usize> = (0..probs.len()).collect();
    sel.sort_by(|&a, &b| (probs[b] + bias[b]).partial_cmp(&(probs[a] + bias[a])).unwrap().then(a.cmp(&b)));
    sel.truncate(cfg.n_used);
    // llama.cpp clamps the sum to the smallest F16 before dividing.
    let sum = if cfg.exp_norm { sel.iter().map(|&e| probs[e]).sum::<f32>().max(6.103515625e-5) } else { 1.0 };
    sel.into_iter().map(|e| (e, probs[e] / sum * cfg.exp_scale)).collect()
}

/// One head's softmax mix of the compressed value rows of `keys`, for the query parts `qa` (against the compressed row)
/// and `qp` (rotary). The sink only takes probability mass: it is in the denominator and has no value.
fn head_mix(keys: &[&[f32]], qa: &[f32], qp: &[f32], sink: f32, scale: f32, o: &mut [f32]) {
    let kr = qa.len();
    let mut sc: Vec<f32> = keys.iter().map(|k| (dot(&k[..kr], qa) + dot(&k[kr..], qp)) * scale).collect();
    let mx = sc.iter().cloned().fold(sink, f32::max);
    let mut z = (sink - mx).exp();
    for s in sc.iter_mut() {
        *s = (*s - mx).exp();
        z += *s;
    }
    for (s, k) in sc.iter().zip(keys) {
        let p = s / z;
        for (o, k) in o.iter_mut().zip(&k[..kr]) { *o += p * k; }
    }
}

/// The key rows position `pos` attends: the indexer's groups plus its own open group, or every row from `w0`.
fn key_rows<'k>(all: &'k [f32], sel: Option<&[usize]>, pos: usize, w0: usize, kp: usize, kw: usize) -> Vec<&'k [f32]> {
    match sel {
        Some(groups) => groups.iter().flat_map(|&g| g * kp..(g + 1) * kp).chain(pos / kp * kp..=pos).map(|t| &all[t * kw..(t + 1) * kw]).collect(),
        None => all.chunks_exact(kw).take(pos + 1).skip(w0).collect(),
    }
}

/// The shared key/value row of a position from its `kv_a` projection: normed compressed KV, then the rotated key part.
fn key_row(l: &Layer, kv: &[f32], pos: usize, kr: usize, eps: f32) -> Vec<f32> {
    let mut krow = vec![0f32; kv.len()];
    rms_norm_w(&kv[..kr], &l.kv_a_norm, eps, &mut krow[..kr]);
    krow[kr..].copy_from_slice(&kv[kr..]);
    l.rope.apply(&mut krow, pos, false);
    krow
}

/// llama.cpp `build_kpool_select` for one position: caches the position's indexer value and gate (`vg`), and returns
/// the complete groups of `kpool` earlier positions the layer keeps (None: all of them fit). The group of `pos` itself
/// is always attended. A group's pooled key is the gate-softmax mix of its members' values, rotated to its first
/// position. `qw` gives the position's unrotated indexer query and head weights; it is asked only for a selection.
#[allow(clippy::too_many_arguments)]
fn kpool_select(ix: &Indexer, rope: &Rope, cfg: &Hy4Config, open: &mut Vec<f32>, pools: &mut Vec<f32>, vg: &[f32], pos: usize, qw: impl FnOnce() -> (Vec<f32>, Vec<f32>)) -> Option<Vec<usize>> {
    let (n, kp) = (ix.k_norm.len(), cfg.kpool);
    open.extend_from_slice(vg);
    let sel = (pools.len() / n > cfg.idx_top).then(|| {
        let (mut q, w) = qw();
        for qh in q.chunks_exact_mut(n) { rope.apply(qh, pos, false); }
        // The reference scales the score by 1/sqrt(n * nh), which does not change the order.
        let mut score: Vec<(f32, usize)> = pools.par_chunks(n).enumerate()
            .map(|(p, pk)| (q.chunks_exact(n).zip(&w).map(|(qh, w)| w * dot(qh, pk).max(0.0)).sum(), p)).collect();
        score.select_nth_unstable_by(cfg.idx_top - 1, |a, b| b.0.total_cmp(&a.0));
        score[..cfg.idx_top].iter().map(|s| s.1).collect()
    });
    if open.len() == kp * 2 * n {
        let mut pk = vec![0f32; n];
        for (d, o) in pk.iter_mut().enumerate() {
            let lg: Vec<f32> = (0..kp).map(|m| open[m * 2 * n + n + d] + ix.ape[m * n + d]).collect();
            let mx = lg.iter().cloned().fold(f32::MIN, f32::max);
            let z: f32 = lg.iter().map(|l| (l - mx).exp()).sum();
            *o = (0..kp).map(|m| (lg[m] - mx).exp() / z * open[m * 2 * n + d]).sum();
        }
        let mut out = vec![0f32; n];
        rms_norm_w(&pk, &ix.comp_norm, cfg.eps, &mut out);
        rope.apply(&mut out, pos + 1 - kp, false);
        pools.extend(out);
        open.clear();
    }
    sel
}

impl<'a> Hy4Model<'a> {
    pub fn new(g: &'a Gguf) -> Result<Self> {
        let cfg = Hy4Config::from_gguf(g)?;
        let hcw = |p: &str| -> Result<Hcw<'a>> {
            Ok(Hcw { f: Mat::load(g, &format!("{p}_fn.weight"))?, base: vec_f32(g, &format!("{p}_base.weight"))?, scale: vec_f32(g, &format!("{p}_scale.weight"))? })
        };
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let m = |n: &str| Mat::load(g, &format!("blk.{il}.{n}.weight"));
            let v = |n: &str| vec_f32(g, &format!("blk.{il}.{n}.weight"));
            let nh = cfg.n_head[il];
            let ffn = if il < cfg.n_dense {
                Ffn::Dense { gate: m("ffn_gate")?, up: m("ffn_up")?, down: m("ffn_down")? }
            } else {
                Ffn::Moe {
                    gate_inp: m("ffn_gate_inp")?,
                    bias: vec_f32(g, &format!("blk.{il}.exp_probs_b.bias"))?,
                    gexps: m("ffn_gate_exps")?,
                    uexps: m("ffn_up_exps")?,
                    dexps: m("ffn_down_exps")?,
                    routed_down: m("ffn_routed_down")?,
                    routed_up: m("ffn_routed_up")?,
                    gate_sh: m("ffn_gate_shexp")?,
                    up_sh: m("ffn_up_shexp")?,
                    down_sh: m("ffn_down_shexp")?,
                }
            };
            let engram = if cfg.engram_index(il).is_some() {
                Some(Engram { gate: v("engram_gate")?, hc_scale: v("engram_hc_scale")?, kv: m("engram_kv")?, kv_b: v("engram_kv_b")?, lookup_norm: v("engram_lookup_norm")?, pre_gamma: v("engram_pre_gamma")? })
            } else {
                None
            };
            let base = if cfg.swa[il] { cfg.rope_base_swa } else { cfg.rope_base };
            let idx = if cfg.kpool > 0 && g.tensors.contains_key(&format!("blk.{il}.indexer.attn_k.weight")) {
                anyhow::ensure!(!cfg.swa[il], "layer {il}: a sliding-window layer cannot own an indexer");
                Some(Indexer { k: m("indexer.attn_k")?, k_norm: v("indexer.k_norm")?, value: m("indexer_compressor_value")?, gate: m("indexer_compressor_gate")?,
                    ape: v("indexer_compressor_ape")?, comp_norm: v("indexer_compressor_norm")?, q_b: m("indexer.attn_q_b")?, proj: m("indexer.proj")? })
            } else {
                None
            };
            layers.push(Layer {
                idx,
                hc_attn: hcw(&format!("blk.{il}.hc_attn"))?,
                hc_ffn: hcw(&format!("blk.{il}.hc_ffn"))?,
                attn_norm: v("attn_norm")?,
                q_a: m("attn_q_a")?,
                q_a_norm: v("attn_q_a_norm")?,
                q_b: m("attn_q_b")?,
                kv_a: m("attn_kv_a_mqa")?,
                kv_a_norm: v("attn_kv_a_norm")?,
                k_b: Mat::load_groups(g, &format!("blk.{il}.attn_k_b.weight"), nh)?,
                v_b: Mat::load_groups(g, &format!("blk.{il}.attn_v_b.weight"), nh)?,
                sinks: v("attn_sinks")?,
                gate: m("attn_gate")?,
                out: m("attn_output")?,
                post_attn_norm: v("post_attention_norm")?,
                ffn_norm: v("ffn_norm")?,
                post_ffw_norm: v("post_ffw_norm")?,
                ffn,
                engram,
                rope: Rope::new(base, 1.0, 0.0, cfg.n_rot, 0, 0.0, 0.0),
                nh,
                swa: cfg.swa[il],
            });
        }
        Ok(Hy4Model {
            tok_embd: host_mat(g, "token_embd.weight")?,
            out_hc: hcw("output_hc")?,
            out_norm: vec_f32(g, "output_norm.weight")?,
            out_w: Mat::load(g, "output.weight")?,
            engram_tab: if cfg.engram.is_some() { Some(host_mat(g, "per_layer_token_embd.weight")?) } else { None },
            keys: vec![vec![]; cfg.n_layer],
            idx_open: vec![vec![]; cfg.n_layer],
            idx_pools: vec![vec![]; cfg.n_layer],
            cpu_share: std::cell::Cell::new(0.5),
            layers,
            cfg,
            toks: vec![],
            trace: std::env::var("STRATA_DS_TRACE").is_ok(),
        })
    }

    pub fn reset(&mut self) {
        for v in self.keys.iter_mut().chain(&mut self.idx_open).chain(&mut self.idx_pools) { v.clear(); }
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

    /// Hyper-connection mixes of the 4 streams: `w.f` on the RMS-normed flat state.
    fn hc_mixes(&self, x: &[f32], w: &Hcw) -> Vec<f32> {
        let mut flat = vec![0f32; x.len()];
        rms_norm(x, self.cfg.eps, &mut flat);
        let mut mixes = vec![0f32; w.f.rows];
        w.f.matvec(&flat, &mut mixes);
        mixes
    }

    /// Reduce weights of a sublayer or of the output head (`pre`).
    fn hc_pre(&self, mixes: &[f32], w: &Hcw) -> [f32; 4] {
        std::array::from_fn(|i| sigmoid(mixes[i] * w.scale[0] + w.base[i]) + self.cfg.hc_eps)
    }

    /// Reduced input of a sublayer and the per-stream gates its output is written back with.
    fn hc_in(&self, il: usize, x: &[f32], w: &Hcw, cur: &mut [f32]) -> [f32; 4] {
        self.hc_in_m(il, x, w, &self.hc_mixes(x, w), cur)
    }

    /// `hc_in` for every token of a prefill chunk, with the sublayer's input norm applied: (inputs, gates).
    fn hc_in_rows(&self, il: usize, xs: &[Vec<f32>], w: &Hcw, norm: &[f32]) -> (Vec<Vec<f32>>, Vec<[f32; 4]>) {
        let flats: Vec<Vec<f32>> = xs.iter().map(|x| {
            let mut f = vec![0f32; x.len()];
            rms_norm(x, self.cfg.eps, &mut f);
            f
        }).collect();
        let mixes = matvec_rows(&w.f, &flats.iter().map(|f| f.as_slice()).collect::<Vec<_>>());
        let mut cur = vec![0f32; norm.len()];
        xs.iter().zip(mixes.chunks_exact(w.f.rows)).map(|(x, m)| {
            let post = self.hc_in_m(il, x, w, m, &mut cur);
            let mut h = vec![0f32; norm.len()];
            rms_norm_w(&cur, norm, self.cfg.eps, &mut h);
            (h, post)
        }).unzip()
    }

    fn hc_in_m(&self, il: usize, x: &[f32], w: &Hcw, mixes: &[f32], cur: &mut [f32]) -> [f32; 4] {
        self.tr("hc_mixes", il as i32, mixes);
        let pre = self.hc_pre(mixes, w);
        self.tr("hc_pre", il as i32, &pre);
        let post: [f32; 4] = std::array::from_fn(|i| sigmoid(mixes[4 + i] * w.scale[1] + w.base[4 + i]) * self.cfg.hc_mag + self.cfg.hc_eps);
        self.tr("hc_post_gate", il as i32, &post);
        hc_collapse(x, &pre, cur);
        post
    }

    /// `attention` for the consecutive positions from `p0` of a prefill chunk: each projection is one GPU call for the
    /// whole chunk instead of one per position.
    fn attention_batch(&mut self, il: usize, curs: &[Vec<f32>], p0: usize) -> Vec<Vec<f32>> {
        let (cfg, l) = (&self.cfg, &self.layers[il]);
        let (nh, kl, kr, nr, eps, nt) = (l.nh, cfg.key_len, cfg.kv_rank, cfg.n_rot, cfg.eps, curs.len());
        let (nope, kw, nq) = (kl - nr, kr + nr, l.q_a.rows);
        let xs: Vec<&[f32]> = curs.iter().map(|c| c.as_slice()).collect();
        let rows = |m: &Mat, x: &[f32], w: usize| matvec_rows(m, &x.chunks_exact(w).collect::<Vec<_>>());
        let normed = |x: &[f32], w: &[f32]| -> Vec<f32> {
            let mut y = vec![0f32; x.len()];
            for (a, b) in x.chunks_exact(w.len()).zip(y.chunks_exact_mut(w.len())) { rms_norm_w(a, w, eps, b); }
            y
        };
        // A matrix of `nh` row groups applied to every token's [nh][w] input (a group reads the first `cols` of its
        // `w`), token-major: one call per head for a long chunk, one per token for a short one.
        let grouped = |m: &Mat<'a>, x: &[f32], w: usize| -> Vec<f32> {
            let gs: Vec<Mat> = (0..nh).map(|h| m.group(h, nh)).collect();
            if nt < nh { return x.chunks_exact(nh * w).flat_map(|xt| matvec_many(&gs, xt, w)).collect(); }
            let r = gs[0].rows;
            let mut y = vec![0f32; nt * nh * r];
            for (h, g) in gs.iter().enumerate() {
                let yh = matvec_rows(g, &x.chunks_exact(nh * w).map(|xt| &xt[h * w..][..g.cols]).collect::<Vec<_>>());
                for (t, yt) in yh.chunks_exact(r).enumerate() { y[(t * nh + h) * r..][..r].copy_from_slice(yt); }
            }
            y
        };
        let qr = normed(&matvec_rows(&l.q_a, &xs), &l.q_a_norm);
        let mut q = rows(&l.q_b, &qr, nq);
        for (t, qt) in q.chunks_exact_mut(nh * kl).enumerate() {
            for qh in qt.chunks_exact_mut(kl) { l.rope.apply(qh, p0 + t, false); }
        }
        for (t, kv) in matvec_rows(&l.kv_a, &xs).chunks_exact(kw).enumerate() {
            self.keys[il].extend(key_row(l, kv, p0 + t, kr, eps));
        }
        let sels: Vec<Option<Vec<usize>>> = match &l.idx {
            Some(ix) => {
                let (n, nw) = (ix.k_norm.len(), ix.proj.rows);
                let ik = normed(&matvec_rows(&ix.k, &xs), &ix.k_norm);
                let (v, g) = (rows(&ix.value, &ik, n), rows(&ix.gate, &ik, n));
                // The query side is needed only once more than `idx_top` groups are complete.
                let ask = self.idx_pools[il].len() / n + nt / cfg.kpool + 1 > cfg.idx_top;
                let (iq, iw) = if ask { (rows(&ix.q_b, &qr, nq), matvec_rows(&ix.proj, &xs)) } else { (vec![], vec![]) };
                (0..nt).map(|t| {
                    let vg = [&v[t * n..(t + 1) * n], &g[t * n..(t + 1) * n]].concat();
                    kpool_select(ix, &l.rope, cfg, &mut self.idx_open[il], &mut self.idx_pools[il], &vg, p0 + t,
                        || (iq[t * nw * n..(t + 1) * nw * n].to_vec(), iw[t * nw..(t + 1) * nw].to_vec()))
                }).collect()
            }
            None => vec![None; nt],
        };
        let qn = grouped(&l.k_b, &q, kl);
        let (scale, all) = (1.0 / (kl as f32).sqrt(), &self.keys[il]);
        let keys: Vec<Vec<&[f32]>> = (0..nt).map(|t| {
            let w0 = if l.swa { (p0 + t + 1).saturating_sub(cfg.n_swa) } else { 0 };
            key_rows(all, sels[t].as_deref(), p0 + t, w0, cfg.kpool, kw)
        }).collect();
        let mut att = vec![0f32; nt * nh * kr];
        att.par_chunks_mut(kr).enumerate().for_each(|(i, o)| {
            head_mix(&keys[i / nh], &qn[i * kr..(i + 1) * kr], &q[i * kl + nope..(i + 1) * kl], l.sinks[i % nh], scale, o);
        });
        let mut a = grouped(&l.v_b, &att, kr);
        for (a, g) in a.iter_mut().zip(&matvec_rows(&l.gate, &xs)) { *a *= sigmoid(*g); }
        let o = rows(&l.out, &a, a.len() / nt);
        normed(&o, &l.post_attn_norm).chunks_exact(l.out.rows).map(|r| r.to_vec()).collect()
    }

    fn engram(&self, il: usize, e: usize, pos: usize, x: &mut [f32]) {
        let (en, l, tab) = (self.cfg.engram.as_ref().unwrap(), self.layers[il].engram.as_ref().unwrap(), self.engram_tab.as_ref().unwrap());
        let n = self.cfg.n_embd;
        let cid = |t: u32| -> i64 { en.token_map.get(t as usize).map_or(-1, |&c| c as i64) };
        let c0 = cid(self.toks[pos]);
        // A token without a compressed id gets no Engram signal at all.
        if c0 >= 0 {
            let (ng, nh, hd) = (en.ngram, en.heads_per_ngram, en.head_dim);
            let ncol = (ng - 1) * nh;
            // Context tokens before the sequence start or without a compressed id read as the pad id.
            let mut toks = vec![en.token_map[en.pad_id] as u64; ng];
            toks[0] = c0 as u64;
            for s in 1..ng.min(pos + 1) {
                let c = cid(self.toks[pos - s]);
                if c >= 0 { toks[s] = c as u64; }
            }
            let mult = &en.multipliers[e * ng..];
            let mut rolling = toks[0].wrapping_mul(mult[0]);
            let mut rows = vec![0f32; ncol * hd];
            for o in 1..ng {
                rolling ^= toks[o].wrapping_mul(mult[o]);
                for h in 0..nh {
                    let col = (o - 1) * nh + h;
                    let id = rolling % en.primes[e * ncol + col] + en.offsets[e * ncol + col];
                    tab.row_into(id as usize, &mut rows[col * hd..(col + 1) * hd]);
                }
            }
            self.tr("engram_rows", il as i32, &rows);
            let mut r = vec![0f32; ncol * hd];
            for c in 0..ncol {
                let s = c * hd..(c + 1) * hd;
                rms_norm_w(&rows[s.clone()], &l.lookup_norm[s.clone()], ENGRAM_EPS, &mut r[s]);
            }
            self.tr("engram_rows_normed", il as i32, &r);
            // 4 keys (one per stream) then the value.
            let mut kv = vec![0f32; l.kv.rows];
            l.kv.matvec(&r, &mut kv);
            for (a, b) in kv.iter_mut().zip(&l.kv_b) { *a += b; }
            self.tr("engram_kv", il as i32, &kv);
            let mut signal = vec![0f32; n];
            rms_norm_w(&kv[4 * n..], &l.pre_gamma, ENGRAM_EPS, &mut signal);
            let (mut xn, mut kn) = (vec![0f32; n], vec![0f32; n]);
            for s in 0..4 {
                rms_norm(&x[s * n..(s + 1) * n], ENGRAM_EPS, &mut xn);
                rms_norm(&kv[s * n..(s + 1) * n], ENGRAM_EPS, &mut kn);
                let d = (0..n).map(|j| xn[j] * kn[j] * l.gate[s * n + j]).sum::<f32>() / (n as f32).sqrt();
                let gate = sigmoid(d.signum() * d.abs().max(1e-6).sqrt());
                for j in 0..n { x[s * n + j] += signal[j] * (gate + l.hc_scale[s * n + j]); }
            }
        }
        self.tr("engram_out", il as i32, x);
    }

    fn attention(&mut self, il: usize, cur: &[f32], pos: usize) -> Vec<f32> {
        let (cfg, l) = (&self.cfg, &self.layers[il]);
        let (nh, kl, kr, nr, eps) = (l.nh, cfg.key_len, cfg.kv_rank, cfg.n_rot, cfg.eps);
        let (nope, kw) = (kl - nr, kr + nr);
        // The three projections of the normed input share one GPU round trip.
        let y = matvec_many(&[l.q_a, l.kv_a, l.gate], cur, 0);
        let (qa, rest) = y.split_at(l.q_a.rows);
        let (kv, gate) = rest.split_at(kw);
        let mut qr = vec![0f32; qa.len()];
        rms_norm_w(qa, &l.q_a_norm, eps, &mut qr);
        self.tr("q_resid", il as i32, &qr);
        // Per head: `nope` no-position dims, then `nr` rotary dims.
        let mut q = vec![0f32; nh * kl];
        l.q_b.matvec(&qr, &mut q);
        for qh in q.chunks_exact_mut(kl) { l.rope.apply(qh, pos, false); }
        // One shared key/value row per position: normed compressed KV, then the rotated key part.
        let krow = key_row(l, kv, pos, kr, eps);
        self.tr("kv_cmpr", il as i32, &krow[..kr]);
        self.tr("k_pe", il as i32, &krow[kr..]);
        self.keys[il].extend_from_slice(&krow);
        // Absorb k_b into the query, so scores are taken against the compressed row directly.
        let kb: Vec<Mat> = (0..nh).map(|h| l.k_b.group(h, nh)).collect();
        let qn = matvec_many(&kb, &q, kl);
        let w0 = if l.swa { (pos + 1).saturating_sub(cfg.n_swa) } else { 0 };
        let sel = l.idx.as_ref().and_then(|ix| {
            let n = ix.k_norm.len();
            let (mut k, mut ik) = (vec![0f32; n], vec![0f32; n]);
            ix.k.matvec(cur, &mut k);
            rms_norm_w(&k, &ix.k_norm, eps, &mut ik);
            let vg = matvec_many(&[ix.value, ix.gate], &ik, 0);
            kpool_select(ix, &l.rope, cfg, &mut self.idx_open[il], &mut self.idx_pools[il], &vg, pos, || {
                let (mut q, mut w) = (vec![0f32; ix.q_b.rows], vec![0f32; ix.proj.rows]);
                ix.q_b.matvec(&qr, &mut q);
                ix.proj.matvec(cur, &mut w);
                (q, w)
            })
        });
        if let Some(groups) = &sel { self.tr("indexer_sel", il as i32, &groups.iter().map(|&g| g as f32).collect::<Vec<_>>()); }
        let keys = key_rows(&self.keys[il], sel.as_deref(), pos, w0, cfg.kpool, kw);
        let scale = 1.0 / (kl as f32).sqrt();
        let mut att = vec![0f32; nh * kr];
        att.par_chunks_mut(kr).enumerate().for_each(|(h, o)| {
            head_mix(&keys, &qn[h * kr..(h + 1) * kr], &q[h * kl + nope..(h + 1) * kl], l.sinks[h], scale, o);
        });
        let vb: Vec<Mat> = (0..nh).map(|h| l.v_b.group(h, nh)).collect();
        let mut a = matvec_many(&vb, &att, kr);
        self.tr("attn_kqv", il as i32, &a);
        for (a, g) in a.iter_mut().zip(gate) { *a *= sigmoid(*g); }
        self.tr("attn_gated", il as i32, &a);
        let mut o = vec![0f32; l.out.rows];
        l.out.matvec(&a, &mut o);
        let mut out = vec![0f32; o.len()];
        rms_norm_w(&o, &l.post_attn_norm, eps, &mut out);
        self.tr("attn_out", il as i32, &out);
        out
    }

    fn ffn(&self, il: usize, cur: &[f32]) -> Vec<f32> {
        let cfg = &self.cfg;
        match &self.layers[il].ffn {
            Ffn::Dense { gate, up, down } => {
                let y = matvec_many(&[*gate, *up], cur, 0);
                let a: Vec<f32> = (0..gate.rows).map(|i| silu(y[i]) * y[gate.rows + i]).collect();
                let mut out = vec![0f32; down.rows];
                down.matvec(&a, &mut out);
                out
            }
            Ffn::Moe { gate_inp, bias, gexps, uexps, dexps, routed_down, routed_up, gate_sh, up_sh, down_sh } => {
                // The router and the shared expert read the full-width input, the routed experts the latent one.
                let y = matvec_many(&[*gate_inp, *routed_down, *gate_sh, *up_sh], cur, 0);
                let (logits, rest) = y.split_at(cfg.n_expert);
                let (rin, sh) = rest.split_at(routed_down.rows);
                self.tr("ffn_moe_logits", il as i32, logits);
                let sel = route(logits, bias, cfg);
                if std::env::var_os("STRATA_DS_NOPREFETCH").is_none() {
                    for &(e, _) in &sel {
                        for m in [gexps, uexps, dexps] { m.prefetch_expert(e, cfg.n_expert); }
                    }
                }
                let ne = cfg.n_expert;
                let nl = dexps.rows / ne;
                // Experts resident in VRAM run on the GPU. Of the others, the CPU computes `cpu_share` from the mapped
                // file while the GPU uploads and runs the rest, both at once. The share follows the two
                // sides' times, so a machine with many cores or the model in RAM leans on the CPU and one with a fast
                // card on the GPU. `STRATA_DS_CPU_SHARE` fixes it; 0 (also `--fixed-experts`) runs every expert on
                // the GPU, replacing the least recently used VRAM slots: the same numerics on every run.
                let (share, fixed) = self.share();
                // A miss the GPU takes replaces the least recently used VRAM slot, so the expert is resident the next
                // time it is picked (`STRATA_DS_NOEVICT`: a copy per use instead). The upload is part of the GPU side.
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
                let clamp = cfg.clamp_exp[il];
                let timed = |list: &[(Mat, Mat, Mat, f32)]| { let t = std::time::Instant::now(); (run_experts(list, rin, clamp, nl), t.elapsed().as_secs_f32()) };
                let gpu_side = || {
                    let t = std::time::Instant::now();
                    let list: Vec<_> = on_gpu.iter().map(|&(g, e, w)| (if g.gpu.is_some() { g } else { on(gexps, e) }, on(uexps, e), on(dexps, e), w)).collect();
                    (run_experts(&list, rin, clamp, nl), t.elapsed().as_secs_f32())
                };
                let ((a, tg), (b, tc)) = rayon::join(gpu_side, || timed(&on_cpu));
                let mix: Vec<f32> = a.iter().zip(&b).map(|(a, b)| a + b).collect();
                if !fixed && n_miss > 0 { self.balance(tg, tc); }
                self.tr("ffn_moe_out", il as i32, &mix);
                let a: Vec<f32> = (0..gate_sh.rows).map(|i| silu(sh[i]) * sh[gate_sh.rows + i]).collect();
                let y = matvec_many(&[*routed_up, *down_sh], &[mix, a].concat(), nl.max(gate_sh.rows));
                let n = routed_up.rows;
                self.tr("ffn_moe_up", il as i32, &y[..n]);
                self.tr("ffn_shexp", il as i32, &y[n..]);
                (0..n).map(|j| y[j] + y[n + j]).collect()
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

    /// Move the share toward the side that finished first (`tg`, `tc`: seconds the GPU and the CPU side took).
    fn balance(&self, tg: f32, tc: f32) {
        if tg + tc > 0.0 { self.cpu_share.set((self.cpu_share.get() + 0.1 * (tg - tc) / (tg + tc)).clamp(0.0, 1.0)); }
    }

    /// `ffn` for the tokens of a prefill chunk. A routed expert is read once and applied to every token that picked
    /// it; the experts are split between the GPU and the CPU like the misses in `ffn`.
    fn ffn_batch(&self, il: usize, curs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let cfg = &self.cfg;
        let xs: Vec<&[f32]> = curs.iter().map(|c| c.as_slice()).collect();
        let split = |y: Vec<f32>, w: usize| -> Vec<Vec<f32>> { y.chunks_exact(w).map(|r| r.to_vec()).collect() };
        match &self.layers[il].ffn {
            Ffn::Dense { gate, up, down } => {
                let (gy, uy) = (matvec_rows(gate, &xs), matvec_rows(up, &xs));
                let a: Vec<f32> = gy.iter().zip(&uy).map(|(g, u)| silu(*g) * u).collect();
                split(matvec_rows(down, &a.chunks_exact(gate.rows).collect::<Vec<_>>()), down.rows)
            }
            Ffn::Moe { gate_inp, bias, gexps, uexps, dexps, routed_down, routed_up, gate_sh, up_sh, down_sh } => {
                let ne = cfg.n_expert;
                let (nf, nl, nin, clamp) = (gexps.rows / ne, dexps.rows / ne, routed_down.rows, cfg.clamp_exp[il]);
                let logits = matvec_rows(gate_inp, &xs);
                let rin = matvec_rows(routed_down, &xs);
                let sels: Vec<Vec<(usize, f32)>> = logits.chunks_exact(ne).map(|l| route(l, bias, cfg)).collect();
                // Per expert: the (token, rank in the token's selection) pairs that use it.
                let mut by_e: std::collections::BTreeMap<usize, Vec<(usize, usize)>> = Default::default();
                for (t, sel) in sels.iter().enumerate() {
                    for (k, &(e, _)) in sel.iter().enumerate() { by_e.entry(e).or_default().push((t, k)); }
                }
                if std::env::var_os("STRATA_DS_NOPREFETCH").is_none() {
                    for &e in by_e.keys() {
                        for m in [gexps, uexps, dexps] { m.prefetch_expert(e, ne); }
                    }
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
                let rows_of = |e: usize| -> Vec<&[f32]> { by_e[&e].iter().map(|&(t, _)| &rin[t * nin..(t + 1) * nin]).collect() };
                let gpu = || {
                    let t = std::time::Instant::now();
                    let out: Vec<(usize, Vec<f32>)> = on_gpu.iter().map(|(e, g, u, d)| {
                        let x = rows_of(*e);
                        let (gy, uy) = (matvec_rows(g, &x), matvec_rows(u, &x));
                        let a: Vec<f32> = gy.iter().zip(&uy).map(|(g, u)| act(*g, *u, clamp)).collect();
                        (*e, matvec_rows(d, &a.chunks_exact(nf).collect::<Vec<_>>()))
                    }).collect();
                    (out, t.elapsed().as_secs_f32())
                };
                let cpu = || {
                    let t = std::time::Instant::now();
                    let out: Vec<(usize, Vec<f32>)> = on_cpu.par_iter().map(|(e, g, u, d)| {
                        let x = rows_of(*e);
                        let (mut gy, mut uy, mut dy) = (vec![0f32; nf], vec![0f32; nf], vec![0f32; x.len() * nl]);
                        for (x, dy) in x.iter().zip(dy.chunks_exact_mut(nl)) {
                            g.matvec(x, &mut gy);
                            u.matvec(x, &mut uy);
                            for (g, u) in gy.iter_mut().zip(&uy) { *g = act(*g, *u, clamp); }
                            d.matvec(&gy, dy);
                        }
                        (*e, dy)
                    }).collect();
                    (out, t.elapsed().as_secs_f32())
                };
                let ((a, tg), (b, tc)) = rayon::join(gpu, cpu);
                if !fixed && n_miss > 0 { self.balance(tg, tc); }
                // Each token sums its experts in its own selection order, as `ffn` does.
                let mut ds: Vec<Vec<&[f32]>> = sels.iter().map(|s| vec![&[][..]; s.len()]).collect();
                for (e, dy) in a.iter().chain(&b) {
                    for (&(t, k), d) in by_e[e].iter().zip(dy.chunks_exact(nl)) { ds[t][k] = d; }
                }
                let mut mix = vec![0f32; curs.len() * nl];
                for (t, m) in mix.chunks_exact_mut(nl).enumerate() {
                    for (d, &(_, w)) in ds[t].iter().zip(&sels[t]) {
                        for j in 0..nl { m[j] += w * d[j]; }
                    }
                }
                let up = matvec_rows(routed_up, &mix.chunks_exact(nl).collect::<Vec<_>>());
                let (sg, su) = (matvec_rows(gate_sh, &xs), matvec_rows(up_sh, &xs));
                let a: Vec<f32> = sg.iter().zip(&su).map(|(g, u)| silu(*g) * u).collect();
                let sh = matvec_rows(down_sh, &a.chunks_exact(gate_sh.rows).collect::<Vec<_>>());
                split(up.iter().zip(&sh).map(|(a, b)| a + b).collect(), routed_up.rows)
            }
        }
    }

    /// Layer-major pass over a chunk of prompt tokens (no logits).
    fn prefill_chunk(&mut self, toks: &[u32]) {
        let t0 = std::time::Instant::now();
        let n = self.cfg.n_embd;
        let p0 = self.toks.len();
        self.toks.extend_from_slice(toks);
        let mut xs: Vec<Vec<f32>> = toks.iter().map(|&t| {
            let mut emb = vec![0f32; n];
            self.tok_embd.row_into(t as usize, &mut emb);
            emb.repeat(4)
        }).collect();
        let mut cur = vec![0f32; n];
        let add = |x: &mut [f32], y: &[f32], post: &[f32; 4]| {
            for s in 0..4 {
                for j in 0..n { x[s * n + j] += y[j] * post[s]; }
            }
        };
        let (mut t_attn, mut t_ffn) = (0f64, 0f64);
        for il in 0..self.cfg.n_layer {
            let t = std::time::Instant::now();
            if let Some(e) = self.cfg.engram_index(il) {
                for (i, x) in xs.iter_mut().enumerate() { self.engram(il, e, p0 + i, x); }
            }
            let (hs, posts) = self.hc_in_rows(il, &xs, &self.layers[il].hc_attn, &self.layers[il].attn_norm);
            for ((x, post), y) in xs.iter_mut().zip(&posts).zip(self.attention_batch(il, &hs, p0)) { add(x, &y, post); }
            t_attn += t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let (hs, posts) = self.hc_in_rows(il, &xs, &self.layers[il].hc_ffn, &self.layers[il].ffn_norm);
            for ((x, post), f) in xs.iter_mut().zip(&posts).zip(self.ffn_batch(il, &hs)) {
                rms_norm_w(&f, &self.layers[il].post_ffw_norm, self.cfg.eps, &mut cur);
                add(x, &cur, post);
            }
            t_ffn += t.elapsed().as_secs_f64();
        }
        if std::env::var_os("STRATA_DS_PROF").is_some() {
            eprintln!("PROF chunk of {} at {p0}: {:.0} ms (attn {:.0}, ffn {:.0}), cpu share {:.2}",
                toks.len(), t0.elapsed().as_secs_f64() * 1e3, t_attn * 1e3, t_ffn * 1e3, self.cpu_share.get());
        }
    }

    pub fn step(&mut self, tok: u32) -> Vec<f32> {
        self.step_opt(tok, true)
    }

    /// Run `toks` through the model; logits only for the last one. All but the last go in chunks of
    /// `STRATA_DS_CHUNK` tokens (default 1024; 1 or a tensor dump: token by token). A chunk reads each routed expert
    /// once, so on a machine whose RAM does not hold the experts a larger chunk means less disk traffic per token.
    pub fn prefill(&mut self, toks: &[u32]) -> Vec<f32> {
        let Some((&last, head)) = toks.split_last() else { return vec![] };
        let chunk: usize = std::env::var("STRATA_DS_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(1024);
        if chunk > 1 && !self.trace && std::env::var_os("STRATA_DS_DUMP").is_none() {
            for c in head.chunks(chunk) { self.prefill_chunk(c); }
        } else {
            for &t in head { self.step_opt(t, false); }
        }
        self.step_opt(last, true)
    }

    fn step_opt(&mut self, tok: u32, want: bool) -> Vec<f32> {
        let t0 = std::time::Instant::now();
        let n = self.cfg.n_embd;
        let pos = self.toks.len();
        self.toks.push(tok);
        let mut emb = vec![0f32; n];
        self.tok_embd.row_into(tok as usize, &mut emb);
        let mut x = emb.repeat(4);
        self.tr("hc_init", -1, &x);
        let (mut cur, mut h) = (vec![0f32; n], vec![0f32; n]);
        let (mut t_attn, mut t_ffn) = (0f64, 0f64);
        for il in 0..self.cfg.n_layer {
            if let Some(e) = self.cfg.engram_index(il) { self.engram(il, e, pos, &mut x); }
            let t = std::time::Instant::now();
            let post = self.hc_in(il, &x, &self.layers[il].hc_attn, &mut cur);
            rms_norm_w(&cur, &self.layers[il].attn_norm, self.cfg.eps, &mut h);
            self.tr("attn_norm", il as i32, &h);
            let att = self.attention(il, &h, pos);
            for s in 0..4 {
                for j in 0..n { x[s * n + j] += att[j] * post[s]; }
            }
            self.tr("hc_attn_out", il as i32, &x);
            t_attn += t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let post = self.hc_in(il, &x, &self.layers[il].hc_ffn, &mut cur);
            rms_norm_w(&cur, &self.layers[il].ffn_norm, self.cfg.eps, &mut h);
            self.tr("ffn_norm", il as i32, &h);
            let f = self.ffn(il, &h);
            rms_norm_w(&f, &self.layers[il].post_ffw_norm, self.cfg.eps, &mut cur);
            self.tr("ffn_out", il as i32, &cur);
            for s in 0..4 {
                for j in 0..n { x[s * n + j] += cur[j] * post[s]; }
            }
            self.tr("l_out", il as i32, &x);
            t_ffn += t.elapsed().as_secs_f64();
        }
        let mut logits = vec![];
        if want {
            let mixes = self.hc_mixes(&x, &self.out_hc);
            self.tr("hc_head_mixes", -1, &mixes);
            let pre = self.hc_pre(&mixes, &self.out_hc);
            self.tr("hc_head_pre", -1, &pre);
            hc_collapse(&x, &pre, &mut cur);
            self.tr("hc_head", -1, &cur);
            rms_norm_w(&cur, &self.out_norm, self.cfg.eps, &mut h);
            self.tr("result_norm", -1, &h);
            logits = vec![0f32; self.out_w.rows];
            self.out_w.matvec(&h, &mut logits);
        }
        if std::env::var_os("STRATA_DS_PROF").is_some() {
            use std::sync::atomic::Ordering::Relaxed;
            let ex = &crate::deepseek::mat::EX_LOOKUPS;
            eprintln!("PROF pos {pos}: {:.0} ms (attn {:.0}, ffn {:.0}), expert cache hit {} miss {}, cpu share {:.2}",
                t0.elapsed().as_secs_f64() * 1e3, t_attn * 1e3, t_ffn * 1e3, ex[0].swap(0, Relaxed), ex[1].swap(0, Relaxed), self.cpu_share.get());
        }
        logits
    }
}
