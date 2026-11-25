//! Kolibri 1.1 forward, mirroring llama.cpp `src/models/kolibri1.cpp` (trace names are its callback names, so
//! `STRATA_DS_DUMP` output lines up with `llama-eval-callback`). Decode is one token at a time; prefill runs a chunk
//! of prompt tokens layer by layer, so a routed expert is read once for all the tokens that picked it.
//! ponytail: attention scores and the indexer run on the CPU; a GPU kernel is the next step for prompt speed.
use super::{KolibriConfig, N_RECENT, N_SINK};
use crate::deepseek::mat::{Mat, matvec_many, matvec_rows, matvec_rows_many, swiglu_chunk, vec_f32};
use crate::deepseek::ops::rms_norm_w;
use crate::gguf::Gguf;
use crate::hy4::forward::{host_mat, run_experts, sigmoid, silu};
use crate::inkling2::forward::{dot_h, dots_h, mix_h};
use anyhow::Result;
use half::f16;
use half::slice::HalfFloatSliceExt;
use rayon::prelude::*;

enum Ffn<'a> {
    Dense { gate: Mat<'a>, up: Mat<'a>, down: Mat<'a> },
    Moe { gate_inp: Mat<'a>, bias: Vec<f32>, gexps: Mat<'a>, uexps: Mat<'a>, dexps: Mat<'a>, gate_sh: Mat<'a>, up_sh: Mat<'a>, down_sh: Mat<'a> },
}

struct Layer<'a> {
    attn_norm: Vec<f32>,
    q: Mat<'a>,
    q_norm: Vec<f32>,
    /// K and V projections and the K norm; None on a layer that reads the K/V of another layer.
    kv: Option<(Mat<'a>, Mat<'a>, Vec<f32>)>,
    out: Mat<'a>,
    post_attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    post_ffw_norm: Vec<f32>,
    ffn: Ffn<'a>,
}

pub struct KolibriModel<'a> {
    pub cfg: KolibriConfig,
    tok_embd: Mat<'a>,
    out_norm: Vec<f32>,
    out_w: Mat<'a>,
    layers: Vec<Layer<'a>>,
    /// Per layer with its own K/V and per K/V head: one f16 row of `head_dim` per position from `first[il]` on, as
    /// llama.cpp keeps them. Sliding layers: K rotated; sparse layers: K and V rounded to NVFP4.
    k: Vec<Vec<Vec<f16>>>,
    v: Vec<Vec<Vec<f16>>>,
    /// Position of row 0 (a sliding layer drops the rows no later query sees).
    first: Vec<usize>,
    /// Per sparse layer with its own K/V: the rounded mean routing key of each complete block of positions.
    blocks: Vec<Vec<f32>>,
    /// Share of a layer's VRAM-cache misses that the CPU computes while the GPU runs the rest (see `ffn`).
    cpu_share: std::cell::Cell<f32>,
    pub toks: Vec<u32>,
    pub trace: bool,
}

/// The E4M3 value nearest to `x` > 0 (ties to even), as ggml encodes and decodes the NVFP4 block scale.
fn e4m3(x: f32) -> f32 {
    if x >= 448.0 { return 448.0; }
    if x < 0.015625 { return (x * 512.0).round_ties_even() / 512.0; }
    let b = x.to_bits();
    let frac = f32::from_bits((b & 0x7f_ffff) | 0x3f80_0000) - 1.0;
    ((1.0 + (frac * 8.0).round_ties_even() / 8.0) * 2f32.powi((b >> 23) as i32 - 127)).min(448.0)
}

/// Nearest E2M1 value, ties to even, saturated at 6.
fn e2m1(x: f32) -> f32 {
    let a = x.abs();
    let r: f32 = if a <= 0.25 { 0.0 } else if a < 0.75 { 0.5 } else if a <= 1.25 { 1.0 } else if a < 1.75 { 1.5 }
        else if a <= 2.5 { 2.0 } else if a < 3.5 { 3.0 } else if a <= 5.0 { 4.0 } else { 6.0 };
    r.copysign(x)
}

/// ggml `round_nvfp4`: every block of 16 values becomes E2M1 values times one E4M3 scale.
pub(crate) fn round_nvfp4(x: &mut [f32]) {
    for g in x.chunks_exact_mut(16) {
        let amax = g.iter().fold(0f32, |m, v| m.max(v.abs()));
        let scale = e4m3((amax / 6.0).max(0.001953125));
        for v in g { *v = e2m1(*v / scale) * scale; }
    }
}

/// (cos, sin) of a position's rotations, with ggml's running product for the angle.
fn rope_tab(pos: usize, n: usize, base: f32) -> Vec<(f32, f32)> {
    let scale = base.powf(-2.0 / n as f32);
    let mut theta = pos as f32;
    (0..n / 2).map(|_| {
        let cs = (theta.cos(), theta.sin());
        theta *= scale;
        cs
    }).collect()
}

/// NeoX RoPE of one head: dimension i pairs with i + n/2.
fn rope(x: &mut [f32], tab: &[(f32, f32)]) {
    let h = tab.len();
    for (i, &(c, s)) in tab.iter().enumerate() {
        let (a, b) = (x[i], x[i + h]);
        x[i] = a * c - b * s;
        x[i + h] = a * s + b * c;
    }
}

/// The routed experts of one token, best first: the bias only picks, the weight is the sigmoid of the logit.
fn route(logits: &[f32], bias: &[f32], n_used: usize) -> Vec<(usize, f32)> {
    let mut sel: Vec<usize> = (0..logits.len()).collect();
    sel.sort_by(|&a, &b| (logits[b] + bias[b]).total_cmp(&(logits[a] + bias[a])).then(a.cmp(&b)));
    sel.truncate(n_used);
    sel.into_iter().map(|e| (e, sigmoid(logits[e]))).collect()
}

/// The `n` best of `scored` (score, index): highest score first, the lower index on a tie. Returns the indices.
fn top_n(mut scored: Vec<(f32, u32)>, n: usize) -> Vec<u32> {
    if scored.len() > n {
        scored.select_nth_unstable_by(n - 1, |a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        scored.truncate(n);
    }
    scored.into_iter().map(|s| s.1).collect()
}

/// The positions a sparse query at `pos` attends, ascending, or None for all of them. `score` rates a position's
/// routing key and `block_score` a block's mean key. A query attends its first `N_SINK` and last `N_RECENT`
/// positions and the `top_k` best of the candidates between them. On an indexer layer the candidates of a long
/// context are narrowed to the `pool_blocks` best complete blocks first; `pool` carries them to the sparse layers
/// that follow (None: every candidate).
fn select(cfg: &KolibriConfig, indexer: bool, pos: usize, pool: &mut Option<Vec<u32>>, score: impl Fn(u32) -> f32 + Sync, block_score: impl Fn(u32) -> f32 + Sync) -> Option<Vec<u32>> {
    if pos < N_SINK + N_RECENT + cfg.top_k { return None; }
    let last = (pos - N_RECENT) as u32;
    if indexer {
        let live = (pos + 1) / cfg.block;
        *pool = (live > cfg.pool_blocks).then(|| {
            let mut b = top_n((0..live as u32).into_par_iter().map(|j| (block_score(j), j)).collect(), cfg.pool_blocks);
            b.sort_unstable();
            b.into_iter().flat_map(|j| j * cfg.block as u32..(j + 1) * cfg.block as u32).filter(|&p| p >= N_SINK as u32 && p <= last).collect()
        });
    }
    let scored: Vec<(f32, u32)> = match pool {
        Some(p) if p.len() <= cfg.top_k => p.iter().map(|&c| (0.0, c)).collect(),
        Some(p) => p.par_iter().map(|&c| (score(c), c)).collect(),
        None => (N_SINK as u32..=last).into_par_iter().map(|c| (score(c), c)).collect(),
    };
    let mut picks = top_n(scored, cfg.top_k);
    picks.sort_unstable();
    Some((0..N_SINK as u32).chain(picks).chain(last + 1..=pos as u32).collect())
}

/// Softmax attention of `n` query heads `q` over the f16 key rows `k` with values `v`, into `o` (zeroed).
fn head_group(q: &[f32], k: &[f16], v: &[f16], hd: usize, scale: f32, o: &mut [f32]) {
    let (nk, n) = (k.len() / hd, q.len() / hd);
    let mut p = vec![0f32; n * nk];
    dots_h(&mut p, scale, k, q, nk);
    for p in p.chunks_exact_mut(nk) {
        let mx = p.iter().cloned().fold(f32::MIN, f32::max);
        let mut z = 0f32;
        for s in p.iter_mut() {
            *s = (*s - mx).exp();
            z += *s;
        }
        for s in p.iter_mut() { *s /= z; }
    }
    mix_h(o, &p, v, nk);
}

impl<'a> KolibriModel<'a> {
    pub fn new(g: &'a Gguf) -> Result<Self> {
        let cfg = KolibriConfig::from_gguf(g)?;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let m = |n: &str| Mat::load(g, &format!("blk.{il}.{n}.weight"));
            let v = |n: &str| vec_f32(g, &format!("blk.{il}.{n}.weight"));
            let ffn = if il < cfg.n_dense {
                Ffn::Dense { gate: m("ffn_gate")?, up: m("ffn_up")?, down: m("ffn_down")? }
            } else {
                Ffn::Moe {
                    gate_inp: m("ffn_gate_inp")?,
                    bias: vec_f32(g, &format!("blk.{il}.exp_probs_b.bias"))?,
                    gexps: m("ffn_gate_exps")?,
                    uexps: m("ffn_up_exps")?,
                    dexps: m("ffn_down_exps")?,
                    gate_sh: m("ffn_gate_shexp")?,
                    up_sh: m("ffn_up_shexp")?,
                    down_sh: m("ffn_down_shexp")?,
                }
            };
            layers.push(Layer {
                attn_norm: v("attn_norm")?,
                q: m("attn_q")?,
                q_norm: v("attn_q_norm")?,
                kv: if cfg.kv_src[il] == il { Some((m("attn_k")?, m("attn_v")?, v("attn_k_norm")?)) } else { None },
                out: m("attn_output")?,
                post_attn_norm: v("post_attention_norm")?,
                ffn_norm: v("ffn_norm")?,
                post_ffw_norm: v("post_ffw_norm")?,
                ffn,
            });
        }
        let per_head = vec![vec![vec![]; cfg.n_head_kv]; cfg.n_layer];
        Ok(KolibriModel {
            tok_embd: host_mat(g, "token_embd.weight")?,
            out_norm: vec_f32(g, "output_norm.weight")?,
            out_w: Mat::load(g, "output.weight")?,
            k: per_head.clone(),
            v: per_head,
            first: vec![0; cfg.n_layer],
            blocks: vec![vec![]; cfg.n_layer],
            cpu_share: std::cell::Cell::new(0.5),
            layers,
            cfg,
            toks: vec![],
            trace: std::env::var("STRATA_DS_TRACE").is_ok(),
        })
    }

    pub fn reset(&mut self) {
        for h in self.k.iter_mut().chain(&mut self.v).flatten() { h.clear(); }
        for b in &mut self.blocks { b.clear(); }
        self.first.fill(0);
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

    /// Attention output (before its norm) of the consecutive positions from `p0`. `pools` holds each position's key
    /// pool from the last indexer layer.
    fn attention(&mut self, il: usize, curs: &[Vec<f32>], p0: usize, pools: &mut [Option<Vec<u32>>]) -> Vec<f32> {
        let cfg = &self.cfg;
        let l = &self.layers[il];
        let (nh, nkv, hd, eps, nt, swa, src) = (cfg.n_head, cfg.n_head_kv, cfg.head_dim, cfg.eps, curs.len(), cfg.swa[il], cfg.kv_src[il]);
        let xs: Vec<&[f32]> = curs.iter().map(|c| c.as_slice()).collect();
        // One token: the projections share a GPU round trip.
        let mats: Vec<Mat> = std::iter::once(l.q).chain(l.kv.iter().flat_map(|kv| [kv.0, kv.1])).collect();
        let mut proj: Vec<Vec<f32>> = if nt == 1 {
            let y = matvec_many(&mats, xs[0], 0);
            let mut at = 0;
            mats.iter().map(|m| { at += m.rows; y[at - m.rows..at].to_vec() }).collect()
        } else {
            mats.iter().map(|m| matvec_rows(m, &xs)).collect()
        };
        let tabs: Vec<Vec<(f32, f32)>> = if swa { (0..nt).map(|t| rope_tab(p0 + t, hd, cfg.rope_base)).collect() } else { vec![] };
        let mut q = std::mem::take(&mut proj[0]);
        let norm_heads = |x: &mut [f32], w: &[f32], per_tok: usize| {
            let mut y = vec![0f32; hd];
            for (i, h) in x.chunks_exact_mut(hd).enumerate() {
                rms_norm_w(h, w, eps, &mut y);
                h.copy_from_slice(&y);
                if swa { rope(h, &tabs[i / per_tok]); }
            }
        };
        if let Some((_, _, k_norm)) = &l.kv {
            let (mut k, mut v) = (std::mem::take(&mut proj[1]), std::mem::take(&mut proj[2]));
            norm_heads(&mut k, k_norm, nkv);
            if nt == 1 { self.tr(if swa { "Kcur_rope" } else { "Kcur_normed" }, il as i32, &k); }
            if !swa {
                round_nvfp4(&mut k);
                round_nvfp4(&mut v);
                if nt == 1 {
                    self.tr("Kcur_nvfp4", il as i32, &k);
                    self.tr("Vcur_nvfp4", il as i32, &v);
                }
            }
            for (rows, store) in [(&k, &mut self.k[il]), (&v, &mut self.v[il])] {
                for (i, h) in rows.chunks_exact(hd).enumerate() { store[i % nkv].extend(h.iter().map(|&x| f16::from_f32(x))); }
            }
            if !swa {
                // The mean routing key of every block that is complete now, summed member by member as ggml does.
                let (k0, bs) = (&self.k[il][0], cfg.block);
                for j in self.blocks[il].len() / hd..k0.len() / hd / bs {
                    let mut mean = vec![0f32; hd];
                    for row in k0[j * bs * hd..(j + 1) * bs * hd].chunks_exact(hd) {
                        for (m, x) in mean.iter_mut().zip(row) { *m += x.to_f32(); }
                    }
                    for m in &mut mean { *m *= 1.0 / bs as f32; }
                    round_nvfp4(&mut mean);
                    self.blocks[il].extend(mean);
                }
            }
        }
        norm_heads(&mut q, &l.q_norm, nh);
        if nt == 1 { self.tr(if swa { "Qcur_rope" } else { "Qcur_normed" }, il as i32, &q); }
        let (ks, vs, first, blocks) = (&self.k[src], &self.v[src], self.first[src], &self.blocks[src]);
        let (per, scale) = (nh / nkv, 1.0 / (hd as f32).sqrt());
        // Sparse layers: the keys each query attends (None: all of them).
        let sels: Vec<Option<Vec<u32>>> = if swa { vec![None; nt] } else {
            let w = 1.0 / (per as f32 * (hd as f32).sqrt());
            pools.par_iter_mut().enumerate().map(|(t, pool)| {
                let rated = |key: &[f16], qr: &Vec<f32>| qr.chunks_exact(hd).map(|qh| dot_h(key, qh).max(0.0) * w).sum::<f32>();
                let route_q = || { let mut r = q[t * nh * hd..][..per * hd].to_vec(); round_nvfp4(&mut r); r };
                // Made only when the context is longer than the keys a query attends.
                let r = std::sync::OnceLock::new();
                select(cfg, cfg.indexer[il], p0 + t, pool,
                    |c| rated(&ks[0][c as usize * hd..][..hd], r.get_or_init(route_q)),
                    |j| { let key: Vec<f16> = blocks[j as usize * hd..][..hd].iter().map(|&x| f16::from_f32(x)).collect(); rated(&key, r.get_or_init(route_q)) })
            }).collect()
        };
        if nt == 1 && let Some(s) = &sels[0] { self.tr("kq_sel", il as i32, &s.iter().map(|&p| p as f32).collect::<Vec<_>>()); }
        // A chunk whose queries all see a run of consecutive keys (a sliding layer, or a sparse one before it selects) runs on the GPU.
        let gpu = nt > 4 && std::env::var_os("STRATA_KOL_CPU_ATTN").is_none();
        // Rows `from..` of a K or V store as the GPU cache: the K/V heads of a position side by side.
        let rows = |hs: &Vec<Vec<f16>>, from: usize| -> Vec<u16> {
            let mut o = Vec::with_capacity((p0 + nt - from) * nkv * hd);
            for r in from - first..p0 + nt - first {
                for h in hs { o.extend_from_slice(h[r * hd..(r + 1) * hd].reinterpret_cast()); }
            }
            o
        };
        if gpu && sels.iter().all(|s| s.is_none()) {
            let from = if swa { (p0 + 1).saturating_sub(cfg.n_swa).max(first) } else { 0 };
            if let Some(att) = crate::deepseek::mat::attn_rows(&q, &rows(ks, from), &rows(vs, from), p0 - from, nt, nh, nkv, hd, if swa { cfg.n_swa } else { 0 }) {
                return matvec_rows(&l.out, &att.chunks_exact(nh * hd).collect::<Vec<_>>());
            }
        }
        // A sparse chunk with selections: each query brings its list of keys (all the earlier ones when it has none).
        if gpu && !swa {
            let ns = sels.iter().enumerate().map(|(t, s)| s.as_ref().map_or(p0 + t + 1, |s| s.len())).max().unwrap();
            let mut idx = vec![0u32; nt * ns];
            let cnt: Vec<i32> = sels.iter().zip(idx.chunks_exact_mut(ns)).enumerate().map(|(t, (s, row))| match s {
                Some(s) => { row[..s.len()].copy_from_slice(s); s.len() as i32 }
                None => { for (i, x) in row[..p0 + t + 1].iter_mut().enumerate() { *x = i as u32; } (p0 + t + 1) as i32 }
            }).collect();
            if let Some(att) = crate::deepseek::mat::attn_sel(&q, &rows(ks, first), &rows(vs, first), &idx, &cnt, nt, nh, nkv, hd) {
                return matvec_rows(&l.out, &att.chunks_exact(nh * hd).collect::<Vec<_>>());
            }
        }
        // One task per token and K/V head; the query heads of a K/V head go three at a time (`dots_h`).
        let group = if per % 3 == 0 { 3 } else { 1 };
        let mut att = vec![0f32; nt * nh * hd];
        att.par_chunks_mut(per * hd).enumerate().for_each(|(i, o)| {
            let (t, g) = (i / nkv, i % nkv);
            let pos = p0 + t;
            let (gk, gv): (Vec<f16>, Vec<f16>);
            let (k, v) = match &sels[t] {
                Some(sel) => {
                    let gather = |rows: &[f16]| sel.iter().flat_map(|&p| &rows[p as usize * hd..][..hd]).copied().collect();
                    (gk, gv) = (gather(&ks[g]), gather(&vs[g]));
                    (&gk[..], &gv[..])
                }
                None => {
                    let from = if swa { (pos + 1).saturating_sub(cfg.n_swa).max(first) } else { 0 };
                    let r = (from - first) * hd..(pos + 1 - first) * hd;
                    (&ks[g][r.clone()], &vs[g][r])
                }
            };
            let qt = &q[i * per * hd..(i + 1) * per * hd];
            for (qh, o) in qt.chunks(group * hd).zip(o.chunks_mut(group * hd)) { head_group(qh, k, v, hd, scale, o); }
        });
        if nt == 1 { self.tr("kqv_out", il as i32, &att); }
        matvec_rows(&l.out, &att.chunks_exact(nh * hd).collect::<Vec<_>>())
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
            Ffn::Moe { gate_inp, bias, gexps, uexps, dexps, gate_sh, up_sh, down_sh } => {
                let ne = cfg.n_expert;
                // The shared expert rides with the routed ones on the GPU (weight 1); a tensor dump keeps it apart.
                let apart = self.trace || std::env::var_os("STRATA_DS_DUMP").is_some() || crate::deepseek::mat::dev_info().is_none();
                let y = if apart { matvec_many(&[*gate_inp, *gate_sh, *up_sh], cur, 0) } else { matvec_many(&[*gate_inp], cur, 0) };
                let (logits, sh) = y.split_at(ne);
                self.tr("ffn_moe_logits", il as i32, logits);
                let sel = route(logits, bias, cfg.n_used);
                if std::env::var_os("STRATA_DS_NOPREFETCH").is_none() {
                    for &(e, _) in &sel {
                        for m in [gexps, uexps, dexps] { m.prefetch_expert(e, ne); }
                    }
                }
                let nl = dexps.rows / ne;
                // Experts resident in VRAM run on the GPU. Of the others, the CPU computes `cpu_share` from the mapped
                // file while the GPU uploads and runs the rest, both at once; the share follows the two sides' times.
                // `STRATA_DS_CPU_SHARE` fixes it; 0 (also `--fixed-experts`) runs every expert on the GPU, replacing
                // the least recently used VRAM slots: the same numerics on every run.
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
                let cpu_side = || { let t = std::time::Instant::now(); (run_experts(&on_cpu, cur, 0.0, nl), t.elapsed().as_secs_f32()) };
                let gpu_side = || {
                    let t = std::time::Instant::now();
                    let mut list: Vec<_> = on_gpu.iter().map(|&(g, e, w)| (if g.gpu.is_some() { g } else { on(gexps, e) }, on(uexps, e), on(dexps, e), w)).collect();
                    if !apart { list.push((*gate_sh, *up_sh, *down_sh, 1.0)); }
                    (run_experts(&list, cur, 0.0, nl), t.elapsed().as_secs_f32())
                };
                let ((a, tg), (b, tc)) = rayon::join(gpu_side, cpu_side);
                if !fixed && n_miss > 0 { self.balance(tg, tc); }
                let mix: Vec<f32> = a.iter().zip(&b).map(|(a, b)| a + b).collect();
                if !apart { return mix; }
                self.tr("ffn_moe_out", il as i32, &mix);
                let a: Vec<f32> = (0..gate_sh.rows).map(|i| silu(sh[i]) * sh[gate_sh.rows + i]).collect();
                let mut s = vec![0f32; down_sh.rows];
                down_sh.matvec(&a, &mut s);
                self.tr("ffn_shexp_out", il as i32, &s);
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

    /// Move the share toward the side that finished first (`tg`, `tc`: seconds the GPU and the CPU side took).
    fn balance(&self, tg: f32, tc: f32) {
        if tg + tc > 0.0 { self.cpu_share.set((self.cpu_share.get() + 0.1 * (tg - tc) / (tg + tc)).clamp(0.0, 1.0)); }
    }

    /// `ffn` for the tokens of a prefill chunk, token-major. A routed expert is read once and applied to every token
    /// that picked it; the experts are split between the GPU and the CPU like the misses in `ffn`.
    fn ffn_batch(&self, il: usize, curs: &[Vec<f32>]) -> Vec<f32> {
        let cfg = &self.cfg;
        let xs: Vec<&[f32]> = curs.iter().map(|c| c.as_slice()).collect();
        let swiglu = |gate: &Mat, up: &Mat, down: &Mat, x: &[&[f32]]| -> Vec<f32> {
            let (gy, uy) = (matvec_rows(gate, x), matvec_rows(up, x));
            let a: Vec<f32> = gy.iter().zip(&uy).map(|(g, u)| silu(*g) * u).collect();
            matvec_rows(down, &a.chunks_exact(gate.rows).collect::<Vec<_>>())
        };
        match &self.layers[il].ffn {
            Ffn::Dense { gate, up, down } => swiglu(gate, up, down, &xs),
            Ffn::Moe { gate_inp, bias, gexps, uexps, dexps, gate_sh, up_sh, down_sh } => {
                let ne = cfg.n_expert;
                let (nf, nl) = (gexps.rows / ne, dexps.rows / ne);
                let logits = matvec_rows(gate_inp, &xs);
                let sels: Vec<Vec<(usize, f32)>> = logits.par_chunks_exact(ne).map(|l| route(l, bias, cfg.n_used)).collect();
                // Per expert: the (token, rank in the token's selection) pairs that use it.
                let mut by_e: std::collections::BTreeMap<usize, Vec<(usize, usize)>> = Default::default();
                for (t, sel) in sels.iter().enumerate() {
                    for (k, &(e, _)) in sel.iter().enumerate() { by_e.entry(e).or_default().push((t, k)); }
                }
                if std::env::var_os("STRATA_DS_NOPREFETCH").is_none() {
                    // In parallel: a read-ahead call waits for the drive's queue.
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
                    // One round trip for all the experts: the per-token weighted sums come back.
                    // The shared expert rides along as one more expert of every token, with weight 1, in slot `n_used`.
                    if std::env::var_os("STRATA_KOL_NOCHUNK").is_none() {
                        let sh = [gate_sh, up_sh, down_sh];
                        let col = |i: usize| -> Vec<Mat> { on_gpu.iter().map(|g| [g.1, g.2, g.3][i]).chain([*sh[i]]).collect() };
                        let mut pairs: Vec<Vec<(usize, usize, f32)>> = on_gpu.iter().map(|(e, ..)| by_e[e].iter().map(|&(t, k)| (t, k, sels[t][k].1)).collect()).collect();
                        pairs.push((0..xs.len()).map(|t| (t, cfg.n_used, 1.0)).collect());
                        if let Some(sum) = swiglu_chunk(&col(0), &col(1), &col(2), &pairs, &xs.concat(), cfg.n_used + 1) {
                            return (vec![], Some(sum), t.elapsed().as_secs_f32());
                        }
                    }
                    // Else two batches for all the experts (gates and ups, then downs).
                    let n_of = |e: &usize| by_e[e].len();
                    let mut x = Vec::with_capacity(2 * on_gpu.iter().map(|(e, ..)| n_of(e)).sum::<usize>() * cfg.n_embd);
                    for (e, ..) in &on_gpu {
                        for _ in 0..2 { for &(t, _) in &by_e[e] { x.extend_from_slice(xs[t]); } }
                    }
                    let y = matvec_rows_many(&on_gpu.iter().flat_map(|(e, g, u, _)| [(*g, n_of(e)), (*u, n_of(e))]).collect::<Vec<_>>(), &x);
                    let mut act = Vec::with_capacity(y.len() / 2);
                    let mut at = 0;
                    for (e, ..) in &on_gpu {
                        let n = n_of(e) * nf;
                        act.extend(y[at..at + n].iter().zip(&y[at + n..at + 2 * n]).map(|(g, u)| silu(*g) * u));
                        at += 2 * n;
                    }
                    let dy = matvec_rows_many(&on_gpu.iter().map(|(e, _, _, d)| (*d, n_of(e))).collect::<Vec<_>>(), &act);
                    let mut at = 0;
                    let out: Vec<(usize, Vec<f32>)> = on_gpu.iter().map(|(e, ..)| { at += n_of(e) * nl; (*e, dy[at - n_of(e) * nl..at].to_vec()) }).collect();
                    (out, None, t.elapsed().as_secs_f32())
                };
                let cpu = || {
                    let t = std::time::Instant::now();
                    // Each weight row is read once for all the tokens of its expert (`matvec_rows`).
                    let out: Vec<(usize, Vec<f32>)> = on_cpu.par_iter().map(|(e, g, u, d)| (*e, swiglu(g, u, d, &rows_of(*e)))).collect();
                    (out, t.elapsed().as_secs_f32())
                };
                let ((a, gsum, tg), (b, tc)) = rayon::join(gpu, cpu);
                if !fixed && n_miss > 0 { self.balance(tg, tc); }
                // Each token sums its experts in its own selection order, as `ffn` does; the shared expert is added last.
                let mut ds: Vec<Vec<&[f32]>> = sels.iter().map(|s| vec![&[][..]; s.len()]).collect();
                for (e, dy) in a.iter().chain(&b) {
                    for (&(t, k), d) in by_e[e].iter().zip(dy.chunks_exact(nl)) { ds[t][k] = d; }
                }
                let mut mix = if gsum.is_some() { vec![0f32; xs.len() * nl] } else { swiglu(gate_sh, up_sh, down_sh, &xs) };
                for (t, m) in mix.chunks_exact_mut(nl).enumerate() {
                    let mut sum = vec![0f32; nl];
                    // An expert of the device's chunk has no row here: its weighted output is in `gsum`.
                    for (d, &(_, w)) in ds[t].iter().zip(&sels[t]).filter(|(d, _)| !d.is_empty()) {
                        for j in 0..nl { sum[j] += w * d[j]; }
                    }
                    if let Some(g) = &gsum { for j in 0..nl { sum[j] += g[t * nl + j]; } }
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
    /// `STRATA_DS_CHUNK` tokens (default 1024, or 1 with a tensor dump, which covers only the tokens that run alone). A chunk reads each routed expert
    /// once, so on a machine whose RAM does not hold the experts a larger chunk means less disk traffic per token.
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
        let normed = |xs: &[Vec<f32>], w: &[f32]| -> Vec<Vec<f32>> {
            xs.iter().map(|x| { let mut h = vec![0f32; n]; rms_norm_w(x, w, eps, &mut h); h }).collect()
        };
        let mut pools: Vec<Option<Vec<u32>>> = vec![None; nt];
        let mut cur = vec![0f32; n];
        let (mut t_attn, mut t_ffn) = (0f64, 0f64);
        for il in 0..self.cfg.n_layer {
            let t = std::time::Instant::now();
            let hs = normed(&xs, &self.layers[il].attn_norm);
            if nt == 1 { self.tr("attn_norm", il as i32, &hs[0]); }
            let att = self.attention(il, &hs, p0, &mut pools);
            if nt == 1 { self.tr("attn_out", il as i32, &att); }
            for (x, a) in xs.iter_mut().zip(att.chunks_exact(n)) {
                rms_norm_w(a, &self.layers[il].post_attn_norm, eps, &mut cur);
                for j in 0..n { x[j] += cur[j]; }
            }
            if nt == 1 { self.tr("ffn_inp", il as i32, &xs[0]); }
            t_attn += t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let hs = normed(&xs, &self.layers[il].ffn_norm);
            if nt == 1 { self.tr("ffn_norm", il as i32, &hs[0]); }
            let f = if nt == 1 { self.ffn(il, &hs[0]) } else { self.ffn_batch(il, &hs) };
            for (x, f) in xs.iter_mut().zip(f.chunks_exact(n)) {
                rms_norm_w(f, &self.layers[il].post_ffw_norm, eps, &mut cur);
                for j in 0..n { x[j] += cur[j]; }
            }
            if nt == 1 { self.tr("l_out", il as i32, &xs[0]); } else if self.trace { self.tr("l_out", il as i32, &xs.concat()); }
            t_ffn += t.elapsed().as_secs_f64();
        }
        // A sliding layer keeps the rows its window still reaches.
        let (hd, keep) = (self.cfg.head_dim, self.cfg.n_swa);
        for il in (0..self.cfg.n_layer).filter(|&il| self.cfg.swa[il]) {
            let rows = self.k[il][0].len() / hd;
            if rows > 4 * keep {
                for h in self.k[il].iter_mut().chain(&mut self.v[il]) { h.drain(..(rows - keep) * hd); }
                self.first[il] += rows - keep;
            }
        }
        let mut logits = vec![];
        if want {
            rms_norm_w(&xs[nt - 1], &self.out_norm, eps, &mut cur);
            self.tr("result_norm", -1, &cur);
            logits = vec![0f32; self.out_w.rows];
            self.out_w.matvec(&cur, &mut logits);
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
    fn nvfp4_rounding_follows_ggml() {
        // Scales: the subnormal floor, a tie in the mantissa (to even), a carry into the exponent, saturation.
        for (x, want) in [(0.001953125, 0.001953125), (0.01, 5.0 / 512.0), (1.0625, 1.0), (1.1875, 1.25), (1.96875, 2.0), (1000.0, 448.0), (0.3, 0.3125)] {
            assert_eq!(e4m3(x), want, "e4m3({x})");
        }
        for (x, want) in [(0.25, 0.0), (0.26, 0.5), (-0.75, -1.0), (1.25, 1.0), (1.75, 2.0), (2.5, 2.0), (3.5, 4.0), (5.0, 4.0), (5.1, 6.0), (-9.0, -6.0)] {
            assert_eq!(e2m1(x), want, "e2m1({x})");
        }
        // amax 6 gives scale 1: the values round on the E2M1 grid; a block of zeros stays zero.
        let mut x = [6.0, -5.1, 3.4, 0.7, 0.2, 1.3, 2.6, -1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        round_nvfp4(&mut x);
        assert_eq!(x[..8], [6.0, -6.0, 3.0, 0.5, 0.0, 1.5, 3.0, -1.5]);
        assert!(x[8..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn rope_is_neox() {
        let mut x = [1.0, 2.0, 0.0, 0.0];
        rope(&mut x, &rope_tab(1, 4, 10000.0));
        let (c1, s1) = (0.01f32.cos(), 0.01f32.sin());
        assert!((x[0] - 1f32.cos()).abs() < 1e-6 && (x[2] - 1f32.sin()).abs() < 1e-6);
        assert!((x[1] - 2.0 * c1).abs() < 1e-6 && (x[3] - 2.0 * s1).abs() < 1e-6);
    }

    /// The GPU chunk attention against `head_group` on the same f16 rows, with a window and keys from before the
    /// chunk. Skips without a GPU (`STRATA_DS_GPU=1` opens one).
    #[test]
    fn gpu_window_attention_matches_cpu() {
        let (nt, nh, nkv, hd, win, base) = (40usize, 6usize, 2usize, 128usize, 7usize, 5usize);
        let mut seed = 1u32;
        let mut rnd = |n: usize| -> Vec<f32> { (0..n).map(|_| { seed = seed.wrapping_mul(1664525).wrapping_add(1013904223); (seed >> 8) as f32 / (1 << 23) as f32 - 1.0 }).collect() };
        let (q, k, v) = (rnd(nt * nh * hd), rnd((base + nt) * nkv * hd), rnd((base + nt) * nkv * hd));
        let bits = |x: &[f32]| -> Vec<u16> { x.iter().map(|&x| f16::from_f32(x).to_bits()).collect() };
        let Some(got) = crate::deepseek::mat::attn_rows(&q, &bits(&k), &bits(&v), base, nt, nh, nkv, hd, win) else { eprintln!("no GPU attention, skipping"); return };
        let (per, scale) = (nh / nkv, 1.0 / (hd as f32).sqrt());
        let mut worst = 0f32;
        for t in 0..nt {
            for h in 0..nh {
                let g = h / per;
                let rows = |x: &[f32]| -> Vec<f16> { ((base + t + 1).saturating_sub(win)..=base + t).flat_map(|p| x[(p * nkv + g) * hd..][..hd].iter().map(|&x| f16::from_f32(x))).collect() };
                let mut o = vec![0f32; hd];
                head_group(&q[(t * nh + h) * hd..][..hd], &rows(&k), &rows(&v), hd, scale, &mut o);
                for (a, b) in o.iter().zip(&got[(t * nh + h) * hd..][..hd]) { worst = worst.max((a - b).abs()); }
            }
        }
        assert!(worst < 5e-3, "GPU window attention differs from the CPU by {worst}");
    }

    /// The GPU attention over key lists against `head_group` on the gathered rows: lists of different lengths, one
    /// shorter than a tile of 32 keys. Skips without a GPU (`STRATA_DS_GPU=1` opens one).
    #[test]
    fn gpu_list_attention_matches_cpu() {
        let (nt, nh, nkv, hd, np) = (9usize, 6usize, 2usize, 128usize, 300usize);
        let mut seed = 7u32;
        let mut next = || { seed = seed.wrapping_mul(1664525).wrapping_add(1013904223); seed >> 8 };
        let mut rnd = |n: usize| -> Vec<f32> { (0..n).map(|_| next() as f32 / (1 << 23) as f32 - 1.0).collect() };
        let (q, k, v) = (rnd(nt * nh * hd), rnd(np * nkv * hd), rnd(np * nkv * hd));
        let lens = [1usize, 5, 31, 32, 33, 64, 100, 257, 300];
        let ns = 300;
        let mut idx = vec![0u32; nt * ns];
        for (t, row) in idx.chunks_exact_mut(ns).enumerate() {
            for (i, x) in row[..lens[t]].iter_mut().enumerate() { *x = ((i * 37 + t * 11) % np) as u32; }
        }
        let cnt: Vec<i32> = lens.iter().map(|&n| n as i32).collect();
        let bits = |x: &[f32]| -> Vec<u16> { x.iter().map(|&x| f16::from_f32(x).to_bits()).collect() };
        let Some(got) = crate::deepseek::mat::attn_sel(&q, &bits(&k), &bits(&v), &idx, &cnt, nt, nh, nkv, hd) else { eprintln!("no GPU attention, skipping"); return };
        let (per, scale) = (nh / nkv, 1.0 / (hd as f32).sqrt());
        let mut worst = 0f32;
        for t in 0..nt {
            for h in 0..nh {
                let g = h / per;
                let rows = |x: &[f32]| -> Vec<f16> { idx[t * ns..][..lens[t]].iter().flat_map(|&p| x[(p as usize * nkv + g) * hd..][..hd].iter().map(|&x| f16::from_f32(x))).collect() };
                let mut o = vec![0f32; hd];
                head_group(&q[(t * nh + h) * hd..][..hd], &rows(&k), &rows(&v), hd, scale, &mut o);
                for (a, b) in o.iter().zip(&got[(t * nh + h) * hd..][..hd]) { worst = worst.max((a - b).abs()); }
            }
        }
        assert!(worst < 1e-4, "GPU list attention differs from the CPU by {worst}");
    }

    fn cfg(top_k: usize, pool_blocks: usize) -> KolibriConfig {
        KolibriConfig { n_layer: 1, n_ctx: 0, n_embd: 0, n_vocab: 0, n_head: 0, n_head_kv: 0, head_dim: 0, rope_base: 0.0, eps: 0.0, n_expert: 0, n_used: 0,
            n_dense: 0, swa: vec![], n_swa: 0, indexer: vec![], kv_src: vec![], top_k, block: 16, pool_blocks }
    }

    #[test]
    fn sparse_selection() {
        let c = cfg(4, 3);
        let none = |_: u32| -> f32 { panic!("not scored") };
        // Short context: every key, nothing is scored.
        assert_eq!(select(&c, true, 139, &mut None, none, none), None);
        // Sinks, the 4 best candidates (ties: the lower position), the recent keys. Candidates are 8..=22 here.
        let score = |p: u32| if p == 22 || p == 9 { 2.0 } else if p == 21 { -1.0 } else { 1.0 };
        let mut pool = None;
        let sel = select(&cfg(4, 9), true, 150, &mut pool, score, none).unwrap();
        assert_eq!(pool, None);
        assert_eq!(sel[..12], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 22]);
        assert_eq!(sel[12..], (23..=150).collect::<Vec<u32>>()[..]);
        // Long context: 13 complete blocks at position 208; the 3 best blocks (1, 3 and, by the tie, 0) hold the pool.
        let block = |j: u32| if j == 3 || j == 1 { 5.0 } else if j == 12 { -1.0 } else { 0.0 };
        let sel = select(&c, true, 208, &mut pool, |p| p as f32, block).unwrap();
        let want: Vec<u32> = (8..32).chain(48..64).collect();
        assert_eq!(pool.as_deref(), Some(&want[..]));
        assert_eq!(sel[8..12], [60, 61, 62, 63]);
        // A sparse layer that is not an indexer selects from the pool it is given.
        let sel = select(&c, false, 208, &mut pool, |p| -(p as f32), none).unwrap();
        assert_eq!(sel[8..12], [8, 9, 10, 11]);
        assert_eq!(sel.len(), 8 + 4 + 128);
    }

    #[test]
    fn routing_adds_the_bias_only_to_pick() {
        let r = route(&[0.0, 2.0, 1.0, 1.0], &[5.0, 0.0, 0.0, 0.0], 3);
        assert_eq!(r.iter().map(|x| x.0).collect::<Vec<_>>(), [0, 1, 2]);
        assert_eq!(r[0].1, 0.5);
    }
}
