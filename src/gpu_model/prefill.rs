//! Chunked prefill: the prompt runs through each layer `chunk` tokens at a time with tensor-core GEMMs. Routed experts
//! run as grouped GEMMs over the chunk's (token, expert) pairs sorted by expert; experts not resident in the cache are
//! copied into a small staging area, a group of them at a time.

use super::{Ffn, GpuModel, Mixer};
use crate::backend::Backend;
use crate::cuda::{GEMM_BM, Gpu};
use crate::expert_cache::MATS;
use crate::model::Model;
use cudarc::driver::CudaSlice;

/// Experts staged per group when a layer routes to experts outside the cache.
const STAGE: usize = 16;

/// Float offsets into the scratch slab (each region holds `chunk` rows). Mixer and FFN regions alias.
struct Lay { h: usize, a: usize, b: usize, c: usize, d: usize, e: usize, rl: usize, wts: usize, ge: usize, ye: usize, shg: usize, shy: usize, sg: usize }

pub(super) struct Scratch {
    pub chunk: usize,
    lay: Lay,
    slab: CudaSlice<f32>,
    /// ids | xrow | inv, `chunk * k` each.
    ints: CudaSlice<i32>,
    tiles: CudaSlice<u64>,
    staging: CudaSlice<u8>,
}

impl Scratch {
    pub fn new(m: &Model, gpu: &Gpu, chunk: usize) -> anyhow::Result<Scratch> {
        let c = &m.cfg;
        let mut stage_bytes = 0;
        for il in 0..c.n_layer {
            if !c.is_moe() { break; }
            let mut b = 0;
            for name in MATS { b += m.w.layer(il, name)?.expert(0, c.n_expert).data.len(); }
            stage_bytes = stage_bytes.max(b);
        }
        let (ne, k) = (c.n_embd, c.n_expert_used.max(1));
        let (nv, kv, channels) = (c.ssm_dt_rank, c.n_head_kv * c.head_dim, c.ssm_d_inner + 2 * c.ssm_n_group * c.ssm_d_state);
        let q = c.n_head * c.head_dim;
        let (h, base) = (ne * chunk, 2 * ne * chunk); // x at 0, then h
        let mut at = base;
        let mut take = |n: usize| { let o = at; at += n * chunk; o };
        let (a, b, c2, d, e) = (take((2 * q).max(channels)), take(q.max(c.ssm_d_inner)), take(channels.max(3 * kv)), take(q.max(c.ssm_d_inner)), take(2 * nv));
        let mix_end = at;
        at = base;
        let mut take = |n: usize| { let o = at; at += n * chunk; o };
        let (rl, wts, ge) = (take(c.n_expert), take(k), take((k * c.n_ff_exp).max(c.n_ff)));
        let (ye, shg, shy, sg) = (take(k * ne), take(c.n_ff_shexp), take(ne), take(1));
        let total = at.max(mix_end);
        let n_tiles = 3 * (chunk * k / GEMM_BM + 2 * c.n_expert);
        Ok(Scratch {
            chunk, lay: Lay { h, a, b, c: c2, d, e, rl, wts, ge, ye, shg, shy, sg },
            slab: gpu.zeros(total)?, ints: gpu.zeros(3 * chunk * k)?, tiles: gpu.zeros(2 * n_tiles)?,
            staging: gpu.zeros(if c.is_moe() { STAGE * stage_bytes } else { 0 })?,
        })
    }

    /// Bytes of VRAM this scratch takes.
    pub fn bytes(&self) -> usize { 4 * (self.slab.len() + self.ints.len()) + 8 * self.tiles.len() + self.staging.len() }
}

impl GpuModel<'_> {
    /// Run `tokens` from the current position; returns the logits after the last one. Short prompts (and models
    /// loaded without a prefill chunk) go through the decode step.
    pub fn prefill(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
        let Some(chunk) = self.pf.as_ref().map(|p| p.chunk) else { return self.step_all(tokens) };
        if tokens.len() < self.short { return self.step_all(tokens); }
        anyhow::ensure!(self.pos + tokens.len() <= self.ctx, "context full ({} tokens; raise --ctx)", self.ctx);
        for ch in tokens.chunks(chunk) { self.prefill_chunk(ch)?; }
        let (g, ne) = (&self.gpu, self.m.cfg.n_embd);
        let last = g.ptr(&self.pf.as_ref().unwrap().slab) + (4 * ne * ((tokens.len() - 1) % chunk)) as u64;
        let h = g.ptr(&self.bufs["h"]);
        // Row 0 of `x` holds the last hidden state, as after `step` (the MTP draft reads it).
        g.stream.memcpy_dtod(&self.pf.as_ref().unwrap().slab.slice(ne * ((tokens.len() - 1) % chunk)..ne * ((tokens.len() - 1) % chunk + 1)), &mut self.x.slice_mut(0..ne))?;
        g.rms_norm(last, 0, self.output_norm.ptr, h, 0, ne, 1, self.m.cfg.eps)?;
        g.gemv(&self.output, h, g.ptr(&self.logits))?;
        Ok(g.stream.clone_dtoh(&self.logits.slice(0..self.m.cfg.n_vocab))?)
    }

    /// A fresh sequence: `prefill(tokens)` from position 0, reusing earlier work. The state after `tokens[..keep]` is
    /// saved; when the saved tokens are a prefix of the next call's `tokens[..keep]`, that call starts from them
    /// (KV past them is rewritten, the recurrent state is copied back). Returns the logits and the tokens reused.
    pub fn prefill_cached(&mut self, tokens: &[u32], keep: usize) -> anyhow::Result<(Vec<f32>, usize)> {
        anyhow::ensure!(keep < tokens.len(), "prefill_cached: keep {keep} of {} tokens", tokens.len());
        let hit = self.saved_ids.len();
        let reused = if hit > 0 && tokens[..keep].starts_with(&self.saved_ids) {
            self.restore_state()?;
            if let Some(h) = self.mtp.as_mut() { h.pos = 0; h.pend.clear(); }
            hit
        } else {
            self.reset()?;
            0
        };
        if keep > reused {
            self.prefill(&tokens[reused..keep])?;
            self.save_state()?;
            self.saved_ids = tokens[..keep].to_vec();
        }
        Ok((self.prefill(&tokens[keep..])?, reused))
    }

    fn step_all(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
        let mut logits = Vec::new();
        for &t in tokens { logits = self.step(t)?; }
        Ok(logits)
    }

    fn prefill_chunk(&mut self, toks: &[u32]) -> anyhow::Result<()> {
        let c = &self.m.cfg;
        let pf = self.pf.as_mut().unwrap();
        let (g, t, n, eps, ne) = (&self.gpu, toks.len(), pf.chunk, c.eps, c.n_embd);
        let emb: Vec<f32> = toks.iter().flat_map(|&tk| self.m.embed(tk)).collect();
        g.stream.memcpy_htod(&emb, &mut pf.slab.slice_mut(0..t * ne))?;
        g.stream.memcpy_htod(&[self.pos as i32, 0], &mut self.step_buf)?;
        let step = g.ptr(&self.step_buf);
        let base = g.ptr(&pf.slab);
        let at = |o: usize| base + 4 * o as u64;
        let l = &pf.lay;
        let (x, h) = (at(0), at(l.h));
        let (dk, nk, nv) = (c.ssm_d_state, c.ssm_n_group, c.ssm_dt_rank);
        let (dv, channels) = (c.ssm_d_inner / nv, c.ssm_d_inner + 2 * nk * dk);
        let (hd, nh, nkv) = (c.head_dim, c.n_head, c.n_head_kv);
        let kv = nkv * hd;
        for (il, ly) in self.layers.iter().enumerate() {
            g.rms_norm(x, ne, ly.attn_norm.ptr, h, ne, ne, t, eps)?;
            match &ly.mixer {
                Mixer::Delta { qkv, gate, beta, alpha, dt_bias, a, conv_w, norm, out, conv, s, .. } => {
                    let (eb, eg) = (at(l.e), at(l.e + n * nv));
                    g.gemm_dense(qkv, t, h, ne, at(l.a), channels, 0)?;
                    g.gemm_dense(gate, t, h, ne, at(l.b), c.ssm_d_inner, 0)?;
                    g.gemm_dense(beta, t, h, ne, eb, nv, 0)?;
                    g.gemm_dense(alpha, t, h, ne, eg, nv, 0)?;
                    g.ssm_gates(eb, eg, dt_bias.ptr, a.ptr, t * nv, nv)?;
                    g.conv_seq(at(l.a), g.ptr(conv), conv_w.ptr, at(l.c), channels, c.ssm_conv, t, 0, 0)?;
                    g.l2_norm(at(l.c), dk, 2 * nk, channels, t * 2 * nk, eps)?;
                    g.deltanet(at(l.c), g.ptr(s), eb, eg, at(l.d), nk, nv, dk, dv, 1.0 / (dk as f32).sqrt(), t, channels, 0, 0)?;
                    g.norm_gate(at(l.d), norm.ptr, at(l.b), dv, t * nv, eps)?;
                    g.gemm_dense(out, t, at(l.d), c.ssm_d_inner, x, ne, 2)?;
                }
                Mixer::Attn { q, k, v, qn, kn, o, kc, vc } => {
                    let (kraw, kk, vv) = (at(l.c), at(l.c + n * kv), at(l.c + 2 * n * kv));
                    g.gemm_dense(q, t, h, ne, at(l.a), 2 * nh * hd, 0)?;
                    g.rms_norm(at(l.a), 2 * hd, qn.ptr, at(l.b), hd, hd, t * nh, eps)?;
                    g.rope_neox(at(l.b), hd, t * nh, nh, c.n_rot, step, c.rope_base)?;
                    g.gemm_dense(k, t, h, ne, kraw, kv, 0)?;
                    g.gemm_dense(v, t, h, ne, vv, kv, 0)?;
                    g.rms_norm(kraw, hd, kn.ptr, kk, hd, hd, t * nkv, eps)?;
                    g.rope_neox(kk, hd, t * nkv, nkv, c.n_rot, step, c.rope_base)?;
                    g.kv_store(kk, vv, g.ptr(kc), g.ptr(vc), kv, t, step, hd)?;
                    g.attn_prefill(at(l.b), g.ptr(kc), g.ptr(vc), at(l.a), at(l.d), step, t, nh, nkv, hd, 1.0 / (hd as f32).sqrt())?;
                    g.gemm_dense(o, t, at(l.d), nh * hd, x, ne, 2)?;
                }
            }
            g.rms_norm(x, ne, ly.post_norm.ptr, h, ne, ne, t, eps)?;
            match &ly.ffn {
                Ffn::Dense { gate, up, down } => {
                    g.gemm_dense(gate, t, h, ne, at(l.ge), c.n_ff, 0)?;
                    g.gemm_dense(up, t, h, ne, at(l.ge), c.n_ff, 1)?;
                    g.gemm_dense(down, t, at(l.ge), c.n_ff, x, ne, 2)?;
                }
                Ffn::Moe { router, shared } => {
                    let k = c.n_expert_used;
                    let ids_d = g.ptr(&pf.ints);
                    g.gemm_dense(router, t, h, ne, at(l.rl), c.n_expert, 0)?;
                    g.router_topk(at(l.rl), c.n_expert, k, c.expert_weights_scale, ids_d, at(l.wts), t)?;
                    let (sh, sg) = match shared {
                        Some(s) => {
                            g.gemm_dense(&s.gate_inp, t, h, ne, at(l.sg), 1, 0)?;
                            g.gemm_dense(&s.gate, t, h, ne, at(l.shg), c.n_ff_shexp, 0)?;
                            g.gemm_dense(&s.up, t, h, ne, at(l.shg), c.n_ff_shexp, 1)?;
                            g.gemm_dense(&s.down, t, at(l.shg), c.n_ff_shexp, at(l.shy), ne, 0)?;
                            (at(l.shy), at(l.sg))
                        }
                        None => (0, 0),
                    };
                    let ids = g.stream.clone_dtoh(&pf.ints.slice(0..t * k))?;
                    let cache = self.cache.as_mut().unwrap();
                    cache.record(il, &ids);
                    // Sort (token, j) pairs by expert: slot s holds token xrow[s]; inv maps t * k + j to its slot.
                    let mut off = vec![0usize; c.n_expert + 1];
                    for &e in &ids { off[e as usize + 1] += 1; }
                    for e in 0..c.n_expert { off[e + 1] += off[e]; }
                    let (mut fill, mut xrow, mut inv) = (off.clone(), vec![0i32; t * k], vec![0i32; t * k]);
                    for (i, &e) in ids.iter().enumerate() {
                        let s = &mut fill[e as usize];
                        xrow[*s] = (i / k) as i32;
                        inv[i] = *s as i32;
                        *s += 1;
                    }
                    g.stream.memcpy_htod(&xrow, &mut pf.ints.slice_mut(n * k..n * k + t * k))?;
                    g.stream.memcpy_htod(&inv, &mut pf.ints.slice_mut(2 * n * k..2 * n * k + t * k))?;
                    let lx = cache.layer(il).unwrap();
                    let (pool, stride) = cache.base(lx);
                    let used = |e: &usize| off[e + 1] > off[*e];
                    let missed: Vec<usize> = (0..c.n_expert).filter(used).filter(|&e| cache.slot(il, e) < 0).collect();
                    // Group 0: resident experts; then staged groups of at most STAGE missed experts.
                    let mut groups: Vec<Vec<(usize, u64)>> = vec![(0..c.n_expert).filter(used)
                        .filter_map(|e| { let s = cache.slot(il, e); (s >= 0).then(|| (e, pool + (s as usize * stride) as u64)) }).collect()];
                    let sp = g.ptr(&pf.staging);
                    for grp in missed.chunks(STAGE) {
                        groups.push(grp.iter().enumerate().map(|(i, &e)| (e, sp + (i * stride) as u64)).collect());
                    }
                    let mut tiles = Vec::new();
                    let mut spans = Vec::new(); // per group: [(start, count); 3]
                    for grp in &groups {
                        let mut sp = [(0, 0); 3];
                        for (mi, mat) in lx.mats.iter().enumerate() {
                            let start = tiles.len() / 2;
                            for &(e, w) in grp {
                                for m0 in (off[e]..off[e + 1]).step_by(GEMM_BM) {
                                    let cnt = GEMM_BM.min(off[e + 1] - m0);
                                    tiles.extend([w + mat.off as u64, m0 as u64 | (cnt as u64) << 32]);
                                }
                            }
                            sp[mi] = (start, tiles.len() / 2 - start);
                        }
                        spans.push(sp);
                    }
                    anyhow::ensure!(tiles.len() <= pf.tiles.len(), "prefill tile list overflow");
                    g.stream.memcpy_htod(&tiles, &mut pf.tiles.slice_mut(0..tiles.len()))?;
                    let tp = |s: usize| g.ptr(&pf.tiles) + 16 * s as u64;
                    let xrow_d = g.ptr(&pf.ints) + 4 * (n * k) as u64;
                    let [mg, mu, md] = lx.mats;
                    for (gi, grp) in groups.iter().enumerate() {
                        if gi > 0 {
                            // ponytail: synchronous pageable copies, not overlapped with compute; pinned double
                            // buffering on a copy stream if prefill of cold layers matters.
                            for (i, &(e, _)) in grp.iter().enumerate() {
                                for (name, mat) in MATS.iter().zip(&lx.mats) {
                                    let src = self.m.w.layer(il, name)?.expert(e, c.n_expert);
                                    let o = i * stride + mat.off;
                                    g.stream.memcpy_htod(src.data, &mut pf.staging.slice_mut(o..o + mat.bytes))?;
                                }
                            }
                        }
                        let sp = spans[gi];
                        g.gemm(mg.ty, mg.rows, mg.cols, tp(sp[0].0), sp[0].1, h, ne, t, xrow_d, at(l.ge), c.n_ff_exp, 0)?;
                        g.gemm(mu.ty, mu.rows, mu.cols, tp(sp[1].0), sp[1].1, h, ne, t, xrow_d, at(l.ge), c.n_ff_exp, 1)?;
                        g.gemm(md.ty, md.rows, md.cols, tp(sp[2].0), sp[2].1, at(l.ge), c.n_ff_exp, t * k, 0, at(l.ye), ne, 0)?;
                    }
                    g.moe_combine_seq(x, at(l.ye), at(l.wts), g.ptr(&pf.ints) + 4 * (2 * n * k) as u64, k, sh, sg, ne, t)?;
                }
            }
        }
        self.pos += t;
        Ok(())
    }
}
