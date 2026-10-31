//! Chunked prefill: the prompt runs through each layer `chunk` tokens at a time with tensor-core GEMMs. Routed experts
//! run as grouped GEMMs over the chunk's (token, expert) pairs sorted by expert; experts not resident in the cache are
//! copied into a small staging area, a group of them at a time, each group's copy beside the previous group's GEMMs.

use super::{Ffn, GpuModel, Mixer};
use crate::backend::{Backend, Buf, GEMM_BM, SAME_X};
use crate::expert_cache::MATS;
use crate::model::Model;

/// Experts staged per group when a layer routes to experts outside the cache. Staging holds two groups.
const STAGE: usize = 16;

/// Float offsets into the scratch slab (each region holds `chunk` rows). Mixer and FFN regions alias.
struct Lay { h: usize, a: usize, b: usize, c: usize, d: usize, e: usize, rl: usize, wts: usize, ge: usize, ye: usize, shg: usize, shy: usize, sg: usize }

pub(super) struct Scratch {
    pub chunk: usize,
    lay: Lay,
    slab: Buf,
    /// ids | xrow | inv, `chunk * k` each.
    ints: Buf,
    tiles: Buf,
    staging: Buf,
}

impl Scratch {
    pub fn new(m: &Model, gpu: &dyn Backend, chunk: usize) -> anyhow::Result<Scratch> {
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
            slab: gpu.zeros::<f32>(total)?, ints: gpu.zeros::<i32>(3 * chunk * k)?, tiles: gpu.zeros::<u64>(2 * n_tiles)?,
            staging: gpu.zeros::<u8>(if c.is_moe() { 2 * STAGE * stage_bytes } else { 0 })?,
        })
    }

    /// Bytes of VRAM this scratch takes.
    pub fn bytes(&self) -> usize { self.slab.len() + self.ints.len() + self.tiles.len() + self.staging.len() }
}

impl GpuModel<'_> {
    /// Run `tokens` from the current position; returns the logits after the last one. Short prompts (and models
    /// loaded without a prefill chunk) go through the decode step.
    pub fn prefill(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
        let Some(chunk) = self.pf.as_ref().map(|p| p.chunk) else { return self.step_all(tokens) };
        if tokens.len() < self.short { return self.step_all(tokens); }
        anyhow::ensure!(self.pos + tokens.len() <= self.ctx, "context full ({} tokens; raise --ctx)", self.ctx);
        for ch in tokens.chunks(chunk) { self.prefill_chunk(ch)?; }
        // The prompt's routing predicts the reply's: place its hot experts now instead of 64 per adapt during decode.
        if self.adapt_every > 0 && let Some(cache) = self.cache.as_mut() { cache.adapt(self.m, &*self.gpu, usize::MAX)?; }
        let (g, ne) = (&self.gpu, self.m.cfg.n_embd);
        let (pf, r) = (self.pf.as_ref().unwrap(), (tokens.len() - 1) % chunk);
        let h = self.bufs["h"].addr();
        // Row 0 of `x` holds the last hidden state, as after `step` (the MTP draft reads it).
        g.dtod(&pf.slab, 4 * ne * r, &self.x, 0, 4 * ne)?;
        g.rms_norm(pf.slab.addr() + (4 * ne * r) as u64, 0, self.output_norm.ptr, h, 0, ne, 1, self.m.cfg.eps)?;
        g.gemv(&self.output, h, self.logits.addr())?;
        g.get::<f32>(&self.logits, 0, self.m.cfg.n_vocab)
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
        let pf = self.pf.as_ref().unwrap();
        let (g, t, n, eps, ne) = (&self.gpu, toks.len(), pf.chunk, c.eps, c.n_embd);
        let emb: Vec<f32> = toks.iter().flat_map(|&tk| self.m.embed(tk)).collect();
        g.put(&emb, &pf.slab, 0)?;
        g.put(&[self.pos as i32, 0], &self.step_buf, 0)?;
        let step = self.step_buf.addr();
        let base = pf.slab.addr();
        let at = |o: usize| base + 4 * o as u64;
        let l = &pf.lay;
        let (x, h) = (at(0), at(l.h));
        let (dk, nk, nv) = (c.ssm_d_state, c.ssm_n_group, c.ssm_dt_rank);
        let (dv, channels) = (c.ssm_dv(), c.ssm_d_inner + 2 * nk * dk);
        let (hd, nh, nkv) = (c.head_dim, c.n_head, c.n_head_kv);
        let kv = nkv * hd;
        for (il, ly) in self.layers.iter().enumerate() {
            g.rms_norm(x, ne, ly.attn_norm.ptr, h, ne, ne, t, eps)?;
            match &ly.mixer {
                Mixer::Delta { qkv, gate, beta, alpha, dt_bias, a, conv_w, norm, out, conv, s, .. } => {
                    let (eb, eg) = (at(l.e), at(l.e + n * nv));
                    g.gemm_dense(qkv, t, h, ne, at(l.a), channels, 0)?;
                    g.conv_seq(at(l.a), conv.addr(), conv_w.ptr, at(l.c), channels, c.ssm_conv, t, 0, 0)?;
                    // The beta and alpha GEMMs (16 blocks each) run beside the norm: batch launches alternate streams.
                    g.batch(&mut || {
                        g.gemm_dense(beta, t, h, ne, eb, nv, SAME_X)?;
                        g.l2_norm(at(l.c), dk, 2 * nk, channels, t * 2 * nk, eps)?;
                        g.gemm_dense(alpha, t, h, ne, eg, nv, SAME_X)
                    })?;
                    g.ssm_gates(eb, eg, dt_bias.ptr, a.ptr, t * nv, nv)?;
                    // The z GEMM runs beside the scan, on the SMs its last wave of blocks leaves idle.
                    g.batch(&mut || {
                        g.deltanet(at(l.c), s.addr(), eb, eg, 0, 0, at(l.d), nk, nv, c.k_interleave, dk, dv, 1.0 / (dk as f32).sqrt(), t, channels, 0, 0)?;
                        g.gemm_dense(gate, t, h, ne, at(l.b), c.ssm_d_inner, SAME_X)
                    })?;
                    g.norm_gate(at(l.d), norm.ptr, at(l.b), dv, t * nv, eps)?;
                    g.gemm_dense(out, t, at(l.d), c.ssm_d_inner, x, ne, 2)?;
                }
                Mixer::Attn { q, k, v, qn, kn, o, kc, vc } => {
                    let (kraw, kk, vv) = (at(l.c), at(l.c + n * kv), at(l.c + 2 * n * kv));
                    let qs = if c.q_gate { 2 * hd } else { hd };
                    g.gemm_dense(q, t, h, ne, at(l.a), nh * qs, 0)?;
                    g.rms_norm(at(l.a), qs, qn.ptr, at(l.b), hd, hd, t * nh, eps)?;
                    g.rope_neox(at(l.b), hd, t * nh, nh, c.n_rot, step, c.rope_base)?;
                    g.gemm_dense(k, t, h, ne, kraw, kv, SAME_X)?;
                    g.gemm_dense(v, t, h, ne, vv, kv, SAME_X)?;
                    g.rms_norm(kraw, hd, kn.ptr, kk, hd, hd, t * nkv, eps)?;
                    g.rope_neox(kk, hd, t * nkv, nkv, c.n_rot, step, c.rope_base)?;
                    g.kv_store(kk, vv, kc.addr(), vc.addr(), kv, t, step, hd)?;
                    g.attn_prefill(at(l.b), kc.addr(), vc.addr(), if c.q_gate { at(l.a) } else { 0 }, at(l.d), step, t, nh, nkv, hd, 1.0 / (hd as f32).sqrt())?;
                    g.gemm_dense(o, t, at(l.d), nh * hd, x, ne, 2)?;
                }
            }
            g.rms_norm(x, ne, ly.post_norm.ptr, h, ne, ne, t, eps)?;
            match &ly.ffn {
                Ffn::Dense { gate, up, down } => {
                    g.gemm_dense(gate, t, h, ne, at(l.ge), c.n_ff, 0)?;
                    g.gemm_dense(up, t, h, ne, at(l.ge), c.n_ff, 1 | SAME_X)?;
                    g.gemm_dense(down, t, at(l.ge), c.n_ff, x, ne, 2)?;
                }
                Ffn::Moe { router, shared } => {
                    let k = c.n_expert_used;
                    let ids_d = pf.ints.addr();
                    // The 1-row shared gate_inp GEMM runs beside the router GEMM (both float: one launch each).
                    g.batch(&mut || {
                        g.gemm_dense(router, t, h, ne, at(l.rl), c.n_expert, 0)?;
                        if let Some(s) = shared { g.gemm_dense(&s.gate_inp, t, h, ne, at(l.sg), 1, SAME_X)?; }
                        Ok(())
                    })?;
                    g.router_topk(at(l.rl), c.n_expert, k, c.expert_weights_scale, ids_d, at(l.wts), t)?;
                    let (sh, sg) = match shared {
                        Some(s) => {
                            g.gemm_dense(&s.gate, t, h, ne, at(l.shg), c.n_ff_shexp, SAME_X)?;
                            g.gemm_dense(&s.up, t, h, ne, at(l.shg), c.n_ff_shexp, 1 | SAME_X)?;
                            (at(l.shy), at(l.sg))
                        }
                        None => (0, 0),
                    };
                    let ids = g.get::<i32>(&pf.ints, 0, t * k)?;
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
                    g.put(&xrow, &pf.ints, n * k)?;
                    g.put(&inv, &pf.ints, 2 * n * k)?;
                    let lx = cache.layer(il).unwrap();
                    let (pool, stride) = cache.base(lx);
                    let used = |e: &usize| off[e + 1] > off[*e];
                    let missed: Vec<usize> = (0..c.n_expert).filter(used).filter(|&e| cache.slot(il, e) < 0).collect();
                    // Group 0: resident experts; then staged groups of at most STAGE missed experts. Staging holds
                    // each matrix in its own section (all gates, then ups, then downs), so experts with consecutive
                    // ids are contiguous there as in the weights and copy in one call.
                    let mut groups: Vec<Vec<(usize, [u64; 3])>> = vec![(0..c.n_expert).filter(used)
                        .filter_map(|e| { let s = cache.slot(il, e); (s >= 0).then(|| (e, lx.mats.each_ref().map(|m| pool + (s as usize * stride + m.off) as u64))) }).collect()];
                    let sp = pf.staging.addr();
                    let half = |j: usize| j % 2 * pf.staging.len() / 2;
                    for (j, grp) in missed.chunks(STAGE).enumerate() {
                        groups.push(grp.iter().enumerate().map(|(i, &e)| (e, lx.mats.each_ref().map(|m| sp + (half(j) + STAGE * m.off + i * m.bytes) as u64))).collect());
                    }
                    let mut tiles = Vec::new();
                    // Per group: [(start, count); 3] and the slot rows its experts cover. The down GEMM reads rows
                    // lo..hi of `ge` only, so its tiles count from lo and it quantizes just those rows.
                    let mut spans = Vec::new();
                    for grp in &groups {
                        let (mut sp, lo, hi) = ([(0, 0); 3], grp.first().map_or(0, |g| off[g.0]), grp.last().map_or(0, |g| off[g.0 + 1]));
                        for mi in 0..3 {
                            let start = tiles.len() / 2;
                            let base = if mi == 2 { lo } else { 0 };
                            for &(e, w) in grp {
                                for m0 in (off[e]..off[e + 1]).step_by(GEMM_BM) {
                                    let cnt = GEMM_BM.min(off[e + 1] - m0);
                                    tiles.extend([w[mi], (m0 - base) as u64 | (cnt as u64) << 32]);
                                }
                            }
                            sp[mi] = (start, tiles.len() / 2 - start);
                        }
                        spans.push((sp, lo, hi));
                    }
                    anyhow::ensure!(8 * tiles.len() <= pf.tiles.len(), "prefill tile list overflow");
                    g.put(&tiles, &pf.tiles, 0)?;
                    let tp = |s: usize| pf.tiles.addr() + 16 * s as u64;
                    let xrow_d = pf.ints.addr() + 4 * (n * k) as u64;
                    let [mg, mu, md] = lx.mats;
                    let copy = |gi: usize| -> anyhow::Result<()> {
                        let grp = &groups[gi];
                        for (name, mat) in MATS.iter().zip(&lx.mats) {
                            let w = self.m.w.layer(il, name)?;
                            let mut i = 0;
                            for run in grp.chunk_by(|a, b| b.0 == a.0 + 1) {
                                let e = run[0].0;
                                g.htod(&w.data[e * mat.bytes..(e + run.len()) * mat.bytes], &pf.staging, half(gi - 1) + STAGE * mat.off + i * mat.bytes)?;
                                i += run.len();
                            }
                        }
                        Ok(())
                    };
                    // Group gi + 1 copies into the staging half group gi - 1 used, beside group gi's up GEMM (the batch
                    // starts after group gi - 1's down GEMM).
                    for gi in 0..groups.len() {
                        let (sp, lo, hi) = spans[gi];
                        g.gemm(mg.ty, mg.rows, mg.cols, tp(sp[0].0), sp[0].1, h, ne, t, xrow_d, at(l.ge), c.n_ff_exp, SAME_X)?;
                        let up = || g.gemm(mu.ty, mu.rows, mu.cols, tp(sp[1].0), sp[1].1, h, ne, t, xrow_d, at(l.ge), c.n_ff_exp, 1 | SAME_X);
                        if gi + 1 < groups.len() { g.batch(&mut || { copy(gi + 1)?; up() })?; } else { up()?; }
                        g.gemm(md.ty, md.rows, md.cols, tp(sp[2].0), sp[2].1, at(l.ge) + (4 * lo * c.n_ff_exp) as u64, c.n_ff_exp, hi - lo, 0, at(l.ye) + (4 * lo * ne) as u64, ne, 0)?;
                    }
                    // The shared down runs last, so the routed gate and up above reuse the int8 h of the shared up.
                    if let Some(s) = shared { g.gemm_dense(&s.down, t, at(l.shg), c.n_ff_shexp, at(l.shy), ne, 0)?; }
                    g.moe_combine_seq(x, at(l.ye), at(l.wts), pf.ints.addr() + 4 * (2 * n * k) as u64, k, sh, sg, ne, t)?;
                }
            }
        }
        self.pos += t;
        Ok(())
    }
}
