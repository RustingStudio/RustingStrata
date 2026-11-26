//! Chunked prefill: the prompt runs through each layer `chunk` tokens at a time with tensor-core GEMMs. Routed experts
//! run as grouped GEMMs over the chunk's (token, expert) pairs sorted by expert; experts not resident in the cache are
//! copied into a small staging area, a group of them at a time, each group's copy beside the previous group's GEMMs.

use super::{Ffn, GpuModel, HcAt, Mixer};
use crate::cpu_worker::TMAX;
use crate::backend::{BATCHED, Backend, Buf, GEMM_BM, SAME_X};
use crate::expert_cache::MATS;
use crate::model::Model;
use crate::weights::Prefetch;

/// Experts staged per group when a layer routes to experts outside the cache. Staging holds three groups.
const STAGE: usize = 16;

/// Float offsets into the scratch slab (each region holds `chunk` rows). Mixer and FFN regions alias, and so do the
/// hyper-connection read and PLE temporaries (`xn` .. `pv`), which run outside both.
struct Lay { h: usize, a: usize, b: usize, c: usize, d: usize, e: usize, rl: usize, wts: usize, ge: usize, ye: usize, shg: usize, shy: usize, sg: usize, hr: usize,
             xn: usize, lo: usize, hg: usize, pk: usize, pq: usize, pg: usize, pn: usize, pv: usize, inj: usize, mix: usize, mo: usize, emb: usize }

pub(super) struct Scratch {
    pub chunk: usize,
    lay: Lay,
    slab: Buf,
    /// `[chunk, 3 * n_embd]` hidden states captured before EAGLE3 `extract_layers`.
    feat: Option<Buf>,
    /// `[chunk, n_ids * n_embd]` post-layer hidden states captured for DFlash `target_layer_ids`.
    dflash_feat: Option<Buf>,
    /// `[chunk, n_embd]` hc stream mean (hc targets), input to the DFlash capture.
    dflash_mean: Option<Buf>,
    /// ids | xrow | inv, `chunk * k` each.
    ints: Buf,
    tiles: Buf,
    staging: Buf,
}

impl Scratch {
    pub fn new(m: &Model, gpu: &dyn Backend, chunk: usize) -> anyhow::Result<Scratch> {
        let c = &m.cfg;
        let mut stage_bytes = 0;
        let max_il = if c.n_nextn > 0 && c.eagle3.is_none() { c.n_layer + 1 } else { c.n_layer };
        for il in 0..max_il {
            if il < c.n_layer && !c.is_moe_layer(il) { continue; }
            let mut b = 0;
            for name in MATS {
                if let Ok(t) = m.w.layer(il, name) {
                    b = (b + t.expert(0, c.n_expert).data.len()).next_multiple_of(16); // as ExpertCache lays out a slot
                }
            }
            stage_bytes = stage_bytes.max(b);
        }
        let (ne, k) = (c.n_embd, c.n_expert_used.max(1));
        let (nv, kv, channels) = (c.ssm_dt_rank, c.n_head_kv * c.head_dim, c.ssm_d_inner + 2 * c.ssm_n_group * c.ssm_d_state);
        let q = c.n_head * c.head_dim;
        let hd = c.hc_n.max(1) * ne;
        let (h, base) = (hd * chunk, (hd + ne) * chunk); // x (hc_n streams a token) at 0, then h
        let mut at = base;
        let mut take = |n: usize| { let o = at; at += n * chunk; o };
        let (a, b, c2, d, e) = (take((2 * q).max(channels).max(2 * ne)), take(q.max(c.ssm_d_inner)), take(channels.max(3 * kv).max(if c.idx_top_k > 0 { (1 + 2 * super::QSA_H) * super::QSA_D } else { 0 })), take(q.max(c.ssm_d_inner)), take(2 * nv));
        let mix_end = at;
        at = base;
        let mut take = |n: usize| { let o = at; at += n * chunk; o };
        let (rl, wts, ge) = (take(c.n_expert), take(k), take((k * c.n_ff_exp).max(c.n_ff)));
        let (ye, shg, shy, sg) = (take(k * ne), take(c.n_ff_shexp), take(ne), take(1));
        let hr = take(if c.hadamard.is_some() { ne } else { 0 });  // Gyro files: the rotated h
        let mut total = at.max(mix_end);
        let (mut xn, mut lo, mut hg, mut pk, mut pq, mut pg, mut pn, mut pv, mut inj, mut mix, mut mo, mut emb) = Default::default();
        if c.hc_n > 0 {
            at = base;
            let mut take = |n: usize| { let o = at; at += n * chunk; o };
            (xn, lo, hg, pk, pq, pg, pn, pv) = (take(hd), take(c.hc_rank), take(hd), take(hd), take(hd), take(hd), take(hd), take(ne));
            at = at.max(total);
            let mut take = |n: usize| { let o = at; at += n * chunk; o };
            (inj, mix, mo, emb) = (take(c.hc_n), take(ne), take(ne), take(ne));
            total = at;
        } else if c.mhc_streams > 0 {
            let md = c.mhc_streams * ne;
            at = base;
            let mut take = |n: usize| { let o = at; at += n * chunk; o };
            xn = take(md);
            at = at.max(total);
            let mut take = |n: usize| { let o = at; at += n * chunk; o };
            (hg, mix, mo) = (take(md), take(ne), take(ne));
            total = at;
        }
        let n_tiles = 3 * (chunk * k / GEMM_BM + 2 * c.n_expert);
        Ok(Scratch {
            chunk, lay: Lay { h, a, b, c: c2, d, e, rl, wts, ge, ye, shg, shy, sg, hr, xn, lo, hg, pk, pq, pg, pn, pv, inj, mix, mo, emb },
            slab: gpu.zeros::<f32>(total)?,
            feat: if c.eagle3.is_some() { Some(gpu.zeros::<f32>(3 * ne * chunk)?) } else { None },
            dflash_feat: c.dflash.as_ref().map(|dc| gpu.zeros::<f32>(dc.target_layer_ids.len() * ne * chunk)).transpose()?,
            dflash_mean: if c.dflash.is_some() && c.hc_n > 0 { Some(gpu.zeros::<f32>(ne * chunk)?) } else { None },
            ints: gpu.zeros::<i32>(3 * chunk * k)?, tiles: gpu.zeros::<u64>(2 * n_tiles)?,
            staging: gpu.zeros::<u8>(if c.is_moe() { 3 * STAGE * stage_bytes } else { 0 })?,
        })
    }

    /// Bytes of VRAM this scratch takes.
    pub fn bytes(&self) -> usize {
        self.slab.len()
            + self.feat.as_ref().map_or(0, Buf::len)
            + self.dflash_feat.as_ref().map_or(0, Buf::len)
            + self.ints.len()
            + self.tiles.len()
            + self.staging.len()
    }
}

impl GpuModel<'_> {
    /// Run `tokens` from the current position; returns the logits after the last one. Short prompts (and models
    /// loaded without a prefill chunk) go through the decode step.
    pub fn prefill(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
        let fresh = self.pos == 0;
        if fresh { self.reset_shortlist(); }
        self.note_short_tokens(tokens);
        if self.prompt_cache_path.is_some() && fresh && tokens.len() > 1 {
            let t0 = std::time::Instant::now();
            let (logits, reused) = self.prefill_cached(tokens, 0, tokens.len() - 1)?;
            if reused > 0 {
                crate::say!(
                    "warm start: reused {reused}/{} prompt tokens from cache (TTFT {:.1} ms)",
                    tokens.len(),
                    t0.elapsed().as_secs_f64() * 1e3
                );
            }
            return Ok(logits);
        }
        let r = self.prefill_with(tokens, &mut |_| Ok(()))?;
        if fresh {
            self.ext_draft_sync_prompt(tokens)?;
        } else if self.tail().ext_draft.is_some() {
            self.tail_mut().ext_draft_pend.extend_from_slice(tokens);
        }
        if let Some(ng) = self.ngram.as_mut() { ng.begin(tokens); }
        Ok(r)
    }

    /// `prefill`, calling `between` between two chunks (`serve --parallel` decodes the other slots there; it must
    /// leave the current slot's state alone).
    pub fn prefill_with(&mut self, tokens: &[u32], between: &mut dyn FnMut(&mut Self) -> anyhow::Result<()>) -> anyhow::Result<Vec<f32>> {
        let short = |s: &mut Self| { let r = s.step_all(tokens); crate::term::prefilled(tokens.len()); r };
        let Some(chunk) = self.pf.as_ref().map(|p| p.chunk) else { return short(self) };
        if tokens.len() < self.short { return short(self); }
        anyhow::ensure!(self.pos + tokens.len() <= self.ctx, "context full ({} tokens; raise --ctx)", self.ctx);
        // STRATA_PF_ADAPT=N: place the hot experts after every N-th chunk of a long prompt, not only after all of it,
        // so the later chunks run with the placement the first ones showed (0 = off, the default; untested on real hardware).
        let every = std::env::var("STRATA_PF_ADAPT").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
        let nchunks = tokens.len().div_ceil(chunk);
        for (i, ch) in tokens.chunks(chunk).enumerate() {
            if i > 0 { between(self)?; }
            self.prefill_chunk(ch, None, None)?;
            crate::term::prefilled(ch.len());
            if every > 0 && (i + 1) % every == 0 && i + 1 < nchunks { self.adapt_experts()?; }
        }
        // The prompt's routing predicts the reply's: place its hot experts now instead of 64 per adapt during decode.
        self.adapt_experts()?;
        // Serve: place the prompt's experts for the prefill, then let decode follow the reply's own routing.
        if self.reset_counts_after_prefill { self.reset_counts(); }
        let s = self.tail();
        let (g, ne) = (&s.gpu, s.m.cfg.n_embd);
        let h = s.bufs["h"].addr();
        // `prefill_chunk` leaves the last hidden state in row 0 of `x`, as after `step` (the MTP draft reads it).
        match (&s.output_norm, &s.output_hc) {
            (Some(on), _) => g.rms_norm(s.x.addr(), 0, on.ptr, h, 0, ne, 1, s.m.cfg.eps)?,
            (None, Some(hr)) => s.hc_read(hr, s.x.addr(), 1, &s.hc_at())?,
            _ => unreachable!(),
        }
        g.gemv(s.output.as_ref().unwrap(), h, s.logits.addr())?;
        g.get::<f32>(&s.logits, 0, s.m.cfg.n_vocab)
    }

    /// Move the hot missed experts into VRAM (all parts of a split model). A fully resident cache has no hot missed
    /// expert to place: skip the scan over all its entries (issue #34 R2).
    fn adapt_experts(&mut self) -> anyhow::Result<()> {
        let adapt = self.adapt_every > 0;
        let mut p = Some(&mut *self);
        while let Some(q) = p {
            if let Some(cache) = q.cache.as_mut() {
                if adapt && !cache.fully_resident() { cache.settle(q.m, &*q.gpu)?; }
            }
            p = q.next.as_deref_mut();
        }
        Ok(())
    }

    /// A fresh sequence: `prefill(tokens)` from position 0, reusing earlier work. The state after `tokens[..keep]` is
    /// saved; when the saved tokens are a prefix of the next call's `tokens[..keep]`, that call starts from them
    /// (KV past them is rewritten, the recurrent state is copied back). The state after `tokens[..sys]` (the system
    /// prompt, `sys <= keep`) is stashed in host memory: a call whose history does not extend the saved one but
    /// starts with the same system prompt starts from there. A saved state that a call replaces is parked in host
    /// memory (up to `cache_ram`), and a later call that continues it takes it back, so a side request with another
    /// prompt does not cost the main conversation its cache. Returns the logits and the tokens reused.
    pub fn prefill_cached(&mut self, tokens: &[u32], sys: usize, keep: usize) -> anyhow::Result<(Vec<f32>, usize)> {
        self.prefill_cached_with(tokens, sys, keep, &mut |_| Ok(()))
    }

    /// `prefill_cached` with `prefill_with`'s `between`.
    pub fn prefill_cached_with(&mut self, tokens: &[u32], sys: usize, keep: usize, between: &mut dyn FnMut(&mut Self) -> anyhow::Result<()>)
        -> anyhow::Result<(Vec<f32>, usize)> {
        anyhow::ensure!(sys <= keep && keep < tokens.len(), "prefill_cached: sys {sys}, keep {keep} of {} tokens", tokens.len());
        self.reset_shortlist();
        self.note_short_tokens(tokens);
        let reused = self.cache_begin(tokens, keep)?;
        if sys > reused {
            self.prefill_with(&tokens[reused..sys], between)?;
            self.stash(&tokens[..sys])?;
        }
        let from = reused.max(sys);
        if keep > from {
            self.prefill_with(&tokens[from..keep], between)?;
            self.save_state()?;
            (self.saved_ids, self.saved_owner) = (tokens[..keep].to_vec(), self.owner);
            if let Err(e) = self.sync_saved_to_prompt_cache() {
                crate::say!("warning: could not save prompt cache: {e:#}");
            }
        }
        let logits = self.prefill_with(&tokens[keep..], between)?;
        self.ext_draft_sync_prompt(tokens)?;
        if let Some(ng) = self.ngram.as_mut() { ng.begin(tokens); }
        Ok((logits, reused))
    }

    /// The start of `prefill_cached`: return to the longest state (saved, stashed or parked) that `tokens[..keep]`
    /// continues, and return its length.
    fn cache_begin(&mut self, tokens: &[u32], keep: usize) -> anyhow::Result<usize> {
        let (hit, own) = (self.saved_ids.len(), self.owner);
        let stashed = self.sys.as_ref().filter(|_| self.sys_owner == own).map_or(0, |s| if tokens[..keep].starts_with(&s.3) { s.3.len() } else { 0 });
        let extends = hit > 0 && hit >= stashed && self.saved_owner == own && tokens[..keep].starts_with(&self.saved_ids);
        // The longest parked state this prompt continues, if it beats the saved one and the stash.
        let best = self.parked.iter().enumerate().filter(|(_, p)| p.owner == own && tokens[..keep].starts_with(&p.ids))
            .max_by_key(|(_, p)| p.ids.len()).map(|(i, p)| (i, p.ids.len()))
            .filter(|&(_, n)| n > stashed && !(extends && n <= hit));
        let reused = if let Some((i, n)) = best {
            self.unpark(i)?;
            n
        } else if extends {
            self.restore_state()?;
            hit
        } else if stashed > 0 {
            self.park()?;
            self.unstash()?;
            stashed
        } else {
            self.park()?;
            self.reset()?;
            0
        };
        // The stash stays valid while its KV entries do: this prompt rewrites them only past `reused`. A parked state
        // brings its own KV, which may have been computed in other chunks.
        if best.is_some() || stashed == 0 && self.sys.as_ref().is_some_and(|s| reused < s.3.len()) { self.sys = None; }
        Ok(reused)
    }

    /// `prefill_cached` for several slots in shared forwards (models with `batch_cap`): `jobs` holds (slot, owner,
    /// tokens, sys, keep) for distinct slots, each prompt within the context. The rows of all prompts go through the
    /// batched forward in job order; `between` runs between two forwards. Returns per job the logits and the tokens
    /// reused. On an error the jobs' slots are reset.
    pub fn prefill_slots(&mut self, jobs: &[(usize, u64, &[u32], usize, usize)], between: &mut dyn FnMut(&mut Self) -> anyhow::Result<()>)
        -> anyhow::Result<Vec<(Vec<f32>, usize)>> {
        let r = self.prefill_slots_run(jobs, between);
        if r.is_err() { for j in jobs { self.use_slot(j.0); self.reset()?; } }
        r
    }

    fn prefill_slots_run(&mut self, jobs: &[(usize, u64, &[u32], usize, usize)], between: &mut dyn FnMut(&mut Self) -> anyhow::Result<()>)
        -> anyhow::Result<Vec<(Vec<f32>, usize)>> {
        use crate::backend::ATT_SPLITS;
        let (cap, max) = (self.batch_cap(), self.bt.as_ref().map_or(0, |b| b.rows));
        anyhow::ensure!((1..=cap).contains(&jobs.len()), "prefill_slots: {} prompts, {cap} at most", jobs.len());
        let dense = self.bt.as_ref().is_some_and(|b| b.dense);
        // Rows of three phases: up to `sys`, up to `keep`, the rest. A model without recurrent state runs them as one
        // (a saved state is a position, and the forwards write its KV); else a phase's forwards end where the
        // states of its prompts are stashed or saved.
        let (mut phases, mut reused) = ([vec![], vec![], vec![]], vec![]);
        for &(sl, owner, tokens, sys, keep) in jobs {
            anyhow::ensure!(sys <= keep && keep < tokens.len() && tokens.len() <= self.ctx, "prefill_slots: sys {sys}, keep {keep} of {} tokens", tokens.len());
            self.use_slot(sl);
            self.owner = owner;
            let n = self.cache_begin(tokens, keep)?;
            if dense {
                if sys > n { self.pos = sys; self.stash(&tokens[..sys])?; }
                if keep > n.max(sys) {
                    self.pos = keep;
                    self.save_state()?;
                    (self.saved_ids, self.saved_owner) = (tokens[..keep].to_vec(), owner);
                }
            }
            self.pos = n;
            let cut = if dense { [n, n, n] } else { [n, sys.max(n), keep.max(n)] };
            for (ph, w) in phases.iter_mut().zip([cut[0]..cut[1], cut[1]..cut[2], cut[2]..tokens.len()]) {
                ph.extend(w.map(|i| (sl, tokens[i], i + 1 == tokens.len())));
            }
            reused.push(n);
        }
        // The attention partials hold `cap` rows at ATT_SPLITS splits; short prompts need fewer splits a row.
        let splits = jobs.iter().map(|j| j.2.len()).max().unwrap().div_ceil(256).clamp(1, ATT_SPLITS);
        let (mut out, mut first) = (vec![], true);
        for (p, rows) in phases.iter().enumerate() {
            for ch in rows.chunks(max.min(cap * ATT_SPLITS / splits)) {
                if !first { between(self)?; }
                first = false;
                out.extend(self.batch_rows(ch, false)?);
            }
            if rows.is_empty() || p == 2 { continue; }
            for &(sl, owner, tokens, sys, keep) in jobs {
                let at = if p == 0 { sys } else { keep };
                if !rows.iter().any(|r| r.0 == sl) { continue; }
                self.use_slot(sl);
                self.owner = owner;
                anyhow::ensure!(self.pos == at, "prefill_slots: slot {sl} at {} after phase {p}, not {at}", self.pos);
                if p == 0 { self.stash(&tokens[..sys])?; } else {
                    self.save_state()?;
                    (self.saved_ids, self.saved_owner) = (tokens[..keep].to_vec(), owner);
                }
            }
        }
        Ok(out.into_iter().zip(reused).map(|((_, l), n)| (l, n)).collect())
    }

    fn step_all(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
        let mut logits = Vec::new();
        // Hyper-connection / mHC models: TMAX tokens per forward (prompts too short for the chunked GEMM prefill).
        if self.m.cfg.hc_n > 0 || self.m.cfg.mhc_streams > 0 {
            // The draft head's entry for a token pairs it with the stack of the token before: row 0 of `x` at the
            // start (as below), then the rows of the chunk.
            let mut last = 0;
            for ch in tokens.chunks(self.tmax()) {
                let fill = self.mtp_synced();
                if fill && self.pos > 0 { let r = self.row_ptr(last); self.mtp_run(&ch[..1], r, false)?; }
                logits = self.verify(ch)?.pop().unwrap();
                if fill && ch.len() > 1 { let r = self.row_ptr(0); self.mtp_run(&ch[1..], r, false)?; }
                last = ch.len() - 1;
            }
            return Ok(logits);
        }
        for &t in tokens {
            // The draft head's entry for `t` pairs it with the hidden state of the token before (row 0 of `x`).
            if self.pos > 0 && self.mtp_synced() { let r = self.row_ptr(0); self.mtp_run(&[t], r, false)?; }
            logits = self.step(t)?;
        }
        Ok(logits)
    }

    /// `input`: the hidden rows and PLE input from the part before (None: embed `toks`).
    /// `rows`: the chunk is a step of a batch (`batch_rows` on a model with recurrent layers or experts): per row its
    /// slot, and the attention splits. Row r runs at the position in `Batch::step` on its own slot's state, the KV
    /// through `Batch::kvt`; the hidden rows are left in the slab, and no position moves.
    fn prefill_chunk(&mut self, toks: &[u32], input: Option<(&[f32], &[f32])>, rows: Option<(&[usize], usize)>) -> anyhow::Result<()> {
        self.dflash_commit_ctx()?;
        // Missed experts are read from the mapped weights in the order the staging copies take them.
        let pre = self.cache.is_some().then(Prefetch::start);
        let owned: (Vec<f32>, Vec<f32>);
        let (emb, pe): (&[f32], &[f32]) = match input {
            Some(v) => v,
            None => {
                let hcn = self.m.cfg.hc_n.max(1);
                let pe = if self.ple_ring.is_some() { toks.iter().flat_map(|&tk| self.m.ple_embed(&mut self.ple_prev, tk)).collect() } else { vec![] };
                owned = (toks.iter().flat_map(|&tk| self.m.embed(tk).repeat(hcn)).collect(), pe);
                (&owned.0, &owned.1)
            }
        };
        let c = &self.m.cfg;
        let pf = self.pf.as_ref().unwrap();
        let (g, mut t, n, eps, ne) = (&self.gpu, toks.len(), pf.chunk, c.eps, c.n_embd);
        let d = c.hc_n.max(1) * ne;
        g.put(emb, &pf.slab, 0)?;
        if !pe.is_empty() { g.put(pe, &pf.slab, pf.lay.emb)?; }
        g.put(&[self.pos as i32, 0], &self.step_buf, 0)?;
        g.put(&[self.pos as i32 + self.rope_offset, 0], &self.rope_buf, 0)?;
        let mut step = self.step_buf.addr();
        let mut rope = self.rope_buf.addr();
        if rows.is_some() { step = self.bt.as_ref().unwrap().step.addr(); }
        let base = pf.slab.addr();
        let at = |o: usize| base + 4 * o as u64;
        let l = &pf.lay;
        let (x, h) = (at(0), at(l.h));
        let ha = HcAt { xn: at(l.xn), lo: at(l.lo), hg: at(l.hg), inj: at(l.inj), h, hq: 0, emb: at(l.emb), pk: at(l.pk), pq: at(l.pq), pg: at(l.pg),
                        pn: at(l.pn), pv: at(l.pv), gemm: true };
        let (dk, nk, nv) = (c.ssm_d_state, c.ssm_n_group, c.ssm_dt_rank);
        let (dv, channels) = (c.ssm_dv(), c.ssm_d_inner + 2 * nk * dk);
        let (hd, nh, nkv) = (c.head_dim, c.n_head, c.n_head_kv);
        let kv = nkv * hd;
        // The draft head runs as one more layer on the chunk while it is in step with the main model.
        let fill = self.mtp_synced();
        let is_eagle = self.mtp.as_ref().is_some_and(|mh| mh.eagle.is_some());
        // ponytail: the chunked prefill does not fill a hyper-connection draft head (it starts without the prompt:
        // lower acceptance at first, same output). Fill it as one more hc layer here if that costs speed.
        let mtp = self.mtp.as_ref().filter(|mh| fill && mh.eagle.is_none() && c.hc_n == 0);
        // STRATA_NAN_TRACE[=N]: from position N (default 0), one line per layer with the largest magnitude of each intermediate buffer, and the first
        // row of one holding a non-finite value. Each probe waits for the GPU: diagnostic runs only.
        let trace = std::env::var("STRATA_NAN_TRACE").is_ok_and(|v| self.pos >= v.parse::<usize>().unwrap_or(0));
        let mut tr = String::new();
        let probe = |tr: &mut String, what: &str, b: &Buf, o: u64, w: usize, rows: usize| -> anyhow::Result<()> {
            if !trace { return Ok(()); }
            let v: Vec<f32> = g.get(b, ((o - b.addr()) / 4) as usize, rows * w)?;
            let max = v.iter().filter(|a| a.is_finite()).fold(0f32, |m, a| m.max(a.abs()));
            tr.push_str(&format!(" {what} {max:.2e}"));
            if let Some(i) = v.iter().position(|a| !a.is_finite()) { tr.push_str(&format!(" (non-finite row {})", i / w)); }
            Ok(())
        };
        for (il, ly) in self.layers.iter().enumerate().map(|(i, ly)| (self.first + i, ly)).chain(mtp.map(|mh| (c.n_layer, &mh.layer))) {
            if il < c.n_layer {
                if let (Some(pff), Some(eg)) = (&pf.feat, self.mtp.as_ref().and_then(|mh| mh.eagle.as_ref()).filter(|_| fill)) {
                    if let Some(idx) = eg.cfg.extract_layers.iter().position(|&el| el == il) {
                        for r in 0..t {
                            g.dtod(&pf.slab, 4 * r * ne, pff, 4 * (r * 3 + idx) * ne, 4 * ne)?;
                        }
                    }
                }
            }
            if let Some(mh) = mtp.filter(|_| il == c.n_layer) {
                // Its entry for token i pairs the token with the hidden state before it (`mtp_run`): h gets the last
                // hidden state of the previous chunk, then rows 0..t - 1; the first token of the sequence has none.
                let skip = (self.pos == 0) as usize;
                if skip == 0 { g.dtod(&self.x, 0, &pf.slab, 4 * l.h, 4 * ne)?; }
                g.dtod(&pf.slab, 0, &pf.slab, 4 * (l.h + (1 - skip) * ne), 4 * (t - 1) * ne)?;
                g.dtod(&pf.slab, 4 * (t - 1) * ne, &self.x, 0, 4 * ne)?;
                let emb: Vec<f32> = toks[skip..].iter().flat_map(|&tk| self.m.embed(tk)).collect();
                g.put(&emb, &pf.slab, 0)?;
                t -= skip;
                g.rms_norm(x, ne, mh.enorm.ptr, at(l.a), 2 * ne, ne, t, eps)?;
                g.rms_norm(h, ne, mh.hnorm.ptr, at(l.a + ne), 2 * ne, ne, t, eps)?;
                g.gemm_dense(mh.eh_proj.as_ref().unwrap(), t, at(l.a), 2 * ne, x, ne, 0)?;
                g.put(&[mh.pos as i32, 0], &self.step_buf, 2)?;
                g.put(&[mh.pos as i32 + self.rope_offset, 0], &self.rope_buf, 2)?;
                step += 8;
                rope += 8;
            }
            // Hyper-connections / mHC: mixer and FFN outputs go to "mix" and "mo", then into the streams.
            let (mx, mact) = if c.hc_n > 0 || ly.mhc.is_some() { (at(l.mix), 0) } else { (x, 2) };
            if let Some(pl) = &ly.ple { self.ple(pl, x, step, t, &ha)?; }
            if let Some((mk, mhc_a, _)) = &ly.mhc {
                let s = c.mhc_streams;
                let md = s * ne;
                let ones = self.mhc_ones.as_ref().unwrap().addr();
                if mk & 1 != 0 {
                    g.hc_write(at(l.hg), x, 0, ne, 0x100 | 0x400 | s, t, false)?;
                }
                g.rms_norm(at(l.hg), md, ones, at(l.xn), md, md, t, 1e-6)?;
                if mhc_a.weight.cols % 64 == 0 {
                    g.gemm_dense(&mhc_a.weight, t, at(l.xn), md, mhc_a.buf.addr() + 128, mhc_a.out_dim, 0)?;
                } else {
                    self.mv(&mhc_a.weight, at(l.xn), md, mhc_a.buf.addr() + 128, mhc_a.out_dim, t)?;
                }
                g.hc_mix(at(l.hg), mhc_a.buf.addr(), h, ne, 0x100 | s | (mhc_a.out_dim << 16), t)?;
                g.rms_norm(h, ne, ly.attn_norm.as_ref().unwrap().ptr, h, ne, ne, t, eps)?;
            } else {
                match (&ly.attn_norm, &ly.hc) {
                    (Some(an), _) => g.rms_norm(x, ne, an.ptr, h, ne, ne, t, eps)?,
                    (None, Some(hr)) => self.hc_read(&hr[0], x, t, &ha)?,
                    _ => unreachable!(),
                }
            }
            match &ly.mixer {
                Mixer::Delta { qkv, gate, beta, alpha, dt_bias, a, conv_w, norm, out, conv, s, .. } => {
                    let (eb, eg) = (at(l.e), at(l.e + n * nv));
                    g.gemm_dense(qkv, t, h, ne, at(l.a), channels, 0)?;
                    // The qkv GEMM above already quantized `h`; when it did not (IQ2_XS, or a shape the int8 GEMM
                    // skips), prepare it now so beta, alpha and the gate can all reuse one copy.
                    g.prequant_gemm_x(gate, t, h, ne)?;
                    match rows {
                        // A row a slot: each its own conv state, in row order.
                        Some((sl, _)) => for (r, &s) in sl.iter().enumerate() {
                            let o = (4 * r * channels) as u64;
                            g.conv_silu(at(l.a) + o, conv[s].addr(), conv_w.ptr, at(l.c) + o, channels, c.ssm_conv, 0, eps)?;
                        },
                        None => g.conv_seq(at(l.a), conv[self.cur].addr(), conv_w.ptr, at(l.c), channels, c.ssm_conv, t, 0, 0)?,
                    }
                    // The beta and alpha GEMMs (16 blocks each) run beside the norm: batch launches alternate streams.
                    g.batch(&mut || {
                        g.gemm_dense(beta, t, h, ne, eb, nv, SAME_X)?;
                        g.l2_norm(at(l.c), dk, 2 * nk, channels, t * 2 * nk, eps)?;
                        g.gemm_dense(alpha, t, h, ne, eg, nv, SAME_X)
                    })?;
                    g.ssm_gates(eb, eg, dt_bias.ptr, a.ptr, t * nv, nv)?;
                    probe(&mut tr, "qkv", &pf.slab, at(l.a), channels, t)?;
                    probe(&mut tr, "conv", &pf.slab, at(l.c), channels, t)?;
                    probe(&mut tr, "beta", &pf.slab, eb, nv, t)?;
                    probe(&mut tr, "alpha", &pf.slab, eg, nv, t)?;
                    // The z GEMM runs beside the scan, on the SMs its last wave of blocks leaves idle.
                    g.batch(&mut || {
                        let scale = 1.0 / (dk as f32).sqrt();
                        match rows {
                            Some((sl, _)) => for (r, &q) in sl.iter().enumerate() {
                                g.deltanet(at(l.c) + (4 * r * channels) as u64, s[q].addr(), eb + (4 * r * nv) as u64, eg + (4 * r * nv) as u64, 0, 0,
                                           at(l.d) + (4 * r * c.ssm_d_inner) as u64, nk, nv, c.k_interleave, dk, dv, scale, 1, channels, 0, 0)?;
                            },
                            None => g.deltanet(at(l.c), s[self.cur].addr(), eb, eg, 0, 0, at(l.d), nk, nv, c.k_interleave, dk, dv, scale, t, channels, 0, 0)?,
                        }
                        g.gemm_dense(gate, t, h, ne, at(l.b), c.ssm_d_inner, SAME_X)
                    })?;
                    probe(&mut tr, "deltanet", &pf.slab, at(l.d), c.ssm_d_inner, t)?;
                    probe(&mut tr, "state", &s[self.cur], s[self.cur].addr(), s[self.cur].len() / 4, 1)?;
                    probe(&mut tr, "z", &pf.slab, at(l.b), c.ssm_d_inner, t)?;
                    if ly.hc.is_some() { g.norm_gate_sig(at(l.d), norm.ptr, at(l.b), dv, t * nv, eps, 0)?; } else { g.norm_gate(at(l.d), norm.ptr, at(l.b), dv, t * nv, eps)?; }
                    g.gemm_dense(out, t, at(l.d), c.ssm_d_inner, mx, ne, mact)?;
                }
                Mixer::Attn { q, k, v, qn, kn, o, sink, .. } if let Some((_, splits)) = rows => {
                    // As `batch_rows` on a dense model: each row reads and extends its own slot's KV.
                    let (kraw, kk, vv) = (at(l.c), at(l.c + n * kv), at(l.c + 2 * n * kv));
                    let qs = if c.q_gate { 2 * hd } else { hd };
                    let bt = self.bt.as_ref().unwrap();
                    let kvt = bt.kvt.addr() + (8 * (il - self.first) * 2 * bt.rows) as u64;
                    g.gemm_dense(q, t, h, ne, at(l.a), nh * qs, BATCHED)?;
                    g.gemm_dense(k, t, h, ne, kraw, kv, SAME_X | BATCHED)?;
                    g.gemm_dense(v, t, h, ne, vv, kv, SAME_X | BATCHED)?;
                    let win = c.swa.get(il).copied().unwrap_or(false).then_some(c.n_swa).unwrap_or(0);
                    g.rms_norm_rope_qk_rows(at(l.a), qs, qn.ptr, at(l.b), nh, kraw, kn.ptr, kk, nkv, hd, eps, c.n_rot, step, c.rope_base, vv, kvt, t)?;
                    g.attn_rows(at(l.b), if c.q_gate { at(l.a) } else { 0 }, at(l.d), bt.attp.addr(), step, nh, nkv, hd, 1.0 / (hd as f32).sqrt(), t,
                                sink.as_ref().map_or(0, |s| s.ptr), win, kvt, splits)?;
                    g.gemm_dense(o, t, at(l.d), nh * hd, mx, ne, mact | BATCHED)?;
                }
                Mixer::Attn { q, k, v, qn, kn, o, kc, vc, qsa, sink } => {
                    let (kraw, kk, vv) = (at(l.c), at(l.c + n * kv), at(l.c + 2 * n * kv));
                    let qs = if c.q_gate { 2 * hd } else { hd };
                    let use_rope = c.rope_layer.get(il).copied().unwrap_or(true);
                    g.gemm_dense(q, t, h, ne, at(l.a), nh * qs, 0)?;
                    g.rms_norm(at(l.a), qs, qn.ptr, at(l.b), hd, hd, t * nh, eps)?;
                    if use_rope { g.rope_neox(at(l.b), hd, t * nh, nh, c.n_rot, rope, c.rope_base)?; }
                    g.gemm_dense(k, t, h, ne, kraw, kv, SAME_X)?;
                    g.gemm_dense(v, t, h, ne, vv, kv, SAME_X)?;
                    g.rms_norm(kraw, hd, kn.ptr, kk, hd, hd, t * nkv, eps)?;
                    if use_rope { g.rope_neox(kk, hd, t * nkv, nkv, c.n_rot, rope, c.rope_base)?; }
                    g.kv_store(kk, vv, kc[self.cur].addr(), vc[self.cur].addr(), kv, t, step, hd)?;
                    // QSA: region c (free after kv_store) holds the raw indexer keys, raw queries and normed queries.
                    let (qm, mw) = match (qsa, &self.qsa) {
                        (Some(a), Some(b)) => {
                            let (ik, iq, iqn) = (at(l.c), at(l.c + n * super::QSA_D), at(l.c + n * (1 + super::QSA_H) * super::QSA_D));
                            let (dq, nq) = (super::QSA_D, super::QSA_H * super::QSA_D);
                            g.gemm_dense(&a.kp, t, h, ne, ik, dq, 0)?;
                            g.gemm_dense(&a.qp, t, h, ne, iq, nq, SAME_X)?;
                            g.qsa_pool(ik, a.ring, a.pool.addr(), a.kn.ptr, step, t, c.n_rot, c.rope_base, eps)?;
                            // Rows with at most `top` complete blocks attend densely: no mask while all of them do.
                            if self.pos + t >= super::QSA_POOL * (b.top + 1) {
                                g.rms_norm(iq, dq, a.qn.ptr, iqn, dq, dq, t * super::QSA_H, eps)?;
                                g.rope_neox(iqn, dq, t * super::QSA_H, super::QSA_H, c.n_rot, rope, c.rope_base)?;
                                for r0 in (0..t).step_by(super::QSA_ROWS) {
                                    g.qsa_select(iqn + (4 * r0 * nq) as u64, a.pool.addr(), b.score.addr(), 0, b.mask.addr() + (4 * r0 * b.mw) as u64,
                                                 step, r0, super::QSA_ROWS.min(t - r0), b.nbs, b.top, 0, b.mw)?;
                                }
                                (b.mask.addr(), b.mw)
                            } else { (0, 0) }
                        }
                        _ => (0, 0),
                    };
                    let sink_addr = sink.as_ref().map_or(0, |s| s.ptr);
                    let win = c.swa.get(il).copied().unwrap_or(false).then_some(c.n_swa).unwrap_or(0);
                    g.attn_prefill_m(at(l.b), kc[self.cur].addr(), vc[self.cur].addr(), if c.q_gate { at(l.a) } else { 0 }, at(l.d), step, t, nh, nkv, hd, 1.0 / (hd as f32).sqrt(), qm, mw, sink_addr, win)?;
                    probe(&mut tr, "q", &pf.slab, at(l.a), nh * qs, t)?;
                    probe(&mut tr, "k", &pf.slab, kk, kv, t)?;
                    probe(&mut tr, "v", &pf.slab, vv, kv, t)?;
                    probe(&mut tr, "attn", &pf.slab, at(l.d), nh * hd, t)?;
                    g.gemm_dense(o, t, at(l.d), nh * hd, mx, ne, mact)?;
                }
            }
            probe(&mut tr, "| mixer", &pf.slab, x, d, t)?;
            if let Some((_, mhc_a, mhc_f)) = &ly.mhc {
                let s = c.mhc_streams;
                let md = s * ne;
                let ones = self.mhc_ones.as_ref().unwrap().addr();
                g.hc_write(at(l.hg), mx, mhc_a.buf.addr(), ne, 0x100 | s | (mhc_a.out_dim << 16), t, false)?;
                g.rms_norm(at(l.hg), md, ones, at(l.xn), md, md, t, 1e-6)?;
                if mhc_f.weight.cols % 64 == 0 {
                    g.gemm_dense(&mhc_f.weight, t, at(l.xn), md, mhc_f.buf.addr() + 128, mhc_f.out_dim, 0)?;
                } else {
                    self.mv(&mhc_f.weight, at(l.xn), md, mhc_f.buf.addr() + 128, mhc_f.out_dim, t)?;
                }
                g.hc_mix(at(l.hg), mhc_f.buf.addr(), h, ne, 0x100 | s | (mhc_f.out_dim << 16), t)?;
                g.rms_norm(h, ne, ly.post_norm.as_ref().unwrap().ptr, h, ne, ne, t, eps)?;
            } else {
                match (&ly.post_norm, &ly.hc) {
                    (Some(pn), _) => g.rms_norm(x, ne, pn.ptr, h, ne, ne, t, eps)?,
                    (None, Some(hr)) => {
                        g.hc_write(x, mx, at(l.inj), ne, c.hc_n, t, false)?;
                        self.hc_read(&hr[1], x, t, &ha)?;
                    }
                    _ => unreachable!(),
                }
            }
            match &ly.ffn {
                Ffn::Dense { gate, up, down } => {
                    g.gemm_dense(gate, t, h, ne, at(l.ge), c.n_ff, 0)?;
                    g.gemm_dense(up, t, h, ne, at(l.ge), c.n_ff, 1 | SAME_X)?;
                    probe(&mut tr, "gate_up", &pf.slab, at(l.ge), c.n_ff, t)?;
                    if let Some((mk, _, mhc_f)) = &ly.mhc {
                        g.gemm_dense(down, t, at(l.ge), c.n_ff, at(l.mo), ne, 0)?;
                        let leave_bit = if mk & 2 != 0 { 0x200 } else { 0 };
                        let code_f = 0x100 | leave_bit | c.mhc_streams | (mhc_f.out_dim << 16);
                        g.hc_write(at(l.hg), at(l.mo), mhc_f.buf.addr(), ne, code_f, t, false)?;
                        if mk & 2 != 0 {
                            g.dtod(&pf.slab, 4 * l.mo, &pf.slab, 0, 4 * t * ne)?;
                        }
                    } else {
                        g.gemm_dense(down, t, at(l.ge), c.n_ff, x, ne, 2)?;
                    }
                }
                Ffn::Moe { router, shared } => {
                    let k = c.n_expert_used;
                    let ids_d = pf.ints.addr();
                    // The 1-row shared gate_inp GEMM runs beside the router GEMM (both float: one launch each).
                    g.batch(&mut || {
                        g.gemm_dense(router, t, h, ne, at(l.rl), c.n_expert, 0)?;
                        if let Some(s) = shared { if let Some(gi) = &s.gate_inp { g.gemm_dense(gi, t, h, ne, at(l.sg), 1, SAME_X)?; } }
                        Ok(())
                    })?;
                    g.router_topk(at(l.rl), c.n_expert, k, c.expert_weights_scale, ids_d, at(l.wts), t, c.router_sigmoid)?;
                    let (sh, sg) = match shared {
                        Some(s) => {
                            g.gemm_dense(&s.gate, t, h, ne, at(l.shg), c.n_ff_shexp, SAME_X)?;
                            g.gemm_dense(&s.up, t, h, ne, at(l.shg), c.n_ff_shexp, 1 | SAME_X)?;
                            (at(l.shy), if s.gate_inp.is_some() { at(l.sg) } else { 0 })
                        }
                        None => (0, 0),
                    };
                    probe(&mut tr, "router", &pf.slab, at(l.rl), c.n_expert, t)?;
                    probe(&mut tr, "weights", &pf.slab, at(l.wts), k, t)?;
                    let ids = g.get::<i32>(&pf.ints, 0, t * k)?;
                    // Ids outside 0..n_expert mean the router ran on corrupt logits (issue #25: one prefill past ~128k
                    // tokens). Fail here with the place, not in the index math below. ponytail: the root cause is still open.
                    if let Some(&bad) = ids.iter().find(|&&e| e < 0 || e as usize >= c.n_expert) {
                        anyhow::bail!("prefill: router gave expert id {bad} at layer {il}, chunk of {t} tokens from position {}; \
                            lower --chunk (for example --chunk 1024) or --ctx, and try STRATA_NAN_TRACE=1 for a non-finite hidden state", self.pos);
                    }
                    let cache = self.cache.as_mut().unwrap();
                    cache.record_prompt(il, &ids);
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
                    let mut missed: Vec<usize> = (0..c.n_expert).filter(used).filter(|&e| cache.slot(il, e) < 0).collect();
                    // A batch step: the CPU runs the missed experts (each read once for all rows, in RAM) beside the
                    // resident GEMMs, instead of a PCIe copy of each that costs more than the CPU's pass.
                    let cpu = if rows.is_some() && c.hadamard.is_none() && !missed.is_empty() {
                        let hv = g.get::<f32>(&pf.slab, l.h, t * ne)?;
                        Some((std::mem::take(&mut missed), hv.chunks_exact(ne).map(<[f32]>::to_vec).collect::<Vec<_>>()))
                    } else { None };
                    if let Some(pre) = &pre {
                        let w: Vec<_> = MATS.iter().map(|name| self.m.w.layer(il, name)).collect::<anyhow::Result<_>>()?;
                        let mut rs: Vec<(usize, usize)> = missed.chunks(STAGE).flat_map(|grp| w.iter().zip(&lx.mats).flat_map(move |(w, mat)| grp.chunk_by(|a, b| *b == a + 1)
                            .map(move |run| (w.data.as_ptr() as usize + run[0] * mat.bytes, run.len() * mat.bytes)))).collect();
                        // Overlap NVMe read-ahead for the next MoE layer's hottest cold experts behind this layer's GEMMs.
                        if let Some(next_il) = (il + 1..c.n_layer).find(|&j| cache.layer(j).is_some()) {
                            rs.extend(cache.cold_ranges_for_next(self.m, next_il, 16));
                        }
                        pre.send(rs);
                    }
                    // Staged groups of at most STAGE missed experts, group j in staging third j % 3. Staging holds
                    // each matrix in its own section (all gates, then ups, then downs), so experts with consecutive
                    // ids are contiguous there as in the weights and copy in one call.
                    let sp = pf.staging.addr();
                    let half = |j: usize| j % 3 * pf.staging.len() / 3;
                    let staged: Vec<Vec<(usize, [u64; 3])>> = missed.chunks(STAGE).enumerate()
                        .map(|(j, grp)| grp.iter().enumerate().map(|(i, &e)| (e, lx.mats.each_ref().map(|m| sp + (half(j) + STAGE * m.off + i * m.bytes) as u64))).collect())
                        .collect();
                    // The resident experts split into one chunk per staged group, of about equal positions, so the
                    // copies run beside them too: step s runs group s - 2 and chunk s beside the copy of group s
                    // (into the third group s - 3 used, done in step s - 1). A step waits only for the copies of the
                    // step two before it (`beside`), so each copy has two steps of GEMMs to hide behind.
                    // Without overlap (Vulkan without a transfer queue) one chunk: more, smaller GEMMs only cost there.
                    let ns = if g.beside_overlaps() { staged.len().max(1) } else { 1 };
                    let res: Vec<(usize, [u64; 3])> = (0..c.n_expert).filter(used)
                        .filter_map(|e| { let s = cache.slot(il, e); (s >= 0).then(|| (e, lx.mats.each_ref().map(|m| pool + (s as usize * stride + m.off) as u64))) }).collect();
                    let total = res.iter().map(|r| off[r.0 + 1] - off[r.0]).sum::<usize>().max(1);
                    let mut chunks = vec![Vec::new(); ns];
                    let mut acc = 0;
                    for &r in &res {
                        chunks[(acc * ns / total).min(ns - 1)].push(r);
                        acc += off[r.0 + 1] - off[r.0];
                    }
                    let mut chunks = chunks.into_iter();
                    let mut groups = vec![];
                    // Per step: its groups and the staged group it copies.
                    let mut steps = vec![];
                    for s in 0..staged.len() + 2 {
                        let g0 = groups.len();
                        groups.extend(s.checked_sub(2).and_then(|j| staged.get(j)).cloned());
                        groups.extend(chunks.next());
                        steps.push((g0..groups.len(), (s < staged.len()).then_some(s)));
                    }
                    let mut tiles = Vec::new();
                    let push = |tiles: &mut Vec<u64>, grp: &[(usize, [u64; 3])], mi: usize, base: usize| {
                        let start = tiles.len() / 2;
                        for &(e, w) in grp {
                            for m0 in (off[e]..off[e + 1]).step_by(GEMM_BM) {
                                tiles.extend([w[mi], (m0 - base) as u64 | (GEMM_BM.min(off[e + 1] - m0) as u64) << 32]);
                            }
                        }
                        (start, tiles.len() / 2 - start)
                    };
                    // Per step: the gate and up tiles of all its groups (one launch each: more blocks, less tail), then
                    // per group its down tiles and the slot rows lo..hi it covers. The down GEMM reads rows lo..hi of
                    // `ge` only, so its tiles count from lo and it quantizes just those rows.
                    let mut spans = Vec::new();
                    for (gs, _) in &steps {
                        let gu = [0, 1].map(|mi| { let s = tiles.len() / 2; for gi in gs.clone() { push(&mut tiles, &groups[gi], mi, 0); } (s, tiles.len() / 2 - s) });
                        let downs: Vec<_> = gs.clone().filter(|&gi| !groups[gi].is_empty()).map(|gi| {
                            let grp = &groups[gi];
                            let (lo, hi) = (off[grp[0].0], off[grp[grp.len() - 1].0 + 1]);
                            (push(&mut tiles, grp, 2, lo), lo, hi)
                        }).collect();
                        spans.push((gu, downs));
                    }
                    anyhow::ensure!(8 * tiles.len() <= pf.tiles.len(), "prefill tile list overflow");
                    g.put(&tiles, &pf.tiles, 0)?;
                    let tp = |s: usize| pf.tiles.addr() + 16 * s as u64;
                    let xrow_d = pf.ints.addr() + 4 * (n * k) as u64;
                    let [mg, mu, md] = lx.mats;
                    // Gyro files: the routed experts read the rotated h and the rotated SwiGLU output (`config::Hadamard`).
                    let rot = self.hada.as_ref().zip(c.hadamard.as_ref());
                    let (hx, same) = match rot {
                        Some((s, r)) => { g.hadamard(h, 0, s[0].addr(), at(l.hr), ne, r.block, t)?; (at(l.hr), 0) }
                        None => (h, SAME_X),
                    };
                    let copy = |j: usize| -> anyhow::Result<()> {
                        let grp = &staged[j];
                        for (name, mat) in MATS.iter().zip(&lx.mats) {
                            let w = self.m.w.layer(il, name)?;
                            let mut i = 0;
                            for run in grp.chunk_by(|a, b| b.0 == a.0 + 1) {
                                let e = run[0].0;
                                let slice = &w.data[e * mat.bytes..(e + run.len()) * mat.bytes];
                                g.htod(slice, &pf.staging, half(j) + STAGE * mat.off + i * mat.bytes)?;
                                // Release cold streamed prefill experts from OS page cache when the model exceeds RAM.
                                crate::weights::cold(slice);
                                i += run.len();
                            }
                        }
                        Ok(())
                    };
                    // `beside` starts a step's copies after the GEMMs before it. They queue last, so the GEMMs need not
                    // wait for the host to stage them.
                    for ((_, cp), (gu, downs)) in steps.iter().zip(&spans) {
                        g.beside(&mut || {
                            if gu[0].1 > 0 {
                                g.gemm(mg.ty, mg.rows, mg.cols, tp(gu[0].0), gu[0].1, hx, ne, t, xrow_d, at(l.ge), c.n_ff_exp, same)?;
                                g.gemm(mu.ty, mu.rows, mu.cols, tp(gu[1].0), gu[1].1, hx, ne, t, xrow_d, at(l.ge), c.n_ff_exp, 1 | SAME_X)?;
                            }
                            for &((s0, n0), lo, hi) in downs {
                                let ge = at(l.ge) + (4 * lo * c.n_ff_exp) as u64;
                                if let Some((s, r)) = rot { g.hadamard(ge, 0, s[1].addr(), ge, c.n_ff_exp, r.block, hi - lo)?; }
                                g.gemm(md.ty, md.rows, md.cols, tp(s0), n0, at(l.ge) + (4 * lo * c.n_ff_exp) as u64, c.n_ff_exp, hi - lo, 0, at(l.ye) + (4 * lo * ne) as u64, ne, 0)?;
                            }
                            if let Some(j) = *cp { copy(j)?; }
                            Ok(())
                        })?;
                    }
                    if let Some((cm, hs)) = &cpu {
                        // Slot s of expert e holds the pair of token xrow[s]; the rows go to `ye` in slot order.
                        let pairs: Vec<(usize, usize, f32)> = { let (xr, of) = (&xrow, &off); cm.iter().flat_map(|&e| (of[e]..of[e + 1]).map(move |s| (xr[s] as usize, e, 0.0))).collect() };
                        let o = crate::cpu_moe::expert_rows(&self.m, il, hs, &pairs);
                        let mut j = 0;
                        for &e in cm {
                            let len = (off[e + 1] - off[e]) * ne;
                            g.put(&o[j..j + len], &pf.slab, l.ye + off[e] * ne)?;
                            j += len;
                        }
                    }
                    g.beside_join()?;
                    // The shared down runs last, so the routed gate and up above reuse the int8 h of the shared up.
                    if let Some(s) = shared { g.gemm_dense(&s.down, t, at(l.shg), c.n_ff_shexp, at(l.shy), ne, 0)?; }
                    // Expert rows are slots (token, j) sorted by expert, not tokens.
                    probe(&mut tr, "experts_gate_up", &pf.slab, at(l.ge), c.n_ff_exp, t * k)?;
                    probe(&mut tr, "experts_down", &pf.slab, at(l.ye), ne, t * k)?;
                    if shared.is_some() {
                        probe(&mut tr, "shared_gate_up", &pf.slab, at(l.shg), c.n_ff_shexp, t)?;
                        probe(&mut tr, "shared_down", &pf.slab, at(l.shy), ne, t)?;
                        probe(&mut tr, "shared_gate", &pf.slab, at(l.sg), 1, t)?;
                    }
                    // "mo" is zero: hc_write clears it after use.
                    let fx = if ly.hc.is_some() { at(l.mo) } else { x };
                    g.moe_combine_seq(fx, at(l.ye), at(l.wts), pf.ints.addr() + 4 * (2 * n * k) as u64, k, sh, sg, ne, t)?;
                    if ly.hc.is_some() { g.hc_write(x, fx, at(l.inj), ne, c.hc_n, t, true)?; }
                }
            }
            probe(&mut tr, "| ffn", &pf.slab, x, d, t)?;
            if trace { crate::say!("nan-trace pos {} layer {il}:{tr}", self.pos); tr.clear(); }
            if il < c.n_layer {
                if let (Some(dff), Some(df)) = (&pf.dflash_feat, self.dflash.as_ref()) {
                    if let Some(idx) = df.cfg.target_layer_ids.iter().position(|&tl| tl == il) {
                        let n_ids = df.cfg.target_layer_ids.len();
                        // hc targets (qwen4exp): collapse the hc streams to their plain mean first
                        // (mirrors llama.cpp's hc_stream_mean); see the step-side note. hc_mix's gate
                        // buffer holds TMAX rows, so tile over TMAX-position blocks.
                        if c.hc_n > 0 {
                            let dfm = pf.dflash_mean.as_ref().expect("dflash_mean with hc");
                            let mut done = 0;
                            while done < t {
                                let rows = (t - done).min(TMAX);
                                g.hc_mix(pf.slab.addr() + 4 * (done * c.hc_n * ne) as u64, df.dfg.addr(), dfm.addr() + 4 * (done * ne) as u64, ne, c.hc_n, rows)?;
                                for r in 0..rows {
                                    g.dtod(dfm, 4 * (done + r) * ne, dff, 4 * ((done + r) * n_ids + idx) * ne, 4 * ne)?;
                                }
                                done += rows;
                            }
                        } else {
                            for r in 0..t {
                                g.dtod(&pf.slab, 4 * r * ne, dff, 4 * (r * n_ids + idx) * ne, 4 * ne)?;
                            }
                        }
                    }
                }
            }
            g.kick()?;
        }
        if rows.is_some() { return Ok(()); }
        if fill && is_eagle {
            g.dtod(&pf.slab, 4 * (t - 1) * d, &self.x, 0, 4 * d)?;
            let (lh, slab_h, old_cur_g) = {
                let pf = self.pf.as_ref().unwrap();
                let eg = self.mtp.as_ref().unwrap().eagle.as_ref().unwrap();
                self.gpu.gemm_dense(&eg.fc, t, pf.feat.as_ref().unwrap().addr(), 3 * ne, pf.slab.addr() + 4 * l.h as u64, ne, 0)?;
                (l.h, pf.slab.addr() + 4 * l.h as u64, eg.cur_g.addr())
            };
            if self.pos > 0 {
                self.mtp_run(&toks[..1], old_cur_g, false)?;
            }
            let tm = self.tmax();
            for (off, ch) in toks[1..].chunks(tm).enumerate() {
                self.mtp_run(ch, slab_h + 4 * (off * tm * ne) as u64, false)?;
            }
            let pf = self.pf.as_ref().unwrap();
            let eg = self.mtp.as_ref().unwrap().eagle.as_ref().unwrap();
            self.gpu.dtod(&pf.slab, 4 * (lh + (t - 1) * ne), &eg.cur_g, 0, 4 * ne)?;
        } else {
            if let Some(mh) = self.mtp.as_mut().filter(|_| fill) { mh.pos += t; }
            // Row 0 of `x` gets the last hidden state (the draft head's layer above did it when it ran: not with
            // hyper-connections).
            if !(fill && c.hc_n == 0) { g.dtod(&pf.slab, 4 * (t - 1) * d, &self.x, 0, 4 * d)?; }
        }
        if let (Some(pf), Some(df)) = (&self.pf, &self.dflash)
            && let Some(dff) = &pf.dflash_feat
        {
            let g = &self.gpu;
            let (l, n, base) = (&pf.lay, pf.chunk, pf.slab.addr());
            let at = |o: usize| base + 4 * o as u64;
            let h = at(l.h);
            let dc = &df.cfg;
            let n_ids = dc.target_layer_ids.len();
            let (df_ne, df_nkv, df_hd, df_eps) = (dc.n_embd, dc.n_head_kv, dc.head_dim, dc.eps);
            let df_kv = df_nkv * df_hd;
            g.gemm_dense(&df.fc, t, dff.addr(), n_ids * ne, at(l.a), df_ne, 0)?;
            g.rms_norm(at(l.a), df_ne, df.hidden_norm.ptr, h, df_ne, df_ne, t, df_eps)?;
            g.put(&[df.pos as i32, 0], &df.ctx_step, 0)?;
            let (kraw, kk, vv) = (at(l.c), at(l.c + n * df_kv), at(l.c + 2 * n * df_kv));
            for (li, dl) in df.layers.iter().enumerate() {
                let first_act = if li == 0 { 0 } else { SAME_X };
                g.gemm_dense(&dl.k, t, h, df_ne, kraw, df_kv, first_act)?;
                g.gemm_dense(&dl.v, t, h, df_ne, vv, df_kv, SAME_X)?;
                g.rms_norm_rope_kv(
                    kraw, df_hd, dl.kn.ptr, kk,
                    df_hd, t * df_nkv, df_eps, df_nkv, dc.n_rot, df.ctx_step.addr(), dc.rope_base,
                    vv, dl.kc[self.cur].addr(), dl.vc[self.cur].addr(),
                )?;
            }
        }
        if let Some(df) = self.dflash.as_mut() {
            df.pos += t;
            df.pend_ctx = 0;
        }
        self.pos += toks.len();
        let Some(nx) = self.next.as_deref_mut() else { return Ok(()) };
        let xs = self.gpu.get::<f32>(&self.pf.as_ref().unwrap().slab, 0, toks.len() * d)?;
        nx.prefill_chunk(toks, Some((&xs, pe)), None)
    }

    /// `step_rows` for up to `batch_cap` distinct slots, on the GEMM kernels of prefill: one decode step a row, each
    /// row reading and extending its own slot's KV. Faster than `step_rows` forwards from some row count up; its
    /// logits differ from theirs in the last bits (as prefill's do), so a greedy token at a near-tie can differ.
    pub fn step_batch(&mut self, rows: &[(usize, u32, bool)]) -> anyhow::Result<Vec<(u32, Vec<f32>)>> { self.batch_rows(rows, true) }

    /// `step_batch` over up to `Batch::rows` rows, where adjacent rows of one slot take consecutive positions (K/V of
    /// every row is stored before attention reads it, and a row reads up to its own position). `all`: an answer for
    /// every row; else only for the rows that ask for logits (a prompt's last row), in row order.
    fn batch_rows(&mut self, rows: &[(usize, u32, bool)], all: bool) -> anyhow::Result<Vec<(u32, Vec<f32>)>> {
        let t = rows.len();
        let pos: Vec<usize> = rows.iter().enumerate().map(|(r, &(s, ..))| *self.slot_pos(s) + rows[..r].iter().filter(|x| x.0 == s).count()).collect();
        let heads: Vec<usize> = (0..t).filter(|&r| all || rows[r].2).collect();
        // Recurrent layers or experts: the layers of `prefill_chunk` in rows mode (`Batch::dense` is the forward below).
        if self.bt.as_ref().is_some_and(|b| !b.dense) {
            let bt = self.bt.as_ref().unwrap();
            anyhow::ensure!((1..=bt.rows).contains(&t) && heads.len() <= bt.cap && self.rope_offset == 0, "step_batch: {t} rows, {} answers", heads.len());
            if pos.iter().any(|&p| p >= self.ctx) { anyhow::bail!("context full ({} tokens; raise --ctx)", self.ctx); }
            let g = &self.gpu;
            g.put(&pos.iter().flat_map(|&p| [p as i32, 0]).collect::<Vec<_>>(), &bt.step, 0)?;
            let mut table = vec![0u64; self.layers.len() * 2 * bt.rows];
            for (i, ly) in self.layers.iter().enumerate() {
                let Mixer::Attn { kc, vc, .. } = &ly.mixer else { continue };
                for (r, &(sl, ..)) in rows.iter().enumerate() { table[(i * bt.rows + r) * 2..][..2].copy_from_slice(&[kc[sl].addr(), vc[sl].addr()]); }
            }
            g.put(&table, &bt.kvt, 0)?;
            let splits = (pos.iter().max().unwrap() + 1).div_ceil(256).clamp(1, crate::backend::ATT_SPLITS);
            anyhow::ensure!(t * splits <= bt.cap * crate::backend::ATT_SPLITS, "step_batch: {t} rows at {splits} splits");
            let (toks, slots): (Vec<u32>, Vec<usize>) = rows.iter().map(|r| (r.1, r.0)).unzip();
            self.prefill_chunk(&toks, None, Some((&slots, splits)))?;
        }
        let res = {
            let (Some(bt), Some(pf)) = (&self.bt, &self.pf) else { anyhow::bail!("step_batch: this model decodes with step_rows") };
            anyhow::ensure!((1..=bt.rows).contains(&t) && heads.len() <= bt.cap && self.rope_offset == 0, "step_batch: {t} rows, {} answers", heads.len());
            if pos.iter().any(|&p| p >= self.ctx) { anyhow::bail!("context full ({} tokens; raise --ctx)", self.ctx); }
            let c = &self.m.cfg;
            let (g, eps, ne, nvoc) = (&self.gpu, c.eps, c.n_embd, c.n_vocab);
            let (hd, nh, nkv) = (c.head_dim, c.n_head, c.n_head_kv);
            let (kv, qs) = (nkv * hd, if c.q_gate { 2 * hd } else { hd });
            let base = pf.slab.addr();
            let at = |o: usize| base + 4 * o as u64;
            let (l, n, step) = (&pf.lay, pf.chunk, bt.step.addr());
            let (x, h) = (at(0), at(l.h));
            if bt.dense {
            g.put(&rows.iter().flat_map(|r| self.m.embed(r.1)).collect::<Vec<f32>>(), &pf.slab, 0)?;
            g.put(&pos.iter().flat_map(|&p| [p as i32, 0]).collect::<Vec<_>>(), &bt.step, 0)?;
            let mut table = vec![0u64; self.layers.len() * 2 * bt.rows];
            for (i, ly) in self.layers.iter().enumerate() {
                let Mixer::Attn { kc, vc, .. } = &ly.mixer else { unreachable!() };
                for (r, &(sl, ..)) in rows.iter().enumerate() { table[(i * bt.rows + r) * 2..][..2].copy_from_slice(&[kc[sl].addr(), vc[sl].addr()]); }
            }
            g.put(&table, &bt.kvt, 0)?;
            // The GQA kernel in every regime: with many rows the GPU is full of blocks already, so a head group takes
            // one block per 256 positions of the longest row (64 rows of 130 positions, 36 layers of Qwen3-4B on a
            // 3060: 17 ms with the splits of `step_rows`, 3.5 ms so).
            let splits = (pos.iter().max().unwrap() + 1).div_ceil(256).clamp(1, crate::backend::ATT_SPLITS);
            anyhow::ensure!(t * splits <= bt.cap * crate::backend::ATT_SPLITS, "step_batch: {t} rows at {splits} splits");
            let (kraw, kk, vv) = (at(l.c), at(l.c + n * kv), at(l.c + 2 * n * kv));
            for (i, ly) in self.layers.iter().enumerate() {
                let (Mixer::Attn { q, k, v, qn, kn, o, sink, .. }, Ffn::Dense { gate, up, down }) = (&ly.mixer, &ly.ffn) else { unreachable!() };
                let kvt = bt.kvt.addr() + (8 * i * 2 * bt.rows) as u64;
                g.rms_norm(x, ne, ly.attn_norm.as_ref().unwrap().ptr, h, ne, ne, t, eps)?;
                g.gemm_dense(q, t, h, ne, at(l.a), nh * qs, BATCHED)?;
                g.gemm_dense(k, t, h, ne, kraw, kv, SAME_X | BATCHED)?;
                g.gemm_dense(v, t, h, ne, vv, kv, SAME_X | BATCHED)?;
                let win = c.swa.get(self.first + i).copied().unwrap_or(false).then_some(c.n_swa).unwrap_or(0);
                g.rms_norm_rope_qk_rows(at(l.a), qs, qn.ptr, at(l.b), nh, kraw, kn.ptr, kk, nkv, hd, eps, c.n_rot, step, c.rope_base, vv, kvt, t)?;
                g.attn_rows(at(l.b), if c.q_gate { at(l.a) } else { 0 }, at(l.d), bt.attp.addr(), step, nh, nkv, hd, 1.0 / (hd as f32).sqrt(), t,
                            sink.as_ref().map_or(0, |s| s.ptr), win, kvt, splits)?;
                g.gemm_dense(o, t, at(l.d), nh * hd, x, ne, 2 | BATCHED)?;
                g.rms_norm(x, ne, ly.post_norm.as_ref().unwrap().ptr, h, ne, ne, t, eps)?;
                g.gemm_dense(gate, t, h, ne, at(l.ge), c.n_ff, BATCHED)?;
                g.gemm_dense(up, t, h, ne, at(l.ge), c.n_ff, 1 | SAME_X | BATCHED)?;
                g.gemm_dense(down, t, at(l.ge), c.n_ff, x, ne, 2 | BATCHED)?;
                g.kick()?;
            }
            }
            // Region a, not h: the next forward's first GEMM reads h, and the backend may keep the quantized rows of
            // the last GEMM input by its address.
            let (on, nr) = (self.output_norm.as_ref().unwrap().ptr, heads.len());
            if nr == 0 { vec![] } else {
            if all { g.rms_norm(x, ne, on, at(l.a), ne, ne, t, eps)?; }
            else { for (j, &r) in heads.iter().enumerate() { g.rms_norm(at(r * ne), ne, on, at(l.a + j * ne), ne, ne, 1, eps)?; } }
            g.gemm_dense(self.output.as_ref().unwrap(), nr, at(l.a), ne, bt.logits.addr(), nvoc, BATCHED)?;
            g.argmax_rows(bt.logits.addr(), nvoc, nr, bt.amax.addr())?;
            let am = g.get::<i32>(&bt.amax, 0, nr)?;
            heads.iter().zip(am).enumerate()
                .map(|(j, (&r, tk))| Ok((tk as u32, if rows[r].2 { g.get::<f32>(&bt.logits, j * nvoc, nvoc)? } else { vec![] })))
                .collect::<anyhow::Result<Vec<_>>>()?
            }
        };
        for &(s, ..) in rows { *self.slot_pos(s) += 1; }
        self.row_tokens += t;
        self.after_experts(t, self.row_tokens)?;
        Ok(res)
    }
}
