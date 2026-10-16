//! GPU decode forward for the architectures in `config::arch`. Routed experts resident in the expert cache run on the GPU; the
//! rest run on the CPU (`cpu_moe`) while the GPU computes hits and the shared expert. Mirrors `model.rs`.

use crate::config::LayerKind;
use crate::cpu_worker::{CpuWorker, TMAX};
use crate::backend::{ATT_SPLITS, Backend, Buf, GpuTensor};
use crate::expert_cache::ExpertCache;
use crate::model::Model;
use anyhow::bail;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::Duration;


mod prefill;

enum Mixer {
    Delta {
        qkv: GpuTensor, gate: GpuTensor, beta: GpuTensor, alpha: GpuTensor, dt_bias: GpuTensor, a: GpuTensor,
        conv_w: GpuTensor, norm: GpuTensor, out: GpuTensor, conv: Buf, s: Buf,
        /// Conv and DeltaNet states after each of the first `k` verify tokens (MTP only), for `rollback`.
        snap: Option<(Buf, Buf)>,
    },
    Attn { q: GpuTensor, k: GpuTensor, v: GpuTensor, qn: GpuTensor, kn: GpuTensor, o: GpuTensor, kc: Buf, vc: Buf },
}

struct Shared { gate_inp: GpuTensor, gate: GpuTensor, up: GpuTensor, down: GpuTensor }

enum Ffn {
    Dense { gate: GpuTensor, up: GpuTensor, down: GpuTensor },
    Moe { router: GpuTensor, shared: Option<Shared> },
}

struct Layer { attn_norm: GpuTensor, post_norm: GpuTensor, mixer: Mixer, ffn: Ffn }

/// MTP draft head: the `nextn` block (layer n_layer) with its own KV cache and CPU worker.
struct MtpHead {
    layer: Layer, eh_proj: GpuTensor, enorm: GpuTensor, hnorm: GpuTensor, head_norm: GpuTensor,
    worker: CpuWorker, step_buf: Buf, emb: Buf, eh: Buf, x: Buf, pos: usize,
    /// Accepted drafts of the last `spec_step` whose draft head entries (from the main model's hidden rows 0..) are
    /// still to be written; the next `spec_step` writes them in its first draft pass.
    pend: Vec<u32>,
}

pub struct GpuModel<'a> {
    m: &'a Model<'a>,
    pub gpu: Box<dyn Backend>,
    layers: Vec<Layer>,
    output_norm: GpuTensor,
    output: GpuTensor,
    /// Scratch activations by name.
    bufs: HashMap<&'static str, Buf>,
    x: Buf,
    worker: Option<CpuWorker>,
    logits: Buf,
    /// Argmax of each logits row (greedy MTP: no full logits copy).
    amax: Buf,
    ids: Buf,
    slots: Buf,
    /// [position, CPU worker seq] for the current token, read by kernels so the graph can replay.
    step_buf: Buf,
    /// DeltaNet (conv, s) copies per recurrent layer and the position, for `restore_state`.
    saved: Vec<(Buf, Buf)>,
    saved_pos: usize,
    /// The tokens behind the saved state when `prefill_cached` saved it (else empty).
    saved_ids: Vec<u32>,
    ctx: usize,
    pub pos: usize,
    pub cache: Option<ExpertCache>,
    /// Swap experts every this many tokens (0 = fixed placement).
    pub adapt_every: usize,
    pub hits: usize,
    pub misses: usize,
    /// Time the CPU worker spent computing missed experts (overlaps GPU work).
    pub cpu_time: Duration,
    /// MTP drafts (made, accepted).
    pub drafts: (usize, usize),
    /// MTP drafts per speculative step (0 = no draft head). May be lowered after `load`, never raised.
    pub k: usize,
    /// Prompts shorter than this go through `step` instead of the batched prefill.
    pub short: usize,
    /// Prefill scratch (None: prompts go through `step`).
    pf: Option<prefill::Scratch>,
    mtp: Option<MtpHead>,
}

impl<'a> GpuModel<'a> {
    /// `chunk`: prefill batch in tokens (0 = no batched prefill). `mtp`: load the MTP draft head (needs `nextn`).
    pub fn load(m: &'a Model<'a>, gpu: Box<dyn Backend>, ctx: usize, chunk: usize, vram_margin: usize, k: usize) -> anyhow::Result<GpuModel<'a>> {
        let c = &m.cfg;
        let up = |name: &str| -> anyhow::Result<GpuTensor> { gpu.upload(&m.w.get(name)?) };
        let lt = |il: usize, s: &str| up(&format!("blk.{il}.{s}"));
        let (free0, _) = gpu.mem_info()?;
        let (dk, nk, nv) = (c.ssm_d_state, c.ssm_n_group, c.ssm_dt_rank);
        let channels = c.ssm_d_inner + 2 * nk * dk;
        let kv = c.n_head_kv * c.head_dim;

        let k = if c.n_nextn > 0 { k.min(TMAX - 1) } else { 0 };
        let mtp = k > 0;
        let load_layer = |il: usize, kind: LayerKind| -> anyhow::Result<Layer> {
            let mixer = if kind == LayerKind::Delta {
                Mixer::Delta {
                    qkv: lt(il, "attn_qkv.weight")?, gate: lt(il, "attn_gate.weight")?, beta: lt(il, "ssm_beta.weight")?,
                    alpha: lt(il, "ssm_alpha.weight")?, dt_bias: lt(il, "ssm_dt.bias")?, a: lt(il, "ssm_a")?,
                    conv_w: lt(il, "ssm_conv1d.weight")?, norm: lt(il, "ssm_norm.weight")?, out: lt(il, "ssm_out.weight")?,
                    conv: gpu.zeros::<f32>((c.ssm_conv - 1) * channels)?, s: gpu.zeros::<f32>(nv * c.ssm_dv() * dk)?,
                    snap: if mtp { Some((gpu.zeros::<f32>(k * (c.ssm_conv - 1) * channels)?, gpu.zeros::<f32>(k * c.ssm_d_inner * dk)?)) } else { None },
                }
            } else {
                Mixer::Attn {
                    q: lt(il, "attn_q.weight")?, k: lt(il, "attn_k.weight")?, v: lt(il, "attn_v.weight")?,
                    qn: lt(il, "attn_q_norm.weight")?, kn: lt(il, "attn_k_norm.weight")?, o: lt(il, "attn_output.weight")?,
                    kc: gpu.zeros::<u16>(gpu.kv_words(ctx * kv))?, vc: gpu.zeros::<u16>(gpu.kv_words(ctx * kv))?,
                }
            };
            let ffn = if c.is_moe() {
                let shared = if m.w.layer(il, "ffn_gate_inp_shexp.weight").is_ok() {
                    Some(Shared {
                        gate_inp: lt(il, "ffn_gate_inp_shexp.weight")?, gate: lt(il, "ffn_gate_shexp.weight")?,
                        up: lt(il, "ffn_up_shexp.weight")?, down: lt(il, "ffn_down_shexp.weight")?,
                    })
                } else { None };
                Ffn::Moe { router: lt(il, "ffn_gate_inp.weight")?, shared }
            } else {
                Ffn::Dense { gate: lt(il, "ffn_gate.weight")?, up: lt(il, "ffn_up.weight")?, down: lt(il, "ffn_down.weight")? }
            };
            Ok(Layer { attn_norm: lt(il, "attn_norm.weight")?, post_norm: lt(il, "post_attention_norm.weight")?, mixer, ffn })
        };
        let layers = c.layers.iter().enumerate().map(|(il, &kind)| load_layer(il, kind)).collect::<anyhow::Result<Vec<_>>>()?;
        let mtp_parts = if mtp {
            let il = c.n_layer;
            Some((load_layer(il, LayerKind::Attn)?, lt(il, "nextn.eh_proj.weight")?, lt(il, "nextn.enorm.weight")?,
                  lt(il, "nextn.hnorm.weight")?, lt(il, "nextn.shared_head_norm.weight")?,
                  gpu.zeros::<i32>(2 * TMAX)?, gpu.zeros::<f32>(TMAX * c.n_embd)?, gpu.zeros::<f32>(TMAX * 2 * c.n_embd)?, gpu.zeros::<f32>(TMAX * c.n_embd)?))
        } else { None };
        let output_norm = up("output_norm.weight")?;
        let output = up("output.weight").or_else(|_| up("token_embd.weight"))?;

        let ke = c.n_expert_used.max(1);
        let ff = (ke * c.n_ff_exp).max(c.n_ff);
        let sizes: [(&'static str, usize); 26] = [
            ("h", c.n_embd), ("hq", c.n_embd * 9 / 32), ("mix", c.n_embd), ("qkv", channels), ("z", c.ssm_d_inner), ("beta", nv), ("g", nv),
            ("y", channels), ("o", c.ssm_d_inner), ("qg", c.n_head * 2 * c.head_dim), ("q", c.n_head * c.head_dim),
            ("kraw", kv), ("k", kv), ("v", kv), ("att", c.n_head * c.head_dim), ("rl", c.n_expert), ("wts", ke),
            ("ge", ff), ("ue", ff), ("ye", ke * c.n_embd), ("shg", c.n_ff_shexp), ("shu", c.n_ff_shexp),
            ("shy", c.n_embd), ("sg", 1), ("wh", ke), ("attp", c.n_head * ATT_SPLITS * (c.head_dim + 2)),
        ];
        let bufs = sizes.into_iter().map(|(name, n)| Ok((name, gpu.zeros::<f32>(TMAX * n)?))).collect::<anyhow::Result<HashMap<_, _>>>()?;
        let (x, logits, ids) = (gpu.zeros::<f32>(TMAX * c.n_embd)?, gpu.zeros::<f32>(TMAX * c.n_vocab)?, gpu.zeros::<i32>(TMAX * ke)?);
        let amax = gpu.zeros::<i32>(TMAX)?;
        let (slots, step_buf) = (gpu.zeros::<i32>(TMAX * ke)?, gpu.zeros::<i32>(2 * TMAX)?);
        let pf = if chunk > 0 { Some(prefill::Scratch::new(m, &*gpu, chunk.min(ctx))?) } else { None };
        let (free1, _) = gpu.mem_info()?;

        let cache = if c.is_moe() {
            let budget = free1.saturating_sub(vram_margin);
            Some(ExpertCache::new(m, &*gpu, budget, c.n_layer + mtp as usize)?)
        } else { None };
        let worker = if c.is_moe() { Some(CpuWorker::new(m, &*gpu, (0..c.n_layer).collect())?) } else { None };
        let mb = |b: usize| b as f64 / 1048576.0;
        eprintln!(
            "plan: weights+state+scratch {:.0} MB (KV {:.0} MB for {ctx} tokens, prefill {:.0} MB), expert slots {:.0} MB, resident experts {}/{}",
            mb(free0 - free1), mb(2 * 2 * gpu.kv_words(ctx * kv) * c.layers.iter().filter(|&&k| k == LayerKind::Attn).count()), mb(pf.as_ref().map_or(0, |p| p.bytes())),
            mb(cache.as_ref().map_or(0, |x| x.slot_bytes_total())),
            cache.as_ref().map_or(0, |x| x.resident_count()), cache.as_ref().map_or(0, |x| x.total_experts()),
        );
        let mtp = match mtp_parts {
            Some((layer, eh_proj, enorm, hnorm, head_norm, step_buf, emb, eh, x)) => Some(MtpHead {
                layer, eh_proj, enorm, hnorm, head_norm, worker: CpuWorker::new(m, &*gpu, vec![c.n_layer])?,
                step_buf, emb, eh, x, pos: 0, pend: vec![],
            }),
            None => None,
        };
        Ok(GpuModel { m, gpu, layers, output_norm, output, bufs, x, worker, logits, amax, ids, slots, step_buf, ctx, pos: 0, saved: vec![], saved_pos: 0, saved_ids: vec![], cache, adapt_every: 32, hits: 0, misses: 0, cpu_time: Duration::ZERO, drafts: (0, 0), k, short: 32, pf, mtp })
    }

    /// Clear KV and recurrent state (and forget the `prefill_cached` prefix, whose KV this zeroes).
    pub fn reset(&mut self) -> anyhow::Result<()> {
        self.saved_ids.clear();
        if let Some(h) = self.mtp.as_mut() { h.pos = 0; h.pend.clear(); }
        for l in self.layers.iter().chain(self.mtp.as_ref().map(|h| &h.layer)) {
            match &l.mixer {
                Mixer::Delta { conv, s, .. } => { self.gpu.memset(conv)?; self.gpu.memset(s)?; }
                Mixer::Attn { kc, vc, .. } => { self.gpu.memset(kc)?; self.gpu.memset(vc)?; }
            }
        }
        self.pos = 0;
        Ok(())
    }

    /// Run one token; returns logits over the vocabulary.
    pub fn step(&mut self, token: u32) -> anyhow::Result<Vec<f32>> {
        if self.pos >= self.ctx { bail!("context full ({} tokens; raise --ctx)", self.ctx); }
        let seq = self.worker.as_mut().map_or(0, |w| w.begin());
        let g = &self.gpu;
        g.put(&self.m.embed(token), &self.x, 0)?;
        g.put(&[self.pos as i32, seq as i32], &self.step_buf, 0)?;
        self.run(1)?;
        let logits = self.gpu.get::<f32>(&self.logits, 0, self.m.cfg.n_vocab)?;
        self.after_forward(1)?;
        Ok(logits)
    }

    /// Run up to `TMAX` tokens at positions pos.. in one forward (each weight read once); returns logits per token.
    /// Hidden state t stays in row t of `x` (see `mtp_draft`).
    pub fn verify(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<Vec<f32>>> {
        let nvoc = self.m.cfg.n_vocab;
        self.verify_run(tokens)?;
        let logits = self.gpu.get::<f32>(&self.logits, 0, tokens.len() * nvoc)?;
        self.after_forward(tokens.len())?;
        Ok(logits.chunks(nvoc).map(|c| c.to_vec()).collect())
    }

    /// `verify` for greedy speculation: `keep` gets the argmax of every row; returns its row n and that row's argmax.
    fn verify_greedy(&mut self, tokens: &[u32], keep: impl FnOnce(&[u32]) -> usize) -> anyhow::Result<(usize, u32)> {
        let (nt, nvoc) = (tokens.len(), self.m.cfg.n_vocab);
        self.verify_run(tokens)?;
        let g = &self.gpu;
        g.argmax_rows(self.logits.addr, nvoc, nt, self.amax.addr)?;
        let am: Vec<u32> = g.get::<i32>(&self.amax, 0, nt)?.into_iter().map(|x| x as u32).collect();
        let n = keep(&am);
        self.after_forward(nt)?;
        Ok((n, am[n]))
    }

    /// Enqueue the forward of `verify` (no logits copy).
    fn verify_run(&mut self, tokens: &[u32]) -> anyhow::Result<()> {
        let nt = tokens.len();
        if !(1..=TMAX).contains(&nt) { bail!("verify: {nt} tokens (1 to {TMAX})"); }
        if self.pos + nt > self.ctx { bail!("context full ({} tokens; raise --ctx)", self.ctx); }
        let seq = self.worker.as_mut().map_or(0, |w| w.begin());
        let g = &self.gpu;
        let emb: Vec<f32> = tokens.iter().flat_map(|&t| self.m.embed(t)).collect();
        g.put(&emb, &self.x, 0)?;
        let steps: Vec<i32> = (0..nt).flat_map(|t| [(self.pos + t) as i32, seq as i32]).collect();
        g.put(&steps, &self.step_buf, 0)?;
        self.run(nt)
    }

    /// Save the DeltaNet and conv states (KV needs no copy: rewinding `pos` is enough).
    pub fn save_state(&mut self) -> anyhow::Result<()> {
        let g = &self.gpu;
        if self.saved.is_empty() {
            for l in &self.layers {
                if let Mixer::Delta { conv, s, .. } = &l.mixer { self.saved.push((g.alloc(conv.len)?, g.alloc(s.len)?)); }
            }
        }
        let mut it = self.saved.iter();
        for l in &self.layers {
            if let Mixer::Delta { conv, s, .. } = &l.mixer {
                let (bc, bs) = it.next().unwrap();
                g.dtod(conv, 0, bc, 0, conv.len)?;
                g.dtod(s, 0, bs, 0, s.len)?;
            }
        }
        self.saved_pos = self.pos;
        self.saved_ids.clear();
        Ok(())
    }

    /// Keep only the first `keep` tokens of the last `verify`, which started at position `start` (1 <= keep <= `k`):
    /// recurrent states come from the snapshots it wrote, KV entries past them are dropped by rewinding `pos`.
    pub fn rollback(&mut self, start: usize, keep: usize) -> anyhow::Result<()> {
        if !(1..=self.k).contains(&keep) || start + keep > self.pos { bail!("rollback: keep {keep} of verify at {start}"); }
        let g = &self.gpu;
        for l in &self.layers {
            if let Mixer::Delta { conv, s, snap: Some((sc, ss)), .. } = &l.mixer {
                // Snapshot `keep - 1` of each state; sizes in bytes.
                let (nc, ns) = (conv.len, s.len);
                g.dtod(sc, (keep - 1) * nc, conv, 0, nc)?;
                g.dtod(ss, (keep - 1) * ns, s, 0, ns)?;
            }
        }
        self.pos = start + keep;
        Ok(())
    }

    /// Return to the state of the last `save_state`.
    pub fn restore_state(&mut self) -> anyhow::Result<()> {
        let g = &self.gpu;
        let mut it = self.saved.iter();
        for l in &self.layers {
            if let Mixer::Delta { conv, s, .. } = &l.mixer {
                let (bc, bs) = it.next().ok_or_else(|| anyhow::anyhow!("restore_state without save_state"))?;
                g.dtod(bc, 0, conv, 0, conv.len)?;
                g.dtod(bs, 0, s, 0, s.len)?;
            }
        }
        self.pos = self.saved_pos;
        Ok(())
    }

    /// Run the forward of `nt` tokens as a recorded graph (graph key `nt`), recorded on first use.
    fn run(&self, nt: usize) -> anyhow::Result<()> { self.gpu.graph(nt as u64, &mut || self.forward(nt)) }

    /// Enqueue (or record into a graph) the forward of `nt` tokens; per-token values come from `step_buf`.
    fn forward(&self, nt: usize) -> anyhow::Result<()> {
        let c = &self.m.cfg;
        let (g, eps, ne, step) = (&self.gpu, c.eps, c.n_embd, self.step_buf.addr);
        let p = |n: &str| self.bufs[n].addr;
        let x = self.x.addr;
        for (il, l) in self.layers.iter().enumerate() { self.layer(il, l, x, step, self.worker.as_ref(), nt)?; }
        g.rms_norm_q(x, ne, self.output_norm.ptr, p("h"), ne, ne, nt, eps, p("hq"))?;
        g.gemvq(&self.output, p("h"), ne, p("hq"), self.logits.addr, c.n_vocab, nt)
    }

    /// y[t] = W x[t] for t < nt (strides in floats).
    fn mv(&self, w: &GpuTensor, x: u64, xs: usize, y: u64, ys: usize, nt: usize) -> anyhow::Result<()> {
        if nt == 1 { self.gpu.gemv(w, x, y) } else { self.gpu.gemvn(w, x, xs, y, ys, nt) }
    }

    /// Last hidden state (before output_norm) of the last `step`.
    pub fn hidden(&self) -> anyhow::Result<Vec<f32>> { Ok(self.gpu.get::<f32>(&self.x, 0, self.m.cfg.n_embd)?) }

    /// MTP draft: logits for the token after `token`, from hidden state `row` of the last `step` or `verify` (the
    /// position that predicted `token`). Mirrors `Model::mtp_step`; the draft head keeps its own position and KV cache.
    pub fn mtp_draft(&mut self, token: u32, row: usize) -> anyhow::Result<Vec<f32>> {
        self.mtp_run(&[token], self.row_ptr(row), true)?;
        Ok(self.gpu.get::<f32>(&self.logits, 0, self.m.cfg.n_vocab)?)
    }

    /// Speculative decode with MTP. `t` is the next token (argmax of the last logits), predicted by hidden row `row`.
    /// Drafts `k` tokens after `t` (each from the previous draft's hidden state), verifies [t, drafts..] in one forward
    /// and keeps the drafts up to the first one the main model disagrees with (`rollback`). Returns the accepted tokens,
    /// the greedy token after the last one (argmax of its logits, first on ties) and its hidden row. Greedy output equals plain `step` decoding (verify is
    /// bit-identical to steps). The draft head entries of accepted drafts wait for the next call's first draft pass,
    /// which runs them with `t` in one batch (their main model rows 0..n precede `t`'s row n).
    pub fn spec_step(&mut self, t: u32, row: usize) -> anyhow::Result<(Vec<u32>, u32, usize)> {
        let k = self.k;
        if self.pos + k + 1 > self.ctx { return Ok((vec![t], crate::ops::argmax(&self.step(t)?) as u32, 0)); }
        let h = self.mtp.as_mut().unwrap();
        let mut first = std::mem::take(&mut h.pend);
        if first.len() != row { first.clear(); }
        first.push(t);
        let nf = first.len();
        let hp = h.pos + nf - 1;
        let mut toks = vec![t];
        let mut hid = self.row_ptr(row + 1 - nf);
        for i in 0..k {
            let d = if i == 0 { self.mtp_run(&first, hid, true)? } else { self.mtp_run(&toks[i..=i], hid, true)? };
            hid = self.mtp.as_ref().unwrap().x.addr + 4 * ((if i == 0 { nf } else { 1 } - 1) * self.m.cfg.n_embd) as u64;
            toks.push(d);
        }
        let start = self.pos;
        let (n, next) = self.verify_greedy(&toks, |am| (0..k).take_while(|&i| am[i] == toks[i + 1]).count())?;
        self.drafts.0 += k;
        self.drafts.1 += n;
        if n < k { self.rollback(start, n + 1)?; }
        // Draft head entries past t were made from draft hidden states: the next call redoes them from main rows.
        let h = self.mtp.as_mut().unwrap();
        h.pos = hp + 1;
        h.pend = toks[1..=n].to_vec();
        toks.truncate(n + 1);
        Ok((toks, next, n))
    }

    /// Address of hidden row `row` of the last `step` or `verify`.
    fn row_ptr(&self, row: usize) -> u64 { self.x.addr + 4 * (row * self.m.cfg.n_embd) as u64 }

    /// Draft head on `tokens` with hidden states at `hid` (rows of n_embd), as one batch at the next draft positions;
    /// returns the argmax of the last token's logits (left in `logits` row 0), or 0 when `head` = false skips the head.
    fn mtp_run(&mut self, tokens: &[u32], hid: u64, head: bool) -> anyhow::Result<u32> {
        let nt = tokens.len();
        let c = &self.m.cfg;
        let Some(h) = self.mtp.as_mut() else { bail!("MTP head not loaded") };
        if h.pos + nt > self.ctx { bail!("context full ({} tokens; raise --ctx)", self.ctx); }
        let seq = h.worker.begin();
        let g = &self.gpu;
        let emb: Vec<f32> = tokens.iter().flat_map(|&t| self.m.embed(t)).collect();
        g.put(&emb, &h.emb, 0)?;
        let steps: Vec<i32> = (0..nt).flat_map(|t| [(h.pos + t) as i32, seq as i32]).collect();
        g.put(&steps, &h.step_buf, 0)?;
        // `run` keys are token counts (below 8); the top bit keeps these keys apart from them.
        let key = { let mut s = std::hash::DefaultHasher::new(); (hid, nt, head).hash(&mut s); s.finish() | 1 << 63 };
        g.graph(key, &mut || self.mtp_forward(hid, nt, head))?;
        let d = if head { g.get::<i32>(&self.amax, 0, 1)?[0] as u32 } else { g.sync()?; 0 };
        let h = self.mtp.as_mut().unwrap();
        h.pos += nt;
        if let Some(cache) = self.cache.as_mut() {
            cache.record(c.n_layer, &h.worker.routing(c.n_layer).iter().map(|x| x.0).collect::<Vec<_>>());
        }
        Ok(d)
    }

    /// Enqueue the draft head (see `mtp_run`).
    fn mtp_forward(&self, hid: u64, nt: usize, head: bool) -> anyhow::Result<()> {
        let (c, g, h) = (&self.m.cfg, &self.gpu, self.mtp.as_ref().unwrap());
        let (eps, ne, hp) = (c.eps, c.n_embd, self.bufs["h"].addr);
        let (eh, x) = (h.eh.addr, h.x.addr);
        g.rms_norm(h.emb.addr, ne, h.enorm.ptr, eh, 2 * ne, ne, nt, eps)?;
        g.rms_norm(hid, ne, h.hnorm.ptr, eh + 4 * ne as u64, 2 * ne, ne, nt, eps)?;
        self.mv(&h.eh_proj, eh, 2 * ne, x, ne, nt)?;
        self.layer(c.n_layer, &h.layer, x, h.step_buf.addr, Some(&h.worker), nt)?;
        if head {
            g.rms_norm(x + 4 * ((nt - 1) * ne) as u64, 0, h.head_norm.ptr, hp, 0, ne, 1, eps)?;
            g.gemv(&self.output, hp, self.logits.addr)?;
            g.argmax_rows(self.logits.addr, c.n_vocab, 1, self.amax.addr)?;
        }
        Ok(())
    }

    /// One decoder block on `nt` tokens at `x` (rows of n_embd); `step` points at [position, worker seq] per token.
    fn layer(&self, il: usize, l: &Layer, x: u64, step: u64, worker: Option<&CpuWorker>, nt: usize) -> anyhow::Result<()> {
        let c = &self.m.cfg;
        let (g, eps, ne) = (&self.gpu, c.eps, c.n_embd);
        let p = |n: &str| self.bufs[n].addr;
        // Inputs from "h" are quantized once ("hq", after each norm) instead of once per GEMV block.
        let mv = |w: &GpuTensor, x: &str, xs: usize, y: &str, ys: usize|
            if x == "h" { g.gemvq(w, p(x), xs, p("hq"), p(y), ys, nt) } else { self.mv(w, p(x), xs, p(y), ys, nt) };
        let (dk, nk, nv) = (c.ssm_d_state, c.ssm_n_group, c.ssm_dt_rank);
        let (dv, channels) = (c.ssm_dv(), c.ssm_d_inner + 2 * nk * dk);
        let (hd, nh, nkv) = (c.head_dim, c.n_head, c.n_head_kv);
        let kv = nkv * hd;
        {
            g.rms_norm_q(x, ne, l.attn_norm.ptr, p("h"), ne, ne, nt, eps, p("hq"))?;
            match &l.mixer {
                Mixer::Delta { qkv, gate, beta, alpha, dt_bias, a, conv_w, norm, out, conv, s, snap } => {
                    let (sc, ss, ns) = snap.as_ref().map_or((0, 0, 0), |(a, b)| (a.addr, b.addr, self.k));
                    mv(qkv, "h", ne, "qkv", channels)?;
                    mv(gate, "h", ne, "z", c.ssm_d_inner)?;
                    mv(beta, "h", ne, "beta", nv)?;
                    mv(alpha, "h", ne, "g", nv)?;
                    g.ssm_gates(p("beta"), p("g"), dt_bias.ptr, a.ptr, nt * nv, nv)?;
                    if nt == 1 {
                        g.conv_silu(p("qkv"), conv.addr, conv_w.ptr, p("y"), channels, c.ssm_conv)?;
                    } else {
                        g.conv_seq(p("qkv"), conv.addr, conv_w.ptr, p("y"), channels, c.ssm_conv, nt, sc, ns)?;
                    }
                    g.l2_norm(p("y"), dk, 2 * nk, channels, nt * 2 * nk, eps)?;
                    g.deltanet(p("y"), s.addr, p("beta"), p("g"), p("o"), nk, nv, c.k_interleave, dk, dv, 1.0 / (dk as f32).sqrt(), nt, channels, ss, ns)?;
                    g.norm_gate(p("o"), norm.ptr, p("z"), dv, nt * nv, eps)?;
                    mv(out, "o", c.ssm_d_inner, "mix", ne)?;
                }
                Mixer::Attn { q, k, v, qn, kn, o, kc, vc } => {
                    let qs = if c.q_gate { 2 * hd } else { hd };
                    mv(q, "h", ne, "qg", nh * qs)?;
                    g.rms_norm(p("qg"), qs, qn.ptr, p("q"), hd, hd, nt * nh, eps)?;
                    g.rope_neox(p("q"), hd, nt * nh, nh, c.n_rot, step, c.rope_base)?;
                    mv(k, "h", ne, "kraw", kv)?;
                    mv(v, "h", ne, "v", kv)?;
                    g.rms_norm(p("kraw"), hd, kn.ptr, p("k"), hd, hd, nt * nkv, eps)?;
                    g.rope_neox(p("k"), hd, nt * nkv, nkv, c.n_rot, step, c.rope_base)?;
                    g.kv_store(p("k"), p("v"), kc.addr, vc.addr, kv, nt, step, hd)?;
                    g.attn_decode(p("q"), kc.addr, vc.addr, if c.q_gate { p("qg") } else { 0 }, p("att"), p("attp"), step, nh, nkv, hd,
                                  1.0 / (hd as f32).sqrt(), nt)?;
                    mv(o, "att", nh * hd, "mix", ne)?;
                }
            }
            g.add(x, p("mix"), nt * ne)?;
            g.rms_norm_q(x, ne, l.post_norm.ptr, p("h"), ne, ne, nt, eps, p("hq"))?;
            match &l.ffn {
                Ffn::Dense { gate, up, down } => {
                    mv(gate, "h", ne, "ge", c.n_ff)?;
                    mv(up, "h", ne, "ue", c.n_ff)?;
                    g.swiglu(p("ge"), p("ue"), nt * c.n_ff)?;
                    mv(down, "ge", c.n_ff, "ye", ne)?;
                    g.add(x, p("ye"), nt * ne)?;
                }
                Ffn::Moe { router, shared } => {
                    let k = c.n_expert_used;
                    let kt = nt * k;
                    let ids = self.ids.addr;
                    mv(router, "h", ne, "rl", c.n_expert)?;
                    g.router_topk(p("rl"), c.n_expert, k, c.expert_weights_scale, ids, p("wts"), nt)?;
                    let cache = self.cache.as_ref().unwrap();
                    let lx = cache.layer(il).unwrap();
                    let (base, stride) = cache.base(lx);
                    let [mg, mu, md] = lx.mats;
                    let slots = self.slots.addr;
                    let w = worker.unwrap();
                    let mb = w.dev + (4 * il * w.layout.stride) as u64;
                    g.map_hits(ids, lx.table_dev.addr, slots, p("wts"), p("wh"), kt)?;
                    g.moe_publish(ids, p("wts"), slots, p("h"), mb, k, ne, w.layout.h, step, nt)?;
                    if mg.ty == mu.ty && (mg.rows, mg.cols) == (mu.rows, mu.cols) {
                        let pair = (mu.off as i64 - mg.off as i64, p("ue") as i64 - p("ge") as i64);
                        g.moe_gemv(mg.ty, mg.rows, mg.cols, base + mg.off as u64, stride, slots, kt, p("h"), ne, k, p("ge"), Some(pair), p("hq"))?;
                    } else {
                        g.moe_gemv(mg.ty, mg.rows, mg.cols, base + mg.off as u64, stride, slots, kt, p("h"), ne, k, p("ge"), None, p("hq"))?;
                        g.moe_gemv(mu.ty, mu.rows, mu.cols, base + mu.off as u64, stride, slots, kt, p("h"), ne, k, p("ue"), None, p("hq"))?;
                    }
                    g.swiglu(p("ge"), p("ue"), kt * c.n_ff_exp)?;
                    g.moe_gemv(md.ty, md.rows, md.cols, base + md.off as u64, stride, slots, kt, p("ge"), c.n_ff_exp, 1, p("ye"), None, 0)?;
                    let (sh, sg) = match shared {
                        Some(s) => {
                            mv(&s.gate_inp, "h", ne, "sg", 1)?;
                            mv(&s.gate, "h", ne, "shg", c.n_ff_shexp)?;
                            mv(&s.up, "h", ne, "shu", c.n_ff_shexp)?;
                            g.swiglu(p("shg"), p("shu"), nt * c.n_ff_shexp)?;
                            mv(&s.down, "shg", c.n_ff_shexp, "shy", ne)?;
                            (p("shy"), p("sg"))
                        }
                        None => (0, 0),
                    };
                    g.moe_combine(x, p("ye"), p("wh"), k, sh, sg, mb, w.layout.y, step, slots, ne, nt)?;
                }
            }
        }
        Ok(())
    }

    /// Advance the position and update expert statistics and placement after a token.
    fn after_forward(&mut self, nt: usize) -> anyhow::Result<()> {
        self.pos += nt;
        if let (Some(cache), Some(w)) = (self.cache.as_mut(), self.worker.as_ref()) {
            for il in 0..self.layers.len() {
                if cache.layer(il).is_none() { continue; }
                let r = w.routing(il);
                let miss = r.iter().filter(|x| x.2 < 0).count();
                self.misses += miss;
                self.hits += r.len() - miss;
                cache.record(il, &r.iter().map(|x| x.0).collect::<Vec<_>>());
            }
            self.cpu_time = w.busy();
            if self.adapt_every > 0 && (self.pos - nt) / self.adapt_every < self.pos / self.adapt_every { cache.adapt(self.m, &*self.gpu, 16)?; }
        }
        Ok(())
    }
}
