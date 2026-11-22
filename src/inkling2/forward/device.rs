//! One decode token on the device: the stream, the depth mix, the loop controllers, the attention rows and the
//! convolution state stay in VRAM. A layer run comes back to the host once, for the router's input; the routed
//! experts then run where `Inkling2Model::plan` puts them, as in `ffn`.
//! `Inkling2Model::run` is the reference and the path of a prompt chunk. The attention rows and the convolution
//! state move between the two when the other side runs next (`to_device`, `to_host`).
//! The state is made at the first decoded token, after the prompt filled the expert cache, so `room` frees the least
//! recently used experts when VRAM is short.
use super::*;
use crate::backend::{Backend, Buf};
use crate::deepseek::mat::dev;
use half::slice::HalfBitsSliceExt;

struct DLayer { res_query: Option<Buf>, norms: Buf, q_clip: Buf, rel_bias: Buf, rel_clip: Buf, convs: [Buf; 4], ffn_query: Buf, ffn_norm: Buf }
/// The controllers of a core layer, Q, K, V and FFN one after another, as f16.
struct DLoop { msg_query: Buf, down: Buf, gain: Buf, forget: Buf }
/// Attention rows as f16 `[row][key head][channel]` and the four convolution histories, one after another. A
/// sliding-window slot is a ring of `n_swa` rows (position p in row p % n_swa); the others grow.
struct DSlot { k: Buf, v: Buf, hist: Buf }

/// Device addresses of the regions of the scratch buffer.
#[derive(Clone, Copy)]
struct Scratch {
    cur: u64, x: u64, blocks: u64, bank: u64, outb: u64, prelude: u64, msg: u64, ins: u64, ins2: u64, gate: u64, nrm: u64,
    q: u64, r: u64, sink: u64, og: u64, k: u64, v: u64, kp: u64, vp: u64, att: u64, proj: u64, h: u64, h2: u64, hn: u64,
    y: u64, d: u64, ids: u64, xh: u64, xq: u64, wc: u64, acc: u64, logits: u64, end: u64,
}

impl Scratch {
    fn at(base: u64, c: &Inkling2Config, rank: usize, vocab: usize) -> Self {
        let (n, q, mut a) = (c.n_embd, c.n_head * c.head_dim, base);
        let mut f = |floats: usize| { a += 4 * floats as u64; a - 4 * floats as u64 };
        Scratch {
            cur: f(n), x: f(n), blocks: f(4 * n), bank: f(4 * n), outb: f(n), prelude: f(n), msg: f(4 * n), ins: f(3 * n), ins2: f(3 * n),
            gate: f(4 * rank), nrm: f(3 * n), q: f(q), r: f(c.n_head * c.n_rel), sink: f(c.n_head), og: f(q), k: f(c.n_head_kv * c.head_dim),
            v: f(c.n_head_kv * c.head_dim), kp: f(c.n_head_kv * c.n_conv), vp: f(c.n_head_kv * c.n_conv), att: f(q), proj: f(n), h: f(n),
            h2: f(n), hn: f(n), y: f(c.n_used * 2 * c.n_ff_exp), d: f(c.n_used * n), ids: f(3 * c.n_used), xh: f(n), xq: f(c.n_used * c.n_ff_exp), wc: f(c.n_used + n), acc: f(n),
            logits: f(vocab), end: f(0),
        }
    }
}

pub(super) struct Device {
    layers: Vec<DLayer>,
    loops: Vec<DLoop>,
    slots: Vec<DSlot>,
    out_query: Buf,
    out_norm: Buf,
    scr: Buf,
    s: Scratch,
    /// Attention partials of every 64 rows; grows with the full-attention rows.
    sc: Buf,
    rank: usize,
    /// Positions whose rows both sides hold, and whether the device has run tokens after them.
    synced: usize,
    ahead: bool,
}

fn up<T: crate::backend::Plain>(be: &dyn Backend, v: &[T]) -> Result<Buf> {
    let b = be.alloc(size_of_val(v))?;
    be.put(v, &b, 0)?;
    Ok(b)
}

fn f(floats: usize) -> u64 { 4 * floats as u64 }

/// Free expert slots until `need` bytes of VRAM are free.
fn room(d: &mut dev::Dev, need: usize) {
    let free = d.parts[0].be.mem_info().map_or(0, |m| m.0);
    if free < need { d.release(0, need - free); }
}

/// Bytes of the attention partials for `rows` full-attention rows.
fn sc_bytes(c: &Inkling2Config, rows: usize) -> usize { 4 * c.n_head * rows.div_ceil(64) * (c.head_dim + 2) }

/// `moe_gemv` operands for matrices at `ptrs`: the lowest address, a step every address is a multiple of from it, and
/// each one's index in steps. None when an index does not fit an i32.
fn slab(ptrs: &[u64]) -> Option<(u64, usize, Vec<i32>)> {
    let base = *ptrs.iter().min()?;
    let gcd = |mut a: u64, mut b: u64| { while b != 0 { (a, b) = (b, a % b); } a };
    let step = ptrs.iter().fold(0, |g, &p| gcd(g, p - base)).max(1);
    Some((base, step as usize, ptrs.iter().map(|&p| i32::try_from((p - base) / step).ok()).collect::<Option<_>>()?))
}

/// The kernels index the scratch buffer and the small tensors with the config's sizes and do not check bounds: a file
/// whose tensors have other lengths runs on the host, where a wrong length panics instead of reading or writing past a
/// buffer on the card.
fn shapes_fit(m: &Inkling2Model, rank: usize) -> bool {
    let c = &m.cfg;
    let (n, q, kw, nh, nc, nf) = (c.n_embd, c.n_head * c.head_dim, c.n_head_kv * c.head_dim, c.n_head, c.n_conv, c.n_ff_exp);
    let mat = |x: &Mat, rows: usize, cols: usize| x.rows == rows && x.cols == cols;
    // Residual blocks held at once: those of the prelude, one pass of the core and the coda (the `blocks` region).
    let starts = |r: std::ops::Range<usize>| r.filter(|i| i % c.block_size == 0).count();
    let blocks = starts(0..c.loop_start) + starts(c.loop_start..c.loop_end) + starts(c.loop_end..c.n_layer);
    blocks <= 4 && rank > 0 && mat(&m.output, m.output.rows, n) && m.out_query.len() == n && m.out_norm.len() == n
        && m.loops.iter().all(|l| {
            l.msg_query.len() == 4 * n && [(&l.down, 2 * n), (&l.gain, n), (&l.forget, n)].iter().all(|(w, cols)| w.len() == 4 && w.iter().all(|x| x.len() == rank * cols))
        })
        && m.layers.iter().all(|l| {
            let (gu, dn) = (l.gate_up.expert_host(0, c.n_expert), l.down.expert_host(0, c.n_expert));
            [(&l.wq, q), (&l.gate, q), (&l.wk, kw), (&l.wv, kw), (&l.rel, nh * c.n_rel), (&l.sink, nh)].iter().all(|(x, r)| mat(x, *r, n))
                && mat(&l.wo, n, q) && mat(&gu, 2 * nf, n) && mat(&dn, n, nf)
                && l.conv_proj.as_ref().is_none_or(|p| p.iter().all(|x| mat(x, c.n_head_kv * nc, n)))
                && l.res_query.as_ref().is_none_or(|r| !r.is_empty() && r.len() % n == 0 && r.len() <= 3 * n)
                && (1..=3).contains(&l.norms.len()) && l.norms.iter().all(|x| x.len() == n)
                && l.q_clip.len() == q && l.rel_clip.len() == nh * c.n_rel && l.rel_bias.len() == c.n_rel * c.rel_extent
                && [kw, kw, n, n].iter().zip(&l.convs).all(|(ch, w)| w.len() == ch * nc)
                && l.ffn_query.len() == n && l.ffn_norm.len() == n
        })
}

/// The layer whose weights slot `il` runs.
fn layer_of(c: &Inkling2Config, il: usize) -> usize {
    if il < c.n_layer { il } else { c.loop_start + (il - c.n_layer) % (c.loop_end - c.loop_start) }
}

/// Floats in the histories of a slot's K, V, attention-output and FFN-output convolutions, and where each starts.
fn hists(c: &Inkling2Config) -> ([usize; 4], usize) {
    let (kw, n, m) = (c.n_head_kv * c.head_dim, c.n_embd, c.n_conv - 1);
    ([0, m * kw, 2 * m * kw, 2 * m * kw + m * n], 2 * m * (kw + n))
}

/// The runs of the positions `from..to` that are contiguous rows of a slot: `(first position, first row, rows)`.
fn runs(from: usize, to: usize, ring: usize) -> impl Iterator<Item = (usize, usize, usize)> {
    let mut a = from.max(to.saturating_sub(ring));
    std::iter::from_fn(move || (a < to).then(|| {
        let b = to.min((a / ring + 1) * ring);
        let r = (a, a % ring, b - a);
        a = b;
        r
    }))
}

impl Device {
    /// None when a token cannot run on the device: no GPU or more than one, a dense matrix that is not in VRAM, a
    /// shape the kernels do not take, a tensor trace, no room, or `STRATA_DS_DEVICE=0`.
    pub(super) fn new(m: &Inkling2Model) -> Option<Self> {
        let (c, pos) = (&m.cfg, m.toks.len());
        let off = std::env::var("STRATA_DS_DEVICE").is_ok_and(|v| v == "0") || m.trace || std::env::var_os("STRATA_DS_DUMP").is_some();
        if off || c.n_conv > 8 || 128 % c.head_dim != 0 || m.loops.is_empty() { return None; }
        let rank = m.loops[0].down[0].len() / (2 * c.n_embd);
        if !shapes_fit(m, rank) { return None; }
        let mut d = dev::get()?.lock().unwrap();
        if d.parts.len() != 1 { return None; }
        let dense = |x: &Mat| !x.host_only();
        if !dense(&m.output) || !m.layers.iter().all(|l| {
            [l.wq, l.wk, l.wv, l.wo, l.gate, l.rel, l.sink].iter().chain(l.conv_proj.iter().flatten()).all(dense)
        }) { return None; }
        let (kw, nr, ext) = (c.n_head_kv * c.head_dim, c.n_rel, c.rel_extent);
        let full = (pos + 1 + 4096).min(c.n_ctx);
        let rows: usize = (0..c.n_slot()).map(|il| if c.swa[layer_of(c, il)] { c.n_swa } else { full }).sum();
        // Rows, partials and scratch, plus 512 MB for the small weights and what the driver needs.
        room(&mut d, 4 * rows * kw + sc_bytes(c, full) + Scratch::at(0, c, rank, m.output.rows).end as usize + (512 << 20));
        let be = &*d.parts[0].be;
        let made = (|| -> Result<Self> {
            let layers = m.layers.iter().map(|l| Ok(DLayer {
                res_query: l.res_query.as_ref().map(|q| up(be, q)).transpose()?,
                norms: up(be, &l.norms.concat())?,
                q_clip: up(be, &l.q_clip)?,
                // The file's order, `[distance][channel]`.
                rel_bias: up(be, &(0..nr * ext).map(|i| l.rel_bias[i % nr * ext + ext - 1 - i / nr]).collect::<Vec<_>>())?,
                rel_clip: up(be, &l.rel_clip)?,
                convs: [up(be, &l.convs[0])?, up(be, &l.convs[1])?, up(be, &l.convs[2])?, up(be, &l.convs[3])?],
                ffn_query: up(be, &l.ffn_query)?,
                ffn_norm: up(be, &l.ffn_norm)?,
            })).collect::<Result<Vec<_>>>()?;
            let half = |w: &[Vec<f16>]| up(be, w.concat().reinterpret_cast());
            let loops = m.loops.iter().map(|l| {
                Ok(DLoop { msg_query: up(be, &l.msg_query)?, down: half(&l.down)?, gain: half(&l.gain)?, forget: half(&l.forget)? })
            }).collect::<Result<Vec<_>>>()?;
            let slots = (0..c.n_slot()).map(|il| {
                let rows = if c.swa[layer_of(c, il)] { c.n_swa } else { full };
                Ok(DSlot { k: be.alloc(2 * rows * kw)?, v: be.alloc(2 * rows * kw)?, hist: be.zeros::<f32>(hists(c).1)? })
            }).collect::<Result<Vec<_>>>()?;
            let scr = be.alloc(Scratch::at(0, c, rank, m.output.rows).end as usize)?;
            let s = Scratch::at(scr.addr(), c, rank, m.output.rows);
            Ok(Device { layers, loops, slots, out_query: up(be, &m.out_query)?, out_norm: up(be, &m.out_norm)?, scr, s, sc: be.alloc(sc_bytes(c, full))?, rank, synced: 0, ahead: false })
        })();
        made.inspect_err(|e| eprintln!("inkling2: no room in VRAM for the decode state ({e:#}); the tokens run as prompt chunks do")).ok()
    }

    /// The model starts over: nothing is shared with the host's rows.
    pub(super) fn forget(&mut self) {
        (self.synced, self.ahead) = (0, false);
    }

    /// Room for the rows of position `pos` in the slots that grow.
    pub(super) fn prepare(&mut self, m: &Inkling2Model, pos: usize) -> Result<()> {
        let (c, kw) = (&m.cfg, m.cfg.n_head_kv * m.cfg.head_dim);
        let mut d = dev::get().unwrap().lock().unwrap();
        let grow = |cap: usize| (pos + 1 + 4096).max(cap + cap / 2).min(c.n_ctx);
        let full = self.slots.iter().enumerate().find(|&(il, _)| !c.swa[layer_of(c, il)]).map(|(_, s)| s.k.len() / (2 * kw));
        if let Some(cap) = full && pos >= cap {
            let n = (0..c.n_slot()).filter(|&il| !c.swa[layer_of(c, il)]).count();
            room(&mut d, 4 * n * grow(cap) * kw + sc_bytes(c, grow(cap)) + (128 << 20));
        }
        let be = &*d.parts[0].be;
        for (il, s) in self.slots.iter_mut().enumerate() {
            let cap = s.k.len() / (2 * kw);
            if c.swa[layer_of(c, il)] || pos < cap { continue; }
            for b in [&mut s.k, &mut s.v] {
                let new = be.alloc(2 * grow(cap) * kw)?;
                be.dtod(b, 0, &new, 0, b.len())?;
                be.sync()?;  // the copy reads the buffer that `*b = new` frees
                *b = new;
            }
            let need = sc_bytes(c, s.k.len() / (2 * kw));
            if self.sc.len() < need {
                be.sync()?;
                self.sc = be.alloc(need)?;
            }
        }
        Ok(())
    }

    /// Copy what the host has run since the last exchange: the rows of the positions `synced..pos` that a token
    /// still sees, and the convolution histories.
    fn to_device(&mut self, m: &Inkling2Model, be: &dyn Backend, pos: usize) -> Result<()> {
        let (c, hd) = (&m.cfg, m.cfg.head_dim);
        let (kw, (at, len)) = (c.n_head_kv * hd, hists(c));
        for (il, (ds, hs)) in self.slots.iter().zip(&m.slots).enumerate() {
            let mut h = vec![0f32; len];
            for (j, x) in hs.hist.iter().enumerate() { h[at[j]..at[j] + x.len()].copy_from_slice(x); }
            be.put(&h, &ds.hist, 0)?;
            let ring = if c.swa[layer_of(c, il)] { c.n_swa } else { c.n_ctx };
            for (a, row, rows) in runs(self.synced, pos, ring) {
                for (host, buf) in [(&hs.k, &ds.k), (&hs.v, &ds.v)] {
                    // The host keeps a run of rows per key head, the row of position `pos - 1` last.
                    let first = host[0].len() / hd - (pos - a);
                    let mut x = vec![f16::ZERO; rows * kw];
                    for (r, x) in x.chunks_exact_mut(hd).enumerate() {
                        x.copy_from_slice(&host[r % c.n_head_kv][(first + r / c.n_head_kv) * hd..][..hd]);
                    }
                    be.put(x.reinterpret_cast(), buf, row * kw)?;
                }
            }
        }
        (self.synced, self.ahead) = (pos, true);
        Ok(())
    }

    /// The reverse of `to_device`, before the host runs tokens after the device did.
    pub(super) fn to_host(&mut self, c: &Inkling2Config, layers: &[Layer], slots: &mut [Slot], pos: usize) {
        if !self.ahead { return; }
        let d = dev::get().unwrap().lock().unwrap();
        let be = &*d.parts[0].be;
        let (hd, (at, len)) = (c.head_dim, hists(c));
        let kw = c.n_head_kv * hd;
        for (il, (ds, hs)) in self.slots.iter().zip(slots).enumerate() {
            let h: Vec<f32> = be.get(&ds.hist, 0, len).expect("inkling2 device state");
            for j in 0..4 { hs.hist[j] = h[at[j]..if j < 3 { at[j + 1] } else { len }].to_vec(); }
            let ring = if layers[layer_of(c, il)].swa { c.n_swa } else { c.n_ctx };
            for (_, row, rows) in runs(self.synced, pos, ring) {
                for (host, buf) in [(&mut hs.k, &ds.k), (&mut hs.v, &ds.v)] {
                    let x: Vec<u16> = be.get(buf, row * kw, rows * kw).expect("inkling2 device state");
                    host.resize(c.n_head_kv, vec![]);
                    for (r, x) in x.chunks_exact(hd).enumerate() { host[r % c.n_head_kv].extend_from_slice(x.reinterpret_cast::<f16>()); }
                }
            }
        }
        (self.synced, self.ahead) = (pos, false);
    }

    /// Run one token at the end of `m.toks` (the caller appends it). Returns the logits of the next one.
    pub(super) fn step(&mut self, m: &Inkling2Model, tok: u32) -> Vec<f32> {
        let (c, pos) = (&m.cfg, m.toks.len());
        assert!(pos < c.n_ctx, "context is limited to {} tokens", c.n_ctx);
        assert!((tok as usize) < m.tok_embd.rows, "token {tok} is outside the vocabulary");
        let (n, core, o) = (c.n_embd, c.loop_end - c.loop_start, self.scr.addr());
        let (mut emb, mut cur) = (vec![0f32; n], vec![0f32; n]);
        m.tok_embd.row_into(tok as usize, &mut emb);
        rms_norm_w(&emb, &m.tok_norm, c.eps, &mut cur);
        let lock = || dev::get().unwrap().lock().unwrap();
        // A copy of n floats inside the scratch buffer.
        let copy = |be: &dyn Backend, scr: &Buf, from: u64, to: u64| be.dtod(scr, (from - o) as usize, scr, (to - o) as usize, 4 * n).unwrap();
        {
            let d = lock();
            let be = &*d.parts[0].be;
            if !self.ahead { self.to_device(m, be, pos).expect("inkling2 device state"); }
            be.put(&cur, &self.scr, ((self.s.cur - o) / 4) as usize).unwrap();
        }
        let mut nb = 0;
        for il in 0..c.loop_start { self.layer(m, il, il, false, pos, &mut nb); }
        // Every pass of the core starts from the prelude output.
        copy(&*lock().parts[0].be, &self.scr, self.s.cur, self.s.prelude);
        let n_pre = nb;
        for pass in 0..c.loop_count {
            // The bank of the previous pass: its blocks after the first core block, and its output.
            let n_bank = nb.saturating_sub(n_pre + 1);
            if pass > 0 {
                let d = lock();
                let (be, s) = (&*d.parts[0].be, self.s);
                for i in 0..n_bank { copy(be, &self.scr, s.blocks + f((n_pre + 1 + i) * n), s.bank + f(i * n)); }
                copy(be, &self.scr, s.cur, s.outb);
                copy(be, &self.scr, s.prelude, s.cur);
                nb = n_pre;
            }
            for ip in c.loop_start..c.loop_end {
                let il = if pass == 0 { ip } else { c.n_layer + (pass - 1) * core + ip - c.loop_start };
                if pass > 0 {
                    let s = self.s;
                    lock().parts[0].be.ink_mix(s.bank, n_bank, s.outb, self.loops[ip - c.loop_start].msg_query.addr(), 4, s.msg, n).unwrap();
                }
                self.layer(m, il, ip, pass > 0, pos, &mut nb);
            }
        }
        for il in c.loop_end..c.n_layer { self.layer(m, il, il, false, pos, &mut nb); }
        let t = std::time::Instant::now();
        let (d, s) = (lock(), self.s);
        let be = &*d.parts[0].be;
        let mut logits = (|| -> Result<Vec<f32>> {
            be.ink_mix(s.blocks, nb, s.cur, self.out_query.addr(), 1, s.h, n)?;
            be.rms_norm(s.h, n, self.out_norm.addr(), s.hn, n, n, 1, c.eps)?;
            be.gemv(d.tensor(m.output.gpu.unwrap()), s.hn, s.logits)?;
            be.get(&self.scr, ((s.logits - o) / 4) as usize, c.n_vocab)
        })().expect("inkling2 device step");
        if c.softcap != 0.0 { logits.par_iter_mut().for_each(|v| *v = c.softcap * (*v / c.softcap).tanh()); }
        m.lap(7, t);
        logits
    }

    /// One layer run of the token: slot `il` (state) with the weights of layer `ip`, as `Inkling2Model::layer`.
    /// `msgs`: the four cross-pass messages are in the scratch buffer.
    fn layer(&mut self, m: &Inkling2Model, il: usize, ip: usize, msgs: bool, pos: usize, nb: &mut usize) {
        let (c, ml, l, ds, s, rank) = (&m.cfg, &m.layers[ip], &self.layers[ip], &self.slots[il], self.s, self.rank);
        let (n, hd, nh, nkv, nr, nc, nf) = (c.n_embd, c.head_dim, c.n_head, c.n_head_kv, c.n_rel, c.n_conv, c.n_ff_exp);
        let (kw, o) = (nkv * hd, self.scr.addr());
        let lp = &self.loops[ip.clamp(c.loop_start, c.loop_end - 1) - c.loop_start];
        let hist = |j: usize| ds.hist.addr() + f(hists(c).0[j]);
        let lock = || dev::get().unwrap().lock().unwrap();
        let t = std::time::Instant::now();
        let mut hn = vec![0f32; n];
        {
            let d = lock();
            let be = &*d.parts[0].be;
            let w = |x: &Mat| d.tensor(x.gpu.unwrap());
            (|| -> Result<()> {
                // Attention inputs: a sliding-window layer mixes one, a full-attention layer one each for Q, K and V.
                // The messages make the three of a sliding-window layer differ.
                let (mut ins, mut n_in) = (s.cur, 1);
                if let Some(q) = &l.res_query {
                    (ins, n_in) = (s.ins, q.len() / (4 * n));
                    be.ink_mix(s.blocks, *nb, s.cur, q.addr(), n_in, ins, n)?;
                }
                if msgs {
                    be.ink_ctrl(lp.down.addr(), lp.gain.addr(), lp.forget.addr(), ins, n_in, s.msg, s.gate, s.ins2, n, rank, 3)?;
                    (ins, n_in) = (s.ins2, 3);
                }
                let n_w = l.norms.len() / (4 * n);
                let n_n = n_in.max(n_w);
                if n_w == 1 {
                    be.rms_norm(ins, n, l.norms.addr(), s.nrm, n, n, n_n, c.eps)?;
                } else {
                    for j in 0..n_n { be.rms_norm(ins + f(j.min(n_in - 1) * n), n, l.norms.addr() + f(j * n), s.nrm + f(j * n), n, n, 1, c.eps)?; }
                }
                let inp = |j: usize| s.nrm + f(j.min(n_n - 1) * n);
                for (x, j, y) in [(&ml.wq, 0, s.q), (&ml.rel, 0, s.r), (&ml.sink, 0, s.sink), (&ml.gate, 0, s.og), (&ml.wk, 1, s.k), (&ml.wv, 2, s.v)] {
                    be.gemv(w(x), inp(j), y)?;
                }
                let mut taps = [0, 0];
                if let Some([kp, vp]) = &ml.conv_proj {
                    be.gemv(w(kp), inp(1), s.kp)?;
                    be.gemv(w(vp), inp(2), s.vp)?;
                    taps = [s.kp, s.vp];
                }
                be.ink_conv(s.k, l.convs[0].addr(), taps[0], hist(0), s.k, kw, nc, hd)?;
                be.ink_conv(s.v, l.convs[1].addr(), taps[1], hist(1), s.v, kw, nc, hd)?;
                // Log-scaled positions sharpen the full-attention layers in a long context.
                let tau = if ml.swa { 1.0 } else { 1.0 + c.log_alpha * ((pos + 1) as f32 / c.log_floor).max(1.0).ln() };
                let (rows, at) = if ml.swa { ((pos + 1).min(c.n_swa), pos % c.n_swa) } else { (pos + 1, pos) };
                let row = 2 * (at * kw) as u64;
                be.ink_clip(s.q, l.q_clip.addr(), hd, nh, c.qk_clip, 0.0, tau, 0)?;
                be.ink_clip(s.k, 0, hd, nkv, c.qk_clip, 0.0, 1.0, ds.k.addr() + row)?;
                be.ink_clip(s.v, 0, hd, nkv, 0.0, c.eps, 1.0, ds.v.addr() + row)?;
                be.ink_clip(s.r, l.rel_clip.addr(), nr, nh, c.rel_clip, 0.0, tau, 0)?;
                be.ink_attn(s.q, s.r, s.sink, s.og, ds.k.addr(), ds.v.addr(), l.rel_bias.addr(), self.sc.addr(), s.att, rows, at, nh, nkv, hd, nr, c.rel_extent)?;
                be.gemv(w(&ml.wo), s.att, s.proj)?;
                be.ink_conv(s.proj, l.convs[2].addr(), 0, hist(2), s.x, n, nc, hd)?;
                // The first layer of each block starts a new residual stream; the earlier one joins the mixed blocks.
                if ip % c.block_size == 0 {
                    be.dtod(&self.scr, (s.cur - o) as usize, &self.scr, (s.blocks - o) as usize + 4 * *nb * n, 4 * n)?;
                    *nb += 1;
                } else {
                    be.add(s.x, s.cur, n)?;
                }
                be.ink_mix(s.blocks, *nb, s.x, l.ffn_query.addr(), 1, s.h, n)?;
                let mut h = s.h;
                if msgs {
                    let at = |b: &Buf, cols: usize| b.addr() + 2 * (3 * rank * cols) as u64;
                    be.ink_ctrl(at(&lp.down, 2 * n), at(&lp.gain, n), at(&lp.forget, n), h, 1, s.msg + f(3 * n), s.gate, s.h2, n, rank, 1)?;
                    h = s.h2;
                }
                be.rms_norm_q(h, n, l.ffn_norm.addr(), s.hn, n, n, 1, c.eps, s.xh, 0)?;
                be.get(&self.scr, ((s.hn - o) / 4) as usize, n).map(|v| hn = v)
            })().expect("inkling2 device step");
        }
        m.lap(0, t);
        // The experts: those of the GPU read the device's copy of the input and leave their outputs there.
        let (on_gpu, on_cpu, adapt) = m.plan(ip, &hn);
        let gpu_side = || {
            let t = std::time::Instant::now();
            let list = Inkling2Model::resident(ml, c.n_expert, &on_gpu);
            let d = lock();
            let (p, mut owned) = (&d.parts[0], vec![]);
            (|| -> Option<()> {
                // One moe_gemv per matrix kind for all of them (8 small gemvs leave most of the card idle).
                let Some((gu, dn, ..)) = list.first() else { return Some(()) };
                let (k, gs) = (list.len(), list.iter().map(|e| e.3).collect::<Vec<f32>>());
                let mut ptrs = |down: bool| list.iter().map(|e| d.weight(p, if down { &e.1 } else { &e.0 }, &mut owned).map(|t| t.ptr)).collect::<Option<Vec<u64>>>();
                let (pg, pd) = (ptrs(false)?, ptrs(true)?);
                match (slab(&pg), slab(&pd)) {
                    (Some((bg, sg, ig)), Some((bd, sd, id))) => {
                        p.be.put(&[ig, id].concat(), &self.scr, ((s.ids - o) / 4) as usize).ok()?;
                        p.be.put(&gs, &self.scr, ((s.ids - o) / 4) as usize + 2 * k).ok()?;
                        // The inputs go in quantized (s.xh from the norm, s.xq here): the gemvs skip that step.
                        p.be.moe_gemv(gu.ty, gu.rows, gu.cols, bg, sg, s.ids, k, s.hn, n, k, s.y, None, s.xh).ok()?;
                        p.be.situ(s.y, nf, k, s.ids + f(2 * k), c.situ_beta, c.situ_linear_beta).ok()?;
                        p.be.quant_xq(s.y, 2 * nf, nf, k, s.xq).ok()?;
                        p.be.moe_gemv(dn.ty, dn.rows, dn.cols, bd, sd, s.ids + 4 * k as u64, k, s.y, 2 * nf, 1, s.d, None, s.xq).ok()?;
                    }
                    // ponytail: addresses too far apart for i32 ids; one gemv per expert, slower but never hit with one slab per layer.
                    _ => {
                        let t = |m: &Mat, at: u64| crate::backend::GpuTensor::new(m.ty, m.rows, m.cols, Buf::new(at, m.data.len(), ()));
                        for i in 0..k { p.be.gemv(&t(gu, pg[i]), s.hn, s.y + f(i * 2 * nf)).ok()?; }
                        p.be.put(&gs, &self.scr, ((s.ids - o) / 4) as usize + 2 * k).ok()?;
                        p.be.situ(s.y, nf, k, s.ids + f(2 * k), c.situ_beta, c.situ_linear_beta).ok()?;
                        for i in 0..k { p.be.gemv(&t(dn, pd[i]), s.y + f(i * 2 * nf), s.d + f(i * n)).ok()?; }
                    }
                }
                // Wait when an expert came from the file: its copy lives until the run ends, and the CPU share
                // follows the time. With every expert in VRAM the queue runs on.
                if adapt || !owned.is_empty() { p.be.sync().ok()?; }
                Some(())
            })().expect("inkling2 device step");
            t.elapsed().as_secs_f32()
        };
        let cpu_side = || {
            let t = std::time::Instant::now();
            (run_experts(&on_cpu, &hn, c), t.elapsed().as_secs_f32())
        };
        let (tg, (cpu, tc)) = rayon::join(gpu_side, cpu_side);
        m.balance(&m.cpu_share, adapt, tg, tc);
        let wc: Vec<f32> = (0..c.n_used).map(|i| on_gpu.get(i).map_or(0.0, |e| e.2)).chain(cpu).collect();
        let d = lock();
        let be = &*d.parts[0].be;
        (|| -> Result<()> {
            be.put(&wc, &self.scr, ((s.wc - o) / 4) as usize)?;
            be.ink_ffn_sum(s.acc, s.d, s.wc, on_gpu.len(), n, s.wc + f(c.n_used))?;
            be.ink_conv(s.acc, l.convs[3].addr(), 0, hist(3), s.acc, n, nc, hd)?;
            be.add(s.x, s.acc, n)
        })().expect("inkling2 device step");
        (self.s.cur, self.s.x) = (s.x, s.cur);
    }
}
