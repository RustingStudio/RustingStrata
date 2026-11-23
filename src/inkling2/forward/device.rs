//! One decode token on the device: the stream, the depth mix, the loop controllers, the attention rows and the
//! convolution state stay in VRAM. A layer run comes back to the host once, for the router's input; the routed
//! experts then run where `Inkling2Model::plan` puts them, as in `ffn`.
//! `Inkling2Model::run` is the reference and the path of a prompt chunk. The attention rows and the convolution
//! state move between the two when the other side runs next (`to_device`, `to_host`).
//! The state is made at the first decoded token, after the prompt filled the expert cache, so `room` frees the least
//! recently used experts when VRAM is short.
use super::*;
use crate::backend::{Backend, Buf, GpuTensor};
use crate::deepseek::mat::dev::{self, SlabBuf};
use crate::quant::GgmlType;
use half::slice::HalfBitsSliceExt;
use std::sync::Arc;

/// Contiguous all-expert weights + F32 router uploaded once for a layer, enabling zero-round-trip GPU MoE.
struct FullExps {
    /// The GPU of the slabs: 0, or 1 when GPU 0 had no room (the FFN of that layer runs there, see `Remote`).
    part: usize,
    _gi: Arc<SlabBuf>,
    gate_inp: GpuTensor,
    bias: Arc<SlabBuf>,
    gu: Arc<SlabBuf>,
    gu_ty: GgmlType,
    gu_rows: usize,
    gu_cols: usize,
    gu_stride: usize,
    dn: Arc<SlabBuf>,
    dn_ty: GgmlType,
    dn_rows: usize,
    dn_cols: usize,
    dn_stride: usize,
    gu_scale: Option<Arc<SlabBuf>>,
    dn_scale: Option<Arc<SlabBuf>>,
}

struct DLayer { res_query: Option<Buf>, norms: Buf, q_clip: Buf, rel_bias: Buf, rel_clip: Buf, convs: [Buf; 4], ffn_query: Buf, ffn_norm: Buf, full_exps: Option<FullExps> }
/// The controllers of a core layer, Q, K, V and FFN one after another, as f16.
struct DLoop { msg_query: Buf, down: Buf, gain: Buf, forget: Buf }
/// Attention rows as f16 `[row][key head][channel]` and the four convolution histories, one after another. A
/// sliding-window slot is a ring of `n_swa` rows (position p in row p % n_swa); the others grow.
struct DSlot { k: Buf, v: Buf, hist: Buf, cap: usize }

/// What GPU 1 needs to run the FFN of the layers whose experts live there: a decode scratch (`s`) and, from the first
/// prompt chunk on, a prefill scratch. The activations come over the host once per layer run (about 10 KB per token).
struct Remote { part: usize, scr: Buf, s: Scratch, pf: Option<PrefillScratch> }

/// Tokens of one prompt chunk on the device: `STRATA_DS_PF_TOKENS` (default 512, at most 2048).
pub(super) fn pf_tokens() -> usize {
    std::env::var("STRATA_DS_PF_TOKENS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(512).clamp(1, 2048)
}

/// Device addresses of the regions of the decode scratch buffer (`sc` lives in `Device::sc_buf` so `--ctx` scales dynamically).
#[derive(Clone, Copy)]
struct Scratch {
    cur: u64, x: u64, blocks: u64, bank: u64, outb: u64, prelude: u64, msg: u64, ins: u64, ins2: u64, gate: u64, nrm: u64, nrm_xq: u64,
    q: u64, r: u64, sink: u64, og: u64, k: u64, v: u64, kp: u64, vp: u64, att: u64, proj: u64, h: u64, h2: u64, hn: u64,
    y: u64, d: u64, ids: u64, xh: u64, xq: u64, wc: u64, rlogits: u64, amax: u64, pos: u64, acc: u64, logits: u64, end: u64,
}

impl Scratch {
    fn at(base: u64, c: &Inkling2Config, rank: usize, vocab: usize) -> Self {
        let (n, q, mut a) = (c.n_embd, c.n_head * c.head_dim, base);
        let xq_row_floats = (n + n.div_ceil(32) * 4).div_ceil(16) * 4;
        let mut f = |floats: usize| { a += 4 * floats as u64; a - 4 * floats as u64 };
        Scratch {
            cur: f(n), x: f(n), blocks: f(4 * n), bank: f(4 * n), outb: f(n), prelude: f(n), msg: f(4 * n), ins: f(3 * n), ins2: f(3 * n),
            gate: f(4 * rank), nrm: f(3 * n), nrm_xq: f(3 * xq_row_floats), q: f(q), r: f(c.n_head * c.n_rel), sink: f(c.n_head), og: f(q), k: f(c.n_head_kv * c.head_dim),
            v: f(c.n_head_kv * c.head_dim), kp: f(c.n_head_kv * c.n_conv), vp: f(c.n_head_kv * c.n_conv), att: f(q), proj: f(n), h: f(n),
            h2: f(n), hn: f(n), y: f(c.n_used * 2 * c.n_ff_exp), d: f(c.n_used * n), ids: f(3 * c.n_used), xh: f(n), xq: f(c.n_used * c.n_ff_exp), wc: f(c.n_used + n), rlogits: f(c.n_expert), amax: f(16), pos: f(1), acc: f(n),
            logits: f(vocab), end: a - base,
        }
    }
}

#[derive(Clone, Copy)]
struct PrefillLayout {
    cur: u64, x: u64, blocks: u64, bank: u64, outb: u64, prelude: u64, msg: u64,
    ins: u64, ins2: u64, gate: u64, nrm: u64, nrm_xq: u64, xh: u64,
    q: u64, r: u64, sink: u64, og: u64, k: u64, k2: u64, v: u64, v2: u64, kp: u64, vp: u64,
    att: u64, proj: u64, h: u64, h2: u64, hn: u64, acc2: u64,
    rlogits: u64, ids: u64, wts: u64, gsc: u64, y: u64, d: u64, xq: u64, end: u64,
}

impl PrefillLayout {
    fn at(base: u64, c: &Inkling2Config, rank: usize, cmax: usize) -> Self {
        let (n, nh, nkv, hd) = (c.n_embd, c.n_head, c.n_head_kv, c.head_dim);
        let (qd, kw, nr, nc, nf, k) = (nh * hd, nkv * hd, c.n_rel, c.n_conv, c.n_ff_exp, c.n_used);
        let xq_row = (n + n.div_ceil(32) * 4).div_ceil(16) * 4;
        let xq_nf_row = (nf + nf.div_ceil(32) * 4).div_ceil(16) * 4;
        let mut a = base;
        let mut f = |floats: usize| { a += 4 * floats as u64; a - 4 * floats as u64 };
        Self {
            cur: f(cmax * n), x: f(cmax * n), blocks: f(4 * cmax * n), bank: f(4 * cmax * n),
            outb: f(cmax * n), prelude: f(cmax * n), msg: f(4 * cmax * n),
            ins: f(3 * cmax * n), ins2: f(3 * cmax * n), gate: f(cmax * 4 * rank),
            nrm: f(3 * cmax * n), nrm_xq: f(3 * cmax * xq_row), xh: f(cmax * xq_row),
            q: f(cmax * qd), r: f(cmax * nh * nr), sink: f(cmax * nh), og: f(cmax * qd),
            k: f(cmax * kw), k2: f(cmax * kw), v: f(cmax * kw), v2: f(cmax * kw),
            kp: f(cmax * nkv * nc), vp: f(cmax * nkv * nc),
            att: f(cmax * qd), proj: f(cmax * n), h: f(cmax * n), h2: f(cmax * n),
            hn: f(cmax * n), acc2: f(cmax * n),
            rlogits: f(cmax * c.n_expert), ids: f(cmax * k), wts: f(cmax * k), gsc: f(cmax * k),
            y: f(cmax * k * 2 * nf), d: f(cmax * k * n),
            xq: f(cmax.min(16) * k * xq_nf_row),
            end: a - base,
        }
    }
}

pub(super) const VERIFY_MAX: usize = 16;
const DSPARK_VISITS: [usize; 3] = [5, 23, 41];

struct PrefillScratch {
    max_nt: usize,
    buf: Buf,
    acc_buf: Buf,
    kc_tmp: Buf,
    vc_tmp: Buf,
    meta: Buf,
    logits_buf: Buf,
    dspark_buf: Buf,
    hist_bak: Buf,
    conv_log: Buf,
    swa_kc_log: Buf,
    swa_vc_log: Buf,
    p: PrefillLayout,
}

impl PrefillScratch {
    /// Bytes `alloc` takes (plus a little for the allocator).
    fn bytes(be: &dyn Backend, c: &Inkling2Config, rank: usize, vocab_rows: usize, max_nt: usize, light: bool) -> usize {
        let (kw, n_slot) = (c.n_head_kv * c.head_dim, c.n_slot());
        let rb = 2 * be.kv_words(kw);
        let lay = PrefillLayout::at(0, c, rank, max_nt).end as usize;
        let swa = (n_slot * VERIFY_MAX * rb).max(1024);
        let extra = if light { 0 } else {
            2 * max_nt * rb + 4 * VERIFY_MAX * vocab_rows + 16 * max_nt * c.n_embd + 4 * n_slot * hists(c).1
                + 4 * n_slot * VERIFY_MAX * (2 * kw + 2 * c.n_embd) + 2 * swa
        };
        lay + 4 * max_nt * c.n_embd + (40 * max_nt * c.n_used).max(1024) + extra + (16 << 20)
    }

    /// `light`: only what the routed experts of a layer need (GPU 1 runs nothing else): the attention logs are tiny.
    fn alloc(be: &dyn Backend, c: &Inkling2Config, rank: usize, vocab_rows: usize, max_nt: usize, light: bool) -> Result<Self> {
        let lay0 = PrefillLayout::at(0, c, rank, max_nt);
        let buf = be.alloc(lay0.end as usize)?;
        let p = PrefillLayout::at(buf.addr(), c, rank, max_nt);
        let acc_buf = be.alloc(4 * max_nt * c.n_embd)?;
        let kw = c.n_head_kv * c.head_dim;
        let rb = 2 * be.kv_words(kw);
        let kv_bytes = if light { 256 } else { max_nt * rb };
        let (kc_tmp, vc_tmp) = (be.alloc(kv_bytes)?, be.alloc(kv_bytes)?);
        let max_np = max_nt * c.n_used;
        let meta = be.alloc((40 * max_np).max(1024))?;
        let lt = |b: usize| if light { 256 } else { b };
        let logits_buf = be.alloc(lt(4 * VERIFY_MAX * vocab_rows))?;
        let dspark_buf = be.alloc(lt(4 * 4 * max_nt * c.n_embd))?;
        let n_slot = c.n_slot();
        let hist_bak = be.alloc(lt(4 * n_slot * hists(c).1))?;
        let conv_log = be.alloc(lt(4 * n_slot * VERIFY_MAX * (2 * kw + 2 * c.n_embd)))?;
        let swa_log_bytes = lt((n_slot * VERIFY_MAX * rb).max(1024));
        let (swa_kc_log, swa_vc_log) = (be.alloc(swa_log_bytes)?, be.alloc(swa_log_bytes)?);
        Ok(Self {
            max_nt, buf, acc_buf, kc_tmp, vc_tmp, meta, logits_buf, dspark_buf,
            hist_bak, conv_log, swa_kc_log, swa_vc_log, p,
        })
    }
}

pub(super) struct Device {
    layers: Vec<DLayer>,
    loops: Vec<DLoop>,
    slots: Vec<DSlot>,
    out_query: Buf,
    out_norm: Buf,
    scr: Buf,
    sc_buf: Buf,
    pf: Option<PrefillScratch>,
    s: Scratch,
    rank: usize,
    all_full: bool,
    /// GPU 1's share of the FFN layers (None when every layer's experts are on GPU 0).
    rem: Option<Remote>,
    /// With `rem`: the loop layers are all on GPU 0, so the loop runs as one graph, and the layers around it launch one by one.
    seg_graph: bool,
    /// Blocks in the depth mix after the loop (what `layer_attn` counted when the loop graph was recorded).
    loop_nb: usize,
    /// Key of the recorded `step_fast` graph, by `argmax` (0: none).
    graph_keys: [u64; 2],
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
        let (kw, nr, ext) = (c.n_head_kv * c.head_dim, c.n_rel, c.rel_extent);
        let full = (2 * (pos + 1)).clamp(4096.min(m.max_ctx), m.max_ctx);
        {
            let mut d = dev::get()?.lock().unwrap();
            // Local patch: several GPUs are fine as long as the plain weights are on GPU 0 (see STRATA_DS_PLAIN_FIRST); the
            // experts that live on another GPU run as a third share in `layer`.
            let dense = |x: &Mat| !x.host_only() && d.on_first(x);
            if !dense(&m.output) || !m.layers.iter().all(|l| {
                [l.wq, l.wk, l.wv, l.wo, l.gate, l.rel, l.sink].iter().chain(l.conv_proj.iter().flatten()).all(dense)
            }) { return None; }
            let rows: usize = (0..c.n_slot()).map(|il| if c.swa[layer_of(c, il)] { c.n_swa } else { full }).sum();
            let row_words = d.parts[0].be.kv_words(kw);
            room(&mut d, 4 * rows * row_words + sc_bytes(c, full) + Scratch::at(0, c, rank, m.output.rows).end as usize + (512 << 20));
        }
        let made = (|| -> Result<Self> {
            let (mut layers, loops, slots, out_query, out_norm, scr, sc_buf) = {
                let d = dev::get().unwrap().lock().unwrap();
                let be = &*d.parts[0].be;
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
                    full_exps: None,
                })).collect::<Result<Vec<_>>>()?;
                let half = |w: &[Vec<f16>]| up(be, w.concat().reinterpret_cast());
                let loops = m.loops.iter().map(|l| {
                    Ok(DLoop { msg_query: up(be, &l.msg_query)?, down: half(&l.down)?, gain: half(&l.gain)?, forget: half(&l.forget)? })
                }).collect::<Result<Vec<_>>>()?;
                let row_words = be.kv_words(kw);
                let slots = (0..c.n_slot()).map(|il| {
                    let rows = if c.swa[layer_of(c, il)] { c.n_swa } else { full };
                    Ok(DSlot { k: be.alloc(2 * rows * row_words)?, v: be.alloc(2 * rows * row_words)?, hist: be.zeros::<f32>(hists(c).1)?, cap: rows })
                }).collect::<Result<Vec<_>>>()?;
                let sc_buf = be.alloc(sc_bytes(c, (pos + 1).max(64)))?;
                let (out_query, out_norm, scr) = (up(be, &m.out_query)?, up(be, &m.out_norm)?, be.alloc(Scratch::at(0, c, rank, m.output.rows).end as usize)?);
                (layers, loops, slots, out_query, out_norm, scr, sc_buf)
            };
            let s = Scratch::at(scr.addr(), c, rank, m.output.rows);

            // Preload full-layer expert slabs so selected layers skip CPU routing, uploads, and host syncs.
            // Core loop layers run 3x per token (75% of FFN compute), so they go first, and to GPU 0, where the attention is.
            // Local patch: with two GPUs the layers that run once (outside the loop) prefer GPU 1, and a layer that does not
            // fit on its GPU goes to the other one: every layer's experts end up in VRAM, and the FFN of a layer on GPU 1
            // runs there (`Remote`). GPU 0 keeps free what the state still needs: prompt chunk scratch and KV growth.
            let cpu_share = std::env::var("STRATA_DS_CPU_SHARE").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.0);
            let (mut n_full, mut slab_bytes, mut per_part) = (0usize, 0usize, [0usize; 2]);
            let mut rem: Option<Remote> = None;
            if cpu_share <= 0.0 && c.n_expert <= 512 && c.n_used <= 32 {
                let (b_res, f_res) = (64 << 20, 384 << 20);
                let (n_parts, pf_cap, vocab) = (dev::get().unwrap().lock().unwrap().parts.len().min(2), pf_tokens(), m.output.rows);
                let (res0, res1) = {
                    let d = dev::get().unwrap().lock().unwrap();
                    let be0 = &*d.parts[0].be;
                    // The KV rows GPU 0 must still be able to take: all of `--ctx` unless STRATA_DS_KV_TOKENS says less (a
                    // lower value frees room for more expert slabs on GPU 0, at the price of a host fallback past it).
                    let kv_rows = std::env::var("STRATA_DS_KV_TOKENS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(m.max_ctx)
                        .min(m.max_ctx).max(full);
                    let n_att = (0..c.n_slot()).filter(|&il| !c.swa[layer_of(c, il)]).count();
                    let kv = 4 * n_att * kv_rows * be0.kv_words(kw) + sc_bytes(c, kv_rows);
                    let pf0 = PrefillScratch::bytes(be0, c, rank, vocab, pf_cap, false);
                    let pf1 = if n_parts > 1 { PrefillScratch::bytes(&*d.parts[1].be, c, rank, vocab, pf_cap, true) } else { 0 };
                    (f_res + pf0 + kv, (192 << 20) + pf1 + Scratch::at(0, c, rank, 0).end as usize)
                };
                let key = |s: &[u8]| Some((s.as_ptr() as usize, s.len()));
                let on_loop = |ip: usize| (c.loop_start..c.loop_end).contains(&ip);
                let order = (c.loop_start..c.loop_end).chain(0..c.loop_start).chain(c.loop_end..c.n_layer);
                for ip in order {
                    let ml = &m.layers[ip];
                    let gi_bytes: Vec<u8> = ml.gate_inp.iter().flat_map(|x| x.to_le_bytes()).collect();
                    let bias_bytes: Vec<u8> = ml.bias.iter().flat_map(|x| x.to_le_bytes()).collect();
                    let gs_bytes: Vec<u8> = ml.gu_scale.iter().flat_map(|x| x.to_le_bytes()).collect();
                    let ds_bytes: Vec<u8> = ml.down_scale.iter().flat_map(|x| x.to_le_bytes()).collect();
                    let place = |part: usize| -> Option<(usize, Arc<SlabBuf>, Arc<SlabBuf>, Arc<SlabBuf>, Arc<SlabBuf>, Option<Arc<SlabBuf>>, Option<Arc<SlabBuf>>)> {
                        let mut d = dev::get().unwrap().lock().unwrap();
                        let fr = if part == 0 { res0 } else { res1 };
                        let gu = d.alloc_slab_on(part, ml.gate_up.data, key(ml.gate_up.data), b_res, fr)?;
                        let dn = d.alloc_slab_on(part, ml.down.data, key(ml.down.data), b_res, fr)?;
                        let gi = d.alloc_slab_on(part, &gi_bytes, None, b_res, fr)?;
                        let bi = d.alloc_slab_on(part, &bias_bytes, None, b_res, fr)?;
                        let gs = if gs_bytes.is_empty() { None } else { Some(d.alloc_slab_on(part, &gs_bytes, None, b_res, fr)?) };
                        let ds = if ds_bytes.is_empty() { None } else { Some(d.alloc_slab_on(part, &ds_bytes, None, b_res, fr)?) };
                        Some((part, gu, dn, gi, bi, gs, ds))
                    };
                    let prefs: &[usize] = match (n_parts > 1, on_loop(ip)) { (false, _) => &[0], (true, true) => &[0, 1], (true, false) => &[1, 0] };
                    let Some((part, gu, dn, gi_slab, bias, gu_scale, dn_scale)) = prefs.iter().find_map(|&p| place(p)) else {
                        break;
                    };
                    let bytes = gu.len() + dn.len() + gi_slab.len() + bias.len()
                        + gu_scale.as_ref().map_or(0, |b| b.len())
                        + dn_scale.as_ref().map_or(0, |b| b.len());
                    slab_bytes += bytes;
                    per_part[part] += bytes;
                    n_full += 1;
                    let (gu_rows, gu_cols) = (ml.gate_up.rows / c.n_expert, ml.gate_up.cols);
                    let (dn_rows, dn_cols) = (ml.down.rows / c.n_expert, ml.down.cols);
                    layers[ip].full_exps = Some(FullExps {
                        part,
                        gate_inp: GpuTensor::new(GgmlType::F32, c.n_expert, c.n_embd, Buf::new(gi_slab.addr(), gi_slab.len(), ())),
                        _gi: gi_slab,
                        bias,
                        gu, gu_ty: ml.gate_up.ty, gu_rows, gu_cols,
                        gu_stride: gu_rows * ml.gate_up.ty.row_bytes(gu_cols as u64),
                        dn, dn_ty: ml.down.ty, dn_rows, dn_cols,
                        dn_stride: dn_rows * ml.down.ty.row_bytes(dn_cols as u64),
                        gu_scale,
                        dn_scale,
                    });
                }
                if layers.iter().any(|l| l.full_exps.as_ref().is_some_and(|fe| fe.part == 1)) {
                    let d = dev::get().unwrap().lock().unwrap();
                    let be1 = &*d.parts[1].be;
                    let r_scr = be1.alloc(Scratch::at(0, c, rank, 0).end as usize)?;
                    let r_s = Scratch::at(r_scr.addr(), c, rank, 0);
                    let pf = PrefillScratch::alloc(be1, c, rank, vocab, pf_cap, true).ok();
                    rem = Some(Remote { part: 1, scr: r_scr, s: r_s, pf });
                }
            }
            if std::env::var_os("STRATA_DS_PROF").is_some() {
                eprintln!("inkling2: preloaded {n_full}/{} expert layers ({:.2} GB: {:.2} on GPU 0, {:.2} on GPU 1) as contiguous slabs",
                    c.n_layer, slab_bytes as f64 / 1e9, per_part[0] as f64 / 1e9, per_part[1] as f64 / 1e9);
            }
            let all_full = rem.is_none() && layers.iter().all(|l| l.full_exps.is_some());
            let core = c.loop_end - c.loop_start;
            let seg_graph = rem.is_some() && std::env::var_os("STRATA_DS_NO_SEG_GRAPH").is_none()
                && layers.iter().all(|l| l.full_exps.is_some())
                && (c.loop_start..c.loop_end).all(|ip| layers[ip].full_exps.as_ref().is_some_and(|fe| fe.part == 0))
                && (c.loop_count * core) % 2 == 0;
            Ok(Device { layers, loops, slots, out_query, out_norm, scr, sc_buf, pf: None, s, rank, all_full, rem, seg_graph, loop_nb: 0, graph_keys: [0; 2], synced: 0, ahead: false })
        })();
        made.inspect_err(|e| eprintln!("inkling2: no room in VRAM for the decode state ({e:#}); the tokens run as prompt chunks do")).ok()
    }

    /// The model starts over: nothing is shared with the host's rows.
    pub(super) fn forget(&mut self) {
        (self.synced, self.ahead) = (0, false);
    }

    /// Save conv histories and SWA ring buffers for fast prefix rollback (full-attention KV rows `< pos` are untouched by later tokens).
    pub(super) fn save_checkpoint(&self, c: &Inkling2Config) -> (Vec<Vec<f32>>, Vec<Option<(Vec<u16>, Vec<u16>)>>, usize, bool) {
        if !self.ahead { return (vec![], vec![], self.synced, false); }
        let d = dev::get().unwrap().lock().unwrap();
        let be = &*d.parts[0].be;
        let len = hists(c).1;
        let kw = c.n_head_kv * c.head_dim;
        let swa_words = c.n_swa * be.kv_words(kw);
        let mut dev_hist = Vec::with_capacity(self.slots.len());
        let mut dev_swa = Vec::with_capacity(self.slots.len());
        for (il, ds) in self.slots.iter().enumerate() {
            dev_hist.push(be.get(&ds.hist, 0, len).expect("inkling2 save_checkpoint hist"));
            if c.swa[layer_of(c, il)] {
                let k: Vec<u16> = be.get(&ds.k, 0, swa_words).expect("inkling2 save_checkpoint swa k");
                let v: Vec<u16> = be.get(&ds.v, 0, swa_words).expect("inkling2 save_checkpoint swa v");
                dev_swa.push(Some((k, v)));
            } else {
                dev_swa.push(None);
            }
        }
        (dev_hist, dev_swa, self.synced, true)
    }

    /// Restore conv histories and SWA ring buffers from a `Checkpoint`.
    pub(super) fn restore_checkpoint(
        &mut self,
        hist: &[Vec<f32>],
        swa: &[Option<(Vec<u16>, Vec<u16>)>],
        synced: usize,
        ahead: bool,
    ) {
        (self.synced, self.ahead) = (synced, ahead);
        if !ahead || hist.is_empty() { return; }
        let d = dev::get().unwrap().lock().unwrap();
        let be = &*d.parts[0].be;
        for ((ds, h), kv) in self.slots.iter().zip(hist).zip(swa) {
            be.put(h, &ds.hist, 0).expect("inkling2 restore_checkpoint hist");
            if let Some((k, v)) = kv {
                be.put(k, &ds.k, 0).expect("inkling2 restore_checkpoint swa k");
                be.put(v, &ds.v, 0).expect("inkling2 restore_checkpoint swa v");
            }
        }
    }

    /// Room for the rows of position `need_pos` in the slots that grow and `sc_buf`.
    pub(super) fn prepare(&mut self, m: &Inkling2Model, need_pos: usize) -> Result<()> {
        let (c, kw) = (&m.cfg, m.cfg.n_head_kv * m.cfg.head_dim);
        let need = need_pos + 1;
        ensure!(need <= m.max_ctx, "context is limited to {} tokens", m.max_ctx);
        let mut d = dev::get().unwrap().lock().unwrap();
        let row_words = d.parts[0].be.kv_words(kw);
        let full_cap = self.slots.iter().enumerate().find(|&(il, _)| !c.swa[layer_of(c, il)]).map(|(_, s)| s.cap);
        if let Some(cap) = full_cap && need > cap {
            let mut new_cap = cap;
            while new_cap < need { new_cap = (new_cap * 2).min(m.max_ctx); }
            let n = (0..c.n_slot()).filter(|&il| !c.swa[layer_of(c, il)]).count();
            room(&mut d, 4 * n * new_cap * row_words + sc_bytes(c, new_cap) + (128 << 20));
        }
        let be = &*d.parts[0].be;
        let need_sc = sc_bytes(c, need);
        if self.sc_buf.len() < need_sc {
            be.sync()?;
            self.sc_buf = be.alloc(need_sc)?;
        }
        for (il, s) in self.slots.iter_mut().enumerate() {
            if c.swa[layer_of(c, il)] || need <= s.cap { continue; }
            let mut new_cap = s.cap;
            while new_cap < need { new_cap = (new_cap * 2).min(m.max_ctx); }
            for b in [&mut s.k, &mut s.v] {
                let new = be.alloc(2 * new_cap * row_words)?;
                be.dtod(b, 0, &new, 0, b.len())?;
                be.sync()?;  // the copy reads the buffer that `*b = new` frees
                *b = new;
            }
            s.cap = new_cap;
        }
        Ok(())
    }

    /// Copy what the host has run since the last exchange: the rows of the positions `synced..pos` that a token
    /// still sees, and the convolution histories.
    fn to_device(&mut self, m: &Inkling2Model, be: &dyn Backend, pos: usize) -> Result<()> {
        let (c, hd) = (&m.cfg, m.cfg.head_dim);
        let (kw, (at, len)) = (c.n_head_kv * hd, hists(c));
        ensure!(self.synced == pos || crate::cuda::kv_mode() == 0,
            "quantized KV cache (--kv) cannot resume mid-context from host slots");
        for (il, (ds, hs)) in self.slots.iter().zip(&m.slots).enumerate() {
            let mut h = vec![0f32; len];
            for (j, x) in hs.hist.iter().enumerate() { h[at[j]..at[j] + x.len()].copy_from_slice(x); }
            be.put(&h, &ds.hist, 0)?;
            let ring = if c.swa[layer_of(c, il)] { c.n_swa } else { m.max_ctx };
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
    pub(super) fn to_host(&mut self, c: &Inkling2Config, layers: &[Layer], slots: &mut [Slot], pos: usize, max_ctx: usize) {
        if !self.ahead { return; }
        assert!(crate::cuda::kv_mode() == 0 || self.synced == pos,
            "quantized KV cache (--kv q8/q4/q2) is only supported on the GPU path");
        let d = dev::get().unwrap().lock().unwrap();
        let be = &*d.parts[0].be;
        let (hd, (at, len)) = (c.head_dim, hists(c));
        let kw = c.n_head_kv * hd;
        for (il, (ds, hs)) in self.slots.iter().zip(slots).enumerate() {
            let h: Vec<f32> = be.get(&ds.hist, 0, len).expect("inkling2 device state");
            for j in 0..4 { hs.hist[j] = h[at[j]..if j < 3 { at[j + 1] } else { len }].to_vec(); }
            let ring = if layers[layer_of(c, il)].swa { c.n_swa } else { max_ctx };
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

    fn step_fast(&mut self, m: &Inkling2Model, tok: u32, argmax: bool) -> (Vec<f32>, u32) {
        let (c, pos) = (&m.cfg, m.toks.len());
        let (n, core, o) = (c.n_embd, c.loop_end - c.loop_start, self.scr.addr());
        let (mut emb, mut cur) = (vec![0f32; n], vec![0f32; n]);
        m.tok_embd.row_into(tok as usize, &mut emb);
        rms_norm_w(&emb, &m.tok_norm, c.eps, &mut cur);
        let d = dev::get().unwrap().lock().unwrap();
        let be = &*d.parts[0].be;
        (|| -> Result<(Vec<f32>, u32)> {
            if !self.ahead { self.to_device(m, be, pos)?; }
            be.put(&cur, &self.scr, ((self.s.cur - o) / 4) as usize)?;
            be.put(&[pos as i32], &self.scr, ((self.s.pos - o) / 4) as usize)?;
            let key = {
                use std::hash::{Hash, Hasher};
                let mut h = std::hash::DefaultHasher::new();
                (argmax, pos / 64, self.sc_buf.addr(), self.s.cur).hash(&mut h);
                for sl in &self.slots { sl.k.addr().hash(&mut h); }
                0x494E_4B32_0000_0000u64 | (h.finish() & 0x0000_FFFF_FFFF_FFFF)
            };
            // The key changes every 64 positions: one graph per kind of step stays (on CUDA a graph holds about
            // 17 MiB of device memory, and the backend stops recording at 32 of them).
            let last = std::mem::replace(&mut self.graph_keys[argmax as usize], key);
            if last != 0 && last != key { be.graph_drop(last)?; }
            if !self.seg_graph {
                be.graph(key, &mut || {
                    let mut nb = 0usize;
                    for il in 0..c.loop_start {
                        self.layer_attn(be, &d, m, il, il, false, pos, true, &mut nb)?;
                        self.layer_ffn_gpu(be, m, il, il)?;
                    }
                    be.dtod(&self.scr, (self.s.cur - o) as usize, &self.scr, (self.s.prelude - o) as usize, 4 * n)?;
                    let n_pre = nb;
                    for pass in 0..c.loop_count {
                        let n_bank = nb.saturating_sub(n_pre + 1);
                        if pass > 0 {
                            let s = self.s;
                            if n_bank > 0 {
                                be.dtod(&self.scr, (s.blocks - o) as usize + 4 * (n_pre + 1) * n, &self.scr, (s.bank - o) as usize, 4 * n_bank * n)?;
                            }
                            be.dtod(&self.scr, (s.cur - o) as usize, &self.scr, (s.outb - o) as usize, 4 * n)?;
                            be.dtod(&self.scr, (s.prelude - o) as usize, &self.scr, (s.cur - o) as usize, 4 * n)?;
                            nb = n_pre;
                        }
                        for ip in c.loop_start..c.loop_end {
                            let il = if pass == 0 { ip } else { c.n_layer + (pass - 1) * core + ip - c.loop_start };
                            if pass > 0 {
                                let s = self.s;
                                be.ink_mix(s.bank, n_bank, s.outb, self.loops[ip - c.loop_start].msg_query.addr(), 4, s.msg, n)?;
                            }
                            self.layer_attn(be, &d, m, il, ip, pass > 0, pos, true, &mut nb)?;
                            self.layer_ffn_gpu(be, m, il, ip)?;
                        }
                    }
                    for il in c.loop_end..c.n_layer {
                        self.layer_attn(be, &d, m, il, il, false, pos, true, &mut nb)?;
                        self.layer_ffn_gpu(be, m, il, il)?;
                    }
                    self.finish_logits(be, &d, m, nb)?;
                    if argmax {
                        be.argmax_rows(self.s.logits, c.n_vocab, 1, self.s.amax)?;
                    }
                    Ok(())
                })?;
            } else {
                // Local patch: layers 0..loop_start and loop_end.. launch one by one (their FFN may be on GPU 1, which needs
                // the host between the attention and the experts); the loop between them is one graph.
                let prof = std::env::var_os("STRATA_DS_PROF").is_some();
                let mut tt = std::time::Instant::now();
                let mut nb = 0usize;
                for il in 0..c.loop_start {
                    self.layer_attn(be, &d, m, il, il, false, pos, true, &mut nb)?;
                    if !self.layer_ffn_remote(&d, m, il, il)? { self.layer_ffn_gpu(be, m, il, il)?; }
                }
                if prof { be.sync()?; m.lap(1, tt); tt = std::time::Instant::now(); }
                let n_pre = nb;
                be.graph(key, &mut || {
                    let mut nb = n_pre;
                    be.dtod(&self.scr, (self.s.cur - o) as usize, &self.scr, (self.s.prelude - o) as usize, 4 * n)?;
                    for pass in 0..c.loop_count {
                        let n_bank = nb.saturating_sub(n_pre + 1);
                        if pass > 0 {
                            let s = self.s;
                            if n_bank > 0 {
                                be.dtod(&self.scr, (s.blocks - o) as usize + 4 * (n_pre + 1) * n, &self.scr, (s.bank - o) as usize, 4 * n_bank * n)?;
                            }
                            be.dtod(&self.scr, (s.cur - o) as usize, &self.scr, (s.outb - o) as usize, 4 * n)?;
                            be.dtod(&self.scr, (s.prelude - o) as usize, &self.scr, (s.cur - o) as usize, 4 * n)?;
                            nb = n_pre;
                        }
                        for ip in c.loop_start..c.loop_end {
                            let il = if pass == 0 { ip } else { c.n_layer + (pass - 1) * core + ip - c.loop_start };
                            if pass > 0 {
                                let s = self.s;
                                be.ink_mix(s.bank, n_bank, s.outb, self.loops[ip - c.loop_start].msg_query.addr(), 4, s.msg, n)?;
                            }
                            self.layer_attn(be, &d, m, il, ip, pass > 0, pos, true, &mut nb)?;
                            self.layer_ffn_gpu(be, m, il, ip)?;
                        }
                    }
                    self.loop_nb = nb;
                    Ok(())
                })?;
                if prof { be.sync()?; m.lap(2, tt); tt = std::time::Instant::now(); }
                let mut nb = self.loop_nb;
                for il in c.loop_end..c.n_layer {
                    self.layer_attn(be, &d, m, il, il, false, pos, true, &mut nb)?;
                    if !self.layer_ffn_remote(&d, m, il, il)? { self.layer_ffn_gpu(be, m, il, il)?; }
                }
                if prof { be.sync()?; m.lap(3, tt); }
                self.finish_logits(be, &d, m, nb)?;
                if argmax {
                    be.argmax_rows(self.s.logits, c.n_vocab, 1, self.s.amax)?;
                }
            }
            if argmax {
                let id = be.get::<i32>(&self.scr, ((self.s.amax - o) / 4) as usize, 1)?[0] as u32;
                Ok((Vec::new(), id))
            } else {
                Ok((be.get(&self.scr, ((self.s.logits - o) / 4) as usize, c.n_vocab)?, 0))
            }
        })().expect("inkling2 device fast step")
    }

    fn step_layers(&mut self, m: &Inkling2Model, tok: u32) -> usize {
        let (c, pos) = (&m.cfg, m.toks.len());
        let (n, core, o) = (c.n_embd, c.loop_end - c.loop_start, self.scr.addr());
        let (mut emb, mut cur) = (vec![0f32; n], vec![0f32; n]);
        m.tok_embd.row_into(tok as usize, &mut emb);
        rms_norm_w(&emb, &m.tok_norm, c.eps, &mut cur);
        let lock = || dev::get().unwrap().lock().unwrap();
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
                if n_bank > 0 {
                    be.dtod(&self.scr, (s.blocks - o) as usize + 4 * (n_pre + 1) * n, &self.scr, (s.bank - o) as usize, 4 * n_bank * n).unwrap();
                }
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
        nb
    }

    fn finish_logits(&self, be: &dyn Backend, d: &dev::Dev, m: &Inkling2Model, nb: usize) -> Result<()> {
        let (c, s) = (&m.cfg, self.s);
        let n = c.n_embd;
        be.ink_mix(s.blocks, nb, s.cur, self.out_query.addr(), 1, s.h, n)?;
        be.rms_norm_q(s.h, n, self.out_norm.addr(), s.hn, n, n, 1, c.eps, s.nrm_xq, 0)?;
        be.gemvq(d.tensor(m.output.gpu.unwrap()), s.hn, n, s.nrm_xq, s.logits, c.n_vocab, 1)?;
        if c.softcap != 0.0 { be.softcap(s.logits, c.n_vocab, c.softcap)?; }
        Ok(())
    }

    /// Run one token at the end of `m.toks` (the caller appends it). Returns the logits of the next one.
    pub(super) fn step(&mut self, m: &Inkling2Model, tok: u32) -> Vec<f32> {
        let pos = m.toks.len();
        assert!(pos < m.max_ctx, "context is limited to {} tokens", m.max_ctx);
        assert!((tok as usize) < m.tok_embd.rows, "token {tok} is outside the vocabulary");
        if self.all_full || self.seg_graph {
            let t = std::time::Instant::now();
            let (logits, _) = self.step_fast(m, tok, false);
            m.lap(7, t);
            return logits;
        }
        let nb = self.step_layers(m, tok);
        let (c, s, o) = (&m.cfg, self.s, self.scr.addr());
        let t = std::time::Instant::now();
        let d = dev::get().unwrap().lock().unwrap();
        let be = &*d.parts[0].be;
        let logits = (|| -> Result<Vec<f32>> {
            self.finish_logits(be, &d, m, nb)?;
            be.get(&self.scr, ((s.logits - o) / 4) as usize, c.n_vocab)
        })().expect("inkling2 device step");
        m.lap(7, t);
        logits
    }

    /// Run one token at the end of `m.toks` and return the greedy argmax token ID (4-byte readback).
    pub(super) fn step_argmax(&mut self, m: &Inkling2Model, tok: u32) -> u32 {
        let pos = m.toks.len();
        assert!(pos < m.max_ctx, "context is limited to {} tokens", m.max_ctx);
        assert!((tok as usize) < m.tok_embd.rows, "token {tok} is outside the vocabulary");
        if self.all_full || self.seg_graph {
            let t = std::time::Instant::now();
            let (_, id) = self.step_fast(m, tok, true);
            m.lap(7, t);
            return id;
        }
        let nb = self.step_layers(m, tok);
        let (c, s, o) = (&m.cfg, self.s, self.scr.addr());
        let t = std::time::Instant::now();
        let d = dev::get().unwrap().lock().unwrap();
        let be = &*d.parts[0].be;
        let next = (|| -> Result<u32> {
            self.finish_logits(be, &d, m, nb)?;
            be.argmax_rows(s.logits, c.n_vocab, 1, s.amax)?;
            Ok(be.get::<i32>(&self.scr, ((s.amax - o) / 4) as usize, 1)?[0] as u32)
        })().expect("inkling2 device step_argmax");
        m.lap(7, t);
        next
    }

    fn ensure_pf(&mut self, be: &dyn Backend, c: &Inkling2Config, vocab_rows: usize, need_nt: usize) -> Result<()> {
        if self.pf.as_ref().is_some_and(|pf| pf.max_nt >= need_nt) { return Ok(()); }
        if self.pf.is_some() {
            be.sync()?;
            self.pf = None;
        }
        let target = need_nt.max(pf_tokens());
        let mut tries = vec![target];
        if need_nt < target { tries.push(need_nt.max(128)); tries.push(need_nt.max(64)); tries.push(need_nt); }
        let mut last_err = None;
        for cap in tries {
            match PrefillScratch::alloc(be, c, self.rank, vocab_rows, cap, false) {
                Ok(pf) => { self.pf = Some(pf); return Ok(()); }
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err.unwrap())
    }

    /// GPU 1's prefill scratch for `need_nt` rows, when some layer's experts live there.
    fn ensure_pf_remote(&mut self, d: &dev::Dev, c: &Inkling2Config, vocab_rows: usize, need_nt: usize) -> Result<()> {
        let rank = self.rank;
        let Some(r) = &mut self.rem else { return Ok(()) };
        if r.pf.as_ref().is_some_and(|pf| pf.max_nt >= need_nt) { return Ok(()); }
        let be1 = &*d.parts[r.part].be;
        if r.pf.is_some() { be1.sync()?; r.pf = None; }
        r.pf = Some(PrefillScratch::alloc(be1, c, rank, vocab_rows, need_nt.max(pf_tokens()), true)?);
        Ok(())
    }

    /// Project a single corrected hidden vector `h_corr` (`[n_embd]`) through the target's LM head on GPU
    /// and return the argmax token ID (`0..c.n_vocab`). No softcap: `softcap * tanh(x / softcap)` keeps the
    /// order of the logits (two that differ by a few ulp may become equal), and the token is a draft that
    /// the target checks.
    pub(super) fn lm_head_argmax(&self, m: &Inkling2Model, h_corr: &[f32]) -> Result<u32> {
        let c = &m.cfg;
        let (s, o) = (self.s, self.scr.addr());
        let d = dev::get().unwrap().lock().unwrap();
        let be = &*d.parts[0].be;
        be.put(h_corr, &self.scr, ((s.hn - o) / 4) as usize)?;
        be.gemvq(d.tensor(m.output.gpu.unwrap()), s.hn, c.n_embd, 0, s.logits, c.n_vocab, 1)?;
        be.argmax_rows(s.logits, c.n_vocab, 1, s.amax)?;
        let id = be.get::<i32>(&self.scr, ((s.amax - o) / 4) as usize, 1)?[0] as u32;
        Ok(id)
    }

    fn record_dspark_visit(&self, visit: usize, cur_a: u64, nt: usize, n: usize) -> Result<()> {
        let Some(k) = DSPARK_VISITS.iter().position(|&v| v == visit) else { return Ok(()) };
        let d = dev::get().unwrap().lock().unwrap();
        let be = &*d.parts[0].be;
        let pf = self.pf.as_ref().unwrap();
        let o = pf.buf.addr();
        be.dtod(&pf.buf, (cur_a - o) as usize, &pf.dspark_buf, k * 4 * nt * n, 4 * nt * n)
    }

    /// Run a chunk of token IDs or raw vision embeddings (`Some(emb)` skips `tok_norm`, matching `inkling2.cpp::build_inputs`).
    /// When `capture_dspark` is true, also returns the 4 target layer hidden states `[5, 23, 41, 48]` (`[4][nt][n_embd]`).
    pub(super) fn prefill_chunk_seq(
        &mut self,
        m: &Inkling2Model,
        seq: &[(u32, Option<&[f32]>)],
        want: bool,
        capture_dspark: bool,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
        let (c, pos0, nt) = (&m.cfg, m.toks.len(), seq.len());
        ensure!(nt > 0 && pos0 + nt <= m.max_ctx, "prefill_chunk: invalid range {pos0} + {nt}");
        let (n, core) = (c.n_embd, c.loop_end - c.loop_start);
        let t0 = std::time::Instant::now();
        let mut cur_host = vec![0f32; nt * n];
        let mut emb = vec![0f32; n];
        for (t, &(tok, raw_emb)) in seq.iter().enumerate() {
            if let Some(e) = raw_emb {
                ensure!(e.len() == n, "prefill_chunk_seq: embedding length {} != {n}", e.len());
                cur_host[t * n..(t + 1) * n].copy_from_slice(e);
            } else {
                ensure!((tok as usize) < m.tok_embd.rows, "token {tok} is outside the vocabulary");
                m.tok_embd.row_into(tok as usize, &mut emb);
                rms_norm_w(&emb, &m.tok_norm, c.eps, &mut cur_host[t * n..(t + 1) * n]);
            }
        }
        let (mut cur_a, mut x_a) = {
            let d = dev::get().unwrap().lock().unwrap();
            let be = &*d.parts[0].be;
            self.ensure_pf(be, c, m.output.rows, nt)?;
            self.ensure_pf_remote(&d, c, m.output.rows, nt)?;
            if !self.ahead { self.to_device(m, be, pos0)?; }
            let pf = self.pf.as_ref().unwrap();
            be.put(&cur_host, &pf.buf, 0)?;
            (pf.p.cur, pf.p.x)
        };
        let mut nb = 0usize;
        let mut visit = 0usize;
        for il in 0..c.loop_start {
            if capture_dspark { self.record_dspark_visit(visit, cur_a, nt, n)?; }
            visit += 1;
            self.prefill_layer(m, il, il, false, pos0, nt, false, &mut nb, &mut cur_a, &mut x_a)?;
        }
        {
            let d = dev::get().unwrap().lock().unwrap();
            let be = &*d.parts[0].be;
            let pf = self.pf.as_ref().unwrap();
            let o = pf.buf.addr();
            be.dtod(&pf.buf, (cur_a - o) as usize, &pf.buf, (pf.p.prelude - o) as usize, 4 * nt * n)?;
        }
        let n_pre = nb;
        for pass in 0..c.loop_count {
            let n_bank = nb.saturating_sub(n_pre + 1);
            if pass > 0 {
                let d = dev::get().unwrap().lock().unwrap();
                let be = &*d.parts[0].be;
                let pf = self.pf.as_ref().unwrap();
                let o = pf.buf.addr();
                if n_bank > 0 {
                    be.dtod(&pf.buf, (pf.p.blocks - o) as usize + 4 * (n_pre + 1) * nt * n,
                            &pf.buf, (pf.p.bank - o) as usize, 4 * n_bank * nt * n)?;
                }
                be.dtod(&pf.buf, (cur_a - o) as usize, &pf.buf, (pf.p.outb - o) as usize, 4 * nt * n)?;
                be.dtod(&pf.buf, (pf.p.prelude - o) as usize, &pf.buf, (cur_a - o) as usize, 4 * nt * n)?;
                nb = n_pre;
            }
            for ip in c.loop_start..c.loop_end {
                let il = if pass == 0 { ip } else { c.n_layer + (pass - 1) * core + ip - c.loop_start };
                if pass > 0 {
                    let d = dev::get().unwrap().lock().unwrap();
                    let be = &*d.parts[0].be;
                    let pf = self.pf.as_ref().unwrap();
                    be.ink_mix_n(pf.p.bank, n_bank, pf.p.outb, self.loops[ip - c.loop_start].msg_query.addr(), 4, pf.p.msg, n, nt)?;
                }
                if capture_dspark { self.record_dspark_visit(visit, cur_a, nt, n)?; }
                visit += 1;
                self.prefill_layer(m, il, ip, pass > 0, pos0, nt, false, &mut nb, &mut cur_a, &mut x_a)?;
            }
        }
        for il in c.loop_end..c.n_layer {
            if capture_dspark { self.record_dspark_visit(visit, cur_a, nt, n)?; }
            visit += 1;
            self.prefill_layer(m, il, il, false, pos0, nt, false, &mut nb, &mut cur_a, &mut x_a)?;
        }
        let dspark_feats = if capture_dspark {
            let d = dev::get().unwrap().lock().unwrap();
            let be = &*d.parts[0].be;
            let pf = self.pf.as_ref().unwrap();
            let o = pf.buf.addr();
            be.ink_mix_n(pf.p.blocks, nb, cur_a, self.out_query.addr(), 1, pf.p.h, n, nt)?;
            be.rms_norm(pf.p.h, n, self.out_norm.addr(), pf.p.hn, n, n, nt, c.eps)?;
            be.dtod(&pf.buf, (pf.p.hn - o) as usize, &pf.dspark_buf, 3 * 4 * nt * n, 4 * nt * n)?;
            Some(be.get(&pf.dspark_buf, 0, 4 * nt * n)?)
        } else {
            None
        };
        let logits = if want {
            let t = std::time::Instant::now();
            let d = dev::get().unwrap().lock().unwrap();
            let be = &*d.parts[0].be;
            let pf = self.pf.as_ref().unwrap();
            let (s, o) = (self.s, self.scr.addr());
            if !capture_dspark {
                be.ink_mix_n(pf.p.blocks, nb, cur_a, self.out_query.addr(), 1, pf.p.h, n, nt)?;
            }
            let last_h = pf.p.h + f((nt - 1) * n);
            be.rms_norm_q(last_h, n, self.out_norm.addr(), s.hn, n, n, 1, c.eps, s.nrm_xq, 0)?;
            be.gemvq(d.tensor(m.output.gpu.unwrap()), s.hn, n, s.nrm_xq, s.logits, c.n_vocab, 1)?;
            if c.softcap != 0.0 { be.softcap(s.logits, c.n_vocab, c.softcap)?; }
            let out = be.get(&self.scr, ((s.logits - o) / 4) as usize, c.n_vocab)?;
            m.lap(7, t);
            out
        } else {
            vec![]
        };
        m.report(pos0, nt, t0);
        Ok((logits, dspark_feats))
    }

    /// Run `1 <= nt <= VERIFY_MAX` candidate tokens `[anchor, drafts..]` on the GPU in a single batched pass,
    /// returning all `nt` logit vectors (`[nt][c.n_vocab]`) and optional DSpark target features (`[4][nt][n_embd]`).
    /// Call `commit_verify(m, pos0, keep, nt)` afterwards to commit the `keep` (`1..=nt`) accepted tokens.
    pub(super) fn verify_forward(
        &mut self,
        m: &Inkling2Model,
        toks: &[u32],
        capture_dspark: bool,
    ) -> Result<(Vec<Vec<f32>>, Option<Vec<f32>>)> {
        let (c, pos0, nt) = (&m.cfg, m.toks.len(), toks.len());
        ensure!(nt > 0 && nt <= VERIFY_MAX && pos0 + nt <= m.max_ctx, "verify_forward: invalid range {pos0} + {nt}");
        let (n, core) = (c.n_embd, c.loop_end - c.loop_start);
        let h_total = hists(c).1;
        let t0 = std::time::Instant::now();
        let mut cur_host = vec![0f32; nt * n];
        let mut emb = vec![0f32; n];
        for (t, &tok) in toks.iter().enumerate() {
            ensure!((tok as usize) < m.tok_embd.rows, "token {tok} is outside the vocabulary");
            m.tok_embd.row_into(tok as usize, &mut emb);
            rms_norm_w(&emb, &m.tok_norm, c.eps, &mut cur_host[t * n..(t + 1) * n]);
        }
        let (mut cur_a, mut x_a) = {
            let d = dev::get().unwrap().lock().unwrap();
            let be = &*d.parts[0].be;
            self.ensure_pf(be, c, m.output.rows, nt)?;
            self.ensure_pf_remote(&d, c, m.output.rows, nt)?;
            if !self.ahead { self.to_device(m, be, pos0)?; }
            let pf = self.pf.as_ref().unwrap();
            for (il, ds) in self.slots.iter().enumerate() {
                be.dtod(&ds.hist, 0, &pf.hist_bak, il * 4 * h_total, 4 * h_total)?;
            }
            be.put(&cur_host, &pf.buf, 0)?;
            (pf.p.cur, pf.p.x)
        };
        let mut nb = 0usize;
        let mut visit = 0usize;
        for il in 0..c.loop_start {
            if capture_dspark { self.record_dspark_visit(visit, cur_a, nt, n)?; }
            visit += 1;
            self.prefill_layer(m, il, il, false, pos0, nt, true, &mut nb, &mut cur_a, &mut x_a)?;
        }
        {
            let d = dev::get().unwrap().lock().unwrap();
            let be = &*d.parts[0].be;
            let pf = self.pf.as_ref().unwrap();
            let o = pf.buf.addr();
            be.dtod(&pf.buf, (cur_a - o) as usize, &pf.buf, (pf.p.prelude - o) as usize, 4 * nt * n)?;
        }
        let n_pre = nb;
        for pass in 0..c.loop_count {
            let n_bank = nb.saturating_sub(n_pre + 1);
            if pass > 0 {
                let d = dev::get().unwrap().lock().unwrap();
                let be = &*d.parts[0].be;
                let pf = self.pf.as_ref().unwrap();
                let o = pf.buf.addr();
                if n_bank > 0 {
                    be.dtod(&pf.buf, (pf.p.blocks - o) as usize + 4 * (n_pre + 1) * nt * n,
                            &pf.buf, (pf.p.bank - o) as usize, 4 * n_bank * nt * n)?;
                }
                be.dtod(&pf.buf, (cur_a - o) as usize, &pf.buf, (pf.p.outb - o) as usize, 4 * nt * n)?;
                be.dtod(&pf.buf, (pf.p.prelude - o) as usize, &pf.buf, (cur_a - o) as usize, 4 * nt * n)?;
                nb = n_pre;
            }
            for ip in c.loop_start..c.loop_end {
                let il = if pass == 0 { ip } else { c.n_layer + (pass - 1) * core + ip - c.loop_start };
                if pass > 0 {
                    let d = dev::get().unwrap().lock().unwrap();
                    let be = &*d.parts[0].be;
                    let pf = self.pf.as_ref().unwrap();
                    be.ink_mix_n(pf.p.bank, n_bank, pf.p.outb, self.loops[ip - c.loop_start].msg_query.addr(), 4, pf.p.msg, n, nt)?;
                }
                if capture_dspark { self.record_dspark_visit(visit, cur_a, nt, n)?; }
                visit += 1;
                self.prefill_layer(m, il, ip, pass > 0, pos0, nt, true, &mut nb, &mut cur_a, &mut x_a)?;
            }
        }
        for il in c.loop_end..c.n_layer {
            if capture_dspark { self.record_dspark_visit(visit, cur_a, nt, n)?; }
            visit += 1;
            self.prefill_layer(m, il, il, false, pos0, nt, true, &mut nb, &mut cur_a, &mut x_a)?;
        }
        let (logits_all, dspark_feats) = {
            let t = std::time::Instant::now();
            let d = dev::get().unwrap().lock().unwrap();
            let be = &*d.parts[0].be;
            let pf = self.pf.as_ref().unwrap();
            let o = pf.buf.addr();
            let out_w = d.tensor(m.output.gpu.unwrap());
            let vrows = m.output.rows;
            be.ink_mix_n(pf.p.blocks, nb, cur_a, self.out_query.addr(), 1, pf.p.h, n, nt)?;
            if Self::small(be, nt, true) {
                be.rms_norm_q(pf.p.h, n, self.out_norm.addr(), pf.p.hn, n, n, nt, c.eps, pf.p.nrm_xq, 0)?;
                be.gemvq(out_w, pf.p.hn, n, pf.p.nrm_xq, pf.logits_buf.addr(), vrows, nt)?;
            } else {
                be.rms_norm(pf.p.h, n, self.out_norm.addr(), pf.p.hn, n, n, nt, c.eps)?;
                be.gemm_dense(out_w, nt, pf.p.hn, n, pf.logits_buf.addr(), vrows, 0)?;
            }
            if c.softcap != 0.0 {
                be.softcap(pf.logits_buf.addr(), nt * vrows, c.softcap)?;
            }
            let dspark = if capture_dspark {
                be.dtod(&pf.buf, (pf.p.hn - o) as usize, &pf.dspark_buf, 3 * 4 * nt * n, 4 * nt * n)?;
                Some(be.get(&pf.dspark_buf, 0, 4 * nt * n)?)
            } else {
                None
            };
            let raw: Vec<f32> = be.get(&pf.logits_buf, 0, nt * vrows)?;
            let logits: Vec<Vec<f32>> = (0..nt).map(|i| raw[i * vrows..i * vrows + c.n_vocab].to_vec()).collect();
            m.lap(7, t);
            (logits, dspark)
        };
        m.report(pos0, nt, t0);
        Ok((logits_all, dspark_feats))
    }

    /// Commit the first `keep` (`1 <= keep <= nt`) tokens of a `verify_forward` pass on the GPU in place.
    pub(super) fn commit_verify(&self, m: &Inkling2Model, pos0: usize, keep: usize, nt: usize) -> Result<()> {
        if keep >= nt { return Ok(()); }
        let c = &m.cfg;
        let (n, kw, hm) = (c.n_embd, c.n_head_kv * c.head_dim, c.n_conv - 1);
        let (h_offs, h_total) = hists(c);
        let conv_widths = [kw, kw, n, n];
        let log_slot_bytes = 4 * VERIFY_MAX * (2 * kw + 2 * n);
        let log_offs = [0, 4 * VERIFY_MAX * kw, 4 * VERIFY_MAX * 2 * kw, 4 * VERIFY_MAX * (2 * kw + n)];
        let d = dev::get().unwrap().lock().unwrap();
        let be = &*d.parts[0].be;
        let pf = self.pf.as_ref().unwrap();
        let rb = 2 * be.kv_words(kw);
        for (il, ds) in self.slots.iter().enumerate() {
            let ip = layer_of(c, il);
            if m.layers[ip].swa {
                let swa_base = il * VERIFY_MAX * rb;
                for t in (keep..nt).rev() {
                    let at = (pos0 + t) % c.n_swa;
                    be.dtod(&pf.swa_kc_log, swa_base + t * rb, &ds.k, at * rb, rb)?;
                    be.dtod(&pf.swa_vc_log, swa_base + t * rb, &ds.v, at * rb, rb)?;
                }
            }
            let bak_base = il * 4 * h_total;
            let log_base = il * log_slot_bytes;
            for j in 0..4 {
                let w = conv_widths[j];
                let dst_off = 4 * h_offs[j];
                if keep >= hm {
                    let src_off = log_base + log_offs[j] + 4 * (keep - hm) * w;
                    be.dtod(&pf.conv_log, src_off, &ds.hist, dst_off, 4 * hm * w)?;
                } else {
                    let old_rows = hm - keep;
                    let bak_off = bak_base + dst_off + 4 * keep * w;
                    be.dtod(&pf.hist_bak, bak_off, &ds.hist, dst_off, 4 * old_rows * w)?;
                    let log_off = log_base + log_offs[j];
                    be.dtod(&pf.conv_log, log_off, &ds.hist, dst_off + 4 * old_rows * w, 4 * keep * w)?;
                }
            }
        }
        Ok(())
    }

    /// Whether a draft check of `nt` rows runs the matvec kernels (`gemvq`, `moe_gemv`) and not the GEMM. Measured on
    /// an RTX 3060: on CUDA a check of 7 rows or more is faster through the GEMM (16 rows take 184 ms against 254 ms),
    /// on Vulkan the matvec kernels are faster up to `VERIFY_MAX` rows. A prompt chunk keeps the old limit, so the
    /// logits of a plain run are unchanged.
    fn small(be: &dyn Backend, nt: usize, verify: bool) -> bool {
        nt <= if verify && be.name() == "cuda" { 6 } else { VERIFY_MAX }
    }

    fn proj_n(be: &dyn Backend, w: &GpuTensor, small: bool, nt: usize, x: u64, xq: u64, y: u64, act: i32) -> Result<()> {
        if small {
            be.gemvq(w, x, w.cols, xq, y, w.rows, nt)
        } else {
            be.gemm_dense(w, nt, x, w.cols, y, w.rows, act)
        }
    }

    fn prefill_layer(
        &self, m: &Inkling2Model, il: usize, ip: usize, msgs: bool,
        pos0: usize, nt: usize, verify_mode: bool, nb: &mut usize, cur_a: &mut u64, x_a: &mut u64,
    ) -> Result<()> {
        let (c, ml, l, ds, rank) = (&m.cfg, &m.layers[ip], &self.layers[ip], &self.slots[il], self.rank);
        let (n, hd, nh, nkv, nr, nc, nf, k) = (c.n_embd, c.head_dim, c.n_head, c.n_head_kv, c.n_rel, c.n_conv, c.n_ff_exp, c.n_used);
        let kw = nkv * hd;
        let lp = &self.loops[ip.clamp(c.loop_start, c.loop_end - 1) - c.loop_start];
        let hist = |j: usize| ds.hist.addr() + f(hists(c).0[j]);
        let pf = self.pf.as_ref().unwrap();
        let (p, buf_o) = (pf.p, pf.buf.addr());
        let log_base = il * 4 * VERIFY_MAX * (2 * kw + 2 * n);

        let has_full_exps = {
            let t_prep = std::time::Instant::now();
            let d = dev::get().unwrap().lock().unwrap();
            let be = &*d.parts[0].be;
            let w = |x: &Mat| d.tensor(x.gpu.unwrap());
            let small = Self::small(be, nt, verify_mode);
            let (mut ins, mut n_in) = (*cur_a, 1);
            if let Some(q) = &l.res_query {
                (ins, n_in) = (p.ins, q.len() / (4 * n));
                be.ink_mix_n(p.blocks, *nb, *cur_a, q.addr(), n_in, ins, n, nt)?;
            }
            if msgs {
                be.ink_ctrl_n(lp.down.addr(), lp.gain.addr(), lp.forget.addr(), ins, n_in, p.msg, p.gate, p.ins2, n, rank, 3, nt)?;
                (ins, n_in) = (p.ins2, 3);
            }
            let n_w = l.norms.len() / (4 * n);
            let n_n = n_in.max(n_w);
            let xq_stride = ((n + n.div_ceil(32) * 4).div_ceil(16) * 4) * nt;
            let norm_xq = |j: usize| p.nrm_xq + f(j.min(n_n - 1) * xq_stride);
            for j in 0..n_n {
                let xj = ins + f(j.min(n_in - 1) * nt * n);
                let wj = l.norms.addr() + f(j.min(n_w - 1) * n);
                let oj = p.nrm + f(j * nt * n);
                if small {
                    be.rms_norm_q(xj, n, wj, oj, n, n, nt, c.eps, norm_xq(j), 0)?;
                } else {
                    be.rms_norm(xj, n, wj, oj, n, n, nt, c.eps)?;
                }
            }
            let inp = |j: usize| p.nrm + f(j.min(n_n - 1) * nt * n);
            let sx = crate::backend::SAME_X;
            Self::proj_n(be, w(&ml.wq), small, nt, inp(0), norm_xq(0), p.q, 0)?;
            Self::proj_n(be, w(&ml.gate), small, nt, inp(0), norm_xq(0), p.og, sx)?;
            Self::proj_n(be, w(&ml.rel), small, nt, inp(0), norm_xq(0), p.r, sx)?;
            Self::proj_n(be, w(&ml.sink), small, nt, inp(0), norm_xq(0), p.sink, sx)?;
            let k_sx = if n_n == 1 { sx } else { 0 };
            Self::proj_n(be, w(&ml.wk), small, nt, inp(1), norm_xq(1), p.k, k_sx)?;
            if let Some([kp, _]) = &ml.conv_proj {
                Self::proj_n(be, w(kp), small, nt, inp(1), norm_xq(1), p.kp, sx)?;
            }
            Self::proj_n(be, w(&ml.wv), small, nt, inp(2), norm_xq(2), p.v, k_sx)?;
            if let Some([_, vp]) = &ml.conv_proj {
                Self::proj_n(be, w(vp), small, nt, inp(2), norm_xq(2), p.vp, sx)?;
            }
            if verify_mode {
                be.dtod(&pf.buf, (p.k - buf_o) as usize, &pf.conv_log, log_base, 4 * nt * kw)?;
                be.dtod(&pf.buf, (p.v - buf_o) as usize, &pf.conv_log, log_base + 4 * VERIFY_MAX * kw, 4 * nt * kw)?;
            }
            let taps = if ml.conv_proj.is_some() { [p.kp, p.vp] } else { [0, 0] };
            be.ink_conv_batch(p.k, l.convs[0].addr(), taps[0], hist(0), p.k2, kw, nc, hd, nt)?;
            be.ink_conv_batch(p.v, l.convs[1].addr(), taps[1], hist(1), p.v2, kw, nc, hd, nt)?;
            let log_alpha = if ml.swa { 0.0 } else { c.log_alpha };
            let rb = 2 * be.kv_words(kw);
            be.ink_clip_n(p.q, l.q_clip.addr(), hd, nh, nt, c.qk_clip, 0.0, 1.0, 0, 0, 1, pos0, log_alpha, c.log_floor)?;
            be.ink_clip_n(p.r, l.rel_clip.addr(), nr, nh, nt, c.rel_clip, 0.0, 1.0, 0, 0, 1, pos0, log_alpha, c.log_floor)?;
            if verify_mode {
                let swa_base = il * VERIFY_MAX * rb;
                if !ml.swa {
                    be.ink_clip_n(p.k2, 0, hd, nkv, nt, c.qk_clip, 0.0, 1.0, ds.k.addr(), pos0, ds.cap, 0, 0.0, 1.0)?;
                    be.ink_clip_n(p.v2, 0, hd, nkv, nt, 0.0, c.eps, 1.0, ds.v.addr(), pos0, ds.cap, 0, 0.0, 1.0)?;
                }
                for t in 0..nt {
                    let pos = pos0 + t;
                    let (rows, at) = if ml.swa {
                        let at = pos % c.n_swa;
                        be.dtod(&ds.k, at * rb, &pf.swa_kc_log, swa_base + t * rb, rb)?;
                        be.dtod(&ds.v, at * rb, &pf.swa_vc_log, swa_base + t * rb, rb)?;
                        be.ink_clip_n(p.k2 + f(t * kw), 0, hd, nkv, 1, c.qk_clip, 0.0, 1.0, ds.k.addr(), at, c.n_swa, 0, 0.0, 1.0)?;
                        be.ink_clip_n(p.v2 + f(t * kw), 0, hd, nkv, 1, 0.0, c.eps, 1.0, ds.v.addr(), at, c.n_swa, 0, 0.0, 1.0)?;
                        ((pos + 1).min(c.n_swa), at)
                    } else {
                        (pos + 1, pos)
                    };
                    be.ink_attn(
                        p.q + f(t * nh * hd), p.r + f(t * nh * nr), p.sink + f(t * nh), p.og + f(t * nh * hd),
                        ds.k.addr(), ds.v.addr(), l.rel_bias.addr(), self.sc_buf.addr(), p.att + f(t * nh * hd),
                        rows, at, nh, nkv, hd, nr, c.rel_extent,
                    )?;
                }
            } else {
                let (kd, vd, at0, kvr) = if ml.swa {
                    (pf.kc_tmp.addr(), pf.vc_tmp.addr(), 0, nt)
                } else {
                    (ds.k.addr(), ds.v.addr(), pos0, ds.cap)
                };
                be.ink_clip_n(p.k2, 0, hd, nkv, nt, c.qk_clip, 0.0, 1.0, kd, at0, kvr, 0, 0.0, 1.0)?;
                be.ink_clip_n(p.v2, 0, hd, nkv, nt, 0.0, c.eps, 1.0, vd, at0, kvr, 0, 0.0, 1.0)?;
                let (kc_new, vc_new, ring, win) = if ml.swa {
                    (kd, vd, c.n_swa, c.n_swa)
                } else {
                    (0, 0, ds.cap, 0)
                };
                be.ink_attn_prefill(p.q, p.r, p.sink, p.og, ds.k.addr(), ds.v.addr(), kc_new, vc_new, l.rel_bias.addr(), p.att,
                    pos0, nt, nh, nkv, hd, nr, c.rel_extent, ring, win)?;
                if ml.swa {
                    let keep = nt.min(c.n_swa);
                    let src0 = nt - keep;
                    let dst0 = (pos0 + src0) % c.n_swa;
                    let first = keep.min(c.n_swa - dst0);
                    be.dtod(&pf.kc_tmp, src0 * rb, &ds.k, dst0 * rb, first * rb)?;
                    be.dtod(&pf.vc_tmp, src0 * rb, &ds.v, dst0 * rb, first * rb)?;
                    if first < keep {
                        be.dtod(&pf.kc_tmp, (src0 + first) * rb, &ds.k, 0, (keep - first) * rb)?;
                        be.dtod(&pf.vc_tmp, (src0 + first) * rb, &ds.v, 0, (keep - first) * rb)?;
                    }
                }
            }
            Self::proj_n(be, w(&ml.wo), small, nt, p.att, 0, p.proj, 0)?;
            if verify_mode {
                be.dtod(&pf.buf, (p.proj - buf_o) as usize, &pf.conv_log, log_base + 4 * VERIFY_MAX * 2 * kw, 4 * nt * n)?;
            }
            be.ink_conv_batch(p.proj, l.convs[2].addr(), 0, hist(2), *x_a, n, nc, hd, nt)?;
            if ip % c.block_size == 0 {
                be.dtod(&pf.buf, (*cur_a - buf_o) as usize, &pf.buf, (p.blocks - buf_o) as usize + 4 * (*nb) * nt * n, 4 * nt * n)?;
                *nb += 1;
            } else {
                be.add(*x_a, *cur_a, nt * n)?;
            }
            be.ink_mix_n(p.blocks, *nb, *x_a, l.ffn_query.addr(), 1, p.h, n, nt)?;
            let mut h = p.h;
            if msgs {
                let at = |b: &Buf, cols: usize| b.addr() + 2 * (3 * rank * cols) as u64;
                be.ink_ctrl_n(at(&lp.down, 2 * n), at(&lp.gain, n), at(&lp.forget, n), h, 1, p.msg + f(3 * nt * n), p.gate, p.h2, n, rank, 1, nt)?;
                h = p.h2;
            }
            if let Some(fe) = &l.full_exps {
                let use_gemm = (!small || fe.part != 0) && fe.gu_cols % 64 == 0 && fe.dn_cols % 64 == 0;
                if use_gemm || fe.gu_ty.float_x() {
                    be.rms_norm(h, n, l.ffn_norm.addr(), p.hn, n, n, nt, c.eps)?;
                } else {
                    be.rms_norm_q(h, n, l.ffn_norm.addr(), p.hn, n, n, nt, c.eps, p.xh, 0)?;
                }
                m.lap(0, t_prep);
                let t_gpu = std::time::Instant::now();
                if fe.part == 0 {
                    Self::pf_ffn_core(be, m, ip, fe, pf, nt, small, use_gemm)?;
                } else {
                    // Experts on GPU 1: the normed rows cross over, the experts run there, the sum comes back.
                    let r = self.rem.as_ref().ok_or_else(|| anyhow::anyhow!("experts on GPU 1 without a scratch there"))?;
                    let pf1 = r.pf.as_ref().ok_or_else(|| anyhow::anyhow!("no prefill scratch on GPU 1"))?;
                    let be1 = &*d.parts[r.part].be;
                    let (o0, o1) = (buf_o, pf1.buf.addr());
                    let hn: Vec<f32> = be.get(&pf.buf, ((p.hn - o0) / 4) as usize, nt * n)?;
                    be1.put(&hn, &pf1.buf, ((pf1.p.hn - o1) / 4) as usize)?;
                    if !use_gemm && !fe.gu_ty.float_x() {
                        let xw = (n + n.div_ceil(32) * 4).div_ceil(16) * 4;
                        let xh: Vec<f32> = be.get(&pf.buf, ((p.xh - o0) / 4) as usize, nt * xw)?;
                        be1.put(&xh, &pf1.buf, ((pf1.p.xh - o1) / 4) as usize)?;
                    }
                    Self::pf_ffn_core(be1, m, ip, fe, pf1, nt, small, use_gemm)?;
                    let acc: Vec<f32> = be1.get(&pf1.acc_buf, 0, nt * n)?;
                    be.put(&acc, &pf.acc_buf, 0)?;
                }
                if verify_mode {
                    be.dtod(&pf.acc_buf, 0, &pf.conv_log, log_base + 4 * VERIFY_MAX * (2 * kw + n), 4 * nt * n)?;
                }
                be.ink_conv_batch(pf.acc_buf.addr(), l.convs[3].addr(), 0, hist(3), p.acc2, n, nc, hd, nt)?;
                be.add(*x_a, p.acc2, nt * n)?;
                m.lap(5, t_gpu);
                true
            } else {
                be.rms_norm(h, n, l.ffn_norm.addr(), p.hn, n, n, nt, c.eps)?;
                m.lap(0, t_prep);
                false
            }
        };
        if !has_full_exps {
            let hn_flat: Vec<f32> = {
                let d = dev::get().unwrap().lock().unwrap();
                d.parts[0].be.get(&pf.buf, ((p.hn - buf_o) / 4) as usize, nt * n)?
            };
            let hn_rows: Vec<Vec<f32>> = hn_flat.chunks_exact(n).map(|r| r.to_vec()).collect();
            let ys = m.ffn_batch(ip, &hn_rows);
            let y_flat: Vec<f32> = ys.into_iter().flatten().collect();
            let d = dev::get().unwrap().lock().unwrap();
            let be = &*d.parts[0].be;
            be.put(&y_flat, &pf.acc_buf, 0)?;
            if verify_mode {
                be.dtod(&pf.acc_buf, 0, &pf.conv_log, log_base + 4 * VERIFY_MAX * (2 * kw + n), 4 * nt * n)?;
            }
            be.ink_conv_batch(pf.acc_buf.addr(), l.convs[3].addr(), 0, hist(3), p.acc2, n, nc, hd, nt)?;
            be.add(*x_a, p.acc2, nt * n)?;
        }
        std::mem::swap(cur_a, x_a);
        Ok(())
    }

    /// The routed experts of a prompt chunk on the GPU `be` runs on: from the normed rows in `pf.p.hn` (and, below the
    /// GEMM size, their quantized copy in `pf.p.xh`) to the weighted sum in `pf.acc_buf`.
    #[allow(clippy::too_many_arguments)]
    fn pf_ffn_core(be: &dyn Backend, m: &Inkling2Model, ip: usize, fe: &FullExps, pf: &PrefillScratch, nt: usize, small: bool, use_gemm: bool) -> Result<()> {
        let (c, ml) = (&m.cfg, &m.layers[ip]);
        let (n, nf, k) = (c.n_embd, c.n_ff_exp, c.n_used);
        let (p, buf_o) = (pf.p, pf.buf.addr());
        Self::proj_n(be, &fe.gate_inp, small, nt, p.hn, 0, p.rlogits, 0)?;
        let gu_sc = fe.gu_scale.as_ref().map_or(0, |b| b.addr());
        let dn_sc = fe.dn_scale.as_ref().map_or(0, |b| b.addr());
        let gsc_out = if gu_sc != 0 { p.gsc } else { 0 };
        be.ink_route(p.rlogits, fe.bias.addr(), p.ids, p.wts, gu_sc, dn_sc, gsc_out, c.n_expert, k, c.exp_scale, nt)?;
        if !use_gemm {
            // Process up to 16 tokens per moe_gemv batch so zero host sync is needed (exact decode speed for nt <= 16).
            for tok0 in (0..nt).step_by(16) {
                let bt = (nt - tok0).min(16);
                let np = bt * k;
                let xh_tok_floats = (n + n.div_ceil(32) * 4).div_ceil(16) * 4;
                let ids_a = p.ids + (4 * tok0 * k) as u64;
                let wts_a = p.wts + f(tok0 * k);
                let gsc_a = if gu_sc != 0 { p.gsc + f(tok0 * k) } else { 0 };
                let hn_a = p.hn + f(tok0 * n);
                let xh_a = p.xh + f(tok0 * xh_tok_floats);
                let y_a = p.y;
                let d_a = p.d;
                let acc_a = pf.acc_buf.addr() + f(tok0 * n);
                be.moe_gemv(fe.gu_ty, fe.gu_rows, fe.gu_cols, fe.gu.addr(), fe.gu_stride, ids_a, np, hn_a, n, k, y_a, None, xh_a)?;
                if fe.dn_ty.float_x() || nf % 32 != 0 {
                    be.situ(y_a, nf, np, gsc_a, c.situ_beta, c.situ_linear_beta)?;
                    be.moe_gemv(fe.dn_ty, fe.dn_rows, fe.dn_cols, fe.dn.addr(), fe.dn_stride, ids_a, np, y_a, 2 * nf, 1, d_a, None, 0)?;
                } else {
                    be.situ_q(y_a, nf, np, gsc_a, c.situ_beta, c.situ_linear_beta, p.xq)?;
                    be.moe_gemv(fe.dn_ty, fe.dn_rows, fe.dn_cols, fe.dn.addr(), fe.dn_stride, ids_a, np, y_a, 2 * nf, 1, d_a, None, p.xq)?;
                }
                be.ink_ffn_sum_n(acc_a, d_a, wts_a, k, n, bt, 0)?;
            }
        } else {
            use crate::backend::GEMM_BM;
            let ids: Vec<i32> = be.get(&pf.buf, ((p.ids - buf_o) / 4) as usize, nt * k)?;
            let mut by_e: Vec<Vec<(usize, usize)>> = vec![vec![]; c.n_expert];
            for (t, row) in ids.chunks_exact(k).enumerate() {
                for (j, &e) in row.iter().enumerate() { by_e[e as usize].push((t, j)); }
            }
            let np = nt * k;
            let mut tiles_gu: Vec<u64> = Vec::with_capacity(2 * (c.n_expert + np / GEMM_BM));
            let mut tiles_dn: Vec<u64> = Vec::with_capacity(2 * (c.n_expert + np / GEMM_BM));
            let mut xrow: Vec<i32> = Vec::with_capacity(np);
            let mut inv: Vec<i32> = vec![0i32; nt * k];
            let mut gsc_rows: Vec<f32> = if ml.gu_scale.is_empty() { Vec::new() } else { Vec::with_capacity(np) };
            let mut m0 = 0usize;
            for e in 0..c.n_expert {
                if by_e[e].is_empty() { continue; }
                let wg = fe.gu.addr() + (e * fe.gu_stride) as u64;
                let wd = fe.dn.addr() + (e * fe.dn_stride) as u64;
                for at in (0..by_e[e].len()).step_by(GEMM_BM) {
                    let cnt = GEMM_BM.min(by_e[e].len() - at) as u64;
                    tiles_gu.extend([wg, (m0 + at) as u64 | (cnt << 32)]);
                    tiles_dn.extend([wd, (m0 + at) as u64 | (cnt << 32)]);
                }
                for &(t, j) in &by_e[e] {
                    inv[t * k + j] = xrow.len() as i32;
                    xrow.push(t as i32);
                    if !ml.gu_scale.is_empty() { gsc_rows.push(ml.gu_scale[e]); }
                }
                m0 += by_e[e].len();
            }
            let n_tiles = tiles_gu.len() / 2;
            let dn_off = 8 * tiles_gu.len();
            let xr_off = dn_off + 8 * tiles_dn.len();
            let inv_off = xr_off + 4 * np;
            ensure!(inv_off + 4 * nt * k <= pf.meta.len(), "prefill meta overflow");
            be.put(&tiles_gu, &pf.meta, 0)?;
            be.put(&tiles_dn, &pf.meta, dn_off / 8)?;
            be.put(&xrow, &pf.meta, xr_off / 4)?;
            be.put(&inv, &pf.meta, inv_off / 4)?;
            let gsc_a = if !gsc_rows.is_empty() {
                be.put(&gsc_rows, &pf.buf, ((p.gsc - buf_o) / 4) as usize)?;
                p.gsc
            } else {
                0
            };
            let ma = pf.meta.addr();
            be.gemm(fe.gu_ty, fe.gu_rows, fe.gu_cols, ma, n_tiles, p.hn, n, nt, ma + xr_off as u64, p.y, 2 * nf, 0)?;
            be.situ(p.y, nf, np, gsc_a, c.situ_beta, c.situ_linear_beta)?;
            be.gemm(fe.dn_ty, fe.dn_rows, fe.dn_cols, ma + dn_off as u64, n_tiles, p.y, 2 * nf, np, 0, p.d, n, 0)?;
            be.memset(&pf.acc_buf)?;
            be.moe_combine_seq(pf.acc_buf.addr(), p.d, p.wts, ma + inv_off as u64, k, 0, 0, n, nt)?;
        }
        Ok(())
    }

    fn layer_attn(&self, be: &dyn Backend, d: &dev::Dev, m: &Inkling2Model, il: usize, ip: usize, msgs: bool, pos: usize, graph: bool, nb: &mut usize) -> Result<()> {
        let (c, ml, l, ds, s, rank) = (&m.cfg, &m.layers[ip], &self.layers[ip], &self.slots[il], self.s, self.rank);
        let (n, hd, nh, nkv, nr, nc) = (c.n_embd, c.head_dim, c.n_head, c.n_head_kv, c.n_rel, c.n_conv);
        let (kw, o) = (nkv * hd, self.scr.addr());
        let lp = &self.loops[ip.clamp(c.loop_start, c.loop_end - 1) - c.loop_start];
        let hist = |j: usize| ds.hist.addr() + f(hists(c).0[j]);
        let w = |x: &Mat| d.tensor(x.gpu.unwrap());
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
        let xq_step = ((n + n.div_ceil(32) * 4).div_ceil(16) * 16) as u64;
        let norm_xq = |j: usize| s.nrm_xq + j.min(n_n - 1) as u64 * xq_step;
        if n_n == 1 {
            be.rms_norm_q(ins, n, l.norms.addr(), s.nrm, n, n, 1, c.eps, norm_xq(0), 0)?;
        } else {
            be.batch(&mut || {
                for j in 0..n_n {
                    be.rms_norm_q(ins + f(j.min(n_in - 1) * n), n, l.norms.addr() + f(j.min(n_w - 1) * n), s.nrm + f(j * n), n, n, 1, c.eps, norm_xq(j), 0)?;
                }
                Ok(())
            })?;
        }
        let inp = |j: usize| s.nrm + f(j.min(n_n - 1) * n);
        be.batch(&mut || {
            be.gemvq2(w(&ml.wq), w(&ml.gate), inp(0), n, norm_xq(0), s.q, s.og, 1)?;
            be.gemvq(w(&ml.rel), inp(0), n, norm_xq(0), s.r, ml.rel.rows, 1)?;
            be.gemvq(w(&ml.sink), inp(0), n, norm_xq(0), s.sink, ml.sink.rows, 1)?;
            if n_n == 1 {
                be.gemvq2(w(&ml.wk), w(&ml.wv), inp(0), n, norm_xq(0), s.k, s.v, 1)?;
            } else {
                be.gemvq(w(&ml.wk), inp(1), n, norm_xq(1), s.k, ml.wk.rows, 1)?;
                be.gemvq(w(&ml.wv), inp(2), n, norm_xq(2), s.v, ml.wv.rows, 1)?;
            }
            if let Some([kp, vp]) = &ml.conv_proj {
                be.gemvq(w(kp), inp(1), n, norm_xq(1), s.kp, kp.rows, 1)?;
                be.gemvq(w(vp), inp(2), n, norm_xq(2), s.vp, vp.rows, 1)?;
            }
            Ok(())
        })?;
        let taps = if ml.conv_proj.is_some() { [s.kp, s.vp] } else { [0, 0] };
        be.batch(&mut || {
            be.ink_conv(s.k, l.convs[0].addr(), taps[0], hist(0), s.k, kw, nc, hd)?;
            be.ink_conv(s.v, l.convs[1].addr(), taps[1], hist(1), s.v, kw, nc, hd)
        })?;
        // Log-scaled positions sharpen the full-attention layers in a long context.
        let log_alpha = if ml.swa { 0.0 } else { c.log_alpha };
        let ring = ds.cap;
        if graph {
            let max_rows = ((pos / 64 + 1) * 64).min(ring);
            be.batch(&mut || {
                be.ink_clip_pos(s.q, l.q_clip.addr(), hd, nh, 1, c.qk_clip, 0.0, 1.0, 0, 1, s.pos, log_alpha, c.log_floor)?;
                be.ink_clip_pos(s.k, 0, hd, nkv, 1, c.qk_clip, 0.0, 1.0, ds.k.addr(), ring, s.pos, 0.0, 1.0)?;
                be.ink_clip_pos(s.v, 0, hd, nkv, 1, 0.0, c.eps, 1.0, ds.v.addr(), ring, s.pos, 0.0, 1.0)?;
                be.ink_clip_pos(s.r, l.rel_clip.addr(), nr, nh, 1, c.rel_clip, 0.0, 1.0, 0, 1, s.pos, log_alpha, c.log_floor)
            })?;
            be.ink_attn_pos(s.q, s.r, s.sink, s.og, ds.k.addr(), ds.v.addr(), l.rel_bias.addr(), self.sc_buf.addr(), s.att, max_rows, ring, s.pos, nh, nkv, hd, nr, c.rel_extent)?;
        } else {
            let (rows, at) = if ml.swa { ((pos + 1).min(c.n_swa), pos % c.n_swa) } else { (pos + 1, pos) };
            be.batch(&mut || {
                be.ink_clip_n(s.q, l.q_clip.addr(), hd, nh, 1, c.qk_clip, 0.0, 1.0, 0, 0, 1, pos, log_alpha, c.log_floor)?;
                be.ink_clip_n(s.k, 0, hd, nkv, 1, c.qk_clip, 0.0, 1.0, ds.k.addr(), at, ring, 0, 0.0, 1.0)?;
                be.ink_clip_n(s.v, 0, hd, nkv, 1, 0.0, c.eps, 1.0, ds.v.addr(), at, ring, 0, 0.0, 1.0)?;
                be.ink_clip_n(s.r, l.rel_clip.addr(), nr, nh, 1, c.rel_clip, 0.0, 1.0, 0, 0, 1, pos, log_alpha, c.log_floor)
            })?;
            be.ink_attn(s.q, s.r, s.sink, s.og, ds.k.addr(), ds.v.addr(), l.rel_bias.addr(), self.sc_buf.addr(), s.att, rows, at, nh, nkv, hd, nr, c.rel_extent)?;
        }
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
        be.rms_norm_q(h, n, l.ffn_norm.addr(), s.hn, n, n, 1, c.eps, s.xh, 0)
    }

    /// The routed experts of one token on the GPU `be` runs on, from the normed input in `s.hn` (and its quantized
    /// copy `s.xh`) to the weighted sum in `s.acc`. All of it stays on that device.
    fn ffn_core(be: &dyn Backend, c: &Inkling2Config, s: Scratch, fe: &FullExps) -> Result<()> {
        let (n, nf, k) = (c.n_embd, c.n_ff_exp, c.n_used);
        be.gemv(&fe.gate_inp, s.hn, s.rlogits)?;
        let gu_sc = fe.gu_scale.as_ref().map_or(0, |b| b.addr());
        let dn_sc = fe.dn_scale.as_ref().map_or(0, |b| b.addr());
        let gsc_a = if gu_sc != 0 { s.ids + f(2 * k) } else { 0 };
        be.ink_route(s.rlogits, fe.bias.addr(), s.ids, s.wc, gu_sc, dn_sc, gsc_a, c.n_expert, k, c.exp_scale, 1)?;
        be.moe_gemv(fe.gu_ty, fe.gu_rows, fe.gu_cols, fe.gu.addr(), fe.gu_stride, s.ids, k, s.hn, n, k, s.y, None, s.xh)?;
        if fe.dn_ty.float_x() {
            be.situ(s.y, nf, k, gsc_a, c.situ_beta, c.situ_linear_beta)?;
            be.moe_gemv(fe.dn_ty, fe.dn_rows, fe.dn_cols, fe.dn.addr(), fe.dn_stride, s.ids, k, s.y, 2 * nf, 1, s.d, None, 0)?;
        } else {
            be.situ_q(s.y, nf, k, gsc_a, c.situ_beta, c.situ_linear_beta, s.xq)?;
            be.moe_gemv(fe.dn_ty, fe.dn_rows, fe.dn_cols, fe.dn.addr(), fe.dn_stride, s.ids, k, s.y, 2 * nf, 1, s.d, None, s.xq)?;
        }
        be.ink_ffn_sum(s.acc, s.d, s.wc, k, n, 0)
    }

    /// What follows the experts of a layer run: the short convolution, the residual add, and the swap of the streams.
    fn ffn_tail(&mut self, be: &dyn Backend, m: &Inkling2Model, il: usize, ip: usize) -> Result<()> {
        let (c, ds, s) = (&m.cfg, &self.slots[il], self.s);
        let (n, hd, nc) = (c.n_embd, c.head_dim, c.n_conv);
        let hist3 = ds.hist.addr() + f(hists(c).0[3]);
        be.ink_conv(s.acc, self.layers[ip].convs[3].addr(), 0, hist3, s.acc, n, nc, hd)?;
        be.add(s.x, s.acc, n)?;
        (self.s.cur, self.s.x) = (s.x, s.cur);
        Ok(())
    }

    fn layer_ffn_gpu(&mut self, be: &dyn Backend, m: &Inkling2Model, il: usize, ip: usize) -> Result<bool> {
        let Some(fe) = &self.layers[ip].full_exps else { return Ok(false) };
        Self::ffn_core(be, &m.cfg, self.s, fe)?;
        self.ffn_tail(be, m, il, ip)?;
        Ok(true)
    }

    /// `layer_ffn_gpu` for a layer whose experts are on GPU 1: the normed input crosses to it through the host, the
    /// experts run there, and the sum comes back (two small copies and two syncs per layer run). False: not that kind of layer.
    fn layer_ffn_remote(&mut self, d: &dev::Dev, m: &Inkling2Model, il: usize, ip: usize) -> Result<bool> {
        let (Some(fe), Some(r)) = (&self.layers[ip].full_exps, &self.rem) else { return Ok(false) };
        if fe.part == 0 { return Ok(false) }
        let (c, s, rs) = (&m.cfg, self.s, r.s);
        let (be0, be1) = (&*d.parts[0].be, &*d.parts[r.part].be);
        let (n, o0, o1) = (c.n_embd, self.scr.addr(), r.scr.addr());
        let xh_words = (n + n.div_ceil(32) * 4).div_ceil(16) * 4;
        let hn: Vec<f32> = be0.get(&self.scr, ((s.hn - o0) / 4) as usize, n)?;
        let xh: Vec<f32> = be0.get(&self.scr, ((s.xh - o0) / 4) as usize, xh_words)?;
        be1.put(&hn, &r.scr, ((rs.hn - o1) / 4) as usize)?;
        be1.put(&xh, &r.scr, ((rs.xh - o1) / 4) as usize)?;
        Self::ffn_core(be1, c, rs, fe)?;
        let acc: Vec<f32> = be1.get(&r.scr, ((rs.acc - o1) / 4) as usize, n)?;
        be0.put(&acc, &self.scr, ((s.acc - o0) / 4) as usize)?;
        self.ffn_tail(be0, m, il, ip)?;
        Ok(true)
    }

    /// One layer run of the token (hybrid path): slot `il` with weights of layer `ip`.
    fn layer(&mut self, m: &Inkling2Model, il: usize, ip: usize, msgs: bool, pos: usize, nb: &mut usize) {
        let (c, ml, ds, s) = (&m.cfg, &m.layers[ip], &self.slots[il], self.s);
        let (n, hd, nc, nf) = (c.n_embd, c.head_dim, c.n_conv, c.n_ff_exp);
        let o = self.scr.addr();
        let hist3 = ds.hist.addr() + f(hists(c).0[3]);
        let lock = || dev::get().unwrap().lock().unwrap();
        let t = std::time::Instant::now();
        let mut hn = vec![];
        {
            let d = lock();
            let be = &*d.parts[0].be;
            self.layer_attn(be, &d, m, il, ip, msgs, pos, false, nb).expect("inkling2 device step");
            if self.layer_ffn_remote(&d, m, il, ip).expect("inkling2 device step remote") || self.layer_ffn_gpu(be, m, il, ip).expect("inkling2 device step") {
                m.lap(0, t);
                return;
            }
            hn = be.get(&self.scr, ((s.hn - o) / 4) as usize, n).expect("inkling2 device step");
        }
        m.lap(0, t);
        // The experts: those of the GPU read the device's copy of the input and leave their outputs there.
        let (on_gpu, on_cpu, adapt) = m.plan(ip, &hn);
        // Local patch: an expert with a matrix on another GPU cannot run in this GPU's queue; `remote` runs through
        // `run_experts` (each matrix where it lives), next to the CPU share.
        let on_first = |x: &Mat| x.gpu.is_none_or(|i| i == dev::STAGED || dev::gpu_of(i) == 0);
        let (list, remote): (Vec<_>, Vec<_>) = Inkling2Model::resident(ml, c.n_expert, &on_gpu).into_iter().partition(|e| on_first(&e.0) && on_first(&e.1));
        let gpu_side = || -> Option<f32> {
            let t = std::time::Instant::now();
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
                        if dn.ty.float_x() {
                            p.be.situ(s.y, nf, k, s.ids + f(2 * k), c.situ_beta, c.situ_linear_beta).ok()?;
                            p.be.moe_gemv(dn.ty, dn.rows, dn.cols, bd, sd, s.ids + 4 * k as u64, k, s.y, 2 * nf, 1, s.d, None, 0).ok()?;
                        } else {
                            p.be.situ_q(s.y, nf, k, s.ids + f(2 * k), c.situ_beta, c.situ_linear_beta, s.xq).ok()?;
                            p.be.moe_gemv(dn.ty, dn.rows, dn.cols, bd, sd, s.ids + 4 * k as u64, k, s.y, 2 * nf, 1, s.d, None, s.xq).ok()?;
                        }
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
            })()?;
            Some(t.elapsed().as_secs_f32())
        };
        let cpu_side = || {
            let t = std::time::Instant::now();
            (run_experts(&on_cpu, &hn, c), t.elapsed().as_secs_f32())
        };
        let remote_side = || (!remote.is_empty()).then(|| run_experts(&remote, &hn, c));
        let (gpu, ((cpu, tc), rem)) = rayon::join(gpu_side, || rayon::join(cpu_side, remote_side));
        let d = lock();
        let be = &*d.parts[0].be;
        match gpu {
            Some(tg) => {
                m.balance(&m.cpu_share, adapt, tg, tc);
                let host: Vec<f32> = match rem { Some(r) => cpu.iter().zip(&r).map(|(a, b)| a + b).collect(), None => cpu };
                let wc: Vec<f32> = (0..c.n_used).map(|i| list.get(i).map_or(0.0, |e| e.2)).chain(host).collect();
                be.put(&wc, &self.scr, ((s.wc - o) / 4) as usize).expect("inkling2 device step");
                be.ink_ffn_sum(s.acc, s.d, s.wc, list.len(), n, s.wc + f(c.n_used)).expect("inkling2 device step");
            }
            None => {
                // No room to stage a routed weight (a card with little free VRAM): the experts of the GPU run on
                // the host too, and the sum goes straight into `s.acc`. The conv and residual stay on the device.
                let y: Vec<f32> = m.experts_host(ip, &on_gpu, &hn).iter().zip(&cpu).map(|(a, b)| a + b).collect();
                be.put(&y, &self.scr, ((s.acc - o) / 4) as usize).expect("inkling2 device step");
            }
        }
        (|| -> Result<()> {
            be.ink_conv(s.acc, self.layers[ip].convs[3].addr(), 0, hist3, s.acc, n, nc, hd)?;
            be.add(s.x, s.acc, n)
        })().expect("inkling2 device step");
        (self.s.cur, self.s.x) = (s.x, s.cur);
    }
}
