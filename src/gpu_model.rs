//! GPU decode forward for the architectures in `config::arch`. Routed experts resident in the expert cache run on the GPU; the
//! rest run on the CPU (`cpu_moe`) while the GPU computes hits and the shared expert. Mirrors `model.rs`.

use crate::config::LayerKind;
use crate::cpu_worker::{CpuWorker, TMAX};
use crate::backend::{ATT_GQA_MIN, ATT_SPLITS, Backend, Buf, GpuTensor};
use crate::expert_cache::ExpertCache;
use crate::model::Model;
use crate::ops::Sampler;
use crate::quant::GgmlType;
use anyhow::{bail, ensure, Context};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::Duration;

/// The decode attention `gq_min` for `nt` tokens from position `pos`. The kernels pick the regime per token; when all
/// tokens fall on one side of ATT_GQA_MIN, i32::MAX (one head per block) or 0 (GQA) gives the same result and lets the
/// backend launch only the kernel that works.
fn gq_min(pos: usize, nt: usize) -> i32 {
    let m = ATT_GQA_MIN as usize;
    if pos + nt < m { i32::MAX } else if pos + 1 >= m { 0 } else { ATT_GQA_MIN }
}


mod prefill;

/// Scratch rows for `hc_read` and `ple`: the decode buffers (`GpuModel::hc_at`) or the prefill slab, where `gemm`
/// runs their matrices as dense GEMMs over the chunk. `hq` 0: no int8 copy of h.
#[derive(Clone, Copy)]
struct HcAt { xn: u64, lo: u64, hg: u64, inj: u64, h: u64, hq: u64, emb: u64, pk: u64, pq: u64, pg: u64, pn: u64, pv: u64, gemm: bool }

enum Mixer {
    Delta {
        qkv: GpuTensor, gate: GpuTensor, beta: GpuTensor, alpha: GpuTensor, dt_bias: GpuTensor, a: GpuTensor,
        conv_w: GpuTensor, norm: GpuTensor, out: GpuTensor, conv: Buf, s: Buf,
        /// Conv and DeltaNet states after each of the first `k` verify tokens (MTP only), for `rollback`.
        snap: Option<(Buf, Buf)>,
    },
    Attn { q: GpuTensor, k: GpuTensor, v: GpuTensor, qn: GpuTensor, kn: GpuTensor, o: GpuTensor, kc: Buf, vc: Buf, qsa: Option<Qsa> },
}

/// QSA indexer of an attention layer (qwen4exp, see `Model::qsa`): projections and norms, the pooled keys (128 floats
/// per block of 4 positions) and this layer's 4 rows of `QsaBufs::ring`.
struct Qsa { kp: GpuTensor, qp: GpuTensor, kn: GpuTensor, qn: GpuTensor, pool: Buf, ring: u64 }

/// QSA scratch shared by the layers: raw indexer keys of each layer's incomplete block (`QSA_RING` floats a layer)
/// and their copy at `save_state`, block scores (rows of `nbs`), decode cell lists (rows of `sst`), prefill masks
/// (rows of `mw` words).
struct QsaBufs { ring: Buf, saved: Buf, score: Buf, sel: Buf, mask: Buf, top: usize, nbs: usize, sst: usize, mw: usize }

/// Positions a QSA pool block covers, the indexer head size and heads (the kernels' QSA_D and QSA_H).
const QSA_POOL: usize = 4;
const QSA_D: usize = 128;
const QSA_H: usize = 4;
const QSA_RING: usize = QSA_POOL * QSA_D;
/// Query rows per `qsa_select` launch in prefill (score rows).
const QSA_ROWS: usize = 64;

struct Shared { gate_inp: GpuTensor, gate: GpuTensor, up: GpuTensor, down: GpuTensor }

enum Ffn {
    Dense { gate: GpuTensor, up: GpuTensor, down: GpuTensor },
    Moe { router: GpuTensor, shared: Option<Shared> },
}

/// qwen4exp hyper-connection read weights (`{p}norm/down/up/inject.weight`, see `Model::hc_read`).
struct HcRead { norm: GpuTensor, down: GpuTensor, up: GpuTensor, inject: Option<GpuTensor> }

/// qwen4exp PLE block weights (see `Model::ple`); `conv` as f32 [channel][tap].
struct Ple { key: GpuTensor, value: GpuTensor, norm_key: GpuTensor, norm_query: GpuTensor, norm_conv: GpuTensor, conv: GpuTensor }

/// `attn_norm` / `post_norm` for a plain residual; `hc` (attention and FFN reads) for hyper-connections.
struct Layer { attn_norm: Option<GpuTensor>, post_norm: Option<GpuTensor>, hc: Option<[HcRead; 2]>, ple: Option<Ple>, mixer: Mixer, ffn: Ffn }

/// MTP draft head: the `nextn` block (layer n_layer) with its own KV cache and CPU worker.
/// A saved conversation state in host memory: its tokens, the PLE tokens before it, the `slot` buffers and the KV
/// rows of its positions in each attention layer.
struct Parked { ids: Vec<u32>, ple_prev: [i64; 2], slot: Vec<Vec<u8>>, kv: Vec<Vec<u8>> }

impl Parked {
    fn bytes(&self) -> usize { self.slot.iter().chain(&self.kv).map(Vec::len).sum() }
}

struct MtpHead {
    layer: Layer, eh_proj: GpuTensor, enorm: GpuTensor, hnorm: GpuTensor, head_norm: GpuTensor,
    worker: CpuWorker, step_buf: Buf, emb: Buf, eh: Buf, x: Buf, pos: usize,
    /// Row 0 of the main model's `x` at `save_state` (the hidden state the next entry pairs with).
    saved_x: Buf,
    /// Accepted drafts of the last `spec_step` whose draft head entries (from the main model's hidden rows 0..) are
    /// still to be written; the next `spec_step` writes them in its first draft pass.
    pend: Vec<u32>,
}

pub struct GpuModel<'a> {
    m: &'a Model<'a>,
    pub gpu: Box<dyn Backend>,
    layers: Vec<Layer>,
    output_norm: Option<GpuTensor>,
    /// Hyper-connection read before the head (qwen4exp, in place of output_norm).
    output_hc: Option<HcRead>,
    /// None on all but the last part of a split model.
    output: Option<GpuTensor>,
    /// qwen4exp PLE: the normalized rows of the last 16 positions (see `ple_conv`), its copy at `save_state`, the two
    /// previous tokens for the n-gram hash and their copy.
    ple_ring: Option<(Buf, Buf)>,
    qsa: Option<QsaBufs>,
    ple_prev: [i64; 2],
    saved_ple_prev: [i64; 2],
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
    /// Host copy of the saved state at the end of the system prompt (`stash`): buffers, position, PLE tokens, the
    /// tokens behind it. A request whose history no longer extends the saved one (a rewritten or compacted
    /// conversation) restarts from it.
    sys: Option<(Vec<Vec<u8>>, usize, [i64; 2], Vec<u32>)>,
    /// Saved states that a request with another history replaced, most recent first (`park`).
    parked: Vec<Parked>,
    /// Host memory for `parked`, in bytes (0 = keep none).
    pub cache_ram: usize,
    ctx: usize,
    pub pos: usize,
    pub cache: Option<ExpertCache>,
    /// Swap experts every this many tokens (0 = fixed placement).
    pub adapt_every: usize,
    pub hits: usize,
    pub misses: usize,
    /// Time the CPU worker spent computing missed experts (overlaps GPU work).
    pub cpu_time: Duration,
    /// Forwards run again from a MoE combine that gave up waiting for the CPU worker (`finish`).
    pub resumed: std::cell::Cell<usize>,
    /// MTP drafts (made, accepted).
    pub drafts: (usize, usize),
    /// Split model, all parts but the last: seconds from the start of this part's forward until its GPU is done, and
    /// for the copy of its hidden rows to the next part.
    pub split_time: (Duration, Duration),
    /// MTP drafts per speculative step (0 = no draft head). May be lowered after `load`, never raised.
    pub k: usize,
    /// Prompts shorter than this go through `step` instead of the batched prefill.
    pub short: usize,
    /// Prefill scratch (None: prompts go through `step`).
    pf: Option<prefill::Scratch>,
    mtp: Option<MtpHead>,
    /// The layers of this part are `first..first + layers.len()`.
    first: usize,
    /// The part on the next device of a model split by layers (`load_split`): this part's hidden state goes to it
    /// through host memory, and the last part computes the logits. Public calls go to the first part.
    next: Option<Box<GpuModel<'a>>>,
}

impl<'a> GpuModel<'a> {
    /// `chunk`: prefill batch in tokens (0 = no batched prefill). `mtp`: load the MTP draft head (needs `nextn`).
    pub fn load(m: &'a Model<'a>, gpu: Box<dyn Backend>, ctx: usize, chunk: usize, vram_margin: usize, k: usize) -> anyhow::Result<GpuModel<'a>> {
        Self::load_part(m, gpu, ctx, chunk, vram_margin, k, 0..m.cfg.n_layer)
    }

    /// `load` with the layers split over `gpus` in proportion to their free VRAM (each device then fills the rest
    /// of its VRAM with expert slots). One device: `load`.
    pub fn load_split(m: &'a Model<'a>, gpus: Vec<Box<dyn Backend>>, ctx: usize, chunk: usize, vram_margin: usize, k: usize) -> anyhow::Result<GpuModel<'a>> {
        let n = m.cfg.n_layer;
        if gpus.len() == 1 { return Self::load(m, gpus.into_iter().next().unwrap(), ctx, chunk, vram_margin, k); }
        let free = gpus.iter().map(|g| Ok(g.mem_info()?.0 as f64)).collect::<anyhow::Result<Vec<_>>>()?;
        let total: f64 = free.iter().sum();
        let mut ends = vec![0];
        let mut acc = 0.0;
        for f in &free { acc += f; ends.push((acc / total * n as f64).round() as usize); }
        // The PLE layers share one history ring, which lives on one device.
        let ple: Vec<usize> = (0..n).filter(|&il| m.w.layer(il, "ple_key.weight").is_ok()).collect();
        let mut parts = vec![];
        for (i, gpu) in gpus.into_iter().enumerate() {
            let r = ends[i]..ends[i + 1];
            ensure!(!r.is_empty(), "device {i} gets no layers (free VRAM {:.0} MB)", free[i] / 1048576.0);
            ensure!(ple.iter().all(|il| r.contains(il)) || !ple.iter().any(|il| r.contains(il)), "the PLE layers would be split across devices");
            eprintln!("device {i} ({}): layers {}..{}", gpu.name(), r.start, r.end);
            // ponytail: no MTP draft head on a split model (it reads the last part's hidden rows from the first).
            parts.push(Self::load_part(m, gpu, ctx, chunk, vram_margin, 0, r)?);
        }
        let mut it = parts.into_iter().rev();
        let mut acc = it.next().unwrap();
        for mut p in it { p.next = Some(Box::new(acc)); acc = p; }
        Ok(acc)
    }

    fn load_part(m: &'a Model<'a>, gpu: Box<dyn Backend>, ctx: usize, chunk: usize, vram_margin: usize, k: usize, range: std::ops::Range<usize>) -> anyhow::Result<GpuModel<'a>> {
        let last = range.end == m.cfg.n_layer;
        // The PLE table stays on the CPU; other F16, Q4_0 and Q5_0 tensors but experts are converted on upload
        // (`Backend::upload`).
        let cpu_only = |n: &str| n == "per_layer_token_embd.weight";
        let f32_up = |n: &str, ty| matches!(ty, GgmlType::F16 | GgmlType::Q4_0 | GgmlType::Q5_0) && !n.contains("_exps.");
        if let Some(t) = m.w.g.tensors.values().find(|t| !cpu_only(&t.name)
            && GgmlType::from_u32(t.ty).is_ok_and(|ty| !ty.gpu(gpu.name()) && !f32_up(&t.name, ty))) {
            anyhow::bail!("tensor {} has type {}, which the {} backend does not support yet", t.name, GgmlType::from_u32(t.ty)?.name(), gpu.name());
        }
        let c = &m.cfg;
        let up = |name: &str| -> anyhow::Result<GpuTensor> { gpu.upload(&m.w.get(name)?) };
        let lt = |il: usize, s: &str| up(&format!("blk.{il}.{s}"));
        let hc_read = |p: &str| -> anyhow::Result<HcRead> {
            let t = |s: &str| up(&format!("{p}{s}.weight"));
            Ok(HcRead { norm: t("norm")?, down: t("down")?, up: t("up")?, inject: t("inject").ok() })
        };
        let hcn = c.hc_n.max(1);
        if c.hc_n > 0 && !c.is_moe() { bail!("hyper-connections without MoE are not supported on the GPU"); }
        let (free0, _) = gpu.mem_info()?;
        let (dk, nk, nv) = (c.ssm_d_state, c.ssm_n_group, c.ssm_dt_rank);
        let channels = c.ssm_d_inner + 2 * nk * dk;
        let kv = c.n_head_kv * c.head_dim;
        let qsa_on = c.idx_top_k > 0 && (0..c.n_layer).any(|il| m.w.layer(il, "indexer.k_proj.weight").is_ok());
        if qsa_on { ensure!(c.idx_head_dim == QSA_D && c.idx_n_head == QSA_H, "QSA indexer: {} heads of {} (the GPU kernels take {QSA_H} of {QSA_D})", c.idx_n_head, c.idx_head_dim); }
        let ring = gpu.zeros::<f32>(if qsa_on { c.n_layer * QSA_RING } else { 1 })?;

        // ponytail: no MTP or batched prefill with hyper-connections yet (their state and row layouts assume one stream).
        let k = if c.n_nextn > 0 && c.hc_n == 0 { k.min(TMAX - 1) } else { 0 };
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
                    qsa: if qsa_on && il < c.n_layer && m.w.layer(il, "indexer.k_proj.weight").is_ok() {
                        Some(Qsa {
                            kp: lt(il, "indexer.k_proj.weight")?, qp: lt(il, "indexer.q_proj.weight")?,
                            kn: lt(il, "indexer.k_norm.weight")?, qn: lt(il, "indexer.q_norm.weight")?,
                            // Rows of 128 floats per position, as the KV rows (`kv_bufs`): ctx / 4 blocks of 512.
                            pool: gpu.zeros::<f32>(ctx * QSA_D / QSA_POOL)?, ring: ring.addr() + (4 * il * QSA_RING) as u64,
                        })
                    } else { None },
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
            let hc = if c.hc_n > 0 { Some([hc_read(&format!("blk.{il}.hc_attn_"))?, hc_read(&format!("blk.{il}.hc_ffn_"))?]) } else { None };
            let ple = if m.w.layer(il, "ple_key.weight").is_ok() {
                let kw = m.w.layer(il, "ple_conv1d.weight")?;
                let conv = gpu.zeros::<f32>(kw.rows * kw.cols)?;
                gpu.put(&kw.to_f32(), &conv, 0)?;
                Some(Ple {
                    key: lt(il, "ple_key.weight")?, value: lt(il, "ple_value.weight")?, norm_key: lt(il, "ple_norm_key.weight")?,
                    norm_query: lt(il, "ple_norm_query.weight")?, norm_conv: lt(il, "ple_norm_conv.weight")?,
                    conv: GpuTensor::new(GgmlType::F32, kw.rows, kw.cols, conv),
                })
            } else { None };
            let (attn_norm, post_norm) = if hc.is_some() { (None, None) } else { (Some(lt(il, "attn_norm.weight")?), Some(lt(il, "post_attention_norm.weight")?)) };
            Ok(Layer { attn_norm, post_norm, hc, ple, mixer, ffn })
        };
        let layers = range.clone().map(|il| load_layer(il, c.layers[il])).collect::<anyhow::Result<Vec<_>>>()?;
        let mtp_parts = if mtp {
            let il = c.n_layer;
            Some((load_layer(il, LayerKind::Attn)?, lt(il, "nextn.eh_proj.weight")?, lt(il, "nextn.enorm.weight")?,
                  lt(il, "nextn.hnorm.weight")?, lt(il, "nextn.shared_head_norm.weight")?,
                  gpu.zeros::<i32>(2 * TMAX)?, gpu.zeros::<f32>(TMAX * c.n_embd)?, gpu.zeros::<f32>(TMAX * 2 * c.n_embd)?, gpu.zeros::<f32>(TMAX * c.n_embd)?))
        } else { None };
        let (output_norm, output_hc) = if !last { (None, None) } else if c.hc_n > 0 { (None, Some(hc_read("output_hc_")?)) } else { (Some(up("output_norm.weight")?), None) };
        let ple_ring = if c.n_embd_ple > 0 { Some((gpu.zeros::<f32>(16 * hcn * c.n_embd)?, gpu.zeros::<f32>(16 * hcn * c.n_embd)?)) } else { None };
        let output = if last { Some(up("output.weight").or_else(|_| up("token_embd.weight"))?) } else { None };

        let ke = c.n_expert_used.max(1);
        let ff = (ke * c.n_ff_exp).max(c.n_ff);
        let mut sizes = vec![
            ("h", c.n_embd), ("hq", c.n_embd * 9 / 32), ("mix", c.n_embd), ("qkv", channels), ("z", c.ssm_d_inner), ("beta", nv), ("g", nv),
            ("y", channels), ("o", c.ssm_d_inner), ("qg", c.n_head * 2 * c.head_dim), ("q", c.n_head * c.head_dim),
            ("kraw", kv), ("k", kv), ("v", kv), ("att", c.n_head * c.head_dim), ("rl", c.n_expert), ("wts", ke),
            ("ge", ff), ("ue", ff), ("ye", ke * c.n_embd), ("shg", c.n_ff_shexp), ("shu", c.n_ff_shexp),
            ("shy", c.n_embd), ("sg", 1), ("wh", ke), ("attp", c.n_head * ATT_SPLITS * (c.head_dim + 2)),
            ("ffq", ff * 9 / 32), ("shq", c.n_ff_shexp * 9 / 32), ("attq", c.n_head * c.head_dim * 9 / 32),
            ("oq", c.ssm_d_inner * 9 / 32),
        ];
        if qsa_on { sizes.extend([("ik", QSA_D), ("iq", QSA_H * QSA_D), ("iqn", QSA_H * QSA_D)]); }
        if c.hc_n > 0 {
            let d = c.hc_n * c.n_embd;
            sizes.extend([("xn", d), ("lo", c.hc_rank), ("hg", d), ("inj", c.hc_n), ("mo", c.n_embd),
                          ("pk", d), ("pq", d), ("pg", d), ("pn", d), ("pv", c.n_embd), ("emb", c.n_embd)]);
        }
        let bufs = sizes.into_iter().map(|(name, n)| Ok((name, gpu.zeros::<f32>(TMAX * n)?))).collect::<anyhow::Result<HashMap<_, _>>>()?;
        let (x, logits, ids) = (gpu.zeros::<f32>(TMAX * hcn * c.n_embd)?, gpu.zeros::<f32>(TMAX * c.n_vocab)?, gpu.zeros::<i32>(TMAX * ke)?);
        let amax = gpu.zeros::<i32>(TMAX)?;
        let (slots, step_buf) = (gpu.zeros::<i32>(TMAX * ke)?, gpu.zeros::<i32>(2 * TMAX)?);
        // Hyper-connection prefill scratch is ~4x larger a token, so a large chunk leaves fewer expert slots: qwen4exp at
        // 943 tokens (warm) prefills 447 tok/s with 1024, 347 with 4096. Long prompts gain (each chunk restages the experts
        // not resident: Flash-Next at 4764 tokens 132 -> 158 tok/s on a 3060), so `calibrate` picks per machine.
        let pf = if chunk > 0 { Some(prefill::Scratch::new(m, &*gpu, chunk.min(ctx))?) } else { None };
        let qsa = if qsa_on {
            let (top, nbs) = (c.idx_top_k / QSA_POOL, ctx / QSA_POOL);
            let (sst, mw) = (1 + c.idx_top_k + QSA_POOL, (nbs + 1).div_ceil(32));
            let rows = pf.as_ref().map_or(0, |p| p.chunk.next_multiple_of(64));
            Some(QsaBufs { saved: gpu.zeros::<f32>(ring.len() / 4)?, ring, score: gpu.zeros::<f32>(QSA_ROWS.max(TMAX) * nbs)?,
                           sel: gpu.zeros::<i32>(TMAX * sst)?, mask: gpu.zeros::<u32>((rows * mw).max(1))?, top, nbs, sst, mw })
        } else { None };
        let (free1, _) = gpu.mem_info()?;
        // Past the budget the driver spills buffers to system memory: RADV then runs a long prefill slowly enough to
        // hit amdgpu's job timeout and lose the device. A margin above 512 MB only leaves fewer expert slots.
        let floor = vram_margin.min(512 << 20);
        ensure!(free1 >= floor, "the model, KV cache and prefill scratch leave {} MB of VRAM free, under the {} MB \
            margin: lower --ctx, use --kv q8 or a smaller --chunk (--vram-margin 0 skips this check)", free1 >> 20, floor >> 20);

        let cache = if c.is_moe() {
            // STRATA_EXPERT_MB caps the expert slots, so that runs place the same experts whatever VRAM is free (a
            // backend comparison, or other programs holding VRAM).
            let cap = std::env::var("STRATA_EXPERT_MB").ok().and_then(|v| v.parse::<usize>().ok()).map_or(usize::MAX, |v| v << 20);
            let budget = free1.saturating_sub(vram_margin).min(cap);
            Some(ExpertCache::new(m, &*gpu, budget, range.start..range.end + mtp as usize)?)
        } else { None };
        let worker = if c.is_moe() { Some(CpuWorker::new(m, &*gpu, range.clone().collect())?) } else { None };
        let mb = |b: usize| b as f64 / 1048576.0;
        eprintln!(
            "plan: weights+state+scratch {:.0} MB (KV {:.0} MB for {ctx} tokens, prefill {:.0} MB), expert slots {:.0} MB, resident experts {}/{}",
            mb(free0.saturating_sub(free1)), mb(2 * 2 * gpu.kv_words(ctx * kv) * c.layers[range.clone()].iter().filter(|&&k| k == LayerKind::Attn).count()), mb(pf.as_ref().map_or(0, |p| p.bytes())),
            mb(cache.as_ref().map_or(0, |x| x.slot_bytes_total())),
            cache.as_ref().map_or(0, |x| x.resident_count()), cache.as_ref().map_or(0, |x| x.total_experts()),
        );
        let mtp = match mtp_parts {
            Some((layer, eh_proj, enorm, hnorm, head_norm, step_buf, emb, eh, x)) => Some(MtpHead {
                layer, eh_proj, enorm, hnorm, head_norm, worker: CpuWorker::new(m, &*gpu, vec![c.n_layer])?,
                step_buf, emb, eh, x, pos: 0, pend: vec![], saved_x: gpu.zeros::<f32>(c.n_embd)?,
            }),
            None => None,
        };
        Ok(GpuModel { m, gpu, layers, output_norm, output_hc, output, ple_ring, qsa, ple_prev: [-1, -1], saved_ple_prev: [-1, -1], bufs, x, worker, logits, amax, ids, slots, step_buf, ctx, pos: 0, saved: vec![], saved_pos: 0, saved_ids: vec![], sys: None, parked: vec![], cache_ram: 0, cache, adapt_every: 8, hits: 0, misses: 0, cpu_time: Duration::ZERO, resumed: Default::default(), drafts: (0, 0), split_time: Default::default(), k, short: 32, pf, mtp, first: range.start, next: None })
    }

    /// Clear KV and recurrent state (and forget the `prefill_cached` prefix, whose KV this zeroes).
    pub fn reset(&mut self) -> anyhow::Result<()> {
        self.saved_ids.clear();
        self.sys = None;
        if let Some(h) = self.mtp.as_mut() { h.pos = 0; h.pend.clear(); }
        for l in self.layers.iter().chain(self.mtp.as_ref().map(|h| &h.layer)) {
            match &l.mixer {
                Mixer::Delta { conv, s, .. } => { self.gpu.memset(conv)?; self.gpu.memset(s)?; }
                Mixer::Attn { kc, vc, .. } => { self.gpu.memset(kc)?; self.gpu.memset(vc)?; }
            }
        }
        self.pos = 0;
        self.ple_prev = [-1, -1];
        if let Some(n) = self.next.as_mut() { n.reset()?; }
        Ok(())
    }

    /// This part and the parts after it.
    pub fn parts(&self) -> impl Iterator<Item = &GpuModel<'a>> { std::iter::successors(Some(self), |p| p.next.as_deref()) }

    /// The part that computes the logits.
    fn tail(&self) -> &GpuModel<'a> { self.parts().last().unwrap() }

    /// Run the forward of `nt` tokens (input in `x`, per-token values in `step_buf`) through this part and the
    /// parts after it. The last part's forward is left running, as `run` leaves it.
    fn pass(&mut self, nt: usize) -> anyhow::Result<()> {
        let t0 = std::time::Instant::now();
        self.run(nt)?;
        if self.next.is_none() { return Ok(()); }
        self.gpu.sync()?;
        let t1 = std::time::Instant::now();
        self.finish(nt, gq_min(self.pos, nt), None)?;
        let c = &self.m.cfg;
        let xs = self.gpu.get::<f32>(&self.x, 0, nt * c.hc_n.max(1) * c.n_embd)?;
        let pe = match self.bufs.get("emb") { Some(e) => self.gpu.get::<f32>(e, 0, nt * c.n_embd)?, None => vec![] };
        let nx = self.next.as_deref_mut().unwrap();
        nx.gpu.put(&xs, &nx.x, 0)?;
        if !pe.is_empty() { nx.gpu.put(&pe, &nx.bufs["emb"], 0)?; }
        let seq = nx.worker.as_mut().map_or(0, |w| w.begin());
        let steps: Vec<i32> = (0..nt).flat_map(|t| [(nx.pos + t) as i32, seq as i32]).collect();
        nx.gpu.put(&steps, &nx.step_buf, 0)?;
        self.split_time.0 += t1 - t0;
        self.split_time.1 += t1.elapsed();
        nx.pass(nt)
    }

    /// Put the embeddings of `tokens` into `x` (repeated per hyper-connection stream) and their PLE input into "emb".
    fn put_embed(&mut self, tokens: &[u32]) -> anyhow::Result<()> {
        let hc = self.m.cfg.hc_n.max(1);
        let emb: Vec<f32> = tokens.iter().flat_map(|&t| self.m.embed(t).repeat(hc)).collect();
        self.gpu.put(&emb, &self.x, 0)?;
        if self.ple_ring.is_some() {
            let pe: Vec<f32> = tokens.iter().flat_map(|&t| self.m.ple_embed(&mut self.ple_prev, t)).collect();
            self.gpu.put(&pe, &self.bufs["emb"], 0)?;
        }
        Ok(())
    }

    /// Run one token; returns logits over the vocabulary.
    pub fn step(&mut self, token: u32) -> anyhow::Result<Vec<f32>> {
        if self.pos >= self.ctx { bail!("context full ({} tokens; raise --ctx)", self.ctx); }
        let seq = self.worker.as_mut().map_or(0, |w| w.begin());
        self.put_embed(&[token])?;
        self.gpu.put(&[self.pos as i32, seq as i32], &self.step_buf, 0)?;
        self.pass(1)?;
        let t = self.tail();
        let mut logits = t.gpu.get::<f32>(&t.logits, 0, self.m.cfg.n_vocab)?;
        if t.finish(1, gq_min(self.pos, 1), None)? { logits = t.gpu.get::<f32>(&t.logits, 0, self.m.cfg.n_vocab)?; }
        self.after_forward(1)?;
        Ok(logits)
    }

    /// `step` that returns the greedy next token (argmax of the logits, first on ties) instead of the logits.
    pub fn step_greedy(&mut self, token: u32) -> anyhow::Result<u32> {
        Ok(self.verify_greedy(&[token], |_| 0)?.1)
    }

    /// Run up to `TMAX` tokens at positions pos.. in one forward (each weight read once); returns logits per token.
    /// Hidden state t stays in row t of `x` (see `mtp_draft`).
    pub fn verify(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<Vec<f32>>> {
        let nvoc = self.m.cfg.n_vocab;
        self.verify_run(tokens)?;
        let t = self.tail();
        let mut logits = t.gpu.get::<f32>(&t.logits, 0, tokens.len() * nvoc)?;
        if t.finish(tokens.len(), gq_min(self.pos, tokens.len()), None)? { logits = t.gpu.get::<f32>(&t.logits, 0, tokens.len() * nvoc)?; }
        self.after_forward(tokens.len())?;
        Ok(logits.chunks(nvoc).map(|c| c.to_vec()).collect())
    }

    /// `verify` for greedy speculation: `keep` gets the argmax of every row; returns its row n and that row's argmax.
    fn verify_greedy(&mut self, tokens: &[u32], keep: impl FnOnce(&[u32]) -> usize) -> anyhow::Result<(usize, u32)> {
        let (nt, nvoc) = (tokens.len(), self.m.cfg.n_vocab);
        self.verify_run(tokens)?;
        let t = self.tail();
        let g = &t.gpu;
        g.argmax_rows(t.logits.addr(), nvoc, nt, t.amax.addr())?;
        let mut am: Vec<u32> = g.get::<i32>(&t.amax, 0, nt)?.into_iter().map(|x| x as u32).collect();
        if t.finish(nt, gq_min(self.pos, nt), None)? {
            g.argmax_rows(t.logits.addr(), nvoc, nt, t.amax.addr())?;
            am = g.get::<i32>(&t.amax, 0, nt)?.into_iter().map(|x| x as u32).collect();
        }
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
        self.put_embed(tokens)?;
        let steps: Vec<i32> = (0..nt).flat_map(|t| [(self.pos + t) as i32, seq as i32]).collect();
        self.gpu.put(&steps, &self.step_buf, 0)?;
        self.pass(nt)
    }

    /// Save the DeltaNet and conv states (KV needs no copy: rewinding `pos` is enough).
    pub fn save_state(&mut self) -> anyhow::Result<()> {
        let g = &self.gpu;
        if self.saved.is_empty() {
            for l in &self.layers {
                if let Mixer::Delta { conv, s, .. } = &l.mixer { self.saved.push((g.alloc(conv.len())?, g.alloc(s.len())?)); }
            }
        }
        let mut it = self.saved.iter();
        for l in &self.layers {
            if let Mixer::Delta { conv, s, .. } = &l.mixer {
                let (bc, bs) = it.next().unwrap();
                g.dtod(conv, 0, bc, 0, conv.len())?;
                g.dtod(s, 0, bs, 0, s.len())?;
            }
        }
        if let Some(h) = &self.mtp { g.dtod(&self.x, 0, &h.saved_x, 0, 4 * self.m.cfg.n_embd)?; }
        if let Some((r, sr)) = &self.ple_ring { g.dtod(r, 0, sr, 0, r.len())?; }
        if let Some(q) = &self.qsa { g.dtod(&q.ring, 0, &q.saved, 0, q.ring.len())?; }
        self.saved_ple_prev = self.ple_prev;
        self.saved_pos = self.pos;
        self.saved_ids.clear();
        if let Some(n) = self.next.as_mut() { n.save_state()?; }
        Ok(())
    }

    /// The buffers `save_state` fills.
    fn slot(&self) -> Vec<&Buf> {
        let mut v: Vec<&Buf> = self.saved.iter().flat_map(|(c, s)| [c, s]).collect();
        v.extend(self.mtp.as_ref().map(|h| &h.saved_x));
        v.extend(self.ple_ring.as_ref().map(|(_, sr)| sr));
        v.extend(self.qsa.as_ref().map(|q| &q.saved));
        v
    }

    /// `save_state`, and keep a host copy of it as the state after `ids` (the system prompt).
    pub fn stash(&mut self, ids: &[u32]) -> anyhow::Result<()> {
        self.save_state()?;
        let mut p = Some(self);
        while let Some(q) = p {
            let data = q.slot().iter().map(|b| q.gpu.get::<u8>(b, 0, b.len())).collect::<anyhow::Result<_>>()?;
            q.saved_ids = ids.to_vec();
            q.sys = Some((data, q.saved_pos, q.saved_ple_prev, q.saved_ids.clone()));
            p = q.next.as_deref_mut();
        }
        Ok(())
    }

    /// The KV caches of the attention layers (and of the draft head) and their QSA pooled keys, whose rows are positions.
    fn kv_bufs(&self) -> Vec<&Buf> {
        self.layers.iter().chain(self.mtp.as_ref().map(|h| &h.layer))
            .flat_map(|l| match &l.mixer {
                Mixer::Attn { kc, vc, qsa, .. } => [kc, vc].into_iter().chain(qsa.as_ref().map(|a| &a.pool)).collect(),
                _ => vec![],
            }).collect()
    }

    /// Copy the saved state to host memory, so a later request that continues it can take it back (`unpark`); the
    /// least recently parked states go when they exceed `cache_ram`.
    fn park(&mut self) -> anyhow::Result<()> {
        let n = self.saved_ids.len();
        // ponytail: a split model parks nothing (each part would need its own copy, in step with the others).
        if n == 0 || self.next.is_some() { return Ok(()) }
        let kv = self.kv_bufs().iter().map(|b| b.len() / self.ctx * n).sum::<usize>();
        if kv + self.slot().iter().map(|b| b.len()).sum::<usize>() > self.cache_ram { return Ok(()) }
        let g = &self.gpu;
        let slot = self.slot().iter().map(|b| g.get::<u8>(b, 0, b.len())).collect::<anyhow::Result<_>>()?;
        let kv = self.kv_bufs().iter().map(|b| g.get::<u8>(b, 0, b.len() / self.ctx * n)).collect::<anyhow::Result<_>>()?;
        self.parked.insert(0, Parked { ids: self.saved_ids.clone(), ple_prev: self.saved_ple_prev, slot, kv });
        while self.parked.iter().map(Parked::bytes).sum::<usize>() > self.cache_ram { self.parked.pop(); }
        Ok(())
    }

    /// Make parked state `i` the saved state and return to it; the saved state is parked in its place.
    fn unpark(&mut self, i: usize) -> anyhow::Result<()> {
        let p = self.parked.remove(i);
        self.park()?;
        for (b, d) in self.slot().iter().zip(&p.slot) { self.gpu.put(d, b, 0)?; }
        for (b, d) in self.kv_bufs().iter().zip(&p.kv) { self.gpu.put(d, b, 0)?; }
        (self.saved_pos, self.saved_ple_prev, self.saved_ids) = (p.ids.len(), p.ple_prev, p.ids);
        self.restore_state()
    }

    /// Park the sequence whose state the model holds now (`ids`: the tokens it has seen), so another one can run.
    /// `adopt` takes it back; the model is left at no sequence (a later `park` is a no-op).
    pub fn park_current(&mut self, ids: &[u32]) -> anyhow::Result<()> {
        anyhow::ensure!(self.next.is_none(), "park_current: split model");
        self.save_state()?;
        self.saved_ids = ids.to_vec();
        self.park()?;
        let ok = self.parked.first().is_some_and(|p| p.ids == ids);
        self.saved_ids.clear();
        anyhow::ensure!(ok, "park_current: state of {} tokens not parked (cache_ram?)", ids.len());
        Ok(())
    }

    /// Make the parked state of `ids` the current one again.
    pub fn adopt(&mut self, ids: &[u32]) -> anyhow::Result<()> {
        let i = self.parked.iter().position(|p| p.ids == ids)
            .ok_or_else(|| anyhow::anyhow!("adopt: no parked state of {} tokens", ids.len()))?;
        self.unpark(i)
    }

    /// Forget the system-prompt stash: its KV rows are not guaranteed while several sequences share the cache.
    pub fn clear_stash(&mut self) { self.sys = None; }

    /// Return to the state of the last `stash`, which also becomes the saved state.
    pub fn unstash(&mut self) -> anyhow::Result<()> {        let mut p = Some(&mut *self);
        while let Some(q) = p {
            let s = q.sys.as_ref().ok_or_else(|| anyhow::anyhow!("unstash without stash"))?;
            for (b, d) in q.slot().iter().zip(&s.0) { q.gpu.put(d, b, 0)?; }
            (q.saved_pos, q.saved_ple_prev, q.saved_ids) = (s.1, s.2, s.3.clone());
            p = q.next.as_deref_mut();
        }
        self.restore_state()
    }

    /// Keep only the first `keep` tokens of the last `verify`, which started at position `start` (1 <= keep <= `k`):
    /// recurrent states come from the snapshots it wrote, KV entries past them are dropped by rewinding `pos`.
    pub fn rollback(&mut self, start: usize, keep: usize) -> anyhow::Result<()> {
        if !(1..=self.k).contains(&keep) || start + keep > self.pos { bail!("rollback: keep {keep} of verify at {start}"); }
        let g = &self.gpu;
        for l in &self.layers {
            if let Mixer::Delta { conv, s, snap: Some((sc, ss)), .. } = &l.mixer {
                // Snapshot `keep - 1` of each state; sizes in bytes.
                let (nc, ns) = (conv.len(), s.len());
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
                g.dtod(bc, 0, conv, 0, conv.len())?;
                g.dtod(bs, 0, s, 0, s.len())?;
            }
        }
        self.pos = self.saved_pos;
        if let Some((r, sr)) = &self.ple_ring { g.dtod(sr, 0, r, 0, r.len())?; }
        if let Some(q) = &self.qsa { g.dtod(&q.saved, 0, &q.ring, 0, q.ring.len())?; }
        self.ple_prev = self.saved_ple_prev;
        // The draft head's entries before the saved position are still the ones its prompt fill wrote (see `mtp_synced`).
        if let Some(h) = self.mtp.as_mut() {
            g.dtod(&h.saved_x, 0, &self.x, 0, 4 * self.m.cfg.n_embd)?;
            h.pos = self.pos.saturating_sub(1);
            h.pend.clear();
        }
        if let Some(n) = self.next.as_mut() { n.restore_state()?; }
        Ok(())
    }

    /// Whether the draft head holds an entry for each token after the first (it is one entry behind the main model):
    /// prefill then extends it with the prompt, so its drafts see the whole context.
    fn mtp_synced(&self) -> bool { self.mtp.as_ref().is_some_and(|h| h.pend.is_empty() && h.pos + 1 == self.pos.max(1)) }

    /// Run the forward of `nt` tokens as a recorded graph (graph key `nt` and the attention regime), recorded on first use.
    fn run(&self, nt: usize) -> anyhow::Result<()> {
        let gq = gq_min(self.pos, nt);
        self.gpu.graph(nt as u64 | (gq as u32 as u64) << 8, &mut || self.forward(nt, gq, None))
    }

    /// Enqueue (or record into a graph) the forward of `nt` tokens; per-token values come from `step_buf`. `from`:
    /// resume at that layer's MoE combine (see `finish`).
    fn forward(&self, nt: usize, gq: i32, from: Option<usize>) -> anyhow::Result<()> {
        let c = &self.m.cfg;
        let (g, eps, ne, step) = (&self.gpu, c.eps, c.n_embd, self.step_buf.addr());
        let p = |n: &str| self.bufs[n].addr();
        let x = self.x.addr();
        let mut res = 0;
        for (i, l) in self.layers.iter().enumerate().skip(from.map_or(0, |il| il - self.first)) {
            let il = self.first + i;
            res = self.layer(il, l, x, step, self.worker.as_ref(), nt, res, gq, from == Some(il))?;
        }
        if self.next.is_some() { return if res != 0 { g.add(x, res, nt * ne) } else { Ok(()) }; }
        match (&self.output_norm, &self.output_hc) {
            (Some(on), _) => g.rms_norm_q(x, ne, on.ptr, p("h"), ne, ne, nt, eps, p("hq"), res)?,
            (None, Some(hr)) => self.hc_read(hr, x, nt, &self.hc_at())?,
            _ => unreachable!(),
        }
        g.gemvq(self.output.as_ref().unwrap(), p("h"), ne, p("hq"), self.logits.addr(), c.n_vocab, nt)
    }

    /// Per-stream RMSNorm of nt stacks (rows of hc * n_embd) at `x` with the hc * n_embd gamma `w` into `out`.
    fn gnorm(&self, x: u64, w: u64, out: u64, nt: usize) -> anyhow::Result<()> {
        let c = &self.m.cfg;
        self.gpu.rms_norm_streams(x, w, out, c.n_embd, c.hc_n, nt, c.eps)
    }

    fn hc_at(&self) -> HcAt {
        let p = |n: &str| self.bufs.get(n).map_or(0, |b| b.addr());
        HcAt { xn: p("xn"), lo: p("lo"), hg: p("hg"), inj: p("inj"), h: p("h"), hq: p("hq"), emb: p("emb"), pk: p("pk"), pq: p("pq"), pg: p("pg"), pn: p("pn"), pv: p("pv"), gemm: false }
    }

    /// y[t] = W x[t] for t < nt: GEMV, or a dense GEMM for `a.gemm`.
    fn hmv(&self, a: &HcAt, w: &GpuTensor, x: u64, xs: usize, y: u64, ys: usize, nt: usize) -> anyhow::Result<()> {
        if a.gemm { self.gpu.gemm_dense(w, nt, x, xs, y, ys, 0) } else { self.mv(w, x, xs, y, ys, nt) }
    }

    /// Hyper-connection read of the stacks at `r` (see `Model::hc_read`): the mixed vector into `a.h` (and `a.hq`),
    /// the injection weights into `a.inj`.
    fn hc_read(&self, hr: &HcRead, r: u64, nt: usize, a: &HcAt) -> anyhow::Result<()> {
        let (c, g) = (&self.m.cfg, &self.gpu);
        let (ne, hc, d) = (c.n_embd, c.hc_n, c.hc_n * c.n_embd);
        self.gnorm(r, hr.norm.ptr, a.xn, nt)?;
        match &hr.inject {
            Some(inj) if a.gemm => {
                g.gemm_dense(&hr.down, nt, a.xn, d, a.lo, c.hc_rank, 0)?;
                g.gemm_dense(inj, nt, a.xn, d, a.inj, hc, 0)?;
            }
            Some(inj) => g.gemvq2(&hr.down, inj, a.xn, d, 0, a.lo, a.inj, nt)?,
            None => self.hmv(a, &hr.down, a.xn, d, a.lo, c.hc_rank, nt)?,
        }
        g.hc_silu(a.lo, 1.0 / hc as f32, nt * c.hc_rank)?;
        self.hmv(a, &hr.up, a.lo, c.hc_rank, a.hg, d, nt)?;
        g.hc_mix(a.xn, a.hg, a.h, ne, hc, nt)?;
        if a.hq != 0 { g.quant_xq(a.h, ne, ne, nt, a.hq)?; }
        Ok(())
    }

    /// The PLE block (see `Model::ple`) on the stacks at `x`, PLE input in `a.emb`.
    fn ple(&self, pl: &Ple, x: u64, step: u64, nt: usize, a: &HcAt) -> anyhow::Result<()> {
        let (c, g) = (&self.m.cfg, &self.gpu);
        let (ne, hc, d) = (c.n_embd, c.hc_n, c.hc_n * c.n_embd);
        self.hmv(a, &pl.key, a.emb, ne, a.xn, d, nt)?;
        self.gnorm(a.xn, pl.norm_key.ptr, a.pk, nt)?;
        self.gnorm(x, pl.norm_query.ptr, a.pq, nt)?;
        self.hmv(a, &pl.value, a.emb, ne, a.pv, ne, nt)?;
        g.ple_gate(a.pk, a.pq, a.pv, a.pg, ne, hc, nt)?;
        self.gnorm(a.pg, pl.norm_conv.ptr, a.pn, nt)?;
        g.ple_conv(x, a.pg, a.pn, pl.conv.ptr, self.ple_ring.as_ref().unwrap().0.addr(), step, d, nt)
    }

    /// y[t] = W x[t] for t < nt (strides in floats).
    fn mv(&self, w: &GpuTensor, x: u64, xs: usize, y: u64, ys: usize, nt: usize) -> anyhow::Result<()> {
        if nt == 1 { self.gpu.gemv(w, x, y) } else { self.gpu.gemvn(w, x, xs, y, ys, nt) }
    }

    /// Last hidden state (before output_norm) of the last `step`.
    pub fn hidden(&self) -> anyhow::Result<Vec<f32>> { let t = self.tail(); t.gpu.get::<f32>(&t.x, 0, self.m.cfg.n_embd) }

    /// MTP draft: logits for the token after `token`, from hidden state `row` of the last `step` or `verify` (the
    /// position that predicted `token`). Mirrors `Model::mtp_step`; the draft head keeps its own position and KV cache.
    pub fn mtp_draft(&mut self, token: u32, row: usize) -> anyhow::Result<Vec<f32>> {
        self.mtp_run(&[token], self.row_ptr(row), true)?;
        Ok(self.gpu.get::<f32>(&self.logits, 0, self.m.cfg.n_vocab)?)
    }

    /// Speculative decode with MTP. `t` is the next token (picked from the last logits), predicted by hidden row `row`.
    /// Drafts `k` tokens after `t` (each the draft head's argmax after the previous one), verifies [t, drafts..] in one
    /// forward and keeps the drafts up to the first one the main model rejects (`rollback`). Returns the accepted
    /// tokens, the token `smp` picks after the last one and its hidden row. Greedy: a draft is kept when it is the
    /// argmax, and the output equals plain `step` decoding (verify is bit-identical to steps). Sampling: a draft is
    /// kept with its probability (`Sampler::accept`), so tokens have the distribution of plain sampling. The draft head entries of accepted drafts wait for the next call's first draft pass,
    /// which runs them with `t` in one batch (their main model rows 0..n precede `t`'s row n).
    pub fn spec_step(&mut self, t: u32, row: usize, smp: &mut Sampler) -> anyhow::Result<(Vec<u32>, u32, usize)> {
        let k = self.k;
        if self.pos + k + 1 > self.ctx { return Ok((vec![t], smp.pick(&self.step(t)?)? as u32, 0)); }
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
            hid = self.mtp.as_ref().unwrap().x.addr() + 4 * ((if i == 0 { nf } else { 1 } - 1) * self.m.cfg.n_embd) as u64;
            toks.push(d);
        }
        let start = self.pos;
        let (n, next) = if smp.temp <= 0.0 {
            self.verify_greedy(&toks, |am| (0..k).take_while(|&i| am[i] == toks[i + 1]).count())?
        } else {
            let l = self.verify(&toks)?;
            let mut n = 0;
            loop {
                if n == k { break (n, smp.pick(&l[n])? as u32); }
                match smp.accept(&l[n], toks[n + 1] as usize)? { None => n += 1, Some(x) => break (n, x as u32) }
            }
        };
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
    fn row_ptr(&self, row: usize) -> u64 { self.x.addr() + 4 * (row * self.m.cfg.n_embd) as u64 }

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
        let gq = gq_min(h.pos, nt);
        let key = { let mut s = std::hash::DefaultHasher::new(); (hid, nt, head, gq).hash(&mut s); s.finish() | 1 << 63 };
        g.graph(key, &mut || self.mtp_forward(hid, nt, head, gq, false))?;
        let mut d = if head { g.get::<i32>(&self.amax, 0, 1)?[0] as u32 } else { g.sync()?; 0 };
        if self.finish(nt, gq, Some((hid, head)))? && head { d = self.gpu.get::<i32>(&self.amax, 0, 1)?[0] as u32; }
        let h = self.mtp.as_mut().unwrap();
        h.pos += nt;
        if let Some(cache) = self.cache.as_mut() {
            cache.record(c.n_layer, &h.worker.routing(c.n_layer).iter().map(|x| x.0).collect::<Vec<_>>());
        }
        Ok(d)
    }

    /// After a forward and its wait: while a MoE combine gave up waiting for the CPU worker (which may be reading
    /// experts from a cold drive; one GPU submission must not outlast the driver's job timeout), wait for the worker
    /// and run the rest of the forward from that combine. `mtp`: the draft head's forward (`mtp_run`'s hid, head)
    /// instead of the model's. Returns whether anything ran again.
    fn finish(&self, nt: usize, gq: i32, mtp: Option<(u64, bool)>) -> anyhow::Result<bool> {
        let mut again = false;
        let mut last = None;
        while self.gpu.stalled() {
            let w = match mtp { Some(_) => &self.mtp.as_ref().unwrap().worker, None => self.worker.as_ref().unwrap() };
            let il = w.stalled_layer().context("a MoE combine gave up waiting, but no layer says so")?;
            // The worker had answered the last time: the GPU does not see the mailbox.
            ensure!(last != Some(il), "moe_combine: the GPU does not see the CPU expert worker's answers. On an AMD RX 9000 \
                (RDNA4) GPU set radv_device_coherent_memory=true in the environment");
            last = Some(il);
            w.wait_done(il)?;
            match mtp {
                Some((hid, head)) => self.mtp_forward(hid, nt, head, gq, true)?,
                None => self.forward(nt, gq, Some(il))?,
            }
            self.gpu.sync()?;
            self.resumed.set(self.resumed.get() + 1);
            again = true;
        }
        Ok(again)
    }

    /// Enqueue the draft head (see `mtp_run`).
    fn mtp_forward(&self, hid: u64, nt: usize, head: bool, gq: i32, resume: bool) -> anyhow::Result<()> {
        let (c, g, h) = (&self.m.cfg, &self.gpu, self.mtp.as_ref().unwrap());
        let (eps, ne, hp) = (c.eps, c.n_embd, self.bufs["h"].addr());
        let (eh, x) = (h.eh.addr(), h.x.addr());
        if !resume {
            g.rms_norm(h.emb.addr(), ne, h.enorm.ptr, eh, 2 * ne, ne, nt, eps)?;
            g.rms_norm(hid, ne, h.hnorm.ptr, eh + 4 * ne as u64, 2 * ne, ne, nt, eps)?;
            self.mv(&h.eh_proj, eh, 2 * ne, x, ne, nt)?;
        }
        let res = self.layer(c.n_layer, &h.layer, x, h.step_buf.addr(), Some(&h.worker), nt, 0, gq, resume)?;
        if res != 0 { g.add(x, res, nt * ne)?; }
        if head {
            g.rms_norm(x + 4 * ((nt - 1) * ne) as u64, 0, h.head_norm.ptr, hp, 0, ne, 1, eps)?;
            g.gemv(self.output.as_ref().unwrap(), hp, self.logits.addr())?;
            g.argmax_rows(self.logits.addr(), c.n_vocab, 1, self.amax.addr())?;
        }
        Ok(())
    }

    /// One decoder block on `nt` tokens at `x` (rows of n_embd); `step` points at [position, worker seq] per token.
    /// One layer on x, after adding `res` (the previous layer's pending FFN output, or 0) to it. Returns this
    /// layer's pending FFN output (a dense FFN's, added by the next norm) or 0. `resume`: only the MoE down GEMV and
    /// combine onward, after a combine that gave up waiting for the CPU worker (see `finish`).
    fn layer(&self, il: usize, l: &Layer, x: u64, step: u64, worker: Option<&CpuWorker>, nt: usize, res: u64, gq: i32, resume: bool) -> anyhow::Result<u64> {
        let c = &self.m.cfg;
        let (g, eps, ne) = (&self.gpu, c.eps, c.n_embd);
        let p = |n: &str| self.bufs[n].addr();
        // Inputs from "h" are quantized once ("hq", after each norm) instead of once per GEMV block.
        let mv = |w: &GpuTensor, x: &str, xs: usize, y: &str, ys: usize|
            if x == "h" { g.gemvq(w, p(x), xs, p("hq"), p(y), ys, nt) } else { self.mv(w, p(x), xs, p(y), ys, nt) };
        let mv2 = |a: &GpuTensor, b: &GpuTensor, ya: &str, yb: &str| g.gemvq2(a, b, p("h"), ne, p("hq"), p(ya), p(yb), nt);
        // The SwiGLU for a down GEMV of type `ty` (float weights read g itself). Returns the xq for the GEMV.
        let sw = |ty: GgmlType, ga: &str, ua: &str, n: usize, rows: usize, q: &str| -> anyhow::Result<u64> {
            if matches!(ty, GgmlType::F32 | GgmlType::BF16) { g.swiglu(p(ga), p(ua), rows * n)?; return Ok(0); }
            g.swiglu_q(p(ga), p(ua), n, rows, p(q))
        };
        let (dk, nk, nv) = (c.ssm_d_state, c.ssm_n_group, c.ssm_dt_rank);
        let (dv, channels) = (c.ssm_dv(), c.ssm_d_inner + 2 * nk * dk);
        let (hd, nh, nkv) = (c.head_dim, c.n_head, c.n_head_kv);
        let kv = nkv * hd;
        if !resume {
            let ha = self.hc_at();
            if let Some(pl) = &l.ple { self.ple(pl, x, step, nt, &ha)?; }
            match (&l.attn_norm, &l.hc) {
                (Some(an), _) => g.rms_norm_q(x, ne, an.ptr, p("h"), ne, ne, nt, eps, p("hq"), res)?,
                (None, Some(hc)) => self.hc_read(&hc[0], x, nt, &ha)?,
                _ => unreachable!(),
            }
            match &l.mixer {
                Mixer::Delta { qkv, gate, beta, alpha, dt_bias, a, conv_w, norm, out, conv, s, snap } => {
                    let (sc, ss, ns) = snap.as_ref().map_or((0, 0, 0), |(a, b)| (a.addr(), b.addr(), self.k));
                    g.batch(&mut || {
                        mv2(qkv, gate, "qkv", "z")?;
                        mv2(beta, alpha, "beta", "g")
                    })?;
                    // One token with 128-wide heads: conv_silu also does the q and k l2 norms.
                    let nn = if nt == 1 && dk == 128 { 2 * nk } else { 0 };
                    if nt == 1 {
                        g.conv_silu(p("qkv"), conv.addr(), conv_w.ptr, p("y"), channels, c.ssm_conv, nn, eps)?;
                    } else {
                        g.conv_seq(p("qkv"), conv.addr(), conv_w.ptr, p("y"), channels, c.ssm_conv, nt, sc, ns)?;
                    }
                    if nn == 0 { g.l2_norm(p("y"), dk, 2 * nk, channels, nt * 2 * nk, eps)?; }
                    g.deltanet(p("y"), s.addr(), p("beta"), p("g"), dt_bias.ptr, a.ptr, p("o"), nk, nv, c.k_interleave, dk, dv, 1.0 / (dk as f32).sqrt(), nt, channels, ss, ns)?;
                    let oq = if l.hc.is_some() { g.norm_gate_sig(p("o"), norm.ptr, p("z"), dv, nt * nv, eps, p("oq"))? }
                             else { g.norm_gate_q(p("o"), norm.ptr, p("z"), dv, nt * nv, eps, p("oq"))? };
                    if oq != 0 { g.gemvq(out, p("o"), c.ssm_d_inner, oq, p("mix"), ne, nt)?; } else { mv(out, "o", c.ssm_d_inner, "mix", ne)?; }
                }
                Mixer::Attn { q, k, v, qn, kn, o, kc, vc, qsa } => {
                    let qs = if c.q_gate { 2 * hd } else { hd };
                    g.batch(&mut || { mv2(q, k, "qg", "kraw")?; mv(v, "h", ne, "v", kv) })?;
                    g.batch(&mut || g.rms_norm_rope_qk(p("qg"), qs, qn.ptr, p("q"), nt * nh, nh, p("kraw"), kn.ptr, p("k"), nt * nkv, nkv,
                                                       hd, eps, c.n_rot, step, c.rope_base, p("v"), kc.addr(), vc.addr()))?;
                    let (sel, sst) = match (qsa, &self.qsa) {
                        (Some(a), Some(b)) => {
                            mv2(&a.qp, &a.kp, "iq", "ik")?;
                            g.qsa_pool(p("ik"), a.ring, a.pool.addr(), a.kn.ptr, step, nt, c.n_rot, c.rope_base, eps)?;
                            g.rms_norm(p("iq"), QSA_D, a.qn.ptr, p("iqn"), QSA_D, QSA_D, nt * QSA_H, eps)?;
                            g.rope_neox(p("iqn"), QSA_D, nt * QSA_H, QSA_H, c.n_rot, step, c.rope_base)?;
                            g.qsa_select(p("iqn"), a.pool.addr(), b.score.addr(), b.sel.addr(), 0, step, 0, nt, b.nbs, b.top, b.sst, 0)?;
                            (b.sel.addr(), b.sst)
                        }
                        _ => (0, 0),
                    };
                    let aq = g.attn_decode_q(p("q"), kc.addr(), vc.addr(), if c.q_gate { p("qg") } else { 0 }, p("att"), p("attp"), step, nh, nkv, hd,
                                             1.0 / (hd as f32).sqrt(), gq, nt, p("attq"), sel, sst)?;
                    if aq != 0 { g.gemvq(o, p("att"), nh * hd, aq, p("mix"), ne, nt)?; } else { mv(o, "att", nh * hd, "mix", ne)?; }
                }
            }
            match (&l.post_norm, &l.hc) {
                (Some(pn), _) => g.rms_norm_q(x, ne, pn.ptr, p("h"), ne, ne, nt, eps, p("hq"), p("mix"))?,
                (None, Some(hc)) => {
                    g.hc_write(x, p("mix"), p("inj"), ne, c.hc_n, nt, false)?;
                    self.hc_read(&hc[1], x, nt, &ha)?;
                }
                _ => unreachable!(),
            }
        }
        {
            match &l.ffn {
                Ffn::Dense { gate, up, down } => {
                    debug_assert!(!resume);
                    g.batch(&mut || mv2(gate, up, "ge", "ue"))?;
                    let q = sw(down.ty, "ge", "ue", down.cols, nt, "ffq")?;
                    g.gemvq(down, p("ge"), down.cols, q, p("ye"), ne, nt)?;
                    return Ok(p("ye"));
                }
                Ffn::Moe { router, shared } => {
                    let k = c.n_expert_used;
                    let kt = nt * k;
                    let ids = self.ids.addr();
                    let cache = self.cache.as_ref().unwrap();
                    let lx = cache.layer(il).unwrap();
                    let (base, stride) = cache.base(lx);
                    let [mg, mu, md] = lx.mats;
                    let slots = self.slots.addr();
                    let (mut sq, mut dq) = (0, 0);
                    let w = worker.unwrap();
                    let mb = w.dev + (4 * il * w.layout.stride) as u64;
                    // Resuming: what the skipped part left (the swiglu_q of Vulkan, the only backend that resumes).
                    if resume { dq = if matches!(md.ty, GgmlType::F32 | GgmlType::BF16) { 0 } else { p("ffq") }; }
                    // The shared expert's GEMVs and swiglu share batches with the routed ones.
                    if !resume {
                    g.batch(&mut || {
                        if let Some(s) = shared {
                            mv2(router, &s.gate_inp, "rl", "sg")?;
                            mv2(&s.gate, &s.up, "shg", "shu")?;
                        } else { mv(router, "h", ne, "rl", c.n_expert)?; }
                        Ok(())
                    })?;
                    g.moe_route(p("rl"), c.n_expert, k, c.expert_weights_scale, lx.table_dev.addr(), ids, p("wts"), slots, p("wh"), p("h"), mb, ne, w.layout.h, step, nt)?;
                    g.batch(&mut || {
                        if mg.ty == mu.ty && (mg.rows, mg.cols) == (mu.rows, mu.cols) {
                            let pair = (mu.off as i64 - mg.off as i64, p("ue") as i64 - p("ge") as i64);
                            g.moe_gemv(mg.ty, mg.rows, mg.cols, base + mg.off as u64, stride, slots, kt, p("h"), ne, k, p("ge"), Some(pair), p("hq"))?;
                        } else {
                            g.moe_gemv(mg.ty, mg.rows, mg.cols, base + mg.off as u64, stride, slots, kt, p("h"), ne, k, p("ge"), None, p("hq"))?;
                            g.moe_gemv(mu.ty, mu.rows, mu.cols, base + mu.off as u64, stride, slots, kt, p("h"), ne, k, p("ue"), None, p("hq"))?;
                        }
                        if let Some(s) = shared { sq = sw(s.down.ty, "shg", "shu", c.n_ff_shexp, nt, "shq")?; }
                        Ok(())
                    })?;
                    g.batch(&mut || {
                        dq = sw(md.ty, "ge", "ue", c.n_ff_exp, kt, "ffq")?;
                        if let Some(s) = shared { g.gemvq(&s.down, p("shg"), c.n_ff_shexp, sq, p("shy"), ne, nt)?; }
                        Ok(())
                    })?;
                    }
                    let (sh, sg) = if shared.is_some() { (p("shy"), p("sg")) } else { (0, 0) };
                    // Hyper-connections: the FFN output collects in "mo" (zero) and goes to the streams from there.
                    let fx = if l.hc.is_some() { p("mo") } else { x };
                    g.moe_down_combine(md.ty, md.rows, md.cols, base + md.off as u64, stride, slots, k, p("ge"), p("ye"), dq,
                                       fx, p("wh"), sh, sg, mb, w.layout.y, step, nt)?;
                    if l.hc.is_some() { g.hc_write(x, p("mo"), p("inj"), ne, c.hc_n, nt, true)?; }
                }
            }
        }
        Ok(0)
    }

    /// Advance the position and update expert statistics and placement after a token.
    fn after_forward(&mut self, nt: usize) -> anyhow::Result<()> {
        self.pos += nt;
        if let (Some(cache), Some(w)) = (self.cache.as_mut(), self.worker.as_ref()) {
            for il in self.first..self.first + self.layers.len() {
                if cache.layer(il).is_none() { continue; }
                let r = w.routing(il);
                let miss = r.iter().filter(|x| x.2 < 0).count();
                self.misses += miss;
                self.hits += r.len() - miss;
                cache.record(il, &r.iter().map(|x| x.0).collect::<Vec<_>>());
            }
            self.cpu_time = w.busy();
            if self.adapt_every > 0 && (self.pos - nt) / self.adapt_every < self.pos / self.adapt_every { cache.adapt(self.m, &*self.gpu, 64)?; }
        }
        if let Some(n) = self.next.as_mut() {
            n.adapt_every = self.adapt_every;
            n.after_forward(nt)?;
        }
        Ok(())
    }
}
