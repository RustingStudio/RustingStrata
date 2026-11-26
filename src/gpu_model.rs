//! GPU decode forward for the architectures in `config::arch`. Routed experts resident in the expert cache run on the GPU; the
//! rest run on the CPU (`cpu_moe`) while the GPU computes hits and the shared expert. Mirrors `model.rs`.

use crate::config::LayerKind;
use crate::cpu_worker::{CpuWorker, TMAX};

/// Slots that `step_rows` decodes in one forward (its graphs are keyed by up to this many slots); `TMAX` is the
/// wider limit of a multi-token verify.
pub const STEP_ROWS: usize = 4;
/// Positions a slot's attention KV starts with when it grows on demand (`GpuModel::kv_need`).
const KV_START: usize = 512;
/// The start when the slots do not all fit at `KV_START`.
const KV_MIN: usize = 256;
use crate::backend::{ATT_GQA_MIN, ATT_SPLITS, Backend, Buf, GpuTensor};
use crate::expert_cache::ExpertCache;
use crate::model::Model;
use crate::ops::Sampler;
use crate::quant::GgmlType;
use anyhow::{bail, ensure, Context};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::Duration;

/// Hits and misses of one layer's routing read from the mailbox (a slot < 0 is an expert on the CPU). R1 prints
/// their sum per run; `record` and the hit rate use the same numbers.
fn tally(routing: &[(i32, f32, i32)]) -> (usize, usize) {
    let misses = routing.iter().filter(|x| x.2 < 0).count();
    (routing.len() - misses, misses)
}

/// The decode attention `gq_min` for `nt` tokens from position `pos`. The kernels pick the regime per token; when all
/// tokens fall on one side of ATT_GQA_MIN, i32::MAX (one head per block) or 0 (GQA) gives the same result and lets the
/// backend launch only the kernel that works.
fn gq_min(pos: usize, nt: usize) -> i32 {
    let m = ATT_GQA_MIN as usize;
    if pos + nt < m { i32::MAX } else if pos + 1 >= m { 0 } else { ATT_GQA_MIN }
}


mod prefill;

/// A tensor `GpuModel::load_part` never uploads: the PLE table stays on the CPU.
fn cpu_only(name: &str) -> bool { name == "per_layer_token_embd.weight" }

/// A tensor `GpuModel::load_part` converts on upload (`Backend::upload`): F16 to F32. Expert tensors (`*_exps.`) are
/// copied raw and keep their type.
fn f32_up(name: &str, ty: GgmlType) -> bool { ty == GgmlType::F16 && !name.contains("_exps.") && !name.starts_with("dflash.") }

/// The weight types a GPU load of `g` leaves for the kernels, in the order first seen: `cpu_only` tensors skipped,
/// `f32_up` tensors rewritten as `backend::upload_type` stores them, and tensors `load_part`'s gate rejects left out
/// (the load itself then fails). Used by the Vulkan shared-memory preflight; no model is loaded.
pub fn gpu_weight_types(g: &crate::gguf::Gguf) -> Vec<GgmlType> {
    let mut v: Vec<GgmlType> = vec![];
    for t in g.tensors.values() {
        if cpu_only(&t.name) { continue; }
        let Ok(ty) = GgmlType::from_u32(t.ty) else { continue };  // load_part rejects the tensor
        let ty = if f32_up(&t.name, ty) { crate::backend::upload_type(ty) } else { ty };
        if ty.gpu("vulkan") && !v.contains(&ty) { v.push(ty); }  // F16 has no kernels: load_part rejects it
    }
    v
}

/// Scratch rows for `hc_read` and `ple`: the decode buffers (`GpuModel::hc_at`) or the prefill slab, where `gemm`
/// runs their matrices as dense GEMMs over the chunk. `hq` 0: no int8 copy of h.
#[derive(Clone, Copy)]
struct HcAt { xn: u64, lo: u64, hg: u64, inj: u64, h: u64, hq: u64, emb: u64, pk: u64, pq: u64, pg: u64, pn: u64, pv: u64, gemm: bool }

enum Mixer {
    Delta {
        qkv: GpuTensor, gate: GpuTensor, beta: GpuTensor, alpha: GpuTensor, dt_bias: GpuTensor, a: GpuTensor,
        /// Conv and DeltaNet states per slot (`use_slot`).
        conv_w: GpuTensor, norm: GpuTensor, out: GpuTensor, conv: Vec<Buf>, s: Vec<Buf>,
        /// Conv and DeltaNet checkpointed snapshots (`sc`, `ss`) + saved verify inputs (`y_s`, `b_s`, `g_s`) for strided rollback.
        snap: Option<(Buf, Buf, Buf, Buf, Buf)>,
    },
    /// `kc`, `vc`: the KV cache per slot.
    Attn { q: GpuTensor, k: GpuTensor, v: GpuTensor, qn: GpuTensor, kn: GpuTensor, o: GpuTensor, kc: Vec<Buf>, vc: Vec<Buf>, qsa: Option<Qsa>, sink: Option<GpuTensor> },
}

/// QSA indexer of an attention layer (qwen4exp, see `Model::qsa`): projections and norms, the pooled keys (128 floats
/// per block of 4 positions) and this layer's 32 rows of `QsaBufs::ring`.
struct Qsa { kp: GpuTensor, qp: GpuTensor, kn: GpuTensor, qn: GpuTensor, pool: Buf, ring: u64 }

/// QSA scratch shared by the layers: raw indexer keys of each layer's last 32 positions (`QSA_RING` floats a layer)
/// and their copy at `save_state`, block scores (rows of `nbs`), decode cell lists (rows of `sst`), prefill masks
/// (rows of `mw` words).
struct QsaBufs { ring: Buf, saved: Buf, score: Buf, sel: Buf, mask: Buf, top: usize, nbs: usize, sst: usize, mw: usize }

/// Positions a QSA pool block covers, the indexer head size and heads (the kernels' QSA_D and QSA_H).
const QSA_POOL: usize = 4;
const QSA_D: usize = 128;
const QSA_H: usize = 4;
const QSA_RING: usize = 32 * QSA_D;
/// Query rows per `qsa_select` launch in prefill (score rows).
const QSA_ROWS: usize = 64;

struct Shared { gate_inp: Option<GpuTensor>, gate: GpuTensor, up: GpuTensor, down: GpuTensor }

enum Ffn {
    Dense { gate: GpuTensor, up: GpuTensor, down: GpuTensor },
    Moe { router: GpuTensor, shared: Option<Shared> },
}

/// qwen4exp hyper-connection read weights (`{p}norm/down/up/inject.weight`, see `Model::hc_read`).
struct HcRead { norm: GpuTensor, down: GpuTensor, up: GpuTensor, inject: Option<GpuTensor> }

/// qwen4exp PLE block weights (see `Model::ple`); `conv` as f32 [channel][tap].
struct Ple { key: GpuTensor, value: GpuTensor, norm_key: GpuTensor, norm_query: GpuTensor, norm_conv: GpuTensor, conv: GpuTensor }

/// `attn_norm` / `post_norm` for a plain residual; `hc` (attention and FFN reads) for hyper-connections.
/// Buffers of `step_batch`: positions (2 ints a row), the KV cache addresses of each row's slot (2 a row, per
/// layer) for `rows` rows; logits and argmax for `cap` rows; attention partials for `cap` rows at `ATT_SPLITS`
/// splits (more rows fit at fewer splits).
/// `dense`: every layer attention and a dense FFN (the forward of `batch_rows`); else `prefill_chunk` in rows mode.
struct Batch { dense: bool, cap: usize, rows: usize, step: Buf, kvt: Buf, logits: Buf, amax: Buf, attp: Buf }

/// SenseNova-Loop manifold-constrained hyper-connection (mHC) router (`mhc_attn` or `mhc_ffn`).
#[derive(Clone)]
struct MhcRouter { weight: GpuTensor, buf: Buf, out_dim: usize }

struct Layer { attn_norm: Option<GpuTensor>, post_norm: Option<GpuTensor>, hc: Option<[HcRead; 2]>, mhc: Option<(u8, MhcRouter, MhcRouter)>, ple: Option<Ple>, mixer: Mixer, ffn: Ffn }

/// MTP draft head: the `nextn` block (layer n_layer) with its own KV cache and CPU worker.
/// Buffers for a copy of the DeltaNet (conv, s) states of `layers` (all or none).
/// `host`: in host memory where the backend has it (`Backend::host_buf`), as for the saved state of each of several
/// slots, which is copied only at a prompt's start and end; the VRAM goes to the slots' live state and the experts.
fn alloc_saved(g: &dyn Backend, layers: &[Layer], host: bool) -> anyhow::Result<Vec<(Buf, Buf)>> {
    let a = |n: usize| -> anyhow::Result<Buf> { match if host { g.host_buf(n)? } else { None } { Some(b) => Ok(b), None => g.alloc(n) } };
    layers.iter().filter_map(|l| match &l.mixer { Mixer::Delta { conv, s, .. } => Some((conv[0].len(), s[0].len())), _ => None })
        .map(|(c, s)| Ok((a(c)?, a(s)?))).collect()
}

/// A saved conversation state in host memory: its tokens, the PLE tokens before it, the `slot` buffers and the KV
/// rows of its positions in each attention layer.
struct Parked { ids: Vec<u32>, owner: u64, parts: Vec<PartCopy> }

/// One part's share of a parked state: its history ring position, saved slot buffers and KV rows.
struct PartCopy { ple_prev: [i64; 2], slot: Vec<Vec<u8>>, kv: Vec<Vec<u8>> }

impl Parked {
    fn bytes(&self) -> usize { self.parts.iter().flat_map(|p| p.slot.iter().chain(&p.kv)).map(Vec::len).sum() }
}

/// Standalone EAGLE3 1-layer GPU weights and scratch buffers.
struct Eagle3Gpu {
    cfg: crate::config::Eagle3Config,
    fc: GpuTensor,
    output: GpuTensor,
    /// `[TMAX, 3 * n_embd]` concatenated hidden states extracted before `extract_layers`.
    feat: Buf,
    /// `[TMAX, n_embd]` output of `fc` for the last target forward.
    cur_g: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    att: Buf,
    attp: Buf,
    ge: Buf,
    ue: Buf,
}

struct DFlashConv {
    base: GpuTensor,
    proj: GpuTensor,
}

struct DFlash2Selector {
    hidden_proj: GpuTensor,
    h_proj: Buf,
    pred_raw: Vec<u8>,
    pred_ty: GgmlType,
    succ_raw: Vec<u8>,
    succ_ty: GgmlType,
    rank: usize,
    top_k: usize,
}

impl DFlash2Selector {
    fn dequant_codebook_row(raw: &[u8], ty: GgmlType, tok: u32, rank: usize, dst: &mut [f32]) {
        let rb = ty.row_bytes(rank as u64);
        let off = tok as usize * rb;
        if off + rb <= raw.len() {
            crate::quant::dequant_row(ty, &raw[off..off + rb], dst);
        } else {
            dst.fill(0.0);
        }
    }

    fn select_path(
        &self,
        anchor_tok: u32,
        k_eff: usize,
        logits: &[f32],
        row_len: usize,
        short_ids: Option<&[u32]>,
        h_proj: &[f32],
        out_mult: f32,
        smp: &mut Sampler,
    ) -> anyhow::Result<(Vec<u32>, Vec<Vec<(usize, f32)>>)> {
        let k_top = self.top_k.min(row_len).max(1);
        let rank = self.rank;
        let mut pred = anchor_tok;
        let mut toks = Vec::with_capacity(k_eff);
        let mut q_dists = Vec::new();
        let mut pred_vec = vec![0f32; rank];
        let mut succ_vec = vec![0f32; rank];
        let mut q_vec = vec![0f32; rank];
        let mut row_cands: Vec<(usize, f32)> = Vec::with_capacity(row_len);
        let mut scores = vec![0f32; k_top];

        for pos in 0..k_eff {
            let row = &logits[pos * row_len..(pos + 1) * row_len];
            row_cands.clear();
            for (idx, &v) in row.iter().enumerate() {
                row_cands.push((idx, v));
            }
            if row_cands.len() > k_top {
                row_cands.select_nth_unstable_by(k_top - 1, |a, b| b.1.total_cmp(&a.1));
                row_cands.truncate(k_top);
            }
            Self::dequant_codebook_row(&self.pred_raw, self.pred_ty, pred, rank, &mut pred_vec);
            let hp = &h_proj[pos * rank..(pos + 1) * rank];
            for r in 0..rank {
                q_vec[r] = pred_vec[r] * hp[r];
            }
            let mut best_i = 0usize;
            let mut best_s = f32::NEG_INFINITY;
            for (j, &(idx, u)) in row_cands.iter().enumerate() {
                let tok_id = short_ids.map_or(idx as u32, |ids| ids[idx]);
                Self::dequant_codebook_row(&self.succ_raw, self.succ_ty, tok_id, rank, &mut succ_vec);
                let dot: f32 = q_vec.iter().zip(&succ_vec).map(|(&a, &b)| a * b).sum();
                let s = u * out_mult + dot;
                scores[j] = s;
                if s > best_s {
                    best_s = s;
                    best_i = j;
                }
            }
            if smp.temp <= 0.0 {
                let chosen_idx = row_cands[best_i].0;
                pred = short_ids.map_or(chosen_idx as u32, |ids| ids[chosen_idx]);
                toks.push(pred);
            } else {
                let (loc_i, q_local) = smp.draft_dist(&scores[..row_cands.len()])?;
                let chosen_idx = row_cands[loc_i as usize].0;
                pred = short_ids.map_or(chosen_idx as u32, |ids| ids[chosen_idx]);
                toks.push(pred);
                let q_mapped = q_local
                    .into_iter()
                    .map(|(j, p)| {
                        let idx = row_cands[j].0;
                        (short_ids.map_or(idx, |ids| ids[idx] as usize), p)
                    })
                    .collect();
                q_dists.push(q_mapped);
            }
        }
        Ok((toks, q_dists))
    }
}

/// One Transformer block of a standalone DFlash block-diffusion draft model.
struct DFlashLayer {
    attn_norm: GpuTensor,
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    qn: GpuTensor,
    kn: GpuTensor,
    o: GpuTensor,
    post_norm: GpuTensor,
    gate: GpuTensor,
    up: GpuTensor,
    down: GpuTensor,
    attn_conv: Option<DFlashConv>,
    mlp_conv: Option<DFlashConv>,
    kc: Vec<Buf>,
    vc: Vec<Buf>,
    is_swa: bool,
}

/// Standalone DFlash parallel block-diffusion draft model GPU weights, KV cache, and scratch.
struct DFlashGpu {
    cfg: crate::config::DFlashConfig,
    fc: GpuTensor,
    hidden_norm: GpuTensor,
    norm: GpuTensor,
    layers: Vec<DFlashLayer>,
    /// `[TMAX, n_target_layers * n_embd]` concatenated post-layer target hidden states.
    feat: Buf,
    /// hc stream gate for the mean collapse (all large => plain mean), `[hc_n * n_embd]`.
    dfg: Buf,
    /// hc-mean collapse output, `[TMAX, n_embd]`.
    dfm: Buf,
    /// `[TMAX, n_embd]` projected & normalized `target_hidden = hidden_norm(fc(feat))` from the last forward.
    th: Buf,
    x: Buf,
    h: Buf,
    h2: Buf,
    conv_dyn: Buf,
    q: Buf,
    k: Buf,
    v_ctx: Buf,
    k_ctx: Buf,
    att: Buf,
    attp: Buf,
    ge: Buf,
    ue: Buf,
    ctx_step: Buf,
    step_swa: Buf,
    step_full: Buf,
    selector: Option<DFlash2Selector>,
    /// Number of verified target tokens already committed into `layers[*].kc/vc`.
    pos: usize,
    /// Number of verified target rows (`0..=TMAX`) in `th` waiting to be committed to `layers[*].kc/vc`.
    pend_ctx: usize,
    short: Option<Shortlist>,
    short_oob: std::cell::Cell<usize>,
    use_short: std::cell::Cell<bool>,
    cur_block: std::cell::Cell<usize>,
    low_streak: std::cell::Cell<u8>,
}

struct Shortlist {
    buf: Buf,
    ids: Vec<u32>,
    in_set: Vec<bool>,
}

/// The draft file's metadata sizes the DFlash buffers and the kernels write its tensors' outputs through raw
/// addresses, so the two must agree (and fit the target's shared buffers) before anything is allocated.
pub(crate) fn dflash_check(m: &Model, dc: &crate::config::DFlashConfig) -> anyhow::Result<()> {
    let c = &m.cfg;
    let (ne, q, kv, hd, ff) = (dc.n_embd, dc.n_head * dc.head_dim, dc.n_head_kv * dc.head_dim, dc.head_dim, dc.n_ff);
    ensure!(dc.n_layer > 0 && ne == c.n_embd && hd > 0 && dc.n_head_kv > 0 && dc.n_head.is_multiple_of(dc.n_head_kv) && dc.n_head > 0
        && dc.n_rot <= hd && ff > 0 && (dc.mask_token_id as usize) < c.n_vocab,
        "draft model: bad DFlash metadata (layers {}, embd {ne} for a {} target, heads {}/{}x{hd}, rope {}, ff {ff}, mask token {})",
        dc.n_layer, c.n_embd, dc.n_head, dc.n_head_kv, dc.n_rot, dc.mask_token_id);
    let dim = |name: String, rows: usize, cols: usize| -> anyhow::Result<()> {
        let t = m.w.get(&name)?;
        ensure!(t.rows == rows && t.cols == cols, "draft model: {name} is {}x{}, its metadata needs {rows}x{cols}", t.rows, t.cols);
        Ok(())
    };
    dim("dflash.fc.weight".into(), ne, dc.target_layer_ids.len() * c.n_embd)?;
    for n in ["hidden_norm", "norm"] { dim(format!("dflash.{n}.weight"), 1, ne)?; }
    let (k, g) = (dc.conv_kernel_size, dc.conv_group_size);
    ensure!(k == 0 || (g > 0 && ne.is_multiple_of(g)), "draft model: conv group size {g} does not divide {ne}");
    for il in 0..dc.n_layer {
        let l = |s: &str, rows, cols| dim(format!("dflash.blk.{il}.{s}.weight"), rows, cols);
        for (s, rows, cols) in [("attn_norm", 1, ne), ("ffn_norm", 1, ne), ("attn_q_norm", 1, hd), ("attn_k_norm", 1, hd), ("attn_q", q, ne),
            ("attn_k", kv, ne), ("attn_v", kv, ne), ("attn_output", ne, q), ("ffn_gate", ff, ne), ("ffn_up", ff, ne), ("ffn_down", ne, ff)] {
            l(s, rows, cols)?;
        }
        if k > 0 {
            for s in ["attn_conv", "mlp_conv"] {
                l(&format!("{s}_proj"), 2 * k * (ne / g), ne)?;
                let b = m.w.get(&format!("dflash.blk.{il}.{s}_base.weight"))?;  // [2, K, H], indexed flat
                ensure!(b.rows * b.cols == 2 * k * ne, "draft model: layer {il} {s}_base has {} values, its metadata needs {}", b.rows * b.cols, 2 * k * ne);
            }
        }
    }
    if dc.selector_rank > 0 { dim("dflash.selector.hidden_proj.weight".into(), dc.selector_rank, ne)?; }
    Ok(())
}

fn build_shortlist(m: &Model, gpu: &dyn Backend, output: Option<&GpuTensor>) -> anyhow::Result<Option<Shortlist>> {
    if std::env::var_os("STRATA_MTP_FULL_VOCAB").is_some() { return Ok(None); }
    let c = &m.cfg;
    if c.n_vocab <= 65536 || !gpu.supports_gemv_rows() { return Ok(None); }
    let Some(out) = output else { return Ok(None); };
    if out.ty.float_x() { return Ok(None); }
    let Ok(crate::gguf::Value::Arr(arr)) = m.w.g.get("tokenizer.ggml.tokens") else { return Ok(None); };
    if arr.len() != c.n_vocab { return Ok(None); }

    let mut u2b = [0xFFFFu16; 512];
    for ch in (b'!'..=b'~').chain(0xA1..=0xAC).chain(0xAE..=0xFF) {
        u2b[ch as usize] = ch as u16;
    }
    let mut next_cp = 256usize;
    for b in 0..=255u8 {
        if u2b[b as usize] == 0xFFFF { u2b[next_cp] = b as u16; next_cp += 1; }
    }
    let decode_bpe = |s: &str| -> Vec<u8> {
        let mut raw = Vec::with_capacity(s.len());
        for ch in s.chars() {
            let cu = ch as usize;
            if cu < 512 && u2b[cu] != 0xFFFF { raw.push(u2b[cu] as u8); }
            else { return s.as_bytes().to_vec(); }
        }
        raw
    };

    let mut ids = Vec::with_capacity(c.n_vocab);
    let mut in_set = vec![false; c.n_vocab];
    for (i, v) in arr.iter().enumerate() {
        let Some(s) = v.as_str() else { return Ok(None); };
        let keep = i < 4096
            || s.starts_with("<|")
            || matches!(s, "<think>" | "</think>")
            || {
                let raw = decode_bpe(s);
                match std::str::from_utf8(&raw) {
                    Ok(txt) => txt.chars().all(|ch| (ch as u32) <= 0x03FF || (0x2000..=0x2BFF).contains(&(ch as u32))),
                    Err(_) => raw.len() <= 2,
                }
            };
        if keep {
            ids.push(i as u32);
            in_set[i] = true;
        }
    }
    if ids.is_empty() || ids.len() >= c.n_vocab * 4 / 5 { return Ok(None); }
    let gpu_ids: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
    let buf = gpu.zeros::<i32>(gpu_ids.len())?;
    gpu.put(&gpu_ids, &buf, 0)?;
    Ok(Some(Shortlist { buf, ids, in_set }))
}

struct MtpHead {
    layer: Layer, eh_proj: Option<GpuTensor>, enorm: GpuTensor, hnorm: GpuTensor,
    /// The head's norm, or with hyper-connections its read (`nextn.hc_head_*`). Rows of `eh` and `x` are then per
    /// stream: hc_n times as wide.
    head_norm: Option<GpuTensor>, hc_head: Option<HcRead>,
    worker: Option<CpuWorker>, step_buf: Buf, emb: Buf, eh: Buf, x: Buf, pos: usize,
    eagle: Option<Eagle3Gpu>,
    /// Row 0 of the main model's `x` (or `eagle.cur_g`) at `save_state`.
    saved_x: Buf,
    /// Accepted drafts of the last `spec_step` whose draft head entries (from the main model's hidden rows 0..) are
    /// still to be written; the next `spec_step` writes them in its first draft pass.
    pend: Vec<u32>,
    /// 12-step rolling window of draft acceptance bits (LSB = latest step), used for adaptive speculative depth.
    adapt_window: u16,
    /// Remaining 1-token bypass steps before probing EAGLE3/MTP again on partially-resident MoE runs.
    adapt_skip: u8,
    /// Sorted ASCII/Latin/Greek/Math/Code/Special vocab subset for fast MTP draft `lm_head` (`0.55 MB` index buffer).
    short: Option<Shortlist>,
    /// Count of out-of-shortlist tokens seen in the current sequence; >4 auto-disables shortlist cleanly.
    short_oob: std::cell::Cell<usize>,
    use_short: std::cell::Cell<bool>,
}

/// What a slot keeps while another slot is current (`use_slot`): the `GpuModel` fields of the same names.
#[derive(Default)]
struct Seq { pos: usize, rope: i32, saved: Vec<(Buf, Buf)>, saved_pos: usize, saved_ids: Vec<u32>, saved_owner: u64, sys: Option<Stash>, sys_owner: u64 }

/// See `GpuModel::sys`.
type Stash = (Vec<Vec<u8>>, usize, [i64; 2], Vec<u32>);

/// Maximum number of intermediate DeltaNet + conv state snapshots written during a multi-token verify pass.
/// Any rollback with `keep <= SNAP_MAX` restores directly via `dtod`; `keep > SNAP_MAX` restores `snap[SNAP_MAX - 1]`
/// and replays the remaining `keep - SNAP_MAX` accepted tokens in one batch pass.
pub const SNAP_MAX: usize = 16;
pub const SNAP_STRIDE: usize = 4;

pub struct GpuModel<'a> {
    m: &'a Model<'a>,
    /// `STRATA_KV_HOST` prefill: VRAM copies of one host layer's K and V cache (see `load_part`).
    kv_stage: Option<(Buf, Buf)>,
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
    /// `ple_prev` before the last forward and that forward's tokens, for `rollback`.
    ple_undo: ([i64; 2], Vec<u32>),
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
    /// [RoPE position, seq] for the current token when the RoPE position differs from the KV row (image prompts).
    rope_buf: Buf,
    /// RoPE position minus KV row for the next token (0 for text-only).
    pub rope_offset: i32,
    /// Set during an image token's `step_embed`: its M-RoPE positions [t, y, x, z] are in `mrope_buf`.
    mrope: bool,
    mrope_buf: Buf,
    /// DeltaNet (conv, s) copies per recurrent layer and the position, for `restore_state`.
    saved: Vec<(Buf, Buf)>,
    saved_pos: usize,
    /// The tokens behind the saved state when `prefill_cached` saved it (else empty).
    saved_ids: Vec<u32>,
    /// Host copy of the saved state at the end of the system prompt (`stash`): buffers, position, PLE tokens, the
    /// tokens behind it. A request whose history no longer extends the saved one (a rewritten or compacted
    /// conversation) restarts from it.
    sys: Option<Stash>,
    /// Saved states that a request with another history replaced, most recent first (`park`).
    parked: Vec<Parked>,
    /// Whose request `prefill_cached` serves (an API key's number): it reuses only states of the same owner, so one
    /// key cannot learn from the cached-token count or the response time what another key sent.
    pub owner: u64,
    /// The owners of the saved and the stashed state.
    saved_owner: u64,
    sys_owner: u64,
    /// Host memory for `parked`, in bytes (0 = keep none).
    pub cache_ram: usize,
    /// Optional disk path for persisting `parked` / `saved` conversation states across restarts (Warm Start).
    pub prompt_cache_path: Option<std::path::PathBuf>,
    ctx: usize,
    pub pos: usize,
    pub cache: Option<ExpertCache>,
    /// Swap experts every this many tokens (0 = fixed placement).
    pub adapt_every: usize,
    /// Experts one `adapt` may swap per pool (default 64; `STRATA_ADAPT_SWAPS`).
    pub adapt_swaps: usize,
    /// Serve policy: after a prefill, forget the prompt's short-term routing counts so decode adapts to the reply
    /// instead of the prompt (`STRATA_RESET_COUNTS`; `serve` turns it on).
    pub reset_counts_after_prefill: bool,
    pub hits: usize,
    pub misses: usize,
    /// Forwards this part ran (decode tokens): the base of the per-token stats `Runner::finish` prints.
    pub tokens: usize,
    /// `moe_route` calls that wrote the mailbox (a GPU mailbox write) and `routing()` calls that read it back.
    pub mailbox_writes: std::cell::Cell<usize>,
    pub mailbox_reads: std::cell::Cell<usize>,
    /// Last token's routing ids of each layer, the next token's prefetch prediction (`STRATA_MOE_PREFETCH=1`).
    pids: Option<Buf>,
    /// Time the CPU worker spent computing missed experts (overlaps GPU work).
    pub cpu_time: Duration,
    /// Forwards run again from a MoE combine that gave up waiting for the CPU worker (`finish`).
    pub resumed: std::cell::Cell<usize>,
    /// LoRA overlays already added to the current forward's residual state (one entry per layer).
    lora_applied: std::cell::RefCell<std::collections::HashSet<usize>>,
    /// MTP drafts (made, accepted).
    pub drafts: (usize, usize),
    /// Split model, all parts but the last: seconds from the start of this part's forward until its GPU is done, and
    /// for the copy of its hidden rows to the next part.
    pub split_time: (Duration, Duration),
    /// MTP drafts per speculative step (0 = no draft head). May be lowered after `load`, never raised.
    pub k: usize,
    /// DeltaNet snapshot slots allocated for speculative rollback (from MTP or `--spec-type`).
    nsnap: usize,
    /// Prompts shorter than this go through `step` instead of the batched prefill.
    pub short: usize,
    /// Prefill scratch (None: prompts go through `step`).
    pf: Option<prefill::Scratch>,
    /// Scratch of `step_batch` (`--parallel` above `STEP_ROWS` on a plain dense model).
    bt: Option<Batch>,
    mtp: Option<MtpHead>,
    /// Standalone DFlash parallel block-diffusion draft model (loaded via `--draft-model`).
    dflash: Option<DFlashGpu>,
    /// Standalone external causal LM draft model (e.g. `nvidia/Qwen3.5-4B-Qwen3.8-27B-Draft`, loaded via `--draft-model`).
    ext_draft: Option<Box<GpuModel<'a>>>,
    ext_draft_pend: Vec<u32>,
    mhc_ones: Option<Buf>,
    /// Gyro files (`config::Hadamard`): the signs of a hidden row and of an expert's inner row, as floats.
    hada: Option<[Buf; 2]>,
    graph_tag: u64,
    /// Speculative N-gram / lookup cache drafter (standalone or combined with `mtp`/`dflash` via `hybrid_spec_step`).
    pub ngram: Option<crate::ngram::NgramDrafter>,
    /// The layers of this part are `first..first + layers.len()`.
    first: usize,
    /// The part on the next device of a model split by layers (`load_split`): this part's hidden state goes to it
    /// through host memory, and the last part computes the logits. Public calls go to the first part.
    next: Option<Box<GpuModel<'a>>>,
    /// The current slot (`load_slots`), whose state the other methods use, and the others' (entry `cur` is stale).
    cur: usize,
    seqs: Vec<Seq>,
    /// Positions each slot's attention KV holds, when the slots' KV grows on demand (`kv_need`); empty when every
    /// KV cache holds `ctx`.
    kv_cap: Vec<usize>,
    /// Positions a growing slot starts with and shrinks back to (`KV_START` or `KV_MIN`).
    kv0: usize,
    /// The (slot, position) of each token of the `step_rows` forward being run (empty otherwise).
    rows: Vec<(usize, usize)>,
    /// Tokens `step_rows` ran, for the placement cadence.
    row_tokens: usize,
    /// Share of a batch step's missed experts that the CPU runs; the rest are staged over PCIe (`prefill_chunk`).
    cpu_frac: f64,
}

/// Layer shares of `--split`: one positive number per device, such as `60,40`.
pub fn split_shares(spec: &str, n: usize) -> anyhow::Result<Vec<f64>> {
    let v = spec.split(',').map(|s| s.trim().parse::<f64>().ok().filter(|x| x.is_finite() && *x > 0.0)
        .with_context(|| format!("--split {spec}: `{}` is not a positive number", s.trim()))).collect::<anyhow::Result<Vec<_>>>()?;
    ensure!(v.len() == n, "--split {spec}: {} share(s) for {n} device(s) (give one per device in --device order)", v.len());
    Ok(v)
}

/// The ends of the layer ranges (after a leading 0) of `n` layers split in proportion to `share`.
fn split_ends(share: &[f64], n: usize) -> Vec<usize> {
    let total: f64 = share.iter().sum();
    let (mut ends, mut acc) = (vec![0], 0.0);
    for f in share { acc += f; ends.push((acc / total * n as f64).round() as usize); }
    ends
}

/// Host-to-VRAM copy rate of `gpu` in MB/s over the path an expert upload takes (pageable host memory, so a fast
/// link reads as the host's copy rate). The second of two 64 MB copies is timed: the first pays the set-up.
fn copy_rate(gpu: &dyn Backend) -> anyhow::Result<f64> {
    let src = vec![1u8; 64 << 20];
    let buf = gpu.alloc(src.len())?;
    let mut secs = 0.0;
    for _ in 0..2 {
        let t = std::time::Instant::now();
        gpu.htod(&src, &buf, 0)?;
        gpu.sync()?;
        secs = t.elapsed().as_secs_f64();
    }
    Ok(64.0 / secs)
}

impl<'a> GpuModel<'a> {
    /// `chunk`: prefill batch in tokens (0 = no batched prefill). `mtp`: load the MTP draft head (needs `nextn`).
    pub fn load(m: &'a Model<'a>, gpu: Box<dyn Backend>, ctx: usize, chunk: usize, vram_margin: usize, k: usize) -> anyhow::Result<GpuModel<'a>> {
        Self::load_part(m, gpu, ctx, chunk, vram_margin, k, 0..m.cfg.n_layer, 1)
    }

    /// `load` with `n` slots: sequences with their own KV cache and recurrent state each (`ctx` tokens a slot), which
    /// `step_rows` decodes together. No MTP draft head; not for hyper-connection, PLE or QSA models.
    pub fn load_slots(m: &'a Model<'a>, gpu: Box<dyn Backend>, ctx: usize, chunk: usize, vram_margin: usize, n: usize) -> anyhow::Result<GpuModel<'a>> {
        Self::load_part(m, gpu, ctx, chunk, vram_margin, 0, 0..m.cfg.n_layer, n)
    }

    /// `load_split` with the same VRAM margin on every device.
    pub fn load_split(m: &'a Model<'a>, gpus: Vec<Box<dyn Backend>>, ctx: usize, chunk: usize, vram_margin: usize, k: usize, ns: usize) -> anyhow::Result<GpuModel<'a>> {
        let margins = vec![vram_margin; gpus.len()];
        Self::load_split_margins(m, gpus, ctx, chunk, &margins, None, k, ns)
    }

    /// `load` with the layers split over `gpus` in proportion to their free VRAM, or to the shares `split` (`--split`,
    /// or the layer counts `calibrate` chose) when given (each device then fills the rest
    /// of its VRAM with expert slots, keeping its entry of `vram_margin` free), and `ns` slots as in `load_slots`.
    /// One device: `load` or `load_slots`.
    pub fn load_split_margins(m: &'a Model<'a>, gpus: Vec<Box<dyn Backend>>, ctx: usize, chunk: usize, vram_margin: &[usize], split: Option<&[f64]>, k: usize, ns: usize) -> anyhow::Result<GpuModel<'a>> {
        ensure!(vram_margin.len() == gpus.len(), "{} VRAM margin(s) for {} device(s)", vram_margin.len(), gpus.len());
        ensure!(split.is_none_or(|s| s.len() == gpus.len()), "{} layer share(s) for {} device(s)", split.map_or(0, |s| s.len()), gpus.len());
        let n = m.cfg.n_layer;
        if gpus.len() == 1 {
            let g = gpus.into_iter().next().unwrap();
            let margin = vram_margin[0];
            return if ns > 1 { Self::load_slots(m, g, ctx, chunk, margin, ns) } else { Self::load(m, g, ctx, chunk, margin, k) };
        }
        let rates = gpus.iter().map(|g| copy_rate(g.as_ref())).collect::<anyhow::Result<Vec<_>>>()?;
        let free = gpus.iter().map(|g| Ok(g.mem_info()?.0 as f64)).collect::<anyhow::Result<Vec<_>>>()?;
        let share = split.map_or_else(|| free.clone(), <[f64]>::to_vec);
        let total: f64 = share.iter().sum();
        // ponytail: EAGLE3/DFlash read features across multiple layers into one device's buffers; no drafts when split.
        let k = if gpus.len() > 1 && m.cfg.eagle3.is_some() { crate::say!("EAGLE3 draft head: not available with a multi-GPU layer split, decoding without drafts"); 0 }
            else if gpus.len() > 1 && m.cfg.dflash.is_some() { crate::say!("DFlash draft model: not available with a multi-GPU layer split, decoding without drafts"); 0 }
            else { k };
        let ends = split_ends(&share, n);
        // The PLE layers share one history ring, which lives on one device.
        let ple: Vec<usize> = (0..n).filter(|&il| m.w.layer(il, "ple_key.weight").is_ok()).collect();
        let mut parts = vec![];
        for (i, gpu) in gpus.into_iter().enumerate() {
            let r = ends[i]..ends[i + 1];
            ensure!(!r.is_empty(), "device {i} gets no layers ({:.0}% of {n}, free VRAM {:.0} MB; see --split)", 100.0 * share[i] / total, free[i] / 1048576.0);
            ensure!(ple.iter().all(|il| r.contains(il)) || !ple.iter().any(|il| r.contains(il)), "the PLE layers would be split across devices");
            crate::say!("device {i} ({}): layers {}..{}, host-to-VRAM copy {:.0} MB/s", gpu.name(), r.start, r.end, rates[i]);
            parts.push(Self::load_part(m, gpu, ctx, chunk, vram_margin[i], k, r, ns)?);
        }
        // Every part holds the state of every slot (a part short of VRAM takes fewer).
        let fit: Vec<usize> = parts.iter().map(|p| p.slots()).collect();
        ensure!(fit.iter().all(|&s| s == fit[0]), "--parallel {ns}: the devices fit {fit:?} slots; lower --parallel or --ctx");
        // One draft length for every part: each caps it by its own backend's verify rows.
        let k = parts.iter().map(|p| p.k).min().unwrap();
        let nsnap = parts.iter().map(|p| p.nsnap).min().unwrap();
        for p in &mut parts { p.k = k; p.nsnap = nsnap; }
        let mut it = parts.into_iter().rev();
        let mut acc = it.next().unwrap();
        for mut p in it { p.next = Some(Box::new(acc)); acc = p; }
        Ok(acc)
    }

    fn load_part(m: &'a Model<'a>, gpu: Box<dyn Backend>, ctx: usize, chunk: usize, vram_margin: usize, k: usize, range: std::ops::Range<usize>, ns: usize) -> anyhow::Result<GpuModel<'a>> {
        let c = &m.cfg;
        ensure!(ns == 1 || c.hc_n == 0 && c.n_embd_ple == 0 && c.idx_top_k == 0, "--parallel: not supported for this model (hyper-connections, PLE or QSA)");
        ensure!((1..=255).contains(&ns), "--parallel {ns}: 1 to 255 slots");
        let last = range.end == m.cfg.n_layer;
        if let Some(t) = m.w.g.tensors.values().find(|t| !cpu_only(&t.name)
            && GgmlType::from_u32(t.ty).is_ok_and(|ty| !ty.gpu(gpu.name()) && !f32_up(&t.name, ty))) {
            anyhow::bail!("tensor {} has type {}, which the {} backend does not support yet", t.name, GgmlType::from_u32(t.ty)?.name(), gpu.name());
        }
        let c = &m.cfg;
        let up = |name: &str| -> anyhow::Result<GpuTensor> { gpu.upload(&m.w.get(name)?) };
        let lt_cache = std::cell::RefCell::new(HashMap::<(usize, String), GpuTensor>::new());
        let lt = |il: usize, s: &str| -> anyhow::Result<GpuTensor> {
            let p = c.phys_layer.get(il).copied().unwrap_or(il);
            if p != il && !m.w.g.tensors.contains_key(&format!("blk.{il}.{s}")) {
                if let Some(cached) = lt_cache.borrow().get(&(p, s.to_string())) {
                    return Ok(cached.clone());
                }
            }
            let gt = gpu.upload(&m.w.layer(il, s)?)?;
            lt_cache.borrow_mut().insert((p, s.to_string()), gt.clone());
            Ok(gt)
        };
        let mhc_rows = chunk.min(ctx).max(TMAX);
        let mhc_cache = std::cell::RefCell::new(HashMap::<(usize, String), MhcRouter>::new());
        let mhc_load = |il: usize, prefix: &str| -> anyhow::Result<MhcRouter> {
            let p = c.phys_layer.get(il).copied().unwrap_or(il);
            if p != il && !m.w.g.tensors.contains_key(&format!("blk.{il}.{prefix}.weight")) {
                if let Some(cached) = mhc_cache.borrow().get(&(p, prefix.to_string())) {
                    return Ok(cached.clone());
                }
            }
            let weight = lt(il, &format!("{prefix}.weight"))?;
            let bias = m.w.layer(il, &format!("{prefix}.bias"))?.to_f32();
            let out_dim = bias.len();
            let buf = gpu.zeros::<f32>(32 + mhc_rows * out_dim.max(24))?;
            gpu.put(&bias, &buf, 0)?;
            let router = MhcRouter { weight, buf, out_dim };
            mhc_cache.borrow_mut().insert((p, prefix.to_string()), router.clone());
            Ok(router)
        };
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
        // Several slots of a model without experts: each slot's attention KV starts at KV_START positions and doubles
        // when a request needs more (`kv_need`), up to `ctx`, so that short requests do not each hold `ctx` of VRAM.
        // ponytail: an MoE keeps `ctx` a slot, as its expert slots take the VRAM the KV would grow into.
        // A host-RAM KV layer (STRATA_KV_HOST) keeps its `ctx`: growing allocates in VRAM.
        let grow = ns > 1 && !c.is_moe() && ctx > KV_START && !(0..c.n_layer).any(crate::host_attn::on_host);
        let mut kv0 = if grow { KV_START } else { ctx };
        // The slots' live state stays in VRAM; more slots than fit would end the load out of memory. A dense model's
        // state may take the VRAM its weights leave (less 512 MB of scratch); an MoE's at most half, so that the other
        // half holds experts. A request past the slots waits for one (`serve_slots`).
        // ponytail: Vulkan keeps each slot's saved state in VRAM too (no `host_buf`), not counted here.
        let ns = if ns > 1 {
            let lay = &c.layers[range.clone()];
            let per = |kv0: usize| lay.iter().map(|&k| if k == LayerKind::Delta { 4 * ((c.ssm_conv - 1) * channels + nv * c.ssm_dv() * dk) } else { 2 * (gpu.kv_words(kv0 * kv) + gpu.kv_words_v(kv0 * kv)) }).sum::<usize>();
            let w: usize = m.w.g.tensors.values().filter(|t| !t.name.contains("_exps.")).map(|t| m.w.g.bytes(t).len()).sum();
            let room = free0.saturating_sub(w + vram_margin + (1 << 29)) / if c.is_moe() { 2 } else { 1 };
            // Short of VRAM for every slot (a BF16 4B on 12 GB): start at KV_MIN positions, which holds a short chat.
            if grow && room / per(kv0).max(1) < ns { kv0 = KV_MIN; }
            let fit = (room / per(kv0).max(1)).clamp(1, ns);
            if fit < ns { crate::say!("--parallel {ns}: the state of {fit} slots fits in VRAM; the other requests wait for a free slot"); }
            fit
        } else { ns };
        let qsa_on = c.idx_top_k > 0 && (0..c.n_layer).any(|il| m.w.layer(il, "indexer.k_proj.weight").is_ok());
        if qsa_on { ensure!(c.idx_head_dim == QSA_D && c.idx_n_head == QSA_H, "QSA indexer: {} heads of {} (the GPU kernels take {QSA_H} of {QSA_D})", c.idx_n_head, c.idx_head_dim); }
        let ring = gpu.zeros::<f32>(if qsa_on { c.n_layer * QSA_RING } else { 1 })?;

        let k = k.min(gpu.verify_rows() - 1);
        // The draft head goes with the last layers (a split model's last part): it reads their hidden rows.
        let mtp = k > 0 && c.n_nextn > 0 && range.end == c.n_layer;
        let nsnap = if k > 0 {
            k.min(SNAP_MAX)
        } else if std::env::var_os("STRATA_SPEC_SNAP").is_some() {
            SNAP_MAX.min(gpu.verify_rows() - 1)
        } else {
            0
        };
        let load_layer = |il: usize, kind: LayerKind| -> anyhow::Result<Layer> {
            let mixer = if kind == LayerKind::Delta {
                Mixer::Delta {
                    qkv: lt(il, "attn_qkv.weight")?, gate: lt(il, "attn_gate.weight")?, beta: lt(il, "ssm_beta.weight")?,
                    alpha: lt(il, "ssm_alpha.weight")?, dt_bias: lt(il, "ssm_dt.bias")?, a: lt(il, "ssm_a")?,
                    conv_w: lt(il, "ssm_conv1d.weight")?, norm: lt(il, "ssm_norm.weight")?, out: lt(il, "ssm_out.weight")?,
                    conv: (0..ns).map(|_| gpu.zeros::<f32>((c.ssm_conv - 1) * channels)).collect::<anyhow::Result<_>>()?,
                    s: (0..ns).map(|_| gpu.zeros::<f32>(nv * c.ssm_dv() * dk)).collect::<anyhow::Result<_>>()?,
                    snap: if nsnap > 0 {
                        let stride = gpu.delta_snap_stride().max(1);
                        Some((
                            gpu.zeros::<f32>(nsnap * (c.ssm_conv - 1) * channels)?,
                            gpu.zeros::<f32>(nsnap.div_ceil(stride) * c.ssm_d_inner * dk)?,
                            gpu.zeros::<f32>(TMAX * channels)?,
                            gpu.zeros::<f32>(TMAX * nv)?,
                            gpu.zeros::<f32>(TMAX * nv)?,
                        ))
                    } else { None },
                }
            } else {
                Mixer::Attn {
                    q: lt(il, "attn_q.weight")?, k: lt(il, "attn_k.weight")?, v: lt(il, "attn_v.weight")?,
                    qn: lt(il, "attn_q_norm.weight")?, kn: lt(il, "attn_k_norm.weight")?, o: lt(il, "attn_output.weight")?,
                    sink: if c.attn_sinks { Some(lt(il, "attn_sinks.weight")?) } else { None },
                    kc: (0..ns).map(|_| kv_alloc(gpu.as_ref(), il, gpu.kv_words(kv0 * kv) * 2)).collect::<anyhow::Result<_>>()?,
                    vc: (0..ns).map(|_| kv_alloc(gpu.as_ref(), il, gpu.kv_words_v(kv0 * kv) * 2)).collect::<anyhow::Result<_>>()?,
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
            let ffn = if c.is_moe_layer(il) {
                let shared = if m.w.layer(il, "ffn_up_shexp.weight").is_ok() {
                    Some(Shared {
                        gate_inp: lt(il, "ffn_gate_inp_shexp.weight").ok(), gate: lt(il, "ffn_gate_shexp.weight")?,
                        up: lt(il, "ffn_up_shexp.weight")?, down: lt(il, "ffn_down_shexp.weight")?,
                    })
                } else { None };
                Ffn::Moe { router: lt(il, "ffn_gate_inp.weight")?, shared }
            } else {
                Ffn::Dense { gate: lt(il, "ffn_gate.weight")?, up: lt(il, "ffn_up.weight")?, down: lt(il, "ffn_down.weight")? }
            };
            let hc = if c.hc_n > 0 { Some([hc_read(&format!("blk.{il}.hc_attn_"))?, hc_read(&format!("blk.{il}.hc_ffn_"))?]) } else { None };
            let mk = c.mhc_layer_kinds.get(il).copied().unwrap_or(0);
            let mhc = if c.mhc_streams > 0 && mk != 0 {
                Some((mk, mhc_load(il, "mhc_attn")?, mhc_load(il, "mhc_ffn")?))
            } else {
                None
            };
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
            Ok(Layer { attn_norm, post_norm, hc, mhc, ple, mixer, ffn })
        };
        let layers = range.clone().map(|il| load_layer(il, c.layers[il])).collect::<anyhow::Result<Vec<_>>>()?;
        let (output_norm, output_hc) = if !last { (None, None) } else if c.hc_n > 0 { (None, Some(hc_read("output_hc_")?)) } else { (Some(up("output_norm.weight")?), None) };
        let ple_ring = if c.n_embd_ple > 0 { Some((gpu.zeros::<f32>(16 * hcn * c.n_embd)?, gpu.zeros::<f32>(16 * hcn * c.n_embd)?)) } else { None };
        let output = if last { Some(up("output.weight").or_else(|_| up("token_embd.weight"))?) } else { None };
        let dflash = if k > 0 && last && ns == 1 {
            if let Some(dc) = c.dflash.clone() {
                dflash_check(m, &dc)?;
                let df_kv = dc.n_head_kv * dc.head_dim;
                let df_q = dc.n_head * dc.head_dim;
                let df_attp = dc.n_head * ATT_SPLITS * (dc.head_dim + 2);
                let mut dl_layers = Vec::with_capacity(dc.n_layer);
                for il in 0..dc.n_layer {
                    let df_lt = |s: &str| {
                        let t = m.w.get(&format!("dflash.blk.{il}.{s}.weight"))?;
                        if t.ty == GgmlType::F16 { gpu.upload(&t) } else { gpu.upload_raw(&t) }
                    };
                    dl_layers.push(DFlashLayer {
                        attn_norm: df_lt("attn_norm")?,
                        q: df_lt("attn_q")?,
                        k: df_lt("attn_k")?,
                        v: df_lt("attn_v")?,
                        qn: df_lt("attn_q_norm")?,
                        kn: df_lt("attn_k_norm")?,
                        o: df_lt("attn_output")?,
                        post_norm: df_lt("ffn_norm")?,
                        gate: df_lt("ffn_gate")?,
                        up: df_lt("ffn_up")?,
                        down: df_lt("ffn_down")?,
                        attn_conv: if dc.conv_kernel_size > 0 {
                            Some(DFlashConv { base: df_lt("attn_conv_base")?, proj: df_lt("attn_conv_proj")? })
                        } else { None },
                        mlp_conv: if dc.conv_kernel_size > 0 {
                            Some(DFlashConv { base: df_lt("mlp_conv_base")?, proj: df_lt("mlp_conv_proj")? })
                        } else { None },
                        kc: (0..ns).map(|_| gpu.zeros::<u16>(gpu.kv_words(ctx * df_kv))).collect::<anyhow::Result<_>>()?,
                        vc: (0..ns).map(|_| gpu.zeros::<u16>(gpu.kv_words_v(ctx * df_kv))).collect::<anyhow::Result<_>>()?,
                        is_swa: dc.layer_swa.get(il).copied().unwrap_or(false),
                    });
                }
                let swa_cnt = dl_layers.iter().filter(|l| l.is_swa).count();
                let short = build_shortlist(m, &*gpu, output.as_ref())?;
                let has_short = short.is_some();
                let selector = if dc.selector_rank > 0 {
                    let pt = m.w.get("dflash.selector.pred_codebook.weight")?;
                    let st = m.w.get("dflash.selector.succ_codebook.weight")?;
                    Some(DFlash2Selector {
                        hidden_proj: up("dflash.selector.hidden_proj.weight")?,
                        h_proj: gpu.zeros::<f32>(TMAX * dc.selector_rank)?,
                        pred_raw: pt.data.to_vec(),
                        pred_ty: pt.ty,
                        succ_raw: st.data.to_vec(),
                        succ_ty: st.ty,
                        rank: dc.selector_rank,
                        top_k: dc.selector_top_k.max(1),
                    })
                } else { None };
                let conv_dyn_len = if dc.conv_kernel_size > 0 && dc.conv_group_size > 0 {
                    TMAX * 2 * dc.conv_kernel_size * (dc.n_embd / dc.conv_group_size)
                } else { 1 };
                crate::say!(
                    "DFlash{} block-diffusion drafter loaded: {} layers (SWA={swa_cnt}, full={}), block_size={}, k={k}, w={}, target_layers={:?}",
                    if selector.is_some() { "2" } else { "" },
                    dc.n_layer,
                    dc.n_layer - swa_cnt,
                    dc.block_size,
                    dl_layers.first().map_or("?", |l| l.q.ty.name()),
                    dc.target_layer_ids,
                );
                Some(DFlashGpu {
                    fc: up("dflash.fc.weight")?,
                    hidden_norm: up("dflash.hidden_norm.weight")?,
                    norm: up("dflash.norm.weight")?,
                    layers: dl_layers,
                    feat: gpu.zeros::<f32>(TMAX * dc.target_layer_ids.len() * c.n_embd)?,
                    // hc_mix indexes its gate buffer per position (g[t*hc*n + c*n + i]), so it needs
                    // TMAX rows; 12.0 => sigmoid ~ 1 => plain stream mean.
                    dfg: { let n = (TMAX * c.hc_n * c.n_embd).max(1); let b = gpu.zeros::<f32>(n)?; gpu.put(&vec![12.0f32; n], &b, 0)?; b },
                    dfm: gpu.zeros::<f32>(TMAX * c.n_embd)?,
                    th: gpu.zeros::<f32>(TMAX * dc.n_embd)?,
                    x: gpu.zeros::<f32>(TMAX * dc.n_embd)?,
                    h: gpu.zeros::<f32>(TMAX * dc.n_embd)?,
                    h2: gpu.zeros::<f32>(if dc.conv_kernel_size > 0 { TMAX * dc.n_embd } else { 1 })?,
                    conv_dyn: gpu.zeros::<f32>(conv_dyn_len)?,
                    q: gpu.zeros::<f32>(TMAX * df_q)?,
                    k: gpu.zeros::<f32>(dc.n_layer.max(1) * TMAX * df_kv)?,
                    k_ctx: gpu.zeros::<f32>(dc.n_layer.max(1) * TMAX * df_kv)?,
                    v_ctx: gpu.zeros::<f32>(dc.n_layer.max(1) * TMAX * df_kv)?,
                    att: gpu.zeros::<f32>(TMAX * df_q)?,
                    attp: gpu.zeros::<f32>(TMAX * df_attp)?,
                    ge: gpu.zeros::<f32>(TMAX * dc.n_ff)?,
                    ue: gpu.zeros::<f32>(TMAX * dc.n_ff)?,
                    ctx_step: gpu.zeros::<i32>(2 * TMAX)?,
                    step_swa: gpu.zeros::<i32>(2 * TMAX)?,
                    step_full: gpu.zeros::<i32>(2 * TMAX)?,
                    selector,
                    pos: 0,
                    pend_ctx: 0,
                    short,
                    short_oob: std::cell::Cell::new(0),
                    use_short: std::cell::Cell::new(has_short),
                    cur_block: std::cell::Cell::new(dc.block_size),
                    low_streak: std::cell::Cell::new(0),
                    cfg: dc,
                })
            } else { None }
        } else { None };
        let mtp_head = if mtp {
            let il = c.n_layer;
            if let Some(ec) = c.eagle3.clone() {
                let eg_kv = ec.n_head_kv * ec.head_dim;
                let layer = Layer {
                    attn_norm: None,
                    post_norm: Some(lt(il, "post_attention_norm.weight")?),
                    hc: None,
                    mhc: None,
                    ple: None,
                    mixer: Mixer::Attn {
                        q: lt(il, "attn_q.weight")?,
                        k: lt(il, "attn_k.weight")?,
                        v: lt(il, "attn_v.weight")?,
                        qn: lt(il, "nextn.enorm.weight")?,
                        kn: lt(il, "nextn.hnorm.weight")?,
                        o: lt(il, "attn_output.weight")?,
                        kc: (0..ns).map(|_| kv_alloc(gpu.as_ref(), il, gpu.kv_words(ctx * eg_kv) * 2)).collect::<anyhow::Result<_>>()?,
                        vc: (0..ns).map(|_| kv_alloc(gpu.as_ref(), il, gpu.kv_words_v(ctx * eg_kv) * 2)).collect::<anyhow::Result<_>>()?,
                        qsa: None,
                        sink: None,
                    },
                    ffn: Ffn::Dense {
                        gate: lt(il, "ffn_gate.weight")?,
                        up: lt(il, "ffn_up.weight")?,
                        down: lt(il, "ffn_down.weight")?,
                    },
                };
                let attp_len = ec.n_head * ATT_SPLITS * (ec.head_dim + 2);
                let eg = Eagle3Gpu {
                    fc: lt(il, "eagle3.fc.weight")?,
                    output: lt(il, "eagle3.output.weight")?,
                    feat: gpu.zeros::<f32>(TMAX * 3 * c.n_embd)?,
                    cur_g: gpu.zeros::<f32>(TMAX * c.n_embd)?,
                    q: gpu.zeros::<f32>(TMAX * ec.n_head * ec.head_dim)?,
                    k: gpu.zeros::<f32>(TMAX * eg_kv)?,
                    v: gpu.zeros::<f32>(TMAX * eg_kv)?,
                    att: gpu.zeros::<f32>(TMAX * ec.n_head * ec.head_dim)?,
                    attp: gpu.zeros::<f32>(TMAX * attp_len)?,
                    ge: gpu.zeros::<f32>(TMAX * ec.n_ff)?,
                    ue: gpu.zeros::<f32>(TMAX * ec.n_ff)?,
                    cfg: ec,
                };
                Some(MtpHead {
                    layer,
                    eh_proj: None,
                    enorm: lt(il, "nextn.enorm.weight")?,
                    hnorm: lt(il, "nextn.hnorm.weight")?,
                    head_norm: Some(lt(il, "nextn.shared_head_norm.weight")?),
                    hc_head: None,
                    worker: None,
                    step_buf: gpu.zeros::<i32>(2 * TMAX)?,
                    emb: gpu.zeros::<f32>(TMAX * c.n_embd)?,
                    eh: gpu.zeros::<f32>(TMAX * 2 * c.n_embd)?,
                    x: gpu.zeros::<f32>(TMAX * c.n_embd)?,
                    pos: 0,
                    eagle: Some(eg),
                    saved_x: gpu.zeros::<f32>(c.n_embd)?,
                    pend: vec![],
                    adapt_window: 0x0FFF,
                    adapt_skip: 0,
                    short: None,
                    short_oob: std::cell::Cell::new(0),
                    use_short: std::cell::Cell::new(false),
                })
            } else {
                let short = build_shortlist(m, &*gpu, output.as_ref())?;
                let has_short = short.is_some();
                Some(MtpHead {
                    layer: load_layer(il, LayerKind::Attn)?,
                    eh_proj: Some(lt(il, "nextn.eh_proj.weight")?),
                    enorm: lt(il, "nextn.enorm.weight")?,
                    hnorm: lt(il, "nextn.hnorm.weight")?,
                    head_norm: if c.hc_n > 0 { None } else { Some(lt(il, "nextn.shared_head_norm.weight")?) },
                    hc_head: if c.hc_n > 0 { Some(hc_read(&format!("blk.{il}.nextn.hc_head_"))?) } else { None },
                    worker: Some(CpuWorker::new(m, &*gpu, vec![c.n_layer])?),
                    step_buf: gpu.zeros::<i32>(2 * TMAX)?,
                    emb: gpu.zeros::<f32>(TMAX * c.n_embd)?,
                    eh: gpu.zeros::<f32>(TMAX * 2 * hcn * c.n_embd)?,
                    x: gpu.zeros::<f32>(TMAX * hcn * c.n_embd)?,
                    pos: 0,
                    eagle: None,
                    saved_x: gpu.zeros::<f32>(hcn * c.n_embd)?,
                    pend: vec![],
                    adapt_window: 0x0FFF,
                    adapt_skip: 0,
                    short,
                    short_oob: std::cell::Cell::new(0),
                    use_short: std::cell::Cell::new(has_short),
                })
            }
        } else { None };

        let ke = c.n_expert_used.max(1);
        let df_ff = c.dflash.as_ref().map_or(0, |d| d.n_ff);
        let df_q = c.dflash.as_ref().map_or(0, |d| d.n_head * d.head_dim);
        let ff = (ke * c.n_ff_exp).max(c.n_ff).max(df_ff);
        let att_dim = (c.n_head * c.head_dim).max(df_q);
        let mut sizes = vec![
            ("h", c.n_embd), ("hq", c.n_embd * 9 / 32), ("mix", c.n_embd), ("qkv", channels), ("z", c.ssm_d_inner), ("beta", nv), ("g", nv),
            ("y", channels), ("o", c.ssm_d_inner), ("qg", c.n_head * 2 * c.head_dim), ("q", c.n_head * c.head_dim),
            ("kraw", kv), ("k", kv), ("v", kv), ("att", c.n_head * c.head_dim), ("rl", c.n_expert), ("wts", ke),
            ("ge", ff), ("ue", ff), ("ye", ke * c.n_embd), ("shg", c.n_ff_shexp), ("shu", c.n_ff_shexp),
            ("shy", c.n_embd), ("sg", 1), ("wh", ke), ("attp", c.n_head * ATT_SPLITS * (c.head_dim + 2)),
            ("ffq", ff * 9 / 32), ("shq", c.n_ff_shexp * 9 / 32), ("attq", att_dim * 9 / 32),
            ("oq", c.ssm_d_inner * 9 / 32),
        ];
        if qsa_on { sizes.extend([("ik", QSA_D), ("iq", QSA_H * QSA_D), ("iqn", QSA_H * QSA_D)]); }
        if c.hc_n > 0 {
            let d = c.hc_n * c.n_embd;
            sizes.extend([("xn", d), ("lo", c.hc_rank), ("hg", d), ("inj", c.hc_n), ("mo", c.n_embd),
                          ("pk", d), ("pq", d), ("pg", d), ("pn", d), ("pv", c.n_embd), ("emb", c.n_embd)]);
        } else if c.mhc_streams > 0 {
            let d = c.mhc_streams * c.n_embd;
            sizes.extend([("mr", d), ("xn", d), ("mo", c.n_embd)]);
        }
        let mhc_ones = if c.mhc_streams > 0 {
            let d = c.mhc_streams * c.n_embd;
            let b = gpu.zeros::<f32>(d)?;
            gpu.put(&vec![1.0f32; d], &b, 0)?;
            Some(b)
        } else {
            None
        };
        let hada = match &c.hadamard {
            Some(r) => {
                // ponytail: the rotated inputs stay floats. A rotated file with int8-input expert types needs a
                // `quant_xq` of "hr" and of the rotated SwiGLU output.
                for (name, info) in &m.w.g.tensors {
                    if name.contains("_exps.") && !GgmlType::from_u32(info.ty)?.float_x() {
                        anyhow::bail!("{name}: a rotated (prism.hadamard) expert matrix of type {} runs only on the CPU", GgmlType::from_u32(info.ty)?.name());
                    }
                }
                sizes.push(("hr", c.n_embd));
                let up = |n: usize| -> anyhow::Result<Buf> { let b = gpu.zeros::<f32>(n)?; gpu.put(r.signs(n), &b, 0)?; Ok(b) };
                Some([up(c.n_embd)?, up(c.n_ff_exp)?])
            }
            None => None,
        };
        let bufs = sizes.into_iter().map(|(name, n)| Ok((name, gpu.zeros::<f32>(TMAX * n)?))).collect::<anyhow::Result<HashMap<_, _>>>()?;
        let (x, logits, ids) = (gpu.zeros::<f32>(TMAX * hcn * c.n_embd)?, gpu.zeros::<f32>(TMAX * c.n_vocab)?, gpu.zeros::<i32>(TMAX * ke)?);
        let amax = gpu.zeros::<i32>(TMAX)?;
        let (slots, step_buf) = (gpu.zeros::<i32>(TMAX * ke)?, gpu.zeros::<i32>(2 * TMAX)?);
        let rope_buf = gpu.zeros::<i32>(2 * TMAX)?;
        let gpu_mrope = gpu.zeros::<i32>(8)?;
        // Hyper-connection prefill scratch is ~4x larger a token, so a large chunk leaves fewer expert slots: qwen4exp at
        // 943 tokens (warm) prefills 447 tok/s with 1024, 347 with 4096. Long prompts gain (each chunk restages the experts
        // not resident: Flash-Next at 4764 tokens 132 -> 158 tok/s on a 3060), so `calibrate` picks per machine.
        let pf = if chunk > 0 && m.loras.is_empty() { Some(prefill::Scratch::new(m, &*gpu, chunk.min(ctx))?) } else { None };
        let qsa = if qsa_on {
            let (top, nbs) = (c.idx_top_k / QSA_POOL, ctx / QSA_POOL);
            let (sst, mw) = (1 + c.idx_top_k + QSA_POOL, (nbs + 1).div_ceil(32));
            let rows = pf.as_ref().map_or(0, |p| p.chunk.next_multiple_of(64));
            Some(QsaBufs { saved: gpu.zeros::<f32>(ring.len() / 4)?, ring, score: gpu.zeros::<f32>(QSA_ROWS.max(TMAX) * nbs)?,
                           sel: gpu.zeros::<i32>(TMAX * sst)?, mask: gpu.zeros::<u32>((rows * mw).max(1))?, top, nbs, sst, mw })
        } else { None };
        // Each slot's `save_state` copy, before the expert plan so that it leaves room for them (one slot allocates it
        // at its first `save_state`).
        let mut seqs: Vec<Seq> = (0..ns).map(|_| Seq::default()).collect();
        if ns > 1 { for q in &mut seqs { q.saved = alloc_saved(&*gpu, &layers, true)?; } }
        let saved = std::mem::take(&mut seqs[0].saved);
        // `step_batch` covers plain dense attention models in one part; the rest decode `STEP_ROWS` rows a forward.
        // A model with DeltaNet layers or experts takes the layers of the prefill (`prefill_chunk` in rows mode).
        let plain = output_norm.is_some() && output.is_some() && range.start == 0 && !mtp && c.hc_n == 0 && c.n_embd_ple == 0 && !qsa_on
            && c.hadamard.is_none() && layers.iter().enumerate().all(|(il, l)| l.attn_norm.is_some() && l.post_norm.is_some() && l.mhc.is_none() && l.ple.is_none()
                && c.rope_layer.get(il).copied().unwrap_or(true));
        let dense = layers.iter().all(|l| matches!(l.mixer, Mixer::Attn { .. }) && matches!(l.ffn, Ffn::Dense { .. }));
        let bt = match &pf {
            Some(p) if ns > STEP_ROWS && plain && gpu.kv_rows() => {
                // ponytail: 128 rows a forward, the size of the CUDA argmax scratch; split the argmax to go past it.
                let cap = ns.min(p.chunk).min(128);
                let rows = p.chunk;
                Some(Batch { dense, cap, rows, step: gpu.zeros::<i32>(2 * rows)?, kvt: gpu.zeros::<u64>(layers.len() * 2 * rows)?, logits: gpu.zeros::<f32>(cap * c.n_vocab)?,
                             amax: gpu.zeros::<i32>(cap)?, attp: gpu.zeros::<f32>(cap * c.n_head * ATT_SPLITS * (c.head_dim + 2))? })
            }
            _ => None,
        };
        let mut ext_draft = if k > 0 && last && ns == 1 && m.draft_model.is_some() {
            let dm = m.draft_model.as_deref().unwrap();
            let d_gpu = gpu.share()?;
            let mut dg = Self::load_part(dm, d_gpu, ctx, chunk, vram_margin, 0, 0..dm.cfg.n_layer, 1)?;
            dg.graph_tag = 1u64 << 62;
            if dg.saved.is_empty() {
                dg.saved = alloc_saved(&*dg.gpu, &dg.layers, false)?;
            }
            Some(Box::new(dg))
        } else {
            None
        };
        // STRATA_KV_HOST: one layer's cache-sized VRAM buffer pair. Prefill copies a host layer's cache prefix into it
        // (a DMA copy, ~10x what the attention kernel reads from mapped memory) and runs the usual kernel on it.
        let kv_stage = match layers.iter().enumerate().filter(|(il, _)| crate::host_attn::on_host(*il))
            .filter_map(|(_, l)| match &l.mixer { Mixer::Attn { kc, .. } => kc.first().map(|b| b.len()), _ => None }).max() {
            Some(b) if std::env::var_os("STRATA_HA_NO_STAGE").is_none() => match (gpu.alloc(b), gpu.alloc(b)) {
                (Ok(k), Ok(v)) => Some((k, v)),
                _ => { crate::say!("host KV: no VRAM for the {} MB prefill staging, prefill reads the cache over PCIe", 2 * b >> 20); None }
            },
            _ => None,
        };
        let (free1, _) = gpu.mem_info()?;
        // Past the budget the driver spills buffers to system memory: RADV then runs a long prefill slowly enough to
        // hit amdgpu's job timeout and lose the device. A margin above 512 MB only leaves fewer expert slots.
        let floor = vram_margin.min(512 << 20);
        ensure!(free1 >= floor, "the model, KV cache and prefill scratch leave {} MB of VRAM free, under the {} MB \
            margin: lower --ctx or --parallel, use --kv q8 or a smaller --chunk (--vram-margin 0 skips this check)", free1 >> 20, floor >> 20);

        let cache = if c.is_moe() {
            // STRATA_EXPERT_MB sets the expert slot budget explicitly (0: no slots, every expert on the CPU, which the
            // split test relies on); otherwise use free VRAM minus margin.
            let budget = match std::env::var("STRATA_EXPERT_MB")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
            {
                Some(mb) => {
                    crate::say!("note: STRATA_EXPERT_MB={mb} override active (reported free device heap: {} MB)", free1 >> 20);
                    mb << 20
                }
                None => free1.saturating_sub(vram_margin),
            };
            Some(ExpertCache::new(m, &*gpu, budget, range.start..range.end + (mtp && c.eagle3.is_none()) as usize)?)
        } else { None };
        let moe_layers: Vec<usize> = range.clone().filter(|&il| c.is_moe_layer(il)).collect();
        let worker = if !moe_layers.is_empty() { Some(CpuWorker::new(m, &*gpu, moe_layers)?) } else { None };
        // R3a: prefetch the next layer's experts predicted by the previous token's routing there. Disabled on gfx11xx/iGPU
        // where the small L2 cache causes prefetching to evict active lines and hurt decode throughput.
        let want_prefetch = std::env::var("STRATA_MOE_PREFETCH").is_ok_and(|v| v != "0");
        if c.is_moe() && want_prefetch && !gpu.moe_prefetch_supported() {
            crate::say!("note: STRATA_MOE_PREFETCH ignored on {} (gfx11xx/iGPU L2 cache too small for MoE expert prefetch)", gpu.device());
        }
        let prefetch = c.is_moe() && gpu.moe_prefetch_supported() && want_prefetch;
        let pids = if prefetch { Some(gpu.zeros::<i32>((c.n_layer + 1) * c.n_expert_used)?) } else { None };
        let mb = |b: usize| b as f64 / 1048576.0;
        crate::say!(
            "plan: weights+state+scratch {:.0} MB (KV {:.0} MB for {kv0} tokens{}{}, prefill {:.0} MB), expert slots {:.0} MB, resident experts {}/{}",
            mb(free0.saturating_sub(free1)), mb((gpu.kv_words(kv0 * kv) + gpu.kv_words_v(kv0 * kv)) * 2 * c.layers[range.clone()].iter().filter(|&&k| k == LayerKind::Attn).count()),
            if ns > 1 { format!(" in each of {ns} slots") } else { String::new() },
            if grow { format!(", growing to {ctx} on demand") } else { String::new() }, mb(pf.as_ref().map_or(0, |p| p.bytes())),
            mb(cache.as_ref().map_or(0, |x| x.slot_bytes_total())),
            cache.as_ref().map_or(0, |x| x.resident_count()), cache.as_ref().map_or(0, |x| x.total_experts()),
        );
        // R3: expert bytes per token and the per-layer working set, against the 6 MB L2 of the V100, so the prefetch
        // decision has numbers in the log (the attained GB/s is printed after a bench run).
        if let Some(ca) = &cache {
            let k = c.n_expert_used;
            let per = (0..ca.layers_len()).map(|il| ca.active_set_bytes(il, k)).max().unwrap_or(0);
            crate::say!("expert traffic: {:.1} MB per layer active set (top-{k}), {:.1} MB per token over {} MoE layers{}",
                mb(per), mb(ca.token_bytes(k)), ca.total_experts() / c.n_expert.max(1),
                if prefetch { ", next-layer L2 prefetch on (STRATA_MOE_PREFETCH)" } else { "" });
        }
        let mtp = mtp_head;
        let adapt_every = std::env::var("STRATA_ADAPT_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
        let adapt_swaps = std::env::var("STRATA_ADAPT_SWAPS").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
        let reset_counts_after_prefill = std::env::var_os("STRATA_RESET_COUNTS").is_some();
        let _ = &mut ext_draft;
        crate::weights::trim_working_set();
        Ok(GpuModel { m, gpu, layers, output_norm, output_hc, output, ple_ring, qsa, ple_prev: [-1, -1], saved_ple_prev: [-1, -1], ple_undo: ([-1, -1], vec![]), bufs, x, worker, logits, amax, ids, slots, step_buf, rope_buf, rope_offset: 0, hada, mrope: false, mrope_buf: gpu_mrope, ctx, pos: 0, saved, saved_pos: 0, saved_ids: vec![], sys: None, parked: vec![], owner: 0, saved_owner: 0, sys_owner: 0, cache_ram: 0, cache, adapt_every, adapt_swaps, reset_counts_after_prefill, hits: 0, misses: 0, tokens: 0, mailbox_writes: Default::default(), mailbox_reads: Default::default(), pids, cpu_time: Duration::ZERO, resumed: Default::default(), lora_applied: Default::default(), drafts: (0, 0), split_time: Default::default(), k, nsnap, short: 32, pf, prompt_cache_path: None, bt, mtp, dflash, ext_draft, ext_draft_pend: Vec::new(), mhc_ones, graph_tag: 0, ngram: None, first: range.start, next: None,
                      cur: 0, seqs, kv_cap: if grow { vec![kv0; ns] } else { vec![] }, kv0, rows: vec![], row_tokens: 0, cpu_frac: 0.5, kv_stage })
    }

    /// Clear KV and recurrent state (and forget the `prefill_cached` prefix, whose KV this zeroes).
    pub fn reset(&mut self) -> anyhow::Result<()> {
        self.saved_ids.clear();
        self.sys = None;
        if let Some(h) = self.mtp.as_mut() {
            h.pos = 0; h.pend.clear(); h.adapt_window = 0x0FFF; h.adapt_skip = 0;
            h.short_oob.set(0); h.use_short.set(h.short.is_some());
        }
        if let Some(df) = self.dflash.as_mut() {
            df.pos = 0;
            df.pend_ctx = 0;
            df.short_oob.set(0);
            df.use_short.set(df.short.is_some());
            df.cur_block.set(df.cfg.block_size);
            df.low_streak.set(0);
            for dl in &df.layers {
                self.gpu.memset(&dl.kc[self.cur])?;
                self.gpu.memset(&dl.vc[self.cur])?;
            }
        }
        if let Some(ed) = self.ext_draft.as_mut() {
            ed.reset()?;
        }
        self.ext_draft_pend.clear();
        if let Some(ng) = self.ngram.as_mut() { ng.reset(); }
        // A slot whose KV grew gives the VRAM back (its new caches are zeroed).
        let shrink = self.kv_cap.get(self.cur).is_some_and(|&n| n > self.kv0);
        if shrink { self.kv_resize(self.cur, self.kv0, false)?; }
        for l in self.layers.iter().chain(self.mtp.as_ref().map(|h| &h.layer)) {
            match &l.mixer {
                Mixer::Delta { conv, s, .. } => { self.gpu.memset(&conv[self.cur])?; self.gpu.memset(&s[self.cur])?; }
                Mixer::Attn { kc, vc, .. } if !shrink => { self.gpu.memset(&kc[self.cur])?; self.gpu.memset(&vc[self.cur])?; }
                Mixer::Attn { .. } => {}
            }
        }
        self.pos = 0;
        self.rope_offset = 0;
        self.ple_prev = [-1, -1];
        if let Some(n) = self.next.as_mut() { n.reset()?; }
        Ok(())
    }

    pub fn has_dflash(&self) -> bool { self.tail().dflash.is_some() }
    pub fn has_ext_draft(&self) -> bool { self.tail().ext_draft.is_some() }

    fn verify_gemm(&self, nt: usize) -> bool {
        let min = std::env::var("STRATA_VERIFY_GEMM").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
        min > 0 && nt >= min && self.rows.is_empty()
    }

    pub(super) fn reset_shortlist(&self) {
        let t = self.tail();
        if let Some(h) = &t.mtp {
            h.short_oob.set(0);
            h.use_short.set(h.short.is_some());
        }
        if let Some(df) = &t.dflash {
            df.short_oob.set(0);
            df.use_short.set(df.short.is_some());
            df.cur_block.set(df.cfg.block_size);
            df.low_streak.set(0);
        }
    }

    pub(super) fn note_short_tokens(&self, tokens: &[u32]) {
        let track = |short: &Option<Shortlist>, oob_c: &std::cell::Cell<usize>, use_c: &std::cell::Cell<bool>| {
            if !use_c.get() { return; }
            let Some(sl) = short else { return };
            let mut oob = oob_c.get();
            for &tok in tokens {
                if !sl.in_set.get(tok as usize).copied().unwrap_or(false) {
                    oob += 1;
                    if oob > 4 {
                        use_c.set(false);
                        break;
                    }
                }
            }
            oob_c.set(oob);
        };
        let t = self.tail();
        if let Some(h) = &t.mtp { track(&h.short, &h.short_oob, &h.use_short); }
        if let Some(df) = &t.dflash { track(&df.short, &df.short_oob, &df.use_short); }
    }

    /// Run an image prompt as a fresh sequence and return its last logits: text tokens through `step`, image tokens
    /// through `step_embed` with their M-RoPE positions. `step` then uses the RoPE offset for text past the image.
    pub fn prefill_image(&mut self, p: &crate::vision::Prompt, embs: &[Vec<Vec<f32>>]) -> anyhow::Result<Vec<f32>> {
        self.prefill_image_with(p, embs, &mut |_| Ok(()))
    }

    /// `prefill_image` with `between` every 64 tokens (the other slots' decode steps, under `--parallel`); this slot
    /// is made current again after it.
    pub fn prefill_image_with(&mut self, p: &crate::vision::Prompt, embs: &[Vec<Vec<f32>>], between: &mut dyn FnMut(&mut Self) -> anyhow::Result<()>)
        -> anyhow::Result<Vec<f32>> {
        let c = &self.m.cfg;
        ensure!(c.n_embd_ple == 0 && c.idx_top_k == 0, "image input is not supported for this model yet ({} with PLE or QSA layers)", c.arch);
        self.reset()?;
        self.rows.clear();
        self.parked.clear();
        // The saved and stashed states of a model without recurrent state are positions in the KV just cleared.
        (self.saved_ids, self.sys) = (vec![], None);
        let cur = self.cur;
        // The draft head has not seen the image rows; image prompts decode without MTP/DFlash.
        let t = self.tail_mut();
        t.mtp = None;
        t.dflash = None;
        t.ext_draft = None;
        t.ext_draft_pend.clear();
        let mut rows = p.images.iter().zip(embs).flat_map(|(span, e)| (span.start..).zip(e)).peekable();
        let mut logits = Vec::new();
        // ponytail: one token per forward (decode speed); a batched prefill needs M-RoPE in the prefill kernels.
        for (i, &id) in p.ids.iter().enumerate() {
            if i > 0 && i % 64 == 0 { between(self)?; self.use_slot(cur); }
            logits = match rows.next_if(|r| r.0 == i) {
                Some((_, x)) => self.step_embed(x, p.positions[i])?,
                None => { self.rope_offset = p.positions[i][0] - self.pos as i32; self.step(id)? }
            };
        }
        self.rope_offset = p.next_position - self.pos as i32;
        Ok(logits)
    }

    /// `step` for an image token: `x` is its embedding (a projector row), `pos` its M-RoPE positions [t, y, x, z].
    fn step_embed(&mut self, x: &[f32], pos: [i32; 4]) -> anyhow::Result<Vec<f32>> {
        if self.pos >= self.ctx { bail!("context full ({} tokens; raise --ctx)", self.ctx); }
        self.kv_need(self.cur, self.pos + 1)?;
        let seq = self.begin_seq();
        self.gpu.put(&x.repeat(self.m.cfg.hc_n.max(1)), &self.x, 0)?;
        self.gpu.put(&[self.pos as i32, seq as i32], &self.step_buf, 0)?;
        self.gpu.put(&pos.map(|p| [p, seq as i32]).concat(), &self.mrope_buf, 0)?;
        self.mrope = true;
        let r = self.pass(1);
        self.mrope = false;
        r?;
        let t = self.tail();
        let mut logits = t.gpu.get::<f32>(&t.logits, 0, self.m.cfg.n_vocab)?;
        if t.finish(1, gq_min(self.pos, 1), None)? { logits = t.gpu.get::<f32>(&t.logits, 0, self.m.cfg.n_vocab)?; }
        self.after_forward(1)?;
        Ok(logits)
    }

    /// Slots (`load_slots`).
    pub fn slots(&self) -> usize { self.seqs.len() }

    /// Make slot `s` current: `prefill`, `prefill_cached`, `step`, `reset` and the rest then use its state.
    pub fn use_slot(&mut self, s: usize) {
        if let Some(n) = self.next.as_mut() { n.use_slot(s); }
        if s == self.cur { return; }
        let mut q = std::mem::take(&mut self.seqs[s]);
        std::mem::swap(&mut self.pos, &mut q.pos);
        std::mem::swap(&mut self.rope_offset, &mut q.rope);
        std::mem::swap(&mut self.saved, &mut q.saved);
        std::mem::swap(&mut self.saved_pos, &mut q.saved_pos);
        std::mem::swap(&mut self.saved_ids, &mut q.saved_ids);
        std::mem::swap(&mut self.saved_owner, &mut q.saved_owner);
        std::mem::swap(&mut self.sys, &mut q.sys);
        std::mem::swap(&mut self.sys_owner, &mut q.sys_owner);
        self.seqs[self.cur] = q;
        self.cur = s;
    }

    /// Whether slot `s` holds an image prompt (its RoPE positions are offset from its KV rows): `step_rows` and
    /// `step_batch` do not take it.
    pub fn slot_shifted(&self, s: usize) -> bool { if s == self.cur { self.rope_offset != 0 } else { self.seqs[s].rope != 0 } }

    /// The tokens behind slot `s`'s saved state and their owner (see `prefill_cached`).
    pub fn slot_saved(&self, s: usize) -> (&[u32], u64) {
        if s == self.cur { (&self.saved_ids, self.saved_owner) } else { (&self.seqs[s].saved_ids, self.seqs[s].saved_owner) }
    }

    /// The most rows `step_batch` takes (0: this model has no batched decode).
    /// Most tokens of one `verify` forward on this backend (4 on CUDA, `TMAX` on Vulkan).
    pub fn tmax(&self) -> usize { self.gpu.verify_rows() }

    pub fn batch_cap(&self) -> usize { self.bt.as_ref().map_or(0, |b| b.cap) }

    pub fn is_moe(&self) -> bool { self.m.cfg.is_moe() }



    /// Context size in tokens (per slot).
    pub fn ctx(&self) -> usize { self.ctx }

    fn slot_pos(&mut self, s: usize) -> &mut usize { if s == self.cur { &mut self.pos } else { &mut self.seqs[s].pos } }

    /// `kv_need` for slot `s`'s next token.
    pub fn kv_step(&mut self, s: usize) -> anyhow::Result<()> { let p = *self.slot_pos(s); self.kv_need(s, p + 1) }

    /// Make slot `s`'s attention KV hold `upto` positions, when the slots' KV grows on demand: it doubles, to at least
    /// `upto` and at most `ctx`, keeping its rows. Past `ctx` it leaves the "context full" error to the caller.
    fn kv_need(&mut self, s: usize, upto: usize) -> anyhow::Result<()> {
        if let Some(n) = self.next.as_mut() { n.kv_need(s, upto)?; }
        let Some(&cap) = self.kv_cap.get(s) else { return Ok(()) };
        if upto <= cap || upto > self.ctx { return Ok(()) }
        self.kv_resize(s, upto.max(2 * cap).next_multiple_of(self.kv0).min(self.ctx), true)
    }

    /// Give slot `s` attention KV caches of `new` positions (zeroed; `keep`: with the old caches' rows). Either all
    /// of them change or, out of VRAM, none. The recorded graphs, which hold the old caches' addresses, go.
    fn kv_resize(&mut self, s: usize, new: usize, keep: bool) -> anyhow::Result<()> {
        let (g, cap) = (&*self.gpu, self.kv_cap[s]);
        let mut fresh = vec![];
        for l in &self.layers {
            if let Mixer::Attn { kc, vc, .. } = &l.mixer {
                for b in [&kc[s], &vc[s]] {
                    fresh.push(g.alloc(b.len() / cap * new).map_err(|e| anyhow::anyhow!(
                        "no VRAM to grow a slot's KV cache to {new} positions (lower --parallel or --ctx): {e:#}"))?);
                }
            }
        }
        g.sync()?;
        let mut fresh = fresh.into_iter();
        for l in &mut self.layers {
            if let Mixer::Attn { kc, vc, .. } = &mut l.mixer {
                for b in [&mut kc[s], &mut vc[s]] {
                    let nb = fresh.next().unwrap();
                    if keep { g.dtod(b, 0, &nb, 0, b.len().min(nb.len()))?; }
                    *b = nb;
                }
            }
        }
        self.kv_cap[s] = new;
        g.sync()?;
        g.graph_clear()
    }

    /// Positions the current slot's attention KV holds.
    fn kv_rows(&self) -> usize { self.kv_cap.get(self.cur).copied().unwrap_or(self.ctx) }

    /// One decode step for each of up to `STEP_ROWS` distinct slots in one forward (each weight read once): `rows` holds
    /// (slot, token, whether to return its logits). Returns per row the greedy next token (argmax, first on ties) and
    /// the logits if asked (else empty: greedy rows copy back only the token). Each forward is graphed by its slots
    /// and attention regimes, so keep the rows in slot order.
    pub fn step_rows(&mut self, rows: &[(usize, u32, bool)]) -> anyhow::Result<Vec<(u32, Vec<f32>)>> {
        let (nt, nvoc) = (rows.len(), self.m.cfg.n_vocab);
        ensure!((1..=STEP_ROWS).contains(&nt) && self.mtp.is_none(), "step_rows: {nt} rows");
        let pos: Vec<usize> = rows.iter().map(|&(s, ..)| *self.slot_pos(s)).collect();
        if pos.iter().any(|&p| p >= self.ctx) { bail!("context full ({} tokens; raise --ctx)", self.ctx); }
        for (&(s, ..), &p) in rows.iter().zip(&pos) { self.kv_need(s, p + 1)?; }
        self.put_embed(&rows.iter().map(|r| r.1).collect::<Vec<_>>())?;
        let res = self.rows_pass(rows, &pos).and_then(|_| {
            let t = self.tail();
            let g = &t.gpu;
            g.argmax_rows(t.logits.addr(), nvoc, nt, t.amax.addr())?;
            let mut am = g.get::<i32>(&t.amax, 0, nt)?;
            if t.finish(nt, 0, None)? {
                g.argmax_rows(t.logits.addr(), nvoc, nt, t.amax.addr())?;
                am = g.get::<i32>(&t.amax, 0, nt)?;
            }
            rows.iter().zip(am).enumerate()
                .map(|(r, (&(.., want), t_))| Ok((t_ as u32, if want { g.get::<f32>(&t.logits, r * nvoc, nvoc)? } else { vec![] }))).collect()
        });
        let every = self.adapt_every;
        let swaps = self.adapt_swaps;
        let reset = self.reset_counts_after_prefill;
        let mut p = Some(&mut *self);
        while let Some(q) = p {
            q.rows.clear();
            if res.is_ok() {
                for &(s, ..) in rows { *q.slot_pos(s) += 1; }
                q.row_tokens += nt;
                q.adapt_every = every;
                q.adapt_swaps = swaps;
                q.reset_counts_after_prefill = reset;
                q.after_experts(nt, q.row_tokens)?;
            }
            p = q.next.as_deref_mut();
        }
        res
    }

    /// The forward of `step_rows` (input in `x`) through this part and the parts after it, as `pass`. Each forward
    /// is graphed by its slots and attention regimes; the last part's is left running.
    fn rows_pass(&mut self, rows: &[(usize, u32, bool)], pos: &[usize]) -> anyhow::Result<()> {
        let nt = rows.len();
        let seq = self.begin_seq();
        self.gpu.put(&pos.iter().flat_map(|&p| [p as i32, seq as i32]).collect::<Vec<_>>(), &self.step_buf, 0)?;
        self.gpu.put(&pos.iter().flat_map(|&p| [p as i32 + self.rope_offset, seq as i32]).collect::<Vec<_>>(), &self.rope_buf, 0)?;
        self.rows = rows.iter().zip(pos).map(|(r, &p)| (r.0, p)).collect();
        let mut key = 1 << 63 | nt as u64;
        for (r, &(sl, p)) in self.rows.iter().enumerate() { key |= (sl as u64) << (8 + 8 * r) | ((gq_min(p, 1) == 0) as u64) << (48 + r); }
        if self.m.loras.is_empty() { self.gpu.graph(key, &mut || self.forward(nt, 0, None))?; } else { self.forward(nt, 0, None)?; }
        if self.next.is_none() { return Ok(()); }
        self.gpu.sync()?;
        self.finish(nt, 0, None)?;
        let xs = self.gpu.get::<f32>(&self.x, 0, nt * self.m.cfg.n_embd)?;
        let nx = self.next.as_deref_mut().unwrap();
        nx.gpu.put(&xs, &nx.x, 0)?;
        nx.rows_pass(rows, pos)
    }

    /// This part and the parts after it.
    /// Layers per part: the split of the layers over the devices.
    pub fn part_layers(&self) -> Vec<usize> { self.parts().map(|p| p.layers.len()).collect() }
    pub fn parts(&self) -> impl Iterator<Item = &GpuModel<'a>> { std::iter::successors(Some(self), |p| p.next.as_deref()) }

    /// The part that computes the logits.
    fn tail(&self) -> &GpuModel<'a> { self.parts().last().unwrap() }
    /// `tail`, which also holds the MTP draft head.
    fn tail_mut(&mut self) -> &mut GpuModel<'a> {
        let mut p = self;
        while p.next.is_some() { p = p.next.as_deref_mut().unwrap(); }
        p
    }

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
        let seq = nx.begin_seq();
        let steps: Vec<i32> = (0..nt).flat_map(|t| [(nx.pos + t) as i32, seq as i32]).collect();
        nx.gpu.put(&steps, &nx.step_buf, 0)?;
        nx.rope_offset = self.rope_offset;
        nx.mrope = self.mrope;
        if self.mrope { nx.gpu.put(&self.gpu.get::<i32>(&self.mrope_buf, 0, 8)?, &nx.mrope_buf, 0)?; }
        nx.gpu.put(&(0..nt).flat_map(|t| [(nx.pos + t) as i32 + nx.rope_offset, seq as i32]).collect::<Vec<_>>(), &nx.rope_buf, 0)?;
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
            self.ple_undo = (self.ple_prev, tokens.to_vec());
            let pe: Vec<f32> = tokens.iter().flat_map(|&t| self.m.ple_embed(&mut self.ple_prev, t)).collect();
            self.gpu.put(&pe, &self.bufs["emb"], 0)?;
        }
        self.note_short_tokens(tokens);
        Ok(())
    }

    /// Run one token; returns logits over the vocabulary.
    pub fn step(&mut self, token: u32) -> anyhow::Result<Vec<f32>> {
        if self.pos >= self.ctx { bail!("context full ({} tokens; raise --ctx)", self.ctx); }
        self.kv_need(self.cur, self.pos + 1)?;
        self.dflash_commit_ctx()?;
        let prof = std::env::var_os("STRATA_VK_PROF").is_some();
        let t0 = prof.then(std::time::Instant::now);
        let seq = self.begin_seq();
        self.put_embed(&[token])?;
        self.gpu.put(&[self.pos as i32, seq as i32], &self.step_buf, 0)?;
        self.gpu.put(&[self.pos as i32 + self.rope_offset, seq as i32], &self.rope_buf, 0)?;
        let t1 = prof.then(std::time::Instant::now);
        self.pass(1)?;
        let t2 = prof.then(std::time::Instant::now);
        let t = self.tail();
        let mut logits = t.gpu.get::<f32>(&t.logits, 0, self.m.cfg.n_vocab)?;
        if t.finish(1, gq_min(self.pos, 1), None)? { logits = t.gpu.get::<f32>(&t.logits, 0, self.m.cfg.n_vocab)?; }
        let t3 = prof.then(std::time::Instant::now);
        self.after_forward(1)?;
        if let (Some(t0), Some(t1), Some(t2), Some(t3)) = (t0, t1, t2, t3) {
            let t4 = std::time::Instant::now();
            crate::say!("step_prof pos={}: embed_htod={:.1}us pass_submit={:.1}us gpu_wait_dtoh={:.1}us after_experts={:.1}us total={:.1}us",
                self.pos, (t1 - t0).as_secs_f64() * 1e6, (t2 - t1).as_secs_f64() * 1e6, (t3 - t2).as_secs_f64() * 1e6, (t4 - t3).as_secs_f64() * 1e6, (t4 - t0).as_secs_f64() * 1e6);
        }
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
        self.dflash_commit_ctx()?;
        let (nt, nvoc) = (tokens.len(), self.m.cfg.n_vocab);
        let prof = std::env::var_os("STRATA_VK_PROF").is_some();
        let t0 = prof.then(std::time::Instant::now);
        let seq = self.begin_seq();
        self.put_embed(tokens)?;
        let steps: Vec<i32> = (0..nt).flat_map(|t| [(self.pos + t) as i32, seq as i32]).collect();
        let rope: Vec<i32> = (0..nt).flat_map(|t| [(self.pos + t) as i32 + self.rope_offset, seq as i32]).collect();
        self.gpu.put(&steps, &self.step_buf, 0)?;
        self.gpu.put(&rope, &self.rope_buf, 0)?;
        let t1 = prof.then(std::time::Instant::now);
        self.pass(nt)?;
        let t2 = prof.then(std::time::Instant::now);
        let t = self.tail();
        let g = &t.gpu;
        g.argmax_rows(t.logits.addr(), nvoc, nt, t.amax.addr())?;
        let mut am: Vec<u32> = g.get::<i32>(&t.amax, 0, nt)?.into_iter().map(|x| x as u32).collect();
        if t.finish(nt, gq_min(self.pos, nt), None)? {
            g.argmax_rows(t.logits.addr(), nvoc, nt, t.amax.addr())?;
            am = g.get::<i32>(&t.amax, 0, nt)?.into_iter().map(|x| x as u32).collect();
        }
        let n = keep(&am);
        let t3 = prof.then(std::time::Instant::now);
        self.after_forward(nt)?;
        if let (Some(t0), Some(t1), Some(t2), Some(t3)) = (t0, t1, t2, t3) {
            let t4 = std::time::Instant::now();
            crate::say!("greedy_prof pos={}: embed_htod={:.1}us pass_submit={:.1}us gpu_wait_dtoh={:.1}us after_experts={:.1}us total={:.1}us",
                self.pos, (t1 - t0).as_secs_f64() * 1e6, (t2 - t1).as_secs_f64() * 1e6, (t3 - t2).as_secs_f64() * 1e6, (t4 - t3).as_secs_f64() * 1e6, (t4 - t0).as_secs_f64() * 1e6);
        }
        Ok((n, am[n]))
    }

    /// Enqueue the forward of `verify` (no logits copy).
    fn verify_run(&mut self, tokens: &[u32]) -> anyhow::Result<()> {
        let nt = tokens.len();
        if !(1..=self.tmax()).contains(&nt) { bail!("verify: {nt} tokens (1 to {})", self.tmax()); }
        if self.pos + nt > self.ctx { bail!("context full ({} tokens; raise --ctx)", self.ctx); }
        self.kv_need(self.cur, self.pos + nt)?;
        self.dflash_commit_ctx()?;
        let seq = self.begin_seq();
        self.put_embed(tokens)?;
        let steps: Vec<i32> = (0..nt).flat_map(|t| [(self.pos + t) as i32, seq as i32]).collect();
        let rope: Vec<i32> = (0..nt).flat_map(|t| [(self.pos + t) as i32 + self.rope_offset, seq as i32]).collect();
        self.gpu.put(&steps, &self.step_buf, 0)?;
        self.gpu.put(&rope, &self.rope_buf, 0)?;
        self.pass(nt)
    }

    /// Save the DeltaNet and conv states (KV needs no copy: rewinding `pos` is enough).
    pub fn save_state(&mut self) -> anyhow::Result<()> {
        self.dflash_commit_ctx()?;
        let g = &self.gpu;
        if self.saved.is_empty() { self.saved = alloc_saved(&**g, &self.layers, self.seqs.len() > 1)?; }
        let mut it = self.saved.iter();
        for l in &self.layers {
            if let Mixer::Delta { conv, s, .. } = &l.mixer {
                let (conv, s) = (&conv[self.cur], &s[self.cur]);
                let (bc, bs) = it.next().unwrap();
                g.dtod(conv, 0, bc, 0, conv.len())?;
                g.dtod(s, 0, bs, 0, s.len())?;
            }
        }
        if let Some(h) = &self.mtp {
            let src = h.eagle.as_ref().map_or(&self.x, |e| &e.cur_g);
            g.dtod(src, 0, &h.saved_x, 0, h.saved_x.len())?;
        }
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
            (q.saved_owner, q.sys_owner) = (q.owner, q.owner);
            p = q.next.as_deref_mut();
        }
        Ok(())
    }

    /// The KV caches of the attention layers (and of the draft head/DFlash) and their QSA pooled keys, whose rows are
    /// positions, each with the positions it holds.
    fn kv_bufs(&self) -> Vec<(&Buf, usize)> {
        let (rows, ctx) = (self.kv_rows(), self.ctx);
        self.layers.iter().map(|l| (l, rows)).chain(self.mtp.as_ref().map(|h| (&h.layer, ctx)))
            .flat_map(|(l, n)| match &l.mixer {
                Mixer::Attn { kc, vc, qsa, .. } => [(&kc[self.cur], n), (&vc[self.cur], n)].into_iter().chain(qsa.as_ref().map(|a| (&a.pool, ctx))).collect(),
                _ => vec![],
            })
            .chain(self.dflash.iter().flat_map(|df| df.layers.iter().flat_map(|dl| [(&dl.kc[self.cur], ctx), (&dl.vc[self.cur], ctx)])))
            .collect()
    }

    /// Snapshot the current saved GPU state into a host-memory `Parked` record if it fits in `cache_ram`.
    fn snapshot_saved(&self) -> anyhow::Result<Option<Parked>> {
        let n = self.saved_ids.len();
        if n == 0 { return Ok(None) }
        let mut size = 0;
        let mut p = Some(self);
        while let Some(q) = p {
            size += q.kv_bufs().iter().map(|(b, r)| b.len() / r * n).sum::<usize>() + q.slot().iter().map(|b| b.len()).sum::<usize>();
            p = q.next.as_deref();
        }
        if size > self.cache_ram { return Ok(None) }
        let mut parts = vec![];
        let mut p = Some(self);
        while let Some(q) = p {
            let g = &q.gpu;
            let slot = q.slot().iter().map(|b| g.get::<u8>(b, 0, b.len())).collect::<anyhow::Result<_>>()?;
            let kv = q.kv_bufs().iter().map(|(b, r)| g.get::<u8>(b, 0, b.len() / r * n)).collect::<anyhow::Result<_>>()?;
            parts.push(PartCopy { ple_prev: q.saved_ple_prev, slot, kv });
            p = q.next.as_deref();
        }
        Ok(Some(Parked { ids: self.saved_ids.clone(), owner: self.saved_owner, parts }))
    }

    fn insert_parked(&mut self, pk: Parked) {
        self.parked.retain(|old| !(old.owner == pk.owner && (pk.ids.starts_with(&old.ids) || old.ids == pk.ids)));
        self.parked.insert(0, pk);
        while self.parked.iter().map(Parked::bytes).sum::<usize>() > self.cache_ram { self.parked.pop(); }
    }

    /// Copy the saved state to host memory, so a later request that continues it can take it back (`unpark`); the
    /// least recently parked states go when they exceed `cache_ram`.
    fn park(&mut self) -> anyhow::Result<()> {
        if let Some(pk) = self.snapshot_saved()? {
            self.insert_parked(pk);
            if self.prompt_cache_path.is_some() {
                if let Err(e) = self.save_prompt_cache() {
                    crate::say!("warning: could not save prompt cache: {e:#}");
                }
            }
        }
        Ok(())
    }

    /// Snapshot the current saved state into `parked` and persist to `prompt_cache_path` (Warm Start).
    pub(crate) fn sync_saved_to_prompt_cache(&mut self) -> anyhow::Result<()> {
        if self.prompt_cache_path.is_none() { return Ok(()) }
        if let Some(pk) = self.snapshot_saved()? {
            self.insert_parked(pk);
            self.save_prompt_cache()?;
        }
        Ok(())
    }

    fn prompt_cache_fingerprint(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.m.cfg.n_layer.hash(&mut h);
        self.m.cfg.n_embd.hash(&mut h);
        self.m.cfg.n_vocab.hash(&mut h);
        self.m.cfg.n_head.hash(&mut h);
        self.m.cfg.n_head_kv.hash(&mut h);
        self.m.cfg.head_dim.hash(&mut h);
        self.m.cfg.ssm_d_inner.hash(&mut h);
        self.m.cfg.ssm_d_state.hash(&mut h);
        if let Some(q) = &self.qsa { q.ring.len().hash(&mut h); }
        let mut p = Some(self);
        while let Some(q) = p {
            q.layers.len().hash(&mut h);
            q.kv_bufs().len().hash(&mut h);
            for (b, r) in q.kv_bufs() { (b.len() / r).hash(&mut h); }
            p = q.next.as_deref();
        }
        h.finish()
    }

    /// Save up to 4 most recent `Parked` states to `self.prompt_cache_path` atomically.
    pub fn save_prompt_cache(&self) -> anyhow::Result<()> {
        use std::io::Write;
        let Some(path) = &self.prompt_cache_path else { return Ok(()) };
        if self.parked.is_empty() { return Ok(()) }
        if let Some(parent) = path.parent() && !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("tmp");
        let f = std::fs::File::create(&tmp)?;
        let mut w = std::io::BufWriter::with_capacity(8 << 20, f);
        w.write_all(b"STRATAPC")?;
        w.write_all(&1u32.to_le_bytes())?;
        w.write_all(&self.prompt_cache_fingerprint().to_le_bytes())?;
        let entries: Vec<&Parked> = self.parked.iter().take(4).collect();
        w.write_all(&(entries.len() as u32).to_le_bytes())?;
        for pk in entries {
            w.write_all(&pk.owner.to_le_bytes())?;
            w.write_all(&(pk.ids.len() as u32).to_le_bytes())?;
            for &id in &pk.ids {
                w.write_all(&id.to_le_bytes())?;
            }
            w.write_all(&(pk.parts.len() as u32).to_le_bytes())?;
            for part in &pk.parts {
                w.write_all(&part.ple_prev[0].to_le_bytes())?;
                w.write_all(&part.ple_prev[1].to_le_bytes())?;
                w.write_all(&(part.slot.len() as u32).to_le_bytes())?;
                for s in &part.slot {
                    w.write_all(&(s.len() as u64).to_le_bytes())?;
                    w.write_all(s)?;
                }
                w.write_all(&(part.kv.len() as u32).to_le_bytes())?;
                for k in &part.kv {
                    w.write_all(&(k.len() as u64).to_le_bytes())?;
                    w.write_all(k)?;
                }
            }
        }
        w.flush()?;
        drop(w);
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Enable Warm Start persistence at `path` and load any existing compatible states into `self.parked`.
    pub fn load_prompt_cache(&mut self, path: &std::path::Path) -> anyhow::Result<()> {
        use std::io::Read;
        self.prompt_cache_path = Some(path.to_path_buf());
        if !path.exists() {
            return Ok(());
        }
        let f = std::fs::File::open(path)?;
        let mut r = std::io::BufReader::with_capacity(8 << 20, f);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        ensure!(&magic == b"STRATAPC", "invalid magic in prompt cache {}", path.display());
        let mut u4 = [0u8; 4];
        let mut u8b = [0u8; 8];
        r.read_exact(&mut u4)?;
        let ver = u32::from_le_bytes(u4);
        ensure!(ver == 1, "unsupported prompt cache version {ver}");
        r.read_exact(&mut u8b)?;
        let fp = u64::from_le_bytes(u8b);
        let expected_fp = self.prompt_cache_fingerprint();
        ensure!(fp == expected_fp, "prompt cache fingerprint mismatch (model or --kv changed)");
        r.read_exact(&mut u4)?;
        let n_entries = u32::from_le_bytes(u4) as usize;
        let n_parts_expected = self.parts().count();
        let mut loaded = Vec::new();
        for _ in 0..n_entries {
            r.read_exact(&mut u8b)?;
            let owner = u64::from_le_bytes(u8b);
            r.read_exact(&mut u4)?;
            let n_ids = u32::from_le_bytes(u4) as usize;
            if n_ids == 0 || n_ids > self.ctx {
                bail!("cached sequence length {n_ids} exceeds --ctx {}", self.ctx);
            }
            let mut ids = Vec::with_capacity(n_ids);
            for _ in 0..n_ids {
                r.read_exact(&mut u4)?;
                ids.push(u32::from_le_bytes(u4));
            }
            r.read_exact(&mut u4)?;
            let n_parts = u32::from_le_bytes(u4) as usize;
            ensure!(n_parts == n_parts_expected, "prompt cache part count {n_parts} != {n_parts_expected}");
            let mut parts = Vec::with_capacity(n_parts);
            for _ in 0..n_parts {
                r.read_exact(&mut u8b)?;
                let p0 = i64::from_le_bytes(u8b);
                r.read_exact(&mut u8b)?;
                let p1 = i64::from_le_bytes(u8b);
                r.read_exact(&mut u4)?;
                let n_slot = u32::from_le_bytes(u4) as usize;
                let mut slot = Vec::with_capacity(n_slot);
                for _ in 0..n_slot {
                    r.read_exact(&mut u8b)?;
                    let len = u64::from_le_bytes(u8b) as usize;
                    let mut buf = vec![0u8; len];
                    r.read_exact(&mut buf)?;
                    slot.push(buf);
                }
                r.read_exact(&mut u4)?;
                let n_kv = u32::from_le_bytes(u4) as usize;
                let mut kv = Vec::with_capacity(n_kv);
                for _ in 0..n_kv {
                    r.read_exact(&mut u8b)?;
                    let len = u64::from_le_bytes(u8b) as usize;
                    let mut buf = vec![0u8; len];
                    r.read_exact(&mut buf)?;
                    kv.push(buf);
                }
                parts.push(PartCopy { ple_prev: [p0, p1], slot, kv });
            }
            loaded.push(Parked { ids, owner, parts });
        }
        let mut total_toks = 0usize;
        let mut total_bytes = 0usize;
        for pk in loaded.into_iter().rev() {
            total_toks += pk.ids.len();
            total_bytes += pk.bytes();
            self.insert_parked(pk);
        }
        if !self.parked.is_empty() {
            crate::say!(
                "warm start: loaded {} cached prefix state(s) ({total_toks} tokens, {:.1} MB) from {}",
                self.parked.len(),
                total_bytes as f64 / 1048576.0,
                path.display()
            );
        }
        Ok(())
    }

    /// Make parked state `i` the saved state and return to it; the saved state is parked in its place.
    fn unpark(&mut self, i: usize) -> anyhow::Result<()> {
        let p = self.parked.remove(i);
        self.park()?;
        self.kv_need(self.cur, p.ids.len())?;
        // A slot that never saved has no buffers to take the state yet.
        if self.saved.is_empty() { self.save_state()?; }
        let mut q = Some(&mut *self);
        for c in p.parts {
            let Some(m) = q else { break };
            for (b, d) in m.slot().iter().zip(&c.slot) { m.gpu.put(d, b, 0)?; }
            for ((b, _), d) in m.kv_bufs().iter().zip(&c.kv) { m.gpu.put(d, b, 0)?; }
            (m.saved_pos, m.saved_ple_prev, m.saved_ids, m.saved_owner) = (p.ids.len(), c.ple_prev, p.ids.clone(), p.owner);
            q = m.next.as_deref_mut();
        }
        self.restore_state()
    }

    /// Return to the state of the last `stash`, which also becomes the saved state.
    pub fn unstash(&mut self) -> anyhow::Result<()> {
        let mut p = Some(&mut *self);
        while let Some(q) = p {
            let s = q.sys.as_ref().ok_or_else(|| anyhow::anyhow!("unstash without stash"))?;
            for (b, d) in q.slot().iter().zip(&s.0) { q.gpu.put(d, b, 0)?; }
            (q.saved_pos, q.saved_ple_prev, q.saved_ids, q.saved_owner) = (s.1, s.2, s.3.clone(), q.sys_owner);
            p = q.next.as_deref_mut();
        }
        self.restore_state()
    }

    fn has_delta(&self) -> bool {
        self.parts().any(|p| p.layers.iter().any(|l| matches!(l.mixer, Mixer::Delta { .. })))
    }

    /// Keep only the first `keep` tokens of the last `verify`, which started at position `start` (1 <= keep <= `k`):
    /// recurrent states come from the snapshots it wrote, KV entries past them are dropped by rewinding `pos`.
    pub fn rollback(&mut self, start: usize, keep: usize) -> anyhow::Result<()> {
        // The PLE n-gram context goes back to the kept tokens. Its ring and the QSA ring are indexed by position
        // with more rows than a verify writes, so the rows a later token reads are still the kept ones.
        if self.ple_ring.is_some() {
            self.ple_prev = self.ple_undo.0;
            let kept: Vec<u32> = self.ple_undo.1.iter().take(keep).copied().collect();
            for t in kept { self.m.ple_embed(&mut self.ple_prev, t); }
        }
        if let Some(df) = self.dflash.as_mut() {
            df.pend_ctx = keep;
        }
        if !self.has_delta() {
            ensure!((1..=self.tmax()).contains(&keep) && start + keep <= self.pos, "rollback: keep {keep} of verify at {start}");
            self.pos = start + keep;
            if let Some(n) = self.next.as_mut() { n.rollback(start, keep)?; }
            return Ok(());
        }
        let nsnap = self.nsnap;
        if !(1..=nsnap).contains(&keep) || start + keep > self.pos { bail!("rollback: keep {keep} of verify at {start}"); }
        if self.layers.iter().any(|l| matches!(l.mixer, Mixer::Delta { .. })) {
            let g = &self.gpu;
            let stride = g.delta_snap_stride().max(1);
            let c = &self.m.cfg;
            let (dk, nk, nv) = (c.ssm_d_state, c.ssm_n_group, c.ssm_dt_rank);
            let (dv, channels) = (c.ssm_dv(), c.ssm_d_inner + 2 * nk * dk);
            let scale = 1.0 / (dk as f32).sqrt();
            let cp_tok = ((keep - 1) / stride) * stride;
            let cp_slot = cp_tok / stride;
            let rem = (keep - 1) - cp_tok;
            let mut do_rollback = || -> anyhow::Result<()> {
                g.batch(&mut || {
                    for l in &self.layers {
                        if let Mixer::Delta { conv, s, snap: Some((sc, ss, ..)), .. } = &l.mixer {
                            let (conv, s) = (&conv[self.cur], &s[self.cur]);
                            // Conv snapshots are kept for every verify position; DeltaNet state snapshots are strided every `stride`.
                            let (nc, ns) = (conv.len(), s.len());
                            g.dtod(sc, (keep - 1) * nc, conv, 0, nc)?;
                            g.dtod(ss, cp_slot * ns, s, 0, ns)?;
                        }
                    }
                    Ok(())
                })?;
                if rem > 0 {
                    let t0 = cp_tok + 1;
                    g.batch(&mut || {
                        for l in &self.layers {
                            if let Mixer::Delta { s, dt_bias, a, snap: Some((_, _, y_s, b_s, g_s)), .. } = &l.mixer {
                                g.deltanet(
                                    y_s.addr() + 4 * (t0 * channels) as u64, s[self.cur].addr(),
                                    b_s.addr() + 4 * (t0 * nv) as u64, g_s.addr() + 4 * (t0 * nv) as u64,
                                    dt_bias.ptr, a.ptr, 0,
                                    nk, nv, c.k_interleave, dk, dv, scale, rem, channels, 0, 0,
                                )?;
                            }
                        }
                        Ok(())
                    })?;
                }
                Ok(())
            };
            if self.m.loras.is_empty() {
                let key = 0x2FC0_0000_0000_0000u64 | self.graph_tag | (keep as u64) | ((self.cur as u64) << 16) | ((self.first as u64) << 24);
                g.graph(key, &mut do_rollback)?;
            } else {
                do_rollback()?;
            }
        }
        self.pos = start + keep;
        if let Some(n) = self.next.as_mut() { n.rollback(start, keep)?; }
        Ok(())
    }

    /// Keep the first `keep` tokens (`1 <= keep < toks.len()`) of a `verify` that started at `start`.
    /// Pure-attention models only rewind `pos`; DeltaNet models restore up to `SNAP_MAX` snapshots and replay any remainder.
    pub fn rollback_tokens(&mut self, start: usize, keep: usize, toks: &[u32]) -> anyhow::Result<()> {
        if let Some(df) = self.dflash.as_mut() {
            df.pend_ctx = keep;
        }
        if !self.has_delta() {
            return self.rollback(start, keep);
        }
        let nsnap = self.nsnap;
        if keep <= nsnap {
            return self.rollback(start, keep);
        }
        self.rollback(start, nsnap)?;
        let rem = &toks[nsnap..keep];
        self.verify_run(rem)?;
        let t = self.tail();
        let _ = t.finish(rem.len(), gq_min(self.pos, rem.len()), None)?;
        t.gpu.sync()?;
        self.after_forward(rem.len())
    }

    /// Return to the state of the last `save_state`.
    pub fn restore_state(&mut self) -> anyhow::Result<()> {
        let g = &self.gpu;
        let mut it = self.saved.iter();
        for l in &self.layers {
            if let Mixer::Delta { conv, s, .. } = &l.mixer {
                let (conv, s) = (&conv[self.cur], &s[self.cur]);
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
            let dst = h.eagle.as_ref().map_or(&self.x, |e| &e.cur_g);
            g.dtod(&h.saved_x, 0, dst, 0, h.saved_x.len())?;
            h.pos = self.pos.saturating_sub(1);
            h.pend.clear();
            h.adapt_window = 0x0FFF;
            h.adapt_skip = 0;
        }
        if let Some(df) = self.dflash.as_mut() {
            df.pos = self.pos;
            df.pend_ctx = 0;
        }
        if let Some(n) = self.next.as_mut() { n.restore_state()?; }
        Ok(())
    }

    /// Whether the draft head holds an entry for each token after the first (it is one entry behind the main model):
    /// prefill then extends it with the prompt, so its drafts see the whole context.
    fn mtp_synced(&self) -> bool { let t = self.tail(); t.mtp.as_ref().is_some_and(|h| h.pend.is_empty() && h.pos + 1 == t.pos.max(1)) }

    /// Run the forward of `nt` tokens as a recorded graph (graph key `nt` and the attention regime), recorded on first use.
    fn run(&self, nt: usize) -> anyhow::Result<()> {
        let gq = gq_min(self.pos, nt);
        if self.m.loras.is_empty() {
            // The RoPE path (fused, offset positions, M-RoPE) is part of the recorded ops.
            let rope = if self.mrope { 2 } else { (self.rope_offset != 0) as u64 };
            self.gpu.graph(self.graph_tag | nt as u64 | (gq as u32 as u64) << 8 | (self.cur as u64) << 40 | rope << 48, &mut || self.forward(nt, gq, None))
        } else { self.forward(nt, gq, None) }
    }

    /// Enqueue (or record into a graph) the forward of `nt` tokens; per-token values come from `step_buf`. `from`:
    /// resume at that layer's MoE combine (see `finish`).
    fn forward(&self, nt: usize, gq: i32, from: Option<usize>) -> anyhow::Result<()> {
        // A resume keeps the marks of layers that already ran; a fresh forward starts clean.
        if from.is_none() { self.lora_applied.borrow_mut().clear(); }
        let c = &self.m.cfg;
        let (g, eps, ne, step) = (&self.gpu, c.eps, c.n_embd, self.step_buf.addr());
        let p = |n: &str| self.bufs[n].addr();
        let x = self.x.addr();
        let vg = self.verify_gemm(nt);
        let mut res = 0;
        for (i, l) in self.layers.iter().enumerate().skip(from.map_or(0, |il| il - self.first)) {
            let il = self.first + i;
            if from != Some(il) {
                if let Some(eg) = self.mtp.as_ref().and_then(|h| h.eagle.as_ref()) {
                    if let Some(idx) = eg.cfg.extract_layers.iter().position(|&el| el == il) {
                        if res != 0 { g.add(x, res, nt * ne)?; res = 0; }
                        g.batch(&mut || {
                            for r in 0..nt {
                                g.dtod(&self.x, 4 * r * ne, &eg.feat, 4 * (r * 3 + idx) * ne, 4 * ne)?;
                            }
                            Ok(())
                        })?;
                    }
                }
            }
            res = self.layer(il, l, x, step, self.worker.as_ref(), nt, res, gq, from == Some(il))?;
            if let Some(df) = &self.dflash {
                if let Some(idx) = df.cfg.target_layer_ids.iter().position(|&tl| tl == il) {
                    if res != 0 { g.add(x, res, nt * ne)?; res = 0; }
                    let n_ids = df.cfg.target_layer_ids.len();
                    // hc targets (qwen4exp): the drafter reads the plain mean of the hc streams
                    // (mirrors llama.cpp's hc_stream_mean); hc_mix with a constant large gate
                    // yields sum * sigmoid(g) / hc = mean.
                    let src = if self.m.cfg.hc_n > 0 {
                        g.hc_mix(self.x.addr(), df.dfg.addr(), df.dfm.addr(), ne, self.m.cfg.hc_n, nt)?;
                        &df.dfm
                    } else {
                        &self.x
                    };
                    g.batch(&mut || {
                        for r in 0..nt {
                            g.dtod(src, 4 * r * ne, &df.feat, 4 * (r * n_ids + idx) * ne, 4 * ne)?;
                        }
                        Ok(())
                    })?;
                }
            }
        }
        if self.next.is_some() { return if res != 0 { g.add(x, res, nt * ne) } else { Ok(()) }; }
        let out_w = self.output.as_ref().unwrap();
        let out_gemm = vg && self.output_hc.is_none() && out_w.cols % 64 == 0;
        match (&self.output_norm, &self.output_hc) {
            (Some(on), _) => {
                let hq = if out_gemm { 0 } else { p("hq") };
                g.rms_norm_q(x, ne, on.ptr, p("h"), ne, ne, nt, eps, hq, res)?;
            }
            (None, Some(hr)) => self.hc_read(hr, x, nt, &self.hc_at())?,
            _ => unreachable!(),
        }
        if let Some(eg) = self.mtp.as_ref().and_then(|h| h.eagle.as_ref()) {
            self.mv(&eg.fc, eg.feat.addr(), 3 * ne, eg.cur_g.addr(), ne, nt)?;
        }
        if let Some(df) = &self.dflash {
            let n_ids = df.cfg.target_layer_ids.len();
            if vg && df.fc.cols % 64 == 0 {
                g.gemm_dense(&df.fc, nt, df.feat.addr(), n_ids * ne, df.x.addr(), ne, 0)?;
            } else {
                self.mv(&df.fc, df.feat.addr(), n_ids * ne, df.x.addr(), ne, nt)?;
            }
            g.rms_norm(df.x.addr(), ne, df.hidden_norm.ptr, df.th.addr(), ne, ne, nt, df.cfg.eps)?;
        }
        if out_gemm {
            g.gemm_dense(out_w, nt, p("h"), ne, self.logits.addr(), c.n_vocab, 0)
        } else {
            g.gemvq(out_w, p("h"), ne, p("hq"), self.logits.addr(), c.n_vocab, nt)
        }
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

    /// Last hidden state (before output_norm; the stack with hyper-connections) of the last `step`.
    pub fn hidden(&self) -> anyhow::Result<Vec<f32>> { let t = self.tail(); t.gpu.get::<f32>(&t.x, 0, self.row_width()) }

    /// MTP draft: logits for the token after `token`, from hidden state `row` of the last `step` or `verify` (the
    /// position that predicted `token`). Mirrors `Model::mtp_step`; the draft head keeps its own position and KV cache.
    pub fn mtp_draft(&mut self, token: u32, row: usize) -> anyhow::Result<Vec<f32>> {
        self.mtp_run_inner(&[token], self.row_ptr(row), true, false)?;
        let t = self.tail();
        Ok(t.gpu.get::<f32>(&t.logits, 0, self.m.cfg.n_vocab)?)
    }

    pub(super) fn dflash_commit_ctx(&mut self) -> anyhow::Result<()> {
        if self.next.is_some() {
            return self.tail_mut().dflash_commit_ctx();
        }
        let Some(df) = self.dflash.as_ref() else { return Ok(()) };
        if df.pend_ctx == 0 { return Ok(()); }
        let nt = df.pend_ctx;
        let start_pos = df.pos;
        anyhow::ensure!(start_pos + nt <= self.ctx, "dflash KV full ({} + {nt} > {})", start_pos, self.ctx);
        let steps: Vec<i32> = (0..nt).flat_map(|t| [(start_pos + t) as i32, 0]).collect();
        self.gpu.put(&steps, &df.ctx_step, 0)?;
        let key = 0xDFC0_0000_0000_0000u64 | nt as u64 | ((self.cur as u64) << 16);
        if self.m.loras.is_empty() {
            self.gpu.graph(key, &mut || self.dflash_commit_forward(nt))?;
        } else {
            self.dflash_commit_forward(nt)?;
        }
        let df = self.dflash.as_mut().unwrap();
        df.pos += nt;
        df.pend_ctx = 0;
        Ok(())
    }

    fn dflash_commit_forward(&self, nt: usize) -> anyhow::Result<()> {
        let df = self.dflash.as_ref().unwrap();
        let dc = &df.cfg;
        let g = &self.gpu;
        let (ne, nkv, hd, eps) = (dc.n_embd, dc.n_head_kv, dc.head_dim, dc.eps);
        let kv = nkv * hd;
        let use_gemm = std::env::var_os("STRATA_DFLASH_GEMM").is_some_and(|v| v != "0") && nt >= 8 && df.layers[0].k.cols % 64 == 0;
        let hq = self.bufs["hq"].addr();
        if !use_gemm {
            g.quant_xq(df.th.addr(), ne, ne, nt, hq)?;
        }
        let off = |li: usize| (4 * li * nt * kv) as u64;
        if use_gemm {
            for (li, dl) in df.layers.iter().enumerate() {
                let first_act = if li == 0 { 0 } else { crate::backend::SAME_X };
                let o = off(li);
                g.gemm_dense(&dl.k, nt, df.th.addr(), ne, df.k_ctx.addr() + o, kv, first_act)?;
                g.gemm_dense(&dl.v, nt, df.th.addr(), ne, df.v_ctx.addr() + o, kv, crate::backend::SAME_X)?;
                g.rms_norm_rope_kv(
                    df.k_ctx.addr() + o, hd, dl.kn.ptr, df.k.addr() + o,
                    hd, nt * nkv, eps, nkv, dc.n_rot, df.ctx_step.addr(), dc.rope_base,
                    df.v_ctx.addr() + o, dl.kc[self.cur].addr(), dl.vc[self.cur].addr(),
                )?;
            }
        } else {
            g.batch(&mut || {
                for (li, dl) in df.layers.iter().enumerate() {
                    let o = off(li);
                    g.gemvq2(&dl.k, &dl.v, df.th.addr(), ne, hq, df.k_ctx.addr() + o, df.v_ctx.addr() + o, nt)?;
                }
                Ok(())
            })?;
            g.batch(&mut || {
                for (li, dl) in df.layers.iter().enumerate() {
                    let o = off(li);
                    g.rms_norm_rope_kv(
                        df.k_ctx.addr() + o, hd, dl.kn.ptr, df.k.addr() + o,
                        hd, nt * nkv, eps, nkv, dc.n_rot, df.ctx_step.addr(), dc.rope_base,
                        df.v_ctx.addr() + o, dl.kc[self.cur].addr(), dl.vc[self.cur].addr(),
                    )?;
                }
                Ok(())
            })?;
        }
        Ok(())
    }

    fn dflash_block_forward(&self, b_len: usize, gq_swa: i32, gq_full: i32, greedy: bool) -> anyhow::Result<()> {
        let df = self.dflash.as_ref().unwrap();
        let dc = &df.cfg;
        let c = &self.m.cfg;
        let g = &self.gpu;
        let (ne, nh, nkv, hd, eps) = (dc.n_embd, dc.n_head, dc.n_head_kv, dc.head_dim, dc.eps);
        let kv = nkv * hd;
        let scale = 1.0 / (hd as f32).sqrt();
        let use_gemm = std::env::var_os("STRATA_DFLASH_GEMM").is_some_and(|v| v != "0") && b_len >= 8 && df.layers[0].q.cols % 64 == 0;
        let hq = self.bufs["hq"].addr();
        let swa_step = df.step_swa.addr();
        let full_step = df.step_full.addr();
        let mut res = 0u64;

        let conv_prepare = |conv: Option<&DFlashConv>, in_buf: u64, is_gemm: bool, quantize: bool| -> anyhow::Result<u64> {
            let Some(cv) = conv else { return Ok(in_buf); };
            if is_gemm && cv.proj.cols % 64 == 0 {
                g.gemm_dense(&cv.proj, b_len, in_buf, ne, df.conv_dyn.addr(), cv.proj.rows, 0)?;
            } else {
                if is_gemm { g.quant_xq(in_buf, ne, ne, b_len, hq)?; }
                g.gemvq(&cv.proj, in_buf, ne, hq, df.conv_dyn.addr(), cv.proj.rows, b_len)?;
            }
            g.dflash2_conv(in_buf, df.conv_dyn.addr(), cv.base.ptr, df.h2.addr(), ne, dc.conv_group_size, dc.conv_kernel_size, b_len, 0)?;
            if quantize { g.quant_xq(df.h2.addr(), ne, ne, b_len, hq)?; }
            Ok(df.h2.addr())
        };
        let conv_finish = |conv: Option<&DFlashConv>, in_buf: u64| -> anyhow::Result<u64> {
            let Some(cv) = conv else { return Ok(in_buf); };
            g.dflash2_conv(in_buf, df.conv_dyn.addr(), cv.base.ptr, df.h2.addr(), ne, dc.conv_group_size, dc.conv_kernel_size, b_len, 1)?;
            Ok(df.h2.addr())
        };

        for dl in &df.layers {
            if use_gemm {
                g.rms_norm_q(df.x.addr(), ne, dl.attn_norm.ptr, df.h.addr(), ne, ne, b_len, eps, 0, res)?;
                res = 0;
                let q_in = conv_prepare(dl.attn_conv.as_ref(), df.h.addr(), true, false)?;
                g.gemm_dense(&dl.q, b_len, q_in, ne, df.q.addr(), nh * hd, 0)?;
                g.batch(&mut || {
                    g.gemm_dense(&dl.k, b_len, q_in, ne, df.k_ctx.addr(), kv, crate::backend::SAME_X)?;
                    g.gemm_dense(&dl.v, b_len, q_in, ne, df.v_ctx.addr(), kv, crate::backend::SAME_X)
                })?;
            } else {
                g.rms_norm_q(df.x.addr(), ne, dl.attn_norm.ptr, df.h.addr(), ne, ne, b_len, eps, hq, res)?;
                res = 0;
                let q_in = conv_prepare(dl.attn_conv.as_ref(), df.h.addr(), false, true)?;
                g.batch(&mut || {
                    g.gemvq(&dl.q, q_in, ne, hq, df.q.addr(), nh * hd, b_len)?;
                    g.gemvq2(&dl.k, &dl.v, q_in, ne, hq, df.k_ctx.addr(), df.v_ctx.addr(), b_len)
                })?;
            }

            g.rms_norm_rope_qk(
                df.q.addr(), hd, dl.qn.ptr, df.q.addr(), b_len * nh, nh,
                df.k_ctx.addr(), dl.kn.ptr, df.k.addr(), b_len * nkv, nkv,
                hd, eps, dc.n_rot, swa_step, dc.rope_base,
                df.v_ctx.addr(), dl.kc[self.cur].addr(), dl.vc[self.cur].addr(),
            )?;

            let (attn_step, attn_gq, win) = if dl.is_swa && dc.conv_kernel_size == 0 {
                (swa_step, gq_swa, dc.sliding_window)
            } else if dl.is_swa {
                (full_step, gq_full, dc.sliding_window)
            } else {
                (full_step, gq_full, 0)
            };
            g.attn_decode_q(
                df.q.addr(), dl.kc[self.cur].addr(), dl.vc[self.cur].addr(), 0,
                df.att.addr(), df.attp.addr(), attn_step, nh, nkv, hd,
                scale, attn_gq, b_len, 0, 0, 0, 0, win,
            )?;

            if use_gemm && dl.o.cols % 64 == 0 {
                let post_res = if dl.attn_conv.is_some() {
                    g.gemm_dense(&dl.o, b_len, df.att.addr(), nh * hd, self.bufs["mix"].addr(), ne, 0)?;
                    conv_finish(dl.attn_conv.as_ref(), self.bufs["mix"].addr())?
                } else {
                    g.gemm_dense(&dl.o, b_len, df.att.addr(), nh * hd, df.x.addr(), ne, 2)?;
                    0
                };
                g.rms_norm_q(df.x.addr(), ne, dl.post_norm.ptr, df.h.addr(), ne, ne, b_len, eps, 0, post_res)?;
                let m_in = conv_prepare(dl.mlp_conv.as_ref(), df.h.addr(), true, false)?;
                g.gemm_dense(&dl.gate, b_len, m_in, ne, df.ge.addr(), dc.n_ff, 0)?;
                g.gemm_dense(&dl.up, b_len, m_in, ne, df.ge.addr(), dc.n_ff, 1 | crate::backend::SAME_X)?;
                if dl.down.cols % 64 == 0 && dl.mlp_conv.is_none() {
                    g.gemm_dense(&dl.down, b_len, df.ge.addr(), dc.n_ff, df.x.addr(), ne, 2)?;
                } else if dl.down.cols % 64 == 0 {
                    g.gemm_dense(&dl.down, b_len, df.ge.addr(), dc.n_ff, self.bufs["ye"].addr(), ne, 0)?;
                    res = conv_finish(dl.mlp_conv.as_ref(), self.bufs["ye"].addr())?;
                } else {
                    g.quant_xq(df.ge.addr(), dc.n_ff, dc.n_ff, b_len, self.bufs["ffq"].addr())?;
                    g.gemvq(&dl.down, df.ge.addr(), dc.n_ff, self.bufs["ffq"].addr(), self.bufs["ye"].addr(), ne, b_len)?;
                    res = conv_finish(dl.mlp_conv.as_ref(), self.bufs["ye"].addr())?;
                }
            } else {
                let attq = self.bufs["attq"].addr();
                g.quant_xq(df.att.addr(), nh * hd, nh * hd, b_len, attq)?;
                g.gemvq(&dl.o, df.att.addr(), nh * hd, attq, self.bufs["mix"].addr(), ne, b_len)?;
                let post_res = conv_finish(dl.attn_conv.as_ref(), self.bufs["mix"].addr())?;
                g.rms_norm_q(df.x.addr(), ne, dl.post_norm.ptr, df.h.addr(), ne, ne, b_len, eps, hq, post_res)?;
                let m_in = conv_prepare(dl.mlp_conv.as_ref(), df.h.addr(), false, true)?;
                g.batch(&mut || g.gemvq2(&dl.gate, &dl.up, m_in, ne, hq, df.ge.addr(), df.ue.addr(), b_len))?;
                let dq = if dl.down.ty.float_x() {
                    g.swiglu(df.ge.addr(), df.ue.addr(), b_len * dc.n_ff)?;
                    0
                } else {
                    g.swiglu_q(df.ge.addr(), df.ue.addr(), dc.n_ff, b_len, self.bufs["ffq"].addr())?
                };
                g.gemvq(&dl.down, df.ge.addr(), dc.n_ff, dq, self.bufs["ye"].addr(), ne, b_len)?;
                res = conv_finish(dl.mlp_conv.as_ref(), self.bufs["ye"].addr())?;
            }
        }

        let n_draft = b_len - 1;
        let x_draft = df.x.addr() + 4 * ne as u64;
        let res_draft = if res != 0 { res + 4 * ne as u64 } else { 0 };
        let out_w = self.output.as_ref().unwrap();
        let use_short = greedy && df.use_short.get() && df.short.is_some();
        let out_gemm = !use_short
            && std::env::var_os("STRATA_DFLASH_GEMM").is_some_and(|v| v != "0")
            && n_draft >= 8
            && out_w.cols % 64 == 0;
        g.rms_norm_q(x_draft, ne, df.norm.ptr, df.h.addr(), ne, ne, n_draft, eps, if out_gemm { 0 } else { hq }, res_draft)?;
        if let Some(sel) = &df.selector {
            if out_gemm {
                g.quant_xq(df.h.addr(), ne, ne, n_draft, hq)?;
            }
            g.gemvq(&sel.hidden_proj, df.h.addr(), ne, hq, sel.h_proj.addr(), sel.rank, n_draft)?;
        }
        if use_short {
            let sl = df.short.as_ref().unwrap();
            g.gemv_rows_n(out_w, df.h.addr(), ne, hq, sl.buf.addr(), sl.ids.len(), self.logits.addr(), n_draft)?;
            if df.selector.is_none() {
                g.argmax_rows(self.logits.addr(), sl.ids.len(), n_draft, self.amax.addr())?;
            }
        } else if out_gemm {
            g.gemm_dense(out_w, n_draft, df.h.addr(), ne, self.logits.addr(), c.n_vocab, 0)?;
            if greedy && df.selector.is_none() {
                g.argmax_rows(self.logits.addr(), c.n_vocab, n_draft, self.amax.addr())?;
            }
        } else {
            g.gemvq(out_w, df.h.addr(), ne, hq, self.logits.addr(), c.n_vocab, n_draft)?;
            if greedy && df.selector.is_none() {
                g.argmax_rows(self.logits.addr(), c.n_vocab, n_draft, self.amax.addr())?;
            }
        }
        Ok(())
    }

    /// Parallel block-diffusion speculative step using DFlash: generates `b_len - 1` draft tokens in one forward pass
    /// over `6` DFlash layers, verifies all `b_len` tokens `[t, d_1..]` on the target model in parallel, and
    /// preserves lossless output (`--temp 0` greedy / `--temp > 0` Leviathan residual rejection sampling).
    pub fn dflash_spec_step(&mut self, t: u32, smp: &mut Sampler) -> anyhow::Result<(Vec<u32>, u32, usize)> {
        let cap_delta = if self.has_delta() { self.nsnap } else { self.tmax() - 1 };
        let ov = std::env::var("STRATA_DFLASH_BLOCK").ok().and_then(|s| s.parse::<usize>().ok());
        let max_k = self.tail().dflash.as_ref().map_or(0, |df| {
            let blk = match ov {
                Some(v) => v,
                None => df.cur_block.get().min(self.k + 1),
            };
            blk.clamp(2, df.cfg.block_size) - 1
        }).min(cap_delta).min(self.tmax() - 1);
        let k_eff = self.ctx.saturating_sub(self.pos + 1).min(max_k);
        if k_eff == 0 || self.rope_offset != 0 || self.tail().dflash.is_none() {
            let next = if smp.temp <= 0.0 { self.step_greedy(t)? } else { smp.pick(&self.step(t)?)? as u32 };
            return Ok((vec![t], next, 0));
        }

        self.dflash_commit_ctx()?;
        let start = self.pos;
        let b_len = k_eff + 1;

        let (mask_id, emb_scale, out_mult) = {
            let df = self.tail().dflash.as_ref().unwrap();
            (df.cfg.mask_token_id, df.cfg.input_embedding_scale, df.cfg.output_multiplier)
        };
        let t_emb = self.m.embed(t);
        let mask_emb = self.m.embed(mask_id);
        let mut x_host = Vec::with_capacity(b_len * t_emb.len());
        if (emb_scale - 1.0).abs() < 1e-6 {
            x_host.extend_from_slice(&t_emb);
            for _ in 0..k_eff { x_host.extend_from_slice(&mask_emb); }
        } else {
            for &v in &t_emb { x_host.push(v * emb_scale); }
            for _ in 0..k_eff {
                for &v in &mask_emb { x_host.push(v * emb_scale); }
            }
        }
        let swa_steps: Vec<i32> = (0..b_len).flat_map(|i| [(start + i) as i32, 0]).collect();
        let full_pos = (start + b_len - 1) as i32;
        let full_steps: Vec<i32> = (0..b_len).flat_map(|_| [full_pos, 0]).collect();
        let gq_swa = gq_min(start, b_len);
        let gq_full = gq_min(start + b_len - 1, 1);
        let greedy = smp.temp <= 0.0;
        let use_short = greedy && self.tail().dflash.as_ref().is_some_and(|df| df.use_short.get() && df.short.is_some());

        {
            let tail = self.tail();
            let df = tail.dflash.as_ref().unwrap();
            tail.gpu.put(&x_host, &df.x, 0)?;
            tail.gpu.put(&swa_steps, &df.step_swa, 0)?;
            tail.gpu.put(&full_steps, &df.step_full, 0)?;
            let key = 0xDFB0_0000_0000_0000u64
                | (b_len as u64)
                | ((gq_swa == 0) as u64) << 8
                | ((gq_full == 0) as u64) << 9
                | (u64::from(greedy)) << 10
                | (u64::from(use_short)) << 11
                | ((tail.cur as u64) << 16);
            if tail.m.loras.is_empty() {
                tail.gpu.graph(key, &mut || tail.dflash_block_forward(b_len, gq_swa, gq_full, greedy))?;
            } else {
                tail.dflash_block_forward(b_len, gq_swa, gq_full, greedy)?;
            }
            // Optional looped multi-step DFlash refinement (`STRATA_DFLASH_STEPS=2..4`): re-anchors the prefix of
            // the diffusion block with Pass-1 predicted embeddings and re-runs the shared 6-layer block drafter.
            let refine_steps = std::env::var("STRATA_DFLASH_STEPS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(1)
                .clamp(1, 4);
            if greedy && refine_steps > 1 && k_eff >= 4 {
                let ne_df = t_emb.len();
                let short_ids = if use_short { df.short.as_ref().map(|s| s.ids.as_slice()) } else { None };
                for pass in 1..refine_steps {
                    let am_pass = tail.gpu.get::<i32>(&tail.amax, 0, k_eff)?;
                    let anchor = ((k_eff * pass) / refine_steps).clamp(1, k_eff - 1);
                    for i in 0..anchor {
                        let tok_i = short_ids.map_or(am_pass[i] as u32, |ids| ids[am_pass[i] as usize]);
                        let emb_i = self.m.embed(tok_i);
                        let dst = &mut x_host[(i + 1) * ne_df..(i + 2) * ne_df];
                        if (emb_scale - 1.0).abs() < 1e-6 {
                            dst.copy_from_slice(&emb_i);
                        } else {
                            for (d, &v) in dst.iter_mut().zip(&emb_i) {
                                *d = v * emb_scale;
                            }
                        }
                    }
                    tail.gpu.put(&x_host, &df.x, 0)?;
                    if tail.m.loras.is_empty() {
                        tail.gpu.graph(key, &mut || tail.dflash_block_forward(b_len, gq_swa, gq_full, greedy))?;
                    } else {
                        tail.dflash_block_forward(b_len, gq_swa, gq_full, greedy)?;
                    }
                }
            }
        }

        let n_voc = self.m.cfg.n_vocab;
        let mut toks = Vec::with_capacity(b_len);
        toks.push(t);
        let mut q_dists = Vec::new();
        if let Some(sel) = &self.tail().dflash.as_ref().unwrap().selector {
            let tail = self.tail();
            let short_ids = if use_short { tail.dflash.as_ref().and_then(|d| d.short.as_ref()).map(|s| s.ids.as_slice()) } else { None };
            let row_len = short_ids.map_or(n_voc, |ids| ids.len());
            let raw_logits = tail.gpu.get::<f32>(&tail.logits, 0, k_eff * row_len)?;
            let h_proj = tail.gpu.get::<f32>(&sel.h_proj, 0, k_eff * sel.rank)?;
            let (sel_toks, sel_qdists) = sel.select_path(t, k_eff, &raw_logits, row_len, short_ids, &h_proj, out_mult, smp)?;
            toks.extend_from_slice(&sel_toks);
            q_dists = sel_qdists;
        } else if greedy {
            let tail = self.tail();
            let am = tail.gpu.get::<i32>(&tail.amax, 0, k_eff)?;
            let short_ids = if use_short { tail.dflash.as_ref().and_then(|d| d.short.as_ref()).map(|s| s.ids.as_slice()) } else { None };
            for x in am {
                toks.push(short_ids.map_or(x as u32, |ids| ids[x as usize]));
            }
        } else {
            let tail = self.tail();
            let mut raw = tail.gpu.get::<f32>(&tail.logits, 0, k_eff * n_voc)?;
            if (out_mult - 1.0).abs() > 1e-6 {
                for v in &mut raw { *v *= out_mult; }
            }
            q_dists.reserve(k_eff);
            for row in raw.chunks_exact(n_voc) {
                let (d_i, q_i) = smp.draft_dist(row)?;
                toks.push(d_i);
                q_dists.push(q_i);
            }
        }

        let (n, next) = if greedy {
            self.verify_greedy(&toks, |am| (0..k_eff).take_while(|&i| am[i] == toks[i + 1]).count())?
        } else {
            let l = self.verify(&toks)?;
            let mut n = 0;
            loop {
                if n == k_eff { break (n, smp.pick(&l[n])? as u32); }
                match smp.accept_dist(&l[n], toks[n + 1] as usize, &q_dists[n])? {
                    None => n += 1,
                    Some(x) => break (n, x as u32),
                }
            }
        };

        self.drafts.0 += k_eff;
        self.drafts.1 += n;
        if n < k_eff { self.rollback(start, n + 1)?; }
        if ov.is_none()
            && let Some(df) = &self.tail().dflash
        {
            let cur = df.cur_block.get().min(df.cfg.block_size);  // not clamp(8, ..): a block size under 8 would panic
            if cur <= 8 {
                if n >= 6 || n == k_eff {
                    df.cur_block.set(12.min(df.cfg.block_size));
                }
                df.low_streak.set(0);
            } else if cur <= 12 {
                if n >= 9 || n == k_eff {
                    df.cur_block.set(16.min(df.cfg.block_size));
                    df.low_streak.set(0);
                } else if n <= 4 {
                    let s = df.low_streak.get() + 1;
                    if s >= 2 {
                        df.cur_block.set(8.min(df.cfg.block_size));
                        df.low_streak.set(0);
                    } else {
                        df.low_streak.set(s);
                    }
                } else {
                    df.low_streak.set(0);
                }
            } else if n <= 5 {
                let s = df.low_streak.get() + 1;
                if s >= 2 {
                    df.cur_block.set(12.min(df.cfg.block_size));
                    df.low_streak.set(0);
                } else {
                    df.low_streak.set(s);
                }
            } else {
                df.low_streak.set(0);
            }
        }
        toks.truncate(n + 1);
        Ok((toks, next, n))
    }

    /// Synchronize the standalone external draft model (`ext_draft`) with a fresh or cached prompt `tokens`.
    pub(super) fn ext_draft_sync_prompt(&mut self, tokens: &[u32]) -> anyhow::Result<()> {
        if self.next.is_some() {
            return self.tail_mut().ext_draft_sync_prompt(tokens);
        }
        let Some(ed) = self.ext_draft.as_mut() else { return Ok(()) };
        if tokens.is_empty() {
            ed.reset()?;
            self.ext_draft_pend.clear();
            return Ok(());
        }
        let prefix = &tokens[..tokens.len() - 1];
        let saved_len = ed.saved_ids.len();
        if saved_len > 0 && prefix.starts_with(&ed.saved_ids) && ed.saved_pos == saved_len {
            ed.restore_state()?;
            if prefix.len() > saved_len {
                ed.prefill_with(&prefix[saved_len..], &mut |_| Ok(()))?;
                ed.saved_ids.extend_from_slice(&prefix[saved_len..]);
                ed.save_state()?;
            }
        } else {
            ed.reset()?;
            if !prefix.is_empty() {
                ed.prefill_with(prefix, &mut |_| Ok(()))?;
                ed.saved_ids = prefix.to_vec();
                ed.save_state()?;
            }
        }
        self.ext_draft_pend = vec![tokens[tokens.len() - 1]];
        Ok(())
    }

    /// Speculative decode using a standalone external causal LM draft model (`--draft-model <model.gguf>`,
    /// e.g. `nvidia/Qwen3.5-4B-Qwen3.8-27B-Draft`). Preserves exact lossless output for both `--temp 0` (greedy)
    /// and `--temp > 0` (Leviathan residual rejection sampling).
    pub fn ext_draft_spec_step(&mut self, t: u32, smp: &mut Sampler) -> anyhow::Result<(Vec<u32>, u32, usize)> {
        let cap_delta = if self.has_delta() { self.nsnap } else { self.tmax() - 1 };
        let k_eff = self.ctx.saturating_sub(self.pos + 1).min(self.k).min(cap_delta).min(self.tmax() - 1);
        if k_eff == 0 || self.rope_offset != 0 || self.tail().ext_draft.is_none() {
            let next = if smp.temp <= 0.0 { self.step_greedy(t)? } else { smp.pick(&self.step(t)?)? as u32 };
            return Ok((vec![t], next, 0));
        }
        let greedy = smp.temp <= 0.0;
        let mut toks = Vec::with_capacity(k_eff + 1);
        toks.push(t);
        let mut q_dists = Vec::new();
        {
            let tail = self.tail_mut();
            let mut pend = std::mem::take(&mut tail.ext_draft_pend);
            pend.push(t);
            let ed = tail.ext_draft.as_mut().unwrap();
            if greedy {
                let first_d = if pend.len() == 1 {
                    ed.step_greedy(pend[0])?
                } else if pend.len() <= ed.tmax() {
                    ed.verify_greedy(&pend, |_| pend.len() - 1)?.1
                } else {
                    for &pt in &pend[..pend.len() - 1] {
                        let _ = ed.step_greedy(pt)?;
                    }
                    ed.step_greedy(*pend.last().unwrap())?
                };
                toks.push(first_d);
            } else {
                let logits = if pend.len() == 1 {
                    ed.step(pend[0])?
                } else if pend.len() <= ed.tmax() {
                    ed.verify(&pend)?.pop().unwrap()
                } else {
                    let mut last_l = Vec::new();
                    for &pt in &pend {
                        last_l = ed.step(pt)?;
                    }
                    last_l
                };
                let (d0, q0) = smp.draft_dist(&logits)?;
                toks.push(d0);
                q_dists.push(q0);
            }
            ed.saved_ids.extend_from_slice(&pend);
            ed.save_state()?;
            for i in 1..k_eff {
                let prev_d = toks[i];
                if greedy {
                    toks.push(ed.step_greedy(prev_d)?);
                } else {
                    let l = ed.step(prev_d)?;
                    let (di, qi) = smp.draft_dist(&l)?;
                    toks.push(di);
                    q_dists.push(qi);
                }
            }
        }
        let start = self.pos;
        let (n, next) = if greedy {
            self.verify_greedy(&toks, |am| (0..k_eff).take_while(|&i| am[i] == toks[i + 1]).count())?
        } else {
            let l = self.verify(&toks)?;
            let mut n = 0;
            loop {
                if n == k_eff { break (n, smp.pick(&l[n])? as u32); }
                match smp.accept_dist(&l[n], toks[n + 1] as usize, &q_dists[n])? {
                    None => n += 1,
                    Some(x) => break (n, x as u32),
                }
            }
        };
        self.drafts.0 += k_eff;
        self.drafts.1 += n;
        if n < k_eff {
            self.rollback(start, n + 1)?;
        }
        {
            let tail = self.tail_mut();
            let ed = tail.ext_draft.as_mut().unwrap();
            if k_eff > 1 {
                if n == k_eff {
                    ed.saved_ids.extend_from_slice(&toks[1..k_eff]);
                    ed.save_state()?;
                    tail.ext_draft_pend = vec![toks[k_eff]];
                } else {
                    ed.restore_state()?;
                    tail.ext_draft_pend = toks[1..=n].to_vec();
                }
            } else {
                tail.ext_draft_pend = toks[1..=n].to_vec();
            }
        }
        toks.truncate(n + 1);
        Ok((toks, next, n))
    }

    /// Speculative decode with MTP or DFlash. `t` is the next token (picked from the last logits), predicted by hidden row `row`.
    /// Drafts `k` tokens after `t`, verifies [t, drafts..] in one forward and keeps the drafts up to the first one the
    /// main model rejects (`rollback`). Greedy equals plain `step` decoding; sampling keeps the exact target distribution.
    pub fn spec_step(&mut self, t: u32, row: usize, smp: &mut Sampler) -> anyhow::Result<(Vec<u32>, u32, usize)> {
        if self.tail().dflash.is_some() {
            return self.dflash_spec_step(t, smp);
        }
        if self.tail().ext_draft.is_some() {
            return self.ext_draft_spec_step(t, smp);
        }
        let k = self.k;
        // No draft head (an image prompt dropped it, see `prefill_image`): plain steps.
        if self.pos + k + 1 > self.ctx || self.tail().mtp.is_none() {
            let next = if smp.temp <= 0.0 { self.step_greedy(t)? } else { smp.pick(&self.step(t)?)? as u32 };
            return Ok((vec![t], next, 0));
        }
        let partial_moe = self.parts().any(|p| p.cache.as_ref().is_some_and(|c| !c.fully_resident()))
            && std::env::var_os("STRATA_MTP_ADAPT").is_none_or(|v| v != "0");
        let h = self.tail_mut().mtp.as_mut().unwrap();
        let mut first = std::mem::take(&mut h.pend);
        if first.len() != row { first.clear(); }
        first.push(t);
        let nf = first.len();
        let hp = h.pos + nf - 1;
        let hits12 = (h.adapt_window & 0x0FFF).count_ones();
        let hits6 = (h.adapt_window & 0x3F).count_ones();
        if partial_moe && h.adapt_skip > 0 {
            h.adapt_skip -= 1;
            let hid = self.row_ptr(row + 1 - nf);
            self.mtp_run(&first, hid, false)?;
            self.tail_mut().mtp.as_mut().unwrap().pos = hp + 1;
            let next = if smp.temp <= 0.0 {
                self.step_greedy(t)?
            } else {
                smp.pick(&self.step(t)?)? as u32
            };
            return Ok((vec![t], next, 0));
        }
        let is_eagle = h.eagle.is_some();
        let k_eff = if partial_moe && k > 1 && hits12 < if is_eagle { 7 } else { 11 } { 1 } else { k };
        let mut toks = vec![t];
        let mut hid = self.row_ptr(row + 1 - nf);
        for i in 0..k_eff {
            let d = if i == 0 { self.mtp_run(&first, hid, true)? } else { self.mtp_run(&toks[i..=i], hid, true)? };
            hid = self.tail().mtp.as_ref().unwrap().x.addr() + 4 * ((if i == 0 { nf } else { 1 } - 1) * self.row_width()) as u64;
            toks.push(d);
        }
        let start = self.pos;
        let (n, next) = if smp.temp <= 0.0 {
            self.verify_greedy(&toks, |am| (0..k_eff).take_while(|&i| am[i] == toks[i + 1]).count())?
        } else {
            let l = self.verify(&toks)?;
            let mut n = 0;
            loop {
                if n == k_eff { break (n, smp.pick(&l[n])? as u32); }
                match smp.accept(&l[n], toks[n + 1] as usize)? { None => n += 1, Some(x) => break (n, x as u32) }
            }
        };
        self.drafts.0 += k_eff;
        self.drafts.1 += n;
        if n < k_eff { self.rollback(start, n + 1)?; }
        // Draft head entries past t were made from draft hidden states: the next call redoes them from main rows.
        let h = self.tail_mut().mtp.as_mut().unwrap();
        h.pos = hp + 1;
        h.pend = toks[1..=n].to_vec();
        h.adapt_window = ((h.adapt_window << 1) | u16::from(n > 0)) & 0x0FFF;
        if partial_moe && n == 0 && (if is_eagle { hits12 <= 3 && hits6 <= 1 } else { hits12 <= 8 || hits6 <= 3 }) {
            h.adapt_skip = if is_eagle { 2 } else { 3 };
        }
        toks.truncate(n + 1);
        Ok((toks, next, n))
    }

    /// Hybrid speculative decode combining context suffix/N-gram bursts (`NgramDrafter`) with DFlash / MTP / EAGLE3 drafting:
    /// uses zero-cost high-confidence suffix bursts when matched, falls back to `dflash_spec_step` / `spec_step` on misses,
    /// keeps DFlash KV / `MtpHead` synchronized across N-gram steps, and feeds verified tokens back into `NgramDrafter`.
    pub fn hybrid_spec_step(&mut self, t: u32, row: usize, smp: &mut Sampler) -> anyhow::Result<(Vec<u32>, u32, usize)> {
        let max_draft = if self.has_delta() {
            self.nsnap
        } else {
            self.ngram.as_ref().map_or(self.k, |ng| ng.cfg.draft_max).min(self.tmax() - 1)
        };
        let room = self.ctx.saturating_sub(self.pos + 1).min(max_draft);
        if self.tail().dflash.is_some() {
            let (draft, order) = match self.ngram.as_mut() {
                Some(ng) if room > 0 => ng.draft_with_order(t, room),
                _ => (Vec::new(), 0),
            };
            // Only bypass DFlash when suffix/ngram has a full-block match (`len >= 14 && order >= 8`).
            let strong = draft.len() >= 14 && order >= 8;
            if strong {
                self.dflash_commit_ctx()?;
                let m = draft.len();
                let mut toks = Vec::with_capacity(1 + m);
                toks.push(t);
                toks.extend_from_slice(&draft);
                let start = self.pos;
                let (n, next) = if smp.temp <= 0.0 {
                    self.verify_greedy(&toks, |am| (0..m).take_while(|&i| am[i] == toks[i + 1]).count())?
                } else {
                    let l = self.verify(&toks)?;
                    let mut n = 0;
                    loop {
                        if n == m { break (n, smp.pick(&l[n])? as u32); }
                        match smp.accept(&l[n], toks[n + 1] as usize)? { None => n += 1, Some(x) => break (n, x as u32) }
                    }
                };
                self.drafts.0 += m;
                self.drafts.1 += n;
                if n < m { self.rollback_tokens(start, n + 1, &toks)?; }
                toks.truncate(n + 1);
                if let Some(ng) = self.ngram.as_mut() { ng.accept(&toks); }
                return Ok((toks, next, n));
            }
            let (toks, next, rw) = self.dflash_spec_step(t, smp)?;
            if let Some(ng) = self.ngram.as_mut() { ng.accept(&toks); }
            return Ok((toks, next, rw));
        }
        let draft = match self.ngram.as_mut() {
            Some(ng) if room > 0 => ng.draft(t, room),
            _ => Vec::new(),
        };
        if draft.is_empty() {
            let (toks, next, rw) = self.spec_step(t, row, smp)?;
            if let Some(ng) = self.ngram.as_mut() { ng.accept(&toks); }
            return Ok((toks, next, rw));
        }
        if self.tail().ext_draft.is_some() {
            let m = draft.len();
            let mut toks = Vec::with_capacity(1 + m);
            toks.push(t);
            toks.extend_from_slice(&draft);
            let start = self.pos;
            let (n, next) = if smp.temp <= 0.0 {
                self.verify_greedy(&toks, |am| (0..m).take_while(|&i| am[i] == toks[i + 1]).count())?
            } else {
                let l = self.verify(&toks)?;
                let mut n = 0;
                loop {
                    if n == m { break (n, smp.pick(&l[n])? as u32); }
                    match smp.accept(&l[n], toks[n + 1] as usize)? { None => n += 1, Some(x) => break (n, x as u32) }
                }
            };
            self.drafts.0 += m;
            self.drafts.1 += n;
            if n < m { self.rollback_tokens(start, n + 1, &toks)?; }
            toks.truncate(n + 1);
            self.tail_mut().ext_draft_pend.extend_from_slice(&toks);
            if let Some(ng) = self.ngram.as_mut() { ng.accept(&toks); }
            return Ok((toks, next, n));
        }
        if self.tail().mtp.is_none() { let (toks, next) = self.ngram_spec_step(t, smp)?; return Ok((toks, next, 0)); }
        // Advance the draft head over [pend.., t] using previous main hidden states before `verify` overwrites them.
        let h = self.tail_mut().mtp.as_mut().unwrap();
        let mut first = std::mem::take(&mut h.pend);
        if first.len() != row { first.clear(); }
        first.push(t);
        let nf = first.len();
        let hp = h.pos + nf - 1;
        let hid = self.row_ptr(row + 1 - nf);
        self.mtp_run(&first, hid, false)?;

        let m = draft.len();
        let mut toks = Vec::with_capacity(1 + m);
        toks.push(t);
        toks.extend_from_slice(&draft);
        let start = self.pos;
        let (n, next) = if smp.temp <= 0.0 {
            self.verify_greedy(&toks, |am| (0..m).take_while(|&i| am[i] == toks[i + 1]).count())?
        } else {
            let l = self.verify(&toks)?;
            let mut n = 0;
            loop {
                if n == m { break (n, smp.pick(&l[n])? as u32); }
                match smp.accept(&l[n], toks[n + 1] as usize)? { None => n += 1, Some(x) => break (n, x as u32) }
            }
        };
        self.drafts.0 += m;
        self.drafts.1 += n;
        if n < m { self.rollback(start, n + 1)?; }
        let h = self.tail_mut().mtp.as_mut().unwrap();
        h.pos = hp + 1;
        h.pend = toks[1..=n].to_vec();
        if n > 0 {
            h.adapt_window = ((h.adapt_window << 1) | 1) & 0x0FFF;
            h.adapt_skip = 0;
        }
        toks.truncate(n + 1);
        if let Some(ng) = self.ngram.as_mut() { ng.accept(&toks); }
        Ok((toks, next, n))
    }

    /// Speculative decode with context N-gram / lookup cache drafting (`NgramDrafter`).
    /// When no N-gram match is found (`draft.is_empty()`), executes a normal 1-token decode step (`step_greedy` or `step`)
    /// with zero GPU overhead. When `m > 0` draft tokens are proposed, verifies `[t, draft..]` in a single pass and
    /// rolls back any rejected suffix via `rollback_tokens`.
    pub fn ngram_spec_step(&mut self, t: u32, smp: &mut Sampler) -> anyhow::Result<(Vec<u32>, u32)> {
        let max_draft = if self.has_delta() && self.nsnap == 0 {
            0
        } else {
            self.ngram.as_ref().map_or(self.k, |ng| ng.cfg.draft_max.max(self.k)).min(self.tmax() - 1)
        };
        let room = self.ctx.saturating_sub(self.pos + 1).min(max_draft);
        let draft = match self.ngram.as_mut() {
            Some(ng) if room > 0 => ng.draft(t, room),
            _ => Vec::new(),
        };
        if draft.is_empty() {
            if let Some(ng) = self.ngram.as_mut() { ng.accept(&[t]); }
            let next = if smp.temp <= 0.0 {
                self.step_greedy(t)?
            } else {
                smp.pick(&self.step(t)?)? as u32
            };
            return Ok((vec![t], next));
        }
        let m = draft.len();
        let mut toks = Vec::with_capacity(1 + m);
        toks.push(t);
        toks.extend_from_slice(&draft);
        let start = self.pos;
        let (n, next) = if smp.temp <= 0.0 {
            self.verify_greedy(&toks, |am| (0..m).take_while(|&i| am[i] == toks[i + 1]).count())?
        } else {
            let l = self.verify(&toks)?;
            let mut n = 0;
            loop {
                if n == m { break (n, smp.pick(&l[n])? as u32); }
                match smp.accept(&l[n], toks[n + 1] as usize)? { None => n += 1, Some(x) => break (n, x as u32) }
            }
        };
        self.drafts.0 += m;
        self.drafts.1 += n;
        if n < m {
            self.rollback_tokens(start, n + 1, &toks)?;
        }
        toks.truncate(n + 1);
        if let Some(ng) = self.ngram.as_mut() {
            ng.accept(&toks);
        }
        Ok((toks, next))
    }

    /// Force-feed `tokens` sequentially while keeping `MtpHead` (`row` / `pend` / `pos`) and `NgramDrafter`
    /// synchronized with the KV cache. Used to inject `</think>\n\n` when `--thinking-budget` expires.
    pub fn force_tokens(&mut self, tokens: &[u32], row: &mut usize) -> anyhow::Result<Vec<f32>> {
        let mut logits = Vec::new();
        for &t in tokens {
            if self.mtp.is_some() {
                let mut first = std::mem::take(&mut self.mtp.as_mut().unwrap().pend);
                if first.len() != *row { first.clear(); }
                first.push(t);
                let nf = first.len();
                let hp = self.mtp.as_ref().unwrap().pos + nf - 1;
                let hid = self.row_ptr(*row + 1 - nf);
                self.mtp_run(&first, hid, false)?;
                self.mtp.as_mut().unwrap().pos = hp + 1;
                *row = 0;
            }
            if self.tail().ext_draft.is_some() {
                self.tail_mut().ext_draft_pend.push(t);
            }
            logits = self.step(t)?;
        }
        if let Some(ng) = self.ngram.as_mut() { ng.accept(tokens); }
        Ok(logits)
    }

    /// Address of hidden row `row` of the last `step` or `verify`.
    fn row_ptr(&self, row: usize) -> u64 {
        let t = self.tail();
        let base = t.mtp.as_ref().and_then(|h| h.eagle.as_ref()).map_or_else(|| t.x.addr(), |e| e.cur_g.addr());
        base + 4 * (row * self.row_width()) as u64
    }

    /// Floats per hidden row: n_embd, or the whole stack with hyper-connections.
    fn row_width(&self) -> usize { self.m.cfg.hc_n.max(1) * self.m.cfg.n_embd }

    /// Draft head on `tokens` with hidden states at `hid` (rows of n_embd), as one batch at the next draft positions;
    /// returns the argmax of the last token's logits (left in `logits` row 0), or 0 when `head` = false skips the head.
    fn mtp_run(&mut self, tokens: &[u32], hid: u64, head: bool) -> anyhow::Result<u32> {
        self.mtp_run_inner(tokens, hid, head, true)
    }

    fn mtp_run_inner(&mut self, tokens: &[u32], hid: u64, head: bool, allow_short: bool) -> anyhow::Result<u32> {
        if self.next.is_some() { return self.tail_mut().mtp_run_inner(tokens, hid, head, allow_short); }
        let nt = tokens.len();
        let c = &self.m.cfg;
        let Some(h) = self.mtp.as_mut() else { bail!("MTP head not loaded") };
        if h.pos + nt > self.ctx { bail!("context full ({} tokens; raise --ctx)", self.ctx); }
        let fast = h.eagle.is_some() || (self.cache.as_ref().is_some_and(|c| c.fully_resident()) && self.gpu.moe_mailbox_optional());
        let short_on = head && allow_short && h.hc_head.is_none() && h.eagle.is_none() && h.short.is_some() && h.use_short.get();
        let seq = if fast { 0 } else { h.worker.as_mut().unwrap().begin() };
        let g = &self.gpu;
        let emb: Vec<f32> = tokens.iter().flat_map(|&t| self.m.embed(t)).collect();
        g.put(&emb, &h.emb, 0)?;
        let steps: Vec<i32> = (0..nt).flat_map(|t| [(h.pos + t) as i32, seq as i32]).collect();
        g.put(&steps, &h.step_buf, 0)?;
        // `run` keys are token counts (below 8); the top bit keeps these keys apart from them.
        let gq = gq_min(h.pos, nt);
        let key = { let mut s = std::hash::DefaultHasher::new(); (hid, nt, head, gq, self.cur, short_on).hash(&mut s); s.finish() | 1 << 63 };
        if self.m.loras.is_empty() { g.graph(key, &mut || self.mtp_forward(hid, nt, head, gq, false, short_on))?; } else { self.mtp_forward(hid, nt, head, gq, false, short_on)?; }
        let is_eagle = self.mtp.as_ref().is_some_and(|h| h.eagle.is_some());
        let mut d = if head { g.get::<i32>(&self.amax, 0, 1)?[0] as u32 } else { if !is_eagle { g.sync()?; } 0 };
        if self.finish(nt, gq, Some((hid, head, short_on)))? && head { d = self.gpu.get::<i32>(&self.amax, 0, 1)?[0] as u32; }
        let h = self.mtp.as_mut().unwrap();
        h.pos += nt;
        if let Some(eg) = &h.eagle {
            if head { d = eg.cfg.d2t.get(d as usize).copied().unwrap_or(d as i32) as u32; }
            return Ok(d);
        }
        if short_on { d = h.short.as_ref().unwrap().ids.get(d as usize).copied().unwrap_or(d); }
        // Fast path: every expert of the draft layer is resident, so nothing can miss and there is nothing to record
        // (R2). The routing counters keep the head's share of the hit rate.
        if fast {
            self.hits += c.n_expert_used * nt;
        } else if let Some(cache) = self.cache.as_mut() {
            // One mailbox read per layer, as in `after_experts` (each is a bulk copy of the routing words).
            let r = h.worker.as_mut().unwrap().routing(c.n_layer);
            self.mailbox_reads.set(self.mailbox_reads.get() + 1);
            let (hits, misses) = tally(&r);
            self.hits += hits;
            self.misses += misses;
            debug_assert!(misses == 0 || !cache.fully_resident(), "a missed expert in a fully resident cache");
            cache.record(c.n_layer, &r.iter().map(|x| x.0).collect::<Vec<_>>());
        }
        Ok(d)
    }

    /// After a forward and its wait: while a MoE combine gave up waiting for the CPU worker (which may be reading
    /// experts from a cold drive; one GPU submission must not outlast the driver's job timeout), wait for the worker
    /// and run the rest of the forward from that combine. `mtp`: the draft head's forward (`mtp_run`'s hid, head)
    /// instead of the model's. Returns whether anything ran again.
    fn finish(&self, nt: usize, gq: i32, mtp: Option<(u64, bool, bool)>) -> anyhow::Result<bool> {
        let mut again = false;
        let mut last = None;
        while self.gpu.stalled() {
            let w = match mtp { Some(_) => self.mtp.as_ref().unwrap().worker.as_ref().unwrap(), None => self.worker.as_ref().unwrap() };
            let il = w.stalled_layer().context("a MoE combine gave up waiting, but no layer says so")?;
            // The worker had answered the last time: the GPU does not see the mailbox.
            ensure!(last != Some(il), "moe_combine: the GPU does not see the CPU expert worker's answers. On an AMD RX 9000 \
                (RDNA4) GPU set radv_device_coherent_memory=true in the environment");
            last = Some(il);
            w.wait_done(il)?;
            match mtp {
                Some((hid, head, short_on)) => self.mtp_forward(hid, nt, head, gq, true, short_on)?,
                None => self.forward(nt, gq, Some(il))?,
            }
            self.gpu.sync()?;
            self.resumed.set(self.resumed.get() + 1);
            again = true;
        }
        Ok(again)
    }

    /// Enqueue the draft head (see `mtp_run`).
    fn mtp_forward(&self, hid: u64, nt: usize, head: bool, gq: i32, resume: bool, short_on: bool) -> anyhow::Result<()> {
        let (c, g, h) = (&self.m.cfg, &self.gpu, self.mtp.as_ref().unwrap());
        if let Some(eg) = &h.eagle {
            let ec = &eg.cfg;
            let (eps, ne, hp) = (ec.eps, c.n_embd, self.bufs["h"].addr());
            let (eh, x) = (h.eh.addr(), h.x.addr());
            let (nh, nkv, hd) = (ec.n_head, ec.n_head_kv, ec.head_dim);
            let kv = nkv * hd;
            g.rms_norm(h.emb.addr(), ne, h.enorm.ptr, eh, 2 * ne, ne, nt, eps)?;
            g.rms_norm(hid, ne, h.hnorm.ptr, eh + 4 * ne as u64, 2 * ne, ne, nt, eps)?;
            if ec.norm_before_residual {
                g.rms_norm(hid, ne, h.hnorm.ptr, x, ne, ne, nt, eps)?;
            } else if hid != x {
                g.memset(&h.x)?;
                g.add(x, hid, nt * ne)?;
            }
            let Mixer::Attn { q, k, v, o, kc, vc, .. } = &h.layer.mixer else { unreachable!() };
            let Ffn::Dense { gate, up, down } = &h.layer.ffn else { unreachable!() };
            g.batch(&mut || {
                self.mv(q, eh, 2 * ne, eg.q.addr(), nh * hd, nt)?;
                self.mv(k, eh, 2 * ne, eg.k.addr(), kv, nt)?;
                self.mv(v, eh, 2 * ne, eg.v.addr(), kv, nt)
            })?;
            g.batch(&mut || {
                g.rope_neox(eg.q.addr(), hd, nt * nh, nh, ec.n_rot, h.step_buf.addr(), ec.rope_base)?;
                g.rope_neox(eg.k.addr(), hd, nt * nkv, nkv, ec.n_rot, h.step_buf.addr(), ec.rope_base)
            })?;
            g.kv_store(eg.k.addr(), eg.v.addr(), kc[self.cur].addr(), vc[self.cur].addr(), kv, nt, h.step_buf.addr(), hd)?;
            if !(crate::host_attn::on_host(self.m.cfg.n_layer)
                && g.attn_host(eg.q.addr(), kc[self.cur].addr(), vc[self.cur].addr(), 0, eg.att.addr(), h.step_buf.addr(), nh, nkv, hd, 1.0 / (hd as f32).sqrt(), nt)?) {
                g.attn_decode_q(eg.q.addr(), kc[self.cur].addr(), vc[self.cur].addr(), 0, eg.att.addr(),
                    eg.attp.addr(), h.step_buf.addr(), nh, nkv, hd, 1.0 / (hd as f32).sqrt(), gq, nt, 0, 0, 0, 0, 0)?;
            }
            self.mv(o, eg.att.addr(), nh * hd, self.bufs["mix"].addr(), ne, nt)?;
            g.rms_norm_q(x, ne, h.layer.post_norm.as_ref().unwrap().ptr, hp, ne, ne, nt, eps, 0, self.bufs["mix"].addr())?;
            g.batch(&mut || {
                self.mv(gate, hp, ne, eg.ge.addr(), ec.n_ff, nt)?;
                self.mv(up, hp, ne, eg.ue.addr(), ec.n_ff, nt)
            })?;
            g.swiglu(eg.ge.addr(), eg.ue.addr(), nt * ec.n_ff)?;
            self.mv(down, eg.ge.addr(), ec.n_ff, self.bufs["ye"].addr(), ne, nt)?;
            g.add(x, self.bufs["ye"].addr(), nt * ne)?;
            if head {
                g.rms_norm(x + 4 * ((nt - 1) * ne) as u64, 0, h.head_norm.as_ref().unwrap().ptr, hp, 0, ne, 1, eps)?;
                g.gemv(&eg.output, hp, self.logits.addr())?;
                g.argmax_rows(self.logits.addr(), ec.draft_vocab, 1, self.amax.addr())?;
            }
            return Ok(());
        }
        let (eps, ne, hp) = (c.eps, c.n_embd, self.bufs["h"].addr());
        let (eh, x) = (h.eh.addr(), h.x.addr());
        // Hyper-connections (llama.cpp qwen4exp `graph_mtp`): every stream of the stack at `hid` is projected on its
        // own from [enorm(embed), hnorm(stream)]; `hnorm` is hc_n gammas. Rows of eh and x are hc times as wide.
        let hc = c.hc_n.max(1);
        let (d, e2) = (hc * ne, hc * 2 * ne);
        if !resume {
            for s in 0..hc {
                let (so, eo) = (4 * (s * ne) as u64, eh + 4 * (s * 2 * ne) as u64);
                g.rms_norm(h.emb.addr(), ne, h.enorm.ptr, eo, e2, ne, nt, eps)?;
                g.rms_norm(hid + so, d, h.hnorm.ptr + so, eo + 4 * ne as u64, e2, ne, nt, eps)?;
                self.mv(h.eh_proj.as_ref().unwrap(), eo, e2, x + so, d, nt)?;
            }
        }
        let res = self.layer(c.n_layer, &h.layer, x, h.step_buf.addr(), h.worker.as_ref(), nt, 0, gq, resume)?;
        if res != 0 { g.add(x, res, nt * ne)?; }
        if head {
            let last = x + 4 * ((nt - 1) * d) as u64;
            if short_on {
                let sl = h.short.as_ref().unwrap();
                g.rms_norm(last, 0, h.head_norm.as_ref().unwrap().ptr, hp, 0, ne, 1, eps)?;
                g.gemv_rows(self.output.as_ref().unwrap(), hp, sl.buf.addr(), sl.ids.len(), self.logits.addr())?;
                g.argmax_rows(self.logits.addr(), sl.ids.len(), 1, self.amax.addr())?;
            } else {
                match &h.hc_head {
                    Some(hr) => {
                        self.hc_read(hr, last, 1, &self.hc_at())?;
                        g.gemvq(self.output.as_ref().unwrap(), hp, ne, self.bufs["hq"].addr(), self.logits.addr(), c.n_vocab, 1)?;
                    }
                    None => {
                        g.rms_norm(last, 0, h.head_norm.as_ref().unwrap().ptr, hp, 0, ne, 1, eps)?;
                        g.gemv(self.output.as_ref().unwrap(), hp, self.logits.addr())?;
                    }
                }
                g.argmax_rows(self.logits.addr(), c.n_vocab, 1, self.amax.addr())?;
            }
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
        let rope = self.rope_buf.addr();
        // Inputs from "h" are quantized once ("hq", after each norm) instead of once per GEMV block.
        let mv = |w: &GpuTensor, x: &str, xs: usize, y: &str, ys: usize|
            if x == "h" { g.gemvq(w, p(x), xs, p("hq"), p(y), ys, nt) } else { self.mv(w, p(x), xs, p(y), ys, nt) };
        let mv2 = |a: &GpuTensor, b: &GpuTensor, ya: &str, yb: &str| g.gemvq2(a, b, p("h"), ne, p("hq"), p(ya), p(yb), nt);
        // The SwiGLU for a down GEMV of type `ty` (float weights read g itself). Returns the xq for the GEMV.
        let sw = |ty: GgmlType, ga: &str, ua: &str, n: usize, rows: usize, q: &str| -> anyhow::Result<u64> {
            if ty.float_x() { g.swiglu(p(ga), p(ua), rows * n)?; return Ok(0); }
            g.swiglu_q(p(ga), p(ua), n, rows, p(q))
        };
        let (dk, nk, nv) = (c.ssm_d_state, c.ssm_n_group, c.ssm_dt_rank);
        let (dv, channels) = (c.ssm_dv(), c.ssm_d_inner + 2 * nk * dk);
        let (hd, nh, nkv) = (c.head_dim, c.n_head, c.n_head_kv);
        let kv = nkv * hd;
        let sink_addr = match &l.mixer { Mixer::Attn { sink: Some(s), .. } => s.ptr, _ => 0 };
        let win = c.swa.get(il).copied().unwrap_or(false).then_some(c.n_swa).unwrap_or(0);
        let use_rope = c.rope_layer.get(il).copied().unwrap_or(true);
        let vg = self.verify_gemm(nt) && l.hc.is_none() && l.mhc.is_none();
        let mixer_gemm = vg && ne % 64 == 0 && !matches!(&l.mixer, Mixer::Attn { qsa: Some(_), .. });
        if !resume {
            let ha = self.hc_at();
            if let Some(pl) = &l.ple { self.ple(pl, x, step, nt, &ha)?; }
            if let Some((mk, mhc_a, _)) = &l.mhc {
                let s = c.mhc_streams;
                let d = s * ne;
                if mk & 1 != 0 {
                    if res != 0 { g.add(x, res, nt * ne)?; }
                    g.hc_write(p("mr"), x, 0, ne, 0x100 | 0x400 | s, nt, false)?;
                }
                let ones = self.mhc_ones.as_ref().unwrap().addr();
                g.rms_norm(p("mr"), d, ones, p("xn"), d, d, nt, 1e-6)?;
                self.mv(&mhc_a.weight, p("xn"), d, mhc_a.buf.addr() + 128, mhc_a.out_dim, nt)?;
                let code_a = 0x100 | s | (mhc_a.out_dim << 16);
                g.hc_mix(p("mr"), mhc_a.buf.addr(), p("h"), ne, code_a, nt)?;
                g.rms_norm_q(p("h"), ne, l.attn_norm.as_ref().unwrap().ptr, p("h"), ne, ne, nt, eps, if mixer_gemm { 0 } else { p("hq") }, 0)?;
            } else {
                match (&l.attn_norm, &l.hc) {
                    (Some(an), _) => g.rms_norm_q(x, ne, an.ptr, p("h"), ne, ne, nt, eps, if mixer_gemm { 0 } else { p("hq") }, res)?,
                    (None, Some(hc)) => self.hc_read(&hc[0], x, nt, &ha)?,
                    _ => unreachable!(),
                }
            }
            match &l.mixer {
                Mixer::Delta { qkv, gate, beta, alpha, dt_bias, a, conv_w, norm, out, conv, s, snap } => {
                    // Every allocated snapshot, not `self.k` of them: the count is baked into the captured graph.
                    let (sc, ss, ns, y_sv, b_sv, g_sv) = snap.as_ref().map_or((0, 0, 0, 0, 0, 0), |(c_b, s_b, y_b, b_b, g_b)| {
                        (c_b.addr(), s_b.addr(), s_b.len() / s[self.cur].len(), y_b.addr(), b_b.addr(), g_b.addr())
                    });
                    let rows = !self.rows.is_empty();
                    let save_snap = nt > 1 && !rows && ns > 0;
                    let (y_dst, b_dst, g_dst) = if save_snap { (y_sv, b_sv, g_sv) } else { (p("y"), p("beta"), p("g")) };
                    let nn = if (nt == 1 || rows) && dk == 128 { 2 * nk } else { 0 };
                    let at = |n: &str, r: usize, w: usize| p(n) + (4 * r * w) as u64;
                    if mixer_gemm && !rows {
                        g.gemm_dense(qkv, nt, p("h"), ne, p("qkv"), channels, 0)?;
                        g.prequant_gemm_x(gate, nt, p("h"), ne)?;
                        // conv_seq is two dependent passes on Vulkan (the second overwrites the state the first reads), and
                        // a batch has no barrier between its dispatches: it runs before the batch, conv_silu inside it.
                        if nt > 1 { g.conv_seq(p("qkv"), conv[self.cur].addr(), conv_w.ptr, y_dst, channels, c.ssm_conv, nt, sc, self.nsnap)?; }
                        g.batch(&mut || {
                            if nt == 1 { g.conv_silu(p("qkv"), conv[self.cur].addr(), conv_w.ptr, y_dst, channels, c.ssm_conv, nn, eps)?; }
                            g.gemm_dense(gate, nt, p("h"), ne, p("z"), c.ssm_d_inner, crate::backend::SAME_X)?;
                            g.gemm_dense(beta, nt, p("h"), ne, b_dst, nv, crate::backend::SAME_X)?;
                            g.gemm_dense(alpha, nt, p("h"), ne, g_dst, nv, crate::backend::SAME_X)
                        })?;
                    } else {
                        mv(qkv, "h", ne, "qkv", channels)?;
                        // One token with 128-wide heads: conv_silu also does the q and k l2 norms.
                        // Overlap conv_silu (reads "qkv", writes "y") with gate/beta/alpha GEMVs (read "hq", write "z"/"beta"/"g").
                        // conv_seq stays out of the batch (see above).
                        if nt > 1 && !rows { g.conv_seq(p("qkv"), conv[self.cur].addr(), conv_w.ptr, y_dst, channels, c.ssm_conv, nt, sc, self.nsnap)?; }
                        g.batch(&mut || {
                            if rows {
                                for (r, &(sl, _)) in self.rows.iter().enumerate() {
                                    g.conv_silu(at("qkv", r, channels), conv[sl].addr(), conv_w.ptr, at("y", r, channels), channels, c.ssm_conv, nn, eps)?;
                                }
                            } else if nt == 1 {
                                g.conv_silu(p("qkv"), conv[self.cur].addr(), conv_w.ptr, y_dst, channels, c.ssm_conv, nn, eps)?;
                            }
                            mv(gate, "h", ne, "z", c.ssm_d_inner)?;
                            g.gemvq2(beta, alpha, p("h"), ne, p("hq"), b_dst, g_dst, nt)
                        })?;
                    }
                    if nn == 0 { g.l2_norm(y_dst, dk, 2 * nk, channels, nt * 2 * nk, eps)?; }
                    let scale = 1.0 / (dk as f32).sqrt();
                    if rows {
                        g.batch(&mut || {
                            for (r, &(sl, _)) in self.rows.iter().enumerate() {
                                g.deltanet(at("y", r, channels), s[sl].addr(), at("beta", r, nv), at("g", r, nv), dt_bias.ptr, a.ptr, at("o", r, c.ssm_d_inner),
                                           nk, nv, c.k_interleave, dk, dv, scale, 1, channels, 0, 0)?;
                            }
                            Ok(())
                        })?;
                    } else {
                        g.deltanet_strided(y_dst, s[self.cur].addr(), b_dst, g_dst, dt_bias.ptr, a.ptr, p("o"), nk, nv, c.k_interleave, dk, dv, scale, nt, channels, ss, ns, g.delta_snap_stride())?;
                    }
                    if vg && c.ssm_d_inner % 64 == 0 {
                        g.norm_gate(p("o"), norm.ptr, p("z"), dv, nt * nv, eps)?;
                        g.gemm_dense(out, nt, p("o"), c.ssm_d_inner, p("mix"), ne, 0)?;
                    } else {
                        let oq = if l.hc.is_some() { g.norm_gate_sig(p("o"), norm.ptr, p("z"), dv, nt * nv, eps, p("oq"))? }
                                 else { g.norm_gate_q(p("o"), norm.ptr, p("z"), dv, nt * nv, eps, p("oq"))? };
                        if oq != 0 { g.gemvq(out, p("o"), c.ssm_d_inner, oq, p("mix"), ne, nt)?; } else { mv(out, "o", c.ssm_d_inner, "mix", ne)?; }
                    }
                }
                Mixer::Attn { q, k, v, qn, kn, o, kc, vc, qsa, .. } => {
                    let qs = if c.q_gate { 2 * hd } else { hd };
                    if mixer_gemm {
                        g.gemm_dense(q, nt, p("h"), ne, p("qg"), nh * qs, 0)?;
                        g.batch(&mut || {
                            g.gemm_dense(k, nt, p("h"), ne, p("kraw"), kv, crate::backend::SAME_X)?;
                            g.gemm_dense(v, nt, p("h"), ne, p("v"), kv, crate::backend::SAME_X)
                        })?;
                    } else {
                        g.batch(&mut || { mv2(q, k, "qg", "kraw")?; mv(v, "h", ne, "v", kv) })?;
                    }
                    if !self.rows.is_empty() {
                        // One token per slot (`step_rows`): each row reads and extends its own slot's KV.
                        let at = |n: &str, r: usize, w: usize| p(n) + (4 * r * w) as u64;
                        g.batch(&mut || {
                            for (r, &(sl, _)) in self.rows.iter().enumerate() {
                                if use_rope && self.rope_offset == 0 {
                                    g.rms_norm_rope_qk(at("qg", r, nh * qs), qs, qn.ptr, at("q", r, nh * hd), nh, nh, at("kraw", r, kv), kn.ptr, at("k", r, kv), nkv, nkv,
                                                       hd, eps, c.n_rot, step + 8 * r as u64, c.rope_base, at("v", r, kv), kc[sl].addr(), vc[sl].addr())?;
                                } else {
                                    g.rms_norm(at("qg", r, nh * qs), qs, qn.ptr, at("q", r, nh * hd), hd, hd, nh, eps)?;
                                    g.rms_norm(at("kraw", r, kv), hd, kn.ptr, at("k", r, kv), hd, hd, nkv, eps)?;
                                }
                            }
                            Ok(())
                        })?;
                        if use_rope && self.rope_offset != 0 {
                            g.batch(&mut || {
                                for r in 0..self.rows.len() {
                                    g.rope_neox(at("q", r, nh * hd), hd, nh, nh, c.n_rot, rope + 8 * r as u64, c.rope_base)?;
                                    g.rope_neox(at("k", r, kv), hd, nkv, nkv, c.n_rot, rope + 8 * r as u64, c.rope_base)?;
                                }
                                Ok(())
                            })?;
                        }
                        if !(use_rope && self.rope_offset == 0) {
                            g.batch(&mut || {
                                for (r, &(sl, _)) in self.rows.iter().enumerate() {
                                    g.kv_store(at("k", r, kv), at("v", r, kv), kc[sl].addr(), vc[sl].addr(), kv, 1, step + 8 * r as u64, hd)?;
                                }
                                Ok(())
                            })?;
                        }
                        for (r, &(sl, pos)) in self.rows.iter().enumerate() {
                            g.attn_decode_q(at("q", r, nh * hd), kc[sl].addr(), vc[sl].addr(), if c.q_gate { at("qg", r, nh * qs) } else { 0 }, at("att", r, nh * hd),
                                            p("attp"), step + 8 * r as u64, nh, nkv, hd, 1.0 / (hd as f32).sqrt(), gq_min(pos, 1), 1, 0, 0, 0,
                                            sink_addr, win)?;
                        }
                        mv(o, "att", nh * hd, "mix", ne)?;
                    } else {
                    if use_rope && self.rope_offset == 0 && !self.mrope {
                        g.batch(&mut || g.rms_norm_rope_qk(p("qg"), qs, qn.ptr, p("q"), nt * nh, nh, p("kraw"), kn.ptr, p("k"), nt * nkv, nkv,
                                                           hd, eps, c.n_rot, step, c.rope_base, p("v"), kc[self.cur].addr(), vc[self.cur].addr()))?;
                    } else {
                        g.batch(&mut || {
                            g.rms_norm(p("qg"), qs, qn.ptr, p("q"), hd, hd, nt * nh, eps)?;
                            g.rms_norm(p("kraw"), hd, kn.ptr, p("k"), hd, hd, nt * nkv, eps)?;
                            Ok(())
                        })?;
                        if use_rope {
                            g.batch(&mut || {
                                // An image token turns each section of pairs by its own position.
                                if let (true, Some(sec)) = (self.mrope, c.mrope_sections) {
                                    g.rope_mrope(p("q"), hd, nt * nh, nh, c.n_rot, self.mrope_buf.addr(), c.rope_base, sec)?;
                                    return g.rope_mrope(p("k"), hd, nt * nkv, nkv, c.n_rot, self.mrope_buf.addr(), c.rope_base, sec);
                                }
                                let rope = if self.mrope { self.mrope_buf.addr() } else { rope };
                                g.rope_neox(p("q"), hd, nt * nh, nh, c.n_rot, rope, c.rope_base)?;
                                g.rope_neox(p("k"), hd, nt * nkv, nkv, c.n_rot, rope, c.rope_base)?;
                                Ok(())
                            })?;
                        }
                        g.kv_store(p("k"), p("v"), kc[self.cur].addr(), vc[self.cur].addr(), kv, nt, step, hd)?;
                    }
                    let (sel, sst) = match (qsa, &self.qsa) {
                        (Some(a), Some(b)) => {
                            mv2(&a.qp, &a.kp, "iq", "ik")?;
                            g.qsa_pool(p("ik"), a.ring, a.pool.addr(), a.kn.ptr, step, nt, c.n_rot, c.rope_base, eps)?;
                            g.rms_norm(p("iq"), QSA_D, a.qn.ptr, p("iqn"), QSA_D, QSA_D, nt * QSA_H, eps)?;
                            g.rope_neox(p("iqn"), QSA_D, nt * QSA_H, QSA_H, c.n_rot, rope, c.rope_base)?;
                            g.qsa_select(p("iqn"), a.pool.addr(), b.score.addr(), b.sel.addr(), 0, step, 0, nt, b.nbs, b.top, b.sst, 0)?;
                            (b.sel.addr(), b.sst)
                        }
                        _ => (0, 0),
                    };
                    let out_gemm = vg && (nh * hd) % 64 == 0;
                    let host = sel == 0 && sink_addr == 0 && win == 0 && crate::host_attn::on_host(il)
                        && g.attn_host(p("q"), kc[self.cur].addr(), vc[self.cur].addr(), if c.q_gate { p("qg") } else { 0 }, p("att"), step, nh, nkv, hd, 1.0 / (hd as f32).sqrt(), nt)?;
                    let aq = if host { 0 } else { g.attn_decode_q(p("q"), kc[self.cur].addr(), vc[self.cur].addr(), if c.q_gate { p("qg") } else { 0 }, p("att"), p("attp"), step, nh, nkv, hd,
                                             1.0 / (hd as f32).sqrt(), gq, nt, if out_gemm { 0 } else { p("attq") }, sel, sst, sink_addr, win)? };
                    if out_gemm {
                        g.gemm_dense(o, nt, p("att"), nh * hd, p("mix"), ne, 0)?;
                    } else if aq != 0 {
                        g.gemvq(o, p("att"), nh * hd, aq, p("mix"), ne, nt)?;
                    } else {
                        mv(o, "att", nh * hd, "mix", ne)?;
                    }
                    }
                }
            }
            let post_hq = match &l.ffn {
                Ffn::Dense { down, .. } if vg && ne % 64 == 0 && down.cols % 64 == 0 => 0,
                _ => p("hq"),
            };
            if let Some((_, mhc_a, mhc_f)) = &l.mhc {
                let s = c.mhc_streams;
                let d = s * ne;
                let code_a = 0x100 | s | (mhc_a.out_dim << 16);
                g.hc_write(p("mr"), p("mix"), mhc_a.buf.addr(), ne, code_a, nt, false)?;
                let ones = self.mhc_ones.as_ref().unwrap().addr();
                g.rms_norm(p("mr"), d, ones, p("xn"), d, d, nt, 1e-6)?;
                self.mv(&mhc_f.weight, p("xn"), d, mhc_f.buf.addr() + 128, mhc_f.out_dim, nt)?;
                let code_f = 0x100 | s | (mhc_f.out_dim << 16);
                g.hc_mix(p("mr"), mhc_f.buf.addr(), p("h"), ne, code_f, nt)?;
                g.rms_norm_q(p("h"), ne, l.post_norm.as_ref().unwrap().ptr, p("h"), ne, ne, nt, eps, post_hq, 0)?;
            } else {
                match (&l.post_norm, &l.hc) {
                    (Some(pn), _) => g.rms_norm_q(x, ne, pn.ptr, p("h"), ne, ne, nt, eps, post_hq, p("mix"))?,
                    (None, Some(hc)) => {
                        g.hc_write(x, p("mix"), p("inj"), ne, c.hc_n, nt, false)?;
                        self.hc_read(&hc[1], x, nt, &ha)?;
                    }
                    _ => unreachable!(),
                }
            }
        }
        {
            match &l.ffn {
                Ffn::Dense { gate, up, down } => {
                    debug_assert!(!resume);
                    if vg && ne % 64 == 0 && down.cols % 64 == 0 {
                        g.gemm_dense(gate, nt, p("h"), ne, p("ge"), down.cols, 0)?;
                        g.gemm_dense(up, nt, p("h"), ne, p("ge"), down.cols, 1 | crate::backend::SAME_X)?;
                        g.gemm_dense(down, nt, p("ge"), down.cols, p("ye"), ne, 0)?;
                    } else {
                        g.batch(&mut || mv2(gate, up, "ge", "ue"))?;
                        let q = sw(down.ty, "ge", "ue", down.cols, nt, "ffq")?;
                        g.gemvq(down, p("ge"), down.cols, q, p("ye"), ne, nt)?;
                    }
                    if let Some((mk, _, mhc_f)) = &l.mhc {
                        let leave_bit = if mk & 2 != 0 { 0x200 } else { 0 };
                        let code_f = 0x100 | leave_bit | c.mhc_streams | (mhc_f.out_dim << 16);
                        g.hc_write(p("mr"), p("ye"), mhc_f.buf.addr(), ne, code_f, nt, false)?;
                        if mk & 2 != 0 {
                            g.dtod(&self.bufs["ye"], 0, &self.x, 0, 4 * nt * ne)?;
                        }
                        return Ok(0);
                    }
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
                    // Fully resident (and the backend takes a null mailbox): no CPU part, so `mb = 0` makes the
                    // kernels skip every mailbox store and the combine never waits (R2). Vulkan reads the mailbox
                    // unconditionally, so `moe_mailbox_optional` keeps its behaviour unchanged.
                    let fast = cache.fully_resident() && g.moe_mailbox_optional();
                    debug_assert!(!fast || (0..c.n_expert).all(|e| cache.slot(il, e) >= 0),
                        "the fast path needs every expert of layer {il} in VRAM");
                    let mb = if fast { 0 } else { w.dev + (4 * il * w.layout.stride) as u64 };
                    if !fast { self.mailbox_writes.set(self.mailbox_writes.get() + 1); }
                    // The previous token's routing of this layer, for the next token's prefetch (R3a).
                    let pid = self.pids.as_ref().map_or(0, |b| b.addr() + (4 * k * il) as u64);
                    // Resuming: what the skipped part left (the swiglu_q of Vulkan, the only backend that resumes).
                    if resume { dq = if md.ty.float_x() { 0 } else { p("ffq") }; }
                    // Gyro files: the routed experts read the rotated h ("hr") and the rotated SwiGLU output, both as
                    // floats (`config::Hadamard`). The CPU worker rotates the plain h of the mailbox itself.
                    let rot = self.hada.as_ref().zip(c.hadamard.as_ref());
                    let hx = if rot.is_some() { p("hr") } else { p("h") };
                    let swd = || match rot {
                        Some((s, r)) => g.hadamard(p("ge"), p("ue"), s[1].addr(), p("ge"), c.n_ff_exp, r.block, kt).map(|_| 0),
                        None => sw(md.ty, "ge", "ue", c.n_ff_exp, kt, "ffq"),
                    };
                    // Prefetch the next layer's predicted experts while this layer runs (knob off by default).
                    if !resume { self.prefetch_next(il, &**g, cache, k)?; }
                    // The shared expert's GEMVs and swiglu share batches with the routed ones.
                    let sh_gemm = vg && ne % 64 == 0 && shared.as_ref().is_some_and(|s| s.gate.cols % 64 == 0 && s.down.cols % 64 == 0);
                    if !resume {
                    if sh_gemm {
                        let s = shared.as_ref().unwrap();
                        g.gemm_dense(&s.gate, nt, p("h"), ne, p("shg"), c.n_ff_shexp, 0)?;
                        g.batch(&mut || {
                            if let Some(gi) = &s.gate_inp { mv2(router, gi, "rl", "sg")?; } else { mv(router, "h", ne, "rl", c.n_expert)?; }
                            g.gemm_dense(&s.up, nt, p("h"), ne, p("shg"), c.n_ff_shexp, 1 | crate::backend::SAME_X)
                        })?;
                    } else {
                        g.batch(&mut || {
                            if let Some(s) = shared {
                                if let Some(gi) = &s.gate_inp { mv2(router, gi, "rl", "sg")?; } else { mv(router, "h", ne, "rl", c.n_expert)?; }
                                mv2(&s.gate, &s.up, "shg", "shu")?;
                            } else { mv(router, "h", ne, "rl", c.n_expert)?; }
                            Ok(())
                        })?;
                    }
                    // ponytail: a launch of its own; it fits the router's batch if it shows in the decode profile.
                    if let Some((s, r)) = rot { g.hadamard(p("h"), 0, s[0].addr(), hx, ne, r.block, nt)?; }
                    g.moe_route(p("rl"), c.n_expert, k, c.expert_weights_scale, lx.table_dev.addr(), ids, p("wts"), slots, p("wh"), p("h"), mb, ne, w.layout.h, step, nt, pid, c.router_sigmoid)?;
                    g.batch(&mut || {
                        if mg.ty == mu.ty && (mg.rows, mg.cols) == (mu.rows, mu.cols) {
                            let pair = (mu.off as i64 - mg.off as i64, p("ue") as i64 - p("ge") as i64);
                            g.moe_gemv(mg.ty, mg.rows, mg.cols, base + mg.off as u64, stride, slots, kt, hx, ne, k, p("ge"), Some(pair), p("hq"))?;
                        } else {
                            g.moe_gemv(mg.ty, mg.rows, mg.cols, base + mg.off as u64, stride, slots, kt, hx, ne, k, p("ge"), None, p("hq"))?;
                            g.moe_gemv(mu.ty, mu.rows, mu.cols, base + mu.off as u64, stride, slots, kt, hx, ne, k, p("ue"), None, p("hq"))?;
                        }
                        if !sh_gemm {
                            if let Some(s) = shared { sq = sw(s.down.ty, "shg", "shu", c.n_ff_shexp, nt, "shq")?; }
                        }
                        Ok(())
                    })?;
                    if sh_gemm {
                        let s = shared.as_ref().unwrap();
                        g.gemm_dense(&s.down, nt, p("shg"), c.n_ff_shexp, p("shy"), ne, 0)?;
                        dq = swd()?;
                    } else {
                        g.batch(&mut || {
                            dq = swd()?;
                            if let Some(s) = shared { g.gemvq(&s.down, p("shg"), c.n_ff_shexp, sq, p("shy"), ne, nt)?; }
                            Ok(())
                        })?;
                    }
                    }
                    let (sh, sg) = match shared { Some(s) => (p("shy"), if s.gate_inp.is_some() { p("sg") } else { 0 }), None => (0, 0) };
                    // Hyper-connections: the FFN output collects in "mo" (zero) and goes to the streams from there.
                    let fx = if l.hc.is_some() { p("mo") } else { x };
                    g.moe_down_combine(md.ty, md.rows, md.cols, base + md.off as u64, stride, slots, k, p("ge"), p("ye"), dq,
                                        fx, p("wh"), sh, sg, mb, w.layout.y, step, nt)?;
                    if !self.m.loras.is_empty() {
                        let use_ge = md.ty.float_x();
                        self.apply_lora_gpu_down(il, nt, k, c.n_ff_exp, c.n_embd, l.hc.is_some(), use_ge)?;
                    }
                    if l.hc.is_some() { g.hc_write(x, p("mo"), p("inj"), ne, c.hc_n, nt, true)?; }
                }
            }
        }
        Ok(0)
    }

    /// Whether this part's experts are all in VRAM and the backend takes `mb = 0`: no worker request, no mailbox
    /// write or read, no `record`, no `adapt` (issue #34 R2). Recomputed on every call (after the load and after every
    /// `adapt`), so the flag can never disagree with the placement tables.
    pub fn expert_fast_path(&self) -> bool {
        self.cache.as_ref().is_some_and(|c| c.fully_resident()) && self.gpu.moe_mailbox_optional()
    }

    /// Serve policy: follow the reply, not the prompt, in the expert placement after each prefill. `serve` turns it on;
    /// a split model's other parts follow through `next`.
    pub fn set_decode_only(&mut self, on: bool) {
        self.reset_counts_after_prefill = on;
        // Fewer, larger swaps than a benchmark run: the reply's hot set is learned in a few steps, while 64 swaps
        // every 8 tokens spend more on uploads and page-cache churn than the extra hits return.
        if std::env::var_os("STRATA_ADAPT_EVERY").is_none() { self.adapt_every = 16; }
        if std::env::var_os("STRATA_ADAPT_SWAPS").is_none() { self.adapt_swaps = 16; }
        if let Some(n) = self.next.as_mut() { n.set_decode_only(on); }
    }

    fn reset_counts(&mut self) {
        let mut p = Some(&mut *self);
        while let Some(q) = p {
            if let Some(c) = q.cache.as_mut() { c.reset_counts(); }
            p = q.next.as_deref_mut();
        }
    }

    /// The worker sequence number to publish in `step_buf`: 0 on the fast path, where the GPU asks the worker nothing.
    fn begin_seq(&mut self) -> u32 {
        let fast = self.expert_fast_path();
        match self.worker.as_mut() { Some(w) if !fast => w.begin(), _ => 0 }
    }

    /// Bytes of routed-expert weights one decode token reads from this part (0: no expert cache).
    pub fn expert_bytes_per_token(&self) -> usize {
        self.cache.as_ref().map_or(0, |c| c.token_bytes(self.m.cfg.n_expert_used))
    }

    /// Hint the next MoE layer's predicted experts into L2 (their routing of the previous token: `pids`) while this
    /// layer runs. R3a, `STRATA_MOE_PREFETCH=1` only (`pids` is None otherwise), a hint with no correctness role.
    fn prefetch_next(&self, il: usize, g: &dyn Backend, cache: &ExpertCache, k: usize) -> anyhow::Result<()> {
        let Some(pids) = &self.pids else { return Ok(()) };
        let last = self.first + self.layers.len();
        let Some(nil) = (il + 1..last).find(|&j| cache.layer(j).is_some()) else { return Ok(()) };
        let Some((table, base, stride, bytes)) = cache.prefetch_geom(nil) else { return Ok(()) };
        g.moe_prefetch(pids.addr() + (4 * k * nil) as u64, k, table, base, stride, bytes)
    }

    /// Advance the position and update expert statistics and placement after a token.
    fn apply_lora_gpu_down(&self, il: usize, nt: usize, k: usize, ff: usize, ne: usize, hc: bool, use_ge: bool) -> anyhow::Result<()> {
        if !self.lora_applied.borrow_mut().insert(il) { return Ok(()); }
        let name = format!("blk.{il}.ffn_down_exps.weight");
        let mut adapters: Vec<(f32, f32, &crate::model::LoraEntry)> = Vec::new();
        for lora in &self.m.loras {
            if let Some(e) = lora.entries.get(&name) {
                adapters.push((lora.scale, lora.alpha, e));
            }
        }
        if adapters.is_empty() { return Ok(()); }
        let kt = nt * k;
        let slots = self.gpu.get::<i32>(&self.slots, 0, kt)?;
        let ids = self.gpu.get::<i32>(&self.ids, 0, kt)?;
        let wts = self.gpu.get::<f32>(&self.bufs["wts"], 0, kt)?;
        // activation: float (ge, post-swiglu) for float weights; dequantized ffq (Q8) for quantized
        let mut acts = vec![0f32; kt * ff];
        if use_ge {
            let ge = self.gpu.get::<f32>(&self.bufs["ge"], 0, kt * ff)?;
            acts.copy_from_slice(&ge);
            // The delta reads the plain SwiGLU output, "ge" holds the rotated one.
            if let Some(r) = &self.m.cfg.hadamard { for a in acts.chunks_exact_mut(ff) { r.unrotate(a); } }
        } else {
            let quant_bytes = kt * ff;
            let quant_words = quant_bytes / 4;
            let scale_words = kt * (ff / 32);
            let packed = self.gpu.get::<f32>(&self.bufs["ffq"], 0, quant_words + scale_words)?;
            // quantized region: first quant_words f32s hold int8 bytes; scales follow
            let mut bytes = vec![0u8; quant_bytes];
            for w in 0..quant_words {
                let v = packed[w].to_bits();
                bytes[w*4] = (v & 0xff) as u8;
                bytes[w*4+1] = ((v >> 8) & 0xff) as u8;
                bytes[w*4+2] = ((v >> 16) & 0xff) as u8;
                bytes[w*4+3] = ((v >> 24) & 0xff) as u8;
            }
            let scales = &packed[quant_words..quant_words + scale_words];
            for slot in 0..kt {
                for blk in 0..(ff / 32) {
                    let scale = scales[slot * (ff / 32) + blk];
                    for i in 0..32 {
                        let byte = bytes[slot * ff + blk * 32 + i] as i8;
                        acts[slot * ff + blk * 32 + i] = byte as f32 * scale;
                    }
                }
            }
        }
        let res_buf = if hc { &self.bufs["mo"] } else { &self.x };
        let mut residual = self.gpu.get::<f32>(res_buf, 0, nt * ne)?;
        for t in 0..nt {
            for j in 0..k {
                let slot = t * k + j;
                if slots[slot] < 0 { continue; }
                let expert_id = ids[slot] as usize;
                let w = wts[slot];
                let act = &acts[slot * ff..(slot + 1) * ff];
                for (scale, alpha, entry) in &adapters {
                    let ei = if entry.n_expert == 1 { 0 } else { expert_id };
                    if ei >= entry.n_expert && entry.n_expert > 1 { continue; }
                    let a = &entry.a[ei];
                    let b = &entry.b[ei];
                    let rank = entry.rank;
                    let in_dim = entry.in_dim;
                    let out_dim = entry.out_dim;
                    let mult = if *alpha > 0.0 { *scale * *alpha / rank as f32 } else { *scale };
                    for o in 0..out_dim {
                        let mut s = 0f32;
                        for r in 0..rank {
                            let mut dot = 0f32;
                            for i in 0..in_dim {
                                dot += a[r * in_dim + i] * act[i];
                            }
                            s += b[o * rank + r] * dot;
                        }
                        residual[t * ne + o] += w * mult * s;
                    }
                }
            }
        }
        self.gpu.put(&residual, res_buf, 0)?;
        Ok(())
    }

    fn after_forward(&mut self, nt: usize) -> anyhow::Result<()> {
        self.pos += nt;
        if let Some(df) = self.dflash.as_mut() {
            df.pend_ctx = nt;
        }
        self.after_experts(nt, self.pos)?;
        if let Some(n) = self.next.as_mut() {
            n.adapt_every = self.adapt_every;
            n.adapt_swaps = self.adapt_swaps;
            n.reset_counts_after_prefill = self.reset_counts_after_prefill;
            n.after_forward(nt)?;
        }
        Ok(())
    }

    /// Expert statistics after a forward of `nt` tokens, and the placement update every `adapt_every` tokens of
    /// `count`.
    fn after_experts(&mut self, nt: usize, count: usize) -> anyhow::Result<()> {
        self.tokens += nt;
        let fast = self.expert_fast_path();
        let k = self.m.cfg.n_expert_used;
        if let (Some(cache), Some(w)) = (self.cache.as_mut(), self.worker.as_ref()) {
            for il in self.first..self.first + self.layers.len() {
                if cache.layer(il).is_none() { continue; }
                // Fast path: every routed expert of the layer is resident by construction, so each token reads k
                // hits and the mailbox is not touched. Reading it back to learn that would cost what R2 removes.
                if fast { self.hits += k * nt; continue; }
                let r = w.routing(il);
                self.mailbox_reads.set(self.mailbox_reads.get() + 1);
                let (hits, misses) = tally(&r);
                self.hits += hits;
                self.misses += misses;
                debug_assert!(misses == 0 || !cache.fully_resident(), "a missed expert in a fully resident cache");
                cache.record(il, &r.iter().map(|x| x.0).collect::<Vec<_>>());
            }
            self.cpu_time = w.busy();
            // No swap can be beneficial with no miss: `adapt` would only scan all 10240 entries (R2).
            if !fast && self.adapt_every > 0 && (count - nt) / self.adapt_every < count / self.adapt_every {
                cache.adapt(self.m, &*self.gpu, self.adapt_swaps)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{GpuModel, split_ends, split_shares, tally};

    /// Layer counts given as shares come back as the same counts: a calibrated split loads as it was measured.
    #[test]
    fn split_ends_keep_layer_counts() {
        assert_eq!(split_ends(&[12.0, 28.0], 40), [0, 12, 40]);
        assert_eq!(split_ends(&[7.0, 13.0, 21.0], 41), [0, 7, 20, 41]);
        assert_eq!(split_ends(&[30.0, 70.0], 48), [0, 14, 48]);
    }

    #[test]
    fn split_shares_take_one_positive_number_per_device() {
        assert_eq!(split_shares("30, 70", 2).unwrap(), [30.0, 70.0]);
        assert_eq!(split_shares("1,2.5,1", 3).unwrap(), [1.0, 2.5, 1.0]);
        for bad in ["30", "30,70,10", "30,0", "30,-1", "30,x", "30,inf", ""] {
            assert!(split_shares(bad, 2).unwrap_err().to_string().starts_with("--split"), "{bad:?}");
        }
    }

    /// R1: a layer's routing gives the hit rate its numbers; a slot < 0 is an expert left on the CPU.
    #[test]
    fn routing_tally_of_a_partial_cache_counts_every_miss() {
        let r = [(4, 0.5, 3), (11, 0.25, -1), (7, 0.25, 0), (9, 0.0, -1)];
        assert_eq!(tally(&r), (2, 2));
        assert_eq!(tally(&[(1, 1.0, -1); 8]), (0, 8));
    }

    /// R1: a fully resident layer cannot miss, which is what the fast path asserts instead of reading the mailbox.
    #[test]
    fn routing_tally_of_a_resident_layer_has_no_miss() {
        let r: Vec<(i32, f32, i32)> = (0..8).map(|e| (e, 0.125, e)).collect();
        assert_eq!(tally(&r), (8, 0));
    }

    /// The fast-path method is off without a cache (CPU experts or a dense model): the compile-time check that the
    /// signature stays usable by `Runner::finish`.
    #[test]
    fn no_cache_has_no_fast_path_bytes() {
        fn _uses(g: &GpuModel) -> (bool, usize) { (g.expert_fast_path(), g.expert_bytes_per_token()) }
    }
}

/// `STRATA_KV_HOST=N`: the KV cache of the attention layers from block N on lives in host RAM (mapped pinned
/// memory the GPU reads over PCIe), freeing VRAM for a longer context. `all` = every layer.
fn kv_alloc(gpu: &dyn Backend, il: usize, bytes: usize) -> anyhow::Result<Buf> {
    if crate::host_attn::on_host(il) { gpu.alloc_host(bytes) } else { gpu.alloc(bytes) }
}
