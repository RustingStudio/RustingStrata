//! Inkling 2 (Thinking Machines), the looped text model: GGUF architecture `inkling2_loop`.
pub mod chat;
pub mod dspark;
pub mod forward;

use crate::gguf::Gguf;
use anyhow::{Context, ensure};

pub const ARCH: &str = "inkling2_loop";

/// Default share of free VRAM for the expert budget. 90% suits a 12 GB card, but on a small one it leaves no room
/// for per-use staging copies and KV growth, so decode falls back to the host on every layer. Query the card once
/// and keep a larger margin there.
pub fn gpu_fraction_default() {
    if std::env::var_os("STRATA_DS_GPU_FRACTION").is_some() { return; }
    let kind = std::env::var("STRATA_DS_BACKEND").ok();
    let total = crate::backend::open(kind.as_deref(), 0).ok()
        .and_then(|b| b.mem_info().ok().map(|(_, t)| t)).unwrap_or(0);
    let frac = if total < 8 << 30 { "0.60" } else { "0.90" };
    unsafe { std::env::set_var("STRATA_DS_GPU_FRACTION", frac); }
}

#[derive(Debug, Clone)]
pub struct Inkling2Config {
    /// Layers in the file. The core layers `loop_start..loop_end` run `loop_count` times.
    pub n_layer: usize,
    /// Prompt plus reply limit: `--ctx`, else the trained context. The rows of the full-attention layers grow with
    /// the text, so RAM and VRAM are the real bound.
    pub n_ctx: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_expert: usize,
    pub n_used: usize,
    pub n_ff_exp: usize,
    /// Logits kept: the vocabulary without the padding rows of the output matrix.
    pub n_vocab: usize,
    pub eps: f32,
    pub n_swa: usize,
    /// Per layer: sliding-window attention (the others attend the whole context).
    pub swa: Vec<bool>,
    pub n_conv: usize,
    pub exp_scale: f32,
    pub situ_beta: f32,
    pub situ_linear_beta: f32,
    /// A layer whose index is a multiple of this starts a new residual stream.
    pub block_size: usize,
    pub softcap: f32,
    pub n_rel: usize,
    pub rel_extent: usize,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub log_floor: f32,
    pub log_alpha: f32,
    pub loop_start: usize,
    pub loop_end: usize,
    pub loop_count: usize,
    /// Model's native context length (`inkling2_loop.context_length`).
    pub ctx: usize,
}

impl Inkling2Config {
    pub fn from_gguf(g: &Gguf) -> anyhow::Result<Self> {
        let k = |s: &str| format!("{ARCH}.{s}");
        let u = |s: &str| -> anyhow::Result<usize> { Ok(g.get(&k(s))?.as_u64().with_context(|| k(s))? as usize) };
        let f = |s: &str| -> anyhow::Result<f32> { g.get(&k(s))?.as_f32().with_context(|| k(s)) };
        let arr = |s: &str| -> anyhow::Result<Vec<usize>> {
            g.get(&k(s))?.as_arr().with_context(|| k(s))?.iter().map(|v| Ok(v.as_u64().with_context(|| k(s))? as usize)).collect()
        };
        // The file stores the head counts per layer; the forward needs one value for all layers.
        let same = |s: &str| -> anyhow::Result<usize> {
            let v = arr(s)?;
            ensure!(!v.is_empty() && v.iter().all(|&x| x == v[0]), "{}: layers differ", k(s));
            Ok(v[0])
        };
        let cfg = Self {
            n_ctx: crate::ctx_limit(u("context_length")?),
            n_layer: u("block_count")?,
            n_embd: u("embedding_length")?,
            n_head: same("attention.head_count")?,
            n_head_kv: same("attention.head_count_kv")?,
            head_dim: u("attention.key_length")?,
            n_expert: u("expert_count")?,
            n_used: u("expert_used_count")?,
            n_ff_exp: u("expert_feed_forward_length")?,
            n_vocab: u("vocab_size_unpadded").or_else(|_| u("vocab_size"))?,
            eps: f("attention.layer_norm_rms_epsilon")?,
            n_swa: u("attention.sliding_window")?,
            swa: arr("attention.sliding_window_pattern")?.into_iter().map(|v| v != 0).collect(),
            n_conv: u("shortconv.l_cache")?,
            exp_scale: f("expert_weights_scale")?,
            situ_beta: f("activation.situ_beta")?,
            situ_linear_beta: f("activation.situ_linear_beta")?,
            block_size: u("attn_res.block_size")?,
            softcap: f("final_logit_softcapping")?,
            n_rel: u("attention.relative_length")?,
            rel_extent: u("attention.relative_extent")?,
            qk_clip: f("attention.qk_clip")?,
            rel_clip: f("attention.relative_clip")?,
            log_floor: u("attention.log_scale_floor")? as f32,
            log_alpha: f("attention.log_scale_alpha")?,
            loop_start: u("loop.start")?,
            loop_end: u("loop.end")?,
            loop_count: u("loop.count")?,
            ctx: g.get(&k("context_length")).ok().and_then(|v| v.as_u64()).map(|x| x as usize).unwrap_or(262_144),
        };
        ensure!(cfg.swa.len() == cfg.n_layer, "sliding_window_pattern: {} entries for {} layers", cfg.swa.len(), cfg.n_layer);
        ensure!(cfg.n_head % cfg.n_head_kv == 0, "head_count {} is not a multiple of head_count_kv {}", cfg.n_head, cfg.n_head_kv);
        ensure!(cfg.loop_start < cfg.loop_end && cfg.loop_end <= cfg.n_layer && cfg.loop_count >= 1, "bad loop range");
        ensure!(cfg.n_conv >= 2 && cfg.block_size >= 1 && cfg.rel_extent >= 1 && cfg.n_swa >= 1, "bad shortconv, block or window size");
        ensure!(u("attention.value_length")? == cfg.head_dim, "key and value head widths differ");
        ensure!(g.meta.get(&k("loop.varying")).and_then(|v| v.as_u64()).unwrap_or(0) == 0, "loop.varying is not supported");
        Ok(cfg)
    }

    /// Layer runs per token: every pass of the core keeps its own attention rows and convolution state.
    pub fn n_slot(&self) -> usize {
        self.n_layer + (self.loop_count - 1) * (self.loop_end - self.loop_start)
    }
}
