//! Inkling 2 (Thinking Machines), the looped text model: GGUF architecture `inkling2_loop` (llama.cpp), or
//! `inkling2` with `num_loops` (the layout of llama.cpp PR #32695, which Unsloth's files use).
pub mod chat;
pub mod dspark;
pub mod forward;

use crate::gguf::Gguf;
use anyhow::{Context, ensure};

pub const ARCH: &str = "inkling2_loop";
/// The same model in the layout of llama.cpp PR #32695: the keys and the loop tensors have other names, and the
/// per-layer arrays have one entry per layer run (prelude, every pass of the core, then the rest).
pub const ARCH_RUNS: &str = "inkling2";

pub fn is_arch(arch: &str) -> bool {
    arch == ARCH || arch == ARCH_RUNS
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
        let runs = g.get("general.architecture").ok().and_then(|v| v.as_str()) == Some(ARCH_RUNS);
        let arch = if runs { ARCH_RUNS } else { ARCH };
        let k = |s: &str| format!("{arch}.{s}");
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
        let (loop_start, loop_end, loop_count) = if runs {
            (u("loop.core_start")?, u("loop.core_end")?, u("num_loops").context("inkling2 files without a looped core are not supported")?)
        } else {
            (u("loop.start")?, u("loop.end")?, u("loop.count")?)
        };
        ensure!(loop_start < loop_end && loop_count >= 1, "bad loop range");
        // `runs`: the file counts layer runs, and the later passes of the core sit between the core and the rest.
        let extra = if runs { (loop_count - 1).checked_mul(loop_end - loop_start).context("bad loop range")? } else { 0 };
        let n_layer = u("block_count")?.checked_sub(extra).context("block_count is below the layer runs of the loop")?;
        let mut swa: Vec<bool> = arr("attention.sliding_window_pattern")?.into_iter().map(|v| v != 0).collect();
        ensure!(swa.len() == n_layer + extra, "sliding_window_pattern: {} entries for {} layers", swa.len(), n_layer + extra);
        ensure!(loop_end <= n_layer, "bad loop range");
        swa.drain(loop_end..loop_end + extra);
        let cfg = Self {
            n_ctx: crate::ctx_limit(u("context_length")?),
            n_layer,
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
            swa,
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
            loop_start,
            loop_end,
            loop_count,
            ctx: g.get(&k("context_length")).ok().and_then(|v| v.as_u64()).map(|x| x as usize).unwrap_or(262_144),
        };
        ensure!(cfg.n_head % cfg.n_head_kv == 0, "head_count {} is not a multiple of head_count_kv {}", cfg.n_head, cfg.n_head_kv);
        ensure!(cfg.loop_end <= cfg.n_layer, "bad loop range");
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
