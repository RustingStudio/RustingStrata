//! Inkling 2 (Thinking Machines), the looped text model: GGUF architecture `inkling2_loop`.
pub mod chat;
pub mod dspark;
pub mod forward;

use crate::gguf::Gguf;
use anyhow::{Context, ensure};

pub const ARCH: &str = "inkling2_loop";
/// The first export of the same model (unsloth's top-level folder) says `inkling2`, counts the layer runs in
/// `block_count` instead of the physical blocks, and names the loop tensors differently: `loop_attn_*` /
/// `loop_mlp_*` instead of `loop_q_ctrl_*` / `loop_ffn_ctrl_*`, and one `loop_res_query` + `loop_kv_res_query`
/// pair instead of a `loop_msg_query` per core layer. The weights are byte-identical, so such a file loads
/// through a name map (see `Inkling2Model::new_with_ctx`).
pub const ARCH_LEGACY: &str = "inkling2";

/// Whether `arch` is Inkling 2, in either the current or the legacy spelling.
pub fn is_arch(arch: &str) -> bool {
    arch == ARCH || arch == ARCH_LEGACY
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
    /// The file uses the legacy layout (`ARCH_LEGACY`): see `Inkling2Model::new_with_ctx` for the tensor names.
    pub legacy: bool,
}

impl Inkling2Config {
    pub fn from_gguf(g: &Gguf) -> anyhow::Result<Self> {
        let legacy = g.meta.contains_key(&format!("{ARCH_LEGACY}.block_count"));
        let arch = if legacy { ARCH_LEGACY } else { ARCH };
        let k = |s: &str| format!("{arch}.{s}");
        let u = |s: &str| -> anyhow::Result<usize> { Ok(g.get(&k(s))?.as_u64().with_context(|| k(s))? as usize) };
        // A key the legacy export spells differently, or does not have.
        let u_or = |s: &str, old: &str| u(s).or_else(|_| u(old));
        let f = |s: &str| -> anyhow::Result<f32> { g.get(&k(s))?.as_f32().with_context(|| k(s)) };
        let arr = |s: &str| -> anyhow::Result<Vec<usize>> {
            g.get(&k(s))?.as_arr().with_context(|| k(s))?.iter().map(|v| Ok(v.as_u64().with_context(|| k(s))? as usize)).collect()
        };
        // The legacy export counts layer runs in `block_count` (48 for 24 blocks whose core 12 run three times),
        // so the physical blocks come from the tensors and the metadata only has to agree.
        let n_layer = g.tensors.keys().filter_map(|n| {
            let rest = n.strip_prefix("blk.")?;
            rest.split('.').next()?.parse::<usize>().ok()
        }).max().map_or(0, |m| m + 1);
        ensure!(n_layer > 0, "no blk.N tensors");
        let loop_start = u_or("loop.start", "loop.core_start")?;
        let loop_end = u_or("loop.end", "loop.core_end")?;
        let loop_count = u_or("loop.count", "num_loops")?;
        let slots = n_layer + (loop_count - 1) * (loop_end.saturating_sub(loop_start));
        let declared = u("block_count")?;
        ensure!(declared == n_layer || declared == slots,
            "block_count {declared}: neither the {n_layer} physical blocks nor the {slots} layer runs");
        // The file stores the head counts per layer run; the forward needs one value for all layers.
        let same = |s: &str| -> anyhow::Result<usize> {
            let v = arr(s)?;
            ensure!(v.len() >= n_layer && v[..n_layer].iter().all(|&x| x == v[0]), "{}: layers differ", k(s));
            Ok(v[0])
        };
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
            swa: {
                let v = arr("attention.sliding_window_pattern")?;
                ensure!(v.len() >= n_layer, "sliding_window_pattern: {} entries for {n_layer} layers", v.len());
                v.into_iter().take(n_layer).map(|v| v != 0).collect()
            },
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
            legacy,
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
