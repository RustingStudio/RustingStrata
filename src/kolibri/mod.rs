//! Aleph Alpha Kolibri 1.1 (`kolibri1`): sliding-window layers with RoPE, every fifth layer a sparse layer without
//! positions that attends the best keys of an indexer over K/V rounded to NVFP4, some sparse layers reading the K/V
//! of an earlier one; a dense FFN in the leading layers, then a sigmoid-routed MoE with a shared expert.
//! CPU forward that mirrors llama.cpp `src/models/kolibri1.cpp`; matvecs and routed experts reuse `deepseek::mat`.
pub mod chat;
pub mod forward;

use crate::gguf::Gguf;
use anyhow::{Context, ensure};

pub const ARCH: &str = "kolibri1";

/// Keys every sparse query attends: the first `N_SINK` positions and its last `N_RECENT`.
pub const N_SINK: usize = 8;
pub const N_RECENT: usize = 128;

#[derive(Debug, Clone)]
pub struct KolibriConfig {
    pub n_layer: usize,
    /// Prompt plus reply limit: `--ctx`, else the trained context.
    pub n_ctx: usize,
    pub n_embd: usize,
    pub n_vocab: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub rope_base: f32,
    pub eps: f32,
    pub n_expert: usize,
    pub n_used: usize,
    /// Leading layers with a dense FFN instead of the MoE.
    pub n_dense: usize,
    /// Sliding-window layers (the others are sparse), and the keys a sliding query sees.
    pub swa: Vec<bool>,
    pub n_swa: usize,
    /// Sparse layers that choose the pool of keys; the other sparse layers select from the pool of the one before.
    pub indexer: Vec<bool>,
    /// The layer whose K/V a layer attends (itself, or an earlier sparse layer).
    pub kv_src: Vec<usize>,
    /// Keys a sparse query selects, and the pool of a long context: `pool_blocks` blocks of `block` positions.
    pub top_k: usize,
    pub block: usize,
    pub pool_blocks: usize,
}

impl KolibriConfig {
    pub fn from_gguf(g: &Gguf) -> anyhow::Result<Self> {
        let k = |s: &str| format!("{ARCH}.{s}");
        let u = |s: &str| -> anyhow::Result<usize> { Ok(g.get(&k(s))?.as_u64().with_context(|| k(s))? as usize) };
        let f = |s: &str| -> anyhow::Result<f32> { g.get(&k(s))?.as_f32().with_context(|| k(s)) };
        let arr_u = |s: &str| -> anyhow::Result<Vec<usize>> {
            g.get(&k(s))?.as_arr().with_context(|| k(s))?.iter().map(|v| Ok(v.as_u64().with_context(|| k(s))? as usize)).collect()
        };
        let n_layer = u("block_count")?;
        let swa: Vec<bool> = arr_u("attention.sliding_window_pattern")?.into_iter().map(|v| v != 0).collect();
        let indexer: Vec<bool> = arr_u("attention.indexer.types")?.into_iter().map(|v| v != 0).collect();
        let kv_src = arr_u("attention.kv_source_map")?;
        let (head_dim, n_head, n_head_kv) = (u("attention.key_length")?, u("attention.head_count")?, u("attention.head_count_kv")?);
        ensure!(swa.len() == n_layer && indexer.len() == n_layer && kv_src.len() == n_layer, "per-layer metadata does not cover {n_layer} layers");
        ensure!(u("expert_gating_func")? == 2 && u("expert_shared_count")? == 1 && u("expert_weights_norm")? == 0, "unsupported Kolibri MoE variant");
        ensure!(head_dim % 16 == 0 && u("attention.value_length")? == head_dim && u("rope.dimension_count")? == head_dim && n_head % n_head_kv == 0, "unsupported Kolibri attention shape");
        let mut seen = false;
        for il in 0..n_layer {
            let src = kv_src[il];
            ensure!(src <= il && kv_src[src] == src && swa[src] == swa[il] && (!swa[il] || (src == il && !indexer[il])) && (swa[il] || indexer[il] || seen),
                "layer {il}: invalid indexer or K/V source");
            seen |= indexer[il];
        }
        Ok(KolibriConfig {
            n_ctx: crate::ctx_limit(u("context_length")?),
            n_layer,
            n_embd: u("embedding_length")?,
            n_vocab: g.tensors.get("token_embd.weight").context("missing token_embd.weight")?.dims[1] as usize,
            n_head,
            n_head_kv,
            head_dim,
            rope_base: f("rope.freq_base")?,
            eps: f("attention.layer_norm_rms_epsilon")?,
            n_expert: u("expert_count")?,
            n_used: u("expert_used_count")?,
            n_dense: u("leading_dense_block_count")?,
            swa,
            n_swa: u("attention.sliding_window")?,
            indexer,
            kv_src,
            top_k: u("attention.indexer.top_k")?,
            block: u("attention.indexer.candidate_block_size")?,
            pool_blocks: u("attention.indexer.candidate_top_k_blocks")?,
        })
    }
}
