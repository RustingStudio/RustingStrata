//! Tencent Hy4-flash (`hy_v4_flash`): hyper-connection streams, gated MLA attention with sinks, latent MoE, Engram.
//! CPU forward that mirrors llama.cpp `src/models/hy-v4-flash.cpp`; matvecs and routed experts reuse `deepseek::mat`.
pub mod chat;
pub mod forward;

use crate::deepseek::EngramConfig;
use crate::gguf::Gguf;
use anyhow::{Context, ensure};

pub const ARCH: &str = "hy_v4_flash";

#[derive(Debug, Clone)]
pub struct Hy4Config {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_vocab: usize,
    /// Heads per layer (32 on the sparse layers, 64 on the sliding-window ones).
    pub n_head: Vec<usize>,
    /// Sliding-window layers.
    pub swa: Vec<bool>,
    pub n_swa: usize,
    /// Per-head query/key width (192 no-position + `n_rot` rotary) and value width.
    pub key_len: usize,
    pub val_len: usize,
    pub kv_rank: usize,
    pub n_rot: usize,
    pub rope_base: f32,
    pub rope_base_swa: f32,
    pub eps: f32,
    pub hc_eps: f32,
    pub hc_mag: f32,
    pub n_expert: usize,
    pub n_used: usize,
    pub exp_scale: f32,
    pub exp_norm: bool,
    pub clamp_exp: Vec<f32>,
    /// Leading layers with a dense FFN instead of the latent MoE.
    pub n_dense: usize,
    pub engram: Option<EngramConfig>,
    /// Sparse-layer indexer: positions per pooled group and the number of groups a position keeps (0: no indexer).
    pub kpool: usize,
    pub idx_top: usize,
}

impl Hy4Config {
    pub fn from_gguf(g: &Gguf) -> anyhow::Result<Self> {
        let k = |s: &str| format!("{ARCH}.{s}");
        let u = |s: &str| -> anyhow::Result<usize> { Ok(g.get(&k(s))?.as_u64().with_context(|| k(s))? as usize) };
        let f = |s: &str| -> anyhow::Result<f32> { g.get(&k(s))?.as_f32().with_context(|| k(s)) };
        let arr_u = |s: &str| -> anyhow::Result<Vec<u64>> {
            g.get(&k(s))?.as_arr().with_context(|| k(s))?.iter().map(|v| v.as_u64().with_context(|| k(s))).collect()
        };
        let n_layer = u("block_count")?;
        let n_head: Vec<usize> = arr_u("attention.head_count")?.into_iter().map(|v| v as usize).collect();
        let swa: Vec<bool> = arr_u("attention.sliding_window_pattern")?.into_iter().map(|v| v != 0).collect();
        let clamp_exp: Vec<f32> = g.get(&k("swiglu_clamp_exp"))?.as_arr().with_context(|| k("swiglu_clamp_exp"))?
            .iter().map(|v| v.as_f32().context("swiglu_clamp_exp")).collect::<anyhow::Result<_>>()?;
        ensure!(n_head.len() == n_layer && swa.len() == n_layer && clamp_exp.len() == n_layer, "per-layer metadata does not cover {n_layer} layers");
        ensure!(u("hyper_connection.count")? == 4, "only 4 hyper-connection streams are supported");
        ensure!(u("expert_gating_func")? == 2 && u("expert_shared_count")? == 1 && u("attention.head_count_kv")? == 1, "unsupported Hy4 variant");
        let engram = if g.meta.contains_key(&k("ple.layers")) {
            let layers: Vec<usize> = arr_u("ple.layers")?.into_iter().map(|v| v as usize).collect();
            let (ngram, hpn) = (u("ple.ngram_size")?, u("ple.heads_per_ngram")?);
            let n_cols = (ngram - 1) * hpn;
            let (mult, offsets, primes) = (arr_u("ple.layer_multipliers")?, arr_u("ple.head_offsets")?, arr_u("ple.head_vocab_sizes")?);
            ensure!(mult.len() == layers.len() * ngram && offsets.len() == layers.len() * n_cols && primes.len() == offsets.len(), "engram metadata is inconsistent");
            let token_map: Vec<i32> = g.get(&k("engram.token_map"))?.as_arr().context("engram.token_map")?.iter()
                .map(|v| v.as_u64().map(|x| x as i32).or(if let crate::gguf::Value::I32(x) = v { Some(*x) } else { None }).context("token_map"))
                .collect::<anyhow::Result<_>>()?;
            let pad_id = u("engram.pad_token_id")?;
            ensure!(pad_id < token_map.len() && token_map[pad_id] >= 0, "engram pad token is not mapped");
            Some(EngramConfig { layers, ngram, heads_per_ngram: hpn, head_dim: u("embedding_length_per_layer_input")?, multipliers: mult, offsets, primes, pad_id, token_map })
        } else {
            None
        };
        let (mut kpool, mut idx_top) = (0, 0);
        if g.meta.contains_key(&k("attention.indexer.top_k")) {
            kpool = u("attention.indexer.kpool")?;
            ensure!(kpool >= 2 && u("attention.indexer.top_k")? / kpool >= 2, "unsupported indexer pool size");
            ensure!(matches!(g.get(&k("attention.indexer.kpool_select_tail")), Ok(crate::gguf::Value::Bool(true))), "only the tail-selecting indexer is supported");
            // One group of the budget is the open tail group, which is always kept.
            idx_top = u("attention.indexer.top_k")? / kpool - 1;
        }
        Ok(Hy4Config {
            n_layer,
            n_embd: u("embedding_length")?,
            n_vocab: g.tensors.get("token_embd.weight").context("missing token_embd.weight")?.dims[1] as usize,
            n_head,
            swa,
            n_swa: u("attention.sliding_window")?,
            key_len: u("attention.key_length_mla")?,
            val_len: u("attention.value_length_mla")?,
            kv_rank: u("attention.kv_lora_rank")?,
            n_rot: u("rope.dimension_count")?,
            rope_base: f("rope.freq_base")?,
            rope_base_swa: f("rope.freq_base_swa")?,
            eps: f("attention.layer_norm_rms_epsilon")?,
            hc_eps: f("hyper_connection.epsilon")?,
            hc_mag: f("hyper_connection.magnitude")?,
            n_expert: u("expert_count")?,
            n_used: u("expert_used_count")?,
            exp_scale: f("expert_weights_scale")?,
            exp_norm: u("expert_weights_norm")? != 0,
            clamp_exp,
            n_dense: u("leading_dense_block_count")?,
            engram,
            kpool,
            idx_top,
        })
    }

    pub fn engram_index(&self, il: usize) -> Option<usize> {
        self.engram.as_ref()?.layers.iter().position(|&l| l == il)
    }
}
