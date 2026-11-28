//! SberDevices GigaChat 4 (`gigachat4`): three Gated DeltaNet layers, then one MLA layer (absorbed, one latent K/V
//! head) with a lightning indexer that picks the keys of a long context; every norm is a gated RMS norm; a dense FFN
//! in the leading layer, then a sigmoid-routed MoE with a shared expert and clamped SwiGLU; soft-capped logits.
//! CPU forward that mirrors llama.cpp `src/models/gigachat4.cpp`; matvecs and routed experts reuse `deepseek::mat`.
//! The MTP block (the last layer of the file) is not run.
pub mod chat;
pub mod forward;

use crate::gguf::Gguf;
use anyhow::{Context, ensure};

pub const ARCH: &str = "gigachat4";

#[derive(Debug, Clone)]
pub struct GigaChat4Config {
    /// Trunk layers (the file's block count less the MTP layers).
    pub n_layer: usize,
    pub n_ctx: usize,
    pub n_embd: usize,
    pub n_vocab: usize,
    pub eps: f32,
    /// Gated DeltaNet layers (the others are MLA).
    pub recurrent: Vec<bool>,
    // MLA
    pub n_head: usize,
    pub q_lora: usize,
    pub kv_lora: usize,
    /// Per-head query/key size before absorption (nope + rope) and the value size per head.
    pub k_mla: usize,
    pub v_mla: usize,
    pub n_rot: usize,
    pub rope_base: f32,
    pub rope_scale: f32,
    pub rope_ctx_orig: u32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub yarn_log_mul: f32,
    // indexer
    pub idx_n_head: usize,
    pub idx_head_dim: usize,
    pub idx_top_k: usize,
    // Gated DeltaNet
    pub ssm_conv: usize,
    pub ssm_d_state: usize,
    pub ssm_n_group: usize,
    pub ssm_dt_rank: usize,
    pub ssm_d_inner: usize,
    // FFN
    pub n_expert: usize,
    pub n_used: usize,
    pub n_dense: usize,
    pub expert_scale: f32,
    pub expert_norm: bool,
    pub clamp_exp: Vec<f32>,
    pub clamp_shexp: Vec<f32>,
    pub softcap: f32,
}

impl GigaChat4Config {
    pub fn from_gguf(g: &Gguf) -> anyhow::Result<Self> {
        let k = |s: &str| format!("{ARCH}.{s}");
        let u = |s: &str| -> anyhow::Result<usize> { Ok(g.get(&k(s))?.as_u64().with_context(|| k(s))? as usize) };
        let f = |s: &str| -> anyhow::Result<f32> { g.get(&k(s))?.as_f32().with_context(|| k(s)) };
        let arr = |s: &str| -> anyhow::Result<Vec<f32>> {
            g.get(&k(s))?.as_arr().with_context(|| k(s))?.iter().map(|v| v.as_f32().or(v.as_u64().map(|x| x as f32)).with_context(|| k(s))).collect()
        };
        let n_layer = u("block_count")? - g.get(&k("nextn_predict_layers")).ok().and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let recurrent: Vec<bool> = arr("attention.recurrent_layers")?.into_iter().map(|v| v != 0.0).collect();
        ensure!(recurrent.len() >= n_layer, "attention.recurrent_layers does not cover {n_layer} layers");
        ensure!(u("expert_gating_func")? == 2 && u("expert_shared_count")? == 1 && u("attention.head_count_kv")? == 1, "unsupported GigaChat 4 variant");
        let (k_mla, v_mla, kv_lora, n_rot) = (u("attention.key_length_mla")?, u("attention.value_length_mla")?, u("attention.kv_lora_rank")?, u("rope.dimension_count")?);
        ensure!(u("attention.key_length")? == kv_lora + n_rot && u("attention.value_length")? == kv_lora && n_rot < k_mla, "unsupported GigaChat 4 attention shape");
        let yarn = g.get(&k("rope.scaling.type")).ok().and_then(|v| v.as_str()) == Some("yarn");
        Ok(GigaChat4Config {
            n_ctx: crate::ctx_limit(u("context_length")?),
            n_layer,
            n_embd: u("embedding_length")?,
            n_vocab: g.tensors.get("token_embd.weight").context("missing token_embd.weight")?.dims[1] as usize,
            eps: f("attention.layer_norm_rms_epsilon")?,
            recurrent,
            n_head: u("attention.head_count")?,
            q_lora: u("attention.q_lora_rank")?,
            kv_lora,
            k_mla,
            v_mla,
            n_rot,
            rope_base: f("rope.freq_base")?,
            rope_scale: if yarn { f("rope.scaling.factor")? } else { 1.0 },
            rope_ctx_orig: if yarn { u("rope.scaling.original_context_length")? as u32 } else { u("context_length")? as u32 },
            beta_fast: f("rope.scaling.yarn_beta_fast").unwrap_or(32.0),
            beta_slow: f("rope.scaling.yarn_beta_slow").unwrap_or(1.0),
            // llama.cpp reads the GGUF value as 0.1 times the multiplier
            yarn_log_mul: f("rope.scaling.yarn_log_multiplier").map_or(0.0, |v| v / 0.1),
            idx_n_head: u("attention.indexer.head_count")?,
            idx_head_dim: u("attention.indexer.key_length")?,
            idx_top_k: u("attention.indexer.top_k")?,
            ssm_conv: u("ssm.conv_kernel")?,
            ssm_d_state: u("ssm.state_size")?,
            ssm_n_group: u("ssm.group_count")?,
            ssm_dt_rank: u("ssm.time_step_rank")?,
            ssm_d_inner: u("ssm.inner_size")?,
            n_expert: u("expert_count")?,
            n_used: u("expert_used_count")?,
            n_dense: u("leading_dense_block_count")?,
            expert_scale: f("expert_weights_scale")?,
            expert_norm: u("expert_weights_norm")? != 0,
            clamp_exp: arr("swiglu_clamp_exp")?,
            clamp_shexp: arr("swiglu_clamp_shexp")?,
            softcap: f("final_logit_softcapping").unwrap_or(0.0),
        })
    }

    /// YaRN magnitude scale of the attention scores (llama.cpp `yarn_kq_scale` with attn_factor 1).
    pub fn mscale(&self) -> f32 {
        if self.rope_scale > 1.0 { 1.0 + 0.1 * self.yarn_log_mul * (self.rope_scale).ln() } else { 1.0 }
    }
}
