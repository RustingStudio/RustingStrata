//! Model hyperparameters read from GGUF metadata.

use crate::gguf::Gguf;
use anyhow::{Context, bail};

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub arch: String,
    pub n_layer: usize,
    pub n_nextn: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub rope_base: f32,
    pub eps: f32,
    pub n_vocab: usize,
    /// Dense FFN size (0 for MoE).
    pub n_ff: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub n_ff_shexp: usize,
    /// 0 = no extra scale.
    pub expert_weights_scale: f32,
    pub ssm_d_inner: usize,
    pub ssm_d_state: usize,
    pub ssm_n_group: usize,
    pub ssm_dt_rank: usize,
    pub ssm_conv: usize,
    pub full_attention_interval: usize,
}

impl ModelConfig {
    pub fn from_gguf(g: &Gguf) -> anyhow::Result<ModelConfig> {
        let arch = g.get("general.architecture")?.as_str().context("general.architecture")?.to_string();
        if arch != "qwen35" && arch != "qwen35moe" { bail!("unsupported architecture '{arch}'"); }
        let opt = |k: &str| g.meta.get(&format!("{arch}.{k}")).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let req = |k: &str| -> anyhow::Result<usize> {
            let key = format!("{arch}.{k}");
            Ok(g.get(&key)?.as_u64().with_context(|| format!("{key}: not an integer"))? as usize)
        };
        let f = |k: &str| g.meta.get(&format!("{arch}.{k}")).and_then(|v| v.as_f32());
        let n_nextn = opt("nextn_predict_layers");
        let emb = g.tensors.get("token_embd.weight").context("missing tensor token_embd.weight")?;
        Ok(ModelConfig {
            n_layer: req("block_count")? - n_nextn,
            n_nextn,
            n_embd: req("embedding_length")?,
            n_head: req("attention.head_count")?,
            n_head_kv: req("attention.head_count_kv")?,
            head_dim: req("attention.key_length")?,
            n_rot: req("rope.dimension_count")?,
            rope_base: f("rope.freq_base").unwrap_or(10000.0),
            eps: f("attention.layer_norm_rms_epsilon").context("missing rms epsilon")?,
            n_vocab: emb.dims[1] as usize,
            n_ff: opt("feed_forward_length"),
            n_expert: opt("expert_count"),
            n_expert_used: opt("expert_used_count"),
            n_ff_exp: opt("expert_feed_forward_length"),
            n_ff_shexp: opt("expert_shared_feed_forward_length"),
            expert_weights_scale: f("expert_weights_scale").unwrap_or(0.0),
            ssm_d_inner: req("ssm.inner_size")?,
            ssm_d_state: req("ssm.state_size")?,
            ssm_n_group: req("ssm.group_count")?,
            ssm_dt_rank: req("ssm.time_step_rank")?,
            ssm_conv: req("ssm.conv_kernel")?,
            full_attention_interval: match opt("full_attention_interval") { 0 => 4, v => v },
            arch,
        })
    }

    pub fn is_recurrent(&self, il: usize) -> bool { (il + 1) % self.full_attention_interval != 0 }
    pub fn is_moe(&self) -> bool { self.n_expert > 0 }
}
