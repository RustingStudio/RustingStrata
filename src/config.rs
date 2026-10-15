//! Model hyperparameters read from GGUF metadata, plus the per-architecture table.

use crate::gguf::Gguf;
use anyhow::{Context, bail};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind { Delta, Attn }

/// What differs between the supported architectures; everything else comes from GGUF metadata.
struct Arch {
    /// DeltaNet layers with full attention every `full_attention_interval` layers; otherwise all attention.
    hybrid: bool,
    /// `attn_q` is [query | sigmoid output gate] per head (2 * head_dim wide).
    q_gate: bool,
    /// DeltaNet value head h reads key head h / (nv / nk) (llama.cpp qwen3next), not h % nk.
    k_interleave: bool,
    /// Canonical tensor suffix to this architecture's suffix.
    aliases: &'static [(&'static str, &'static str)],
}

fn arch(name: &str) -> Option<Arch> {
    const FFN_NORM: &[(&str, &str)] = &[("post_attention_norm.weight", "ffn_norm.weight")];
    Some(match name {
        "qwen35" | "qwen35moe" => Arch { hybrid: true, q_gate: true, k_interleave: false, aliases: &[] },
        "qwen3next" => Arch { hybrid: true, q_gate: true, k_interleave: true, aliases: &[] },
        "qwen3" | "qwen3moe" => Arch { hybrid: false, q_gate: false, k_interleave: false, aliases: FFN_NORM },
        _ => return None,
    })
}

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
    /// DeltaNet sizes, 0 without DeltaNet layers.
    pub ssm_d_inner: usize,
    pub ssm_d_state: usize,
    pub ssm_n_group: usize,
    pub ssm_dt_rank: usize,
    pub ssm_conv: usize,
    /// Mixer of each main layer (MTP layers are attention).
    pub layers: Vec<LayerKind>,
    pub q_gate: bool,
    pub k_interleave: bool,
    pub aliases: &'static [(&'static str, &'static str)],
}

impl ModelConfig {
    pub fn from_gguf(g: &Gguf) -> anyhow::Result<ModelConfig> {
        let arch = g.get("general.architecture")?.as_str().context("general.architecture")?.to_string();
        let Some(a) = self::arch(&arch) else {
            bail!("unsupported architecture '{arch}' (supported: qwen3, qwen3moe, qwen3next, qwen35, qwen35moe)")
        };
        let opt = |k: &str| g.meta.get(&format!("{arch}.{k}")).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let req = |k: &str| -> anyhow::Result<usize> {
            let key = format!("{arch}.{k}");
            Ok(g.get(&key)?.as_u64().with_context(|| format!("{key}: not an integer"))? as usize)
        };
        let f = |k: &str| g.meta.get(&format!("{arch}.{k}")).and_then(|v| v.as_f32());
        let n_nextn = opt("nextn_predict_layers");
        let n_layer = req("block_count")? - n_nextn;
        let layers: Vec<LayerKind> = if a.hybrid {
            let every = match opt("full_attention_interval") { 0 => 4, v => v };
            (0..n_layer).map(|il| if (il + 1) % every == 0 { LayerKind::Attn } else { LayerKind::Delta }).collect()
        } else {
            vec![LayerKind::Attn; n_layer]
        };
        let has_delta = layers.contains(&LayerKind::Delta);
        let ssm = |k: &str| if has_delta { req(k) } else { Ok(opt(k)) };
        let (n_embd, n_head) = (req("embedding_length")?, req("attention.head_count")?);
        let head_dim = match opt("attention.key_length") { 0 => n_embd / n_head, v => v };
        let n_expert = opt("expert_count");
        anyhow::ensure!(n_expert <= 512, "{n_expert} experts (at most 512 supported)");
        let emb = g.tensors.get("token_embd.weight").context("missing tensor token_embd.weight")?;
        Ok(ModelConfig {
            n_layer,
            n_nextn,
            n_embd,
            n_head,
            n_head_kv: req("attention.head_count_kv")?,
            head_dim,
            n_rot: match opt("rope.dimension_count") { 0 => head_dim, v => v },
            rope_base: f("rope.freq_base").unwrap_or(10000.0),
            eps: f("attention.layer_norm_rms_epsilon").context("missing rms epsilon")?,
            n_vocab: emb.dims[1] as usize,
            n_ff: opt("feed_forward_length"),
            n_expert,
            n_expert_used: opt("expert_used_count"),
            n_ff_exp: opt("expert_feed_forward_length"),
            n_ff_shexp: opt("expert_shared_feed_forward_length"),
            expert_weights_scale: f("expert_weights_scale").unwrap_or(0.0),
            ssm_d_inner: ssm("ssm.inner_size")?,
            ssm_d_state: ssm("ssm.state_size")?,
            ssm_n_group: ssm("ssm.group_count")?,
            ssm_dt_rank: ssm("ssm.time_step_rank")?,
            ssm_conv: ssm("ssm.conv_kernel")?,
            layers,
            q_gate: a.q_gate,
            k_interleave: a.k_interleave,
            aliases: a.aliases,
            arch,
        })
    }

    pub fn is_moe(&self) -> bool { self.n_expert > 0 }
    /// DeltaNet value head size (0 without DeltaNet layers).
    pub fn ssm_dv(&self) -> usize { self.ssm_d_inner / self.ssm_dt_rank.max(1) }
}
