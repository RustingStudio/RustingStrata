//! KRAFTON Raon-1 mini (`raon1`): 4-stream Raon-mHC, hybrid RDA (Raon Delta Attention) + gated NoPE MLA,
//! SiTU-GLU dense/MoE FFN, and per-layer reasoning effort steering.
//! Mirrors `krafton-ai/llama.cpp` `src/models/raon1.cpp` and `KRAFTON/Raon-1-mini` `modeling_raon1_mini.py`.
pub mod chat;
pub mod forward;

use crate::gguf::{Gguf, Value};
use anyhow::{Context, ensure};

pub const ARCH: &str = "raon1";

#[derive(Debug, Clone)]
pub struct RaonConfig {
    pub n_layer: usize,
    pub n_ctx: usize,
    pub n_embd: usize,
    pub n_vocab: usize,
    pub n_head: usize,
    /// `true` for RDA (linear delta attention) layers (`head_count_kv == 0`), `false` for MLA layers (`head_count_kv == 1`).
    pub is_rda: Vec<bool>,
    pub d_conv: usize,
    pub dt_rank: usize,
    pub rda_hd: usize,
    pub gate_lower_bound: Option<f32>,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_rope_hd: usize,
    pub mla_key_len: usize,
    pub mla_val_len: usize,
    pub eps: f32,
    pub latent_eps: f32,
    pub hc: usize,
    pub hc_eps: f32,
    pub hc_mag: f32,
    pub situ_beta: f32,
    pub situ_linear_beta: f32,
    pub n_dense: usize,
    pub n_ff: usize,
    pub n_ff_exp: usize,
    pub n_expert: usize,
    pub n_used: usize,
    pub n_shared: usize,
    pub exp_scale: f32,
    pub exp_norm: bool,
    pub effort_names: Vec<String>,
    pub effort_alpha: Vec<f32>,
    pub effort_default: Option<usize>,
    pub effort_layers: Vec<bool>,
}

impl RaonConfig {
    pub fn from_gguf(g: &Gguf) -> anyhow::Result<Self> {
        let k = |s: &str| format!("{ARCH}.{s}");
        let u = |s: &str| -> anyhow::Result<usize> { Ok(g.get(&k(s))?.as_u64().with_context(|| k(s))? as usize) };
        let f = |s: &str| -> anyhow::Result<f32> { g.get(&k(s))?.as_f32().with_context(|| k(s)) };
        let arr_u = |s: &str| -> anyhow::Result<Vec<u64>> {
            g.get(&k(s))?.as_arr().with_context(|| k(s))?.iter().map(|v| v.as_u64().with_context(|| k(s))).collect()
        };

        let n_layer = u("block_count")?;
        let n_embd = u("embedding_length")?;
        let n_head = u("attention.head_count")?;
        let head_count_kv = arr_u("attention.head_count_kv")?;
        ensure!(head_count_kv.len() == n_layer, "raon1.attention.head_count_kv len {} != {n_layer}", head_count_kv.len());
        let mut is_rda = Vec::with_capacity(n_layer);
        let mut n_rda = 0usize;
        for (i, &kv) in head_count_kv.iter().enumerate() {
            ensure!(kv == 0 || kv == 1, "raon1 layer {i} head_count_kv must be 0 (RDA) or 1 (MLA), got {kv}");
            is_rda.push(kv == 0);
            if kv == 0 { n_rda += 1; }
        }
        ensure!(n_rda > 0 && n_rda < n_layer, "raon1 requires both RDA and MLA layers (got {n_rda}/{n_layer} RDA)");

        let d_conv = u("ssm.conv_kernel")?;
        ensure!(d_conv == 4, "raon1 ssm.conv_kernel must be 4, got {d_conv}");
        let dt_rank = u("ssm.time_step_rank")?;
        let rda_hd = u("kda.head_dim")?;
        ensure!(matches!(rda_hd, 16 | 32 | 64 | 128), "unsupported raon1 kda.head_dim {rda_hd}");
        let gate_lower_bound = g.meta.get(&k("kda.gate_lower_bound")).and_then(Value::as_f32).filter(|v| v.is_finite());

        let q_lora_rank = u("attention.q_lora_rank")?;
        let kv_lora_rank = u("attention.kv_lora_rank")?;
        let qk_rope_hd = u("rope.dimension_count")?;
        let mla_key_len = u("attention.key_length_mla")?;
        let mla_val_len = u("attention.value_length_mla")?;
        ensure!(mla_key_len > qk_rope_hd, "raon1 MLA key_len {mla_key_len} <= rope dim {qk_rope_hd}");

        let hc = u("hyper_connection.count")?;
        ensure!((2..=8).contains(&hc), "raon1 hyper_connection.count {hc} not in 2..=8");
        let n_dense = u("leading_dense_block_count")?;
        ensure!(n_dense <= n_layer, "raon1 leading_dense_block_count {n_dense} > {n_layer}");
        let exp_norm = match g.get(&k("expert_weights_norm"))? {
            Value::Bool(b) => *b,
            v => v.as_u64().with_context(|| k("expert_weights_norm"))? != 0,
        };
        let n_ff_exp = match g.get(&k("expert_feed_forward_length"))? {
            Value::Arr(a) => a.first().and_then(Value::as_u64).context("empty expert_feed_forward_length")? as usize,
            v => v.as_u64().with_context(|| k("expert_feed_forward_length"))? as usize,
        };

        let mut effort_names = Vec::new();
        let mut effort_alpha = Vec::new();
        let mut effort_default = None;
        let mut effort_layers = vec![false; n_layer];
        if let Some(alpha_v) = g.meta.get(&k("reasoning_effort.alpha")) {
            effort_alpha = alpha_v.as_arr().context("reasoning_effort.alpha")?.iter()
                .map(|v| v.as_f32().context("reasoning_effort.alpha entry"))
                .collect::<anyhow::Result<_>>()?;
            effort_names = if let Some(nv) = g.meta.get(&k("reasoning_effort.names")) {
                nv.as_arr().context("reasoning_effort.names")?.iter()
                    .map(|v| v.as_str().map(str::to_string).context("reasoning_effort.names entry"))
                    .collect::<anyhow::Result<_>>()?
            } else {
                vec!["low".into(), "medium".into(), "high".into()]
            };
            ensure!(effort_names.len() == effort_alpha.len() && !effort_alpha.is_empty(), "raon1 reasoning_effort mismatch");
            if let Some(dv) = g.meta.get(&k("reasoning_effort.default")).and_then(Value::as_str) {
                effort_default = effort_names.iter().position(|s| s == dv);
            }
            for il in arr_u("reasoning_effort.layers")? {
                let il = il as usize;
                ensure!(il >= n_dense && il < n_layer, "raon1 reasoning_effort layer {il} out of MoE range");
                effort_layers[il] = true;
            }
        }

        let n_vocab = g.tensors.get("token_embd.weight").context("missing token_embd.weight")?.dims[1] as usize;
        Ok(RaonConfig {
            n_layer,
            n_ctx: crate::ctx_limit(g.meta.get(&k("context_length")).and_then(Value::as_u64).map_or(32768, |v| v as usize)),
            n_embd,
            n_vocab,
            n_head,
            is_rda,
            d_conv,
            dt_rank,
            rda_hd,
            gate_lower_bound,
            q_lora_rank,
            kv_lora_rank,
            qk_rope_hd,
            mla_key_len,
            mla_val_len,
            eps: f("attention.layer_norm_rms_epsilon")?,
            latent_eps: f("attention.latent_norm_rms_epsilon")?,
            hc,
            hc_eps: f("hyper_connection.epsilon")?,
            hc_mag: f("hyper_connection.magnitude")?,
            situ_beta: f("activation.situ_beta")?,
            situ_linear_beta: f("activation.situ_linear_beta")?,
            n_dense,
            n_ff: u("feed_forward_length")?,
            n_ff_exp,
            n_expert: u("expert_count")?,
            n_used: u("expert_used_count")?,
            n_shared: u("expert_shared_count")?,
            exp_scale: f("expert_weights_scale")?,
            exp_norm,
            effort_names,
            effort_alpha,
            effort_default,
            effort_layers,
        })
    }

    /// Resolve an effort name (`none|low|medium|high|xhigh|max`) to its steering alpha (0.0 when disabled).
    pub fn effort_alpha_for(&self, effort: Option<&str>, think: bool) -> f32 {
        if self.effort_alpha.is_empty() {
            return 0.0;
        }
        if let Some(name) = effort.map(str::trim).filter(|s| !s.is_empty()) {
            if let Some(idx) = self.effort_names.iter().position(|n| n == name) {
                return self.effort_alpha[idx];
            }
        }
        if !think {
            if let Some(idx) = self.effort_names.iter().position(|n| n == "none") {
                return self.effort_alpha[idx];
            }
            return 0.0;
        }
        self.effort_default.map_or(0.0, |i| self.effort_alpha[i])
    }
}
