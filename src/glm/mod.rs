//! GLM-5.3-Flash (GGUF arch `glm5-next`): hyperparameters and tensor-table validation.
//! Mirrors llama.cpp `src/models/glm5-next.cpp` (hparams and tensor creation).

pub mod forward;
use crate::gguf::Gguf;
use anyhow::{Context, ensure};

/// Token mixer of a layer: Kimi Delta Attention (linear) or DSA (MLA with a k-pool indexer).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind { Kda, Dsa }

#[derive(Clone, Debug)]
pub struct GlmConfig {
    /// All blocks in the file, including the trailing NextN (MTP) block.
    pub n_layer_file: usize,
    /// Blocks used for normal decoding (file blocks minus NextN blocks).
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_vocab: usize,
    pub n_ff: usize,
    pub n_dense: usize,
    pub eps: f32,
    pub kda_eps: f32,
    pub kind: Vec<LayerKind>,
    pub conv_k: usize,
    pub kda_head_dim: usize,
    pub gate_lower_bound: f32,
    pub q_lora: usize,
    pub kv_lora: usize,
    /// MLA per-head widths before absorption (k = v = 256).
    pub k_mla: usize,
    pub v_mla: usize,
    pub idx_n_head: usize,
    pub idx_head_dim: usize,
    pub idx_top_k: usize,
    pub kpool: usize,
    pub hc_n: usize,
    pub hc_eps: f32,
    pub hc_iters: usize,
    pub n_expert: usize,
    pub n_used: usize,
    pub n_ff_exp: usize,
    pub n_ff_shexp: usize,
    pub n_shared: usize,
    pub weights_norm: bool,
    pub weights_scale: f32,
    pub clamp_exp: Vec<f32>,
    pub clamp_shexp: Vec<f32>,
    pub n_nextn: usize,
}

impl GlmConfig {
    pub fn from_gguf(g: &Gguf) -> anyhow::Result<GlmConfig> {
        let arch = g.get("general.architecture")?.as_str().unwrap_or("");
        ensure!(arch == "glm5-next", "architecture '{arch}' is not glm5-next");
        let k = |s: &str| format!("glm5-next.{s}");
        let u = |s: &str| -> anyhow::Result<usize> { Ok(g.get(&k(s))?.as_u64().with_context(|| k(s))? as usize) };
        let f = |s: &str| -> anyhow::Result<f32> { g.get(&k(s))?.as_f32().with_context(|| k(s)) };
        let arr_f = |s: &str| -> anyhow::Result<Vec<f32>> {
            g.get(&k(s))?.as_arr().with_context(|| k(s))?.iter().map(|v| v.as_f32().with_context(|| k(s))).collect()
        };
        let n_layer_file = u("block_count")?;
        let n_nextn = u("nextn_predict_layers")?;
        ensure!(n_nextn <= n_layer_file, "nextn_predict_layers > block_count");
        let kv_heads: Vec<usize> = g.get(&k("attention.head_count_kv"))?.as_arr().context("head_count_kv")?
            .iter().map(|v| v.as_u64().context("head_count_kv entry")).collect::<anyhow::Result<Vec<_>>>()?
            .into_iter().map(|x| x as usize).collect();
        ensure!(kv_heads.len() >= n_layer_file, "head_count_kv has {} entries for {n_layer_file} layers", kv_heads.len());
        let kind = kv_heads.iter().take(n_layer_file).map(|&h| if h == 0 { LayerKind::Kda } else { LayerKind::Dsa }).collect();
        let emb = g.tensors.get("token_embd.weight").context("missing token_embd.weight")?;
        let c = GlmConfig {
            n_layer_file,
            n_layer: n_layer_file - n_nextn,
            n_embd: u("embedding_length")?,
            n_head: u("attention.head_count")?,
            n_vocab: emb.dims[1] as usize,
            n_ff: u("feed_forward_length")?,
            n_dense: u("leading_dense_block_count")?,
            eps: f("attention.layer_norm_rms_epsilon")?,
            kda_eps: f("attention.layer_norm_epsilon")?,
            kind,
            conv_k: u("ssm.conv_kernel")?,
            kda_head_dim: u("kda.head_dim")?,
            gate_lower_bound: f("kda.gate_lower_bound")?,
            q_lora: u("attention.q_lora_rank")?,
            kv_lora: u("attention.kv_lora_rank")?,
            k_mla: u("attention.key_length_mla")?,
            v_mla: u("attention.value_length_mla")?,
            idx_n_head: u("attention.indexer.head_count")?,
            idx_head_dim: u("attention.indexer.key_length")?,
            idx_top_k: u("attention.indexer.top_k")?,
            kpool: u("attention.indexer.kpool")?,
            hc_n: u("hyper_connection.count")?,
            hc_eps: f("hyper_connection.epsilon")?,
            hc_iters: u("hyper_connection.sinkhorn_iterations")?,
            n_expert: u("expert_count")?,
            n_used: u("expert_used_count")?,
            n_ff_exp: u("expert_feed_forward_length")?,
            n_ff_shexp: u("expert_shared_feed_forward_length")?,
            n_shared: u("expert_shared_count")?,
            weights_norm: g.get(&k("expert_weights_norm"))?.as_u64().unwrap_or(0) != 0,
            weights_scale: f("expert_weights_scale")?,
            clamp_exp: arr_f("swiglu_clamp_exp")?,
            clamp_shexp: arr_f("swiglu_clamp_shexp")?,
            n_nextn,
        };
        ensure!(u("expert_gating_func")? == 2, "only sigmoid expert gating (2) is supported");
        ensure!(c.hc_n == 4, "only 4 hyper-connection streams are supported");
        ensure!(c.n_shared == 1 && c.n_ff_shexp == c.n_ff_exp, "unexpected shared-expert shape");
        ensure!(c.n_head * c.kda_head_dim == 8192 && c.kda_head_dim == 128, "unexpected KDA dimensions");
        ensure!(c.clamp_exp.len() >= n_layer_file && c.clamp_shexp.len() >= n_layer_file, "swiglu clamp arrays too short");
        ensure!(c.n_dense <= c.n_layer, "leading_dense_block_count out of range");
        Ok(c)
    }

    pub fn n_kda(&self) -> usize { self.kind[..self.n_layer].iter().filter(|&&k| k == LayerKind::Kda).count() }
    pub fn n_dsa(&self) -> usize { self.n_layer - self.n_kda() }

    /// Every tensor of the model with its expected dims (GGUF order: fastest first), as llama.cpp creates them.
    pub fn expected_tensors(&self) -> Vec<(String, Vec<u64>)> {
        let (e, h, nh) = (self.n_embd as u64, self.hc_n as u64, self.n_head as u64);
        let (ff, ffe, ne) = (self.n_ff as u64, self.n_ff_exp as u64, self.n_expert as u64);
        let (ql, kl, kd) = (self.q_lora as u64, self.kv_lora as u64, self.k_mla as u64);
        let (vd, di) = (self.v_mla as u64, nh * self.kda_head_dim as u64);
        let (ni, nih, hd) = (self.idx_head_dim as u64, self.idx_n_head as u64, self.kda_head_dim as u64);
        let mix = (2 + h) * h;
        let mut v: Vec<(String, Vec<u64>)> = vec![
            ("token_embd.weight".into(), vec![e, self.n_vocab as u64]),
            ("output_norm.weight".into(), vec![e]),
            ("output.weight".into(), vec![e, self.n_vocab as u64]),
        ];
        for il in 0..self.n_layer_file {
            let b = |n: &str| format!("blk.{il}.{n}");
            let mut t = |n: &str, d: Vec<u64>| v.push((b(n), d));
            t("attn_norm.weight", vec![e]);
            t("ffn_norm.weight", vec![e]);
            for s in ["attn", "ffn"].into_iter().filter(|_| il < self.n_layer) { // the NextN block has no hyper-connections
                t(&format!("hc_{s}_fn.weight"), vec![h * e, mix]);
                t(&format!("hc_{s}_base.weight"), vec![mix]);
                t(&format!("hc_{s}_scale.weight"), vec![3]);
            }
            match self.kind[il] {
                LayerKind::Kda => {
                    t("attn_q.weight", vec![e, di]);
                    t("attn_k.weight", vec![e, di]);
                    t("attn_v.weight", vec![e, di]);
                    t("attn_output.weight", vec![di, e]);
                    t("ssm_a", vec![nh]);
                    t("ssm_beta.weight", vec![e, nh]);
                    for s in ["q", "k", "v"] { t(&format!("ssm_conv1d_{s}.weight"), vec![self.conv_k as u64, 1, di]); }
                    t("ssm_dt.bias", vec![di]);
                    t("ssm_f_a.weight", vec![e, hd]);
                    t("ssm_f_b.weight", vec![hd, di]);
                    t("ssm_g_a.weight", vec![e, hd]);
                    t("ssm_g_b.weight", vec![hd, di]);
                    t("ssm_norm.weight", vec![hd]);
                }
                LayerKind::Dsa => {
                    t("attn_q_a.weight", vec![e, ql]);
                    t("attn_q_a_norm.weight", vec![ql]);
                    t("attn_q_b.weight", vec![ql, nh * kd]);
                    t("attn_kv_a_mqa.weight", vec![e, kl]);
                    t("attn_kv_a_norm.weight", vec![kl]);
                    t("attn_k_b.weight", vec![kd, kl, nh]);
                    t("attn_v_b.weight", vec![kl, vd, nh]);
                    t("attn_output.weight", vec![nh * vd, e]);
                    t("indexer.attn_k.weight", vec![e, ni]);
                    t("indexer.attn_q_b.weight", vec![ql, nih * ni]);
                    t("indexer.k_norm.weight", vec![ni]);
                    t("indexer.k_norm.bias", vec![ni]);
                    t("indexer.proj.weight", vec![e, nih]);
                    t("indexer_compressor_ape.weight", vec![ni, self.kpool as u64]);
                    t("indexer_compressor_gate.weight", vec![e, ni]);
                }
            }
            if il < self.n_dense {
                t("ffn_gate.weight", vec![e, ff]);
                t("ffn_up.weight", vec![e, ff]);
                t("ffn_down.weight", vec![ff, e]);
            } else {
                t("ffn_gate_inp.weight", vec![e, ne]);
                t("exp_probs_b.bias", vec![ne]);
                t("ffn_gate_exps.weight", vec![e, ffe, ne]);
                t("ffn_up_exps.weight", vec![e, ffe, ne]);
                t("ffn_down_exps.weight", vec![ffe, e, ne]);
                let fs = ffe * self.n_shared as u64;
                t("ffn_gate_shexp.weight", vec![e, fs]);
                t("ffn_up_shexp.weight", vec![e, fs]);
                t("ffn_down_shexp.weight", vec![fs, e]);
            }
            if il >= self.n_layer {
                t("nextn.eh_proj.weight", vec![2 * e, e]);
                t("nextn.enorm.weight", vec![e]);
                t("nextn.hnorm.weight", vec![e]);
                t("nextn.shared_head_norm.weight", vec![e]);
            }
        }
        v
    }

    /// Check every expected tensor exists with the expected dims and a supported quant type, and nothing else is in the file.
    pub fn validate(&self, g: &Gguf) -> anyhow::Result<()> {
        let mut errs = vec![];
        let exp = self.expected_tensors();
        for (name, dims) in &exp {
            match g.tensors.get(name) {
                None => errs.push(format!("missing tensor {name}")),
                Some(t) => {
                    // Per-expert and 2-D conv tensors may carry trailing 1s; compare ignoring them.
                    let trim = |d: &[u64]| { let mut d = d.to_vec(); while d.len() > 1 && d.last() == Some(&1) { d.pop(); } d };
                    if trim(&t.dims) != trim(dims) { errs.push(format!("{name}: dims {:?}, expected {:?}", t.dims, dims)); }
                    if let Err(e) = crate::quant::GgmlType::from_u32(t.ty) { errs.push(format!("{name}: {e}")); }
                }
            }
        }
        if g.tensors.len() != exp.len() {
            let known: std::collections::HashSet<&str> = exp.iter().map(|(n, _)| n.as_str()).collect();
            for n in g.tensors.keys().filter(|n| !known.contains(n.as_str())) { errs.push(format!("unexpected tensor {n}")); }
        }
        ensure!(errs.is_empty(), "{} tensor problems:\n{}", errs.len(), errs.iter().take(40).cloned().collect::<Vec<_>>().join("\n"));
        Ok(())
    }
}

/// `rustingstrata generate` for glm5-next: raw prompt (no template), greedy, prints the continuation. CPU reference only.
pub fn generate(g: &Gguf, prompt: &str, n: usize) -> anyhow::Result<()> {
    use std::io::Write;
    let tok = crate::tokenizer::Tokenizer::from_gguf(g)?;
    let mut m = forward::GlmModel::new(g)?;
    let eos = tok.eos_ids();
    let ids = tok.encode(prompt)?;
    if std::env::var_os("STRATA_GLM_TRACE").is_some() { eprintln!("prompt ids {ids:?}"); }
    let mut logits = m.prefill(&ids);
    let (mut out, mut shown) = (vec![], 0);
    for i in 0..n {
        let next = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
        if eos.contains(&next) { break; }
        out.push(next);
        let s = tok.decode(&out)?;
        if !s.ends_with('\u{FFFD}') && s.len() > shown && s.is_char_boundary(shown) {
            print!("{}", &s[shown..]);
            std::io::stdout().flush()?;
            shown = s.len();
        }
        if i + 1 < n { logits = m.step(next); }
    }
    println!();
    Ok(())
}
