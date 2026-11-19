//! Scalar CPU reference forward for the architectures in `config::arch`, one token at a time.
//! Mirrors llama.cpp `src/models/qwen35moe.cpp`; this is the correctness oracle, not the fast path.

use crate::config::{DFlashConfig, Eagle3Config, LayerKind, ModelConfig};
use crate::gguf::Gguf;
use crate::ops;
use crate::weights::{Tensor, Weights};
use anyhow::Context;
use std::collections::HashMap;
use std::path::Path;

#[derive(Clone)]
pub struct LoraEntry {
    pub rank: usize,
    pub in_dim: usize,
    pub out_dim: usize,
    pub n_expert: usize,
    pub a: Vec<Vec<f32>>,
    pub b: Vec<Vec<f32>>,
}

#[derive(Clone)]
pub struct Lora {
    pub scale: f32,
    pub alpha: f32,
    pub entries: HashMap<String, LoraEntry>,
}

pub struct Model<'a> {
    pub cfg: ModelConfig,
    pub w: Weights<'a>,
    pub loras: Vec<Lora>,
    pub draft_model: Option<Box<Model<'a>>>,
    ple: Option<Ple>,
}

impl Model<'_> {
    /// Host memory this process owns outright: baked/synthesized tensors and LoRA factors. The mmapped GGUF is
    /// evictable page cache and is not counted, so models larger than RAM still load.
    pub fn ram_estimate(&self) -> usize {
        let lora: usize = self.loras.iter().flat_map(|l| l.entries.values()).map(|e| (e.a.len() * e.in_dim + e.b.len() * e.out_dim) * 4).sum();
        let dm = self.draft_model.as_ref().map_or(0, |d| d.ram_estimate());
        self.w.built.values().map(|t| t.data.len()).sum::<usize>() + lora + dm
    }
}

/// qwen4exp per-layer embedding (PLE): n-gram hash constants from the GGUF.
struct Ple { mult: Vec<u64>, off: Vec<u64>, vocab: Vec<u64>, eos: i64, heads_per_ngram: usize }

// PLE geometry, checked against the GGUF at load: n-grams up to 3 tokens, `heads_per_ngram` hash heads per n-gram order,
// a 4-tap causal conv dilated by the n-gram size, so it reads rows t-9, t-6, t-3, t.
const PLE_NGRAM: usize = 3;
const PLE_KERNEL: usize = 4;
const PLE_HIST: usize = (PLE_KERNEL - 1) * PLE_NGRAM;
/// QSA: cells per pooled indexer key.
const QSA_POOL: usize = 4;

pub enum LayerState {
    /// conv: last (ssm_conv - 1) inputs per channel, oldest first; s: dt_rank x head_v x head_k.
    Recurrent { conv: Vec<f32>, s: Vec<f32> },
    /// k, v: pos x n_head_kv x head_dim; ik: pos x idx_head_dim raw QSA indexer keys (empty without an indexer).
    Attn { k: Vec<f32>, v: Vec<f32>, ik: Vec<f32> },
}

pub struct State {
    pub layers: Vec<LayerState>,
    pub pos: usize,
    /// M-RoPE positions for the next token: `[t, y, x, z]`. Text tokens use `pos` in all four slots.
    pub rope_pos: [i32; 4],
    /// qwen4exp PLE: the last PLE_HIST normalized rows (hc * n_embd each, oldest first), and the two previous
    /// tokens (newest first, -1 for none) for the n-gram hash.
    pub ple_hist: Vec<f32>,
    pub ple_prev: [i64; 2],
}

impl State {
    pub fn new(c: &ModelConfig) -> State {
        let channels = c.ssm_d_inner + 2 * c.ssm_n_group * c.ssm_d_state;
        let layers = c.layers.iter()
            .map(|k| match k {
                LayerKind::Delta => LayerState::Recurrent {
                    conv: vec![0.0; (c.ssm_conv - 1) * channels],
                    s: vec![0.0; c.ssm_dt_rank * c.ssm_dv() * c.ssm_d_state],
                },
                LayerKind::Attn => LayerState::Attn { k: Vec::new(), v: Vec::new(), ik: Vec::new() },
            })
            .collect();
        let ple = if c.n_embd_ple > 0 { PLE_HIST * c.hc_n * c.n_embd } else { 0 };
        State { layers, pos: 0, rope_pos: [0; 4], ple_hist: vec![0.0; ple], ple_prev: [-1, -1] }
    }
}

/// MTP draft state: the KV cache of the nextn block.
pub struct Mtp { pub kv: LayerState, pub pos: usize }

impl Default for Mtp {
    fn default() -> Mtp { Mtp { kv: LayerState::Attn { k: Vec::new(), v: Vec::new(), ik: Vec::new() }, pos: 0 } }
}

fn mv(t: &Tensor, x: &[f32]) -> Vec<f32> {
    let mut y = vec![0f32; t.rows];
    t.matvec(x, &mut y);
    y
}

impl<'a> Model<'a> {
    pub fn load(g: &'a Gguf) -> anyhow::Result<Model<'a>> { Self::load_with_lora(g, &[]) }
    pub fn load_with_lora(g: &'a Gguf, loras: &[(String, f32)]) -> anyhow::Result<Model<'a>> {
        Self::load_with_aux(g, None, loras)
    }
    pub fn load_with_aux(g: &'a Gguf, aux: Option<&'a Gguf>, loras: &[(String, f32)]) -> anyhow::Result<Model<'a>> {
        let mut cfg = ModelConfig::from_gguf(g)?;
        // A DFlash draft given on the command line replaces the target's own MTP head; without this the draft was
        // ignored without a word for a target GGUF that carries one.
        if cfg.n_nextn > 0 && aux.is_some_and(|ag| ag.tensors.contains_key("dflash.fc.weight")) {
            eprintln!("note: DFlash draft model given: the target's own MTP head is not used");
            cfg.n_nextn = 0;
        }
        let is_standalone_draft = aux.is_some_and(|ag| {
            !ag.tensors.contains_key("fc.weight")
                && !ag.tensors.contains_key("dflash.fc.weight")
                && !ag.tensors.contains_key(&format!("blk.{}.nextn.eh_proj.weight", cfg.n_layer))
                && ag.tensors.contains_key("token_embd.weight")
                && ModelConfig::from_gguf(ag).is_ok()
        });
        let w = Weights::new_with_aux(g, if is_standalone_draft { None } else { aux })?;
        let mut draft_model = None;
        if is_standalone_draft {
            let ag = aux.unwrap();
            let dm = Model::load(ag)?;
            anyhow::ensure!(
                dm.cfg.n_vocab == cfg.n_vocab,
                "standalone draft model vocab ({}) != target vocab ({})",
                dm.cfg.n_vocab,
                cfg.n_vocab
            );
            eprintln!(
                "note: loaded standalone external draft model (arch={}, layers={}, embd={}, vocab={})",
                dm.cfg.arch, dm.cfg.n_layer, dm.cfg.n_embd, dm.cfg.n_vocab
            );
            draft_model = Some(Box::new(dm));
        } else if cfg.n_nextn == 0 {
            if let Some(ag) = aux.filter(|_| w.layer(cfg.n_layer, "eagle3.fc.weight").is_ok()) {
                let il = cfg.n_layer;
                let wq = w.layer(il, "attn_q.weight")?;
                let wk = w.layer(il, "attn_k.weight")?;
                let gate = w.layer(il, "ffn_gate.weight")?;
                let out = w.layer(il, "eagle3.output.weight")?;
                let n_head = ag.meta.get("eagle3.attention.head_count").and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(cfg.n_head);
                let head_dim = wq.rows / n_head.max(1);
                let n_head_kv = (wk.rows / head_dim.max(1)).max(1);
                let n_ff = gate.rows;
                let draft_vocab = out.rows;
                let n_rot = ag.meta.get("eagle3.rope.dimension_count").and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(head_dim);
                let rope_base = ag.meta.get("eagle3.rope.freq_base").and_then(|v| v.as_f32()).unwrap_or(cfg.rope_base);
                let eps = ag.meta.get("eagle3.attention.layer_norm_rms_epsilon").and_then(|v| v.as_f32()).unwrap_or(cfg.eps);
                let norm_before_residual = matches!(ag.meta.get("eagle3.norm_before_residual"), Some(crate::gguf::Value::Bool(true)));
                let rope_neox = !matches!(ag.meta.get("eagle3.rope.neox"), Some(crate::gguf::Value::Bool(false)));
                let extract_layers = match ag.meta.get("eagle3.target_layers").or_else(|| ag.meta.get("eagle3.extract_layers")).and_then(|v| v.as_arr()) {
                    Some(arr) if arr.len() == 3 => {
                        let mut ls = [0usize; 3];
                        for (i, v) in arr.iter().enumerate() {
                            ls[i] = v.as_u64().context("bad eagle3 extract layer")? as usize;
                        }
                        ls
                    }
                    _ => [2, cfg.n_layer / 2, cfg.n_layer.saturating_sub(3)],
                };
                let d2t = if let Some(t) = ag.tensors.get("d2t") {
                    let n = t.dims[0] as usize;
                    anyhow::ensure!(n == draft_vocab, "d2t len {n} != draft_vocab {draft_vocab}");
                    let mut ids: Vec<i32> = match t.ty {
                        26 => ag.raw_bytes(t, 4 * n)?.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect(),
                        27 => ag.raw_bytes(t, 8 * n)?.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap()) as i32).collect(),
                        ty => anyhow::bail!("unsupported EAGLE3 d2t tensor type {ty}"),
                    };
                    if ids.windows(2).any(|w| w[0] >= w[1]) {
                        for (i, v) in ids.iter_mut().enumerate() { *v += i as i32; }
                    }
                    ids
                } else {
                    (0..draft_vocab as i32).collect()
                };
                cfg.n_nextn = 1;
                cfg.eagle3 = Some(Eagle3Config {
                    extract_layers, n_ff, n_head, n_head_kv, head_dim, n_rot, rope_base, eps,
                    draft_vocab, norm_before_residual, rope_neox, d2t,
                });
                eprintln!(
                    "note: loaded standalone EAGLE3 draft head (extract_layers={extract_layers:?}, heads={n_head}x{head_dim}, kv={n_head_kv}, ff={n_ff}, draft_vocab={draft_vocab})"
                );
            } else if aux.is_some_and(|ag| ag.tensors.contains_key("dflash.fc.weight")) {
                let ag = aux.unwrap();
                let u = |k: &str| ag.meta.get(k).and_then(|v| v.as_u64()).map(|v| v as usize);
                let f = |k: &str| ag.meta.get(k).and_then(|v| v.as_f32());
                let n_layer = u("dflash.block_count").context("dflash.block_count")?;
                let block_size = u("dflash.block_size").unwrap_or(16).max(2);
                let mask_token_id = u("dflash.mask_token_id").context("dflash.mask_token_id")? as u32;
                let n_embd = u("dflash.embedding_length").unwrap_or(cfg.n_embd);
                let n_ff = u("dflash.feed_forward_length").context("dflash.feed_forward_length")?;
                let n_head = u("dflash.attention.head_count").unwrap_or(cfg.n_head);
                let n_head_kv = u("dflash.attention.head_count_kv").unwrap_or(cfg.n_head_kv);
                let head_dim = u("dflash.attention.key_length").unwrap_or(n_embd / n_head.max(1));
                let n_rot = u("dflash.rope.dimension_count").unwrap_or(head_dim);
                let rope_base = f("dflash.rope.freq_base").unwrap_or(cfg.rope_base);
                let eps = f("dflash.attention.layer_norm_rms_epsilon").unwrap_or(cfg.eps);
                let sliding_window = u("dflash.attention.sliding_window").unwrap_or(0);
                let input_embedding_scale = f("dflash.input_embedding_scale").unwrap_or(1.0);
                let output_multiplier = f("dflash.output_multiplier").unwrap_or(1.0);
                let conv_kernel_size = u("dflash.conv_kernel_size").unwrap_or(0);
                let conv_group_size = u("dflash.conv_group_size").unwrap_or(0);
                let selector_rank = u("dflash.selector_rank").unwrap_or(0);
                let selector_top_k = u("dflash.selector_top_k").unwrap_or(0);
                let target_layer_ids = ag.meta.get("dflash.target_layers").and_then(|v| v.as_arr())
                    .context("dflash.target_layers")?
                    .iter().map(|v| v.as_u64().map(|x| x as usize).context("bad dflash target layer")).collect::<anyhow::Result<Vec<_>>>()?;
                anyhow::ensure!(target_layer_ids.iter().all(|&id| id < cfg.n_layer), "dflash target_layer_ids out of range for {} layers: {target_layer_ids:?}", cfg.n_layer);
                let layer_swa = ag.meta.get("dflash.attention.sliding_window_pattern").and_then(|v| v.as_arr())
                    .filter(|a| a.len() >= n_layer)
                    .map(|a| a[..n_layer].iter().map(|v| v.as_u64().unwrap_or(0) != 0).collect())
                    .unwrap_or_else(|| vec![false; n_layer]);
                cfg.dflash = Some(DFlashConfig {
                    target_layer_ids: target_layer_ids.clone(),
                    block_size,
                    mask_token_id,
                    n_layer,
                    n_embd,
                    n_ff,
                    n_head,
                    n_head_kv,
                    head_dim,
                    n_rot,
                    rope_base,
                    eps,
                    sliding_window,
                    layer_swa,
                    input_embedding_scale,
                    output_multiplier,
                    conv_kernel_size,
                    conv_group_size,
                    selector_rank,
                    selector_top_k,
                });
                eprintln!(
                    "note: loaded standalone DFlash{} draft model (layers={n_layer}, block={block_size}, mask={mask_token_id}, target_layers={target_layer_ids:?}, heads={n_head}x{head_dim}, kv={n_head_kv}, ff={n_ff})",
                    if selector_rank > 0 { "2" } else { "" }
                );
            } else if aux.is_some_and(|ag| ag.tensors.contains_key(&format!("blk.{}.nextn.eh_proj.weight", cfg.n_layer))) {
                cfg.n_nextn = 1;
                eprintln!("note: loaded standalone MTP draft head from companion GGUF (layer {})", cfg.n_layer);
            }
        }
        if !w.validate(cfg.n_layer)? && cfg.n_nextn > 0 {
            eprintln!("note: the MTP (nextn) layer has a quant type this engine does not support; loading without MTP");
            cfg.n_nextn = 0;
        }
        let emb_t = w.get("token_embd.weight")?;
        // Pre-touch mmap pages of token_embd.weight so decode step put_embed() never stalls on SSD page faults.
        let mut _touch = 0u8;
        for off in (0..emb_t.data.len()).step_by(4096) {
            _touch ^= unsafe { std::ptr::read_volatile(&emb_t.data[off]) };
        }
        std::hint::black_box(_touch);
        let ple = if cfg.n_embd_ple > 0 {
            let p = |k: &str| format!("{}.ple.{k}", cfg.arch);
            let arr = |k: &str| -> anyhow::Result<Vec<u64>> {
                g.get(&p(k))?.as_arr().with_context(|| p(k))?.iter().map(|v| v.as_u64().with_context(|| p(k))).collect()
            };
            let int = |k: &str| -> anyhow::Result<u64> { g.get(&p(k))?.as_u64().with_context(|| p(k)) };
            let heads_per_ngram = int("heads_per_ngram")? as usize;
            anyhow::ensure!(int("ngram_size")? == PLE_NGRAM as u64 && heads_per_ngram > 0
                && int("conv_kernel")? == PLE_KERNEL as u64, "unsupported PLE geometry");
            let ple = Ple { mult: arr("layer_multipliers")?, off: arr("head_offsets")?, vocab: arr("head_vocab_sizes")?, eos: int("eos_token_id")? as i64, heads_per_ngram };
            let heads = (PLE_NGRAM - 1) * heads_per_ngram;
            anyhow::ensure!(ple.mult.len() == PLE_NGRAM && ple.off.len() == heads && ple.vocab.len() == heads
                && ple.vocab.iter().all(|&v| v > 0) && heads * cfg.n_embd_ple == cfg.n_embd, "bad PLE metadata");
            w.get("per_layer_token_embd.weight")?;
            Some(ple)
        } else {
            w.get("output_norm.weight")?;
            None
        };
        let mut m = Model { cfg, w, loras: Vec::new(), draft_model, ple };
        m.apply_lora(loras)?;
        Ok(m)
    }

    pub fn apply_lora(&mut self, specs: &[(String, f32)]) -> anyhow::Result<()> {
        // Load everything before touching the current overlays, so a bad adapter leaves the model unchanged.
        let loras: Vec<Lora> = specs.iter().map(|(path, scale)| load_lora(path, *scale)).collect::<anyhow::Result<_>>()?;
        // Routed-expert adapters are not baked (`bake_lora` checks the others): they index activations and the
        // residual row by their own dimensions at inference, so these must be the base tensor's.
        let ne = self.cfg.n_expert.max(1);
        for (name, e) in loras.iter().flat_map(|l| &l.entries).filter(|(n, _)| n.ends_with("ffn_down_exps.weight")) {
            let Ok(base) = self.w.get(name) else { continue };
            anyhow::ensure!(e.in_dim == base.cols && e.out_dim * ne == base.rows && (e.n_expert == 1 || e.n_expert == ne),
                "lora: target {name} is {}x{} with {ne} experts, adapter is {}x{} with {}", base.rows / ne, base.cols, e.out_dim, e.in_dim, e.n_expert);
        }
        self.w.clear_baked();
        self.loras = loras;
        self.w.bake_lora(&self.loras);
        Ok(())
    }

    pub fn lora_delta(&self, name: &str, x: &[f32], y: &mut [f32], expert: Option<usize>) {
        for lora in &self.loras {
            if let Some(e) = lora.entries.get(name) {
                let ei = if e.n_expert == 1 { 0 } else { expert.unwrap_or(0) };
                if ei >= e.n_expert && e.n_expert > 1 { continue; }
                let a = &e.a[ei];
                let b = &e.b[ei];
                let rank = e.rank;
                let in_dim = e.in_dim;
                let out_dim = e.out_dim;
                for r in 0..rank {
                    let mut s = 0f32;
                    for i in 0..in_dim {
                        if i < x.len() { s += a[r * in_dim + i] * x[i]; }
                    }
                    let coef = if lora.alpha > 0.0 { lora.scale * lora.alpha / rank as f32 * s } else { lora.scale * s };
                    for o in 0..out_dim {
                        if o < y.len() { y[o] += coef * b[o * rank + r]; }
                    }
                }
            }
        }
    }

    fn t(&self, il: usize, suffix: &str) -> Tensor<'a> {
        self.w.layer(il, suffix).unwrap_or_else(|e| panic!("{e:#}"))
    }
    fn f(&self, il: usize, suffix: &str) -> Vec<f32> { self.t(il, suffix).to_f32() }

    pub fn embed(&self, token: u32) -> Vec<f32> {
        let mut x = vec![0f32; self.cfg.n_embd];
        let idx = (token as usize).min(self.cfg.n_vocab.saturating_sub(1));
        self.w.get("token_embd.weight").unwrap().row(idx, &mut x);
        x
    }

    /// Run one token; returns logits over the vocabulary.
    pub fn step(&self, st: &mut State, token: u32) -> Vec<f32> { self.step_h(st, token).0 }

    /// Run one token; returns logits and the last hidden state (before output_norm).
    pub fn step_h(&self, st: &mut State, token: u32) -> (Vec<f32>, Vec<f32>) {
        let rope_pos = st.rope_pos;
        self.step_embed_h(st, self.embed(token), rope_pos, token)
    }

    /// Feed a precomputed input embedding at explicit M-RoPE positions. `token` is used only by qwen4exp's PLE;
    /// pass the placeholder id for image tokens. The KV row still advances by one.
    pub fn step_embed(&self, st: &mut State, x: Vec<f32>, rope_pos: [i32; 4], token: u32) -> Vec<f32> {
        self.step_embed_h(st, x, rope_pos, token).0
    }

    pub fn step_embed_h(&self, st: &mut State, mut x: Vec<f32>, rope_pos: [i32; 4], token: u32) -> (Vec<f32>, Vec<f32>) {
        if self.cfg.hc_n > 0 { return self.step_embed_hc(st, x, rope_pos, token); }
        if self.cfg.mhc_streams > 0 { return self.step_embed_mhc(st, x, rope_pos); }
        let kv_pos = st.pos;
        for il in 0..self.cfg.n_layer { self.block(il, &mut x, &mut st.layers[il], kv_pos, rope_pos); }
        st.pos += 1;
        st.rope_pos = [rope_pos[0] + 1; 4];
        (self.head(&x, &self.w.get("output_norm.weight").unwrap().to_f32()), x)
    }

    /// SenseNova-Loop (`sensenova_loop`): shared-weight looped layers with manifold-constrained hyper-connections (mHC)
    /// and 20-step log-space Sinkhorn-Knopp doubly-stochastic stream mixing.
    fn step_embed_mhc(&self, st: &mut State, mut x: Vec<f32>, rope_pos: [i32; 4]) -> (Vec<f32>, Vec<f32>) {
        let c = &self.cfg;
        let kv_pos = st.pos;
        let s = c.mhc_streams;
        let mut r = Vec::new();
        for il in 0..c.n_layer {
            let mk = c.mhc_layer_kinds.get(il).copied().unwrap_or(0);
            if mk == 0 {
                self.block(il, &mut x, &mut st.layers[il], kv_pos, rope_pos);
                continue;
            }
            if mk & 1 != 0 {
                r = x.repeat(s);
            }
            let (mut branch, post, mix) = self.mhc_route(il, "mhc_attn", &r, s);
            let mut h = vec![0f32; c.n_embd];
            ops::rms_norm(&branch, &self.f(il, "attn_norm.weight"), c.eps, &mut h);
            let attn_out = match &mut st.layers[il] {
                LayerState::Recurrent { conv, s: dstate } => self.deltanet(il, &h, conv, dstate),
                LayerState::Attn { k, v, ik } => self.attention(il, &h, k, v, ik, kv_pos, rope_pos),
            };
            Self::mhc_merge(&mut r, &attn_out, &post, &mix, s);
            let (branch_f, post_f, mix_f) = self.mhc_route(il, "mhc_ffn", &r, s);
            branch = branch_f;
            ops::rms_norm(&branch, &self.f(il, "post_attention_norm.weight"), c.eps, &mut h);
            let ffn_out = if c.is_moe_layer(il) { self.moe(il, &h) } else { self.dense_ffn(il, &h) };
            Self::mhc_merge(&mut r, &ffn_out, &post_f, &mix_f, s);
            if mk & 2 != 0 {
                let inv_s = 1.0 / s as f32;
                for i in 0..c.n_embd {
                    let mut sum = 0.0f32;
                    for sc in 0..s { sum += r[sc * c.n_embd + i]; }
                    x[i] = sum * inv_s;
                }
            }
        }
        st.pos += 1;
        st.rope_pos = [rope_pos[0] + 1; 4];
        (self.head(&x, &self.w.get("output_norm.weight").unwrap().to_f32()), x)
    }

    fn mhc_route(&self, il: usize, prefix: &str, r: &[f32], streams: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = self.cfg.n_embd;
        let sq_mean = r.iter().map(|&v| v * v).sum::<f32>() / (r.len().max(1) as f32);
        let scale = 1.0 / (sq_mean + 1e-6).sqrt();
        let norm: Vec<f32> = r.iter().map(|&v| v * scale).collect();
        let w = self.t(il, &format!("{prefix}.weight"));
        let b = self.f(il, &format!("{prefix}.bias"));
        let mut logits = mv(&w, &norm);
        for (l, &bi) in logits.iter_mut().zip(&b) { *l += bi; }
        let pre: Vec<f32> = logits[..streams].iter().map(|&v| ops::sigmoid(v)).collect();
        let post: Vec<f32> = logits[streams..2 * streams].iter().map(|&v| 2.0 * ops::sigmoid(v)).collect();
        let mix = if logits.len() <= 2 * streams {
            let mut id = vec![0.0f32; streams * streams];
            for i in 0..streams { id[i * streams + i] = 1.0; }
            id
        } else {
            let mut m = logits[2 * streams..2 * streams + streams * streams].to_vec();
            for _ in 0..20 {
                for r0 in 0..streams {
                    let row = &mut m[r0 * streams..(r0 + 1) * streams];
                    let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let lse = mx + row.iter().map(|&v| (v - mx).exp()).sum::<f32>().ln();
                    for v in row.iter_mut() { *v -= lse; }
                }
                for c0 in 0..streams {
                    let mut mx = f32::NEG_INFINITY;
                    for r0 in 0..streams { mx = mx.max(m[r0 * streams + c0]); }
                    let mut se = 0.0f32;
                    for r0 in 0..streams { se += (m[r0 * streams + c0] - mx).exp(); }
                    let lse = mx + se.ln();
                    for r0 in 0..streams { m[r0 * streams + c0] -= lse; }
                }
            }
            m.into_iter().map(f32::exp).collect()
        };
        let mut branch = vec![0.0f32; n];
        for (sc, &p) in pre.iter().enumerate() {
            for (bi, &rv) in branch.iter_mut().zip(&r[sc * n..(sc + 1) * n]) {
                *bi += p * rv;
            }
        }
        (branch, post, mix)
    }

    fn mhc_merge(r: &mut [f32], branch: &[f32], post: &[f32], mix: &[f32], streams: usize) {
        let n = branch.len();
        let old = r.to_vec();
        for r0 in 0..streams {
            let p = post[r0];
            let dst = &mut r[r0 * n..(r0 + 1) * n];
            for i in 0..n {
                let mut acc = p * branch[i];
                for c0 in 0..streams {
                    acc += mix[r0 * streams + c0] * old[c0 * n + i];
                }
                dst[i] = acc;
            }
        }
    }

    /// qwen4exp: the residual is a stack of hc_n streams (hyper-connections). Every half-layer reads the stack
    /// down to one vector, runs its block and writes the output back into every stream with a per-stream weight.
    /// Layers holding `ple_key` first add the per-layer n-gram embedding (PLE) to the stack. Returns the logits
    /// and the stack before the head's read (the MTP draft input). Mirrors original Strata's `Engine::enqueue` (llama.cpp qwen4exp.cpp).
    fn step_embed_hc(&self, st: &mut State, x: Vec<f32>, rope_pos: [i32; 4], token: u32) -> (Vec<f32>, Vec<f32>) {
        let c = &self.cfg;
        let mut r = x.repeat(c.hc_n);
        let emb = self.ple_embed(&mut st.ple_prev, token);
        for il in 0..c.n_layer {
            if self.w.layer(il, "ple_key.weight").is_ok() { self.ple(il, &mut r, &emb, &mut st.ple_hist); }
            let (h, inj) = self.hc_read(&r, &format!("blk.{il}.hc_attn_"));
            let mix = match &mut st.layers[il] {
                LayerState::Recurrent { conv, s } => self.deltanet(il, &h, conv, s),
                LayerState::Attn { k, v, ik } => self.attention(il, &h, k, v, ik, st.pos, rope_pos),
            };
            Self::hc_write(&mut r, &mix, &inj);
            let (h, inj) = self.hc_read(&r, &format!("blk.{il}.hc_ffn_"));
            Self::hc_write(&mut r, &self.moe(il, &h), &inj);
        }
        st.pos += 1;
        st.rope_pos = [rope_pos[0] + 1; 4];
        let (h, _) = self.hc_read(&r, "output_hc_");
        (mv(&self.w.get("output.weight").unwrap(), &h), r)
    }

    /// Hyper-connection read (`build_hc_mix`) of the stack `r` with the tensors `{p}norm/down/up/inject.weight`:
    /// xn = per-stream RMSNorm(r), lo = silu(down·xn / hc), mixed = mean over streams of xn ⊙ sigmoid(up·lo),
    /// inject = inject·xn (empty when there is no inject tensor, as in the head).
    fn hc_read(&self, r: &[f32], p: &str) -> (Vec<f32>, Vec<f32>) {
        let (n, hc) = (self.cfg.n_embd, self.cfg.hc_n as f32);
        let t = |s: &str| self.w.get(&format!("{p}{s}.weight"));
        let mut xn = vec![0f32; r.len()];
        self.gnorm(r, &t("norm").unwrap().to_f32(), &mut xn);
        let lo: Vec<f32> = mv(&t("down").unwrap(), &xn).into_iter().map(|v| ops::silu(v / hc)).collect();
        let mut mixed = vec![0f32; n];
        for (i, g) in mv(&t("up").unwrap(), &lo).into_iter().enumerate() { mixed[i % n] += xn[i] * ops::sigmoid(g) / hc; }
        (mixed, t("inject").map(|w| mv(&w, &xn)).unwrap_or_default())
    }

    /// Hyper-connection write (`build_hc_combine`): stream c += out * 2 sigmoid(inject[c] / hc).
    fn hc_write(r: &mut [f32], out: &[f32], inj: &[f32]) {
        let hc = inj.len() as f32;
        for (rc, &j) in r.chunks_exact_mut(out.len()).zip(inj) {
            let w = 2.0 * ops::sigmoid(j / hc);
            for (a, b) in rc.iter_mut().zip(out) { *a += b * w; }
        }
    }

    /// Per-stream RMSNorm of an hc_n x n_embd stack with an hc_n x n_embd gamma.
    fn gnorm(&self, x: &[f32], w: &[f32], out: &mut [f32]) {
        let n = self.cfg.n_embd;
        for ((xc, wc), oc) in x.chunks_exact(n).zip(w.chunks_exact(n)).zip(out.chunks_exact_mut(n)) {
            ops::rms_norm(xc, wc, self.cfg.eps, oc);
        }
    }

    /// The PLE table rows of `token`: its 2- and 3-grams hashed (XOR of token x multiplier, mod each head's vocab,
    /// plus the head's offset), 8 heads per order. EOS or the sequence start cut the context: that position and
    /// every older one read as EOS. Advances `prev` (the last two tokens, -1 before the start).
    /// ponytail: image tokens are not mapped to `ple.image_token_id` (text only).
    pub fn ple_rows(&self, prev: &mut [i64; 2], token: u32) -> Vec<u64> {
        let Some(p) = &self.ple else { return Vec::new() };
        let mut ctx = [token as i64; PLE_NGRAM];
        let mut cut = false;
        for s in 1..PLE_NGRAM {
            let t = prev[s - 1];
            cut = cut || t < 0 || t == p.eos;
            ctx[s] = if cut { p.eos } else { t };
        }
        *prev = [token as i64, prev[0]];
        let mut rows = Vec::with_capacity(p.vocab.len());
        for n in 2..=PLE_NGRAM {
            let mut m = (ctx[0] as u64).wrapping_mul(p.mult[0]);
            for j in 1..n { m ^= (ctx[j] as u64).wrapping_mul(p.mult[j]); }
            for h in (n - 2) * p.heads_per_ngram..(n - 1) * p.heads_per_ngram { rows.push(m % p.vocab[h] + p.off[h]); }
        }
        rows
    }

    /// The PLE input of `token`: its 16 table rows, head after head (empty for models without PLE).
    pub fn ple_embed(&self, prev: &mut [i64; 2], token: u32) -> Vec<f32> {
        let rows = self.ple_rows(prev, token);
        if rows.is_empty() { return Vec::new() }
        let (table, d) = (self.w.get("per_layer_token_embd.weight").unwrap(), self.cfg.n_embd_ple);
        let mut emb = vec![0f32; self.cfg.n_embd];
        for (r, e) in rows.into_iter().zip(emb.chunks_exact_mut(d)) { table.row(r as usize, e); }
        emb
    }

    /// The PLE block (`build_ple`) on the stack `r`: a per-stream gate from key·query scales the value, and the
    /// normalized gated rows feed a causal conv dilated by the n-gram size. r += gated + silu(conv).
    fn ple(&self, il: usize, r: &mut [f32], emb: &[f32], hist: &mut [f32]) {
        let (n, d) = (self.cfg.n_embd, r.len());
        let mut key = vec![0f32; d];
        self.gnorm(&mv(&self.t(il, "ple_key.weight"), emb), &self.f(il, "ple_norm_key.weight"), &mut key);
        let mut query = vec![0f32; d];
        self.gnorm(r, &self.f(il, "ple_norm_query.weight"), &mut query);
        let value = mv(&self.t(il, "ple_value.weight"), emb);
        let mut gated = vec![0f32; d];
        for (c, gc) in gated.chunks_exact_mut(n).enumerate() {
            let s = key[c * n..(c + 1) * n].iter().zip(&query[c * n..(c + 1) * n]).map(|(a, b)| a * b).sum::<f32>() / (n as f32).sqrt();
            let sign = if s > 0.0 { 1.0 } else if s < 0.0 { -1.0 } else { 0.0 };
            let gate = ops::sigmoid(sign * s.abs().max(1e-6).sqrt());
            for (o, v) in gc.iter_mut().zip(&value) { *o = v * gate; }
        }
        let mut norm = vec![0f32; d];
        self.gnorm(&gated, &self.f(il, "ple_norm_conv.weight"), &mut norm);
        let kw = self.f(il, "ple_conv1d.weight"); // [channel][tap]
        for i in 0..d {
            let mut acc = kw[i * PLE_KERNEL + PLE_KERNEL - 1] * norm[i];
            for k in 0..PLE_KERNEL - 1 { acc += kw[i * PLE_KERNEL + k] * hist[k * PLE_NGRAM * d + i]; }
            r[i] += gated[i] + ops::silu(acc);
        }
        hist.copy_within(d.., 0);
        hist[(PLE_HIST - 1) * d..].copy_from_slice(&norm);
    }

    /// One decoder block (mixer and FFN, each with residual) on x at position `pos`.
    fn block(&self, il: usize, x: &mut [f32], ls: &mut LayerState, pos: usize, rope_pos: [i32; 4]) {
        let c = &self.cfg;
        let mut h = vec![0f32; c.n_embd];
        ops::rms_norm(x, &self.f(il, "attn_norm.weight"), c.eps, &mut h);
        let mix = match ls {
            LayerState::Recurrent { conv, s } => self.deltanet(il, &h, conv, s),
            LayerState::Attn { k, v, ik } => self.attention(il, &h, k, v, ik, pos, rope_pos),
        };
        for (a, b) in x.iter_mut().zip(&mix) { *a += b; }
        ops::rms_norm(x, &self.f(il, "post_attention_norm.weight"), c.eps, &mut h);
        let ffn = if c.is_moe_layer(il) { self.moe(il, &h) } else { self.dense_ffn(il, &h) };
        for (a, b) in x.iter_mut().zip(&ffn) { *a += b; }
    }

    /// Logits of the output head on rms_norm(x) with weights `norm`.
    fn head(&self, x: &[f32], norm: &[f32]) -> Vec<f32> {
        let mut h = vec![0f32; self.cfg.n_embd];
        ops::rms_norm(x, norm, self.cfg.eps, &mut h);
        let head = self.w.get("output.weight").or_else(|_| self.w.get("token_embd.weight")).unwrap();
        mv(&head, &h)
    }

    /// MTP draft (`nextn` block, stored as layer n_layer): logits for the token after `token`, from `hid`, the main
    /// model's last hidden state at the position that predicted `token`. Input is eh_proj [enorm(embed), hnorm(hid)].
    pub fn mtp_step(&self, ms: &mut Mtp, hid: &[f32], token: u32) -> (Vec<f32>, Vec<f32>) {
        let (c, il) = (&self.cfg, self.cfg.n_layer);
        let n = c.n_embd;
        if c.hc_n > 0 { return self.mtp_step_hc(ms, hid, token); }
        let mut eh = vec![0f32; 2 * n];
        ops::rms_norm(&self.embed(token), &self.f(il, "nextn.enorm.weight"), c.eps, &mut eh[..n]);
        ops::rms_norm(hid, &self.f(il, "nextn.hnorm.weight"), c.eps, &mut eh[n..]);
        let mut x = mv(&self.t(il, "nextn.eh_proj.weight"), &eh);
        let rope_pos = [ms.pos as i32; 4];
        self.block(il, &mut x, &mut ms.kv, ms.pos, rope_pos);
        ms.pos += 1;
        (self.head(&x, &self.f(il, "nextn.shared_head_norm.weight")), x)
    }

    /// qwen4exp draft step (llama.cpp `graph_mtp`): `hid` is the hc_n x n_embd stack. Every stream is projected
    /// on its own from [enorm(embed), hnorm(stream)], then the nextn block runs as a hyper-connection layer with
    /// plain attention (no PLE, no QSA). Returns the logits and the new stack, which is `hid` of the next draft.
    fn mtp_step_hc(&self, ms: &mut Mtp, hid: &[f32], token: u32) -> (Vec<f32>, Vec<f32>) {
        let (c, il) = (&self.cfg, self.cfg.n_layer);
        let n = c.n_embd;
        let mut e = vec![0f32; n];
        ops::rms_norm(&self.embed(token), &self.f(il, "nextn.enorm.weight"), c.eps, &mut e);
        let mut hn = vec![0f32; hid.len()];
        self.gnorm(hid, &self.f(il, "nextn.hnorm.weight"), &mut hn);
        let proj = self.t(il, "nextn.eh_proj.weight");
        let mut r: Vec<f32> = hn.chunks_exact(n).flat_map(|h| mv(&proj, &[&e, h].concat())).collect();
        let (h, inj) = self.hc_read(&r, &format!("blk.{il}.hc_attn_"));
        let LayerState::Attn { k, v, ik } = &mut ms.kv else { unreachable!("the nextn block is an attention layer") };
        Self::hc_write(&mut r, &self.attention(il, &h, k, v, ik, ms.pos, [ms.pos as i32; 4]), &inj);
        let (h, inj) = self.hc_read(&r, &format!("blk.{il}.hc_ffn_"));
        Self::hc_write(&mut r, &self.moe(il, &h), &inj);
        ms.pos += 1;
        let (h, _) = self.hc_read(&r, &format!("blk.{il}.nextn.hc_head_"));
        (mv(&self.w.get("output.weight").unwrap(), &h), r)
    }

    fn deltanet(&self, il: usize, h: &[f32], conv: &mut [f32], s: &mut [f32]) -> Vec<f32> {
        let c = &self.cfg;
        let (dk, nk, nv) = (c.ssm_d_state, c.ssm_n_group, c.ssm_dt_rank);
        let dv = c.ssm_dv();
        let channels = c.ssm_d_inner + 2 * nk * dk;
        let kw = c.ssm_conv;

        let qkv = mv(&self.t(il, "attn_qkv.weight"), h);
        let z = mv(&self.t(il, "attn_gate.weight"), h);
        let beta: Vec<f32> = mv(&self.t(il, "ssm_beta.weight"), h).into_iter().map(ops::sigmoid).collect();
        let (dt_bias, a) = (self.f(il, "ssm_dt.bias"), self.f(il, "ssm_a"));
        let g: Vec<f32> = mv(&self.t(il, "ssm_alpha.weight"), h).iter().enumerate()
            .map(|(i, &al)| ops::softplus(al + dt_bias[i]) * a[i]).collect();

        // Causal depthwise conv over [state..., current], then SiLU.
        let w = self.f(il, "ssm_conv1d.weight");
        let mut y = vec![0f32; channels];
        for ch in 0..channels {
            let mut acc = 0.0;
            for k in 0..kw - 1 { acc += conv[k * channels + ch] * w[ch * kw + k]; }
            acc += qkv[ch] * w[ch * kw + kw - 1];
            y[ch] = ops::silu(acc);
        }
        conv.copy_within(channels.., 0);
        conv[(kw - 2) * channels..].copy_from_slice(&qkv);

        let (q, rest) = y.split_at_mut(nk * dk);
        let (k, v) = rest.split_at_mut(nk * dk);
        for hd in q.chunks_exact_mut(dk).chain(k.chunks_exact_mut(dk)) { ops::l2_norm(hd, c.eps); }

        let norm_w = self.f(il, "ssm_norm.weight");
        let scale = 1.0 / (dk as f32).sqrt();
        let mut out = vec![0f32; nv * dv];
        let mut o = vec![0f32; dv];
        for vh in 0..nv {
            let kh = if c.k_interleave { vh / (nv / nk) } else { vh % nk };
            let (qh, khv, vv) = (&q[kh * dk..(kh + 1) * dk], &k[kh * dk..(kh + 1) * dk], &v[vh * dv..(vh + 1) * dv]);
            let sm = &mut s[vh * dv * dk..(vh + 1) * dv * dk]; // row j (value dim) x col i (key dim)
            let decay = g[vh].exp();
            for x in sm.iter_mut() { *x *= decay; }
            for j in 0..dv {
                let row = &mut sm[j * dk..(j + 1) * dk];
                let sk: f32 = row.iter().zip(khv).map(|(a, b)| a * b).sum();
                let d = beta[vh] * (vv[j] - sk);
                for (r, kk) in row.iter_mut().zip(khv) { *r += d * kk; }
                o[j] = row.iter().zip(qh.iter()).map(|(a, b)| a * b).sum::<f32>() * scale;
            }
            let oh = &mut out[vh * dv..(vh + 1) * dv];
            ops::rms_norm(&o, &norm_w, c.eps, oh);
            // qwen4exp gates with sigmoid(z), the others with silu(z)
            let gate = if c.arch == "qwen4exp" { ops::sigmoid } else { ops::silu };
            for (x, zz) in oh.iter_mut().zip(&z[vh * dv..(vh + 1) * dv]) { *x *= gate(*zz); }
        }
        mv(&self.t(il, "ssm_out.weight"), &out)
    }

    /// RoPE for one head: 1-D when the model has no M-RoPE sections, otherwise `[t, y, x, z]`.
    fn rope(&self, x: &mut [f32], kv_pos: usize, rope_pos: [i32; 4]) {
        match self.cfg.mrope_sections {
            Some(sections) => ops::rope_mrope(x, self.cfg.n_rot, rope_pos, sections, self.cfg.rope_base),
            None => ops::rope_neox(x, self.cfg.n_rot, kv_pos, self.cfg.rope_base),
        }
    }

    /// QSA (qwen4exp sparse attention): the cells row `pos` attends to, or None for all of them. Keys are pooled
    /// in blocks of QSA_POOL cells: mean of the raw indexer keys, RMS-normed and rotated to the block's first
    /// position. Each complete block scores sum over heads of relu(q_h . k) / sqrt(head_dim); the top
    /// idx_top_k / QSA_POOL blocks (ties to the lower block) and the incomplete tail block are kept. Dense while
    /// there are no more complete blocks than that. Mirrors llama.cpp qwen4exp.cpp `build_qsa_sel`.
    fn qsa(&self, il: usize, h: &[f32], ik: &mut Vec<f32>, pos: usize, rope_pos: [i32; 4]) -> Option<Vec<usize>> {
        let c = &self.cfg;
        let (nh, d) = (c.idx_n_head, c.idx_head_dim);
        if c.idx_top_k == 0 || il >= c.n_layer || self.w.layer(il, "indexer.k_proj.weight").is_err() { return None; }
        ik.extend_from_slice(&mv(&self.t(il, "indexer.k_proj.weight"), h));
        let (nb, top) = ((pos + 1) / QSA_POOL, c.idx_top_k / QSA_POOL);
        if nb <= top { return None; }
        let (qn, kn) = (self.f(il, "indexer.q_norm.weight"), self.f(il, "indexer.k_norm.weight"));
        let qr = mv(&self.t(il, "indexer.q_proj.weight"), h);
        let mut q = vec![0f32; nh * d];
        for i in 0..nh {
            ops::rms_norm(&qr[i * d..(i + 1) * d], &qn, c.eps, &mut q[i * d..(i + 1) * d]);
            self.rope(&mut q[i * d..(i + 1) * d], pos, rope_pos);
        }
        let mut score = vec![0f32; nb];
        let (mut mean, mut k) = (vec![0f32; d], vec![0f32; d]);
        for (b, s) in score.iter_mut().enumerate() {
            mean.fill(0.0);
            for t in b * QSA_POOL..(b + 1) * QSA_POOL {
                for (m, x) in mean.iter_mut().zip(&ik[t * d..(t + 1) * d]) { *m += x / QSA_POOL as f32; }
            }
            ops::rms_norm(&mean, &kn, c.eps, &mut k);
            self.rope(&mut k, b * QSA_POOL, [(b * QSA_POOL) as i32; 4]);
            *s = q.chunks(d).map(|qh| qh.iter().zip(&k).map(|(a, b)| a * b).sum::<f32>().max(0.0)).sum::<f32>()
                / (d as f32).sqrt();
        }
        let mut blocks = ops::top_k(&score, top);
        blocks.sort_unstable();
        Some(blocks.iter().flat_map(|b| b * QSA_POOL..(b + 1) * QSA_POOL).chain(nb * QSA_POOL..pos + 1).collect())
    }

    fn attention(&self, il: usize, h: &[f32], kc: &mut Vec<f32>, vc: &mut Vec<f32>, ik: &mut Vec<f32>, pos: usize, rope_pos: [i32; 4])
        -> Vec<f32> {
        let c = &self.cfg;
        let cells = self.qsa(il, h, ik, pos, rope_pos).unwrap_or_else(|| {
            if c.swa.get(il).copied().unwrap_or(false) {
                ((pos + 1).saturating_sub(c.n_swa)..=pos).collect()
            } else {
                (0..pos + 1).collect()
            }
        });
        let use_rope = c.rope_layer.get(il).copied().unwrap_or(true);
        let sinks = if c.attn_sinks { self.f(il, "attn_sinks.weight") } else { Vec::new() };
        let (hd, nh, nkv) = (c.head_dim, c.n_head, c.n_head_kv);
        let qs = if c.q_gate { 2 * hd } else { hd };  // per-head row of attn_q: query, then the gate if any
        let qg = mv(&self.t(il, "attn_q.weight"), h);
        let (qn, kn) = (self.f(il, "attn_q_norm.weight"), self.f(il, "attn_k_norm.weight"));
        let mut q = vec![0f32; nh * hd];
        for i in 0..nh {
            let qh = &mut q[i * hd..(i + 1) * hd];
            ops::rms_norm(&qg[i * qs..i * qs + hd], &qn, c.eps, qh);
            if use_rope { self.rope(qh, pos, rope_pos); }
        }
        let kraw = mv(&self.t(il, "attn_k.weight"), h);
        let v = mv(&self.t(il, "attn_v.weight"), h);
        let mut k = vec![0f32; nkv * hd];
        for i in 0..nkv {
            let kh = &mut k[i * hd..(i + 1) * hd];
            ops::rms_norm(&kraw[i * hd..(i + 1) * hd], &kn, c.eps, kh);
            if use_rope { self.rope(kh, pos, rope_pos); }
        }
        kc.extend_from_slice(&k);
        vc.extend_from_slice(&v);
        let scale = 1.0 / (hd as f32).sqrt();
        let mut out = vec![0f32; nh * hd];
        let mut sc = vec![0f32; cells.len()];
        for i in 0..nh {
            let g = i / (nh / nkv);
            let qh = &q[i * hd..(i + 1) * hd];
            for (&t, s) in cells.iter().zip(sc.iter_mut()) {
                let kt = &kc[(t * nkv + g) * hd..(t * nkv + g + 1) * hd];
                *s = qh.iter().zip(kt).map(|(a, b)| a * b).sum::<f32>() * scale;
            }
            ops::softmax_sink(&mut sc, sinks.get(i).copied());
            let oh = &mut out[i * hd..(i + 1) * hd];
            for (&t, p) in cells.iter().zip(&sc) {
                let vt = &vc[(t * nkv + g) * hd..(t * nkv + g + 1) * hd];
                for (o, vv) in oh.iter_mut().zip(vt) { *o += p * vv; }
            }
            if c.q_gate {
                let gate = &qg[i * qs + hd..(i + 1) * qs];
                for (o, gg) in oh.iter_mut().zip(gate) { *o *= ops::sigmoid(*gg); }
            }
        }
        mv(&self.t(il, "attn_output.weight"), &out)
    }

    fn ffn(&self, il: usize, gate: &Tensor, up: &Tensor, down: &Tensor, down_name: &str, expert: Option<usize>, h: &[f32]) -> Vec<f32> {
        let mut a = mv(gate, h);
        for (x, u) in a.iter_mut().zip(mv(up, h)) { *x = ops::silu(*x) * u; }
        let mut y = mv(down, &a);
        if down_name == "ffn_down_exps.weight" {
            self.lora_delta(&format!("blk.{il}.{down_name}"), &a, &mut y, expert);
        }
        y
    }

    fn dense_ffn(&self, il: usize, h: &[f32]) -> Vec<f32> {
        self.ffn(il, &self.t(il, "ffn_gate.weight"), &self.t(il, "ffn_up.weight"), &self.t(il, "ffn_down.weight"), "ffn_down.weight", None, h)
    }

    fn moe(&self, il: usize, h: &[f32]) -> Vec<f32> {
        let mut y = self.routed_experts(il, h);
        if self.w.layer(il, "ffn_up_shexp.weight").is_ok() {
            // Flint's shared expert is always on; Qwen's has a sigmoid gate.
            let s = match self.w.layer(il, "ffn_gate_inp_shexp.weight") {
                Ok(sg) => ops::sigmoid(mv(&sg, h)[0]),
                Err(_) => 1.0,
            };
            let o = self.ffn(il, &self.t(il, "ffn_gate_shexp.weight"), &self.t(il, "ffn_up_shexp.weight"), &self.t(il, "ffn_down_shexp.weight"), "ffn_down_shexp.weight", Some(0), h);
            for (a, b) in y.iter_mut().zip(o) { *a += s * b; }
        }
        y
    }

    /// Router scores, top-k and normalized weights: (expert, weight) pairs.
    pub fn route(&self, il: usize, h: &[f32]) -> Vec<(usize, f32)> {
        let c = &self.cfg;
        let mut p = mv(&self.t(il, "ffn_gate_inp.weight"), h);
        if c.router_sigmoid {
            for v in &mut p { *v = ops::sigmoid(*v); }
        } else {
            ops::softmax(&mut p);
        }
        let top = ops::top_k(&p, c.n_expert_used);
        let sum = top.iter().map(|&e| p[e]).sum::<f32>().max(6.103515625e-5);
        let mut scale = 1.0 / sum;
        if c.expert_weights_scale != 0.0 { scale *= c.expert_weights_scale; }
        top.into_iter().map(|e| (e, p[e] * scale)).collect()
    }

    /// Router, top-k and the weighted sum of routed experts (no shared expert).
    pub fn routed_experts(&self, il: usize, h: &[f32]) -> Vec<f32> {
        let (gate, up, down) = (self.t(il, "ffn_gate_exps.weight"), self.t(il, "ffn_up_exps.weight"), self.t(il, "ffn_down_exps.weight"));
        let n = self.cfg.n_expert;
        let mut y = vec![0f32; self.cfg.n_embd];
        for (e, w) in self.route(il, h) {
            let o = self.ffn(il, &gate.expert(e, n), &up.expert(e, n), &down.expert(e, n), "ffn_down_exps.weight", Some(e), h);
            for (a, b) in y.iter_mut().zip(o) { *a += w * b; }
        }
        y
    }
}


pub fn load_lora(path: &str, scale: f32) -> anyhow::Result<Lora> {
    use crate::quant::GgmlType;
    use crate::weights::Tensor;
    anyhow::ensure!(scale.is_finite(), "lora: {path}: scale must be finite");
    let g = Gguf::open(Path::new(path))?;
    let alpha = g.meta.get("adapter.lora.alpha").and_then(|v| v.as_f32()).unwrap_or(0.0);
    let mut bases = std::collections::BTreeSet::new();
    for name in g.tensors.keys() {
        if let Some(b) = name.strip_suffix(".lora_a") { bases.insert(b.to_string()); }
        if let Some(b) = name.strip_suffix(".lora_b") { bases.insert(b.to_string()); }
    }
    let mut entries = HashMap::new();
    for base in bases {
        let a_info = g.tensors.get(&format!("{base}.lora_a")).context("missing lora_a")?;
        let b_info = g.tensors.get(&format!("{base}.lora_b")).context("missing lora_b")?;
        let a_t = Tensor { ty: GgmlType::from_u32(a_info.ty)?, cols: a_info.dims[0] as usize, rows: a_info.dims[1..].iter().product::<u64>() as usize, data: g.bytes(a_info) };
        let b_t = Tensor { ty: GgmlType::from_u32(b_info.ty)?, cols: b_info.dims[0] as usize, rows: b_info.dims[1..].iter().product::<u64>() as usize, data: g.bytes(b_info) };
        let a_flat = a_t.to_f32();
        let b_flat = b_t.to_f32();
        let a_dims = &a_info.dims;
        let b_dims = &b_info.dims;
        anyhow::ensure!(a_dims.len() >= 2 && b_dims.len() >= 2, "lora: {base}: rank or dimensions missing");
        let in_dim = a_dims[0] as usize;
        let rank = a_dims[1] as usize;
        let n_expert = *a_dims.get(2).unwrap_or(&1) as usize;
        let out_dim = b_dims[1] as usize;
        anyhow::ensure!(rank > 0 && in_dim > 0 && out_dim > 0 && n_expert > 0, "lora: {base}: empty dimension");
        anyhow::ensure!(b_dims[0] as usize == rank, "lora: {base}: lora_b rank {} != lora_a rank {rank}", b_dims[0]);
        anyhow::ensure!(*b_dims.get(2).unwrap_or(&1) as usize == n_expert, "lora: {base}: expert count differs between lora_a and lora_b");
        anyhow::ensure!(a_flat.len() >= n_expert * rank * in_dim, "lora: {base}: truncated lora_a");
        anyhow::ensure!(b_flat.len() >= n_expert * out_dim * rank, "lora: {base}: truncated lora_b");
        let mut a_experts = vec![vec![0f32; rank * in_dim]; n_expert];
        let mut b_experts = vec![vec![0f32; out_dim * rank]; n_expert];
        for e in 0..n_expert {
            for r in 0..rank {
                for i in 0..in_dim {
                    a_experts[e][r * in_dim + i] = a_flat[(r + e * rank) * in_dim + i];
                }
                for o in 0..out_dim {
                    b_experts[e][o * rank + r] = b_flat[(o + e * out_dim) * rank + r];
                }
            }
        }
        entries.insert(base, LoraEntry { rank, in_dim, out_dim, n_expert, a: a_experts, b: b_experts });
    }
    Ok(Lora { scale, alpha, entries })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w_str(b: &mut Vec<u8>, s: &str) { b.extend((s.len() as u64).to_le_bytes()); b.extend(s.as_bytes()); }

    fn tiny_adapter(tag: &str) -> std::path::PathBuf {
        let (in_dim, rank, out_dim) = (4u64, 2u64, 3u64);
        let (mut b, mut off) = (Vec::new(), 0u64);
        b.extend(b"GGUF");
        b.extend(3u32.to_le_bytes());
        b.extend(2u64.to_le_bytes());
        b.extend(1u64.to_le_bytes());
        w_str(&mut b, "adapter.lora.alpha");
        b.extend(6u32.to_le_bytes());
        b.extend(2.0f32.to_le_bytes());
        for (name, dims) in [("blk.0.ffn_down_exps.weight.lora_a", [in_dim, rank, 1]), ("blk.0.ffn_down_exps.weight.lora_b", [rank, out_dim, 1])] {
            w_str(&mut b, name);
            b.extend(3u32.to_le_bytes());
            for d in dims { b.extend(d.to_le_bytes()); }
            b.extend(0u32.to_le_bytes());  // F32
            b.extend(off.to_le_bytes());
            off += (dims[0] * dims[1] * dims[2] * 4).next_multiple_of(32);
        }
        while b.len() % 32 != 0 { b.push(0); }
        let a: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let w: Vec<f32> = (0..6).map(|i| 10.0 + i as f32).collect();
        b.extend(a.iter().flat_map(|f| f.to_le_bytes()));
        b.extend(w.iter().flat_map(|f| f.to_le_bytes()));
        let p = std::env::temp_dir().join(format!("strata-lora-{}-{tag}.gguf", std::process::id()));
        std::fs::write(&p, &b).unwrap();
        p
    }

    #[test]
    fn load_lora_adapter() {
        let p = tiny_adapter("parse");
        let l = load_lora(p.to_str().unwrap(), 0.5).unwrap();
        std::fs::remove_file(&p).ok();
        assert_eq!((l.scale, l.alpha), (0.5, 2.0));
        let e = &l.entries["blk.0.ffn_down_exps.weight"];
        assert_eq!((e.rank, e.in_dim, e.out_dim, e.n_expert), (2, 4, 3, 1));
        assert_eq!(e.a[0], (0..8).map(|i| i as f32).collect::<Vec<_>>());
        assert_eq!(e.b[0], (0..6).map(|i| 10.0 + i as f32).collect::<Vec<_>>());
    }

    /// A routed-expert adapter whose dimensions are not the base tensor's is refused (needs `STRATA_MODEL`, a MoE model).
    #[test]
    fn routed_expert_lora_dimensions_are_checked() {
        let Some(m) = std::env::var_os("STRATA_MODEL").filter(|m| Path::new(m).exists()) else { return };
        let g = Gguf::open(Path::new(&m)).unwrap();
        if !g.tensors.contains_key("blk.0.ffn_down_exps.weight") { return; }
        let p = tiny_adapter("dims");
        let r = Model::load_with_lora(&g, &[(p.to_str().unwrap().to_string(), 1.0)]);
        std::fs::remove_file(&p).ok();
        let e = format!("{:#}", r.err().expect("a 3x4 adapter on the expert down matrix must be refused"));
        assert!(e.contains("blk.0.ffn_down_exps.weight") && e.contains("adapter is 3x4"), "{e}");
    }

    #[test]
    fn load_eagle3_aux_config() {
        let (mut hdr, mut off) = (Vec::new(), 0u64);
        let push_tensor = |h: &mut Vec<u8>, body: &mut Vec<u8>, off: &mut u64, name: &str, dims: &[u64], ty: u32, data: &[u8]| {
            w_str(h, name);
            h.extend((dims.len() as u32).to_le_bytes());
            for &d in dims { h.extend(d.to_le_bytes()); }
            h.extend(ty.to_le_bytes());
            h.extend((*off).to_le_bytes());
            body.extend_from_slice(data);
            let pad = (32 - data.len() % 32) % 32;
            body.extend(std::iter::repeat_n(0u8, pad));
            *off += (data.len() + pad) as u64;
        };
        let mut body = Vec::new();
        let mut kvs = Vec::new();
        let add_u32 = |b: &mut Vec<u8>, k: &str, v: u32| { w_str(b, k); b.extend(4u32.to_le_bytes()); b.extend(v.to_le_bytes()); };
        let add_f32 = |b: &mut Vec<u8>, k: &str, v: f32| { w_str(b, k); b.extend(6u32.to_le_bytes()); b.extend(v.to_le_bytes()); };
        let add_str = |b: &mut Vec<u8>, k: &str, v: &str| { w_str(b, k); b.extend(8u32.to_le_bytes()); w_str(b, v); };
        // Build tiny target GGUF (`qwen3moe` with 1 layer, embd=4, n_vocab=8, nh=2, nkv=1, hd=2)
        add_str(&mut kvs, "general.architecture", "qwen3moe");
        add_u32(&mut kvs, "qwen3moe.block_count", 1);
        add_u32(&mut kvs, "qwen3moe.embedding_length", 4);
        add_u32(&mut kvs, "qwen3moe.attention.head_count", 2);
        add_u32(&mut kvs, "qwen3moe.attention.head_count_kv", 1);
        add_f32(&mut kvs, "qwen3moe.attention.layer_norm_rms_epsilon", 1e-6);
        let z4 = vec![0u8; 4 * 4];
        let z4x8 = vec![0u8; 32 * 4];
        push_tensor(&mut hdr, &mut body, &mut off, "token_embd.weight", &[4, 8], 0, &z4x8);
        push_tensor(&mut hdr, &mut body, &mut off, "output_norm.weight", &[4], 0, &z4);
        push_tensor(&mut hdr, &mut body, &mut off, "blk.0.attn_norm.weight", &[4], 0, &z4);
        let mut tgt = Vec::new();
        tgt.extend(b"GGUF");
        tgt.extend(3u32.to_le_bytes());
        tgt.extend(3u64.to_le_bytes());
        tgt.extend(6u64.to_le_bytes());
        tgt.extend(kvs);
        tgt.extend(hdr);
        while tgt.len() % 32 != 0 { tgt.push(0); }
        tgt.extend(body);

        // Build tiny EAGLE3 aux GGUF (`draft_vocab=3`, draft `heads=2x4`, `kv=1`, `ff=6`, `d2t` relative `[0, 0, 5]` -> abs `[0, 1, 7]`)
        let (mut ahdr, mut abody, mut aoff) = (Vec::new(), Vec::new(), 0u64);
        let mut akvs = Vec::new();
        add_str(&mut akvs, "general.architecture", "eagle3");
        add_u32(&mut akvs, "eagle3.attention.head_count", 2);
        add_f32(&mut akvs, "eagle3.rope.freq_base", 1000000.0);
        w_str(&mut akvs, "eagle3.target_layers");
        akvs.extend(9u32.to_le_bytes());
        akvs.extend(5u32.to_le_bytes());
        akvs.extend(3u64.to_le_bytes());
        for v in [0i32, 0, 0] { akvs.extend(v.to_le_bytes()); }
        let fc_d = vec![0u8; 12 * 4 * 4];
        let norm_d = vec![0u8; 4 * 4];
        let qk_d = vec![0u8; 8 * 8 * 4];
        let kv_d = vec![0u8; 8 * 4 * 4];
        let o_d = vec![0u8; 8 * 4 * 4];
        let ff_d = vec![0u8; 4 * 6 * 4];
        let out_d = vec![0u8; 4 * 3 * 4];
        let d2t_d: Vec<u8> = [0i64, 0, 5].into_iter().flat_map(|v| v.to_le_bytes()).collect();
        push_tensor(&mut ahdr, &mut abody, &mut aoff, "fc.weight", &[12, 4], 0, &fc_d);
        push_tensor(&mut ahdr, &mut abody, &mut aoff, "output.weight", &[4, 3], 0, &out_d);
        push_tensor(&mut ahdr, &mut abody, &mut aoff, "output_norm.weight", &[4], 0, &norm_d);
        push_tensor(&mut ahdr, &mut abody, &mut aoff, "blk.0.attn_norm.weight", &[4], 0, &norm_d);
        push_tensor(&mut ahdr, &mut abody, &mut aoff, "blk.0.attn_norm_2.weight", &[4], 0, &norm_d);
        push_tensor(&mut ahdr, &mut abody, &mut aoff, "blk.0.ffn_norm.weight", &[4], 0, &norm_d);
        push_tensor(&mut ahdr, &mut abody, &mut aoff, "blk.0.attn_q.weight", &[8, 8], 0, &qk_d);
        push_tensor(&mut ahdr, &mut abody, &mut aoff, "blk.0.attn_k.weight", &[8, 4], 0, &kv_d);
        push_tensor(&mut ahdr, &mut abody, &mut aoff, "blk.0.attn_v.weight", &[8, 4], 0, &kv_d);
        push_tensor(&mut ahdr, &mut abody, &mut aoff, "blk.0.attn_output.weight", &[8, 4], 0, &o_d);
        push_tensor(&mut ahdr, &mut abody, &mut aoff, "blk.0.ffn_gate.weight", &[4, 6], 0, &ff_d);
        push_tensor(&mut ahdr, &mut abody, &mut aoff, "blk.0.ffn_up.weight", &[4, 6], 0, &ff_d);
        push_tensor(&mut ahdr, &mut abody, &mut aoff, "blk.0.ffn_down.weight", &[6, 4], 0, &ff_d);
        push_tensor(&mut ahdr, &mut abody, &mut aoff, "d2t", &[3], 27, &d2t_d);
        let mut aux = Vec::new();
        aux.extend(b"GGUF");
        aux.extend(3u32.to_le_bytes());
        aux.extend(14u64.to_le_bytes());
        aux.extend(4u64.to_le_bytes());
        aux.extend(akvs);
        aux.extend(ahdr);
        while aux.len() % 32 != 0 { aux.push(0); }
        aux.extend(abody);

        let pt = std::env::temp_dir().join(format!("strata-e3-tgt-{}.gguf", std::process::id()));
        let pa = std::env::temp_dir().join(format!("strata-e3-aux-{}.gguf", std::process::id()));
        std::fs::write(&pt, &tgt).unwrap();
        std::fs::write(&pa, &aux).unwrap();
        let gt = Gguf::open(&pt).unwrap();
        let ga = Gguf::open(&pa).unwrap();
        let m = Model::load_with_aux(&gt, Some(&ga), &[]).unwrap();
        std::fs::remove_file(&pt).ok();
        std::fs::remove_file(&pa).ok();
        let ec = m.cfg.eagle3.as_ref().expect("eagle3 config loaded");
        assert_eq!(m.cfg.n_nextn, 1);
        assert_eq!((ec.n_head, ec.head_dim, ec.n_head_kv, ec.n_ff, ec.draft_vocab), (2, 4, 1, 6, 3));
        assert_eq!(ec.d2t, vec![0, 1, 7]);
    }

    #[test]
    fn eagle3_safetensors_auto_converts_and_loads() {
        let dir = std::env::temp_dir().join(format!("strata-e3-st-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // Build tiny target GGUF (`qwen3moe` 1 layer, embd=4, n_vocab=8, nh=2, nkv=1, hd=2)
        let (mut hdr, mut body, mut off) = (Vec::new(), Vec::new(), 0u64);
        let push_tensor = |h: &mut Vec<u8>, b: &mut Vec<u8>, o: &mut u64, name: &str, dims: &[u64], ty: u32, data: &[u8]| {
            w_str(h, name);
            h.extend((dims.len() as u32).to_le_bytes());
            for &d in dims { h.extend(d.to_le_bytes()); }
            h.extend(ty.to_le_bytes());
            h.extend((*o).to_le_bytes());
            b.extend_from_slice(data);
            let pad = (32 - data.len() % 32) % 32;
            b.extend(std::iter::repeat_n(0u8, pad));
            *o += (data.len() + pad) as u64;
        };
        let mut kvs = Vec::new();
        let add_u32 = |b: &mut Vec<u8>, k: &str, v: u32| { w_str(b, k); b.extend(4u32.to_le_bytes()); b.extend(v.to_le_bytes()); };
        let add_f32 = |b: &mut Vec<u8>, k: &str, v: f32| { w_str(b, k); b.extend(6u32.to_le_bytes()); b.extend(v.to_le_bytes()); };
        let add_str = |b: &mut Vec<u8>, k: &str, v: &str| { w_str(b, k); b.extend(8u32.to_le_bytes()); w_str(b, v); };
        add_str(&mut kvs, "general.architecture", "qwen3moe");
        add_u32(&mut kvs, "qwen3moe.block_count", 1);
        add_u32(&mut kvs, "qwen3moe.embedding_length", 4);
        add_u32(&mut kvs, "qwen3moe.attention.head_count", 2);
        add_u32(&mut kvs, "qwen3moe.attention.head_count_kv", 1);
        add_f32(&mut kvs, "qwen3moe.attention.layer_norm_rms_epsilon", 1e-6);
        let z4 = vec![0u8; 4 * 4];
        let z4x8 = vec![0u8; 32 * 4];
        push_tensor(&mut hdr, &mut body, &mut off, "token_embd.weight", &[4, 8], 0, &z4x8);
        push_tensor(&mut hdr, &mut body, &mut off, "output_norm.weight", &[4], 0, &z4);
        push_tensor(&mut hdr, &mut body, &mut off, "blk.0.attn_norm.weight", &[4], 0, &z4);
        let mut tgt = Vec::new();
        tgt.extend(b"GGUF");
        tgt.extend(3u32.to_le_bytes());
        tgt.extend(3u64.to_le_bytes());
        tgt.extend(6u64.to_le_bytes());
        tgt.extend(kvs);
        tgt.extend(hdr);
        while tgt.len() % 32 != 0 { tgt.push(0); }
        tgt.extend(body);
        let pt = dir.join("target.gguf");
        std::fs::write(&pt, &tgt).unwrap();

        // Build synthetic HuggingFace EAGLE3 `config.json` + `model.safetensors`
        let cfg_path = dir.join("config.json");
        std::fs::write(
            &cfg_path,
            r#"{
                "hidden_size": 4,
                "intermediate_size": 6,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "norm_before_residual": false,
                "draft_vocab_size": 3,
                "eagle_aux_hidden_state_layer_ids": [0, 0, 0]
            }"#,
        ).unwrap();

        let d2t_bytes: Vec<u8> = [0i64, 0, 5].into_iter().flat_map(|v| v.to_le_bytes()).collect();
        let st_specs: Vec<(&str, &str, Vec<u64>, Vec<u8>)> = vec![
            ("fc.weight", "BF16", vec![4, 12], vec![0u8; 4 * 12 * 2]),
            ("midlayer.input_layernorm.weight", "BF16", vec![4], vec![0u8; 4 * 2]),
            ("midlayer.hidden_norm.weight", "F32", vec![4], vec![0u8; 4 * 4]),
            ("midlayer.self_attn.q_proj.weight", "BF16", vec![8, 8], vec![0u8; 8 * 8 * 2]),
            ("midlayer.self_attn.k_proj.weight", "BF16", vec![4, 8], vec![0u8; 4 * 8 * 2]),
            ("midlayer.self_attn.v_proj.weight", "BF16", vec![4, 8], vec![0u8; 4 * 8 * 2]),
            ("midlayer.self_attn.o_proj.weight", "BF16", vec![4, 8], vec![0u8; 4 * 8 * 2]),
            ("midlayer.post_attention_layernorm.weight", "F32", vec![4], vec![0u8; 4 * 4]),
            ("midlayer.mlp.gate_proj.weight", "BF16", vec![6, 4], vec![0u8; 6 * 4 * 2]),
            ("midlayer.mlp.up_proj.weight", "BF16", vec![6, 4], vec![0u8; 6 * 4 * 2]),
            ("midlayer.mlp.down_proj.weight", "BF16", vec![4, 6], vec![0u8; 4 * 6 * 2]),
            ("norm.weight", "F32", vec![4], vec![0u8; 4 * 4]),
            ("lm_head.weight", "BF16", vec![3, 4], vec![0u8; 3 * 4 * 2]),
            ("d2t", "I64", vec![3], d2t_bytes),
        ];
        let mut st_hdr = serde_json::Map::new();
        let mut st_body = Vec::new();
        for (name, dt, shape, data) in st_specs {
            let start = st_body.len();
            st_body.extend_from_slice(&data);
            let end = st_body.len();
            st_hdr.insert(
                name.to_string(),
                serde_json::json!({"dtype": dt, "shape": shape, "data_offsets": [start, end]}),
            );
        }
        let hdr_bytes = serde_json::to_vec(&st_hdr).unwrap();
        let mut st_file = Vec::with_capacity(8 + hdr_bytes.len() + st_body.len());
        st_file.extend_from_slice(&(hdr_bytes.len() as u64).to_le_bytes());
        st_file.extend_from_slice(&hdr_bytes);
        st_file.extend_from_slice(&st_body);
        let st_path = dir.join("model.safetensors");
        std::fs::write(&st_path, &st_file).unwrap();

        let aux_gguf = dir.join("model.gguf");
        crate::gguf::convert_safetensors_eagle3_to_gguf(&st_path, &cfg_path, &aux_gguf, 48).unwrap();

        let gt = Gguf::open(&pt).unwrap();
        let ga = Gguf::open(&aux_gguf).unwrap();
        let m = Model::load_with_aux(&gt, Some(&ga), &[]).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        let ec = m.cfg.eagle3.as_ref().expect("eagle3 config loaded from converted safetensors");
        assert_eq!(m.cfg.n_nextn, 1);
        assert_eq!((ec.n_head, ec.head_dim, ec.n_head_kv, ec.n_ff, ec.draft_vocab), (2, 4, 1, 6, 3));
        assert_eq!(ec.d2t, vec![0, 1, 7]);
    }

    #[test]
    fn dflash_safetensors_auto_converts_and_loads() {
        let dir = std::env::temp_dir().join(format!("strata_dflash_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mut kvs = Vec::new();
        let w_s = |b: &mut Vec<u8>, s: &str| { b.extend((s.len() as u64).to_le_bytes()); b.extend(s.as_bytes()); };
        let w_kv = |b: &mut Vec<u8>, k: &str, t: u32, p: &[u8]| { w_s(b, k); b.extend(t.to_le_bytes()); b.extend(p); };
        let s = |b: &mut Vec<u8>, k: &str, v: &str| { let mut p = Vec::new(); w_s(&mut p, v); w_kv(b, k, 8, &p); };
        s(&mut kvs, "general.architecture", "qwen3");
        w_kv(&mut kvs, "qwen3.block_count", 4, &1u32.to_le_bytes());
        w_kv(&mut kvs, "qwen3.embedding_length", 4, &32u32.to_le_bytes());
        w_kv(&mut kvs, "qwen3.attention.head_count", 4, &2u32.to_le_bytes());
        w_kv(&mut kvs, "qwen3.attention.head_count_kv", 4, &1u32.to_le_bytes());
        w_kv(&mut kvs, "qwen3.attention.key_length", 4, &16u32.to_le_bytes());
        w_kv(&mut kvs, "qwen3.attention.layer_norm_rms_epsilon", 6, &1e-6f32.to_le_bytes());
        let mut hdr = Vec::new();
        let mut body = Vec::new();
        for (name, dims, nb) in [
            ("token_embd.weight", vec![32u64, 4], 32 * 4 * 4),
            ("output_norm.weight", vec![32u64], 32 * 4),
            ("output.weight", vec![32u64, 4], 32 * 4 * 4),
        ] {
            w_s(&mut hdr, name);
            hdr.extend((dims.len() as u32).to_le_bytes());
            for d in dims { hdr.extend(d.to_le_bytes()); }
            hdr.extend(0u32.to_le_bytes());
            hdr.extend((body.len() as u64).to_le_bytes());
            body.resize(body.len() + nb, 0);
            while body.len() % 32 != 0 { body.push(0); }
        }
        let mut tgt = Vec::new();
        tgt.extend(b"GGUF");
        tgt.extend(3u32.to_le_bytes());
        tgt.extend(3u64.to_le_bytes());
        tgt.extend(7u64.to_le_bytes());
        tgt.extend(kvs);
        tgt.extend(hdr);
        while tgt.len() % 32 != 0 { tgt.push(0); }
        tgt.extend(body);
        let pt = dir.join("target.gguf");
        std::fs::write(&pt, &tgt).unwrap();

        let cfg_path = dir.join("config.json");
        std::fs::write(
            &cfg_path,
            r#"{
                "architectures": ["DFlashDraftModel"],
                "dflash_config": { "block_size": 16, "mask_token_id": 2, "target_layer_ids": [0] },
                "hidden_size": 32,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "layer_types": ["sliding_attention", "full_attention"],
                "sliding_window": 4096,
                "rms_norm_eps": 1e-6,
                "rope_parameters": { "rope_theta": 10000000.0 }
            }"#,
        ).unwrap();

        let mut st_specs: Vec<(String, &str, Vec<u64>, Vec<u8>)> = vec![
            ("fc.weight".into(), "BF16", vec![32, 32], vec![0u8; 32 * 32 * 2]),
            ("hidden_norm.weight".into(), "BF16", vec![32], vec![0u8; 32 * 2]),
            ("norm.weight".into(), "BF16", vec![32], vec![0u8; 32 * 2]),
        ];
        for il in 0..2 {
            st_specs.push((format!("layers.{il}.input_layernorm.weight"), "BF16", vec![32], vec![0u8; 32 * 2]));
            st_specs.push((format!("layers.{il}.post_attention_layernorm.weight"), "BF16", vec![32], vec![0u8; 32 * 2]));
            st_specs.push((format!("layers.{il}.self_attn.q_norm.weight"), "BF16", vec![16], vec![0u8; 16 * 2]));
            st_specs.push((format!("layers.{il}.self_attn.k_norm.weight"), "BF16", vec![16], vec![0u8; 16 * 2]));
            st_specs.push((format!("layers.{il}.self_attn.q_proj.weight"), "BF16", vec![32, 32], vec![0u8; 32 * 32 * 2]));
            st_specs.push((format!("layers.{il}.self_attn.k_proj.weight"), "BF16", vec![16, 32], vec![0u8; 16 * 32 * 2]));
            st_specs.push((format!("layers.{il}.self_attn.v_proj.weight"), "BF16", vec![16, 32], vec![0u8; 16 * 32 * 2]));
            st_specs.push((format!("layers.{il}.self_attn.o_proj.weight"), "BF16", vec![32, 32], vec![0u8; 32 * 32 * 2]));
            st_specs.push((format!("layers.{il}.mlp.gate_proj.weight"), "BF16", vec![32, 32], vec![0u8; 32 * 32 * 2]));
            st_specs.push((format!("layers.{il}.mlp.up_proj.weight"), "BF16", vec![32, 32], vec![0u8; 32 * 32 * 2]));
            st_specs.push((format!("layers.{il}.mlp.down_proj.weight"), "BF16", vec![32, 32], vec![0u8; 32 * 32 * 2]));
        }
        let mut st_hdr = serde_json::Map::new();
        let mut st_body = Vec::new();
        for (name, dt, shape, data) in st_specs {
            let start = st_body.len();
            st_body.extend_from_slice(&data);
            st_hdr.insert(name, serde_json::json!({"dtype": dt, "shape": shape, "data_offsets": [start, st_body.len()]}));
        }
        let hdr_bytes = serde_json::to_vec(&st_hdr).unwrap();
        let mut st_file = Vec::with_capacity(8 + hdr_bytes.len() + st_body.len());
        st_file.extend_from_slice(&(hdr_bytes.len() as u64).to_le_bytes());
        st_file.extend_from_slice(&hdr_bytes);
        st_file.extend_from_slice(&st_body);
        let st_path = dir.join("model.safetensors");
        std::fs::write(&st_path, &st_file).unwrap();

        let aux_gguf = dir.join("model-dflash-q8_0.gguf");
        crate::gguf::convert_safetensors_dflash_to_gguf(&st_path, &cfg_path, &aux_gguf, 1).unwrap();
        let gt = Gguf::open(&pt).unwrap();
        let ga = Gguf::open(&aux_gguf).unwrap();
        let m = Model::load_with_aux(&gt, Some(&ga), &[]).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        let df = m.cfg.dflash.as_ref().expect("dflash config loaded");
        assert_eq!(m.cfg.n_nextn, 0);
        assert_eq!((df.n_layer, df.block_size, df.mask_token_id, df.n_embd, df.n_ff, df.n_head, df.n_head_kv, df.head_dim, df.sliding_window), (2, 16, 2, 32, 32, 2, 1, 16, 4096));
        assert_eq!(df.target_layer_ids, vec![0]);
        assert_eq!(df.layer_swa, vec![true, false]);
        // Metadata that disagrees with the tensors is refused before any GPU buffer is sized from it.
        crate::gpu_model::dflash_check(&m, df).unwrap();
        for bad in [
            crate::config::DFlashConfig { n_ff: 16, ..df.clone() },
            crate::config::DFlashConfig { head_dim: 8, n_rot: 8, ..df.clone() },
            crate::config::DFlashConfig { n_embd: 16, ..df.clone() },
            crate::config::DFlashConfig { conv_kernel_size: 3, ..df.clone() },
            crate::config::DFlashConfig { mask_token_id: u32::MAX, ..df.clone() },
        ] {
            assert!(crate::gpu_model::dflash_check(&m, &bad).is_err());
        }
    }

    #[test]
    fn sensenova_loop_mhc_cpu_matches_gpu() {
        let dir = std::env::temp_dir().join(format!("strata_snloop_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mut kvs = Vec::new();
        let w_s = |b: &mut Vec<u8>, s: &str| { b.extend((s.len() as u64).to_le_bytes()); b.extend(s.as_bytes()); };
        let w_kv = |b: &mut Vec<u8>, k: &str, t: u32, p: &[u8]| { w_s(b, k); b.extend(t.to_le_bytes()); b.extend(p); };
        let s = |b: &mut Vec<u8>, k: &str, v: &str| { let mut p = Vec::new(); w_s(&mut p, v); w_kv(b, k, 8, &p); };
        s(&mut kvs, "general.architecture", "sensenova_loop");
        w_kv(&mut kvs, "sensenova_loop.block_count", 4, &4u32.to_le_bytes());
        w_kv(&mut kvs, "sensenova_loop.embedding_length", 4, &64u32.to_le_bytes());
        w_kv(&mut kvs, "sensenova_loop.feed_forward_length", 4, &64u32.to_le_bytes());
        w_kv(&mut kvs, "sensenova_loop.attention.head_count", 4, &1u32.to_le_bytes());
        w_kv(&mut kvs, "sensenova_loop.attention.head_count_kv", 4, &1u32.to_le_bytes());
        w_kv(&mut kvs, "sensenova_loop.attention.key_length", 4, &64u32.to_le_bytes());
        w_kv(&mut kvs, "sensenova_loop.attention.layer_norm_rms_epsilon", 6, &1e-6f32.to_le_bytes());
        w_kv(&mut kvs, "sensenova_loop.mhc.streams", 4, &2u32.to_le_bytes());
        // recurrent_layers = [false, false, false, false]
        w_s(&mut kvs, "sensenova_loop.attention.recurrent_layers");
        kvs.extend(9u32.to_le_bytes());
        kvs.extend(7u32.to_le_bytes());
        kvs.extend(4u64.to_le_bytes());
        kvs.extend([0u8, 0, 0, 0]);
        // mhc.layer_kinds = [1 (enter), 2 (leave), 1 (enter visit 2), 2 (leave visit 2)]
        w_s(&mut kvs, "sensenova_loop.mhc.layer_kinds");
        kvs.extend(9u32.to_le_bytes());
        kvs.extend(4u32.to_le_bytes());
        kvs.extend(4u64.to_le_bytes());
        for v in [1u32, 2, 1, 2] { kvs.extend(v.to_le_bytes()); }

        let mut hdr = Vec::new();
        let mut body = Vec::new();
        let mut seed = 1usize;
        let mut add_t = |name: &str, dims: &[u64], scale: f32| {
            w_s(&mut hdr, name);
            hdr.extend((dims.len() as u32).to_le_bytes());
            for &d in dims { hdr.extend(d.to_le_bytes()); }
            hdr.extend(0u32.to_le_bytes()); // F32
            hdr.extend((body.len() as u64).to_le_bytes());
            let n: usize = dims.iter().product::<u64>() as usize;
            for i in 0..n {
                let v = if scale == 0.0 {
                    1.0f32
                } else {
                    (((seed * 31 + i * 17 + 7) % 97) as f32 - 48.0) * scale
                };
                body.extend_from_slice(&v.to_le_bytes());
            }
            seed += 1;
            while body.len() % 32 != 0 { body.push(0); }
        };
        add_t("token_embd.weight", &[64, 8], 0.02);
        add_t("output_norm.weight", &[64], 0.0);
        add_t("output.weight", &[64, 8], 0.02);
        // Only physical layers 0 and 1 are stored in the GGUF; virtual layers 2 and 3 loop over 0 and 1!
        for il in 0..2 {
            add_t(&format!("blk.{il}.attn_norm.weight"), &[64], 0.0);
            add_t(&format!("blk.{il}.post_attention_norm.weight"), &[64], 0.0);
            add_t(&format!("blk.{il}.attn_q_norm.weight"), &[64], 0.0);
            add_t(&format!("blk.{il}.attn_k_norm.weight"), &[64], 0.0);
            add_t(&format!("blk.{il}.attn_q.weight"), &[64, 128], 0.01); // q_gate = true -> 2 * n_head * head_dim = 128
            add_t(&format!("blk.{il}.attn_k.weight"), &[64, 64], 0.01);
            add_t(&format!("blk.{il}.attn_v.weight"), &[64, 64], 0.01);
            add_t(&format!("blk.{il}.attn_output.weight"), &[64, 64], 0.01);
            add_t(&format!("blk.{il}.ffn_gate.weight"), &[64, 64], 0.01);
            add_t(&format!("blk.{il}.ffn_up.weight"), &[64, 64], 0.01);
            add_t(&format!("blk.{il}.ffn_down.weight"), &[64, 64], 0.01);
            add_t(&format!("blk.{il}.mhc_attn.weight"), &[128, 8], 0.01);
            add_t(&format!("blk.{il}.mhc_attn.bias"), &[8], 0.01);
            add_t(&format!("blk.{il}.mhc_ffn.weight"), &[128, 8], 0.01);
            add_t(&format!("blk.{il}.mhc_ffn.bias"), &[8], 0.01);
        }
        let n_tensors = 3 + 2 * 15;
        let mut gg = Vec::new();
        gg.extend(b"GGUF");
        gg.extend(3u32.to_le_bytes());
        gg.extend((n_tensors as u64).to_le_bytes());
        gg.extend(11u64.to_le_bytes());
        gg.extend(kvs);
        gg.extend(hdr);
        while gg.len() % 32 != 0 { gg.push(0); }
        gg.extend(body);
        let p = dir.join("snloop.gguf");
        std::fs::write(&p, &gg).unwrap();

        let g = Gguf::open(&p).unwrap();
        let m = Model::load(&g).unwrap();
        assert_eq!(m.cfg.phys_layer, vec![0, 1, 0, 1]);
        assert_eq!(m.cfg.mhc_streams, 2);

        let mut st = State::new(&m.cfg);
        let mut cpu_logits = Vec::new();
        for &tok in &[1u32, 3, 2, 4] {
            cpu_logits = m.step(&mut st, tok);
        }

        if let Ok(bk) = crate::backend::open(None, 0) {
            let mut gm = crate::gpu_model::GpuModel::load(&m, bk, 64, 0, 0, 0).unwrap();
            let mut gpu_logits = Vec::new();
            for &tok in &[1u32, 3, 2, 4] {
                gpu_logits = gm.step(tok).unwrap();
            }
            let max_err = cpu_logits.iter().zip(&gpu_logits).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            eprintln!("sensenova_loop decode vs cpu max_err = {max_err}");
            assert!(max_err < 1e-2, "sensenova_loop CPU vs GPU decode max_err = {max_err} (cpu={cpu_logits:?}, gpu={gpu_logits:?})");

            if let Ok(bk2) = crate::backend::open(None, 0) {
                let mut gm_pf = crate::gpu_model::GpuModel::load(&m, bk2, 64, 64, 0, 0).unwrap();
                gm_pf.short = 2;
                let pf_logits = gm_pf.prefill(&[1u32, 3, 2, 4]).unwrap();
                let max_err_pf = cpu_logits.iter().zip(&pf_logits).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                let max_err_gpu = gpu_logits.iter().zip(&pf_logits).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                eprintln!("sensenova_loop prefill vs cpu max_err = {max_err_pf}, prefill vs gpu_decode max_err = {max_err_gpu}");
                assert!(max_err_pf < 1e-2, "sensenova_loop CPU vs GPU chunked prefill max_err = {max_err_pf} (cpu={cpu_logits:?}, pf={pf_logits:?})");
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn standalone_external_draft_model_spec_step_matches_greedy() {
        let dir = std::env::temp_dir().join(format!("strata_extdraft_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let make_qwen3 = |path: &std::path::Path, seed_off: usize| {
            let mut kvs = Vec::new();
            let w_s = |b: &mut Vec<u8>, s: &str| { b.extend((s.len() as u64).to_le_bytes()); b.extend(s.as_bytes()); };
            let w_kv = |b: &mut Vec<u8>, k: &str, t: u32, p: &[u8]| { w_s(b, k); b.extend(t.to_le_bytes()); b.extend(p); };
            let s = |b: &mut Vec<u8>, k: &str, v: &str| { let mut p = Vec::new(); w_s(&mut p, v); w_kv(b, k, 8, &p); };
            s(&mut kvs, "general.architecture", "qwen3");
            w_kv(&mut kvs, "qwen3.block_count", 4, &1u32.to_le_bytes());
            w_kv(&mut kvs, "qwen3.embedding_length", 4, &64u32.to_le_bytes());
            w_kv(&mut kvs, "qwen3.feed_forward_length", 4, &64u32.to_le_bytes());
            w_kv(&mut kvs, "qwen3.attention.head_count", 4, &1u32.to_le_bytes());
            w_kv(&mut kvs, "qwen3.attention.head_count_kv", 4, &1u32.to_le_bytes());
            w_kv(&mut kvs, "qwen3.attention.key_length", 4, &64u32.to_le_bytes());
            w_kv(&mut kvs, "qwen3.attention.layer_norm_rms_epsilon", 6, &1e-6f32.to_le_bytes());

            let mut hdr = Vec::new();
            let mut body = Vec::new();
            let mut seed = 1 + seed_off;
            let mut add_t = |name: &str, dims: &[u64], scale: f32| {
                w_s(&mut hdr, name);
                hdr.extend((dims.len() as u32).to_le_bytes());
                for &d in dims { hdr.extend(d.to_le_bytes()); }
                hdr.extend(0u32.to_le_bytes());
                hdr.extend((body.len() as u64).to_le_bytes());
                let n: usize = dims.iter().product::<u64>() as usize;
                for i in 0..n {
                    let v = if scale == 0.0 { 1.0f32 } else { (((seed * 29 + i * 13 + 3) % 89) as f32 - 44.0) * scale };
                    body.extend_from_slice(&v.to_le_bytes());
                }
                seed += 1;
                while body.len() % 32 != 0 { body.push(0); }
            };
            add_t("token_embd.weight", &[64, 8], 0.02);
            add_t("output_norm.weight", &[64], 0.0);
            add_t("output.weight", &[64, 8], 0.02);
            add_t("blk.0.attn_norm.weight", &[64], 0.0);
            add_t("blk.0.ffn_norm.weight", &[64], 0.0);
            add_t("blk.0.attn_q_norm.weight", &[64], 0.0);
            add_t("blk.0.attn_k_norm.weight", &[64], 0.0);
            add_t("blk.0.attn_q.weight", &[64, 64], 0.01);
            add_t("blk.0.attn_k.weight", &[64, 64], 0.01);
            add_t("blk.0.attn_v.weight", &[64, 64], 0.01);
            add_t("blk.0.attn_output.weight", &[64, 64], 0.01);
            add_t("blk.0.ffn_gate.weight", &[64, 64], 0.01);
            add_t("blk.0.ffn_up.weight", &[64, 64], 0.01);
            add_t("blk.0.ffn_down.weight", &[64, 64], 0.01);
            let mut gg = Vec::new();
            gg.extend(b"GGUF");
            gg.extend(3u32.to_le_bytes());
            gg.extend(14u64.to_le_bytes());
            gg.extend(8u64.to_le_bytes());
            gg.extend(kvs);
            gg.extend(hdr);
            while gg.len() % 32 != 0 { gg.push(0); }
            gg.extend(body);
            std::fs::write(path, &gg).unwrap();
        };
        let pt = dir.join("target.gguf");
        let pd = dir.join("draft.gguf");
        make_qwen3(&pt, 0);
        make_qwen3(&pd, 0);

        let gt = Gguf::open(&pt).unwrap();
        let gd = Gguf::open(&pd).unwrap();
        let m = Model::load_with_aux(&gt, Some(&gd), &[]).unwrap();
        assert!(m.draft_model.is_some());

        if let Ok(bk1) = crate::backend::open(None, 0)
            && let Ok(bk2) = crate::backend::open(None, 0)
        {
            let mut gm_base = crate::gpu_model::GpuModel::load(&m, bk1, 64, 0, 0, 0).unwrap();
            let mut gm_spec = crate::gpu_model::GpuModel::load(&m, bk2, 64, 0, 0, 3).unwrap();
            assert!(gm_spec.has_ext_draft());

            let l_base = gm_base.prefill(&[1, 2]).unwrap();
            let l_spec = gm_spec.prefill(&[1, 2]).unwrap();
            let mut next_b = crate::ops::argmax(&l_base) as u32;
            let mut next_s = crate::ops::argmax(&l_spec) as u32;
            assert_eq!(next_b, next_s);

            let mut base_toks = Vec::new();
            for _ in 0..8 {
                base_toks.push(next_b);
                next_b = gm_base.step_greedy(next_b).unwrap();
            }
            let mut spec_toks = Vec::new();
            let mut smp = crate::ops::Sampler::greedy();
            let mut row = 0;
            while spec_toks.len() < 8 {
                let (toks, nx, rw) = gm_spec.spec_step(next_s, row, &mut smp).unwrap();
                spec_toks.extend_from_slice(&toks);
                next_s = nx;
                row = rw;
            }
            spec_toks.truncate(8);
            assert_eq!(base_toks, spec_toks);

            // Now test when external draft model has different weights (forces rejections and state rollbacks).
            make_qwen3(&pd, 42);
            let gd2 = Gguf::open(&pd).unwrap();
            let m2 = Model::load_with_aux(&gt, Some(&gd2), &[]).unwrap();
            if let Ok(bk3) = crate::backend::open(None, 0) {
                let mut gm_diff = crate::gpu_model::GpuModel::load(&m2, bk3, 64, 0, 0, 3).unwrap();
                let l_diff = gm_diff.prefill(&[1, 2]).unwrap();
                let mut next_d = crate::ops::argmax(&l_diff) as u32;
                let mut diff_toks = Vec::new();
                let mut row_d = 0;
                while diff_toks.len() < 8 {
                    let (toks, nx, rw) = gm_diff.spec_step(next_d, row_d, &mut smp).unwrap();
                    diff_toks.extend_from_slice(&toks);
                    next_d = nx;
                    row_d = rw;
                }
                diff_toks.truncate(8);
                assert_eq!(base_toks, diff_toks);
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
