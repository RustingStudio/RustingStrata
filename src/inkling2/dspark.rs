//! Inkling 2 DSpark speculative drafter (`general.architecture = "dflash"` with `dspark_fuse.weight`),
//! matching `llama.cpp` PR #32218 (`src/models/dflash.cpp`, `src/models/dspark.cpp`, `common/speculative.cpp`).
//!
//! Target features from slot visits `[5, 23, 41, 48]` are fused per layer via `dspark_fuse.weight` + `dflare_norm`
//! and projected through `attn_k_ctx` / `attn_v_ctx` into the 3-layer draft KV cache; row 3 (`dspark_norm`)
//! produces the committed context vector. To draft a block of `block_size = 8` tokens (`anchor` + 7 `mask_embd`),
//! the 3 non-causal SWA/full transformer layers compute `hidden[0..8]`, and the chained Markov RNN head
//! (`markov_w1`, `joint_proj`, `dspark_hidden_out`) predicts up to 7 draft tokens through the target's LM head.

use crate::deepseek::mat::{Mat, matvec_rows, matvec_rows_sets};
use crate::deepseek::ops::rms_norm_w;
use crate::gguf::Gguf;
use crate::quant::GgmlType;
use crate::weights::dot;
use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use std::path::Path;

use super::forward::Inkling2Model;

struct DsparkLayer {
    swa: bool,
    dflare_norm: Vec<f32>,
    attn_k_ctx: Mat<'static>,
    attn_v_ctx: Mat<'static>,
    attn_k_ctx_norm: Vec<f32>,
    attn_norm: Vec<f32>,
    wq: Mat<'static>,
    wk: Mat<'static>,
    wv: Mat<'static>,
    wo: Mat<'static>,
    attn_q_norm: Vec<f32>,
    attn_k_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    ffn_gate: Mat<'static>,
    ffn_up: Mat<'static>,
    ffn_down: Mat<'static>,
}

struct MarkovW2Gpu {
    w2: crate::backend::GpuTensor,
    bias: crate::backend::Buf,
}

enum MarkovHead {
    /// Legacy Unsloth DSpark: projects `latent` (`[rank]`) to hidden space (`[n_embd]`), adds `h_row`, then runs target LM head.
    HiddenOut { weight: Mat<'static>, bias: Vec<f32> },
    /// Mainline llama.cpp DSpark: adds low-rank logit delta `markov_w2 @ latent + bias` onto batched target LM head logits.
    Logits { w2: Mat<'static>, bias: Vec<f32>, gpu: std::sync::OnceLock<Option<MarkovW2Gpu>> },
}

pub struct DsparkDrafter {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub rank: usize,
    pub block_size: usize,
    pub n_swa: usize,
    pub eps: f32,
    pub rope_theta: f32,
    pub target_layers: [usize; 4],
    fuse: Vec<f32>,
    dspark_norm: Vec<f32>,
    token_embd_norm: Option<Vec<f32>>,
    mask_embd: Vec<f32>,
    output_norm: Vec<f32>,
    markov_w1: Mat<'static>,
    joint_proj: Mat<'static>,
    joint_proj_b: Vec<f32>,
    markov_head: MarkovHead,
    layers: Vec<DsparkLayer>,
    k_cache: Vec<Vec<f32>>,
    v_cache: Vec<Vec<f32>>,
    ctx_cache: Vec<f32>,
    pub kv_len: usize,
    // Kept last so its mmap outlives all `'static` matrix views above.
    _gguf: Box<Gguf>,
}

fn host_mat<'a>(g: &'a Gguf, name: &str) -> Result<Mat<'a>> {
    let t = g.tensors.get(name).with_context(|| format!("dspark: missing tensor {name}"))?;
    let rows = t.dims[1..].iter().product::<u64>() as usize;
    Ok(Mat { data: g.bytes(t), ty: GgmlType::from_u32(t.ty)?, rows, cols: t.dims[0] as usize, cpu: true, gpu: None, scale: 1.0 })
}

fn f32s(g: &Gguf, name: &str) -> Result<Vec<f32>> {
    let m = host_mat(g, name)?;
    let mut v = vec![0f32; m.rows * m.cols];
    for (r, row) in v.chunks_exact_mut(m.cols).enumerate() { m.row_into(r, row); }
    Ok(v)
}

impl DsparkDrafter {
    pub fn open(path: &Path) -> Result<Self> {
        let g = Box::new(Gguf::open(path)?);
        // SAFETY: `g` is boxed and stored as `_gguf` (the last field of `DsparkDrafter`), so its mmap lives as long
        // as the struct and is dropped after all `Mat<'static>` slices that point into it.
        let g_ref: &'static Gguf = unsafe { &*(&*g as *const Gguf) };
        let arch = g_ref.get("general.architecture")?.as_str().unwrap_or("");
        ensure!(arch == "dflash", "{}: expected architecture 'dflash', got '{arch}'", path.display());
        let u = |k: &str| -> Result<usize> { Ok(g_ref.get(k)?.as_u64().with_context(|| k.to_string())? as usize) };
        let f = |k: &str| -> Result<f32> { g_ref.get(k)?.as_f32().with_context(|| k.to_string()) };

        let n_layer = u("dflash.block_count")?;
        let n_embd = u("dflash.embedding_length")?;
        let n_ff = u("dflash.feed_forward_length")?;
        let n_head = u("dflash.attention.head_count")?;
        let n_head_kv = u("dflash.attention.head_count_kv")?;
        let head_dim = u("dflash.attention.key_length")?;
        let rope_theta = f("dflash.rope.freq_base").unwrap_or(1_000_000.0);
        let eps = f("dflash.attention.layer_norm_rms_epsilon").unwrap_or(1e-6);
        let block_size = u("dflash.block_size").unwrap_or(8);
        let n_swa = u("dflash.attention.sliding_window").unwrap_or(1024);
        let swa: Vec<bool> = g_ref
            .get("dflash.attention.sliding_window_pattern")?
            .as_arr()
            .context("dflash.attention.sliding_window_pattern")?
            .iter()
            .map(|v| match v {
                crate::gguf::Value::Bool(b) => *b,
                other => other.as_u64().unwrap_or(0) != 0,
            })
            .collect();
        ensure!(swa.len() == n_layer, "dspark: sliding_window_pattern len {} != {n_layer}", swa.len());
        let tl_arr = g_ref.get("dflash.target_layers")?.as_arr().context("dflash.target_layers")?;
        ensure!(tl_arr.len() == 4, "dspark: expected 4 target_layers, got {}", tl_arr.len());
        let mut target_layers = [0usize; 4];
        for (i, v) in tl_arr.iter().enumerate() {
            target_layers[i] = v.as_u64().context("dflash.target_layers entry")? as usize;
        }

        let markov_w1 = host_mat(g_ref, "markov_w1.weight")?;
        let rank = markov_w1.cols;
        let fuse = if g_ref.tensors.contains_key("dspark_fuse.weight") {
            let f = f32s(g_ref, "dspark_fuse.weight")?;
            ensure!(f.len() == 4 * (n_layer + 1), "dspark_fuse.weight len {} != {}", f.len(), 4 * (n_layer + 1));
            f
        } else {
            let mut f = Vec::with_capacity(4 * (n_layer + 1));
            for il in 0..n_layer {
                let row = f32s(g_ref, &format!("blk.{il}.ctx_mix.weight"))?;
                ensure!(row.len() == 4, "blk.{il}.ctx_mix.weight len {} != 4", row.len());
                f.extend_from_slice(&row);
            }
            let row = f32s(g_ref, "markov_ctx_mix.weight")?;
            ensure!(row.len() == 4, "markov_ctx_mix.weight len {} != 4", row.len());
            f.extend_from_slice(&row);
            f
        };
        let dspark_norm = if g_ref.tensors.contains_key("dspark_norm.weight") {
            f32s(g_ref, "dspark_norm.weight")?
        } else {
            f32s(g_ref, "markov_ctx_norm.weight")?
        };
        let token_embd_norm = if g_ref.tensors.contains_key("token_embd_norm.weight") {
            Some(f32s(g_ref, "token_embd_norm.weight")?)
        } else {
            None
        };
        let mask_embd = f32s(g_ref, "mask_embd.weight")?;
        ensure!(mask_embd.len() >= n_embd && mask_embd.len() % n_embd == 0, "dspark: mask_embd.weight len {} is not rows of {n_embd}", mask_embd.len());
        let output_norm = f32s(g_ref, "output_norm.weight")?;
        let joint_proj = host_mat(g_ref, "joint_proj.weight")?;
        let joint_proj_b = f32s(g_ref, "joint_proj.bias")?;
        let markov_head = if g_ref.tensors.contains_key("markov_w2.weight") {
            let (w2, bias) = (host_mat(g_ref, "markov_w2.weight")?, f32s(g_ref, "markov_w2.bias")?);
            ensure!(w2.cols == rank, "dspark: markov_w2.weight has {} columns, markov_w1 gives rank {rank}", w2.cols);
            ensure!(bias.len() == w2.rows, "dspark: markov_w2.bias len {} != {} rows of markov_w2.weight", bias.len(), w2.rows);
            MarkovHead::Logits { w2, bias, gpu: std::sync::OnceLock::new() }
        } else {
            MarkovHead::HiddenOut {
                weight: host_mat(g_ref, "dspark_hidden_out.weight")?,
                bias: f32s(g_ref, "dspark_hidden_out.bias")?,
            }
        };

        let mut layers = Vec::with_capacity(n_layer);
        for il in 0..n_layer {
            let p = |s: &str| format!("blk.{il}.{s}");
            let m = |s: &str| Mat::load(g_ref, &p(s));
            let v = |s: &str| f32s(g_ref, &p(s));
            let dflare_norm = if g_ref.tensors.contains_key(&p("dflare_norm.weight")) {
                v("dflare_norm.weight")?
            } else {
                v("ctx_norm.weight")?
            };
            layers.push(DsparkLayer {
                swa: swa[il],
                dflare_norm,
                attn_k_ctx: m("attn_k_ctx.weight")?,
                attn_v_ctx: m("attn_v_ctx.weight")?,
                attn_k_ctx_norm: v("attn_k_ctx_norm.weight")?,
                attn_norm: v("attn_norm.weight")?,
                wq: m("attn_q.weight")?,
                wk: m("attn_k.weight")?,
                wv: m("attn_v.weight")?,
                wo: m("attn_output.weight")?,
                attn_q_norm: v("attn_q_norm.weight")?,
                attn_k_norm: v("attn_k_norm.weight")?,
                ffn_norm: v("ffn_norm.weight")?,
                ffn_gate: m("ffn_gate.weight")?,
                ffn_up: m("ffn_up.weight")?,
                ffn_down: m("ffn_down.weight")?,
            });
        }

        Ok(Self {
            n_layer,
            n_embd,
            n_head,
            n_head_kv,
            head_dim,
            n_ff,
            rank,
            block_size,
            n_swa,
            eps,
            rope_theta,
            target_layers,
            fuse,
            dspark_norm,
            token_embd_norm,
            mask_embd,
            output_norm,
            markov_w1,
            joint_proj,
            joint_proj_b,
            markov_head,
            layers,
            k_cache: vec![Vec::new(); n_layer],
            v_cache: vec![Vec::new(); n_layer],
            ctx_cache: Vec::new(),
            kv_len: 0,
            _gguf: g,
        })
    }

    pub fn reset(&mut self) {
        self.truncate(0);
    }

    pub fn truncate(&mut self, len: usize) {
        if len < self.kv_len {
            let kw = self.n_head_kv * self.head_dim;
            for il in 0..self.n_layer {
                self.k_cache[il].truncate(len * kw);
                self.v_cache[il].truncate(len * kw);
            }
            self.ctx_cache.truncate(len * self.n_embd);
            self.kv_len = len;
        }
    }

    /// Mix the 4 target layer features (`[4][nt][n_embd]`) for `keep` tokens (`0..keep`) using row `row` of `dspark_fuse`
    /// and RMS-normalize each token with `norm_w`.
    fn mix_features(&self, keep: usize, nt: usize, target_feats: &[f32], row: usize, norm_w: &[f32]) -> Vec<f32> {
        let n = self.n_embd;
        let mut w = self.fuse[row * 4..(row + 1) * 4].to_vec();
        crate::ops::softmax(&mut w);
        let mut out = vec![0.0f32; keep * n];
        out.par_chunks_mut(n).enumerate().for_each(|(t, dst)| {
            let mut mixed = vec![0.0f32; n];
            for k in 0..4 {
                let wk = w[k];
                let src = &target_feats[(k * nt + t) * n..(k * nt + t + 1) * n];
                for (m, &s) in mixed.iter_mut().zip(src) { *m += wk * s; }
            }
            rms_norm_w(&mixed, norm_w, self.eps, dst);
        });
        out
    }

    /// Inject the first `keep` tokens (`pos0 .. pos0 + keep`) of a target batch (`nt` tokens, `target_feats` `[4][nt][n_embd]`)
    /// into the DSpark KV cache and `committed_context`.
    pub fn inject_subset(&mut self, pos0: usize, keep: usize, nt: usize, target_feats: &[f32]) {
        if keep == 0 { return; }
        let (n, kw, hd, nkv) = (self.n_embd, self.n_head_kv * self.head_dim, self.head_dim, self.n_head_kv);
        assert_eq!(target_feats.len(), 4 * nt * n, "dspark: invalid target_feats size");
        self.truncate(pos0);
        for il in 0..self.n_layer {
            let ctx = self.mix_features(keep, nt, target_feats, il, &self.layers[il].dflare_norm);
            let l = &self.layers[il];
            let kv = matvec_rows_sets(&[(l.attn_k_ctx, 0), (l.attn_v_ctx, 0)], &ctx, keep);
            let (k_part, v_part) = kv.split_at(keep * kw);
            let mut k_vec = k_part.to_vec();
            let k_norm = &l.attn_k_ctx_norm;
            let eps = self.eps;
            let theta = self.rope_theta;
            k_vec.par_chunks_mut(kw).enumerate().for_each(|(t, row)| {
                let pos = pos0 + t;
                let mut tmp = vec![0.0f32; hd];
                for kh in 0..nkv {
                    let h = &mut row[kh * hd..(kh + 1) * hd];
                    tmp.copy_from_slice(h);
                    rms_norm_w(&tmp, k_norm, eps, h);
                    crate::ops::rope_neox(h, hd, pos, theta);
                }
            });
            self.k_cache[il].extend_from_slice(&k_vec);
            self.v_cache[il].extend_from_slice(v_part);
        }
        let committed = self.mix_features(keep, nt, target_feats, self.n_layer, &self.dspark_norm);
        self.ctx_cache.extend_from_slice(&committed);
        self.kv_len = pos0 + keep;
    }

    /// Draft up to `max_draft` (`<= 7`) tokens starting after `anchor_tok` (which will sit at `pos0`).
    /// Uses `committed_context[pos0 - 1]` and the committed KV cache `[0..pos0]`.
    pub fn draft(&self, target: &mut Inkling2Model, anchor_tok: u32, pos0: usize, max_draft: usize) -> Vec<u32> {
        let count = max_draft.min(self.block_size.saturating_sub(1));
        if count == 0 || pos0 == 0 || pos0 > self.kv_len {
            return Vec::new();
        }
        let (bs, n, nh, nkv, hd, nf) = (self.block_size, self.n_embd, self.n_head, self.n_head_kv, self.head_dim, self.n_ff);
        let (qd, kw, group) = (nh * hd, nkv * hd, nh / nkv);
        let scale = 1.0 / (hd as f32).sqrt();

        // 1. Build 8-token input: slot 0 is normed target token embedding of `anchor_tok`, slots 1..8 are `mask_embd`.
        let mut inp = vec![0.0f32; bs * n];
        if let Some(norm_w) = &self.token_embd_norm {
            let mut emb = vec![0.0f32; n];
            target.embed_raw(anchor_tok, &mut emb);
            rms_norm_w(&emb, norm_w, self.eps, &mut inp[..n]);
        } else {
            target.embed_normed(anchor_tok, &mut inp[..n]);
        }
        let per_pos_mask = self.mask_embd.len() >= (bs - 1) * n;
        for i in 1..bs {
            let src = if per_pos_mask { &self.mask_embd[(i - 1) * n..i * n] } else { &self.mask_embd[..n] };
            inp[i * n..(i + 1) * n].copy_from_slice(src);
        }

        // 2. Run the 3 non-causal DSpark transformer layers over the 8-token block.
        let mut h = vec![0.0f32; bs * n];
        for il in 0..self.n_layer {
            let l = &self.layers[il];
            for i in 0..bs {
                rms_norm_w(&inp[i * n..(i + 1) * n], &l.attn_norm, self.eps, &mut h[i * n..(i + 1) * n]);
            }
            let qkv = matvec_rows_sets(&[(l.wq, 0), (l.wk, 0), (l.wv, 0)], &h, bs);
            let (q_slice, kv_slice) = qkv.split_at(bs * qd);
            let (k_slice, v_slice) = kv_slice.split_at(bs * kw);
            let mut q = q_slice.to_vec();
            let mut k_blk = k_slice.to_vec();
            let v_blk = v_slice;

            let (eps, theta) = (self.eps, self.rope_theta);
            q.par_chunks_mut(qd).enumerate().for_each(|(i, row)| {
                let pos = pos0 + i;
                let mut tmp = vec![0.0f32; hd];
                for head in row.chunks_exact_mut(hd) {
                    tmp.copy_from_slice(head);
                    rms_norm_w(&tmp, &l.attn_q_norm, eps, head);
                    crate::ops::rope_neox(head, hd, pos, theta);
                }
            });
            k_blk.par_chunks_mut(kw).enumerate().for_each(|(i, row)| {
                let pos = pos0 + i;
                let mut tmp = vec![0.0f32; hd];
                for head in row.chunks_exact_mut(hd) {
                    tmp.copy_from_slice(head);
                    rms_norm_w(&tmp, &l.attn_k_norm, eps, head);
                    crate::ops::rope_neox(head, hd, pos, theta);
                }
            });

            // Non-causal attention: every query in 0..bs sees committed KV in `[first..pos0]` plus all `0..bs` block keys.
            let first = if l.swa { pos0.saturating_sub(self.n_swa) } else { 0 };
            let n_ctx = pos0 - first;
            let n_tot = n_ctx + bs;
            let kc = &self.k_cache[il][first * kw..pos0 * kw];
            let vc = &self.v_cache[il][first * kw..pos0 * kw];
            let mut attn = vec![0.0f32; bs * qd];
            attn.par_chunks_mut(hd).enumerate().for_each(|(idx, oh)| {
                let (i, qh) = (idx / nh, idx % nh);
                let kh = qh / group;
                let q_vec = &q[i * qd + qh * hd..i * qd + (qh + 1) * hd];
                let mut scores = vec![0.0f32; n_tot];
                for s in 0..n_ctx {
                    let k_vec = &kc[s * kw + kh * hd..s * kw + (kh + 1) * hd];
                    scores[s] = dot(q_vec, k_vec) * scale;
                }
                for s in 0..bs {
                    let k_vec = &k_blk[s * kw + kh * hd..s * kw + (kh + 1) * hd];
                    scores[n_ctx + s] = dot(q_vec, k_vec) * scale;
                }
                crate::ops::softmax(&mut scores);
                for s in 0..n_ctx {
                    let v_vec = &vc[s * kw + kh * hd..s * kw + (kh + 1) * hd];
                    let w = scores[s];
                    for (a, &b) in oh.iter_mut().zip(v_vec) { *a += w * b; }
                }
                for s in 0..bs {
                    let v_vec = &v_blk[s * kw + kh * hd..s * kw + (kh + 1) * hd];
                    let w = scores[n_ctx + s];
                    for (a, &b) in oh.iter_mut().zip(v_vec) { *a += w * b; }
                }
            });

            let attn_rows: Vec<&[f32]> = attn.chunks_exact(qd).collect();
            let attn_out = matvec_rows(&l.wo, &attn_rows);
            for (a, &b) in inp.iter_mut().zip(&attn_out) { *a += b; }

            for i in 0..bs {
                rms_norm_w(&inp[i * n..(i + 1) * n], &l.ffn_norm, self.eps, &mut h[i * n..(i + 1) * n]);
            }
            let gu = matvec_rows_sets(&[(l.ffn_gate, 0), (l.ffn_up, 0)], &h, bs);
            let (gate_s, up_s) = gu.split_at(bs * nf);
            let mut act = gate_s.to_vec();
            act.par_iter_mut().zip(up_s).for_each(|(g, &u)| *g = crate::ops::silu(*g) * u);
            let act_rows: Vec<&[f32]> = act.chunks_exact(nf).collect();
            let down = matvec_rows(&l.ffn_down, &act_rows);
            for (a, &b) in inp.iter_mut().zip(&down) { *a += b; }
        }

        let mut hidden = vec![0.0f32; bs * n];
        for i in 0..bs {
            rms_norm_w(&inp[i * n..(i + 1) * n], &self.output_norm, self.eps, &mut hidden[i * n..(i + 1) * n]);
        }

        // 3. Chained Markov RNN Head across draft positions `1..=count`.
        let r = self.rank;
        let ctx_feat = &self.ctx_cache[(pos0 - 1) * n..pos0 * n];
        let mut state = vec![0.0f32; r];
        let mut prev_embd = vec![0.0f32; r];
        let mut joint_in = vec![0.0f32; 2 * r + 2 * n];
        joint_in[2 * r + n..].copy_from_slice(ctx_feat);
        let mut joint = vec![0.0f32; 3 * r];
        let mut latent = vec![0.0f32; r];
        let mut prev = anchor_tok;
        let mut drafts = Vec::with_capacity(count);

        let markov_step = |prev_tok: u32,
                           h_row: &[f32],
                           state: &mut [f32],
                           prev_embd: &mut [f32],
                           joint_in: &mut [f32],
                           joint: &mut [f32],
                           latent: &mut [f32]| {
            self.markov_w1.row_into(prev_tok as usize, prev_embd);
            joint_in[..r].copy_from_slice(state);
            joint_in[r..2 * r].copy_from_slice(prev_embd);
            joint_in[2 * r..2 * r + n].copy_from_slice(h_row);
            self.joint_proj.matvec(joint_in, joint);
            for (v, &b) in joint.iter_mut().zip(&self.joint_proj_b) { *v += b; }
            for k in 0..r {
                let gate = crate::ops::sigmoid(joint[k]);
                let cand = joint[r + k].tanh();
                state[k] = cand + gate * (state[k] - cand);
                latent[k] = joint[2 * r + k].tanh();
            }
        };

        match &self.markov_head {
            MarkovHead::HiddenOut { weight, bias } => {
                let mut corr = vec![0.0f32; n];
                for step in 0..count {
                    let h_row = &hidden[(step + 1) * n..(step + 2) * n];
                    markov_step(prev, h_row, &mut state, &mut prev_embd, &mut joint_in, &mut joint, &mut latent);
                    weight.matvec(&latent, &mut corr);
                    for ((c, &b), &h_v) in corr.iter_mut().zip(bias).zip(h_row) {
                        *c = *c + b + h_v;
                    }
                    prev = target.lm_head_argmax(&corr);
                    drafts.push(prev);
                }
            }
            MarkovHead::Logits { w2, bias, gpu } => {
                let gpu_w2 = gpu.get_or_init(|| {
                    if w2.rows > target.output_rows() || w2.cols > n {
                        return None;
                    }
                    let dm = crate::deepseek::mat::dev::get()?;
                    let d = dm.lock().ok()?;
                    let be = &*d.parts[0].be;
                    let w2_gpu = be.upload(&crate::weights::Tensor {
                        ty: w2.ty,
                        rows: w2.rows,
                        cols: w2.cols,
                        data: w2.data,
                        scale: w2.scale,
                    }).ok()?;
                    let bias_buf = be.alloc(bias.len() * 4).ok()?;
                    be.put(bias, &bias_buf, 0).ok()?;
                    Some(MarkovW2Gpu { w2: w2_gpu, bias: bias_buf })
                });
                let mut use_gpu = gpu_w2.is_some()
                    && target.lm_head_batch_gpu(&hidden[n..(1 + count) * n], count);
                let mut cpu_base: Option<Vec<f32>> = None;
                let mut delta = Vec::new();
                let n_vocab = target.cfg.n_vocab;
                let nv = n_vocab.min(w2.rows).min(bias.len());

                for step in 0..count {
                    let h_row = &hidden[(step + 1) * n..(step + 2) * n];
                    markov_step(prev, h_row, &mut state, &mut prev_embd, &mut joint_in, &mut joint, &mut latent);
                    if use_gpu {
                        let g = gpu_w2.as_ref().unwrap();
                        if let Some(id) = target.markov_w2_argmax_gpu(&g.w2, g.bias.addr(), bias.len(), &latent, step) {
                            prev = id;
                            drafts.push(prev);
                            continue;
                        }
                        use_gpu = false;
                    }
                    let base_all = cpu_base.get_or_insert_with(|| target.lm_head_batch_cpu(&hidden[n..(1 + count) * n], count));
                    if delta.is_empty() { delta.resize(w2.rows, 0.0); }
                    w2.matvec(&latent, &mut delta);
                    let base = &base_all[step * n_vocab..(step + 1) * n_vocab];
                    let (mut best_i, mut best_v) = (0usize, f32::NEG_INFINITY);
                    for i in 0..nv {
                        let v = base[i] + delta[i] + bias[i];
                        if v > best_v {
                            best_v = v;
                            best_i = i;
                        }
                    }
                    prev = best_i as u32;
                    drafts.push(prev);
                }
            }
        }
        drafts
    }
}
