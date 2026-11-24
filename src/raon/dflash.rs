//! DFlash2 parallel block-diffusion drafter for KRAFTON Raon-1 mini (`dflash-raon-1-mini-Q8_0.gguf`).
//! Uses `GroupedDynamicCausalConv` (prepare/finish), sliding-window bidirectional block attention with
//! QK-RMSNorm + NeoX RoPE, SwiGLU FFN, and low-rank `CandidateSelector` (`top_k = 16`).
use super::forward::RaonModel;
use crate::deepseek::mat::{Mat, matvec_rows, vec_f32};
use crate::deepseek::ops::rms_norm_w;
use crate::gguf::Gguf;
use crate::weights::dot;
use anyhow::{Context, Result};
use rayon::prelude::*;

struct DfConv<'a> {
    base: Vec<f32>, // [2, K, H]
    proj: Mat<'a>,  // [H, 2*K*G]
}

struct DfLayer<'a> {
    attn_norm: Vec<f32>,
    attn_conv: Option<DfConv<'a>>,
    wq: Mat<'a>,
    wk: Mat<'a>,
    wv: Mat<'a>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    wo: Mat<'a>,
    ffn_norm: Vec<f32>,
    ffn_conv: Option<DfConv<'a>>,
    gate: Mat<'a>,
    up: Mat<'a>,
    down: Mat<'a>,
}

struct Selector<'a> {
    hidden: Mat<'a>,      // [H, rank]
    pred: Mat<'a>,        // [rank, vocab]
    succ: Mat<'a>,        // [rank, vocab]
    rank: usize,
    top_k: usize,
}

pub struct RaonDFlash<'a> {
    pub block_size: usize,
    pub mask_token_id: u32,
    pub target_layers: Vec<usize>, // 0-indexed Raon layer outputs to capture
    n_embd: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    eps: f32,
    rope_base: f32,
    conv_k: usize,
    conv_gs: usize,
    swa_window: usize,
    fc: Mat<'a>,
    enc_norm: Vec<f32>,
    out_norm: Vec<f32>,
    layers: Vec<DfLayer<'a>>,
    selector: Option<Selector<'a>>,
    // Per-layer cached context K/V: [pos * n_head_kv * head_dim]
    k_cache: Vec<Vec<f32>>,
    v_cache: Vec<Vec<f32>>,
    ctx_len: usize,
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// NeoX-style rotary embedding in-place on `n_heads` heads of dimension `hd` at position `pos`.
fn rope_neox_inplace(v: &mut [f32], n_heads: usize, hd: usize, pos: usize, base: f32) {
    let half = hd / 2;
    let p = pos as f32;
    for h in 0..n_heads {
        let s = &mut v[h * hd..(h + 1) * hd];
        for i in 0..half {
            let inv_freq = base.powf(-2.0 * i as f32 / hd as f32);
            let ang = p * inv_freq;
            let (si, co) = ang.sin_cos();
            let (a, b) = (s[i], s[i + half]);
            s[i] = a * co - b * si;
            s[i + half] = a * si + b * co;
        }
    }
}

/// DFlash2 `GroupedDynamicCausalConv` (matches `src/vulkan/shaders/dflash2_conv.comp`).
/// `x`: `[T, H]`, `dyn_w`: `[T, 2, K, G]`, `base`: `[2, K, H]`, `part`: 0 (prepare) or 1 (finish).
fn dflash2_conv(x: &[f32], dyn_w: &[f32], base: &[f32], h: usize, gs: usize, k: usize, t_len: usize, part: usize) -> Vec<f32> {
    let g_cnt = h / gs;
    let mut out = vec![0.0f32; t_len * h];
    for t in 0..t_len {
        let dst = &mut out[t * h..(t + 1) * h];
        for off in 0..k {
            if t >= off {
                let src = &x[(t - off) * h..(t - off + 1) * h];
                let b_row = &base[(part * k + off) * h..(part * k + off + 1) * h];
                let d_row = &dyn_w[((t * 2 + part) * k + off) * g_cnt..((t * 2 + part) * k + off + 1) * g_cnt];
                for i in 0..h {
                    dst[i] += (b_row[i] + d_row[i / gs]) * src[i];
                }
            }
        }
    }
    out
}

impl<'a> RaonDFlash<'a> {
    pub fn load(g: &'a Gguf) -> Result<Self> {
        let u = |k: &str, def: usize| -> usize { g.get(k).ok().and_then(|v| v.as_u64()).map_or(def, |v| v as usize) };
        let f = |k: &str, def: f32| -> f32 { g.get(k).ok().and_then(|v| v.as_f32()).unwrap_or(def) };

        let n_layer = u("dflash.block_count", 5);
        let n_embd = u("dflash.embedding_length", 2560);
        let n_head = u("dflash.attention.head_count", 20);
        let n_head_kv = u("dflash.attention.head_count_kv", 5);
        let head_dim = u("dflash.attention.key_length", 128);
        let eps = f("dflash.attention.layer_norm_rms_epsilon", 1e-5);
        let rope_base = f("dflash.rope.freq_base", 10_000_000.0);
        let block_size = u("dflash.block_size", 8);
        let conv_k = u("dflash.conv_kernel_size", 0);
        let conv_gs = u("dflash.conv_group_size", 16);
        let swa_window = u("dflash.attention.sliding_window", 2048);
        let mask_token_id = u("tokenizer.ggml.mask_token_id", 259_839) as u32;

        // `dflash.target_layers` in GGUF stores 1-indexed hidden-state IDs (e.g. [2, 7, 12, 17, 22]),
        // corresponding to 0-indexed Raon layer outputs [1, 6, 11, 16, 21].
        let target_layers: Vec<usize> = g.get("dflash.target_layers")
            .ok()
            .and_then(|v| v.as_arr())
            .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|id| (id as usize).saturating_sub(1)).collect())
            .filter(|v: &Vec<usize>| !v.is_empty())
            .unwrap_or_else(|| vec![1, 6, 11, 16, 21]);

        let hm = |name: &str| Mat::load_host(g, name);
        let fc = hm("fc.weight")?;
        let enc_norm = vec_f32(g, "enc.output_norm.weight")?;
        let out_norm = vec_f32(g, "output_norm.weight")?;

        let mut layers = Vec::with_capacity(n_layer);
        for il in 0..n_layer {
            let attn_conv = if conv_k > 0 && g.tensors.contains_key(&format!("blk.{il}.attn_conv_base")) {
                Some(DfConv {
                    base: vec_f32(g, &format!("blk.{il}.attn_conv_base"))?,
                    proj: hm(&format!("blk.{il}.attn_conv_proj.weight"))?,
                })
            } else {
                None
            };
            let ffn_conv = if conv_k > 0 && g.tensors.contains_key(&format!("blk.{il}.ffn_conv_base")) {
                Some(DfConv {
                    base: vec_f32(g, &format!("blk.{il}.ffn_conv_base"))?,
                    proj: hm(&format!("blk.{il}.ffn_conv_proj.weight"))?,
                })
            } else {
                None
            };
            layers.push(DfLayer {
                attn_norm: vec_f32(g, &format!("blk.{il}.attn_norm.weight"))?,
                attn_conv,
                wq: hm(&format!("blk.{il}.attn_q.weight"))?,
                wk: hm(&format!("blk.{il}.attn_k.weight"))?,
                wv: hm(&format!("blk.{il}.attn_v.weight"))?,
                q_norm: vec_f32(g, &format!("blk.{il}.attn_q_norm.weight"))?,
                k_norm: vec_f32(g, &format!("blk.{il}.attn_k_norm.weight"))?,
                wo: hm(&format!("blk.{il}.attn_output.weight"))?,
                ffn_norm: vec_f32(g, &format!("blk.{il}.ffn_norm.weight"))?,
                ffn_conv,
                gate: hm(&format!("blk.{il}.ffn_gate.weight"))?,
                up: hm(&format!("blk.{il}.ffn_up.weight"))?,
                down: hm(&format!("blk.{il}.ffn_down.weight"))?,
            });
        }

        let selector_rank = u("dflash.selector_rank", 0);
        let selector_top_k = u("dflash.selector_top_k", 16);
        let selector = if selector_rank > 0 && g.tensors.contains_key("selector_hidden.weight") {
            Some(Selector {
                hidden: hm("selector_hidden.weight").context("selector_hidden.weight")?,
                pred: hm("selector_predecessor.weight").context("selector_predecessor.weight")?,
                succ: hm("selector_successor.weight").context("selector_successor.weight")?,
                rank: selector_rank,
                top_k: selector_top_k,
            })
        } else {
            None
        };

        Ok(Self {
            block_size,
            mask_token_id,
            target_layers,
            n_embd,
            n_head,
            n_head_kv,
            head_dim,
            eps,
            rope_base,
            conv_k,
            conv_gs,
            swa_window,
            fc,
            enc_norm,
            out_norm,
            k_cache: vec![Vec::new(); n_layer],
            v_cache: vec![Vec::new(); n_layer],
            layers,
            selector,
            ctx_len: 0,
        })
    }

    pub fn reset(&mut self) {
        for k in &mut self.k_cache { k.clear(); }
        for v in &mut self.v_cache { v.clear(); }
        self.ctx_len = 0;
    }

    /// Project newly accepted target hidden features into each drafter layer's K/V context cache.
    pub fn commit_ctx(&mut self, feats: &[Vec<f32>]) {
        if feats.is_empty() { return; }
        let nt = feats.len();
        let frefs: Vec<&[f32]> = feats.iter().map(Vec::as_slice).collect();
        let fc_out = matvec_rows(&self.fc, &frefs);
        let mut h = vec![0.0f32; nt * self.n_embd];
        for t in 0..nt {
            rms_norm_w(
                &fc_out[t * self.n_embd..(t + 1) * self.n_embd],
                &self.enc_norm,
                self.eps,
                &mut h[t * self.n_embd..(t + 1) * self.n_embd],
            );
        }
        let hrefs: Vec<&[f32]> = h.chunks_exact(self.n_embd).collect();
        let kv_dim = self.n_head_kv * self.head_dim;
        let hd = self.head_dim;
        for (il, dl) in self.layers.iter().enumerate() {
            let mut ks = matvec_rows(&dl.wk, &hrefs);
            let vs = matvec_rows(&dl.wv, &hrefs);
            let mut kn = vec![0.0f32; hd];
            for t in 0..nt {
                let row = &mut ks[t * kv_dim..(t + 1) * kv_dim];
                for kh in 0..self.n_head_kv {
                    let s = &mut row[kh * hd..(kh + 1) * hd];
                    rms_norm_w(s, &dl.k_norm, self.eps, &mut kn);
                    s.copy_from_slice(&kn);
                }
                rope_neox_inplace(row, self.n_head_kv, hd, self.ctx_len + t, self.rope_base);
            }
            self.k_cache[il].extend_from_slice(&ks);
            self.v_cache[il].extend_from_slice(&vs);
        }
        self.ctx_len += nt;
    }

    /// Draft `k_eff` (`<= block_size - 1`) speculative tokens following `anchor`.
    pub fn draft_block(&mut self, anchor: u32, target: &RaonModel, k_eff: usize) -> Vec<u32> {
        let k_eff = k_eff.min(self.block_size.saturating_sub(1));
        if k_eff == 0 { return Vec::new(); }
        let b_len = k_eff + 1;
        let (ne, nh, nkv, hd, eps) = (self.n_embd, self.n_head, self.n_head_kv, self.head_dim, self.eps);
        let kv_dim = nkv * hd;
        let gqa = nh / nkv;
        let scale = 1.0 / (hd as f32).sqrt();
        let start = self.ctx_len;

        // Input block embeddings: [anchor, mask, mask, ...]
        let mut x = vec![0.0f32; b_len * ne];
        target.embed_tok(anchor, &mut x[..ne]);
        let mut mask_emb = vec![0.0f32; ne];
        target.embed_tok(self.mask_token_id, &mut mask_emb);
        for t in 1..b_len {
            x[t * ne..(t + 1) * ne].copy_from_slice(&mask_emb);
        }

        let mut h = vec![0.0f32; b_len * ne];
        for (il, dl) in self.layers.iter().enumerate() {
            // 1. Attention norm + optional DFlash2 conv prepare
            for t in 0..b_len {
                rms_norm_w(&x[t * ne..(t + 1) * ne], &dl.attn_norm, eps, &mut h[t * ne..(t + 1) * ne]);
            }
            let (q_in, attn_dyn) = if let Some(cv) = &dl.attn_conv {
                let hrefs: Vec<&[f32]> = h.chunks_exact(ne).collect();
                let dyn_w = matvec_rows(&cv.proj, &hrefs);
                let pre = dflash2_conv(&h, &dyn_w, &cv.base, ne, self.conv_gs, self.conv_k, b_len, 0);
                (pre, Some(dyn_w))
            } else {
                (h.clone(), None)
            };

            let qrefs: Vec<&[f32]> = q_in.chunks_exact(ne).collect();
            let mut qs = matvec_rows(&dl.wq, &qrefs);
            let mut ks_blk = matvec_rows(&dl.wk, &qrefs);
            let vs_blk = matvec_rows(&dl.wv, &qrefs);

            let mut tmp = vec![0.0f32; hd];
            for t in 0..b_len {
                let qrow = &mut qs[t * nh * hd..(t + 1) * nh * hd];
                for qh in 0..nh {
                    let s = &mut qrow[qh * hd..(qh + 1) * hd];
                    rms_norm_w(s, &dl.q_norm, eps, &mut tmp);
                    s.copy_from_slice(&tmp);
                }
                rope_neox_inplace(qrow, nh, hd, start + t, self.rope_base);

                let krow = &mut ks_blk[t * kv_dim..(t + 1) * kv_dim];
                for kh in 0..nkv {
                    let s = &mut krow[kh * hd..(kh + 1) * hd];
                    rms_norm_w(s, &dl.k_norm, eps, &mut tmp);
                    s.copy_from_slice(&tmp);
                }
                rope_neox_inplace(krow, nkv, hd, start + t, self.rope_base);
            }

            // Bidirectional block attention over [ctx_start..start] + [0..b_len]
            let ctx_start = start.saturating_sub(self.swa_window);
            let n_ctx = start - ctx_start;
            let total_kv = n_ctx + b_len;
            let k_ctx = &self.k_cache[il];
            let v_ctx = &self.v_cache[il];

            let mut att = vec![0.0f32; b_len * nh * hd];
            att.par_chunks_mut(hd).enumerate().for_each(|(idx, oh)| {
                let (t, qh) = (idx / nh, idx % nh);
                let kh = qh / gqa;
                let qv = &qs[(t * nh + qh) * hd..(t * nh + qh + 1) * hd];
                let mut scores = Vec::with_capacity(total_kv);
                let mut mx = f32::NEG_INFINITY;
                for p in ctx_start..start {
                    let kv = &k_ctx[p * kv_dim + kh * hd..p * kv_dim + (kh + 1) * hd];
                    let s = dot(qv, kv) * scale;
                    if s > mx { mx = s; }
                    scores.push(s);
                }
                for p in 0..b_len {
                    let kv = &ks_blk[p * kv_dim + kh * hd..p * kv_dim + (kh + 1) * hd];
                    let s = dot(qv, kv) * scale;
                    if s > mx { mx = s; }
                    scores.push(s);
                }
                let mut sum = 0.0f32;
                for s in &mut scores { *s = (*s - mx).exp(); sum += *s; }
                let inv = 1.0 / sum;
                for (si, &s) in scores[..n_ctx].iter().enumerate() {
                    let w = s * inv;
                    let p = ctx_start + si;
                    let vv = &v_ctx[p * kv_dim + kh * hd..p * kv_dim + (kh + 1) * hd];
                    for j in 0..hd { oh[j] += w * vv[j]; }
                }
                for (p, &s) in scores[n_ctx..].iter().enumerate() {
                    let w = s * inv;
                    let vv = &vs_blk[p * kv_dim + kh * hd..p * kv_dim + (kh + 1) * hd];
                    for j in 0..hd { oh[j] += w * vv[j]; }
                }
            });

            let arefs: Vec<&[f32]> = att.chunks_exact(nh * hd).collect();
            let mut o = matvec_rows(&dl.wo, &arefs);
            if let (Some(cv), Some(dyn_w)) = (&dl.attn_conv, &attn_dyn) {
                o = dflash2_conv(&o, dyn_w, &cv.base, ne, self.conv_gs, self.conv_k, b_len, 1);
            }
            for (xv, &ov) in x.iter_mut().zip(&o) { *xv += ov; }

            // 2. FFN norm + optional DFlash2 conv prepare + SwiGLU + conv finish
            for t in 0..b_len {
                rms_norm_w(&x[t * ne..(t + 1) * ne], &dl.ffn_norm, eps, &mut h[t * ne..(t + 1) * ne]);
            }
            let (m_in, ffn_dyn) = if let Some(cv) = &dl.ffn_conv {
                let hrefs: Vec<&[f32]> = h.chunks_exact(ne).collect();
                let dyn_w = matvec_rows(&cv.proj, &hrefs);
                let pre = dflash2_conv(&h, &dyn_w, &cv.base, ne, self.conv_gs, self.conv_k, b_len, 0);
                (pre, Some(dyn_w))
            } else {
                (h.clone(), None)
            };
            let mrefs: Vec<&[f32]> = m_in.chunks_exact(ne).collect();
            let mut gy = matvec_rows(&dl.gate, &mrefs);
            let uy = matvec_rows(&dl.up, &mrefs);
            for (g, u) in gy.iter_mut().zip(&uy) { *g = silu(*g) * *u; }
            let nf = dl.gate.rows;
            let grefs: Vec<&[f32]> = gy.chunks_exact(nf).collect();
            let mut fo = matvec_rows(&dl.down, &grefs);
            if let (Some(cv), Some(dyn_w)) = (&dl.ffn_conv, &ffn_dyn) {
                fo = dflash2_conv(&fo, dyn_w, &cv.base, ne, self.conv_gs, self.conv_k, b_len, 1);
            }
            for (xv, &ov) in x.iter_mut().zip(&fo) { *xv += ov; }
        }

        // 3. Final norm + shared target lm_head on draft positions [1..b_len]
        let mut h_draft = vec![0.0f32; k_eff * ne];
        for i in 0..k_eff {
            rms_norm_w(&x[(i + 1) * ne..(i + 2) * ne], &self.out_norm, eps, &mut h_draft[i * ne..(i + 1) * ne]);
        }
        let drefs: Vec<&[f32]> = h_draft.chunks_exact(ne).collect();
        let logits = target.lm_head_rows(&drefs);
        let n_voc = logits.len() / k_eff;

        if let Some(sel) = &self.selector {
            let h_proj = matvec_rows(&sel.hidden, &drefs);
            let k_top = sel.top_k.min(n_voc).max(1);
            let rank = sel.rank;
            let mut pred = anchor;
            let mut out_toks = Vec::with_capacity(k_eff);
            let mut pred_vec = vec![0.0f32; rank];
            let mut succ_vec = vec![0.0f32; rank];
            let mut q_vec = vec![0.0f32; rank];
            let mut cands: Vec<(usize, f32)> = Vec::with_capacity(n_voc);
            for pos in 0..k_eff {
                let row = &logits[pos * n_voc..(pos + 1) * n_voc];
                cands.clear();
                for (idx, &v) in row.iter().enumerate() { cands.push((idx, v)); }
                if cands.len() > k_top {
                    cands.select_nth_unstable_by(k_top - 1, |a, b| b.1.total_cmp(&a.1));
                    cands.truncate(k_top);
                }
                sel.pred.row_into(pred as usize, &mut pred_vec);
                let hp = &h_proj[pos * rank..(pos + 1) * rank];
                for r in 0..rank { q_vec[r] = pred_vec[r] * hp[r]; }
                let mut best_tok = cands[0].0 as u32;
                let mut best_s = f32::NEG_INFINITY;
                for &(idx, u) in &cands {
                    sel.succ.row_into(idx, &mut succ_vec);
                    let s = u + dot(&q_vec, &succ_vec);
                    if s > best_s {
                        best_s = s;
                        best_tok = idx as u32;
                    }
                }
                pred = best_tok;
                out_toks.push(pred);
            }
            out_toks
        } else {
            (0..k_eff)
                .map(|i| {
                    let row = &logits[i * n_voc..(i + 1) * n_voc];
                    row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32
                })
                .collect()
        }
    }
}
