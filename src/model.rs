//! Scalar CPU reference forward for the architectures in `config::arch`, one token at a time.
//! Mirrors llama.cpp `src/models/qwen35moe.cpp`; this is the correctness oracle, not the fast path.

use crate::config::{LayerKind, ModelConfig};
use crate::gguf::Gguf;
use crate::ops;
use crate::weights::{Tensor, Weights};

pub struct Model<'a> {
    pub cfg: ModelConfig,
    pub w: Weights<'a>,
}

pub enum LayerState {
    /// conv: last (ssm_conv - 1) inputs per channel, oldest first; s: dt_rank x head_v x head_k.
    Recurrent { conv: Vec<f32>, s: Vec<f32> },
    /// k, v: pos x n_head_kv x head_dim.
    Attn { k: Vec<f32>, v: Vec<f32> },
}

pub struct State {
    pub layers: Vec<LayerState>,
    pub pos: usize,
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
                LayerKind::Attn => LayerState::Attn { k: Vec::new(), v: Vec::new() },
            })
            .collect();
        State { layers, pos: 0 }
    }
}

/// MTP draft state: the KV cache of the nextn block.
pub struct Mtp { pub kv: LayerState, pub pos: usize }

impl Default for Mtp {
    fn default() -> Mtp { Mtp { kv: LayerState::Attn { k: Vec::new(), v: Vec::new() }, pos: 0 } }
}

fn mv(t: &Tensor, x: &[f32]) -> Vec<f32> {
    let mut y = vec![0f32; t.rows];
    t.matvec(x, &mut y);
    y
}

impl<'a> Model<'a> {
    pub fn load(g: &'a Gguf) -> anyhow::Result<Model<'a>> {
        let cfg = ModelConfig::from_gguf(g)?;
        let w = Weights::new(g)?;
        w.validate()?;
        w.get("token_embd.weight")?;
        w.get("output_norm.weight")?;
        Ok(Model { cfg, w })
    }

    fn t(&self, il: usize, suffix: &str) -> Tensor<'a> {
        self.w.layer(il, suffix).unwrap_or_else(|e| panic!("{e:#}"))
    }
    fn f(&self, il: usize, suffix: &str) -> Vec<f32> { self.t(il, suffix).to_f32() }

    pub fn embed(&self, token: u32) -> Vec<f32> {
        let mut x = vec![0f32; self.cfg.n_embd];
        self.w.get("token_embd.weight").unwrap().row(token as usize, &mut x);
        x
    }

    /// Run one token; returns logits over the vocabulary.
    pub fn step(&self, st: &mut State, token: u32) -> Vec<f32> { self.step_h(st, token).0 }

    /// Run one token; returns logits and the last hidden state (before output_norm).
    pub fn step_h(&self, st: &mut State, token: u32) -> (Vec<f32>, Vec<f32>) {
        let mut x = self.embed(token);
        for il in 0..self.cfg.n_layer { self.block(il, &mut x, &mut st.layers[il], st.pos); }
        st.pos += 1;
        (self.head(&x, &self.w.get("output_norm.weight").unwrap().to_f32()), x)
    }

    /// One decoder block (mixer and FFN, each with residual) on x at position `pos`.
    fn block(&self, il: usize, x: &mut [f32], ls: &mut LayerState, pos: usize) {
        let c = &self.cfg;
        let mut h = vec![0f32; c.n_embd];
        ops::rms_norm(x, &self.f(il, "attn_norm.weight"), c.eps, &mut h);
        let mix = match ls {
            LayerState::Recurrent { conv, s } => self.deltanet(il, &h, conv, s),
            LayerState::Attn { k, v } => self.attention(il, &h, k, v, pos),
        };
        for (a, b) in x.iter_mut().zip(&mix) { *a += b; }
        ops::rms_norm(x, &self.f(il, "post_attention_norm.weight"), c.eps, &mut h);
        let ffn = if c.is_moe() { self.moe(il, &h) } else { self.dense_ffn(il, &h) };
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
        let mut eh = vec![0f32; 2 * n];
        ops::rms_norm(&self.embed(token), &self.f(il, "nextn.enorm.weight"), c.eps, &mut eh[..n]);
        ops::rms_norm(hid, &self.f(il, "nextn.hnorm.weight"), c.eps, &mut eh[n..]);
        let mut x = mv(&self.t(il, "nextn.eh_proj.weight"), &eh);
        self.block(il, &mut x, &mut ms.kv, ms.pos);
        ms.pos += 1;
        (self.head(&x, &self.f(il, "nextn.shared_head_norm.weight")), x)
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
            for (x, zz) in oh.iter_mut().zip(&z[vh * dv..(vh + 1) * dv]) { *x *= ops::silu(*zz); }
        }
        mv(&self.t(il, "ssm_out.weight"), &out)
    }

    fn attention(&self, il: usize, h: &[f32], kc: &mut Vec<f32>, vc: &mut Vec<f32>, pos: usize) -> Vec<f32> {
        let c = &self.cfg;
        let (hd, nh, nkv) = (c.head_dim, c.n_head, c.n_head_kv);
        let qs = if c.q_gate { 2 * hd } else { hd };  // per-head row of attn_q: query, then the gate if any
        let qg = mv(&self.t(il, "attn_q.weight"), h);
        let (qn, kn) = (self.f(il, "attn_q_norm.weight"), self.f(il, "attn_k_norm.weight"));
        let mut q = vec![0f32; nh * hd];
        for i in 0..nh {
            let qh = &mut q[i * hd..(i + 1) * hd];
            ops::rms_norm(&qg[i * qs..i * qs + hd], &qn, c.eps, qh);
            ops::rope_neox(qh, c.n_rot, pos, c.rope_base);
        }
        let kraw = mv(&self.t(il, "attn_k.weight"), h);
        let v = mv(&self.t(il, "attn_v.weight"), h);
        let mut k = vec![0f32; nkv * hd];
        for i in 0..nkv {
            let kh = &mut k[i * hd..(i + 1) * hd];
            ops::rms_norm(&kraw[i * hd..(i + 1) * hd], &kn, c.eps, kh);
            ops::rope_neox(kh, c.n_rot, pos, c.rope_base);
        }
        kc.extend_from_slice(&k);
        vc.extend_from_slice(&v);
        let n = pos + 1;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut out = vec![0f32; nh * hd];
        let mut sc = vec![0f32; n];
        for i in 0..nh {
            let g = i / (nh / nkv);
            let qh = &q[i * hd..(i + 1) * hd];
            for (t, s) in sc.iter_mut().enumerate() {
                let kt = &kc[(t * nkv + g) * hd..(t * nkv + g + 1) * hd];
                *s = qh.iter().zip(kt).map(|(a, b)| a * b).sum::<f32>() * scale;
            }
            ops::softmax(&mut sc);
            let oh = &mut out[i * hd..(i + 1) * hd];
            for (t, p) in sc.iter().enumerate() {
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

    fn ffn(gate: &Tensor, up: &Tensor, down: &Tensor, h: &[f32]) -> Vec<f32> {
        let mut a = mv(gate, h);
        for (x, u) in a.iter_mut().zip(mv(up, h)) { *x = ops::silu(*x) * u; }
        mv(down, &a)
    }

    fn dense_ffn(&self, il: usize, h: &[f32]) -> Vec<f32> {
        Self::ffn(&self.t(il, "ffn_gate.weight"), &self.t(il, "ffn_up.weight"), &self.t(il, "ffn_down.weight"), h)
    }

    fn moe(&self, il: usize, h: &[f32]) -> Vec<f32> {
        let mut y = self.routed_experts(il, h);
        if let Ok(sg) = self.w.layer(il, "ffn_gate_inp_shexp.weight") {
            let s = ops::sigmoid(mv(&sg, h)[0]);
            let o = Self::ffn(&self.t(il, "ffn_gate_shexp.weight"), &self.t(il, "ffn_up_shexp.weight"), &self.t(il, "ffn_down_shexp.weight"), h);
            for (a, b) in y.iter_mut().zip(o) { *a += s * b; }
        }
        y
    }

    /// Router softmax, top-k and normalized weights: (expert, weight) pairs.
    pub fn route(&self, il: usize, h: &[f32]) -> Vec<(usize, f32)> {
        let c = &self.cfg;
        let mut p = mv(&self.t(il, "ffn_gate_inp.weight"), h);
        ops::softmax(&mut p);
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
            let o = Self::ffn(&gate.expert(e, n), &up.expert(e, n), &down.expert(e, n), h);
            for (a, b) in y.iter_mut().zip(o) { *a += w * b; }
        }
        y
    }
}
