//! GPU decode forward for `qwen35` / `qwen35moe` with a fixed expert placement: routed experts of the first N MoE
//! layers live in VRAM (layer order, as many as fit), the remaining layers run their routed experts on the CPU
//! reference path. Mirrors `model.rs` step for step.

use crate::cuda::{Gpu, GpuTensor};
use crate::model::Model;
use anyhow::bail;
use cudarc::driver::CudaSlice;
use std::collections::HashMap;
use std::time::{Duration, Instant};

enum Mixer {
    Delta {
        qkv: GpuTensor, gate: GpuTensor, beta: GpuTensor, alpha: GpuTensor, dt_bias: GpuTensor, a: GpuTensor,
        conv_w: GpuTensor, norm: GpuTensor, out: GpuTensor, conv: CudaSlice<f32>, s: CudaSlice<f32>,
    },
    Attn { q: GpuTensor, k: GpuTensor, v: GpuTensor, qn: GpuTensor, kn: GpuTensor, o: GpuTensor, kc: CudaSlice<u16>, vc: CudaSlice<u16> },
}

struct Shared { gate_inp: GpuTensor, gate: GpuTensor, up: GpuTensor, down: GpuTensor }

enum Ffn {
    Dense { gate: GpuTensor, up: GpuTensor, down: GpuTensor },
    /// `experts` is None when this layer's routed experts run on the CPU.
    Moe { router: GpuTensor, experts: Option<[GpuTensor; 3]>, shared: Option<Shared> },
}

struct Layer { attn_norm: GpuTensor, post_norm: GpuTensor, mixer: Mixer, ffn: Ffn }

pub struct GpuModel<'a> {
    m: &'a Model<'a>,
    pub gpu: Gpu,
    layers: Vec<Layer>,
    output_norm: GpuTensor,
    output: GpuTensor,
    /// Scratch activations, addressed by name through `ptrs`.
    bufs: Vec<CudaSlice<f32>>,
    ptrs: HashMap<&'static str, u64>,
    x: CudaSlice<f32>,
    cpu: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    ids: CudaSlice<i32>,
    ctx: usize,
    pub pos: usize,
    pub resident_layers: usize,
    /// Time spent in CPU expert layers (including the device sync before them).
    pub cpu_time: Duration,
}

impl<'a> GpuModel<'a> {
    pub fn load(m: &'a Model<'a>, gpu: Gpu, ctx: usize, vram_margin: usize) -> anyhow::Result<GpuModel<'a>> {
        let c = &m.cfg;
        let up = |name: &str| -> anyhow::Result<GpuTensor> { gpu.upload(&m.w.get(name)?) };
        let lt = |il: usize, s: &str| up(&format!("blk.{il}.{s}"));
        let (free0, _) = gpu.ctx.mem_get_info()?;
        let (dk, nk, nv) = (c.ssm_d_state, c.ssm_n_group, c.ssm_dt_rank);
        let channels = c.ssm_d_inner + 2 * nk * dk;
        let kv = c.n_head_kv * c.head_dim;

        let mut layers = Vec::with_capacity(c.n_layer);
        for il in 0..c.n_layer {
            let mixer = if c.is_recurrent(il) {
                Mixer::Delta {
                    qkv: lt(il, "attn_qkv.weight")?, gate: lt(il, "attn_gate.weight")?, beta: lt(il, "ssm_beta.weight")?,
                    alpha: lt(il, "ssm_alpha.weight")?, dt_bias: lt(il, "ssm_dt.bias")?, a: lt(il, "ssm_a")?,
                    conv_w: lt(il, "ssm_conv1d.weight")?, norm: lt(il, "ssm_norm.weight")?, out: lt(il, "ssm_out.weight")?,
                    conv: gpu.zeros((c.ssm_conv - 1) * channels)?, s: gpu.zeros(nv * (c.ssm_d_inner / nv) * dk)?,
                }
            } else {
                Mixer::Attn {
                    q: lt(il, "attn_q.weight")?, k: lt(il, "attn_k.weight")?, v: lt(il, "attn_v.weight")?,
                    qn: lt(il, "attn_q_norm.weight")?, kn: lt(il, "attn_k_norm.weight")?, o: lt(il, "attn_output.weight")?,
                    kc: gpu.zeros(ctx * kv)?, vc: gpu.zeros(ctx * kv)?,
                }
            };
            let ffn = if c.is_moe() {
                let shared = if m.w.layer(il, "ffn_gate_inp_shexp.weight").is_ok() {
                    Some(Shared {
                        gate_inp: lt(il, "ffn_gate_inp_shexp.weight")?, gate: lt(il, "ffn_gate_shexp.weight")?,
                        up: lt(il, "ffn_up_shexp.weight")?, down: lt(il, "ffn_down_shexp.weight")?,
                    })
                } else { None };
                Ffn::Moe { router: lt(il, "ffn_gate_inp.weight")?, experts: None, shared }
            } else {
                Ffn::Dense { gate: lt(il, "ffn_gate.weight")?, up: lt(il, "ffn_up.weight")?, down: lt(il, "ffn_down.weight")? }
            };
            layers.push(Layer { attn_norm: lt(il, "attn_norm.weight")?, post_norm: lt(il, "post_attention_norm.weight")?, mixer, ffn });
        }
        let output_norm = up("output_norm.weight")?;
        let output = up("output.weight").or_else(|_| up("token_embd.weight"))?;

        let k = c.n_expert_used.max(1);
        let ff = (k * c.n_ff_exp).max(c.n_ff);
        let sizes: [(&'static str, usize); 23] = [
            ("h", c.n_embd), ("mix", c.n_embd), ("qkv", channels), ("z", c.ssm_d_inner), ("beta", nv), ("g", nv),
            ("y", channels), ("o", c.ssm_d_inner), ("qg", c.n_head * 2 * c.head_dim), ("q", c.n_head * c.head_dim),
            ("kraw", kv), ("k", kv), ("v", kv), ("att", c.n_head * c.head_dim), ("rl", c.n_expert), ("wts", k),
            ("ge", ff), ("ue", ff), ("ye", k * c.n_embd), ("shg", c.n_ff_shexp), ("shu", c.n_ff_shexp),
            ("shy", c.n_embd), ("sg", 1),
        ];
        let mut bufs = Vec::new();
        let mut ptrs = HashMap::new();
        for (name, n) in sizes {
            let b = gpu.zeros::<f32>(n)?;
            ptrs.insert(name, gpu.ptr(&b));
            bufs.push(b);
        }
        let (x, cpu, logits, ids) = (gpu.zeros(c.n_embd)?, gpu.zeros(c.n_embd)?, gpu.zeros(c.n_vocab)?, gpu.zeros::<i32>(k)?);
        let (free1, _) = gpu.ctx.mem_get_info()?;

        // Fixed placement: whole layers of routed experts, in layer order, while VRAM minus the margin allows.
        let mut resident_layers = 0;
        let mut expert_bytes = 0;
        for (il, l) in layers.iter_mut().enumerate() {
            let Ffn::Moe { experts, .. } = &mut l.ffn else { continue };
            let names = ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"];
            let need: usize = names.iter().map(|s| m.w.layer(il, s).map(|t| t.data.len()).unwrap_or(0)).sum();
            let (free, _) = gpu.ctx.mem_get_info()?;
            if free < need.saturating_add(vram_margin) { break; }
            *experts = Some([lt(il, names[0])?, lt(il, names[1])?, lt(il, names[2])?]);
            resident_layers += 1;
            expert_bytes += need;
        }
        let mb = |b: usize| b as f64 / 1048576.0;
        eprintln!(
            "plan: weights+state+scratch {:.0} MB (KV {:.0} MB for {ctx} tokens), routed experts {:.0} MB, resident expert layers {resident_layers}/{}",
            mb(free0 - free1), mb(2 * 2 * ctx * kv * (c.n_layer / c.full_attention_interval)), mb(expert_bytes),
            if c.is_moe() { c.n_layer } else { 0 },
        );
        Ok(GpuModel { m, gpu, layers, output_norm, output, bufs, ptrs, x, cpu, logits, ids, ctx, pos: 0, resident_layers, cpu_time: Duration::ZERO })
    }

    /// Clear KV and recurrent state.
    pub fn reset(&mut self) -> anyhow::Result<()> {
        for l in &mut self.layers {
            match &mut l.mixer {
                Mixer::Delta { conv, s, .. } => { self.gpu.stream.memset_zeros(conv)?; self.gpu.stream.memset_zeros(s)?; }
                Mixer::Attn { kc, vc, .. } => { self.gpu.stream.memset_zeros(kc)?; self.gpu.stream.memset_zeros(vc)?; }
            }
        }
        self.pos = 0;
        Ok(())
    }

    /// Run one token; returns logits over the vocabulary.
    pub fn step(&mut self, token: u32) -> anyhow::Result<Vec<f32>> {
        let c = &self.m.cfg;
        if self.pos >= self.ctx { bail!("context full ({} tokens)", self.ctx); }
        let (g, pos, eps, ne) = (&self.gpu, self.pos, c.eps, c.n_embd);
        g.stream.memcpy_htod(&self.m.embed(token), &mut self.x)?;
        let p = |n: &str| self.ptrs[n];
        let x = g.ptr(&self.x);
        let (dk, nk, nv) = (c.ssm_d_state, c.ssm_n_group, c.ssm_dt_rank);
        let dv = c.ssm_d_inner / nv;
        let (hd, nh, nkv) = (c.head_dim, c.n_head, c.n_head_kv);
        let mut cpu_time = Duration::ZERO;
        for (il, l) in self.layers.iter().enumerate() {
            g.rms_norm(x, 0, l.attn_norm.ptr, p("h"), 0, ne, 1, eps)?;
            match &l.mixer {
                Mixer::Delta { qkv, gate, beta, alpha, dt_bias, a, conv_w, norm, out, conv, s } => {
                    g.gemv(qkv, p("h"), p("qkv"))?;
                    g.gemv(gate, p("h"), p("z"))?;
                    g.gemv(beta, p("h"), p("beta"))?;
                    g.gemv(alpha, p("h"), p("g"))?;
                    g.ssm_gates(p("beta"), p("g"), dt_bias.ptr, a.ptr, nv)?;
                    g.conv_silu(p("qkv"), g.ptr(conv), conv_w.ptr, p("y"), c.ssm_d_inner + 2 * nk * dk, c.ssm_conv)?;
                    g.l2_norm(p("y"), dk, 2 * nk, eps)?;
                    g.deltanet_step(p("y"), g.ptr(s), p("beta"), p("g"), p("o"), nk, nv, dk, dv, 1.0 / (dk as f32).sqrt())?;
                    g.norm_gate(p("o"), norm.ptr, p("z"), dv, nv, eps)?;
                    g.gemv(out, p("o"), p("mix"))?;
                }
                Mixer::Attn { q, k, v, qn, kn, o, kc, vc } => {
                    g.gemv(q, p("h"), p("qg"))?;
                    g.rms_norm(p("qg"), 2 * hd, qn.ptr, p("q"), hd, hd, nh, eps)?;
                    g.rope_neox(p("q"), hd, nh, c.n_rot, pos, c.rope_base)?;
                    g.gemv(k, p("h"), p("kraw"))?;
                    g.gemv(v, p("h"), p("v"))?;
                    g.rms_norm(p("kraw"), hd, kn.ptr, p("k"), hd, hd, nkv, eps)?;
                    g.rope_neox(p("k"), hd, nkv, c.n_rot, pos, c.rope_base)?;
                    g.kv_store(p("k"), p("v"), g.ptr(kc), g.ptr(vc), nkv * hd, pos)?;
                    g.attn_decode(p("q"), g.ptr(kc), g.ptr(vc), p("qg"), p("att"), pos + 1, nh, nkv, hd, 1.0 / (hd as f32).sqrt())?;
                    g.gemv(o, p("att"), p("mix"))?;
                }
            }
            g.add(x, p("mix"), ne)?;
            g.rms_norm(x, 0, l.post_norm.ptr, p("h"), 0, ne, 1, eps)?;
            match &l.ffn {
                Ffn::Dense { gate, up, down } => {
                    g.gemv(gate, p("h"), p("ge"))?;
                    g.gemv(up, p("h"), p("ue"))?;
                    g.swiglu(p("ge"), p("ue"), c.n_ff)?;
                    g.gemv(down, p("ge"), p("ye"))?;
                    g.add(x, p("ye"), ne)?;
                }
                Ffn::Moe { router, experts, shared } => {
                    let (sh, sg) = match shared {
                        Some(s) => {
                            g.gemv(&s.gate_inp, p("h"), p("sg"))?;
                            g.gemv(&s.gate, p("h"), p("shg"))?;
                            g.gemv(&s.up, p("h"), p("shu"))?;
                            g.swiglu(p("shg"), p("shu"), c.n_ff_shexp)?;
                            g.gemv(&s.down, p("shg"), p("shy"))?;
                            (p("shy"), p("sg"))
                        }
                        None => (0, 0),
                    };
                    let k = c.n_expert_used;
                    if let Some([eg, eu, ed]) = experts {
                        let ids = g.ptr(&self.ids);
                        g.gemv(router, p("h"), p("rl"))?;
                        g.router_topk(p("rl"), c.n_expert, k, c.expert_weights_scale, ids, p("wts"))?;
                        g.moe_gemv(eg, c.n_expert, ids, k, p("h"), 0, p("ge"))?;
                        g.moe_gemv(eu, c.n_expert, ids, k, p("h"), 0, p("ue"))?;
                        g.swiglu(p("ge"), p("ue"), k * c.n_ff_exp)?;
                        g.moe_gemv(ed, c.n_expert, ids, k, p("ge"), c.n_ff_exp, p("ye"))?;
                        g.moe_combine(x, p("ye"), p("wts"), k, sh, sg, 0, ne)?;
                    } else {
                        // ponytail: synchronous CPU round trip per layer; M3 overlaps it with GPU work.
                        let t0 = Instant::now();
                        let h = g.stream.clone_dtoh(&self.bufs[0])?;
                        let y = self.m.routed_experts(il, &h);
                        g.stream.memcpy_htod(&y, &mut self.cpu)?;
                        cpu_time += t0.elapsed();
                        g.moe_combine(x, 0, 0, 0, sh, sg, g.ptr(&self.cpu), ne)?;
                    }
                }
            }
        }
        g.rms_norm(x, 0, self.output_norm.ptr, p("h"), 0, ne, 1, eps)?;
        g.gemv(&self.output, p("h"), g.ptr(&self.logits))?;
        let logits = g.stream.clone_dtoh(&self.logits)?;
        self.pos += 1;
        self.cpu_time += cpu_time;
        Ok(logits)
    }
}
