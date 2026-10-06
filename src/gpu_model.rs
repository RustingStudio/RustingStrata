//! GPU decode forward for `qwen35` / `qwen35moe`. Routed experts resident in the expert cache run on the GPU; the
//! rest run on the CPU (`cpu_moe`) while the GPU computes hits and the shared expert. Mirrors `model.rs`.

use crate::cpu_worker::CpuWorker;
use crate::cuda::{Gpu, GpuTensor};
use crate::expert_cache::ExpertCache;
use crate::model::Model;
use anyhow::bail;
use cudarc::driver::{CudaGraph, CudaSlice, sys};
use std::collections::HashMap;
use std::time::Duration;

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
    Moe { router: GpuTensor, shared: Option<Shared> },
}

struct Layer { attn_norm: GpuTensor, post_norm: GpuTensor, mixer: Mixer, ffn: Ffn }

pub struct GpuModel<'a> {
    m: &'a Model<'a>,
    pub gpu: Gpu,
    layers: Vec<Layer>,
    output_norm: GpuTensor,
    output: GpuTensor,
    /// Scratch activations by name.
    bufs: HashMap<&'static str, CudaSlice<f32>>,
    x: CudaSlice<f32>,
    worker: Option<CpuWorker>,
    logits: CudaSlice<f32>,
    ids: CudaSlice<i32>,
    slots: CudaSlice<i32>,
    /// [position, CPU worker seq] for the current token, read by kernels so the graph can replay.
    step_buf: CudaSlice<i32>,
    /// The whole token forward, captured on the first step (None until then, or with STRATA_NO_GRAPH set).
    graph: Option<CudaGraph>,
    ctx: usize,
    pub pos: usize,
    pub cache: Option<ExpertCache>,
    /// Swap experts every this many tokens (0 = fixed placement).
    pub adapt_every: usize,
    pub hits: usize,
    pub misses: usize,
    /// Time the CPU worker spent computing missed experts (overlaps GPU work).
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
                Ffn::Moe { router: lt(il, "ffn_gate_inp.weight")?, shared }
            } else {
                Ffn::Dense { gate: lt(il, "ffn_gate.weight")?, up: lt(il, "ffn_up.weight")?, down: lt(il, "ffn_down.weight")? }
            };
            layers.push(Layer { attn_norm: lt(il, "attn_norm.weight")?, post_norm: lt(il, "post_attention_norm.weight")?, mixer, ffn });
        }
        let output_norm = up("output_norm.weight")?;
        let output = up("output.weight").or_else(|_| up("token_embd.weight"))?;

        let k = c.n_expert_used.max(1);
        let ff = (k * c.n_ff_exp).max(c.n_ff);
        let sizes: [(&'static str, usize); 24] = [
            ("h", c.n_embd), ("mix", c.n_embd), ("qkv", channels), ("z", c.ssm_d_inner), ("beta", nv), ("g", nv),
            ("y", channels), ("o", c.ssm_d_inner), ("qg", c.n_head * 2 * c.head_dim), ("q", c.n_head * c.head_dim),
            ("kraw", kv), ("k", kv), ("v", kv), ("att", c.n_head * c.head_dim), ("rl", c.n_expert), ("wts", k),
            ("ge", ff), ("ue", ff), ("ye", k * c.n_embd), ("shg", c.n_ff_shexp), ("shu", c.n_ff_shexp),
            ("shy", c.n_embd), ("sg", 1), ("wh", k),
        ];
        let bufs = sizes.into_iter().map(|(name, n)| Ok((name, gpu.zeros::<f32>(n)?))).collect::<anyhow::Result<HashMap<_, _>>>()?;
        let (x, logits, ids) = (gpu.zeros(c.n_embd)?, gpu.zeros(c.n_vocab)?, gpu.zeros::<i32>(k)?);
        let (slots, step_buf) = (gpu.zeros::<i32>(k)?, gpu.zeros::<i32>(2)?);
        let (free1, _) = gpu.ctx.mem_get_info()?;

        let cache = if c.is_moe() {
            let budget = free1.saturating_sub(vram_margin);
            Some(ExpertCache::new(m, &gpu, budget)?)
        } else { None };
        let worker = if c.is_moe() { Some(CpuWorker::new(m, &gpu.ctx, (0..c.n_layer).collect())?) } else { None };
        let mb = |b: usize| b as f64 / 1048576.0;
        eprintln!(
            "plan: weights+state+scratch {:.0} MB (KV {:.0} MB for {ctx} tokens), expert slots {:.0} MB, resident experts {}/{}",
            mb(free0 - free1), mb(2 * 2 * ctx * kv * (c.n_layer / c.full_attention_interval)),
            mb(cache.as_ref().map_or(0, |x| x.slot_bytes_total())),
            cache.as_ref().map_or(0, |x| x.resident_count()), cache.as_ref().map_or(0, |x| x.total_experts()),
        );
        Ok(GpuModel { m, gpu, layers, output_norm, output, bufs, x, worker, logits, ids, slots, step_buf, graph: None, ctx, pos: 0, cache, adapt_every: 32, hits: 0, misses: 0, cpu_time: Duration::ZERO })
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
        if self.pos >= self.ctx { bail!("context full ({} tokens)", self.ctx); }
        let seq = self.worker.as_mut().map_or(0, |w| w.begin());
        let g = &self.gpu;
        g.stream.memcpy_htod(&self.m.embed(token), &mut self.x)?;
        g.stream.memcpy_htod(&[self.pos as i32, seq as i32], &mut self.step_buf)?;
        match &self.graph {
            Some(graph) => graph.launch()?,
            None if std::env::var_os("STRATA_NO_GRAPH").is_some() => self.forward()?,
            None => {
                g.stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                let r = self.forward();
                let graph = g.stream.end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)?;
                r?;
                let graph = graph.ok_or_else(|| anyhow::anyhow!("empty CUDA graph"))?;
                graph.launch()?;
                self.graph = Some(graph);
            }
        }
        let logits = self.gpu.stream.clone_dtoh(&self.logits)?;
        self.after_forward(logits)
    }

    /// Enqueue (or record into a graph) the token forward; per-token values come from `step_buf`.
    fn forward(&self) -> anyhow::Result<()> {
        let c = &self.m.cfg;
        let (g, eps, ne, step) = (&self.gpu, c.eps, c.n_embd, self.gpu.ptr(&self.step_buf));
        let p = |n: &str| g.ptr(&self.bufs[n]);
        let x = g.ptr(&self.x);
        let (dk, nk, nv) = (c.ssm_d_state, c.ssm_n_group, c.ssm_dt_rank);
        let dv = c.ssm_d_inner / nv;
        let (hd, nh, nkv) = (c.head_dim, c.n_head, c.n_head_kv);
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
                    g.rope_neox(p("q"), hd, nh, c.n_rot, step, c.rope_base)?;
                    g.gemv(k, p("h"), p("kraw"))?;
                    g.gemv(v, p("h"), p("v"))?;
                    g.rms_norm(p("kraw"), hd, kn.ptr, p("k"), hd, hd, nkv, eps)?;
                    g.rope_neox(p("k"), hd, nkv, c.n_rot, step, c.rope_base)?;
                    g.kv_store(p("k"), p("v"), g.ptr(kc), g.ptr(vc), nkv * hd, step)?;
                    g.attn_decode(p("q"), g.ptr(kc), g.ptr(vc), p("qg"), p("att"), step, nh, nkv, hd, 1.0 / (hd as f32).sqrt())?;
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
                Ffn::Moe { router, shared } => {
                    let k = c.n_expert_used;
                    let ids = g.ptr(&self.ids);
                    g.gemv(router, p("h"), p("rl"))?;
                    g.router_topk(p("rl"), c.n_expert, k, c.expert_weights_scale, ids, p("wts"))?;
                    let cache = self.cache.as_ref().unwrap();
                    let lx = cache.layer(il).unwrap();
                    let (base, stride) = cache.base(g, lx);
                    let [mg, mu, md] = lx.mats;
                    let slots = g.ptr(&self.slots);
                    let w = self.worker.as_ref().unwrap();
                    let mb = w.dev + (4 * il * w.layout.stride) as u64;
                    g.map_hits(ids, g.ptr(&lx.table_dev), slots, p("wts"), p("wh"), k)?;
                    g.moe_publish(ids, p("wts"), slots, p("h"), mb, k, ne, w.layout.h, step)?;
                    g.moe_gemv(mg.ty, mg.rows, mg.cols, base + mg.off as u64, stride, slots, k, p("h"), 0, p("ge"))?;
                    g.moe_gemv(mu.ty, mu.rows, mu.cols, base + mu.off as u64, stride, slots, k, p("h"), 0, p("ue"))?;
                    g.swiglu(p("ge"), p("ue"), k * c.n_ff_exp)?;
                    g.moe_gemv(md.ty, md.rows, md.cols, base + md.off as u64, stride, slots, k, p("ge"), c.n_ff_exp, p("ye"))?;
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
                    g.moe_combine(x, p("ye"), p("wh"), k, sh, sg, mb, w.layout.y, step, slots, ne)?;
                }
            }
        }
        g.rms_norm(x, 0, self.output_norm.ptr, p("h"), 0, ne, 1, eps)?;
        g.gemv(&self.output, p("h"), g.ptr(&self.logits))?;
        Ok(())
    }

    /// Advance the position and update expert statistics and placement after a token.
    fn after_forward(&mut self, logits: Vec<f32>) -> anyhow::Result<Vec<f32>> {
        self.pos += 1;
        if let (Some(cache), Some(w)) = (self.cache.as_mut(), self.worker.as_ref()) {
            for il in 0..self.layers.len() {
                if cache.layer(il).is_none() { continue; }
                let r = w.routing(il);
                let miss = r.iter().filter(|x| x.2 < 0).count();
                self.misses += miss;
                self.hits += r.len() - miss;
                cache.record(il, &r.iter().map(|x| x.0).collect::<Vec<_>>());
            }
            self.cpu_time = w.busy();
            if self.adapt_every > 0 && self.pos % self.adapt_every == 0 { cache.adapt(self.m, &self.gpu, 16)?; }
        }
        Ok(logits)
    }
}
