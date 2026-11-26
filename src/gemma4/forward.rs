//! Single-token CPU reference forward for Gemma 4 26B-A4B (mirrors llama.cpp gemma4.cpp). Matvecs use the GPU when enabled.
use super::Gemma4Config;
use crate::deepseek::mat::{Mat, matvec_many, matvec_rows, matvec_rows_many, matvec_rows_sets, vec_f32};
use crate::deepseek::ops::{rms_norm, rms_norm_w};
use crate::gguf::Gguf;
use anyhow::Result;
use rayon::prelude::*;

struct Layer<'a> {
    attn_norm: Vec<f32>,
    wq: Mat<'a>,
    wk: Mat<'a>,
    /// full-attention layers have no V projection: V = K before its norm
    wv: Option<Mat<'a>>,
    wo: Mat<'a>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    post_attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    gate: Mat<'a>,
    up: Mat<'a>,
    down: Mat<'a>,
    post_ffw_norm: Vec<f32>,
    post_ffw_norm_1: Option<Vec<f32>>,
    pre_ffw_norm_2: Option<Vec<f32>>,
    post_ffw_norm_2: Option<Vec<f32>>,
    router: Option<Mat<'a>>,
    router_scale: Option<Vec<f32>>,
    gate_up: Option<Mat<'a>>,
    down_exps: Option<Mat<'a>>,
    down_scale: Option<Vec<f32>>,
    out_scale: f32,
}

#[derive(Default)]
struct Cache {
    k: Vec<f32>,
    v: Vec<f32>,
}

pub struct Gemma4Model<'a> {
    pub cfg: Gemma4Config,
    tok_embd: Mat<'a>,
    out_norm: Vec<f32>,
    /// proportional-rope frequency factors of the full-attention layers
    rope_freqs: Vec<f32>,
    layers: Vec<Layer<'a>>,
    cache: Vec<Cache>,
    pub toks: Vec<u32>,
    pub trace: bool,
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * x * (1.0 + 0.044715 * x * x)).tanh())
}

/// NEOX rope on one head: pairs (i, i + n_rot/2), theta = pos * base^(-2i/n_rot) / factor[i].
/// (sin, cos) of each pair's angle at `pos`: one table serves every head of a token.
fn rope_table(pos: usize, n_rot: usize, base: f32, factors: Option<&[f32]>) -> Vec<(f32, f32)> {
    (0..n_rot / 2).map(|i| {
        let mut theta = pos as f32 * base.powf(-2.0 * i as f32 / n_rot as f32);
        if let Some(f) = factors { theta /= f[i]; }
        theta.sin_cos()
    }).collect()
}

fn rope_by(x: &mut [f32], table: &[(f32, f32)]) {
    let h = table.len();
    for (i, &(s, c)) in table.iter().enumerate() {
        let (a, b) = (x[i], x[i + h]);
        x[i] = a * c - b * s;
        x[i + h] = a * s + b * c;
    }
}

pub static PROF: [std::sync::atomic::AtomicU64; 12] = [const { std::sync::atomic::AtomicU64::new(0) }; 12];
fn lap(i: usize, t: &mut std::time::Instant) { PROF[i].fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); *t = std::time::Instant::now(); }

impl<'a> Gemma4Model<'a> {
    pub fn new(g: &'a Gguf) -> Result<Self> {
        let cfg = Gemma4Config::from_gguf(g)?;
        let mut layers = vec![];
        for il in 0..cfg.n_layer {
            let m = |n: &str| Mat::load(g, &format!("blk.{il}.{n}"));
            let v = |n: &str| vec_f32(g, &format!("blk.{il}.{n}"));
            let om = |n: &str| -> Result<Option<Mat<'a>>> { Ok(if g.tensors.contains_key(&format!("blk.{il}.{n}")) { Some(m(n)?) } else { None }) };
            let ov = |n: &str| -> Result<Option<Vec<f32>>> { Ok(if g.tensors.contains_key(&format!("blk.{il}.{n}")) { Some(v(n)?) } else { None }) };
            layers.push(Layer {
                attn_norm: v("attn_norm.weight")?,
                wq: m("attn_q.weight")?,
                wk: m("attn_k.weight")?,
                wv: if g.tensors.contains_key(&format!("blk.{il}.attn_v.weight")) { Some(m("attn_v.weight")?) } else { None },
                wo: m("attn_output.weight")?,
                q_norm: v("attn_q_norm.weight")?,
                k_norm: v("attn_k_norm.weight")?,
                post_attn_norm: v("post_attention_norm.weight")?,
                ffn_norm: v("ffn_norm.weight")?,
                gate: m("ffn_gate.weight")?,
                up: m("ffn_up.weight")?,
                down: m("ffn_down.weight")?,
                post_ffw_norm: v("post_ffw_norm.weight")?,
                post_ffw_norm_1: ov("post_ffw_norm_1.weight")?,
                pre_ffw_norm_2: ov("pre_ffw_norm_2.weight")?,
                post_ffw_norm_2: ov("post_ffw_norm_2.weight")?,
                router: om("ffn_gate_inp.weight")?,
                router_scale: ov("ffn_gate_inp.scale")?,
                gate_up: om("ffn_gate_up_exps.weight")?,
                down_exps: om("ffn_down_exps.weight")?,
                down_scale: ov("ffn_down_exps.scale")?,
                out_scale: v("layer_output_scale.weight")?[0],
            });
        }
        let n = cfg.n_layer;
        Ok(Gemma4Model {
            tok_embd: Mat::load(g, "token_embd.weight")?,
            out_norm: vec_f32(g, "output_norm.weight")?,
            rope_freqs: vec_f32(g, "rope_freqs.weight")?,
            layers,
            cache: (0..n).map(|_| Cache::default()).collect(),
            toks: vec![],
            trace: std::env::var("STRATA_G4_TRACE").is_ok(),
            cfg,
        })
    }

    pub fn reset(&mut self) {
        self.cache.iter_mut().for_each(|c| { c.k.clear(); c.v.clear(); });
        self.toks.clear();
    }

    fn tr(&self, name: &str, il: i32, v: &[f32]) {
        if let Ok(dir) = std::env::var("STRATA_G4_DUMP") {
            use std::io::Write;
            let fname = if il < 0 { format!("{dir}/{name}.f32") } else { format!("{dir}/{name}-{il}.f32") };
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::OpenOptions::new().create(true).append(true).open(fname).unwrap().write_all(&bytes).unwrap();
        }
        if self.trace {
            let n = v.len();
            let f = |i: usize| format!("{:12.4}", v[i]);
            eprintln!("TRACE {name}-{il} [{} {} {} ... {} {} {}] sum={:.6}", f(0), f(1), f(2), f(n - 3), f(n - 2), f(n - 1), v.iter().sum::<f32>());
        }
    }

    /// Attention of the `nt` tokens at positions `p0..` (the cache holds `p0` earlier rows). `h`: a normed row a
    /// token. Returns the output projection, a row a token. One token: the matvecs of decode; more: one pass of
    /// each matrix over all rows.
    fn attn(&mut self, il: usize, h: &[f32], p0: usize, nt: usize) -> Vec<f32> {
        let cfg = &self.cfg;
        let (nh, nkv, hd, eps) = (cfg.n_head, cfg.n_kv[il], cfg.hd(il), cfg.eps);
        let swa = cfg.swa[il];
        let l = &self.layers[il];
        let (base, factors) = if swa { (cfg.base_swa, None) } else { (cfg.base_full, Some(&self.rope_freqs[..])) };
        let ms: Vec<_> = [Some(l.wq), Some(l.wk), l.wv].into_iter().flatten().collect();
        let mut tp = std::time::Instant::now();
        let qkv = if nt == 1 { matvec_many(&ms, h, 0) } else { matvec_rows_sets(&ms.iter().map(|&m| (m, 0)).collect::<Vec<_>>(), h, nt) };
        lap(6, &mut tp);
        let (qw, kw) = (nh * hd, nkv * hd);
        let mut q = qkv[..nt * qw].to_vec();
        let mut kraw = qkv[nt * qw..nt * (qw + kw)].to_vec();
        let mut vraw = if l.wv.is_some() { qkv[nt * (qw + kw)..].to_vec() } else { kraw.clone() };
        // per head: RMS norm (weighted for Q and K), then rope at the token's position for Q and K
        let heads = |x: &mut [f32], per: usize, w: Option<&[f32]>, rot: bool| {
            x.par_chunks_mut(per).enumerate().for_each(|(t, xt)| {
                let mut tmp = vec![0f32; hd];
                let table = if rot { rope_table(p0 + t, hd, base, factors) } else { vec![] };
                for hx in xt.chunks_exact_mut(hd) {
                    match w { Some(w) => rms_norm_w(hx, w, eps, &mut tmp), None => rms_norm(hx, eps, &mut tmp) }
                    hx.copy_from_slice(&tmp);
                    if rot { rope_by(hx, &table); }
                }
            });
        };
        heads(&mut q, qw, Some(&l.q_norm), true);
        heads(&mut kraw, kw, Some(&l.k_norm), true);
        heads(&mut vraw, kw, None, false);
        lap(11, &mut tp);
        let c = &mut self.cache[il];
        c.k.extend_from_slice(&kraw);
        c.v.extend_from_slice(&vraw);
        let rep = nh / nkv;
        let (ck, cv, window) = (&c.k, &c.v, cfg.window);
        let mut o = vec![0f32; nt * qw];
        // a key head's `rep` query heads share one pass over its K rows and one over its V rows
        o.par_chunks_mut(rep * hd).enumerate().for_each(|(g, og)| {
            let (pos, kvh) = (p0 + g / nkv, g % nkv);
            let lo = if swa { (pos + 1).saturating_sub(window) } else { 0 };
            let np = pos + 1 - lo;
            let at = |p: usize| (p * nkv + kvh) * hd..(p * nkv + kvh + 1) * hd;
            let mut sc = vec![0f32; rep * np];
            for i in 0..np {
                let kp = &ck[at(lo + i)];
                for r in 0..rep { sc[r * np + i] = crate::weights::dot(&q[(g * rep + r) * hd..(g * rep + r + 1) * hd], kp); }
            }
            for s in sc.chunks_exact_mut(np) {
                let mx = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                s.iter_mut().for_each(|s| { *s = (*s - mx).exp(); sum += *s; });
                s.iter_mut().for_each(|s| *s /= sum);
            }
            for i in 0..np {
                let vp = &cv[at(lo + i)];
                for (r, oh) in og.chunks_exact_mut(hd).enumerate() {
                    let w = sc[r * np + i];
                    oh.iter_mut().zip(vp).for_each(|(o, v)| *o += w * v);
                }
            }
        });
        lap(7, &mut tp);
        if nt == 1 {
            self.tr("attn_core", il as i32, &o);
            let mut out = vec![0f32; cfg.n_embd];
            self.layers[il].wo.matvec(&o, &mut out);
            lap(8, &mut tp);
            return out;
        }
        matvec_rows(&self.layers[il].wo, &o.chunks_exact(qw).collect::<Vec<_>>())
    }

    /// Dense FFN + routed experts for one token (`x` = attention output with residual).
    fn ffn(&self, il: usize, x: &[f32]) -> Vec<f32> {
        let cfg = &self.cfg;
        let (n, eps, nff, nf) = (cfg.n_embd, cfg.eps, cfg.n_ff, cfg.n_ff_exp);
        let l = &self.layers[il];
        let mut nrm = vec![0f32; n];
        rms_norm_w(x, &l.ffn_norm, eps, &mut nrm);
        let act = |g: &[f32], w: usize, o: &mut [f32]| for j in 0..w { o[j] = gelu(g[j]) * g[w + j]; };
        if cfg.n_expert == 0 {
            let mut g = vec![0f32; nff];
            act(&matvec_many(&[l.gate, l.up], &nrm, 0), nff, &mut g);
            let mut mlp = vec![0f32; n];
            l.down.matvec(&g, &mut mlp);
            let mut out = vec![0f32; n];
            rms_norm_w(&mlp, &l.post_ffw_norm, eps, &mut out);
            self.tr("ffn_out", il as i32, &out);
            return out;
        }
        // router sees the unweighted-normed attention output scaled by 1/sqrt(n_embd) and the learned scale
        let mut tp = std::time::Instant::now();
        let mut t = vec![0f32; n];
        rms_norm(x, eps, &mut t);
        let s = 1.0 / (n as f32).sqrt();
        t.iter_mut().zip(l.router_scale.as_ref().expect("expert FFN has no router scale")).for_each(|(t, w)| *t *= s * w);
        let mut logits = vec![0f32; cfg.n_expert];
        l.router.as_ref().expect("expert FFN has no router").matvec(&t, &mut logits);
        lap(10, &mut tp);
        self.tr("ffn_moe_logits", il as i32, &logits);
        let sel = route(&logits, cfg.n_used);
        let ids: Vec<f32> = sel.iter().map(|s| s.0 as f32).collect();
        self.tr("ffn_moe_topk", il as i32, &ids);

        let mut moe_in = vec![0f32; n];
        rms_norm_w(x, l.pre_ffw_norm_2.as_ref().expect("expert FFN has no pre_ffw_norm_2"), eps, &mut moe_in);
        let mut cpu_sel = vec![];
        let mut gpu_sel = vec![];
        for (e, w) in sel {
            let (gu_m, dn_m) = (l.gate_up.as_ref().expect("expert FFN has no gate_up").expert(e, cfg.n_expert), l.down_exps.as_ref().expect("expert FFN has no down_exps").expert(e, cfg.n_expert));
            let sc = w * l.down_scale.as_ref().expect("expert FFN has no down_scale")[e];
            if gu_m.gpu.is_some() || dn_m.gpu.is_some() { gpu_sel.push((gu_m, dn_m, sc)); } else { cpu_sel.push((gu_m, dn_m, sc)); }
        }
        // The dense matrices and the experts on the card share two round trips (gate and up, then down), and the
        // experts that have no room on the card run on the CPU in the same time.
        let on_card = || {
            let mut tp = std::time::Instant::now();
            let (k, w) = (gpu_sel.len(), nff.max(nf));
            let ms: Vec<Mat> = [l.gate, l.up].into_iter().chain(gpu_sel.iter().map(|g| g.0)).collect();
            let gu = matvec_many(&ms, &[nrm.repeat(2), moe_in.repeat(k)].concat(), n);
            let mut a = vec![0f32; (1 + k) * w];
            act(&gu, nff, &mut a);
            for i in 0..k { act(&gu[2 * nff + i * 2 * nf..], nf, &mut a[(1 + i) * w..]); }
            let ms: Vec<Mat> = std::iter::once(l.down).chain(gpu_sel.iter().map(|g| g.1)).collect();
            let d = matvec_many(&ms, &a, w);
            lap(4, &mut tp);
            d
        };
        let on_cpu = || {
            let mut tp = std::time::Instant::now();
            let k = cpu_sel.len();
            if k == 0 { return vec![]; }
            for (g, d, _) in &cpu_sel { g.prefetch_expert(0, 1); d.prefetch_expert(0, 1); }
            use crate::cpu_moe::{int_dot, quantize_q8k};
            let (Some(dgu), Some(ddn)) = (int_dot(cpu_sel[0].0.ty), int_dot(cpu_sel[0].1.ty)) else {
                // An expert type with no integer kernel (BF16 experts of a UD-Q8_K_XL file): each expert through
                // Mat::matvec, which takes every type.
                return cpu_sel.iter().flat_map(|(g, d, _)| {
                    let mut gu = vec![0f32; 2 * nf];
                    g.matvec(&moe_in, &mut gu);
                    let mut a = vec![0f32; nf];
                    act(&gu, nf, &mut a);
                    let mut o = vec![0f32; n];
                    d.matvec(&a, &mut o);
                    o
                }).collect();
            };
            let xq = quantize_q8k(&moe_in);
            let mut gu = vec![0f32; k * 2 * nf];
            gu.par_chunks_mut(16).enumerate().for_each(|(c, ys)| for (j, v) in ys.iter_mut().enumerate() {
                let i = c * 16 + j;
                let (m, r) = (&cpu_sel[i / (2 * nf)].0, i % (2 * nf));
                let rb = m.data.len() / m.rows;
                *v = dgu(&m.data[r * rb..(r + 1) * rb], &xq);
            });
            let aq: Vec<_> = gu.chunks_exact(2 * nf).map(|g| {
                let mut a = vec![0f32; nf];
                act(g, nf, &mut a);
                quantize_q8k(&a)
            }).collect();
            let mut dv = vec![0f32; k * n];
            dv.par_chunks_mut(16).enumerate().for_each(|(c, ys)| for (j, v) in ys.iter_mut().enumerate() {
                let i = c * 16 + j;
                let (m, r) = (&cpu_sel[i / n].1, i % n);
                let rb = m.data.len() / m.rows;
                *v = ddn(&m.data[r * rb..(r + 1) * rb], &aq[i / n]);
            });
            lap(5, &mut tp);
            dv
        };
        let (d, dv) = rayon::join(on_card, on_cpu);
        let mut mlp_n = vec![0f32; n];
        rms_norm_w(&d[..n], l.post_ffw_norm_1.as_ref().expect("expert FFN has no post_ffw_norm_1"), eps, &mut mlp_n);
        self.tr("ffn_mlp", il as i32, &mlp_n);
        let mut moe = vec![0f32; n];
        for (dn, (_, _, sc)) in d[n..].chunks_exact(n).zip(&gpu_sel).chain(dv.chunks_exact(n).zip(&cpu_sel)) {
            moe.iter_mut().zip(dn).for_each(|(m, d)| *m += sc * d);
        }
        let mut moe_n = vec![0f32; n];
        rms_norm_w(&moe, l.post_ffw_norm_2.as_ref().expect("expert FFN has no post_ffw_norm_2"), eps, &mut moe_n);
        self.tr("ffn_moe", il as i32, &moe_n);
        let sum: Vec<f32> = mlp_n.iter().zip(&moe_n).map(|(a, b)| a + b).collect();
        let mut out = vec![0f32; n];
        rms_norm_w(&sum, &l.post_ffw_norm, eps, &mut out);
        out
    }

    /// `ffn` for the `nt` tokens of a prompt chunk (`x`: a row a token). Each matrix runs once on all rows, and
    /// each routed expert once on the rows of the tokens that picked it.
    fn ffn_chunk(&self, il: usize, x: &[f32], nt: usize) -> Vec<f32> {
        let cfg = &self.cfg;
        let (n, eps, nff) = (cfg.n_embd, cfg.eps, cfg.n_ff);
        let l = &self.layers[il];
        // `f(row, out)` on every row
        let rows = |x: &[f32], f: &(dyn Fn(&[f32], &mut [f32]) + Sync)| {
            let mut o = vec![0f32; x.len()];
            o.par_chunks_mut(n).zip(x.par_chunks(n)).for_each(|(o, x)| f(x, o));
            o
        };
        fn split(x: &[f32], w: usize) -> Vec<&[f32]> { x.chunks_exact(w).collect() }
        let nrm = rows(x, &|x, o| rms_norm_w(x, &l.ffn_norm, eps, o));
        let gu = matvec_rows_sets(&[(l.gate, 0), (l.up, 0)], &nrm, nt);
        let (g, u) = gu.split_at(nt * nff);
        let act: Vec<f32> = g.par_iter().zip(u).map(|(g, u)| gelu(*g) * u).collect();
        let mlp = matvec_rows(&l.down, &act.chunks_exact(nff).collect::<Vec<_>>());
        if cfg.n_expert == 0 { return rows(&mlp, &|x, o| rms_norm_w(x, &l.post_ffw_norm, eps, o)); }
        let mlp_n = rows(&mlp, &|x, o| rms_norm_w(x, l.post_ffw_norm_1.as_ref().expect("expert FFN has no post_ffw_norm_1"), eps, o));

        let (s, rs) = (1.0 / (n as f32).sqrt(), l.router_scale.as_ref().expect("expert FFN has no router scale"));
        let rin = rows(x, &|x, o| { rms_norm(x, eps, o); o.iter_mut().zip(rs).for_each(|(t, w)| *t *= s * w); });
        let logits = matvec_rows(l.router.as_ref().expect("expert FFN has no router"), &split(&rin, n));
        let sels: Vec<Vec<(usize, f32)>> = logits.chunks_exact(cfg.n_expert).map(|r| route(r, cfg.n_used)).collect();
        let moe_in = rows(x, &|x, o| rms_norm_w(x, l.pre_ffw_norm_2.as_ref().expect("expert FFN has no pre_ffw_norm_2"), eps, o));

        // per expert the (token, slot) pairs that picked it; experts with a VRAM copy share GPU round trips
        let (nf, k) = (cfg.n_ff_exp, cfg.n_used);
        let mut by_e: std::collections::BTreeMap<usize, Vec<(usize, usize)>> = Default::default();
        for (t, sel) in sels.iter().enumerate() { for (j, &(e, _)) in sel.iter().enumerate() { by_e.entry(e).or_default().push((t, j)); } }
        let (gate_up, down_exps) = (l.gate_up.as_ref().expect("expert FFN has no gate_up"), l.down_exps.as_ref().expect("expert FFN has no down_exps"));
        let mut d = vec![0f32; nt * k * n];
        let (mut on_gpu, mut on_cpu) = (vec![], vec![]);
        for (e, uses) in by_e {
            let (gm, dm) = (gate_up.expert(e, cfg.n_expert), down_exps.expert(e, cfg.n_expert));
            if gm.gpu.is_some() && dm.gpu.is_some() { on_gpu.push((gm, dm, uses)); } else { on_cpu.push((gm, dm, uses)); }
        }
        let xin = |uses: &[(usize, usize)]| -> Vec<f32> { uses.iter().flat_map(|&(t, _)| moe_in[t * n..(t + 1) * n].iter().copied()).collect() };
        let acts = |y: &[f32]| -> Vec<f32> { y.par_chunks_exact(2 * nf).flat_map_iter(|g| (0..nf).map(move |j| gelu(g[j]) * g[nf + j])).collect() };
        let mut put = |uses: &[(usize, usize)], y: &[f32]| for (i, &(t, j)) in uses.iter().enumerate() {
            d[(t * k + j) * n..(t * k + j + 1) * n].copy_from_slice(&y[i * n..(i + 1) * n]);
        };
        if !on_gpu.is_empty() {
            let x: Vec<f32> = on_gpu.iter().flat_map(|e| xin(&e.2)).collect();
            let a = acts(&matvec_rows_many(&on_gpu.iter().map(|e| (e.0, e.2.len())).collect::<Vec<_>>(), &x));
            let y = matvec_rows_many(&on_gpu.iter().map(|e| (e.1, e.2.len())).collect::<Vec<_>>(), &a);
            let mut at = 0;
            for (_, _, uses) in &on_gpu { put(uses, &y[at..at + uses.len() * n]); at += uses.len() * n; }
        }
        for (gm, dm, uses) in &on_cpu {
            let a = acts(&matvec_rows(gm, &split(&xin(uses), n)));
            put(uses, &matvec_rows(dm, &split(&a, nf)));
        }
        let ds = l.down_scale.as_ref().expect("expert FFN has no down_scale");
        let (n1, n2) = (l.post_ffw_norm_2.as_ref().expect("expert FFN has no post_ffw_norm_2"), &l.post_ffw_norm);
        let mut out = vec![0f32; nt * n];
        out.par_chunks_mut(n).enumerate().for_each(|(t, o)| {
            let mut moe = vec![0f32; n];
            for (j, &(e, w)) in sels[t].iter().enumerate() {
                moe.iter_mut().zip(&d[(t * k + j) * n..(t * k + j + 1) * n]).for_each(|(m, d)| *m += w * ds[e] * d);
            }
            let mut sum = vec![0f32; n];
            rms_norm_w(&moe, n1, eps, &mut sum);
            sum.iter_mut().zip(&mlp_n[t * n..(t + 1) * n]).for_each(|(s, m)| *s += m);
            rms_norm_w(&sum, n2, eps, o);
        });
        out
    }

    /// Layer-major pass over a chunk of prompt tokens (no logits). `x`: the input row of each token.
    fn prefill_chunk(&mut self, toks: &[u32], mut x: Vec<f32>) {
        let (n, nt, p0, eps) = (self.cfg.n_embd, toks.len(), self.toks.len(), self.cfg.eps);
        self.toks.extend_from_slice(toks);
        for il in 0..self.cfg.n_layer {
            let mut nrm = vec![0f32; nt * n];
            nrm.par_chunks_mut(n).zip(x.par_chunks(n)).for_each(|(o, x)| rms_norm_w(x, &self.layers[il].attn_norm, eps, o));
            let mut t0 = std::time::Instant::now();
            let a = self.attn(il, &nrm, p0, nt);
            lap(0, &mut t0);
            let l = &self.layers[il];
            let mut an = vec![0f32; nt * n];
            an.par_chunks_mut(n).zip(a.par_chunks(n)).zip(x.par_chunks(n)).for_each(|((o, a), x)| {
                rms_norm_w(a, &l.post_attn_norm, eps, o);
                o.iter_mut().zip(x).for_each(|(o, x)| *o += x);
            });
            let f = self.ffn_chunk(il, &an, nt);
            lap(1, &mut t0);
            let os = self.layers[il].out_scale;
            x = an.iter().zip(&f).map(|(a, f)| (a + f) * os).collect();
        }
    }

    /// Feed a prompt and return the logits of its last token. An entry with an embedding (a vision projector
    /// row) uses it as its input, unscaled as llama.cpp does for embedding batches. The tokens before the last go
    /// in chunks of `STRATA_G4_CHUNK` (default 256; 1: one token at a time, which the traces use).
    pub fn prefill_seq(&mut self, seq: &[(u32, Option<&[f32]>)]) -> Vec<f32> {
        let Some((&(id, emb), head)) = seq.split_last() else { return vec![] };
        let chunk: usize = std::env::var("STRATA_G4_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
        let chunk = if self.trace || std::env::var_os("STRATA_G4_DUMP").is_some() { 1 } else { chunk.max(1) };
        for c in head.chunks(chunk) {
            if c.len() == 1 {
                match c[0] { (id, Some(e)) => self.step_embed_opt(id, e, false), (id, None) => self.step_opt(id, false) };
                continue;
            }
            let x: Vec<f32> = c.iter().flat_map(|&(id, e)| match e { Some(e) => e.to_vec(), None => self.embed(id) }).collect();
            self.prefill_chunk(&c.iter().map(|c| c.0).collect::<Vec<_>>(), x);
            crate::term::prefilled(c.len());
        }
        match emb { Some(e) => self.step_embed_opt(id, e, true), None => self.step_opt(id, true) }
    }

    pub fn step(&mut self, tok: u32) -> Vec<f32> {
        self.step_opt(tok, true)
    }

    pub fn prefill(&mut self, toks: &[u32]) -> Vec<f32> {
        self.prefill_seq(&toks.iter().map(|&t| (t, None)).collect::<Vec<_>>())
    }

    /// Feed one token whose input embedding is supplied by the caller (the vision projector). The token
    /// id only records the sequence; the embedding is used as-is, as llama.cpp does for embedding batches.
    pub fn step_embed(&mut self, tok: u32, emb: &[f32]) -> Vec<f32> { self.step_embed_opt(tok, emb, true) }

    /// `step_embed` without the output head unless `want` is set (prompt ingestion).
    pub fn step_embed_opt(&mut self, tok: u32, emb: &[f32], want: bool) -> Vec<f32> {
        assert_eq!(emb.len(), self.cfg.n_embd, "step_embed: embedding width");
        self.forward(tok, emb.to_vec(), want)
    }

    /// `step` without the output head unless `want` is set (prompt ingestion).
    pub fn step_opt_pub(&mut self, tok: u32, want: bool) -> Vec<f32> { self.step_opt(tok, want) }

    /// The input row of a token: its embedding times sqrt(n_embd).
    fn embed(&self, tok: u32) -> Vec<f32> {
        let n = self.cfg.n_embd;
        let mut x = vec![0f32; n];
        self.tok_embd.row_into(tok as usize, &mut x);
        let sc = (n as f32).sqrt();
        x.iter_mut().for_each(|v| *v *= sc);
        x
    }

    fn step_opt(&mut self, tok: u32, want: bool) -> Vec<f32> {
        let x = self.embed(tok);
        self.forward(tok, x, want)
    }

    /// One token, on a thread of the pool, where the fork in `ffn` costs no hand-over of the card's half.
    fn forward(&mut self, tok: u32, x: Vec<f32>, want: bool) -> Vec<f32> {
        rayon::scope(|_| self.forward_in(tok, x, want))
    }

    fn forward_in(&mut self, tok: u32, mut x: Vec<f32>, want: bool) -> Vec<f32> {
        let n = self.cfg.n_embd;
        let pos = self.toks.len();
        self.toks.push(tok);
        self.tr("inp_scaled", -1, &x);
        let mut nrm = vec![0f32; n];
        for il in 0..self.cfg.n_layer {
            rms_norm_w(&x, &self.layers[il].attn_norm, self.cfg.eps, &mut nrm);
            let mut t0 = std::time::Instant::now();
            let a = self.attn(il, &nrm, pos, 1);
            lap(0, &mut t0);
            let mut an = vec![0f32; n];
            rms_norm_w(&a, &self.layers[il].post_attn_norm, self.cfg.eps, &mut an);
            an.iter_mut().zip(&x).for_each(|(a, x)| *a += x);
            self.tr("attn_out", il as i32, &an);
            let f = self.ffn(il, &an);
            lap(1, &mut t0);
            let os = self.layers[il].out_scale;
            x = an.iter().zip(&f).map(|(a, f)| (a + f) * os).collect();
            self.tr("l_out", il as i32, &x);
        }
        if !want { return vec![]; }
        let mut t0 = std::time::Instant::now();
        rms_norm_w(&x, &self.out_norm, self.cfg.eps, &mut nrm);
        self.tr("result_norm", -1, &nrm);
        let mut logits = vec![0f32; self.cfg.n_vocab];
        self.tok_embd.matvec(&nrm, &mut logits);
        let cap = self.cfg.softcap;
        if cap > 0.0 { logits.iter_mut().for_each(|l| *l = cap * (*l / cap).tanh()); }
        lap(2, &mut t0);
        self.tr("result_output", -1, &logits);
        logits
    }
}

/// softmax over all experts, top `k` by probability (ties to the lower index), weights = their probs normalised by the sum (clamped like ggml).
pub fn route(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = logits.iter().map(|&l| (l - mx).exp()).collect();
    let s: f32 = p.iter().sum();
    p.iter_mut().for_each(|v| *v /= s);
    let mut idx: Vec<usize> = (0..p.len()).collect();
    idx.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap().then(a.cmp(&b)));
    idx.truncate(k);
    let sum = idx.iter().map(|&i| p[i]).sum::<f32>().max(6.103_515_6e-5);
    idx.into_iter().map(|i| (i, p[i] / sum)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn route_normalises_top_k() {
        let r = route(&[0.0, 2.0, 1.0, 2.0], 2);
        assert_eq!(r.iter().map(|x| x.0).collect::<Vec<_>>(), vec![1, 3]);
        assert!((r.iter().map(|x| x.1).sum::<f32>() - 1.0).abs() < 1e-6);
    }
    /// A prompt fed as one chunk gives the bits of the token-by-token pass on the CPU (the dense tiny model and the
    /// 26B with experts; a model that is not on this machine is skipped).
    #[test]
    fn chunked_prefill_matches_token_by_token() {
        for path in ["/home/vasylt/models/gemma4-uv-0.8b/gemma-4-unified-0.8B-tiny.Q2_K.gguf", "/home/vasylt/models/gemma4-26b/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"] {
            let Ok(g) = Gguf::open(std::path::Path::new(path)) else { continue };
            let toks: Vec<u32> = (0..12).map(|i| 2 + i * 977).collect();
            let (mut a, mut b) = (Gemma4Model::new(&g).unwrap(), Gemma4Model::new(&g).unwrap());
            for &t in &toks[..11] { a.step_opt(t, false); }
            let x = toks[..11].iter().flat_map(|&t| b.embed(t)).collect();
            b.prefill_chunk(&toks[..11], x);
            assert_eq!(a.step(toks[11]), b.step(toks[11]), "{path}");
        }
    }
    #[test]
    fn rope_pos0_is_identity() {
        let mut x = [1.0, 2.0, 3.0, 4.0];
        rope_by(&mut x, &rope_table(0, 4, 10000.0, None));
        assert_eq!(x, [1.0, 2.0, 3.0, 4.0]);
    }
}
