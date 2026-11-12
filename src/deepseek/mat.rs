//! Quantised matrix view over the GGUF mmap with an f32 matvec (slow reference path).
use crate::gguf::{Gguf, TensorInfo};
use crate::quant::{GgmlType, dequant_row};
use rayon::prelude::*;
use std::sync::{Mutex, OnceLock};

/// Optional GPU offload of the non-expert matvecs (STRATA_DS_GPU=1). Weights stay resident in VRAM; the activation is
/// copied up and the result down per call, so everything else in the forward stays the CPU reference code.
/// STRATA_DS_BACKEND (cuda or vulkan, default cuda) and STRATA_DS_DEVICES (such as 0,1) pick the GPUs; with several,
/// the weights fill them in order and the expert budget is divided in proportion to free VRAM.
mod dev {
    use super::*;
    use crate::backend::{Backend, Buf, GpuTensor};
    use crate::weights::Tensor;
    /// Scratch slot for experts that missed the VRAM cache: copied up per use, so the CPU never dequantises them.
    pub const STAGE: usize = 16 << 20;
    pub const STAGED: u32 = u32::MAX;
    /// A tensor id holds its GPU in the top 8 bits (so STAGED, GPU 255, is never a real id) and its index there below.
    const SHIFT: u32 = 24;
    pub fn gpu_of(id: u32) -> usize { (id >> SHIFT) as usize }
    struct Part { be: Box<dyn Backend>, ts: Vec<GpuTensor>, x: Buf, y: Buf, stage: Buf, bytes: usize, budget: usize, room: usize }
    pub struct Dev { parts: Vec<Part>, ex: std::collections::HashMap<usize, usize>, exs: Vec<(u32, usize, u64, f32)>, now: u64, cur: usize, lookups: u64 }
    static DEV: OnceLock<Option<Mutex<Dev>>> = OnceLock::new();
    pub fn get() -> Option<&'static Mutex<Dev>> {
        DEV.get_or_init(|| {
            if std::env::var_os("STRATA_DS_GPU").is_none() { return None; }
            let kind = std::env::var("STRATA_DS_BACKEND").ok();
            let ids: Vec<usize> = std::env::var("STRATA_DS_DEVICES").ok()
                .map(|v| v.split(',').filter_map(|d| d.trim().parse().ok()).collect()).filter(|v: &Vec<usize>| !v.is_empty()).unwrap_or(vec![0]);
            // `STRATA_DS_GPU_EXPERTS` is the absolute weight budget in GB (plain weights and expert cache). Unset:
            // `STRATA_DS_GPU_FRACTION` of free VRAM (80% by default, 90% for GLM), which keeps the old defaults on a
            // 12 GB card and uses most of an 80 GB one.
            let gb: Option<f64> = std::env::var("STRATA_DS_GPU_EXPERTS").ok().and_then(|v| v.parse().ok());
            let frac: f64 = std::env::var("STRATA_DS_GPU_FRACTION").ok().and_then(|v| v.parse().ok()).unwrap_or(0.8);
            let mut bes = vec![];
            for d in ids {
                match crate::backend::open(kind.as_deref().or(Some("cuda")), d) {
                    Ok(be) => bes.push(be),
                    Err(e) => { eprintln!("GPU {d}: {e:#}; running without it"); return None; }
                }
            }
            let free: Vec<usize> = bes.iter().map(|b| b.mem_info().map_or(1, |m| m.0.max(1))).collect();
            let total: usize = free.iter().sum();
            let parts = bes.into_iter().zip(&free).map(|(be, &f)| {
                // Room for a prefill chunk of activations and outputs, so row batches reuse them instead of allocating per expert.
                let (x, y, stage) = (be.alloc(4 * 5_000_000).ok()?, be.alloc(4 * 5_000_000).ok()?, be.alloc(STAGE).ok()?);
                let budget = match gb {
                    Some(g) => (g * 1e9 * f as f64 / total as f64) as usize,
                    None => (f.saturating_sub(56 << 20) as f64 * frac) as usize,
                };
                // Room for the plain weights before moving on to the next GPU: what is free, less the expert cache and a margin.
                let room = f.saturating_sub(budget + (768 << 20));
                Some(Part { be, ts: vec![], x, y, stage, bytes: 0, budget, room })
            }).collect::<Option<Vec<_>>>()?;
            Some(Mutex::new(Dev { parts, ex: Default::default(), exs: vec![], now: 0, cur: 0, lookups: 0 }))
        }).as_ref()
    }
    impl Dev {
        pub fn bytes(&self) -> usize { self.parts.iter().map(|p| p.bytes).sum() }
        pub fn info(&self) -> (&'static str, usize, usize) { (self.parts[0].be.name(), self.ex.len(), self.bytes()) }
        /// Bytes still available under the expert budget (plain weights already subtracted).
        pub fn budget_left(&self) -> usize { self.parts.iter().map(|p| p.budget.saturating_sub(p.bytes)).sum() }
        /// The GPU that takes `len` more bytes of plain weights: the current one while they fit, then the next; the last takes all.
        pub fn pick(&mut self, len: usize) -> usize {
            while self.cur + 1 < self.parts.len() && self.parts[self.cur].bytes + len > self.parts[self.cur].room { self.cur += 1; }
            self.cur
        }
        pub fn upload_on(&mut self, d: usize, data: &[u8], ty: GgmlType, rows: usize, cols: usize) -> Option<u32> {
            let p = &mut self.parts[d];
            let t = p.be.upload(&Tensor { ty, cols, rows, data }).ok()?;
            p.bytes += data.len();
            p.ts.push(t);
            Some((d as u32) << SHIFT | (p.ts.len() as u32 - 1))
        }
        pub fn upload(&mut self, data: &[u8], ty: GgmlType, rows: usize, cols: usize) -> Option<u32> {
            let d = self.pick(data.len());
            self.upload_on(d, data, ty, rows, cols)
        }
        fn tensor(&self, i: u32) -> &GpuTensor { &self.parts[gpu_of(i)].ts[(i & (1 << SHIFT) - 1) as usize] }
        /// VRAM copy of one routed expert: uploaded on first use while under a GPU's STRATA_DS_GPU_EXPERTS share (GB), then the
        /// least recently used slot of the same shape is overwritten. Slots touched in the last 24 lookups are kept (one layer's
        /// 18 experts are in flight together). No slot to take: STAGED (copied up per use).
        pub fn expert(&mut self, data: &[u8], ty: GgmlType, rows: usize, cols: usize) -> Option<u32> {
            let k = data.as_ptr() as usize;
            self.now += 1;
            self.lookups += 1;
            let now = self.now;
            if self.lookups % 4096 == 0 { for e in &mut self.exs { e.3 *= 0.5; } }
            if let Some(&p) = self.ex.get(&k) {
                self.exs[p].2 = now;
                self.exs[p].3 += 1.0;
                EX_LOOKUPS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Some(self.exs[p].0);
            }
            EX_LOOKUPS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let room = |p: &Part| (p.bytes + data.len() <= p.budget).then(|| p.budget - p.bytes);
            if let Some(d) = (0..self.parts.len()).filter_map(|d| room(&self.parts[d]).map(|r| (r, d))).max().map(|(_, d)| d) {
                let i = self.upload_on(d, data, ty, rows, cols)?;
                self.ex.insert(k, self.exs.len());
                self.exs.push((i, k, now, 1.0));
                return Some(i);
            }
            if self.parts.iter().all(|p| p.budget == 0) || data.len() > STAGE || std::env::var_os("STRATA_DS_NOEVICT").is_some() { return None; }
            let lfu = std::env::var_os("STRATA_DS_LFU").is_some();
            let victim = self.exs.iter().enumerate()
                .filter(|(_, e)| e.2 + 24 <= now && { let t = self.tensor(e.0); t.ty == ty && t.rows == rows && t.cols == cols })
                .min_by(|(_, a), (_, b)| if lfu { a.3.total_cmp(&b.3).then(a.2.cmp(&b.2)) } else { a.2.cmp(&b.2) })
                .map(|(p, _)| p);
            let Some(p) = victim else { return Some(STAGED) };
            if lfu && self.exs[p].3 > 4.0 { return Some(STAGED); }
            let (i, old, _, _) = self.exs[p];
            let (be, t) = (&self.parts[gpu_of(i)].be, self.tensor(i));
            if be.htod(data, &Buf::new(t.ptr, data.len(), ()), 0).is_err() { return Some(STAGED); }
            self.ex.remove(&old);
            self.ex.insert(k, p);
            self.exs[p] = (i, k, now, 1.0);
            Some(i)
        }
        pub fn gemv(&self, i: u32, x: &[f32], y: &mut [f32], staged: Option<(&[u8], GgmlType, usize, usize)>) {
            let p = &self.parts[if i == STAGED { 0 } else { gpu_of(i) }];
            p.be.put(x, &p.x, 0).unwrap();
            if let Some((d, ty, rows, cols)) = staged {
                p.be.htod(d, &p.stage, 0).unwrap();
                let w = GpuTensor::new(ty, rows, cols, Buf::new(p.stage.addr(), d.len(), ()));
                p.be.gemv(&w, p.x.addr(), p.y.addr()).unwrap();
            } else {
                p.be.gemv(self.tensor(i), p.x.addr(), p.y.addr()).unwrap();
            }
            let got: Vec<f32> = p.be.get(&p.y, 0, y.len()).unwrap();
            y.copy_from_slice(&got);
        }
        /// One gemv per matrix, all queued before a single readback; all matrices sit on one GPU. Job i reads
        /// x[i*xs..] (xs = 0: every job reads x) and writes its rows after job i-1's. Staged experts are uploaded
        /// once each into a temporary buffer, so a layer with both resident and staged experts still batches.
        pub fn batch_mats(&self, ms: &[Mat], x: &[f32], xs: usize) -> Option<Vec<f32>> {
            let first = self.part_of(ms.first()?)?;
            if ms.iter().any(|m| self.part_of(m) != Some(first)) { return None; }
            let p = &self.parts[first];
            let nx = if xs == 0 { x.len() } else { ms.len() * xs };
            if nx > p.x.len() / 4 { return None; }
            let xsrc = &x[..nx];
            p.be.put(xsrc, &p.x, 0).ok()?;
            let mut owned = vec![];
            let mut off = 0;
            for (i, m) in ms.iter().enumerate() {
                if off + m.rows > p.y.len() / 4 { return None; }
                let w = match m.gpu? {
                    STAGED => {
                        let b = p.be.alloc(m.data.len()).ok()?;
                        p.be.htod(m.data, &b, 0).ok()?;
                        owned.push(GpuTensor::new(m.ty, m.rows, m.cols, b));
                        owned.last().unwrap()
                    }
                    t => self.tensor(t),
                };
                p.be.gemv(w, p.x.addr() + (4 * i * xs) as u64, p.y.addr() + (4 * off) as u64).ok()?;
                off += m.rows;
            }
            let y = p.be.get(&p.y, 0, off).ok()?;
            drop(owned);
            Some(y)
        }
        fn part_of(&self, m: &Mat) -> Option<usize> {
            let i = m.gpu?;
            Some(if i == STAGED { 0 } else { gpu_of(i) })
        }
        /// W x[t] for every row of `xs` (one matrix, several activations). Uses the tensor-core GEMM when there are
        /// enough rows and the GEMV otherwise; staged weights are uploaded once for the whole batch.
        pub fn rows(&self, m: &Mat, xs: &[&[f32]]) -> Option<Vec<f32>> {
            let i = m.gpu?;
            let d = if i == STAGED { 0 } else { gpu_of(i) };
            let p = &self.parts[d];
            if xs.is_empty() { return Some(vec![]); }
            if xs.iter().any(|x| x.len() != m.cols) { return None; }
            let n = xs.len();
            let mut flat = Vec::with_capacity(n * m.cols);
            for x in xs { flat.extend_from_slice(x); }
            let mut owned = vec![];
            let w = if i == STAGED {
                let b = p.be.alloc(m.data.len()).ok()?;
                p.be.htod(m.data, &b, 0).ok()?;
                owned.push(GpuTensor::new(m.ty, m.rows, m.cols, b));
                &owned[0]
            } else { self.tensor(i) };
            // Reuse the scratch pair when the chunk fits; otherwise allocate a GEMM-sized pair for this call.
            let y = if n * m.cols <= p.x.len() / 4 && n * m.rows <= p.y.len() / 4 {
                p.be.put(&flat, &p.x, 0).ok()?;
                if n <= 4 {
                    p.be.gemvn(w, p.x.addr(), m.cols, p.y.addr(), m.rows, n).ok()?;
                } else {
                    p.be.gemm_dense(w, n, p.x.addr(), m.cols, p.y.addr(), m.rows, 0).ok()?;
                }
                p.be.get(&p.y, 0, n * m.rows).ok()?
            } else {
                let xb = p.be.zeros::<f32>(n * m.cols).ok()?;
                let yb = p.be.zeros::<f32>(n * m.rows).ok()?;
                p.be.put(&flat, &xb, 0).ok()?;
                p.be.gemm_dense(w, n, xb.addr(), m.cols, yb.addr(), m.rows, 0).ok()?;
                p.be.get(&yb, 0, n * m.rows).ok()?
            };
            drop(owned);
            Some(y)
        }
    }
}
/// Matvecs of several matrices on one activation (xs = 0) or on consecutive slices x[i*xs..] (xs > 0); outputs concatenated.
/// One GPU round trip when the matrices share a GPU (staged experts included). Otherwise each runs on its own, as
/// `Mat::matvec`.
pub fn matvec_many(ms: &[Mat], x: &[f32], xs: usize) -> Vec<f32> {
    if let Some(y) = matvec_many_opt(ms, x, xs) { return y; }
    let mut y = vec![0f32; ms.iter().map(|m| m.rows).sum()];
    let mut off = 0;
    for (i, m) in ms.iter().enumerate() {
        m.matvec(&x[i * xs..i * xs + m.cols], &mut y[off..off + m.rows]);
        off += m.rows;
    }
    y
}

/// The batched path of `matvec_many`, or None when the matrices cannot share one GPU batch (or batching is off).
pub fn matvec_many_opt(ms: &[Mat], x: &[f32], xs: usize) -> Option<Vec<f32>> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("STRATA_DS_BATCH").ok().as_deref() != Some("0")) { return None; }
    dev::get().and_then(|d| d.lock().unwrap().batch_mats(ms, x, xs))
}

/// W x[t] for each row of `xs` (all rows of one matrix). Batches on the GPU when the weight is resident.
pub fn matvec_rows(m: &Mat, xs: &[&[f32]]) -> Vec<f32> {
    if let Some(y) = dev::get().and_then(|d| d.lock().unwrap().rows(m, xs)) { return y; }
    let mut y = vec![0f32; xs.len() * m.rows];
    for (i, x) in xs.iter().enumerate() {
        m.matvec(x, &mut y[i * m.rows..(i + 1) * m.rows]);
    }
    y
}
/// Expert cache lookups since the last PROF print: [hit, miss].
pub static EX_LOOKUPS: [std::sync::atomic::AtomicU64; 2] = [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];
pub fn gpu_bytes() -> usize { dev::get().map_or(0, |d| d.lock().unwrap().bytes()) }
/// Expert-budget bytes not yet used (0 without a GPU).
pub fn budget_left() -> usize { dev::get().map_or(0, |d| d.lock().unwrap().budget_left()) }
/// (backend, resident expert slots, bytes of all uploaded weights) when the GPU path is open.
pub fn dev_info() -> Option<(&'static str, usize, usize)> {
    dev::get().map(|d| d.lock().unwrap().info())
}

/// Row-major weight: `rows` rows of `cols` elements (ggml dims[0] = cols, dims[1] = rows).
/// A 3-D expert tensor is addressed with `expert(e)`.
#[derive(Clone, Copy)]
pub struct Mat<'a> {
    pub data: &'a [u8],
    pub ty: GgmlType,
    pub rows: usize,
    pub cols: usize,
    /// Routed-expert tensor: llama.cpp runs these on the CPU (Q8_K activations); the rest runs on CUDA (Q8_1 blocks of 32).
    pub cpu: bool,
    /// Index of the VRAM copy (STRATA_DS_GPU).
    pub gpu: Option<u32>,
}

impl<'a> Mat<'a> {
    pub fn load(g: &'a Gguf, name: &str) -> anyhow::Result<Self> {
        let t: &TensorInfo = g.tensors.get(name).ok_or_else(|| anyhow::anyhow!("missing tensor {name}"))?;
        let ty = GgmlType::from_u32(t.ty)?;
        let cols = t.dims[0] as usize;
        let rows = t.dims[1..].iter().product::<u64>() as usize;
        let cpu = name.contains("_exps");
        let data = g.bytes(t);
        let gpu = if cpu || t.dims.len() != 2 { None } else { dev::get().and_then(|d| d.lock().unwrap().upload(data, ty, rows, cols)) };
        Ok(Mat { data, ty, rows, cols, cpu, gpu })
    }

    /// Like `load` for a matrix used as `n` row groups (grouped output projection): each group gets its own VRAM copy.
    pub fn load_groups(g: &'a Gguf, name: &str, n: usize) -> anyhow::Result<Self> {
        let mut m = Self::load(g, name)?;
        m.gpu = None;
        if let Some(d) = dev::get() {
            let (gr, rb) = (m.rows / n, m.ty.row_bytes(m.cols as u64));
            let mut d = d.lock().unwrap();
            let gpu = d.pick(m.data.len());  // the groups get consecutive ids on one GPU
            let ids: Vec<_> = (0..n).map(|i| d.upload_on(gpu, &m.data[i * gr * rb..(i + 1) * gr * rb], m.ty, gr, m.cols)).collect();
            if ids.iter().all(|i| i.is_some()) { m.gpu = ids[0]; }
        }
        Ok(m)
    }

    /// Row group `i` of `n` (see `load_groups`).
    pub fn group(&self, i: usize, n: usize) -> Mat<'a> {
        let rows = self.rows / n;
        let sz = rows * self.ty.row_bytes(self.cols as u64);
        Mat { data: &self.data[i * sz..(i + 1) * sz], rows, gpu: self.gpu.map(|b| b + i as u32), ..*self }
    }

    /// Slice of a [cols, rows, n_expert] tensor for expert e.
    pub fn expert(&self, e: usize, n_expert: usize) -> Mat<'a> {
        let rows = self.rows / n_expert;
        let sz = rows * self.ty.row_bytes(self.cols as u64);
        let data = &self.data[e * sz..(e + 1) * sz];
        let gpu = dev::get().and_then(|d| d.lock().unwrap().expert(data, self.ty, rows, self.cols));
        Mat { data, ty: self.ty, rows, cols: self.cols, cpu: self.cpu, gpu }
    }

    /// Whether this matrix has no VRAM copy and must be read from the mapped file (CPU or staged expert).
    pub fn host_only(&self) -> bool { self.gpu.is_none() || self.gpu == Some(dev::STAGED) }

    /// Ask the kernel to read expert e's pages now (madvise WILLNEED), so all selected experts of a layer queue on the SSD together.
    #[cfg(target_os = "linux")]
    pub fn prefetch_expert(&self, e: usize, n_expert: usize) {
        unsafe extern "C" { fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32; }
        let sz = self.rows / n_expert * self.ty.row_bytes(self.cols as u64);
        let a = self.data[e * sz..].as_ptr() as usize;
        let start = a & !4095;
        unsafe { madvise(start as *mut std::ffi::c_void, a + sz - start, 3) };
    }
    /// No madvise outside Linux (the MSVC linker has no such symbol, issue #21): the pages are read when touched.
    #[cfg(not(target_os = "linux"))]
    pub fn prefetch_expert(&self, _e: usize, _n_expert: usize) {}

    pub fn row_into(&self, r: usize, out: &mut [f32]) {
        let rb = self.ty.row_bytes(self.cols as u64);
        dequant_row(self.ty, &self.data[r * rb..(r + 1) * rb], out);
    }

    /// y = W x
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(y.len(), self.rows);
        if let (Some(i), Some(d)) = (self.gpu, dev::get()) {
            return d.lock().unwrap().gemv(i, x, y, (i == dev::STAGED).then_some((self.data, self.ty, self.rows, self.cols)));
        }
        let rb = self.ty.row_bytes(self.cols as u64);
        // Integer Q8_K dots are several times faster than dequantising each row, and match llama.cpp's CPU path.
        // `STRATA_CPU_REF=1` keeps the f32 reference for debugging and exact old numerics.
        if (self.cpu || self.rows >= 512) && std::env::var_os("STRATA_CPU_REF").is_none() && self.cols % self.ty.block_elems() == 0 {
            if let Some(dot) = crate::cpu_moe::int_dot(self.ty) {
                let xq = crate::cpu_moe::quantize_q8k(x);
                y.par_chunks_mut(32).enumerate().for_each(|(c, ys)| {
                    for (k, o) in ys.iter_mut().enumerate() {
                        let r = c * 32 + k;
                        *o = dot(&self.data[r * rb..(r + 1) * rb], &xq);
                    }
                });
                return;
            }
        }
        let xq = q8_roundtrip(self.ty, self.cpu, x);
        let x = &xq[..];
        y.par_iter_mut().enumerate().for_each_init(
            || vec![0f32; self.cols],
            |tmp, (r, o)| {
                dequant_row(self.ty, &self.data[r * rb..(r + 1) * rb], tmp);
                *o = tmp.iter().zip(x).map(|(a, b)| a * b).sum();
            },
        );
    }
}

/// ggml quantises the activation to Q8_K (k-quants, IQ1/IQ2) or Q8_0 (Q8_0, IQ4_NL) before a quantised dot.
/// Reproduce that rounding so the f32 reference tracks llama.cpp; other types see the activation unchanged.
fn q8_roundtrip(ty: GgmlType, cpu: bool, x: &[f32]) -> Vec<f32> {
    use GgmlType::*;
    let mut out = x.to_vec();
    match ty {
        Q5K | IQ1M | IQ2XXS | IQ2XS if cpu => {
            for b in out.chunks_exact_mut(256) {
                let (mut amax, mut max) = (0f32, 0f32);
                for &v in b.iter() {
                    if v.abs() > amax { amax = v.abs(); max = v; }
                }
                if amax == 0.0 { b.fill(0.0); continue; }
                let iscale = -127.0 / max;
                for v in b.iter_mut() { *v = (iscale * *v).round_ties_even().min(127.0) / iscale; }
            }
        }
        Q5K | IQ1M | IQ2XXS | IQ2XS | Q8_0 | IQ4NL => {
            for b in out.chunks_exact_mut(32) {
                let amax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
                let d = amax / 127.0;
                if d == 0.0 { continue; }
                let dh = half::f16::from_f32(d).to_f32();
                for v in b.iter_mut() { *v = (*v / d).round() * dh; }
            }
        }
        _ => {}
    }
    out
}

/// Small F32/BF16 tensor fully dequantised.
pub fn vec_f32(g: &Gguf, name: &str) -> anyhow::Result<Vec<f32>> {
    let m = Mat::load(g, name)?;
    let mut v = vec![0f32; m.rows * m.cols];
    for r in 0..m.rows {
        m.row_into(r, &mut v[r * m.cols..(r + 1) * m.cols]);
    }
    Ok(v)
}
