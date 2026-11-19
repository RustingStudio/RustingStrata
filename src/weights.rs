//! Named tensor views over the mmapped GGUF, page cache advice for them, and a reference parallel matvec.

use crate::config::ModelConfig;
use crate::gguf::Gguf;
use crate::quant::{GgmlType, dequant_row};
use anyhow::Context;
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub struct Tensor<'a> {
    pub ty: GgmlType,
    /// Elements per row (GGUF dims[0]).
    pub cols: usize,
    /// Product of the remaining dims; for 3-D expert tensors this is out_rows * n_expert.
    pub rows: usize,
    pub data: &'a [u8],
}

impl<'a> Tensor<'a> {
    fn row_bytes(&self) -> usize { self.ty.row_bytes(self.cols as u64) }

    pub fn row(&self, r: usize, out: &mut [f32]) {
        let rb = self.row_bytes();
        dequant_row(self.ty, &self.data[r * rb..(r + 1) * rb], out);
    }

    /// y[r] = row_r · x for all rows.
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!((x.len(), y.len()), (self.cols, self.rows));
        y.par_chunks_mut(16).enumerate().for_each_init(|| vec![0f32; self.cols], |buf, (ci, ys)| {
            for (k, yv) in ys.iter_mut().enumerate() {
                self.row(ci * 16 + k, buf);
                *yv = buf.iter().zip(x).map(|(a, b)| a * b).sum();
            }
        });
    }

    /// out[t][r] = row_r · x[t] for all rows and tokens. `x` is `n_tokens * cols`, `out` is `n_tokens * rows`.
    pub fn matmul(&self, x: &[f32], n_tokens: usize, out: &mut [f32]) {
        assert_eq!((x.len(), out.len()), (n_tokens * self.cols, n_tokens * self.rows));
        // Row-outer: each weight row is decoded once, whatever n_tokens is. Blocks of rows that fit the L2 cache stay
        // there while every token passes over them.
        let block = (1 << 16) / self.cols.max(1) + 1;
        let mut by_row = vec![0f32; out.len()];
        by_row.par_chunks_mut(block * n_tokens).enumerate().for_each_init(|| vec![0f32; block * self.cols], |buf, (b, ys)| {
            let nr = ys.len() / n_tokens;
            for (r, w) in buf.chunks_exact_mut(self.cols).take(nr).enumerate() { self.row(b * block + r, w); }
            for (t, xt) in x.chunks_exact(self.cols).enumerate() {
                for (r, w) in buf.chunks_exact(self.cols).take(nr).enumerate() { ys[r * n_tokens + t] = dot(w, xt); }
            }
        });
        out.par_chunks_mut(self.rows).enumerate().for_each(|(t, ys)| {
            for (r, yv) in ys.iter_mut().enumerate() { *yv = by_row[r * n_tokens + t]; }
        });
    }

    /// Rows belonging to expert `e` of an [cols, out, n_expert] tensor.
    pub fn expert(&self, e: usize, n_expert: usize) -> Tensor<'a> {
        let rows = self.rows / n_expert;
        let rb = self.row_bytes();
        Tensor { rows, data: &self.data[e * rows * rb..(e + 1) * rows * rb], ..*self }
    }

    pub fn to_f32(&self) -> Vec<f32> {
        let mut v = vec![0f32; self.rows * self.cols];
        for (r, out) in v.chunks_exact_mut(self.cols).enumerate() { self.row(r, out); }
        v
    }
}

pub struct Weights<'a> {
    pub g: &'a Gguf,
    pub aux_g: Option<&'a Gguf>,
    aliases: &'static [(&'static str, &'static str)],
    /// Loader-owned overrides: synthesized `ssm_beta`/`ssm_alpha` and LoRA-baked tensors.
    pub built: HashMap<String, Tensor<'a>>,
    /// Heap storage the baked tensors in `built` point into. Kept beside `built` so a reload frees it instead of
    /// leaking the F32 copy; `built` is cleared before `baked` so no tensor outlives its bytes.
    baked: HashMap<String, Box<[u8]>>,
    pub lora_baked: std::collections::HashSet<String>,
}

impl<'a> Weights<'a> {
    pub fn new(g: &'a Gguf) -> anyhow::Result<Weights<'a>> {
        Self::new_with_aux(g, None)
    }

    pub fn new_with_aux(g: &'a Gguf, aux_g: Option<&'a Gguf>) -> anyhow::Result<Weights<'a>> {
        let c = ModelConfig::from_gguf(g)?;
        let mut w = Weights { g, aux_g, aliases: c.aliases, built: HashMap::new(), baked: HashMap::new(), lora_baked: std::collections::HashSet::new() };
        // llama.cpp src/models/qwen3next.cpp: per key head group, r = nv / nk beta rows, then r alpha rows.
        for il in 0..c.n_layer {
            let Ok(ba) = w.layer(il, "ssm_ba.weight") else { continue };
            let (nk, nv, rb) = (c.ssm_n_group, c.ssm_dt_rank, ba.row_bytes());
            anyhow::ensure!(nk > 0 && nv > 0 && nv % nk == 0 && ba.rows == 2 * nv && ba.data.len() == ba.rows * rb,
                "blk.{il}.ssm_ba: {} rows, {} bytes for group_count {nk}, time_step_rank {nv}", ba.rows, ba.data.len());
            let r = nv / nk;
            let (mut b, mut a) = (Vec::with_capacity(ba.data.len() / 2), Vec::with_capacity(ba.data.len() / 2));
            for grp in ba.data.chunks_exact(2 * r * rb) {
                b.extend_from_slice(&grp[..r * rb]);
                a.extend_from_slice(&grp[r * rb..]);
            }
            // ponytail: leaked, a few KB per layer per load; own the bytes in Weights if one process loads many models.
            let leak = |v: Vec<u8>| -> &'a [u8] { Box::leak(v.into_boxed_slice()) };
            let half = Tensor { rows: ba.rows / 2, ..ba };
            w.built.insert(format!("blk.{il}.ssm_beta.weight"), Tensor { data: leak(b), ..half });
            w.built.insert(format!("blk.{il}.ssm_alpha.weight"), Tensor { data: leak(a), ..half });
        }
        // Standalone EAGLE3 draft GGUF: remap layer-0 draft weights onto blk.{n_layer}.* so they never shadow layer 0.
        if let Some(ag) = aux_g.filter(|ag| ag.tensors.contains_key("fc.weight")) {
            let nl = c.n_layer;
            for (name, t) in &ag.tensors {
                if matches!(name.as_str(), "d2t" | "t2d" | "token_embd.weight") { continue; }
                let Ok(ty) = GgmlType::from_u32(t.ty) else { continue };
                let dst = match name.as_str() {
                    "fc.weight" => format!("blk.{nl}.eagle3.fc.weight"),
                    "output.weight" => format!("blk.{nl}.eagle3.output.weight"),
                    "output_norm.weight" => format!("blk.{nl}.nextn.shared_head_norm.weight"),
                    "blk.0.attn_norm.weight" => format!("blk.{nl}.nextn.enorm.weight"),
                    "blk.0.attn_norm_2.weight" => format!("blk.{nl}.nextn.hnorm.weight"),
                    "blk.0.ffn_norm.weight" => format!("blk.{nl}.post_attention_norm.weight"),
                    _ if name.starts_with("blk.0.") => format!("blk.{nl}.{}", &name["blk.0.".len()..]),
                    _ => continue,
                };
                w.built.insert(dst, Tensor {
                    ty,
                    cols: t.dims[0] as usize,
                    rows: t.dims[1..].iter().product::<u64>() as usize,
                    data: ag.bytes(t),
                });
            }
        }
        Ok(w)
    }

    /// `name` with the architecture's alias for a canonical suffix tried second.
    pub fn get(&self, name: &str) -> anyhow::Result<Tensor<'a>> {
        if let Some(t) = self.built.get(name) { return Ok(*t); }
        let find_t = |g: &'a Gguf| -> Option<(&'a crate::gguf::TensorInfo, &'a Gguf)> {
            if let Some(t) = g.tensors.get(name) {
                return Some((t, g));
            }
            if let Some(t) = self.aliases.iter().find_map(|&(c, a)| g.tensors.get(&format!("{}{a}", name.strip_suffix(c)?))) {
                return Some((t, g));
            }
            None
        };
        let (t, owner_g) = find_t(self.g)
            .or_else(|| self.aux_g.and_then(find_t))
            .with_context(|| format!("missing tensor {name}"))?;
        let ty = GgmlType::from_u32(t.ty).with_context(|| format!("tensor {name}"))?;
        Ok(Tensor {
            ty,
            cols: t.dims[0] as usize,
            rows: t.dims[1..].iter().product::<u64>() as usize,
            data: owner_g.bytes(t),
        })
    }

    pub fn layer(&self, il: usize, suffix: &str) -> anyhow::Result<Tensor<'a>> { self.get(&format!("blk.{il}.{suffix}")) }

    /// Refuse files containing quant types this engine does not implement, except in the MTP (nextn) layers, blocks
    /// `n_layer` on: false if one of those has such a type, so that the model loads without MTP.
    /// Drop the LoRA-baked overrides and the heap copies they point into; synthesized `ssm_*` tensors stay.
    pub fn clear_baked(&mut self) {
        for name in std::mem::take(&mut self.lora_baked) { self.built.remove(&name); }
        self.baked.clear();
    }

    pub fn bake_lora(&mut self, loras: &[crate::model::Lora]) {
        for lora in loras {
            for (name, entry) in &lora.entries {
                if name.ends_with("ffn_down_exps.weight") { continue; }
                if entry.n_expert != 1 {
                    eprintln!("lora: target {name} has {} experts, expected 1; skipping", entry.n_expert);
                    continue;
                }
                let base = match self.get(name) { Ok(t) => t, Err(_) => { eprintln!("lora: target {name} not found, skipping"); continue; } };
                if base.cols != entry.in_dim || base.rows != entry.out_dim {
                    eprintln!("lora: target {name} is {}x{}, adapter is {}x{}; skipping", base.rows, base.cols, entry.out_dim, entry.in_dim);
                    continue;
                }
                let mut w = base.to_f32();
                let (a, b) = (&entry.a[0], &entry.b[0]);
                let (rank, in_dim, out_dim) = (entry.rank, entry.in_dim, entry.out_dim);
                let mult = if lora.alpha > 0.0 { lora.scale * lora.alpha / rank as f32 } else { lora.scale };
                for o in 0..out_dim {
                    for i in 0..in_dim {
                        let mut d = 0f32;
                        for r in 0..rank { d += b[o * rank + r] * a[r * in_dim + i]; }
                        w[o * in_dim + i] += mult * d;
                    }
                }
                self.built.remove(name);
                self.baked.remove(name);
                let bytes: Box<[u8]> = w.iter().flat_map(|f| f.to_le_bytes()).collect();
                // `bytes` lives in `self.baked` until `clear_baked`, which removes the tensor below first.
                let data: &'a [u8] = unsafe { std::slice::from_raw_parts(bytes.as_ptr(), bytes.len()) };
                let t = Tensor { ty: crate::quant::GgmlType::F32, cols: base.cols, rows: base.rows, data };
                self.baked.insert(name.clone(), bytes);
                self.built.insert(name.clone(), t);
                self.lora_baked.insert(name.clone());
            }
        }
    }
    pub fn validate(&self, n_layer: usize) -> anyhow::Result<bool> {
        let mut nextn = true;
        let iter = self.g.tensors.values().chain(self.aux_g.into_iter().flat_map(|ag| ag.tensors.values()));
        for t in iter {
            if matches!(t.name.as_str(), "d2t" | "t2d") { continue; }
            if let Err(e) = GgmlType::from_u32(t.ty) {
                let il = t.name.strip_prefix("blk.").and_then(|r| r.split('.').next()).and_then(|i| i.parse::<usize>().ok());
                if il.is_some_and(|il| il >= n_layer) { nextn = false; continue; }
                return Err(e).with_context(|| format!("tensor {}", t.name));
            }
        }
        Ok(nextn)
    }
}

#[cfg(target_os = "linux")]
fn madvise(p: usize, n: usize, advice: i32) {
    unsafe extern "C" { fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32; }
    unsafe { madvise(p as *mut _, n, advice) };
}
#[cfg(not(target_os = "linux"))]
fn madvise(_: usize, _: usize, _: i32) {}

/// Starts reading mapped bytes p..p + n into the page cache without waiting (MADV_WILLNEED).
pub fn will_need(p: usize, n: usize) {
    // One call reads at most one readahead window (read_ahead_kb, 128-256 KB on NVMe).
    let (mut a, end) = (p & !4095, p + n);
    while a < end {
        let l = (end - a).min(256 << 10);
        madvise(a, l, 3);
        a += l;
    }
}

/// Bytes of every GGUF mapped by this process.
static MAPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether the mapped model files fit in RAM with 4 GiB to spare: then their pages stay in the page cache (`cold` is
/// a no-op) and `preload` reads them. `STRATA_PRELOAD=0`/`1` overrides.
// ponytail: MemTotal, not what other programs left free, and files of dropped models still count; use
// MemAvailable and subtract on drop if either one decides wrongly.
pub fn fits_ram() -> bool {
    match std::env::var("STRATA_PRELOAD").ok().as_deref() {
        Some("0") => false,
        Some(_) => true,
        None => { static RAM: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new(); *RAM.get_or_init(crate::memory::total_ram) }.is_some_and(|t| MAPPED.load(std::sync::atomic::Ordering::Relaxed) + (4 << 30) <= t),
    }
}

/// Counts a newly mapped model file; when the files fit in RAM (`fits_ram`), a background thread starts reading it
/// into the page cache, so experts that run on the CPU do not wait for the drive on their first use (issue #49).
pub fn preload(b: &[u8]) {
    MAPPED.fetch_add(b.len() as u64, std::sync::atomic::Ordering::Relaxed);
    let (p, n) = (b.as_ptr() as usize, b.len());
    // The thread holds no borrow: MADV_WILLNEED on a range unmapped in the meantime fails without a fault.
    if fits_ram() { std::thread::spawn(move || will_need(p, n)); }
}

/// Drops the whole pages of `b` (MADV_DONTNEED): bytes now on the GPU, whose host copy competes with the experts that
/// missed VRAM for page cache. Expert files exceed the page cache, so pages read once at load would otherwise outlive
/// the experts a prompt reads from it. `STRATA_COLD_KEEP` only marks them reclaimable (MADV_COLD) instead. Nothing is
/// dropped when the model fits in RAM (`fits_ram`).
pub fn cold(b: &[u8]) {
    if fits_ram() { return; }
    let (a, e) = ((b.as_ptr() as usize + 4095) & !4095, (b.as_ptr() as usize + b.len()) & !4095);
    if e > a { madvise(a, e - a, if std::env::var_os("STRATA_COLD_KEEP").is_some() { 20 } else { 4 }); }
}

/// Reads ranges of mapped weights into the page cache from a background thread (`will_need`), in the order sent, so
/// a cold read keeps many drive reads in flight instead of one page fault at a time. Dropping it joins the thread.
pub struct Prefetch { tx: Option<std::sync::mpsc::Sender<Vec<(usize, usize)>>>, th: Option<std::thread::JoinHandle<()>> }

impl Prefetch {
    pub fn start() -> Prefetch {
        let (tx, rx) = std::sync::mpsc::channel::<Vec<(usize, usize)>>();
        let th = std::thread::spawn(move || for rs in rx { for (p, n) in rs { will_need(p, n); } });
        Prefetch { tx: Some(tx), th: Some(th) }
    }
    /// (address, bytes) ranges of mapped weights.
    pub fn send(&self, rs: Vec<(usize, usize)>) { let _ = self.tx.as_ref().unwrap().send(rs); }
}

impl Drop for Prefetch {
    fn drop(&mut self) {
        self.tx = None;
        if let Some(t) = self.th.take() { let _ = t.join(); }
    }
}

/// Dot product with 16 independent partial sums, so the compiler vectorizes it (a single running f32 sum cannot
/// be). 32 and 64 sums measured slower.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    let (ca, cb) = (a.chunks_exact(16), b.chunks_exact(16));
    let tail: f32 = ca.remainder().iter().zip(cb.remainder()).map(|(x, y)| x * y).sum();
    for (x, y) in ca.zip(cb) { for i in 0..16 { acc[i] += x[i] * y[i]; } }
    acc.iter().sum::<f32>() + tail
}
