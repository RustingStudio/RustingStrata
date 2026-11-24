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
    /// Optional per-tensor post-dequant multiplier (e.g. NVFP4 `<stem>.scale` F32 scalar, else `1.0`).
    pub scale: f32,
}

impl<'a> Tensor<'a> {
    fn row_bytes(&self) -> usize { self.ty.row_bytes(self.cols as u64) }

    pub fn row(&self, r: usize, out: &mut [f32]) {
        let rb = self.row_bytes();
        dequant_row(self.ty, &self.data[r * rb..(r + 1) * rb], out);
        if self.scale != 1.0 {
            for v in out.iter_mut() { *v *= self.scale; }
        }
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
    phys_layer: Vec<usize>,
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
        let mut w = Weights {
            g,
            aux_g,
            aliases: c.aliases,
            phys_layer: c.phys_layer.clone(),
            built: HashMap::new(),
            baked: HashMap::new(),
            lora_baked: std::collections::HashSet::new(),
        };
        // llama.cpp src/models/qwen3next.cpp: per key head group, r = nv / nk beta rows, then r alpha rows.
        for il in 0..c.n_layer {
            let p = c.phys_layer.get(il).copied().unwrap_or(il);
            if p != il { continue; }
            let leak = |v: Vec<u8>| -> &'a [u8] { Box::leak(v.into_boxed_slice()) };
            if let Ok(qkvg) = w.layer(il, "attn_qkv_gate.weight") {
                let channels = c.ssm_d_inner + 2 * c.ssm_n_group * c.ssm_d_state;
                let rb = qkvg.row_bytes();
                anyhow::ensure!(qkvg.rows == channels + c.ssm_d_inner,
                    "blk.{il}.attn_qkv_gate: {} rows, expected {}", qkvg.rows, channels + c.ssm_d_inner);
                let split_at = channels * rb;
                w.built.insert(format!("blk.{il}.attn_qkv.weight"), Tensor { rows: channels, data: &qkvg.data[..split_at], ..qkvg });
                w.built.insert(format!("blk.{il}.attn_gate.weight"), Tensor { rows: c.ssm_d_inner, data: &qkvg.data[split_at..], ..qkvg });
            }
            if let Ok(ba) = w.layer(il, "ssm_beta_alpha.weight") {
                let (nv, rb) = (c.ssm_dt_rank, ba.row_bytes());
                anyhow::ensure!(nv > 0 && ba.rows == 2 * nv, "blk.{il}.ssm_beta_alpha: {} rows for {nv}", ba.rows);
                let split_at = nv * rb;
                let half = Tensor { rows: nv, ..ba };
                w.built.insert(format!("blk.{il}.ssm_beta.weight"), Tensor { data: &ba.data[..split_at], ..half });
                w.built.insert(format!("blk.{il}.ssm_alpha.weight"), Tensor { data: &ba.data[split_at..], ..half });
            }
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
                    scale: 1.0,
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
        let scale = t.name.strip_suffix(".weight")
            .and_then(|stem| owner_g.tensors.get(&format!("{stem}.scale")))
            .and_then(|st| {
                if st.ty == 0 && owner_g.bytes(st).len() == 4 {
                    Some(f32::from_le_bytes(owner_g.bytes(st).try_into().unwrap()))
                } else {
                    None
                }
            })
            .unwrap_or(1.0);
        Ok(Tensor {
            ty,
            cols: t.dims[0] as usize,
            rows: t.dims[1..].iter().product::<u64>() as usize,
            data: owner_g.bytes(t),
            scale,
        })
    }

    pub fn layer(&self, il: usize, suffix: &str) -> anyhow::Result<Tensor<'a>> {
        let p = self.phys_layer.get(il).copied().unwrap_or(il);
        if p != il
            && let Ok(t) = self.get(&format!("blk.{il}.{suffix}"))
        {
            return Ok(t);
        }
        self.get(&format!("blk.{p}.{suffix}"))
    }

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
                    crate::say!("lora: target {name} has {} experts, expected 1; skipping", entry.n_expert);
                    continue;
                }
                let base = match self.get(name) { Ok(t) => t, Err(_) => { crate::say!("lora: target {name} not found, skipping"); continue; } };
                if base.cols != entry.in_dim || base.rows != entry.out_dim {
                    crate::say!("lora: target {name} is {}x{}, adapter is {}x{}; skipping", base.rows, base.cols, entry.out_dim, entry.in_dim);
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
                let t = Tensor { ty: crate::quant::GgmlType::F32, cols: base.cols, rows: base.rows, data, scale: 1.0 };
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
    // One call reads at most one readahead window and silently skips the rest of its range (16 MB steps on a 4 MB
    // window read 25% of a file). The window is read_ahead_kb, 128 KB unless the system or the file system raised it.
    let (mut a, end) = (p & !4095, p + n);
    while a < end {
        let l = (end - a).min(128 << 10);
        madvise(a, l, 3);
        a += l;
    }
}

/// Bytes of the GGUF files this process has mapped now.
static MAPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether the mapped model files fit in RAM with 4 GiB to spare: then their pages stay in the page cache (`cold` is
/// a no-op) and `preload` reads them. `STRATA_PRELOAD=0`/`1` overrides.
// ponytail: MemTotal, not what other programs left free; use MemAvailable if that decides wrongly.
pub fn fits_ram() -> bool {
    match std::env::var("STRATA_PRELOAD").ok().as_deref() {
        Some("0") => false,
        Some(_) => true,
        None => { static RAM: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new(); *RAM.get_or_init(crate::memory::total_ram) }.is_some_and(|t| MAPPED.load(std::sync::atomic::Ordering::Relaxed) + (4 << 30) <= t),
    }
}

/// Whether `path` is on a network file system (SMB, NFS, 9p, Ceph, Gluster or a FUSE mount), from
/// `/proc/self/mountinfo`. `preload` asks for the whole file at once: a server on the network then gets every read
/// together, and they take the link from all other programs (issue #94). So such a file is read when its pages are
/// touched. `STRATA_PRELOAD=1` reads it ahead all the same.
pub fn net_fs(path: &std::path::Path) -> bool {
    if std::env::var_os("STRATA_PRELOAD").is_some() { return false; }
    let (Ok(p), Ok(mounts)) = (path.canonicalize(), std::fs::read_to_string("/proc/self/mountinfo")) else { return false };
    net_fs_in(&p, &mounts)
}

/// `net_fs` for the text of a mountinfo file: the type of the longest mount point that is a prefix of `p` decides.
// ponytail: a mount point with a space is written with \040 and does not match, so the mount above it decides.
fn net_fs_in(p: &std::path::Path, mounts: &str) -> bool {
    mounts.lines().filter_map(|l| { let (a, b) = l.split_once(" - ")?; Some((a.split(' ').nth(4)?, b.split(' ').next()?)) })
        .filter(|(m, _)| p.starts_with(m)).max_by_key(|(m, _)| m.len())
        .is_some_and(|(_, t)| ["cifs", "smb3", "smbfs", "nfs", "nfs4", "9p", "ceph", "glusterfs"].contains(&t) || t.starts_with("fuse."))
}

#[test]
fn net_fs_reads_mountinfo() {
    let m = "22 1 8:2 / / rw,relatime shared:1 - ext4 /dev/sda2 rw\n\
             40 22 0:35 / /mnt/nas rw,relatime shared:20 - cifs //nas/models rw,vers=3.1.1\n\
             41 40 8:17 / /mnt/nas/local rw - xfs /dev/sdb1 rw\n\
             42 22 0:40 / /home/u/remote rw,nosuid - fuse.sshfs u@host:/ rw\n\
             43 22 8:33 / /mnt/win rw - fuseblk /dev/sdc1 rw\n";
    let on = |p: &str| net_fs_in(std::path::Path::new(p), m);
    assert!(on("/mnt/nas/a.gguf") && on("/home/u/remote/a.gguf"));
    assert!(!on("/mnt/nas/local/a.gguf") && !on("/mnt/nasty/a.gguf") && !on("/home/u/a.gguf") && !on("/mnt/win/a.gguf"));
}

/// Counts a newly mapped model file; with `read`, when the files fit in RAM (`fits_ram`), a background thread starts
/// reading it into the page cache, so experts that run on the CPU do not wait for the drive on their first use
/// (issue #49).
pub fn preload(b: &[u8], read: bool) {
    MAPPED.fetch_add(b.len() as u64, std::sync::atomic::Ordering::Relaxed);
    let (p, n) = (b.as_ptr() as usize, b.len());
    // The thread holds no borrow: MADV_WILLNEED on a range unmapped in the meantime fails without a fault.
    if read && fits_ram() {
        PRELOADING.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::thread::spawn(move || { will_need(p, n); PRELOADING.fetch_sub(1, std::sync::atomic::Ordering::Relaxed); });
    }
}

/// Takes the bytes of an unmapped model file off the count.
pub fn unmapped(n: usize) {
    MAPPED.fetch_sub(n as u64, std::sync::atomic::Ordering::Relaxed);
}

/// `preload` threads still queueing reads.
static PRELOADING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Whether every read `preload` asked for is queued: from then on a readahead request for a part of the model adds
/// nothing. False until then, and always when the model does not fit in RAM.
pub fn preloaded() -> bool {
    static FITS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    PRELOADING.load(std::sync::atomic::Ordering::Relaxed) == 0 && *FITS.get_or_init(fits_ram)
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
