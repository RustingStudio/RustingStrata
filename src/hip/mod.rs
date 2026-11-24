//! HIP (ROCm) backend: the same `kernels.cu` as `cuda`, with `STRATA_HIP` selecting its portable code paths, compiled
//! at run time by hiprtc for the device's `gfx` target and launched through the HIP runtime loaded by name (see
//! `ffi` and `compat.h`). No ROCm headers, no `hipcc`, no C++ build step: `cargo build --release` and `--backend hip`.
//!
//! The device side is one source: `kernels.cu` keeps its CUDA paths behind `#if !STRATA_HIP`, and `compat.h` supplies
//! what CUDA's headers give it for free. On RDNA the kernel is compiled as wave32 (lane = threadIdx.x & 31), which is
//! the assumption the source already makes about a warp.
//!
//! Status: the host layer and the wave32 device build are in place; ops are being wired up one family at a time
//! against the CPU reference (`cargo test --release hip`). Everything not yet ported returns an error naming the op,
//! so a run says exactly which kernel is missing instead of computing silently wrong numbers.

use crate::backend::{ATT_SPLITS, ATT_SPLITS_1, Backend, Buf, GpuTensor, GEMM_BM, SAME_X};
use crate::cuda::{AMAX_PARTS, FA_BK, attn_prefill_smem_bk, dyn_shmem_need, kname, rpw_from, gemv_block_from, tuned};
use crate::quant::GgmlType;
use anyhow::{Context, bail};
use std::any::Any;
use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::sync::{Arc, LazyLock, Mutex};

/// `STRATA_HIP_SYNC`: one sync per launch (see `Gpu::launch`).
static SYNC_EACH: LazyLock<bool> = LazyLock::new(|| std::env::var_os("STRATA_HIP_SYNC").is_some());
/// `STRATA_NO_GRAPH` / `STRATA_HIP_GRAPH=0`: run the forward every time instead of capturing it.
static NO_GRAPH: LazyLock<bool> = LazyLock::new(|| {
    std::env::var_os("STRATA_NO_GRAPH").is_some() || std::env::var("STRATA_HIP_GRAPH").map(|v| v == "0").unwrap_or(false)
});
/// `STRATA_HIP_TRACE`: name every launch before it goes out, so a fault that kills the process names the kernel.
static TRACE: LazyLock<bool> = LazyLock::new(|| std::env::var_os("STRATA_HIP_TRACE").is_some());

mod ffi;

pub struct Gpu {
    /// The runtime, shared with every allocation and graph so none of them outlives the library that frees it.
    hip: Arc<ffi::Hip>,
    device: usize,
    stream: ffi::Stream,
    /// The copy queue of `htod_side`, and the events its marks are recorded in (reused round robin: a wait already
    /// queued keeps the record it saw).
    side: ffi::Stream,
    marks: Vec<Event>,
    next_mark: std::sync::atomic::AtomicUsize,
    /// Host ranges `pin_host` registered (start, end, the address registered): `htod` copies from them directly.
    pinned: std::sync::RwLock<Vec<(usize, usize, usize)>>,
    fns: HashMap<String, ffi::Func>,
    /// "gfx1201": the target hiprtc compiled for.
    arch: String,
    /// `--kv q8` (the kernels were compiled with `KV_MODE` 0 or 1; HIP has no q4/q2 cache rows).
    pub kv_q8: bool,
    /// Opt-in shared memory per block on this device (64 KiB on RDNA).
    max_smem: usize,
    /// `argmax_rows` partials.
    am: Buf,
    /// Recorded graphs by `Backend::graph` key.
    graphs: Mutex<HashMap<u64, Graph>>,
    /// Rope frequency tables, `base` and `n_rot` to the device buffer that holds them.
    fr: Mutex<(HashMap<(u32, usize), Buf>, Vec<(f32, usize)>)>,
    /// Pinned staging for large `htod`, used as a ring: the copy returns before its DMA, so the host fills the next
    /// region while the card reads the previous one (expert streaming in prefill).
    staging: Mutex<Staging>,
}

// The HIP handles are raw pointers but the runtime is thread-safe for the calls the backend makes, and `Backend`
// requires `Send`: transfers, launches and synchronization may come from any thread.
unsafe impl Send for Gpu {}

struct Graph {
    exec: ffi::GraphExec,
    graph: ffi::Graph,
    hip: Arc<ffi::Hip>,
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            (self.hip.graph_exec_destroy)(self.exec);
            (self.hip.graph_destroy)(self.graph);
        }
    }
}

/// One launch: grid and block dimensions and the dynamic shared memory the kernel asks for.
#[derive(Clone, Copy)]
pub struct LaunchConfig {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}

pub fn cfg(grid: (u32, u32), block: u32, smem: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: (grid.0, grid.1, 1), block_dim: (block, 1, 1), shared_mem_bytes: smem }
}

/// Launch kernel `$name` with `cfg`. Every argument is evaluated into a local, so `hipModuleLaunchKernel` reads it
/// from a live address; the type of the expression gives the parameter width, as the kernel's signature expects.
macro_rules! launch {
    ($g:expr, $name:expr, $cfg:expr, $($a:expr),* $(,)?) => {{
        let c = $cfg;
        $g.launch(&$name, c, &mut [ $(&mut { $a } as &mut dyn Any),* ])
            .with_context(|| format!("launch {}", $name))?;
    }};
}

/// `kernels.cu` assembled for hiprtc: the compatibility layer, the quant tables, the compile-time switches, the file.
fn source(kv_q8: bool) -> String {
    use crate::quant::tables::*;
    let join = |v: Vec<String>| v.join(",");
    format!(
        "{}\n\
         __device__ const unsigned long long iq2xs_grid[512] = {{{}}};\n\
         __device__ const unsigned long long iq2xxs_grid[256] = {{{}}};\n\
         __device__ const unsigned long long iq2s_grid[1024] = {{{}}};\n\
         __device__ const unsigned long long iq1s_grid[2048] = {{{}}};\n\
         __device__ const unsigned int iq3xxs_grid[256] = {{{}}};\n\
         __device__ const unsigned int iq3s_grid[512] = {{{}}};\n\
         __device__ const unsigned char ksigns_iq2xs[128] = {{{}}};\n\
         __device__ const signed char kvalues_iq4nl[16] = {{{}}};\n\
         __device__ const float tq_lut[4096] = {{{}}};\n\
         #define LLAMA_Q8 {}\n#define KV_MODE {}\n#define K_MODE KV_MODE\n#define V_MODE KV_MODE\n#define TQ_KBITS 4\n#define TQ_VBITS 4\n#define GEMM_TC 0\n#define GEMM_WMMA (STRATA_WMMA && {})\n\
         #define GEMM_BM {}\n#define GEMM_BK {}\n#define GEMM_MIN_BLOCKS 1\n#define GEMV_LOADW {}\n\
         #define FA_BQ {}\n#define FA_BK {}\n{}",
        include_str!("compat.h"),
        join(IQ2XS_GRID.iter().map(|v| format!("{v:#x}ULL")).collect()),
        join(IQ2XXS_GRID.iter().map(|v| format!("{v:#x}ULL")).collect()),
        join(IQ2S_GRID.iter().map(|v| format!("{v:#x}ULL")).collect()),
        join(IQ1S_GRID.iter().map(|v| format!("{v:#x}ULL")).collect()),
        join(IQ3XXS_GRID.iter().map(|v| format!("{v:#x}ULL")).collect()),
        join(IQ3S_GRID.iter().map(|v| format!("{v:#x}ULL")).collect()),
        join(KSIGNS_IQ2XS.iter().map(|v| v.to_string()).collect()),
        join(KVALUES_IQ4NL.iter().map(|v| v.to_string()).collect()),
        join(crate::quant::TQ_LUT.iter().flatten().map(|v| format!("{v:e}f")).collect()),
        crate::cuda::llama_numerics() as u8,
        kv_q8 as u8,
        // `STRATA_HIP_WMMA=0` falls back to the CUDA-core multiplications (A/B comparison, and a workaround if a
        // card's matrix cores misbehave).
        (std::env::var("STRATA_HIP_WMMA").map(|v| v != "0").unwrap_or(true)) as u8,
        GEMM_BM,
        crate::cuda::GEMM_BK,
        tuned(std::env::var("STRATA_GEMV_LOADW").ok().as_deref(), 0, |v| v <= 1),
        crate::cuda::FA_BQ,
        crate::cuda::FA_BK,
        include_str!("../cuda/kernels.cu"),
    )
}

/// Cache key for a compiled code object: the source, the target and the hiprtc version.
fn cache_key(src: &str, arch: &str, rtc: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (src, arch, rtc).hash(&mut h);
    h.finish()
}

impl Gpu {
    pub fn new(device: usize) -> anyhow::Result<Gpu> {
        let hip = Arc::new(ffi::Hip::open()?);
        hip.check(unsafe { (hip.init)(0) }, "hipInit")
            .map_err(|e| anyhow::anyhow!("{e} (check that `rocminfo` lists the card and that this user can open /dev/kfd)"))?;
        let n = {
            let mut n = 0;
            hip.check(unsafe { (hip.device_count)(&mut n) }, "hipGetDeviceCount")?;
            n as usize
        };
        anyhow::ensure!(device < n, "HIP device {device}: {n} device(s) present");
        hip.check(unsafe { (hip.set_device)(device as i32) }, "hipSetDevice")?;
        let arch = hip.arch(device as i32)?;
        let warp = hip.attribute(ffi::ATTR_WARP_SIZE, device as i32)?;
        anyhow::ensure!(warp == 32, "H100 HIP backend needs wave32; {arch} reports warp size {warp}");
        // Opt-in (75) is what a kernel may ask for; fall back to the plain per-block limit (74) if a runtime does not
        // know the attribute, and never trust a figure below the smallest tile the prefill attention needs.
        let mut max_smem = hip.attribute(ffi::ATTR_SHARED_PER_BLOCK_OPTIN, device as i32).unwrap_or(0) as usize;
        if max_smem < (16 << 10) {
            max_smem = hip.attribute(ffi::ATTR_SHARED_PER_BLOCK, device as i32).unwrap_or(16 << 10) as usize;
        }
        anyhow::ensure!(max_smem >= (16 << 10), "HIP device reports {max_smem} B of shared memory a block");
        // The HIP kernels hold an f16 or q8 cache (KV_MODE 0 or 1): refuse q4 and q2 rather than run f16 unasked.
        anyhow::ensure!(crate::cuda::kv_mode() <= 1, "--kv q4 and q2 are not available on the HIP backend (f16 or q8)");
        let kv_q8 = crate::cuda::kv_q8();
        let rtc = ffi::Rtc::get().map_err(|e| anyhow::anyhow!("{e}")).context("HIP init").context("hiprtc is required to compile the kernels")?;
        let (maj, min) = rtc.version();
        let tag = format!("{maj}.{min}");
        crate::say!("hip: {} ({arch}, wave{warp}, {} B shared memory a block)",
                  hip.device_name(device as i32), max_smem);
        let src = source(kv_q8);
        let obj = compile_cached(rtc, &src, &arch, &tag)?;
        let mut stream: ffi::Stream = std::ptr::null_mut();
        hip.check(unsafe { (hip.stream_create)(&mut stream, ffi::STREAM_NON_BLOCKING) }, "hipStreamCreate")?;
        let mut side: ffi::Stream = std::ptr::null_mut();
        hip.check(unsafe { (hip.stream_create)(&mut side, ffi::STREAM_NON_BLOCKING) }, "hipStreamCreate(side)")?;
        let marks = (0..MARKS).map(|_| Event::new(&hip)).collect::<anyhow::Result<Vec<_>>>()?;
        let mut module: ffi::Module = std::ptr::null_mut();
        hip.check(unsafe { (hip.module_load)(&mut module, obj.as_ptr().cast()) }, "hipModuleLoadData")?;
        let mut fns = HashMap::new();
        for name in kernel_names() {
            let c = std::ffi::CString::new(name.clone())?;
            let mut f: ffi::Func = std::ptr::null_mut();
            hip.check(unsafe { (hip.function)(&mut f, module, c.as_ptr()) }, &format!("hipModuleGetFunction({name})"))?;
            // The kernels ask for up to 49 KiB of dynamic shared memory, more than HIP allows by default. The ceiling
            // is the device's opt-in minus what the kernel already uses statically, which the runtime only checks
            // here, so an over-ask is halved instead of failing the backend.
            if let Some((_, need)) = dyn_shmem_need(&name) {
                let mut want = need.min(max_smem);
                loop {
                    let r = unsafe { (hip.func_set_attribute)(f, ffi::ATTR_MAX_DYNAMIC_SHARED, want as i32) };
                    if r == ffi::HIP_SUCCESS { break }
                    if want < 8192 {
                        bail!("hipFuncSetAttribute({name}, {want}): {}", hip.err(r));
                    }
                    want /= 2;
                }
            }
            fns.insert(name, f);
        }
        // argmax_part writes an int2 (8 bytes) per partial: AMAX_PARTS per row, up to 64 rows.
        let am_buf = alloc_raw(&hip, 64 * AMAX_PARTS * 8)?;
        // Pinned ring for large host-to-device copies.
        let mut host: *mut c_void = std::ptr::null_mut();
        hip.check(unsafe { (hip.host_malloc)(&mut host, RING, ffi::HOST_MALLOC_PORTABLE | ffi::HOST_MALLOC_MAPPED) }, "hipHostMalloc(ring)")?;
        let mut dev: *mut c_void = std::ptr::null_mut();
        hip.check(unsafe { (hip.host_device_ptr)(&mut dev, host, 0) }, "hipHostGetDevicePointer(ring)")?;
        let staging = Staging { _pin: Buf::new(dev as u64, RING, PinnedHost { hip: hip.clone(), p: host }), host: host.cast(), head: 0, busy: VecDeque::new() };
        Ok(Gpu { hip, device, stream, side, marks, next_mark: Default::default(), pinned: Default::default(), fns, arch, kv_q8, max_smem, am: am_buf,
                 graphs: Mutex::new(HashMap::new()), fr: Mutex::new((HashMap::new(), Vec::new())), staging: Mutex::new(staging) })
    }

    pub fn arch(&self) -> &str { &self.arch }

    /// Launch kernel `name` with `cfg`; `args` are the kernel parameters, in order, read from the given addresses.
    pub fn launch(&self, name: &str, c: LaunchConfig, args: &mut [&mut dyn Any]) -> anyhow::Result<()> {
        let f = *self.fns.get(name).ok_or_else(|| anyhow::anyhow!("HIP kernel {name} is not loaded"))?;
        if *TRACE { eprintln!("hip launch {name} grid {:?} block {:?} smem {}", c.grid_dim, c.block_dim, c.shared_mem_bytes); }
        let mut params: Vec<*mut c_void> = args.iter_mut().map(|a| (*a as *mut dyn Any).cast::<c_void>()).collect();
        let r = unsafe {
            (self.hip.launch)(f, c.grid_dim.0, c.grid_dim.1, c.grid_dim.2, c.block_dim.0, c.block_dim.1, c.block_dim.2,
                              c.shared_mem_bytes, self.stream, params.as_mut_ptr(), std::ptr::null_mut())
        };
        if r != ffi::HIP_SUCCESS { bail!("{}: {}", name, self.hip.err(r)) }
        // `STRATA_HIP_SYNC`: wait after every launch, so a hardware exception or bad argument names the kernel that
        // raised it instead of the sync at the end of the pass. Debugging only: no overlap and no graphs.
        if *SYNC_EACH {
            self.hip.check(unsafe { (self.hip.stream_sync)(self.stream) }, &format!("kernel {name} failed"))?;
        }
        Ok(())
    }

    fn wait(&self) -> anyhow::Result<()> {
        self.hip.check(unsafe { (self.hip.stream_sync)(self.side) }, "hipStreamSynchronize(side)")?;
        self.hip.check(unsafe { (self.hip.stream_sync)(self.stream) }, "hipStreamSynchronize")
    }

    /// Record the point `stream` has reached in the next mark event.
    fn mark(&self, stream: ffi::Stream) -> anyhow::Result<usize> {
        let i = self.next_mark.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % MARKS;
        self.marks[i].record(stream)?;
        Ok(i)
    }

    /// `htod` on `stream`: queue the copy and return once `src` may be reused. From pageable memory the runtime
    /// stages the buffer inside the call, and large copies go through the pinned ring so the DMA does not stop the
    /// host (expert streaming during prefill).
    fn htod_on(&self, stream: ffi::Stream, src: &[u8], dst: &Buf, off: usize) -> anyhow::Result<()> {
        anyhow::ensure!(off + src.len() <= dst.len(), "htod: {} bytes at {off} into {}", src.len(), dst.len());
        let (s0, s1) = (src.as_ptr() as usize, src.as_ptr() as usize + src.len());
        if src.len() < RING_MIN || self.pinned.read().unwrap().iter().any(|&(a, b, _)| a <= s0 && s1 <= b) {
            let r = unsafe { (self.hip.memcpy_async)((dst.addr() + off as u64) as *mut c_void, src.as_ptr().cast(), src.len(),
                                                     ffi::MEMCPY_H2D, stream) };
            return self.hip.check(r, "hipMemcpyAsync(htod)");
        }
        let mut g = self.staging.lock().unwrap();
        for (i, c) in src.chunks(RING).enumerate() {
            if g.head + c.len() > RING { g.head = 0; }
            let (a, b) = (g.head, g.head + c.len());
            // Regions still being read by an earlier copy have to complete before the host overwrites them.
            while g.busy.iter().any(|&(s, e, _)| s < b && a < e) {
                let (_, _, ev) = g.busy.pop_front().unwrap();
                self.hip.check(unsafe { (self.hip.event_sync)(ev.ev) }, "hipEventSynchronize")?;
            }
            unsafe { std::ptr::copy_nonoverlapping(c.as_ptr(), g.host.add(a), c.len()) };
            let r = unsafe { (self.hip.memcpy_async)((dst.addr() + (off + i * RING) as u64) as *mut c_void,
                                                     g.host.add(a) as *const c_void, c.len(), ffi::MEMCPY_H2D, stream) };
            self.hip.check(r, "hipMemcpyAsync(htod ring)")?;
            let ev = Event::new(&self.hip)?;
            ev.record(stream)?;
            g.busy.push_back((a, b, ev));
            g.head = b;
        }
        Ok(())
    }

    fn wait_mark(&self, stream: ffi::Stream, i: usize) -> anyhow::Result<()> {
        self.hip.check(unsafe { (self.hip.stream_wait_event)(stream, self.marks[i % MARKS].ev, 0) }, "hipStreamWaitEvent")
    }

    fn ptr(&self, b: &Buf) -> u64 { b.addr() }

    /// qwen35moe and the other models this backend targets dequantize in the kernel, so the ops below are the
    /// generic ones.
    fn float_ty(ty: GgmlType) -> bool { matches!(ty, GgmlType::F32 | GgmlType::BF16) }

    fn gemv_cols(prefix: &str, ty: GgmlType, cols: usize) -> anyhow::Result<(i32, String)> {
        let b = ty.block_elems().max(32);
        if cols % b != 0 { bail!("{} tensor with {cols} columns: GEMV needs a multiple of {b}", ty.name()); }
        Ok((cols as i32, kname(prefix, ty) + if cols % 256 != 0 { "_p" } else { "" }))
    }

    fn rpw(rows: usize, cols: usize, k: usize) -> usize {
        let d = if k > 1 { if cols <= 512 { 4 } else { (2048 / cols).max(1) } } else if rows >= 65536 { 8 } else if rows >= 8192 || rows == 2048 { 2 } else { 1 };
        let fam = if k > 1 { "STRATA_RPW_MOE" } else { "STRATA_RPW_DENSE" };
        let family = std::env::var(fam).ok();
        rpw_from(std::env::var("STRATA_RPW").ok().as_deref(), family.as_deref(), d)
    }

    fn gemv_block(k: usize) -> u32 {
        let fam = if k > 1 { "STRATA_GEMV_BLOCK_MOE" } else { "STRATA_GEMV_BLOCK_DENSE" };
        let family = std::env::var(fam).ok();
        gemv_block_from(std::env::var("STRATA_GEMV_BLOCK").ok().as_deref(), family.as_deref())
    }

    fn xq_smem(ty: GgmlType, cols: usize) -> u32 {
        if Self::float_ty(ty) { 0 } else { (cols + cols / 8) as u32 }
    }

    fn gemv_cfg(ty: GgmlType, rows: usize, cols: usize, k: usize) -> (LaunchConfig, i32) {
        if Self::float_ty(ty) {
            let wpb = if rows >= 4096 { 8 } else { 1 };
            return (cfg((rows.div_ceil(wpb) as u32, k as u32), 32 * wpb as u32, 0), 1);
        }
        let (rpw, wpb) = (Self::rpw(rows, cols, k), Self::gemv_block(k) as usize / 32);
        (cfg((rows.div_ceil(wpb * rpw) as u32, k as u32), (32 * wpb) as u32, Self::xq_smem(ty, cols)), rpw as i32)
    }

    /// Whether the stream is capturing a graph: no allocation and no synchronous copy may be issued inside one.
    fn capturing(&self) -> bool {
        let mut status = 0;
        unsafe { (self.hip.stream_is_capturing)(self.stream, &mut status) };
        status != 0
    }

    /// The rope frequencies base^(-2i / n_rot) as `rms_norm_rope` computes them in f64, made once by `rope_freq`.
    /// While a graph is captured a missing table gives 0 (the kernel computes the frequencies itself) and is made
    /// after the capture, which `graph` then repeats.
    fn rope_fr(&self, base: f32, n_rot: usize) -> anyhow::Result<u64> {
        let mut fr = self.fr.lock().unwrap();
        if let Some(b) = fr.0.get(&(base.to_bits(), n_rot)) { return Ok(b.addr()) }
        if self.capturing() {
            fr.1.push((base, n_rot));
            return Ok(0);
        }
        let buf = alloc_raw(&self.hip, 8 * (n_rot / 2).max(1))?;
        let p = buf.addr();
        let (bs, nr) = (base, n_rot as i32);
        launch!(self, "rope_freq", cfg((1, 1), 256, 0), p, bs, nr);
        self.wait()?;
        fr.0.insert((base.to_bits(), n_rot), buf);
        Ok(p)
    }

    /// `gemvq`; with z = 2 also the matrix of `pair.2` rows at byte offset `pair.0` into the output at byte offset
    /// `pair.1` (token stride `pair.2`).
    fn gemvq_pair(&self, w: &GpuTensor, x: u64, x_stride: usize, xq: u64, y: u64, y_stride: usize, nt: usize,
                  pair: (i64, i64, i32, f32), z: u32) -> anyhow::Result<()> {
        // gemvn takes 4 inputs at most; a larger batch (Inkling 2's draft verify, up to 16 rows) runs 4 at a time and
        // quantizes x itself, since xq holds the scales of all nt rows after all the rows (as on CUDA).
        if nt > 4 {
            for t0 in (0..nt).step_by(4) {
                let (xt, yt) = (x + 4 * (t0 * x_stride) as u64, y + 4 * (t0 * y_stride) as u64);
                let y2 = if z == 2 { pair.1 + 4 * t0 as i64 * (pair.2 as i64 - y_stride as i64) } else { 0 };
                self.gemvq_pair(w, xt, x_stride, 0, yt, y_stride, (nt - t0).min(4), (pair.0, y2, pair.2, pair.3), z)?;
            }
            return Ok(());
        }
        if nt == 0 { bail!("gemvn: no inputs"); }
        let (ncols, k1) = Self::gemv_cols("gemv", w.ty, w.cols)?;
        let (mut c, rpw) = Self::gemv_cfg(w.ty, w.rows, w.cols, 1);
        let per = c.shared_mem_bytes;
        c.shared_mem_bytes = if xq == 0 { per * nt as u32 } else { 0 };
        c.grid_dim.2 = z;
        // gemvn caches one x row per token in shared memory; when that exceeds the block's limit, run a token at a time.
        if xq == 0 && c.shared_mem_bytes as usize > self.max_smem {
            c.shared_mem_bytes = per;
            for t in 0..nt {
                let (xt, yt) = (x + 4 * t as u64 * x_stride as u64, y + 4 * t as u64 * y_stride as u64);
                launch!(self, &k1, c, w.ptr, xt, yt, w.rows as i32, ncols, rpw, xq, pair.0, pair.1, pair.2, w.scale, pair.3);
            }
            return Ok(());
        }
        if nt == 1 {
            launch!(self, &k1, c, w.ptr, x, y, w.rows as i32, ncols, rpw, xq, pair.0, pair.1, pair.2, w.scale, pair.3);
            return Ok(());
        }
        let kn = k1.replacen("gemv", "gemvn", 1);
        launch!(self, &kn, c, w.ptr, x, x_stride as i32, y, y_stride as i32, w.rows as i32, ncols, rpw, nt as i32, xq, pair.0, pair.1, pair.2, w.scale, pair.3);
        Ok(())
    }

    /// Prefill GEMM over a tile list (or one dense matrix when `tiles` is 0). Without int8 tensor cores the f16
    /// `gemm_*`/`gemms_*` kernels run; `gemm_tc` is compiled out of this build (`GEMM_TC 0`).
    fn gemm_raw(&self, ty: GgmlType, rows: usize, cols: usize, tiles: u64, n_tiles: usize, wd: u64, nd: usize,
                x: u64, x_stride: usize, _n_x: usize, xrow: u64, y: u64, y_stride: usize, act: i32, w_scale: f32) -> anyhow::Result<()> {
        if cols % 64 != 0 { bail!("{} matrix with {cols} columns: GEMM needs a multiple of 64", ty.name()); }
        if n_tiles == 0 { return Ok(()) }
        // Only the activation reaches the kernel: SAME_X and BATCHED (the same kernels here) are flags for the host.
        let (same, act) = (act & SAME_X != 0, act & 3);
        let _ = same;
        let grid = (rows.div_ceil(64) as u32, n_tiles as u32);
        let sm = tiles == 0 && (grid.0 * grid.1 < 112 || nd <= 32);
        let grid = if sm { (grid.0, nd.div_ceil(32) as u32) } else { grid };
        launch!(self, &kname(if sm { "gemms" } else { "gemm" }, ty), cfg(grid, 2 * GEMM_BM as u32, 0),
                tiles, wd, nd as i32, x, x_stride as i32, xrow, y, y_stride as i32, rows as i32, (cols / 64) as i32, act, w_scale);
        // Tile lists: gemm_* skips the tiles of at most 32 positions, which gemms_* takes.
        if tiles != 0 {
            launch!(self, &kname("gemms", ty), cfg(grid, 2 * GEMM_BM as u32, 0),
                    tiles, wd, nd as i32, x, x_stride as i32, xrow, y, y_stride as i32, rows as i32, (cols / 64) as i32, act, w_scale);
        }
        Ok(())
    }

    /// Decode attention: `attn_decode_gq` plus `attn_merge`, which also writes `out` quantized into xq.
    fn attn(&self, q: u64, kc: u64, vc: u64, qg: u64, out: u64, part: u64, step: u64, nh: usize, nkv: usize, hd: usize,
            scale: f32, gq_min: i32, nt: usize, xq: u64, sel: u64, sst: usize, sink: u64, win: usize, kvt: u64, ns1: usize) -> anyhow::Result<()> {
        if ![64, 128, 256, 512].contains(&hd) { bail!("attention head_dim {hd} unsupported (64, 128, 256 or 512)"); }
        let group = nh / nkv;
        let g = [8, 4, 2, 1].into_iter().find(|&g| group % g == 0 && g * hd <= 2048).unwrap();
        let ns = if kvt != 0 { ns1 } else { ATT_SPLITS };
        let pz = (nh * ns * (hd + 2)) as i32;
        if gq_min != 0 {
            launch!(self, &format!("attn_decode_{hd}"), cfg(((nh * nt) as u32, ns1 as u32), 256, (8 * hd * 4) as u32),
                    q, kc, vc, part, step, nh as i32, nkv as i32, hd as i32, scale, gq_min, pz, nt as i32, sel, sst as i32, sink, win as i32, kvt);
        }
        // No cp.async double buffering here: that path is the sm_80+ one (`ATT_RUN_CP`), so the K/V tile comes
        // through registers and the kernel needs no extra shared memory.
        let g4 = false;
        let (name, cp) = if g4 { ("attn_decode_gqa4_128".to_string(), 0) } else { (format!("attn_decode_gqa_{hd}"), 0) };
        if gq_min != i32::MAX {
            launch!(self, &name, cfg(((nh / g * nt) as u32, ns as u32), 256, (8 * hd * 4 + cp) as u32),
                    q, kc, vc, part, step, nh as i32, nkv as i32, hd as i32, scale, gq_min, pz, nt as i32, sel, sst as i32, sink, win as i32, kvt);
        }
        launch!(self, "attn_merge", LaunchConfig { grid_dim: (nh as u32, 1, nt as u32), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                part, qg, out, hd as i32, ns as i32, step, ns1 as i32, gq_min, xq, sink);
        Ok(())
    }

    fn grid(n: usize) -> LaunchConfig { cfg((n.div_ceil(256) as u32, 1), 256, 0) }
}

/// Pinned host allocation freed with the runtime that made it.
struct PinnedHost { hip: Arc<ffi::Hip>, p: *mut c_void }

unsafe impl Send for PinnedHost {}
unsafe impl Sync for PinnedHost {}

impl Drop for PinnedHost {
    fn drop(&mut self) { unsafe { (self.hip.host_free)(self.p) }; }
}

/// Pinned host memory used as a ring for `htod`: a region is free again once the copy that read it has completed.
struct Staging {
    /// Owns the pinned allocation; dropping it frees the ring.
    _pin: Buf,
    host: *mut u8,
    head: usize,
    busy: VecDeque<(usize, usize, Event)>,
}

impl Drop for Gpu {
    /// Unlock what `pin_host` locked, before the mapping it belongs to goes.
    fn drop(&mut self) {
        let _ = self.wait();
        for &(_, _, p) in self.pinned.read().unwrap().iter() { unsafe { (self.hip.host_unregister)(p as *mut c_void) }; }
    }
}

/// A `hipEvent_t` that destroys itself.
struct Event { hip: Arc<ffi::Hip>, ev: ffi::Event }

impl Event {
    fn new(hip: &Arc<ffi::Hip>) -> anyhow::Result<Event> {
        let mut ev: ffi::Event = std::ptr::null_mut();
        hip.check(unsafe { (hip.event_create)(&mut ev, ffi::EVENT_DISABLE_TIMING) }, "hipEventCreate")?;
        Ok(Event { hip: hip.clone(), ev })
    }
    fn record(&self, stream: ffi::Stream) -> anyhow::Result<()> {
        self.hip.check(unsafe { (self.hip.event_record)(self.ev, stream) }, "hipEventRecord")
    }
}

impl Drop for Event {
    fn drop(&mut self) { unsafe { (self.hip.event_destroy)(self.ev) }; }
}

/// Recorded graphs kept before a forward runs uncaptured (CUDA's cap, same reason).
const GRAPHS_MAX: usize = 32;

const RING: usize = 64 << 20;
/// Mark events of the copy queue (`Gpu::mark`).
const MARKS: usize = 64;
const RING_MIN: usize = 256 << 10;

fn alloc_raw(hip: &Arc<ffi::Hip>, bytes: usize) -> anyhow::Result<Buf> {
    let mut p: *mut c_void = std::ptr::null_mut();
    hip.check(unsafe { (hip.malloc)(&mut p, bytes.max(1)) }, "hipMalloc")?;
    // The Buf owns the allocation through its address; the free happens when the owner (the backend or the tensor)
    // drops, so the owner is a small RAII handle.
    struct DevAlloc { hip: Arc<ffi::Hip>, p: *mut c_void }
    unsafe impl Send for DevAlloc {}
    unsafe impl Sync for DevAlloc {}
    impl Drop for DevAlloc {
        fn drop(&mut self) { unsafe { (self.hip.free)(self.p) }; }
    }
    Ok(Buf::new(p as u64, bytes.max(1), DevAlloc { hip: hip.clone(), p }))
}

fn compile_cached(rtc: &ffi::Rtc, src: &str, arch: &str, tag: &str) -> anyhow::Result<Vec<u8>> {
    let dir = std::env::var_os("XDG_CACHE_HOME").map(Into::into)
        .unwrap_or_else(|| std::path::PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache"))
        .join("rusting_strata");
    let path = dir.join(format!("{:016x}.amdgcn", cache_key(src, arch, tag)));
    if let Ok(b) = std::fs::read(&path) { return Ok(b) }
    crate::say!("compiling HIP kernels for {arch} with hiprtc {tag} (first run only)");
    let b = rtc.compile(src, "kernels.cu", arch, &[])?;
    let _ = std::fs::create_dir_all(&dir).and_then(|_| std::fs::write(&path, &b));
    Ok(b)
}

/// Free VRAM as the kernel driver counts it, for the amdgpu card whose VRAM size is `total` (to 256 MiB). `hipMemGetInfo` counts
/// only compute allocations, so the desktop and other graphics clients look free to it and a budget taken from it
/// squeezes them out (the compositor then fails to pin its framebuffer). Vulkan's memory budget already counts them.
fn drm_free(total: usize) -> Option<usize> {
    let read = |p: std::path::PathBuf| std::fs::read_to_string(p).ok()?.trim().parse::<usize>().ok();
    std::fs::read_dir("/sys/class/drm").ok()?.flatten().find_map(|e| {
        let dev = e.path().join("device");
        let t = read(dev.join("mem_info_vram_total"))?;
        if t.abs_diff(total) > 256 << 20 { return None }
        Some(t.saturating_sub(read(dev.join("mem_info_vram_used"))?))
    })
}

/// The kernel names to load: `cuda::kernel_names(false)` has no int8-mma GEMM, which needs no separate list here.
fn kernel_names() -> Vec<String> { crate::cuda::kernel_names(false) }

impl Backend for Gpu {
    fn name(&self) -> &'static str { "hip" }

    fn device(&self) -> String { self.hip.device_name(self.device as i32) }

    fn alloc(&self, bytes: usize) -> anyhow::Result<Buf> {
        let b = alloc_raw(&self.hip, bytes)?;
        self.memset(&b)?;
        Ok(b)
    }

    /// `htod_on` the kernels' stream.
    fn htod(&self, src: &[u8], dst: &Buf, off: usize) -> anyhow::Result<()> { self.htod_on(self.stream, src, dst, off) }

    fn htod_side(&self, src: &[u8], dst: &Buf, off: usize) -> anyhow::Result<()> { self.htod_on(self.side, src, dst, off) }

    /// `hipHostRegister` of the pages under `data`, read-only when the runtime takes that flag (the model file is
    /// mapped read-only).
    fn pin_host(&self, data: &[u8]) -> anyhow::Result<bool> {
        const PAGE: usize = 4096;
        let (a, b) = (data.as_ptr() as usize, data.as_ptr() as usize + data.len());
        if self.pinned.read().unwrap().iter().any(|&(s, e, _)| s <= a && b <= e) { return Ok(true); }
        let (pa, pb) = (a & !(PAGE - 1), b.div_ceil(PAGE) * PAGE);
        let mut last = 0;
        for flags in [ffi::HOST_REGISTER_READ_ONLY | ffi::HOST_REGISTER_PORTABLE, ffi::HOST_REGISTER_PORTABLE] {
            last = unsafe { (self.hip.host_register)(pa as *mut c_void, pb - pa, flags) };
            if last == ffi::HIP_SUCCESS {
                self.pinned.write().unwrap().push((pa, pb, pa));
                return Ok(true);
            }
        }
        self.hip.check(last, "hipHostRegister").map(|_| false)
    }
    fn side_mark(&self) -> anyhow::Result<usize> { self.mark(self.side) }
    fn main_waits(&self, m: usize) -> anyhow::Result<()> { self.wait_mark(self.stream, m) }
    fn main_mark(&self) -> anyhow::Result<usize> { self.mark(self.stream) }
    fn side_waits(&self, m: usize) -> anyhow::Result<()> { self.wait_mark(self.side, m) }

    fn dtoh(&self, src: &Buf, off: usize, dst: &mut [u8]) -> anyhow::Result<()> {
        anyhow::ensure!(off + dst.len() <= src.len(), "dtoh: {} bytes at {off} from {}", dst.len(), src.len());
        self.wait()?;
        let r = unsafe { (self.hip.memcpy)(dst.as_mut_ptr().cast(), (src.addr() + off as u64) as *const c_void, dst.len(),
                                           ffi::MEMCPY_D2H) };
        self.hip.check(r, "hipMemcpyAsync(dtoh)")
    }

    fn dtod(&self, src: &Buf, src_off: usize, dst: &Buf, dst_off: usize, bytes: usize) -> anyhow::Result<()> {
        anyhow::ensure!(src_off + bytes <= src.len() && dst_off + bytes <= dst.len(), "dtod: {bytes} bytes out of range");
        let r = unsafe { (self.hip.memcpy_async)((dst.addr() + dst_off as u64) as *mut c_void,
                                                 (src.addr() + src_off as u64) as *const c_void, bytes, ffi::MEMCPY_D2D, self.stream) };
        self.hip.check(r, "hipMemcpyAsync(dtod)")
    }

    /// On the stream, so it is ordered against the copies and launches around it (the plain `hipMemset` goes to the
    /// null stream and can land after a copy that was queued for the same buffer).
    fn memset(&self, dst: &Buf) -> anyhow::Result<()> {
        self.hip.check(unsafe { (self.hip.memset_async)(dst.addr() as *mut c_void, 0, dst.len(), self.stream) }, "hipMemsetAsync")
    }

    fn mem_info(&self) -> anyhow::Result<(usize, usize)> {
        let (mut free, mut total) = (0usize, 0usize);
        self.hip.check(unsafe { (self.hip.mem_get_info)(&mut free, &mut total) }, "hipMemGetInfo")?;
        Ok((free.min(drm_free(total).unwrap_or(usize::MAX)), total))
    }

    fn sync(&self) -> anyhow::Result<()> { self.wait() }

    /// Recorded with `hipStreamBeginCapture`: the same launches, replayed by the driver with no host work.
    /// HIP refuses an allocation on a capturing stream ("operation not permitted when stream is capturing"), where
    /// CUDA's stream-ordered allocator allows it, so anything a step allocates has to be made before the capture (the
    /// rope table defers itself, see `rope_fr`). `STRATA_HIP_GRAPH=0` turns capture off.
    fn graph(&self, key: u64, f: &mut dyn FnMut() -> anyhow::Result<()>) -> anyhow::Result<()> {
        if *NO_GRAPH { return f() }
        if let Some(g) = self.graphs.lock().unwrap().get(&key) {
            let r = unsafe { (self.hip.graph_launch)(g.exec, self.stream) };
            return self.hip.check(r, "hipGraphLaunch");
        }
        // Each graph holds device memory and `step_rows` keys one per combination of sequence slots: past the cap a
        // forward runs uncaptured (the CUDA backend caps for the same reason).
        if self.graphs.lock().unwrap().len() >= GRAPHS_MAX { return f() }
        if self.capturing() { return f() }  // never nest captures
        self.hip.check(unsafe { (self.hip.stream_begin_capture)(self.stream, ffi::CAPTURE_MODE_THREAD_LOCAL) }, "hipStreamBeginCapture")?;
        let r = f();
        let mut graph: ffi::Graph = std::ptr::null_mut();
        let end = unsafe { (self.hip.stream_end_capture)(self.stream, &mut graph) };
        r?;
        self.hip.check(end, "hipStreamEndCapture")?;
        anyhow::ensure!(!graph.is_null(), "empty HIP graph");
        let mut exec: ffi::GraphExec = std::ptr::null_mut();
        self.hip.check(unsafe { (self.hip.graph_instantiate)(&mut exec, graph, 0) }, "hipGraphInstantiate")?;
        self.hip.check(unsafe { (self.hip.graph_launch)(exec, self.stream) }, "hipGraphLaunch")?;
        // A rope table the capture lacked: make it, and capture again next time (this graph computes it per call).
        let want = std::mem::take(&mut self.fr.lock().unwrap().1);
        if !want.is_empty() {
            for (base, n_rot) in want { self.rope_fr(base, n_rot)?; }
            return Ok(());
        }
        self.graphs.lock().unwrap().insert(key, Graph { exec, graph, hip: self.hip.clone() });
        Ok(())
    }

    fn graph_clear(&self) -> anyhow::Result<()> {
        self.sync()?;
        self.graphs.lock().unwrap().clear();
        Ok(())
    }

    /// Pinned host memory mapped into the device address space (the CPU expert worker's mailbox).
    fn host_mailbox(&self, words: usize) -> anyhow::Result<(Buf, *mut u32)> {
        let n = words.max(1);
        let mut host: *mut c_void = std::ptr::null_mut();
        self.hip.check(unsafe { (self.hip.host_malloc)(&mut host, 4 * n, ffi::HOST_MALLOC_PORTABLE | ffi::HOST_MALLOC_MAPPED) }, "hipHostMalloc")?;
        let mut dev: *mut c_void = std::ptr::null_mut();
        self.hip.check(unsafe { (self.hip.host_device_ptr)(&mut dev, host, 0) }, "hipHostGetDevicePointer")?;
        unsafe { std::ptr::write_bytes(host as *mut u8, 0, 4 * n) };
        let owner = PinnedHost { hip: self.hip.clone(), p: host };
        Ok((Buf::new(dev as u64, 4 * n, owner), host.cast()))
    }

    fn kv_words(&self, n: usize) -> usize { if self.kv_q8 { n * 17 / 32 } else { n } }

    // ---- kernels -------------------------------------------------------------------------------------------------

    fn gemm(&self, ty: GgmlType, rows: usize, cols: usize, tiles: u64, n_tiles: usize, x: u64, x_stride: usize, n_x: usize,
            xrow: u64, y: u64, y_stride: usize, act: i32) -> anyhow::Result<()> {
        self.gemm_raw(ty, rows, cols, tiles, n_tiles, 0, 0, x, x_stride, n_x, xrow, y, y_stride, act, 1.0f32)
    }

    fn gemm_dense(&self, w: &GpuTensor, n: usize, x: u64, x_stride: usize, y: u64, y_stride: usize, act: i32) -> anyhow::Result<()> {
        self.gemm_raw(w.ty, w.rows, w.cols, 0, n.div_ceil(GEMM_BM), w.ptr, n, x, x_stride, n, 0, y, y_stride, act, w.scale)
    }

    fn int8_gemm(&self, _ty: GgmlType) -> bool { false }

    fn gemv(&self, w: &GpuTensor, x: u64, y: u64) -> anyhow::Result<()> {
        let (ncols, kn) = Self::gemv_cols("gemv", w.ty, w.cols)?;
        let (c, rpw) = Self::gemv_cfg(w.ty, w.rows, w.cols, 1);
        launch!(self, &kn, c, w.ptr, x, y, w.rows as i32, ncols, rpw, 0u64, 0i64, 0i64, 0i32, w.scale, 1.0f32);
        Ok(())
    }

    fn gemvn(&self, w: &GpuTensor, x: u64, x_stride: usize, y: u64, y_stride: usize, nt: usize) -> anyhow::Result<()> {
        self.gemvq(w, x, x_stride, 0, y, y_stride, nt)
    }

    fn gemvq(&self, w: &GpuTensor, x: u64, x_stride: usize, xq: u64, y: u64, y_stride: usize, nt: usize) -> anyhow::Result<()> {
        self.gemvq_pair(w, x, x_stride, xq, y, y_stride, nt, (0, 0, 0, 1.0f32), 1)
    }

    fn gemvq2(&self, a: &GpuTensor, b: &GpuTensor, x: u64, x_stride: usize, xq: u64, ya: u64, yb: u64, nt: usize) -> anyhow::Result<()> {
        if a.ty != b.ty || a.cols != b.cols {
            self.gemvq(a, x, x_stride, xq, ya, a.rows, nt)?;
            return self.gemvq(b, x, x_stride, xq, yb, b.rows, nt);
        }
        let ((a, ya), (b, yb)) = if a.rows >= b.rows { ((a, ya), (b, yb)) } else { ((b, yb), (a, ya)) };
        self.gemvq_pair(a, x, x_stride, xq, ya, a.rows, nt, (b.ptr as i64 - a.ptr as i64, yb as i64 - ya as i64, b.rows as i32, b.scale), 2)
    }

    fn moe_gemv(&self, ty: GgmlType, rows: usize, cols: usize, w: u64, stride: usize, ids: u64, k: usize, x: u64, x_stride: usize,
                xg: usize, y: u64, pair: Option<(i64, i64)>, xq: u64) -> anyhow::Result<()> {
        let (ncols, kn) = Self::gemv_cols("moe_gemv", ty, cols)?;
        let (mut c, mut rpw) = Self::gemv_cfg(ty, rows, cols, k);
        if rpw == 1 && !Self::float_ty(ty) {
            rpw = 2;
            c.grid_dim.0 = rows.div_ceil(2 * (Self::gemv_block(k) as usize / 32)) as u32;
        }
        let (w2, y2) = pair.unwrap_or((0, 0));
        c.grid_dim.2 = if pair.is_some() { 2 } else { 1 };
        if xq != 0 { c.shared_mem_bytes = 0; }
        launch!(self, &kn, c, w, ids, stride as i64, x, x_stride as i32, xg as i32, y, rows as i32, ncols, rpw, w2, y2, xq);
        Ok(())
    }

    fn dequant(&self, w: &GpuTensor, rows: usize, out: u64) -> anyhow::Result<()> {
        launch!(self, &kname("dequant", w.ty), cfg((rows as u32, 1), 256, 0), w.ptr, out, Self::gemv_cols("dequant", w.ty, w.cols)?.0, w.scale);
        Ok(())
    }

    fn quant_xq(&self, x: u64, x_stride: usize, n: usize, nt: usize, out: u64) -> anyhow::Result<()> {
        launch!(self, "quant_xq", cfg((nt as u32, 1), 256, 0), x, x_stride as i32, n as i32, out);
        Ok(())
    }

    fn rms_norm(&self, x: u64, x_stride: usize, w: u64, out: u64, out_stride: usize, n: usize, rows: usize, eps: f32) -> anyhow::Result<()> {
        self.rms_norm_q(x, x_stride, w, out, out_stride, n, rows, eps, 0, 0)
    }

    fn rms_norm_q(&self, x: u64, x_stride: usize, w: u64, out: u64, out_stride: usize, n: usize, rows: usize, eps: f32, xq: u64, res: u64) -> anyhow::Result<()> {
        launch!(self, "rms_norm", cfg((rows as u32, 1), 512, 0), x, x_stride as i32, w, out, out_stride as i32, n as i32, eps, xq, res, 1i32);
        Ok(())
    }

    fn rms_norm_streams(&self, x: u64, w: u64, out: u64, n: usize, hc: usize, nt: usize, eps: f32) -> anyhow::Result<()> {
        launch!(self, "rms_norm", cfg(((nt * hc) as u32, 1), 512, 0), x, n as i32, w, out, n as i32, n as i32, eps, 0u64, 0u64, hc as i32);
        Ok(())
    }

    fn argmax_rows(&self, x: u64, n: usize, rows: usize, out: u64) -> anyhow::Result<()> {
        let part = self.ptr(&self.am);
        anyhow::ensure!(rows * AMAX_PARTS * 8 <= self.am.len(), "argmax_rows: {rows} rows");  // len is in bytes
        launch!(self, "argmax_part", cfg((AMAX_PARTS as u32, rows as u32), 256, 0), x, n as i32, part);
        launch!(self, "argmax_rows", cfg((rows as u32, 1), AMAX_PARTS as u32, 0), part, n as i32, out);
        Ok(())
    }

    fn l2_norm(&self, x: u64, n: usize, per: usize, stride: usize, rows: usize, eps: f32) -> anyhow::Result<()> {
        launch!(self, "l2_norm", cfg((rows as u32, 1), 128, 0), x, n as i32, per as i32, stride as i32, eps);
        Ok(())
    }

    fn rope_neox(&self, x: u64, stride: usize, heads: usize, per: usize, n_rot: usize, step: u64, base: f32) -> anyhow::Result<()> {
        self.rope_pairs(x, stride, heads, per, n_rot, step, base, 0, n_rot / 2)
    }

    fn rope_pairs(&self, x: u64, stride: usize, heads: usize, per: usize, n_rot: usize, step: u64, base: f32, lo: usize, hi: usize) -> anyhow::Result<()> {
        launch!(self, "rope_neox", cfg(((heads / per) as u32, 1), (n_rot / 2) as u32, 0), x, stride as i32, per as i32, n_rot as i32, step, base, lo as i32, hi as i32);
        Ok(())
    }

    fn rms_norm_rope(&self, x: u64, x_stride: usize, w: u64, out: u64, n: usize, rows: usize, eps: f32, per: usize, n_rot: usize, step: u64, base: f32) -> anyhow::Result<()> {
        self.rms_norm_rope_kv(x, x_stride, w, out, n, rows, eps, per, n_rot, step, base, 0, 0, 0)
    }

    fn rms_norm_rope_kv(&self, x: u64, x_stride: usize, w: u64, out: u64, n: usize, rows: usize, eps: f32, per: usize, n_rot: usize,
                        step: u64, base: f32, v: u64, kc: u64, vc: u64) -> anyhow::Result<()> {
        anyhow::ensure!(n_rot <= n.min(512) && n % 32 == 0, "rms_norm_rope: head of {n}, n_rot {n_rot}");
        let fr = self.rope_fr(base, n_rot)?;
        launch!(self, "rms_norm_rope", cfg((rows as u32, 1), 256, 0), x, x_stride as i32, w, out, n as i32, eps, per as i32, n_rot as i32,
                step, base, v, kc, vc, 0i32, 0u64, 0i32, 0u64, 0u64, 0i32, fr, 0u64, 0u64);
        Ok(())
    }

    fn rms_norm_rope_qk(&self, q: u64, q_stride: usize, qw: u64, qo: u64, qrows: usize, qper: usize, k: u64, kw: u64, ko: u64,
                        krows: usize, kper: usize, n: usize, eps: f32, n_rot: usize, step: u64, base: f32, v: u64, kc: u64, vc: u64) -> anyhow::Result<()> {
        anyhow::ensure!(n_rot <= n.min(512) && n % 32 == 0, "rms_norm_rope: head of {n}, n_rot {n_rot}");
        let fr = self.rope_fr(base, n_rot)?;
        launch!(self, "rms_norm_rope", cfg(((qrows + krows) as u32, 1), 256, 0), k, n as i32, kw, ko, n as i32, eps, kper as i32, n_rot as i32,
                step, base, v, kc, vc, qrows as i32, q, q_stride as i32, qw, qo, qper as i32, fr, 0u64, 0u64);
        Ok(())
    }

    fn kv_rows(&self) -> bool { true }

    fn rms_norm_rope_qk_rows(&self, q: u64, q_stride: usize, qw: u64, qo: u64, nh: usize, k: u64, kw: u64, ko: u64, nkv: usize,
                             hd: usize, eps: f32, n_rot: usize, step: u64, base: f32, v: u64, kvt: u64, nt: usize) -> anyhow::Result<()> {
        anyhow::ensure!(n_rot <= hd.min(512) && hd % 32 == 0, "rms_norm_rope: head of {hd}, n_rot {n_rot}");
        let fr = self.rope_fr(base, n_rot)?;
        launch!(self, "rms_norm_rope", cfg(((nt * (nh + nkv)) as u32, 1), 256, 0), k, hd as i32, kw, ko, hd as i32, eps, nkv as i32, n_rot as i32,
                step, base, v, 0u64, 0u64, (nt * nh) as i32, q, q_stride as i32, qw, qo, nh as i32, fr, kvt, 0u64);
        Ok(())
    }

    fn attn_rows(&self, q: u64, qg: u64, out: u64, part: u64, step: u64, nh: usize, nkv: usize, hd: usize, scale: f32, nt: usize,
                 sink: u64, win: usize, kvt: u64, splits: usize) -> anyhow::Result<()> {
        self.attn(q, 0, 0, qg, out, part, step, nh, nkv, hd, scale, 0, nt, 0, 0, 0, sink, win, kvt, splits)
    }

    fn add(&self, x: u64, y: u64, n: usize) -> anyhow::Result<()> {
        launch!(self, "add", Self::grid(n), x, y, n as i32);
        Ok(())
    }

    fn swiglu(&self, g: u64, u: u64, n: usize) -> anyhow::Result<()> {
        launch!(self, "swiglu", Self::grid(n), g, u, n as i32);
        Ok(())
    }

    fn swiglu_q(&self, g: u64, u: u64, n: usize, rows: usize, xq: u64) -> anyhow::Result<u64> {
        launch!(self, "swiglu_q", cfg((n.div_ceil(256) as u32, rows as u32), 32, 0), g, u, n as i32, xq);
        Ok(xq)
    }

    fn ssm_gates(&self, beta: u64, g: u64, dt_bias: u64, a: u64, n: usize, nv: usize) -> anyhow::Result<()> {
        launch!(self, "ssm_gates", Self::grid(n), beta, g, dt_bias, a, n as i32, nv as i32);
        Ok(())
    }

    fn conv_silu(&self, qkv: u64, state: u64, w: u64, y: u64, ch: usize, kw: usize, nn: usize, eps: f32) -> anyhow::Result<()> {
        launch!(self, "conv_silu", cfg((ch.div_ceil(128) as u32, 1), 128, 0), qkv, state, w, y, ch as i32, kw as i32, nn as i32, eps, 0u64);
        Ok(())
    }

    fn deltanet(&self, y: u64, s: u64, beta: u64, g: u64, dt_bias: u64, a: u64, o: u64, nk: usize, nv: usize, interleave: bool,
                dk: usize, dv: usize, scale: f32, t: usize, ych: usize, snap: u64, nsnap: usize) -> anyhow::Result<()> {
        let rows = 512 / dk.max(1);
        anyhow::ensure!(matches!(dk, 32 | 64 | 128 | 256) && dv % rows == 0, "deltanet: head dims {dk}/{dv} unsupported");
        launch!(self, "deltanet_seq", cfg((nv as u32, dv.div_ceil(8 * rows) as u32), 256, 0),
                y, s, beta, g, o, nk as i32, nv as i32, interleave as i32, dk as i32, dv as i32, scale, t as i32, ych as i32,
                snap, nsnap as i32, dt_bias, a, 0u64);
        Ok(())
    }

    fn conv_seq(&self, qkv: u64, state: u64, w: u64, y: u64, ch: usize, kw: usize, t: usize, snap: u64, nsnap: usize) -> anyhow::Result<()> {
        launch!(self, "conv_seq", Self::grid(t * ch), qkv, state, w, y, ch as i32, kw as i32, t as i32);
        launch!(self, "conv_state", Self::grid(ch), qkv, state, ch as i32, kw as i32, t as i32, snap, nsnap as i32);
        Ok(())
    }
    fn graph_drop(&self, key: u64) -> anyhow::Result<()> {
        self.graphs.lock().unwrap().remove(&key);
        Ok(())
    }

    fn situ(&self, y: u64, n: usize, rows: usize, scale: u64, beta: f32, lb: f32) -> anyhow::Result<()> {
        launch!(self, "situ", Self::grid(n * rows), y, n as i32, rows as i32, scale, beta, lb);
        Ok(())
    }

    fn situ_q(&self, y: u64, n: usize, rows: usize, s: u64, beta: f32, lb: f32, xq: u64) -> anyhow::Result<()> {
        anyhow::ensure!(n % 32 == 0, "situ_q: n ({n}) must be a multiple of 32");
        launch!(self, "situ_q", cfg((n.div_ceil(256) as u32, rows as u32), 32, 0), y, n as i32, s, beta, lb, xq);
        Ok(())
    }

    fn softcap(&self, x: u64, n: usize, cap: f32) -> anyhow::Result<()> {
        launch!(self, "softcap", Self::grid(n), x, n as i32, cap);
        Ok(())
    }

    fn hadamard(&self, x: u64, u: u64, s: u64, out: u64, n: usize, block: usize, rows: usize) -> anyhow::Result<()> {
        anyhow::ensure!(block <= 1024, "prism.hadamard.block_size {block}: at most 1024 on the GPU");
        launch!(self, "hadamard", cfg(((rows * n / block) as u32, 1), block as u32, 0), x, u, s, out, n as i32, block as i32, 1.0 / (block as f32).sqrt());
        Ok(())
    }

    // ---- Inkling 2 (the same kernels and launch shapes as src/cuda) ----

    fn ink_mix_n(&self, blocks: u64, nb: usize, cur: u64, q: u64, nq: usize, out: u64, n: usize, nt: usize) -> anyhow::Result<()> {
        launch!(self, "ink_mix", cfg((nq as u32, nt as u32), 256, 0), blocks, nb as i32, cur, q, out, n as i32, nt as i32);
        Ok(())
    }

    fn ink_ctrl_n(&self, down: u64, gain: u64, forget: u64, cur: u64, nins: usize, msg: u64, gate: u64, out: u64, n: usize, rank: usize, count: usize, nt: usize) -> anyhow::Result<()> {
        let g1 = LaunchConfig { grid_dim: (rank as u32, count as u32, nt as u32), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let g2 = LaunchConfig { grid_dim: (n.div_ceil(256) as u32, count as u32, nt as u32), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        launch!(self, "ink_ctrl_gate", g1, down, cur, nins as i32, msg, gate, n as i32, rank as i32, nt as i32);
        launch!(self, "ink_ctrl_out", g2, gain, forget, gate, cur, nins as i32, msg, out, n as i32, rank as i32, nt as i32);
        Ok(())
    }

    fn ink_conv(&self, cur: u64, w: u64, taps: u64, hist: u64, out: u64, n: usize, n_conv: usize, per: usize) -> anyhow::Result<()> {
        launch!(self, "ink_conv", Self::grid(n), cur, w, taps, hist, out, n as i32, n_conv as i32, per as i32);
        Ok(())
    }

    fn ink_conv_batch(&self, cur: u64, w: u64, taps: u64, hist: u64, out: u64, n: usize, n_conv: usize, per: usize, nt: usize) -> anyhow::Result<()> {
        anyhow::ensure!(cur != out, "ink_conv_batch: out must not alias cur");
        anyhow::ensure!(n_conv <= 8, "ink_conv_batch: n_conv={n_conv} > 8");
        launch!(self, "ink_conv_batch", cfg((n.div_ceil(256) as u32, nt as u32), 256, 0), cur, w, taps, hist, out, n as i32, n_conv as i32, per as i32, nt as i32, 0i32);
        launch!(self, "ink_conv_batch", Self::grid(n), cur, w, taps, hist, out, n as i32, n_conv as i32, per as i32, nt as i32, 1i32);
        Ok(())
    }

    fn ink_clip_n(&self, x: u64, scale: u64, w: usize, rows: usize, nt: usize, thr: f32, eps: f32, mul: f32,
                  dst: u64, dst_at0: usize, dst_ring: usize, pos0: usize, log_alpha: f32, log_floor: f32) -> anyhow::Result<()> {
        anyhow::ensure!(w <= 128, "ink_clip_n: w={w} > 128");
        anyhow::ensure!(crate::cuda::kv_mode() == 0 || dst == 0 || w % 32 == 0, "ink_clip_n: quantized KV requires w % 32 == 0 (got {w})");
        let ring = if dst != 0 { dst_ring.max(1) } else { 1 };
        let fl = if log_alpha == 0.0 && log_floor == 0.0 { 1.0 } else { log_floor };
        launch!(self, "ink_clip", cfg((rows as u32, nt as u32), 128, 0),
            x, scale, w as i32, thr, eps, mul, dst, dst_at0 as i32, ring as i32, pos0 as i32, log_alpha, fl, 0u64);
        Ok(())
    }

    fn ink_clip_pos(&self, x: u64, scale: u64, w: usize, rows: usize, nt: usize, thr: f32, eps: f32, mul: f32,
                    dst: u64, dst_ring: usize, pos_ptr: u64, log_alpha: f32, log_floor: f32) -> anyhow::Result<()> {
        anyhow::ensure!(w <= 128, "ink_clip_pos: w={w} > 128");
        anyhow::ensure!(crate::cuda::kv_mode() == 0 || dst == 0 || w % 32 == 0, "ink_clip_pos: quantized KV requires w % 32 == 0 (got {w})");
        let ring = if dst != 0 { dst_ring.max(1) } else { 1 };
        let fl = if log_alpha == 0.0 && log_floor == 0.0 { 1.0 } else { log_floor };
        launch!(self, "ink_clip", cfg((rows as u32, nt as u32), 128, 0),
            x, scale, w as i32, thr, eps, mul, dst, 0i32, ring as i32, 0i32, log_alpha, fl, pos_ptr);
        Ok(())
    }

    fn ink_attn(&self, q: u64, r: u64, sink: u64, gate: u64, k: u64, v: u64, bias: u64, sc: u64, out: u64, rows: usize, at: usize, nh: usize, nkv: usize, hd: usize, nr: usize, ext: usize) -> anyhow::Result<()> {
        let chunks = rows.div_ceil(64) as u32;
        launch!(self, "ink_attn", cfg((nh as u32, chunks), 128, 0), q, r, sink, k, v, bias, sc, rows as i32, at as i32, 1i32, nkv as i32, hd as i32, nr as i32, ext as i32, 0u64);
        launch!(self, "ink_attn_merge", cfg((nh as u32, 1), 128, 0), sc, gate, out, chunks as i32, hd as i32);
        Ok(())
    }

    fn ink_attn_pos(&self, q: u64, r: u64, sink: u64, gate: u64, k: u64, v: u64, bias: u64, sc: u64, out: u64, max_rows: usize, ring: usize, pos_ptr: u64, nh: usize, nkv: usize, hd: usize, nr: usize, ext: usize) -> anyhow::Result<()> {
        let chunks = max_rows.div_ceil(64) as u32;
        launch!(self, "ink_attn", cfg((nh as u32, chunks), 128, 0), q, r, sink, k, v, bias, sc, 0i32, 0i32, ring.max(1) as i32, nkv as i32, hd as i32, nr as i32, ext as i32, pos_ptr);
        launch!(self, "ink_attn_merge", cfg((nh as u32, 1), 128, 0), sc, gate, out, chunks as i32, hd as i32);
        Ok(())
    }

    fn ink_attn_prefill(&self, q: u64, r: u64, sink: u64, gate: u64, kc: u64, vc: u64, kc_new: u64, vc_new: u64, bias: u64, out: u64,
                        pos: usize, nt: usize, nh: usize, nkv: usize, hd: usize, nr: usize, ext: usize, ring: usize, win: usize) -> anyhow::Result<()> {
        let kn = match hd {
            32 => "ink_attn_prefill_32",
            64 => "ink_attn_prefill_64",
            128 => "ink_attn_prefill_128",
            _ => bail!("ink_attn_prefill: unsupported hd={hd}"),
        };
        launch!(self, kn, cfg((nt.div_ceil(4) as u32, nh as u32), 128, 0),
            q, r, sink, gate, kc, vc, kc_new, vc_new, bias, out,
            pos as i32, nt as i32, nh as i32, nkv as i32, nr as i32, ext as i32, ring.max(1) as i32, win as i32);
        Ok(())
    }

    fn ink_ffn_sum_n(&self, acc: u64, d: u64, wts: u64, ne: usize, n: usize, nt: usize, cpu: u64) -> anyhow::Result<()> {
        launch!(self, "ink_ffn_sum", cfg((n.div_ceil(256) as u32, nt as u32), 256, 0), acc, d, wts, ne as i32, n as i32, cpu);
        Ok(())
    }

    fn ink_pool(&self) -> bool { true }
    fn ink_slots(&self, ids: u64, wts: u64, table: u64, slots: u64, wh: u64, h: u64, mb: u64, nh: usize, h_off: usize, step: u64, k: usize) -> anyhow::Result<()> {
        anyhow::ensure!(k <= 32, "ink_slots: k={k}");
        launch!(self, "ink_slots", cfg((1, 1), 256, 0), ids, wts, table, slots, wh, h, mb, nh as i32, h_off as i32, step, k as i32);
        Ok(())
    }
    fn ink_ffn_sum_mb(&self, acc: u64, d: u64, wh: u64, k: usize, n: usize, mb: u64, y_off: usize, step: u64, slots: u64) -> anyhow::Result<()> {
        launch!(self, "ink_ffn_sum_mb", cfg((n.div_ceil(256) as u32, 1), 256, 0), acc, d, wh, k as i32, n as i32, mb, y_off as i32, step, slots);
        Ok(())
    }
    fn ink_route(&self, logits: u64, bias: u64, ids: u64, wts: u64, gu_sc: u64, dn_sc: u64, gsc_out: u64, n_expert: usize, k: usize, exp_scale: f32, rows: usize) -> anyhow::Result<()> {
        anyhow::ensure!(n_expert <= 512 && k <= 32, "ink_route: n_expert={n_expert} k={k}");
        launch!(self, "ink_route", cfg((rows as u32, 1), 32, 0), logits, bias, ids, wts, gu_sc, dn_sc, gsc_out, n_expert as i32, k as i32, exp_scale);
        Ok(())
    }

    fn dflash2_conv(&self, x: u64, dyn_w: u64, base: u64, out: u64, h: usize, gs: usize, k: usize, t: usize, part: usize) -> anyhow::Result<()> {
        launch!(self, "dflash2_conv", cfg((h.div_ceil(256) as u32, t as u32), 256, 0), x, dyn_w, base, out, h as i32, gs as i32, k as i32, t as i32, part as i32);
        Ok(())
    }

    fn attn_prefill_m(&self, q: u64, kc: u64, vc: u64, qg: u64, out: u64, step: u64, t: usize, nh: usize, nkv: usize, hd: usize,
                      scale: f32, qm: u64, mw: usize, sink: u64, win: usize) -> anyhow::Result<()> {
        if !matches!(hd, 64 | 128 | 256) { bail!("attention prefill: head_dim {hd} unsupported (64, 128 or 256)"); }
        // The tile depth has to be the one the kernel picks for itself (`FA_BK_FOR(HD)`, which takes 16 keys rather than
        // 32 at head_dim 256): 32 keys plus the score staging tile need 78 KiB, more than a 64 KiB workgroup may use,
        // while 16 keys need 58 KiB. A smaller depth here would under-allocate what the kernel addresses.
        let bk = if hd == 256 { 16 } else { FA_BK };
        // The HIP kernel stages its matrix-core scores through an extra FA_BQ x (bk + 8) f32 tile (see attn_prefill_t);
        // the qs/ks/vs/ps sizes the shared helper computes are the same as CUDA's.
        let smem = (attn_prefill_smem_bk(hd, bk) + 4 * crate::cuda::FA_BQ * (bk + 8)) as u32;
        anyhow::ensure!(smem as usize <= self.max_smem, "attn_prefill_{hd}: {smem} B of shared memory, {} allowed", self.max_smem);
        launch!(self, &format!("attn_prefill_{hd}"), cfg((t.div_ceil(64) as u32, nh as u32), 128, smem),
                q, kc, vc, qg, out, step, t as i32, nh as i32, nkv as i32, scale, qm, mw as i32, sink, win as i32);
        Ok(())
    }

    fn qsa_pool(&self, ik: u64, ring: u64, pool: u64, w: u64, step: u64, t: usize, n_rot: usize, base: f32, eps: f32) -> anyhow::Result<()> {
        launch!(self, "qsa_pool", cfg(((t / 4 + 1) as u32, 1), 128, 0), ik, ring, pool, w, step, t as i32, n_rot as i32, base, eps);
        Ok(())
    }

    fn qsa_select(&self, q: u64, pool: u64, score: u64, sel: u64, mask: u64, step: u64, r0: usize, rows: usize, nbs: usize, top: usize, sst: usize, mw: usize) -> anyhow::Result<()> {
        launch!(self, "qsa_score", cfg((64, rows.div_ceil(8) as u32), 256, 0), q, pool, score, step, r0 as i32, rows as i32, nbs as i32, top as i32);
        launch!(self, "qsa_select", cfg((rows as u32, 1), 1024, 0), score, sel, mask, step, r0 as i32, nbs as i32, top as i32, sst as i32, mw as i32);
        Ok(())
    }

    fn moe_combine_seq(&self, x: u64, ye: u64, wts: u64, inv: u64, k: usize, sh: u64, sg: u64, n: usize, t: usize) -> anyhow::Result<()> {
        launch!(self, "moe_combine_seq", Self::grid(n * t), x, ye, wts, inv, k as i32, sh, sg, n as i32, t as i32);
        Ok(())
    }

    fn norm_gate(&self, o: u64, w: u64, z: u64, n: usize, heads: usize, eps: f32) -> anyhow::Result<()> {
        self.norm_gate_q(o, w, z, n, heads, eps, 0).map(|_| ())
    }

    fn norm_gate_q(&self, o: u64, w: u64, z: u64, n: usize, heads: usize, eps: f32, xq: u64) -> anyhow::Result<u64> {
        let xq = if crate::cuda::llama_numerics() || n % 32 != 0 { 0 } else { xq };
        launch!(self, "norm_gate", cfg((heads as u32, 1), 128, 0), o, w, z, n as i32, eps, xq, 0i32);
        Ok(xq)
    }

    fn norm_gate_sig(&self, o: u64, w: u64, z: u64, n: usize, heads: usize, eps: f32, xq: u64) -> anyhow::Result<u64> {
        let xq = if crate::cuda::llama_numerics() || n % 32 != 0 { 0 } else { xq };
        launch!(self, "norm_gate", cfg((heads as u32, 1), 128, 0), o, w, z, n as i32, eps, xq, 1i32);
        Ok(xq)
    }

    fn hc_silu(&self, x: u64, sc: f32, n: usize) -> anyhow::Result<()> {
        launch!(self, "hc_silu", Self::grid(n), x, sc, n as i32);
        Ok(())
    }

    fn hc_mix(&self, xn: u64, g: u64, out: u64, n: usize, hc: usize, nt: usize) -> anyhow::Result<()> {
        launch!(self, "hc_mix", cfg((n.div_ceil(256) as u32, nt as u32), 256, 0), xn, g, out, n as i32, hc as i32);
        Ok(())
    }

    fn hc_write(&self, r: u64, out: u64, inj: u64, n: usize, hc: usize, nt: usize, clear: bool) -> anyhow::Result<()> {
        launch!(self, "hc_write", cfg((n.div_ceil(256) as u32, nt as u32), 256, 0), r, out, inj, n as i32, hc as i32, clear as i32);
        Ok(())
    }

    fn ple_gate(&self, key: u64, query: u64, value: u64, gated: u64, n: usize, hc: usize, nt: usize) -> anyhow::Result<()> {
        launch!(self, "ple_gate", cfg((hc as u32, nt as u32), 256, 0), key, query, value, gated, n as i32);
        Ok(())
    }

    fn ple_conv(&self, r: u64, gated: u64, norm: u64, kw: u64, ring: u64, step: u64, d: usize, nt: usize) -> anyhow::Result<()> {
        launch!(self, "ple_conv", Self::grid(d), r, gated, norm, kw, ring, step, d as i32, nt as i32);
        Ok(())
    }

    fn kv_store(&self, k: u64, v: u64, kc: u64, vc: u64, n: usize, t: usize, step: u64, hd: usize) -> anyhow::Result<()> {
        anyhow::ensure!(!self.kv_q8 || (hd % 32 == 0 && n % hd == 0), "q8 KV cache: head_dim {hd} is not a multiple of 32");
        launch!(self, "kv_store", Self::grid(n * t), k, v, kc, vc, n as i32, (n * t) as i32, step, hd as i32);
        Ok(())
    }

    fn attn_decode_gq(&self, q: u64, kc: u64, vc: u64, qg: u64, out: u64, part: u64, step: u64, nh: usize, nkv: usize, hd: usize,
                      scale: f32, gq_min: i32, nt: usize, sink: u64, win: usize) -> anyhow::Result<()> {
        self.attn(q, kc, vc, qg, out, part, step, nh, nkv, hd, scale, gq_min, nt, 0, 0, 0, sink, win, 0, ATT_SPLITS_1)
    }

    fn attn_decode_q(&self, q: u64, kc: u64, vc: u64, qg: u64, out: u64, part: u64, step: u64, nh: usize, nkv: usize, hd: usize,
                     scale: f32, gq_min: i32, nt: usize, xq: u64, sel: u64, sst: usize, sink: u64, win: usize) -> anyhow::Result<u64> {
        let xq = if crate::cuda::llama_numerics() { 0 } else { xq };
        self.attn(q, kc, vc, qg, out, part, step, nh, nkv, hd, scale, gq_min, nt, xq, sel, sst, sink, win, 0, ATT_SPLITS_1)?;
        Ok(xq)
    }

    fn router_topk(&self, p: u64, n: usize, k: usize, extra_scale: f32, ids: u64, wts: u64, t: usize, sigmoid: bool) -> anyhow::Result<()> {
        anyhow::ensure!(n <= 512 && k <= 32, "router_topk: {n} experts, top {k}");
        launch!(self, "router_topk", cfg((t as u32, 1), 32, 0), p, n as i32, k as i32, extra_scale, ids, wts, sigmoid as i32);
        Ok(())
    }

    fn moe_route(&self, p: u64, n: usize, k: usize, extra_scale: f32, table: u64, ids: u64, wts: u64, slots: u64, wh: u64, h: u64,
                 mb: u64, nh: usize, h_off: usize, step: u64, t: usize, pid: u64, sigmoid: bool) -> anyhow::Result<()> {
        anyhow::ensure!(n <= 512 && k <= 32, "moe_route: {n} experts, top {k}");
        launch!(self, "moe_route", cfg((1, 1), 256, 0), p, n as i32, k as i32, extra_scale, table, ids, wts, slots, wh, h, mb,
                nh as i32, h_off as i32, step, t as i32, pid, sigmoid as i32);
        Ok(())
    }

    fn moe_mailbox_optional(&self) -> bool { true }
    fn moe_prefetch_supported(&self) -> bool { false }

    fn moe_combine(&self, x: u64, ye: u64, w: u64, k: usize, sh: u64, sg: u64, mb: u64, y_off: usize, step: u64, slots: u64, n: usize, t: usize) -> anyhow::Result<()> {
        let mut c = Self::grid(n);
        c.grid_dim.1 = t as u32;
        launch!(self, "moe_combine", c, x, ye, w, k as i32, sh, sg, mb, y_off as i32, step, slots, n as i32);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole plumbing on the device: allocate, upload, launch a real kernel, read it back. `add` is the
    /// simplest kernel in the file, so a failure here is the backend, not the math.
    #[test]
    fn add_kernel_round_trips() {
        let g = match Gpu::new(0) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping: HIP unavailable: {e:#}");
                return;
            }
        };
        let n = 1024;
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| 0.5 * i as f32).collect();
        let want: Vec<f32> = (0..n).map(|i| 1.5 * i as f32).collect();
        let g: &dyn Backend = &g;
        let bx = g.alloc(4 * n).unwrap();
        let by = g.alloc(4 * n).unwrap();
        g.put(&x, &bx, 0).unwrap();
        g.put(&y, &by, 0).unwrap();
        // `Backend::add` is x += y, so the result lands in x.
        g.add(bx.addr(), by.addr(), n).unwrap();
        g.sync().unwrap();
        let got: Vec<f32> = g.get(&bx, 0, n).unwrap();
        assert!(got.iter().zip(&want).all(|(a, b)| a == b), "first mismatch at {:?}", got.iter().zip(&want).position(|(a, b)| a != b));
    }

    /// `argmax_rows` on 64 rows, the most a batch step has. The partials are 8 bytes each, AMAX_PARTS per row; the
    /// buffer used to be sized as if they were one byte, so rows past 8 wrote past its end.
    #[test]
    fn argmax_rows_takes_64_rows() {
        let Ok(gpu) = Gpu::new(0) else { eprintln!("skipping: no HIP device"); return };
        let g: &dyn Backend = &gpu;
        let (rows, n) = (64usize, 1000usize);
        // Each row is a permutation of 0..n, so its maximum is unique and the reference needs no tie rule.
        let x: Vec<f32> = (0..rows * n).map(|k| { let (r, i) = (k / n, k % n); ((i * 7919 + r * 31) % n) as f32 }).collect();
        let want: Vec<i32> = (0..rows).map(|r| {
            let row = &x[r * n..(r + 1) * n];
            let mut best = 0;
            for i in 1..n { if row[i] > row[best] { best = i; } }
            best as i32
        }).collect();
        let bx = g.alloc(4 * rows * n).unwrap();
        let bo = g.alloc(4 * rows).unwrap();
        g.put(&x, &bx, 0).unwrap();
        g.argmax_rows(bx.addr(), n, rows, bo.addr()).unwrap();
        g.sync().unwrap();
        let got: Vec<i32> = g.get(&bo, 0, rows).unwrap();
        assert_eq!(got, want);
    }

    /// `argmax_rows` on values 0..16, so most rows tie for the maximum: the first index wins, as in the CPU reference.
    #[test]
    fn argmax_rows_breaks_ties_at_the_first_index() {
        let Ok(gpu) = Gpu::new(0) else { eprintln!("skipping: no HIP device"); return };
        let g: &dyn Backend = &gpu;
        let (rows, n) = (64usize, 1000usize);
        let mut st = 77u32;
        let x: Vec<f32> = (0..rows * n).map(|_| { st = st.wrapping_mul(1664525).wrapping_add(1013904223); ((st >> 16) % 16) as f32 }).collect();
        let want: Vec<i32> = (0..rows).map(|r| {
            let row = &x[r * n..(r + 1) * n];
            let mut best = 0;
            for i in 1..n { if row[i] > row[best] { best = i; } }
            best as i32
        }).collect();
        assert!(want.iter().enumerate().any(|(r, &b)| x[r * n..(r + 1) * n].iter().filter(|&&v| v == x[r * n + b as usize]).count() > 1), "no row has a tie");
        let bx = g.alloc(4 * rows * n).unwrap();
        let bo = g.alloc(4 * rows).unwrap();
        g.put(&x, &bx, 0).unwrap();
        g.argmax_rows(bx.addr(), n, rows, bo.addr()).unwrap();
        g.sync().unwrap();
        let got: Vec<i32> = g.get(&bo, 0, rows).unwrap();
        assert_eq!(got, want);
    }

    /// `--backend hip` reaches this backend through `backend::open`, the path the CLI takes.
    #[test]
    fn opens_through_backend_open() {
        let Ok(b) = crate::backend::open(Some("hip"), 0) else { eprintln!("skipping: no HIP device"); return };
        assert_eq!(b.name(), "hip");
    }

    /// `norm_gate` with the quantize-to-int8 epilogue: the first kernel of the qwen35 attention layer, and the one
    /// that faulted first on the card, so it gets its own comparison against the reference math.
    #[test]
    fn norm_gate_matches_the_reference() {
        let g = match Gpu::new(0) { Ok(g) => g, Err(e) => { eprintln!("skipping: HIP unavailable: {e:#}"); return } };
        let (heads, n) = (4usize, 128usize);
        let mut st = 12345u32;
        let mut rnd = || { st = st.wrapping_mul(1664525).wrapping_add(1013904223); (st >> 8) as f32 / 16777216.0 - 0.5 };
        let o: Vec<f32> = (0..heads * n).map(|_| rnd()).collect();
        let w: Vec<f32> = (0..n).map(|_| rnd() + 1.0).collect();
        let z: Vec<f32> = (0..heads * n).map(|_| rnd()).collect();
        let want: Vec<f32> = (0..heads).flat_map(|h| {
            let row = &o[h * n..(h + 1) * n];
            let ss: f32 = row.iter().map(|v| v * v).sum();
            let sc = 1.0 / (ss / n as f32 + 1e-6).sqrt();
            let silu = |v: f32| v / (1.0 + (-v).exp());
            (0..n).map(|i| row[i] * sc * w[i] * silu(z[h * n + i])).collect::<Vec<_>>()
        }).collect();
        let gx = g.alloc(4 * heads * n).unwrap();
        let gw = g.alloc(4 * n).unwrap();
        let gz = g.alloc(4 * heads * n).unwrap();
        let xq_bytes = heads * n + heads * n / 32 * 4;
        let gxq = g.alloc(xq_bytes).unwrap();
        let g: &dyn Backend = &g;
        g.put(&o, &gx, 0).unwrap();
        g.put(&w, &gw, 0).unwrap();
        g.put(&z, &gz, 0).unwrap();
        g.norm_gate(gx.addr(), gw.addr(), gz.addr(), n, heads, 1e-6).unwrap();
        g.sync().unwrap();
        eprintln!("plain norm_gate ok: {:?}", &g.get::<f32>(&gx, 0, 2).unwrap());
        g.put(&o, &gx, 0).unwrap();
        g.norm_gate_q(gx.addr(), gw.addr(), gz.addr(), n, heads, 1e-6, gxq.addr()).unwrap();
        g.sync().unwrap();
        let got: Vec<f32> = g.get(&gx, 0, heads * n).unwrap();
        let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(worst < 2e-5, "max |diff| = {worst}");
    }

    /// `gemm_dense` against a CPU product, both tile regimes: the 16-position warp tile (`SM`) the small shapes take
    /// and the 32-position one, which is what the matrix-core path in `gemm_q` computes.
    #[test]
    fn gemm_dense_matches_cpu_in_both_tile_regimes() {
        let Ok(g) = Gpu::new(0) else { eprintln!("skipping: no HIP device"); return };
        for (rows, n, cols) in [(128usize, 64usize, 64usize), (8192, 128, 64)] {
            let mut st = 7u32;
            let mut rnd = || { st = st.wrapping_mul(1664525).wrapping_add(1013904223); (st >> 9) as f32 / 8388608.0 - 0.5 };
            let w: Vec<f32> = (0..rows * cols).map(|_| rnd()).collect();
            let x: Vec<f32> = (0..n * cols).map(|_| rnd()).collect();
            let mut want = vec![0f32; n * rows];
            for m in 0..n {
                for r in 0..rows {
                    let mut a = 0f32;
                    for k in 0..cols { a += x[m * cols + k] * w[r * cols + k]; }
                    want[m * rows + r] = a;
                }
            }
            let gw = g.alloc(4 * rows * cols).unwrap();
            let gx = g.alloc(4 * n * cols).unwrap();
            let gy = g.alloc(4 * n * rows).unwrap();
            let g: &dyn Backend = &g;
            g.put(&w, &gw, 0).unwrap();
            g.put(&x, &gx, 0).unwrap();
            let t = GpuTensor::new(GgmlType::F32, rows, cols, gw);
            g.gemm_dense(&t, n, gx.addr(), cols, gy.addr(), rows, 0).unwrap();
            g.sync().unwrap();
            let got: Vec<f32> = g.get(&gy, 0, n * rows).unwrap();
            // A batch step's flag picks no other kernel here, so it gives the same bits. The host masks the flag off
            // (`act & 3`), so this call stores over y. If the flag ever reached the kernel as `act` 8, the kernel would add
            // to the ones and the bit comparison below would fail.
            g.put(&vec![1f32; n * rows], &gy, 0).unwrap();
            g.gemm_dense(&t, n, gx.addr(), cols, gy.addr(), rows, crate::backend::BATCHED).unwrap();
            g.sync().unwrap();
            let batched: Vec<f32> = g.get(&gy, 0, n * rows).unwrap();
            assert!(got.iter().zip(&batched).all(|(a, b)| a.to_bits() == b.to_bits()), "rows {rows}: BATCHED changed the result");
            let mut worst = 0f32;
            let mut at = 0;
            for i in 0..n * rows {
                let d = (got[i] - want[i]).abs();
                if d > worst { worst = d; at = i; }
            }
            // f32 products summed over 64 terms: a few ulps either way, and f16 inputs quantize harder, so compare by
            // relative error against the magnitude of the row's terms.
            assert!(worst < 1e-2 * cols as f32, "rows {rows}: max |diff| = {worst} at {at} (got {}, want {})", got[at], want[at]);
        }
    }

    /// `attn_prefill_t` end to end (score product, causal softmax, P·V, the epilogue) against a CPU reference, at
    /// head_dim 128 with three query tiles and five key tiles and at head_dim 256 (the HIP build's 16-key tiles).
    /// A wrong lane/lane-staging mapping, a lost half of a fragment or a missing key shows up as a large per-element
    /// difference on a specific query or head — which is how the transposed-V bug below was found — rather than as
    /// the near-tie token flip a parity run sees.
    #[test]
    fn attn_prefill_matches_cpu() {
        use half::f16;
        let Ok(gpu) = Gpu::new(0) else { eprintln!("skipping: no HIP device"); return };
        let g: &dyn Backend = &gpu;
        let (nh, nkv) = (4usize, 2usize);
        for (hd, t) in [(128usize, 130usize), (256, 70)] {
            let scale = 1.0 / (hd as f32).sqrt();
            let mut st = 99u32;
            let mut rnd = || { st = st.wrapping_mul(1664525).wrapping_add(1013904223); (st >> 9) as f32 / 8388608.0 - 0.5 };
            let q: Vec<f32> = (0..t * nh * hd).map(|_| rnd()).collect();
            let kh: Vec<u16> = (0..t * nkv * hd).map(|_| f16::from_f32(rnd()).to_bits()).collect();
            let vh: Vec<u16> = (0..t * nkv * hd).map(|_| f16::from_f32(rnd()).to_bits()).collect();
            let half = |h: u16| f16::from_bits(h).to_f32();
            // Reference: the kernel scales Q before rounding it to f16, sums in f32, rounds the weights to f16 to
            // feed P·V, and divides by the row sum at the end (`out = o * inv` with `inv = sc / den`).
            let mut want = vec![0f32; t * nh * hd];
            for h in 0..nh {
                let kvh = h / (nh / nkv);
                for ti in 0..t {
                    let mut sc = vec![0f32; ti + 1];
                    for (j, s) in sc.iter_mut().enumerate() {
                        let mut a = 0f32;
                        for d in 0..hd {
                            a += half(f16::from_f32(q[ti * nh * hd + h * hd + d] * scale).to_bits())
                                * half(kh[(j * nkv + kvh) * hd + d]);
                        }
                        *s = a;
                    }
                    let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let e: Vec<f32> = sc.iter().map(|s| (s - mx).exp()).collect();
                    let den: f32 = e.iter().sum();
                    for d in 0..hd {
                        let mut a = 0f32;
                        for (j, w) in e.iter().enumerate() {
                            a += half(f16::from_f32(w / den).to_bits()) * half(vh[(j * nkv + kvh) * hd + d]);
                        }
                        want[ti * nh * hd + h * hd + d] = a;
                    }
                }
            }
            let gq = g.alloc(4 * q.len()).unwrap();
            let gk = g.alloc(2 * kh.len()).unwrap();
            let gv = g.alloc(2 * vh.len()).unwrap();
            let go = g.alloc(4 * q.len()).unwrap();
            let gs = g.alloc(4).unwrap();
            g.put(&q, &gq, 0).unwrap();
            g.put(&kh, &gk, 0).unwrap();
            g.put(&vh, &gv, 0).unwrap();
            g.put(&[0i32], &gs, 0).unwrap();
            g.attn_prefill_m(gq.addr(), gk.addr(), gv.addr(), 0, go.addr(), gs.addr(), t, nh, nkv, hd, scale, 0, 0, 0, 0).unwrap();
            g.sync().unwrap();
            let got: Vec<f32> = g.get(&go, 0, q.len()).unwrap();
            let mut worst = 0f32;
            let (mut at, mut gv_) = (0, 0f32);
            for i in 0..q.len() {
                let d = (got[i] - want[i]).abs();
                if d > worst { worst = d; at = i; gv_ = got[i]; }
            }
            eprintln!("hd {hd} T {t}: max |diff| = {worst:e} at {at} (row {}, head {}, d {}): got {gv_} want {}",
                      at / (nh * hd), (at / hd) % nh, at % hd, want[at]);
            assert!(worst < 5e-3, "hd {hd}: max |diff| = {worst} at {at}");
        }
    }

    /// The generated source compiles for this device through hiprtc (the CUDA source with `STRATA_HIP` taken).
    #[test]
    fn hiprtc_compiles_the_kernels() {
        let Ok(rtc) = ffi::Rtc::get() else {
            eprintln!("skipping: no hiprtc");
            return;
        };
        let Ok(hip) = ffi::Hip::open() else {
            eprintln!("skipping: no HIP runtime");
            return;
        };
        let arch = hip.arch(0).unwrap();
        rtc.compile(&source(false), "kernels.cu", &arch, &[]).expect("hiprtc compiles kernels.cu for the device");
    }

    /// The gfx11 (RDNA3) WMMA branch compiles, whatever card is present: its intrinsics, operand width and
    /// `STRATA_WMMA_ROW` layout differ from gfx12's. Numerics on gfx11 need an RDNA3 card to check.
    #[test]
    fn hiprtc_compiles_the_kernels_for_gfx11() {
        let Ok(rtc) = ffi::Rtc::get() else {
            eprintln!("skipping: no hiprtc");
            return;
        };
        rtc.compile(&source(false), "kernels.cu", "gfx1100", &[]).expect("hiprtc compiles kernels.cu for gfx1100");
    }
}
