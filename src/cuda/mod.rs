//! CUDA context, NVRTC compile with a PTX disk cache, and launch wrappers for `kernels.cu`.
//! One stream, event tracking off: buffers are passed to kernels as raw device pointers.

use crate::quant::{GgmlType, tables::*};
use anyhow::{Context, bail};
use crate::backend::{Backend, Buf};
use cudarc::driver::{CudaContext, CudaFunction, CudaGraph, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg, result, sys};
use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

const QTYPES: [GgmlType; 16] = {
    use GgmlType::*;
    [F32, BF16, Q8_0, Q2K, Q3K, Q4K, Q5K, Q6K, IQ1S, IQ1M, IQ2XXS, IQ2XS, IQ2S, IQ3XXS, IQ3S, IQ4XS]
};
const KERNELS: [&str; 26] = [
    "rms_norm", "quant_xq", "argmax_rows", "l2_norm", "rope_neox", "add", "swiglu", "ssm_gates", "conv_silu", "deltanet_seq", "norm_gate",
    "kv_store", "attn_decode", "attn_decode_gqa", "attn_merge", "router_topk", "moe_combine", "map_hits", "moe_publish", "conv_seq", "conv_state",
    "moe_combine_seq", "attn_prefill_64", "attn_prefill_128", "attn_prefill_256", "quant_q8",
];

pub use crate::backend::{ATT_GQA_MIN, ATT_SPLITS, GEMM_BM, GpuTensor};

/// Splits of the one-head-per-block decode attention used below ATT_GQA_MIN positions.
const ATT_SPLITS_1: usize = 16;

pub struct Gpu {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    fns: HashMap<String, CudaFunction>,
    /// q8 rows of the last integer GEMM's X (see quant_q8 in kernels.cu), grown on demand.
    q8: std::sync::Mutex<CudaSlice<u8>>,
    /// The kernels were compiled for a q8 KV cache (`--kv q8`).
    pub kv_q8: bool,
    /// Recorded graphs by `Backend::graph` key.
    graphs: std::sync::Mutex<HashMap<u64, CudaGraph>>,
}

/// `--llama-numerics`: quantize activations like llama.cpp's CPU q8_K (one scale per 256) for every integer matmul,
/// and run attention as its CPU flash attention does (LLAMA_Q8 in kernels.cu).
pub fn llama_numerics() -> bool { std::env::var_os("STRATA_LLAMA_NUMERICS").is_some() }

/// `--kv q8`: KV cache rows in ggml q8_0 form (KV_Q8 in kernels.cu), read once by Gpu::new.
fn kv_q8() -> bool { std::env::var_os("STRATA_KV_Q8").is_some() }

fn source(kv_q8: bool) -> String {
    let join = |v: Vec<String>| v.join(",");
    format!(
        "__device__ const unsigned long long iq2xs_grid[512] = {{{}}};\n\
         __device__ const unsigned long long iq2xxs_grid[256] = {{{}}};\n\
         __device__ const unsigned long long iq2s_grid[1024] = {{{}}};\n\
         __device__ const unsigned long long iq1s_grid[2048] = {{{}}};\n\
         __device__ const unsigned int iq3xxs_grid[256] = {{{}}};\n\
         __device__ const unsigned int iq3s_grid[512] = {{{}}};\n\
         __device__ const unsigned char ksigns_iq2xs[128] = {{{}}};\n\
         __device__ const signed char kvalues_iq4nl[16] = {{{}}};\n#define LLAMA_Q8 {}\n#define KV_Q8 {}\n{}",
        join(IQ2XS_GRID.iter().map(|v| format!("{v:#x}ULL")).collect()),
        join(IQ2XXS_GRID.iter().map(|v| format!("{v:#x}ULL")).collect()),
        join(IQ2S_GRID.iter().map(|v| format!("{v:#x}ULL")).collect()),
        join(IQ1S_GRID.iter().map(|v| format!("{v:#x}ULL")).collect()),
        join(IQ3XXS_GRID.iter().map(|v| format!("{v:#x}u")).collect()),
        join(IQ3S_GRID.iter().map(|v| format!("{v:#x}u")).collect()),
        join(KSIGNS_IQ2XS.iter().map(|v| v.to_string()).collect()),
        join(KVALUES_IQ4NL.iter().map(|v| v.to_string()).collect()),
        llama_numerics() as u8,
        kv_q8 as u8,
        include_str!("kernels.cu"),
    )
}

/// Compile `src` for `arch`, reusing `$XDG_CACHE_HOME/rusting_strata/<hash>.ptx` when present.
fn ptx(src: &str, arch: &'static str) -> anyhow::Result<Ptx> {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (src, arch).hash(&mut h);
    let dir = std::env::var_os("XDG_CACHE_HOME").map(Into::into)
        .unwrap_or_else(|| std::path::PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache"))
        .join("rusting_strata");
    let path = dir.join(format!("{:016x}.ptx", h.finish()));
    if let Ok(s) = std::fs::read_to_string(&path) {
        // Refresh mtime so pruning below keeps PTX that is still in use.
        let _ = std::fs::File::options().append(true).open(&path).and_then(|f| f.set_modified(std::time::SystemTime::now()));
        return Ok(Ptx::from_src(s));
    }
    prune_ptx(&dir, std::time::Duration::from_secs(7 * 86400));
    let opts = CompileOptions { arch: Some(arch), options: vec!["-std=c++17".into()], ..Default::default() };
    let p = compile_ptx_with_opts(src, opts).map_err(|e| anyhow::anyhow!("NVRTC compile of kernels.cu failed: {e}"))?;
    let _ = std::fs::create_dir_all(&dir).and_then(|_| std::fs::write(&path, p.to_src()));
    Ok(p)
}

/// Delete cached `.ptx` files in `dir` not used for `age`; every kernel edit leaves a stale ~4 MB file behind.
fn prune_ptx(dir: &std::path::Path, age: std::time::Duration) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        let old = e.metadata().and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok()).is_some_and(|d| d > age);
        if old && p.extension().is_some_and(|x| x == "ptx") { let _ = std::fs::remove_file(p); }
    }
}

pub fn cfg(grid: (u32, u32), block: u32, smem: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: (grid.0, grid.1, 1), block_dim: (block, 1, 1), shared_mem_bytes: smem }
}

/// Launch kernel `$name` with `cfg`; every argument is evaluated into a local so the builder can borrow it.
macro_rules! launch {
    ($g:expr, $name:expr, $cfg:expr, $($a:expr),* $(,)?) => {{
        let f = $g.f($name);
        let mut b = $g.stream.launch_builder(f);
        $( let v = $a; b.arg(&v); )*
        unsafe { b.launch($cfg) }.with_context(|| format!("launch {}", $name))?;
    }};
}

fn kname(prefix: &str, ty: GgmlType) -> String { format!("{prefix}_{}", ty.name().to_lowercase()) }

impl Gpu {
    pub fn new() -> anyhow::Result<Gpu> {
        let kv_q8 = kv_q8();
        anyhow::ensure!(!(kv_q8 && llama_numerics()), "--kv q8 does not combine with --llama-numerics (its attention reads f16 KV)");
        let ctx = CudaContext::new(0).context("CUDA init")?;
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream()?;
        let (maj, min) = ctx.compute_capability()?;
        let arch: &'static str = Box::leak(format!("sm_{maj}{min}").into_boxed_str());
        let module = ctx.load_module(ptx(&source(kv_q8), arch)?).context("load PTX")?;
        let mut fns = HashMap::new();
        let names = KERNELS.iter().map(|s| s.to_string())
            .chain(QTYPES.iter().flat_map(|&t| ["dequant", "gemv", "gemvn", "moe_gemv", "gemm"].map(|p| kname(p, t))))
            .chain(QTYPES.iter().filter(|&&t| !Self::float_ty(t)).map(|&t| kname("gemmi", t)));
        for n in names {
            let f = module.load_function(&n).with_context(|| format!("kernel {n}"))?;
            // gemvn caches nt quantized inputs: 4 x 17408 columns (Qwen3.8-27B ffn_down) need 77 KB.
            if n.starts_with("attn_prefill") || n.starts_with("gemvn") {
                f.set_attribute(cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 96 << 10)?;
            }
            fns.insert(n, f);
        }
        let q8 = std::sync::Mutex::new(stream.alloc_zeros::<u8>(1)?);
        Ok(Gpu { ctx, stream, fns, q8, kv_q8, graphs: Default::default() })
    }

    fn f(&self, name: &str) -> &CudaFunction { &self.fns[name] }

    pub fn ptr<T>(&self, s: &CudaSlice<T>) -> u64 { s.device_ptr(&self.stream).0 }

    pub fn zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(&self, n: usize) -> anyhow::Result<CudaSlice<T>> {
        Ok(self.stream.alloc_zeros::<T>(n.max(1))?)
    }

    /// First `rows` rows of `w`, dequantized on the GPU (test helper).
    pub fn dequant(&self, w: &GpuTensor, rows: usize) -> anyhow::Result<Vec<f32>> {
        let out = self.zeros::<f32>(rows * w.cols)?;
        launch!(self, &kname("dequant", w.ty), cfg((rows as u32, 1), 256, 0), w.ptr, self.ptr(&out), (w.cols / 256) as i32);
        Ok(self.stream.clone_dtoh(&out)?)
    }

    fn nsb(w: &GpuTensor) -> anyhow::Result<i32> {
        if w.cols % 256 != 0 { bail!("{} tensor with {} columns: GEMV needs a multiple of 256", w.ty.name(), w.cols); }
        Ok((w.cols / 256) as i32)
    }

    /// Whether the prefill GEMM for `ty` runs on int8 tensor cores (X quantized to int8 per 32) rather than f16.
    /// `STRATA_MMQ=q5_k,q6_k,..` overrides the default list.
    pub fn mmq(ty: GgmlType) -> bool {
        // IQ2_XS (and IQ2_S) has a scale per 16: its int8 path spends as long scaling mma results as in the mma, and loses to f16 (#31).
        !Self::float_ty(ty) && std::env::var("STRATA_MMQ").map(|v| v.split(',').any(|t| t == ty.name())).unwrap_or(!matches!(ty, GgmlType::IQ2XS | GgmlType::IQ2S) || llama_numerics())
    }

    fn gemm_raw(&self, ty: GgmlType, rows: usize, cols: usize, tiles: u64, n_tiles: usize, wd: u64, nd: usize, x: u64, x_stride: usize, n_x: usize, xrow: u64, y: u64, y_stride: usize, act: i32) -> anyhow::Result<()> {
        if cols % 256 != 0 { bail!("{} matrix with {cols} columns: GEMM needs a multiple of 256", ty.name()); }
        if n_tiles == 0 { return Ok(()); }
        let grid = (rows.div_ceil(64) as u32, n_tiles as u32);
        if !Self::mmq(ty) {
            launch!(self, &kname("gemm", ty), cfg(grid, 2 * GEMM_BM as u32, 0),
                tiles, wd, nd as i32, x, x_stride as i32, xrow, y, y_stride as i32, rows as i32, (cols / 256) as i32, act);
            return Ok(());
        }
        // ponytail: X is quantized on every call, also when several GEMMs read the same x; reuse if it shows in profiles.
        let mut q8 = self.q8.lock().unwrap();
        let need = n_x * (cols + cols / 8 + cols / 4);
        if q8.len() < need { *q8 = self.stream.alloc_zeros::<u8>(need)?; }
        let xq = self.ptr(&*q8);
        launch!(self, "quant_q8", cfg((n_x as u32, 1), 256, 0), x, x_stride as i32, xq, cols as i32);
        launch!(self, &kname("gemmi", ty), cfg(grid, 2 * GEMM_BM as u32, 0),
            tiles, wd, nd as i32, xq, xrow, y, y_stride as i32, rows as i32, (cols / 256) as i32, act);
        Ok(())
    }

    /// Shared memory for x quantized to int8 plus per-32 scales (quantized weight types only).
    fn xq_smem(ty: GgmlType, cols: usize) -> u32 {
        if Self::float_ty(ty) { 0 } else { (cols + cols / 8) as u32 }
    }

    fn float_ty(ty: GgmlType) -> bool { matches!(ty, GgmlType::F32 | GgmlType::BF16) }

    /// Rows per warp for the integer GEMV (env STRATA_RPW overrides). Experts: about 8 super-blocks of work per
    /// warp. Dense: 2 from 2048 rows, else 1 (fewer blocks hurt small matrices; measured in LOOP_LOG #50).
    fn rpw(rows: usize, cols: usize, k: usize) -> usize {
        // The vocab head: more rows per block amortize gemvn's per-block x quantization (nt = 2: 1158 -> 1027 us).
        let d = if k > 1 { (2048 / cols).max(1) } else if rows >= 65536 { 8 } else if rows >= 2048 { 2 } else { 1 };
        std::env::var("STRATA_RPW").ok().and_then(|v| v.parse().ok()).unwrap_or(d)
    }

    /// Launch config and rows-per-warp argument for a GEMV with `rows` x `cols` per matrix and `k` matrices.
    fn gemv_cfg(ty: GgmlType, rows: usize, cols: usize, k: usize) -> (LaunchConfig, i32) {
        // Float types: one warp per row and block, so small matrices (32 rows) still spread over many SMs.
        if Self::float_ty(ty) { return (cfg((rows as u32, k as u32), 32, 0), 1); }
        let rpw = Self::rpw(rows, cols, k);
        (cfg((rows.div_ceil(8 * rpw) as u32, k as u32), 256, Self::xq_smem(ty, cols)), rpw as i32)
    }

    /// Quantize `nt` rows of n floats (row t at x + 4 * t * x_stride bytes) for `gemvq` / `moe_gemv` into `out`
    /// (n + n / 8 bytes per row).
    pub fn quant_xq(&self, x: u64, x_stride: usize, n: usize, nt: usize, out: u64) -> anyhow::Result<()> {
        launch!(self, "quant_xq", cfg((nt as u32, 1), 256, 0), x, x_stride as i32, n as i32, out);
        Ok(())
    }

    fn grid(n: usize) -> LaunchConfig { cfg((n.div_ceil(256) as u32, 1), 256, 0) }

    /// `attn_decode` with an explicit GQA threshold (tests use 0 to force the GQA kernel).
    pub fn attn_decode_gq(&self, q: u64, kc: u64, vc: u64, qg: u64, out: u64, part: u64, step: u64, nh: usize, nkv: usize, hd: usize, scale: f32, gq_min: i32, nt: usize) -> anyhow::Result<()> {
        if ![64, 128, 256, 512].contains(&hd) { bail!("attention head_dim {hd} unsupported (64, 128, 256 or 512)"); }
        let group = nh / nkv;  // query heads per KV head, served by one block (up to 8, G * hd <= 2048)
        let g = [8, 4, 2, 1].into_iter().find(|&g| group % g == 0 && g * hd <= 2048).unwrap();
        let pz = (nh * ATT_SPLITS * (hd + 2)) as i32;
        launch!(self, "attn_decode", cfg(((nh * nt) as u32, ATT_SPLITS_1 as u32), 256, (8 * hd * 4) as u32), q, kc, vc, part, step, nh as i32, nkv as i32, hd as i32, scale, gq_min, pz, nt as i32);
        launch!(self, "attn_decode_gqa", cfg(((nh / g * nt) as u32, ATT_SPLITS as u32), 256, (8 * hd * 4) as u32), q, kc, vc, part, step, nh as i32, nkv as i32, hd as i32, scale, gq_min, pz, nt as i32);
        launch!(self, "attn_merge", LaunchConfig { grid_dim: (nh as u32, 1, nt as u32), block_dim: (256, 1, 1), shared_mem_bytes: 0 }, part, qg, out, hd as i32, ATT_SPLITS as i32, step, ATT_SPLITS_1 as i32, gq_min);
        Ok(())
    }
}

impl Backend for Gpu {
    fn alloc(&self, bytes: usize) -> anyhow::Result<Buf> {
        let s = self.stream.alloc_zeros::<u8>(bytes.max(1))?;
        Ok(Buf::new(self.ptr(&s), s.len(), s))
    }

    // Pageable copies, then a stream sync, as cudarc's memcpy_htod / clone_dtoh do.
    fn htod(&self, src: &[u8], dst: &Buf, off: usize) -> anyhow::Result<()> {
        anyhow::ensure!(off + src.len() <= dst.len, "htod: {} bytes at {off} into {}", src.len(), dst.len);
        self.ctx.bind_to_thread()?;
        unsafe { result::memcpy_htod_async(dst.addr + off as u64, src, self.stream.cu_stream()) }?;
        Ok(self.stream.synchronize()?)
    }

    fn dtoh(&self, src: &Buf, off: usize, dst: &mut [u8]) -> anyhow::Result<()> {
        anyhow::ensure!(off + dst.len() <= src.len, "dtoh: {} bytes at {off} from {}", dst.len(), src.len);
        self.ctx.bind_to_thread()?;
        unsafe { result::memcpy_dtoh_async(dst, src.addr + off as u64, self.stream.cu_stream()) }?;
        Ok(self.stream.synchronize()?)
    }

    fn dtod(&self, src: &Buf, src_off: usize, dst: &Buf, dst_off: usize, bytes: usize) -> anyhow::Result<()> {
        anyhow::ensure!(src_off + bytes <= src.len && dst_off + bytes <= dst.len, "dtod: {bytes} bytes out of range");
        self.ctx.bind_to_thread()?;
        Ok(unsafe { result::memcpy_dtod_async(dst.addr + dst_off as u64, src.addr + src_off as u64, bytes, self.stream.cu_stream()) }?)
    }

    fn memset(&self, dst: &Buf) -> anyhow::Result<()> {
        self.ctx.bind_to_thread()?;
        Ok(unsafe { result::memset_d8_async(dst.addr, 0, dst.len, self.stream.cu_stream()) }?)
    }

    fn mem_info(&self) -> anyhow::Result<(usize, usize)> { Ok(self.ctx.mem_get_info()?) }

    fn sync(&self) -> anyhow::Result<()> { Ok(self.stream.synchronize()?) }

    /// CUDA graph captured from the stream. `STRATA_NO_GRAPH` runs `f` every time instead.
    fn graph(&self, key: u64, f: &mut dyn FnMut() -> anyhow::Result<()>) -> anyhow::Result<()> {
        if let Some(g) = self.graphs.lock().unwrap().get(&key) { return Ok(g.launch()?); }
        if std::env::var_os("STRATA_NO_GRAPH").is_some() { return f(); }
        self.stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let r = f();
        let g = self.stream.end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)?;
        r?;
        let g = g.ok_or_else(|| anyhow::anyhow!("empty CUDA graph"))?;
        g.launch()?;
        self.graphs.lock().unwrap().insert(key, g);
        Ok(())
    }

    /// Pinned host memory mapped into the device address space.
    fn host_mailbox(&self, words: usize) -> anyhow::Result<(Buf, *mut u32)> {
        let n = words.max(1);
        let flags = sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE;
        let mut mem = unsafe { self.ctx.alloc_pinned_with_flags::<u32>(n, flags)? };
        let host = mem.as_mut_ptr()?;
        unsafe { std::ptr::write_bytes(host, 0, n) };
        let mut dev = 0;
        let r = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, host as *mut _, 0) };
        anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuMemHostGetDevicePointer: {r:?}");
        Ok((Buf::new(dev, 4 * n, mem), host))
    }

    /// Tensor-core GEMM over a tile list (see `GemmTile` in kernels.cu): for each tile, positions m0 .. m0 + cnt get
    /// y[m, 0..rows] = W_tile x[xrow[m]] (x row m when `xrow` is 0). `tiles` is a device array of [w, m0 | cnt << 32].
    /// `act`: 0 stores, 1 computes y = silu(y) * result (SwiGLU with the gate already in y), 2 adds to y.
    /// `n_x`: rows of x the tiles read (integer weight types quantize them to int8 first).
    fn gemm(&self, ty: GgmlType, rows: usize, cols: usize, tiles: u64, n_tiles: usize, x: u64, x_stride: usize, n_x: usize, xrow: u64, y: u64, y_stride: usize, act: i32) -> anyhow::Result<()> {
        self.gemm_raw(ty, rows, cols, tiles, n_tiles, 0, 0, x, x_stride, n_x, xrow, y, y_stride, act)
    }

    /// y[m] (op) W x[m] for positions m < n, x and y rows `x_stride` / `y_stride` floats apart.
    fn gemm_dense(&self, w: &GpuTensor, n: usize, x: u64, x_stride: usize, y: u64, y_stride: usize, act: i32) -> anyhow::Result<()> {
        self.gemm_raw(w.ty, w.rows, w.cols, 0, n.div_ceil(GEMM_BM), w.ptr, n, x, x_stride, n, 0, y, y_stride, act)
    }

    /// y = W x (all rows of `w`).
    fn gemv(&self, w: &GpuTensor, x: u64, y: u64) -> anyhow::Result<()> {
        let nsb = Self::nsb(w)?;
        let (c, rpw) = Self::gemv_cfg(w.ty, w.rows, w.cols, 1);
        launch!(self, &kname("gemv", w.ty), c, w.ptr, x, y, w.rows as i32, nsb, rpw);
        Ok(())
    }

    /// y[t] = W x[t] for t < nt (x[t] at x + 4 * t * x_stride bytes, y[t] at y + 4 * t * y_stride); each weight
    /// block is read once for all nt.
    fn gemvn(&self, w: &GpuTensor, x: u64, x_stride: usize, y: u64, y_stride: usize, nt: usize) -> anyhow::Result<()> {
        self.gemvq(w, x, x_stride, 0, y, y_stride, nt)
    }

    /// `gemvn` that reads x pre-quantized by `quant_xq` at `xq` (0 = quantize x per block; float weights read x).
    fn gemvq(&self, w: &GpuTensor, x: u64, x_stride: usize, xq: u64, y: u64, y_stride: usize, nt: usize) -> anyhow::Result<()> {
        if !(1..=4).contains(&nt) { bail!("gemvn: {nt} inputs (1 to 4)"); }
        let nsb = Self::nsb(w)?;
        let (mut c, rpw) = Self::gemv_cfg(w.ty, w.rows, w.cols, 1);
        c.shared_mem_bytes = if xq == 0 { c.shared_mem_bytes * nt as u32 } else { 0 };
        launch!(self, &kname("gemvn", w.ty), c, w.ptr, x, x_stride as i32, y, y_stride as i32, w.rows as i32, nsb, rpw, nt as i32, xq);
        Ok(())
    }

    /// For j < k with ids[j] >= 0: y[j] = W_j x[j / xg], where W_j is the `rows` x `cols` matrix at w + ids[j] * stride bytes.
    /// `pair` = (w2, y2) also computes the same product for the matrices at byte offset w2 into the output at byte offset y2.
    /// `xq` (not 0): x pre-quantized by `quant_xq` (k / xg rows), read in place of x.
    fn moe_gemv(&self, ty: GgmlType, rows: usize, cols: usize, w: u64, stride: usize, ids: u64, k: usize, x: u64, x_stride: usize, xg: usize, y: u64, pair: Option<(i64, i64)>, xq: u64) -> anyhow::Result<()> {
        if cols % 256 != 0 { bail!("{} expert with {cols} columns: GEMV needs a multiple of 256", ty.name()); }
        let (mut c, mut rpw) = Self::gemv_cfg(ty, rows, cols, k);
        // At least 2 rows per warp: an expert's x quantization (once per block) then covers 16 rows.
        if rpw == 1 && !Self::float_ty(ty) { rpw = 2; c.grid_dim.0 = rows.div_ceil(16) as u32; }
        let (w2, y2) = pair.unwrap_or((0, 0));
        c.grid_dim.2 = if pair.is_some() { 2 } else { 1 };
        if xq != 0 { c.shared_mem_bytes = 0; }
        launch!(self, &kname("moe_gemv", ty), c, w, ids, stride as i64, x, x_stride as i32, xg as i32, y, rows as i32, (cols / 256) as i32, rpw, w2, y2, xq);
        Ok(())
    }

    fn map_hits(&self, ids: u64, table: u64, slots: u64, wts: u64, wh: u64, k: usize) -> anyhow::Result<()> {
        launch!(self, "map_hits", cfg((1, 1), 32, 0), ids, table, slots, wts, wh, k as i32);
        Ok(())
    }

    /// `rows` rows: out[r] = rmsnorm(x[r]) * w.
    fn rms_norm(&self, x: u64, x_stride: usize, w: u64, out: u64, out_stride: usize, n: usize, rows: usize, eps: f32) -> anyhow::Result<()> {
        self.rms_norm_q(x, x_stride, w, out, out_stride, n, rows, eps, 0)
    }

    /// `rms_norm` that also writes its output quantized as `quant_xq` does to `xq` (when not 0).
    fn rms_norm_q(&self, x: u64, x_stride: usize, w: u64, out: u64, out_stride: usize, n: usize, rows: usize, eps: f32, xq: u64) -> anyhow::Result<()> {
        launch!(self, "rms_norm", cfg((rows as u32, 1), 512, 0), x, x_stride as i32, w, out, out_stride as i32, n as i32, eps, xq);
        Ok(())
    }

    /// out[r] = argmax of row r of x (`rows` rows of n floats), first index on ties.
    fn argmax_rows(&self, x: u64, n: usize, rows: usize, out: u64) -> anyhow::Result<()> {
        launch!(self, "argmax_rows", cfg((rows as u32, 1), 1024, 0), x, n as i32, out);
        Ok(())
    }

    /// Row r: vector r % per (n floats) of token r / per, tokens `stride` floats apart.
    fn l2_norm(&self, x: u64, n: usize, per: usize, stride: usize, rows: usize, eps: f32) -> anyhow::Result<()> {
        launch!(self, "l2_norm", cfg((rows as u32, 1), 128, 0), x, n as i32, per as i32, stride as i32, eps);
        Ok(())
    }

    /// `heads` rows of `stride` floats, `per` per token; token t is at position step[0] + t.
    fn rope_neox(&self, x: u64, stride: usize, heads: usize, per: usize, n_rot: usize, step: u64, base: f32) -> anyhow::Result<()> {
        launch!(self, "rope_neox", cfg(((heads / per) as u32, 1), (n_rot / 2) as u32, 0), x, stride as i32, per as i32, n_rot as i32, step, base);
        Ok(())
    }

    fn add(&self, x: u64, y: u64, n: usize) -> anyhow::Result<()> {
        launch!(self, "add", Self::grid(n), x, y, n as i32);
        Ok(())
    }

    fn swiglu(&self, g: u64, u: u64, n: usize) -> anyhow::Result<()> {
        launch!(self, "swiglu", Self::grid(n), g, u, n as i32);
        Ok(())
    }

    fn ssm_gates(&self, beta: u64, g: u64, dt_bias: u64, a: u64, n: usize, nv: usize) -> anyhow::Result<()> {
        launch!(self, "ssm_gates", Self::grid(n), beta, g, dt_bias, a, n as i32, nv as i32);
        Ok(())
    }

    fn conv_silu(&self, qkv: u64, state: u64, w: u64, y: u64, ch: usize, kw: usize) -> anyhow::Result<()> {
        launch!(self, "conv_silu", Self::grid(ch), qkv, state, w, y, ch as i32, kw as i32);
        Ok(())
    }

    /// Scan `t` tokens (rows of y `ych` floats apart) through the DeltaNet state `s`.
    /// `snap`: room for `nsnap` states; the state after each of the first min(nsnap, t - 1) tokens is kept there.
    fn deltanet(&self, y: u64, s: u64, beta: u64, g: u64, o: u64, nk: usize, nv: usize, dk: usize, dv: usize, scale: f32, t: usize, ych: usize, snap: u64, nsnap: usize) -> anyhow::Result<()> {
        let rows = 512 / dk.max(1);  // state rows per warp
        anyhow::ensure!(matches!(dk, 32 | 64 | 128 | 256) && dv % rows == 0, "deltanet: head dims {dk}/{dv} unsupported");
        launch!(self, "deltanet_seq", cfg((nv as u32, dv.div_ceil(8 * rows) as u32), 256, 0),
            y, s, beta, g, o, nk as i32, nv as i32, dk as i32, dv as i32, scale, t as i32, ych as i32, snap, nsnap as i32);
        Ok(())
    }

    /// Causal conv + SiLU over `t` tokens of `ch` channels, then roll the conv state forward (snapshots as in `deltanet`).
    fn conv_seq(&self, qkv: u64, state: u64, w: u64, y: u64, ch: usize, kw: usize, t: usize, snap: u64, nsnap: usize) -> anyhow::Result<()> {
        launch!(self, "conv_seq", Self::grid(t * ch), qkv, state, w, y, ch as i32, kw as i32, t as i32);
        launch!(self, "conv_state", Self::grid(ch), qkv, state, ch as i32, kw as i32, t as i32, snap, nsnap as i32);
        Ok(())
    }

    fn attn_prefill(&self, q: u64, kc: u64, vc: u64, qg: u64, out: u64, step: u64, t: usize, nh: usize, nkv: usize, hd: usize, scale: f32) -> anyhow::Result<()> {
        if !matches!(hd, 64 | 128 | 256) { bail!("attention prefill: head_dim {hd} unsupported (64, 128 or 256)"); }
        let smem = (2 * (64 + 32 + 32) * (hd + 8)) as u32;
        launch!(self, &format!("attn_prefill_{hd}"), cfg((t.div_ceil(64) as u32, nh as u32), 128, smem),
            q, kc, vc, qg, out, step, t as i32, nh as i32, nkv as i32, scale);
        Ok(())
    }

    fn moe_combine_seq(&self, x: u64, ye: u64, wts: u64, inv: u64, k: usize, sh: u64, sg: u64, n: usize, t: usize) -> anyhow::Result<()> {
        launch!(self, "moe_combine_seq", Self::grid(n * t), x, ye, wts, inv, k as i32, sh, sg, n as i32, t as i32);
        Ok(())
    }

    fn norm_gate(&self, o: u64, w: u64, z: u64, n: usize, heads: usize, eps: f32) -> anyhow::Result<()> {
        launch!(self, "norm_gate", cfg((heads as u32, 1), 128, 0), o, w, z, n as i32, eps);
        Ok(())
    }

    /// Store `t` tokens of `n` values at positions step[0] ...
    /// u16 words of a KV cache for `n` values: f16, or q8 with one f16 scale per 32 values.
    fn kv_words(&self, n: usize) -> usize { if self.kv_q8 { n * 17 / 32 } else { n } }

    fn kv_store(&self, k: u64, v: u64, kc: u64, vc: u64, n: usize, t: usize, step: u64, hd: usize) -> anyhow::Result<()> {
        // q8: each warp quantizes one 32-value block of one row.
        anyhow::ensure!(!self.kv_q8 || (hd % 32 == 0 && n % hd == 0), "q8 KV cache: head_dim {hd} is not a multiple of 32");
        launch!(self, "kv_store", Self::grid(n * t), k, v, kc, vc, n as i32, (n * t) as i32, step, hd as i32);
        Ok(())
    }

    /// Decode attention for `nt` query tokens, token t over positions 0 ..= step[2 t] (rows of q, qg, out and step
    /// per token, `part` holds `nh * ATT_SPLITS * (hd + 2)` floats per token). One launch serves all tokens, their
    /// blocks side by side, so they read the same K/V rows from L2. Below ATT_GQA_MIN positions one block serves one head
    /// (ATT_SPLITS_1 splits), from it on one block serves all query heads of a KV head (ATT_SPLITS splits). Both
    /// kernels are launched; the step decides which one works.
    fn attn_decode(&self, q: u64, kc: u64, vc: u64, qg: u64, out: u64, part: u64, step: u64, nh: usize, nkv: usize, hd: usize, scale: f32, nt: usize) -> anyhow::Result<()> {
        self.attn_decode_gq(q, kc, vc, qg, out, part, step, nh, nkv, hd, scale, ATT_GQA_MIN, nt)
    }

    /// Route `t` tokens (logit rows of n, ids / wts rows of k).
    fn router_topk(&self, p: u64, n: usize, k: usize, extra_scale: f32, ids: u64, wts: u64, t: usize) -> anyhow::Result<()> {
        anyhow::ensure!(n <= 512 && k <= 32, "router_topk: {n} experts, top {k}");
        launch!(self, "router_topk", cfg((t as u32, 1), 32, 0), p, n as i32, k as i32, extra_scale, ids, wts);
        Ok(())
    }

    /// Write routing and (on a miss) `h` to the layer mailbox at `mb` and raise its request flag.
    fn moe_publish(&self, ids: u64, wts: u64, slots: u64, h: u64, mb: u64, k: usize, n: usize, h_off: usize, step: u64, t: usize) -> anyhow::Result<()> {
        launch!(self, "moe_publish", cfg((1, 1), 256, 0), ids, wts, slots, h, mb, k as i32, n as i32, h_off as i32, step, t as i32);
        Ok(())
    }

    /// `mb` = 0: no CPU part. Otherwise adds the CPU result at word `y_off` of the mailbox once its done flag is `seq`.
    fn moe_combine(&self, x: u64, ye: u64, w: u64, k: usize, sh: u64, sg: u64, mb: u64, y_off: usize, step: u64, slots: u64, n: usize, t: usize) -> anyhow::Result<()> {
        let mut c = Self::grid(n);
        c.grid_dim.1 = t as u32;
        launch!(self, "moe_combine", c, x, ye, w, k as i32, sh, sg, mb, y_off as i32, step, slots, n as i32);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn prune_ptx_drops_only_stale_ptx() {
        let d = std::env::temp_dir().join(format!("strata_prune_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let day = std::time::Duration::from_secs(86400);
        for (n, ago) in [("old.ptx", 8), ("new.ptx", 1), ("old.txt", 8)] {
            let f = std::fs::File::create(d.join(n)).unwrap();
            f.set_modified(std::time::SystemTime::now() - day * ago).unwrap();
        }
        super::prune_ptx(&d, day * 7);
        let mut left: Vec<_> = std::fs::read_dir(&d).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
        left.sort();
        std::fs::remove_dir_all(&d).unwrap();
        assert_eq!(left, ["new.ptx", "old.txt"]);
    }
}
