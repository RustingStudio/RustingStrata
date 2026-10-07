//! CUDA context, NVRTC compile with a PTX disk cache, and launch wrappers for `kernels.cu`.
//! One stream, event tracking off: buffers are passed to kernels as raw device pointers.

use crate::quant::{GgmlType, tables::*};
use crate::weights::Tensor;
use anyhow::{Context, bail};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

const QTYPES: [GgmlType; 11] = {
    use GgmlType::*;
    [F32, BF16, Q8_0, Q2K, Q3K, Q4K, Q5K, Q6K, IQ2XS, IQ3XXS, IQ4XS]
};
const KERNELS: [&str; 22] = [
    "rms_norm", "l2_norm", "rope_neox", "add", "swiglu", "ssm_gates", "conv_silu", "deltanet_seq", "norm_gate",
    "kv_store", "attn_decode", "attn_merge", "router_topk", "moe_combine", "map_hits", "moe_publish", "conv_seq", "conv_state",
    "moe_combine_seq", "attn_prefill_64", "attn_prefill_128", "attn_prefill_256",
];

/// Blocks per head in decode attention (positions split across them).
pub const ATT_SPLITS: usize = 16;

/// Positions per GEMM tile (GEMM_BM in kernels.cu).
pub const GEMM_BM: usize = 128;

pub struct GpuTensor {
    pub ty: GgmlType,
    pub rows: usize,
    pub cols: usize,
    pub ptr: u64,
    _buf: CudaSlice<u8>,
}

impl GpuTensor {
    pub fn bytes(&self) -> usize { self.ty.row_bytes(self.cols as u64) * self.rows }
}

pub struct Gpu {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    fns: HashMap<String, CudaFunction>,
}

fn source() -> String {
    let join = |v: Vec<String>| v.join(",");
    format!(
        "__device__ const unsigned long long iq2xs_grid[512] = {{{}}};\n\
         __device__ const unsigned int iq3xxs_grid[256] = {{{}}};\n\
         __device__ const unsigned char ksigns_iq2xs[128] = {{{}}};\n\
         __device__ const signed char kvalues_iq4nl[16] = {{{}}};\n{}",
        join(IQ2XS_GRID.iter().map(|v| format!("{v:#x}ULL")).collect()),
        join(IQ3XXS_GRID.iter().map(|v| format!("{v:#x}u")).collect()),
        join(KSIGNS_IQ2XS.iter().map(|v| v.to_string()).collect()),
        join(KVALUES_IQ4NL.iter().map(|v| v.to_string()).collect()),
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
    if let Ok(s) = std::fs::read_to_string(&path) { return Ok(Ptx::from_src(s)); }
    let opts = CompileOptions { arch: Some(arch), options: vec!["-std=c++17".into()], ..Default::default() };
    let p = compile_ptx_with_opts(src, opts).map_err(|e| anyhow::anyhow!("NVRTC compile of kernels.cu failed: {e}"))?;
    let _ = std::fs::create_dir_all(&dir).and_then(|_| std::fs::write(&path, p.to_src()));
    Ok(p)
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
        let ctx = CudaContext::new(0).context("CUDA init")?;
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream()?;
        let (maj, min) = ctx.compute_capability()?;
        let arch: &'static str = Box::leak(format!("sm_{maj}{min}").into_boxed_str());
        let module = ctx.load_module(ptx(&source(), arch)?).context("load PTX")?;
        let mut fns = HashMap::new();
        let names = KERNELS.iter().map(|s| s.to_string())
            .chain(QTYPES.iter().flat_map(|&t| ["dequant", "gemv", "moe_gemv", "gemm"].map(|p| kname(p, t))));
        for n in names {
            let f = module.load_function(&n).with_context(|| format!("kernel {n}"))?;
            if n.starts_with("attn_prefill") {
                f.set_attribute(cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 96 << 10)?;
            }
            fns.insert(n, f);
        }
        Ok(Gpu { ctx, stream, fns })
    }

    fn f(&self, name: &str) -> &CudaFunction { &self.fns[name] }

    pub fn ptr<T>(&self, s: &CudaSlice<T>) -> u64 { s.device_ptr(&self.stream).0 }

    pub fn upload(&self, t: &Tensor) -> anyhow::Result<GpuTensor> {
        let buf = self.stream.clone_htod(t.data)?;
        Ok(GpuTensor { ty: t.ty, rows: t.rows, cols: t.cols, ptr: self.ptr(&buf), _buf: buf })
    }

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

    /// Tensor-core GEMM over a tile list (see `GemmTile` in kernels.cu): for each tile, positions m0 .. m0 + cnt get
    /// y[m, 0..rows] = W_tile x[xrow[m]] (x row m when `xrow` is 0). `tiles` is a device array of [w, m0 | cnt << 32].
    /// `act`: 0 stores, 1 computes y = silu(y) * result (SwiGLU with the gate already in y), 2 adds to y.
    pub fn gemm(&self, ty: GgmlType, rows: usize, cols: usize, tiles: u64, n_tiles: usize, x: u64, x_stride: usize, xrow: u64, y: u64, y_stride: usize, act: i32) -> anyhow::Result<()> {
        self.gemm_raw(ty, rows, cols, tiles, n_tiles, 0, 0, x, x_stride, xrow, y, y_stride, act)
    }

    /// y[m] (op) W x[m] for positions m < n, x and y rows `x_stride` / `y_stride` floats apart.
    pub fn gemm_dense(&self, w: &GpuTensor, n: usize, x: u64, x_stride: usize, y: u64, y_stride: usize, act: i32) -> anyhow::Result<()> {
        self.gemm_raw(w.ty, w.rows, w.cols, 0, n.div_ceil(GEMM_BM), w.ptr, n, x, x_stride, 0, y, y_stride, act)
    }

    fn gemm_raw(&self, ty: GgmlType, rows: usize, cols: usize, tiles: u64, n_tiles: usize, wd: u64, nd: usize, x: u64, x_stride: usize, xrow: u64, y: u64, y_stride: usize, act: i32) -> anyhow::Result<()> {
        if cols % 256 != 0 { bail!("{} matrix with {cols} columns: GEMM needs a multiple of 256", ty.name()); }
        if n_tiles == 0 { return Ok(()); }
        launch!(self, &kname("gemm", ty), cfg((rows.div_ceil(64) as u32, n_tiles as u32), 2 * GEMM_BM as u32, 0),
            tiles, wd, nd as i32, x, x_stride as i32, xrow, y, y_stride as i32, rows as i32, (cols / 256) as i32, act);
        Ok(())
    }

    /// Shared memory for x quantized to int8 plus per-32 scales (quantized weight types only).
    fn xq_smem(ty: GgmlType, cols: usize) -> u32 {
        if Self::float_ty(ty) { 0 } else { (cols + cols / 8) as u32 }
    }

    fn float_ty(ty: GgmlType) -> bool { matches!(ty, GgmlType::F32 | GgmlType::BF16) }

    /// Rows per warp for the integer GEMV: about 8 super-blocks of work per warp (env STRATA_RPW overrides).
    fn rpw(cols: usize) -> usize {
        std::env::var("STRATA_RPW").ok().and_then(|v| v.parse().ok()).unwrap_or((2048 / cols).max(1))
    }

    /// Launch config and rows-per-warp argument for a GEMV with `rows` x `cols` per matrix and `k` matrices.
    fn gemv_cfg(ty: GgmlType, rows: usize, cols: usize, k: usize) -> (LaunchConfig, i32) {
        let rpw = if Self::float_ty(ty) { 1 } else { Self::rpw(cols) };
        (cfg((rows.div_ceil(8 * rpw) as u32, k as u32), 256, Self::xq_smem(ty, cols)), rpw as i32)
    }

    /// y = W x (all rows of `w`).
    pub fn gemv(&self, w: &GpuTensor, x: u64, y: u64) -> anyhow::Result<()> {
        let nsb = Self::nsb(w)?;
        let (c, rpw) = Self::gemv_cfg(w.ty, w.rows, w.cols, 1);
        launch!(self, &kname("gemv", w.ty), c, w.ptr, x, y, w.rows as i32, nsb, rpw);
        Ok(())
    }

    /// For j < k with ids[j] >= 0: y[j] = W_j x[j], where W_j is the `rows` x `cols` matrix at w + ids[j] * stride bytes.
    pub fn moe_gemv(&self, ty: GgmlType, rows: usize, cols: usize, w: u64, stride: usize, ids: u64, k: usize, x: u64, x_stride: usize, y: u64) -> anyhow::Result<()> {
        if cols % 256 != 0 { bail!("{} expert with {cols} columns: GEMV needs a multiple of 256", ty.name()); }
        let (c, rpw) = Self::gemv_cfg(ty, rows, cols, k);
        launch!(self, &kname("moe_gemv", ty), c, w, ids, stride as i64, x, x_stride as i32, y, rows as i32, (cols / 256) as i32, rpw);
        Ok(())
    }

    pub fn map_hits(&self, ids: u64, table: u64, slots: u64, wts: u64, wh: u64, k: usize) -> anyhow::Result<()> {
        launch!(self, "map_hits", cfg((1, 1), 32, 0), ids, table, slots, wts, wh, k as i32);
        Ok(())
    }

    /// `rows` rows: out[r] = rmsnorm(x[r]) * w.
    pub fn rms_norm(&self, x: u64, x_stride: usize, w: u64, out: u64, out_stride: usize, n: usize, rows: usize, eps: f32) -> anyhow::Result<()> {
        launch!(self, "rms_norm", cfg((rows as u32, 1), 256, 0), x, x_stride as i32, w, out, out_stride as i32, n as i32, eps);
        Ok(())
    }

    /// Row r: vector r % per (n floats) of token r / per, tokens `stride` floats apart.
    pub fn l2_norm(&self, x: u64, n: usize, per: usize, stride: usize, rows: usize, eps: f32) -> anyhow::Result<()> {
        launch!(self, "l2_norm", cfg((rows as u32, 1), 128, 0), x, n as i32, per as i32, stride as i32, eps);
        Ok(())
    }

    /// `heads` rows of `stride` floats, `per` per token; token t is at position step[0] + t.
    pub fn rope_neox(&self, x: u64, stride: usize, heads: usize, per: usize, n_rot: usize, step: u64, base: f32) -> anyhow::Result<()> {
        launch!(self, "rope_neox", cfg((heads as u32, 1), (n_rot / 2) as u32, 0), x, stride as i32, per as i32, n_rot as i32, step, base);
        Ok(())
    }

    fn grid(n: usize) -> LaunchConfig { cfg((n.div_ceil(256) as u32, 1), 256, 0) }

    pub fn add(&self, x: u64, y: u64, n: usize) -> anyhow::Result<()> {
        launch!(self, "add", Self::grid(n), x, y, n as i32);
        Ok(())
    }

    pub fn swiglu(&self, g: u64, u: u64, n: usize) -> anyhow::Result<()> {
        launch!(self, "swiglu", Self::grid(n), g, u, n as i32);
        Ok(())
    }

    pub fn ssm_gates(&self, beta: u64, g: u64, dt_bias: u64, a: u64, n: usize, nv: usize) -> anyhow::Result<()> {
        launch!(self, "ssm_gates", Self::grid(n), beta, g, dt_bias, a, n as i32, nv as i32);
        Ok(())
    }

    pub fn conv_silu(&self, qkv: u64, state: u64, w: u64, y: u64, ch: usize, kw: usize) -> anyhow::Result<()> {
        launch!(self, "conv_silu", Self::grid(ch), qkv, state, w, y, ch as i32, kw as i32);
        Ok(())
    }

    /// Scan `t` tokens (rows of y `ych` floats apart) through the DeltaNet state `s`.
    pub fn deltanet(&self, y: u64, s: u64, beta: u64, g: u64, o: u64, nk: usize, nv: usize, dk: usize, dv: usize, scale: f32, t: usize, ych: usize) -> anyhow::Result<()> {
        anyhow::ensure!(dk % 32 == 0 && dk <= 256, "deltanet: head dim {dk} unsupported");
        launch!(self, "deltanet_seq", cfg((nv as u32, dv.div_ceil(8) as u32), 256, 0),
            y, s, beta, g, o, nk as i32, nv as i32, dk as i32, dv as i32, scale, t as i32, ych as i32);
        Ok(())
    }

    /// Causal conv + SiLU over `t` tokens of `ch` channels, then roll the conv state forward.
    pub fn conv_seq(&self, qkv: u64, state: u64, w: u64, y: u64, ch: usize, kw: usize, t: usize) -> anyhow::Result<()> {
        launch!(self, "conv_seq", Self::grid(t * ch), qkv, state, w, y, ch as i32, kw as i32, t as i32);
        launch!(self, "conv_state", Self::grid(ch), qkv, state, ch as i32, kw as i32, t as i32);
        Ok(())
    }

    pub fn attn_prefill(&self, q: u64, kc: u64, vc: u64, qg: u64, out: u64, step: u64, t: usize, nh: usize, nkv: usize, hd: usize, scale: f32) -> anyhow::Result<()> {
        if !matches!(hd, 64 | 128 | 256) { bail!("attention prefill: head_dim {hd} unsupported (64, 128 or 256)"); }
        let smem = (2 * ((64 + 32) * (hd + 8) + hd * 40)) as u32;
        launch!(self, &format!("attn_prefill_{hd}"), cfg((t.div_ceil(64) as u32, nh as u32), 128, smem),
            q, kc, vc, qg, out, step, t as i32, nh as i32, nkv as i32, scale);
        Ok(())
    }

    pub fn moe_combine_seq(&self, x: u64, ye: u64, wts: u64, inv: u64, k: usize, sh: u64, sg: u64, n: usize, t: usize) -> anyhow::Result<()> {
        launch!(self, "moe_combine_seq", Self::grid(n * t), x, ye, wts, inv, k as i32, sh, sg, n as i32, t as i32);
        Ok(())
    }

    pub fn norm_gate(&self, o: u64, w: u64, z: u64, n: usize, heads: usize, eps: f32) -> anyhow::Result<()> {
        launch!(self, "norm_gate", cfg((heads as u32, 1), 128, 0), o, w, z, n as i32, eps);
        Ok(())
    }

    /// Store `t` tokens of `n` values at positions step[0] ...
    pub fn kv_store(&self, k: u64, v: u64, kc: u64, vc: u64, n: usize, t: usize, step: u64) -> anyhow::Result<()> {
        launch!(self, "kv_store", Self::grid(n * t), k, v, kc, vc, n as i32, (n * t) as i32, step);
        Ok(())
    }

    /// Decode attention over positions 0 ..= step[0], split across ATT_SPLITS blocks per head; `part` holds
    /// `nh * ATT_SPLITS * (hd + 2)` floats.
    pub fn attn_decode(&self, q: u64, kc: u64, vc: u64, qg: u64, out: u64, part: u64, step: u64, nh: usize, nkv: usize, hd: usize, scale: f32) -> anyhow::Result<()> {
        if ![64, 128, 256, 512].contains(&hd) { bail!("attention head_dim {hd} unsupported (64, 128, 256 or 512)"); }
        launch!(self, "attn_decode", cfg((nh as u32, ATT_SPLITS as u32), 256, (8 * hd * 4) as u32),
            q, kc, vc, part, step, nh as i32, nkv as i32, hd as i32, scale);
        launch!(self, "attn_merge", cfg((nh as u32, 1), 256, 0), part, qg, out, hd as i32, ATT_SPLITS as i32);
        Ok(())
    }

    /// Route `t` tokens (logit rows of n, ids / wts rows of k).
    pub fn router_topk(&self, p: u64, n: usize, k: usize, extra_scale: f32, ids: u64, wts: u64, t: usize) -> anyhow::Result<()> {
        launch!(self, "router_topk", cfg((t as u32, 1), 32, 0), p, n as i32, k as i32, extra_scale, ids, wts);
        Ok(())
    }

    /// Write routing and (on a miss) `h` to the layer mailbox at `mb` and raise its request flag.
    pub fn moe_publish(&self, ids: u64, wts: u64, slots: u64, h: u64, mb: u64, k: usize, n: usize, h_off: usize, step: u64) -> anyhow::Result<()> {
        launch!(self, "moe_publish", cfg((1, 1), 256, 0), ids, wts, slots, h, mb, k as i32, n as i32, h_off as i32, step);
        Ok(())
    }

    /// `mb` = 0: no CPU part. Otherwise adds the CPU result at word `y_off` of the mailbox once its done flag is `seq`.
    pub fn moe_combine(&self, x: u64, ye: u64, w: u64, k: usize, sh: u64, sg: u64, mb: u64, y_off: usize, step: u64, slots: u64, n: usize) -> anyhow::Result<()> {
        launch!(self, "moe_combine", Self::grid(n), x, ye, w, k as i32, sh, sg, mb, y_off as i32, step, slots, n as i32);
        Ok(())
    }
}
