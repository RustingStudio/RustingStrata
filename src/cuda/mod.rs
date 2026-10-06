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
const KERNELS: [&str; 13] = [
    "rms_norm", "l2_norm", "rope_neox", "add", "swiglu", "ssm_gates", "conv_silu", "deltanet_step", "norm_gate",
    "kv_store", "attn_decode", "router_topk", "moe_combine",
];

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
        let stream = ctx.default_stream();
        let (maj, min) = ctx.compute_capability()?;
        let arch: &'static str = Box::leak(format!("sm_{maj}{min}").into_boxed_str());
        let module = ctx.load_module(ptx(&source(), arch)?).context("load PTX")?;
        let mut fns = HashMap::new();
        let names = KERNELS.iter().map(|s| s.to_string())
            .chain(QTYPES.iter().flat_map(|&t| ["dequant", "gemv", "moe_gemv"].map(|p| kname(p, t))));
        for n in names {
            let f = module.load_function(&n).with_context(|| format!("kernel {n}"))?;
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

    /// y = W x (all rows of `w`).
    pub fn gemv(&self, w: &GpuTensor, x: u64, y: u64) -> anyhow::Result<()> {
        let nsb = Self::nsb(w)?;
        launch!(self, &kname("gemv", w.ty), cfg((w.rows.div_ceil(8) as u32, 1), 256, 0), w.ptr, x, y, w.rows as i32, nsb);
        Ok(())
    }

    /// For k slots: y[k] = W_{ids[k]} x[k], where `w` is an [cols, rows, n_expert] expert tensor.
    pub fn moe_gemv(&self, w: &GpuTensor, n_expert: usize, ids: u64, k: usize, x: u64, x_stride: usize, y: u64) -> anyhow::Result<()> {
        let rows = w.rows / n_expert;
        let nsb = Self::nsb(w)?;
        launch!(self, &kname("moe_gemv", w.ty), cfg((rows.div_ceil(8) as u32, k as u32), 256, 0),
            w.ptr, ids, x, x_stride as i32, y, rows as i32, nsb);
        Ok(())
    }

    /// `rows` rows: out[r] = rmsnorm(x[r]) * w.
    pub fn rms_norm(&self, x: u64, x_stride: usize, w: u64, out: u64, out_stride: usize, n: usize, rows: usize, eps: f32) -> anyhow::Result<()> {
        launch!(self, "rms_norm", cfg((rows as u32, 1), 256, 0), x, x_stride as i32, w, out, out_stride as i32, n as i32, eps);
        Ok(())
    }

    pub fn l2_norm(&self, x: u64, n: usize, rows: usize, eps: f32) -> anyhow::Result<()> {
        launch!(self, "l2_norm", cfg((rows as u32, 1), 128, 0), x, n as i32, eps);
        Ok(())
    }

    pub fn rope_neox(&self, x: u64, stride: usize, heads: usize, n_rot: usize, pos: usize, base: f32) -> anyhow::Result<()> {
        launch!(self, "rope_neox", cfg((heads as u32, 1), (n_rot / 2) as u32, 0), x, stride as i32, n_rot as i32, pos as i32, base);
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

    pub fn ssm_gates(&self, beta: u64, g: u64, dt_bias: u64, a: u64, n: usize) -> anyhow::Result<()> {
        launch!(self, "ssm_gates", Self::grid(n), beta, g, dt_bias, a, n as i32);
        Ok(())
    }

    pub fn conv_silu(&self, qkv: u64, state: u64, w: u64, y: u64, ch: usize, kw: usize) -> anyhow::Result<()> {
        launch!(self, "conv_silu", Self::grid(ch), qkv, state, w, y, ch as i32, kw as i32);
        Ok(())
    }

    pub fn deltanet_step(&self, y: u64, s: u64, beta: u64, g: u64, o: u64, nk: usize, nv: usize, dk: usize, dv: usize, scale: f32) -> anyhow::Result<()> {
        launch!(self, "deltanet_step", cfg((nv as u32, 1), dv as u32, (2 * dk * 4) as u32),
            y, s, beta, g, o, nk as i32, dk as i32, dv as i32, scale);
        Ok(())
    }

    pub fn norm_gate(&self, o: u64, w: u64, z: u64, n: usize, heads: usize, eps: f32) -> anyhow::Result<()> {
        launch!(self, "norm_gate", cfg((heads as u32, 1), 128, 0), o, w, z, n as i32, eps);
        Ok(())
    }

    pub fn kv_store(&self, k: u64, v: u64, kc: u64, vc: u64, n: usize, pos: usize) -> anyhow::Result<()> {
        launch!(self, "kv_store", Self::grid(n), k, v, kc, vc, n as i32, pos as i32);
        Ok(())
    }

    pub fn attn_decode(&self, q: u64, kc: u64, vc: u64, qg: u64, out: u64, n_pos: usize, nh: usize, nkv: usize, hd: usize, scale: f32) -> anyhow::Result<()> {
        if hd % 32 != 0 || hd > 512 { bail!("attention head_dim {hd} unsupported (multiple of 32, at most 512)"); }
        launch!(self, "attn_decode", cfg((nh as u32, 1), 256, (8 * hd * 4) as u32),
            q, kc, vc, qg, out, n_pos as i32, nh as i32, nkv as i32, hd as i32, scale);
        Ok(())
    }

    pub fn router_topk(&self, p: u64, n: usize, k: usize, extra_scale: f32, ids: u64, wts: u64) -> anyhow::Result<()> {
        launch!(self, "router_topk", cfg((1, 1), 32, 0), p, n as i32, k as i32, extra_scale, ids, wts);
        Ok(())
    }

    pub fn moe_combine(&self, x: u64, ye: u64, w: u64, k: usize, sh: u64, sg: u64, cpu: u64, n: usize) -> anyhow::Result<()> {
        launch!(self, "moe_combine", Self::grid(n), x, ye, w, k as i32, sh, sg, cpu, n as i32);
        Ok(())
    }
}
