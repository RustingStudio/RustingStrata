//! CUDA context, NVRTC compile with a cubin disk cache, and launch wrappers for `kernels.cu`.
//! One stream (plus a side stream for `batch`), event tracking off: buffers are passed to kernels as raw device pointers.

use crate::quant::{GgmlType, tables::*};
use anyhow::{Context, bail};
use crate::backend::{Backend, Buf, SAME_X};
use cudarc::driver::{CudaContext, CudaEvent, CudaFunction, CudaGraph, CudaSlice, CudaStream, DevicePtr, PinnedHostSlice, LaunchConfig, PushKernelArg, result, sys};
use cudarc::nvrtc::Ptx;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

const QTYPES: [GgmlType; 22] = {
    use GgmlType::*;
    [F32, BF16, Q8_0, Q2K, Q3K, Q4K, Q5K, Q6K, IQ1S, IQ1M, IQ2XXS, IQ2XS, IQ2S, IQ3XXS, IQ3S, IQ4XS, IQ4NL, Q2_0, Q4_1, Q5_1, Q4_0, Q5_0]
};
const KERNELS: [&str; 35] = [
    "rope_freq", "qsa_pool", "qsa_score", "qsa_select", "rms_norm", "quant_xq", "argmax_part", "argmax_rows", "l2_norm", "rope_neox", "rms_norm_rope", "add", "swiglu", "swiglu_q", "ssm_gates", "conv_silu", "deltanet_seq", "norm_gate",
    "kv_store", "attn_merge", "router_topk", "moe_combine", "moe_route", "conv_seq", "conv_state",
    "moe_combine_seq", "attn_prefill_64", "attn_prefill_128", "attn_prefill_256", "quant_q8",
    "hc_silu", "hc_mix", "hc_write", "ple_gate", "ple_conv",
];

pub use crate::backend::{ATT_GQA_MIN, ATT_SPLITS, GEMM_BM, GpuTensor};
use crate::backend::ATT_SPLITS_1;

pub struct Gpu {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    /// Takes every second launch inside `batch`, so a captured graph runs them as parallel branches.
    side: Arc<CudaStream>,
    /// Launches so far in the current `batch` (usize::MAX outside one).
    nb: AtomicUsize,
    fns: HashMap<String, CudaFunction>,
    /// The int8 prefill GEMM (`gemmi_*`/`gemmis_*`) is compiled in and loaded: sm_80+ only (see `int8_gemm_arch`).
    i8_gemm: bool,
    /// q8 rows of the last integer GEMM's X (see quant_q8 in kernels.cu), grown on demand, and which x they hold
    /// (x, x_stride, n_x, cols) for a `SAME_X` call; None after a call without it that did not quantize.
    q8: std::sync::Mutex<(CudaSlice<u8>, Option<[u64; 4]>)>,
    /// The kernels were compiled for a q8 KV cache (`--kv q8`).
    pub kv_q8: bool,
    /// `argmax_rows` partials, AMAX_PARTS per row.
    am: CudaSlice<[i32; 2]>,
    /// Recorded graphs by `Backend::graph` key.
    graphs: std::sync::Mutex<HashMap<u64, Graph>>,
    /// Pinned staging for large `htod`, used as a ring: the copy returns before its DMA, so the host fills the next
    /// one while the GPU works (expert streaming in prefill). Head offset and the regions queued copies still read.
    ring: std::sync::Mutex<(PinnedHostSlice<u8>, usize, VecDeque<(usize, usize, CudaEvent)>)>,
    /// Recorded by `wait`.
    done: CudaEvent,
    /// The side stream after the last `beside`'s copies; the next `beside` makes the stream wait for it.
    lag: std::sync::Mutex<Option<CudaEvent>>,
    /// rope_freq tables by (base bits, n_rot), and the ones a graph capture asked for (see `rope_fr`).
    fr: std::sync::Mutex<(HashMap<(u32, usize), CudaSlice<f64>>, Vec<(f32, usize)>)>,
}

const RING: usize = 64 << 20;

/// Smaller copies take the driver's pageable path (staged before the call returns).
const RING_MIN: usize = 256 << 10;
/// `nb` inside `beside`: launches stay on the main stream, copies go to the side stream.
const BESIDE: usize = usize::MAX - 1;

/// A recorded graph. Its handles are plain CUDA driver handles, which any thread may launch once the context is
/// bound (cudarc leaves `CudaGraph` without `Send` only because it holds raw pointers).
struct Graph(CudaGraph);

const AMAX_PARTS: usize = 64;
unsafe impl Send for Graph {}

/// `--llama-numerics`: quantize activations like llama.cpp's CPU q8_K (one scale per 256) for every integer matmul,
/// and run attention as its CPU flash attention does (LLAMA_Q8 in kernels.cu).
pub fn llama_numerics() -> bool { std::env::var_os("STRATA_LLAMA_NUMERICS").is_some() }

/// `--kv q8`: KV cache rows in ggml q8_0 form (KV_Q8 in kernels.cu), read once by Gpu::new and Vk::new.
pub fn kv_q8() -> bool { std::env::var_os("STRATA_KV_Q8").is_some() }

/// `STRATA_GEMM_FMA`: force the CUDA-core FMA fallback in the f16 prefill kernels instead of the sm_70 tensor-core
/// path (`GEMM_TC` in kernels.cu). It covers both the f16 prefill GEMM and the prefill attention (`attn_prefill_t`):
/// the attention's P staging tile is part of every request, so the shared-memory size does not depend on it.
/// Ignored on sm_80+, which always uses `mma.m16n8k16`.
pub fn gemm_fma() -> bool { std::env::var_os("STRATA_GEMM_FMA").is_some() }

fn source(kv_q8: bool, gemm_tc: bool) -> String {
    let join = |v: Vec<String>| v.join(",");
    format!(
        "__device__ const unsigned long long iq2xs_grid[512] = {{{}}};\n\
         __device__ const unsigned long long iq2xxs_grid[256] = {{{}}};\n\
         __device__ const unsigned long long iq2s_grid[1024] = {{{}}};\n\
         __device__ const unsigned long long iq1s_grid[2048] = {{{}}};\n\
         __device__ const unsigned int iq3xxs_grid[256] = {{{}}};\n\
         __device__ const unsigned int iq3s_grid[512] = {{{}}};\n\
         __device__ const unsigned char ksigns_iq2xs[128] = {{{}}};\n\
         __device__ const signed char kvalues_iq4nl[16] = {{{}}};\n#define LLAMA_Q8 {}\n#define KV_Q8 {}\n#define GEMM_TC {}\n{}",
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
        gemm_tc as u8,
        include_str!("kernels.cu"),
    )
}

/// The NVRTC version `libloading` actually resolves, or `None` when the library is absent or `nvrtcVersion` fails.
/// Device-free: no `libcuda` call, so it also works in a CPU-only test.
fn nvrtc_version() -> Option<(i32, i32)> {
    if !unsafe { cudarc::nvrtc::sys::is_culib_present() } { return None; }
    // A library without `nvrtcVersion` makes cudarc's loader panic: turn that into `None` rather than crash.
    std::panic::catch_unwind(|| {
        let (mut major, mut minor) = (0, 0);
        let r = unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) };
        (r == cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS).then_some((major, minor))
    }).ok().flatten()
}

/// The user-facing error for a device `cc` with an NVRTC that cannot target it, `None` when the pair is fine.
/// CUDA 13 dropped Volta, so an sm_70 device must be compiled by NVRTC 12.x; saying it here keeps NVRTC's own
/// `invalid value for --gpu-architecture` from being the first thing the user sees.
fn arch_support_error(cc: (i32, i32), nvrtc: (i32, i32)) -> Option<String> {
    (cc.0 < 8 && nvrtc.0 >= 13).then(|| format!(
        "NVRTC {}.{} cannot compile for sm_{}{}: 13.x removed Volta (sm_70) support; install toolkit/NVRTC 12.x",
        nvrtc.0, nvrtc.1, cc.0, cc.1))
}

/// The startup line for the device: arch always, plus the sm_70 prefill GEMM path (tensor cores vs FMA fallback).
/// `gemm_fma` is passed explicitly so the log reflects the cubin actually compiled, not the env at print time.
fn device_log(device: usize, name: &str, arch: &str, cc: (i32, i32), gemm_fma: bool) -> String {
    let path = match (cc.0 < 8, gemm_fma) {
        (true, false) => " (Volta, m8n8k4)",
        (true, true) => " (Volta fallback)",
        (false, _) => "",  // sm_80+ always uses mma.m16n8k16
    };
    format!("cuda device {device}: {name}, {arch}{path}")
}

/// Whether the device has int8 tensor cores, i.e. whether the `gemmi_*`/`gemmis_*` prefill GEMM exists in its cubin
/// and may be launched. Volta (sm_70) does not: the s8 `mma` shapes need sm_80, so the kernels are compiled out of
/// the sm_70 build (see the `#if __CUDA_ARCH__ >= 800` guards in kernels.cu) and dropped from the loaded name list.
/// This is the CUDA launch gate only: `Gpu::mmq` stays the pure type policy shared with Vulkan and the tests.
fn int8_gemm_arch(cc: (i32, i32)) -> bool { cc.0 >= 8 }

/// The dynamic shared memory a launch of kernel `name` can request, with its family for the startup log: the fixed
/// maximum of the attention prefill (`2 * (64 + 32 + 32)` rows of `hd + 8` halves plus the 64 x (32 + 8) half P
/// staging tile of the sm_70 attention path, `attn_prefill_m`) and of the decode GQA kernels (8 query slices per
/// head group plus the cp.async K/V chunks at head_dim 128/256, `attn`), `usize::MAX` for `gemvn_*` whose request
/// depends on the tensor shape (`xq_smem * nt`, `gemv_cfg`), and `None` for every kernel that never asks for more
/// than the 48 KiB default (`attn_decode_*`, `attn_merge`, the rest).
fn dyn_shmem_need(name: &str) -> Option<(&'static str, usize)> {
    let family = if name.starts_with("attn_prefill") { "attn_prefill_*" }
        else if name.starts_with("attn_decode_gqa") { "attn_decode_gqa*" }
        else if name.starts_with("gemvn") { "gemvn_*" }
        else { return None };
    let need = match family {
        "attn_prefill_*" => 2 * (64 + 32 + 32) * (name["attn_prefill_".len()..].parse::<usize>().ok()? + 8)
            + 2 * 64 * (32 + 8),  // the sm_70 P staging tile, requested unconditionally
        "attn_decode_gqa*" => {
            let hd = name["attn_decode_gqa".len()..].trim_start_matches('4');
            let hd = hd.strip_prefix('_')?.parse::<usize>().ok()?;
            // attn_decode_gqa4_128 copies one K/V chunk, the generic kernel two: the larger value covers both, and
            // both are below the 48 KiB default at head_dim 64 and 128.
            let cp = if hd == 128 || hd == 256 { 2 * 2 * 32 * hd * 2 } else { 0 };
            8 * hd * 4 + cp
        }
        // The only shape-dependent request: picking the device ceiling keeps every legal launch possible.
        _ => usize::MAX,
    };
    Some((family, need))
}

/// The dynamic shared memory to opt in to for a function with `stat` bytes of static shared memory, or `None` to
/// leave the attribute alone because the 48 KiB default already covers the launch. `need` is clamped to the device
/// opt-in minus the static part: `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` cannot exceed what the
/// device allows for the block, and on Volta the opt-in ceiling is shared with the static usage (up to 2112 B in
/// `attn_decode_gqa_*`). A clamp below `need` makes the load succeed and a genuinely infeasible launch fail where
/// it is attempted, instead of the whole module failing to load.
fn dyn_shmem_optin(need: usize, stat: usize, optin: usize) -> Option<usize> {
    const DEFAULT: usize = 48 << 10;
    (need > DEFAULT).then(|| need.min(optin.saturating_sub(stat)))
}

/// One startup line per kernel family: the device's dynamic shared memory opt-in (not a table) and what the load set
/// for the family's largest static usage. Quotable in an issue or PR as the numbers actually observed on the card.
fn smem_log(family: &str, optin: usize, stat: usize, set: Option<usize>) -> String {
    let dynamic = set.map_or_else(|| "default (48 KiB)".to_string(), |v| format!("{v} B"));
    format!("cuda shared memory: opt-in {optin} B per block, {family}: {stat} B static, dynamic max {dynamic}")
}

/// The kernel names to load from the cubin: the full list on sm_80+, without the int8 prefill GEMM (`gemmi_*` and
/// `gemmis_*`, 20 non-float types x 2) on a device where that code is not compiled in. Pure so a CPU test can pin
/// both lists (282 / 242 names).
fn kernel_names(sm80: bool) -> Vec<String> {
    let int8 = move |&&t: &&GgmlType| sm80 && !matches!(t, GgmlType::F32 | GgmlType::BF16);
    KERNELS.iter().map(|s| s.to_string())
        .chain(QTYPES.iter().flat_map(|&t| ["dequant", "gemv", "gemvn", "moe_gemv", "gemm", "gemms"].map(|p| kname(p, t))))
        .chain(QTYPES.iter().flat_map(|&t| ["gemv", "gemvn", "moe_gemv"].map(|p| kname(p, t) + "_p")))
        .chain(QTYPES.iter().filter(int8).flat_map(|&t| [kname("gemmi", t), kname("gemmis", t)]))
        .chain([64, 128, 256, 512].iter().flat_map(|h| [format!("attn_decode_{h}"), format!("attn_decode_gqa_{h}")]))
        .chain(["attn_decode_gqa4_128".to_string()])
        .collect()
}

/// Cubin cache key: the compiled source, the target arch and the NVRTC version that produced it. The version is part
/// of it so a cubin written by another toolkit is never loaded (a 13.x sm_70 cubin must not hide the unsupported
/// toolkit, and machine code from an older compiler must not survive an upgrade). Detected NVRTC absent -> tag
/// `"unverified"`: any cached entry is then only reused without a version check, never in place of the missing-
/// library error below.
fn cache_key(src: &str, arch: &str, nvrtc: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (src, arch, nvrtc).hash(&mut h);
    h.finish()
}

/// Compile `src` to a cubin for `arch`, reusing `$XDG_CACHE_HOME/rusting_strata/<hash>.cubin` when present. A cubin,
/// unlike PTX, needs no driver JIT at load (~6 s per start whenever the driver's own JIT cache missed). The hash
/// covers `(src, arch, NVRTC version)`; when NVRTC cannot be loaded the tag is `"unverified"`, so a cache entry is
/// no longer reusable and the run reports the missing library instead of silently reusing a cubin.
fn ptx(src: &str, arch: &'static str, cc: (i32, i32)) -> anyhow::Result<Ptx> {
    let nvrtc = nvrtc_version();
    // A device/NVRTC mismatch is fatal: no "CUDA init" context, or `backend::open` and the tests swallow it as a skip.
    if let Some(nv) = nvrtc {
        if let Some(msg) = arch_support_error(cc, nv) {
            return Err(anyhow::anyhow!("{msg}"));
        }
    }
    let tag = nvrtc.map(|(maj, min)| format!("{maj}.{min}")).unwrap_or_else(|| "unverified".to_string());
    let dir = std::env::var_os("XDG_CACHE_HOME").map(Into::into)
        .unwrap_or_else(|| std::path::PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache"))
        .join("rusting_strata");
    let path = dir.join(format!("{:016x}.cubin", cache_key(src, arch, &tag)));
    if let Ok(b) = std::fs::read(&path) {
        // Refresh mtime so pruning below keeps a cubin that is still in use.
        let _ = std::fs::File::options().append(true).open(&path).and_then(|f| f.set_modified(std::time::SystemTime::now()));
        return Ok(Ptx::from_binary(b));
    }
    // cudarc panics on a missing library; "CUDA init" lets `backend::open` fall back to Vulkan.
    if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
        return Err(anyhow::anyhow!(
            "the NVRTC library (libnvrtc.so) is not installed; sm_70 needs the CUDA 12.x toolkit/NVRTC (13.x dropped Volta support)"
        )).context("CUDA init");
    }
    prune_ptx(&dir, std::time::Duration::from_secs(7 * 86400));
    eprintln!("compiling CUDA kernels for {arch} with NVRTC {tag} (first run only, ~15 s)");
    let b = cubin(src, arch)?;
    let _ = std::fs::create_dir_all(&dir).and_then(|_| std::fs::write(&path, &b));
    Ok(Ptx::from_binary(b))
}

fn cubin(src: &str, arch: &str) -> anyhow::Result<Vec<u8>> {
    use cudarc::nvrtc::{result as nr, sys as ns};
    let c = std::ffi::CString::new(src)?;
    let prog = nr::create_program(&c, None)?;
    let out = (|| unsafe {
        if let Err(e) = nr::compile_program(prog, &[format!("--gpu-architecture={arch}"), "-std=c++17".into()]) {
            let log = nr::get_program_log(prog).map(|l| std::ffi::CStr::from_ptr(l.as_ptr()).to_string_lossy().into_owned());
            bail!("NVRTC compile of kernels.cu failed: {e}\n{}", log.unwrap_or_default());
        }
        let mut n = 0;
        ns::nvrtcGetCUBINSize(prog, &mut n).result()?;
        let mut b = vec![0u8; n];
        ns::nvrtcGetCUBIN(prog, b.as_mut_ptr() as *mut std::ffi::c_char).result()?;
        Ok(b)
    })();
    unsafe { nr::destroy_program(prog) }?;
    out
}

/// Delete cached `.ptx`/`.cubin` files in `dir` not used for `age`; every kernel edit leaves a stale file behind.
fn prune_ptx(dir: &std::path::Path, age: std::time::Duration) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        let old = e.metadata().and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok()).is_some_and(|d| d > age);
        if old && p.extension().is_some_and(|x| x == "ptx" || x == "cubin") { let _ = std::fs::remove_file(p); }
    }
}

pub fn cfg(grid: (u32, u32), block: u32, smem: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: (grid.0, grid.1, 1), block_dim: (block, 1, 1), shared_mem_bytes: smem }
}

/// Launch kernel `$name` with `cfg`; every argument is evaluated into a local so the builder can borrow it.
macro_rules! launch {
    ($g:expr, $name:expr, $cfg:expr, $($a:expr),* $(,)?) => {{
        let f = $g.f($name);
        let mut b = $g.st().launch_builder(f);
        $( let v = $a; b.arg(&v); )*
        unsafe { b.launch($cfg) }.with_context(|| format!("launch {}", $name))?;
    }};
}

fn kname(prefix: &str, ty: GgmlType) -> String { format!("{prefix}_{}", ty.name().to_lowercase()) }

impl Gpu {
    pub fn new(device: usize) -> anyhow::Result<Gpu> {
        let kv_q8 = kv_q8();
        anyhow::ensure!(!(kv_q8 && llama_numerics()), "--kv q8 does not combine with --llama-numerics (its attention reads f16 KV)");
        if !unsafe { sys::is_culib_present() } {
            return Err(anyhow::anyhow!("no NVIDIA driver (libcuda.so)")).context("CUDA init");
        }
        let ctx = CudaContext::new(device).context("CUDA init")?;
        unsafe { ctx.disable_event_tracking() };
        let (maj, min) = ctx.compute_capability()?;
        let arch: &'static str = Box::leak(format!("sm_{maj}{min}").into_boxed_str());
        let gemm_fma = gemm_fma();
        eprintln!("{}", device_log(device, &ctx.name().unwrap_or_default(), arch, (maj, min), gemm_fma));
        let stream = ctx.new_stream()?;
        let module = ctx.load_module(ptx(&source(kv_q8, !gemm_fma), arch, (maj, min))?).context("load cubin")?;
        let i8_gemm = int8_gemm_arch((maj, min));
        // The per-block dynamic shared memory ceiling the driver reports: 96 KiB on Volta per the roadmap, but
        // possibly less (reserve), and shared with the function's static part. Never a constant.
        let optin = ctx.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)? as usize;
        let mut fns = HashMap::new();
        // Per family: the largest static usage seen and the largest value set, logged once after the load.
        let mut smem: BTreeMap<&'static str, (usize, Option<usize>)> = BTreeMap::new();
        for n in kernel_names(i8_gemm) {
            let f = module.load_function(&n).with_context(|| format!("kernel {n}"))?;
            if let Some((family, need)) = dyn_shmem_need(&n) {
                let stat = f.shared_size_bytes().with_context(|| format!("static shared memory of {n}"))?.max(0) as usize;
                let set = dyn_shmem_optin(need, stat, optin);
                if let Some(v) = set {
                    f.set_attribute(sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, v as i32)
                        .with_context(|| format!("set dynamic shared memory on {n}: {stat} B static, {optin} B device opt-in"))?;
                }
                let e = smem.entry(family).or_insert((stat, None));
                e.0 = e.0.max(stat);
                e.1 = e.1.max(set);
            }
            fns.insert(n, f);
        }
        for (family, (stat, set)) in &smem { eprintln!("{}", smem_log(family, optin, *stat, *set)); }
        let q8 = std::sync::Mutex::new((stream.alloc_zeros::<u8>(1)?, None));
        let am = stream.alloc_zeros(64 * AMAX_PARTS)?;
        let ring = std::sync::Mutex::new((unsafe { ctx.alloc_pinned::<u8>(RING)? }, 0, VecDeque::new()));
        let done = ctx.new_event(None)?;
        Ok(Gpu { ctx, side: stream.fork()?, stream, nb: AtomicUsize::new(usize::MAX), fns, i8_gemm, q8, kv_q8, am, graphs: Default::default(), ring, done, lag: Default::default(), fr: Default::default() })
    }

    fn f(&self, name: &str) -> &CudaFunction { &self.fns[name] }

    /// The rope frequencies base^(-2i / n_rot) as rms_norm_rope computes them in f64 (slow on consumer cards), made
    /// once by the rope_freq kernel. While a graph is captured, a missing table gives 0 (the kernel computes them
    /// itself) and is made after the capture, which `graph` then repeats.
    fn rope_fr(&self, base: f32, n_rot: usize) -> anyhow::Result<u64> {
        let mut fr = self.fr.lock().unwrap();
        if let Some(t) = fr.0.get(&(base.to_bits(), n_rot)) { return Ok(self.ptr(t)); }
        if self.stream.capture_status()? != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
            fr.1.push((base, n_rot));
            return Ok(0);
        }
        let t = self.stream.alloc_zeros::<f64>((n_rot / 2).max(1))?;
        let p = self.ptr(&t);
        let mut b = self.stream.launch_builder(self.f("rope_freq"));
        let (bs, nr) = (base, n_rot as i32);
        b.arg(&p); b.arg(&bs); b.arg(&nr);
        unsafe { b.launch(cfg((1, 1), 256, 0)) }.context("launch rope_freq")?;
        self.stream.synchronize()?;
        fr.0.insert((base.to_bits(), n_rot), t);
        Ok(p)
    }

    /// The stream for the next launch: inside `batch`, odd launches go to `side`.
    fn st(&self) -> &Arc<CudaStream> {
        let n = self.nb.load(Relaxed);
        if n >= BESIDE { return &self.stream; }
        self.nb.store(n + 1, Relaxed);
        if n % 2 == 1 { &self.side } else { &self.stream }
    }

    pub fn ptr<T>(&self, s: &CudaSlice<T>) -> u64 { s.device_ptr(&self.stream).0 }

    pub fn zeros_slice<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(&self, n: usize) -> anyhow::Result<CudaSlice<T>> {
        Ok(self.stream.alloc_zeros::<T>(n.max(1))?)
    }

    /// GEMV column count (a multiple of 32 and of the type's block) and kernel name: a partial last super-block
    /// (cols % 256 != 0) takes the `_p` kernel, keeping its tail out of the full-row one.
    fn gemv_cols(prefix: &str, ty: GgmlType, cols: usize) -> anyhow::Result<(i32, String)> {
        let b = ty.block_elems().max(32);
        if cols % b != 0 { bail!("{} tensor with {cols} columns: GEMV needs a multiple of {b}", ty.name()); }
        Ok((cols as i32, kname(prefix, ty) + if cols % 256 != 0 { "_p" } else { "" }))
    }

    /// Whether the prefill GEMM for `ty` runs on int8 tensor cores (X quantized to int8 per 32) rather than f16.
    /// `STRATA_MMQ=q5_k,q6_k,..` overrides the default list.
    /// Type policy only: the CUDA launcher additionally requires the arch to have int8 tensor cores (`gemm_raw`), so
    /// a consumer of the decision must use `Backend::mmq`, which the CUDA backend overrides with `int8_gemm_arch`.
    pub fn mmq(ty: GgmlType) -> bool {
        // IQ2_XS (and IQ2_S) has a scale per 16: its int8 path spends as long scaling mma results as in the mma, and loses to f16 (#31).
        !Self::float_ty(ty) && std::env::var("STRATA_MMQ").map(|v| v.split(',').any(|t| t == ty.name())).unwrap_or(!matches!(ty, GgmlType::IQ2XS | GgmlType::IQ2S) || llama_numerics())
    }

    fn gemm_raw(&self, ty: GgmlType, rows: usize, cols: usize, tiles: u64, n_tiles: usize, wd: u64, nd: usize, x: u64, x_stride: usize, n_x: usize, xrow: u64, y: u64, y_stride: usize, act: i32) -> anyhow::Result<()> {
        // Partial super-blocks (Q2_0's 640 columns, BF16's 320) are fine: K steps are 64 deep.
        if cols % 64 != 0 { bail!("{} matrix with {cols} columns: GEMM needs a multiple of 64", ty.name()); }
        if n_tiles == 0 { return Ok(()); }
        let grid = (rows.div_ceil(64) as u32, n_tiles as u32);
        // A dense GEMM of few blocks (hc_*_inject: 4 rows, 8 blocks) runs as gemms_*/gemmis_*, 32 positions a block
        // instead of 128: four times the blocks, the same sums.
        let sm = tiles == 0 && grid.0 * grid.1 < 112;
        let grid = if sm { (grid.0, nd.div_ceil(32) as u32) } else { grid };
        let (same, act) = (act & SAME_X != 0, act & !SAME_X);
        let mut q8 = self.q8.lock().unwrap();
        // The int8 path loads q8 rows 16 bytes at a time: their stride (cols * 11 / 8) is aligned only for cols % 128 == 0.
        // Volta has no int8 tensor cores: there the kernels are not compiled (`#if __CUDA_ARCH__ >= 800`) and not
        // loaded, so this gate must dominate `STRATA_MMQ` or the launch would panic on `self.fns[...]`; sm_70 takes
        // the f16 `gemm_*`/`gemms_*` below instead.
        if !self.i8_gemm || !Self::mmq(ty) || cols % 128 != 0 {
            if !same { q8.1 = None; }
            launch!(self, &kname(if sm { "gemms" } else { "gemm" }, ty), cfg(grid, 2 * GEMM_BM as u32, 0),
                tiles, wd, nd as i32, x, x_stride as i32, xrow, y, y_stride as i32, rows as i32, (cols / 64) as i32, act);
            // Tile lists: gemm_* skips the tiles of at most 32 positions, which gemms_* takes.
            if tiles != 0 {
                launch!(self, &kname("gemms", ty), cfg(grid, 2 * GEMM_BM as u32, 0),
                    tiles, wd, nd as i32, x, x_stride as i32, xrow, y, y_stride as i32, rows as i32, (cols / 64) as i32, act);
            }
            return Ok(());
        }
        let need = n_x * (cols + cols / 8 + cols / 4);
        if q8.0.len() < need { q8.0 = self.stream.alloc_zeros::<u8>(need)?; }
        let xq = self.ptr(&q8.0);
        let key = Some([x, x_stride as u64, n_x as u64, cols as u64]);
        if !(same && q8.1 == key) {
            anyhow::ensure!(self.nb.load(Relaxed) >= BESIDE, "gemm inside batch: x must be quantized already (SAME_X)");
            launch!(self, "quant_q8", cfg((n_x as u32, 1), 256, 0), x, x_stride as i32, xq, cols as i32);
            q8.1 = key;
        }
        launch!(self, &kname(if sm { "gemmis" } else { "gemmi" }, ty), cfg(grid, 2 * GEMM_BM as u32, 0),
            tiles, wd, nd as i32, xq, xrow, y, y_stride as i32, rows as i32, (cols / 64) as i32, act);
        if tiles != 0 {
            launch!(self, &kname("gemmis", ty), cfg(grid, 2 * GEMM_BM as u32, 0),
                tiles, wd, nd as i32, xq, xrow, y, y_stride as i32, rows as i32, (cols / 64) as i32, act);
        }
        Ok(())
    }

    /// Shared memory for x quantized to int8 plus per-32 scales (quantized weight types only).
    fn xq_smem(ty: GgmlType, cols: usize) -> u32 {
        if Self::float_ty(ty) { 0 } else { (cols + cols / 8) as u32 }
    }

    fn float_ty(ty: GgmlType) -> bool { matches!(ty, GgmlType::F32 | GgmlType::BF16) }

    /// Rows per warp for the integer GEMV (env STRATA_RPW overrides). Experts: about 8 super-blocks of work per
    /// warp. Dense: 2 for 2048 and from 8192 rows, else 1 (measured per shape: the 4B's 2560 and
    /// 4096 rows are ~10% faster with 1, the default model's 2048 rows ~5% slower).
    fn rpw(rows: usize, cols: usize, k: usize) -> usize {
        // The vocab head: more rows per block amortize gemvn's per-block x quantization (nt = 2: 1158 -> 1027 us).
        // Experts of 512 columns (35B down, 2 superblocks a row): 16 rows per warp, 116.1 -> 116.6 tok/s.
        let d = if k > 1 { if cols <= 512 { 16 } else { (2048 / cols).max(1) } } else if rows >= 65536 { 8 } else if rows >= 8192 || rows == 2048 { 2 } else { 1 };
        std::env::var("STRATA_RPW").ok().and_then(|v| v.parse().ok()).unwrap_or(d)
    }

    /// Launch config and rows-per-warp argument for a GEMV with `rows` x `cols` per matrix and `k` matrices.
    fn gemv_cfg(ty: GgmlType, rows: usize, cols: usize, k: usize) -> (LaunchConfig, i32) {
        // Float types: one warp per row. Small matrices (32 rows) take one row per block to spread over many SMs;
        // from 4096 rows 8 per block, as an SM holds only 16 blocks (hc_attn_up 10240 x 320: 28.7 -> 24.8 us).
        if Self::float_ty(ty) {
            let wpb = if rows >= 4096 { 8 } else { 1 };
            return (cfg((rows.div_ceil(wpb) as u32, k as u32), 32 * wpb as u32, 0), 1);
        }
        let rpw = Self::rpw(rows, cols, k);
        (cfg((rows.div_ceil(8 * rpw) as u32, k as u32), 256, Self::xq_smem(ty, cols)), rpw as i32)
    }

    fn grid(n: usize) -> LaunchConfig { cfg((n.div_ceil(256) as u32, 1), 256, 0) }

    /// `gemvq`; with z = 2 also the matrix of `pair.2` rows at byte offset `pair.0` into the output at byte offset
    /// `pair.1` (token stride `pair.2`).
    fn gemvq_pair(&self, w: &GpuTensor, x: u64, x_stride: usize, xq: u64, y: u64, y_stride: usize, nt: usize, pair: (i64, i64, i32), z: u32) -> anyhow::Result<()> {
        if !(1..=4).contains(&nt) { bail!("gemvn: {nt} inputs (1 to 4)"); }
        let (ncols, k1) = Self::gemv_cols("gemv", w.ty, w.cols)?;
        let (mut c, rpw) = Self::gemv_cfg(w.ty, w.rows, w.cols, 1);
        c.shared_mem_bytes = if xq == 0 { c.shared_mem_bytes * nt as u32 } else { 0 };
        c.grid_dim.2 = z;
        // One token: gemv, whose kernel has no nt branches (gemvn's nt = 4 path costs it registers).
        if nt == 1 {
            launch!(self, &k1, c, w.ptr, x, y, w.rows as i32, ncols, rpw, xq, pair.0, pair.1, pair.2);
            return Ok(());
        }
        let kn = k1.replacen("gemv", "gemvn", 1);
        launch!(self, &kn, c, w.ptr, x, x_stride as i32, y, y_stride as i32, w.rows as i32, ncols, rpw, nt as i32, xq, pair.0, pair.1, pair.2);
        Ok(())
    }

    /// attn_decode_gq; attn_merge also writes `out` quantized into xq when not null.
    fn attn(&self, q: u64, kc: u64, vc: u64, qg: u64, out: u64, part: u64, step: u64, nh: usize, nkv: usize, hd: usize, scale: f32, gq_min: i32, nt: usize, xq: u64, sel: u64, sst: usize) -> anyhow::Result<()> {
        if ![64, 128, 256, 512].contains(&hd) { bail!("attention head_dim {hd} unsupported (64, 128, 256 or 512)"); }
        let group = nh / nkv;  // query heads per KV head, served by one block (up to 8, G * hd <= 2048)
        let g = [8, 4, 2, 1].into_iter().find(|&g| group % g == 0 && g * hd <= 2048).unwrap();
        let pz = (nh * ATT_SPLITS * (hd + 2)) as i32;
        // gq_min i32::MAX or 0 fixes the regime, so the kernel that would exit at once is not launched.
        if gq_min != 0 { launch!(self, &format!("attn_decode_{hd}"), cfg(((nh * nt) as u32, ATT_SPLITS_1 as u32), 256, (8 * hd * 4) as u32), q, kc, vc, part, step, nh as i32, nkv as i32, hd as i32, scale, gq_min, pz, nt as i32, sel, sst as i32); }
        // attn_decode_gqa with the f16 cache at head_dim 128 or 256 also holds 2 chunks of K and V for its cp.async,
        // attn_decode_gqa4_128 one.
        let cp = !self.kv_q8 && (hd == 128 || hd == 256);
        let g4 = cp && g == 4 && hd == 128;
        let (name, cp) = if g4 { ("attn_decode_gqa4_128".to_string(), 2 * 32 * hd * 2) } else { (format!("attn_decode_gqa_{hd}"), if cp { 2 * 2 * 32 * hd * 2 } else { 0 }) };
        if gq_min != i32::MAX { launch!(self, &name, cfg(((nh / g * nt) as u32, ATT_SPLITS as u32), 256, (8 * hd * 4 + cp) as u32), q, kc, vc, part, step, nh as i32, nkv as i32, hd as i32, scale, gq_min, pz, nt as i32, sel, sst as i32); }
        launch!(self, "attn_merge", LaunchConfig { grid_dim: (nh as u32, 1, nt as u32), block_dim: (256, 1, 1), shared_mem_bytes: 0 }, part, qg, out, hd as i32, ATT_SPLITS as i32, step, ATT_SPLITS_1 as i32, gq_min, xq);
        Ok(())
    }

    /// Stream sync that polls an event with long pauses between queries. The driver's own sync (and the sync inside
    /// a copy to pageable memory) queries in a tight loop, which slows the CPU expert threads (+2% decode with many
    /// expert misses, 35B on a 7600X; blocking gains 4% but adds its wake-up latency to every token and prefill layer).
    fn wait(&self) -> anyhow::Result<()> {
        self.ctx.bind_to_thread()?;
        self.done.record(&self.stream)?;
        while !self.done.try_is_complete()? {
            for _ in 0..256 { std::hint::spin_loop(); }
        }
        Ok(())
    }
}

impl Backend for Gpu {
    fn name(&self) -> &'static str { "cuda" }
    fn device(&self) -> String { self.ctx.name().unwrap_or_default() }
    /// The type policy gated on this device: `gemm_raw` takes the f16 `gemm_*` path where the arch has no int8
    /// tensor cores (Volta), so a test comparing kernels must not expect `gemmi_*` there.
    fn mmq(&self, ty: GgmlType) -> bool { self.i8_gemm && Gpu::mmq(ty) }
    /// sm_70 runs `mma16816` on CUDA cores (the `__CUDA_ARCH__ < 800` branch in kernels.cu): the same f16 products
    /// summed in another order than `mma.sync`, so a cross-backend comparison is by tolerance, not bit for bit.
    fn volta(&self) -> bool { !self.i8_gemm }
    fn alloc(&self, bytes: usize) -> anyhow::Result<Buf> {
        let s = self.stream.alloc_zeros::<u8>(bytes.max(1))?;
        Ok(Buf::new(self.ptr(&s), s.len(), s))
    }

    // Returns once `src` is copied (a copy still reading `src` after return broke greedy output).
    fn htod(&self, src: &[u8], dst: &Buf, off: usize) -> anyhow::Result<()> {
        anyhow::ensure!(off + src.len() <= dst.len(), "htod: {} bytes at {off} into {}", src.len(), dst.len());
        self.ctx.bind_to_thread()?;
        // Inside `batch` or `beside` a copy goes to the side stream, beside the next launch on the main stream.
        let st = if self.nb.load(Relaxed) == usize::MAX { &self.stream } else { &self.side };
        // No stream sync: from pageable memory (every caller's) the driver has staged `src` when the call returns.
        if src.len() < RING_MIN { return Ok(unsafe { result::memcpy_htod_async(dst.addr() + off as u64, src, st.cu_stream()) }?); }
        let mut g = self.ring.lock().unwrap();
        let (pin, head, busy) = &mut *g;
        let base = pin.as_mut_ptr()?;
        for (i, c) in src.chunks(RING).enumerate() {
            if *head + c.len() > RING { *head = 0; }
            let (a, b) = (*head, *head + c.len());
            while busy.iter().any(|&(s, e, _)| s < b && a < e) { busy.pop_front().unwrap().2.synchronize()?; }
            unsafe {
                let p = base.add(a);
                std::ptr::copy_nonoverlapping(c.as_ptr(), p, c.len());
                result::memcpy_htod_async(dst.addr() + (off + i * RING) as u64, std::slice::from_raw_parts(p, c.len()), st.cu_stream())?;
            }
            busy.push_back((a, b, st.record_event(None)?));
            *head = b;
        }
        Ok(())
    }

    // As cudarc's clone_dtoh: copy, then a stream sync.

    fn dtoh(&self, src: &Buf, off: usize, dst: &mut [u8]) -> anyhow::Result<()> {
        anyhow::ensure!(off + dst.len() <= src.len(), "dtoh: {} bytes at {off} from {}", dst.len(), src.len());
        self.ctx.bind_to_thread()?;
        // A copy into pageable memory waits for the stream inside the call, spinning; wait for the work first.
        self.wait()?;
        unsafe { result::memcpy_dtoh_async(dst, src.addr() + off as u64, self.stream.cu_stream()) }?;
        Ok(self.stream.synchronize()?)
    }

    fn dtod(&self, src: &Buf, src_off: usize, dst: &Buf, dst_off: usize, bytes: usize) -> anyhow::Result<()> {
        anyhow::ensure!(src_off + bytes <= src.len() && dst_off + bytes <= dst.len(), "dtod: {bytes} bytes out of range");
        self.ctx.bind_to_thread()?;
        Ok(unsafe { result::memcpy_dtod_async(dst.addr() + dst_off as u64, src.addr() + src_off as u64, bytes, self.stream.cu_stream()) }?)
    }

    fn memset(&self, dst: &Buf) -> anyhow::Result<()> {
        self.ctx.bind_to_thread()?;
        Ok(unsafe { result::memset_d8_async(dst.addr(), 0, dst.len(), self.stream.cu_stream()) }?)
    }

    fn mem_info(&self) -> anyhow::Result<(usize, usize)> { Ok(self.ctx.mem_get_info()?) }

    fn sync(&self) -> anyhow::Result<()> { self.wait() }

    // Every op batched on CUDA is one launch, so alternating launches never splits an op over the two streams. Copies
    // do not count: they all go to the side stream.
    fn batch(&self, f: &mut dyn FnMut() -> anyhow::Result<()>) -> anyhow::Result<()> {
        self.side.join(&self.stream)?;
        self.nb.store(0, Relaxed);
        let r = f();
        self.nb.store(usize::MAX, Relaxed);
        self.stream.join(&self.side)?;
        r
    }

    fn beside_overlaps(&self) -> bool { true }

    fn beside(&self, f: &mut dyn FnMut() -> anyhow::Result<()>) -> anyhow::Result<()> {
        self.side.join(&self.stream)?;
        self.nb.store(BESIDE, Relaxed);
        let r = f();
        self.nb.store(usize::MAX, Relaxed);
        let e = self.side.record_event(None)?;
        if let Some(p) = self.lag.lock().unwrap().replace(e) { self.stream.wait(&p)?; }
        r
    }

    fn beside_join(&self) -> anyhow::Result<()> {
        self.lag.lock().unwrap().take();
        Ok(self.stream.join(&self.side)?)
    }

    /// CUDA graph captured from the stream. `STRATA_NO_GRAPH` runs `f` every time instead.
    fn graph(&self, key: u64, f: &mut dyn FnMut() -> anyhow::Result<()>) -> anyhow::Result<()> {
        if let Some(g) = self.graphs.lock().unwrap().get(&key) { return Ok(g.0.launch()?); }
        if std::env::var_os("STRATA_NO_GRAPH").is_some() { return f(); }
        self.stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let r = f();
        let g = self.stream.end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)?;
        r?;
        let g = g.ok_or_else(|| anyhow::anyhow!("empty CUDA graph"))?;
        g.launch()?;
        // A rope table the capture lacked: make it, and capture again next time (this graph computes it per call).
        let want = std::mem::take(&mut self.fr.lock().unwrap().1);
        if !want.is_empty() {
            for (base, n_rot) in want { self.rope_fr(base, n_rot)?; }
            return Ok(());
        }
        self.graphs.lock().unwrap().insert(key, Graph(g));
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

    /// Tensor-core GEMM (see `GemmTile` in kernels.cu); integer weight types quantize x to int8 first.
    fn gemm(&self, ty: GgmlType, rows: usize, cols: usize, tiles: u64, n_tiles: usize, x: u64, x_stride: usize, n_x: usize, xrow: u64, y: u64, y_stride: usize, act: i32) -> anyhow::Result<()> {
        self.gemm_raw(ty, rows, cols, tiles, n_tiles, 0, 0, x, x_stride, n_x, xrow, y, y_stride, act)
    }

    fn gemm_dense(&self, w: &GpuTensor, n: usize, x: u64, x_stride: usize, y: u64, y_stride: usize, act: i32) -> anyhow::Result<()> {
        self.gemm_raw(w.ty, w.rows, w.cols, 0, n.div_ceil(GEMM_BM), w.ptr, n, x, x_stride, n, 0, y, y_stride, act)
    }

    fn gemv(&self, w: &GpuTensor, x: u64, y: u64) -> anyhow::Result<()> {
        let (ncols, kn) = Self::gemv_cols("gemv", w.ty, w.cols)?;
        let (c, rpw) = Self::gemv_cfg(w.ty, w.rows, w.cols, 1);
        launch!(self, &kn, c, w.ptr, x, y, w.rows as i32, ncols, rpw, 0u64, 0i64, 0i64, 0i32);
        Ok(())
    }

    fn gemvn(&self, w: &GpuTensor, x: u64, x_stride: usize, y: u64, y_stride: usize, nt: usize) -> anyhow::Result<()> {
        self.gemvq(w, x, x_stride, 0, y, y_stride, nt)
    }

    fn gemvq(&self, w: &GpuTensor, x: u64, x_stride: usize, xq: u64, y: u64, y_stride: usize, nt: usize) -> anyhow::Result<()> {
        self.gemvq_pair(w, x, x_stride, xq, y, y_stride, nt, (0, 0, 0), 1)
    }

    // One launch for both matrices.
    fn gemvq2(&self, a: &GpuTensor, b: &GpuTensor, x: u64, x_stride: usize, xq: u64, ya: u64, yb: u64, nt: usize) -> anyhow::Result<()> {
        if a.ty != b.ty || a.cols != b.cols {
            self.gemvq(a, x, x_stride, xq, ya, a.rows, nt)?;
            return self.gemvq(b, x, x_stride, xq, yb, b.rows, nt);
        }
        // The launch is sized for the larger matrix.
        let ((a, ya), (b, yb)) = if a.rows >= b.rows { ((a, ya), (b, yb)) } else { ((b, yb), (a, ya)) };
        self.gemvq_pair(a, x, x_stride, xq, ya, a.rows, nt, (b.ptr as i64 - a.ptr as i64, yb as i64 - ya as i64, b.rows as i32), 2)
    }

    fn moe_gemv(&self, ty: GgmlType, rows: usize, cols: usize, w: u64, stride: usize, ids: u64, k: usize, x: u64, x_stride: usize, xg: usize, y: u64, pair: Option<(i64, i64)>, xq: u64) -> anyhow::Result<()> {
        let (ncols, kn) = Self::gemv_cols("moe_gemv", ty, cols)?;
        let (mut c, mut rpw) = Self::gemv_cfg(ty, rows, cols, k);
        // At least 2 rows per warp: an expert's x quantization (once per block) then covers 16 rows.
        if rpw == 1 && !Self::float_ty(ty) { rpw = 2; c.grid_dim.0 = rows.div_ceil(16) as u32; }
        let (w2, y2) = pair.unwrap_or((0, 0));
        c.grid_dim.2 = if pair.is_some() { 2 } else { 1 };
        if xq != 0 { c.shared_mem_bytes = 0; }
        launch!(self, &kn, c, w, ids, stride as i64, x, x_stride as i32, xg as i32, y, rows as i32, ncols, rpw, w2, y2, xq);
        Ok(())
    }

    fn dequant(&self, w: &GpuTensor, rows: usize, out: u64) -> anyhow::Result<()> {
        launch!(self, &kname("dequant", w.ty), cfg((rows as u32, 1), 256, 0), w.ptr, out, Self::gemv_cols("dequant", w.ty, w.cols)?.0);
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
        // One block per row reads 1 MB (a 248320-token vocabulary) in ~75 us; AMAX_PARTS blocks per row share it.
        let part = self.ptr(&self.am);
        anyhow::ensure!(rows * AMAX_PARTS <= self.am.len(), "argmax_rows: {rows} rows");
        launch!(self, "argmax_part", cfg((AMAX_PARTS as u32, rows as u32), 256, 0), x, n as i32, part);
        launch!(self, "argmax_rows", cfg((rows as u32, 1), AMAX_PARTS as u32, 0), part, n as i32, out);
        Ok(())
    }

    fn l2_norm(&self, x: u64, n: usize, per: usize, stride: usize, rows: usize, eps: f32) -> anyhow::Result<()> {
        launch!(self, "l2_norm", cfg((rows as u32, 1), 128, 0), x, n as i32, per as i32, stride as i32, eps);
        Ok(())
    }

    fn rope_neox(&self, x: u64, stride: usize, heads: usize, per: usize, n_rot: usize, step: u64, base: f32) -> anyhow::Result<()> {
        launch!(self, "rope_neox", cfg(((heads / per) as u32, 1), (n_rot / 2) as u32, 0), x, stride as i32, per as i32, n_rot as i32, step, base);
        Ok(())
    }

    fn rms_norm_rope(&self, x: u64, x_stride: usize, w: u64, out: u64, n: usize, rows: usize, eps: f32, per: usize, n_rot: usize, step: u64, base: f32) -> anyhow::Result<()> {
        self.rms_norm_rope_kv(x, x_stride, w, out, n, rows, eps, per, n_rot, step, base, 0, 0, 0)
    }

    fn rms_norm_rope_kv(&self, x: u64, x_stride: usize, w: u64, out: u64, n: usize, rows: usize, eps: f32, per: usize, n_rot: usize, step: u64, base: f32, v: u64, kc: u64, vc: u64) -> anyhow::Result<()> {
        anyhow::ensure!(n_rot <= n.min(512) && n % 32 == 0, "rms_norm_rope: head of {n}, n_rot {n_rot}");
        let fr = self.rope_fr(base, n_rot)?;
        launch!(self, "rms_norm_rope", cfg((rows as u32, 1), 256, 0), x, x_stride as i32, w, out, n as i32, eps, per as i32, n_rot as i32, step, base, v, kc, vc,
                0i32, 0u64, 0i32, 0u64, 0u64, 0i32, fr);
        Ok(())
    }

    // Both in one launch: the q rows come first.
    fn rms_norm_rope_qk(&self, q: u64, q_stride: usize, qw: u64, qo: u64, qrows: usize, qper: usize, k: u64, kw: u64, ko: u64, krows: usize, kper: usize,
                        n: usize, eps: f32, n_rot: usize, step: u64, base: f32, v: u64, kc: u64, vc: u64) -> anyhow::Result<()> {
        anyhow::ensure!(n_rot <= n.min(512) && n % 32 == 0, "rms_norm_rope: head of {n}, n_rot {n_rot}");
        let fr = self.rope_fr(base, n_rot)?;
        launch!(self, "rms_norm_rope", cfg(((qrows + krows) as u32, 1), 256, 0), k, n as i32, kw, ko, n as i32, eps, kper as i32, n_rot as i32, step, base, v, kc, vc,
                qrows as i32, q, q_stride as i32, qw, qo, qper as i32, fr);
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
    fn swiglu_q(&self, g: u64, u: u64, n: usize, rows: usize, xq: u64) -> anyhow::Result<u64> {
        launch!(self, "swiglu_q", cfg((n.div_ceil(256) as u32, rows as u32), 32, 0), g, u, n as i32, xq);
        Ok(xq)
    }

    fn ssm_gates(&self, beta: u64, g: u64, dt_bias: u64, a: u64, n: usize, nv: usize) -> anyhow::Result<()> {
        launch!(self, "ssm_gates", Self::grid(n), beta, g, dt_bias, a, n as i32, nv as i32);
        Ok(())
    }

    fn conv_silu(&self, qkv: u64, state: u64, w: u64, y: u64, ch: usize, kw: usize, nn: usize, eps: f32) -> anyhow::Result<()> {
        launch!(self, "conv_silu", cfg((ch.div_ceil(128) as u32, 1), 128, 0), qkv, state, w, y, ch as i32, kw as i32, nn as i32, eps);
        Ok(())
    }

    fn deltanet(&self, y: u64, s: u64, beta: u64, g: u64, dt_bias: u64, a: u64, o: u64, nk: usize, nv: usize, interleave: bool, dk: usize, dv: usize, scale: f32, t: usize, ych: usize, snap: u64, nsnap: usize) -> anyhow::Result<()> {
        let rows = 512 / dk.max(1);  // state rows per warp
        anyhow::ensure!(matches!(dk, 32 | 64 | 128 | 256) && dv % rows == 0, "deltanet: head dims {dk}/{dv} unsupported");
        launch!(self, "deltanet_seq", cfg((nv as u32, dv.div_ceil(8 * rows) as u32), 256, 0),
            y, s, beta, g, o, nk as i32, nv as i32, interleave as i32, dk as i32, dv as i32, scale, t as i32, ych as i32, snap, nsnap as i32,
            dt_bias, a);
        Ok(())
    }

    fn conv_seq(&self, qkv: u64, state: u64, w: u64, y: u64, ch: usize, kw: usize, t: usize, snap: u64, nsnap: usize) -> anyhow::Result<()> {
        launch!(self, "conv_seq", Self::grid(t * ch), qkv, state, w, y, ch as i32, kw as i32, t as i32);
        launch!(self, "conv_state", Self::grid(ch), qkv, state, ch as i32, kw as i32, t as i32, snap, nsnap as i32);
        Ok(())
    }

    fn attn_prefill_m(&self, q: u64, kc: u64, vc: u64, qg: u64, out: u64, step: u64, t: usize, nh: usize, nkv: usize, hd: usize, scale: f32, qm: u64, mw: usize) -> anyhow::Result<()> {
        if !matches!(hd, 64 | 128 | 256) { bail!("attention prefill: head_dim {hd} unsupported (64, 128 or 256)"); }
        let smem = (2 * (64 + 32 + 32) * (hd + 8) + 2 * 64 * (32 + 8)) as u32;
        launch!(self, &format!("attn_prefill_{hd}"), cfg((t.div_ceil(64) as u32, nh as u32), 128, smem),
            q, kc, vc, qg, out, step, t as i32, nh as i32, nkv as i32, scale, qm, mw as i32);
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
        // llama numerics take one scale per 256 values, more than a head.
        let xq = if llama_numerics() || n % 32 != 0 { 0 } else { xq };
        launch!(self, "norm_gate", cfg((heads as u32, 1), 128, 0), o, w, z, n as i32, eps, xq, 0i32);
        Ok(xq)
    }

    fn norm_gate_sig(&self, o: u64, w: u64, z: u64, n: usize, heads: usize, eps: f32, xq: u64) -> anyhow::Result<u64> {
        let xq = if llama_numerics() || n % 32 != 0 { 0 } else { xq };
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

    fn kv_words(&self, n: usize) -> usize { if self.kv_q8 { n * 17 / 32 } else { n } }

    fn kv_store(&self, k: u64, v: u64, kc: u64, vc: u64, n: usize, t: usize, step: u64, hd: usize) -> anyhow::Result<()> {
        // q8: each warp quantizes one 32-value block of one row.
        anyhow::ensure!(!self.kv_q8 || (hd % 32 == 0 && n % hd == 0), "q8 KV cache: head_dim {hd} is not a multiple of 32");
        launch!(self, "kv_store", Self::grid(n * t), k, v, kc, vc, n as i32, (n * t) as i32, step, hd as i32);
        Ok(())
    }

    /// One launch serves all tokens, their blocks side by side, so they read the same K/V rows from L2. Below ATT_GQA_MIN positions one block serves one head
    /// (ATT_SPLITS_1 splits), from it on one block serves all query heads of a KV head (ATT_SPLITS splits). Both
    /// kernels are launched (only one when gq_min is 0 or i32::MAX); the step decides which one works.
    fn attn_decode_gq(&self, q: u64, kc: u64, vc: u64, qg: u64, out: u64, part: u64, step: u64, nh: usize, nkv: usize, hd: usize, scale: f32, gq_min: i32, nt: usize) -> anyhow::Result<()> {
        self.attn(q, kc, vc, qg, out, part, step, nh, nkv, hd, scale, gq_min, nt, 0, 0, 0)
    }

    fn attn_decode_q(&self, q: u64, kc: u64, vc: u64, qg: u64, out: u64, part: u64, step: u64, nh: usize, nkv: usize, hd: usize, scale: f32, gq_min: i32, nt: usize, xq: u64, sel: u64, sst: usize) -> anyhow::Result<u64> {
        let xq = if llama_numerics() { 0 } else { xq };
        self.attn(q, kc, vc, qg, out, part, step, nh, nkv, hd, scale, gq_min, nt, xq, sel, sst)?;
        Ok(xq)
    }

    fn router_topk(&self, p: u64, n: usize, k: usize, extra_scale: f32, ids: u64, wts: u64, t: usize) -> anyhow::Result<()> {
        anyhow::ensure!(n <= 512 && k <= 32, "router_topk: {n} experts, top {k}");
        launch!(self, "router_topk", cfg((t as u32, 1), 32, 0), p, n as i32, k as i32, extra_scale, ids, wts);
        Ok(())
    }

    fn moe_route(&self, p: u64, n: usize, k: usize, extra_scale: f32, table: u64, ids: u64, wts: u64, slots: u64, wh: u64, h: u64, mb: u64, nh: usize, h_off: usize, step: u64, t: usize) -> anyhow::Result<()> {
        anyhow::ensure!(n <= 512 && k <= 32, "moe_route: {n} experts, top {k}");
        launch!(self, "moe_route", cfg((1, 1), 256, 0), p, n as i32, k as i32, extra_scale, table, ids, wts, slots, wh, h, mb, nh as i32, h_off as i32, step, t as i32);
        Ok(())
    }

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

    const SRC: &str = "extern \"C\" __global__ void k(float* x){x[0]=1.0f;}";

    /// The resolved NVRTC is the host's 12.9; a machine without NVRTC skips (documented, not ignored).
    #[test]
    fn nvrtc_version_is_detected() {
        if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
            eprintln!("skipping: NVRTC library not present");
            return;
        }
        assert_eq!(super::nvrtc_version(), Some((12, 9)));
    }

    /// R2/R4 with no device and no model: 12.9 accepts a trivial kernel for sm_70.
    #[test]
    fn nvrtc_compiles_trivial_kernel_for_sm70() {
        if !unsafe { cudarc::nvrtc::sys::is_culib_present() } { return; }
        super::cubin(SRC, "sm_70").expect("NVRTC 12.x must compile for sm_70 without a GPU");
    }

    /// M1-T2..M1-T6, CPU-only (no device, no model, no `STRATA_MODEL`): the full generated source must produce a
    /// cubin for sm_70 **and** for sm_80. The last sm_70 blocker, `mma.m16n8k16`, is a CUDA-core fallback since
    /// M1-T4 (`src/cuda/kernels.cu:mma16816`), so no blocker list remains: any sm_70 diagnostic is a regression.
    /// The sm_80 cubin is the compile-level proof that the `#else` tensor-core path is unaffected.
    #[test]
    fn nvrtc_compiles_full_source_for_sm70() {
        if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
            eprintln!("skipping: NVRTC library not present");
            return;
        }
        match super::nvrtc_version() {
            Some((12, _)) => {}
            v => {
                eprintln!("skipping: NVRTC {v:?} is not 12.x; CUDA 13.x cannot target sm_70 and its diagnostics differ");
                return;
            }
        }
        let src = super::source(false, true);
        super::cubin(&src, "sm_70").unwrap_or_else(|e| panic!("sm_70 must compile: M1-T4 removed the last blocker: {e:#}"));
        super::cubin(&src, "sm_80").expect("the sm_80 path must keep compiling (cubin for sm_80)");
        // Same sm_70 source with the FMA fallback selected: both variants must produce a cubin (M2-T1).
        super::cubin(&super::source(false, false), "sm_70").expect("the STRATA_GEMM_FMA sm_70 source must compile");
    }

    #[test]
    fn arch_support_error_rejects_cuda13_on_sm70() {
        let msg = super::arch_support_error((7, 0), (13, 0)).expect("13.x cannot target sm_70");
        assert!(msg.contains("12.x"), "message must name the required toolkit: {msg}");
        assert!(msg.contains("sm_70"), "message must name the arch: {msg}");
    }

    #[test]
    fn arch_support_error_allows_cuda12_on_sm70() {
        assert_eq!(super::arch_support_error((7, 0), (12, 9)), None);
    }

    #[test]
    fn arch_support_error_allows_sm80_on_cuda13() {
        assert_eq!(super::arch_support_error((8, 0), (13, 0)), None);
    }

    #[test]
    fn cache_key_depends_on_nvrtc_tag() {
        let (a, b, c) = (
            super::cache_key(SRC, "sm_70", "12.9"),
            super::cache_key(SRC, "sm_70", "13.0"),
            super::cache_key(SRC, "sm_70", "unverified"),
        );
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
        assert_eq!(a, super::cache_key(SRC, "sm_70", "12.9"));
        assert_ne!(a, super::cache_key(SRC, "sm_80", "12.9"));
    }

    #[test]
    fn arch_log_marks_volta() {
        assert_eq!(
            super::device_log(0, "Tesla V100-SXM2-32GB", "sm_70", (7, 0), false),
            "cuda device 0: Tesla V100-SXM2-32GB, sm_70 (Volta, m8n8k4)");
        assert_eq!(
            super::device_log(0, "Tesla V100-SXM2-32GB", "sm_70", (7, 0), true),
            "cuda device 0: Tesla V100-SXM2-32GB, sm_70 (Volta fallback)");
        assert_eq!(super::device_log(0, "NVIDIA GeForce RTX 3060", "sm_86", (8, 6), false),
            "cuda device 0: NVIDIA GeForce RTX 3060, sm_86");
        assert_eq!(super::device_log(0, "NVIDIA GeForce RTX 3060", "sm_86", (8, 6), true),
            "cuda device 0: NVIDIA GeForce RTX 3060, sm_86");
    }

    /// M1-T5, CPU-only: the int8 prefill GEMM needs int8 tensor cores, which Volta (7.x) lacks; every sm_80+ part
    /// keeps it.
    #[test]
    fn int8_gemm_arch() {
        assert!(!super::int8_gemm_arch((7, 0)));
        assert!(!super::int8_gemm_arch((7, 5)));
        assert!(super::int8_gemm_arch((8, 0)));
        assert!(super::int8_gemm_arch((8, 6)));
        assert!(super::int8_gemm_arch((9, 0)));
    }

    /// M1-T5, CPU-only: the loaded name list drops `gemmi_*`/`gemmis_*` only on a device where kernels.cu compiles
    /// them out. 282 names today (sm_80, unchanged); 20 non-float types x 2 fewer names = 242 on sm_70.
    #[test]
    fn kernel_names() {
        let (sm70, sm80) = (super::kernel_names(false), super::kernel_names(true));
        assert_eq!(sm80.len(), 282);
        assert_eq!(sm70.len(), 242);
        assert!(sm70.iter().all(|n| !n.starts_with("gemmi") && !n.starts_with("gemmis")),
            "sm_70 must not load the int8 prefill GEMM: {:?}",
            sm70.iter().filter(|n| n.starts_with("gemmi")).collect::<Vec<_>>());
        assert!(sm80.contains(&"gemmi_q4_k".to_string()));
        assert!(sm80.contains(&"gemmis_q2_k".to_string()));
    }

    /// M1-T7, CPU-only: the per-family dynamic shared memory a launch can ask for, from the launcher's tile shapes.
    #[test]
    fn dyn_shmem_need_per_family() {
        // attn_prefill: 2 * (64 + 32 + 32) rows of hd + 8 halves, 2 bytes each, plus the 64 x 40 half P staging tile.
        assert_eq!(super::dyn_shmem_need("attn_prefill_64"), Some(("attn_prefill_*", 2 * 128 * 72 + 5120)));
        assert_eq!(super::dyn_shmem_need("attn_prefill_128"), Some(("attn_prefill_*", 2 * 128 * 136 + 5120)));
        assert_eq!(super::dyn_shmem_need("attn_prefill_256"), Some(("attn_prefill_*", 72704)));
        // attn_decode_gqa: 8 query slices per head group, plus 2 chunks of 32 K and V positions at head_dim 128/256.
        assert_eq!(super::dyn_shmem_need("attn_decode_gqa_64"), Some(("attn_decode_gqa*", 2048)));
        assert_eq!(super::dyn_shmem_need("attn_decode_gqa_128"), Some(("attn_decode_gqa*", 36864)));
        assert_eq!(super::dyn_shmem_need("attn_decode_gqa_256"), Some(("attn_decode_gqa*", 73728)));
        assert_eq!(super::dyn_shmem_need("attn_decode_gqa_512"), Some(("attn_decode_gqa*", 16384)));
        // gqa4_128 copies one K/V chunk; the generic figure covers it.
        assert_eq!(super::dyn_shmem_need("attn_decode_gqa4_128"), Some(("attn_decode_gqa*", 36864)));
        // gemvn caches nt quantized inputs of a runtime column count: the whole legal budget, not a guess.
        assert_eq!(super::dyn_shmem_need("gemvn_q2_k"), Some(("gemvn_*", usize::MAX)));
        assert_eq!(super::dyn_shmem_need("gemvn_f32"), Some(("gemvn_*", usize::MAX)));
        // Every other kernel stays within the 48 KiB default (attn_decode_* maxes out at 16 KiB).
        for n in ["attn_decode_256", "attn_merge", "gemv_q4_k", "gemm_q2_k", "gemms_q2_k", "quant_q8", "rope_freq"] {
            assert_eq!(super::dyn_shmem_need(n), None, "{n}");
        }
    }

    /// M1-T7, CPU-only: the opt-in value is clamped to the device ceiling minus the static shared memory, and the
    /// attribute is only touched where the launch needs more than the 48 KiB default.
    #[test]
    fn dyn_shmem_optin_clamps_to_optin_and_static() {
        const DEFAULT: usize = 48 << 10;
        // At or below the 48 KiB default: no attribute, the default covers the launch.
        assert_eq!(super::dyn_shmem_optin(16384, 0, 96 << 10), None);
        assert_eq!(super::dyn_shmem_optin(DEFAULT, 0, 96 << 10), None);
        assert_eq!(super::dyn_shmem_optin(DEFAULT + 1, 0, 96 << 10), Some(DEFAULT + 1));
        // static == 0: the device ceiling is entirely available.
        assert_eq!(super::dyn_shmem_optin(67584, 0, 96 << 10), Some(67584));
        // Static comes out of the same ceiling (attn_decode_gqa_256: 2112 B static, 73728 B needed).
        assert_eq!(super::dyn_shmem_optin(73728, 2112, 96 << 10), Some(73728));
        // A shape-dependent request takes ceiling - static.
        assert_eq!(super::dyn_shmem_optin(usize::MAX, 2112, 96 << 10), Some((96 << 10) - 2112));
        // A need past the ceiling (attn_decode_gqa_256 plus a static part on a 96 KiB opt-in) is clamped, so the
        // load succeeds and the launch fails where it is attempted.
        assert_eq!(super::dyn_shmem_optin(100000, 4096, 96 << 10), Some((96 << 10) - 4096));
        // Static alone above the opt-in: clamp at 0, never underflow.
        assert_eq!(super::dyn_shmem_optin(usize::MAX, 100 << 10, 96 << 10), Some(0));
    }

    #[test]
    fn smem_log_reports_device_numbers() {
        assert_eq!(super::smem_log("attn_prefill_*", 98304, 0, Some(72704)),
            "cuda shared memory: opt-in 98304 B per block, attn_prefill_*: 0 B static, dynamic max 72704 B");
        assert_eq!(super::smem_log("attn_decode_*", 98304, 2112, None),
            "cuda shared memory: opt-in 98304 B per block, attn_decode_*: 2112 B static, dynamic max default (48 KiB)");
    }
}
