//! HIP runtime and hiprtc, loaded by name like `cuda/nvrtc.rs` loads NVRTC: no ROCm headers, link flags or `hipcc`
//! at build time, and no C++ build step. Every entry point the backend uses is resolved once per process.

use anyhow::{Context, bail};
use libloading::Library;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::path::PathBuf;
use std::sync::OnceLock;

pub type Stream = *mut c_void;
pub type Module = *mut c_void;
pub type Func = *mut c_void;
pub type Event = *mut c_void;
pub type Graph = *mut c_void;
pub type GraphExec = *mut c_void;
pub type RtcProgram = *mut c_void;

// hipError_t / hiprtcResult
pub const HIP_SUCCESS: c_int = 0;

// Values the ROCm 7 headers give these enums (probe-printed; see the module docs of `hip`).
pub const MEMCPY_H2D: c_int = 1;
pub const MEMCPY_D2H: c_int = 2;
pub const MEMCPY_D2D: c_int = 3;
pub const HOST_MALLOC_PORTABLE: u32 = 1;
pub const HOST_MALLOC_MAPPED: u32 = 2;
pub const EVENT_DISABLE_TIMING: u32 = 2;
pub const STREAM_NON_BLOCKING: u32 = 1;
pub const CAPTURE_MODE_THREAD_LOCAL: c_int = 1;
pub const ATTR_MAX_DYNAMIC_SHARED: c_int = 8;
pub const ATTR_SHARED_PER_BLOCK_OPTIN: c_int = 75;
pub const ATTR_SHARED_PER_BLOCK: c_int = 74;
pub const ATTR_WARP_SIZE: c_int = 87;
pub const ATTR_INTEGRATED: c_int = 16;
pub const ATTR_MULTIPROCESSOR_COUNT: c_int = 63;
pub const ATTR_MAX_THREADS_PER_BLOCK: c_int = 56;

/// Room for `hipDeviceProp_t`, which `hipGetDeviceProperties` writes whole. ROCm 7's `hipDeviceProp_tR0600` is 1472
/// bytes; the size is not exported and changes between versions, so the buffer leaves ample room for a larger one.
/// The architecture name is found by shape (`find_gfx`), and `STRATA_HIP_ARCH` overrides it.
const PROP_BUF: usize = 16 << 10;

/// The first `gfxNNNN`-looking string at a 4-byte offset in `bytes` (at least 6 characters, at most 16), read only
/// within `bytes`.
fn find_gfx(bytes: &[u8]) -> Option<String> {
    for i in (0..bytes.len()).step_by(4) {
        let w = &bytes[i..(i + 16).min(bytes.len())];
        if !w.starts_with(b"gfx") { continue }
        let end = w.iter().position(|&b| b == 0 || !b.is_ascii_alphanumeric()).unwrap_or(w.len());
        if end < 6 { continue }
        return String::from_utf8(w[..end].to_vec()).ok();
    }
    None
}

type Fn = unsafe extern "C" fn();

/// Library names to try, in order. `STRATA_HIP` (a path or a name) replaces the whole list.
fn candidates(vars: Option<PathBuf>, default_suffix: &[&str]) -> Vec<PathBuf> {
    if let Some(p) = vars { return vec![p]; }
    let mut v: Vec<PathBuf> = Vec::new();
    for s in default_suffix { v.push(PathBuf::from(s)); }
    // A ROCm install that is not on the loader path (`/opt/rocm` may be a symlink chain).
    for root in ["/opt/rocm/lib", "/opt/rocm/core-10.1/lib", "/usr/lib/x86_64-linux-gnu", "/usr/local/lib"] {
        for s in default_suffix { v.push(PathBuf::from(root).join(s)); }
    }
    v
}

pub struct Hip {
    pub init: unsafe extern "C" fn(c_int) -> c_int,
    pub device_count: unsafe extern "C" fn(*mut c_int) -> c_int,
    pub set_device: unsafe extern "C" fn(c_int) -> c_int,
    pub device_name: unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int,
    pub device_attribute: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int,
    pub mem_get_info: unsafe extern "C" fn(*mut usize, *mut usize) -> c_int,
    pub malloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int,
    pub free: unsafe extern "C" fn(*mut c_void) -> c_int,
    pub memset: unsafe extern "C" fn(*mut c_void, c_int, usize) -> c_int,
    pub memset_async: unsafe extern "C" fn(*mut c_void, c_int, usize, Stream) -> c_int,
    pub memcpy: unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int,
    pub memcpy_async: unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int, Stream) -> c_int,
    pub stream_create: unsafe extern "C" fn(*mut Stream, u32) -> c_int,
    pub stream_destroy: unsafe extern "C" fn(Stream) -> c_int,
    pub stream_sync: unsafe extern "C" fn(Stream) -> c_int,
    pub stream_wait_event: unsafe extern "C" fn(Stream, Event, u32) -> c_int,
    pub stream_begin_capture: unsafe extern "C" fn(Stream, c_int) -> c_int,
    pub stream_end_capture: unsafe extern "C" fn(Stream, *mut Graph) -> c_int,
    pub stream_is_capturing: unsafe extern "C" fn(Stream, *mut c_int) -> c_int,
    pub graph_instantiate: unsafe extern "C" fn(*mut GraphExec, Graph, u64) -> c_int,
    pub graph_launch: unsafe extern "C" fn(GraphExec, Stream) -> c_int,
    pub graph_exec_destroy: unsafe extern "C" fn(GraphExec) -> c_int,
    pub graph_destroy: unsafe extern "C" fn(Graph) -> c_int,
    pub event_create: unsafe extern "C" fn(*mut Event, u32) -> c_int,
    pub event_record: unsafe extern "C" fn(Event, Stream) -> c_int,
    pub event_sync: unsafe extern "C" fn(Event) -> c_int,
    pub event_query: unsafe extern "C" fn(Event) -> c_int,
    pub event_destroy: unsafe extern "C" fn(Event) -> c_int,
    pub host_malloc: unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> c_int,
    pub host_free: unsafe extern "C" fn(*mut c_void) -> c_int,
    pub host_device_ptr: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, u32) -> c_int,
    pub module_load: unsafe extern "C" fn(*mut Module, *const c_void) -> c_int,
    pub module_unload: unsafe extern "C" fn(Module) -> c_int,
    pub function: unsafe extern "C" fn(*mut Func, Module, *const c_char) -> c_int,
    pub launch: unsafe extern "C" fn(Func, u32, u32, u32, u32, u32, u32, u32, Stream, *mut *mut c_void, *mut *mut c_void) -> c_int,
    pub func_set_attribute: unsafe extern "C" fn(*const c_void, c_int, c_int) -> c_int,
    pub error_string: unsafe extern "C" fn(c_int) -> *const c_char,
    pub last_error: unsafe extern "C" fn() -> c_int,
    _lib: Library,
}

impl Hip {
    /// The runtime, or an error naming what was missing. "HIP init" lets `backend::open` fall back to Vulkan.
    pub fn open() -> anyhow::Result<Hip> {
        let want = candidates(std::env::var_os("STRATA_HIP").map(Into::into),
                              &["libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"]);
        let mut tried = Vec::new();
        for path in &want {
            if let Ok(lib) = unsafe { Library::new(path) } {
                if let Some(h) = unsafe { Self::bind(lib) } { return Ok(h); }
                tried.push(format!("{}: missing symbols", path.display()));
            } else {
                tried.push(format!("{}: not loadable", path.display()));
            }
        }
        Err(anyhow::anyhow!("no HIP runtime: {}", tried.join(", "))).context("HIP init")
    }

    unsafe fn sym<T: Copy>(lib: &Library, name: &str) -> Option<T> {
        let s = unsafe { lib.get::<T>(format!("{name}\0").as_bytes()).ok() }?;
        Some(*s)
    }

    unsafe fn bind(lib: Library) -> Option<Hip> {
        macro_rules! g { ($n:literal) => { unsafe { Self::sym(&lib, $n)? } } }
        Some(Hip {
            init: g!("hipInit"),
            device_count: g!("hipGetDeviceCount"),
            set_device: g!("hipSetDevice"),
            device_name: g!("hipDeviceGetName"),
            device_attribute: g!("hipDeviceGetAttribute"),
            mem_get_info: g!("hipMemGetInfo"),
            malloc: g!("hipMalloc"),
            free: g!("hipFree"),
            memset: g!("hipMemset"),
            memset_async: g!("hipMemsetAsync"),
            memcpy: g!("hipMemcpy"),
            memcpy_async: g!("hipMemcpyAsync"),
            stream_create: g!("hipStreamCreateWithFlags"),
            stream_destroy: g!("hipStreamDestroy"),
            stream_sync: g!("hipStreamSynchronize"),
            stream_wait_event: g!("hipStreamWaitEvent"),
            stream_begin_capture: g!("hipStreamBeginCapture"),
            stream_end_capture: g!("hipStreamEndCapture"),
            stream_is_capturing: g!("hipStreamIsCapturing"),
            graph_instantiate: g!("hipGraphInstantiateWithFlags"),
            graph_launch: g!("hipGraphLaunch"),
            graph_exec_destroy: g!("hipGraphExecDestroy"),
            graph_destroy: g!("hipGraphDestroy"),
            event_create: g!("hipEventCreateWithFlags"),
            event_record: g!("hipEventRecord"),
            event_sync: g!("hipEventSynchronize"),
            event_query: g!("hipEventQuery"),
            event_destroy: g!("hipEventDestroy"),
            host_malloc: g!("hipHostMalloc"),
            host_free: g!("hipHostFree"),
            host_device_ptr: g!("hipHostGetDevicePointer"),
            module_load: g!("hipModuleLoadData"),
            module_unload: g!("hipModuleUnload"),
            function: g!("hipModuleGetFunction"),
            launch: g!("hipModuleLaunchKernel"),
            func_set_attribute: g!("hipFuncSetAttribute"),
            error_string: g!("hipGetErrorString"),
            last_error: g!("hipGetLastError"),
            _lib: lib,
        })
    }

    pub fn err(&self, code: c_int) -> String {
        let s = unsafe { (self.error_string)(code) };
        if s.is_null() { return format!("hip error {code}") }
        unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned()
    }

    pub fn check(&self, code: c_int, what: &str) -> anyhow::Result<()> {
        if code == HIP_SUCCESS { return Ok(()) }
        bail!("{what}: {}", self.err(code))
    }

    /// The device's `gfxNNNN` target, from `hipDeviceProp_t`'s `gcnArchName`. `STRATA_HIP_ARCH` overrides it.
    /// ROCm 7 dropped the `hipDeviceAttributeArch`/`GcnArchName` attributes, so the property struct is the only
    /// source; it is version-specific, hence the field-length and shape check before the value is trusted.
    pub fn arch(&self, device: c_int) -> anyhow::Result<String> {
        if let Ok(a) = std::env::var("STRATA_HIP_ARCH") { return Ok(a) }
        if let Some(a) = self.prop_arch(device) { return Ok(a) }
        bail!("cannot read the device's gfx architecture; set STRATA_HIP_ARCH (e.g. gfx1201)")
    }

    /// The device property struct is version-specific (`hipDeviceProp_t` and `hipDeviceProp_tR0600` differ, and ROCm 7
    /// resolves `hipGetDeviceProperties` to the latter), so the target name is found by shape inside the struct
    /// instead of at a fixed offset: the first `gfxNNNN`-looking string, validated.
    fn prop_arch(&self, device: c_int) -> Option<String> {
        // Resolved here and not in `bind`: this is the only use, and a missing symbol must not fail the backend.
        let f: unsafe extern "C" fn(*mut c_void, c_int) -> c_int = unsafe { Self::sym(&self._lib, "hipGetDeviceProperties") }?;
        let mut buf = vec![0u64; PROP_BUF / 8];
        let r = unsafe { f(buf.as_mut_ptr().cast(), device) };
        if r != HIP_SUCCESS { return None }
        find_gfx(unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), PROP_BUF) })
    }

    pub fn device_name(&self, device: c_int) -> String {
        let mut buf = [0 as c_char; 256];
        let r = unsafe { (self.device_name)(buf.as_mut_ptr(), 256, device) };
        if r != HIP_SUCCESS { return "AMD GPU".into() }
        unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned()
    }

    pub fn attribute(&self, attr: c_int, device: c_int) -> anyhow::Result<c_int> {
        let mut v = 0;
        self.check(unsafe { (self.device_attribute)(&mut v, attr, device) }, "hipDeviceGetAttribute")?;
        Ok(v)
    }
}

pub struct Rtc {
    version: unsafe extern "C" fn(*mut c_int, *mut c_int) -> c_int,
    create: unsafe extern "C" fn(*mut RtcProgram, *const c_char, *const c_char, c_int, *const *const c_char, *const *const c_char) -> c_int,
    compile: unsafe extern "C" fn(RtcProgram, c_int, *const *const c_char) -> c_int,
    log_size: unsafe extern "C" fn(RtcProgram, *mut usize) -> c_int,
    log: unsafe extern "C" fn(RtcProgram, *mut c_char) -> c_int,
    code_size: unsafe extern "C" fn(RtcProgram, *mut usize) -> c_int,
    code: unsafe extern "C" fn(RtcProgram, *mut c_char) -> c_int,
    destroy: unsafe extern "C" fn(*mut RtcProgram) -> c_int,
    error_string: unsafe extern "C" fn(c_int) -> *const c_char,
    lib_path: PathBuf,
    _lib: Library,
}

static RTC: OnceLock<Result<Rtc, String>> = OnceLock::new();

impl Rtc {
    /// The library, or which paths were tried and why each failed.
    pub fn get() -> Result<&'static Rtc, &'static str> { RTC.get_or_init(Self::open).as_ref().map_err(|e| e.as_str()) }

    fn open() -> Result<Rtc, String> {
        let want = candidates(std::env::var_os("STRATA_HIPRTC").map(Into::into),
                              &["libhiprtc.so", "libhiprtc.so.7", "libhiprtc.so.6"]);
        let mut tried = Vec::new();
        for path in &want {
            let lib = match unsafe { Library::new(path) } {
                Ok(l) => l,
                Err(e) => { tried.push(format!("{}: not loadable ({e})", path.display())); continue }
            };
            unsafe {
                macro_rules! g { ($n:literal) => { match Self::sym(&lib, $n) { Some(f) => f, None => { tried.push(format!("{}: no {}", path.display(), $n)); continue } } } }
                return Ok(Rtc {
                    version: g!("hiprtcVersion"),
                    create: g!("hiprtcCreateProgram"),
                    compile: g!("hiprtcCompileProgram"),
                    log_size: g!("hiprtcGetProgramLogSize"),
                    log: g!("hiprtcGetProgramLog"),
                    code_size: g!("hiprtcGetCodeSize"),
                    code: g!("hiprtcGetCode"),
                    destroy: g!("hiprtcDestroyProgram"),
                    error_string: g!("hiprtcGetErrorString"),
                    lib_path: path.clone(),
                    _lib: lib,
                });
            }
        }
        Err(format!("no hiprtc: {}", tried.join(", ")))
    }

    unsafe fn sym<T: Copy>(lib: &Library, name: &str) -> Option<T> {
        let s = unsafe { lib.get::<T>(format!("{name}\0").as_bytes()).ok() }?;
        Some(*s)
    }

    pub fn version(&self) -> (i32, i32) {
        let (mut maj, mut min) = (0, 0);
        unsafe { (self.version)(&mut maj, &mut min) };
        (maj, min)
    }

    fn err(&self, code: c_int) -> String {
        let s = unsafe { (self.error_string)(code) };
        if s.is_null() { return format!("hiprtc error {code}") }
        unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned()
    }

    /// Directories holding `hip/hip_runtime.h`: hiprtc compiles a string in a scratch directory, so the ROCm headers
    /// have to be named. Found next to the loaded `libhiprtc` and through the environment.
    fn include_dirs(&self) -> Vec<PathBuf> {
        let mut v = Vec::new();
        for k in ["ROCM_PATH", "HIP_PATH"] {
            if let Some(p) = std::env::var_os(k) { v.push(PathBuf::from(p).join("include")); }
        }
        if let Some(dir) = self.lib_path.parent() {
            // libhiprtc.so -> <root>/lib, headers in <root>/include (also <root>/include/hip).
            v.push(dir.join("../include"));
        }
        for p in ["/opt/rocm/include", "/usr/include", "/usr/local/include"] { v.push(PathBuf::from(p)); }
        v.retain(|p| p.join("hip/hip_runtime.h").exists());
        v
    }

    /// Compile `src` to an AMDGPU code object for `arch` (`gfx1201`). `name` only labels diagnostics.
    pub fn compile(&self, src: &str, name: &str, arch: &str, extra: &[&str]) -> anyhow::Result<Vec<u8>> {
        let csrc = std::ffi::CString::new(src).context("kernel source contains a NUL byte")?;
        let cname = std::ffi::CString::new(name)?;
        let mut prog: RtcProgram = std::ptr::null_mut();
        let code = unsafe { (self.create)(&mut prog, csrc.as_ptr(), cname.as_ptr(), 0, std::ptr::null(), std::ptr::null()) };
        if code != HIP_SUCCESS { bail!("hiprtcCreateProgram: {}", self.err(code)) }
        let guard = ProgGuard { prog, destroy: self.destroy };
        let mut opts: Vec<std::ffi::CString> = vec![
            std::ffi::CString::new(format!("--offload-arch={arch}"))?,
            std::ffi::CString::new("-O3")?,
            std::ffi::CString::new("--std=c++17")?,
        ];
        for dir in self.include_dirs() { opts.push(std::ffi::CString::new(format!("-I{}", dir.display()))?); }
        for e in extra { opts.push(std::ffi::CString::new(*e)?); }
        let ptrs: Vec<*const c_char> = opts.iter().map(|o| o.as_ptr()).collect();
        let code = unsafe { (self.compile)(guard.prog, ptrs.len() as c_int, ptrs.as_ptr()) };
        let mut logsize = 0usize;
        unsafe { (self.log_size)(guard.prog, &mut logsize) };
        if logsize > 1 {
            let mut log = vec![0 as c_char; logsize];
            unsafe { (self.log)(guard.prog, log.as_mut_ptr()) };
            let text = unsafe { CStr::from_ptr(log.as_ptr()) }.to_string_lossy();
            if code != HIP_SUCCESS { bail!("hiprtc ({arch}): {}\n{}", self.err(code), text.trim_end()) }
            let text = text.trim_end();
            if !text.is_empty() { eprintln!("hiprtc {name} ({arch}): {text}") }
        } else if code != HIP_SUCCESS {
            bail!("hiprtc ({arch}): {}", self.err(code))
        }
        let mut n = 0usize;
        unsafe { (self.code_size)(guard.prog, &mut n) };
        anyhow::ensure!(n > 0, "hiprtc produced no code for {arch}");
        let mut code_obj = vec![0u8; n];
        let r = unsafe { (self.code)(guard.prog, code_obj.as_mut_ptr().cast()) };
        if r != HIP_SUCCESS { bail!("hiprtcGetCode: {}", self.err(r)) }
        Ok(code_obj)
    }
}

struct ProgGuard {
    prog: RtcProgram,
    destroy: unsafe extern "C" fn(*mut RtcProgram) -> c_int,
}

impl Drop for ProgGuard {
    fn drop(&mut self) { unsafe { (self.destroy)(&mut self.prog) }; }
}

#[cfg(test)]
mod tests {
    use super::find_gfx;

    /// The name is found at the end of the buffer without reading past it, a short `gfx` is passed over, and a
    /// buffer without one gives None.
    #[test]
    fn find_gfx_reads_only_inside_the_buffer() {
        let mut b = vec![0u8; 64];
        b[8..11].copy_from_slice(b"gfx");
        b[56..63].copy_from_slice(b"gfx1201");
        assert_eq!(find_gfx(&b).as_deref(), Some("gfx1201"));
        assert_eq!(find_gfx(&b[..60]), None);
        let mut tail = vec![0u8; 61];
        tail[52..61].copy_from_slice(b"gfx1100ab");
        assert_eq!(find_gfx(&tail).as_deref(), Some("gfx1100ab"));
        assert_eq!(find_gfx(&[0u8; 32]), None);
    }
}
