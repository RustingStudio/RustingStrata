//! NVRTC, loaded here and not through cudarc: cudarc fixes its library name list at build time to one CUDA version. On Linux
//! `libnvrtc.so` covers every toolkit, but Windows DLL names carry the version (`nvrtc64_120_0.dll`), so a binary built for
//! CUDA 13 never found a 12.x NVRTC there.

use anyhow::{Context, bail};
use libloading::Library;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

type Prog = *mut c_void;

struct Nvrtc {
    version: unsafe extern "C" fn(*mut c_int, *mut c_int) -> c_int,
    create: unsafe extern "C" fn(*mut Prog, *const c_char, *const c_char, c_int, *const *const c_char, *const *const c_char) -> c_int,
    compile: unsafe extern "C" fn(Prog, c_int, *const *const c_char) -> c_int,
    log_size: unsafe extern "C" fn(Prog, *mut usize) -> c_int,
    log: unsafe extern "C" fn(Prog, *mut c_char) -> c_int,
    cubin_size: unsafe extern "C" fn(Prog, *mut usize) -> c_int,
    cubin: unsafe extern "C" fn(Prog, *mut c_char) -> c_int,
    destroy: unsafe extern "C" fn(*mut Prog) -> c_int,
    _lib: Library,
}

/// Library names and paths to try, in order. `STRATA_NVRTC` (a path or a name) replaces the whole list.
pub fn candidates() -> Vec<PathBuf> {
    names(cfg!(windows), std::env::var_os("STRATA_NVRTC").map(Into::into), std::env::var_os("CUDA_PATH").map(Into::into))
}

fn names(windows: bool, forced: Option<PathBuf>, cuda_path: Option<PathBuf>) -> Vec<PathBuf> {
    if let Some(p) = forced { return vec![p]; }
    let mut v: Vec<PathBuf> = cudarc::get_lib_name_candidates("nvrtc").into_iter().map(Into::into).collect();
    if windows {
        // Every CUDA 12.x names its NVRTC `nvrtc64_120_0.dll`; cudarc's list has only the build's own major (13).
        const DLLS: [&str; 2] = ["nvrtc64_130_0.dll", "nvrtc64_120_0.dll"];
        v.extend(DLLS.map(PathBuf::from));
        // The toolkit installer sets CUDA_PATH; its DLL directory (`bin`, `bin\x64` since 13) is not always on PATH.
        if let Some(root) = cuda_path {
            for dir in ["bin", "bin/x64"] { for dll in DLLS { v.push(root.join(dir).join(dll)); } }
        }
    }
    v
}

/// `None` when the file does not load or lacks one of the NVRTC symbols used here.
fn open(path: &Path) -> Option<Nvrtc> {
    unsafe {
        let lib = Library::new(path).ok()?;
        // NVRTC loads `nvrtc-builtins64_*.dll` by bare name at compile time. Windows does not search the directory of a DLL
        // loaded by full path, so load the builtins first: a bare-name load then resolves to the module already in memory.
        if let Some(dir) = path.parent().filter(|d| cfg!(windows) && d.is_absolute()) {
            for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
                if e.file_name().to_string_lossy().starts_with("nvrtc-builtins") {
                    if let Ok(l) = Library::new(e.path()) { std::mem::forget(l); }
                }
            }
        }
        Some(Nvrtc {
            version: *lib.get(b"nvrtcVersion\0").ok()?,
            create: *lib.get(b"nvrtcCreateProgram\0").ok()?,
            compile: *lib.get(b"nvrtcCompileProgram\0").ok()?,
            log_size: *lib.get(b"nvrtcGetProgramLogSize\0").ok()?,
            log: *lib.get(b"nvrtcGetProgramLog\0").ok()?,
            cubin_size: *lib.get(b"nvrtcGetCUBINSize\0").ok()?,
            cubin: *lib.get(b"nvrtcGetCUBIN\0").ok()?,
            destroy: *lib.get(b"nvrtcDestroyProgram\0").ok()?,
            _lib: lib,
        })
    }
}

fn lib() -> Option<&'static Nvrtc> {
    static LIB: OnceLock<Option<Nvrtc>> = OnceLock::new();
    LIB.get_or_init(|| candidates().iter().find_map(|p| open(p))).as_ref()
}

pub fn present() -> bool { lib().is_some() }

/// The version of the NVRTC that was loaded, `None` when no library loads. Device-free: no driver call.
pub fn version() -> Option<(i32, i32)> {
    let (mut major, mut minor) = (0, 0);
    (unsafe { (lib()?.version)(&mut major, &mut minor) } == 0).then_some((major, minor))
}

/// The error for a machine with no usable NVRTC: names what was searched, so a wrong DLL name is visible to the user.
pub fn missing() -> anyhow::Error {
    let tried: Vec<String> = candidates().iter().map(|p| p.display().to_string()).collect();
    anyhow::anyhow!(
        "the CUDA NVRTC library is not installed or not on the library path (tried: {}); install the CUDA toolkit or its \
         NVRTC package (12.x for sm_70: 13.x dropped Volta support), or set STRATA_NVRTC to the library file",
        tried.join(", "))
}

/// Compile `src` to a cubin for `arch`.
pub fn cubin(src: &str, arch: &str) -> anyhow::Result<Vec<u8>> {
    let n = lib().ok_or_else(missing)?;
    let src = CString::new(src)?;
    let opts = [CString::new(format!("--gpu-architecture={arch}"))?, CString::new("-std=c++17")?];
    let ptrs: Vec<*const c_char> = opts.iter().map(|o| o.as_ptr()).collect();
    let ok = |r: c_int, what: &str| if r == 0 { Ok(()) } else { Err(anyhow::anyhow!("{what} failed: NVRTC error {r}")) };
    unsafe {
        let mut prog: Prog = std::ptr::null_mut();
        ok((n.create)(&mut prog, src.as_ptr(), std::ptr::null(), 0, std::ptr::null(), std::ptr::null()), "nvrtcCreateProgram")?;
        let out = (|| {
            let r = (n.compile)(prog, ptrs.len() as c_int, ptrs.as_ptr());
            if r != 0 {
                let mut len = 0;
                let mut log = vec![0u8; if (n.log_size)(prog, &mut len) == 0 { len.max(1) } else { 1 }];
                (n.log)(prog, log.as_mut_ptr() as *mut c_char);
                let log = CStr::from_bytes_until_nul(&log).map(|l| l.to_string_lossy().into_owned()).unwrap_or_default();
                bail!("NVRTC compile of kernels.cu failed: NVRTC error {r}\n{log}");
            }
            let mut len = 0;
            ok((n.cubin_size)(prog, &mut len), "nvrtcGetCUBINSize")?;
            let mut b = vec![0u8; len];
            ok((n.cubin)(prog, b.as_mut_ptr() as *mut c_char), "nvrtcGetCUBIN")?;
            Ok(b)
        })();
        (n.destroy)(&mut prog);
        out.context("NVRTC")
    }
}

#[cfg(test)]
mod tests {
    use super::names;
    use std::path::PathBuf;

    /// Issue 35: the CUDA 12.x Windows DLL, by bare name and under CUDA_PATH, must be on the list.
    #[test]
    fn windows_names_cover_cuda12_and_cuda_path() {
        let v = names(true, None, Some(PathBuf::from("C:/cuda")));
        for want in ["nvrtc64_120_0.dll", "nvrtc64_130_0.dll", "C:/cuda/bin/nvrtc64_120_0.dll", "C:/cuda/bin/x64/nvrtc64_130_0.dll"] {
            assert!(v.contains(&PathBuf::from(want)), "{want} missing from {v:?}");
        }
        assert!(!names(false, None, Some(PathBuf::from("/opt/cuda"))).iter().any(|p| p.to_string_lossy().contains("120_0")));
        assert_eq!(names(true, Some(PathBuf::from("x.dll")), None), [PathBuf::from("x.dll")]);
    }
}
