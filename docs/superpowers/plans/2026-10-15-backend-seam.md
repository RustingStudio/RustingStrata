# Backend Seam Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put a `Backend` trait between the model code and CUDA, so `cudarc` is used only under `src/cuda/`, with no change in output or speed.

**Architecture:** New `src/backend.rs` holds `Buf` (device address, byte length, and an owner that frees the memory on drop), the moved `GpuTensor`, the shared tuning constants, and `trait Backend`: memory ops, a recorded-graph call, a host-mapped mailbox, and the kernel ops the model uses (same signatures as today). `Gpu` implements it. `GpuModel` holds `Box<dyn Backend>`, and the model files use `Buf` in place of `CudaSlice`.

**Tech Stack:** Rust 2024, cudarc 0.19.10 (driver + nvrtc, dynamic loading), anyhow.

**Spec:** `docs/superpowers/specs/2026-10-15-vulkan-and-arch-table-design.md`, Part 1.

## Global Constraints

- No behavior change. Greedy output for a fixed prompt must be byte-identical to v1.0.0 (tag `v1.0.0`).
- Speed gate (spec): `strata bench --gpu --pp 512,4096 --tg 128 --reps 3` inside the v1.0.0 ranges: prefill 4K 2089-2096, decode after 4K 95.4-95.8, short decode 100.6-101.5 tok/s. CPU-load caveats from `docs/LOOP_LOG.md` apply, so the gate is an interleaved A/B run against a v1.0.0 build on the same machine state.
- Parity gate (spec): `tools/parity.py` agreement unchanged (default `--gpu` at v1.0.0: 631-634 of 640, either-mode count).
- `cargo test --release` passes.
- After the plan, `grep -rn cudarc src --include=*.rs | grep -v '^src/cuda/'` prints nothing.
- No new dependencies. No cargo features.
- The trait is extracted, not redesigned. Op signatures stay exactly as in `src/cuda/mod.rs`.
- Persisted text (comments, docs, commits) is in plain English. Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Do not commit the user's staged changes (`Cargo.toml`, `LICENSE.md`, `README.md`). Commit with `git commit -- <paths>` and list only this task's files.
- Shell blocks are bash. The interactive shell on this machine is fish, so run multi-line blocks with `bash -c '...'` or from a script.
- GPU tests load a 12 GB model. Run them one at a time (`--test-threads=1` is not needed: they hold `common::gpu_lock()`), and with no other large GPU process running (check `nvidia-smi`).

## Deviations from the spec (decided here)

- The spec lists `record(key, f)` plus `replay(key) -> bool`. This plan uses one method, `graph(key, f)`: it replays the graph for `key` if one exists, and otherwise records `f` and runs it. Both callers (`run`, `mtp_run`) did exactly that sequence, so one method removes the duplicated capture code. `STRATA_NO_GRAPH` is handled inside `graph`, so it now also covers the MTP draft head (a debug variable, so this does not change normal runs).
- `host_mailbox` returns `(Buf, *mut u32)`, not `(*mut u32, u64)`. The `Buf` owns the pinned memory, so `CpuWorker` keeps it alive the same way it keeps `PinnedHostSlice` alive today.
- `dtod` takes offsets and a byte count (`src, src_off, dst, dst_off, bytes`). `rollback` and `prefill` copy sub-ranges.
- `upload` is a provided trait method (`alloc` + `htod`). Typed helpers (`zeros::<T>`, `put`, `get`) live in `impl dyn Backend`, because generic methods cannot be called through a trait object.
- Ops used only by tests (`dequant`, `quant_xq`, `attn_decode_gq`, `mmq`) stay inherent on `Gpu`. Part 3 moves them into the trait when the per-backend kernel tests need them.
- `tests/cuda.rs` keeps its name until Part 3 adds a second backend. This plan adds `tests/backend.rs` for the new memory, graph and mailbox calls.

## File map

| File | Change |
|---|---|
| `src/backend.rs` | New: `Buf`, `Plain`, `GpuTensor` (moved), constants (moved), `trait Backend`, typed helpers. |
| `src/lib.rs` | `pub mod backend;` |
| `src/cuda/mod.rs` | `impl Backend for Gpu` (memory, graph, mailbox, and the op methods moved out of `impl Gpu`); re-export of the moved items. |
| `src/cpu_worker.rs` | Mailbox from `Backend::host_mailbox`; no cudarc. |
| `src/expert_cache.rs` | `Buf` and `&dyn Backend`; `base` loses its `gpu` parameter. |
| `src/gpu_model.rs`, `src/gpu_model/prefill.rs` | `Box<dyn Backend>`, `Buf`, `graph()`; no cudarc. |
| `src/main.rs` | `Box::new(Gpu::new()?)` at the two `GpuModel::load` sites. |
| `tests/backend.rs` | New: memory round trip, graph record/replay, host mailbox. |
| `tests/cuda.rs`, `tests/kv_q8.rs`, `tests/llama_numerics.rs`, `tests/gpu_model.rs` | Import `Backend`; `Box::new(gpu)` at `GpuModel::load`. |
| `docs/LOOP_LOG.md` | Entry #85 with the gate measurements. |

---

### Task 1: `Backend` trait, CUDA implementation, seam tests

**Files:**
- Create: `src/backend.rs`, `tests/backend.rs`
- Modify: `src/lib.rs`, `src/cuda/mod.rs`, `tests/cuda.rs`, `tests/kv_q8.rs`, `tests/llama_numerics.rs`, `src/gpu_model.rs` (import only), `src/gpu_model/prefill.rs` (import only)

**Interfaces:**
- Produces (used by Tasks 2-4):
  - `pub struct Buf { pub addr: u64, pub len: usize, .. }` with `Buf::new(addr: u64, len: usize, owner: impl Any + Send + Sync) -> Buf`. `len` is in bytes.
  - `pub unsafe trait Plain`, implemented for `f32, i32, u32, u16, u8, u64`.
  - `pub struct GpuTensor { pub ty, pub rows, pub cols, pub ptr: u64, .. }` with `bytes()`.
  - `pub const ATT_SPLITS: usize = 32; pub const ATT_GQA_MIN: i32 = 2048; pub const GEMM_BM: usize = 128;`
  - `trait Backend`: `alloc`, `htod`, `dtoh`, `dtod`, `memset`, `mem_info`, `sync`, `graph`, `host_mailbox`, `upload` (provided), `kv_words`, plus 26 ops (listed in Step 3).
  - `impl dyn Backend`: `zeros::<T>(n) -> Buf`, `put(src: &[T], dst: &Buf, at: usize)`, `get::<T>(src: &Buf, at: usize, n: usize) -> Vec<T>`. `at` and `n` count elements of `T`.

- [ ] **Step 1: Write the failing test**

Create `tests/backend.rs`:

```rust
use rusting_strata::{backend::Backend, cuda::Gpu};

/// Skip only when there is no usable CUDA device; kernel compile errors fail the test.
fn backend() -> Option<Box<dyn Backend>> {
    match Gpu::new() {
        Ok(g) => Some(Box::new(g)),
        Err(e) if format!("{e}") == "CUDA init" => { eprintln!("SKIP: no GPU: {e:#}"); None }
        Err(e) => panic!("{e:#}"),
    }
}

#[test]
fn memory_ops_round_trip() {
    let Some(g) = backend() else { return };
    let a = g.zeros::<f32>(8).unwrap();
    assert_eq!(a.len, 32);
    assert_eq!(g.get::<f32>(&a, 0, 8).unwrap(), [0.0; 8]);
    g.put(&[1.0f32, 2.0, 3.0], &a, 2).unwrap();
    assert_eq!(g.get::<f32>(&a, 0, 6).unwrap(), [0.0, 0.0, 1.0, 2.0, 3.0, 0.0]);
    let b = g.zeros::<f32>(4).unwrap();
    g.dtod(&a, 8, &b, 4, 12).unwrap();
    assert_eq!(g.get::<f32>(&b, 0, 4).unwrap(), [0.0, 1.0, 2.0, 3.0]);
    g.memset(&a).unwrap();
    assert_eq!(g.get::<f32>(&a, 0, 8).unwrap(), [0.0; 8]);
    assert!(g.put(&[0f32; 9], &a, 0).is_err(), "a copy past the end must fail");
    let (free, total) = g.mem_info().unwrap();
    assert!(free > 0 && free <= total);
}

#[test]
fn graph_records_once_and_replays() {
    let Some(g) = backend() else { return };
    if std::env::var_os("STRATA_NO_GRAPH").is_some() { return; }
    let (x, y) = (g.zeros::<f32>(4).unwrap(), g.zeros::<f32>(4).unwrap());
    g.put(&[1.0f32, 2.0, 3.0, 4.0], &y, 0).unwrap();
    let mut calls = 0;
    for _ in 0..3 { g.graph(5, &mut || { calls += 1; g.add(x.addr, y.addr, 4) }).unwrap(); }
    assert_eq!(calls, 1, "f runs only when recording");
    assert_eq!(g.get::<f32>(&x, 0, 4).unwrap(), [3.0, 6.0, 9.0, 12.0]);
}

#[test]
fn host_mailbox_is_shared_with_kernels() {
    let Some(g) = backend() else { return };
    let (mb, host) = g.host_mailbox(4).unwrap();
    let ones = g.zeros::<f32>(4).unwrap();
    g.put(&[1.0f32; 4], &ones, 0).unwrap();
    unsafe { host.cast::<f32>().write_volatile(2.0) };
    g.add(mb.addr, ones.addr, 4).unwrap(); // the kernel reads and writes host memory
    g.sync().unwrap();
    let got: Vec<f32> = (0..4).map(|i| unsafe { host.cast::<f32>().add(i).read_volatile() }).collect();
    assert_eq!(got, [3.0, 1.0, 1.0, 1.0]);
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test --release --test backend`
Expected: compile error, `unresolved import rusting_strata::backend`.

- [ ] **Step 3: Write `src/backend.rs`**

```rust
//! Backend seam: device memory as `Buf` (an address plus whatever keeps the memory alive) and the kernel ops the
//! model runs. Every op takes device addresses, so model code never sees backend types. CUDA implements it in `cuda`.

use crate::quant::GgmlType;
use crate::weights::Tensor;
use std::any::Any;

/// Blocks per head in decode attention (positions split across them). Scratch sizes depend on it.
pub const ATT_SPLITS: usize = 32;
/// Positions from which decode attention shares K/V loads across a GQA group (break-even measured near 2K).
pub const ATT_GQA_MIN: i32 = 2048;
const _: () = assert!(ATT_SPLITS <= 256); // attn_merge keeps one weight per split in a 256-thread block
/// Positions per GEMM tile. Prefill cuts its tile lists at this size.
pub const GEMM_BM: usize = 128;

/// `len` bytes of device memory at device address `addr`, freed when the owner drops.
pub struct Buf {
    pub addr: u64,
    pub len: usize,
    _owner: Box<dyn Any + Send + Sync>,
}

impl Buf {
    pub fn new(addr: u64, len: usize, owner: impl Any + Send + Sync) -> Buf { Buf { addr, len, _owner: Box::new(owner) } }
}

pub struct GpuTensor {
    pub ty: GgmlType,
    pub rows: usize,
    pub cols: usize,
    pub ptr: u64,
    _buf: Buf,
}

impl GpuTensor {
    pub fn bytes(&self) -> usize { self.ty.row_bytes(self.cols as u64) * self.rows }
}

/// Values copied to and from device memory as raw bytes.
/// Safety: no padding bytes, and every bit pattern is a valid value.
pub unsafe trait Plain: Copy + Default + 'static {}
unsafe impl Plain for f32 {}
unsafe impl Plain for i32 {}
unsafe impl Plain for u32 {}
unsafe impl Plain for u16 {}
unsafe impl Plain for u8 {}
unsafe impl Plain for u64 {}

/// Device memory, recorded graphs and kernel ops. Host copies are synchronous; device work runs in queue order.
/// The ops are documented in `cuda/mod.rs`.
pub trait Backend {
    /// Zeroed buffer of `bytes` (at least 1).
    fn alloc(&self, bytes: usize) -> anyhow::Result<Buf>;
    fn htod(&self, src: &[u8], dst: &Buf, off: usize) -> anyhow::Result<()>;
    fn dtoh(&self, src: &Buf, off: usize, dst: &mut [u8]) -> anyhow::Result<()>;
    fn dtod(&self, src: &Buf, src_off: usize, dst: &Buf, dst_off: usize, bytes: usize) -> anyhow::Result<()>;
    /// Zero all of `dst`.
    fn memset(&self, dst: &Buf) -> anyhow::Result<()>;
    /// Free and total device memory in bytes.
    fn mem_info(&self) -> anyhow::Result<(usize, usize)>;
    /// Wait for all queued device work.
    fn sync(&self) -> anyhow::Result<()>;
    /// Run the ops `f` queues as the graph recorded under `key`, recording it on first use. `f` runs only when
    /// recording, so its ops must read per-call values from device memory.
    fn graph(&self, key: u64, f: &mut dyn FnMut() -> anyhow::Result<()>) -> anyhow::Result<()>;
    /// Zeroed host memory of `words` u32 that kernels read and write at the `Buf` address. The host pointer is
    /// valid while the `Buf` lives.
    fn host_mailbox(&self, words: usize) -> anyhow::Result<(Buf, *mut u32)>;

    fn upload(&self, t: &Tensor) -> anyhow::Result<GpuTensor> {
        let b = self.alloc(t.data.len())?;
        self.htod(t.data, &b, 0)?;
        Ok(GpuTensor { ty: t.ty, rows: t.rows, cols: t.cols, ptr: b.addr, _buf: b })
    }

    fn kv_words(&self, n: usize) -> usize;
    fn gemm(&self, ty: GgmlType, rows: usize, cols: usize, tiles: u64, n_tiles: usize, x: u64, x_stride: usize, n_x: usize, xrow: u64, y: u64, y_stride: usize, act: i32) -> anyhow::Result<()>;
    fn gemm_dense(&self, w: &GpuTensor, n: usize, x: u64, x_stride: usize, y: u64, y_stride: usize, act: i32) -> anyhow::Result<()>;
    fn gemv(&self, w: &GpuTensor, x: u64, y: u64) -> anyhow::Result<()>;
    fn gemvn(&self, w: &GpuTensor, x: u64, x_stride: usize, y: u64, y_stride: usize, nt: usize) -> anyhow::Result<()>;
    fn gemvq(&self, w: &GpuTensor, x: u64, x_stride: usize, xq: u64, y: u64, y_stride: usize, nt: usize) -> anyhow::Result<()>;
    fn moe_gemv(&self, ty: GgmlType, rows: usize, cols: usize, w: u64, stride: usize, ids: u64, k: usize, x: u64, x_stride: usize, xg: usize, y: u64, pair: Option<(i64, i64)>, xq: u64) -> anyhow::Result<()>;
    fn map_hits(&self, ids: u64, table: u64, slots: u64, wts: u64, wh: u64, k: usize) -> anyhow::Result<()>;
    fn rms_norm(&self, x: u64, x_stride: usize, w: u64, out: u64, out_stride: usize, n: usize, rows: usize, eps: f32) -> anyhow::Result<()>;
    fn rms_norm_q(&self, x: u64, x_stride: usize, w: u64, out: u64, out_stride: usize, n: usize, rows: usize, eps: f32, xq: u64) -> anyhow::Result<()>;
    fn argmax_rows(&self, x: u64, n: usize, rows: usize, out: u64) -> anyhow::Result<()>;
    fn l2_norm(&self, x: u64, n: usize, per: usize, stride: usize, rows: usize, eps: f32) -> anyhow::Result<()>;
    fn rope_neox(&self, x: u64, stride: usize, heads: usize, per: usize, n_rot: usize, step: u64, base: f32) -> anyhow::Result<()>;
    fn add(&self, x: u64, y: u64, n: usize) -> anyhow::Result<()>;
    fn swiglu(&self, g: u64, u: u64, n: usize) -> anyhow::Result<()>;
    fn ssm_gates(&self, beta: u64, g: u64, dt_bias: u64, a: u64, n: usize, nv: usize) -> anyhow::Result<()>;
    fn conv_silu(&self, qkv: u64, state: u64, w: u64, y: u64, ch: usize, kw: usize) -> anyhow::Result<()>;
    fn deltanet(&self, y: u64, s: u64, beta: u64, g: u64, o: u64, nk: usize, nv: usize, dk: usize, dv: usize, scale: f32, t: usize, ych: usize, snap: u64, nsnap: usize) -> anyhow::Result<()>;
    fn conv_seq(&self, qkv: u64, state: u64, w: u64, y: u64, ch: usize, kw: usize, t: usize, snap: u64, nsnap: usize) -> anyhow::Result<()>;
    fn attn_prefill(&self, q: u64, kc: u64, vc: u64, qg: u64, out: u64, step: u64, t: usize, nh: usize, nkv: usize, hd: usize, scale: f32) -> anyhow::Result<()>;
    fn moe_combine_seq(&self, x: u64, ye: u64, wts: u64, inv: u64, k: usize, sh: u64, sg: u64, n: usize, t: usize) -> anyhow::Result<()>;
    fn norm_gate(&self, o: u64, w: u64, z: u64, n: usize, heads: usize, eps: f32) -> anyhow::Result<()>;
    fn kv_store(&self, k: u64, v: u64, kc: u64, vc: u64, n: usize, t: usize, step: u64, hd: usize) -> anyhow::Result<()>;
    fn attn_decode(&self, q: u64, kc: u64, vc: u64, qg: u64, out: u64, part: u64, step: u64, nh: usize, nkv: usize, hd: usize, scale: f32, nt: usize) -> anyhow::Result<()>;
    fn router_topk(&self, p: u64, n: usize, k: usize, extra_scale: f32, ids: u64, wts: u64, t: usize) -> anyhow::Result<()>;
    fn moe_publish(&self, ids: u64, wts: u64, slots: u64, h: u64, mb: u64, k: usize, n: usize, h_off: usize, step: u64, t: usize) -> anyhow::Result<()>;
    fn moe_combine(&self, x: u64, ye: u64, w: u64, k: usize, sh: u64, sg: u64, mb: u64, y_off: usize, step: u64, slots: u64, n: usize, t: usize) -> anyhow::Result<()>;
}

/// Typed copies. Positions and counts are in elements of `T`.
impl<'a> dyn Backend + 'a {
    /// Zeroed buffer of `n` values of `T`.
    pub fn zeros<T: Plain>(&self, n: usize) -> anyhow::Result<Buf> { self.alloc(n * size_of::<T>()) }

    /// Copy `src` into `dst` from element `at` on.
    pub fn put<T: Plain>(&self, src: &[T], dst: &Buf, at: usize) -> anyhow::Result<()> {
        let b = unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<u8>(), size_of_val(src)) };
        self.htod(b, dst, at * size_of::<T>())
    }

    /// `n` values of `T` from `src`, starting at element `at`.
    pub fn get<T: Plain>(&self, src: &Buf, at: usize, n: usize) -> anyhow::Result<Vec<T>> {
        let mut v = vec![T::default(); n];
        let b = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), n * size_of::<T>()) };
        self.dtoh(src, at * size_of::<T>(), b)?;
        Ok(v)
    }
}
```

Add `pub mod backend;` to `src/lib.rs`, before `pub mod cuda;`.

- [ ] **Step 4: Implement `Backend` for `Gpu` in `src/cuda/mod.rs`**

4a. Imports and moved items. Replace the cudarc import line and delete the `ATT_SPLITS`, `ATT_GQA_MIN`, the `const _` assert, `GEMM_BM`, `GpuTensor` and `impl GpuTensor` definitions (lines 23-44 today, except `ATT_SPLITS_1`, which stays). Add:

```rust
pub use crate::backend::{ATT_GQA_MIN, ATT_SPLITS, GEMM_BM, GpuTensor};
use crate::backend::{Backend, Buf};
use cudarc::driver::{CudaContext, CudaFunction, CudaGraph, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg, result, sys};
```

The `pub use` keeps `rusting_strata::cuda::{ATT_SPLITS, ATT_GQA_MIN, GEMM_BM}` working for the tests.

4b. Add a graph table to `Gpu`:

```rust
    /// Recorded graphs by `Backend::graph` key.
    graphs: std::sync::Mutex<HashMap<u64, CudaGraph>>,
```

and `graphs: Default::default()` in the struct literal at the end of `Gpu::new`.

4c. Delete the inherent `upload` (the trait provides it). Change the `.with_context` in `ptx`/`new` only if the compiler asks; nothing else in `new` changes.

4d. Move these methods out of `impl Gpu` into a new `impl Backend for Gpu` block, bodies unchanged, with `pub` removed: `kv_words`, `gemm`, `gemm_dense`, `gemv`, `gemvn`, `gemvq`, `moe_gemv`, `map_hits`, `rms_norm`, `rms_norm_q`, `argmax_rows`, `l2_norm`, `rope_neox`, `add`, `swiglu`, `ssm_gates`, `conv_silu`, `deltanet`, `conv_seq`, `attn_prefill`, `moe_combine_seq`, `norm_gate`, `kv_store`, `attn_decode`, `router_topk`, `moe_publish`, `moe_combine`. Keep their doc comments.
These stay in `impl Gpu`: `new`, `f`, `ptr`, `zeros`, `dequant`, `nsb`, `mmq`, `gemm_raw`, `xq_smem`, `float_ty`, `rpw`, `gemv_cfg`, `grid`, `quant_xq`, `attn_decode_gq`. Calls such as `Self::nsb(w)` and `self.attn_decode_gq(..)` inside the moved bodies still resolve to the inherent methods.

4e. At the top of `impl Backend for Gpu`, add the memory, graph and mailbox methods:

```rust
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

    // ... the 27 moved methods from 4d follow here ...
}
```

If `cu_stream()` or `bind_to_thread()` has a different name in cudarc 0.19.10, check `~/.cargo/registry/src/index.crates.io-*/cudarc-0.19.10/src/driver/safe/core.rs` (`CudaStream::cu_stream` at line 787, `CudaContext::bind_to_thread` at line 350).

4f. Make existing callers compile. The ops are now trait methods, so add `use crate::backend::Backend;` to `src/gpu_model.rs` and `src/gpu_model/prefill.rs`, and `use rusting_strata::backend::Backend;` to `tests/cuda.rs`, `tests/kv_q8.rs` and `tests/llama_numerics.rs`. `GpuModel` still holds a concrete `Gpu` in this task.

- [ ] **Step 5: Run the new tests**

Run: `cargo test --release --test backend`
Expected: 3 passed.

- [ ] **Step 6: Run the kernel tests**

Run: `cargo test --release --test cuda --test kv_q8 --test llama_numerics --test gpu_model`
Expected: all pass. The ops moved without changes, so a failure here means a body was edited by mistake.

- [ ] **Step 7: Commit**

```bash
git add src/backend.rs tests/backend.rs
git commit -m "feat: Backend trait with the CUDA implementation

Buf, GpuTensor and the shared tuning constants move to src/backend.rs.
Gpu implements Backend: memory copies, recorded graphs, the host mailbox
and the kernel ops the model uses, with unchanged signatures.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- src/backend.rs tests/backend.rs src/lib.rs src/cuda/mod.rs src/gpu_model.rs src/gpu_model/prefill.rs tests/cuda.rs tests/kv_q8.rs tests/llama_numerics.rs
```

---

### Task 2: CPU worker mailbox through `host_mailbox`

**Files:**
- Modify: `src/cpu_worker.rs`, `src/gpu_model.rs:163,173`

**Interfaces:**
- Consumes: `Backend::host_mailbox(&self, words) -> anyhow::Result<(Buf, *mut u32)>` (Task 1).
- Produces: `CpuWorker::new(m: &Model, gpu: &dyn Backend, moe_layers: Vec<usize>) -> anyhow::Result<CpuWorker>`. `pub dev: u64` stays (the mailbox device address).

- [ ] **Step 1: Change `CpuWorker::new`**

In `src/cpu_worker.rs`, replace the import `use cudarc::driver::{PinnedHostSlice, sys};` with `use crate::backend::{Backend, Buf};`. Change the field `_mem: PinnedHostSlice<u32>` to `_mem: Buf`. Replace the start of `new` (signature through the `ensure!` on `cuMemHostGetDevicePointer`) with:

```rust
    pub fn new(m: &Model, gpu: &dyn Backend, moe_layers: Vec<usize>) -> anyhow::Result<CpuWorker> {
        let c = &m.cfg;
        let layout = Mailbox::new(c.n_expert_used, c.n_embd);
        let len = layout.stride * moe_layers.iter().max().map_or(0, |&il| il + 1);
        let (mem, host) = gpu.host_mailbox(len)?;
        let (host, dev) = (host as usize, mem.addr);
```

The rest of `new` is unchanged; the final struct literal still reads `CpuWorker { _mem: mem, host, dev, .. }`. `Drop` joins the thread before the fields drop, so the mailbox outlives the worker thread as before.

- [ ] **Step 2: Update the two callers in `src/gpu_model.rs`**

`CpuWorker::new(m, &gpu.ctx, (0..c.n_layer).collect())` becomes `CpuWorker::new(m, &gpu, (0..c.n_layer).collect())`, and `CpuWorker::new(m, &gpu.ctx, vec![c.n_layer])` becomes `CpuWorker::new(m, &gpu, vec![c.n_layer])`. `&Gpu` coerces to `&dyn Backend`.

- [ ] **Step 3: Run the MoE GPU tests**

Run: `cargo test --release --test gpu_model`
Expected: 7 passed. The default model is MoE, so every test goes through the mailbox, and `mtp_spec_matches_greedy` also uses the draft head's own worker.

- [ ] **Step 4: Commit**

```bash
git commit -m "refactor: CPU worker mailbox from Backend::host_mailbox

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- src/cpu_worker.rs src/gpu_model.rs
```

---

### Task 3: Expert cache on `Buf`

**Files:**
- Modify: `src/expert_cache.rs`, `src/gpu_model.rs:494,499,541`, `src/gpu_model/prefill.rs:193`

**Interfaces:**
- Consumes: `Buf`, `dyn Backend` helpers `zeros`, `put`, and `Backend::htod` (Task 1).
- Produces: `ExpertCache::new(m, gpu: &dyn Backend, budget, n_layers)`, `adapt(&mut self, m, gpu: &dyn Backend, max_swaps)`, `base(&self, lx: &LayerExperts) -> (u64, usize)` (no `gpu` parameter), `LayerExperts::table_dev: Buf`.

- [ ] **Step 1: Port `src/expert_cache.rs`**

Imports: replace `use crate::cuda::Gpu;` and `use cudarc::driver::CudaSlice;` with `use crate::backend::{Backend, Buf};`.

Types: `pub table_dev: CudaSlice<i32>` becomes `pub table_dev: Buf`; `struct Pool { buf: CudaSlice<u8>, .. }` becomes `struct Pool { buf: Buf, .. }`.

Every `gpu: &Gpu` parameter (`new`, `place`, `sync_table`, `adapt`) becomes `gpu: &dyn Backend`.

Body changes:

```rust
// new(): pool allocation
pools.push(Pool { buf: gpu.zeros::<u8>(slots * slot_bytes)?, slot_bytes, owner: Vec::with_capacity(slots) });

// new(): per-layer table
table_dev: { let b = gpu.zeros::<i32>(n)?; gpu.put(&vec![-1i32; n], &b, 0)?; b },

// new(): capacity check (len is bytes, as CudaSlice<u8>::len() was)
if pool.owner.len() * pool.slot_bytes + pool.slot_bytes > pool.buf.len { continue; }

// place(): expert copy
gpu.htod(t.data, &pool.buf, o)?;

// sync_table()
if let Some(lx) = self.layers[il].as_ref() { gpu.put(&lx.table, &lx.table_dev, 0)?; }

// base()
pub fn base(&self, lx: &LayerExperts) -> (u64, usize) {
    let p = &self.pools[lx.pool];
    (p.buf.addr, p.slot_bytes)
}

// slot_bytes_total()
pub fn slot_bytes_total(&self) -> usize { self.pools.iter().map(|p| p.buf.len).sum() }
```

`zeros::<u8>` rounds a 0-byte pool up to 1 byte, as `gpu.zeros::<u8>((..).max(1))` did.

- [ ] **Step 2: Update callers**

`src/gpu_model.rs`:
- `let (base, stride) = cache.base(g, lx);` becomes `let (base, stride) = cache.base(lx);`
- `g.map_hits(ids, g.ptr(&lx.table_dev), ..)` becomes `g.map_hits(ids, lx.table_dev.addr, ..)`
- `ExpertCache::new(m, &gpu, ..)` and `cache.adapt(self.m, &self.gpu, 16)` compile unchanged (`&Gpu` coerces).

`src/gpu_model/prefill.rs`: `let (pool, stride) = cache.base(g, lx);` becomes `let (pool, stride) = cache.base(lx);`.

- [ ] **Step 3: Run the GPU model tests**

Run: `cargo test --release --test gpu_model`
Expected: 7 passed. `gpu_forward_matches_cpu_reference` runs with a full cache and with a 512 MB margin (part of the experts on the CPU), and `prefill_matches_reference` covers the staged experts.

- [ ] **Step 4: Commit**

```bash
git commit -m "refactor: expert cache on Buf and dyn Backend

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- src/expert_cache.rs src/gpu_model.rs src/gpu_model/prefill.rs
```

---

### Task 4: `GpuModel` on `Box<dyn Backend>`

**Files:**
- Modify: `src/gpu_model.rs`, `src/gpu_model/prefill.rs`, `src/main.rs:124,396`, `tests/gpu_model.rs` (7 `GpuModel::load` calls)

**Interfaces:**
- Consumes: everything from Tasks 1-3.
- Produces: `GpuModel::load(m: &'a Model<'a>, gpu: Box<dyn Backend>, ctx: usize, chunk: usize, vram_margin: usize, k: usize)`, `pub gpu: Box<dyn Backend>`. Part 3 constructs a Vulkan backend and passes it here.

- [ ] **Step 1: Switch the callers first (the build fails until Step 4)**

```bash
sed -i 's/GpuModel::load(&m, gpu,/GpuModel::load(\&m, Box::new(gpu),/' tests/gpu_model.rs
sed -i 's/GpuModel::load(m, Gpu::new()?,/GpuModel::load(m, Box::new(Gpu::new()?),/; s/GpuModel::load(&m, Gpu::new()?,/GpuModel::load(\&m, Box::new(Gpu::new()?),/' src/main.rs
grep -n "GpuModel::load" tests/gpu_model.rs src/main.rs
```

Expected: 7 lines in `tests/gpu_model.rs` and 2 in `src/main.rs`, all with `Box::new(`.

- [ ] **Step 2: Run the build to see it fail**

Run: `cargo build --release --tests`
Expected: `mismatched types: expected Gpu, found Box<_>` at the `GpuModel::load` calls.

- [ ] **Step 3: Port `src/gpu_model.rs` declarations and `load`**

Imports: delete `use crate::cuda::{Gpu, GpuTensor};` and `use cudarc::driver::{CudaGraph, CudaSlice, sys};`. Add:

```rust
use crate::backend::{ATT_SPLITS, Backend, Buf, GpuTensor};
use std::hash::{Hash, Hasher};
```

Types:
- `Mixer::Delta`: `conv: Buf, s: Buf`, `snap: Option<(Buf, Buf)>`.
- `Mixer::Attn`: `kc: Buf, vc: Buf`.
- `MtpHead`: `step_buf: Buf, emb: Buf, eh: Buf, x: Buf`.
- `GpuModel`: `pub gpu: Box<dyn Backend>`, `bufs: HashMap<&'static str, Buf>`, `x: Buf`, `logits: Buf`, `amax: Buf`, `ids: Buf`, `slots: Buf`, `step_buf: Buf`, `saved: Vec<(Buf, Buf)>`. Delete the `graphs` and `mtp_graphs` fields and their doc comments, and remove `graphs: (0..=TMAX).map(|_| None).collect(), mtp_graphs: Default::default(), ` from the struct literal at the end of `load`.

`load`:
- Signature: `gpu: Box<dyn Backend>`.
- `gpu.ctx.mem_get_info()?` (two places) becomes `gpu.mem_info()?`.
- Every `gpu.zeros(` without a turbofish becomes `gpu.zeros::<f32>(`: the `conv`, `s`, both `snap` buffers, the `MtpHead` `emb`, `eh`, `x`, and `x`, `logits` in the `(x, logits, ids)` line. The KV caches become `gpu.zeros::<u16>(gpu.kv_words(ctx * kv))?`. Existing `::<i32>` and `::<f32>` calls stay.
- `crate::cuda::ATT_SPLITS` becomes `ATT_SPLITS`.
- `prefill::Scratch::new(m, &gpu, ..)`, `ExpertCache::new(m, &gpu, ..)` and both `CpuWorker::new(m, &gpu, ..)` become `&*gpu`.

- [ ] **Step 4: Port the `GpuModel` methods**

Mechanical rules for the whole file (they cover every remaining use):
- `g.ptr(&X)`, `self.gpu.ptr(&X)` and `gpu.ptr(&X)` become `X.addr`. Examples: `g.ptr(&self.bufs[n])` becomes `self.bufs[n].addr`, `g.ptr(conv)` becomes `conv.addr`, `self.gpu.ptr(&self.mtp.as_ref().unwrap().x)` becomes `self.mtp.as_ref().unwrap().x.addr`.
- `stream.clone_dtoh(&B.slice(0..n))?` becomes `get::<T>(&B, 0, n)?` with `T` the old element type (`f32` for `logits` and `x`, `i32` for `amax`).
- `X.len()` on a former `CudaSlice` becomes `X.len` (bytes, not elements; see `rollback` below).

Methods whose bodies change beyond those rules, in full:

```rust
    pub fn reset(&mut self) -> anyhow::Result<()> {
        self.saved_ids.clear();
        if let Some(h) = self.mtp.as_mut() { h.pos = 0; h.pend.clear(); }
        for l in self.layers.iter().chain(self.mtp.as_ref().map(|h| &h.layer)) {
            match &l.mixer {
                Mixer::Delta { conv, s, .. } => { self.gpu.memset(conv)?; self.gpu.memset(s)?; }
                Mixer::Attn { kc, vc, .. } => { self.gpu.memset(kc)?; self.gpu.memset(vc)?; }
            }
        }
        self.pos = 0;
        Ok(())
    }

    pub fn step(&mut self, token: u32) -> anyhow::Result<Vec<f32>> {
        if self.pos >= self.ctx { bail!("context full ({} tokens; raise --ctx)", self.ctx); }
        let seq = self.worker.as_mut().map_or(0, |w| w.begin());
        let g = &self.gpu;
        g.put(&self.m.embed(token), &self.x, 0)?;
        g.put(&[self.pos as i32, seq as i32], &self.step_buf, 0)?;
        self.run(1)?;
        let logits = self.gpu.get::<f32>(&self.logits, 0, self.m.cfg.n_vocab)?;
        self.after_forward(1)?;
        Ok(logits)
    }
```

`verify`: `let logits = self.gpu.get::<f32>(&self.logits, 0, tokens.len() * nvoc)?;`

`verify_greedy`:

```rust
        g.argmax_rows(self.logits.addr, nvoc, nt, self.amax.addr)?;
        let am: Vec<u32> = g.get::<i32>(&self.amax, 0, nt)?.into_iter().map(|x| x as u32).collect();
```

`verify_run`: `g.put(&emb, &self.x, 0)?;` and `g.put(&steps, &self.step_buf, 0)?;`

```rust
    pub fn save_state(&mut self) -> anyhow::Result<()> {
        let g = &self.gpu;
        if self.saved.is_empty() {
            for l in &self.layers {
                if let Mixer::Delta { conv, s, .. } = &l.mixer { self.saved.push((g.alloc(conv.len)?, g.alloc(s.len)?)); }
            }
        }
        let mut it = self.saved.iter();
        for l in &self.layers {
            if let Mixer::Delta { conv, s, .. } = &l.mixer {
                let (bc, bs) = it.next().unwrap();
                g.dtod(conv, 0, bc, 0, conv.len)?;
                g.dtod(s, 0, bs, 0, s.len)?;
            }
        }
        self.saved_pos = self.pos;
        self.saved_ids.clear();
        Ok(())
    }

    pub fn rollback(&mut self, start: usize, keep: usize) -> anyhow::Result<()> {
        if !(1..=self.k).contains(&keep) || start + keep > self.pos { bail!("rollback: keep {keep} of verify at {start}"); }
        let g = &self.gpu;
        for l in &self.layers {
            if let Mixer::Delta { conv, s, snap: Some((sc, ss)), .. } = &l.mixer {
                // Snapshot `keep - 1` of each state; sizes in bytes.
                let (nc, ns) = (conv.len, s.len);
                g.dtod(sc, (keep - 1) * nc, conv, 0, nc)?;
                g.dtod(ss, (keep - 1) * ns, s, 0, ns)?;
            }
        }
        self.pos = start + keep;
        Ok(())
    }

    pub fn restore_state(&mut self) -> anyhow::Result<()> {
        let g = &self.gpu;
        let mut it = self.saved.iter();
        for l in &self.layers {
            if let Mixer::Delta { conv, s, .. } = &l.mixer {
                let (bc, bs) = it.next().ok_or_else(|| anyhow::anyhow!("restore_state without save_state"))?;
                g.dtod(bc, 0, conv, 0, conv.len)?;
                g.dtod(bs, 0, s, 0, s.len)?;
            }
        }
        self.pos = self.saved_pos;
        Ok(())
    }

    /// Run the forward of `nt` tokens as a recorded graph (graph key `nt`), recorded on first use.
    fn run(&self, nt: usize) -> anyhow::Result<()> { self.gpu.graph(nt as u64, &mut || self.forward(nt)) }
```

`hidden`: `Ok(self.gpu.get::<f32>(&self.x, 0, self.m.cfg.n_embd)?)`. `mtp_draft`: `Ok(self.gpu.get::<f32>(&self.logits, 0, self.m.cfg.n_vocab)?)`.

`mtp_run`, from `let seq = h.worker.begin();` to the line before `let h = self.mtp.as_mut().unwrap();`:

```rust
        let seq = h.worker.begin();
        let g = &self.gpu;
        let emb: Vec<f32> = tokens.iter().flat_map(|&t| self.m.embed(t)).collect();
        g.put(&emb, &h.emb, 0)?;
        let steps: Vec<i32> = (0..nt).flat_map(|t| [(h.pos + t) as i32, seq as i32]).collect();
        g.put(&steps, &h.step_buf, 0)?;
        // `run` keys are token counts (below 8); the top bit keeps these keys apart from them.
        let key = { let mut s = std::hash::DefaultHasher::new(); (hid, nt, head).hash(&mut s); s.finish() | 1 << 63 };
        g.graph(key, &mut || self.mtp_forward(hid, nt, head))?;
        let d = if head { g.get::<i32>(&self.amax, 0, 1)?[0] as u32 } else { g.sync()?; 0 };
```

`after_forward`: `cache.adapt(self.m, &*self.gpu, 16)?`.

`forward`, `mtp_forward`, `layer`, `spec_step` and `row_ptr` change only by the `.addr` rule.

- [ ] **Step 5: Port `src/gpu_model/prefill.rs`**

Imports: replace `use crate::cuda::{GEMM_BM, Gpu};` and `use cudarc::driver::CudaSlice;` with `use crate::backend::{Backend, Buf, GEMM_BM};`.

`Scratch`: fields `slab: Buf, ints: Buf, tiles: Buf, staging: Buf`. In `new`, change the parameter to `gpu: &dyn Backend` and the allocations to:

```rust
            slab: gpu.zeros::<f32>(total)?, ints: gpu.zeros::<i32>(3 * chunk * k)?, tiles: gpu.zeros::<u64>(2 * n_tiles)?,
            staging: gpu.zeros::<u8>(if c.is_moe() { STAGE * stage_bytes } else { 0 })?,
```

```rust
    /// Bytes of VRAM this scratch takes.
    pub fn bytes(&self) -> usize { self.slab.len + self.ints.len + self.tiles.len + self.staging.len }
```

`prefill`, after the `for ch in tokens.chunks(chunk)` loop:

```rust
        let (g, ne) = (&self.gpu, self.m.cfg.n_embd);
        let (pf, r) = (self.pf.as_ref().unwrap(), (tokens.len() - 1) % chunk);
        let h = self.bufs["h"].addr;
        // Row 0 of `x` holds the last hidden state, as after `step` (the MTP draft reads it).
        g.dtod(&pf.slab, 4 * ne * r, &self.x, 0, 4 * ne)?;
        g.rms_norm(pf.slab.addr + (4 * ne * r) as u64, 0, self.output_norm.ptr, h, 0, ne, 1, self.m.cfg.eps)?;
        g.gemv(&self.output, h, self.logits.addr)?;
        g.get::<f32>(&self.logits, 0, self.m.cfg.n_vocab)
```

`prefill_chunk`:
- `let pf = self.pf.as_mut().unwrap();` becomes `let pf = self.pf.as_ref().unwrap();`
- `g.stream.memcpy_htod(&emb, &mut pf.slab.slice_mut(0..t * ne))?;` becomes `g.put(&emb, &pf.slab, 0)?;`
- `g.stream.memcpy_htod(&[self.pos as i32, 0], &mut self.step_buf)?;` becomes `g.put(&[self.pos as i32, 0], &self.step_buf, 0)?;`
- `let ids = g.stream.clone_dtoh(&pf.ints.slice(0..t * k))?;` becomes `let ids = g.get::<i32>(&pf.ints, 0, t * k)?;`
- `g.stream.memcpy_htod(&xrow, &mut pf.ints.slice_mut(n * k..n * k + t * k))?;` becomes `g.put(&xrow, &pf.ints, n * k)?;`
- `g.stream.memcpy_htod(&inv, &mut pf.ints.slice_mut(2 * n * k..2 * n * k + t * k))?;` becomes `g.put(&inv, &pf.ints, 2 * n * k)?;`
- `g.stream.memcpy_htod(&tiles, &mut pf.tiles.slice_mut(0..tiles.len()))?;` becomes `g.put(&tiles, &pf.tiles, 0)?;`
- `anyhow::ensure!(tiles.len() <= pf.tiles.len(), ..)` becomes `anyhow::ensure!(8 * tiles.len() <= pf.tiles.len, ..)` (bytes).
- `g.stream.memcpy_htod(src.data, &mut pf.staging.slice_mut(o..o + mat.bytes))?;` becomes `g.htod(src.data, &pf.staging, o)?;`
- The `.addr` rule for the rest (`step`, `base`, `conv`, `s`, `kc`, `vc`, `ids_d`, `sp`, `tp`, `xrow_d`, the `inv` address in `moe_combine_seq`).

- [ ] **Step 6: Build and check the cudarc gate**

Run: `cargo build --release --tests && grep -rn cudarc src --include=*.rs | grep -v '^src/cuda/'`
Expected: the build succeeds with no new warnings, and grep prints nothing.

- [ ] **Step 7: Run the whole test suite**

Run: `cargo test --release`
Expected: all tests pass, including the 3 in `tests/backend.rs`.

- [ ] **Step 8: Commit**

```bash
git commit -m "refactor: GpuModel on Box<dyn Backend>

Model state is Buf, the forward and MTP graphs go through Backend::graph,
and cudarc is now used only under src/cuda/.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- src/gpu_model.rs src/gpu_model/prefill.rs src/main.rs tests/gpu_model.rs
```

---

### Task 5: Gate run and log entry

**Files:**
- Modify: `docs/LOOP_LOG.md` (new entry #85 after #84, before `## Next candidates`)

**Interfaces:**
- Consumes: the release binary from Task 4.

- [ ] **Step 1: Build a v1.0.0 binary next to the new one**

```bash
git worktree add /tmp/strata-v1 v1.0.0
cargo build --release --manifest-path /tmp/strata-v1/Cargo.toml
cargo build --release
M=/mnt/ollama/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf
```

- [ ] **Step 2: Greedy output is byte-identical**

```bash
# Generated text goes to stdout, timings to stderr. Greedy is the default (no --temp).
for b in v1:/tmp/strata-v1/target/release/strata head:target/release/strata; do
  bin=${b#*:} out=/tmp/greedy-${b%%:*}.txt
  $bin generate --model $M --gpu --prompt "The capital of France is" -n 64 2>/dev/null > $out
  $bin generate --model $M --gpu --mtp 1 --prompt "The capital of France is" -n 64 2>/dev/null >> $out
done
cmp /tmp/greedy-v1.txt /tmp/greedy-head.txt && echo IDENTICAL
```

Expected: `IDENTICAL`. A difference means an op or copy changed; find it with `tests/gpu_model.rs` before going on.

- [ ] **Step 3: Interleaved speed A/B**

Check `nvidia-smi` and `uptime` first: no other GPU process over 1 GB, load under 5.

```bash
for r in 1 2 3; do
  for b in /tmp/strata-v1/target/release/strata target/release/strata; do
    echo "== $b round $r"; $b bench --model $M --gpu --pp 512,4096 --tg 128 --reps 1 2>&1 | tail -4
  done
done
```

Expected: per metric, the new binary's mean is within the v1.0.0 binary's min-max range, or within 1% of its mean. Also inside the README ranges if machine load allows (prefill 4K 2089-2096, decode after 4K 95.4-95.8, short decode 100.6-101.5 tok/s). If the new binary is slower beyond noise, profile `step` (the `Box<dyn Backend>` call happens once per graph replay, so a slowdown points at a changed copy, such as an extra sync).

- [ ] **Step 4: Parity**

Run: `python -I tools/parity.py $M --extra "--gpu" --tag seam`
Expected: either-mode count 631 or more of 640 (v1.0.0 default `--gpu`: 631-634).

- [ ] **Step 5: Clean up and write the log entry**

```bash
git worktree remove /tmp/strata-v1
```

Append to `docs/LOOP_LOG.md` after entry #84, with the measured values in place of the brackets:

```markdown
### 2026-10-15 #85 — backend seam (Part 1 of the Vulkan and architecture-table spec)

`src/backend.rs` puts a `Backend` trait between the model and CUDA: `Buf` (device address plus owner) replaces
`CudaSlice` in `gpu_model.rs`, `gpu_model/prefill.rs`, `expert_cache.rs` and `cpu_worker.rs`; the forward and MTP
graphs go through `Backend::graph`; the CPU mailbox comes from `Backend::host_mailbox`. cudarc is used only under
`src/cuda/`.
- Greedy output (64 tokens, MTP 0 and 1) byte-identical to v1.0.0.
- Interleaved A/B against v1.0.0, 3 rounds, load [load]: 4K prefill [new] vs [v1], decode after 512 [new] vs [v1],
  after 4K [new] vs [v1] tok/s.
- Parity `--gpu`: [n]/640 either mode (strict [m]).
- `cargo test --release`: [count] pass.
```

- [ ] **Step 6: Commit**

```bash
git commit -m "docs: loop log #85, backend seam gate

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- docs/LOOP_LOG.md
```
