//! Backend seam: device memory as `Buf` (an address plus whatever keeps the memory alive) and the kernel ops the
//! model runs. Every op takes device addresses, so model code never sees backend types. `cuda` and `vulkan` implement it.

use crate::quant::GgmlType;
use crate::weights::Tensor;
use std::any::Any;

/// Blocks per head in decode attention (positions split across them). Scratch sizes depend on it.
pub const ATT_SPLITS: usize = 32;
/// Splits of the one-head-per-block decode attention used below ATT_GQA_MIN positions.
pub const ATT_SPLITS_1: usize = 16;
/// Positions from which decode attention shares K/V loads across a GQA group (break-even measured near 2K).
pub const ATT_GQA_MIN: i32 = 2048;
const _: () = assert!(ATT_SPLITS <= 256); // attn_merge keeps one weight per split in a 256-thread block
/// Positions per GEMM tile. Prefill cuts its tile lists at this size.
pub const GEMM_BM: usize = 128;

/// `len` bytes of device memory at device address `addr`, freed when the last clone drops.
#[derive(Clone)]
pub struct Buf {
    addr: u64,
    len: usize,
    _owner: std::sync::Arc<dyn Any + Send + Sync>,
}

impl Buf {
    pub fn new(addr: u64, len: usize, owner: impl Any + Send + Sync) -> Buf { Buf { addr, len, _owner: std::sync::Arc::new(owner) } }
    pub fn addr(&self) -> u64 { self.addr }
    pub fn len(&self) -> usize { self.len }
}

#[derive(Clone)]
pub struct GpuTensor {
    pub ty: GgmlType,
    pub rows: usize,
    pub cols: usize,
    pub ptr: u64,
    pub scale: f32,
    _buf: Buf,
}

impl GpuTensor {
    /// `rows` x `cols` of `ty` stored in `buf` from its start.
    pub fn new(ty: GgmlType, rows: usize, cols: usize, buf: Buf) -> GpuTensor { GpuTensor { ty, rows, cols, ptr: buf.addr, scale: 1.0, _buf: buf } }
    pub fn with_scale(mut self, scale: f32) -> GpuTensor { self.scale = scale; self }
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
/// Pointer arguments are device addresses; strides are in floats unless named otherwise.
/// The backend `kind` ("cuda" or "vulkan") on GPU `device`; with no kind, CUDA if it loads, else Vulkan.
pub fn open(kind: Option<&str>, device: usize) -> anyhow::Result<Box<dyn Backend>> {
    match kind {
        Some("cuda") => Ok(Box::new(crate::cuda::Gpu::new(device)?)),
        Some("vulkan") => Ok(Box::new(crate::vulkan::Vk::new(device)?)),
        Some(k) => anyhow::bail!("unknown backend {k:?} (cuda or vulkan)"),
        None => match crate::cuda::Gpu::new(device) {
            Ok(g) => Ok(Box::new(g)),
            Err(c) if format!("{c}") == "CUDA init" => {
                let v = crate::vulkan::Vk::new(device).map_err(|v| anyhow::anyhow!("no GPU backend loads: CUDA: {c:#}; Vulkan: {v:#}"))?;
                eprintln!("CUDA unavailable ({c:#}), using Vulkan");
                Ok(Box::new(v))
            }
            Err(c) => Err(c),
        },
    }
}

/// Which usable Vulkan device each GPU index opens, so that an index is the same card on both backends. `cuda` is the
/// identifier of every CUDA device in CUDA's order (`None`: the driver gives none), `vk` the one of every usable
/// Vulkan device in the order Vulkan alone would count them. Index i is the position in `vk` of CUDA device i's
/// identifier, `None` when no usable Vulkan device has it. A CUDA device without identifier takes the next Vulkan
/// device that no CUDA device names, and the rest of those follow the CUDA devices: with no CUDA device, or none
/// that Vulkan knows, the order is `vk`'s own.
pub fn device_order<T: PartialEq>(cuda: &[Option<T>], vk: &[T]) -> Vec<Option<usize>> {
    let mut rest = (0..vk.len()).filter(|&i| !cuda.iter().flatten().any(|c| *c == vk[i]));
    let mut order: Vec<_> = cuda.iter().map(|c| match c { Some(c) => vk.iter().position(|v| v == c), None => rest.next() }).collect();
    order.extend(rest.map(Some));
    order
}

/// `gemm` `act` flag: x is unchanged since the previous gemm call, which read the same x, so its int8 copy is reused.
pub const SAME_X: i32 = 4;

/// The weight type `Backend::upload` stores a tensor of type `ty` as: F16 as F32 (no F16 kernels; F32 holds every F16
/// value exactly). Expert tensors are copied raw by `GpuModel::load_part`, not through `upload`.
pub fn upload_type(ty: GgmlType) -> GgmlType { if ty == GgmlType::F16 { GgmlType::F32 } else { ty } }

pub trait Backend: Send {
    /// "cuda" or "vulkan".
    fn name(&self) -> &'static str;
    /// The GPU's name as the driver reports it.
    fn device(&self) -> String;
    /// Create a lightweight handle sharing the same underlying GPU device/context for an auxiliary draft model.
    fn share(&self) -> anyhow::Result<Box<dyn Backend>> {
        anyhow::bail!("backend {} does not support device sharing", self.name())
    }
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
    /// Free the graph recorded under `key`, if there is one. For a caller whose keys change as it runs (a graph holds
    /// device memory, and CUDA stops recording at a fixed number of them).
    fn graph_drop(&self, _key: u64) -> anyhow::Result<()> { Ok(()) }
    /// Queue the ops of `f`, none of which reads what another writes, with no ordering among them (Vulkan: no
    /// barriers between them); ops after the batch see all their results.
    fn batch(&self, f: &mut dyn FnMut() -> anyhow::Result<()>) -> anyhow::Result<()> { f() }
    /// Queue the ops of `f` in order, its host copies beside its other ops (CUDA: on a side stream), so none of
    /// those ops may read what a copy writes. The copies see the ops before `f`. Ops after `f` see the copies of the
    /// `beside` before this one, not yet these: they run on beside the next `beside` too (`beside_join` waits).
    fn beside(&self, f: &mut dyn FnMut() -> anyhow::Result<()>) -> anyhow::Result<()> { f() }
    /// Ops after this see the copies of every `beside` so far.
    fn beside_join(&self) -> anyhow::Result<()> { Ok(()) }
    /// Submit the ops queued so far without waiting. Long prefills call it per layer: amdgpu resets the GPU when one
    /// submission runs over its timeout (10 s).
    fn kick(&self) -> anyhow::Result<()> { Ok(()) }
    /// Whether `beside` really overlaps its copies with its other ops (else it queues them in order).
    fn beside_overlaps(&self) -> bool { false }
    /// Whether a MoE combine gave up waiting for the CPU worker since the last call (checked after a wait). Its layer
    /// added nothing and the rest of that submission was skipped: the caller waits for the worker and runs the
    /// forward again from that combine. Clears the flag. Backends whose combines never give up return false.
    fn stalled(&self) -> bool { false }
    /// Whether the prefill GEMM for `ty` runs on int8 tensor cores on this backend: the type policy of `Gpu::mmq`,
    /// gated on the device. Tests build their reference math from this, since the CUDA launcher takes the f16
    /// `gemm_*` path where the arch has no int8 tensor cores (sm_70) even for an int8 type.
    fn mmq(&self, ty: GgmlType) -> bool { crate::cuda::Gpu::mmq(ty) }
    /// Whether this backend runs the Volta (sm_70) CUDA fallbacks, whose kernels sum the same f16 products in
    /// another order than `mma.sync`. A cross-backend comparison against or from such a backend is by tolerance.
    fn volta(&self) -> bool { false }
    /// Zeroed host memory of `words` u32 that kernels read and write at the `Buf` address. The host pointer is
    /// valid while the `Buf` lives.
    fn host_mailbox(&self, words: usize) -> anyhow::Result<(Buf, *mut u32)>;

    /// Refuse, before any weight is uploaded, if a pipeline these weight types (`GpuModel`'s upload types) need would
    /// exceed the device's shared-memory limit; `verbose` also lists each checked (kernel, specialization) pair.
    /// Vulkan computes it from the committed SPIR-V and the specialization constants; backends without a Workgroup
    /// shared-memory limit have nothing to check.
    fn preflight(&self, _types: &[GgmlType], _verbose: bool) -> anyhow::Result<()> { Ok(()) }

    fn upload(&self, t: &Tensor) -> anyhow::Result<GpuTensor> {
        let gt = match upload_type(t.ty) {
            GgmlType::F32 if t.ty == GgmlType::F16 => {
                let mut f = vec![0f32; t.rows * t.cols];
                crate::quant::dequant_row(GgmlType::F16, t.data, &mut f);
                let b = self.alloc(4 * f.len())?;
                self.htod(&f.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(), &b, 0)?;
                GpuTensor::new(GgmlType::F32, t.rows, t.cols, b)
            }
            ty => {
                let b = self.alloc(t.data.len())?;
                self.htod(t.data, &b, 0)?;
                GpuTensor::new(ty, t.rows, t.cols, b)
            }
        };
        Ok(gt.with_scale(t.scale))
    }

    fn upload_raw(&self, t: &Tensor) -> anyhow::Result<GpuTensor> {
        let b = self.alloc(t.data.len())?;
        self.htod(t.data, &b, 0)?;
        Ok(GpuTensor::new(t.ty, t.rows, t.cols, b).with_scale(t.scale))
    }

    /// u16 words of a KV cache for `n` values: f16, or q8 with one f16 scale per 32 values.
    fn kv_words(&self, n: usize) -> usize;
    /// GEMM over a tile list: for each tile, positions m0 .. m0 + cnt get y[m, 0..rows] = W_tile x[xrow[m]] (x row m
    /// when `xrow` is 0). `tiles` is a device array of [w, m0 | cnt << 32]. `act`: 0 stores, 1 computes
    /// y = silu(y) * result (SwiGLU with the gate already in y), 2 adds to y; `| SAME_X` as below. `n_x`: rows of x
    /// the tiles read.
    fn gemm(&self, ty: GgmlType, rows: usize, cols: usize, tiles: u64, n_tiles: usize, x: u64, x_stride: usize, n_x: usize, xrow: u64, y: u64, y_stride: usize, act: i32) -> anyhow::Result<()>;
    /// Whether `gemm` and `gemm_dense` run on int8 tensor cores for `ty` (the alternative is f16 products), so tests
    /// and callers build their reference the way this backend will.
    fn int8_gemm(&self, ty: GgmlType) -> bool;
    /// y[m] (op) W x[m] for positions m < n, x and y rows `x_stride` / `y_stride` floats apart.
    fn gemm_dense(&self, w: &GpuTensor, n: usize, x: u64, x_stride: usize, y: u64, y_stride: usize, act: i32) -> anyhow::Result<()>;
    fn prequant_gemm_x(&self, _w: &GpuTensor, _n: usize, _x: u64, _x_stride: usize) -> anyhow::Result<()> { Ok(()) }
    /// y = W x (all rows of `w`).
    fn gemv(&self, w: &GpuTensor, x: u64, y: u64) -> anyhow::Result<()>;
    /// y[t] = W x[t] for t < nt (1 to 4; x[t] at x + 4 * t * x_stride bytes, y[t] at y + 4 * t * y_stride); each
    /// weight block is read once for all nt.
    fn gemvn(&self, w: &GpuTensor, x: u64, x_stride: usize, y: u64, y_stride: usize, nt: usize) -> anyhow::Result<()>;
    /// `gemvn` that reads x pre-quantized by `quant_xq` at `xq` (0 = quantize x itself; float weights read x).
    fn gemvq(&self, w: &GpuTensor, x: u64, x_stride: usize, xq: u64, y: u64, y_stride: usize, nt: usize) -> anyhow::Result<()>;
    /// `gemvq` of `a` into `ya` and of `b` into `yb` on the same x (columns alike; output rows `a.rows` / `b.rows` apart).
    fn gemvq2(&self, a: &GpuTensor, b: &GpuTensor, x: u64, x_stride: usize, xq: u64, ya: u64, yb: u64, nt: usize) -> anyhow::Result<()> {
        self.gemvq(a, x, x_stride, xq, ya, a.rows, nt)?;
        self.gemvq(b, x, x_stride, xq, yb, b.rows, nt)
    }
    /// Whether this backend supports indexed-row GEMV (`gemv_rows`).
    fn supports_gemv_rows(&self) -> bool { false }
    /// `y[i] = W[ids[i]] x` for `i < n_rows` (indexed subset of rows of `w`).
    fn gemv_rows(&self, _w: &GpuTensor, _x: u64, _ids: u64, _n_rows: usize, _y: u64) -> anyhow::Result<()> {
        anyhow::bail!("gemv_rows unsupported on this backend")
    }
    /// `y[t * n_rows + i] = W[ids[i]] x[t]` for `t < nt`, `i < n_rows`.
    fn gemv_rows_n(&self, w: &GpuTensor, x: u64, x_stride: usize, _xq: u64, ids: u64, n_rows: usize, y: u64, nt: usize) -> anyhow::Result<()> {
        for t in 0..nt {
            self.gemv_rows(w, x + (t * x_stride * 4) as u64, ids, n_rows, y + (t * n_rows * 4) as u64)?;
        }
        Ok(())
    }
    /// For j < k with ids[j] >= 0: y[j] = W_j x[j / xg], where W_j is the `rows` x `cols` matrix at w + ids[j] * stride
    /// bytes. `pair` = (w2, y2) also computes the same product for the matrices at byte offset w2 into the output at
    /// byte offset y2. `xq` (not 0): x pre-quantized by `quant_xq` (k / xg rows), read in place of x.
    fn moe_gemv(&self, ty: GgmlType, rows: usize, cols: usize, w: u64, stride: usize, ids: u64, k: usize, x: u64, x_stride: usize, xg: usize, y: u64, pair: Option<(i64, i64)>, xq: u64) -> anyhow::Result<()>;
    /// out = the first `rows` rows of `w` as floats (a test helper).
    fn dequant(&self, w: &GpuTensor, rows: usize, out: u64) -> anyhow::Result<()>;
    /// Quantize `nt` rows of n floats (row t at x + 4 * t * x_stride bytes) for `gemvq` / `moe_gemv` into `out`:
    /// the nt rows of n int8 values, then their f32 scales (one per 32 values), nt * (n + n / 8) bytes in all.
    fn quant_xq(&self, x: u64, x_stride: usize, n: usize, nt: usize, out: u64) -> anyhow::Result<()>;
    /// `rows` rows: out[r] = rmsnorm(x[r]) * w.
    fn rms_norm(&self, x: u64, x_stride: usize, w: u64, out: u64, out_stride: usize, n: usize, rows: usize, eps: f32) -> anyhow::Result<()>;
    /// `rms_norm` that also writes its output quantized as `quant_xq` does to `xq` (when not 0), and first adds `res`
    /// (rows `x_stride` apart, when not 0) into x in place.
    fn rms_norm_q(&self, x: u64, x_stride: usize, w: u64, out: u64, out_stride: usize, n: usize, rows: usize, eps: f32, xq: u64, res: u64) -> anyhow::Result<()>;
    /// Per-stream RMSNorm of nt stacks of hc rows of n (contiguous) with the hc x n gamma `w`.
    fn rms_norm_streams(&self, x: u64, w: u64, out: u64, n: usize, hc: usize, nt: usize, eps: f32) -> anyhow::Result<()> {
        for s in 0..hc {
            let o = 4 * (s * n) as u64;
            self.rms_norm(x + o, hc * n, w + o, out + o, hc * n, n, nt, eps)?;
        }
        Ok(())
    }
    /// out[r] = argmax of row r of x (`rows` rows of n floats), first index on ties.
    fn argmax_rows(&self, x: u64, n: usize, rows: usize, out: u64) -> anyhow::Result<()>;
    /// Row r: vector r % per (n floats) of token r / per, tokens `stride` floats apart.
    fn l2_norm(&self, x: u64, n: usize, per: usize, stride: usize, rows: usize, eps: f32) -> anyhow::Result<()>;
    /// `heads` rows of `stride` floats, `per` per token; token t is at position step[0] + t.
    fn rope_neox(&self, x: u64, stride: usize, heads: usize, per: usize, n_rot: usize, step: u64, base: f32) -> anyhow::Result<()> {
        self.rope_pairs(x, stride, heads, per, n_rot, step, base, 0, n_rot / 2)
    }
    /// `rope_neox` of pairs `lo..hi` only (M-RoPE: one call per section, each with its own position buffer).
    fn rope_pairs(&self, x: u64, stride: usize, heads: usize, per: usize, n_rot: usize, step: u64, base: f32, lo: usize, hi: usize) -> anyhow::Result<()>;
    /// M-RoPE (`ops::rope_mrope`): section s of the pairs turns by position `pos[2 * s]` (`pos`: four [position, _]
    /// i32 pairs); pairs past the sections use the last position.
    fn rope_mrope(&self, x: u64, stride: usize, heads: usize, per: usize, n_rot: usize, pos: u64, base: f32, sections: [usize; 4]) -> anyhow::Result<()> {
        let (half, mut lo) = (n_rot / 2, 0);
        for (s, n) in sections.into_iter().enumerate() {
            let hi = if s == 3 { half } else { (lo + n).min(half) };
            if hi > lo { self.rope_pairs(x, stride, heads, per, n_rot, pos + 8 * s as u64, base, lo, hi)?; }
            lo = hi;
        }
        Ok(())
    }
    /// `rms_norm` of `rows` heads of n (output rows of n), then `rope_neox` on them.
    fn rms_norm_rope(&self, x: u64, x_stride: usize, w: u64, out: u64, n: usize, rows: usize, eps: f32, per: usize, n_rot: usize, step: u64, base: f32) -> anyhow::Result<()> {
        self.rms_norm(x, x_stride, w, out, n, n, rows, eps)?;
        self.rope_neox(out, n, rows, per, n_rot, step, base)
    }
    /// `rms_norm_rope` of the k heads (`per` per token), then `kv_store` of them and v (rows of n, like out).
    fn rms_norm_rope_kv(&self, x: u64, x_stride: usize, w: u64, out: u64, n: usize, rows: usize, eps: f32, per: usize, n_rot: usize, step: u64, base: f32, v: u64, kc: u64, vc: u64) -> anyhow::Result<()>;
    /// `rms_norm_rope` of the q heads (`qrows` rows, `qper` per token), then `rms_norm_rope_kv` of the k heads (rows of n).
    fn rms_norm_rope_qk(&self, q: u64, q_stride: usize, qw: u64, qo: u64, qrows: usize, qper: usize, k: u64, kw: u64, ko: u64, krows: usize, kper: usize,
                        n: usize, eps: f32, n_rot: usize, step: u64, base: f32, v: u64, kc: u64, vc: u64) -> anyhow::Result<()> {
        self.rms_norm_rope(q, q_stride, qw, qo, n, qrows, eps, qper, n_rot, step, base)?;
        self.rms_norm_rope_kv(k, n, kw, ko, n, krows, eps, kper, n_rot, step, base, v, kc, vc)
    }
    /// x += y.
    fn add(&self, x: u64, y: u64, n: usize) -> anyhow::Result<()>;
    /// g = silu(g) * u.
    fn swiglu(&self, g: u64, u: u64, n: usize) -> anyhow::Result<()>;
    /// Beta*tanh(gate/beta) * sigmoid(gate) * lb*tanh(up/lb) as in `situ`, directly quantized as `quant_xq` into `xq`.
    fn situ_q(&self, _y: u64, _n: usize, _rows: usize, _scale: u64, _beta: f32, _lb: f32, _xq: u64) -> anyhow::Result<()> { anyhow::bail!("{}: situ_q not implemented", self.name()) }
    /// `x[i] = cap * tanh(x[i] / cap)` for `i < n`.
    fn softcap(&self, _x: u64, _n: usize, _cap: f32) -> anyhow::Result<()> { anyhow::bail!("{}: softcap not implemented", self.name()) }
    /// Inkling's activation on `rows` rows of 2n floats at `y`, n gates then n ups: each gate becomes
    /// beta*tanh(gate/beta) * sigmoid(gate) * lb*tanh(up/lb) (`lb` not above 0: times the up as it is).
    fn situ(&self, _y: u64, _n: usize, _rows: usize, _scale: u64, _beta: f32, _lb: f32) -> anyhow::Result<()> { anyhow::bail!("{}: SITU not implemented", self.name()) }
    // Inkling 2's decode token and batched prefill on the device (`inkling2::device`); the host math is in `inkling2::forward`.
    /// Depth mix: `out[j]` (n floats each, j < nq) = the softmax-weighted sum of the RMS-normed states, scored by
    /// query row j of `q`. The states are the `nb` rows of `blocks` (the newest last) and `cur`.
    fn ink_mix(&self, blocks: u64, nb: usize, cur: u64, q: u64, nq: usize, out: u64, n: usize) -> anyhow::Result<()> {
        self.ink_mix_n(blocks, nb, cur, q, nq, out, n, 1)
    }
    /// Batched depth mix across `nt` tokens: `blocks` `[nb][nt][n]`, `cur` `[nt][n]`, `q` `[nq][n]`, `out` `[nq][nt][n]`.
    fn ink_mix_n(&self, _blocks: u64, _nb: usize, _cur: u64, _q: u64, _nq: usize, _out: u64, _n: usize, _nt: usize) -> anyhow::Result<()> {
        anyhow::bail!("{}: Inkling 2 device decode not implemented", self.name())
    }
    /// `count` loop controllers: `out[j] = cur_j - tanh(forget)*cur_j + tanh(gain)*msg[j]`, `cur_j` row
    /// `min(j, nins - 1)` of `cur`. `down` (`[rank][2n]`), `gain` and `forget` (`[rank][n]`) are f16, a controller
    /// after another. `gate`: `count * rank` floats of scratch.
    fn ink_ctrl(&self, down: u64, gain: u64, forget: u64, cur: u64, nins: usize, msg: u64, gate: u64, out: u64, n: usize, rank: usize, count: usize) -> anyhow::Result<()> {
        self.ink_ctrl_n(down, gain, forget, cur, nins, msg, gate, out, n, rank, count, 1)
    }
    /// Batched loop controllers across `nt` tokens: `cur` `[min(nins, count)][nt][n]`, `msg`/`out` `[count][nt][n]`,
    /// `gate` `[nt][count][rank]`.
    fn ink_ctrl_n(&self, _down: u64, _gain: u64, _forget: u64, _cur: u64, _nins: usize, _msg: u64, _gate: u64, _out: u64, _n: usize, _rank: usize, _count: usize, _nt: usize) -> anyhow::Result<()> {
        anyhow::bail!("{}: Inkling 2 device decode not implemented", self.name())
    }
    /// Short causal convolution with a residual, `out` may be `cur`: weights `w` `[channel][lag]`, the last
    /// `n_conv - 1` inputs in `hist` (advanced by `cur`), and input-dependent `taps` `[channel / per][lag]` or 0.
    fn ink_conv(&self, _cur: u64, _w: u64, _taps: u64, _hist: u64, _out: u64, _n: usize, _n_conv: usize, _per: usize) -> anyhow::Result<()> { anyhow::bail!("{}: Inkling 2 device decode not implemented", self.name()) }
    /// Batched short causal convolution with a residual over `nt` tokens: `cur` and `out` (`[nt][n]`) must not alias;
    /// `taps` (`[nt][n / per][n_conv]`) may be 0. Advances `hist` by the last `n_conv - 1` inputs across `hist + cur`.
    fn ink_conv_batch(&self, _cur: u64, _w: u64, _taps: u64, _hist: u64, _out: u64, _n: usize, _n_conv: usize, _per: usize, _nt: usize) -> anyhow::Result<()> {
        anyhow::bail!("{}: Inkling 2 batched conv not implemented", self.name())
    }
    /// `rows` rows of `w` (at most 128) floats at `x`, in place: times `scale` (per value, or 0), then each row
    /// scaled to an L2 norm of at most `thr`, or with `thr` 0 to an RMS of 1 (`eps`); then times `mul`. `dst` not 0:
    /// the result also goes there in the active KV layout.
    fn ink_clip(&self, x: u64, scale: u64, w: usize, rows: usize, thr: f32, eps: f32, mul: f32, dst: u64) -> anyhow::Result<()> {
        self.ink_clip_n(x, scale, w, rows, 1, thr, eps, mul, dst, 0, 1, 0, 0.0, 1.0)
    }
    /// Batched `ink_clip` over `nt` tokens of `rows` rows each (`x`: `[nt][rows][w]`, `scale`: `[rows][w]` or 0).
    /// When `dst != 0`, stores token `tok` at KV row `(dst_at0 + tok) % dst_ring`. When `log_alpha != 0.0`,
    /// scales `mul` by `1 + log_alpha * ln(max((pos0 + tok + 1) / log_floor, 1))`.
    fn ink_clip_n(
        &self, _x: u64, _scale: u64, _w: usize, _rows: usize, _nt: usize,
        _thr: f32, _eps: f32, _mul: f32,
        _dst: u64, _dst_at0: usize, _dst_ring: usize,
        _pos0: usize, _log_alpha: f32, _log_floor: f32,
    ) -> anyhow::Result<()> {
        anyhow::bail!("{}: Inkling 2 device decode not implemented", self.name())
    }
    /// Graph-capturable `ink_clip_n`: reads `pos0` from device `i32` pointer `pos_ptr` (`dst_at0 = pos0 % dst_ring`).
    fn ink_clip_pos(
        &self, _x: u64, _scale: u64, _w: usize, _rows: usize, _nt: usize,
        _thr: f32, _eps: f32, _mul: f32,
        _dst: u64, _dst_ring: usize,
        _pos_ptr: u64, _log_alpha: f32, _log_floor: f32,
    ) -> anyhow::Result<()> {
        anyhow::bail!("{}: Inkling 2 device decode not implemented", self.name())
    }
    /// Attention of one token over `rows` key and value rows (`nkv * hd` wide in active KV format; the token's own row is `at`, the
    /// row before it `at - 1` modulo `rows`). Logit: q.k / sqrt(hd) + r.bias[distance] (distance < `ext`) - sink, a
    /// softmax with one more logit of 0, times sigmoid(gate). `hd` divides 128. `sc`: `nh * rows.div_ceil(64) * (hd + 2)`
    /// floats of scratch.
    fn ink_attn(&self, _q: u64, _r: u64, _sink: u64, _gate: u64, _k: u64, _v: u64, _bias: u64, _sc: u64, _out: u64, _rows: usize, _at: usize, _nh: usize, _nkv: usize, _hd: usize, _nr: usize, _ext: usize) -> anyhow::Result<()> { anyhow::bail!("{}: Inkling 2 device decode not implemented", self.name()) }
    /// Graph-capturable `ink_attn`: reads `pos` from device `i32` pointer `pos_ptr`, sets `rows = min(pos + 1, ring)` and `at = pos % ring`,
    /// dispatching `max_rows.div_ceil(64)` chunks.
    fn ink_attn_pos(&self, _q: u64, _r: u64, _sink: u64, _gate: u64, _k: u64, _v: u64, _bias: u64, _sc: u64, _out: u64, _max_rows: usize, _ring: usize, _pos_ptr: u64, _nh: usize, _nkv: usize, _hd: usize, _nr: usize, _ext: usize) -> anyhow::Result<()> { anyhow::bail!("{}: Inkling 2 device decode not implemented", self.name()) }
    /// Batched causal Inkling 2 attention over `nt` query tokens starting at position `pos` (`hd` 32/64/128).
    /// When `kc_new != 0`, keys/values for `kp >= pos` read row `kp - pos` of `kc_new`/`vc_new` so SWA history `< pos` in `kc`/`vc` is preserved.
    /// `q`/`gate`/`out`: `[nt][nh][hd]`, `r`: `[nt][nh][nr]`, `sink`: `[nt][nh]`, `bias`: `[ext][nr]`.
    fn ink_attn_prefill(
        &self, _q: u64, _r: u64, _sink: u64, _gate: u64, _kc: u64, _vc: u64, _kc_new: u64, _vc_new: u64, _bias: u64, _out: u64,
        _pos: usize, _nt: usize, _nh: usize, _nkv: usize, _hd: usize, _nr: usize, _ext: usize, _ring: usize, _win: usize,
    ) -> anyhow::Result<()> {
        anyhow::bail!("{}: Inkling 2 prefill attention not implemented", self.name())
    }
    /// `acc[c] = (cpu ? cpu[c] : 0) + sum over e < ne of wts[e] * d[e * n + c]`.
    fn ink_ffn_sum(&self, acc: u64, d: u64, wts: u64, ne: usize, n: usize, cpu: u64) -> anyhow::Result<()> {
        self.ink_ffn_sum_n(acc, d, wts, ne, n, 1, cpu)
    }
    /// Batched `ink_ffn_sum` over `nt` tokens (`acc`/`cpu`: `[nt][n]`, `d`: `[nt][ne][n]`, `wts`: `[nt][ne]`).
    fn ink_ffn_sum_n(&self, _acc: u64, _d: u64, _wts: u64, _ne: usize, _n: usize, _nt: usize, _cpu: u64) -> anyhow::Result<()> {
        anyhow::bail!("{}: Inkling 2 device decode not implemented", self.name())
    }
    /// Inkling 2 sigmoid top-k router for `rows` tokens: picks top `k` from `logits` (`[rows, n_expert]`) using
    /// `sigmoid(logit) + bias`, writes indices `[rows, k]` to `ids`, normalized unbiased probs * `exp_scale` (* `dn_sc[id]`
    /// if non-null) to `wts`, and optional `gu_sc[id]` (or `1.0`) to `gsc_out`.
    fn ink_route(&self, _logits: u64, _bias: u64, _ids: u64, _wts: u64, _gu_sc: u64, _dn_sc: u64, _gsc_out: u64, _n_expert: usize, _k: usize, _exp_scale: f32, _rows: usize) -> anyhow::Result<()> { anyhow::bail!("{}: Inkling 2 router not implemented", self.name()) }
    /// Gyro files (`config::Hadamard`): each of `rows` rows of n floats at `x` (times silu(x) * u first when `u` is
    /// not 0) becomes H (s x) per `block` values, written to `out`, which may be `x`. `s`: the row's n signs as floats.
    fn hadamard(&self, _x: u64, _u: u64, _s: u64, _out: u64, _n: usize, _block: usize, _rows: usize) -> anyhow::Result<()> {
        anyhow::bail!("this file rotates its expert inputs (prism.hadamard), which the {} backend does not do", self.name())
    }
    /// The SwiGLU of `rows` rows of n for a GEMV with quantized weights: either g = silu(g) * u (returns 0), or
    /// silu(g) * u quantized into `xq` as `quant_xq` does (returns xq, for the GEMV's xq; g is then left as is).
    fn swiglu_q(&self, g: u64, u: u64, n: usize, rows: usize, _xq: u64) -> anyhow::Result<u64> {
        self.swiglu(g, u, rows * n)?;
        Ok(0)
    }
    fn ssm_gates(&self, beta: u64, g: u64, dt_bias: u64, a: u64, n: usize, nv: usize) -> anyhow::Result<()>;
    /// One token's causal conv and SiLU; then the first `nn` groups of 128 channels are l2-normalized as `l2_norm` does.
    fn conv_silu(&self, qkv: u64, state: u64, w: u64, y: u64, ch: usize, kw: usize, nn: usize, eps: f32) -> anyhow::Result<()>;
    /// Scan `t` tokens (rows of y `ych` floats apart) through the DeltaNet state `s`.
    /// `snap`: room for `nsnap` states; the state after each of the first min(nsnap, t - 1) tokens is kept there.
    /// `interleave`: value head h reads key head h / (nv / nk), else h % nk.
    /// `dt_bias` not 0: beta and g are raw, and the kernel applies `ssm_gates` (with `dt_bias` and `a`) as it reads them.
    fn deltanet(&self, y: u64, s: u64, beta: u64, g: u64, dt_bias: u64, a: u64, o: u64, nk: usize, nv: usize, interleave: bool, dk: usize, dv: usize, scale: f32, t: usize, ych: usize, snap: u64, nsnap: usize) -> anyhow::Result<()>;
    /// Stride (in tokens) between saved DeltaNet checkpoints in `snap` (`1` saves every token, `4` saves every 4th token).
    fn delta_snap_stride(&self) -> usize { 1 }
    fn deltanet_strided(&self, y: u64, s: u64, beta: u64, g: u64, dt_bias: u64, a: u64, o: u64, nk: usize, nv: usize, interleave: bool, dk: usize, dv: usize, scale: f32, t: usize, ych: usize, snap: u64, nsnap: usize, _snap_stride: usize) -> anyhow::Result<()> {
        self.deltanet(y, s, beta, g, dt_bias, a, o, nk, nv, interleave, dk, dv, scale, t, ych, snap, nsnap)
    }
    /// Causal conv + SiLU over `t` tokens of `ch` channels, then roll the conv state forward (snapshots as in `deltanet`).
    fn conv_seq(&self, qkv: u64, state: u64, w: u64, y: u64, ch: usize, kw: usize, t: usize, snap: u64, nsnap: usize) -> anyhow::Result<()>;
    /// Grouped dynamic causal convolution for DFlash2 (`part = 0` prepare, `part = 1` finish).
    fn dflash2_conv(&self, x: u64, dyn_w: u64, base: u64, out: u64, h: usize, gs: usize, k: usize, t: usize, part: usize) -> anyhow::Result<()>;
    /// `qg` 0: no output gate.
    fn attn_prefill(&self, q: u64, kc: u64, vc: u64, qg: u64, out: u64, step: u64, t: usize, nh: usize, nkv: usize, hd: usize, scale: f32) -> anyhow::Result<()> {
        self.attn_prefill_m(q, kc, vc, qg, out, step, t, nh, nkv, hd, scale, 0, 0, 0, 0)
    }
    /// `attn_prefill` where query row i sees only the keys whose 4-cell block has its bit set in row i of `qm` (`mw`
    /// words a row, rows up to `t` rounded up to 64; see `qsa_select`). `qm` 0: no mask. `sink` (0 = none): per-head
    /// attention-sink logits. `win` (0 = none): sliding-window width.
    fn attn_prefill_m(&self, q: u64, kc: u64, vc: u64, qg: u64, out: u64, step: u64, t: usize, nh: usize, nkv: usize, hd: usize, scale: f32, qm: u64, mw: usize, sink: u64, win: usize) -> anyhow::Result<()>;
    /// QSA indexer keys (see `Model::qsa`): pool the blocks of 4 positions that the `t` raw keys (rows of 128) at
    /// positions step[0] .. complete into `pool` (rows of 128 per block: mean, rms norm with `w`, NEOX rope at the
    /// block's first position). `ring`: 4 rows of 128, the raw keys of the incomplete block (row = position % 4).
    fn qsa_pool(&self, _ik: u64, _ring: u64, _pool: u64, _w: u64, _step: u64, _t: usize, _n_rot: usize, _base: f32, _eps: f32) -> anyhow::Result<()> {
        anyhow::bail!("{}: QSA sparse attention not implemented", self.name())
    }
    /// QSA block selection for `rows` query rows at positions step[0] + r0 + i (rows of 4 normed, roped heads of 128 in
    /// `q`): scores into `score` (rows of `nbs`), then either cell lists into `sel` (rows of `sst` ints, `mw` 0) for
    /// `attn_decode_q`, or block masks into `mask` (rows of `mw` words) for `attn_prefill_m`. `top`: blocks kept.
    fn qsa_select(&self, _q: u64, _pool: u64, _score: u64, _sel: u64, _mask: u64, _step: u64, _r0: usize, _rows: usize, _nbs: usize, _top: usize, _sst: usize, _mw: usize) -> anyhow::Result<()> {
        anyhow::bail!("{}: QSA sparse attention not implemented", self.name())
    }
    fn moe_combine_seq(&self, x: u64, ye: u64, wts: u64, inv: u64, k: usize, sh: u64, sg: u64, n: usize, t: usize) -> anyhow::Result<()>;
    fn norm_gate(&self, o: u64, w: u64, z: u64, n: usize, heads: usize, eps: f32) -> anyhow::Result<()>;
    /// `norm_gate`, then either returns 0, or also writes o (all heads of a token as one row) quantized as `quant_xq`
    /// does to `xq` and returns xq, for the GEMV's xq.
    fn norm_gate_q(&self, o: u64, w: u64, z: u64, n: usize, heads: usize, eps: f32, _xq: u64) -> anyhow::Result<u64> {
        self.norm_gate(o, w, z, n, heads, eps)?;
        Ok(0)
    }
    /// `norm_gate_q` gating with sigmoid(z) in place of silu(z) (qwen4exp).
    fn norm_gate_sig(&self, _o: u64, _w: u64, _z: u64, _n: usize, _heads: usize, _eps: f32, _xq: u64) -> anyhow::Result<u64> {
        anyhow::bail!("{}: sigmoid-gated norm not implemented", self.name())
    }
    // qwen4exp hyper-connections (see `Model::hc_read`); the residual of a token is hc rows of n.
    /// x = silu(x * sc) over n values.
    fn hc_silu(&self, _x: u64, _sc: f32, _n: usize) -> anyhow::Result<()> { anyhow::bail!("{}: hyper-connections not implemented", self.name()) }
    /// out[t] = sum over streams c of xn[t][c] * sigmoid(g[t][c]) / hc, for t < nt.
    fn hc_mix(&self, _xn: u64, _g: u64, _out: u64, _n: usize, _hc: usize, _nt: usize) -> anyhow::Result<()> { anyhow::bail!("{}: hyper-connections not implemented", self.name()) }
    /// Stream c of r[t] += out[t] * 2 sigmoid(inj[t][c] / hc), for t < nt; `clear`: zero out afterwards.
    fn hc_write(&self, _r: u64, _out: u64, _inj: u64, _n: usize, _hc: usize, _nt: usize, _clear: bool) -> anyhow::Result<()> { anyhow::bail!("{}: hyper-connections not implemented", self.name()) }
    /// PLE gate (see `Model::ple`): key and query rows of hc * n per token, value rows of n.
    fn ple_gate(&self, _key: u64, _query: u64, _value: u64, _gated: u64, _n: usize, _hc: usize, _nt: usize) -> anyhow::Result<()> { anyhow::bail!("{}: PLE not implemented", self.name()) }
    /// PLE conv and residual add for nt tokens at positions step[0] .. (rows of d); `ring`: 16 rows of d.
    fn ple_conv(&self, _r: u64, _gated: u64, _norm: u64, _kw: u64, _ring: u64, _step: u64, _d: usize, _nt: usize) -> anyhow::Result<()> { anyhow::bail!("{}: PLE not implemented", self.name()) }
    /// Store `t` tokens of `n` values at positions step[0] ...
    fn kv_store(&self, k: u64, v: u64, kc: u64, vc: u64, n: usize, t: usize, step: u64, hd: usize) -> anyhow::Result<()>;
    /// Decode attention for `nt` query tokens, token t over positions 0 ..= step[2 t] (rows of q, qg, out and step per
    /// token, `part` holds `nh * ATT_SPLITS * (hd + 2)` floats per token). `qg` 0: no output gate.
    fn attn_decode(&self, q: u64, kc: u64, vc: u64, qg: u64, out: u64, part: u64, step: u64, nh: usize, nkv: usize, hd: usize, scale: f32, nt: usize) -> anyhow::Result<()> {
        self.attn_decode_gq(q, kc, vc, qg, out, part, step, nh, nkv, hd, scale, ATT_GQA_MIN, nt, 0, 0)
    }
    /// `attn_decode_gq`, also writing `out` quantized for `gemvq` into `xq` where supported; returns the xq to use (0: none).
    /// `sel` not 0: QSA cell lists from `qsa_select` (rows of `sst` ints per token). `sink` (0 = none): per-head
    /// attention-sink logits. `win` (0 = none): sliding-window width.
    fn attn_decode_q(&self, q: u64, kc: u64, vc: u64, qg: u64, out: u64, part: u64, step: u64, nh: usize, nkv: usize, hd: usize, scale: f32, gq_min: i32, nt: usize, _xq: u64, sel: u64, _sst: usize, sink: u64, win: usize) -> anyhow::Result<u64> {
        anyhow::ensure!(sel == 0, "{}: QSA sparse attention not implemented", self.name());
        self.attn_decode_gq(q, kc, vc, qg, out, part, step, nh, nkv, hd, scale, gq_min, nt, sink, win)?;
        Ok(0)
    }
    /// Whether `rms_norm_rope_qk_rows` and `attn_rows` exist here (`GpuModel::step_batch` needs them).
    fn kv_rows(&self) -> bool { false }
    /// `rms_norm_rope_qk` for `nt` tokens of different sequences: token t is at position `step[2t]` and stores its K
    /// and V in the caches whose addresses are `kvt[2t]` and `kvt[2t + 1]` (u64 each).
    fn rms_norm_rope_qk_rows(&self, _q: u64, _q_stride: usize, _qw: u64, _qo: u64, _nh: usize, _k: u64, _kw: u64, _ko: u64, _nkv: usize, _hd: usize, _eps: f32, _n_rot: usize,
                             _step: u64, _base: f32, _v: u64, _kvt: u64, _nt: usize) -> anyhow::Result<()> {
        anyhow::bail!("{}: no per-row KV kernels", self.name())
    }
    /// `attn_decode_gq` with `gq_min` 0 for `nt` tokens of different sequences, with the caches of
    /// `rms_norm_rope_qk_rows`; `part` holds `nt` rows of the split partials. `splits` (1 ..= ATT_SPLITS): blocks a
    /// head group; many short rows run faster with few.
    fn attn_rows(&self, _q: u64, _qg: u64, _out: u64, _part: u64, _step: u64, _nh: usize, _nkv: usize, _hd: usize, _scale: f32, _nt: usize, _sink: u64, _win: usize, _kvt: u64,
                 _splits: usize) -> anyhow::Result<()> {
        anyhow::bail!("{}: no per-row KV kernels", self.name())
    }
    /// `attn_decode` that shares K/V loads across a GQA group from `gq_min` positions on (tests use 0 to force it).
    fn attn_decode_gq(&self, q: u64, kc: u64, vc: u64, qg: u64, out: u64, part: u64, step: u64, nh: usize, nkv: usize, hd: usize, scale: f32, gq_min: i32, nt: usize, sink: u64, win: usize) -> anyhow::Result<()>;
    /// Route `t` tokens (logit rows of n, ids / wts rows of k). `sigmoid` selects sigmoid scores instead of softmax.
    fn router_topk(&self, p: u64, n: usize, k: usize, extra_scale: f32, ids: u64, wts: u64, t: usize, sigmoid: bool) -> anyhow::Result<()>;
    /// Decode routing: `router_topk`, the expert cache lookup (slots[j] = table[ids[j]], -1 = not resident; wh[j] =
    /// wts[j] on a hit, else 0), then the routing and (on a miss) `h` (rows of nh) written to the layer mailbox at
    /// `mb` and its request flag raised. `mb` = 0 skips every mailbox store where `moe_mailbox_optional` holds. `pid`
    /// (0 = none) receives the last token's ids, the prediction the next token's prefetch uses.
    fn moe_route(&self, p: u64, n: usize, k: usize, extra_scale: f32, table: u64, ids: u64, wts: u64, slots: u64, wh: u64, h: u64, mb: u64, nh: usize, h_off: usize, step: u64, t: usize, pid: u64, sigmoid: bool) -> anyhow::Result<()>;
    /// `mb` = 0: no CPU part. Otherwise adds the CPU result at word `y_off` of the mailbox once its done flag is `seq`.
    fn moe_combine(&self, x: u64, ye: u64, w: u64, k: usize, sh: u64, sg: u64, mb: u64, y_off: usize, step: u64, slots: u64, n: usize, t: usize) -> anyhow::Result<()>;
    /// Whether `moe_route`/`moe_combine` accept `mb = 0`, i.e. a fully resident expert cache costs the CPU worker
    /// nothing (no mailbox traffic at all). False where the kernels read the mailbox unconditionally (Vulkan).
    /// Most tokens of one `gemvn`/verify forward: 4 unless the backend's kernels take `cpu_worker::TMAX`.
    fn verify_rows(&self) -> usize { 4 }
    fn moe_mailbox_optional(&self) -> bool { false }

    /// Whether `moe_prefetch` does anything here (CUDA: yes; Vulkan: no kernel, the knob is ignored).
    fn moe_prefetch_supported(&self) -> bool { false }

    /// Hint into L2 the cached slots of the experts `ids` of a layer (`table` maps expert to slot, -1 = on the CPU):
    /// `STRATA_MOE_PREFETCH`, a performance knob only. A no-op where `moe_prefetch_supported` is false.
    fn moe_prefetch(&self, ids: u64, k: usize, table: u64, base: u64, stride: usize, bytes: usize) -> anyhow::Result<()> {
        let _ = (ids, k, table, base, stride, bytes);
        Ok(())
    }

    /// `moe_gemv` of the down experts (`k` slots per token, x pre-quantized at `xq` when not 0) into `ye`, then
    /// `moe_combine` of ye into x.
    fn moe_down_combine(&self, ty: GgmlType, rows: usize, cols: usize, w: u64, stride: usize, slots: u64, k: usize, h: u64, ye: u64, xq: u64,
                        x: u64, wh: u64, sh: u64, sg: u64, mb: u64, y_off: usize, step: u64, t: usize) -> anyhow::Result<()> {
        self.moe_gemv(ty, rows, cols, w, stride, slots, t * k, h, cols, 1, ye, None, xq)?;
        self.moe_combine(x, ye, wh, k, sh, sg, mb, y_off, step, slots, rows, t)
    }
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

#[cfg(test)]
mod tests {
    use super::device_order;

    /// Identifiers are letters here. Two cards of one model have one name and two identifiers, so names play no part.
    #[test]
    fn a_gpu_index_is_the_same_card_on_cuda_and_vulkan() {
        // CUDA counts the faster card first, Vulkan in bus order (the 5060 Ti and 3060 of the report).
        assert_eq!(device_order(&[Some('T'), Some('S')], &['S', 'T']), [Some(1), Some(0)]);
        // Two cards of the same model, and an integrated GPU that CUDA does not know: it follows the CUDA devices.
        assert_eq!(device_order(&[Some('B'), Some('A')], &['A', 'I', 'B']), [Some(2), Some(0), Some(1)]);
        // CUDA device 0 is not usable on Vulkan (filtered out): index 0 fails, index 1 is still CUDA's card 1.
        assert_eq!(device_order(&[Some('A'), Some('B')], &['B']), [None, Some(0)]);
        // A CUDA device without identifier takes the first Vulkan device that no CUDA device names.
        assert_eq!(device_order(&[None, Some('B')], &['B', 'A']), [Some(1), Some(0)]);
        // No CUDA (AMD, Intel, llvmpipe), or no identifier at all: Vulkan's own order, as before.
        assert_eq!(device_order(&[], &['A', 'B']), [Some(0), Some(1)]);
        assert_eq!(device_order(&[None, None], &['A', 'B']), [Some(0), Some(1)]);
        assert_eq!(device_order::<char>(&[Some('A')], &[]), [None]);
    }
}
