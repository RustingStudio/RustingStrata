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

/// `len` bytes of device memory at device address `addr`, freed when the owner drops.
pub struct Buf {
    addr: u64,
    len: usize,
    _owner: Box<dyn Any + Send + Sync>,
}

impl Buf {
    pub fn new(addr: u64, len: usize, owner: impl Any + Send + Sync) -> Buf { Buf { addr, len, _owner: Box::new(owner) } }
    pub fn addr(&self) -> u64 { self.addr }
    pub fn len(&self) -> usize { self.len }
}

pub struct GpuTensor {
    pub ty: GgmlType,
    pub rows: usize,
    pub cols: usize,
    pub ptr: u64,
    _buf: Buf,
}

impl GpuTensor {
    /// `rows` x `cols` of `ty` stored in `buf` from its start.
    pub fn new(ty: GgmlType, rows: usize, cols: usize, buf: Buf) -> GpuTensor { GpuTensor { ty, rows, cols, ptr: buf.addr, _buf: buf } }
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
            Err(c) if format!("{c}") == "CUDA init" => Ok(Box::new(crate::vulkan::Vk::new(device)
                .map_err(|v| anyhow::anyhow!("no GPU backend loads: CUDA: {c:#}; Vulkan: {v:#}"))?)),
            Err(c) => Err(c),
        },
    }
}

pub trait Backend: Send {
    /// "cuda" or "vulkan".
    fn name(&self) -> &'static str;
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
    /// Queue the ops of `f`, none of which reads what another writes, with no ordering among them (Vulkan: no
    /// barriers between them); ops after the batch see all their results.
    fn batch(&self, f: &mut dyn FnMut() -> anyhow::Result<()>) -> anyhow::Result<()> { f() }
    /// Zeroed host memory of `words` u32 that kernels read and write at the `Buf` address. The host pointer is
    /// valid while the `Buf` lives.
    fn host_mailbox(&self, words: usize) -> anyhow::Result<(Buf, *mut u32)>;

    fn upload(&self, t: &Tensor) -> anyhow::Result<GpuTensor> {
        let b = self.alloc(t.data.len())?;
        self.htod(t.data, &b, 0)?;
        Ok(GpuTensor::new(t.ty, t.rows, t.cols, b))
    }

    /// u16 words of a KV cache for `n` values: f16, or q8 with one f16 scale per 32 values.
    fn kv_words(&self, n: usize) -> usize;
    /// GEMM over a tile list: for each tile, positions m0 .. m0 + cnt get y[m, 0..rows] = W_tile x[xrow[m]] (x row m
    /// when `xrow` is 0). `tiles` is a device array of [w, m0 | cnt << 32]. `act`: 0 stores, 1 computes
    /// y = silu(y) * result (SwiGLU with the gate already in y), 2 adds to y. `n_x`: rows of x the tiles read.
    fn gemm(&self, ty: GgmlType, rows: usize, cols: usize, tiles: u64, n_tiles: usize, x: u64, x_stride: usize, n_x: usize, xrow: u64, y: u64, y_stride: usize, act: i32) -> anyhow::Result<()>;
    /// y[m] (op) W x[m] for positions m < n, x and y rows `x_stride` / `y_stride` floats apart.
    fn gemm_dense(&self, w: &GpuTensor, n: usize, x: u64, x_stride: usize, y: u64, y_stride: usize, act: i32) -> anyhow::Result<()>;
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
    /// out[r] = argmax of row r of x (`rows` rows of n floats), first index on ties.
    fn argmax_rows(&self, x: u64, n: usize, rows: usize, out: u64) -> anyhow::Result<()>;
    /// Row r: vector r % per (n floats) of token r / per, tokens `stride` floats apart.
    fn l2_norm(&self, x: u64, n: usize, per: usize, stride: usize, rows: usize, eps: f32) -> anyhow::Result<()>;
    /// `heads` rows of `stride` floats, `per` per token; token t is at position step[0] + t.
    fn rope_neox(&self, x: u64, stride: usize, heads: usize, per: usize, n_rot: usize, step: u64, base: f32) -> anyhow::Result<()>;
    /// `rms_norm` of `rows` heads of n (output rows of n), then `rope_neox` on them.
    fn rms_norm_rope(&self, x: u64, x_stride: usize, w: u64, out: u64, n: usize, rows: usize, eps: f32, per: usize, n_rot: usize, step: u64, base: f32) -> anyhow::Result<()> {
        self.rms_norm(x, x_stride, w, out, n, n, rows, eps)?;
        self.rope_neox(out, n, rows, per, n_rot, step, base)
    }
    /// x += y.
    fn add(&self, x: u64, y: u64, n: usize) -> anyhow::Result<()>;
    /// g = silu(g) * u.
    fn swiglu(&self, g: u64, u: u64, n: usize) -> anyhow::Result<()>;
    /// y[t] = W (silu(g[t]) * u[t]) for t < nt (rows of `w.cols` floats; y rows `y_stride` apart); g may be overwritten.
    fn swiglu_gemv(&self, g: u64, u: u64, w: &GpuTensor, y: u64, y_stride: usize, nt: usize) -> anyhow::Result<()> {
        self.swiglu(g, u, nt * w.cols)?;
        if nt == 1 { self.gemv(w, g, y) } else { self.gemvn(w, g, w.cols, y, y_stride, nt) }
    }
    fn ssm_gates(&self, beta: u64, g: u64, dt_bias: u64, a: u64, n: usize, nv: usize) -> anyhow::Result<()>;
    /// One token's causal conv and SiLU; then the first `nn` groups of 128 channels are l2-normalized as `l2_norm` does.
    fn conv_silu(&self, qkv: u64, state: u64, w: u64, y: u64, ch: usize, kw: usize, nn: usize, eps: f32) -> anyhow::Result<()>;
    /// Scan `t` tokens (rows of y `ych` floats apart) through the DeltaNet state `s`.
    /// `snap`: room for `nsnap` states; the state after each of the first min(nsnap, t - 1) tokens is kept there.
    /// `interleave`: value head h reads key head h / (nv / nk), else h % nk.
    /// `dt_bias` not 0: beta and g are raw, and the kernel applies `ssm_gates` (with `dt_bias` and `a`) as it reads them.
    fn deltanet(&self, y: u64, s: u64, beta: u64, g: u64, dt_bias: u64, a: u64, o: u64, nk: usize, nv: usize, interleave: bool, dk: usize, dv: usize, scale: f32, t: usize, ych: usize, snap: u64, nsnap: usize) -> anyhow::Result<()>;
    /// Causal conv + SiLU over `t` tokens of `ch` channels, then roll the conv state forward (snapshots as in `deltanet`).
    fn conv_seq(&self, qkv: u64, state: u64, w: u64, y: u64, ch: usize, kw: usize, t: usize, snap: u64, nsnap: usize) -> anyhow::Result<()>;
    /// `qg` 0: no output gate.
    fn attn_prefill(&self, q: u64, kc: u64, vc: u64, qg: u64, out: u64, step: u64, t: usize, nh: usize, nkv: usize, hd: usize, scale: f32) -> anyhow::Result<()>;
    fn moe_combine_seq(&self, x: u64, ye: u64, wts: u64, inv: u64, k: usize, sh: u64, sg: u64, n: usize, t: usize) -> anyhow::Result<()>;
    fn norm_gate(&self, o: u64, w: u64, z: u64, n: usize, heads: usize, eps: f32) -> anyhow::Result<()>;
    /// Store `t` tokens of `n` values at positions step[0] ...
    fn kv_store(&self, k: u64, v: u64, kc: u64, vc: u64, n: usize, t: usize, step: u64, hd: usize) -> anyhow::Result<()>;
    /// Decode attention for `nt` query tokens, token t over positions 0 ..= step[2 t] (rows of q, qg, out and step per
    /// token, `part` holds `nh * ATT_SPLITS * (hd + 2)` floats per token). `qg` 0: no output gate.
    fn attn_decode(&self, q: u64, kc: u64, vc: u64, qg: u64, out: u64, part: u64, step: u64, nh: usize, nkv: usize, hd: usize, scale: f32, nt: usize) -> anyhow::Result<()> {
        self.attn_decode_gq(q, kc, vc, qg, out, part, step, nh, nkv, hd, scale, ATT_GQA_MIN, nt)
    }
    /// `attn_decode` that shares K/V loads across a GQA group from `gq_min` positions on (tests use 0 to force it).
    fn attn_decode_gq(&self, q: u64, kc: u64, vc: u64, qg: u64, out: u64, part: u64, step: u64, nh: usize, nkv: usize, hd: usize, scale: f32, gq_min: i32, nt: usize) -> anyhow::Result<()>;
    /// Route `t` tokens (logit rows of n, ids / wts rows of k).
    fn router_topk(&self, p: u64, n: usize, k: usize, extra_scale: f32, ids: u64, wts: u64, t: usize) -> anyhow::Result<()>;
    /// Decode routing: `router_topk`, the expert cache lookup (slots[j] = table[ids[j]], -1 = not resident; wh[j] =
    /// wts[j] on a hit, else 0), then the routing and (on a miss) `h` (rows of nh) written to the layer mailbox at
    /// `mb` and its request flag raised.
    fn moe_route(&self, p: u64, n: usize, k: usize, extra_scale: f32, table: u64, ids: u64, wts: u64, slots: u64, wh: u64, h: u64, mb: u64, nh: usize, h_off: usize, step: u64, t: usize) -> anyhow::Result<()>;
    /// `mb` = 0: no CPU part. Otherwise adds the CPU result at word `y_off` of the mailbox once its done flag is `seq`.
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
