//! CPU side of the expert split, without per-layer stream syncs. For each MoE layer the GPU writes routing ids,
//! weights, cache slots and (when some expert missed) the normed activation into host-mapped pinned memory, then
//! raises the layer's request flag. This worker spins on the flag, computes the missed experts with `cpu_moe`, writes
//! their weighted sum back and raises the done flag, which the GPU combine kernel waits on only if the layer missed.

use crate::model::Model;
use cudarc::driver::{PinnedHostSlice, sys};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering, fence};
use std::sync::{Arc, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Word offsets inside one layer's mailbox. Flags sit on separate cache lines.
pub const REQ: usize = 0;
/// Tokens in the request (1 for decode, up to `TMAX` for a multi-token verify).
pub const NTOK: usize = 1;
pub const DONE: usize = 16;
pub const IDS: usize = 32;

/// Most tokens one request may carry.
pub const TMAX: usize = 4;

/// Per layer: ids, weights and slots for up to TMAX tokens of `k` entries (token t at t * k; the three arrays are
/// each NTOK * k long), then TMAX activations of `n` floats at `h` and TMAX results at `y`.
pub struct Mailbox { pub k: usize, pub n: usize, pub h: usize, pub y: usize, pub stride: usize }

impl Mailbox {
    pub fn new(k: usize, n: usize) -> Mailbox {
        let h = (IDS + 3 * TMAX * k).next_multiple_of(16);
        Mailbox { k, n, h, y: h + TMAX * n, stride: (h + 2 * TMAX * n).next_multiple_of(16) }
    }
}

pub struct CpuWorker {
    _mem: PinnedHostSlice<u32>,
    host: usize,
    /// Device address of the mailboxes (layer `il` at `dev + 4 * il * layout.stride`).
    pub dev: u64,
    pub layout: Mailbox,
    seq: u32,
    tx: Option<mpsc::Sender<u32>>,
    stop: Arc<AtomicBool>,
    busy_ns: Arc<AtomicU64>,
    thread: Option<JoinHandle<()>>,
}

impl CpuWorker {
    /// Mailboxes cover layers up to the largest of `moe_layers`. Safety: `m` must outlive the worker (it is joined on drop).
    pub fn new(m: &Model, ctx: &Arc<cudarc::driver::CudaContext>, moe_layers: Vec<usize>) -> anyhow::Result<CpuWorker> {
        let c = &m.cfg;
        let layout = Mailbox::new(c.n_expert_used, c.n_embd);
        let len = layout.stride * moe_layers.iter().max().map_or(0, |&il| il + 1);
        let flags = sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE;
        let mut mem = unsafe { ctx.alloc_pinned_with_flags::<u32>(len, flags)? };
        let host = mem.as_mut_ptr()? as usize;
        unsafe { std::ptr::write_bytes(host as *mut u32, 0, len) };
        let mut dev = 0;
        let r = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, host as *mut _, 0) };
        anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuMemHostGetDevicePointer: {r:?}");

        let (tx, rx) = mpsc::channel::<u32>();
        let (stop, busy_ns) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicU64::new(0)));
        let (st, bn) = (stop.clone(), busy_ns.clone());
        let mp = m as *const Model as usize;
        let lay = Mailbox::new(c.n_expert_used, c.n_embd);
        let thread = std::thread::Builder::new().name("strata-cpu-experts".into()).spawn(move || {
            let m = unsafe { &*(mp as *const Model) };
            let w = |i: usize| host as *mut u32 as usize + 4 * i;
            while let Ok(seq) = rx.recv() {
                for &il in &moe_layers {
                    let b = il * lay.stride;
                    let at = |o: usize| (w(b + o)) as *mut u32;
                    // A newer seq here means the GPU already moved on to the next token, so it did not wait for this
                    // layer (no miss): skip it. Waiting for the old seq would hang, and the next miss would trap.
                    let mut r;
                    while { r = unsafe { at(REQ).read_volatile() }; r != seq } {
                        if st.load(Ordering::Relaxed) { return; }
                        if r != 0 && (r.wrapping_sub(seq) as i32) > 0 { break; }
                        std::hint::spin_loop();
                    }
                    if r != seq { continue; }
                    fence(Ordering::Acquire);
                    let t0 = Instant::now();
                    let rd = |o: usize| unsafe { at(o).read_volatile() };
                    let nt = (rd(NTOK) as usize).clamp(1, TMAX);
                    let (k, kt, n) = (lay.k, lay.k * nt, lay.n);
                    for t in 0..nt {
                        let sel: Vec<(usize, f32)> = (t * k..(t + 1) * k).filter(|&j| (rd(IDS + 2 * kt + j) as i32) < 0)
                            .map(|j| (rd(IDS + j) as usize, f32::from_bits(rd(IDS + kt + j)))).collect();
                        if sel.is_empty() { continue; }
                        let h = unsafe { std::slice::from_raw_parts(at(lay.h + t * n) as *const f32, n) }.to_vec();
                        let y = crate::cpu_moe::experts(m, il, &h, &sel);
                        unsafe { std::ptr::copy_nonoverlapping(y.as_ptr(), at(lay.y + t * n) as *mut f32, n) };
                    }
                    fence(Ordering::Release);
                    unsafe { at(DONE).write_volatile(seq) };
                    bn.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                }
            }
        })?;
        Ok(CpuWorker { _mem: mem, host, dev, layout, seq: 0, tx: Some(tx), stop, busy_ns, thread: Some(thread) })
    }

    /// Start a new token: returns the sequence number the GPU must publish with.
    pub fn begin(&mut self) -> u32 {
        self.seq = self.seq.wrapping_add(1).max(1);
        let _ = self.tx.as_ref().unwrap().send(self.seq);
        self.seq
    }

    /// Routing of layer `il` from the last finished request, all its tokens: (expert id, routed weight, slot or -1).
    pub fn routing(&self, il: usize) -> Vec<(i32, f32, i32)> {
        let b = il * self.layout.stride;
        let rd = |o: usize| unsafe { (self.host as *const u32).add(b + o).read_volatile() };
        let k = self.layout.k * (rd(NTOK) as usize).clamp(1, TMAX);
        (0..k).map(|j| (rd(IDS + j) as i32, f32::from_bits(rd(IDS + k + j)), rd(IDS + 2 * k + j) as i32)).collect()
    }

    /// Total time the worker spent computing experts.
    pub fn busy(&self) -> Duration { Duration::from_nanos(self.busy_ns.load(Ordering::Relaxed)) }
}

impl Drop for CpuWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.tx = None;
        if let Some(t) = self.thread.take() { let _ = t.join(); }
    }
}
