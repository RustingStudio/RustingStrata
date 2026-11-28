//! The CPU side of the expert pool in a decode token (see `pool`): the mailboxes and flags of `crate::cpu_worker`,
//! one per layer run whose experts are not all in VRAM, answered with `run_experts` as `plan`'s CPU experts are.
//! `ink_slots` raises a run's request flag when one of its experts missed, else its skip flag;
//! `ink_ffn_sum_mb` waits for the done flag only after a miss.
use super::*;
use crate::backend::{Backend, Buf};
use crate::cpu_worker::{DONE, IDS, Mailbox, REQ, SKIP, read_wc};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering, fence};
use std::sync::{Arc, mpsc};
use std::thread::JoinHandle;

/// A layer's routed experts in the mapped file, for the worker thread.
#[derive(Clone)]
pub(super) struct HostExperts { pub gate_up: Mat<'static>, pub down: Mat<'static>, pub gu_scale: Vec<f32> }

pub(super) struct InkWorker {
    _mem: Buf,
    host: usize,
    /// Device address of the mailboxes: run `il` at `dev + 4 * il * layout.stride`.
    pub dev: u64,
    pub layout: Mailbox,
    seq: u32,
    tx: Option<mpsc::Sender<u32>>,
    stop: Arc<AtomicBool>,
    busy_ns: Arc<AtomicU64>,
    thread: Option<JoinHandle<()>>,
}

impl InkWorker {
    /// `runs`: (mailbox index, layer) of every layer run with a mailbox, in the order a token runs them;
    /// `experts[ip]` the host experts of each such layer.
    pub fn new(be: &dyn Backend, cfg: &Inkling2Config, runs: Vec<(usize, usize)>, experts: Vec<Option<HostExperts>>) -> Result<Self> {
        let layout = Mailbox::new(cfg.n_used, cfg.n_embd);
        let len = layout.stride * runs.iter().map(|r| r.0 + 1).max().unwrap_or(1);
        let (mem, host) = be.host_mailbox(len)?;
        let (host, dev) = (host as usize, mem.addr());
        let (tx, rx) = mpsc::channel::<u32>();
        let (stop, busy_ns) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicU64::new(0)));
        let (st, bn, cfg) = (stop.clone(), busy_ns.clone(), cfg.clone());
        let lay = Mailbox::new(cfg.n_used, cfg.n_embd);
        // Its own pool, as `CpuWorker`'s: the expert passes start on the spinning thread at once.
        let pool = rayon::ThreadPoolBuilder::new().num_threads(rayon::current_num_threads())
            .thread_name(|i| format!("inkling-experts-{i}")).build()?;
        let thread = std::thread::Builder::new().name("inkling-cpu-experts".into()).spawn(move || pool.install(move || {
            let w = |i: usize| (host + 4 * i) as *mut u32;
            let ne = cfg.n_expert;
            while let Ok(seq) = rx.recv() {
                for &(il, ip) in &runs {
                    let at = |o: usize| w(il * lay.stride + o);
                    // A newer seq anywhere: the GPU moved on without waiting for this one.
                    let ahead = |v: u32| v != 0 && (v.wrapping_sub(seq) as i32) >= 0;
                    let mut r;
                    while { r = unsafe { at(REQ).read_volatile() }; r != seq } {
                        if st.load(Ordering::Relaxed) { return; }
                        if ahead(r) || ahead(unsafe { at(SKIP).read_volatile() }) { break; }
                        std::hint::spin_loop();
                    }
                    if r != seq { continue; }
                    fence(Ordering::SeqCst);
                    let t0 = std::time::Instant::now();
                    let k = lay.k;
                    let mut v = vec![0u32; (3 * k).next_multiple_of(4)];
                    unsafe { read_wc(at(IDS), &mut v) };
                    let mut h = vec![0u32; lay.n.next_multiple_of(4)];
                    unsafe { read_wc(at(lay.h), &mut h) };
                    let h: Vec<f32> = h[..lay.n].iter().map(|&b| f32::from_bits(b)).collect();
                    let ex = experts[ip].as_ref().expect("a mailbox run of a layer without host experts");
                    let list: Vec<_> = (0..k).filter(|&j| (v[2 * k + j] as i32) < 0).map(|j| {
                        let e = v[j] as usize;
                        (ex.gate_up.expert_host(e, ne), ex.down.expert_host(e, ne), f32::from_bits(v[k + j]), ex.gu_scale[e])
                    }).collect();
                    let y = run_experts(&list, &h, &cfg);
                    unsafe { std::ptr::copy_nonoverlapping(y.as_ptr(), at(lay.y) as *mut f32, lay.n) };
                    fence(Ordering::SeqCst);
                    unsafe { at(DONE).write_volatile(seq) };
                    fence(Ordering::SeqCst);
                    bn.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                }
            }
        }))?;
        Ok(InkWorker { _mem: mem, host, dev, layout, seq: 0, tx: Some(tx), stop, busy_ns, thread: Some(thread) })
    }

    /// Start a token: the sequence number the kernels must find in `step[1]`.
    pub fn begin(&mut self) -> u32 {
        self.seq = self.seq.wrapping_add(1).max(1);
        let _ = self.tx.as_ref().unwrap().send(self.seq);
        self.seq
    }

    /// Mailbox of run `il` on the device.
    pub fn mb(&self, il: usize) -> u64 { self.dev + (4 * il * self.layout.stride) as u64 }

    /// The routing of run `il` in the last token: (expert, slot or -1) for each of its k experts.
    pub fn routing(&self, il: usize) -> Vec<(usize, i32)> {
        let k = self.layout.k;
        let mut v = vec![0u32; (3 * k).next_multiple_of(4)];
        unsafe { read_wc((self.host as *const u32).add(il * self.layout.stride + IDS), &mut v) };
        (0..k).map(|j| (v[j] as usize, v[2 * k + j] as i32)).collect()
    }

    /// Time the worker spent computing experts.
    pub fn busy(&self) -> std::time::Duration { std::time::Duration::from_nanos(self.busy_ns.load(Ordering::Relaxed)) }
}

impl Drop for InkWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.tx = None;
        if let Some(t) = self.thread.take() { let _ = t.join(); }
    }
}
