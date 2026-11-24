//! VRAM slots for the routed experts of the layers without a slab (see `device`): one pool of equal slots per
//! expert layout, a table per layer (expert -> slot, -1 = on the CPU) that `ink_slots` reads, and usage counts that
//! decide which experts hold the slots. Routes skew hard: in the coda of Inkling 2 Mini, 100 of 768 experts take
//! 60-67% of a text's uses. The table changes between tokens only, with the queue idle.
use super::*;
use super::worker::{HostExperts, InkWorker};
use crate::backend::{Buf, GpuTensor};
use crate::deepseek::mat::dev::{self, SlabBuf};
use crate::quant::GgmlType;
use std::sync::Arc;

/// Share of a usage count kept per token: a count forgets in a few hundred tokens.
const KEEP: f32 = 0.995;
/// Experts copied into free slots per token while a pool fills, and into a less used expert's slot once it is full.
/// A copy of an 8.5 MB expert takes about 0.7 ms over PCIe 3.0.
const FILLS: usize = 4;
const SWAPS: usize = 2;
/// Share of a long count kept per token: the saved profile, which places the next run's experts at load.
const LONG_KEEP: f32 = 0.9999;
/// Slots past each pool's own for a prompt chunk's experts that are not in it (`stage`).
pub(super) const STAGE: usize = 32;

pub(super) struct PoolLayer {
    pub pool: usize,
    pub table_dev: Buf,
    table: Vec<i32>,
    pub gate_inp: GpuTensor,
    _gi: Arc<SlabBuf>,
    pub bias: Arc<SlabBuf>,
    pub gu_scale: Option<Arc<SlabBuf>>,
    pub dn_scale: Option<Arc<SlabBuf>>,
    /// (type, rows, cols) of one expert's gate_up and down matrices; down starts `dn_off` bytes into a slot.
    pub gu: (GgmlType, usize, usize),
    pub dn: (GgmlType, usize, usize),
    pub dn_off: usize,
    host: HostExperts,
}

/// `owner.len()` slots of `bytes`, then `STAGE` more for a prompt chunk.
struct Slots { buf: Option<Arc<SlabBuf>>, bytes: usize, owner: Vec<Option<(usize, usize)>> }

pub(super) struct Pool {
    slots: Vec<Slots>,
    pub layers: Vec<Option<PoolLayer>>,
    /// Decayed uses, [layer * n_expert + expert].
    counts: Vec<f32>,
    /// The same with a memory of some ten thousand tokens: saved when the pool goes, read at the next load.
    long: Vec<f32>,
    profile: std::path::PathBuf,
    ne: usize,
    pub worker: InkWorker,
    /// (mailbox, layer) of the runs with a pool layer, in token order.
    runs: Vec<(usize, usize)>,
    /// Since the last PROF line: expert uses found in a slot, uses on the CPU, experts copied in.
    pub stats: [usize; 3],
    pub tokens: usize,
}

impl Pool {
    /// The pool for layers `need` with what the weight budget has left, or None when nothing fits or the backend
    /// has no pool kernels.
    pub fn new(m: &Inkling2Model, d: &mut dev::Dev, need: &[usize], runs: Vec<(usize, usize)>) -> Option<Pool> {
        let (c, ne) = (&m.cfg, m.cfg.n_expert);
        if !d.parts[0].be.ink_pool() { return None; }
        let (b_res, f_res) = (64 << 20, 384 << 20);
        let bytes = |x: &[f32]| -> Vec<u8> { x.iter().flat_map(|v| v.to_le_bytes()).collect() };
        // Layout of a slot per layer, and the layers of each layout.
        let mut kinds: Vec<((GgmlType, usize, usize, GgmlType, usize, usize), usize, Vec<usize>)> = vec![];
        let mut layers: Vec<Option<PoolLayer>> = (0..c.n_layer).map(|_| None).collect();
        let mut host: Vec<Option<HostExperts>> = vec![None; c.n_layer];
        for &ip in need {
            let ml = &m.layers[ip];
            let (gu, dn) = (ml.gate_up.expert_host(0, ne), ml.down.expert_host(0, ne));
            let dn_off = gu.data.len().next_multiple_of(256);
            let slot = (dn_off + dn.data.len()).next_multiple_of(256);
            let key = (gu.ty, gu.rows, gu.cols, dn.ty, dn.rows, dn.cols);
            let g = kinds.iter().position(|k| k.0 == key).unwrap_or_else(|| { kinds.push((key, slot, vec![])); kinds.len() - 1 });
            kinds[g].2.push(ip);
            let gi = d.alloc_slab_on(0, &bytes(&ml.gate_inp), None, b_res, f_res)?;
            let bias = d.alloc_slab_on(0, &bytes(&ml.bias), None, b_res, f_res)?;
            let sc = |v: &[f32], d: &mut dev::Dev| if v.is_empty() { Some(None) } else { d.alloc_slab_on(0, &bytes(v), None, b_res, f_res).map(Some) };
            let (gu_scale, dn_scale) = (sc(&ml.gu_scale, d)?, sc(&ml.down_scale, d)?);
            let be = &*d.parts[0].be;
            let table = vec![-1i32; ne];
            let table_dev = be.alloc(4 * ne).ok()?;
            be.put(&table, &table_dev, 0).ok()?;
            // SAFETY: the mapped file outlives the model, which owns the device state and so the worker (joined on drop).
            let st = |x: Mat| unsafe { std::mem::transmute::<Mat<'_>, Mat<'static>>(x) };
            let he = HostExperts { gate_up: st(ml.gate_up), down: st(ml.down), gu_scale: ml.gu_scale.clone() };
            host[ip] = Some(he.clone());
            layers[ip] = Some(PoolLayer {
                pool: g, table_dev, table,
                gate_inp: GpuTensor::new(GgmlType::F32, ne, c.n_embd, Buf::new(gi.addr(), gi.len(), ())), _gi: gi,
                bias, gu_scale, dn_scale,
                gu: (gu.ty, gu.rows, gu.cols), dn: (dn.ty, dn.rows, dn.cols), dn_off, host: he,
            });
        }
        // Slots in proportion to each layout's experts, from what the budget and the card have left.
        let total: usize = kinds.iter().map(|k| k.1 * ne * k.2.len()).sum();
        let free = d.parts[0].be.mem_info().map_or(0, |m| m.0).saturating_sub(f_res + (64 << 20));
        let room = d.budget_left().saturating_sub(b_res).min(free);
        let frac = (room as f64 / total.max(1) as f64).min(1.0);
        let mut slots = vec![];
        for k in &kinds {
            let n = (frac * (ne * k.2.len()) as f64) as usize;
            let n = n.saturating_sub(STAGE).min(ne * k.2.len());
            let buf = d.alloc_room((n + STAGE) * k.1, b_res, f_res);
            let n = if buf.is_some() { n } else { 0 };
            slots.push(Slots { buf, bytes: k.1, owner: vec![None; n] });
        }
        if std::env::var_os("STRATA_DS_PROF").is_some() {
            let desc: Vec<String> = kinds.iter().zip(&slots).map(|(k, s)| format!("{:?} layers {:?}: {} of {} experts ({:.2} GB)",
                k.0.0, k.2, s.owner.len(), ne * k.2.len(), (s.owner.len() * k.1) as f64 / 1e9)).collect();
            eprintln!("inkling2: expert pool {}", desc.join("; "));
        }
        if slots.iter().all(|s| s.buf.is_none()) { return None; }
        let worker = InkWorker::new(&*d.parts[0].be, c, runs.clone(), host).inspect_err(|e| eprintln!("inkling2: no CPU expert worker ({e:#})")).ok()?;
        let profile = crate::expert_cache::cache_dir().join(format!("inkling2-profile-{}x{}-{}-{:?}.f32", c.n_layer, ne, c.n_embd, m.layers[0].gate_up.ty));
        let long: Vec<f32> = std::fs::read(&profile).ok().filter(|b| b.len() == 4 * c.n_layer * ne)
            .map(|b| b.chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect()).unwrap_or_default();
        let mut pool = Pool { slots, layers, counts: vec![0.0; c.n_layer * ne], long, profile, ne, worker, runs, stats: [0; 3], tokens: 0 };
        pool.fill(d).inspect_err(|e| eprintln!("inkling2: expert pool fill: {e:#}")).ok()?;
        Some(pool)
    }

    pub fn layer(&self, ip: usize) -> Option<&PoolLayer> { self.layers[ip].as_ref() }

    /// The first placement: the saved profile's most used experts, else in every layer the same share of experts
    /// for each time a token runs it (a core layer runs three times).
    fn fill(&mut self, d: &mut dev::Dev) -> Result<()> {
        let ne = self.ne;
        let per: Vec<f32> = (0..self.layers.len()).map(|ip| self.runs.iter().filter(|r| r.1 == ip).count() as f32).collect();
        let runs = |ip: usize| per[ip];
        let have = !self.long.is_empty();
        if !have { self.long = vec![0.0; self.counts.len()]; }
        for g in 0..self.slots.len() {
            let mut cand: Vec<(f32, usize, usize)> = self.layers.iter().enumerate()
                .filter(|(_, l)| l.as_ref().is_some_and(|l| l.pool == g))
                .flat_map(|(ip, _)| (0..ne).map(move |e| (ip, e)))
                .map(|(ip, e)| (if have { self.long[ip * ne + e] } else { runs(ip) / (e + 1) as f32 }, ip, e)).collect();
            cand.sort_by(|a, b| b.0.total_cmp(&a.0));
            for (slot, &(_, ip, e)) in cand.iter().take(self.slots[g].owner.len()).enumerate() { self.place(d, g, slot, ip, e)?; }
        }
        // Counts start from the profile at their steady size, so the first tokens move no more than later ones.
        let steady = 1.0 / (1.0 - KEEP);
        for (ip, l) in self.long.chunks(ne).enumerate() {
            let s: f32 = l.iter().sum();
            if s > 0.0 { for e in 0..ne { self.counts[ip * ne + e] = l[e] / s * steady * 8.0; } }
        }
        for pl in self.layers.iter().flatten() { d.parts[0].be.put(&pl.table, &pl.table_dev, 0)?; }
        Ok(())
    }

    /// Copy expert `e` of layer `ip` into slot `slot` of pool `g` and point the table (on the host) at it.
    fn place(&mut self, d: &mut dev::Dev, g: usize, slot: usize, ip: usize, e: usize) -> Result<()> {
        let ne = self.ne;
        let at = self.upload(&*d.parts[0].be, ip, e, slot)?;
        let pl = self.layers[ip].as_mut().unwrap();
        let (gu, dn) = (pl.host.gate_up.expert_host(e, ne), pl.host.down.expert_host(e, ne));
        // A prompt chunk's host lookups find the slot too (`Dev::expert`).
        d.pin(gu.data, gu.ty, gu.rows, gu.cols, gu.scale, at);
        d.pin(dn.data, dn.ty, dn.rows, dn.cols, dn.scale, at + pl.dn_off as u64);
        pl.table[e] = slot as i32;
        self.slots[g].owner[slot] = Some((ip, e));
        Ok(())
    }

    /// Copy expert `e` of layer `ip` into slot `slot` (a staging slot too) of its pool: its address.
    pub fn upload(&self, be: &dyn crate::backend::Backend, ip: usize, e: usize, slot: usize) -> Result<u64> {
        let pl = self.layers[ip].as_ref().unwrap();
        let sl = &self.slots[pl.pool];
        let at = sl.buf.as_ref().unwrap().addr() + (slot * sl.bytes) as u64;
        let (gu, dn) = (pl.host.gate_up.expert_host(e, self.ne), pl.host.down.expert_host(e, self.ne));
        be.htod(gu.data, &Buf::new(at, gu.data.len(), ()), 0)?;
        be.htod(dn.data, &Buf::new(at + pl.dn_off as u64, dn.data.len(), ()), 0)?;
        Ok(at)
    }

    /// Slot of expert `e` of layer `ip` (-1: not in the pool), and the first of the pool's `STAGE` staging slots.
    pub fn slot(&self, ip: usize, e: usize) -> i32 { self.layers[ip].as_ref().unwrap().table[e] }
    pub fn stage(&self, pl: &PoolLayer) -> usize { self.slots[pl.pool].owner.len() }
    /// Whether layer `ip`'s pool got VRAM (its staging slots exist, so a prompt chunk can run there).
    pub fn has_room(&self, ip: usize) -> bool { self.layers[ip].as_ref().is_some_and(|pl| self.slots[pl.pool].buf.is_some()) }

    /// Base address and slot size of layer `ip`'s pool.
    pub fn base(&self, pl: &PoolLayer) -> (u64, usize) { let s = &self.slots[pl.pool]; (s.buf.as_ref().map_or(0, |b| b.addr()), s.bytes) }

    /// After a token, with the queue idle: count its routing, copy the most used missed experts into free slots, and
    /// give a full pool's least used slot to a missed expert used more often (by more than one).
    pub fn adapt(&mut self, d: &mut dev::Dev) -> Result<()> {
        let ne = self.ne;
        self.tokens += 1;
        for v in &mut self.counts { *v *= KEEP; }
        for v in &mut self.long { *v *= LONG_KEEP; }
        let mut missed = vec![];
        for &(il, ip) in &self.runs {
            for (e, s) in self.worker.routing(il) {
                if e >= ne { continue; }
                self.counts[ip * ne + e] += 1.0;
                self.long[ip * ne + e] += 1.0;
                if s < 0 { missed.push((ip, e)); self.stats[1] += 1; } else { self.stats[0] += 1; }
            }
        }
        missed.sort_by(|a, b| self.counts[b.0 * ne + b.1].total_cmp(&self.counts[a.0 * ne + a.1]));
        missed.dedup();
        let (mut fills, mut swaps, mut changed) = (0, 0, vec![]);
        for (ip, e) in missed {
            let g = self.layers[ip].as_ref().unwrap().pool;
            if self.layers[ip].as_ref().unwrap().table[e] >= 0 { continue; }
            let slot = match self.slots[g].owner.iter().position(|o| o.is_none()) {
                Some(s) if fills < FILLS => { fills += 1; s }
                Some(_) => continue,
                None if swaps < SWAPS => {
                    let Some((s, (vip, ve))) = self.slots[g].owner.iter().enumerate().filter_map(|(s, o)| o.map(|o| (s, o)))
                        .min_by(|a, b| self.counts[a.1.0 * ne + a.1.1].total_cmp(&self.counts[b.1.0 * ne + b.1.1])) else { continue };
                    if self.counts[ip * ne + e] <= self.counts[vip * ne + ve] + 1.0 { continue; }
                    let vl = self.layers[vip].as_mut().unwrap();
                    vl.table[ve] = -1;
                    d.unpin(vl.host.gate_up.expert_host(ve, ne).data);
                    d.unpin(vl.host.down.expert_host(ve, ne).data);
                    changed.push(vip);
                    swaps += 1;
                    s
                }
                None => continue,
            };
            self.place(d, g, slot, ip, e)?;
            changed.push(ip);
            self.stats[2] += 1;
        }
        changed.sort_unstable();
        changed.dedup();
        for ip in changed {
            let pl = self.layers[ip].as_ref().unwrap();
            d.parts[0].be.put(&pl.table, &pl.table_dev, 0)?;
        }
        Ok(())
    }
}

impl Drop for Pool {
    /// Save the long counts: the next load places from them.
    fn drop(&mut self) {
        if self.tokens < 16 { return; }
        let bytes: Vec<u8> = self.long.iter().flat_map(|v| v.to_le_bytes()).collect();
        if let Some(dir) = self.profile.parent() { let _ = std::fs::create_dir_all(dir); }
        if let Err(e) = std::fs::write(&self.profile, bytes) { eprintln!("inkling2: expert profile {}: {e}", self.profile.display()); }
    }
}
