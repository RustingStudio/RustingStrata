//! Per-expert VRAM cache for routed experts. Experts of equal byte size share a pool of slots; each slot holds one
//! expert's gate, up and down matrices back to back. A per-layer table maps expert id to slot (-1 = on the CPU).
//! Routing counts decay over time; `adapt` swaps the coldest resident experts for the hottest missed ones.

use crate::cuda::Gpu;
use crate::model::Model;
use crate::quant::GgmlType;
use cudarc::driver::CudaSlice;
use std::collections::BTreeMap;
use std::path::PathBuf;

pub const MATS: [&str; 3] = ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"];

#[derive(Clone, Copy)]
pub struct Mat { pub ty: GgmlType, pub rows: usize, pub cols: usize, pub off: usize, pub bytes: usize }

pub struct LayerExperts {
    pub pool: usize,
    pub mats: [Mat; 3],
    table: Vec<i32>,
    pub table_dev: CudaSlice<i32>,
}

struct Pool { buf: CudaSlice<u8>, slot_bytes: usize, owner: Vec<(usize, usize)> }

pub struct ExpertCache {
    pools: Vec<Pool>,
    layers: Vec<Option<LayerExperts>>,
    n_expert: usize,
    /// Decayed routing counts, [layer * n_expert + expert].
    counts: Vec<f32>,
    profile: PathBuf,
    pub swaps: usize,
}

fn profile_path(m: &Model) -> PathBuf {
    let dir = std::env::var_os("XDG_CACHE_HOME").map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache"));
    let (c, n) = (&m.cfg, m.w.g.tensors.len());
    dir.join("rusting_strata").join(format!("profile-{}-{}x{}-{n}.f32", c.arch, c.n_layer, c.n_expert))
}

impl ExpertCache {
    /// Allocate slots worth at most `budget` bytes and fill them from the saved routing profile, or in layer order.
    pub fn new(m: &Model, gpu: &Gpu, budget: usize) -> anyhow::Result<ExpertCache> {
        let c = &m.cfg;
        let n = c.n_expert;
        let mut layers: Vec<Option<(usize, [Mat; 3])>> = Vec::new();
        let mut classes = BTreeMap::<usize, usize>::new(); // slot bytes -> experts of that size
        for il in 0..c.n_layer {
            if !c.is_moe() { layers.push(None); continue; }
            let mut off = 0;
            let mut mats = [Mat { ty: GgmlType::F32, rows: 0, cols: 0, off: 0, bytes: 0 }; 3];
            for (i, name) in MATS.iter().enumerate() {
                let t = m.w.layer(il, name)?.expert(0, n);
                mats[i] = Mat { ty: t.ty, rows: t.rows, cols: t.cols, off, bytes: t.data.len() };
                off += t.data.len();
            }
            *classes.entry(off).or_default() += n;
            layers.push(Some((off, mats)));
        }
        let total: usize = classes.iter().map(|(b, k)| b * k).sum();
        let frac = (budget as f64 / total.max(1) as f64).min(1.0);
        let profile = profile_path(m);
        let counts = std::fs::read(&profile).ok()
            .filter(|b| b.len() == 4 * c.n_layer * n)
            .map(|b| b.chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect())
            .unwrap_or_else(|| vec![0f32; c.n_layer * n]);

        let mut pools = Vec::new();
        let mut out: Vec<Option<LayerExperts>> = Vec::new();
        for (&slot_bytes, &k) in &classes {
            let slots = (frac * k as f64) as usize;
            pools.push(Pool { buf: gpu.zeros::<u8>((slots * slot_bytes).max(1))?, slot_bytes, owner: Vec::with_capacity(slots) });
        }
        for l in &layers {
            out.push(match l {
                None => None,
                Some((sb, mats)) => Some(LayerExperts {
                    pool: classes.keys().position(|k| k == sb).unwrap(), mats: *mats, table: vec![-1; n],
                    table_dev: gpu.stream.clone_htod(&vec![-1i32; n])?,
                }),
            });
        }
        let mut cache = ExpertCache { pools, layers: out, n_expert: n, counts, profile, swaps: 0 };
        // Initial fill: hottest first by profile; ties (and no profile) keep layer order.
        let mut order: Vec<usize> = (0..c.n_layer * n).filter(|&i| cache.layers[i / n].is_some()).collect();
        order.sort_by(|&a, &b| cache.counts[b].total_cmp(&cache.counts[a]).then(a.cmp(&b)));
        for i in order {
            let p = cache.layers[i / n].as_ref().unwrap().pool;
            let pool = &cache.pools[p];
            if pool.owner.len() * pool.slot_bytes + pool.slot_bytes > pool.buf.len() { continue; }
            let slot = pool.owner.len();
            cache.pools[p].owner.push((i / n, i % n));
            cache.place(m, gpu, i / n, i % n, slot, false)?;
        }
        for il in 0..cache.layers.len() { cache.sync_table(gpu, il)?; }
        Ok(cache)
    }

    /// Copy expert `e` of layer `il` into `slot` of its pool and point the table at it.
    fn place(&mut self, m: &Model, gpu: &Gpu, il: usize, e: usize, slot: usize, sync: bool) -> anyhow::Result<()> {
        let lx = self.layers[il].as_mut().unwrap();
        let pool = &mut self.pools[lx.pool];
        for (name, mat) in MATS.iter().zip(&lx.mats) {
            let t = m.w.layer(il, name)?.expert(e, self.n_expert);
            let o = slot * pool.slot_bytes + mat.off;
            gpu.stream.memcpy_htod(t.data, &mut pool.buf.slice_mut(o..o + mat.bytes))?;
        }
        lx.table[e] = slot as i32;
        if sync { self.sync_table(gpu, il)?; }
        Ok(())
    }

    fn sync_table(&mut self, gpu: &Gpu, il: usize) -> anyhow::Result<()> {
        if let Some(lx) = self.layers[il].as_mut() { gpu.stream.memcpy_htod(&lx.table, &mut lx.table_dev)?; }
        Ok(())
    }

    pub fn layer(&self, il: usize) -> Option<&LayerExperts> { self.layers[il].as_ref() }
    pub fn base(&self, gpu: &Gpu, lx: &LayerExperts) -> (u64, usize) {
        let p = &self.pools[lx.pool];
        (gpu.ptr(&p.buf), p.slot_bytes)
    }
    /// Cache slot of expert `e` in layer `il`, or -1.
    pub fn slot(&self, il: usize, e: usize) -> i32 { self.layers[il].as_ref().map_or(-1, |l| l.table[e]) }
    pub fn resident(&self, il: usize, e: usize) -> bool { self.layers[il].as_ref().is_some_and(|l| l.table[e] >= 0) }
    pub fn resident_count(&self) -> usize { self.pools.iter().map(|p| p.owner.len()).sum() }
    pub fn slot_bytes_total(&self) -> usize { self.pools.iter().map(|p| p.buf.len()).sum() }
    pub fn total_experts(&self) -> usize { self.layers.iter().flatten().count() * self.n_expert }

    pub fn record(&mut self, il: usize, ids: &[i32]) {
        for &e in ids { self.counts[il * self.n_expert + e as usize] += 1.0; }
    }

    /// Decay counts, then per pool swap up to `max_swaps` cold residents for clearly hotter missed experts.
    pub fn adapt(&mut self, m: &Model, gpu: &Gpu, max_swaps: usize) -> anyhow::Result<()> {
        for c in &mut self.counts { *c *= 0.95; }
        let n = self.n_expert;
        for p in 0..self.pools.len() {
            let mut hot: Vec<usize> = (0..self.counts.len())
                .filter(|&i| self.layers[i / n].as_ref().is_some_and(|l| l.pool == p && l.table[i % n] < 0)).collect();
            hot.sort_by(|&a, &b| self.counts[b].total_cmp(&self.counts[a]));
            let mut cold: Vec<usize> = (0..self.pools[p].owner.len()).collect();
            let cnt = |s: usize, me: &Self| { let (il, e) = me.pools[p].owner[s]; me.counts[il * n + e] };
            cold.sort_by(|&a, &b| cnt(a, self).total_cmp(&cnt(b, self)));
            let mut touched = Vec::new();
            for (&h, &slot) in hot.iter().zip(&cold).take(max_swaps) {
                if self.counts[h] < 1.5 * cnt(slot, self) + 2.0 { break; }
                let (oil, oe) = self.pools[p].owner[slot];
                self.layers[oil].as_mut().unwrap().table[oe] = -1;
                self.pools[p].owner[slot] = (h / n, h % n);
                self.place(m, gpu, h / n, h % n, slot, false)?;
                touched.extend([oil, h / n]);
                self.swaps += 1;
            }
            touched.sort();
            touched.dedup();
            for il in touched { self.sync_table(gpu, il)?; }
        }
        Ok(())
    }

    pub fn save_profile(&self) -> anyhow::Result<()> {
        std::fs::create_dir_all(self.profile.parent().unwrap())?;
        std::fs::write(&self.profile, self.counts.iter().flat_map(|c| c.to_le_bytes()).collect::<Vec<u8>>())?;
        Ok(())
    }
}
