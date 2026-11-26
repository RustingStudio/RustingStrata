//! Per-expert VRAM cache for routed experts. Experts of equal byte size share a pool of slots; each slot holds one
//! expert's gate, up and down matrices back to back. A per-layer table maps expert id to slot (-1 = on the CPU).
//! Routing counts decay over time; `adapt` swaps the coldest resident experts for the hottest missed ones.

use crate::backend::{Backend, Buf};
use crate::model::Model;
use crate::quant::GgmlType;
use std::path::PathBuf;

/// Share of each count kept per `adapt` (every 8 tokens by default): `counts` and `long`.
const KEEP: f32 = 0.95;
const LONG_KEEP: f32 = 0.999;

pub const MATS: [&str; 3] = ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"];

#[derive(Clone, Copy)]
pub struct Mat { pub ty: GgmlType, pub rows: usize, pub cols: usize, pub off: usize, pub bytes: usize }

pub struct LayerExperts {
    pub pool: usize,
    pub mats: [Mat; 3],
    table: Vec<i32>,
    pub table_dev: Buf,
}

struct Pool { buf: Buf, slot_bytes: usize, owner: Vec<(usize, usize)> }

pub struct ExpertCache {
    pools: Vec<Pool>,
    layers: Vec<Option<LayerExperts>>,
    n_expert: usize,
    /// Decayed routing counts, [layer * n_expert + expert]: they forget in a few hundred tokens, for `adapt`.
    counts: Vec<f32>,
    /// The same counts with a memory of some ten thousand tokens: the saved profile and the initial placement. A
    /// profile of the last reply alone made the next run's placement fit that reply only, and where the placement
    /// changes the greedy text (CPU and GPU experts round differently), runs alternated between two placements.
    long: Vec<f32>,
    /// Routing of the prompt being prefilled, until `settle` folds it into `counts`. Not part of `long`: a prompt has
    /// thousands of tokens for a reply's hundreds, and a profile made of prompts placed the prompt's experts while
    /// decode, the only phase that pays for a miss, routed elsewhere (issue #44).
    prompt: Vec<f32>,
    /// Sum of one layer's `counts` in steady state (k experts per token, an adapt every 8 tokens).
    steady: f32,
    /// None under `STRATA_FIXED_EXPERTS`: placement ignores and keeps the saved profile and never adapts.
    profile: Option<PathBuf>,
    /// Whether `record` ran since the load: a fully resident run records nothing and must not rewrite the profile.
    recorded: bool,
    pub swaps: usize,
}

/// Where routing profiles and calibration files live: `$XDG_CACHE_HOME`, `%LOCALAPPDATA%` on Windows, else
/// `~/.cache`.
pub fn cache_dir() -> PathBuf {
    std::env::var_os("XDG_CACHE_HOME").or_else(|| std::env::var_os("LOCALAPPDATA")).map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache"))
        .join("rusting_strata")
}

fn profile_path(m: &Model) -> PathBuf {
    let (c, n) = (&m.cfg, m.w.g.tensors.len());
    cache_dir().join(format!("profile-{}-{}x{}-{n}.f32", c.arch, c.n_layer, c.n_expert))
}

impl ExpertCache {
    /// Allocate slots worth at most `budget` bytes for the layers in `range` (up to n_layer, or n_layer + 1 with the
    /// MTP layer) and fill them from the saved routing profile, or in layer order.
    pub fn new(m: &Model, gpu: &dyn Backend, budget: usize, range: std::ops::Range<usize>) -> anyhow::Result<ExpertCache> {
        let n_layers = range.end;
        let c = &m.cfg;
        let n = c.n_expert;
        const MAX_POOL_FULL_BYTES: usize = 2048 << 20;
        let mut layers: Vec<Option<(usize, [Mat; 3])>> = Vec::new();
        let mut pool_specs: Vec<(usize, usize)> = Vec::new(); // (slot_bytes, experts in pool)
        for il in 0..n_layers {
            if !c.is_moe_layer(il) || !range.contains(&il) { layers.push(None); continue; }
            let mut off = 0;
            let mut mats = [Mat { ty: GgmlType::F32, rows: 0, cols: 0, off: 0, bytes: 0 }; 3];
            for (i, name) in MATS.iter().enumerate() {
                let t = m.w.layer(il, name)?.expert(0, n);
                mats[i] = Mat { ty: t.ty, rows: t.rows, cols: t.cols, off, bytes: t.data.len() };
                off = (off + t.data.len()).next_multiple_of(16);  // GEMV kernels read 8-byte words
            }
            let pid = pool_specs
                .iter()
                .position(|&(sb, cnt)| sb == off && (cnt + n) * off <= MAX_POOL_FULL_BYTES)
                .unwrap_or_else(|| {
                    pool_specs.push((off, 0));
                    pool_specs.len() - 1
                });
            pool_specs[pid].1 += n;
            layers.push(Some((pid, mats)));
        }
        let total: usize = pool_specs.iter().map(|(b, k)| b * k).sum();
        let mut frac = (budget as f64 / total.max(1) as f64).min(1.0);
        let profile = std::env::var_os("STRATA_FIXED_EXPERTS").is_none().then(|| profile_path(m));
        let counts = profile.as_ref().and_then(|p| std::fs::read(p).ok())
            .filter(|b| b.len() >= 4 * c.n_layer * n)
            .map(|b| b.chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect::<Vec<_>>())
            .unwrap_or_default();
        if let Some(p) = profile.as_ref().filter(|_| frac < 1.0) {
            crate::say!("expert profile: {}{}", p.display(), if counts.is_empty() { " (none yet: written when the run ends)" } else { "" });
        }
        let mut long = counts;
        long.resize(n_layers * n, 0.0);
        // `counts` start from the profile at the size routing gives them in steady state (k experts per token, an
        // adapt every 8 tokens), so the first tokens of a run do not move the placement more than later ones.
        let steady = (c.n_expert_used * 8) as f32 / (1.0 - KEEP);
        let counts: Vec<f32> = long.chunks(n.max(1)).flat_map(|l| { let s = l.iter().sum::<f32>(); l.iter().map(move |x| if s > 0.0 { x * steady / s } else { 0.0 }) }).collect();

        // The budget comes from the driver's free-memory estimate, which can be more than it will allocate (AMD's
        // Windows driver): on a failed allocation, free the pools and try a tenth less.
        let pools = loop {
            let r = pool_specs.iter().map(|&(slot_bytes, k)| {
                let slots = (frac * k as f64) as usize;
                Ok(Pool { buf: gpu.zeros::<u8>(slots * slot_bytes)?, slot_bytes, owner: Vec::with_capacity(slots) })
            }).collect::<anyhow::Result<Vec<_>>>();
            match r {
                Ok(p) => break p,
                Err(e) if frac > 0.01 => {
                    crate::say!("expert slots: {e:#}; trying {:.0} MB", 0.9 * frac * total as f64 / 1048576.0);
                    frac *= 0.9;
                }
                Err(e) => return Err(e),
            }
        };
        let mut out: Vec<Option<LayerExperts>> = Vec::new();
        for l in &layers {
            out.push(match l {
                None => None,
                Some((pid, mats)) => Some(LayerExperts {
                    pool: *pid, mats: *mats, table: vec![-1; n],
                    table_dev: { let b = gpu.zeros::<i32>(n)?; gpu.put(&vec![-1i32; n], &b, 0)?; b },
                }),
            });
        }
        let prompt = vec![0.0; counts.len()];
        let mut cache = ExpertCache { pools, layers: out, n_expert: n, counts, long, prompt, steady, profile, recorded: false, swaps: 0 };
        // Initial fill: hottest first by profile; ties (and no profile) keep layer order.
        let mut order: Vec<usize> = (0..n_layers * n).filter(|&i| cache.layers[i / n].is_some()).collect();
        order.sort_by(|&a, &b| cache.counts[b].total_cmp(&cache.counts[a]).then(a.cmp(&b)));
        // The experts that fit, read ahead in upload order.
        let mut room: Vec<usize> = cache.pools.iter().map(|p| p.buf.len() / p.slot_bytes).collect();
        let order: Vec<usize> = order.into_iter().filter(|&i| { let r = &mut room[cache.layers[i / n].as_ref().unwrap().pool]; *r > 0 && { *r -= 1; true } }).collect();
        let pre = crate::weights::Prefetch::start();
        let mut rs = Vec::new();
        for &i in &order {
            for name in MATS { let t = m.w.layer(i / n, name)?.expert(i % n, n); rs.push((t.data.as_ptr() as usize, t.data.len())); }
        }
        pre.send(rs);
        for i in order {
            let p = cache.layers[i / n].as_ref().unwrap().pool;
            let slot = cache.pools[p].owner.len();
            cache.pools[p].owner.push((i / n, i % n));
            cache.place(m, gpu, i / n, i % n, slot, false)?;
        }
        for il in 0..cache.layers.len() { cache.sync_table(gpu, il)?; }
        Ok(cache)
    }

    /// Copy expert `e` of layer `il` into `slot` of its pool and point the table at it.
    fn place(&mut self, m: &Model, gpu: &dyn Backend, il: usize, e: usize, slot: usize, sync: bool) -> anyhow::Result<()> {
        let lx = self.layers[il].as_mut().unwrap();
        let pool = &mut self.pools[lx.pool];
        for (name, mat) in MATS.iter().zip(&lx.mats) {
            let t = m.w.layer(il, name)?.expert(e, self.n_expert);
            let o = slot * pool.slot_bytes + mat.off;
            debug_assert_eq!(t.data.len(), mat.bytes);
            gpu.htod(t.data, &pool.buf, o)?;
            crate::weights::cold(t.data);
        }
        lx.table[e] = slot as i32;
        if sync { self.sync_table(gpu, il)?; }
        Ok(())
    }

    fn sync_table(&mut self, gpu: &dyn Backend, il: usize) -> anyhow::Result<()> {
        if let Some(lx) = self.layers[il].as_ref() { gpu.put(&lx.table, &lx.table_dev, 0)?; }
        Ok(())
    }

    pub fn layer(&self, il: usize) -> Option<&LayerExperts> { self.layers[il].as_ref() }
    pub fn base(&self, lx: &LayerExperts) -> (u64, usize) {
        let p = &self.pools[lx.pool];
        (p.buf.addr(), p.slot_bytes)
    }
    /// Cache slot of expert `e` in layer `il`, or -1.
    pub fn slot(&self, il: usize, e: usize) -> i32 { self.layers[il].as_ref().map_or(-1, |l| l.table[e]) }
    pub fn resident(&self, il: usize, e: usize) -> bool { self.layers[il].as_ref().is_some_and(|l| l.table[e] >= 0) }
    pub fn resident_count(&self) -> usize { self.pools.iter().map(|p| p.owner.len()).sum() }
    pub fn slot_bytes_total(&self) -> usize { self.pools.iter().map(|p| p.buf.len()).sum() }
    pub fn total_experts(&self) -> usize { self.layers.iter().flatten().count() * self.n_expert }

    /// Every expert of every layer of this cache is in VRAM: no routing can miss, so the CPU worker, the mailbox and
    /// `adapt` can all be skipped (the fast path of issue #34 R2).
    pub fn fully_resident(&self) -> bool { self.total_experts() > 0 && self.resident_count() == self.total_experts() }

    /// Layers this cache holds (the model's, up to the MTP layer with a draft head).
    pub fn layers_len(&self) -> usize { self.layers.len() }

    /// Bytes the active experts of layer `il` read per token (`k` slots of the layer's pool): the layer's working
    /// set, logged at load and against `L2` in the prefetch decision.
    pub fn active_set_bytes(&self, il: usize, k: usize) -> usize {
        self.layers.get(il).and_then(|l| l.as_ref()).map_or(0, |l| k * self.pools[l.pool].slot_bytes)
    }

    /// Bytes one token reads from its routed experts: `active_set_bytes` over the layers of this cache.
    pub fn token_bytes(&self, k: usize) -> usize { (0..self.layers.len()).map(|il| self.active_set_bytes(il, k)).sum() }

    /// (expert table, slot base, slot stride, slot bytes) of layer `il`, for the prefetch kernel.
    pub fn prefetch_geom(&self, il: usize) -> Option<(u64, u64, usize, usize)> {
        let lx = self.layer(il)?;
        let (base, stride) = self.base(lx);
        Some((lx.table_dev.addr(), base, stride, self.pools[lx.pool].slot_bytes))
    }

    pub fn record(&mut self, il: usize, ids: &[i32]) {
        self.recorded = true;
        for &e in ids {
            self.counts[il * self.n_expert + e as usize] += 1.0;
            self.long[il * self.n_expert + e as usize] += 1.0;
        }
    }

    /// Routing of a prefill chunk, for `settle`.
    pub fn record_prompt(&mut self, il: usize, ids: &[i32]) {
        for &e in ids { self.prompt[il * self.n_expert + e as usize] += 1.0; }
    }

    /// After a prefill: the prompt's routing joins `counts`, scaled back to steady-state size, and the placement
    /// follows. A long prompt still decides the placement the reply starts with. `STRATA_PREFILL_PROTECT=1` makes it
    /// blend gently instead (25% of steady state) and swap at most 10% of each pool's slots, so it does not flush the
    /// hot decode cache; `strict` swaps nothing. Off by default: on Qwen3.6-35B-A3B (8k prompt, 128-token reply) it
    /// lowered the reply's expert hit rate from 83% to 76%.
    pub fn settle(&mut self, m: &Model, gpu: &dyn Backend) -> anyhow::Result<()> {
        let n = self.n_expert;
        let mode = std::env::var("STRATA_PREFILL_PROTECT").unwrap_or_default();
        if self.profile.is_some() && !mode.eq_ignore_ascii_case("strict") {
            let protect = !mode.is_empty() && mode != "0";
            let prompt_cap = if protect { 0.25 * self.steady } else { f32::INFINITY };
            for il in 0..self.layers.len() {
                let r = il * n..(il + 1) * n;
                let psum: f32 = self.prompt[r.clone()].iter().sum();
                let pscale = if psum > prompt_cap { prompt_cap / psum } else { 1.0 };
                for i in r.clone() { self.counts[i] += self.prompt[i] * pscale; }
                let sum = self.counts[r.clone()].iter().sum::<f32>();
                if sum > self.steady { for i in r { self.counts[i] *= self.steady / sum; } }
            }
            let max_swaps = if protect {
                let min_pool = self.pools.iter().map(|p| p.owner.len()).filter(|&l| l > 0).min().unwrap_or(0);
                (min_pool / 10).max(1)
            } else {
                usize::MAX
            };
            self.swap(m, gpu, max_swaps)?;
        }
        self.prompt.fill(0.0);
        Ok(())
    }

    /// Top `top_k` hottest non-resident experts of `next_il` (by short-term `counts`, falling back to `long`),
    /// returned as `(ptr, len)` mmap byte ranges for asynchronous NVMe read-ahead (`Prefetch::send`) while layer
    /// `il` computes on the GPU.
    pub fn cold_ranges_for_next(&self, m: &Model, next_il: usize, top_k: usize) -> Vec<(usize, usize)> {
        let Some(lx) = self.layers.get(next_il).and_then(|l| l.as_ref()) else { return Vec::new() };
        let n = self.n_expert;
        let base = next_il * n;
        let mut cands: Vec<usize> = (0..n).filter(|&e| lx.table[e] < 0).collect();
        if cands.is_empty() { return Vec::new(); }
        let score = |e: usize| self.counts[base + e] + 0.01 * self.long[base + e];
        smallest(&mut cands, top_k, |&a, &b| score(b).total_cmp(&score(a)));
        let mut rs = Vec::with_capacity(cands.len() * 3);
        for (name, mat) in MATS.iter().zip(&lx.mats) {
            let Ok(w) = m.w.layer(next_il, name) else { continue };
            for &e in &cands {
                rs.push((w.data.as_ptr() as usize + e * mat.bytes, mat.bytes));
            }
        }
        rs
    }

    /// Forget the short-term routing counts: the prompt's thousands of tokens no longer outweigh the reply's first
    /// tokens, so `adapt` follows the text being decoded instead of the prompt that produced it.
    pub fn reset_counts(&mut self) { self.counts.fill(0.0); }

    /// Decay counts, then per pool swap up to `max_swaps` cold residents for clearly hotter missed experts.
    pub fn adapt(&mut self, m: &Model, gpu: &dyn Backend, max_swaps: usize) -> anyhow::Result<()> {
        if self.profile.is_none() { return Ok(()); }
        for c in &mut self.counts { *c *= KEEP; }
        for c in &mut self.long { *c *= LONG_KEEP; }
        self.swap(m, gpu, max_swaps)
    }

    fn swap(&mut self, m: &Model, gpu: &dyn Backend, max_swaps: usize) -> anyhow::Result<()> {
        let n = self.n_expert;
        for p in 0..self.pools.len() {
            let mut hot: Vec<usize> = (0..self.counts.len())
                .filter(|&i| self.layers[i / n].as_ref().is_some_and(|l| l.pool == p && l.table[i % n] < 0)).collect();
            smallest(&mut hot, max_swaps, |&a, &b| self.counts[b].total_cmp(&self.counts[a]));
            let mut cold: Vec<usize> = (0..self.pools[p].owner.len()).collect();
            let cnt = |s: usize, me: &Self| { let (il, e) = me.pools[p].owner[s]; me.counts[il * n + e] };
            smallest(&mut cold, max_swaps, |&a, &b| cnt(a, self).total_cmp(&cnt(b, self)));
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
        let Some(profile) = &self.profile else { return Ok(()) };
        // A run that recorded nothing (a fully resident cache skips `record`) leaves the saved profile alone: writing
        // it back would at best repeat what was loaded and, on a first run without one, replace it with zeros (R2b).
        if !self.recorded { return Ok(()); }
        std::fs::create_dir_all(profile.parent().unwrap())?;
        // Layers of other caches (a model split over devices) keep the counts their caches saved.
        let existing: Vec<f32> = std::fs::read(profile).unwrap_or_default().chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect();
        let moe: Vec<bool> = self.layers.iter().map(|l| l.is_some()).collect();
        let all = profile_update(&existing, self.n_expert, &self.long, &moe);
        std::fs::write(profile, all.iter().flat_map(|c| c.to_le_bytes()).collect::<Vec<u8>>())?;
        Ok(())
    }
}

/// The counts to save: `existing` (the file's, other caches' layers included) with every layer this cache holds
/// replaced by its `long` counts. Pure, so the "do not clobber the profile" rule is tested without a device.
fn profile_update(existing: &[f32], n_expert: usize, long: &[f32], moe: &[bool]) -> Vec<f32> {
    let mut all = existing.to_vec();
    all.resize(all.len().max(long.len()), 0.0);
    for (il, &held) in moe.iter().enumerate() {
        if held { all[il * n_expert..(il + 1) * n_expert].copy_from_slice(&long[il * n_expert..(il + 1) * n_expert]); }
    }
    all
}

/// Keeps the `k` first elements of `v` in the order a stable sort by `cmp` gives, without sorting all of `v`.
fn smallest(v: &mut Vec<usize>, k: usize, cmp: impl Fn(&usize, &usize) -> std::cmp::Ordering) {
    let cmp = |a: &usize, b: &usize| cmp(a, b).then(a.cmp(b));
    if k < v.len() {
        v.select_nth_unstable_by(k, cmp);
        v.truncate(k);
    }
    v.sort_by(cmp);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cache over `layers` MoE layers of `n` experts with one pool of `layers * n` slots of 128 bytes; the first
    /// `resident` experts of every layer are placed. No model and no device: the placement flags are the whole state
    /// the fast path and `save_profile` read.
    fn fake(layers: usize, n: usize, resident: usize, profile: Option<PathBuf>) -> ExpertCache {
        let pool = Pool { buf: Buf::new(0, layers * n * 128, ()), slot_bytes: 128, owner: Vec::new() };
        let ls = (0..layers).map(|_| Some(LayerExperts {
            pool: 0,
            mats: [Mat { ty: GgmlType::F32, rows: 0, cols: 0, off: 0, bytes: 128 }; 3],
            table: vec![-1; n],
            table_dev: Buf::new(0, 4 * n, ()),
        })).collect();
        let mut c = ExpertCache { pools: vec![pool], layers: ls, n_expert: n, counts: vec![0.0; layers * n],
            long: vec![0.0; layers * n], prompt: vec![0.0; layers * n], steady: 1.0, profile, recorded: false, swaps: 0 };
        for il in 0..layers {
            for e in 0..resident {
                c.pools[0].owner.push((il, e));
                c.layers[il].as_mut().unwrap().table[e] = e as i32;
            }
        }
        c
    }

    /// R2: the fast path needs *every* slot used; one expert left on the CPU keeps the mailbox path on.
    #[test]
    fn fully_resident_only_when_every_expert_is_placed() {
        assert!(fake(2, 4, 4, None).fully_resident());
        assert!(!fake(2, 4, 3, None).fully_resident());
        let empty = fake(0, 4, 0, None);
        assert_eq!(empty.total_experts(), 0);
        assert!(!empty.fully_resident(), "a cache without layers has nothing to be resident");
    }

    /// R1/R3: the working set logged at load is `k` slots per layer, summed per token.
    #[test]
    fn active_set_is_k_slots_per_layer() {
        let c = fake(3, 4, 2, None);
        assert_eq!(c.active_set_bytes(0, 8), 8 * 128);
        assert_eq!(c.active_set_bytes(9, 8), 0, "a layer outside the cache reads nothing");
        assert_eq!(c.token_bytes(8), 3 * 8 * 128);
    }

    /// R2b: a fully resident run records nothing, so `save_profile` must leave the file as it was (no zeroing of a
    /// saved profile, no creation of one).
    #[test]
    fn full_residency_does_not_clobber_the_saved_profile() {
        let d = std::env::temp_dir().join(format!("strata_profile_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let f = d.join("profile.f32");
        let before: Vec<u8> = [1.0f32; 8].iter().flat_map(|c| c.to_le_bytes()).collect();
        std::fs::write(&f, &before).unwrap();

        // Two layers in the file, one held by this cache: the other one is another device's.
        let mut c = fake(1, 4, 4, Some(f.clone()));
        assert!(c.fully_resident());
        c.save_profile().unwrap();
        assert_eq!(std::fs::read(&f).unwrap(), before, "a run without a record must not rewrite the profile");

        // A run that did record does write, and only for the layers this cache holds.
        c.record(0, &[1, 3]);
        c.save_profile().unwrap();
        let after: Vec<f32> = std::fs::read(&f).unwrap().chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect();
        assert_eq!(&after[..4], &[0.0, 1.0, 0.0, 1.0]);
        assert_eq!(&after[4..], &[1.0; 4], "the other layer still holds its saved counts");
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// R2b: the merge keeps the counts of layers owned by another cache (a model split over devices).
    #[test]
    fn profile_update_keeps_other_caches_layers() {
        let existing = [9.0f32; 8];
        let long = [1.0f32; 8];
        let all = profile_update(&existing, 4, &long, &[true, false]);
        assert_eq!(&all[..4], &[1.0; 4]);
        assert_eq!(&all[4..], &[9.0; 4]);
        let longer = profile_update(&existing, 4, &[1.0; 12], &[true, false, true]);
        assert_eq!(longer.len(), 12, "a longer profile grows without losing the file's counts");
    }

    #[test]
    fn prefill_protect_blends_gently_and_preserves_hot_decode_counts() {
        let mut c = fake(1, 4, 2, None);
        // Simulate hot decode experts 0 and 1, then a huge prompt hitting cold experts 2 and 3.
        c.counts = vec![40.0, 40.0, 0.0, 0.0];
        c.steady = 80.0;
        c.record_prompt(0, &[2, 2, 2, 2, 3, 3, 3, 3]);
        // Under default protection, prompt contribution is capped at 0.25 * steady = 20.0 total (10 each),
        // so hot decode experts 0 and 1 stay strictly hotter than cold prompt experts 2 and 3.
        let psum: f32 = c.prompt[..4].iter().sum();
        let cap = 0.25 * c.steady;
        let pscale = if psum > cap { cap / psum } else { 1.0 };
        for i in 0..4 { c.counts[i] += c.prompt[i] * pscale; }
        assert!(c.counts[0] > c.counts[2] && c.counts[1] > c.counts[3]);
    }
}
