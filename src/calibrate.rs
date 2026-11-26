//! Universal calibration search.
//!
//! This module is the model- and backend-agnostic half of `rustingstrata calibrate`. It knows nothing about GGUFs,
//! GPUs or tokenizers: the caller supplies a `measure` closure that, given a set of knobs and a workload matrix,
//! returns throughput numbers. That makes the same search usable with every engine the binary supports.
//!
//! The search is deliberately small-budget and noise-aware:
//!   * a deterministic low-discrepancy coarse sweep over the joint knob space,
//!   * multi-start coordinate descent from the best coarse points (escapes the local minima of a single greedy walk),
//!   * a final joint polish,
//!   * a hard sample budget with early stopping,
//!   * a baseline measured with the settings a user gets today, not the first point of the search space.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::Instant;

/// A point in the knob space. Values are numeric: integer knobs are rounded by the caller when they are applied.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Knobs(pub BTreeMap<String, f64>);

impl Knobs {
    pub fn get(&self, name: &str) -> Option<f64> { self.0.get(name).copied() }
    pub fn i(&self, name: &str) -> Option<usize> { self.get(name).map(|v| v.round().max(0.0) as usize) }
    pub fn set(&mut self, name: &str, value: f64) { self.0.insert(name.to_string(), value); }
    /// A short, stable label used in progress lines and in the JSON report.
    pub fn label(&self) -> String {
        self.0.iter().map(|(k, v)| format!("{k}={}", if v.fract() == 0.0 { format!("{v:.0}") } else { format!("{v:.2}") }))
            .collect::<Vec<_>>().join(" ")
    }
}

/// One discrete knob. `reload` marks knobs whose change needs a fresh model load; the measurement closure can use it
/// to avoid reloading on cheap changes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KnobSpec {
    pub name: String,
    pub values: Vec<f64>,
    #[serde(default)]
    pub reload: bool,
}

impl KnobSpec {
    pub fn new(name: &str, values: impl IntoIterator<Item = f64>, reload: bool) -> Self {
        Self { name: name.into(), values: values.into_iter().collect(), reload }
    }
}

/// A prompt/decode pair to benchmark. `task` is a label the measurement closure interprets ("prose", "coding",
/// "user:PATH"); it is passed through so the report records which corpus produced the numbers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Workload {
    pub pp: usize,
    pub tg: usize,
    pub task: String,
}

impl Workload {
    pub fn new(pp: usize, tg: usize, task: &str) -> Self { Self { pp, tg, task: task.into() } }
    pub fn label(&self) -> String { format!("{}pp/{}tg/{}", self.pp, self.tg, self.task) }
}

/// Throughput of one workload measurement. Higher is better.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Measurement {
    pub prefill_tok_s: f64,
    pub decode_tok_s: f64,
    #[serde(default)]
    pub drafts_offered: usize,
    #[serde(default)]
    pub drafts_accepted: usize,
}

impl Measurement {
    pub fn valid(&self) -> bool {
        [self.prefill_tok_s, self.decode_tok_s].iter().all(|v| v.is_finite() && *v > 0.0)
    }
}

/// Relative importance of prefill throughput and decode throughput. Each is divided by its own baseline, so the
/// units cancel. A weighted geometric mean (rather than an arithmetic one) cannot hide a regression in one signal
/// behind a large gain in the other. Time to first token is deliberately not a separate signal: for greedy decoding
/// it is the prefill time, so giving it a weight would just reweight prefill.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Objective {
    pub prefill: f64,
    pub decode: f64,
}

impl Objective {
    /// Balanced default: prompt handling and reply generation count equally.
    pub fn balanced() -> Self { Self { prefill: 0.5, decode: 0.5 } }
    fn normalized(&self) -> Self {
        let sum = self.prefill + self.decode;
        if sum <= 0.0 || !sum.is_finite() { return Self::balanced(); }
        Self { prefill: self.prefill / sum, decode: self.decode / sum }
    }
    /// Score of one measurement relative to its baseline.
    pub fn score(&self, m: &Measurement, base: &Measurement) -> f64 {
        let clamp = |v: f64| v.clamp(1e-3, 1e3);
        let pf = clamp(m.prefill_tok_s / base.prefill_tok_s);
        let dec = clamp(m.decode_tok_s / base.decode_tok_s);
        let o = self.normalized();
        pf.powf(o.prefill) * dec.powf(o.decode)
    }
}

/// Score of a full workload matrix: geometric mean over workloads, so each prompt depth counts equally. Every depth
/// is compared with its own baseline, so the ratio means what it says at every prompt length.
pub fn aggregate_score(ms: &[Measurement], bases: &[Measurement], obj: &Objective) -> f64 {
    let fallback = bases.first().cloned().unwrap_or_default();
    let mut acc = 0.0;
    let mut n = 0;
    for (i, m) in ms.iter().enumerate() {
        if !m.valid() { continue; }
        let base = bases.get(i).unwrap_or(&fallback);
        acc += obj.score(m, base).ln();
        n += 1;
    }
    if n == 0 { 0.0 } else { (acc / n as f64).exp() }
}

/// One evaluated candidate. `error` is set (and `score` is zero) when a measurement failed, e.g. an OOM at a large
/// chunk: failed points stay in the trace so the report explains why a region of the space was skipped.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Trial {
    pub knobs: Knobs,
    pub measurements: Vec<Measurement>,
    pub score: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub elapsed_ms: u128,
    /// How many workloads were measured. Coarse trials use the first workload only; they must not be compared with
    /// full-matrix scores, nor can they win the search.
    pub workloads: usize,
}

/// Budget and stopping settings.
#[derive(Clone, Debug)]
pub struct SearchConfig {
    /// Hard cap on measurement closure calls. Reaching it always stops the search.
    pub samples: usize,
    /// Number of coarse low-discrepancy points (capped at a third of the budget).
    pub coarse: usize,
    /// Multi-start count for coordinate descent.
    pub starts: usize,
    /// Stop after this many full refinement rounds without an improvement.
    pub patience: usize,
    /// Deterministic seed (also used for the low-discrepancy sequence).
    pub seed: u64,
    /// Wall-clock end: no sample starts after it, and the best so far is the result (stop reason "time").
    pub deadline: Option<Instant>,
}

impl Default for SearchConfig {
    fn default() -> Self { Self { samples: 24, coarse: 8, starts: 3, patience: 2, seed: 1, deadline: None } }
}

/// Result of a search, including the full trace for the JSON report.
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResult {
    pub best: Trial,
    pub baseline: Vec<Measurement>,
    pub trials: Vec<Trial>,
    pub samples_used: usize,
    pub stop_reason: String,
    pub elapsed_ms: u128,
    pub seed: u64,
}

/// Radical inverse in a prime base: a deterministic, well-spread point set in `[0,1)^dims`. No dependencies, same
/// points on every run for a given seed, so a calibration can be reproduced from its recorded seed.
pub fn low_discrepancy(index: usize, dims: usize, seed: u64) -> Vec<f64> {
    const PRIMES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    (0..dims).map(|d| {
        let base = PRIMES.get(d).copied().unwrap_or(37);
        let mut n = index as u64 + seed.wrapping_mul(base);
        let mut f = 1.0 / base as f64;
        let mut out = 0.0;
        while n > 0 {
            out += f * (n % base) as f64;
            n /= base;
            f /= base as f64;
        }
        out
    }).collect()
}

/// Map a unit point to knob values by nearest candidate (a small jitter-free quantization of the unit cube).
fn quantize(space: &[KnobSpec], unit: &[f64]) -> Knobs {
    let mut k = Knobs::default();
    for (spec, u) in space.iter().zip(unit) {
        let i = (u.clamp(0.0, 0.999999) * spec.values.len() as f64) as usize;
        k.set(&spec.name, spec.values[i.min(spec.values.len() - 1)]);
    }
    k
}

fn default_knobs(space: &[KnobSpec]) -> Knobs {
    let mut k = Knobs::default();
    for s in space { if let Some(v) = s.values.first() { k.set(&s.name, *v); } }
    k
}

/// The gain over the baseline a candidate needs to be written (2%, about the spread of one bench run).
pub const MIN_GAIN: f64 = 1.02;

/// Run the search. `measure` is called once per sample with the workload matrix; it should apply the knobs, take its
/// own repetitions and return one measurement per workload. `baseline` has one measurement per workload, taken with
/// `baseline_knobs` (the settings a user gets without a calibration file). The baseline competes in the search: a
/// candidate slower than today's defaults must not be written as the winner.
pub fn search<F>(space: &[KnobSpec], workloads: &[Workload], cfg: &SearchConfig, obj: &Objective,
                baseline_knobs: &Knobs, baseline: Vec<Measurement>, mut measure: F) -> Result<SearchResult>
where F: FnMut(&Knobs, &[Workload]) -> Result<Vec<Measurement>>
{
    anyhow::ensure!(!space.is_empty() && !workloads.is_empty(), "calibration needs a knob space and at least one workload");
    let fallback = baseline.iter().find(|m| m.valid()).cloned().unwrap_or_else(|| Measurement { prefill_tok_s: 1.0, decode_tok_s: 1.0, ..Default::default() });
    let mut baselines = baseline.clone();
    while baselines.len() < workloads.len() { baselines.push(fallback.clone()); }
    let start = Instant::now();
    let mut trials: Vec<Trial> = Vec::new();
    // The baseline is a real trial, measured outside the sample budget. Its ratios against itself are 1.0, so the
    // search can never pick a slower point unless the baseline measurement itself failed. It enters at MIN_GAIN: a
    // candidate must beat today's defaults by more than run-to-run noise, or noise picks the winner.
    trials.push(Trial {
        knobs: baseline_knobs.clone(),
        measurements: baseline.clone(),
        score: aggregate_score(&baseline, &baselines, obj) * MIN_GAIN,
        error: if baseline.iter().any(|m| m.valid()) { None } else { Some("baseline measurement failed".to_string()) },
        elapsed_ms: 0,
        workloads: workloads.len(),
    });
    let mut used = 0usize;
    let late = || cfg.deadline.is_some_and(|d| Instant::now() >= d);
    let full = |used: usize| used >= cfg.samples || late();
    let mut n = 0usize;

    let mut evaluate = |knobs: &Knobs, subset: &[Workload]| -> Result<Trial> {
        let t0 = Instant::now();
        let (measurements, error) = match measure(knobs, subset) {
            Ok(ms) if ms.iter().any(|m| m.valid()) => (ms, None),
            Ok(ms) => (ms, Some("no valid measurement (all rates were zero or non-finite)".to_string())),
            Err(e) => (vec![], Some(format!("{e:#}"))),
        };
        let score = if error.is_some() { 0.0 } else { aggregate_score(&measurements, &baselines, obj) };
        n += 1;
        let what = error.as_deref().map_or_else(|| format!("score {score:.3}"), |e| format!("failed: {}", e.lines().next().unwrap_or("")));
        crate::say!("sample {n}/{}: {} on {} workload(s), {what} ({:.0} s)", cfg.samples, knobs.label(), subset.len(), t0.elapsed().as_secs_f64());
        Ok(Trial { knobs: knobs.clone(), measurements, score, error, elapsed_ms: t0.elapsed().as_millis(), workloads: subset.len() })
    };

    // One bench child per distinct (knob point, workload set): repeats cost a full model load and consume budget
    // without adding information. `seen` is shared by every stage, including all starts.
    let mut seen: std::collections::HashSet<(String, usize)> = std::collections::HashSet::new();
    // The baseline was measured outside the budget: never spend a sample repeating it.
    seen.insert((baseline_knobs.label(), workloads.len()));
    let mut eval_unique = |knobs: &Knobs, subset: &[Workload]| -> Result<Option<Trial>> {
        if !seen.insert((knobs.label(), subset.len())) { return Ok(None); }
        evaluate(knobs, subset).map(Some)
    };

    // Stage A: coarse joint sweep, on the cheapest (first) workload. These scores are only used to pick starting
    // points for Stage B; they are never compared with full-matrix scores or allowed to win.
    let coarse = cfg.coarse.min(cfg.samples / 3).min(16).max(1);
    for i in 0..coarse {
        if full(used) { break }
        let k = quantize(space, &low_discrepancy(i, space.len(), cfg.seed));
        if let Some(trial) = eval_unique(&k, &workloads[..1])? {
            used += 1;
            trials.push(trial);
        }
        if full(used) { break }
    }

    // Stage B: multi-start coordinate descent on the full matrix.
    let mut seeds: Vec<(Knobs, f64)> = trials.iter().filter(|t| t.error.is_none())
        .map(|t| (t.knobs.clone(), t.score)).collect();
    seeds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    seeds.dedup_by(|a, b| a.0 == b.0);
    seeds.truncate(cfg.starts.max(1));
    if seeds.is_empty() { seeds.push((default_knobs(space), 0.0)); }

    let mut converged_all = true;
    for (cur, _seed_score) in seeds.into_iter() {
        let mut stop_reason = "budget".to_string();
        // Re-measure the seed on the full matrix before comparing it with full-matrix candidates. Otherwise a point
        // that only wins on the shortest prompt would start with an inflated score and reject real improvements.
        let mut cur_trial = if full(used) { break } else if let Some(t) = eval_unique(&cur, workloads)? {
            used += 1;
            trials.push(t.clone());
            t
        } else {
            trials.iter().find(|t| t.workloads == workloads.len() && t.knobs == cur).cloned()
                .unwrap_or_else(|| Trial { knobs: cur.clone(), measurements: vec![], score: 0.0, error: None,
                                          elapsed_ms: 0, workloads: workloads.len() })
        };
        let mut stale = 0;
        for _ in 0..cfg.patience.max(1) + 2 {
            let mut improved = false;
            for s in space {
                let original = cur_trial.knobs.get(&s.name);
                let mut local_best = cur_trial.clone();
                for &v in &s.values {
                    if Some(v) == original || full(used) { continue }
                    let mut candidate = cur_trial.knobs.clone();
                    candidate.set(&s.name, v);
                    if let Some(t) = eval_unique(&candidate, workloads)? {
                        used += 1;
                        trials.push(t.clone());
                        if t.error.is_none() && t.score > local_best.score { local_best = t; }
                    }
                    if full(used) { break; }
                }
                match local_best.error.clone() {
                    None if local_best.score > cur_trial.score => { cur_trial = local_best; improved = true; }
                    _ => {}
                }
                if full(used) { break; }
            }
            if !improved { stale += 1; if stale >= cfg.patience.max(1) { stop_reason = "converged".into(); } }
            else { stale = 0; }
            if full(used) || stop_reason == "converged" { break; }
        }
        if full(used) { converged_all = false; break; }
        if stop_reason != "converged" { converged_all = false; }
    }

    // Stage C: one coordinate pass from the overall winner over the cheap knobs.
    let best_so_far = trials.iter().filter(|t| t.error.is_none() && t.workloads == workloads.len())
        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal)).cloned();
    let mut best = best_so_far.unwrap_or_else(|| Trial {
        knobs: default_knobs(space), measurements: vec![], score: 0.0,
        error: Some("no successful full-matrix measurement".into()), elapsed_ms: 0, workloads: 0,
    });
    if !full(used) {
        for s in space.iter().filter(|s| !s.reload) {
            let original = best.knobs.get(&s.name);
            for &v in &s.values {
                if Some(v) == original || full(used) { continue }
                let mut candidate = best.knobs.clone();
                candidate.set(&s.name, v);
                if let Some(t) = eval_unique(&candidate, workloads)? {
                    used += 1;
                    trials.push(t.clone());
                    if t.error.is_none() && t.score > best.score { best = t; }
                }
            }
        }
    }
    let stop_reason = if late() { "time" } else if full(used) || !converged_all { "budget" } else { "converged" }.to_string();
    Ok(SearchResult {
        best, baseline, trials, samples_used: used, stop_reason,
        elapsed_ms: start.elapsed().as_millis(), seed: cfg.seed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn space() -> Vec<KnobSpec> {
        vec![KnobSpec::new("chunk", [256.0, 1024.0, 4096.0], false),
             KnobSpec::new("mtp", [0.0, 1.0, 2.0], false)]
    }
    fn workloads() -> Vec<Workload> { vec![Workload::new(128, 16, "prose"), Workload::new(512, 16, "prose")] }
    fn base() -> Measurement { Measurement { prefill_tok_s: 100.0, decode_tok_s: 20.0, ..Default::default() } }
    fn baseline_knobs() -> Knobs { let mut k = Knobs::default(); k.set("chunk", 256.0); k.set("mtp", 0.0); k }

    #[test]
    fn objective_is_a_geometric_mean_of_two_signals() {
        let o = Objective::balanced();
        assert!((o.score(&base(), &base()) - 1.0).abs() < 1e-9);
        let better = Measurement { prefill_tok_s: 200.0, decode_tok_s: 10.0, ..Default::default() };
        assert!((o.score(&better, &base()) - 1.0).abs() < 1e-9, "2x prefill and 0.5x decode cancel");
        let gain = Measurement { prefill_tok_s: 200.0, decode_tok_s: 40.0, ..Default::default() };
        assert!((o.score(&gain, &base()) - 2.0).abs() < 1e-9);
        let mixed = Measurement { prefill_tok_s: 1000.0, decode_tok_s: 1.0, ..Default::default() };
        assert!(o.score(&mixed, &base()) < 1.0, "a big regression must not be hidden by one gain");
    }

    #[test]
    fn low_discrepancy_is_deterministic_and_spread() {
        assert_eq!(low_discrepancy(7, 3, 42), low_discrepancy(7, 3, 42));
        let pts: Vec<_> = (0..16).map(|i| low_discrepancy(i, 2, 1)).collect();
        let xs: std::collections::BTreeSet<_> = pts.iter().map(|p| format!("{:.2}", p[0])).collect();
        assert!(xs.len() >= 8, "coarse points should cover the first dimension: {xs:?}");
    }

    #[test]
    fn search_respects_the_sample_budget_and_finds_the_best_region() {
        let mut calls = 0;
        let cfg = SearchConfig { samples: 12, coarse: 4, starts: 2, patience: 1, seed: 5, deadline: None };
        let r = search(&space(), &workloads(), &cfg, &Objective::balanced(), &baseline_knobs(), vec![base(), base()], |k, _| {
            calls += 1;
            let boost = if k.i("chunk") == Some(1024) { 2.0 } else { 1.0 } * if k.i("mtp") == Some(1) { 1.5 } else { 1.0 };
            Ok(vec![Measurement { prefill_tok_s: 100.0 * boost, decode_tok_s: 20.0 * boost, ..Default::default() }])
        }).unwrap();
        assert!(calls <= cfg.samples, "used {calls} above budget {}", cfg.samples);
        assert_eq!(r.samples_used, calls);
        assert_eq!(r.best.knobs.i("chunk"), Some(1024));
        assert_eq!(r.best.knobs.i("mtp"), Some(1));
        assert!(r.best.score > 1.0);
    }

    #[test]
    fn every_start_gets_a_full_descent_not_one_round() {
        // A flat landscape never converges: each of the three starts must run several full rounds. With the old
        // stop_reason leak, starts 2 and 3 got a single round after start 1 converged.
        // A bigger space gives each start real room to move: scores improve with both knob values, so no start
        // converges immediately. (Measured points are memoized, so call-count-based landscapes are not useful.)
        let big: Vec<KnobSpec> = vec![
            KnobSpec::new("chunk", (1..=8).map(|i| i as f64 * 100.0), false),
            KnobSpec::new("mtp", (0..8).map(|i| i as f64), false),
        ];
        let cfg = SearchConfig { samples: 40, coarse: 3, starts: 3, patience: 2, seed: 9, deadline: None };
        let r = search(&big, &workloads(), &cfg, &Objective::balanced(), &baseline_knobs(), vec![base(), base()], |k, _| {
            let boost = 1.0 + k.i("chunk").unwrap() as f64 / 1000.0 + k.i("mtp").unwrap() as f64 / 100.0;
            Ok(vec![Measurement { prefill_tok_s: 100.0 * boost, decode_tok_s: 20.0 * boost, ..Default::default() }])
        }).unwrap();
        assert!(r.samples_used > 25, "each start should get several rounds, not one: {}", r.samples_used);
    }

    #[test]
    fn short_prompt_only_gains_cannot_win_the_search() {
        let cfg = SearchConfig { samples: 14, coarse: 6, starts: 3, patience: 1, seed: 3, deadline: None };
        let r = search(&space(), &workloads(), &cfg, &Objective::balanced(), &baseline_knobs(), vec![base(), base()], |k, subset| {
            let mtp = k.i("mtp").unwrap();
            Ok(subset.iter().map(|w| {
                let boost = if mtp == 1 { if w.pp == 128 { 2.0 } else { 0.25 } } else { 1.0 };
                Measurement { prefill_tok_s: 100.0 * boost, decode_tok_s: 20.0 * boost, ..Default::default() }
            }).collect())
        }).unwrap();
        assert_ne!(r.best.knobs.i("mtp"), Some(1), "the short-prompt-only gain must not win");
        assert_eq!(r.best.measurements.len(), 2);
    }

    #[test]
    fn failed_measurements_are_recorded_and_never_win() {
        let cfg = SearchConfig { samples: 8, coarse: 4, starts: 1, patience: 1, seed: 2, deadline: None };
        let r = search(&space(), &workloads(), &cfg, &Objective::balanced(), &baseline_knobs(), vec![base(), base()], |k, _| {
            if k.i("mtp") == Some(1) { anyhow::bail!("out of memory") }
            Ok(vec![base(), base()])
        }).unwrap();
        assert_ne!(r.best.knobs.i("mtp"), Some(1));
        assert!(r.trials.iter().any(|t| t.error.as_deref().is_some_and(|e| e.contains("out of memory"))));
    }

    #[test]
    fn an_empty_baseline_still_allows_the_search_to_run() {
        let cfg = SearchConfig { samples: 10, coarse: 3, starts: 1, patience: 1, seed: 6, deadline: None };
        let r = search(&space(), &workloads(), &cfg, &Objective::balanced(), &baseline_knobs(), vec![], |_, _| {
            Ok(vec![Measurement { prefill_tok_s: 100.0, decode_tok_s: 20.0, ..Default::default() }])
        }).unwrap();
        assert!(r.best.error.is_none());
        assert!(r.samples_used > 0);
    }

    #[test]
    fn the_same_knob_point_is_never_measured_twice() {
        let cfg = SearchConfig { samples: 60, coarse: 6, starts: 3, patience: 2, seed: 7, deadline: None };
        let r = search(&space(), &workloads(), &cfg, &Objective::balanced(), &baseline_knobs(), vec![base(), base()], |k, _| {
            let boost = 1.0 + k.i("chunk").unwrap() as f64 / 10000.0 + k.i("mtp").unwrap() as f64 / 10.0;
            Ok(vec![Measurement { prefill_tok_s: 100.0 * boost, decode_tok_s: 20.0 * boost, ..Default::default() }])
        }).unwrap();
        let keys: std::collections::HashSet<_> = r.trials.iter()
            .map(|t| (t.knobs.label(), t.workloads)).collect();
        assert_eq!(keys.len(), r.trials.len(), "a repeated sample never adds information");
        assert!(r.trials.len() > cfg.coarse, "the search should still do real refinement");
    }

    #[test]
    fn every_start_converging_is_recorded_as_converged() {
        // A flat landscape: no move ever improves, so all starts stop on patience before the budget is used.
        let cfg = SearchConfig { samples: 60, coarse: 3, starts: 2, patience: 2, seed: 4, deadline: None };
        let r = search(&space(), &workloads(), &cfg, &Objective::balanced(), &baseline_knobs(), vec![base(), base()], |_, _| {
            Ok(vec![base(), base()])
        }).unwrap();
        assert!(r.samples_used < cfg.samples, "should stop early: {}", r.samples_used);
        assert_eq!(r.stop_reason, "converged");
    }

    #[test]
    fn deadline_stops_the_search() {
        let space = vec![KnobSpec::new("chunk", [256.0, 512.0, 1024.0], true)];
        let ws = vec![Workload::new(128, 16, "prose")];
        let mut calls = 0;
        let cfg = SearchConfig { samples: 20, deadline: Some(Instant::now()), ..Default::default() };
        let r = search(&space, &ws, &cfg, &Objective::balanced(), &Knobs::default(), vec![], |_, _| {
            calls += 1;
            Ok(vec![Measurement { prefill_tok_s: 1.0, decode_tok_s: 1.0, ..Default::default() }])
        }).unwrap();
        assert_eq!((calls, r.samples_used, r.stop_reason.as_str()), (0, 0, "time"));
    }

    #[test]
    fn a_slower_candidate_can_never_beat_the_baseline() {
        let cfg = SearchConfig { samples: 20, coarse: 6, starts: 3, patience: 2, seed: 11, deadline: None };
        let base_knobs = baseline_knobs();
        let r = search(&space(), &workloads(), &cfg, &Objective::balanced(), &base_knobs,
                       vec![base(), base()], |_, _| {
            Ok(vec![Measurement { prefill_tok_s: 50.0, decode_tok_s: 10.0, ..Default::default() }])
        }).unwrap();
        assert_eq!(r.best.knobs, base_knobs, "a slower point must not be written");
        assert!((r.best.score - MIN_GAIN).abs() < 1e-9, "the baseline wins at score MIN_GAIN");
        assert!(r.trials.iter().any(|t| t.knobs == base_knobs && t.score == MIN_GAIN));
    }

    #[test]
    fn aggregate_score_ignores_invalid_workloads() {
        let bad = Measurement::default();
        let bases = [base(), base()];
        assert!((aggregate_score(&[base(), bad.clone()], &bases, &Objective::balanced()) - 1.0).abs() < 1e-9);
        assert_eq!(aggregate_score(&[bad], &bases, &Objective::balanced()), 0.0);
    }
}
