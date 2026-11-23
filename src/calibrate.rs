//! Universal calibration search.
//!
//! This module is the model- and backend-agnostic half of `rustingstrata calibrate`. It knows nothing about GGUFs,
//! GPUs or tokenizers: the caller supplies a `measure` closure that, given a set of knobs and a workload matrix,
//! returns throughput and time-to-first-token numbers. That makes the same search usable with every engine the
//! binary supports, including the ones that run on their own code path.
//!
//! The search is deliberately small-budget and noise-aware:
//!   * a deterministic low-discrepancy coarse sweep over the joint knob space,
//!   * multi-start coordinate descent from the best coarse points (escapes the local minima of a single greedy walk),
//!   * a final joint polish,
//!   * a hard sample budget with early stopping,
//!   * a weighted geometric-mean objective over prefill throughput, decode throughput and TTFT.

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
/// to keep a runner alive when only cheap knobs change.
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

/// Throughput and latency of one workload measurement. Higher is better for the rates, lower for `ttft_ms`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Measurement {
    pub prefill_tok_s: f64,
    pub decode_tok_s: f64,
    pub ttft_ms: f64,
    #[serde(default)]
    pub drafts_offered: usize,
    #[serde(default)]
    pub drafts_accepted: usize,
}

impl Measurement {
    fn valid(&self) -> bool {
        [self.prefill_tok_s, self.decode_tok_s, self.ttft_ms].iter().all(|v| v.is_finite() && *v > 0.0)
    }
}

/// Relative importance of the three signals. Weights are normalized to sum to one; the preset profiles are the
/// supported way to set them, `--weights` is the explicit one.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Objective {
    pub prefill: f64,
    pub decode: f64,
    pub ttft: f64,
}

impl Objective {
    pub fn balanced() -> Self { Self { prefill: 0.25, decode: 0.35, ttft: 0.40 } }
    pub fn interactive() -> Self { Self { prefill: 0.20, decode: 0.30, ttft: 0.50 } }
    pub fn batch() -> Self { Self { prefill: 0.50, decode: 0.40, ttft: 0.10 } }
    pub fn only(which: &str) -> Result<Self> {
        Ok(match which {
            "prefill" => Self { prefill: 1.0, decode: 0.0, ttft: 0.0 },
            "decode" => Self { prefill: 0.0, decode: 1.0, ttft: 0.0 },
            "ttft" => Self { prefill: 0.0, decode: 0.0, ttft: 1.0 },
            other => anyhow::bail!("unknown objective {other:?} (expected prefill, decode or ttft)"),
        })
    }
    fn normalized(&self) -> Self {
        let sum = self.prefill + self.decode + self.ttft;
        if sum <= 0.0 || !sum.is_finite() { return Self::balanced(); }
        Self { prefill: self.prefill / sum, decode: self.decode / sum, ttft: self.ttft / sum }
    }
    /// Score of one measurement relative to the baseline. A weighted geometric mean: a balanced gain is worth more
    /// than a large gain in one dimension paired with a regression in another, and the units cancel out.
    pub fn score(&self, m: &Measurement, base: &Measurement) -> f64 {
        let clamp = |v: f64| v.clamp(1e-3, 1e3);
        let pf = clamp(m.prefill_tok_s / base.prefill_tok_s);
        let dec = clamp(m.decode_tok_s / base.decode_tok_s);
        let ttft = clamp(base.ttft_ms / m.ttft_ms);
        let o = self.normalized();
        pf.powf(o.prefill) * dec.powf(o.decode) * ttft.powf(o.ttft)
    }
}

/// Score of a full workload matrix: geometric mean over workloads, so each prompt depth counts equally. Each depth
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
}

impl Default for SearchConfig {
    fn default() -> Self { Self { samples: 24, coarse: 8, starts: 3, patience: 2, seed: 1 } }
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

/// Run the search. `measure` is called once per sample with the whole workload matrix; it should apply the knobs,
/// warm up, take its own repetitions and return one measurement per workload.
pub fn search<F>(space: &[KnobSpec], workloads: &[Workload], cfg: &SearchConfig, obj: &Objective,
                baseline: Vec<Measurement>, mut measure: F) -> Result<SearchResult>
where F: FnMut(&Knobs, &[Workload]) -> Result<Vec<Measurement>>
{
    anyhow::ensure!(!space.is_empty() && !workloads.is_empty(), "calibration needs a knob space and at least one workload");
    let fallback = baseline.iter().find(|m| m.valid()).cloned().unwrap_or_else(|| Measurement { prefill_tok_s: 1.0, decode_tok_s: 1.0, ttft_ms: 1.0, ..Default::default() });
    let mut baselines = baseline.clone();
    while baselines.len() < workloads.len() { baselines.push(fallback.clone()); }
    let start = Instant::now();
    let mut trials: Vec<Trial> = Vec::new();
    let mut used = 0usize;

    let mut evaluate = |knobs: &Knobs, subset: &[Workload]| -> Result<Trial> {
        let t0 = Instant::now();
        let (measurements, error) = match measure(knobs, subset) {
            Ok(ms) if ms.iter().any(|m| m.valid()) => (ms, None),
            Ok(ms) => (ms, Some("no valid measurement (all rates were zero or non-finite)".to_string())),
            Err(e) => (vec![], Some(format!("{e:#}"))),
        };
        let score = if error.is_some() { 0.0 } else { aggregate_score(&measurements, &baselines, obj) };
        Ok(Trial { knobs: knobs.clone(), measurements, score, error, elapsed_ms: t0.elapsed().as_millis(), workloads: subset.len() })
    };

    // Stage A: coarse joint sweep, on the cheapest (first) workload. These scores are only used to pick starting
    // points for Stage B; they are never compared with full-matrix scores or allowed to win.
    let coarse = cfg.coarse.min(cfg.samples / 3).min(16).max(1);
    for i in 0..coarse {
        let k = quantize(space, &low_discrepancy(i, space.len(), cfg.seed));
        let trial = evaluate(&k, &workloads[..1])?;
        used += 1;
        trials.push(trial);
        if used >= cfg.samples { break }
    }

    // Stage B: multi-start coordinate descent on the full matrix.
    let mut seeds: Vec<(Knobs, f64)> = trials.iter().filter(|t| t.error.is_none())
        .map(|t| (t.knobs.clone(), t.score)).collect();
    seeds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    seeds.dedup_by(|a, b| a.0 == b.0);
    seeds.truncate(cfg.starts.max(1));
    if seeds.is_empty() { seeds.push((default_knobs(space), 0.0)); }

    let mut stop_reason = "budget".to_string();
    for (cur, _seed_score) in seeds.into_iter() {
        // Re-measure the seed on the full matrix before comparing it with full-matrix candidates. Otherwise a point
        // that only wins on the shortest prompt would start with an inflated score and reject real improvements.
        let mut cur_trial = if used < cfg.samples {
            let t = evaluate(&cur, workloads)?;
            used += 1;
            trials.push(t.clone());
            t
        } else { break };
        let mut stale = 0;
        for _ in 0..cfg.patience.max(1) + 2 {
            let mut improved = false;
            for s in space {
                let original = cur_trial.knobs.get(&s.name);
                let mut local_best = cur_trial.clone();
                for &v in &s.values {
                    if Some(v) == original { continue }
                    if used >= cfg.samples { break; }
                    let mut candidate = cur_trial.knobs.clone();
                    candidate.set(&s.name, v);
                    let t = evaluate(&candidate, workloads)?;
                    used += 1;
                    trials.push(t.clone());
                    if t.error.is_none() && t.score > local_best.score { local_best = t; }
                    if used >= cfg.samples { break; }
                }
                match local_best.error.clone() {
                    None if local_best.score > cur_trial.score => { cur_trial = local_best; improved = true; }
                    _ => {}
                }
                if used >= cfg.samples { break; }
            }
            if !improved { stale += 1; if stale >= cfg.patience.max(1) { stop_reason = "converged".into(); } }
            else { stale = 0; }
            if used >= cfg.samples { break; }
            if stop_reason == "converged" { break; }
        }
        if used >= cfg.samples { break; }
    }
    if used >= cfg.samples { stop_reason = "budget".into(); }

    // Stage C: one coordinate pass from the overall winner over the cheap knobs (reload knobs are handled by the
    // coarse sweep and the seeds, since each change there requires a fresh model load).
    let best_so_far = trials.iter().filter(|t| t.error.is_none() && t.workloads == workloads.len())
        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal)).cloned();
    let mut best = best_so_far.unwrap_or_else(|| Trial {
        knobs: default_knobs(space), measurements: vec![], score: 0.0,
        error: Some("no successful full-matrix measurement".into()), elapsed_ms: 0, workloads: 0,
    });
    if used < cfg.samples {
        for s in space.iter().filter(|s| !s.reload) {
            let original = best.knobs.get(&s.name);
            for &v in &s.values {
                if Some(v) == original || used >= cfg.samples { continue }
                let mut candidate = best.knobs.clone();
                candidate.set(&s.name, v);
                let t = evaluate(&candidate, workloads)?;
                used += 1;
                trials.push(t.clone());
                if t.error.is_none() && t.score > best.score { best = t; }
            }
        }
    }
    if used >= cfg.samples { stop_reason = "budget".into(); }
    Ok(SearchResult {
        best, baseline, trials, samples_used: used, stop_reason,
        elapsed_ms: start.elapsed().as_millis(), seed: cfg.seed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn space() -> Vec<KnobSpec> {
        vec![KnobSpec::new("chunk", [128.0, 512.0, 2048.0], false),
             KnobSpec::new("kv", [0.0, 1.0], true)]
    }
    fn workloads() -> Vec<Workload> { vec![Workload::new(128, 16, "prose"), Workload::new(512, 16, "prose")] }
    fn base() -> Measurement { Measurement { prefill_tok_s: 100.0, decode_tok_s: 20.0, ttft_ms: 100.0, ..Default::default() } }

    #[test]
    fn objective_is_a_geometric_mean_and_normalized() {
        let o = Objective::balanced();
        let same = Measurement { prefill_tok_s: 100.0, decode_tok_s: 20.0, ttft_ms: 100.0, ..Default::default() };
        assert!((o.score(&same, &base()) - 1.0).abs() < 1e-9);
        let better = Measurement { prefill_tok_s: 200.0, decode_tok_s: 10.0, ttft_ms: 50.0, ..Default::default() };
        // 2^0.25 * 0.5^0.35 * 2^0.40
        let expected = 2f64.powf(0.25) * 0.5f64.powf(0.35) * 2f64.powf(0.40);
        assert!((o.score(&better, &base()) - expected).abs() < 1e-9);
        assert!(o.score(&better, &base()) > 1.0, "balanced gains should win");
        let mixed = Measurement { prefill_tok_s: 1000.0, decode_tok_s: 2.0, ttft_ms: 500.0, ..Default::default() };
        assert!(o.score(&mixed, &base()) < 1.0, "a big regression must not be hidden by one gain");
    }

    #[test]
    fn objective_presets_and_unknown_name() {
        assert!((Objective::interactive().ttft - 0.5).abs() < 1e-9);
        assert!((Objective::batch().prefill - 0.5).abs() < 1e-9);
        assert!(Objective::only("decode").unwrap().decode == 1.0);
        assert!(Objective::only("nope").is_err());
    }

    #[test]
    fn low_discrepancy_is_deterministic_and_spread() {
        let a = low_discrepancy(7, 3, 42);
        let b = low_discrepancy(7, 3, 42);
        assert_eq!(a, b);
        assert!(a.iter().all(|v| (0.0..1.0).contains(v)));
        let pts: Vec<_> = (0..16).map(|i| low_discrepancy(i, 2, 1)).collect();
        let xs: std::collections::BTreeSet<_> = pts.iter().map(|p| format!("{:.2}", p[0])).collect();
        assert!(xs.len() >= 8, "coarse points should cover the first dimension: {xs:?}");
    }

    #[test]
    fn search_respects_the_sample_budget_and_finds_the_best_region() {
        // Synthetic landscape: chunk 512 with kv 1 is best. The closure never sees more calls than the budget.
        let mut calls = 0;
        let cfg = SearchConfig { samples: 12, coarse: 4, starts: 2, patience: 1, seed: 5 };
        let r = search(&space(), &workloads(), &cfg, &Objective::balanced(), vec![base(), base()], |k, _| {
            calls += 1;
            let chunk = k.i("chunk").unwrap();
            let kv = k.i("kv").unwrap();
            let boost = if chunk == 512 { 2.0 } else { 1.0 } * if kv == 1 { 1.5 } else { 1.0 };
            Ok(vec![Measurement { prefill_tok_s: 100.0 * boost, decode_tok_s: 20.0 * boost, ttft_ms: 100.0 / boost, ..Default::default() }])
        }).unwrap();
        assert!(calls <= cfg.samples, "used {calls} above budget {}", cfg.samples);
        assert_eq!(r.samples_used, calls);
        assert_eq!(r.best.knobs.i("chunk"), Some(512));
        assert_eq!(r.best.knobs.i("kv"), Some(1));
        assert!(r.best.score > 1.0);
        assert_eq!(r.stop_reason, "budget");
    }

    #[test]
    fn short_prompt_only_gains_cannot_win_the_search() {
        // kv=1 wins at 128 tokens but loses badly at 512. A coarse trial measured only on the first workload must
        // not become the final best point.
        let cfg = SearchConfig { samples: 14, coarse: 6, starts: 3, patience: 1, seed: 3 };
        let r = search(&space(), &workloads(), &cfg, &Objective::balanced(), vec![base(), base()], |k, subset| {
            let kv = k.i("kv").unwrap();
            Ok(subset.iter().map(|w| {
                let boost = if kv == 1 { if w.pp == 128 { 2.0 } else { 0.25 } } else { 1.0 };
                Measurement { prefill_tok_s: 100.0 * boost, decode_tok_s: 20.0 * boost,
                              ttft_ms: 100.0 / boost, ..Default::default() }
            }).collect())
        }).unwrap();
        assert_eq!(r.best.knobs.i("kv"), Some(0));
        assert!(r.best.measurements.len() == 2, "the winner must be measured on the full matrix");
        assert!(r.trials.iter().filter(|t| t.workloads == 1).all(|t| t.score < r.best.score + 1e-9 || t.knobs != r.best.knobs));
    }

    #[test]
    fn failed_measurements_are_recorded_and_never_win() {
        let cfg = SearchConfig { samples: 8, coarse: 4, starts: 1, patience: 1, seed: 2 };
        let r = search(&space(), &workloads(), &cfg, &Objective::balanced(), vec![base(), base()], |k, _| {
            if k.i("kv") == Some(1) { anyhow::bail!("out of memory") }
            Ok(vec![base(), base()])
        }).unwrap();
        assert_eq!(r.best.knobs.i("kv"), Some(0));
        assert!(r.trials.iter().any(|t| t.error.as_deref().is_some_and(|e| e.contains("out of memory"))));
    }

    #[test]
    fn aggregate_score_ignores_invalid_workloads() {
        let bad = Measurement::default();
        let bases = [base(), base()];
        assert!((aggregate_score(&[base(), bad.clone()], &bases, &Objective::balanced()) - 1.0).abs() < 1e-9);
        assert_eq!(aggregate_score(&[bad], &bases, &Objective::balanced()), 0.0);
    }
}
