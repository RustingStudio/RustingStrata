//! Context-lookup speculative decoding (port of `llama.cpp` `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`,
//! `ngram-mod`, and `ngram-cache` with static/dynamic lookup caches).
//!
//! All drafting runs on the CPU in microseconds with zero VRAM overhead, producing draft token sequences that
//! `GpuModel::ngram_spec_step` verifies in a single batched forward pass on the GPU.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

pub const LLAMA_TOKEN_NULL: i32 = -1;
pub const LLAMA_NGRAM_MIN: usize = 1;
pub const LLAMA_NGRAM_MAX: usize = 4;
pub const LLAMA_NGRAM_STATIC: usize = 2;

const LCG_FACTOR: u32 = 2654435761;
const COMMON_NGRAM_MAX_VALUES: usize = 4;
const COMMON_NGRAM_HASH_MAP_SIZE: usize = 262_144;
const COMMON_NGRAM_MAX_VALUE_COUNT: u16 = 16_380;

const DRAFT_MIN_SAMPLE_SIZE_LAX: [i32; LLAMA_NGRAM_MAX] = [2, 2, 1, 1];
const DRAFT_MIN_PERCENT_LAX: [i32; LLAMA_NGRAM_MAX] = [66, 50, 50, 50];
const DRAFT_MIN_SAMPLE_SIZE_STRICT: [i32; LLAMA_NGRAM_MAX] = [4, 3, 2, 2];
const DRAFT_MIN_PERCENT_STRICT: [i32; LLAMA_NGRAM_MAX] = [75, 66, 66, 66];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecType {
    None,
    Suffix,
    NgramSimple,
    NgramMapK,
    NgramMapK4v,
    NgramMod,
    NgramCache,
}

impl SpecType {
    pub fn parse_list(s: &str) -> anyhow::Result<Vec<SpecType>> {
        let mut out = Vec::new();
        for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let ty = match part {
                "none" => return Ok(vec![]),
                "suffix" => SpecType::Suffix,
                "ngram-simple" => SpecType::NgramSimple,
                "ngram-map-k" => SpecType::NgramMapK,
                "ngram-map-k4v" => SpecType::NgramMapK4v,
                "ngram-mod" => SpecType::NgramMod,
                "ngram-cache" => SpecType::NgramCache,
                other => anyhow::bail!(
                    "unknown --spec-type {other:?} (expected: none, suffix, ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache)"
                ),
            };
            if !out.contains(&ty) {
                out.push(ty);
            }
        }
        Ok(out)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            SpecType::None => "none",
            SpecType::Suffix => "suffix",
            SpecType::NgramSimple => "ngram-simple",
            SpecType::NgramMapK => "ngram-map-k",
            SpecType::NgramMapK4v => "ngram-map-k4v",
            SpecType::NgramMod => "ngram-mod",
            SpecType::NgramCache => "ngram-cache",
        }
    }
}

#[derive(Debug, Clone)]
pub struct NgramConfig {
    pub types: Vec<SpecType>,
    pub draft_max: usize,
    pub size_n: usize,
    pub size_m: usize,
    pub min_hits: u16,
    pub mod_n_match: usize,
    pub mod_n_min: usize,
    pub lookup_cache_static: Option<PathBuf>,
    pub lookup_cache_dynamic: Option<PathBuf>,
    /// Target model vocabulary size (0 = unchecked). Tokens >= `n_vocab` are filtered from caches and drafts.
    pub n_vocab: usize,
}

impl Default for NgramConfig {
    fn default() -> Self {
        Self {
            types: Vec::new(),
            draft_max: 16,
            size_n: 12,
            size_m: 48,
            min_hits: 1,
            mod_n_match: 24,
            mod_n_min: 4,
            lookup_cache_static: None,
            lookup_cache_dynamic: None,
            n_vocab: 0,
        }
    }
}

impl NgramConfig {
    pub fn is_enabled(&self) -> bool {
        !self.types.is_empty() && self.draft_max > 0
    }
}

// ---------------------------------------------------------------------------
// 1. N-gram Simple (`common_ngram_simple_draft` in `llama.cpp`)
// ---------------------------------------------------------------------------

pub fn ngram_simple_draft(tokens: &[u32], sampled: u32, size_n: usize, size_m: usize, draft_max: usize) -> Vec<u32> {
    ngram_simple_draft_eos(tokens, sampled, size_n, size_m, draft_max, &[])
}

pub fn ngram_simple_draft_eos(
    tokens: &[u32],
    sampled: u32,
    size_n: usize,
    size_m: usize,
    draft_max: usize,
    eos: &[u32],
) -> Vec<u32> {
    let cur_len = tokens.len();
    let n = size_n.max(1);
    let m = size_m.max(1);
    if cur_len <= n + 1 || draft_max == 0 {
        return Vec::new();
    }
    let pat_prefix = &tokens[cur_len + 1 - n..cur_len];
    let first_tok = if n == 1 { sampled } else { pat_prefix[0] };

    // Search backwards from `cur_len - n - 1` down to `1` (matching llama.cpp's `j > 0` loop).
    // If a candidate window crosses an EOS/turn token (such as `<|im_end|>` at the end of a user turn),
    // remember it as a fallback and keep searching for an earlier clean match.
    let Some(start_j) = cur_len.checked_sub(n + 1) else {
        return Vec::new();
    };
    let mut match_pos = 0usize;
    let mut fallback_pos = 0usize;
    for j in (1..=start_j).rev() {
        if tokens[j + n - 1] == sampled && tokens[j] == first_tok && tokens[j..j + n - 1] == *pat_prefix {
            let avail = cur_len.saturating_sub(j + n);
            let take = m.min(avail).min(draft_max);
            if !eos.is_empty() && tokens[j + n..j + n + take].iter().any(|t| eos.contains(t)) {
                if fallback_pos == 0 {
                    fallback_pos = j;
                }
                continue;
            }
            match_pos = j;
            break;
        }
    }
    if match_pos == 0 {
        match_pos = fallback_pos;
    }
    if match_pos == 0 {
        return Vec::new();
    }
    let avail = cur_len.saturating_sub(match_pos + n);
    let copy_max = m.min(avail);
    if copy_max < n.min(draft_max) {
        return Vec::new();
    }
    let take = copy_max.min(draft_max);
    tokens[match_pos + n..match_pos + n + take].to_vec()
}

/// Multi-order rolling suffix matcher (n = 16 -> 4, plus unanimous-repeat n = 3) returning `(draft, matched_n)`.
pub fn suffix_draft_with_order(tokens: &[u32], sampled: u32, max_draft: usize, eos: &[u32]) -> (Vec<u32>, usize) {
    let cur_len = tokens.len();
    if max_draft == 0 || cur_len < 6 { return (Vec::new(), 0); }
    for n in [16usize, 12, 8, 6, 4] {
        if cur_len <= n + 1 { continue; }
        let d = ngram_simple_draft_eos(tokens, sampled, n, 48, max_draft, eos);
        if d.len() >= n.min(4).min(max_draft) {
            return (d, n);
        }
    }
    if cur_len > 10 && max_draft >= 4 {
        let p = &tokens[cur_len - 2..cur_len];
        let mut hits = [0usize; 2];
        let mut nh = 0usize;
        for j in (1..=cur_len - 7).rev() {
            if tokens[j] == p[0] && tokens[j + 1] == p[1] && tokens[j + 2] == sampled {
                hits[nh] = j + 3;
                nh += 1;
                if nh == 2 { break; }
            }
        }
        if nh == 2 && hits[0] + 4 <= cur_len && hits[1] + 4 <= cur_len {
            let avail = (cur_len - hits[0]).min(cur_len - hits[1]).min(max_draft);
            let same = (0..avail).take_while(|&i| tokens[hits[0] + i] == tokens[hits[1] + i]).count();
            if same >= 4 {
                return (tokens[hits[0]..hits[0] + same].to_vec(), 3);
            }
        }
    }
    (Vec::new(), 0)
}

// ---------------------------------------------------------------------------
// 2. N-gram Map (`common_ngram_map` in `llama.cpp`: `ngram-map-k` & `ngram-map-k4v`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default)]
struct NgramMapValue {
    value_idx: usize,
    value_num: u16,
    n_accepted: i16,
}

#[derive(Debug, Clone)]
struct NgramMapKey {
    key_idx: usize,
    stat_idx: usize,
    key_num: u16,
    values: [NgramMapValue; COMMON_NGRAM_MAX_VALUES],
}

#[derive(Debug, Clone)]
pub struct NgramMap {
    size_key: usize,
    size_value: usize,
    key_only: bool,
    min_hits: u16,
    keys: Vec<NgramMapKey>,
    size_last_begin: usize,
    last_draft_created: bool,
    last_draft_key_idx: usize,
    last_draft_value_idx: usize,
    idx_last_check: usize,
    key_map: Vec<u32>,
    key_map_last_idx: usize,
}

fn ngram_map_hash(tokens: &[u32]) -> u32 {
    let mut hash = 0u32;
    for &t in tokens {
        hash = hash.wrapping_mul(LCG_FACTOR).wrapping_add(t);
    }
    hash
}

impl NgramMap {
    pub fn new(size_key: usize, size_value: usize, key_only: bool, min_hits: u16) -> Self {
        Self {
            size_key: size_key.max(1),
            size_value: size_value.max(1),
            key_only,
            min_hits: min_hits.max(1),
            keys: Vec::new(),
            size_last_begin: 0,
            last_draft_created: false,
            last_draft_key_idx: 0,
            last_draft_value_idx: 0,
            idx_last_check: 0,
            key_map: vec![0; COMMON_NGRAM_HASH_MAP_SIZE],
            key_map_last_idx: 0,
        }
    }

    pub fn begin(&mut self, tokens: &[u32]) {
        let size_begin = tokens.len();
        let mut idx_begin_cleanup = self.size_last_begin;
        if idx_begin_cleanup > size_begin {
            idx_begin_cleanup = size_begin.saturating_sub(self.size_key + self.size_value);
        }
        if !self.key_map.is_empty() && size_begin < self.idx_last_check {
            for entry in &mut self.key_map {
                if *entry != 0 && (*entry as usize) >= idx_begin_cleanup {
                    *entry = 0;
                }
            }
            self.key_map_last_idx = idx_begin_cleanup.saturating_sub(1);
        }
        if size_begin < self.idx_last_check && !self.keys.is_empty() {
            let key_only = self.key_only;
            self.keys.retain_mut(|key| {
                if key.key_idx >= idx_begin_cleanup {
                    return false;
                }
                if key_only {
                    return true;
                }
                for j in (0..COMMON_NGRAM_MAX_VALUES).rev() {
                    if key.values[j].value_idx != 0 && key.values[j].value_idx >= idx_begin_cleanup {
                        for k in j..COMMON_NGRAM_MAX_VALUES - 1 {
                            key.values[k] = key.values[k + 1];
                        }
                        key.values[COMMON_NGRAM_MAX_VALUES - 1] = NgramMapValue::default();
                    }
                }
                key.values[0].value_idx != 0
            });
        }
        self.idx_last_check = size_begin;
        self.size_last_begin = size_begin;
    }

    pub fn draft(&mut self, inp: &[u32], sampled: u32, draft_max: usize) -> Vec<u32> {
        self.last_draft_created = false;
        self.last_draft_key_idx = 0;
        self.last_draft_value_idx = 0;

        let cur_len = inp.len();
        let n = self.size_key;
        let m = self.size_value;
        if cur_len < 2 * n + m || draft_max == 0 {
            return Vec::new();
        }
        if self.idx_last_check > cur_len || self.size_last_begin > cur_len {
            self.begin(inp);
        }
        self.idx_last_check = cur_len;

        let mut key_tokens = Vec::with_capacity(n);
        key_tokens.extend_from_slice(&inp[cur_len + 1 - n..cur_len]);
        key_tokens.push(sampled);

        let mut match_pos = 0usize;
        if !self.key_map.is_empty() {
            let idx_hash = (ngram_map_hash(&key_tokens) as usize) % self.key_map.len();
            let idx_key = self.key_map[idx_hash] as usize;
            if idx_key != 0 && idx_key + n + m + 1 < cur_len && inp[idx_key..idx_key + n] == key_tokens[..] {
                match_pos = idx_key;
            }
        }
        if match_pos == 0 && self.size_last_begin > n + m + 1 {
            let hi = self.size_last_begin - n - m - 1;
            for j in (self.key_map_last_idx + 1..=hi).rev() {
                if inp[j..j + n] == key_tokens[..] {
                    match_pos = j;
                    break;
                }
            }
        }
        if match_pos == 0 && cur_len > n + m + 1 {
            let hi = cur_len - n - m - 1;
            let lo = self.size_last_begin.max(self.key_map_last_idx);
            if hi > lo {
                for j in (lo + 1..=hi).rev() {
                    if inp[j..j + n] == key_tokens[..] {
                        match_pos = j;
                        break;
                    }
                }
            }
        }

        if !self.key_map.is_empty() {
            if self.size_last_begin > n + m + 1 {
                let hi = self.size_last_begin - n - m - 1;
                for j in (self.key_map_last_idx + 1..=hi).rev() {
                    let idx_hash = (ngram_map_hash(&inp[j..j + n]) as usize) % self.key_map.len();
                    if self.key_map[idx_hash] == 0 {
                        self.key_map[idx_hash] = j as u32;
                    }
                }
            }
            if cur_len > n + m + 1 {
                let hi = cur_len - n - m - 1;
                let lo = self.size_last_begin.max(self.key_map_last_idx);
                if hi > lo {
                    for j in (lo + 1..=hi).rev() {
                        let idx_hash = (ngram_map_hash(&inp[j..j + n]) as usize) % self.key_map.len();
                        if self.key_map[idx_hash] == 0 {
                            self.key_map[idx_hash] = j as u32;
                        }
                    }
                }
                self.key_map_last_idx = self.key_map_last_idx.max(hi);
            }
        }

        if match_pos == 0 {
            return Vec::new();
        }

        let mut key_offset = self.keys.len();
        for (i, k) in self.keys.iter().enumerate() {
            if k.key_idx + n <= inp.len() && inp[k.key_idx..k.key_idx + n] == key_tokens[..] {
                key_offset = i;
                break;
            }
        }
        if key_offset == self.keys.len() {
            let mut values = [NgramMapValue::default(); COMMON_NGRAM_MAX_VALUES];
            for v in &mut values {
                v.n_accepted = m as i16;
            }
            self.keys.push(NgramMapKey {
                key_idx: match_pos,
                stat_idx: 0,
                key_num: 0,
                values,
            });
        }

        let curr_key = &mut self.keys[key_offset];
        curr_key.key_num = (curr_key.key_num + 1).min(COMMON_NGRAM_MAX_VALUE_COUNT);

        if self.key_only {
            let n_draft = m
                .min(curr_key.values[0].n_accepted.max(0) as usize)
                .min(draft_max)
                .min(cur_len.saturating_sub(match_pos + n));
            self.last_draft_created = n_draft > 0;
            self.last_draft_key_idx = key_offset;
            self.last_draft_value_idx = 0;
            return inp[match_pos + n..match_pos + n + n_draft].to_vec();
        }

        if curr_key.key_num < self.min_hits {
            return Vec::new();
        }

        for i in curr_key.stat_idx..=match_pos {
            if i + n + m > cur_len || inp[i..i + n] != key_tokens[..] {
                continue;
            }
            let idx_begin_value_key = i + n;
            let mut idx_value = None;
            for v in 0..COMMON_NGRAM_MAX_VALUES {
                let idx_v = curr_key.values[v].value_idx;
                if idx_v == 0 {
                    curr_key.values[v].value_idx = idx_begin_value_key;
                    curr_key.values[v].value_num = 0;
                    curr_key.values[v].n_accepted = m as i16;
                    idx_value = Some(v);
                    break;
                }
                if idx_v + m <= cur_len && inp[idx_begin_value_key..idx_begin_value_key + m] == inp[idx_v..idx_v + m] {
                    idx_value = Some(v);
                    break;
                }
            }
            if let Some(v) = idx_value {
                curr_key.values[v].value_num = (curr_key.values[v].value_num + 1).min(COMMON_NGRAM_MAX_VALUE_COUNT);
            }
        }
        curr_key.stat_idx = match_pos;

        let mut max_occur = 0u16;
        let mut slot_max = 0usize;
        for v in 0..COMMON_NGRAM_MAX_VALUES {
            if curr_key.values[v].value_num > max_occur {
                max_occur = curr_key.values[v].value_num;
                slot_max = v;
            }
        }
        let sum_occur: u32 = (0..COMMON_NGRAM_MAX_VALUES)
            .filter(|&v| v != slot_max)
            .map(|v| curr_key.values[v].value_num as u32)
            .sum();
        if sum_occur > 0 && (max_occur as u32) < 2 * sum_occur {
            return Vec::new();
        }

        let n_draft = m
            .min(curr_key.values[slot_max].n_accepted.max(0) as usize)
            .min(draft_max)
            .min(cur_len.saturating_sub(match_pos + n));
        self.last_draft_created = n_draft > 0;
        self.last_draft_key_idx = key_offset;
        self.last_draft_value_idx = slot_max;
        inp[match_pos + n..match_pos + n + n_draft].to_vec()
    }

    pub fn accept(&mut self, n_accepted: usize) {
        if !self.last_draft_created {
            return;
        }
        if let Some(key) = self.keys.get_mut(self.last_draft_key_idx) {
            if let Some(val) = key.values.get_mut(self.last_draft_value_idx) {
                val.n_accepted = n_accepted as i16;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. N-gram Mod (`common_ngram_mod` in `llama.cpp`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct NgramMod {
    n: usize,
    n_min: usize,
    n_max: usize,
    entries: Vec<i32>,
    used: usize,
    i_last: usize,
    n_draft_last: usize,
    n_low: usize,
}

impl NgramMod {
    pub const EMPTY: i32 = -1;
    pub const DEFAULT_SIZE: usize = 4 * 1024 * 1024;

    pub fn new(n: usize, n_min: usize, n_max: usize) -> Self {
        Self {
            n: n.max(1),
            n_min,
            n_max,
            entries: vec![Self::EMPTY; Self::DEFAULT_SIZE],
            used: 0,
            i_last: 0,
            n_draft_last: 0,
            n_low: 0,
        }
    }

    fn idx(&self, tokens: &[u32]) -> usize {
        let mut res = 0u64;
        for &t in &tokens[..self.n] {
            res = res.wrapping_mul(6364136223846793005u64).wrapping_add(t as u64);
        }
        (res as usize) % self.entries.len()
    }

    fn add(&mut self, window: &[u32]) {
        let i = self.idx(window);
        if self.entries[i] == Self::EMPTY {
            self.used += 1;
        }
        self.entries[i] = window[self.n] as i32;
    }

    fn get(&self, window: &[u32]) -> i32 {
        self.entries[self.idx(window)]
    }

    pub fn reset_table(&mut self) {
        self.entries.fill(Self::EMPTY);
        self.used = 0;
    }

    pub fn begin(&mut self, prompt: &[u32]) {
        self.i_last = 0;
        self.n_draft_last = 0;
        let n = self.n;
        if prompt.len() <= n {
            return;
        }
        for i in 0..prompt.len() - n {
            self.add(&prompt[i..=i + n]);
        }
        self.i_last = prompt.len() - n;
        if (self.used as f64) / (self.entries.len() as f64) > 0.25 {
            self.reset_table();
        }
    }

    pub fn draft(&mut self, prompt: &[u32], sampled: u32, draft_max: usize) -> Vec<u32> {
        self.n_draft_last = 0;
        let cur_len = prompt.len();
        let n = self.n;
        let max_len = self.n_max.min(draft_max);
        if cur_len < n || max_len == 0 {
            return Vec::new();
        }
        if self.i_last + 32 < cur_len && cur_len > n {
            for i in self.i_last..cur_len - n {
                self.add(&prompt[i..=i + n]);
            }
            self.i_last = cur_len - n;
        }
        let mut buf = Vec::with_capacity(n + max_len);
        buf.extend_from_slice(&prompt[cur_len + 1 - n..cur_len]);
        buf.push(sampled);
        for i in 0..max_len {
            let tok = self.get(&buf[i..i + n]);
            if tok == Self::EMPTY {
                if i < self.n_min.min(max_len) {
                    return Vec::new();
                }
                break;
            }
            buf.push(tok as u32);
        }
        let out = buf[n..].to_vec();
        self.n_draft_last = out.len();
        out
    }

    pub fn accept(&mut self, n_accepted: usize) {
        if self.n_draft_last > 0 {
            let f_acc = (n_accepted as f64) / (self.n_draft_last as f64);
            if f_acc < 0.25 {
                self.n_low += 1;
                if self.n_low >= 5 {
                    self.reset_table();
                    self.n_low = 0;
                    self.i_last = 0;
                }
            } else {
                self.n_low = 0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 4. N-gram Cache (`common_ngram_cache` in `llama.cpp`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramKey(pub [i32; LLAMA_NGRAM_MAX]);

impl Default for NgramKey {
    fn default() -> Self {
        Self([LLAMA_TOKEN_NULL; LLAMA_NGRAM_MAX])
    }
}

impl NgramKey {
    pub fn from_slice(tokens: &[u32]) -> Self {
        let mut k = [LLAMA_TOKEN_NULL; LLAMA_NGRAM_MAX];
        for (i, &t) in tokens.iter().take(LLAMA_NGRAM_MAX).enumerate() {
            k[i] = t as i32;
        }
        Self(k)
    }
}

impl std::hash::Hash for NgramKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let mut h = (self.0[0] as u64).wrapping_mul(11400714819323198485u64);
        for i in 1..LLAMA_NGRAM_MAX {
            h ^= (self.0[i] as u64).wrapping_mul(11400714819323198485u64);
        }
        state.write_u64(h);
    }
}

pub type NgramCachePart = HashMap<u32, i32>;
pub type NgramCacheMap = HashMap<NgramKey, NgramCachePart>;

pub fn ngram_cache_update(cache: &mut NgramCacheMap, ngram_min: usize, ngram_max: usize, inp: &[u32], nnew: usize) {
    let inp_size = inp.len();
    if inp_size < 2 || nnew == 0 {
        return;
    }
    let nmin = ngram_min.clamp(1, LLAMA_NGRAM_MAX);
    let nmax = ngram_max.clamp(nmin, LLAMA_NGRAM_MAX);
    for ngram_size in nmin..=nmax {
        let i_start = inp_size.saturating_sub(nnew).max(ngram_size);
        for i in i_start..inp_size {
            let ngram = NgramKey::from_slice(&inp[i - ngram_size..i]);
            let token = inp[i];
            let part = cache.entry(ngram).or_default();
            *part.entry(token).or_insert(0) += 1;
        }
    }
}

fn get_seq_token(inp: &[u32], draft: &[u32], i: usize) -> u32 {
    if i < inp.len() {
        inp[i]
    } else {
        draft[1 + i - inp.len()]
    }
}

fn try_draft_static(nc_static: &NgramCacheMap, ngram_static: NgramKey) -> Option<u32> {
    let part = nc_static.get(&ngram_static)?;
    let mut max_count = 0i32;
    let mut sum_count = 0i32;
    let mut max_token = None;
    for (&tok, &cnt) in part {
        if cnt > max_count {
            max_count = cnt;
            max_token = Some(tok);
        }
        sum_count += cnt;
    }
    if sum_count < DRAFT_MIN_SAMPLE_SIZE_LAX[LLAMA_NGRAM_STATIC - 1] {
        return None;
    }
    if 100 * max_count < DRAFT_MIN_PERCENT_LAX[LLAMA_NGRAM_STATIC - 1] * sum_count {
        return None;
    }
    max_token
}

fn try_draft_primary(
    nc_primary: &NgramCacheMap,
    ngrams_primary: &[NgramKey],
    part_static: Option<&NgramCachePart>,
    min_sample_size: &[i32; LLAMA_NGRAM_MAX],
    min_percent: &[i32; LLAMA_NGRAM_MAX],
) -> Option<u32> {
    for &ngram in ngrams_primary.iter().rev() {
        let Some(part_primary) = nc_primary.get(&ngram) else {
            continue;
        };
        let idx = ngram
            .0
            .iter()
            .take_while(|&&t| t != LLAMA_TOKEN_NULL)
            .count()
            .saturating_sub(1)
            .min(LLAMA_NGRAM_MAX - 1);
        let mut max_count_primary = 0i64;
        let mut max_count_static = 0i64;
        let mut sum_count_primary = 0i32;
        let mut max_token = None;

        for (&token, &count_primary) in part_primary {
            let count_static = part_static
                .and_then(|ps| ps.get(&token))
                .map_or(1i64, |&c| 100 * (c as i64));
            let cp = count_primary as i64;
            if cp * count_static > max_count_primary * max_count_static {
                max_token = Some(token);
                max_count_primary = cp;
                max_count_static = count_static;
            }
            sum_count_primary += count_primary;
        }
        if sum_count_primary < min_sample_size[idx] {
            continue;
        }
        if 100 * (max_count_primary as i32) < min_percent[idx] * sum_count_primary {
            continue;
        }
        if max_token.is_some() {
            return max_token;
        }
    }
    None
}

pub fn ngram_cache_draft(
    inp: &[u32],
    sampled: u32,
    n_draft: usize,
    ngram_min: usize,
    ngram_max: usize,
    nc_context: &NgramCacheMap,
    nc_dynamic: &NgramCacheMap,
    nc_static: &NgramCacheMap,
) -> Vec<u32> {
    if inp.len() + 1 < LLAMA_NGRAM_STATIC || n_draft == 0 {
        return Vec::new();
    }
    let mut full_inp = Vec::with_capacity(inp.len() + 1);
    full_inp.extend_from_slice(inp);
    full_inp.push(sampled);
    let inp_size = full_inp.len();

    let mut draft = vec![sampled];
    let nmin = ngram_min.clamp(1, LLAMA_NGRAM_MAX);
    let nmax = ngram_max.clamp(nmin, LLAMA_NGRAM_MAX);

    while draft.len() - 1 < n_draft {
        let total_len = inp_size + draft.len() - 1;
        let ngram_static = if total_len >= LLAMA_NGRAM_STATIC {
            let ngram_start_static = total_len - LLAMA_NGRAM_STATIC;
            let mut k = NgramKey::default();
            for j in 0..LLAMA_NGRAM_STATIC {
                k.0[j] = get_seq_token(&full_inp, &draft, ngram_start_static + j) as i32;
            }
            Some(k)
        } else {
            None
        };
        let part_static = ngram_static.as_ref().and_then(|k| nc_static.get(k));

        let mut ngrams_cd = Vec::with_capacity(nmax - nmin + 1);
        for ngram_size_cd in nmin..=nmax {
            if total_len < ngram_size_cd {
                break;
            }
            let start_cd = total_len - ngram_size_cd;
            let mut key = NgramKey::default();
            for j in 0..ngram_size_cd {
                key.0[j] = get_seq_token(&full_inp, &draft, start_cd + j) as i32;
            }
            ngrams_cd.push(key);
        }

        let drafted = try_draft_primary(
            nc_context,
            &ngrams_cd,
            part_static,
            &DRAFT_MIN_SAMPLE_SIZE_LAX,
            &DRAFT_MIN_PERCENT_LAX,
        )
        .or_else(|| {
            try_draft_primary(
                nc_dynamic,
                &ngrams_cd,
                part_static,
                &DRAFT_MIN_SAMPLE_SIZE_STRICT,
                &DRAFT_MIN_PERCENT_STRICT,
            )
        })
        .or_else(|| ngram_static.and_then(|k| try_draft_static(nc_static, k)));

        match drafted {
            Some(tok) => draft.push(tok),
            None => break,
        }
    }
    draft.into_iter().skip(1).collect()
}

pub fn ngram_cache_merge(target: &mut NgramCacheMap, add: &NgramCacheMap) {
    for (&ngram, part) in add {
        let dst = target.entry(ngram).or_default();
        for (&tok, &cnt) in part {
            *dst.entry(tok).or_insert(0) += cnt;
        }
    }
}

/// Save `cache` in `llama.cpp`'s `common_ngram_cache_save` binary format.
pub fn ngram_cache_save(cache: &NgramCacheMap, path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() && !parent.as_os_str().is_empty() {
        std::fs::create_dir_all(parent)?;
    }
    let mut out = BufWriter::new(File::create(path)?);
    for (ngram, part) in cache {
        if part.is_empty() {
            continue;
        }
        for &t in &ngram.0 {
            out.write_all(&t.to_le_bytes())?;
        }
        let ntokens = part.len() as i32;
        out.write_all(&ntokens.to_le_bytes())?;
        for (&tok, &cnt) in part {
            out.write_all(&(tok as i32).to_le_bytes())?;
            out.write_all(&cnt.to_le_bytes())?;
        }
    }
    out.flush()?;
    Ok(())
}

/// Load a lookup cache in `llama.cpp`'s binary format (`common_ngram_cache_load`).
pub fn ngram_cache_load_binary(bytes: &[u8]) -> anyhow::Result<NgramCacheMap> {
    let mut r = std::io::Cursor::new(bytes);
    let mut cache = NgramCacheMap::new();
    let mut buf4 = [0u8; 4];
    while (r.position() as usize) < bytes.len() {
        let mut key = [LLAMA_TOKEN_NULL; LLAMA_NGRAM_MAX];
        for k in &mut key {
            r.read_exact(&mut buf4)?;
            *k = i32::from_le_bytes(buf4);
        }
        r.read_exact(&mut buf4)?;
        let ntokens = i32::from_le_bytes(buf4);
        anyhow::ensure!(ntokens > 0 && ntokens <= 1_000_000, "invalid ntokens {ntokens} in binary ngram cache");
        let mut part = NgramCachePart::with_capacity(ntokens as usize);
        for _ in 0..ntokens {
            r.read_exact(&mut buf4)?;
            let tok = i32::from_le_bytes(buf4);
            r.read_exact(&mut buf4)?;
            let cnt = i32::from_le_bytes(buf4);
            anyhow::ensure!(tok >= 0 && cnt > 0, "invalid token/count ({tok}, {cnt}) in binary ngram cache");
            part.insert(tok as u32, cnt);
        }
        cache.insert(NgramKey(key), part);
    }
    Ok(cache)
}

/// Load a lookup cache from `path`: tries `llama.cpp` binary format first, or if the file is valid UTF-8 text and a
/// tokenizer is supplied, tokenizes the text corpus and extracts 1..=4 n-grams into the cache.
pub fn ngram_cache_load(path: &Path, tok: Option<&crate::tokenizer::Tokenizer>) -> anyhow::Result<NgramCacheMap> {
    let bytes = std::fs::read(path)?;
    if bytes.is_empty() {
        return Ok(NgramCacheMap::new());
    }
    if let Ok(cache) = ngram_cache_load_binary(&bytes) && !cache.is_empty() {
        return Ok(cache);
    }
    if let (Ok(text), Some(tok)) = (std::str::from_utf8(&bytes), tok) {
        let ids = tok.encode(text)?;
        let mut cache = NgramCacheMap::new();
        ngram_cache_update(&mut cache, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, &ids, ids.len());
        return Ok(cache);
    }
    ngram_cache_load_binary(&bytes)
}

/// Drop cache entries whose key or candidate tokens exceed `n_vocab` (e.g. a cache saved by a larger-vocab model).
pub fn ngram_cache_filter_vocab(cache: &mut NgramCacheMap, n_vocab: usize) -> usize {
    if n_vocab == 0 {
        return 0;
    }
    let mut pruned = 0usize;
    cache.retain(|key, part| {
        let key_ok = key.0.iter().all(|&t| t == LLAMA_TOKEN_NULL || (t >= 0 && (t as usize) < n_vocab));
        if !key_ok {
            pruned += part.len().max(1);
            return false;
        }
        let before = part.len();
        part.retain(|&tok, _| (tok as usize) < n_vocab);
        pruned += before - part.len();
        !part.is_empty()
    });
    pruned
}

// ---------------------------------------------------------------------------
// Unified Drafter (`NgramDrafter`) owned by `GpuModel`
// ---------------------------------------------------------------------------

pub struct NgramDrafter {
    pub cfg: NgramConfig,
    pub history: Vec<u32>,
    eos: Vec<u32>,
    map_k: Option<NgramMap>,
    map_k4v: Option<NgramMap>,
    mod_state: Option<NgramMod>,
    cache_size: usize,
    nc_context: NgramCacheMap,
    nc_dynamic: NgramCacheMap,
    nc_static: NgramCacheMap,
    last_used: Option<SpecType>,
    last_draft_len: usize,
    last_matched_n: usize,
    short_cooldown: usize,
}

impl NgramDrafter {
    pub fn new(cfg: NgramConfig, tok: Option<&crate::tokenizer::Tokenizer>) -> anyhow::Result<Self> {
        let eos = tok.map(|t| t.eos_ids()).unwrap_or_default();
        let mut nc_static = match &cfg.lookup_cache_static {
            Some(p) => ngram_cache_load(p, tok)?,
            None => NgramCacheMap::new(),
        };
        let mut nc_dynamic = match &cfg.lookup_cache_dynamic {
            Some(p) if p.exists() => ngram_cache_load(p, tok).unwrap_or_default(),
            _ => NgramCacheMap::new(),
        };
        let pruned = ngram_cache_filter_vocab(&mut nc_static, cfg.n_vocab)
            + ngram_cache_filter_vocab(&mut nc_dynamic, cfg.n_vocab);
        if pruned > 0 {
            eprintln!(
                "note: pruned {pruned} out-of-vocabulary entries (n_vocab={}) from ngram lookup cache",
                cfg.n_vocab
            );
        }
        let map_k = cfg
            .types
            .contains(&SpecType::NgramMapK)
            .then(|| NgramMap::new(cfg.size_n, cfg.size_m, true, cfg.min_hits));
        let map_k4v = cfg
            .types
            .contains(&SpecType::NgramMapK4v)
            .then(|| NgramMap::new(cfg.size_n, cfg.size_m, false, cfg.min_hits));
        let mod_state = cfg
            .types
            .contains(&SpecType::NgramMod)
            .then(|| NgramMod::new(cfg.mod_n_match, cfg.mod_n_min, cfg.draft_max));

        Ok(Self {
            cfg,
            history: Vec::new(),
            eos,
            map_k,
            map_k4v,
            mod_state,
            cache_size: 0,
            nc_context: NgramCacheMap::new(),
            nc_dynamic,
            nc_static,
            last_used: None,
            last_draft_len: 0,
            last_matched_n: 0,
            short_cooldown: 0,
        })
    }

    fn flush_dynamic_context(&mut self) {
        if self.cfg.lookup_cache_dynamic.is_some() {
            if self.cache_size < self.history.len() {
                let nnew = self.history.len() - self.cache_size;
                ngram_cache_update(&mut self.nc_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, &self.history, nnew);
                self.cache_size = self.history.len();
            }
            if !self.nc_context.is_empty() {
                ngram_cache_merge(&mut self.nc_dynamic, &self.nc_context);
            }
        }
    }

    pub fn reset(&mut self) {
        self.flush_dynamic_context();
        self.history.clear();
        self.nc_context.clear();
        self.cache_size = 0;
        self.last_used = None;
        self.last_draft_len = 0;
        self.last_matched_n = 0;
        self.short_cooldown = 0;
        if let Some(m) = &mut self.map_k {
            m.begin(&[]);
        }
        if let Some(m) = &mut self.map_k4v {
            m.begin(&[]);
        }
    }

    /// Initialize or replace history with `prompt` (called by `prefill` and `prefill_cached`).
    pub fn begin(&mut self, prompt: &[u32]) {
        if !prompt.starts_with(&self.history) {
            self.flush_dynamic_context();
            self.nc_context.clear();
            self.cache_size = 0;
        }
        self.history.clear();
        self.history.extend_from_slice(prompt);
        self.last_used = None;
        self.last_draft_len = 0;
        self.last_matched_n = 0;
        self.short_cooldown = 0;
        if let Some(m) = &mut self.map_k {
            m.begin(&self.history);
        }
        if let Some(m) = &mut self.map_k4v {
            m.begin(&self.history);
        }
        if let Some(m) = &mut self.mod_state {
            m.begin(&self.history);
        }
    }

    /// Like `draft`, also returning the matched prefix order (`0` for non-suffix drafters).
    pub fn draft_with_order(&mut self, sampled: u32, max_draft: usize) -> (Vec<u32>, usize) {
        let d = self.draft(sampled, max_draft);
        let n = if d.is_empty() { 0 } else { self.last_matched_n };
        (d, n)
    }

    /// Propose up to `max_draft` tokens following `sampled` (which is not yet in `self.history`).
    pub fn draft(&mut self, sampled: u32, max_draft: usize) -> Vec<u32> {
        let cap = max_draft.min(self.cfg.draft_max);
        self.last_used = None;
        self.last_draft_len = 0;
        self.last_matched_n = 0;
        if cap == 0 {
            return Vec::new();
        }
        let mut out = Vec::new();
        for &ty in &self.cfg.types {
            let (res, mn) = match ty {
                SpecType::None => (Vec::new(), 0),
                SpecType::Suffix => {
                    if self.short_cooldown == 0 { suffix_draft_with_order(&self.history, sampled, cap, &self.eos) } else { (Vec::new(), 0) }
                }
                SpecType::NgramSimple => {
                    let mut r = ngram_simple_draft_eos(&self.history, sampled, self.cfg.size_n, self.cfg.size_m, cap, &self.eos);
                    let mut mn = if r.is_empty() { 0 } else { self.cfg.size_n };
                    if r.is_empty() && self.short_cooldown == 0 {
                        if self.cfg.size_n > 6 {
                            r = ngram_simple_draft_eos(&self.history, sampled, 6, self.cfg.size_m, cap, &self.eos);
                            if !r.is_empty() { mn = 6; }
                        }
                        if r.is_empty() && self.cfg.size_n > 4 {
                            r = ngram_simple_draft_eos(&self.history, sampled, 4, self.cfg.size_m, cap.min(8), &self.eos);
                            if !r.is_empty() { mn = 4; }
                        }
                    }
                    (r, mn)
                }
                SpecType::NgramMapK => (self.map_k.as_mut().map_or_else(Vec::new, |m| m.draft(&self.history, sampled, cap)), 0),
                SpecType::NgramMapK4v => (self.map_k4v.as_mut().map_or_else(Vec::new, |m| m.draft(&self.history, sampled, cap)), 0),
                SpecType::NgramMod => (self.mod_state.as_mut().map_or_else(Vec::new, |m| m.draft(&self.history, sampled, cap)), 0),
                SpecType::NgramCache => {
                    let mut inp = self.history.clone();
                    inp.push(sampled);
                    if self.cache_size < inp.len() {
                        let nnew = inp.len() - self.cache_size;
                        ngram_cache_update(&mut self.nc_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, &inp, nnew);
                        self.cache_size = inp.len();
                    }
                    let exact = if self.short_cooldown == 0 {
                        ngram_simple_draft_eos(&self.history, sampled, 6, self.cfg.size_m, cap, &self.eos)
                    } else {
                        Vec::new()
                    };
                    if !exact.is_empty() {
                        (exact, 6)
                    } else if self.short_cooldown == 0 || !self.nc_static.is_empty() || !self.nc_dynamic.is_empty() {
                        let cache_cap = if self.nc_static.is_empty() && self.nc_dynamic.is_empty() {
                            cap.min(6)
                        } else {
                            cap
                        };
                        (ngram_cache_draft(
                            &self.history,
                            sampled,
                            cache_cap,
                            2,
                            LLAMA_NGRAM_MAX,
                            &self.nc_context,
                            &self.nc_dynamic,
                            &self.nc_static,
                        ), 0)
                    } else {
                        (Vec::new(), 0)
                    }
                }
            };
            if !res.is_empty() {
                self.last_used = Some(ty);
                self.last_matched_n = mn;
                out = res;
                break;
            }
        }
        if out.is_empty()
            && !self.cfg.types.contains(&SpecType::NgramCache)
            && (!self.nc_static.is_empty() || !self.nc_dynamic.is_empty())
        {
            let res = ngram_cache_draft(
                &self.history,
                sampled,
                cap,
                3,
                LLAMA_NGRAM_MAX,
                &NgramCacheMap::new(),
                &self.nc_dynamic,
                &self.nc_static,
            );
            if !res.is_empty() {
                self.last_used = Some(SpecType::NgramCache);
                out = res;
            }
        }
        if let Some(pos) = out.iter().position(|t| self.eos.contains(t)) {
            out.truncate(pos + 1);
        }
        if self.cfg.n_vocab > 0 {
            if let Some(pos) = out.iter().position(|&t| (t as usize) >= self.cfg.n_vocab) {
                out.truncate(pos);
            }
        }
        self.last_draft_len = out.len();
        out
    }

    /// Record the tokens accepted in this step (`[sampled, accepted_drafts..]`) and notify the active drafter.
    pub fn accept(&mut self, accepted_tokens: &[u32]) {
        let n_accepted_drafts = accepted_tokens.len().saturating_sub(1);
        if self.short_cooldown > 0 {
            self.short_cooldown -= 1;
        }
        if self.last_draft_len > 0 && n_accepted_drafts == 0 {
            self.short_cooldown = 8;
        }
        match self.last_used {
            Some(SpecType::NgramMapK) => {
                if let Some(m) = &mut self.map_k {
                    m.accept(n_accepted_drafts);
                }
            }
            Some(SpecType::NgramMapK4v) => {
                if let Some(m) = &mut self.map_k4v {
                    m.accept(n_accepted_drafts);
                }
            }
            Some(SpecType::NgramMod) => {
                if let Some(m) = &mut self.mod_state {
                    m.accept(n_accepted_drafts);
                }
            }
            _ => {}
        }
        self.history.extend_from_slice(accepted_tokens);
    }

    /// Persist the dynamic lookup cache to disk if configured.
    pub fn save_dynamic(&mut self) -> anyhow::Result<()> {
        self.flush_dynamic_context();
        let Some(path) = &self.cfg.lookup_cache_dynamic else {
            return Ok(());
        };
        if !self.nc_dynamic.is_empty() {
            ngram_cache_save(&self.nc_dynamic, path)?;
        }
        Ok(())
    }
}

impl Drop for NgramDrafter {
    fn drop(&mut self) {
        let _ = self.save_dynamic();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ngram_simple_finds_exact_match_and_respects_draft_max() {
        // Build a 100-token sequence with a repeated 30-token block.
        let mut tokens: Vec<u32> = (1..=80).collect();
        // Repeat tokens 20..32 (12 tokens) at the end: 20..31 in history, 31 as `sampled`.
        tokens.extend(20..31);
        let draft = ngram_simple_draft(&tokens, 31, 12, 48, 16);
        assert_eq!(draft, (32..48).collect::<Vec<u32>>());
    }

    #[test]
    fn ngram_map_k_and_k4v_draft_and_adapt_acceptance() {
        let mut tokens: Vec<u32> = (1..=100).collect();
        tokens.extend(10..21);
        let mut map = NgramMap::new(12, 16, true, 1);
        map.begin(&tokens);
        let d1 = map.draft(&tokens, 21, 16);
        assert_eq!(d1, (22..38).collect::<Vec<u32>>());
        map.accept(5);
        let d2 = map.draft(&tokens, 21, 16);
        assert_eq!(d2.len(), 5);
    }

    #[test]
    fn ngram_mod_hashes_and_drafts_long_sequences() {
        let mut mod_drafter = NgramMod::new(8, 2, 16);
        let prompt: Vec<u32> = (100..200).collect();
        mod_drafter.begin(&prompt);
        let mut ctx = prompt.clone();
        ctx.extend_from_slice(&prompt[10..17]);
        let draft = mod_drafter.draft(&ctx, prompt[17], 12);
        assert_eq!(draft, prompt[18..30].to_vec());
    }

    #[test]
    fn ngram_cache_roundtrip_and_draft() {
        let mut cache = NgramCacheMap::new();
        let seq = vec![10u32, 20, 30, 40, 50, 10, 20, 30, 40, 50, 10, 20, 30, 40, 50];
        ngram_cache_update(&mut cache, 1, 4, &seq, seq.len());
        let tmp = std::env::temp_dir().join("strata_test_ngram_cache.bin");
        ngram_cache_save(&cache, &tmp).unwrap();
        let loaded = ngram_cache_load(&tmp, None).unwrap();
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(cache.len(), loaded.len());
        let draft = ngram_cache_draft(&[10, 20], 30, 4, 1, 4, &loaded, &NgramCacheMap::new(), &NgramCacheMap::new());
        assert_eq!(draft, vec![40, 50, 10, 20]);
    }

    #[test]
    fn test_ngram_cache_draft_short_history() {
        let empty = NgramCacheMap::new();
        let draft = ngram_cache_draft(&[], 42, 4, 1, 4, &empty, &empty, &empty);
        assert!(draft.is_empty());
        let mut c = NgramCacheMap::new();
        ngram_cache_update(&mut c, 1, 4, &[], 1);
        ngram_cache_update(&mut c, 1, 4, &[42], 1);
        assert!(c.is_empty());
    }

    #[test]
    fn test_ngram_cache_vocab_filter() {
        let mut cache = NgramCacheMap::new();
        // Token ID 231622 comes from a 248320-vocab model; target is 151936-vocab Qwen3-30B-A3B.
        let cross_seq = vec![10u32, 20, 231_622, 10, 20, 231_622, 10, 20, 231_622];
        ngram_cache_update(&mut cache, 1, 4, &cross_seq, cross_seq.len());
        let valid_seq = vec![10u32, 20, 30, 10, 20, 30, 10, 20, 30];
        ngram_cache_update(&mut cache, 1, 4, &valid_seq, valid_seq.len());
        let tmp = std::env::temp_dir().join("strata_test_ngram_vocab_filter.bin");
        ngram_cache_save(&cache, &tmp).unwrap();
        let cfg = NgramConfig {
            types: vec![SpecType::NgramCache],
            n_vocab: 151_936,
            lookup_cache_static: Some(tmp.clone()),
            ..Default::default()
        };
        let mut drafter = NgramDrafter::new(cfg, None).unwrap();
        let _ = std::fs::remove_file(&tmp);
        drafter.begin(&[10]);
        let drafted = drafter.draft(20, 8);
        assert!(drafted.iter().all(|&t| (t as usize) < 151_936));
        assert_eq!(drafted.first().copied(), Some(30));
    }

    #[test]
    fn test_ngram_cache_try_draft_primary_indexes_by_ngram_len() {
        let mut cache = NgramCacheMap::new();
        // Single occurrence of [10, 20, 30] -> 40. With nmin=2, a 3-token context key must consult
        // DRAFT_MIN_SAMPLE_SIZE_LAX[2] (= 1), not index [1] (= 2).
        let seq = vec![10u32, 20, 30, 40];
        ngram_cache_update(&mut cache, 2, 4, &seq, seq.len());
        let draft = ngram_cache_draft(&[10, 20], 30, 2, 2, 4, &cache, &NgramCacheMap::new(), &NgramCacheMap::new());
        assert_eq!(draft, vec![40]);
    }
}
