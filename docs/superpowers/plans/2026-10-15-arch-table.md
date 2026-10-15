# Architecture Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run `qwen3`, `qwen3moe` and `qwen3next` GGUFs on the CPU oracle and on CUDA. Per-architecture differences live in one table in `src/config.rs`, not in per-model code.

**Architecture:** `config::arch(name)` returns four facts per architecture: `hybrid`, `q_gate`, `k_interleave` and tensor-name `aliases`. `ModelConfig` exposes them as `layers: Vec<LayerKind>`, `q_gate`, `k_interleave` and `aliases`. `Weights::new` resolves aliases. For `qwen3next` it also splits the fused `ssm_ba` tensor into the canonical `ssm_beta` and `ssm_alpha` tensors at load time. The CPU oracle, GPU decode and GPU prefill read only these fields. Two kernel flags are new: gate pointer 0 means "no attention gate", and `interleave` selects the `qwen3next` key-head mapping. IQ4_NL is added because `qwen3next` shared-expert weights use it.

**Tech Stack:** Rust 2024, cudarc 0.19 with NVRTC (`src/cuda/kernels.cu`), rayon, `tokenizers`. Reference: llama.cpp at `~/Clones/llama.cpp` (CUDA build in `build-cuda/bin`).

**Spec:** `docs/superpowers/specs/2026-10-15-vulkan-and-arch-table-design.md`, section "Part 2: Architecture table".

## Global Constraints

- One `match` on the architecture name in `src/config.rs`. No per-architecture modules. All other code uses canonical tensor names (`post_attention_norm.weight`, `ssm_beta.weight`, `ssm_alpha.weight`).
- Order: `qwen3`, then `qwen3moe`, then `qwen3next`. Each goes into the CPU oracle first, then CUDA.
- Gate per architecture (spec, verbatim): "CPU oracle and CUDA produce tokens identical to llama.cpp for "The capital of France is" (24 tokens, greedy); `tools/parity.py` greedy agreement with llama.cpp at least 98.5% on the 20-prompt set (small models have more near-ties)."
- No behavior change for `qwen35` / `qwen35moe`. The full default suite (`cargo test --release`, default model) stays green after every task.
- `Backend::htod` keeps its stream sync (LOOP_LOG #85). Do not touch it.
- Do not edit `README.md`, `Cargo.toml` or `LICENSE.md`. The owner has uncommitted edits to them on `main`.
- Commit with explicit paths only: `git commit -m "..." -- <paths>`. Every commit message ends with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Code style: match the surrounding code. Comments are sparse. A deliberate shortcut with a known ceiling gets a `ponytail:` comment.
- Models go to `~/models` (home has the free space). There is no `hf` CLI. Download with `curl -L -o <file> https://huggingface.co/<repo>/resolve/main/<file>`.

## Deviations from the spec (decided while planning)

1. **`ssm_ba` split at load time, no reorder kernel.** Splitting the fused matrix is a row permutation of quantized rows. `Weights::new` does it once on the CPU, so the CPU oracle, decode and prefill all see ordinary `ssm_beta` / `ssm_alpha` tensors. No kernel or scratch buffer is needed.
2. **Legacy `ssm_in` is not supported.** The unsloth `qwen3next` GGUFs ship `attn_qkv` + `attn_gate`. An old file fails with `missing tensor blk.0.attn_qkv.weight`.
3. **No `qwen3next` alias.** llama.cpp names that tensor `blk.N.post_attention_norm` for both `qwen35` and `qwen3next`, and the 80B header confirms it.
4. **`router_topk` already handles 512 experts** (`TOPK_R 16` × 32 lanes, `ensure!(n <= 512)`). Only a config check is added.
5. **New flag `k_interleave`.** In `qwen3next`, DeltaNet value head `h` reads key head `h / (nv / nk)`. In `qwen35` it reads `h % nk`.
6. **New quant type IQ4_NL** (ggml id 20). The `qwen3next` shared-expert gate and up tensors use it.
7. **New pre-tokenizer `qwen2`.** All three new architectures use it.

## Models (verified from GGUF headers)

| Arch | File | Repo | Size |
|---|---|---|---|
| qwen3 | `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | `unsloth/Qwen3-4B-Instruct-2507-GGUF` | 2.5 GB |
| qwen3moe | `Qwen3-30B-A3B-Instruct-2507-UD-Q2_K_XL.gguf` | `unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF` | 11.8 GB |
| qwen3next | `Qwen3-Next-80B-A3B-Instruct-UD-IQ1_S.gguf` | `unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF` | 22.9 GB |

Facts: qwen3 4B has 36 blocks, heads 32/8, head_dim 128, no `rope.dimension_count`, and no `output.weight` (tied embeddings, already handled). qwen3moe has 48 blocks, heads 32/4, 128 experts, `n_ff_exp` 768, and no shared expert. qwen3next has 48 blocks, n_embd 2048, heads 16/2, head_dim 256, rope dim 64, 512 experts with 10 used, `n_ff_exp` 512, shared expert 512, ssm inner 4096 / state 128 / groups 16 / dt rank 32 / conv 4, and `ssm_ba` [2048, 64].

## Gate commands (used by Tasks 2, 3 and 5)

Reference server (one terminal or background job, from the repo root):

```bash
LD_LIBRARY_PATH=~/Clones/llama.cpp/build-cuda/bin ~/Clones/llama.cpp/build-cuda/bin/llama-server -m $M --port 8199 --n-cpu-moe 99
```

Greedy-24 gate (tool added in Task 2):

```bash
cargo build --release --features cuda
python -I tools/greedy24.py $M                 # CPU oracle
python -I tools/greedy24.py $M --extra=--gpu   # CUDA
```

Parity gate. The reference cache and the alt file are per model, so always name a fresh `--ref` and a missing `--alt`:

```bash
python -I tools/parity.py $M --extra=--gpu --tag $ARCH --ref target/parity-ref-$ARCH.json --alt target/none.json
```

If strata runs out of VRAM next to llama-server, wait until all 20 `ref` lines print, stop the server, and rerun the same command. The reference stays cached. If agreement is below 98.5%, list every miss with its `top2_gap_at_miss` and report it. Do not change thresholds; the controller rules.

---

### Task 1: Architecture table, aliases and the qwen2 tokenizer

**Files:**
- Modify: `src/config.rs` (whole file)
- Modify: `src/weights.rs:50-75` (`Weights` struct, `new`, `get`)
- Modify: `src/model.rs:26-40` (`State::new`), `src/model.rs:57-64` (`Model::load`), `src/model.rs:129`
- Modify: `src/gpu_model.rs:99-128` (`load_layer`, layer list), `src/gpu_model.rs:105`, `src/gpu_model.rs:131` (MTP `load_layer`), `src/gpu_model.rs:162` (KV print), `src/gpu_model.rs:410`
- Modify: `src/gpu_model/prefill.rs:120`
- Modify: `src/tokenizer.rs:13-14`, `src/tokenizer.rs:26`, `src/tokenizer.rs:40`
- Modify: `tests/gguf.rs:47-53`, `tests/quant.rs:34`, `tests/cuda.rs` lines 38, 82, 123, 157, 204, 245, `tests/tokenizer.rs`
- Create: `tests/fixtures/tokens-qwen2.json`

**Interfaces:**
- Produces: `pub enum LayerKind { Delta, Attn }` (Clone, Copy, Debug, PartialEq, Eq) in `config`; `ModelConfig { layers: Vec<LayerKind>, q_gate: bool, k_interleave: bool, aliases: &'static [(&'static str, &'static str)], .. }`; `ModelConfig::ssm_dv(&self) -> usize`; `Weights::new(g: &'a Gguf) -> anyhow::Result<Weights<'a>>`. `ModelConfig::is_recurrent` and `ModelConfig::full_attention_interval` are removed.

- [ ] **Step 1: Download the qwen3 model**

```bash
mkdir -p ~/models && cd ~/models && curl -L -o Qwen3-4B-Instruct-2507-Q4_K_M.gguf https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
```

Expected: a file of about 2.5 GB. Below, `Q3=~/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf`.

- [ ] **Step 2: Write the failing tests**

In `tests/gguf.rs`, change the import to `use rusting_strata::{config::{LayerKind, ModelConfig}, gguf::Gguf, weights::Weights};`. In `real_file_config`, replace the two lines `assert_eq!((c.n_rot, c.full_attention_interval, c.n_vocab), ...)` and `assert!(c.is_recurrent(0) ...)` with:

```rust
    assert_eq!((c.n_rot, c.n_vocab), (64, 248320));
    assert_eq!(&c.layers[..4], &[LayerKind::Delta, LayerKind::Delta, LayerKind::Delta, LayerKind::Attn]);
    assert!(c.q_gate && !c.k_interleave);
```

Append to `tests/gguf.rs`:

```rust
/// Arch-table fields of whichever model STRATA_MODEL names; canonical tensor names resolve through the aliases.
#[test]
fn arch_table_fields() {
    let Some(p) = common::model_path() else { return };
    let g = Gguf::open(&p).unwrap();
    let c = ModelConfig::from_gguf(&g).unwrap();
    let attn = c.layers.iter().filter(|&&k| k == LayerKind::Attn).count();
    assert_eq!(c.layers.len(), c.n_layer);
    match c.arch.as_str() {
        "qwen3" | "qwen3moe" => assert!(attn == c.n_layer && !c.q_gate && !c.k_interleave && c.ssm_dv() == 0),
        "qwen3next" => assert!(attn == c.n_layer / 4 && c.q_gate && c.k_interleave),
        _ => assert!(attn == c.n_layer / 4 && c.q_gate && !c.k_interleave),
    }
    let w = Weights::new(&g).unwrap();
    w.layer(0, "post_attention_norm.weight").unwrap();
    w.layer(c.n_layer - 1, "post_attention_norm.weight").unwrap();
}
```

In `tests/tokenizer.rs`, choose the fixture by pre-tokenizer. Replace the `let fx ...` statement with:

```rust
    let pre = g.get("tokenizer.ggml.pre").unwrap().as_str().unwrap().to_string();
    let fx: serde_json::Value = serde_json::from_str(match pre.as_str() {
        "qwen2" => include_str!("fixtures/tokens-qwen2.json"),
        _ => include_str!("fixtures/tokens.json"),
    }).unwrap();
```

- [ ] **Step 3: Generate the qwen2 token fixture from llama.cpp**

```bash
LD_LIBRARY_PATH=~/Clones/llama.cpp/build-cuda/bin python -I tools/gen_token_fixtures.py $Q3 ~/Clones/llama.cpp/build-cuda/bin/llama-tokenize > tests/fixtures/tokens-qwen2.json
```

Expected: a JSON array of 7 `{text, ids}` objects. Check that the ids differ from `tests/fixtures/tokens.json` (different vocabulary).

- [ ] **Step 4: Run the tests to verify they fail**

Run: `cargo test --release --test gguf --test tokenizer 2>&1 | tail -20`
Expected: compile errors (`LayerKind`, `Weights::new`, `ssm_dv` not found).

- [ ] **Step 5: Replace `src/config.rs`**

```rust
//! Model hyperparameters read from GGUF metadata, plus the per-architecture table.

use crate::gguf::Gguf;
use anyhow::{Context, bail};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind { Delta, Attn }

/// What differs between the supported architectures; everything else comes from GGUF metadata.
struct Arch {
    /// DeltaNet layers with full attention every `full_attention_interval` layers; otherwise all attention.
    hybrid: bool,
    /// `attn_q` is [query | sigmoid output gate] per head (2 * head_dim wide).
    q_gate: bool,
    /// DeltaNet value head h reads key head h / (nv / nk) (llama.cpp qwen3next), not h % nk.
    k_interleave: bool,
    /// Canonical tensor suffix to this architecture's suffix.
    aliases: &'static [(&'static str, &'static str)],
}

fn arch(name: &str) -> Option<Arch> {
    const FFN_NORM: &[(&str, &str)] = &[("post_attention_norm.weight", "ffn_norm.weight")];
    Some(match name {
        "qwen35" | "qwen35moe" => Arch { hybrid: true, q_gate: true, k_interleave: false, aliases: &[] },
        "qwen3next" => Arch { hybrid: true, q_gate: true, k_interleave: true, aliases: &[] },
        "qwen3" | "qwen3moe" => Arch { hybrid: false, q_gate: false, k_interleave: false, aliases: FFN_NORM },
        _ => return None,
    })
}

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub arch: String,
    pub n_layer: usize,
    pub n_nextn: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub rope_base: f32,
    pub eps: f32,
    pub n_vocab: usize,
    /// Dense FFN size (0 for MoE).
    pub n_ff: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub n_ff_shexp: usize,
    /// 0 = no extra scale.
    pub expert_weights_scale: f32,
    /// DeltaNet sizes, 0 without DeltaNet layers.
    pub ssm_d_inner: usize,
    pub ssm_d_state: usize,
    pub ssm_n_group: usize,
    pub ssm_dt_rank: usize,
    pub ssm_conv: usize,
    /// Mixer of each main layer (MTP layers are attention).
    pub layers: Vec<LayerKind>,
    pub q_gate: bool,
    pub k_interleave: bool,
    pub aliases: &'static [(&'static str, &'static str)],
}

impl ModelConfig {
    pub fn from_gguf(g: &Gguf) -> anyhow::Result<ModelConfig> {
        let arch = g.get("general.architecture")?.as_str().context("general.architecture")?.to_string();
        let Some(a) = self::arch(&arch) else {
            bail!("unsupported architecture '{arch}' (supported: qwen3, qwen3moe, qwen3next, qwen35, qwen35moe)")
        };
        let opt = |k: &str| g.meta.get(&format!("{arch}.{k}")).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let req = |k: &str| -> anyhow::Result<usize> {
            let key = format!("{arch}.{k}");
            Ok(g.get(&key)?.as_u64().with_context(|| format!("{key}: not an integer"))? as usize)
        };
        let f = |k: &str| g.meta.get(&format!("{arch}.{k}")).and_then(|v| v.as_f32());
        let n_nextn = opt("nextn_predict_layers");
        let n_layer = req("block_count")? - n_nextn;
        let layers: Vec<LayerKind> = if a.hybrid {
            let every = match opt("full_attention_interval") { 0 => 4, v => v };
            (0..n_layer).map(|il| if (il + 1) % every == 0 { LayerKind::Attn } else { LayerKind::Delta }).collect()
        } else {
            vec![LayerKind::Attn; n_layer]
        };
        let has_delta = layers.contains(&LayerKind::Delta);
        let ssm = |k: &str| if has_delta { req(k) } else { Ok(opt(k)) };
        let (n_embd, n_head) = (req("embedding_length")?, req("attention.head_count")?);
        let head_dim = match opt("attention.key_length") { 0 => n_embd / n_head, v => v };
        let n_expert = opt("expert_count");
        anyhow::ensure!(n_expert <= 512, "{n_expert} experts (at most 512 supported)");
        let emb = g.tensors.get("token_embd.weight").context("missing tensor token_embd.weight")?;
        Ok(ModelConfig {
            n_layer,
            n_nextn,
            n_embd,
            n_head,
            n_head_kv: req("attention.head_count_kv")?,
            head_dim,
            n_rot: match opt("rope.dimension_count") { 0 => head_dim, v => v },
            rope_base: f("rope.freq_base").unwrap_or(10000.0),
            eps: f("attention.layer_norm_rms_epsilon").context("missing rms epsilon")?,
            n_vocab: emb.dims[1] as usize,
            n_ff: opt("feed_forward_length"),
            n_expert,
            n_expert_used: opt("expert_used_count"),
            n_ff_exp: opt("expert_feed_forward_length"),
            n_ff_shexp: opt("expert_shared_feed_forward_length"),
            expert_weights_scale: f("expert_weights_scale").unwrap_or(0.0),
            ssm_d_inner: ssm("ssm.inner_size")?,
            ssm_d_state: ssm("ssm.state_size")?,
            ssm_n_group: ssm("ssm.group_count")?,
            ssm_dt_rank: ssm("ssm.time_step_rank")?,
            ssm_conv: ssm("ssm.conv_kernel")?,
            layers,
            q_gate: a.q_gate,
            k_interleave: a.k_interleave,
            aliases: a.aliases,
            arch,
        })
    }

    pub fn is_moe(&self) -> bool { self.n_expert > 0 }
    /// DeltaNet value head size (0 without DeltaNet layers).
    pub fn ssm_dv(&self) -> usize { self.ssm_d_inner / self.ssm_dt_rank.max(1) }
}
```

- [ ] **Step 6: `Weights::new` and alias lookup in `src/weights.rs`**

Add `use crate::config::ModelConfig;`. Replace the `Weights` struct and `get` with:

```rust
pub struct Weights<'a> {
    pub g: &'a Gguf,
    aliases: &'static [(&'static str, &'static str)],
}

impl<'a> Weights<'a> {
    pub fn new(g: &'a Gguf) -> anyhow::Result<Weights<'a>> {
        let c = ModelConfig::from_gguf(g)?;
        Ok(Weights { g, aliases: c.aliases })
    }

    /// `name` with the architecture's alias for a canonical suffix tried second.
    pub fn get(&self, name: &str) -> anyhow::Result<Tensor<'a>> {
        let t = self.g.tensors.get(name)
            .or_else(|| self.aliases.iter().find_map(|&(c, a)| self.g.tensors.get(&format!("{}{a}", name.strip_suffix(c)?))))
            .with_context(|| format!("missing tensor {name}"))?;
```

The rest of `get` (`let ty = ...` and the `Ok(Tensor { .. })`) stays. `layer` and `validate` stay.

Replace every `Weights { g: &g }` with `Weights::new(&g).unwrap()`: `tests/quant.rs:34` (`rusting_strata::weights::Weights::new(&g).unwrap()`) and `tests/cuda.rs` lines 38, 82, 123, 157, 204, 245.

- [ ] **Step 7: Use `layers` and `ssm_dv` in the CPU oracle (`src/model.rs`)**

Import `LayerKind` next to `ModelConfig` (`use crate::config::{LayerKind, ModelConfig};`). In `State::new`, replace the `head_v` line and the `is_recurrent` map with:

```rust
        let layers = c.layers.iter()
            .map(|k| match k {
                LayerKind::Delta => LayerState::Recurrent {
                    conv: vec![0.0; (c.ssm_conv - 1) * channels],
                    s: vec![0.0; c.ssm_dt_rank * c.ssm_dv() * c.ssm_d_state],
                },
                LayerKind::Attn => LayerState::Attn { k: Vec::new(), v: Vec::new() },
            })
            .collect();
```

In `Model::load`:

```rust
        let cfg = ModelConfig::from_gguf(g)?;
        let w = Weights::new(g)?;
        w.validate()?;
```

At line 129 replace `c.ssm_d_inner / nv` with `c.ssm_dv()`. Update the module doc on line 1: "Scalar CPU reference forward for the architectures in `config::arch`, one token at a time."

- [ ] **Step 8: Use `layers` and `ssm_dv` on the GPU**

In `src/gpu_model.rs`, import `LayerKind`. Change `load_layer` to take `kind: LayerKind` and test `if kind == LayerKind::Delta`. Line 105: `s: gpu.zeros::<f32>(nv * c.ssm_dv() * dk)?`. Line 128:

```rust
        let layers = c.layers.iter().enumerate().map(|(il, &kind)| load_layer(il, kind)).collect::<anyhow::Result<Vec<_>>>()?;
```

MTP (line 131): `load_layer(il, LayerKind::Attn)?`. KV print (line 162): replace `(c.n_layer / c.full_attention_interval)` with `c.layers.iter().filter(|&&k| k == LayerKind::Attn).count()`. Line 410 and `src/gpu_model/prefill.rs:120`: `let (dv, channels) = (c.ssm_dv(), c.ssm_d_inner + 2 * nk * dk);`. Update the module doc on `gpu_model.rs:1` the same way as `model.rs`.

- [ ] **Step 9: The qwen2 pre-tokenizer (`src/tokenizer.rs`)**

Below `QWEN35_RE` add:

```rust
/// llama.cpp `LLAMA_VOCAB_PRE_TYPE_QWEN2` pre-tokenizer regex (qwen3, qwen3moe, qwen3next).
const QWEN2_RE: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
```

Replace line 26 with:

```rust
        let re = match pre { "qwen35" => QWEN35_RE, "qwen2" => QWEN2_RE, _ => bail!("unsupported pre-tokenizer '{pre}'") };
```

and use `re` instead of `QWEN35_RE` in the `Split::new` call.

- [ ] **Step 10: Run the tests**

```bash
cargo build --release --features cuda 2>&1 | grep -E '^(warning|error)' ; cargo test --release --features cuda 2>&1 | grep -E 'test result|FAILED|panicked'
STRATA_MODEL=$Q3 cargo test --release --test gguf arch_table_fields
STRATA_MODEL=$Q3 cargo test --release --test tokenizer
./target/release/strata info --model $Q3 | head -40
```

Expected: no warnings; the default suite is fully green; both `STRATA_MODEL` tests pass; `info` shows `arch: "qwen3"`, 36 `Attn` layers, `q_gate: false`, `head_dim: 128`, `n_rot: 128`, all `ssm_*` 0.

- [ ] **Step 11: Commit**

```bash
git commit -m "feat: architecture table with layer kinds, tensor aliases and the qwen2 tokenizer" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- src/config.rs src/weights.rs src/model.rs src/gpu_model.rs src/gpu_model/prefill.rs src/tokenizer.rs tests/gguf.rs tests/quant.rs tests/cuda.rs tests/tokenizer.rs tests/fixtures/tokens-qwen2.json
```

(Add files with `git add` first if git refuses untracked paths.)

---

### Task 2: qwen3 dense (attention without the output gate)

**Files:**
- Modify: `src/model.rs:180-222` (`attention`)
- Modify: `src/gpu_model.rs:433-443` (`Mixer::Attn` decode)
- Modify: `src/gpu_model/prefill.rs:140-149` (`Mixer::Attn` prefill)
- Modify: `src/cuda/kernels.cu` `attn_merge` (~line 1340), `attn_prefill_t` (~line 1567), `attn_prefill_llama` (~line 1614)
- Modify: `src/backend.rs:95-99` (doc lines of `attn_prefill` / `attn_decode`)
- Modify: `tests/cuda.rs` `attn_prefill_matches_decode`
- Create: `tools/greedy24.py`

**Interfaces:**
- Consumes: `ModelConfig::q_gate`, `ModelConfig::layers` (Task 1).
- Produces: in `attn_decode`, `attn_decode_gq` and `attn_prefill`, `qg == 0` means no gate (out = attention output). The signatures do not change. `tools/greedy24.py <model> [--extra=ARGS] [--port 8199] [--bin target/release/strata] [--prompt TEXT] [-n 24]` exits 0 when strata and llama-server agree.

- [ ] **Step 1: Write the failing kernel test**

In `tests/cuda.rs` `attn_prefill_matches_decode`, after `let got = gpu.stream.clone_dtoh(&dout).unwrap();`, add:

```rust
        // qg = 0: no output gate. The gated output is the ungated output times sigmoid(gate).
        let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
        let gated = |plain: &[f32]| -> Vec<f32> {
            plain.iter().enumerate().map(|(i, o)| o * sig(qg[(i / hd) * 2 * hd + hd + i % hd])).collect()
        };
        gpu.attn_prefill(gpu.ptr(&dq), gpu.ptr(&dk), gpu.ptr(&dv), 0, gpu.ptr(&dout), gpu.ptr(&step), t, nh, nkv, hd, scale).unwrap();
        let e = rel_l2(&gated(&gpu.stream.clone_dtoh(&dout).unwrap()), &got);
        assert!(e <= 1e-6, "attn_prefill without gate nh {nh} hd {hd}: {e}");
```

Inside the `for gq_min` loop, after `let want = gpu.stream.clone_dtoh(&ddec).unwrap();`, add:

```rust
            for i in 0..t {
                let s = gpu.stream.clone_htod(&[(pos0 + i) as i32]).unwrap();
                let off = |p: u64, per: usize| p + (i * per * 4) as u64;
                gpu.attn_decode_gq(off(gpu.ptr(&dq), nh * hd), gpu.ptr(&dk), gpu.ptr(&dv), 0,
                                   off(gpu.ptr(&ddec), nh * hd), gpu.ptr(&part), gpu.ptr(&s), nh, nkv, hd, scale, gq_min, 1).unwrap();
            }
            let e = rel_l2(&gated(&gpu.stream.clone_dtoh(&ddec).unwrap()), &want);
            assert!(e <= 1e-6, "attn_decode without gate nh {nh} hd {hd} gq_min {gq_min}: {e}");
```

(`qg` as a row of `nh * 2 * hd` per token: element `i` of the flat `[t, nh, hd]` output belongs to row `i / hd` of `[t * nh]` heads, and the gate of head row `r` starts at `r * 2 * hd + hd`.) The batched `dbat` comparison below this point must still use `want` from the gated run. Leave it as is.

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test --release --features cuda --test cuda attn_prefill_matches_decode`
Expected: FAIL. The kernels read the gate through a null pointer (illegal address), or the error check fails.

- [ ] **Step 3: Kernels: gate pointer 0 means no gate**

`attn_merge`: replace the `qg += ...` line part and the output line:

```cuda
    if (qg) qg += (size_t)blockIdx.z * gridDim.x * 2 * hd;
    out += (size_t)blockIdx.z * gridDim.x * hd;
```

```cuda
        out[h * hd + d] = o / L * (qg ? sigmoid(qg[h * 2 * hd + hd + d]) : 1.0f);
```

`attn_prefill_t`:

```cuda
        const float* gate = qg ? qg + (size_t)t * nh * 2 * HD + h * 2 * HD + HD : nullptr;
```

```cuda
                orow[col] = o[d][2 * r + e] / l[r] * (gate ? sigmoid(gate[col]) : 1.0f);
```

`attn_prefill_llama`:

```cuda
        const float* gate = qg ? qg + (size_t)t * nh * 2 * HD + h * 2 * HD + HD : nullptr;
```

```cuda
            if (d < HD) out[(size_t)t * nh * HD + h * HD + d] = acc[r] / l * (gate ? sigmoid(gate[d]) : 1.0f);
```

Update the comment above `attn_merge` to say "* sigmoid(gate), gate skipped when qg is null". In `src/backend.rs`, add to the doc comments of `attn_prefill` and `attn_decode`: "`qg` 0: no output gate."

- [ ] **Step 4: Run the kernel test**

Run: `cargo test --release --features cuda --test cuda attn_prefill_matches_decode`
Expected: PASS.

- [ ] **Step 5: CPU oracle without the gate (`src/model.rs` `attention`)**

```rust
        let qs = if c.q_gate { 2 * hd } else { hd };  // per-head row of attn_q: query, then the gate if any
```

Use it in the query norm: `ops::rms_norm(&qg[i * qs..i * qs + hd], &qn, c.eps, qh);`. Replace the two gate lines with:

```rust
            if c.q_gate {
                let gate = &qg[i * qs + hd..(i + 1) * qs];
                for (o, gg) in oh.iter_mut().zip(gate) { *o *= ops::sigmoid(*gg); }
            }
```

- [ ] **Step 6: GPU decode and prefill without the gate**

`src/gpu_model.rs` `Mixer::Attn`:

```rust
                    let qs = if c.q_gate { 2 * hd } else { hd };
                    mv(q, "h", ne, "qg", nh * qs)?;
                    g.rms_norm(p("qg"), qs, qn.ptr, p("q"), hd, hd, nt * nh, eps)?;
```

In the `attn_decode` call, pass `if c.q_gate { p("qg") } else { 0 }` instead of `p("qg")`.

`src/gpu_model/prefill.rs` `Mixer::Attn`:

```rust
                    let qs = if c.q_gate { 2 * hd } else { hd };
                    g.gemm_dense(q, t, h, ne, at(l.a), nh * qs, 0)?;
                    g.rms_norm(at(l.a), qs, qn.ptr, at(l.b), hd, hd, t * nh, eps)?;
```

In the `attn_prefill` call, pass `if c.q_gate { at(l.a) } else { 0 }` instead of `at(l.a)`.

- [ ] **Step 7: Create `tools/greedy24.py`**

```python
# Greedy gate: strata and llama-server produce the same greedy continuation of one prompt.
# Usage: python -I tools/greedy24.py <model.gguf> [--extra=--gpu] [--port 8199] [--bin target/release/strata]
# Needs llama-server running on --port with the same model. Exit status 0 = identical.
import argparse, json, subprocess, sys, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("--extra", default=""); ap.add_argument("--port", type=int, default=8199)
ap.add_argument("--bin", default="target/release/strata"); ap.add_argument("--prompt", default="The capital of France is")
ap.add_argument("-n", type=int, default=24)
a = ap.parse_args()
body = {"prompt": a.prompt, "n_predict": a.n, "temperature": 0, "top_k": 1, "cache_prompt": False}
r = urllib.request.Request(f"http://127.0.0.1:{a.port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
want = json.load(urllib.request.urlopen(r))["content"]
got = subprocess.run([a.bin, "generate", "--model", a.model, "--prompt", a.prompt, "-n", str(a.n), *a.extra.split()],
                     capture_output=True, text=True, check=True).stdout
print("llama.cpp:", repr(want)); print("strata:   ", repr(got))
sys.exit(0 if want and want in got else 1)
```

`strata generate` may print the prompt before the continuation; the substring check covers both cases.

- [ ] **Step 8: Unit gates on the qwen3 model**

```bash
cargo build --release --features cuda 2>&1 | grep -E '^(warning|error)'
STRATA_MODEL=$Q3 cargo test --release --test model greedy_continuation_is_sane
STRATA_MODEL=$Q3 cargo test --release --features cuda --test gpu_model 2>&1 | grep -E 'test |test result'
```

Expected: the CPU test passes ("Paris"). In the GPU tests, each test that compares against the CPU oracle passes. A test that needs MTP or MoE may fail on a dense model with no MTP head. List those failures with their first panic line in the report. Do not change them.

- [ ] **Step 9: Spec gate on qwen3**

Start the reference server with `M=$Q3`, then:

```bash
python -I tools/greedy24.py $Q3
python -I tools/greedy24.py $Q3 --extra=--gpu
python -I tools/parity.py $Q3 --extra=--gpu --tag qwen3 --ref target/parity-ref-qwen3.json --alt target/none.json
```

Expected: both greedy runs exit 0, and `TOP1 AGREEMENT` is at least 98.5%. Stop the server afterwards.

- [ ] **Step 10: Regression on the default model, then commit**

Run: `cargo test --release --features cuda 2>&1 | grep -E 'test result|FAILED|panicked'`. Expected: all green.

```bash
git add tools/greedy24.py
git commit -m "feat: qwen3 dense: attention without the output gate" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- src/model.rs src/gpu_model.rs src/gpu_model/prefill.rs src/cuda/kernels.cu src/backend.rs tests/cuda.rs tools/greedy24.py
```

---

### Task 3: qwen3moe

No code change is expected. `qwen3moe` is `qwen3` attention plus the existing MoE path without a shared expert (routing: softmax, top-k, renormalize, as in `Model::route`). This task proves it, and fixes only what the gates expose.

**Files:**
- Modify: only what a failing gate points to (record each fix in the report).

**Interfaces:**
- Consumes: everything from Tasks 1-2.

- [ ] **Step 1: Download**

```bash
cd ~/models && curl -L -o Qwen3-30B-A3B-Instruct-2507-UD-Q2_K_XL.gguf https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/resolve/main/Qwen3-30B-A3B-Instruct-2507-UD-Q2_K_XL.gguf
```

Below, `QM=~/models/Qwen3-30B-A3B-Instruct-2507-UD-Q2_K_XL.gguf`.

- [ ] **Step 2: Config, kernels per quant type, CPU oracle**

```bash
./target/release/strata info --model $QM | head -40
STRATA_MODEL=$QM cargo test --release --test gguf arch_table_fields
STRATA_MODEL=$QM cargo test --release --features cuda --test cuda dequant_and_gemv_match_reference_for_every_type
STRATA_MODEL=$QM cargo test --release --test model greedy_continuation_is_sane
```

Expected: `info` shows `arch: "qwen3moe"`, 48 `Attn` layers, `n_expert: 128`, `n_expert_used: 8`, `n_ff_exp: 768`, `n_ff_shexp: 0`. All three tests pass.

- [ ] **Step 3: Spec gate**

Start the reference server with `M=$QM`, then:

```bash
python -I tools/greedy24.py $QM
python -I tools/greedy24.py $QM --extra=--gpu
python -I tools/parity.py $QM --extra=--gpu --tag qwen3moe --ref target/parity-ref-qwen3moe.json --alt target/none.json
```

Expected: both greedy runs exit 0, and `TOP1 AGREEMENT` is at least 98.5%.

If a gate fails, debug against the CPU oracle first: the oracle is the reference for the GPU, and llama.cpp is the reference for the oracle. Any fix gets a test that fails without it.

- [ ] **Step 4: Commit (only if code changed)**

```bash
git commit -m "fix: <what the qwen3moe gate exposed>" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- <changed paths>
```

---

### Task 4: IQ4_NL quant type

**Files:**
- Modify: `src/quant/mod.rs` (enum, `from_u32`, `block_elems`, `block_bytes`, `name`, `dequant_row`, new fn `iq4nl`, bail message)
- Modify: `src/cuda/mod.rs:13-16` (`QTYPES`)
- Modify: `src/cuda/kernels.cu` (`deq8<20>` after `deq8<23>`; `QI<20>` after `QI<23>`; `ITYPES` line ~491)
- Modify: `tests/quant.rs` (`rejects_unknown`, new `iq4_nl_block`)

**Interfaces:**
- Produces: `GgmlType::IQ4NL` (ggml id 20, 32 elements per 18-byte block, `name()` "IQ4_NL", CUDA kernel suffix `iq4_nl`).

- [ ] **Step 1: Write the failing tests (`tests/quant.rs`)**

Replace `rejects_unknown` and add a hand-built block test:

```rust
#[test] fn rejects_unknown() { assert!(GgmlType::from_u32(2).is_err()); } // Q4_0 is not supported

/// One IQ4_NL block: f16 scale, then 16 bytes holding elements j (low nibble) and j + 16 (high nibble).
#[test]
fn iq4_nl_block() {
    const K: [f32; 16] = [-127., -104., -83., -65., -49., -35., -22., -10., 1., 13., 25., 38., 53., 69., 89., 113.];
    let ty = GgmlType::from_u32(20).unwrap();
    assert_eq!((ty.name(), ty.row_bytes(32)), ("IQ4_NL", 18));
    let mut b = half::f16::from_f32(0.5).to_le_bytes().to_vec();
    b.extend((0..16u8).map(|j| j | (15 - j) << 4));
    let mut y = [0f32; 32];
    dequant_row(ty, &b, &mut y);
    for j in 0..16 { assert_eq!((y[j], y[j + 16]), (0.5 * K[j], 0.5 * K[15 - j]), "element {j}"); }
}
```

If `half` is not a dev-dependency visible to tests, write the scale bytes as `[0x00, 0x38]` (f16 0.5) instead.

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test --release --test quant`
Expected: `iq4_nl_block` FAILS with "unsupported quant type IQ4_NL".

- [ ] **Step 3: CPU support (`src/quant/mod.rs`)**

- Add `IQ4NL` to the enum, after `IQ4XS`.
- `from_u32`: add `20 => IQ4NL`.
- Bail message: `supported: F32 BF16 Q8_0 Q2_K-Q6_K IQ1_S IQ1_M IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S IQ4_NL IQ4_XS`.
- `block_elems`: `match self { F32 | BF16 => 1, Q8_0 | IQ4NL => 32, _ => 256 }`.
- `block_bytes`: add `IQ4NL => 18`.
- `name`: add `IQ4NL => "IQ4_NL"`.
- `dequant_row`: add `IQ4NL => iq4nl(blk, y)`.

Add the function after `iq4xs`:

```rust
fn iq4nl(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    for j in 0..16 {
        y[j] = d * KVALUES_IQ4NL[(b[2 + j] & 0xF) as usize] as f32;
        y[j + 16] = d * KVALUES_IQ4NL[(b[2 + j] >> 4) as usize] as f32;
    }
}
```

Run: `cargo test --release --test quant`. Expected: PASS.

- [ ] **Step 4: CUDA support**

`src/cuda/mod.rs`: `const QTYPES: [GgmlType; 17]`, and append `IQ4NL` to the list.

`src/cuda/kernels.cu`, after `deq8<23>`:

```cuda
template <> __device__ __forceinline__ void deq8<20>(const u8* b, int sub, float* o) {  // IQ4_NL, 8 blocks of 18
    const u8* p = b + 18 * (sub >> 2);
    int r = (sub & 3) * 8;
    float d = ldh(p);
    const u8* q = p + 2 + (r & 15);
    for (int i = 0; i < 8; i++) o[i] = d * (float)kvalues_iq4nl[r < 16 ? q[i] & 0xF : q[i] >> 4];
}
```

After `QI<23>`:

```cuda
template <> struct QI<20> {  // IQ4_NL, 8 blocks of 18 (2-byte aligned: ld16 only)
    static constexpr bool MIN = false;
    __device__ static void get(const u8* b, int sub, int& lo, int& hi, float& d, float& m) {
        const u8* p = b + 18 * (sub >> 2);
        int r = (sub & 3) * 8;
        d = ldh(p);
        const u8* q = p + 2 + (r & 15);
        u32 w0 = ld16(q) | (ld16(q + 2) << 16), w1 = ld16(q + 4) | (ld16(q + 6) << 16), sft = r < 16 ? 0 : 4;
        int v[8];
        for (int i = 0; i < 4; i++) { v[i] = kvalues_iq4nl[(w0 >> (8 * i + sft)) & 0xF]; v[i + 4] = kvalues_iq4nl[(w1 >> (8 * i + sft)) & 0xF]; }
        lo = pack4(v[0], v[1], v[2], v[3]); hi = pack4(v[4], v[5], v[6], v[7]);
    }
};
```

`ITYPES`: append `X(20, iq4_nl, 144)` after `X(23, iq4_xs, 136)` (144 = 8 blocks × 18 bytes per 256 elements).

- [ ] **Step 5: CUDA test with a model that has IQ4_NL**

No model on disk has IQ4_NL until Task 5 downloads one. Do Task 5 Step 1 (the download) now if it has not happened, then run:

```bash
cargo build --release --features cuda 2>&1 | grep -E '^(warning|error)'
STRATA_MODEL=~/models/Qwen3-Next-80B-A3B-Instruct-UD-IQ1_S.gguf cargo test --release --features cuda --test cuda dequant_and_gemv_match_reference_for_every_type -- --nocapture 2>&1 | grep -iE 'iq4_nl|test result'
cargo test --release --features cuda 2>&1 | grep -E 'test result|FAILED|panicked'
```

Expected: the every-type test passes and its output lists IQ4_NL. The default suite is green.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: IQ4_NL quant type (CPU and CUDA)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- src/quant/mod.rs src/cuda/mod.rs src/cuda/kernels.cu tests/quant.rs
```

---

### Task 5: qwen3next

**Files:**
- Modify: `src/weights.rs` (`built` map, `ssm_ba` split in `new`, lookup in `get`)
- Modify: `src/model.rs:161` (key head of a value head)
- Modify: `src/backend.rs:93` (`deltanet` signature), `src/cuda/mod.rs:409-414` (`deltanet` launch), `src/cuda/kernels.cu` `deltanet_seq` (~line 1008)
- Modify: `src/gpu_model.rs:429`, `src/gpu_model/prefill.rs:135` (`deltanet` calls)
- Modify: `tests/cuda.rs` `deltanet_chunk_matches_steps`, `tests/gguf.rs` (new test)

**Interfaces:**
- Consumes: `ModelConfig::k_interleave`, `ssm_n_group`, `ssm_dt_rank` (Task 1); IQ4_NL (Task 4).
- Produces: `Backend::deltanet(&self, y: u64, s: u64, beta: u64, g: u64, o: u64, nk: usize, nv: usize, interleave: bool, dk: usize, dv: usize, scale: f32, t: usize, ych: usize, snap: u64, nsnap: usize)`. Value head `vh` reads key head `vh / (nv / nk)` when `interleave`, else `vh % nk`.

- [ ] **Step 1: Download (skip if Task 4 already did)**

```bash
cd ~/models && curl -L -o Qwen3-Next-80B-A3B-Instruct-UD-IQ1_S.gguf https://huggingface.co/unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF/resolve/main/Qwen3-Next-80B-A3B-Instruct-UD-IQ1_S.gguf
```

Expected: about 22.9 GB. Below, `QN=~/models/Qwen3-Next-80B-A3B-Instruct-UD-IQ1_S.gguf`.

- [ ] **Step 2: Write the failing tests**

Append to `tests/gguf.rs` (add `weights::Tensor` to the import):

```rust
/// qwen3next fuses beta and alpha in `ssm_ba`: per key head group, r = nv / nk beta rows, then r alpha rows.
#[test]
fn qwen3next_ssm_ba_split() {
    let Some(p) = common::model_path() else { return };
    let g = Gguf::open(&p).unwrap();
    let c = ModelConfig::from_gguf(&g).unwrap();
    if c.arch != "qwen3next" { return; }
    let w = Weights::new(&g).unwrap();
    let (ba, b, a) = (w.layer(0, "ssm_ba.weight").unwrap(), w.layer(0, "ssm_beta.weight").unwrap(), w.layer(0, "ssm_alpha.weight").unwrap());
    assert_eq!((b.rows, a.rows, b.cols, a.cols), (c.ssm_dt_rank, c.ssm_dt_rank, ba.cols, ba.cols));
    let row = |t: &Tensor, i: usize| { let mut v = vec![0f32; t.cols]; t.row(i, &mut v); v };
    let r = c.ssm_dt_rank / c.ssm_n_group;
    for grp in 0..c.ssm_n_group {
        for j in 0..r {
            assert_eq!(row(&b, grp * r + j), row(&ba, grp * 2 * r + j), "beta group {grp} row {j}");
            assert_eq!(row(&a, grp * r + j), row(&ba, grp * 2 * r + r + j), "alpha group {grp} row {j}");
        }
    }
}
```

In `tests/cuda.rs` `deltanet_chunk_matches_steps`, wrap everything after the `let (dy, db, dg) = ...` line in `for ilv in [false, true] { ... }`. In the GPU call, pass `nk, nv, ilv, dk, dv, ...`. In the CPU reference, change `let kh = vh % nk;` to:

```rust
            let kh = if ilv { vh / (nv / nk) } else { vh % nk };
```

Add `ilv` to the assertion message. (With `nk = 2, nv = 4`, the two mappings differ: 0,0,1,1 against 0,1,0,1.)

- [ ] **Step 3: Run them to verify they fail**

```bash
cargo test --release --features cuda --test cuda deltanet_chunk_matches_steps
STRATA_MODEL=$QN cargo test --release --test gguf qwen3next_ssm_ba_split
```

Expected: the first fails to compile (wrong argument count). The second fails with `missing tensor blk.0.ssm_beta.weight`.

- [ ] **Step 4: Split `ssm_ba` in `Weights::new` (`src/weights.rs`)**

Add `use std::collections::HashMap;`. The struct gains a field:

```rust
pub struct Weights<'a> {
    pub g: &'a Gguf,
    aliases: &'static [(&'static str, &'static str)],
    /// Tensors built at load time: qwen3next `ssm_ba` split into `ssm_beta` and `ssm_alpha`.
    built: HashMap<String, Tensor<'a>>,
}
```

`new`:

```rust
    pub fn new(g: &'a Gguf) -> anyhow::Result<Weights<'a>> {
        let c = ModelConfig::from_gguf(g)?;
        let mut w = Weights { g, aliases: c.aliases, built: HashMap::new() };
        // llama.cpp src/models/qwen3next.cpp: per key head group, r = nv / nk beta rows, then r alpha rows.
        for il in 0..c.n_layer {
            let Ok(ba) = w.layer(il, "ssm_ba.weight") else { continue };
            let (r, rb) = (c.ssm_dt_rank / c.ssm_n_group, ba.ty.row_bytes(ba.cols as u64));
            let (mut b, mut a) = (Vec::with_capacity(ba.data.len() / 2), Vec::with_capacity(ba.data.len() / 2));
            for grp in ba.data.chunks_exact(2 * r * rb) {
                b.extend_from_slice(&grp[..r * rb]);
                a.extend_from_slice(&grp[r * rb..]);
            }
            // ponytail: leaked, a few MB per load; own the bytes in Weights if one process loads many models.
            let leak = |v: Vec<u8>| -> &'a [u8] { Box::leak(v.into_boxed_slice()) };
            let half = Tensor { rows: ba.rows / 2, ..ba };
            w.built.insert(format!("blk.{il}.ssm_beta.weight"), Tensor { data: leak(b), ..half });
            w.built.insert(format!("blk.{il}.ssm_alpha.weight"), Tensor { data: leak(a), ..half });
        }
        Ok(w)
    }
```

At the top of `get`, add: `if let Some(t) = self.built.get(name) { return Ok(*t); }`.

Run: `STRATA_MODEL=$QN cargo test --release --test gguf qwen3next_ssm_ba_split arch_table_fields`. Expected: PASS.

- [ ] **Step 5: Key head mapping in the CPU oracle (`src/model.rs:161`)**

```rust
            let kh = if c.k_interleave { vh / (nv / nk) } else { vh % nk };
```

(`c` is `&self.cfg`. Bind it at the top of `deltanet` if it is not bound there yet.)

- [ ] **Step 6: Key head mapping on CUDA**

`src/backend.rs:93`: insert `interleave: bool` after `nv: usize` and document it: "`interleave`: value head h reads key head h / (nv / nk), else h % nk."

`src/cuda/mod.rs` `deltanet`: the same signature change, and pass `interleave as i32` after `nv as i32` in the launch.

`src/cuda/kernels.cu` `deltanet_seq`: add `int ilv` after `int nv` in the parameter list, and change the key head:

```cuda
    int L = dk >> 4, R = 32 / L, lane = threadIdx.x & 31, s = lane % L, vh = blockIdx.x, kh = ilv ? vh / (nv / nk) : vh % nk;
```

Call sites: `src/gpu_model.rs:429` and `src/gpu_model/prefill.rs:135`. Pass `c.k_interleave` after `nv`.

Run: `cargo test --release --features cuda --test cuda deltanet_chunk_matches_steps`. Expected: PASS for both `ilv` values.

- [ ] **Step 7: Config and CPU oracle on the 80B**

```bash
cargo build --release --features cuda 2>&1 | grep -E '^(warning|error)'
./target/release/strata info --model $QN | head -45
STRATA_MODEL=$QN cargo test --release --test gguf arch_table_fields
STRATA_MODEL=$QN cargo test --release --test model greedy_continuation_is_sane
```

Expected: `info` shows `arch: "qwen3next"`, layers Delta, Delta, Delta, Attn repeating (12 Attn), `head_dim: 256`, `n_rot: 64`, `n_expert: 512`, `n_expert_used: 10`, `k_interleave: true`. Both tests pass. The CPU test is slow (80B scalar oracle); allow several minutes.

- [ ] **Step 8: Spec gate**

Start the reference server with `M=$QN`. Use `--n-cpu-moe 99`; RAM is 30 GB, so close other large processes first. Then:

```bash
python -I tools/greedy24.py $QN
python -I tools/greedy24.py $QN --extra=--gpu
python -I tools/parity.py $QN --extra=--gpu --tag qwen3next --ref target/parity-ref-qwen3next.json --alt target/none.json
```

Expected: both greedy runs exit 0, and `TOP1 AGREEMENT` is at least 98.5%. IQ1_S is a very rough quant, so near-ties are more frequent. Report the misses as described in "Gate commands" if the number falls short.

- [ ] **Step 9: Regression on the default model, then commit**

Run: `cargo test --release --features cuda 2>&1 | grep -E 'test result|FAILED|panicked'`. Expected: all green.

```bash
git commit -m "feat: qwen3next: ssm_ba split and interleaved key heads" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- src/weights.rs src/model.rs src/backend.rs src/cuda/mod.rs src/cuda/kernels.cu src/gpu_model.rs src/gpu_model/prefill.rs tests/cuda.rs tests/gguf.rs
```

---

### Task 6: LOOP_LOG entry

**Files:**
- Modify: `docs/LOOP_LOG.md` (new entry before `## Next candidates`)

- [ ] **Step 1: Write the entry**

Insert a section `### 2026-10-15 #86 — architecture table (Part 2 of the Vulkan and architecture-table spec)` directly before `## Next candidates`. Follow the style of entry #85 above it. It contains:

- What changed: the arch table (`hybrid`, `q_gate`, `k_interleave`, `aliases`), `LayerKind`, the `ssm_ba` load-time split, the gate-off kernels, `interleave` in `deltanet`, IQ4_NL, and the qwen2 pre-tokenizer.
- The deviations from the spec, copied from this plan's "Deviations from the spec" list.
- A results table: for each of qwen3, qwen3moe and qwen3next, the model file, greedy-24 CPU and GPU (pass or fail), and parity agreement (with the number of misses whose top-2 gap is under 0.1 nats).
- Default model regression: the suite result.
- Open items: README's supported-architecture list is not updated (the owner has uncommitted README edits); legacy `ssm_in` files are unsupported.

- [ ] **Step 2: Commit**

```bash
git commit -m "docs: loop log #86, architecture table" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- docs/LOOP_LOG.md
```
