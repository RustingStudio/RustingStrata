# M0 + M1: Baseline and CPU Reference Forward — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record the tuned llama.cpp numbers to beat (M0), then build a pure-CPU, scalar-but-correct Rust forward pass
for `qwen35moe` / `qwen35` GGUFs that matches llama.cpp greedy top-1 ≥ 99% (M1). M1 becomes the golden oracle for every
later GPU/CPU-optimized kernel.

**Architecture:** One crate `rusting_strata`. `gguf` parses and mmaps the file; `quant` holds block layouts + scalar
dequant ported from ggml's reference C; `weights` gives named tensor views with a rayon-parallel `matvec`; `ops` holds
scalar math; `model` runs the hybrid DeltaNet/attention/MoE stack one token at a time. A `strata score` subcommand
feeds token ids and prints argmax per position, which `tools/parity.py` compares to `llama-server`.

**Tech Stack:** Rust 1.95 (edition 2024), `memmap2`, `rayon`, `half`, `tokenizers`, `serde_json`, `clap`, `anyhow`.
Python 3 + numpy + `~/llama.cpp/gguf-py` for fixture/oracle scripts only.

**Spec:** `docs/superpowers/specs/2026-10-06-rusting-strata-v1-design.md`

## Global Constraints

- Pure Rust engine; no ggml/llama.cpp linkage. llama.cpp is only an external oracle and baseline.
- v1 quant types: IQ2_XS, IQ3_XXS, IQ4_XS, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, BF16, F32. Any other type → load error
  naming tensor and type.
- Nothing in `model` hardcodes sizes; everything comes from `ModelConfig`.
- Target file: `$STRATA_MODEL`, default
  `/mnt/ollama/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf`.
  Tests that need it skip (print `SKIP` and return) when the file is missing.
- Correctness gate for M1: greedy top-1 agreement ≥ 99% vs llama.cpp on 20 prompts.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Reference semantics (verified in llama.cpp source, `src/models/qwen35moe.cpp`, `delta-net-base.cpp`)

Notation: GGUF dims `[ne0, ne1, ne2]`; a 2-D weight `[in, out]` has `out` rows of `in` elements; `W·x` = per-row dot.

- `n_layer = block_count - nextn_predict_layers` (MTP block is not run in the main pass).
- Layer `il` is recurrent (Gated DeltaNet) iff `(il + 1) % full_attention_interval != 0`.
- RMSNorm: `y = x / sqrt(mean(x²) + eps) * w`, eps = `attention.layer_norm_rms_epsilon`. Weights are used as stored.
- L2 norm: `y = x / max(sqrt(Σx²), eps)` (same eps).
- **Block:** `h = rms(x, attn_norm)`; `x = x + mixer(h)`; `x = x + ffn(rms(x, post_attention_norm))`.
  Final: `rms(x, output_norm)`, `logits = output·x`.
- **DeltaNet mixer** (sizes for 35B: `d_inner`=4096, `d_state`=128, `n_group`=16 (k heads), `dt_rank`=32 (v heads),
  `head_v = d_inner/dt_rank` = 128, conv channels `= d_inner + 2·n_group·d_state` = 8192):
  - `qkv = attn_qkv·h` (8192, order q[2048] | k[2048] | v[4096]); `z = attn_gate·h` (4096).
  - `beta = sigmoid(ssm_beta·h)` [32]; `g = softplus(ssm_alpha·h + ssm_dt.bias) * ssm_a` [32] (ssm_a is already −exp(A_log)).
  - Conv: per channel `c`, window `[s0, s1, s2, qkv[c]]` (oldest first), `y[c] = Σ_k w[c·4 + k]·win[k]`
    (`ssm_conv1d` is `[4, 8192]` F32), new state = `[s1, s2, qkv[c]]`; then `silu`.
  - `q`, `k`: 16 heads × 128, L2-normed per head; `v`: 32 heads × 128. V-head `vh` uses k/q head `vh % 16`.
  - Per v-head, state `S[j][i]` (j = value dim, i = key dim, 128×128 f32):
    `S *= exp(g)`; `sk[j] = Σ_i S[j][i]·k[i]`; `d[j] = beta·(v[j] − sk[j])`; `S[j][i] += d[j]·k[i]`;
    `o[j] = Σ_i S[j][i]·q[i] / sqrt(128)`.
  - Per head: `o = rms(o, ssm_norm[128]) * silu(z_head)`; output `ssm_out·o` (4096 → 2048).
- **Attention mixer** (every 4th layer; 16 q heads, 2 kv heads, head dim 256):
  - `qg = attn_q·h` (8192), per head `[q 256 | gate 256]`; `q = rms(q_head, attn_q_norm)`.
  - `k = rms(attn_k·h per head, attn_k_norm)`, `v = attn_v·h`.
  - RoPE (IMRoPE with all sections at the same text position = NEOX rope on the first `rope.dimension_count`=64 dims):
    for `i in 0..32`: `θ = pos · base^(−2i/64)`, rotate pair `(x[i], x[i+32])`; dims 64..255 unchanged. base = `rope.freq_base`.
  - Causal attention, scale `1/sqrt(256)`, q head `h` uses kv head `h / (n_head / n_head_kv)`.
  - `o *= sigmoid(gate)`; output `attn_output·o` (4096 → 2048).
- **MoE FFN:** `p = softmax(ffn_gate_inp·h)` over 256; top-8 by p; `w = p_top / max(Σp_top, 6.1035e-5)`;
  `y = Σ w_e · down_e(silu(gate_e·h) ⊙ up_e·h)`. Shared: `y += sigmoid(ffn_gate_inp_shexp·h) · down_sh(silu(gate_sh·h) ⊙ up_sh·h)`.
  No `expert_weights_scale` key in the file → no extra scale (apply it only if present and ≠ 0, 1).
- **Dense FFN (`qwen35`):** `down(silu(gate·h) ⊙ up·h)`.

## File Structure

```
RustingStrata/
  Cargo.toml
  src/lib.rs          module list
  src/main.rs         CLI: `strata info`, `strata score`, `strata generate`
  src/gguf.rs         GGUF header/metadata/tensor index parse + mmap
  src/config.rs       ModelConfig from metadata
  src/quant/mod.rs    GgmlType, block sizes, dequant_row dispatch
  src/quant/tables.rs generated IQ tables (from ggml-common.h, MIT)
  src/weights.rs      Tensor view, row access, matvec, expert slicing
  src/ops.rs          scalar math: rms, l2, silu, sigmoid, softplus, softmax, rope, topk
  src/model.rs        qwen35 / qwen35moe CPU reference forward + state
  src/tokenizer.rs    GPT-2 BPE from GGUF via `tokenizers`
  tools/gen_tables.py     ggml-common.h → src/quant/tables.rs
  tools/gen_fixtures.py   real file → tests/fixtures/<type>.{raw,f32}
  tools/parity.py         llama-server vs strata score
  tools/baseline.sh       M0 llama.cpp sweep
  tests/quant.rs  tests/gguf.rs  tests/ops.rs  tests/model.rs  tests/tokenizer.rs
  tests/fixtures/
  docs/baseline.md  docs/m1-parity.md
```

---

### Task 1: M0 — llama.cpp CUDA baseline

**Files:**
- Create: `tools/baseline.sh`, `docs/baseline.md`

**Interfaces:** Produces `docs/baseline.md` with the numbers every later speed gate compares against, and a CUDA
`llama-server` at `~/llama.cpp/build-cuda/bin/` that `tools/parity.py` uses.

- [ ] **Step 1: Build llama.cpp with CUDA in a separate build dir** (existing `build/` is CPU-only; leave it alone)

```bash
cd ~/llama.cpp && cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build-cuda -j 12 --target llama-bench llama-server llama-tokenize llama-cli
```
Expected: binaries in `~/llama.cpp/build-cuda/bin/`, `ldd .../libggml-cuda.so` shows libcudart.

- [ ] **Step 2: Write `tools/baseline.sh`**

```bash
#!/usr/bin/env bash
# Sweep llama.cpp's --n-cpu-moe on the target file; print llama-bench tables.
set -euo pipefail
M=${STRATA_MODEL:-/mnt/ollama/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf}
B=~/llama.cpp/build-cuda/bin
for n in 0 4 8 12 16 20 24; do
  echo "== n-cpu-moe $n"
  $B/llama-bench -m "$M" -ngl 99 --n-cpu-moe $n -fa 1 -p 4096 -n 128 -r 2 -o md 2>/dev/null | tail -n +3 || echo "OOM"
done
```
(If `llama-bench` lacks `--n-cpu-moe`, use `-ot "blk\.(N..39)\.ffn_.*_exps=CPU"` equivalents.)

- [ ] **Step 3: Run sweep, then measure the best setting at 32K depth and with MTP**

Run: `bash tools/baseline.sh | tee /tmp/claude-baseline.txt`
Then for the best `n`: `llama-bench ... -d 32768 -p 4096 -n 128` (decode/prefill at 32K depth), and
`llama-server -m $M -ngl 99 --n-cpu-moe <best> -fa on --spec-type mtp` (check `llama-server --help | grep -i -A2 "mtp\|draft"`
for the exact flag) measuring decode tok/s from the `timings` field of a 256-token `/completion` at temperature 0.

- [ ] **Step 4: Write `docs/baseline.md`** — table: setting, prefill 4K, prefill at 32K depth, decode short, decode at
  32K depth, decode with MTP; hardware line; llama.cpp commit (`git -C ~/llama.cpp log -1 --format=%h`).

- [ ] **Step 5: Commit**

```bash
git add tools/baseline.sh docs/baseline.md && git commit -m "docs: llama.cpp baseline on reference machine"
```

**Gate:** if llama.cpp decode with MTP is already > 2× what the spec's design can plausibly reach, stop and report to
the user before M2 (spec "Risks").

---

### Task 2: Crate scaffold + GGUF parser + ModelConfig

**Files:**
- Create: `Cargo.toml`, `src/lib.rs`, `src/main.rs`, `src/gguf.rs`, `src/config.rs`, `tests/gguf.rs`, `.gitignore`

**Interfaces:**
- Produces:
  - `gguf::Gguf::open(path: &Path) -> anyhow::Result<Gguf>`
  - `Gguf { pub meta: BTreeMap<String, Value>, pub tensors: BTreeMap<String, TensorInfo>, mmap: Mmap, data_offset: usize }`
  - `TensorInfo { pub name: String, pub dims: Vec<u64>, pub ty: u32, pub offset: u64 }`
  - `Gguf::bytes(&self, t: &TensorInfo) -> &[u8]` (exact byte size from type)
  - `Value` enum: `U8..I64, F32, F64, Bool, Str(String), Arr(Vec<Value>)` with `as_u64()`, `as_f32()`, `as_str()`, `as_arr()`
  - `Gguf::get(&self, key) -> anyhow::Result<&Value>`
  - `config::ModelConfig::from_gguf(&Gguf) -> anyhow::Result<ModelConfig>` with fields listed in Step 3.

- [ ] **Step 1: Cargo.toml + .gitignore**

```toml
[package]
name = "rusting_strata"
version = "0.1.0"
edition = "2024"

[dependencies]
anyhow = "1"
memmap2 = "0.9"
rayon = "1"
half = "2"
tokenizers = { version = "0.20", default-features = false, features = ["onig"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
clap = { version = "4", features = ["derive"] }

[profile.release]
opt-level = 3
debug = 1
codegen-units = 1
lto = "thin"

[profile.test]
opt-level = 3
```
`.gitignore`: `/target`

- [ ] **Step 2: Write failing test `tests/gguf.rs`** — synthetic GGUF built in the test, plus a real-file check.

```rust
use rusting_strata::{config::ModelConfig, gguf::Gguf};
use std::io::Write;

fn w_str(b: &mut Vec<u8>, s: &str) { b.extend((s.len() as u64).to_le_bytes()); b.extend(s.as_bytes()); }

#[test]
fn parses_synthetic_file() {
    let mut b = Vec::new();
    b.extend(b"GGUF"); b.extend(3u32.to_le_bytes());
    b.extend(1u64.to_le_bytes()); // tensors
    b.extend(2u64.to_le_bytes()); // kv
    w_str(&mut b, "general.architecture"); b.extend(8u32.to_le_bytes()); w_str(&mut b, "qwen35moe");
    w_str(&mut b, "qwen35moe.block_count"); b.extend(4u32.to_le_bytes()); b.extend(41u32.to_le_bytes());
    w_str(&mut b, "t"); b.extend(2u32.to_le_bytes()); b.extend(4u64.to_le_bytes()); b.extend(2u64.to_le_bytes());
    b.extend(0u32.to_le_bytes()); b.extend(0u64.to_le_bytes()); // F32, offset 0
    while b.len() % 32 != 0 { b.push(0); }
    for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] { b.extend(v.to_le_bytes()); }
    let mut f = tempfile_path("syn.gguf"); f.1.write_all(&b).unwrap(); drop(f.1);
    let g = Gguf::open(&f.0).unwrap();
    assert_eq!(g.get("general.architecture").unwrap().as_str(), Some("qwen35moe"));
    assert_eq!(g.get("qwen35moe.block_count").unwrap().as_u64(), Some(41));
    let t = &g.tensors["t"];
    assert_eq!(t.dims, vec![4, 2]);
    let bytes = g.bytes(t);
    assert_eq!(bytes.len(), 32);
    assert_eq!(f32::from_le_bytes(bytes[28..32].try_into().unwrap()), 8.0);
}

fn tempfile_path(name: &str) -> (std::path::PathBuf, std::fs::File) {
    let p = std::env::temp_dir().join(format!("strata-{}-{name}", std::process::id()));
    let f = std::fs::File::create(&p).unwrap();
    (p, f)
}

pub fn model_path() -> Option<std::path::PathBuf> {
    let p = std::env::var("STRATA_MODEL").unwrap_or_else(|_| "/mnt/ollama/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf".into());
    let p = std::path::PathBuf::from(p);
    if p.exists() { Some(p) } else { eprintln!("SKIP: model file missing"); None }
}

#[test]
fn real_file_config() {
    let Some(p) = model_path() else { return };
    let g = Gguf::open(&p).unwrap();
    let c = ModelConfig::from_gguf(&g).unwrap();
    assert_eq!(c.arch, "qwen35moe");
    assert_eq!((c.n_layer, c.n_nextn, c.n_embd, c.n_head, c.n_head_kv, c.head_dim), (40, 1, 2048, 16, 2, 256));
    assert_eq!((c.n_expert, c.n_expert_used, c.n_ff_exp, c.n_ff_shexp), (256, 8, 512, 512));
    assert_eq!((c.ssm_d_inner, c.ssm_d_state, c.ssm_n_group, c.ssm_dt_rank, c.ssm_conv), (4096, 128, 16, 32, 4));
    assert_eq!((c.n_rot, c.full_attention_interval, c.n_vocab), (64, 4, 248320));
    assert!(c.is_recurrent(0) && c.is_recurrent(2) && !c.is_recurrent(3));
    assert_eq!(g.bytes(&g.tensors["blk.0.ffn_down_exps.weight"]).len(), 2048 * 256 * (512 / 256) * 98); // IQ3_XXS 98 B/256
}
```

- [ ] **Step 3: Implement `src/gguf.rs` and `src/config.rs`**

`gguf.rs`: read magic `GGUF`, version (accept 2 and 3), `n_tensors: u64`, `n_kv: u64`; KV value types
`0 U8,1 I8,2 U16,3 I16,4 U32,5 I32,6 F32,7 Bool,8 Str,9 Arr(elem_type u32, n u64),10 U64,11 I64,12 F64`. Tensor info:
name, `n_dims: u32`, dims `u64`×n, `ty: u32`, `offset: u64`. Alignment = `general.alignment` (default 32);
`data_offset` = header end rounded up. Large string arrays (vocab, 248K entries) must parse fast: read via a cursor
over the mmap, not `Read` calls. `bytes()` uses `quant::GgmlType::from_u32(ty)?.row_bytes(dims[0]) * Π dims[1..]`;
for unsupported types `bytes()` is not callable — `Gguf::open` does not validate types, `Weights` does (Task 5).

`config.rs`:
```rust
pub struct ModelConfig {
    pub arch: String, pub n_layer: usize, pub n_nextn: usize, pub n_embd: usize,
    pub n_head: usize, pub n_head_kv: usize, pub head_dim: usize, pub n_rot: usize,
    pub rope_base: f32, pub eps: f32, pub n_vocab: usize,
    pub n_ff: usize,                       // dense FFN (0 for MoE)
    pub n_expert: usize, pub n_expert_used: usize, pub n_ff_exp: usize, pub n_ff_shexp: usize,
    pub expert_weights_scale: f32,         // 0 = none
    pub ssm_d_inner: usize, pub ssm_d_state: usize, pub ssm_n_group: usize, pub ssm_dt_rank: usize, pub ssm_conv: usize,
    pub full_attention_interval: usize,
}
impl ModelConfig {
    pub fn is_recurrent(&self, il: usize) -> bool { (il + 1) % self.full_attention_interval != 0 }
    pub fn is_moe(&self) -> bool { self.n_expert > 0 }
}
```
Keys are `<arch>.<name>`; arch ∈ {`qwen35`, `qwen35moe`} else error `unsupported architecture '<x>'`.
`n_layer = block_count − nextn_predict_layers (default 0)`, `head_dim = attention.key_length`,
`n_vocab = token_embd.weight.dims[1]`, optional keys default to 0 (`full_attention_interval` defaults to 4).

`src/lib.rs`: `pub mod gguf; pub mod config; pub mod quant; pub mod weights; pub mod ops; pub mod model; pub mod tokenizer;`
(add modules as their tasks land; keep the crate compiling). `src/main.rs`: clap with `info --model` printing config +
tensor type histogram.

- [ ] **Step 4: Run** `cargo test --test gguf` — Expected: 2 passed (real-file test passes when model present).
- [ ] **Step 5: Commit** `feat: GGUF parser and ModelConfig`

---

### Task 3: Quant layouts + scalar dequant (K-quants, Q8_0, BF16, F32) with python-oracle fixtures

**Files:**
- Create: `src/quant/mod.rs`, `tools/gen_fixtures.py`, `tests/quant.rs`, `tests/fixtures/*`

**Interfaces:**
- Produces:
  - `quant::GgmlType` enum `{F32, BF16, Q8_0, Q2K, Q3K, Q4K, Q5K, Q6K, IQ2XS, IQ3XXS, IQ4XS}` with
    `from_u32(u32) -> anyhow::Result<Self>` (ids: F32 0, Q8_0 8, Q2_K 10, Q3_K 11, Q4_K 12, Q5_K 13, Q6_K 14,
    IQ2_XS 17, IQ3_XXS 18, IQ4_XS 23, BF16 30; anything else → `Err("unsupported quant type id N")`),
    `block_elems() -> usize`, `block_bytes() -> usize`, `row_bytes(n: u64) -> usize`, `name() -> &'static str`.
  - `quant::dequant_row(ty: GgmlType, src: &[u8], dst: &mut [f32])` — `dst.len()` elements, `src` exactly `row_bytes`.

Block layouts (bytes per block; f16 fields little-endian, `QK_K = 256`):

| type | elems | bytes | field order |
|---|---|---|---|
| F32 | 1 | 4 | |
| BF16 | 1 | 2 | |
| Q8_0 | 32 | 34 | d f16, qs i8[32] |
| Q2_K | 256 | 84 | scales u8[16], qs u8[64], d f16, dmin f16 |
| Q3_K | 256 | 110 | hmask u8[32], qs u8[64], scales u8[12], d f16 |
| Q4_K | 256 | 144 | d f16, dmin f16, scales u8[12], qs u8[128] |
| Q5_K | 256 | 176 | d f16, dmin f16, scales u8[12], qh u8[32], qs u8[128] |
| Q6_K | 256 | 210 | ql u8[128], qh u8[64], scales i8[16], d f16 |
| IQ2_XS | 256 | 74 | d f16, qs u16[32], scales u8[8] |
| IQ3_XXS | 256 | 98 | d f16, qs u8[96] (64 grid idx + 32 scale/sign bytes) |
| IQ4_XS | 256 | 136 | d f16, scales_h u16, scales_l u8[4], qs u8[128] |

- [ ] **Step 1: `tools/gen_fixtures.py`** — oracle is gguf-py's numpy dequant (independent of the C code we port).

```python
# Usage: python -I tools/gen_fixtures.py <model.gguf> <out_dir>
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path.home() / "llama.cpp/gguf-py"))
from gguf import GGUFReader, GGMLQuantizationType as Q
from gguf.quants import dequantize
from gguf.constants import GGML_QUANT_SIZES

WANT = {Q.F32, Q.BF16, Q.Q8_0, Q.Q2_K, Q.Q3_K, Q.Q4_K, Q.Q5_K, Q.Q6_K, Q.IQ2_XS, Q.IQ3_XXS, Q.IQ4_XS}
r = GGUFReader(sys.argv[1]); out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
done = set()
for t in r.tensors:
    q = t.tensor_type
    if q not in WANT or q in done: continue
    blk, tsz = GGML_QUANT_SIZES[q]
    nblocks = min(8, int(t.shape[0]) // blk)          # first row, up to 8 blocks
    raw = np.frombuffer(t.data.tobytes()[: nblocks * tsz], dtype=np.uint8)
    deq = dequantize(raw, q).astype(np.float32).reshape(-1)
    (out / f"{q.name}.raw").write_bytes(raw.tobytes())
    (out / f"{q.name}.f32").write_bytes(deq.tobytes())
    print(q.name, t.name, nblocks, "blocks"); done.add(q)
missing = WANT - done
print("missing:", [m.name for m in missing])
```
Run: `python -I tools/gen_fixtures.py "$STRATA_MODEL" tests/fixtures`. Expected: all 11 types written, `missing: []`
(F32/BF16 come from norm/router tensors).

- [ ] **Step 2: Failing test `tests/quant.rs`**

```rust
use rusting_strata::quant::{dequant_row, GgmlType};

fn check(name: &str, ty: GgmlType) {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/");
    let raw = std::fs::read(format!("{dir}{name}.raw")).unwrap();
    let want: Vec<f32> = std::fs::read(format!("{dir}{name}.f32")).unwrap()
        .chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(raw.len(), ty.row_bytes(want.len() as u64), "{name} size");
    let mut got = vec![0f32; want.len()];
    dequant_row(ty, &raw, &mut got);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!((g - w).abs() <= 1e-6 * w.abs().max(1.0), "{name}[{i}]: got {g} want {w}");
    }
}

#[test] fn f32_()    { check("F32", GgmlType::F32) }
#[test] fn bf16()    { check("BF16", GgmlType::BF16) }
#[test] fn q8_0()    { check("Q8_0", GgmlType::Q8_0) }
#[test] fn q2_k()    { check("Q2_K", GgmlType::Q2K) }
#[test] fn q3_k()    { check("Q3_K", GgmlType::Q3K) }
#[test] fn q4_k()    { check("Q4_K", GgmlType::Q4K) }
#[test] fn q5_k()    { check("Q5_K", GgmlType::Q5K) }
#[test] fn q6_k()    { check("Q6_K", GgmlType::Q6K) }
#[test] fn rejects_unknown() { assert!(GgmlType::from_u32(16).is_err()); } // IQ2_XXS not in v1
```

- [ ] **Step 3: Run** `cargo test --test quant` — Expected: compile error (module missing).

- [ ] **Step 4: Implement `src/quant/mod.rs`** — direct ports of `ggml-quants.c` `dequantize_row_*`:

```rust
use half::{bf16, f16};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType { F32, BF16, Q8_0, Q2K, Q3K, Q4K, Q5K, Q6K, IQ2XS, IQ3XXS, IQ4XS }
use GgmlType::*;

impl GgmlType {
    pub fn from_u32(id: u32) -> anyhow::Result<Self> {
        Ok(match id { 0 => F32, 30 => BF16, 8 => Q8_0, 10 => Q2K, 11 => Q3K, 12 => Q4K, 13 => Q5K, 14 => Q6K,
            17 => IQ2XS, 18 => IQ3XXS, 23 => IQ4XS, _ => anyhow::bail!("unsupported quant type id {id}") })
    }
    pub fn block_elems(self) -> usize { match self { F32 | BF16 => 1, Q8_0 => 32, _ => 256 } }
    pub fn block_bytes(self) -> usize {
        match self { F32 => 4, BF16 => 2, Q8_0 => 34, Q2K => 84, Q3K => 110, Q4K => 144, Q5K => 176, Q6K => 210,
            IQ2XS => 74, IQ3XXS => 98, IQ4XS => 136 }
    }
    pub fn row_bytes(self, n: u64) -> usize { n as usize / self.block_elems() * self.block_bytes() }
    pub fn name(self) -> &'static str {
        match self { F32 => "F32", BF16 => "BF16", Q8_0 => "Q8_0", Q2K => "Q2_K", Q3K => "Q3_K", Q4K => "Q4_K",
            Q5K => "Q5_K", Q6K => "Q6_K", IQ2XS => "IQ2_XS", IQ3XXS => "IQ3_XXS", IQ4XS => "IQ4_XS" }
    }
}

#[inline] fn h(b: &[u8], o: usize) -> f32 { f16::from_le_bytes([b[o], b[o + 1]]).to_f32() }

fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 { (q[j] & 63, q[j + 4] & 63) }
    else { ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)) }
}

pub fn dequant_row(ty: GgmlType, src: &[u8], dst: &mut [f32]) {
    let (be, bb) = (ty.block_elems(), ty.block_bytes());
    for (blk, y) in src.chunks_exact(bb).zip(dst.chunks_exact_mut(be)) {
        match ty {
            F32 => y[0] = f32::from_le_bytes(blk.try_into().unwrap()),
            BF16 => y[0] = bf16::from_le_bytes([blk[0], blk[1]]).to_f32(),
            Q8_0 => { let d = h(blk, 0); for j in 0..32 { y[j] = blk[2 + j] as i8 as f32 * d; } }
            Q2K => q2k(blk, y), Q3K => q3k(blk, y), Q4K => q4k(blk, y), Q5K => q5k(blk, y), Q6K => q6k(blk, y),
            IQ2XS => iq2xs(blk, y), IQ3XXS => iq3xxs(blk, y), IQ4XS => iq4xs(blk, y),
        }
    }
}

fn q2k(b: &[u8], y: &mut [f32]) {
    let (sc, d, min) = (&b[0..16], h(b, 80), h(b, 82));
    let mut q = 16; let (mut is, mut o) = (0, 0);
    for _ in 0..2 {
        for shift in [0, 2, 4, 6] {
            for half in 0..2 {
                let s = sc[is]; is += 1;
                let (dl, ml) = (d * (s & 0xF) as f32, min * (s >> 4) as f32);
                for l in 0..16 { y[o] = dl * ((b[q + l + 16 * half] >> shift) & 3) as f32 - ml; o += 1; }
            }
        }
        q += 32;
    }
}

fn q3k(b: &[u8], y: &mut [f32]) {
    let (hm, d_all) = (&b[0..32], h(b, 108));
    let mut aux = [0u32; 4];
    for i in 0..3 { aux[i] = u32::from_le_bytes(b[96 + 4 * i..100 + 4 * i].try_into().unwrap()); }
    let (k1, k2, tmp) = (0x03030303u32, 0x0f0f0f0fu32, aux[2]);
    aux[2] = ((aux[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4);
    aux[3] = ((aux[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4);
    aux[0] = (aux[0] & k2) | ((tmp & k1) << 4);
    aux[1] = (aux[1] & k2) | (((tmp >> 2) & k1) << 4);
    let scales: Vec<i8> = aux.iter().flat_map(|a| a.to_le_bytes()).map(|x| x as i8).collect();
    let (mut q, mut is, mut o, mut m) = (32usize, 0, 0, 1u8);
    for _ in 0..2 {
        for shift in [0, 2, 4, 6] {
            for half in 0..2 {
                let dl = d_all * (scales[is] as i32 - 32) as f32; is += 1;
                for l in 0..16 {
                    let i = l + 16 * half;
                    let v = ((b[q + i] >> shift) & 3) as i32 - if hm[i] & m != 0 { 0 } else { 4 };
                    y[o] = dl * v as f32; o += 1;
                }
            }
            m <<= 1;
        }
        q += 32;
    }
}

fn q4k(b: &[u8], y: &mut [f32]) {
    let (d, min, sc) = (h(b, 0), h(b, 2), &b[4..16]);
    let (mut q, mut o) = (16, 0);
    for j in 0..4 {
        let (s1, m1) = scale_min_k4(2 * j, sc); let (s2, m2) = scale_min_k4(2 * j + 1, sc);
        let (d1, m1, d2, m2) = (d * s1 as f32, min * m1 as f32, d * s2 as f32, min * m2 as f32);
        for l in 0..32 { y[o + l] = d1 * (b[q + l] & 0xF) as f32 - m1; }
        for l in 0..32 { y[o + 32 + l] = d2 * (b[q + l] >> 4) as f32 - m2; }
        q += 32; o += 64;
    }
}

fn q5k(b: &[u8], y: &mut [f32]) {
    let (d, min, sc, qh) = (h(b, 0), h(b, 2), &b[4..16], &b[16..48]);
    let (mut q, mut o, mut u1, mut u2) = (48, 0, 1u8, 2u8);
    for j in 0..4 {
        let (s1, m1) = scale_min_k4(2 * j, sc); let (s2, m2) = scale_min_k4(2 * j + 1, sc);
        let (d1, m1, d2, m2) = (d * s1 as f32, min * m1 as f32, d * s2 as f32, min * m2 as f32);
        for l in 0..32 { y[o + l] = d1 * ((b[q + l] & 0xF) as f32 + if qh[l] & u1 != 0 { 16.0 } else { 0.0 }) - m1; }
        for l in 0..32 { y[o + 32 + l] = d2 * ((b[q + l] >> 4) as f32 + if qh[l] & u2 != 0 { 16.0 } else { 0.0 }) - m2; }
        q += 32; o += 64; u1 <<= 2; u2 <<= 2;
    }
}

fn q6k(b: &[u8], y: &mut [f32]) {
    let d = h(b, 208);
    for n in 0..2 {
        let (ql, qh, sc) = (&b[64 * n..], &b[128 + 32 * n..], &b[192 + 8 * n..]);
        let yo = 128 * n;
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
            let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
            y[yo + l] = d * sc[is] as i8 as f32 * q1 as f32;
            y[yo + l + 32] = d * sc[is + 2] as i8 as f32 * q2 as f32;
            y[yo + l + 64] = d * sc[is + 4] as i8 as f32 * q3 as f32;
            y[yo + l + 96] = d * sc[is + 6] as i8 as f32 * q4 as f32;
        }
    }
}

// IQ types: Task 4.
fn iq2xs(_: &[u8], _: &mut [f32]) { unimplemented!("Task 4") }
fn iq3xxs(_: &[u8], _: &mut [f32]) { unimplemented!("Task 4") }
fn iq4xs(_: &[u8], _: &mut [f32]) { unimplemented!("Task 4") }
```

- [ ] **Step 5: Run** `cargo test --test quant` — Expected: 9 passed.
- [ ] **Step 6: Commit** `feat: K-quant, Q8_0, BF16 dequant with gguf-py fixtures`

---

### Task 4: IQ2_XS, IQ3_XXS, IQ4_XS dequant + generated tables

**Files:**
- Create: `tools/gen_tables.py`, `src/quant/tables.rs`
- Modify: `src/quant/mod.rs` (replace the three `unimplemented!` fns), `tests/quant.rs` (3 tests)

**Interfaces:** Consumes `GgmlType`, `dequant_row`. Produces `quant::tables::{IQ2XS_GRID: [u64; 512],
IQ3XXS_GRID: [u32; 256], KSIGNS_IQ2XS: [u8; 128], KMASK_IQ2XS: [u8; 8], KVALUES_IQ4NL: [i8; 16]}` (used by CPU/CUDA
kernels later).

- [ ] **Step 1: `tools/gen_tables.py`**

```python
# Usage: python -I tools/gen_tables.py ~/llama.cpp/ggml/src/ggml-common.h > src/quant/tables.rs
import re, sys
src = open(sys.argv[1]).read()
def arr(name):
    m = re.search(r'\b' + name + r'\s*\[[^\]]*\]\s*=\s*\{(.*?)\};', src, re.S) or \
        re.search(name + r',\s*\d+\)\s*(.*?)GGML_TABLE_END', src, re.S)
    body = re.sub(r'//.*', '', m.group(1))
    return [x.strip() for x in body.replace('{', ' ').replace('}', ' ').split(',') if x.strip()]
spec = [("iq2xs_grid", "IQ2XS_GRID", "u64"), ("iq3xxs_grid", "IQ3XXS_GRID", "u32"),
        ("ksigns_iq2xs", "KSIGNS_IQ2XS", "u8"), ("kmask_iq2xs", "KMASK_IQ2XS", "u8"),
        ("kvalues_iq4nl", "KVALUES_IQ4NL", "i8")]
print("// Generated by tools/gen_tables.py from llama.cpp ggml-common.h (MIT License, ggml authors).")
for c, r, t in spec:
    v = arr(c)
    print(f"pub static {r}: [{t}; {len(v)}] = [{', '.join(v)}];")
```
Run it; Expected lengths: 512, 256, 128, 8, 16 (the script prints them in the type; `cargo build` errors if wrong).
Values may carry C suffixes like `0x...ULL` — strip `ULL|UL|U` in the script if the build complains.

- [ ] **Step 2: Add failing tests** to `tests/quant.rs`:
```rust
#[test] fn iq2_xs()  { check("IQ2_XS", GgmlType::IQ2XS) }
#[test] fn iq3_xxs() { check("IQ3_XXS", GgmlType::IQ3XXS) }
#[test] fn iq4_xs()  { check("IQ4_XS", GgmlType::IQ4XS) }
```
Run: `cargo test --test quant iq` — Expected: panics `not implemented: Task 4`.

- [ ] **Step 3: Implement** (ports of `dequantize_row_iq2_xs/iq3_xxs/iq4_xs`):

```rust
pub mod tables;
use tables::*;

fn iq2xs(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    let qs = |i: usize| u16::from_le_bytes([b[2 + 2 * i], b[3 + 2 * i]]);
    let scales = &b[66..74];
    let mut o = 0;
    for ib in 0..8 {
        let db = [d * (0.5 + (scales[ib] & 0xF) as f32) * 0.25, d * (0.5 + (scales[ib] >> 4) as f32) * 0.25];
        for l in 0..4 {
            let q = qs(4 * ib + l);
            let grid = IQ2XS_GRID[(q & 511) as usize].to_le_bytes();
            let signs = KSIGNS_IQ2XS[(q >> 9) as usize];
            for j in 0..8 {
                let s = if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                y[o + j] = db[l / 2] * grid[j] as f32 * s;
            }
            o += 8;
        }
    }
}

fn iq3xxs(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    let (qs, ss) = (&b[2..66], &b[66..98]);
    let mut o = 0;
    for ib in 0..8 {
        let aux = u32::from_le_bytes(ss[4 * ib..4 * ib + 4].try_into().unwrap());
        let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
        for l in 0..4 {
            let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
            let g1 = IQ3XXS_GRID[qs[8 * ib + 2 * l] as usize].to_le_bytes();
            let g2 = IQ3XXS_GRID[qs[8 * ib + 2 * l + 1] as usize].to_le_bytes();
            for j in 0..4 {
                y[o + j] = db * g1[j] as f32 * if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                y[o + j + 4] = db * g2[j] as f32 * if signs & KMASK_IQ2XS[j + 4] != 0 { -1.0 } else { 1.0 };
            }
            o += 8;
        }
    }
}

fn iq4xs(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    let sh = u16::from_le_bytes([b[2], b[3]]);
    let (sl, qs) = (&b[4..8], &b[8..136]);
    for ib in 0..8 {
        let ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) as i32 | ((((sh >> (2 * ib)) & 3) as i32) << 4);
        let dl = d * (ls - 32) as f32;
        let q = &qs[16 * ib..16 * ib + 16];
        for j in 0..16 {
            y[32 * ib + j] = dl * KVALUES_IQ4NL[(q[j] & 0xF) as usize] as f32;
            y[32 * ib + j + 16] = dl * KVALUES_IQ4NL[(q[j] >> 4) as usize] as f32;
        }
    }
}
```

- [ ] **Step 4: Run** `cargo test --test quant` — Expected: 12 passed.
- [ ] **Step 5: Commit** `feat: IQ2_XS, IQ3_XXS, IQ4_XS dequant`

---

### Task 5: Weights — typed tensor views and parallel matvec

**Files:**
- Create: `src/weights.rs`, add a `matvec` test to `tests/quant.rs`

**Interfaces:**
- Consumes: `Gguf`, `TensorInfo`, `GgmlType`, `dequant_row`.
- Produces:
  - `weights::Tensor<'a> { pub ty: GgmlType, pub cols: usize, pub rows: usize, pub data: &'a [u8] }`
    (`cols = dims[0]`, `rows = Π dims[1..]`; for 3-D experts rows = out × n_expert).
  - `Tensor::row(&self, r: usize, out: &mut [f32])` — dequant one row.
  - `Tensor::matvec(&self, x: &[f32], y: &mut [f32])` — `y[r] = row_r · x` for all rows, rayon-parallel in chunks of 16
    rows, each worker reuses one `Vec<f32>` row buffer.
  - `Tensor::expert(&self, e: usize, n_expert: usize) -> Tensor<'a>` — sub-view of rows `[e·rows/n, (e+1)·rows/n)`.
  - `Tensor::to_f32(&self) -> Vec<f32>` (small tensors: norms, conv, ssm_a, dt bias).
  - `weights::Weights<'a> { g: &'a Gguf }` with `get(&self, name: &str) -> anyhow::Result<Tensor<'a>>`
    (errors: `missing tensor <name>`, `tensor <name>: unsupported quant type id N`),
    `layer(&self, il: usize, suffix: &str) -> anyhow::Result<Tensor<'a>>` = `get(format!("blk.{il}.{suffix}"))`.
  - `Weights::validate(&self) -> anyhow::Result<()>` — every tensor's type is supported (load-time refusal).

- [ ] **Step 1: Failing test**

```rust
#[test]
fn matvec_matches_rowwise_dequant() {
    let Some(p) = gguf_test_model() else { return };
    let g = rusting_strata::gguf::Gguf::open(&p).unwrap();
    let w = rusting_strata::weights::Weights { g: &g };
    w.validate().unwrap();
    let t = w.layer(0, "ffn_down_exps.weight").unwrap().expert(7, 256); // IQ3_XXS, 2048 rows x 512
    assert_eq!((t.rows, t.cols), (2048, 512));
    let x: Vec<f32> = (0..512).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
    let mut y = vec![0f32; 2048]; t.matvec(&x, &mut y);
    let mut row = vec![0f32; 512];
    for r in [0, 1, 1000, 2047] {
        t.row(r, &mut row);
        let want: f32 = row.iter().zip(&x).map(|(a, b)| a * b).sum();
        assert!((y[r] - want).abs() < 1e-4, "row {r}");
    }
}
fn gguf_test_model() -> Option<std::path::PathBuf> { /* same as tests/gguf.rs model_path */ 
    let p = std::path::PathBuf::from(std::env::var("STRATA_MODEL").unwrap_or_else(|_| "/mnt/ollama/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf".into()));
    p.exists().then_some(p)
}
```

- [ ] **Step 2: Run** — Expected: compile failure (no `weights`).
- [ ] **Step 3: Implement `src/weights.rs`** per the interface. `matvec`:

```rust
pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
    use rayon::prelude::*;
    assert_eq!((x.len(), y.len()), (self.cols, self.rows));
    let rb = self.ty.row_bytes(self.cols as u64);
    y.par_chunks_mut(16).enumerate().for_each_init(|| vec![0f32; self.cols], |buf, (ci, ys)| {
        for (k, yv) in ys.iter_mut().enumerate() {
            let r = ci * 16 + k;
            dequant_row(self.ty, &self.data[r * rb..(r + 1) * rb], buf);
            *yv = buf.iter().zip(x).map(|(a, b)| a * b).sum();
        }
    });
}
```
- [ ] **Step 4: Run** `cargo test --test quant` — Expected: 13 passed.
- [ ] **Step 5: Commit** `feat: tensor views and parallel reference matvec`

---

### Task 6: Scalar ops

**Files:** Create `src/ops.rs`, `tests/ops.rs`

**Interfaces:** Produces (all in-place or out-param, f32):
`rms_norm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32])`, `l2_norm(x: &mut [f32], eps: f32)`, `silu(x: f32) -> f32`,
`sigmoid(x: f32) -> f32`, `softplus(x: f32) -> f32`, `softmax(x: &mut [f32])`,
`rope_neox(x: &mut [f32], n_rot: usize, pos: usize, base: f32)`, `top_k(p: &[f32], k: usize) -> Vec<usize>`
(descending, ties → lower index), `argmax(x: &[f32]) -> usize`.

- [ ] **Step 1: Failing tests (hand-computed values)**

```rust
use rusting_strata::ops::*;
#[test] fn rms() { let mut o = [0f32; 2]; rms_norm(&[3.0, 4.0], &[1.0, 2.0], 0.0, &mut o);
    let s = (12.5f32).sqrt(); assert!((o[0] - 3.0 / s).abs() < 1e-6 && (o[1] - 8.0 / s).abs() < 1e-6); }
#[test] fn l2() { let mut x = [3.0, 4.0]; l2_norm(&mut x, 1e-6); assert!((x[0] - 0.6).abs() < 1e-6); }
#[test] fn softplus_large() { assert!((softplus(30.0) - 30.0).abs() < 1e-5); assert!((softplus(0.0) - 2f32.ln()).abs() < 1e-6); }
#[test] fn rope_pos0_identity() { let mut x: Vec<f32> = (0..8).map(|i| i as f32).collect(); let y = x.clone();
    rope_neox(&mut x, 4, 0, 1e7); assert_eq!(x, y); }
#[test] fn rope_rotates_pairs() { let mut x = vec![1.0, 0.0, 0.0, 0.0, 9.0];  // n_rot 4: pairs (0,2),(1,3)
    rope_neox(&mut x, 4, 1, 10000.0); // theta0 = 1 rad, theta1 = 1/100 rad
    assert!((x[0] - 1f32.cos()).abs() < 1e-6 && (x[2] - 1f32.sin()).abs() < 1e-6 && x[4] == 9.0); }
#[test] fn topk() { assert_eq!(top_k(&[0.1, 0.5, 0.2, 0.5], 2), vec![1, 3]); }
```
- [ ] **Step 2: Run** — Expected: compile failure.
- [ ] **Step 3: Implement.** Rope: for `i in 0..n_rot/2`: `θ = pos · base^(−2i/n_rot)`, `(a, b) = (x[i], x[i + n_rot/2])`,
  `x[i] = a·cosθ − b·sinθ`, `x[i + n_rot/2] = a·sinθ + b·cosθ`; compute θ in f64 then cast. softplus: `if x > 20 { x }
  else { x.exp().ln_1p() }` (ggml uses the same threshold). L2: `scale = 1 / max(sqrt(Σx²), eps)`.
- [ ] **Step 4: Run** `cargo test --test ops` — Expected: 6 passed.
- [ ] **Step 5: Commit** `feat: scalar reference ops`

---

### Task 7: Tokenizer from GGUF

**Files:** Create `src/tokenizer.rs`, `tests/tokenizer.rs`, `tests/fixtures/tokens.json`

**Interfaces:** Produces `tokenizer::Tokenizer::from_gguf(&Gguf) -> anyhow::Result<Tokenizer>`,
`encode(&self, text: &str) -> anyhow::Result<Vec<u32>>` (special tokens like `<|im_start|>` recognised),
`decode(&self, ids: &[u32]) -> anyhow::Result<String>`, `eos_ids(&self) -> Vec<u32>`.

- [ ] **Step 1: Fixture from llama.cpp** — strings: `"Hello world"`, `"fn main() {\n    println!(\"hi\");\n}"`,
  `"Привіт, як справи?"`, `"<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"`, `"   spaces\t\ttabs\n\n\n"`,
  `"数字 12345 and émojis 🚀"`. For each run
  `~/llama.cpp/build-cuda/bin/llama-tokenize -m "$STRATA_MODEL" -p "<s>" --ids --log-disable --no-bos`
  and store `[{"text": ..., "ids": [...]}]` in `tests/fixtures/tokens.json` (a short python wrapper with
  `subprocess.run([...], capture_output=True)` and `ast.literal_eval` on the printed list; commit the JSON).
- [ ] **Step 2: Failing test** — for each fixture: `encode(text) == ids` and `decode(ids) == text`.
- [ ] **Step 3: Implement** with `tokenizers`:
  - vocab = `tokenizer.ggml.tokens` (index = id), merges = `tokenizer.ggml.merges` split on first `' '`.
  - `BPE::builder().vocab_and_merges(vocab_map, merges).build()`.
  - Pre-tokenizer `Sequence[Split(Regex(QWEN35_RE), Isolated, invert=false), ByteLevel(add_prefix_space=false, trim_offsets=false, use_regex=false)]`,
    `QWEN35_RE` = the regex llama.cpp uses for `qwen35` (`src/llama-vocab.cpp`, `LLAMA_VOCAB_PRE_TYPE_QWEN35`):
    `(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`.
  - Decoder ByteLevel. Tokens with `tokenizer.ggml.token_type` 3 (control) or 4 (user-defined) → `add_special_tokens`.
  - `eos_ids`: `tokenizer.ggml.eos_token_id` plus ids of `<|im_end|>` and `<|endoftext|>` if present.
- [ ] **Step 4: Run** `cargo test --test tokenizer` — Expected: pass. If onig rejects `(?!\S)`, switch the
  `tokenizers` feature to `fancy-regex` (`default-features = false, features = ["fancy-regex"]`).
- [ ] **Step 5: Commit** `feat: GGUF BPE tokenizer`

---

### Task 8: qwen35 / qwen35moe CPU reference forward

**Files:** Create `src/model.rs`, `tests/model.rs`

**Interfaces:**
- Consumes: `ModelConfig`, `Weights`, `Tensor`, ops.
- Produces:
  - `model::Model<'a> { pub cfg: ModelConfig, w: Weights<'a>, /* cached small f32 tensors per layer */ }`
    with `Model::load(g: &'a Gguf) -> anyhow::Result<Model<'a>>` (calls `validate`, pre-converts all F32/small tensors
    — norms, `ssm_a`, `ssm_dt.bias`, `ssm_conv1d`, `ssm_alpha`, `ssm_beta`, `ffn_gate_inp`, `ffn_gate_inp_shexp` — to `Vec<f32>`).
  - `model::State` — per-layer `enum LayerState { Recurrent { conv: Vec<f32> /* (k-1)·channels, oldest first */, s: Vec<f32> /* dt_rank·128·128 */ }, Attn { k: Vec<f32>, v: Vec<f32> /* pos·n_head_kv·head_dim */ } }`,
    `pos: usize`; `State::new(cfg)`.
  - `Model::step(&self, st: &mut State, token: u32) -> Vec<f32>` — runs one token, returns logits (n_vocab),
    increments `st.pos`.
  - `Model::embed(&self, token: u32) -> Vec<f32>`.

- [ ] **Step 1: Failing test** — determinism + sanity on the real file (slow, `--release` profile via `[profile.test]`):

```rust
#[test]
fn greedy_continuation_is_sane() {
    let Some(p) = model_path() else { return };
    let g = rusting_strata::gguf::Gguf::open(&p).unwrap();
    let m = rusting_strata::model::Model::load(&g).unwrap();
    let tok = rusting_strata::tokenizer::Tokenizer::from_gguf(&g).unwrap();
    let ids = tok.encode("The capital of France is").unwrap();
    let mut st = rusting_strata::model::State::new(&m.cfg);
    let mut logits = vec![];
    for &t in &ids { logits = m.step(&mut st, t); }
    let next = rusting_strata::ops::argmax(&logits) as u32;
    assert_eq!(tok.decode(&[next]).unwrap().trim(), "Paris");
}
```
- [ ] **Step 2: Run** — Expected: compile failure.
- [ ] **Step 3: Implement `step`** exactly per **Reference semantics** above. Structure:

```rust
pub fn step(&self, st: &mut State, token: u32) -> Vec<f32> {
    let c = &self.cfg;
    let mut x = self.embed(token);
    let mut h = vec![0f32; c.n_embd];
    for il in 0..c.n_layer {
        ops::rms_norm(&x, &self.l[il].attn_norm, c.eps, &mut h);
        let mix = if c.is_recurrent(il) { self.deltanet(il, &h, st) } else { self.attention(il, &h, st) };
        for (a, b) in x.iter_mut().zip(&mix) { *a += b; }
        ops::rms_norm(&x, &self.l[il].post_norm, c.eps, &mut h);
        let f = if c.is_moe() { self.moe(il, &h) } else { self.dense_ffn(il, &h) };
        for (a, b) in x.iter_mut().zip(&f) { *a += b; }
    }
    st.pos += 1;
    ops::rms_norm(&x.clone(), &self.output_norm, c.eps, &mut x);
    let out = self.w.get("output.weight").unwrap();
    let mut logits = vec![0f32; out.rows]; out.matvec(&x, &mut logits); logits
}
```
`deltanet`, `attention`, `moe`, `dense_ffn` are private methods, each a direct transcription of the formulas in
**Reference semantics** (the DeltaNet per-head loop is `for vh in 0..dt_rank { let kh = vh % n_group; ... }`; the
attention loop reads `st` KV for positions `0..=pos` after appending the current k/v). MoE experts: top-8 ids, for each
`gate = ffn_gate_exps.expert(e, n)`, `up`, `down` matvecs on 512-row slices. If `token_embd` lacks a name match
(`output.weight` missing in a tied model) → use `token_embd.weight` as the head.

- [ ] **Step 4: Run** `cargo test --test model -- --nocapture` — Expected: PASS ("Paris"). If it fails, debug with
  `~/llama.cpp/build-cuda/bin/llama-eval-callback -m $STRATA_MODEL -p "The capital of France is" -n 1 --no-warmup`,
  which prints sums of every named tensor (`attn_norm-0`, `linear_attn_out-0`, `ffn_moe_out-0`, `l_out-N`, ...);
  print the same sums from Rust for token 0 and find the first divergence.
- [ ] **Step 5: Commit** `feat: CPU reference forward for qwen35/qwen35moe`

---

### Task 9: `strata score` / `generate` CLI + 20-prompt parity gate

**Files:**
- Modify: `src/main.rs`
- Create: `tools/parity.py`, `tools/prompts.json`, `docs/m1-parity.md`

**Interfaces:**
- `strata score --model <gguf> --input <json>`: input `{"prompt": [ids], "continuation": [ids]}`; runs all prompt ids
  then continuation[..n-1]; prints JSON `{"argmax": [..], "top5": [[id, logprob]..]..]}` — one entry per position that
  predicts a continuation token (n entries).
- `strata generate --model <gguf> --prompt <text> -n <N>`: greedy, streams decoded text, prints tok/s at the end.

- [ ] **Step 1: `tools/prompts.json`** — 20 prompts: 7 code (Rust, Python, C, SQL, a regex question, a bug-fix
  request, a shell one-liner), 7 chat (general knowledge, Ukrainian, Russian, Chinese, a math word problem, a
  definition, a recipe), 6 long-doc (≥ 400 tokens each: a pasted README section, a license text, a log excerpt, a
  Wikipedia-style paragraph set, a JSON blob, a markdown table) — all wrapped in the chat template
  `<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n` (thinking off, keeps answers short).

- [ ] **Step 2: `tools/parity.py`**

```python
# Usage: python -I tools/parity.py <model.gguf> [--n 32] [--limit 20]
# Needs llama-server already running: ~/llama.cpp/build-cuda/bin/llama-server -m M -ngl 99 --n-cpu-moe 20 --port 8199
import json, subprocess, sys, urllib.request, argparse, pathlib
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--n", type=int, default=32)
ap.add_argument("--limit", type=int, default=20); a = ap.parse_args()
def post(path, body):
    r = urllib.request.Request("http://127.0.0.1:8199" + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r))
prompts = json.load(open(pathlib.Path(__file__).with_name("prompts.json")))[: a.limit]
agree = total = 0; rows = []
for p in prompts:
    ids = post("/tokenize", {"content": p, "add_special": False, "parse_special": True})["tokens"]
    r = post("/completion", {"prompt": ids, "n_predict": a.n, "temperature": 0, "cache_prompt": False,
                             "return_tokens": True, "samplers": ["top_k"], "top_k": 1})
    cont = r["tokens"]
    out = subprocess.run(["target/release/strata", "score", "--model", a.model, "--input", "-"],
                         input=json.dumps({"prompt": ids, "continuation": cont}), capture_output=True, text=True, check=True)
    mine = json.loads(out.stdout)["argmax"]
    k = sum(x == y for x, y in zip(mine, cont)); agree += k; total += len(cont)
    rows.append((p[:40].replace("\n", " "), len(ids), k, len(cont))); print(rows[-1], flush=True)
print(f"TOP1 AGREEMENT {agree}/{total} = {100 * agree / total:.2f}%")
sys.exit(0 if agree >= 0.99 * total else 1)
```

- [ ] **Step 3: Implement `score` and `generate`** in `src/main.rs` (score: logprob = `logit − logsumexp(logits)`).
- [ ] **Step 4: Build and run gate**

```bash
cargo build --release
~/llama.cpp/build-cuda/bin/llama-server -m "$STRATA_MODEL" -ngl 99 --n-cpu-moe 20 -fa on --port 8199 &
python -I tools/parity.py "$STRATA_MODEL"
```
Expected: `TOP1 AGREEMENT ... ≥ 99.00%`, exit 0. Disagreements at near-ties (top-2 logprob within 0.05 in our top5)
are expected noise from quantized-activation differences; if agreement is < 99%, find the first divergent prompt and
diff tensor sums against `llama-eval-callback` as in Task 8 Step 4.

- [ ] **Step 5: Write `docs/m1-parity.md`** (agreement table, CPU reference tok/s from `strata generate`, llama.cpp
  commit) and **commit** `feat: score/generate CLI, M1 parity gate passed`.

---

## Next plans (written after M1 lands, using its measurements)

- `m2-cuda-static-placement`: NVRTC kernels per quant type (parity vs `quant::dequant_row` and `Model::step`), fixed
  expert placement, first speed number vs `docs/baseline.md`.
- `m3-adaptive-cache-cpu-experts`, `m4-chunked-prefill`, `m5-mtp`, `m6-server-calibrate`.
