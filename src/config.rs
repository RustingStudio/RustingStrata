//! Model hyperparameters read from GGUF metadata, plus the per-architecture table.

use crate::gguf::Gguf;
use anyhow::{Context, bail, ensure};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind { Delta, Attn }

/// Standalone EAGLE3 1-layer draft head parameters.
#[derive(Clone, Debug)]
pub struct Eagle3Config {
    pub extract_layers: [usize; 3],
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub rope_base: f32,
    pub eps: f32,
    pub draft_vocab: usize,
    pub norm_before_residual: bool,
    pub rope_neox: bool,
    pub d2t: Vec<i32>,
}

/// Standalone DFlash parallel block-diffusion draft model parameters.
#[derive(Clone, Debug)]
pub struct DFlashConfig {
    pub target_layer_ids: Vec<usize>,
    pub block_size: usize,
    pub mask_token_id: u32,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub rope_base: f32,
    pub eps: f32,
    pub sliding_window: usize,
    pub layer_swa: Vec<bool>,
    pub input_embedding_scale: f32,
    pub output_multiplier: f32,
    pub conv_kernel_size: usize,
    pub conv_group_size: usize,
    pub selector_rank: usize,
    pub selector_top_k: usize,
}

impl DFlashConfig {
    pub fn is_dflash2(&self) -> bool { self.selector_rank > 0 }
}

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
    const SENSENOVA_LOOP: &[(&str, &str)] = &[
        ("mhc_attn.weight", "hc_attn_fn.weight"),
        ("mhc_attn.bias", "hc_attn_fn.bias"),
        ("mhc_ffn.weight", "hc_ffn_fn.weight"),
        ("mhc_ffn.bias", "hc_ffn_fn.bias"),
    ];
    Some(match name {
        "qwen35" | "qwen35moe" => Arch { hybrid: true, q_gate: true, k_interleave: false, aliases: &[] },
        // SenseNova-Loop: Qwen3.5 hybrid DeltaNet/Attn with shared-weight looped middle layers and mHC Sinkhorn routing.
        "sensenova_loop" => Arch { hybrid: true, q_gate: true, k_interleave: false, aliases: SENSENOVA_LOOP },
        // Qwen3.8-Flash-Next: qwen35-style DeltaNet pairing, plus hyper-connections and PLE read from metadata.
        "qwen4exp" => Arch { hybrid: true, q_gate: true, k_interleave: false, aliases: &[] },
        "qwen3next" => Arch { hybrid: true, q_gate: true, k_interleave: true, aliases: &[] },
        // Recursive Flint: Qwen-style attention with sliding windows, per-head QK norms and attention sinks.
        "flint" => Arch { hybrid: false, q_gate: false, k_interleave: false, aliases: FFN_NORM },
        "qwen3" | "qwen3moe" => Arch { hybrid: false, q_gate: false, k_interleave: false, aliases: FFN_NORM },
        _ => return None,
    })
}

/// `prism.hadamard.*` of agentionai's Gyro files. Every routed expert matrix is stored as `W diag(s) H^T` over blocks
/// of `block` input columns (H the normalized Sylvester-Walsh-Hadamard matrix, s a vector of +1 and -1), so its input
/// x has to become `H (s x)` first: the layer input before gate and up, the SwiGLU output before down. The router and
/// the shared expert see the unrotated values.
#[derive(Clone)]
pub struct Hadamard { pub block: usize, signs: Vec<Vec<f32>> }

impl std::fmt::Debug for Hadamard {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Hadamard {{ block: {}, widths: {:?} }}", self.block, self.signs.iter().map(|s| s.len()).collect::<Vec<_>>())
    }
}

impl Hadamard {
    fn from_gguf(g: &Gguf) -> anyhow::Result<Option<Hadamard>> {
        use crate::gguf::Value;
        let key = |k: &str| g.meta.get(&format!("prism.hadamard.{k}"));
        let Some(version) = key("version") else { return Ok(None) };
        let text = |k: &str| key(k).and_then(|v| v.as_str()).unwrap_or("");
        let arr = |k: &str| key(k).and_then(|v| v.as_arr()).unwrap_or(&[]);
        let block = key("block_size").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        anyhow::ensure!(version.as_u64() == Some(1) && text("transform") == "normalized-sylvester-walsh-hadamard" && text("axis") == "input-last-dimension"
            && block > 1 && block.is_power_of_two() && arr("inverse_weight_names").is_empty(),
            "unsupported prism.hadamard metadata (version {version:?}, transform {:?}, axis {:?}, block_size {block})", text("transform"), text("axis"));
        // ponytail: all routed experts rotated or none (true for the Gyro files); a table per layer if a file ever
        // rotates only some of them.
        let mut named: Vec<&str> = arr("weight_names").iter().filter_map(|v| v.as_str()).collect();
        let mut routed: Vec<&str> = g.tensors.keys().map(|n| n.as_str())
            .filter(|n| n.starts_with("blk.") && ["gate", "up", "down"].iter().any(|r| n.ends_with(&format!(".ffn_{r}_exps.weight")))).collect();
        named.sort_unstable();
        routed.sort_unstable();
        anyhow::ensure!(named == routed, "prism.hadamard.weight_names lists {} tensors, the file has {} routed expert tensors: only files with every routed expert rotated are supported", named.len(), routed.len());
        let mut widths: Vec<usize> = routed.iter().map(|n| g.tensors[*n].dims[0] as usize).collect();
        widths.sort_unstable();
        widths.dedup();
        anyhow::ensure!(widths.iter().all(|w| w % block == 0), "prism.hadamard.block_size {block} does not divide the expert widths {widths:?}");
        let signs = match text("sign_mode") {
            "identity" => widths.iter().map(|&w| vec![1f32; w]).collect(),
            "explicit" => {
                let sw: Vec<usize> = arr("sign_widths").iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect();
                let sv: Vec<f32> = arr("sign_values").iter().map(|v| match *v {
                    Value::I8(v) => v as f32, Value::I16(v) => v as f32, Value::I32(v) => v as f32, Value::I64(v) => v as f32,
                    _ => v.as_f32().unwrap_or(0.0),
                }).collect();
                anyhow::ensure!(sw.iter().sum::<usize>() == sv.len() && sv.iter().all(|v| v.abs() == 1.0), "prism.hadamard sign_widths / sign_values are inconsistent");
                let mut at = 0;
                let all: Vec<Vec<f32>> = sw.iter().map(|&w| { at += w; sv[at - w..at].to_vec() }).collect();
                anyhow::ensure!(widths.iter().all(|w| all.iter().any(|s| s.len() == *w)), "prism.hadamard has no sign vector for one of the widths {widths:?}");
                all
            }
            mode => bail!("unsupported prism.hadamard.sign_mode {mode:?}"),
        };
        Ok(Some(Hadamard { block, signs }))
    }

    /// The signs for an input of `n` values.
    pub fn signs(&self, n: usize) -> &[f32] {
        self.signs.iter().find(|s| s.len() == n).unwrap_or_else(|| panic!("no prism.hadamard signs for width {n}"))
    }

    /// `x` becomes `H (s x)`, block by block.
    pub fn rotate(&self, x: &mut [f32]) {
        let signs = self.signs(x.len());
        for (v, s) in x.iter_mut().zip(signs) { *v *= s; }
        Self::fwht(x, self.block);
    }

    /// `x` becomes `H x` per `block` values: the fast Walsh-Hadamard transform in natural (Sylvester) order, where
    /// entry (r, c) of H is (-1)^popcount(r & c) / sqrt(block).
    fn fwht(x: &mut [f32], block: usize) {
        let scale = 1.0 / (block as f32).sqrt();
        for v in x.chunks_exact_mut(block) {
            let mut len = 1;
            while len < block {
                for i in (0..block).step_by(2 * len) {
                    for j in i..i + len { let (a, c) = (v[j], v[j + len]); v[j] = a + c; v[j + len] = a - c; }
                }
                len *= 2;
            }
            for a in v { *a *= scale; }
        }
    }

    pub fn rotated(&self, x: &[f32]) -> Vec<f32> { let mut y = x.to_vec(); self.rotate(&mut y); y }

    /// The inverse of `rotate`: H is its own inverse, so `x` becomes `s (H x)`.
    pub fn unrotate(&self, x: &mut [f32]) {
        let signs = self.signs(x.len());
        Self::fwht(x, self.block);
        for (v, s) in x.iter_mut().zip(signs) { *v *= s; }
    }
}

#[test]
fn hadamard_rotation_is_the_matrix_product() {
    let h = Hadamard { block: 8, signs: vec![(0..16).map(|i| if i % 3 == 0 { -1.0 } else { 1.0 }).collect()] };
    let x: Vec<f32> = (0..16).map(|i| (i * i) as f32 - 20.0).collect();
    let y = h.rotated(&x);
    for r in 0..16 {
        let want: f32 = (0..8).map(|c| { let i: usize = r / 8 * 8 + c; (if (r % 8 & c as usize).count_ones() % 2 == 1 { -1.0 } else { 1.0 }) * h.signs(16)[i] * x[i] }).sum::<f32>() / 8f32.sqrt();
        assert!((y[r] - want).abs() < 1e-4, "{r}: {} vs {want}", y[r]);
    }
    let mut back = y;
    h.unrotate(&mut back);
    assert!(back.iter().zip(&x).all(|(a, b)| (a - b).abs() < 1e-4), "unrotate: {back:?} vs {x:?}");
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
    /// Qwen3-VL/3.5 M-RoPE section sizes (sum = `n_rot/2`); `None` uses 1-D RoPE.
    pub mrope_sections: Option<[usize; 4]>,
    pub rope_base: f32,
    pub eps: f32,
    pub n_vocab: usize,
    /// Dense FFN size (0 for MoE).
    pub n_ff: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub n_ff_shexp: usize,
    /// Leading layers with a dense FFN; the rest are MoE (0 when the whole model is MoE).
    pub n_dense: usize,
    /// 0 = no extra scale.
    pub expert_weights_scale: f32,
    /// Router activation is sigmoid (Flint) instead of softmax.
    pub router_sigmoid: bool,
    /// The rotation of the routed experts' inputs (Gyro files), if the file asks for one.
    pub hadamard: Option<Hadamard>,
    /// Sliding-window width for the layers flagged in `swa` (0 = all layers are dense).
    pub n_swa: usize,
    /// Per-layer sliding-window attention (Flint). Empty for architectures without it.
    pub swa: Vec<bool>,
    /// Per-layer RoPE; Flint's full-attention layers use no positional encoding.
    pub rope_layer: Vec<bool>,
    /// Per-head learned attention sinks (`blk.N.attn_sinks.weight`).
    pub attn_sinks: bool,
    /// DeltaNet sizes, 0 without DeltaNet layers.
    pub ssm_d_inner: usize,
    pub ssm_d_state: usize,
    pub ssm_n_group: usize,
    pub ssm_dt_rank: usize,
    pub ssm_conv: usize,
    /// Hyper-connection residual streams and their low rank (qwen4exp; 0 = plain residual).
    pub hc_n: usize,
    pub hc_rank: usize,
    /// SenseNova-Loop manifold-constrained hyper-connection (mHC) stream count (0 = none) and per-layer phase flags
    /// (`0` = plain, `1` = enter, `4` = carry, `2` = leave, `3` = enter+leave).
    pub mhc_streams: usize,
    pub mhc_layer_kinds: Vec<u8>,
    /// Physical weight layer index for each virtual execution layer `0..n_layer` (`phys_layer[il] == il` unless looped).
    pub phys_layer: Vec<usize>,
    /// Per-layer (n-gram) embedding width (qwen4exp; 0 = none).
    pub n_embd_ple: usize,
    /// QSA sparse attention indexer (qwen4exp): heads, head size, and the number of cells kept per row
    /// (0 = dense attention).
    pub idx_n_head: usize,
    pub idx_head_dim: usize,
    pub idx_top_k: usize,
    /// Mixer of each main layer (MTP layers are attention).
    pub layers: Vec<LayerKind>,
    pub q_gate: bool,
    pub k_interleave: bool,
    pub aliases: &'static [(&'static str, &'static str)],
    pub eagle3: Option<Eagle3Config>,
    pub dflash: Option<DFlashConfig>,
}

/// Architectures with their own forward and server outside `ModelConfig` (`src/main.rs` dispatches on them).
pub const OWN_ARCHS: [&str; 6] =
    ["gemma4", "glm5-next", "deepseek41", crate::hy4::ARCH, crate::inkling2::ARCH, crate::inkling2::ARCH_LEGACY];

impl ModelConfig {
    pub fn from_gguf(g: &Gguf) -> anyhow::Result<ModelConfig> {
        let arch = g.get("general.architecture")?.as_str().context("general.architecture")?.to_string();
        let Some(a) = self::arch(&arch) else {
            ensure!(!OWN_ARCHS.contains(&arch.as_str()), "'{arch}' models run with chat, generate, bench and serve only");
            bail!("unsupported architecture '{arch}' (supported: qwen3, qwen3moe, qwen3next, qwen35, qwen35moe, qwen4exp, flint, sensenova_loop)")
        };
        let opt = |k: &str| g.meta.get(&format!("{arch}.{k}")).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let req = |k: &str| -> anyhow::Result<usize> {
            let key = format!("{arch}.{k}");
            Ok(g.get(&key)?.as_u64().with_context(|| format!("{key}: not an integer"))? as usize)
        };
        let f = |k: &str| g.meta.get(&format!("{arch}.{k}")).and_then(|v| v.as_f32());
        let n_nextn = opt("nextn_predict_layers");
        let n_layer = req("block_count")? - n_nextn;
        let flags = |k: &str| -> anyhow::Result<Option<Vec<bool>>> {
            let key = format!("{arch}.{k}");
            let Some(v) = g.meta.get(&key) else { return Ok(None) };
            let a = v.as_arr().with_context(|| format!("{key}: not an array"))?;
            anyhow::ensure!(a.len() >= n_layer, "{key} has {} entries for {n_layer} layers", a.len());
            Ok(Some(a[..n_layer].iter().map(|v| match v {
                crate::gguf::Value::Bool(b) => *b,
                _ => v.as_u64().unwrap_or(0) != 0,
            }).collect()))
        };
        let layers: Vec<LayerKind> = if let Some(rec) = flags("attention.recurrent_layers")? {
            rec.into_iter().map(|r| if r { LayerKind::Delta } else { LayerKind::Attn }).collect()
        } else if a.hybrid {
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
        let n_dense = opt("leading_dense_block_count");
        anyhow::ensure!(n_dense <= n_layer, "{n_dense} leading dense blocks for {n_layer} layers");
        let mhc_streams = opt("mhc.streams");
        let loop_start = opt("loop.start");
        let loop_end = opt("loop.end");
        let loop_count = opt("loop.count").max(1);
        let mhc_layer_kinds: Vec<u8> = match g.meta.get(&format!("{arch}.mhc.layer_kinds")).and_then(|v| v.as_arr()) {
            Some(arr) if arr.len() >= n_layer => arr[..n_layer].iter().map(|v| v.as_u64().unwrap_or(0) as u8).collect(),
            _ if mhc_streams > 0 && opt("mhc.end") > opt("mhc.start") => {
                let (ms, me) = (opt("mhc.start"), opt("mhc.end"));
                let total_mhc = (ms + (me - ms) * loop_count).min(n_layer);
                (0..n_layer).map(|il| {
                    if il < ms || il >= total_mhc { 0 }
                    else if il == ms { 1 }
                    else if il + 1 == total_mhc { 2 }
                    else { 4 }
                }).collect()
            }
            _ => Vec::new(),
        };
        anyhow::ensure!(mhc_streams == 0 || matches!(mhc_streams, 2 | 4), "unsupported mhc.streams {mhc_streams} (expected 2 or 4)");
        let has_blk = |il: usize| {
            g.tensors.contains_key(&format!("blk.{il}.attn_norm.weight"))
                || g.tensors.contains_key(&format!("blk.{il}.hc_attn_norm.weight"))
        };
        let loop_span = if loop_end > loop_start {
            Some((loop_start, loop_end))
        } else if !mhc_layer_kinds.is_empty() {
            let s = mhc_layer_kinds.iter().position(|&k| k & 1 != 0);
            let e = s.and_then(|st| mhc_layer_kinds[st..].iter().position(|&k| k & 2 != 0).map(|o| st + o + 1));
            s.zip(e).filter(|&(st, en)| en > st)
        } else {
            None
        };
        let first_missing = (0..n_layer).find(|&il| !has_blk(il));
        let phys_layer: Vec<usize> = (0..n_layer).map(|il| {
            if has_blk(il) {
                il
            } else if let Some((st, en)) = loop_span {
                st + (il.saturating_sub(st)) % (en - st)
            } else if let Some(fm) = first_missing.filter(|&fm| fm > 0) {
                let miss_end = (fm..n_layer).find(|&j| has_blk(j)).unwrap_or(n_layer);
                let span = (miss_end - fm).min(fm).max(1);
                (fm - span) + (il - fm) % span
            } else {
                il
            }
        }).collect();
        let swa = flags("attention.sliding_window_pattern")?.unwrap_or_else(|| vec![false; n_layer]);
        let rope_layer = flags("attention.rope_pattern")?.unwrap_or_else(|| vec![true; n_layer]);
        let n_swa = opt("attention.sliding_window");
        anyhow::ensure!(!swa.contains(&true) || n_swa > 0, "sliding-window layers without attention.sliding_window");
        let emb = g.tensors.get("token_embd.weight").context("missing tensor token_embd.weight")?;
        let c = ModelConfig {
            n_layer,
            n_nextn,
            n_embd,
            n_head,
            n_head_kv: req("attention.head_count_kv")?,
            head_dim,
            n_rot: match opt("rope.dimension_count") { 0 => head_dim, v => v },
            mrope_sections: g.meta.get(&format!("{arch}.rope.dimension_sections")).and_then(|v| v.as_arr()).and_then(|a| {
                if a.len() < 4 { return None; }
                let mut s = [0usize; 4];
                for (o, v) in s.iter_mut().zip(a) { *o = v.as_u64()? as usize; }
                Some(s)
            }),
            rope_base: f("rope.freq_base").unwrap_or(10000.0),
            eps: f("attention.layer_norm_rms_epsilon").context("missing rms epsilon")?,
            n_vocab: emb.dims[1] as usize,
            n_ff: opt("feed_forward_length"),
            n_expert,
            n_expert_used: opt("expert_used_count"),
            n_ff_exp: opt("expert_feed_forward_length"),
            n_ff_shexp: opt("expert_shared_feed_forward_length"),
            n_dense,
            expert_weights_scale: f("expert_weights_scale").unwrap_or(0.0),
            router_sigmoid: arch == "flint" && opt("expert_gating_func") == 2,
            hadamard: Hadamard::from_gguf(g)?,
            n_swa,
            swa,
            rope_layer,
            attn_sinks: g.tensors.contains_key("blk.0.attn_sinks.weight"),
            ssm_d_inner: ssm("ssm.inner_size")?,
            ssm_d_state: ssm("ssm.state_size")?,
            ssm_n_group: ssm("ssm.group_count")?,
            ssm_dt_rank: ssm("ssm.time_step_rank")?,
            ssm_conv: ssm("ssm.conv_kernel")?,
            hc_n: opt("hyper_connection.count"),
            hc_rank: opt("hyper_connection.low_rank"),
            mhc_streams,
            mhc_layer_kinds,
            phys_layer,
            n_embd_ple: opt("embedding_length_per_layer_input"),
            idx_n_head: opt("attention.indexer.head_count"),
            idx_head_dim: opt("attention.indexer.key_length"),
            idx_top_k: opt("attention.indexer.top_k"),
            layers,
            q_gate: a.q_gate,
            k_interleave: a.k_interleave,
            aliases: a.aliases,
            arch,
            eagle3: None,
            dflash: None,
        };
        // Kernels divide by these and index value heads by nv / nk key heads.
        let (nk, nv) = (c.ssm_n_group, c.ssm_dt_rank);
        anyhow::ensure!(!has_delta || (nk > 0 && nv > 0 && nv % nk == 0 && c.ssm_d_inner % nv == 0),
            "bad ssm dims: group_count {nk}, time_step_rank {nv}, inner_size {}", c.ssm_d_inner);
        Ok(c)
    }

    pub fn is_moe(&self) -> bool { self.n_expert > 0 }
    /// Layer `il` uses the MoE FFN (dense-lead architectures start with `n_dense` dense layers).
    pub fn is_moe_layer(&self, il: usize) -> bool { self.n_expert > 0 && il >= self.n_dense }
    /// DeltaNet value head size (0 without DeltaNet layers).
    pub fn ssm_dv(&self) -> usize { self.ssm_d_inner / self.ssm_dt_rank.max(1) }
}
