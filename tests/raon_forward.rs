//! End-to-end tests for KRAFTON Raon-1 mini (`raon1`):
//! 1. Synthetic 2-layer (1 RDA + 1 MLA, dense + MoE + shared + effort steering) GGUF built in memory:
//!    verifies `RaonConfig::from_gguf`, `RaonModel::new`, chunked `prefill` vs token-by-token `step` parity,
//!    reasoning effort steering, and `reset()` determinism.
//! 2. Optional real-model test gated by `STRATA_RAON_MODEL`.
use rusting_strata::{gguf::Gguf, raon::forward::RaonModel, tokenizer::Tokenizer};
use std::io::Write;

fn write_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn kv_u32(buf: &mut Vec<u8>, k: &str, v: u32) {
    write_str(buf, k);
    buf.extend_from_slice(&4u32.to_le_bytes());
    buf.extend_from_slice(&v.to_le_bytes());
}

fn kv_f32(buf: &mut Vec<u8>, k: &str, v: f32) {
    write_str(buf, k);
    buf.extend_from_slice(&6u32.to_le_bytes());
    buf.extend_from_slice(&v.to_le_bytes());
}

fn kv_bool(buf: &mut Vec<u8>, k: &str, v: bool) {
    write_str(buf, k);
    buf.extend_from_slice(&7u32.to_le_bytes());
    buf.push(u8::from(v));
}

fn kv_str(buf: &mut Vec<u8>, k: &str, v: &str) {
    write_str(buf, k);
    buf.extend_from_slice(&8u32.to_le_bytes());
    write_str(buf, v);
}

fn kv_u32_arr(buf: &mut Vec<u8>, k: &str, vs: &[u32]) {
    write_str(buf, k);
    buf.extend_from_slice(&9u32.to_le_bytes());
    buf.extend_from_slice(&4u32.to_le_bytes());
    buf.extend_from_slice(&(vs.len() as u64).to_le_bytes());
    for &v in vs { buf.extend_from_slice(&v.to_le_bytes()); }
}

fn kv_f32_arr(buf: &mut Vec<u8>, k: &str, vs: &[f32]) {
    write_str(buf, k);
    buf.extend_from_slice(&9u32.to_le_bytes());
    buf.extend_from_slice(&6u32.to_le_bytes());
    buf.extend_from_slice(&(vs.len() as u64).to_le_bytes());
    for &v in vs { buf.extend_from_slice(&v.to_le_bytes()); }
}

fn kv_str_arr(buf: &mut Vec<u8>, k: &str, vs: &[&str]) {
    write_str(buf, k);
    buf.extend_from_slice(&9u32.to_le_bytes());
    buf.extend_from_slice(&8u32.to_le_bytes());
    buf.extend_from_slice(&(vs.len() as u64).to_le_bytes());
    for s in vs { write_str(buf, s); }
}

/// Build a tiny valid `raon1` GGUF file on disk with 2 layers (layer 0 = RDA + Dense, layer 1 = MLA + MoE + effort steering).
fn build_synthetic_raon_gguf(path: &std::path::Path) {
    let (n_layer, n_embd, n_vocab, n_head) = (2usize, 32usize, 48usize, 2usize);
    let (d_conv, dt_rank, rda_hd) = (4usize, 8usize, 16usize);
    let d_inner = n_head * rda_hd; // 32
    let (q_rank, kv_rank, rope_hd, mla_kl, mla_vl) = (16usize, 16usize, 8usize, 24usize, 16usize);
    let nope = mla_kl - rope_hd; // 16
    let (hc, n_ff, n_ff_exp, n_exp, n_used) = (4usize, 48usize, 16usize, 4usize, 2usize);
    let hc_mix = 2 * hc + (hc - 1) * (hc - 1); // 17

    let mut kv = Vec::new();
    let mut n_kv = 0u64;
    macro_rules! add_kv { ($f:ident, $k:expr, $v:expr) => { $f(&mut kv, $k, $v); n_kv += 1; }; }
    add_kv!(kv_str, "general.architecture", "raon1");
    add_kv!(kv_u32, "raon1.block_count", n_layer as u32);
    add_kv!(kv_u32, "raon1.context_length", 128);
    add_kv!(kv_u32, "raon1.embedding_length", n_embd as u32);
    add_kv!(kv_u32, "raon1.feed_forward_length", n_ff as u32);
    add_kv!(kv_u32, "raon1.attention.head_count", n_head as u32);
    add_kv!(kv_u32_arr, "raon1.attention.head_count_kv", &[0, 1]);
    add_kv!(kv_f32, "raon1.attention.layer_norm_rms_epsilon", 1e-5);
    add_kv!(kv_f32, "raon1.attention.latent_norm_rms_epsilon", 1e-6);
    add_kv!(kv_u32, "raon1.ssm.conv_kernel", d_conv as u32);
    add_kv!(kv_u32, "raon1.ssm.time_step_rank", dt_rank as u32);
    add_kv!(kv_u32, "raon1.kda.head_dim", rda_hd as u32);
    add_kv!(kv_f32, "raon1.kda.gate_lower_bound", -5.0);
    add_kv!(kv_u32, "raon1.attention.q_lora_rank", q_rank as u32);
    add_kv!(kv_u32, "raon1.attention.kv_lora_rank", kv_rank as u32);
    add_kv!(kv_u32, "raon1.rope.dimension_count", rope_hd as u32);
    add_kv!(kv_u32, "raon1.attention.key_length_mla", mla_kl as u32);
    add_kv!(kv_u32, "raon1.attention.value_length_mla", mla_vl as u32);
    add_kv!(kv_u32, "raon1.expert_count", n_exp as u32);
    add_kv!(kv_u32, "raon1.expert_used_count", n_used as u32);
    add_kv!(kv_u32, "raon1.expert_feed_forward_length", n_ff_exp as u32);
    add_kv!(kv_u32, "raon1.expert_shared_count", 1);
    add_kv!(kv_u32, "raon1.leading_dense_block_count", 1);
    add_kv!(kv_f32, "raon1.expert_weights_scale", 1.0);
    add_kv!(kv_bool, "raon1.expert_weights_norm", true);
    add_kv!(kv_f32, "raon1.activation.situ_beta", 4.0);
    add_kv!(kv_f32, "raon1.activation.situ_linear_beta", 25.0);
    add_kv!(kv_u32, "raon1.hyper_connection.count", hc as u32);
    add_kv!(kv_f32, "raon1.hyper_connection.epsilon", 1e-5);
    add_kv!(kv_f32, "raon1.hyper_connection.magnitude", 2.0);
    add_kv!(kv_str_arr, "raon1.reasoning_effort.names", &["none", "low", "high"]);
    add_kv!(kv_f32_arr, "raon1.reasoning_effort.alpha", &[0.0, 0.3, 0.05]);
    add_kv!(kv_str, "raon1.reasoning_effort.default", "high");
    add_kv!(kv_u32_arr, "raon1.reasoning_effort.layers", &[1]);

    let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
    let mut seed = 1u32;
    let mut add_t = |name: String, dims: Vec<u64>, scale: f32, bias: f32| {
        let len: usize = dims.iter().map(|&d| d as usize).product();
        let vals: Vec<f32> = (0..len).map(|i| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let u = ((seed >> 8) as f32) / ((1u32 << 24) as f32) * 2.0 - 1.0;
            bias + scale * u + 0.001 * ((i % 7) as f32 - 3.0)
        }).collect();
        tensors.push((name, dims, vals));
    };

    add_t("token_embd.weight".into(), vec![n_embd as u64, n_vocab as u64], 0.2, 0.0);
    add_t("output_norm.weight".into(), vec![n_embd as u64], 0.05, 1.0);
    add_t("output.weight".into(), vec![n_embd as u64, n_vocab as u64], 0.15, 0.0);

    for il in 0..n_layer {
        add_t(format!("blk.{il}.attn_norm.weight"), vec![n_embd as u64], 0.05, 1.0);
        add_t(format!("blk.{il}.ffn_norm.weight"), vec![n_embd as u64], 0.05, 1.0);
        for site in ["hc_attn", "hc_ffn"] {
            add_t(format!("blk.{il}.{site}_fn.weight"), vec![(hc * n_embd) as u64, hc_mix as u64], 0.05, 0.0);
            add_t(format!("blk.{il}.{site}_base.weight"), vec![hc_mix as u64], 0.1, 0.0);
            add_t(format!("blk.{il}.{site}_scale.weight"), vec![4], 0.1, 0.2);
        }
        if il == 0 {
            // RDA + Dense
            for nm in ["attn_q", "attn_k", "attn_v", "ssm_g"] {
                add_t(format!("blk.{il}.{nm}.weight"), vec![n_embd as u64, d_inner as u64], 0.1, 0.0);
            }
            for nm in ["ssm_conv1d_q", "ssm_conv1d_k", "ssm_conv1d_v"] {
                add_t(format!("blk.{il}.{nm}.weight"), vec![d_conv as u64, 1, d_inner as u64, 1], 0.2, 0.0);
            }
            add_t(format!("blk.{il}.ssm_f_a.weight"), vec![n_embd as u64, dt_rank as u64], 0.1, 0.0);
            add_t(format!("blk.{il}.ssm_f_b.weight"), vec![dt_rank as u64, d_inner as u64], 0.1, 0.0);
            add_t(format!("blk.{il}.ssm_beta.weight"), vec![n_embd as u64, n_head as u64], 0.1, 0.0);
            add_t(format!("blk.{il}.ssm_rda.weight"), vec![n_embd as u64, n_head as u64], 0.1, 0.0);
            add_t(format!("blk.{il}.ssm_a"), vec![n_head as u64], 0.1, -1.0);
            add_t(format!("blk.{il}.ssm_dt.bias"), vec![d_inner as u64], 0.1, 0.0);
            add_t(format!("blk.{il}.ssm_norm.weight"), vec![rda_hd as u64], 0.05, 1.0);
            add_t(format!("blk.{il}.attn_output.weight"), vec![d_inner as u64, n_embd as u64], 0.1, 0.0);

            add_t(format!("blk.{il}.ffn_gate.weight"), vec![n_embd as u64, n_ff as u64], 0.1, 0.0);
            add_t(format!("blk.{il}.ffn_up.weight"), vec![n_embd as u64, n_ff as u64], 0.1, 0.0);
            add_t(format!("blk.{il}.ffn_down.weight"), vec![n_ff as u64, n_embd as u64], 0.1, 0.0);
        } else {
            // MLA + MoE + shared + effort steering
            add_t(format!("blk.{il}.attn_q_a.weight"), vec![n_embd as u64, q_rank as u64], 0.1, 0.0);
            add_t(format!("blk.{il}.attn_q_a_norm.weight"), vec![q_rank as u64], 0.05, 1.0);
            add_t(format!("blk.{il}.attn_q_b.weight"), vec![q_rank as u64, (n_head * mla_kl) as u64], 0.1, 0.0);
            add_t(format!("blk.{il}.attn_kv_a_mqa.weight"), vec![n_embd as u64, (kv_rank + rope_hd) as u64], 0.1, 0.0);
            add_t(format!("blk.{il}.attn_kv_a_norm.weight"), vec![kv_rank as u64], 0.05, 1.0);
            add_t(format!("blk.{il}.attn_k_b.weight"), vec![nope as u64, kv_rank as u64, n_head as u64], 0.1, 0.0);
            add_t(format!("blk.{il}.attn_v_b.weight"), vec![kv_rank as u64, mla_vl as u64, n_head as u64], 0.1, 0.0);
            add_t(format!("blk.{il}.attn_gate.weight"), vec![n_embd as u64, (n_head * mla_vl) as u64], 0.1, 0.0);
            add_t(format!("blk.{il}.attn_output.weight"), vec![(n_head * mla_vl) as u64, n_embd as u64], 0.1, 0.0);

            add_t(format!("blk.{il}.ffn_gate_inp.weight"), vec![n_embd as u64, n_exp as u64], 0.1, 0.0);
            add_t(format!("blk.{il}.exp_probs_b.bias"), vec![n_exp as u64], 0.05, 0.0);
            add_t(format!("blk.{il}.ffn_gate_exps.weight"), vec![n_embd as u64, n_ff_exp as u64, n_exp as u64], 0.1, 0.0);
            add_t(format!("blk.{il}.ffn_up_exps.weight"), vec![n_embd as u64, n_ff_exp as u64, n_exp as u64], 0.1, 0.0);
            add_t(format!("blk.{il}.ffn_down_exps.weight"), vec![n_ff_exp as u64, n_embd as u64, n_exp as u64], 0.1, 0.0);
            add_t(format!("blk.{il}.ffn_gate_shexp.weight"), vec![n_embd as u64, n_ff_exp as u64], 0.1, 0.0);
            add_t(format!("blk.{il}.ffn_up_shexp.weight"), vec![n_embd as u64, n_ff_exp as u64], 0.1, 0.0);
            add_t(format!("blk.{il}.ffn_down_shexp.weight"), vec![n_ff_exp as u64, n_embd as u64], 0.1, 0.0);
            add_t(format!("blk.{il}.ffn_effort_steer.weight"), vec![n_embd as u64], 0.5, 0.2);
        }
    }

    let mut hdr = Vec::new();
    hdr.extend_from_slice(b"GGUF");
    hdr.extend_from_slice(&3u32.to_le_bytes());
    hdr.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    hdr.extend_from_slice(&n_kv.to_le_bytes());
    hdr.extend_from_slice(&kv);

    let mut data_buf = Vec::new();
    for (name, dims, vals) in &tensors {
        let pad = (32 - (data_buf.len() % 32)) % 32;
        data_buf.resize(data_buf.len() + pad, 0);
        let off = data_buf.len() as u64;
        write_str(&mut hdr, name);
        hdr.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for &d in dims { hdr.extend_from_slice(&d.to_le_bytes()); }
        hdr.extend_from_slice(&0u32.to_le_bytes()); // F32
        hdr.extend_from_slice(&off.to_le_bytes());
        for &v in vals { data_buf.extend_from_slice(&v.to_le_bytes()); }
    }
    let pad = (32 - (hdr.len() % 32)) % 32;
    hdr.resize(hdr.len() + pad, 0);

    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&hdr).unwrap();
    f.write_all(&data_buf).unwrap();
}

#[test]
fn synthetic_raon_chunked_prefill_matches_step_and_effort_steers() {
    let dir = std::env::temp_dir().join(format!("strata_raon_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tiny_raon.gguf");
    build_synthetic_raon_gguf(&path);

    let g = Gguf::open(&path).unwrap();
    let mut m = RaonModel::new(&g).unwrap();
    assert_eq!(m.cfg.n_layer, 2);
    assert_eq!(m.cfg.is_rda, vec![true, false]);

    let prompt = [3u32, 7, 11, 2, 19, 5, 23, 1];
    // 1. Chunked prefill
    let logits_chunk = m.prefill(&prompt);
    assert_eq!(logits_chunk.len(), 48);
    assert!(logits_chunk.iter().all(|v| v.is_finite()));

    // 2. Reset + token-by-token step
    m.reset();
    let mut logits_seq = Vec::new();
    for &t in &prompt {
        logits_seq = m.step(t);
    }
    let max_diff = logits_chunk.iter().zip(&logits_seq).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_diff < 1e-4, "chunked prefill vs step max_diff={max_diff}");

    // 3. Effort steering changes logits when alpha changes ("none" = 0.0 vs "low" = 0.3)
    m.reset();
    m.set_effort(Some("none"), false);
    let logits_none = m.prefill(&prompt);
    m.reset();
    m.set_effort(Some("low"), true);
    let logits_low = m.prefill(&prompt);
    let steer_diff = logits_none.iter().zip(&logits_low).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(steer_diff > 1e-3, "effort steering had no effect: diff={steer_diff}");

    // 4. Speculative verify_greedy with partial mismatch must match token-by-token step bit-for-bit.
    m.reset();
    m.set_effort(Some("none"), false);
    let l_base = m.prefill(&prompt[..4]);
    let mut m_step = RaonModel::new(&g).unwrap();
    m_step.set_effort(Some("none"), false);
    let _ = m_step.prefill(&prompt[..4]);
    let anchor = l_base.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
    let l1 = m_step.step(anchor);
    let t1 = l1.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
    let l2 = m_step.step(t1);
    let wrong = (t1 + 1) % 48;
    let (acc, l_spec) = m.verify_greedy(anchor, &[t1, wrong, 0], &[]);
    assert_eq!(acc, vec![t1], "should accept matching first draft token and stop at mismatch");
    let spec_diff = l2.iter().zip(&l_spec).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(spec_diff < 1e-4, "verify_greedy rollback logits differ from step: {spec_diff}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn real_raon_model_generates_coherent_tokens() {
    let Ok(p) = std::env::var("STRATA_RAON_MODEL") else { eprintln!("STRATA_RAON_MODEL unset, skipping"); return };
    let g = Gguf::open(std::path::Path::new(&p)).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let mut m = RaonModel::new(&g).unwrap();
    let ids = tok.encode("The capital of France is").unwrap();
    let logits = m.prefill(&ids);
    assert!(logits.iter().all(|v| v.is_finite()));
    let top = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
    let piece = tok.decode(&[top]).unwrap();
    assert!(!piece.trim().is_empty(), "empty top token");
}
