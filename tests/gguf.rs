mod common;
use rusting_strata::{config::{LayerKind, ModelConfig}, gguf::Gguf, weights::Weights};

fn w_str(b: &mut Vec<u8>, s: &str) {
    b.extend((s.len() as u64).to_le_bytes());
    b.extend(s.as_bytes());
}

#[test]
fn parses_synthetic_file() {
    let mut b = Vec::new();
    b.extend(b"GGUF");
    b.extend(3u32.to_le_bytes());
    b.extend(1u64.to_le_bytes()); // tensors
    b.extend(2u64.to_le_bytes()); // kv
    w_str(&mut b, "general.architecture");
    b.extend(8u32.to_le_bytes());
    w_str(&mut b, "qwen35moe");
    w_str(&mut b, "qwen35moe.block_count");
    b.extend(4u32.to_le_bytes());
    b.extend(41u32.to_le_bytes());
    w_str(&mut b, "t");
    b.extend(2u32.to_le_bytes());
    b.extend(4u64.to_le_bytes());
    b.extend(2u64.to_le_bytes());
    b.extend(0u32.to_le_bytes()); // F32
    b.extend(0u64.to_le_bytes()); // offset
    while b.len() % 32 != 0 { b.push(0); }
    for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] { b.extend(v.to_le_bytes()); }
    let p = std::env::temp_dir().join(format!("strata-{}-syn.gguf", std::process::id()));
    std::fs::write(&p, &b).unwrap();
    let g = Gguf::open(&p).unwrap();
    std::fs::remove_file(&p).ok();
    assert_eq!(g.get("general.architecture").unwrap().as_str(), Some("qwen35moe"));
    assert_eq!(g.get("qwen35moe.block_count").unwrap().as_u64(), Some(41));
    let t = &g.tensors["t"];
    assert_eq!(t.dims, vec![4, 2]);
    let bytes = g.bytes(t);
    assert_eq!(bytes.len(), 32);
    assert_eq!(f32::from_le_bytes(bytes[28..32].try_into().unwrap()), 8.0);
}

#[test]
fn real_file_config() {
    let Some(p) = common::model_path() else { return };
    let g = Gguf::open(&p).unwrap();
    let c = ModelConfig::from_gguf(&g).unwrap();
    assert_eq!(c.arch, "qwen35moe");
    assert_eq!((c.n_layer, c.n_nextn, c.n_embd, c.n_head, c.n_head_kv, c.head_dim), (40, 1, 2048, 16, 2, 256));
    assert_eq!((c.n_expert, c.n_expert_used, c.n_ff_exp, c.n_ff_shexp), (256, 8, 512, 512));
    assert_eq!((c.ssm_d_inner, c.ssm_d_state, c.ssm_n_group, c.ssm_dt_rank, c.ssm_conv), (4096, 128, 16, 32, 4));
    assert_eq!((c.n_rot, c.n_vocab), (64, 248320));
    assert_eq!(&c.layers[..4], &[LayerKind::Delta, LayerKind::Delta, LayerKind::Delta, LayerKind::Attn]);
    assert!(c.q_gate && !c.k_interleave);
    // IQ3_XXS: 98 bytes per 256 elements; [512, 2048, 256].
    assert_eq!(g.bytes(&g.tensors["blk.0.ffn_down_exps.weight"]).len(), 2048 * 256 * 2 * 98);
}

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
