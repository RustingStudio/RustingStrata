//! Recursive Flint (GGUF arch `flint`): the metadata/array parsing and tokenizer. Set `STRATA_FLINT_MODEL` to a
//! Flint GGUF to run these.

use rusting_strata::{config::ModelConfig, gguf::Gguf, tokenizer::Tokenizer};

fn flint_path() -> Option<std::path::PathBuf> {
    let Ok(p) = std::env::var("STRATA_FLINT_MODEL") else { eprintln!("SKIP: STRATA_FLINT_MODEL is not set"); return None };
    let p = std::path::PathBuf::from(p);
    if p.exists() { Some(p) } else { eprintln!("SKIP: Flint model missing: {}", p.display()); None }
}

#[test]
fn flint_config_and_tokenizer() {
    let Some(p) = flint_path() else { return };
    let g = Gguf::open(&p).unwrap();
    let c = ModelConfig::from_gguf(&g).unwrap();
    assert_eq!(c.arch, "flint");
    assert_eq!((c.n_layer, c.n_embd, c.n_head, c.n_head_kv, c.head_dim, c.n_rot, c.n_vocab),
        (49, 2048, 32, 4, 128, 128, 150144));
    assert_eq!((c.n_dense, c.n_expert, c.n_expert_used), (2, 256, 6));
    assert_eq!((c.n_ff, c.n_ff_exp, c.n_ff_shexp, c.n_swa), (6144, 512, 1024, 513));
    assert_eq!((c.swa.iter().filter(|&&x| x).count(), c.rope_layer.iter().filter(|&&x| !x).count()), (35, 14));
    assert!(c.attn_sinks && c.router_sigmoid && !c.q_gate && !c.k_interleave);
    assert!(!c.is_moe_layer(1) && c.is_moe_layer(2));

    let tok = Tokenizer::from_gguf(&g).unwrap();
    assert_eq!(tok.encode("The capital of France is").unwrap(), [630, 6963, 297, 9865, 332]);
    assert_eq!(tok.encode("<|user|>Hi<|assistant|>").unwrap(), [150003, 10340, 150004]);
    let eos = tok.eos_ids();
    for id in [150001, 150003, 150005, 150018] { assert!(eos.contains(&id), "eog {id} missing: {eos:?}"); }
}
