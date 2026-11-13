mod common;
use rusting_strata::{gguf::Gguf, tokenizer::Tokenizer};

#[test]
fn matches_llama_cpp() {
    let Some(p) = common::model_path() else { return };
    let g = Gguf::open(&p).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let pre = g.get("tokenizer.ggml.pre").unwrap().as_str().unwrap().to_string();
    let fx: serde_json::Value = serde_json::from_str(match pre.as_str() {
        "qwen2" => include_str!("fixtures/tokens-qwen2.json"),
        _ => include_str!("fixtures/tokens.json"),
    }).unwrap();
    for case in fx.as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let want: Vec<u32> = case["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        assert_eq!(tok.encode(text).unwrap(), want, "encode {text:?}");
        assert_eq!(tok.decode(&want).unwrap(), text, "decode {text:?}");
    }
    assert!(!tok.eos_ids().is_empty());
}

#[test]
fn vision_special_tokens_are_atomic() {
    let Some(p) = common::model_path() else { return };
    let g = Gguf::open(&p).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let (Some(start), Some(pad), Some(end)) = (
        tok.token_id("<|vision_start|>"), tok.token_id("<|image_pad|>"), tok.token_id("<|vision_end|>"),
    ) else { eprintln!("skip: model has no vision tokens"); return };
    assert_eq!(tok.encode("<|vision_start|><|image_pad|><|vision_end|>").unwrap(), [start, pad, end]);
}
