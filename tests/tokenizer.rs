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

/// `/tokenize` and `/detokenize` in llama.cpp's shapes: the pieces of a text are its bytes, split by token.
#[test]
fn tokenize_endpoints() {
    let Some(p) = common::model_path() else { return };
    let g = Gguf::open(&p).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let text = "Hello wörld 你好 <|im_end|>\n";
    let v = rusting_strata::serve::tokenize(&tok, serde_json::json!({"content": text}).to_string().as_bytes()).unwrap();
    let ids: Vec<u32> = v["tokens"].as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect();
    assert_eq!(ids, tok.encode(text).unwrap());
    let v = rusting_strata::serve::tokenize(&tok, serde_json::json!({"content": text, "with_pieces": true}).to_string().as_bytes()).unwrap();
    let mut bytes = vec![];
    for (t, &id) in v["tokens"].as_array().unwrap().iter().zip(&ids) {
        assert_eq!(t["id"], id);
        match &t["piece"] {
            serde_json::Value::String(s) => bytes.extend_from_slice(s.as_bytes()),
            serde_json::Value::Array(a) => bytes.extend(a.iter().map(|b| b.as_u64().unwrap() as u8)),
            x => panic!("piece {x}"),
        }
    }
    assert_eq!(bytes, text.as_bytes());
    let back = rusting_strata::serve::detokenize(&tok, serde_json::json!({"tokens": ids}).to_string().as_bytes()).unwrap();
    assert_eq!(back["content"], text);
    assert!(rusting_strata::serve::detokenize(&tok, br#"{"tokens": [-1]}"#).is_err());
    assert!(rusting_strata::serve::tokenize(&tok, br#"{"content": 1}"#).is_err());
}
