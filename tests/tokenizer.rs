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

/// A grammar on the real vocabulary: whatever the logits, the picked tokens spell what it allows, greedy and
/// sampled, and the reply can end only once it is whole. `logit_bias` bans and boosts by id and by text.
#[test]
fn grammar_and_logit_bias_constrain_picks() {
    use rusting_strata::grammar::{BiasKey, Constraint, Grammar};
    use rusting_strata::ops::Sampler;
    let Some(p) = common::model_path() else { return };
    let g = Gguf::open(&p).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let gram = std::sync::Arc::new(Grammar::parse("root ::= \"{\" ws \"\\\"n\\\":\" ws [0-9]+ ws \"}\"\nws ::= [ ]?").unwrap());
    let eos = tok.eos_ids()[0];
    for temp in [0.0, 1.0] {
        let mut s = Sampler { temp, rng: 5, limit: Constraint::of(&[], Some(gram.clone()), &tok).unwrap(), ..Sampler::greedy() };
        let mut out = vec![];
        let mut rng = 1u64;
        let t0 = std::time::Instant::now();
        for _ in 0..40 {
            // Noise, with the end token and a letter favoured so the mask has work to do.
            let mut l: Vec<f32> = (0..tok.vocab_size()).map(|_| { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; (rng >> 40) as f32 / 1e6 }).collect();
            l[eos as usize] = 50.0;
            l[tok.encode("x").unwrap()[0] as usize] = 40.0;
            let t = s.pick(&l).unwrap() as u32;
            if t == eos { break; }
            out.push(t);
        }
        let text = tok.decode(&out).unwrap();
        eprintln!("temp {temp}: {text:?} in {:.1} ms per token", t0.elapsed().as_secs_f64() * 1e3 / (out.len() + 1) as f64);
        let body = text.trim_start_matches('{').trim_end_matches('}').trim();
        assert!(text.starts_with('{') && text.ends_with('}') && body.starts_with("\"n\":") && body[4..].trim().parse::<u64>().is_ok(), "{text:?}");
    }
    // Bias: the best token banned by id, another boosted through its text.
    let l: Vec<f32> = (0..tok.vocab_size()).map(|i| if i == 100 { 9.0 } else { 0.0 }).collect();
    let hi = tok.encode("hi").unwrap()[0];
    let mut s = Sampler { limit: Constraint::of(&[(BiasKey::Id(100), f32::NEG_INFINITY), (BiasKey::Text("hi".into()), 5.0)], None, &tok).unwrap(), ..Sampler::greedy() };
    assert_eq!(s.pick(&l).unwrap() as u32, hi);
    assert!(Constraint::of(&[(BiasKey::Id(u32::MAX), 1.0)], None, &tok).is_err());
}
