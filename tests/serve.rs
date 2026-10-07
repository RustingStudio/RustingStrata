use rusting_strata::serve::{chat_prompt, chat_request, read_request};

#[test]
fn chat_prompt_matches_template() {
    let m = vec![("system".to_string(), "Be brief.".to_string()), ("user".to_string(), "Hi".to_string())];
    assert_eq!(chat_prompt(&m, false),
        "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    assert!(chat_prompt(&m, true).ends_with("<|im_start|>assistant\n<think>\n"));
}

#[test]
fn parses_request() {
    let body = r#"{"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}],"stream":true,"max_tokens":7}"#;
    let raw = format!("POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\ncontent-length: {}\r\n\r\n{body}", body.len());
    let (method, path, b) = read_request(&mut raw.as_bytes()).unwrap();
    assert_eq!((method.as_str(), path.as_str()), ("POST", "/v1/chat/completions"));
    let q = chat_request(&b).unwrap();
    assert_eq!(q.messages, vec![("user".to_string(), "ab".to_string())]);
    assert!(q.stream && !q.think);
    assert_eq!(q.max_tokens, Some(7));
    assert!(chat_request(br#"{"messages":[]}"#).is_err());
}
