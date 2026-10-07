use rusting_strata::serve::{chat_prompt, chat_request, host_ok, key_ok, read_request};

#[test]
fn chat_prompt_matches_template() {
    let m = vec![("system".to_string(), "Be brief.".to_string()), ("user".to_string(), "Hi".to_string())];
    assert_eq!(chat_prompt(&m, false),
        "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    assert!(chat_prompt(&m, true).ends_with("<|im_start|>assistant\n<think>\n"));
}

#[test]
fn parses_request() {
    let body = r#"{"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}],"stream":true,"max_tokens":7,"temperature":0.7,"top_p":0.8,"seed":5}"#;
    let raw = format!("POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\ncontent-length: {}\r\n\r\n{body}", body.len());
    let r = read_request(&mut raw.as_bytes()).unwrap();
    assert_eq!((r.method.as_str(), r.path.as_str(), r.host.as_str(), r.json), ("POST", "/v1/chat/completions", "x", true));
    let q = chat_request(&r.body).unwrap();
    assert_eq!(q.messages, vec![("user".to_string(), "ab".to_string())]);
    assert!(q.stream && !q.think);
    assert_eq!(q.max_tokens, Some(7));
    assert_eq!((q.temperature, q.top_k, q.top_p, q.seed), (Some(0.7), None, Some(0.8), Some(5)));
    assert!(chat_request(br#"{"messages":[]}"#).is_err());
}

#[test]
fn rejects_bad_hosts_and_huge_headers() {
    assert!(host_ok("localhost:8080", "127.0.0.1:8080") && host_ok("127.0.0.1", "127.0.0.1:8080") && host_ok("[::1]:9", "x:1"));
    assert!(host_ok("box.lan:8080", "box.lan:8080"));
    assert!(!host_ok("evil.example:8080", "127.0.0.1:8080") && !host_ok("", "127.0.0.1:8080"));
    let raw = format!("GET / HTTP/1.1\r\nX: {}\r\n\r\n", "a".repeat(70_000));
    assert!(read_request(&mut raw.as_bytes()).is_err());
}

#[test]
fn checks_api_key() {
    assert!(key_ok("Bearer s3cret", "s3cret"));
    assert!(!key_ok("Bearer s3cres", "s3cret") && !key_ok("Bearer s3cret2", "s3cret") && !key_ok("s3cret", "s3cret") && !key_ok("", "s3cret"));
}
