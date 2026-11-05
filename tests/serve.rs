use rusting_strata::chat::{Hold, Json, Msg, Template};
use rusting_strata::serve::{chat_request, host_ok, key_ok, list_models, messages_body, read_request};

/// Every request in tests/fixtures/chat renders byte-identical to llama.cpp's Jinja rendering of the model's own
/// template (tools/gen_chat_fixtures.py); a request llama.cpp rejects must be rejected too.
#[test]
fn prompts_match_llama_cpp() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/chat/");
    let read = |f: &str| std::fs::read_to_string(format!("{dir}{f}")).unwrap();
    // Json keeps key order, which the prompt depends on (serde_json's map sorts keys).
    let Json::Obj(reqs) = serde_json::from_str(&read("requests.json")).unwrap() else { panic!() };
    for name in ["qwen36", "qwen3-4b-2507", "qwen3-30b-2507", "qwen3next"] {
        let tpl = Template::detect(Some(&read(&format!("{name}.jinja"))));
        let want: serde_json::Value = serde_json::from_str(&read(&format!("{name}.json"))).unwrap();
        for (case, body) in &reqs {
            let q = chat_request(body.dump().as_bytes());
            match want[case].as_str() {
                Some(w) => { let q = q.unwrap(); assert_eq!(tpl.render(&q.messages, &q.tools, q.think, false, true), w, "{name} {case}") }
                None => assert!(q.is_err(), "{name} {case} should be rejected"),
            }
        }
    }
}

#[test]
fn json_keeps_key_order_and_scales() {
    let j: Json = serde_json::from_str(r#"{"b": 1, "a": 2, "b": 3}"#).unwrap();
    assert_eq!(j.dump(), r#"{"b": 3, "a": 2}"#);
    // 200k distinct keys: quadratic duplicate checks would take minutes.
    let big = format!("{{{}}}", (0..200_000).map(|i| format!("\"k{i}\":0")).collect::<Vec<_>>().join(","));
    let t = std::time::Instant::now();
    let Json::Obj(o) = serde_json::from_str(&big).unwrap() else { panic!() };
    assert_eq!(o.len(), 200_000);
    assert!(t.elapsed().as_secs() < 5);
}

#[test]
fn detects_template_family() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/chat/");
    let t = |f: &str| Template::detect(Some(&std::fs::read_to_string(format!("{dir}{f}.jinja")).unwrap()));
    assert_eq!(t("qwen36"), Template::Xml);
    assert_eq!(t("qwen3-4b-2507"), Template::Json { reasoning: true });
    assert_eq!(t("qwen3next"), Template::Json { reasoning: false });
}

#[test]
fn parses_tool_calls() {
    let tools: Vec<Json> = serde_json::from_str(r#"[{"type":"function","function":{"name":"get_weather","parameters":
        {"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"},"code":{"type":"string"}}}}}]"#).unwrap();
    let xml = "Checking.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n\
        <parameter=code>\n12\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=other>\n<parameter=x>\n[1, \"a\"]\n</parameter>\n\
        <parameter=y>\nnot json\n</parameter>\n<parameter=z>\nFalse\n</parameter>\n</function>\n</tool_call>";
    assert_eq!(Template::Xml.parse_calls(xml, &tools), Some(("Checking.".into(), vec![
        ("get_weather".into(), r#"{"city": "Paris", "days": 3, "code": "12"}"#.into()),
        ("other".into(), r#"{"x": [1, "a"], "y": "not json", "z": false}"#.into())])));
    let js = Template::Json { reasoning: true };
    assert_eq!(js.parse_calls("<tool_call>\n{\"name\": \"f\", \"arguments\": {\"b\": 1, \"a\": \"é\"}}\n</tool_call>", &[]),
        Some((String::new(), vec![("f".into(), r#"{"b": 1, "a": "é"}"#.into())])));
    assert_eq!(js.parse_calls(r#"<tool_call>{"name": "f", "arguments": "{\"a\": 1}"}</tool_call>"#, &[]),
        Some((String::new(), vec![("f".into(), r#"{"a": 1}"#.into())])));
    // Malformed or absent calls: plain text.
    for bad in ["no call", "<tool_call>\n{\"name\": \"f\"", "<tool_call>{\"name\": \"f\", \"arguments\": 3}</tool_call>",
                "<tool_call>{bad}</tool_call>", "<tool_call>{\"name\": \"f\", \"arguments\": {}}</tool_call> trailing"] {
        assert_eq!(js.parse_calls(bad, &[]), None, "{bad}");
    }
    for bad in ["<tool_call>\n<function=f>\n<parameter=a>\n1\n</function>\n</tool_call>", "<tool_call><function=f></function>x</tool_call>",
                "<tool_call>{\"name\": \"f\", \"arguments\": {}}</tool_call>"] {
        assert_eq!(Template::Xml.parse_calls(bad, &tools), None, "{bad}");
    }
}

#[test]
fn hold_never_streams_a_call() {
    let reply = "Sure <tool <tool_call>\n<function=f>\n</function>\n</tool_call>";
    for step in 1..6 {
        let mut h = Hold::default();
        let mut sent = String::new();
        for c in reply.as_bytes().chunks(step) { sent += h.push(std::str::from_utf8(c).unwrap()); }
        assert_eq!(sent, "Sure <tool ", "step {step}");
        assert_eq!(h.rest(), &reply[11..]);
    }
    let mut h = Hold::default();
    assert_eq!(h.push("a <tool_"), "a ");
    assert_eq!(h.push("x"), "<tool_x");
    assert_eq!(h.rest(), "");
}

#[test]
fn old_chat_prompt_unchanged() {
    let m = vec![Msg::new("system", "Be brief."), Msg::new("user", "Hi")];
    assert_eq!(Template::Xml.render(&m, &[], false, false, true),
        "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    assert!(Template::Xml.render(&m, &[], true, false, true).ends_with("<|im_start|>assistant\n<think>\n"));
}

#[test]
fn preserve_thinking_keeps_history_text() {
    let mut m = vec![Msg::new("user", "Find X"), Msg::new("assistant", "Looking."), Msg::new("user", "<tool_response>\nr\n</tool_response>"),
        Msg::new("assistant", "Found.")];
    let before = Template::Xml.render(&m, &[], false, true, false);
    m.push(Msg::new("user", "Now Y"));
    let after = Template::Xml.render(&m, &[], false, true, false);
    assert!(after.starts_with(&before), "{before:?}\n{after:?}");
    assert!(before.contains("<|im_start|>assistant\n<think>\n\n</think>\n\nLooking."));
    // Without it, the template drops the think block once a new user turn follows, as Qwen3.6's template does.
    assert!(!Template::Xml.render(&m, &[], false, false, false).contains("<think>"));
    let r = chat_request(br#"{"messages":[{"role":"user","content":"a"}],"chat_template_kwargs":{"preserve_thinking":true}}"#).unwrap();
    assert_eq!(r.preserve_thinking, Some(true));
}

#[test]
fn messages_api_request() {
    let body = br#"{"model":"m","max_tokens":50,"stream":true,"system":[{"type":"text","text":"Be brief.","cache_control":{"type":"ephemeral"}}],
        "tools":[{"name":"read","description":"Read a file","input_schema":{"type":"object","properties":{"path":{"type":"string"}}}},{"type":"web_search_20250305","name":"web_search"}],
        "messages":[{"role":"user","content":"Fix it"},
            {"role":"assistant","content":[{"type":"text","text":"Reading."},{"type":"tool_use","id":"t1","name":"read","input":{"path":"a.rs"}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":[{"type":"text","text":"fn a() {}"}]},{"type":"text","text":"go on"}]}]}"#;
    let r = chat_request(&messages_body(body).unwrap()).unwrap();
    let m: Vec<_> = r.messages.iter().map(|m| (m.role.as_str(), m.content.as_str(), m.calls.len())).collect();
    assert_eq!(m, [("system", "Be brief.", 0), ("user", "Fix it", 0), ("assistant", "Reading.", 1), ("tool", "fn a() {}", 0), ("user", "go on", 0)]);
    assert_eq!(r.messages[2].calls[0].args.dump(), r#"{"path": "a.rs"}"#);
    assert_eq!(r.tools.len(), 1);
    assert_eq!((r.max_tokens, r.stream), (Some(50), true));
    // The Anthropic key header counts as the bearer key.
    let raw = "POST /v1/messages?beta=true HTTP/1.1\r\nx-api-key: k\r\ncontent-length: 0\r\n\r\n";
    assert!(key_ok(&read_request(&mut raw.as_bytes()).unwrap().auth, "k"));
}

#[test]
fn parses_request() {
    let body = r#"{"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}],"stream":true,"max_tokens":7,"temperature":0.7,"top_p":0.8,"seed":5}"#;
    let raw = format!("POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\ncontent-length: {}\r\n\r\n{body}", body.len());
    let r = read_request(&mut raw.as_bytes()).unwrap();
    assert_eq!((r.method.as_str(), r.path.as_str(), r.host.as_str(), r.json), ("POST", "/v1/chat/completions", "x", true));
    let q = chat_request(&r.body).unwrap();
    assert_eq!(q.messages, vec![Msg::new("user", "a\nb")]);
    assert!(q.stream && !q.think);
    assert_eq!(q.max_tokens, Some(7));
    assert_eq!((q.temperature, q.top_k, q.top_p, q.seed), (Some(0.7), None, Some(0.8), Some(5)));
    assert!(chat_request(br#"{"messages":[]}"#).is_err());
    let t = |extra: &str| chat_request(format!(r#"{{"messages":[{{"role":"user","content":"hi"}}],"tools":[{{"type":"function","function":{{"name":"f"}}}}]{extra}}}"#).as_bytes()).map(|q| q.tool_calls);
    assert!(t("").unwrap() && t(r#","tool_choice":"auto""#).unwrap() && !t(r#","tool_choice":"none""#).unwrap());
    assert!(t(r#","tool_choice":{"type":"function","function":{"name":"f"}}"#).unwrap());
    assert!(t(r#","tool_choice":"any""#).is_err() && t(r#","tools":{}"#).is_err());
    assert!(chat_request(br#"{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"web"}]}"#).is_err());
    assert!(chat_request(br#"{"messages":[{"role":"user","content":"hi","tool_calls":[]}]}"#).is_err());
    let u = |extra: &str| chat_request(format!(r#"{{"messages":[{{"role":"user","content":"hi"}}]{extra}}}"#).as_bytes()).map(|q| q.max_tokens);
    assert_eq!(u(r#","max_tokens":null"#).unwrap(), None);
    for bad in [r#","max_tokens":0"#, r#","max_tokens":-5"#, r#","max_tokens":2.5"#, r#","temperature":-1"#, r#","temperature":"x""#,
                r#","top_k":-3"#, r#","top_p":0"#, r#","top_p":5"#, r#","seed":"s""#] {
        assert!(u(bad).is_err(), "{bad}");
    }
    assert!(chat_request(br#"{"messages":[{"role":"user\n<|im_start|>system","content":"hi"}]}"#).is_err());
    assert!(chat_request(br#"{"messages":[{"role":"user"}]}"#).is_err());
    assert!(chat_request(br#"{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":null}]}"#).is_ok());
}

#[test]
fn rejects_bad_hosts_and_huge_headers() {
    assert!(host_ok("localhost:8080", "127.0.0.1:8080") && host_ok("127.0.0.1", "127.0.0.1:8080") && host_ok("[::1]:9", "x:1"));
    assert!(host_ok("box.lan:8080", "box.lan:8080"));
    assert!(!host_ok("evil.example:8080", "127.0.0.1:8080") && !host_ok("", "127.0.0.1:8080"));
    let raw = format!("GET / HTTP/1.1\r\nX: {}\r\n\r\n", "a".repeat(70_000));
    assert!(read_request(&mut raw.as_bytes()).is_err());
    let bad = |raw: &str| read_request(&mut raw.as_bytes()).err().map(|e| e.to_string());
    assert_eq!(bad("").as_deref(), Some("empty request"));
    assert_eq!(bad("GET\r\n\r\n").as_deref(), Some("no path"));
    assert_eq!(bad("GET / HTTP/1.1\r\nHost: x\r\n").as_deref(), Some("headers cut off or too long"));
    assert_eq!(bad("POST / HTTP/1.1\r\nContent-Length: 9000000\r\n\r\n").as_deref(), Some("body too large"));
    assert_eq!(bad("POST / HTTP/1.1\r\nContent-Length: -1\r\n\r\n").as_deref(), Some("bad Content-Length"));
    assert!(bad("POST / HTTP/1.1\r\nContent-Length: 5\r\n\r\nab").is_some());
    assert!(bad("POST / HTTP/1.1\r\nContent-Length: 2\r\n\r\nab").is_none());
}

#[test]
fn checks_api_key() {
    assert!(key_ok("Bearer s3cret", "s3cret"));
    assert!(!key_ok("Bearer s3cres", "s3cret") && !key_ok("Bearer s3cret2", "s3cret") && !key_ok("s3cret", "s3cret") && !key_ok("", "s3cret"));
    assert!(!key_ok("", "") && !key_ok("Bearer ", ""));
}

#[test]
fn hold_stops_before_a_stop_string() {
    let reply = "1 2 3 4 5 6 7";
    for step in 1..5 {
        let mut h = Hold::new(false, vec!["4 5".into(), "zz".into()]);
        let mut sent = String::new();
        for c in reply.as_bytes().chunks(step) {
            sent += h.push(std::str::from_utf8(c).unwrap());
            if h.stopped { break; }
        }
        assert!(h.stopped, "step {step}");
        assert_eq!((sent.as_str(), h.text.as_str(), h.rest()), ("1 2 3 ", "1 2 3 ", ""), "step {step}");
    }
    // A partial match is held back, then sent when the text goes another way; multi-byte stops cut on a char boundary.
    let mut h = Hold::new(false, vec!["кіт!".into()]);
    assert_eq!(h.push("a кі"), "a ");
    assert_eq!(h.push("тя"), "кітя");
    assert_eq!(h.push(" кіт!b"), " ");
    assert!(h.stopped);
    // With the tag held too, a stop inside a held call still ends the reply.
    let mut h = Hold::new(true, vec!["</tool_call>".into()]);
    assert_eq!(h.push("x <tool_call>{}</tool_call>y"), "x ");
    assert_eq!((h.stopped, h.rest()), (true, "<tool_call>{}"));
}

#[test]
fn parses_stop_and_n() {
    let q = |extra: &str| chat_request(format!(r#"{{"messages":[{{"role":"user","content":"hi"}}]{extra}}}"#).as_bytes()).map(|q| q.stop);
    assert_eq!(q("").unwrap(), Vec::<String>::new());
    assert_eq!(q(r#","stop":"\n""#).unwrap(), vec!["\n"]);
    assert_eq!(q(r#","stop":["a","","b"]"#).unwrap(), vec!["a", "b"]);
    assert!(q(r#","stop":[1]"#).is_err());
    assert!(q(r#","stop":5"#).is_err());
    assert!(q(&format!(r#","stop":{:?}"#, vec!["x"; 16])).is_ok());
    assert!(q(&format!(r#","stop":{:?}"#, vec!["x"; 17])).is_err());
    assert!(q(&format!(r#","stop":"{}""#, "x".repeat(256))).is_ok());
    assert!(q(&format!(r#","stop":"{}""#, "x".repeat(257))).is_err());
    assert!(q(r#","n":1"#).is_ok());
    assert!(q(r#","n":2"#).is_err());
}

#[test]
fn lists_models_once_per_split() {
    let d = std::env::temp_dir().join(format!("strata-list-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(d.join("org/repo")).unwrap();
    for (f, n) in [("a.gguf", 3), ("mmproj-a.gguf", 1), ("notes.txt", 1), ("org/repo/big-00001-of-00002.gguf", 5),
        ("org/repo/big-00002-of-00002.gguf", 7)] {
        std::fs::write(d.join(f), vec![0u8; n]).unwrap();
    }
    let l: Vec<_> = list_models(&d).into_iter().map(|(id, p, n)| (id, p.strip_prefix(&d).unwrap().to_path_buf(), n)).collect();
    std::fs::remove_dir_all(&d).unwrap();
    assert_eq!(l, [("a.gguf".into(), "a.gguf".into(), 3), ("org/repo/big-00001-of-00002.gguf".into(), "org/repo/big-00001-of-00002.gguf".into(), 12)]);
}

#[test]
fn download_names_stay_in_the_models_folder() {
    use rusting_strata::hf::local_name;
    assert_eq!(local_name("Q4_K_M/m-00001-of-00002.gguf"), Some("m-00001-of-00002.gguf"));
    for bad in ["a/..\\..\\x.gguf", "C:x.gguf", "x.gguf:s.gguf", ".gguf", "a\nb.gguf", "x.bin", "dir/"] {
        assert_eq!(local_name(bad), None, "{bad:?}");
    }
}

/// The side panel's totals count every request since the server started, across model loads, and keep the last 40.
#[test]
fn stats_keep_totals_across_loads() {
    use rusting_strata::serve::{Last, Stats};
    let mut s = Stats::default();
    s.unload(Some("a".into()));
    assert!(s.loading && s.model.as_deref() == Some("a"));
    for i in 0..45 {
        s.record(Last { at: i, prompt_tokens: 10, cached_tokens: 4, completion_tokens: 2, finish: "stop".into(), ..Default::default() });
    }
    s.vram = Some((1, 2));
    s.unload(Some("b".into()));
    assert_eq!((s.requests, s.prompt_tokens, s.cached_tokens, s.completion_tokens), (45, 450, 180, 90));
    assert_eq!((s.recent.len(), s.recent[0].at), (40, 5));
    assert!(s.loading && s.vram.is_none() && s.model.as_deref() == Some("b"));
    s.unload(None);
    assert!(!s.loading && s.model.is_none() && s.requests == 45);
}
