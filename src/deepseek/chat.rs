//! `rustingstrata chat` for DeepSeek-V4.1: greedy, CUDA for the matvecs and expert staging, KV kept across turns.
use super::forward::DsModel;
use crate::{chat::{Msg, Template}, gguf::Gguf, tokenizer::Tokenizer};
use std::io::{BufRead, Write};

fn gpu_defaults() {
    // GPU on by default; VRAM expert cache 8 GB (rest of the experts stream through one staging slot).
    for (k, v) in [("STRATA_DS_GPU", "1"), ("STRATA_DS_GPU_EXPERTS", "9")] {
        if std::env::var_os(k).is_none() { unsafe { std::env::set_var(k, v) } }
    }
}

/// `rustingstrata generate`: raw prompt (no template), greedy, prints the continuation.
pub fn generate(g: &Gguf, prompt: &str, n: usize) -> anyhow::Result<()> {
    gpu_defaults();
    let tok = Tokenizer::from_gguf(g)?;
    let mut m = DsModel::new(g)?;
    let eos = tok.eos_ids();
    let mut logits = vec![];
    logits = m.prefill(&tok.encode(prompt)?);
    let (mut out, mut shown) = (vec![], 0);
    for i in 0..n {
        let next = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
        if eos.contains(&next) { break; }
        out.push(next);
        let s = tok.decode(&out)?;
        if !s.ends_with('\u{FFFD}') && s.len() > shown && s.is_char_boundary(shown) {
            print!("{}", &s[shown..]);
            std::io::stdout().flush()?;
            shown = s.len();
        }
        if i + 1 < n { logits = m.step(next); }
    }
    println!();
    Ok(())
}

pub fn chat(g: &Gguf, system: Option<String>, n: usize, think: bool) -> anyhow::Result<()> {
    gpu_defaults();
    let tok = Tokenizer::from_gguf(g)?;
    let mut m = DsModel::new(g)?;
    let eos = tok.eos_ids();
    let tpl = Template::Ds;
    let mut first = true;
    let stdin = std::io::stdin();
    loop {
        print!("> ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 { break; }
        let line = line.trim_end();
        if line.is_empty() { continue; }
        // First turn: the full template. Later turns: the cache already holds everything up to the reply, so append only
        // the end-of-sentence token and the next user turn (earlier think blocks stay in the cache).
        let text = if first {
            let mut msgs: Vec<Msg> = system.iter().map(|s| Msg::new("system", s)).collect();
            msgs.push(Msg::new("user", line));
            tpl.render(&msgs, &[], think, false, true)
        } else {
            format!("<｜end▁of▁sentence｜><｜User｜>{line}<｜Assistant｜>{}", if think { "<think>" } else { "</think>" })
        };
        first = false;
        let mut logits = vec![];
        logits = m.prefill(&tok.encode(&text)?);
        let (mut out, mut shown) = (vec![], 0);
        for i in 0..n {
            let next = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
            if eos.contains(&next) { break; }
            out.push(next);
            let s = tok.decode(&out)?;
            if !s.ends_with('\u{FFFD}') && s.len() > shown && s.is_char_boundary(shown) {
                print!("{}", &s[shown..]);
                std::io::stdout().flush()?;
                shown = s.len();
            }
            if i + 1 < n { logits = m.step(next); }
        }
        println!();
    }
    Ok(())
}

/// Prompt plus reply cap per request (ponytail: fixed; the KV cache is untested far beyond this).
const CTX: usize = 4096;

/// `rustingstrata serve` for DeepSeek: one request at a time, OpenAI `/v1/chat/completions` (JSON or SSE), `/v1/models`, `/health`.
/// Earlier reasoning is kept by default, like llama-server (`chat_template_kwargs.preserve_thinking:false` drops it).
/// ponytail: greedy only, no tools/stop strings; the KV cache is reused only when the new prompt extends the last sequence.
pub fn serve(g: &Gguf, name: &str, addr: &str, api_key: Option<&str>) -> anyhow::Result<()> {
    use serde_json::json;
    gpu_defaults();
    let tok = Tokenizer::from_gguf(g)?;
    let mut m = DsModel::new(g)?;
    let eos = tok.eos_ids();
    let l = std::net::TcpListener::bind(addr)?;
    eprintln!("deepseek serve on http://{addr}/ model {name}");
    for c in l.incoming() {
        let Ok(mut c) = c else { continue };
        let _ = c.set_read_timeout(Some(std::time::Duration::from_secs(30)));
        let _ = c.set_write_timeout(Some(std::time::Duration::from_secs(30)));
        let Ok(r) = crate::serve::read_request(&mut std::io::BufReader::new(&c)) else { continue };
        let send = |c: &mut std::net::TcpStream, status: &str, ty: &str, b: &str| {
            let _ = write!(c, "HTTP/1.1 {status}\r\nContent-Type: {ty}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len());
        };
        let path = r.path.split('?').next().unwrap_or("");
        // /health stays open for load balancers, like the main server.
        // Without a key the Host check stops DNS rebinding, as in the main server.
        if path != "/health" && api_key.is_none() && !crate::serve::host_ok(&r.host, addr, &[]) {
            send(&mut c, "403 Forbidden", "application/json", r#"{"error":{"message":"bad Host header"}}"#);
            continue;
        }
        if path != "/health" && api_key.is_some_and(|k| !crate::serve::key_ok(&r.auth, k)) {
            send(&mut c, "401 Unauthorized", "application/json", r#"{"error":{"message":"missing or wrong API key","type":"invalid_api_key"}}"#);
            continue;
        }
        match (r.method.as_str(), path) {
            ("GET", "/health") => send(&mut c, "200 OK", "application/json", r#"{"status":"ok"}"#),
            ("GET", "/v1/models") => send(&mut c, "200 OK", "application/json", &json!({"object": "list", "data": [{"id": name, "object": "model", "owned_by": "strata"}]}).to_string()),
            ("POST", "/v1/chat/completions") => {
                let q = match crate::serve::chat_request(&r.body) {
                    Ok(q) => q,
                    Err(e) => { send(&mut c, "400 Bad Request", "application/json", &json!({"error": {"message": format!("{e:#}"), "type": "invalid_request_error"}}).to_string()); continue }
                };
                let ids = match tok.encode(&Template::Ds.render(&q.messages, &[], q.think, q.preserve_thinking.unwrap_or(true), true)) {
                    Ok(i) if !i.is_empty() => i,
                    _ => { send(&mut c, "400 Bad Request", "application/json", r#"{"error":{"message":"cannot encode prompt"}}"#); continue }
                };
                if ids.len() > CTX {
                    send(&mut c, "400 Bad Request", "application/json", &json!({"error": {"message": format!("prompt is {} tokens, the limit is {CTX}", ids.len()), "type": "invalid_request_error"}}).to_string());
                    continue;
                }
                let keep = if m.toks.len() < ids.len() && ids.starts_with(&m.toks) { m.toks.len() } else { m.reset(); 0 };
                let mut logits = vec![];
                logits = m.prefill(&ids[keep..]);
                let n = q.max_tokens.unwrap_or(256).min(CTX - ids.len());
                if q.stream { let _ = write!(c, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n"); }
                let chunk = |d: serde_json::Value, fin: Option<&str>| json!({"object": "chat.completion.chunk", "model": name, "choices": [{"index": 0, "delta": d, "finish_reason": fin}]});
                let (mut out, mut shown, mut done) = (vec![], 0, "length");
                for i in 0..n {
                    let next = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
                    if eos.contains(&next) { done = "stop"; break; }
                    out.push(next);
                    let s = tok.decode(&out).unwrap_or_default();
                    if q.stream && !s.ends_with('\u{FFFD}') && s.len() > shown && s.is_char_boundary(shown) {
                        let ok = write!(c, "data: {}\n\n", chunk(json!({"content": &s[shown..]}), None)).is_ok();
                        shown = s.len();
                        if !ok { break; }
                    }
                    if i + 1 < n { logits = m.step(next); }
                }
                let text = tok.decode(&out).unwrap_or_default();
                if q.stream { let _ = write!(c, "data: {}\n\ndata: [DONE]\n\n", chunk(json!({}), Some(done))); }
                else {
                    let b = json!({"object": "chat.completion", "model": name, "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": done}],
                        "usage": {"prompt_tokens": ids.len(), "completion_tokens": out.len(), "total_tokens": ids.len() + out.len()}});
                    send(&mut c, "200 OK", "application/json", &b.to_string());
                }
            }
            _ => send(&mut c, "404 Not Found", "application/json", r#"{"error":{"message":"not found"}}"#),
        }
    }
    Ok(())
}
