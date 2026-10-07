//! Chat prompt format and the small HTTP/1.1 subset `strata serve` speaks (std::net, one request per connection).

use anyhow::{Context, bail};
use serde_json::Value;
use std::io::BufRead;

/// ChatML prompt in the model's template: (role, content) turns, then the assistant header. `think` false
/// appends the empty think block the template uses for `enable_thinking = false`.
pub fn chat_prompt(msgs: &[(String, String)], think: bool) -> String {
    let mut s: String = msgs.iter().map(|(r, c)| format!("<|im_start|>{r}\n{c}<|im_end|>\n")).collect();
    s += if think { "<|im_start|>assistant\n<think>\n" } else { "<|im_start|>assistant\n<think>\n\n</think>\n\n" };
    s
}

/// Reads one request: method, path, body (Content-Length only, no chunked uploads).
pub fn read_request(r: &mut impl BufRead) -> anyhow::Result<(String, String, Vec<u8>)> {
    let mut line = String::new();
    r.read_line(&mut line)?;
    let mut it = line.split_whitespace();
    let (method, path) = (it.next().context("empty request")?.to_string(), it.next().context("no path")?.to_string());
    let mut len = 0;
    loop {
        line.clear();
        if r.read_line(&mut line)? == 0 { bail!("headers cut off"); }
        let l = line.trim_end();
        if l.is_empty() { break; }
        if let Some((k, v)) = l.split_once(':') && k.eq_ignore_ascii_case("content-length") { len = v.trim().parse()?; }
    }
    if len > 64 << 20 { bail!("body too large"); }
    let mut body = vec![0; len];
    r.read_exact(&mut body)?;
    Ok((method, path, body))
}

/// The parts of an OpenAI chat completion request that strata uses. Decoding is greedy: sampling fields are ignored.
pub struct ChatRequest {
    pub messages: Vec<(String, String)>,
    pub max_tokens: Option<usize>,
    pub stream: bool,
    pub think: bool,
}

pub fn chat_request(body: &[u8]) -> anyhow::Result<ChatRequest> {
    let v: Value = serde_json::from_slice(body)?;
    let messages = v["messages"].as_array().context("missing messages")?.iter().map(|m| {
        let role = m["role"].as_str().context("message without role")?;
        // content is a string or a list of {"type": "text", "text": ..} parts
        let content = match &m["content"] {
            Value::String(s) => s.clone(),
            Value::Array(parts) => parts.iter().filter_map(|p| p["text"].as_str()).collect(),
            Value::Null => String::new(),
            _ => bail!("bad content"),
        };
        Ok((role.to_string(), content))
    }).collect::<anyhow::Result<Vec<_>>>()?;
    if messages.is_empty() { bail!("no messages"); }
    let max_tokens = v["max_completion_tokens"].as_u64().or(v["max_tokens"].as_u64()).map(|n| n as usize);
    let think = v["chat_template_kwargs"]["enable_thinking"].as_bool().unwrap_or(false);
    Ok(ChatRequest { messages, max_tokens, stream: v["stream"].as_bool().unwrap_or(false), think })
}
