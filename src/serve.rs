//! The small HTTP/1.1 subset `strata serve` speaks (std::net, one request per connection) and its chat request.

use crate::chat::{Json, Msg, tool_def};
use anyhow::{Context, bail};
use serde_json::Value;
use std::io::{BufRead, Read};
use std::path::{Path, PathBuf};

pub struct Request {
    pub method: String,
    pub path: String,
    pub host: String,
    pub auth: String,
    pub json: bool,
    pub body: Vec<u8>,
}

/// Reads one request (Content-Length bodies only, no chunked uploads). Headers are capped at 64 KiB and the body at
/// 8 MiB.
pub fn read_request(r: &mut impl BufRead) -> anyhow::Result<Request> {
    let mut h = r.take(64 << 10);
    let mut line = String::new();
    h.read_line(&mut line)?;
    let mut it = line.split_whitespace();
    let (method, path) = (it.next().context("empty request")?.to_string(), it.next().context("no path")?.to_string());
    let (mut len, mut host, mut auth, mut json) = (0, String::new(), String::new(), false);
    loop {
        line.clear();
        if h.read_line(&mut line)? == 0 { bail!("headers cut off or too long"); }
        let l = line.trim_end();
        if l.is_empty() { break; }
        let Some((k, v)) = l.split_once(':') else { continue };
        let (k, v) = (k.to_ascii_lowercase(), v.trim());
        match k.as_str() {
            "content-length" => len = v.parse().context("bad Content-Length")?,
            "host" => host = v.to_string(),
            "authorization" => auth = v.to_string(),
            "content-type" => json = v.starts_with("application/json"),
            _ => {}
        }
    }
    if len > 8 << 20 { bail!("body too large"); }
    let mut body = vec![0; len];
    r.read_exact(&mut body)?;
    Ok(Request { method, path, host, auth, json, body })
}

/// Host header check against DNS rebinding: a loopback name, or the host the server was bound to.
pub fn host_ok(host: &str, bound: &str) -> bool {
    let name = |s: &str| s.rsplit_once(':').filter(|(_, p)| p.parse::<u16>().is_ok()).map_or(s, |x| x.0).to_ascii_lowercase();
    let n = name(host);
    matches!(n.as_str(), "localhost" | "127.0.0.1" | "[::1]") || n == name(bound)
}

/// `Authorization` header check for `--api-key`, without an early exit on the first differing byte.
pub fn key_ok(auth: &str, key: &str) -> bool {
    if key.is_empty() { return false; }
    let a = auth.strip_prefix("Bearer ").unwrap_or("").as_bytes();
    a.len() == key.len() && a.iter().zip(key.as_bytes()).fold(0, |x, (p, q)| x | (p ^ q)) == 0
}

/// GGUF models under `dir` and two folder levels below it, as (name relative to `dir` with `/` separators, path,
/// bytes over all shards), sorted by name; see `models`.
pub fn list_models(dir: &Path) -> Vec<(String, PathBuf, u64)> {
    let (mut files, mut dirs) = (Vec::new(), vec![(dir.to_path_buf(), 0)]);
    while let Some((d, depth)) = dirs.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for p in rd.flatten().map(|e| e.path()) {
            if p.is_dir() { if depth < 2 { dirs.push((p, depth + 1)); } continue; }
            let id = p.strip_prefix(dir).unwrap_or(&p).to_string_lossy().replace('\\', "/");
            files.push((id, std::fs::metadata(&p).map_or(0, |m| m.len())));
        }
    }
    let mut out: Vec<_> = models(&files).into_iter().map(|(id, n)| (id.clone(), dir.join(id), n)).collect();
    out.sort();
    out
}

/// The models among files given as (name, bytes): .gguf files, a split model once by its first shard with the
/// bytes of all its shards, vision projectors (`mmproj*`) left out.
pub fn models(files: &[(String, u64)]) -> Vec<(String, u64)> {
    let parts = |stem: &str| files.iter().filter(|(m, _)| shard(m).is_some_and(|(s, _)| s == stem)).map(|x| x.1).sum();
    files.iter().filter(|(n, _)| n.ends_with(".gguf") && !n.rsplit('/').next().unwrap_or(n).starts_with("mmproj"))
        .filter_map(|(n, b)| match shard(n) {
            Some((stem, 1)) => Some((n.clone(), parts(stem))),
            Some(_) => None,
            None => Some((n.clone(), *b)),
        }).collect()
}

/// "dir/name-00002-of-00003.gguf" gives ("dir/name", 2).
pub fn shard(n: &str) -> Option<(&str, u32)> {
    let (head, total) = n.strip_suffix(".gguf")?.rsplit_once("-of-")?;
    let (stem, i) = head.rsplit_once('-')?;
    let digits = |s: &str| s.len() == 5 && s.bytes().all(|b| b.is_ascii_digit());
    if !digits(total) || !digits(i) { return None; }
    Some((stem, i.parse().ok()?))
}

/// The parts of an OpenAI chat completion request that strata uses. Sampling fields left out fall back to the
/// server's flags (greedy by default).
pub struct ChatRequest {
    pub messages: Vec<Msg>,
    /// Tool definitions as the template prints them (`chat::tool_def`).
    pub tools: Vec<Json>,
    /// Whether to look for tool calls in the reply (`tool_choice` other than "none"). "required" and a named
    /// function are not enforced: the model may still answer in text.
    pub tool_calls: bool,
    pub max_tokens: Option<usize>,
    pub stream: bool,
    pub think: bool,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub seed: Option<u64>,
    /// The reply ends before the first of these strings (`stop`).
    pub stop: Vec<String>,
}

pub fn chat_request(body: &[u8]) -> anyhow::Result<ChatRequest> {
    let v: Value = serde_json::from_slice(body)?;
    // Messages and tools are read again in key order: the template prints tools and call arguments as given.
    let j: Json = serde_json::from_slice(body)?;
    let Json::Arr(msgs) = j.get("messages") else { bail!("missing messages") };
    let messages = msgs.iter().map(Msg::from_json).collect::<anyhow::Result<Vec<_>>>()?;
    if messages.is_empty() { bail!("no messages"); }
    let tools = match j.get("tools") {
        Json::Null => vec![],
        Json::Arr(a) => a.iter().map(tool_def).collect::<anyhow::Result<_>>()?,
        _ => bail!("tools must be an array"),
    };
    let tool_calls = match &v["tool_choice"] {
        Value::Null => true,
        Value::String(s) if ["auto", "required", "none"].contains(&s.as_str()) => s != "none",
        Value::Object(o) if o.get("type").and_then(Value::as_str) == Some("function") => true,
        _ => bail!("tool_choice must be \"auto\", \"required\", \"none\" or a function"),
    };
    // Optional fields: absent or null means the default, anything else must be valid.
    let int = |k: &str, min: u64| match &v[k] {
        Value::Null => Ok(None),
        x => x.as_u64().filter(|&n| n >= min).map(Some).with_context(|| format!("{k} must be an integer >= {min}")),
    };
    let num = |k: &str, ok: fn(f64) -> bool, want: &str| match &v[k] {
        Value::Null => Ok(None),
        x => x.as_f64().filter(|&f| ok(f)).map(|f| Some(f as f32)).with_context(|| format!("{k} must be {want}")),
    };
    let max_tokens = int("max_completion_tokens", 1)?.or(int("max_tokens", 1)?).map(|n| n as usize);
    if int("n", 1)?.is_some_and(|n| n != 1) { bail!("only n = 1 is supported"); }
    let stop = match &v["stop"] {
        Value::Null => Some(vec![]),
        Value::String(s) => Some(vec![s.clone()]),
        Value::Array(a) => a.iter().map(|s| s.as_str().map(str::to_owned)).collect(),
        _ => None,
    }.context("stop must be a string or an array of strings")?.into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>();
    // Hold checks every prefix of every stop string at each token, so both are bounded.
    if stop.len() > 16 || stop.iter().any(|s| s.len() > 256) { bail!("stop takes at most 16 strings of at most 256 bytes"); }
    let think = v["chat_template_kwargs"]["enable_thinking"].as_bool().unwrap_or(false);
    Ok(ChatRequest { messages, tools, tool_calls, max_tokens, stream: v["stream"].as_bool().unwrap_or(false), think,
        temperature: num("temperature", |f| f >= 0.0, "a number >= 0")?, top_k: int("top_k", 0)?.map(|n| n as usize),
        top_p: num("top_p", |f| f > 0.0 && f <= 1.0, "a number in (0, 1]")?, seed: int("seed", 0)?, stop })
}
