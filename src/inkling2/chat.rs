//! `rustingstrata generate|chat|serve` for Inkling 2: greedy, GPU for the matvecs and the expert cache, state kept across turns.
//! A reply is a list of blocks, each `<|message_model|>`, a content token, the body and `<|end_message|>`; the model
//! writes the block headers itself, so the prompt has no generation prefix.
use super::forward::Inkling2Model;
use crate::{chat::{Json, Msg}, gguf::Gguf, tokenizer::Tokenizer};
use std::io::{BufRead, Write};

const USER: &str = "<|message_user|>";
const MODEL: &str = "<|message_model|>";
const SYSTEM: &str = "<|message_system|>";
const TOOL: &str = "<|message_tool|>";
const TEXT: &str = "<|content_text|>";
const THINKING: &str = "<|content_thinking|>";
const INVOKE: &str = "<|content_invoke_tool_json|>";
const END: &str = "<|end_message|>";
/// Ends an assistant turn. The model stops on it, so a later prompt adds it after the kept reply.
const DONE: &str = "<|content_model_end_sampling|>";

pub use crate::deepseek::chat::gpu_defaults;

/// The template's `tojson(sort_keys=true, separators=(",", ":"))`.
// ponytail: also sorts the keys inside a tool call's argument values (the template sorts only the top level).
fn compact(j: &Json) -> String {
    serde_json::from_str::<serde_json::Value>(&j.dump()).map_or_else(|_| j.dump(), |v| v.to_string())
}

/// The GGUF's chat template. `think` picks the effort line: 0.9 (the template's default, "high") or 0 ("none").
// ponytail: no other effort levels and no image parts. A tool message takes the name of the call it answers by
// position (the template matches `tool_call_id`).
pub fn render(msgs: &[Msg], tools: &[Json], think: bool) -> String {
    let mut s = String::new();
    if !tools.is_empty() {
        let specs: Vec<String> = tools.iter().map(|t| {
            let f = t.get("function");
            format!(r#"{{"description":{},"name":{},"parameters":{},"type":"function"}}"#, compact(f.get("description")), compact(f.get("name")), compact(f.get("parameters")))
        }).collect();
        s += &format!("{SYSTEM}tool_declare<|content_xml|>[{}]{END}", specs.join(","));
    }
    // The effort line goes before the first message that is not a system message.
    let effort = format!("{SYSTEM}{TEXT}Thinking effort level: {}{END}", if think { "0.9" } else { "0" });
    let mut effort_done = false;
    let mut called = vec![].into_iter();
    for m in msgs {
        if !effort_done && m.role != "system" {
            s += &effort;
            effort_done = true;
        }
        match m.role.as_str() {
            "assistant" => {
                let reasoning = m.reasoning.as_deref().filter(|r| !r.is_empty());
                if let Some(r) = reasoning { s += &format!("{MODEL}{THINKING}{r}{END}"); }
                if !m.content.is_empty() || (m.calls.is_empty() && reasoning.is_none()) { s += &format!("{MODEL}{TEXT}{}{END}", m.content); }
                for c in &m.calls {
                    s += &format!("{MODEL}{}{INVOKE}{{\"name\":{},\"args\":{}}}{END}", c.name, compact(&Json::Str(c.name.clone())), compact(&c.args));
                }
                s += DONE;
                called = m.calls.iter().map(|c| c.name.as_str()).collect::<Vec<_>>().into_iter();
            }
            "tool" => s += &format!("{TOOL}{}{TEXT}{}{END}", called.next().unwrap_or(""), m.content),
            "system" => s += &format!("{SYSTEM}{TEXT}{}{END}", m.content),
            _ => s += &format!("{USER}{TEXT}{}{END}", m.content),
        }
    }
    if !effort_done { s += &effort; }
    s
}

/// What a piece of the reply belongs to.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Part { Text, Think, Call }

/// Name and JSON arguments of a call block body, `{"name":..,"args":{..}}`.
pub fn parse_call(body: &str) -> Option<(String, String)> {
    let v: serde_json::Value = serde_json::from_str(body.trim()).ok()?;
    Some((v["name"].as_str()?.to_string(), v.get("args").map_or("{}".into(), |a| a.to_string())))
}

/// Greedy decode of up to `n` tokens after `logits`. `piece` gets each new piece of a block's body and returns false
/// to stop; `start` is true on the first piece of a block. Returns the token count and whether the model ended the
/// reply itself.
fn decode(m: &mut Inkling2Model, tok: &Tokenizer, mut logits: Vec<f32>, n: usize, mut piece: impl FnMut(Part, bool, &str) -> bool) -> (usize, bool) {
    let eos = tok.eos_ids();
    let id = |t: &str| tok.token_id(t);
    let (model, end) = (id(MODEL), id(END));
    let kinds = [(id(TEXT), Part::Text), (id(THINKING), Part::Think), (id(INVOKE), Part::Call)];
    // Text outside a block is content. `None`: inside a block header (the tool name of a call).
    let (mut part, mut start, mut buf, mut shown) = (Some(Part::Text), true, vec![], 0);
    for i in 0..n {
        let next = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
        if eos.contains(&next) { return (i, true); }
        let mark = if Some(next) == model { Some(None) } else if Some(next) == end { Some(Some(Part::Text)) } else { kinds.iter().find(|k| k.0 == Some(next)).map(|k| Some(k.1)) };
        if let Some(p) = mark {
            (part, start, shown) = (p, true, 0);
            buf.clear();
        } else if let Some(p) = part {
            buf.push(next);
            let s = tok.decode(&buf).unwrap_or_default();
            if !s.ends_with('\u{FFFD}') && s.len() > shown && s.is_char_boundary(shown) {
                if !piece(p, start, &s[shown..]) { return (i + 1, false); }
                (shown, start) = (s.len(), false);
            }
        }
        if i + 1 < n { logits = m.step(next); }
    }
    (n, false)
}

/// `rustingstrata generate`: raw prompt (no template), greedy, prints the continuation with its block tokens.
pub fn generate(g: &Gguf, prompt: &str, n: usize) -> anyhow::Result<()> {
    gpu_defaults();
    let tok = Tokenizer::from_gguf(g)?;
    let mut m = Inkling2Model::new(g)?;
    let max_ctx = m.cfg.n_ctx;
    let ids = tok.encode(prompt)?;
    anyhow::ensure!(!ids.is_empty() && ids.len() < max_ctx, "prompt is {} tokens, the limit is {max_ctx}", ids.len());
    let trace = std::env::var_os("STRATA_DS_TRACE").is_some();
    if trace { eprintln!("prompt tokens: {ids:?}"); }
    let mut logits = m.prefill(&ids);
    let (eos, mut out, mut shown) = (tok.eos_ids(), vec![], 0);
    for i in 0..n.min(max_ctx - ids.len()) {
        let next = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
        if eos.contains(&next) { break; }
        out.push(next);
        let s = tok.decode(&out).unwrap_or_default();
        if !s.ends_with('\u{FFFD}') && s.len() > shown && s.is_char_boundary(shown) {
            print!("{}", &s[shown..]);
            std::io::stdout().flush()?;
            shown = s.len();
        }
        if i + 1 < n { logits = m.step(next); }
    }
    if trace { eprintln!("\noutput tokens: {out:?}"); }
    println!();
    Ok(())
}

pub fn chat(g: &Gguf, system: Option<String>, n: usize, think: bool) -> anyhow::Result<()> {
    gpu_defaults();
    let tok = Tokenizer::from_gguf(g)?;
    let mut m = Inkling2Model::new(g)?;
    let max_ctx = m.cfg.n_ctx;
    let stdin = std::io::stdin();
    let mut ended = true;
    loop {
        print!("> ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 { break; }
        let line = line.trim_end();
        if line.is_empty() { continue; }
        // First turn: the full template. Later turns: the state already holds everything up to the reply, so append
        // only the end of the turn (and of a block cut short by the token limit) and the next user message.
        let text = if m.toks.is_empty() {
            let msgs: Vec<Msg> = system.iter().map(|s| Msg::new("system", s)).chain([Msg::new("user", line)]).collect();
            render(&msgs, &[], think)
        } else {
            format!("{}{DONE}{USER}{TEXT}{line}{END}", if ended { "" } else { END })
        };
        let ids = tok.encode(&text)?;
        if m.toks.len() + ids.len() >= max_ctx {
            println!("context is full ({max_ctx} tokens); restart the chat");
            break;
        }
        let logits = m.prefill(&ids);
        let left = max_ctx - m.toks.len();
        let mut last = Part::Text;
        (_, ended) = decode(&mut m, &tok, logits, n.min(left), |p, start, s| {
            // Thinking is shown dim.
            if start && p != last {
                print!("{}", match p { Part::Think => "\x1b[2m", _ if last == Part::Think => "\x1b[0m\n", _ => "\n" });
                last = p;
            }
            print!("{s}");
            std::io::stdout().flush().is_ok()
        });
        println!("{}", if last == Part::Think { "\x1b[0m" } else { "" });
    }
    Ok(())
}

/// `rustingstrata serve` for Inkling 2: one request at a time, OpenAI `/v1/chat/completions` (JSON or SSE), `/v1/models`, `/health`.
/// Thinking is on unless the request turns it off (`chat_template_kwargs.enable_thinking: false`); it arrives as `reasoning_content`.
/// ponytail: greedy only, no stop strings; the state is reused only when the new prompt extends the last sequence, so
/// a client must send `reasoning_content` back to skip the prompt of the earlier turns.
pub fn serve(g: &Gguf, name: &str, addr: &str, api_key: Option<&str>) -> anyhow::Result<()> {
    use anyhow::Context;
    use serde_json::json;
    gpu_defaults();
    let tok = Tokenizer::from_gguf(g)?;
    let mut m = Inkling2Model::new(g)?;
    let max_ctx = m.cfg.n_ctx;
    crate::serve::own(name, addr, api_key, "inkling2", |c, q| {
        let ids = tok.encode(&render(&q.messages, &q.tools, q.think.unwrap_or(true))).ok().filter(|i| !i.is_empty()).context("cannot encode prompt")?;
        anyhow::ensure!(ids.len() < max_ctx, "prompt is {} tokens, the limit is {max_ctx}", ids.len());
        let t = std::time::Instant::now();
        let keep = if m.toks.len() < ids.len() && ids.starts_with(&m.toks) { m.toks.len() } else { m.reset(); 0 };
        let logits = m.prefill(&ids[keep..]);
        let prefill_s = t.elapsed().as_secs_f64();
        let n = q.max_tokens.unwrap_or(256).min(max_ctx - ids.len());
        if q.stream { let _ = write!(c, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n"); }
        let chunk = |d: serde_json::Value, fin: Option<&str>| json!({"object": "chat.completion.chunk", "model": name, "choices": [{"index": 0, "delta": d, "finish_reason": fin}]});
        let tools = q.tool_calls && !q.tools.is_empty();
        let (mut content, mut reasoning, mut bodies) = (String::new(), String::new(), Vec::<String>::new());
        let (count, stopped) = decode(&mut m, &tok, logits, n, |p, start, s| {
            let key = match p {
                Part::Call if tools => {
                    if start { bodies.push(String::new()); }
                    *bodies.last_mut().unwrap() += s;
                    return true;
                }
                Part::Think => { reasoning += s; "reasoning_content" }
                _ => { content += s; "content" }
            };
            !q.stream || write!(c, "data: {}\n\n", chunk(json!({key: s}), None)).is_ok()
        });
        let calls: Vec<_> = bodies.iter().filter_map(|b| parse_call(b)).enumerate().map(|(k, (f, a))| {
            let mut c = json!({"id": format!("call_{k}"), "type": "function", "function": {"name": f, "arguments": a}});
            if q.stream { c["index"] = k.into(); }
            c
        }).collect();
        let done = if !calls.is_empty() { "tool_calls" } else if stopped { "stop" } else { "length" };
        if q.stream {
            if !calls.is_empty() { let _ = write!(c, "data: {}\n\n", chunk(json!({"tool_calls": calls}), None)); }
            let _ = write!(c, "data: {}\n\ndata: [DONE]\n\n", chunk(json!({}), Some(done)));
        } else {
            let mut message = json!({"role": "assistant", "content": content});
            if !reasoning.is_empty() { message["reasoning_content"] = reasoning.into(); }
            if !calls.is_empty() { message["tool_calls"] = calls.into(); }
            let b = json!({"object": "chat.completion", "model": name, "choices": [{"index": 0, "message": message, "finish_reason": done}],
                "usage": {"prompt_tokens": ids.len(), "completion_tokens": count, "total_tokens": ids.len() + count}});
            crate::serve::send(c, "200 OK", "application/json", &b.to_string());
        }
        Ok(crate::serve::Done { prompt: ids.len(), cached: keep, completion: count, prefill_s, finish: done })
    })
}

#[test]
fn template_and_calls() {
    use crate::chat::Call;
    assert_eq!(render(&[Msg::new("system", "be brief"), Msg::new("user", "hi")], &[], true),
        "<|message_system|><|content_text|>be brief<|end_message|>\
         <|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>\
         <|message_user|><|content_text|>hi<|end_message|>");
    let obj = |v: &str| -> Json { serde_json::from_str(v).unwrap() };
    let tool = obj(r#"{"type":"function","function":{"name":"get","description":"weather","parameters":{"type":"object","properties":{}}}}"#);
    let mut call = Msg::new("assistant", "");
    call.reasoning = Some("plan".into());
    call.calls = vec![Call { name: "get".into(), args: obj(r#"{"n":3,"city":"Kyiv"}"#) }];
    assert_eq!(render(&[Msg::new("user", "hi"), call, Msg::new("tool", "sun")], &[tool], false),
        "<|message_system|>tool_declare<|content_xml|>[{\"description\":\"weather\",\"name\":\"get\",\"parameters\":{\"properties\":{},\"type\":\"object\"},\"type\":\"function\"}]<|end_message|>\
         <|message_system|><|content_text|>Thinking effort level: 0<|end_message|>\
         <|message_user|><|content_text|>hi<|end_message|>\
         <|message_model|><|content_thinking|>plan<|end_message|>\
         <|message_model|>get<|content_invoke_tool_json|>{\"name\":\"get\",\"args\":{\"city\":\"Kyiv\",\"n\":3}}<|end_message|><|content_model_end_sampling|>\
         <|message_tool|>get<|content_text|>sun<|end_message|>");
    assert_eq!(parse_call(r#"{"name":"get","args":{"city":"Kyiv","n":3}}"#), Some(("get".into(), r#"{"city":"Kyiv","n":3}"#.into())));
}
