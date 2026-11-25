//! `rustingstrata generate|chat|serve` for Kolibri 1.1: greedy, CUDA for the matvecs and expert staging, KV kept across turns.
use super::forward::KolibriModel;
use crate::{chat::{Json, Msg}, gguf::Gguf, tokenizer::Tokenizer};
use std::io::{BufRead, Write};

const END: &str = "<|im_end|>\n";
const THINK_END: &str = "</think>";
const CALL: &str = "<tool_call>";
const CALL_END: &str = "</tool_call>";

pub use crate::deepseek::chat::gpu_defaults;

/// The template's reasoning effort: `reasoning_effort` when given, else "none" when thinking is switched off, else high.
pub fn effort(think: Option<bool>, effort: Option<&str>) -> &str {
    effort.unwrap_or(if think == Some(false) { "none" } else { "high" })
}

/// The GGUF's chat template with `add_generation_prompt`. `effort` is its `reasoning_effort` ("none" disables
/// thinking); `preserve` is its `preserve_thinking`.
pub fn render(msgs: &[Msg], tools: &[Json], effort: &str, preserve: bool) -> String {
    let sentence = match effort {
        "none" => "Reasoning is disabled. Proceed straight to answering according to the user's instructions.",
        "minimal" | "low" => "Reasoning effort is set to low. Think briefly through only the essential steps in the user's language, then proceed directly to the answer.",
        "medium" => "Reasoning effort is set to medium. Think through the task methodically in the user's language, check key assumptions, and provide a well-supported answer.",
        _ => "Reasoning effort is set to high. Think carefully through the task in the user's language, validate key assumptions, consider plausible alternatives, and prioritize correctness and clarity.",
    };
    let mut s = String::from("<|im_start|>system\n");
    if let Some(m) = msgs.first().filter(|m| m.role == "system") { s += &format!("{}\n\n", m.content); }
    s += &format!("# Reasoning effort\n\n{sentence}");
    if !tools.is_empty() {
        s += "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>";
        for t in tools { s += &format!("\n{}", t.dump()); }
        s += "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";
    }
    s += END;
    // Replies after the last real user turn keep their thinking.
    let last_query = msgs.iter().rposition(|m| m.role == "user" && !(m.content.starts_with("<tool_response>") && m.content.ends_with("</tool_response>"))).unwrap_or(msgs.len().saturating_sub(1));
    for (i, m) in msgs.iter().enumerate() {
        match m.role.as_str() {
            "system" if i == 0 => {}
            "assistant" => {
                let (mut content, mut reasoning) = (m.content.as_str(), m.reasoning.as_deref());
                if reasoning.is_none() && let Some((r, c)) = content.rsplit_once(THINK_END) {
                    reasoning = Some(r.trim_end_matches('\n').rsplit("<think>").next().unwrap_or(""));
                    content = c;
                }
                s += "<|im_start|>assistant\n";
                if i > last_query || preserve {
                    s += &format!("<think>\n{}\n</think>\n\n", reasoning.filter(|r| !r.trim().is_empty()).map_or("", |r| r.trim_matches('\n')));
                }
                let content = content.trim_start_matches('\n');
                s += content;
                for (k, c) in m.calls.iter().enumerate() {
                    if k > 0 || !content.is_empty() { s += "\n"; }
                    s += &format!("{CALL}\n{{\"name\": \"{}\", \"arguments\": {}}}\n{CALL_END}", c.name, c.args.dump());
                }
                s += END;
            }
            "tool" => {
                // Consecutive tool messages share one user turn.
                if i == 0 || msgs[i - 1].role != "tool" { s += "<|im_start|>user"; }
                s += &format!("\n<tool_response>\n{}\n</tool_response>", m.content);
                if msgs.get(i + 1).is_none_or(|x| x.role != "tool") { s += END; }
            }
            role => s += &format!("<|im_start|>{role}\n{}{END}", m.content),
        }
    }
    s += "<|im_start|>assistant\n";
    if effort == "none" { s += "<think>\n\n</think>\n\n"; }
    s
}

/// The tool calls of a reply: the text before the first call and (name, JSON arguments) per call. The thinking part
/// is skipped, so a tag inside it is not a call.
pub fn parse_calls(text: &str) -> Option<(String, Vec<(String, String)>)> {
    let from = text.rfind(THINK_END).map_or(0, |p| p + THINK_END.len());
    let start = from + text[from..].find(CALL)?;
    let mut calls = vec![];
    for c in text[start..].split(CALL).skip(1) {
        let v: serde_json::Value = serde_json::from_str(c.split(CALL_END).next()?.trim()).ok()?;
        let args = match &v["arguments"] { serde_json::Value::String(s) => s.clone(), a => a.to_string() };
        calls.push((v["name"].as_str()?.to_string(), args));
    }
    (!calls.is_empty()).then(|| (text[from..start].trim().to_string(), calls))
}

/// Greedy decode of up to `n` tokens after `logits`; `piece` gets each new piece of text and returns false to stop.
/// Returns the tokens and whether the model ended the reply itself.
fn decode(m: &mut KolibriModel, tok: &Tokenizer, mut logits: Vec<f32>, n: usize, mut piece: impl FnMut(&str) -> bool) -> (Vec<u32>, bool) {
    let eos = tok.eos_ids();
    let (mut out, mut shown) = (vec![], 0);
    for i in 0..n {
        let next = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
        if eos.contains(&next) { return (out, true); }
        out.push(next);
        crate::term::decoded(1);
        let s = tok.decode(&out).unwrap_or_default();
        if !s.ends_with('\u{FFFD}') && s.len() > shown && s.is_char_boundary(shown) {
            if !piece(&s[shown..]) { break; }
            shown = s.len();
        }
        if i + 1 < n { logits = m.step(next); }
    }
    (out, false)
}

fn show(s: &str) -> bool {
    print!("{s}");
    std::io::stdout().flush().is_ok()
}

/// `rustingstrata generate`: raw prompt (no template), greedy, prints the continuation.
pub fn generate(g: &Gguf, prompt: &str, n: usize) -> anyhow::Result<()> {
    gpu_defaults();
    let tok = Tokenizer::from_gguf(g)?;
    let mut m = KolibriModel::new(g)?;
    let max_ctx = m.cfg.n_ctx;
    let ids = tok.encode(prompt)?;
    anyhow::ensure!(!ids.is_empty() && ids.len() < max_ctx, "prompt is {} tokens, the limit is {max_ctx}", ids.len());
    if std::env::var_os("STRATA_DS_TRACE").is_some() { crate::say!("prompt tokens: {ids:?}"); }
    let logits = m.prefill(&ids);
    let (out, _) = decode(&mut m, &tok, logits, n.min(max_ctx - ids.len()), show);
    if std::env::var_os("STRATA_DS_TRACE").is_some() { crate::say!("\noutput tokens: {out:?}"); }
    println!();
    Ok(())
}

pub fn chat(g: &Gguf, system: Option<String>, n: usize, think: bool) -> anyhow::Result<()> {
    gpu_defaults();
    let tok = Tokenizer::from_gguf(g)?;
    let mut m = KolibriModel::new(g)?;
    let max_ctx = m.cfg.n_ctx;
    let effort = if think { "high" } else { "none" };
    let stdin = std::io::stdin();
    loop {
        print!("{}", crate::term::prompt());
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 { break; }
        let line = line.trim_end();
        if line.is_empty() { continue; }
        // First turn: the full template. Later turns: the cache already holds everything up to the reply, so append only
        // the end token and the next user turn (earlier think blocks stay in the cache).
        let user = [Msg::new("user", line)];
        let text = if m.toks.is_empty() {
            let msgs: Vec<Msg> = system.iter().map(|s| Msg::new("system", s)).chain(user).collect();
            render(&msgs, &[], effort, false)
        } else {
            format!("{END}{}", render(&user, &[], effort, false).split_once(END).unwrap().1)
        };
        let ids = tok.encode(&text)?;
        if m.toks.len() + ids.len() >= max_ctx {
            println!("context is full ({max_ctx} tokens); restart the chat");
            break;
        }
        let logits = m.prefill(&ids);
        let left = max_ctx - m.toks.len();
        decode(&mut m, &tok, logits, n.min(left), show);
        println!();
    }
    Ok(())
}

/// The chat of `rustingstrata serve` for Kolibri 1.1 (see `serve::Own`): one request at a time, JSON or SSE.
/// ponytail: greedy only, no stop strings; thinking arrives inside `content` up to `</think>` (split it into
/// `reasoning_content` when a client needs that); the KV cache is reused only when the new prompt extends the last
/// sequence.
pub fn serve<'a>(g: &'a Gguf, tok: &'a Tokenizer, name: &'a str) -> anyhow::Result<crate::serve::Own<'a>> {
    use anyhow::Context;
    use serde_json::json;
    gpu_defaults();
    let mut m = KolibriModel::new(g)?;
    let max_ctx = m.cfg.n_ctx;
    Ok(Box::new(move |c: &mut std::net::TcpStream, q: &crate::serve::ChatRequest| {
        let text = render(&q.messages, &q.tools, effort(q.think, q.reasoning_effort.as_deref()), q.preserve_thinking.unwrap_or(false));
        let ids = tok.encode(&text).ok().filter(|i| !i.is_empty()).context("cannot encode prompt")?;
        anyhow::ensure!(ids.len() < max_ctx, "prompt is {} tokens, the limit is {max_ctx}", ids.len());
        let t = std::time::Instant::now();
        let keep = if m.toks.len() < ids.len() && ids.starts_with(&m.toks) { m.toks.len() } else { m.reset(); 0 };
        if q.stream && !q.head_sent { let _ = write!(c, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n"); }
        let logits = crate::serve::alive(c, q.stream, || m.prefill(&ids[keep..]));
        let prefill_s = t.elapsed().as_secs_f64();
        let n = q.max_tokens.unwrap_or(usize::MAX).min(max_ctx - ids.len());
        let chunk = |d: serde_json::Value, fin: Option<&str>| json!({"object": "chat.completion.chunk", "model": name, "choices": [{"index": 0, "delta": d, "finish_reason": fin}]});
        let tools = q.tool_calls && !q.tools.is_empty();
        let mut hold = crate::chat::Hold::new(tools, vec![]).tagged(CALL);
        let mut alive = std::time::Instant::now();
        let (out, stopped) = decode(&mut m, &tok, logits, n, |p| {
            let t = hold.push(p);
            !q.stream || if t.is_empty() { crate::serve::held(c, &mut alive) } else { write!(c, "data: {}\n\n", chunk(json!({"content": t}), None)).is_ok() }
        });
        let calls = if tools { parse_calls(&hold.text) } else { None };
        let done = if calls.is_some() { "tool_calls" } else if stopped { "stop" } else { "length" };
        let calls_json = |(content, calls): &(String, Vec<(String, String)>), stream: bool| {
            let c: Vec<_> = calls.iter().enumerate().map(|(k, (f, a))| {
                let mut c = json!({"id": format!("call_{k}"), "type": "function", "function": {"name": f, "arguments": a}});
                if stream { c["index"] = k.into(); }
                c
            }).collect();
            if stream { json!({"tool_calls": c}) } else { json!({"role": "assistant", "content": if content.is_empty() { None } else { Some(content) }, "tool_calls": c}) }
        };
        let usage = json!({"prompt_tokens": ids.len(), "completion_tokens": out.len(), "total_tokens": ids.len() + out.len(), "prompt_tokens_details": {"cached_tokens": keep}});
        if q.stream {
            match &calls {
                Some(x) => { let _ = write!(c, "data: {}\n\n", chunk(calls_json(x, true), None)); }
                None if !hold.rest().is_empty() => { let _ = write!(c, "data: {}\n\n", chunk(json!({"content": hold.rest()}), None)); }
                None => {}
            }
            let _ = write!(c, "data: {}\n\n", chunk(json!({}), Some(done)));
            if q.include_usage { let _ = write!(c, "data: {}\n\n", json!({"object": "chat.completion.chunk", "model": name, "choices": [], "usage": usage})); }
            let _ = write!(c, "data: [DONE]\n\n");
        } else {
            let message = match &calls { Some(x) => calls_json(x, false), None => json!({"role": "assistant", "content": hold.text}) };
            let b = json!({"object": "chat.completion", "model": name, "choices": [{"index": 0, "message": message, "finish_reason": done}], "usage": usage});
            crate::serve::send(c, "200 OK", "application/json", &b.to_string());
        }
        Ok(crate::serve::Done { prompt: ids.len(), cached: keep, completion: out.len(), prefill_s, finish: done })
    }))
}

#[test]
fn template_and_calls() {
    let user = [Msg::new("user", "hi")];
    assert_eq!(render(&user, &[], "none", false),
        "<|im_start|>system\n# Reasoning effort\n\nReasoning is disabled. Proceed straight to answering according to the user's instructions.<|im_end|>\n\
         <|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    // A reply before the last user turn loses its thinking; a tool round after it keeps the think block.
    let mut a = Msg::new("assistant", "<think>\nplan\n</think>\n\nok");
    let mut msgs = vec![Msg::new("system", "S"), Msg::new("user", "q"), a.clone(), Msg::new("user", "q2")];
    let r = render(&msgs, &[], "low", false);
    assert!(r.starts_with("<|im_start|>system\nS\n\n# Reasoning effort\n\nReasoning effort is set to low."), "{r}");
    assert!(r.ends_with("<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\nok<|im_end|>\n<|im_start|>user\nq2<|im_end|>\n<|im_start|>assistant\n"), "{r}");
    a.calls = vec![crate::chat::Call { name: "get".into(), args: serde_json::from_str(r#"{"city": "Kyiv"}"#).unwrap() }];
    msgs.extend([a, Msg::new("tool", "sun"), Msg::new("tool", "rain")]);
    assert!(render(&msgs, &[], "high", false).ends_with("<|im_start|>assistant\n<think>\nplan\n</think>\n\nok\n<tool_call>\n{\"name\": \"get\", \"arguments\": {\"city\": \"Kyiv\"}}\n</tool_call><|im_end|>\n\
         <|im_start|>user\n<tool_response>\nsun\n</tool_response>\n<tool_response>\nrain\n</tool_response><|im_end|>\n<|im_start|>assistant\n"));
    let reply = "<think>\n<tool_call> no\n</think>\n\nok\n<tool_call>\n{\"name\": \"get\", \"arguments\": {\"city\": \"Kyiv\", \"n\": 3}}\n</tool_call>\n<tool_call>\n{\"name\": \"b\", \"arguments\": {}}\n</tool_call>";
    assert_eq!(parse_calls(reply), Some(("ok".into(), vec![("get".into(), r#"{"city":"Kyiv","n":3}"#.into()), ("b".into(), "{}".into())])));
}
