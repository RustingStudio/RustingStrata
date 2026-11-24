//! `rustingstrata generate|chat|serve` for Hy4-flash: greedy, CUDA for the matvecs and expert staging, KV kept across turns.
use super::forward::Hy4Model;
use crate::{chat::{Json, Msg}, gguf::Gguf, tokenizer::Tokenizer};
use std::io::{BufRead, Write};

const START: &str = "<｜hy_start:opensource｜>";
const MID: &str = "<｜hy_middle:opensource｜>";
const END: &str = "<｜hy_end:opensource｜>";
const THINK: &str = "<think:opensource>";
const THINK_END: &str = "</think:opensource>";
const CALLS: &str = "<tool_calls:opensource>";
const CALLS_END: &str = "</tool_calls:opensource>";
const CALL: &str = "<tool_call:opensource>";
const CALL_END: &str = "</tool_call:opensource>";
const KEY: &str = "<arg_key:opensource>";
const KEY_END: &str = "</arg_key:opensource>";
const VAL: &str = "<arg_value:opensource>";
const VAL_END: &str = "</arg_value:opensource>";

pub use crate::deepseek::chat::gpu_defaults;

/// The GGUF's chat template with `add_generation_prompt`. `think` picks `reasoning_effort` high or no_think;
/// `preserve` is the template's `preserved_thinking` (on when tools are given, unless said otherwise).
// ponytail: tool responses keep the message order (the template reorders them by call id); no image/video parts.
pub fn render(msgs: &[Msg], tools: &[Json], think: bool, preserve: Option<bool>) -> String {
    let preserve = preserve.unwrap_or(!tools.is_empty());
    let mode = format!("<｜reasoning_mode:opensource｜>reasoning_effort:{}", if think { "high" } else { "no_think" });
    let tools_prompt = || {
        format!("# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{}\n</tools>\n\n\
            For function call returns, you should first print {CALLS}\nFor each function call, you should return object like:\n\
            {CALL}{{function-name}}{KEY}{{arg-key-1}}{KEY_END}{VAL}{{arg-value-1}}{VAL_END}{KEY}{{arg-key-2}}{KEY_END}{VAL}{{arg-value-2}}{VAL_END}...{CALL_END}\n\
            At the end of function call returns, you should print {CALLS_END}",
            tools.iter().map(|t| t.get("function").dump()).collect::<Vec<_>>().join("\n"))
    };
    let mut s = String::new();
    if msgs.first().is_none_or(|m| m.role != "system") {
        s += &format!("{START}system{MID}{}{mode}{END}", if tools.is_empty() { String::new() } else { tools_prompt() });
    }
    let last_user = msgs.iter().rposition(|m| m.role == "user");
    for (i, m) in msgs.iter().enumerate() {
        // Consecutive tool messages share one turn.
        let (tool, prev_tool, next_tool) = (m.role == "tool", i > 0 && msgs[i - 1].role == "tool", msgs.get(i + 1).is_some_and(|x| x.role == "tool"));
        if !(tool && prev_tool) { s += &format!("{START}{}{MID}", m.role); }
        match m.role.as_str() {
            "system" if i == 0 => {
                if !tools.is_empty() {
                    s += &tools_prompt();
                    if !m.content.is_empty() { s += "\n\n"; }
                }
                s += &m.content;
                s += &mode;
            }
            "assistant" => {
                let r = m.reasoning.as_deref().filter(|r| !r.is_empty() && think && (preserve || last_user.is_none_or(|u| i > u))).unwrap_or("");
                s += &format!("{THINK}{r}{THINK_END}{}", m.content);
                if !m.calls.is_empty() {
                    s += CALLS;
                    for c in &m.calls {
                        s += &format!("{CALL}{}", c.name);
                        if let Json::Obj(args) = &c.args {
                            for (k, v) in args {
                                s += &format!("{KEY}{k}{KEY_END}{VAL}{}{VAL_END}", v.as_str().map_or_else(|| v.dump(), str::to_string));
                            }
                        }
                        s += CALL_END;
                    }
                    s += CALLS_END;
                }
            }
            "tool" => s += &format!("<tool_response:opensource>{}</tool_response:opensource>", m.content),
            _ => s += &m.content,
        }
        if !(tool && next_tool) { s += END; }
    }
    if msgs.last().is_none_or(|m| m.role != "assistant") {
        s += &format!("{START}assistant{MID}{THINK}{}", if think { "" } else { THINK_END });
    }
    s
}

/// The tool calls of a reply: the text before the call block and (name, JSON arguments) per call. An argument is
/// JSON when it parses as JSON, else a string. The thinking part is skipped, so a tag inside it is not a call.
pub fn parse_calls(text: &str) -> Option<(String, Vec<(String, String)>)> {
    let from = text.rfind(THINK_END).map_or(0, |p| p + THINK_END.len());
    let start = from + text[from..].find(CALLS)?;
    let body = text[start + CALLS.len()..].split(CALLS_END).next()?;
    let mut calls = vec![];
    for c in body.split(CALL).skip(1) {
        let c = c.split(CALL_END).next()?;
        let mut parts = c.split(KEY);
        let name = parts.next()?.trim().to_string();
        let mut args = serde_json::Map::new();
        for p in parts {
            let (k, rest) = p.split_once(KEY_END)?;
            let v = rest.split_once(VAL)?.1.split(VAL_END).next()?;
            args.insert(k.to_string(), serde_json::from_str(v).unwrap_or_else(|_| v.into()));
        }
        calls.push((name, serde_json::Value::Object(args).to_string()));
    }
    (!calls.is_empty()).then(|| (text[from..start].trim().to_string(), calls))
}

/// Greedy decode of up to `n` tokens after `logits`; `piece` gets each new piece of text and returns false to stop.
/// Returns the tokens and whether the model ended the reply itself.
fn decode(m: &mut Hy4Model, tok: &Tokenizer, mut logits: Vec<f32>, n: usize, mut piece: impl FnMut(&str) -> bool) -> (Vec<u32>, bool) {
    let eos = tok.eos_ids();
    let (mut out, mut shown) = (vec![], 0);
    for i in 0..n {
        let next = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
        if eos.contains(&next) { return (out, true); }
        out.push(next);
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
    let mut m = Hy4Model::new(g)?;
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
    let mut m = Hy4Model::new(g)?;
    let max_ctx = m.cfg.n_ctx;
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
            render(&msgs, &[], think, None)
        } else {
            format!("{END}{}", render(&user, &[], think, None).split_once(END).unwrap().1)
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

/// The chat of `rustingstrata serve` for Hy4-flash (see `serve::Own`): one request at a time, JSON or SSE.
/// ponytail: greedy only, no stop strings; thinking (`"think": true`) arrives inside `content` up to `</think:opensource>`;
/// the KV cache is reused only when the new prompt extends the last sequence.
pub fn serve<'a>(g: &'a Gguf, tok: &'a Tokenizer, name: &'a str) -> anyhow::Result<crate::serve::Own<'a>> {
    use anyhow::Context;
    use serde_json::json;
    gpu_defaults();
    let mut m = Hy4Model::new(g)?;
    let max_ctx = m.cfg.n_ctx;
    Ok(Box::new(move |c: &mut std::net::TcpStream, q: &crate::serve::ChatRequest| {
        let ids = tok.encode(&render(&q.messages, &q.tools, q.think.unwrap_or(false), q.preserve_thinking)).ok().filter(|i| !i.is_empty()).context("cannot encode prompt")?;
        anyhow::ensure!(ids.len() < max_ctx, "prompt is {} tokens, the limit is {max_ctx}", ids.len());
        let t = std::time::Instant::now();
        let keep = if m.toks.len() < ids.len() && ids.starts_with(&m.toks) { m.toks.len() } else { m.reset(); 0 };
        let logits = m.prefill(&ids[keep..]);
        let prefill_s = t.elapsed().as_secs_f64();
        let n = q.max_tokens.unwrap_or(256).min(max_ctx - ids.len());
        if q.stream { let _ = write!(c, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n"); }
        let chunk = |d: serde_json::Value, fin: Option<&str>| json!({"object": "chat.completion.chunk", "model": name, "choices": [{"index": 0, "delta": d, "finish_reason": fin}]});
        let tools = q.tool_calls && !q.tools.is_empty();
        let mut hold = crate::chat::Hold::new(tools, vec![]).tagged(CALLS);
        let (out, stopped) = decode(&mut m, &tok, logits, n, |p| {
            let t = hold.push(p);
            !q.stream || t.is_empty() || write!(c, "data: {}\n\n", chunk(json!({"content": t}), None)).is_ok()
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
        if q.stream {
            match &calls {
                Some(x) => { let _ = write!(c, "data: {}\n\n", chunk(calls_json(x, true), None)); }
                None if !hold.rest().is_empty() => { let _ = write!(c, "data: {}\n\n", chunk(json!({"content": hold.rest()}), None)); }
                None => {}
            }
            let _ = write!(c, "data: {}\n\ndata: [DONE]\n\n", chunk(json!({}), Some(done)));
        } else {
            let message = match &calls { Some(x) => calls_json(x, false), None => json!({"role": "assistant", "content": hold.text}) };
            let b = json!({"object": "chat.completion", "model": name, "choices": [{"index": 0, "message": message, "finish_reason": done}],
                "usage": {"prompt_tokens": ids.len(), "completion_tokens": out.len(), "total_tokens": ids.len() + out.len()}});
            crate::serve::send(c, "200 OK", "application/json", &b.to_string());
        }
        Ok(crate::serve::Done { prompt: ids.len(), cached: keep, completion: out.len(), prefill_s, finish: done })
    }))
}

#[test]
fn template_and_calls() {
    let user = [Msg::new("user", "hi")];
    assert_eq!(render(&user, &[], false, None),
        "<｜hy_start:opensource｜>system<｜hy_middle:opensource｜><｜reasoning_mode:opensource｜>reasoning_effort:no_think<｜hy_end:opensource｜>\
         <｜hy_start:opensource｜>user<｜hy_middle:opensource｜>hi<｜hy_end:opensource｜>\
         <｜hy_start:opensource｜>assistant<｜hy_middle:opensource｜><think:opensource></think:opensource>");
    let reply = "<think:opensource>plan</think:opensource>ok<tool_calls:opensource><tool_call:opensource>get<arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Kyiv</arg_value:opensource><arg_key:opensource>n</arg_key:opensource><arg_value:opensource>3</arg_value:opensource></tool_call:opensource></tool_calls:opensource>";
    assert_eq!(parse_calls(reply), Some(("ok".into(), vec![("get".into(), r#"{"city":"Kyiv","n":3}"#.into())])));
}
