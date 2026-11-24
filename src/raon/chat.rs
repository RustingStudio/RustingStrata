//! `rustingstrata generate|chat|serve` for KRAFTON Raon-1 mini (`raon1`).
use super::forward::RaonModel;
use crate::{chat::{Json, Msg}, gguf::Gguf, tokenizer::Tokenizer};
use std::io::{BufRead, Write};

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";
const THINK_OPEN: &str = "<think>\n";
const THINK_EMPTY: &str = "<think></think>\n\n";
const THINK_END: &str = "</think>";
const CALL_TAG: &str = "<tool_call>";

pub use crate::deepseek::chat::gpu_defaults;

/// Render messages + optional tools using KRAFTON Raon-1 mini's ChatML SFT template.
pub fn render(msgs: &[Msg], tools: &[Json], think: bool, preserve: Option<bool>) -> String {
    let truncate_history = preserve.map(|p| !p).unwrap_or(false);
    let mut s = String::new();
    let has_tools = !tools.is_empty();
    let first_sys = msgs.first().is_some_and(|m| m.role == "system");

    if has_tools {
        s += &format!("{IM_START}system\n# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for t in tools {
            s += "\n";
            s += &t.dump();
        }
        s += "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n\
              <tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n\
              <parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n\
              </parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n\
              - Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n\
              - Required parameters MUST be specified\n\
              - You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n\
              - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n\
              </IMPORTANT>";
        if first_sys {
            let c = msgs[0].content.trim();
            if !c.is_empty() {
                s += "\n\n";
                s += c;
            }
        }
        s += &format!("{IM_END}\n");
    } else if first_sys {
        s += &format!("{IM_START}system\n{}{IM_END}\n", msgs[0].content.trim());
    }

    let mut last_query_idx = msgs.len().saturating_sub(1);
    for (i, m) in msgs.iter().enumerate().rev() {
        if m.role == "user" {
            let c = m.content.trim();
            if !(c.starts_with("<tool_response>") && c.ends_with("</tool_response>")) {
                last_query_idx = i;
                break;
            }
        }
    }

    for (i, m) in msgs.iter().enumerate() {
        let content = m.content.trim();
        match m.role.as_str() {
            "system" => {
                if i > 0 {
                    s += &format!("{IM_START}system\n{content}{IM_END}\n");
                }
            }
            "user" => {
                s += &format!("{IM_START}user\n{content}{IM_END}\n");
            }
            "assistant" => {
                let (reasoning, body) = if let Some(r) = &m.reasoning {
                    (r.as_str(), content)
                } else if content.starts_with("<think>") && content.contains(THINK_END) {
                    let (a, b) = content.split_once(THINK_END).unwrap();
                    (a[7..].trim_matches('\n'), b.trim_start_matches('\n'))
                } else {
                    ("", content)
                };
                s += &format!("{IM_START}assistant\n");
                let keep_r = !truncate_history || i > last_query_idx;
                if keep_r && !reasoning.trim().is_empty() {
                    s += &format!("<think>\n{}\n</think>\n\n", reasoning.trim());
                } else {
                    s += THINK_EMPTY;
                }
                s += body;
                for (ci, c) in m.calls.iter().enumerate() {
                    if ci == 0 {
                        if !body.trim().is_empty() { s += "\n\n"; }
                    } else {
                        s += "\n";
                    }
                    s += &format!("<tool_call>\n<function={}>\n", c.name);
                    if let Json::Obj(args) = &c.args {
                        for (k, v) in args {
                            let val = v.as_str().map_or_else(|| v.dump(), str::to_string);
                            s += &format!("<parameter={k}>\n{val}\n</parameter>\n");
                        }
                    }
                    s += "</function>\n</tool_call>";
                }
                s += &format!("{IM_END}\n");
            }
            "tool" => {
                let prev_tool = i > 0 && msgs[i - 1].role == "tool";
                let next_tool = msgs.get(i + 1).is_some_and(|x| x.role == "tool");
                if !prev_tool { s += &format!("{IM_START}user"); }
                s += &format!("\n<tool_response>\n{content}\n</tool_response>");
                if !next_tool { s += &format!("{IM_END}\n"); }
            }
            _ => {}
        }
    }

    s += &format!("{IM_START}assistant\n");
    s += if think { THINK_OPEN } else { THINK_EMPTY };
    s
}

/// Parse `<tool_call><function=name><parameter=k>v</parameter></function></tool_call>` blocks.
pub fn parse_calls(text: &str) -> Option<(String, Vec<(String, String)>)> {
    let from = text.rfind(THINK_END).map_or(0, |p| p + THINK_END.len());
    let start = from + text[from..].find(CALL_TAG)?;
    let mut calls = Vec::new();
    for block in text[start..].split(CALL_TAG).skip(1) {
        let Some(inner) = block.split("</tool_call>").next() else { continue; };
        let Some((_, after_fn)) = inner.split_once("<function=") else { continue; };
        let Some((name, rest)) = after_fn.split_once('>') else { continue; };
        let Some(body) = rest.split("</function>").next() else { continue; };
        let mut args = serde_json::Map::new();
        for part in body.split("<parameter=").skip(1) {
            let Some((k, vrest)) = part.split_once('>') else { continue; };
            let Some(val) = vrest.split("</parameter>").next() else { continue; };
            let val = val.trim();
            args.insert(k.trim().to_string(), serde_json::from_str(val).unwrap_or_else(|_| val.into()));
        }
        calls.push((name.trim().to_string(), serde_json::Value::Object(args).to_string()));
    }
    (!calls.is_empty()).then(|| (text[from..start].trim().to_string(), calls))
}

fn decode(m: &mut RaonModel, tok: &Tokenizer, mut logits: Vec<f32>, n: usize, mut piece: impl FnMut(&str) -> bool) -> (Vec<u32>, bool) {
    let eos = tok.eos_ids();
    let spec_on = std::env::var("STRATA_RAON_SPEC").ok().as_deref() != Some("0");
    let (mut out, mut shown) = (Vec::new(), 0);
    let mut emit = |out: &[u32], shown: &mut usize| -> bool {
        let s = tok.decode(out).unwrap_or_default();
        if !s.ends_with('\u{FFFD}') && s.len() > *shown && s.is_char_boundary(*shown) {
            if !piece(&s[*shown..]) { return false; }
            *shown = s.len();
        }
        true
    };
    while out.len() < n {
        let next = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
        if eos.contains(&next) { return (out, true); }
        out.push(next);
        if !emit(&out, &mut shown) { break; }
        if out.len() >= n { break; }
        let rem = n - out.len();
        if spec_on && rem >= 2 {
            let max_d = rem.min(3);
            let (draft, order) = crate::ngram::suffix_draft_with_order(&m.toks, next, max_d, &eos);
            if draft.len() >= 2 && order >= 3 {
                let (accepted, next_logits) = m.verify_greedy(next, &draft, &eos);
                logits = next_logits;
                for t in accepted {
                    out.push(t);
                    if !emit(&out, &mut shown) { return (out, false); }
                    if out.len() >= n { return (out, false); }
                }
                continue;
            }
        }
        logits = m.step(next);
    }
    (out, false)
}

fn show(s: &str) -> bool {
    print!("{s}");
    std::io::stdout().flush().is_ok()
}

pub fn generate(g: &Gguf, prompt: &str, n: usize) -> anyhow::Result<()> {
    gpu_defaults();
    let tok = Tokenizer::from_gguf(g)?;
    let mut m = RaonModel::new(g)?;
    let max_ctx = m.cfg.n_ctx;
    let ids = tok.encode(prompt)?;
    anyhow::ensure!(!ids.is_empty() && ids.len() < max_ctx, "prompt is {} tokens, the limit is {max_ctx}", ids.len());
    let logits = m.prefill(&ids);
    decode(&mut m, &tok, logits, n.min(max_ctx - ids.len()), show);
    println!();
    Ok(())
}

pub fn chat(g: &Gguf, system: Option<String>, n: usize, think: bool) -> anyhow::Result<()> {
    gpu_defaults();
    let tok = Tokenizer::from_gguf(g)?;
    let mut m = RaonModel::new(g)?;
    m.set_effort(None, think);
    let max_ctx = m.cfg.n_ctx;
    let stdin = std::io::stdin();
    let mut history: Vec<Msg> = system.into_iter().map(|s| Msg::new("system", &s)).collect();
    loop {
        print!("{}", crate::term::prompt());
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 { break; }
        let line = line.trim_end();
        if line.is_empty() { continue; }
        history.push(Msg::new("user", line));
        let text = render(&history, &[], think, None);
        let ids = tok.encode(&text)?;
        if ids.len() >= max_ctx {
            println!("context is full ({max_ctx} tokens); restart the chat");
            break;
        }
        let keep = if m.toks.len() < ids.len() && ids.starts_with(&m.toks) { m.toks.len() } else { m.reset(); 0 };
        let logits = m.prefill(&ids[keep..]);
        let (out, _) = decode(&mut m, &tok, logits, n.min(max_ctx - ids.len()), show);
        println!();
        let reply = tok.decode(&out).unwrap_or_default();
        let stored = if think { format!("{THINK_OPEN}{reply}") } else { reply };
        history.push(Msg::new("assistant", &stored));
    }
    Ok(())
}

pub fn serve<'a>(g: &'a Gguf, tok: &'a Tokenizer, name: &'a str) -> anyhow::Result<crate::serve::Own<'a>> {
    use anyhow::Context;
    use serde_json::json;
    gpu_defaults();
    let mut m = RaonModel::new(g)?;
    let max_ctx = m.cfg.n_ctx;
    Ok(Box::new(move |c: &mut std::net::TcpStream, q: &crate::serve::ChatRequest| {
        let think = q.think.unwrap_or(false);
        m.set_effort(q.reasoning_effort.as_deref(), think);
        let ids = tok.encode(&render(&q.messages, &q.tools, think, q.preserve_thinking))
            .ok().filter(|i| !i.is_empty()).context("cannot encode prompt")?;
        anyhow::ensure!(ids.len() < max_ctx, "prompt is {} tokens, the limit is {max_ctx}", ids.len());
        let t = std::time::Instant::now();
        let keep = if m.toks.len() < ids.len() && ids.starts_with(&m.toks) { m.toks.len() } else { m.reset(); 0 };
        if q.stream {
            let _ = write!(c, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n");
        }
        let logits = crate::serve::alive(c, q.stream, || m.prefill(&ids[keep..]));
        let prefill_s = t.elapsed().as_secs_f64();
        let n = q.max_tokens.unwrap_or(usize::MAX).min(max_ctx - ids.len());
        let chunk = |d: serde_json::Value, fin: Option<&str>| json!({"object": "chat.completion.chunk", "model": name, "choices": [{"index": 0, "delta": d, "finish_reason": fin}]});
        let tools = q.tool_calls && !q.tools.is_empty();
        let mut hold = crate::chat::Hold::new(tools, vec![]).tagged(CALL_TAG);
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
    assert_eq!(render(&user, &[], false, None), "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think></think>\n\n");
    assert_eq!(render(&user, &[], true, None), "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n");
    let reply = "<think>\nplan\n</think>\n\nok\n\n<tool_call>\n<function=get>\n<parameter=city>\nSeoul\n</parameter>\n<parameter=n>\n3\n</parameter>\n</function>\n</tool_call>";
    assert_eq!(parse_calls(reply), Some(("ok".into(), vec![("get".into(), r#"{"city":"Seoul","n":3}"#.into())])));
    // One malformed tool_call block must not discard valid sibling blocks.
    let mixed = "<tool_call>malformed</tool_call>\n<tool_call>\n<function=ping>\n</function>\n</tool_call>";
    assert_eq!(parse_calls(mixed), Some(("".into(), vec![("ping".into(), "{}".into())])));
    // Multi-turn history with think=true preserves reasoning block properly.
    let hist = [Msg::new("user", "hi"), Msg::new("assistant", "<think>\nplan\n</think>\n\nhello"), Msg::new("user", "next")];
    let rendered = render(&hist, &[], true, Some(false));
    assert!(rendered.contains("<|im_start|>assistant\n<think></think>\n\nhello<|im_end|>\n"));
}
