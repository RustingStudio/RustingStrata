//! `rustingstrata chat` for DeepSeek-V4.1: greedy, CUDA for the matvecs and expert staging, KV kept across turns.
use super::forward::DsModel;
use crate::{chat::{Msg, Template}, gguf::{Gguf, Value}, tokenizer::Tokenizer};
use std::io::{BufRead, Write};

pub fn gpu_defaults() {
    // GPU on by default. The expert cache is sized from free VRAM unless STRATA_DS_GPU_EXPERTS says otherwise.
    if std::env::var_os("STRATA_DS_GPU").is_none() { unsafe { std::env::set_var("STRATA_DS_GPU", "1") } }
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
        print!("{}", crate::term::prompt());
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

/// Prompt plus reply limit per request: `--ctx`, else `STRATA_DS_CTX`, else the model's trained context.
/// ponytail: only checked against llama.cpp up to a 1425-token prompt; the candidate-block prune (beyond 4096 compressed rows) is not implemented.
fn ctx_limit(g: &Gguf) -> usize {
    let arch = g.meta.get("general.architecture").and_then(Value::as_str).unwrap_or_default();
    crate::ctx_limit(std::env::var("STRATA_DS_CTX").ok().and_then(|v| v.parse().ok())
        .or_else(|| g.meta.get(&format!("{arch}.context_length")).and_then(Value::as_u64).map(|v| v as usize)).unwrap_or(usize::MAX))
}

/// The chat of `rustingstrata serve` for DeepSeek (see `serve::Own`): one request at a time, JSON or SSE.
/// Earlier reasoning is kept by default, like llama-server (`chat_template_kwargs.preserve_thinking:false` drops it).
/// ponytail: greedy only, no tools/stop strings; the KV cache is reused only when the new prompt extends the last sequence.
pub fn serve<'a>(g: &'a Gguf, tok: &'a Tokenizer, name: &'a str) -> anyhow::Result<crate::serve::Own<'a>> {
    use anyhow::Context;
    use serde_json::json;
    gpu_defaults();
    let mut m = DsModel::new(g)?;
    let eos = tok.eos_ids();
    let ctx = ctx_limit(g);
    Ok(Box::new(move |c: &mut std::net::TcpStream, q: &crate::serve::ChatRequest| {
        let ids = tok.encode(&Template::Ds.render(&q.messages, &q.tools, q.think.unwrap_or(false), q.preserve_thinking.unwrap_or(true), true)).ok().filter(|i| !i.is_empty()).context("cannot encode prompt")?;
        anyhow::ensure!(ids.len() <= ctx, "prompt is {} tokens, the limit is {ctx}", ids.len());
        let t = std::time::Instant::now();
        let keep = if m.toks.len() < ids.len() && ids.starts_with(&m.toks) { m.toks.len() } else { m.reset(); 0 };
        if q.stream { let _ = write!(c, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n"); }
        let mut logits = crate::serve::alive(c, q.stream, || m.prefill(&ids[keep..]));
        let prefill_s = t.elapsed().as_secs_f64();
        let n = q.max_tokens.unwrap_or(usize::MAX).min(ctx - ids.len());
        let chunk = |d: serde_json::Value, fin: Option<&str>| json!({"object": "chat.completion.chunk", "model": name, "choices": [{"index": 0, "delta": d, "finish_reason": fin}]});
        let tools = q.tool_calls && !q.tools.is_empty();
        let mut hold = crate::chat::Hold::new(tools, vec![]).tagged(Template::Ds.call_tag());
        let (mut out, mut shown, mut done) = (vec![], 0, "length");
        let mut alive = std::time::Instant::now();
        for i in 0..n {
            let next = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
            if eos.contains(&next) { done = "stop"; break; }
            out.push(next);
            let s = tok.decode(&out).unwrap_or_default();
            if !s.ends_with('\u{FFFD}') && s.len() > shown && s.is_char_boundary(shown) {
                let t = hold.push(&s[shown..]);
                shown = s.len();
                if q.stream && !(if t.is_empty() { crate::serve::held(c, &mut alive) } else { write!(c, "data: {}\n\n", chunk(json!({"content": t}), None)).is_ok() }) { break; }
            }
            if i + 1 < n { logits = m.step(next); }
        }
        let s = tok.decode(&out).unwrap_or_default();
        if s.len() > shown && s.is_char_boundary(shown) { hold.push(&s[shown..]); }
        let calls = if tools { Template::Ds.parse_calls(&hold.text, &q.tools) } else { None };
        let done = if calls.is_some() { "tool_calls" } else { done };
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
        }
        else {
            let message = match &calls { Some(x) => calls_json(x, false), None => json!({"role": "assistant", "content": hold.text}) };
            let b = json!({"object": "chat.completion", "model": name, "choices": [{"index": 0, "message": message, "finish_reason": done}],
                "usage": {"prompt_tokens": ids.len(), "completion_tokens": out.len(), "total_tokens": ids.len() + out.len()}});
            crate::serve::send(c, "200 OK", "application/json", &b.to_string());
        }
        Ok(crate::serve::Done { prompt: ids.len(), cached: keep, completion: out.len(), prefill_s, finish: done })
    }))
}
