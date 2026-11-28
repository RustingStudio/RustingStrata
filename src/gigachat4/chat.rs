//! `rustingstrata generate|chat|serve` for GigaChat 4: greedy, CUDA for the matvecs and expert staging, the cache kept
//! across turns.
//! ponytail: no tool calls (the template's `<｜GCML｜...>` call format and `function result` turns); add them with a
//! `parse_calls` like `kolibri::chat` when a client needs them.
use super::forward::GigaChat4Model;
use crate::{chat::Msg, gguf::Gguf, tokenizer::Tokenizer};
use std::io::{BufRead, Write};

const SEP: &str = "<|role_sep|>\n";
const END: &str = "<|message_sep|>\n\n";
const THINK_END: &str = "</think>";

pub use crate::deepseek::chat::gpu_defaults;

/// The GGUF's chat template with `add_generation_prompt`; `think` is its `reasoning` (off: every think block is empty
/// and the reply starts after an empty one).
pub fn render(msgs: &[Msg], think: bool) -> String {
    let mut s = String::from("<s>");
    let sys = msgs.first().filter(|m| m.role == "system");
    s += &format!("system{SEP}{}{END}", sys.map_or("", |m| m.content.as_str()));
    for m in &msgs[sys.is_some() as usize..] {
        match m.role.as_str() {
            "assistant" => {
                let (mut content, mut reasoning) = (m.content.as_str(), m.reasoning.as_deref().unwrap_or(""));
                if let Some((r, c)) = content.rsplit_once(THINK_END) {
                    reasoning = r.rsplit("<think>").next().unwrap_or("");
                    content = c.trim_start_matches('\n');
                }
                s += &format!("assistant{SEP}<think>{}</think>\n\n{content}{END}", if think { reasoning } else { "" });
            }
            role => s += &format!("{role}{SEP}{}{END}", m.content),
        }
    }
    s += &format!("assistant{SEP}<think>");
    if !think { s += "</think>\n\n"; }
    s
}

/// Greedy decode of up to `n` tokens after `logits`; `piece` gets each new piece of text and returns false to stop.
/// Returns the tokens and whether the model ended the reply itself.
fn decode(m: &mut GigaChat4Model, tok: &Tokenizer, lim: &mut Option<crate::grammar::Constraint>, mut logits: Vec<f32>, n: usize, mut piece: impl FnMut(&str) -> bool) -> (Vec<u32>, bool) {
    let eos = tok.eos_ids();
    let (mut out, mut shown) = (vec![], 0);
    for i in 0..n {
        let next = crate::grammar::greedy(lim, &mut logits);
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

/// `rustingstrata generate`: raw prompt (BOS, no template), greedy, prints the continuation.
pub fn generate(g: &Gguf, prompt: &str, n: usize) -> anyhow::Result<()> {
    gpu_defaults();
    let tok = Tokenizer::from_gguf(g)?;
    let mut m = GigaChat4Model::new(g)?;
    let max_ctx = m.cfg.n_ctx;
    let ids = tok.encode_special(prompt, true)?;
    anyhow::ensure!(!ids.is_empty() && ids.len() < max_ctx, "{}", crate::serve::too_long(ids.len(), max_ctx));
    if std::env::var_os("STRATA_DS_TRACE").is_some() { crate::say!("prompt tokens: {ids:?}"); }
    let logits = m.prefill(&ids);
    let (out, _) = decode(&mut m, &tok, &mut None, logits, n.min(max_ctx - ids.len()), show);
    if std::env::var_os("STRATA_DS_TRACE").is_some() { crate::say!("\noutput tokens: {out:?}"); }
    println!();
    Ok(())
}

pub fn chat(g: &Gguf, system: Option<String>, n: usize, think: bool) -> anyhow::Result<()> {
    gpu_defaults();
    let tok = Tokenizer::from_gguf(g)?;
    let mut m = GigaChat4Model::new(g)?;
    let max_ctx = m.cfg.n_ctx;
    let stdin = std::io::stdin();
    loop {
        print!("{}", crate::term::prompt());
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 { break; }
        let line = line.trim_end();
        if line.is_empty() { continue; }
        // First turn: the full template. Later turns: the cache holds everything up to the reply, so append the end of
        // the turn and the next user turn.
        let user = [Msg::new("user", line)];
        let text = if m.toks.is_empty() {
            let msgs: Vec<Msg> = system.iter().map(|s| Msg::new("system", s)).chain(user).collect();
            render(&msgs, think)
        } else {
            let r = render(&user, think);
            format!("{END}{}", &r[r.find("user").unwrap()..])
        };
        let ids = tok.encode(&text)?;
        if m.toks.len() + ids.len() >= max_ctx {
            println!("context is full ({max_ctx} tokens); restart the chat");
            break;
        }
        let logits = m.prefill(&ids);
        let left = max_ctx - m.toks.len();
        decode(&mut m, &tok, &mut None, logits, n.min(left), show);
        println!();
    }
    Ok(())
}

/// The chat of `rustingstrata serve` for GigaChat 4 (see `serve::Own`): one request at a time, JSON or SSE.
/// ponytail: greedy only, no stop strings, no tool calls; thinking arrives inside `content` up to `</think>`; the
/// cache is reused only when the new prompt extends the last sequence.
pub fn serve<'a>(g: &'a Gguf, tok: &'a Tokenizer, name: &'a str) -> anyhow::Result<crate::serve::Own<'a>> {
    use anyhow::Context;
    use serde_json::json;
    gpu_defaults();
    let mut m = GigaChat4Model::new(g)?;
    let max_ctx = m.cfg.n_ctx;
    Ok(Box::new(move |c: &mut std::net::TcpStream, q: &crate::serve::ChatRequest| {
        let mut lim = crate::grammar::Constraint::new(q, tok)?;
        let think = q.think.unwrap_or(true) && q.reasoning_effort.as_deref() != Some("none");
        let text = render(&q.messages, think);
        let ids = tok.encode(&text).ok().filter(|i| !i.is_empty()).context("cannot encode prompt")?;
        anyhow::ensure!(ids.len() < max_ctx, "{}", crate::serve::too_long(ids.len(), max_ctx));
        let t = std::time::Instant::now();
        let keep = if m.toks.len() < ids.len() && ids.starts_with(&m.toks) { m.toks.len() } else { m.reset(); 0 };
        if q.stream && !q.head_sent { let _ = write!(c, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n"); }
        let logits = crate::serve::alive(c, q.stream, || m.prefill(&ids[keep..]));
        let prefill_s = t.elapsed().as_secs_f64();
        let n = q.max_tokens.unwrap_or(usize::MAX).min(max_ctx - ids.len());
        let chunk = |d: serde_json::Value, fin: Option<&str>| json!({"object": "chat.completion.chunk", "model": name, "choices": [{"index": 0, "delta": d, "finish_reason": fin}]});
        let mut text = String::new();
        let (out, stopped) = decode(&mut m, tok, &mut lim, logits, n, |p| {
            text += p;
            !q.stream || write!(c, "data: {}\n\n", chunk(json!({"content": p}), None)).is_ok()
        });
        let done = if stopped { "stop" } else { "length" };
        let d = crate::serve::Done { prompt: ids.len(), cached: keep, completion: out.len(), prefill_s, finish: done };
        let timings = d.timings(t);
        let usage = json!({"prompt_tokens": ids.len(), "completion_tokens": out.len(), "total_tokens": ids.len() + out.len(), "prompt_tokens_details": {"cached_tokens": keep}});
        if q.stream {
            let mut last = chunk(json!({}), Some(done));
            last["usage"] = usage.clone();
            last["timings"] = timings;
            let _ = write!(c, "data: {last}\n\n");
            if q.include_usage { let _ = write!(c, "data: {}\n\n", json!({"object": "chat.completion.chunk", "model": name, "choices": [], "usage": usage})); }
            let _ = write!(c, "data: [DONE]\n\n");
        } else {
            let b = json!({"object": "chat.completion", "model": name, "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": done}], "usage": usage, "timings": timings});
            crate::serve::send(c, "200 OK", "application/json", &b.to_string());
        }
        Ok(d)
    }))
}

#[test]
fn template() {
    // the strings from the GGUF's jinja template (rendered with jinja2)
    let user = [Msg::new("user", "hi")];
    assert_eq!(render(&user, true), "<s>system<|role_sep|>\n<|message_sep|>\n\nuser<|role_sep|>\nhi<|message_sep|>\n\nassistant<|role_sep|>\n<think>");
    assert_eq!(render(&user, false), "<s>system<|role_sep|>\n<|message_sep|>\n\nuser<|role_sep|>\nhi<|message_sep|>\n\nassistant<|role_sep|>\n<think></think>\n\n");
    let mut a = Msg::new("assistant", "ok");
    a.reasoning = Some("plan".into());
    let msgs = [Msg::new("user", "q"), a, Msg::new("user", "q2")];
    assert_eq!(render(&msgs, true), "<s>system<|role_sep|>\n<|message_sep|>\n\nuser<|role_sep|>\nq<|message_sep|>\n\nassistant<|role_sep|>\n<think>plan</think>\n\nok<|message_sep|>\n\nuser<|role_sep|>\nq2<|message_sep|>\n\nassistant<|role_sep|>\n<think>");
    let msgs = [Msg::new("system", "S"), Msg::new("user", "q"), Msg::new("assistant", "<think>plan</think>\n\nok"), Msg::new("user", "q2")];
    assert_eq!(render(&msgs, false), "<s>system<|role_sep|>\nS<|message_sep|>\n\nuser<|role_sep|>\nq<|message_sep|>\n\nassistant<|role_sep|>\n<think></think>\n\nok<|message_sep|>\n\nuser<|role_sep|>\nq2<|message_sep|>\n\nassistant<|role_sep|>\n<think></think>\n\n");
}
