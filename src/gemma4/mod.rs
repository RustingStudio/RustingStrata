//! Gemma 4: sliding-window + full attention, dense FFN beside routed experts (or dense-only when the model has
//! no router), final logit softcapping. CPU reference forward mirroring llama.cpp `gemma4.cpp`; matvecs go to the
//! GPU when STRATA_DS_GPU is set (see `gpu_defaults`).
use crate::gguf::Gguf;
use anyhow::{Context, ensure};

pub mod forward;

pub struct Gemma4Config {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_vocab: usize,
    pub n_ff: usize,
    pub n_expert: usize,
    pub n_used: usize,
    pub n_ff_exp: usize,
    pub eps: f32,
    pub softcap: f32,
    pub window: usize,
    pub n_ctx_train: usize,
    /// per layer: sliding-window layer?
    pub swa: Vec<bool>,
    pub n_kv: Vec<usize>,
    pub hd_swa: usize,
    pub hd_full: usize,
    pub base_swa: f32,
    pub base_full: f32,
}

impl Gemma4Config {
    pub fn from_gguf(g: &Gguf) -> anyhow::Result<Gemma4Config> {
        let arch = g.get("general.architecture")?.as_str().unwrap_or("");
        ensure!(arch == "gemma4", "architecture '{arch}' is not gemma4");
        let k = |s: &str| format!("gemma4.{s}");
        let u = |s: &str| -> anyhow::Result<usize> { Ok(g.get(&k(s))?.as_u64().with_context(|| k(s))? as usize) };
        let f = |s: &str| -> anyhow::Result<f32> { g.get(&k(s))?.as_f32().with_context(|| k(s)) };
        let arr = |s: &str| -> anyhow::Result<Vec<usize>> {
            g.get(&k(s))?.as_arr().with_context(|| k(s))?.iter().map(|v| v.as_u64().map(|x| x as usize).with_context(|| k(s))).collect()
        };
        let n_layer = u("block_count")?;
        let swa: Vec<bool> = arr("attention.sliding_window_pattern")?.into_iter().map(|b| b != 0).collect();
        let n_kv = arr("attention.head_count_kv")?;
        ensure!(swa.len() == n_layer && n_kv.len() == n_layer, "per-layer arrays do not match block_count");
        ensure!(u("attention.shared_kv_layers").unwrap_or(0) == 0, "shared-KV layers (E2B/E4B) are not supported");
        ensure!(u("embedding_length_per_layer_input").unwrap_or(0) == 0, "per-layer embeddings (E2B/E4B) are not supported");
        let emb = g.tensors.get("token_embd.weight").context("missing token_embd.weight")?;
        Ok(Gemma4Config {
            n_layer,
            n_embd: u("embedding_length")?,
            n_head: u("attention.head_count")?,
            n_vocab: emb.dims[1] as usize,
            n_ff: u("feed_forward_length")?,
            n_expert: u("expert_count").unwrap_or(0),
            n_used: u("expert_used_count").unwrap_or(0),
            n_ff_exp: u("expert_feed_forward_length").unwrap_or(0),
            eps: f("attention.layer_norm_rms_epsilon")?,
            softcap: f("final_logit_softcapping").unwrap_or(0.0),
            window: u("attention.sliding_window")?,
            n_ctx_train: u("context_length")?,
            swa,
            n_kv,
            hd_swa: u("attention.key_length_swa")?,
            hd_full: u("attention.key_length")?,
            base_swa: f("rope.freq_base_swa")?,
            base_full: f("rope.freq_base")?,
        })
    }
    pub fn hd(&self, il: usize) -> usize { if self.swa[il] { self.hd_swa } else { self.hd_full } }
}

/// Gemma 4 chat template for plain text messages: `<bos>` is added by the caller. Roles: system, user,
/// assistant (rendered as `model`). Matches the model's `tokenizer.chat_template` for no tools.
pub fn chat_prompt(msgs: &[(&str, &str)], think: bool) -> String {
    let mut s = String::new();
    let first_system = msgs.first().is_some_and(|(r, _)| *r == "system" || *r == "developer");
    if think || first_system {
        s.push_str("<|turn>system\n");
        if think { s.push_str("<|think|>\n"); }
        if first_system { s.push_str(msgs[0].1.trim()); }
        s.push_str("<turn|>\n");
    }
    for (role, text) in if first_system { &msgs[1..] } else { msgs } {
        let r = if *role == "assistant" { "model" } else { role };
        let text = if r == "model" { strip_thinking(text) } else { text.trim().to_string() };
        s.push_str(&format!("<|turn>{r}\n{text}<turn|>\n"));
    }
    s.push_str("<|turn>model\n");
    if !think { s.push_str("<|channel>thought\n<channel|>"); }
    s
}

/// Remove thinking blocks from an assistant message before it is fed back, like the template's `strip_thinking`.
fn strip_thinking(text: &str) -> String {
    let mut out = String::new();
    for part in text.split("<channel|>") {
        if let Some((before, _)) = part.split_once("<|channel>") { out.push_str(before); } else { out.push_str(part); }
    }
    out.trim().to_string()
}

fn gpu_defaults() {
    if std::env::var_os("STRATA_G4_CPU").is_some() { return; }
    for (k, v) in [("STRATA_DS_GPU", "1"), ("STRATA_DS_GPU_EXPERTS", "11"), ("STRATA_DS_NOEVICT", "1")] {
        if std::env::var_os(k).is_none() { unsafe { std::env::set_var(k, v) } }
    }
}

/// `rustingstrata generate` for gemma4: raw prompt (BOS prepended, no template), greedy.
/// STRATA_G4_FORCE=<file of comma-separated ids> teacher-forces those tokens, STRATA_G4_TOP5 prints top-5 log-probs per step.
pub fn generate(g: &Gguf, prompt: &str, n: usize) -> anyhow::Result<()> {
    gpu_defaults();
    let tok = crate::tokenizer::Tokenizer::from_gguf(g)?;
    let seq = prompt_seq(&tok, prompt, None)?;
    let mut m = forward::Gemma4Model::new(g)?;
    let logits = feed(&mut m, &seq)?;
    decode(&tok, &mut m, logits, n)
}

/// Image prompt: each `<image>` becomes `<|image>` (BOI) + one soft token per projector output row +
/// `<image|>` (EOI). The soft tokens are fed as the projector's embeddings, unscaled like llama.cpp's
/// embedding batches. Positions are the token index (Gemma 4 uses normal RoPE, not M-RoPE).
pub fn generate_image(
    g: &Gguf,
    prompt: &str,
    grids: &[(usize, usize)],
    embeddings: &[Vec<Vec<f32>>],
    n: usize,
) -> anyhow::Result<()> {
    gpu_defaults();
    let tok = crate::tokenizer::Tokenizer::from_gguf(g)?;
    let seq = prompt_seq(&tok, prompt, Some((grids, embeddings)))?;
    if std::env::var_os("STRATA_G4_TRACE").is_some() { eprintln!("prompt ids {} tokens", seq.len()); }
    let mut m = forward::Gemma4Model::new(g)?;
    let logits = feed(&mut m, &seq)?;
    decode(&tok, &mut m, logits, n)
}

/// Build the BOS + prompt token sequence, expanding each `<image>` into BOI + one soft token per projector
/// row + EOI. When the prompt has no marker, all images are placed first.
fn prompt_seq<'a>(
    tok: &crate::tokenizer::Tokenizer,
    prompt: &str,
    images: Option<(&[(usize, usize)], &'a [Vec<Vec<f32>>])>,
) -> anyhow::Result<Vec<(u32, Option<&'a [f32]>)>> {
    let (grids, embeddings) = images.unwrap_or((&[], &[]));
    anyhow::ensure!(embeddings.len() == grids.len(), "image count mismatch");
    let begin = tok.token_id("<|image>").context("tokenizer has no <|image>")?;
    let soft = tok.token_id("<|image|>").context("tokenizer has no <|image|>")?;
    let end = tok.token_id("<image|>").context("tokenizer has no <image|>")?;

    let parts: Vec<&str> = prompt.split("<image>").collect();
    let n_mark = parts.len() - 1;
    anyhow::ensure!(n_mark == 0 || n_mark == grids.len(),
        "prompt has {n_mark} <image> placeholder(s) for {} image(s)", grids.len());
    let mut ids = vec![2u32];
    if n_mark == 0 {
        for _ in grids { ids.extend([begin, soft, end]); }
        ids.extend(tok.encode(prompt)?);
    } else {
        for (i, part) in parts.iter().enumerate() {
            if i > 0 { ids.extend([begin, soft, end]); }
            ids.extend(tok.encode(part)?);
        }
    }

    let mut seq = Vec::with_capacity(ids.len());
    let (mut img, mut row) = (0usize, 0usize);
    for &id in &ids {
        if id == soft && img < grids.len() {
            let (nx, ny) = grids[img];
            let ntok = nx * ny;
            anyhow::ensure!(embeddings[img].len() == ntok, "image {img}: {} rows for {ntok} tokens", embeddings[img].len());
            for _ in 0..ntok {
                seq.push((soft, Some(&embeddings[img][row][..])));
                row += 1;
            }
            img += 1;
            row = 0;
        } else {
            seq.push((id, None));
        }
    }
    anyhow::ensure!(img == grids.len(), "used {img} of {} image(s)", grids.len());
    Ok(seq)
}

/// Feed `seq`, returning the logits for its last token. Embedding rows are passed unscaled.
fn feed(m: &mut forward::Gemma4Model, seq: &[(u32, Option<&[f32]>)]) -> anyhow::Result<Vec<f32>> {
    if let Some((_, Some(e))) = seq.iter().find(|(_, e)| e.is_some()) {
        anyhow::ensure!(e.len() == m.cfg.n_embd,
            "projector produces {} features but the text model expects {}", e.len(), m.cfg.n_embd);
    }
    let (head, last) = seq.split_at(seq.len() - 1);
    for &(id, emb) in head {
        match emb {
            Some(e) => { m.step_embed_opt(id, e, false); }
            None => { m.step_opt_pub(id, false); }
        }
    }
    let (id, emb) = last[0];
    Ok(match emb {
        Some(e) => m.step_embed_opt(id, e, true),
        None => m.step_opt_pub(id, true),
    })
}

/// `rustingstrata chat` for gemma4: greedy, KV kept across turns while the history extends it.
/// `images` are encoded once by the caller and placed at the first `<image>` marker (or before the first user
/// turn when the marker is absent).
pub fn chat(
    g: &Gguf,
    system: Option<String>,
    n: usize,
    think: bool,
    images: Option<(&[(usize, usize)], &[Vec<Vec<f32>>])>,
) -> anyhow::Result<()> {
    use std::io::{BufRead, Write};
    gpu_defaults();
    let tok = crate::tokenizer::Tokenizer::from_gguf(g)?;
    let mut m = forward::Gemma4Model::new(g)?;
    let stop = stop_ids(&tok);
    let ctx = m.cfg.n_ctx_train;
    let mut msgs: Vec<(String, String)> = system.into_iter().map(|s| ("system".to_string(), s)).collect();
    let mut first_user = true;
    loop {
        print!("> ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if std::io::stdin().lock().read_line(&mut line)? == 0 { break; }
        if line.trim().is_empty() { continue; }
        let mut text = line.trim_end().to_string();
        if first_user {
            first_user = false;
            if let Some((grids, _)) = images && !text.contains("<image>") {
                text = format!("{}{}", "<image>".repeat(grids.len()), text);
            }
        }
        msgs.push(("user".into(), text));
        let view: Vec<(&str, &str)> = msgs.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
        let seq = prompt_seq(&tok, &chat_prompt(&view, think), images)?;
        let ids: Vec<u32> = seq.iter().map(|(id, _)| *id).collect();
        if ids.len() >= ctx {
            eprintln!("context full ({} tokens)", ids.len());
            break;
        }
        let mut keep = if m.toks.len() < ids.len() && ids.starts_with(&m.toks) { m.toks.len() } else { m.reset(); 0 };
        if keep >= seq.len() { m.reset(); keep = 0; }
        let logits = feed(&mut m, &seq[keep..])?;
        let room = n.min(ctx - ids.len());
        let (out, _) = decode_greedy(&tok, &mut m, &stop, logits, room, &mut |t| {
            print!("{t}");
            let _ = std::io::stdout().flush();
            true
        });
        println!();
        msgs.push(("assistant".into(), tok.decode(&out)?));
    }
    Ok(())
}

/// `rustingstrata serve` for gemma4: one request at a time, OpenAI `/v1/chat/completions` (JSON or SSE),
/// `/v1/models`, `/health`. Greedy only (sampling and stop strings are not applied); prompt + reply limited
/// to the trained context; KV reused when the prompt extends it. Images in messages are encoded with
/// `mmproj` when given.
pub fn serve(
    g: &Gguf,
    name: &str,
    addr: &str,
    api_key: Option<&str>,
    mmproj: Option<&std::path::Path>,
) -> anyhow::Result<()> {
    use serde_json::json;
    use std::io::Write;
    gpu_defaults();
    let tok = crate::tokenizer::Tokenizer::from_gguf(g)?;
    let mut m = forward::Gemma4Model::new(g)?;
    let projector = mmproj.map(crate::vision::VisionProjector::open).transpose()?;
    let stop = stop_ids(&tok);
    let ctx = m.cfg.n_ctx_train;
    let l = std::net::TcpListener::bind(addr)?;
    eprintln!("gemma4 serve on http://{addr}/ model {name}");
    for c in l.incoming() {
        let Ok(mut c) = c else { continue };
        let _ = c.set_read_timeout(Some(std::time::Duration::from_secs(30)));
        let _ = c.set_write_timeout(Some(std::time::Duration::from_secs(30)));
        let Ok(r) = crate::serve::read_request(&mut std::io::BufReader::new(&c)) else { continue };
        let send = |c: &mut std::net::TcpStream, status: &str, ty: &str, b: &str| {
            let _ = write!(c, "HTTP/1.1 {status}\r\nContent-Type: {ty}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len());
        };
        let path = r.path.split('?').next().unwrap_or("");
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
                if !q.tools.is_empty() || q.messages.iter().any(|m| m.role == "tool") {
                    send(&mut c, "400 Bad Request", "application/json", r#"{"error":{"message":"tools are not supported for gemma4","type":"invalid_request_error"}}"#);
                    continue;
                }
                let encoded = (|| -> anyhow::Result<(Vec<(usize, usize)>, Vec<Vec<Vec<f32>>>)> {
                    let mut grids = vec![];
                    let mut embs = vec![];
                    for msg in &q.messages {
                        for src in &msg.images {
                            let p = projector.as_ref().context("this server has no --mmproj")?;
                            let img = crate::vision::load_image(src)?;
                            let (feats, nx, ny) = crate::vision::encode_image(p, &img)?;
                            grids.push((nx, ny));
                            embs.push(feats);
                        }
                    }
                    Ok((grids, embs))
                })();
                let (grids, embs) = match encoded {
                    Ok(v) => v,
                    Err(e) => { send(&mut c, "400 Bad Request", "application/json", &json!({"error": {"message": format!("{e:#}"), "type": "invalid_request_error"}}).to_string()); continue }
                };
                let view: Vec<(&str, &str)> = q.messages.iter().map(|m| (m.role.as_str(), m.content.as_str())).collect();
                let images = if grids.is_empty() { None } else { Some((&grids[..], &embs[..])) };
                let seq = match prompt_seq(&tok, &chat_prompt(&view, q.think), images) {
                    Ok(s) => s,
                    Err(e) => { send(&mut c, "400 Bad Request", "application/json", &json!({"error": {"message": format!("{e:#}"), "type": "invalid_request_error"}}).to_string()); continue }
                };
                let ids: Vec<u32> = seq.iter().map(|(id, _)| *id).collect();
                if ids.len() >= ctx {
                    send(&mut c, "400 Bad Request", "application/json", &json!({"error": {"message": format!("prompt is {} tokens, the limit is {}", ids.len(), ctx - 1), "type": "invalid_request_error"}}).to_string());
                    continue;
                }
                let keep = if m.toks.len() < ids.len() && ids.starts_with(&m.toks) { m.toks.len() } else { m.reset(); 0 };
                let logits = feed(&mut m, &seq[keep..])?;
                let n = q.max_tokens.unwrap_or(256).min(ctx - ids.len());
                if q.stream { let _ = write!(c, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n"); }
                let stream = q.stream;
                let chunk = |d: serde_json::Value, fin: Option<&str>| json!({"object": "chat.completion.chunk", "model": name, "choices": [{"index": 0, "delta": d, "finish_reason": fin}]});
                let (out, done) = decode_greedy(&tok, &mut m, &stop, logits, n, &mut |t| {
                    !stream || t.is_empty() || write!(c, "data: {}\n\n", chunk(json!({"content": t}), None)).is_ok()
                });
                if stream {
                    let _ = write!(c, "data: {}\n\ndata: [DONE]\n\n", chunk(json!({}), Some(done)));
                } else {
                    let b = json!({"object": "chat.completion", "model": name, "choices": [{"index": 0, "message": {"role": "assistant", "content": tok.decode(&out)?}, "finish_reason": done}],
                        "usage": {"prompt_tokens": ids.len(), "completion_tokens": out.len(), "total_tokens": ids.len() + out.len()}});
                    send(&mut c, "200 OK", "application/json", &b.to_string());
                }
            }
            _ => send(&mut c, "404 Not Found", "application/json", r#"{"error":{"message":"not found"}}"#),
        }
    }
    Ok(())
}

/// EOS plus the end-of-turn token that llama.cpp treats as EOG for gemma4.
fn stop_ids(tok: &crate::tokenizer::Tokenizer) -> Vec<u32> {
    let mut e = tok.eos_ids();
    if let Some(t) = tok.token_id("<turn|>") && !e.contains(&t) { e.push(t); }
    e
}

fn decode_greedy(
    tok: &crate::tokenizer::Tokenizer,
    m: &mut forward::Gemma4Model,
    stop: &[u32],
    mut logits: Vec<f32>,
    n: usize,
    emit: &mut dyn FnMut(&str) -> bool,
) -> (Vec<u32>, &'static str) {
    let (mut out, mut shown, mut done) = (vec![], 0, "length");
    for i in 0..n {
        let next = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
        if stop.contains(&next) { done = "stop"; break; }
        out.push(next);
        let s = tok.decode(&out).unwrap_or_default();
        if !s.ends_with('\u{FFFD}') && s.len() > shown && s.is_char_boundary(shown) {
            let ok = emit(&s[shown..]);
            shown = s.len();
            if !ok { done = "stop"; break; }
        }
        if i + 1 < n { logits = m.step(next); }
    }
    (out, done)
}

fn decode(tok: &crate::tokenizer::Tokenizer, m: &mut forward::Gemma4Model, mut logits: Vec<f32>, n: usize) -> anyhow::Result<()> {
    use std::io::Write;
    let eos = tok.eos_ids();
    forward::PROF.iter().for_each(|a| a.store(0, std::sync::atomic::Ordering::Relaxed));
    let t_dec = std::time::Instant::now();
    let (mut out, mut shown) = (vec![], 0);
    let forced: Vec<u32> = std::env::var("STRATA_G4_FORCE").ok().and_then(|p| std::fs::read_to_string(p).ok()).map(|t| t.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_default();
    for i in 0..n {
        let mut next = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
        if std::env::var_os("STRATA_G4_TOP5").is_some() {
            let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let lz = mx as f64 + logits.iter().map(|&l| ((l - mx) as f64).exp()).sum::<f64>().ln();
            let mut idx: Vec<usize> = (0..logits.len()).collect();
            idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            let t: Vec<String> = idx[..5].iter().map(|&k| format!("{k}:{:.4}", logits[k] as f64 - lz)).collect();
            eprintln!("TOP5 {i} {}", t.join(" "));
        }
        if let Some(&f) = forced.get(i) { next = f; }
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
    if std::env::var_os("STRATA_G4_PROF").is_some() {
        let p: Vec<f64> = forward::PROF.iter().map(|a| a.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6).collect();
        eprintln!("PROF decode {:.1} ms/token over {n} tokens", t_dec.elapsed().as_secs_f64() * 1e3 / n as f64);
        eprintln!("PROF ms attn={:.0} ffn(total)={:.0} head={:.0} prefetch={:.0} gpu_exp={:.0} cpu_exp={:.0}", p[0], p[1], p[2], p[3], p[4], p[5]);
    }
    Ok(())
}

#[cfg(test)]
mod chat_tests {
    use super::chat_prompt;

    #[test]
    fn prompt_matches_template() {
        assert_eq!(chat_prompt(&[("user", "Hi")], false), "<|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>");
        assert_eq!(chat_prompt(&[("system", "Be brief."), ("user", "Hi")], false),
            "<|turn>system\nBe brief.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>");
        assert_eq!(chat_prompt(&[("system", "Be brief."), ("user", "Hi")], true),
            "<|turn>system\n<|think|>\nBe brief.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n");
        assert_eq!(chat_prompt(&[("user", "Hi")], true),
            "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n");
        assert_eq!(
            chat_prompt(&[("user", "Hi"), ("assistant", "<|channel>thought\nsecret\n<channel|>Hello")], false),
            "<|turn>user\nHi<turn|>\n<|turn>model\nHello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
        );
    }
}
