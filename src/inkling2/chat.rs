//! `rustingstrata generate|chat|serve` for Inkling 2: greedy, GPU for the matvecs and the expert cache, state kept across turns.
//! A reply is a list of blocks, each `<|message_model|>`, a content token, the body and `<|end_message|>`; the model
//! writes the block headers itself, so the prompt has no generation prefix.
use super::forward::{Checkpoint, Inkling2Model};
use crate::{chat::{Hold, Json, Msg}, gguf::Gguf, ops::Sampler, tokenizer::Tokenizer};
use anyhow::{Context, bail, ensure};
use std::io::{BufRead, Write};

/// Runtime options for Inkling 2 `generate`, `chat`, and `serve`.
#[derive(Clone, Debug)]
pub struct Inkling2Opts {
    pub max_ctx: usize,
    pub sampler: Sampler,
    pub thinking_budget: usize,
    pub think_default: bool,
    pub mmproj: Option<std::path::PathBuf>,
    pub draft_model: Option<std::path::PathBuf>,
    pub ngram_cfg: crate::ngram::NgramConfig,
}

impl Default for Inkling2Opts {
    fn default() -> Self {
        Self {
            max_ctx: 0,
            sampler: Sampler::greedy(),
            thinking_budget: 0,
            think_default: true,
            mmproj: None,
            draft_model: None,
            ngram_cfg: crate::ngram::NgramConfig::default(),
        }
    }
}

impl Inkling2Opts {
    /// Load `Inkling2Model` with optional `DsparkDrafter` (loaded before `Device::new` so its weights are counted in VRAM budget)
    /// and `NgramDrafter`.
    pub fn build_model<'a>(&self, g: &'a Gguf, tok: &Tokenizer) -> anyhow::Result<Inkling2Model<'a>> {
        let dspark = match &self.draft_model {
            Some(p) => Some(super::dspark::DsparkDrafter::open(p)?),
            None => None,
        };
        let mut m = Inkling2Model::new_with_ctx(g, self.max_ctx)?;
        m.dspark = dspark;
        m.spec_draft_max = if self.ngram_cfg.is_enabled() {
            self.ngram_cfg.draft_max
        } else if self.ngram_cfg.draft_max != 16 {
            self.ngram_cfg.draft_max
        } else {
            3
        };
        if self.ngram_cfg.is_enabled() {
            let mut cfg = self.ngram_cfg.clone();
            cfg.n_vocab = m.cfg.n_vocab;
            m.ngram = Some(crate::ngram::NgramDrafter::new(cfg, Some(tok))?);
        }
        Ok(m)
    }
}

const USER: &str = "<|message_user|>";
const MODEL: &str = "<|message_model|>";
const SYSTEM: &str = "<|message_system|>";
const TOOL: &str = "<|message_tool|>";
const TEXT: &str = "<|content_text|>";
const IMAGE: &str = "<|content_image|>";
const IMAGE_PAD: &str = "<|unused_200054|>";
const THINKING: &str = "<|content_thinking|>";
const INVOKE: &str = "<|content_invoke_tool_json|>";
const END: &str = "<|end_message|>";
/// Ends an assistant turn. The model stops on it, so a later prompt adds it after the kept reply.
const DONE: &str = "<|content_model_end_sampling|>";

pub fn gpu_defaults() {
    crate::deepseek::chat::gpu_defaults();
    if std::env::var_os("STRATA_DS_GPU_FRACTION").is_none() { unsafe { std::env::set_var("STRATA_DS_GPU_FRACTION", "0.6,0.9") } }
    // Local patch: with several GPUs the plain weights stay on GPU 0, where the device decode state lives.
    if std::env::var_os("STRATA_DS_PLAIN_FIRST").is_none() { unsafe { std::env::set_var("STRATA_DS_PLAIN_FIRST", "1") } }
}

/// The template's `tojson(sort_keys=true, separators=(",", ":"))`.
// ponytail: also sorts the keys inside a tool call's argument values (the template sorts only the top level).
fn compact(j: &Json) -> String {
    serde_json::from_str::<serde_json::Value>(&j.dump()).map_or_else(|_| j.dump(), |v| v.to_string())
}

/// Map an effort level string or float (`none|minimal|low|medium|high|xhigh|max` or `0.0..=0.99`)
/// and optional `think` boolean to the GGUF template's effort string (`"0"`, `"0.1"`, `"0.2"`, `"0.7"`, `"0.9"`, `"0.99"`).
pub fn parse_effort(effort: Option<&str>, think: Option<bool>, default_think: bool) -> anyhow::Result<String> {
    if think == Some(false) && effort.is_none() {
        return Ok("0".into());
    }
    let Some(raw) = effort.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(if think.unwrap_or(default_think) { "0.9".into() } else { "0".into() });
    };
    let num: f32 = match raw {
        "none" => 0.0,
        "minimal" => 0.1,
        "low" => 0.2,
        "medium" => 0.7,
        "high" => 0.9,
        "xhigh" | "max" => 0.99,
        other => other.parse::<f32>().map_err(|_| anyhow::anyhow!("Unknown reasoning_effort: {other}"))?,
    };
    if !(0.0..=0.99).contains(&num) {
        bail!("reasoning_effort must be in [0.0, 0.99]");
    }
    Ok(if num == 0.0 { "0".into() } else { format!("{num}") })
}

/// The GGUF's chat template. `think` picks the effort line: 0.9 (the template's default, "high") or 0 ("none").
pub fn render(msgs: &[Msg], tools: &[Json], think: bool) -> String {
    render_with_effort(msgs, tools, if think { "0.9" } else { "0" })
}

/// The GGUF's chat template with an explicit validated `Thinking effort level: <effort>` string.
pub fn render_with_effort(msgs: &[Msg], tools: &[Json], effort_level: &str) -> String {
    let mut s = String::new();
    if !tools.is_empty() {
        let specs: Vec<String> = tools.iter().map(|t| {
            let f = t.get("function");
            format!(r#"{{"description":{},"name":{},"parameters":{},"type":"function"}}"#, compact(f.get("description")), compact(f.get("name")), compact(f.get("parameters")))
        }).collect();
        s += &format!("{SYSTEM}tool_declare<|content_xml|>[{}]{END}", specs.join(","));
    }
    // The effort line goes before the first message that is not a system message.
    let effort = format!("{SYSTEM}{TEXT}Thinking effort level: {effort_level}{END}");
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
            "tool" => {
                let prefix = format!("{TOOL}{}", called.next().unwrap_or(""));
                render_user_or_tool(&mut s, &prefix, &m.content, m.images.len());
            }
            "system" => s += &format!("{SYSTEM}{TEXT}{}{END}", m.content),
            _ => render_user_or_tool(&mut s, USER, &m.content, m.images.len()),
        }
    }
    if !effort_done { s += &effort; }
    s
}

/// Render a user or tool message, splitting any `<image>` markers or `m.images` into
/// `<|message_user|><|content_image|><|unused_200054|><|end_message|>` blocks per the GGUF Jinja template.
fn render_user_or_tool(s: &mut String, rtok: &str, content: &str, n_images: usize) {
    if content.contains("<image>") {
        let parts: Vec<&str> = content.split("<image>").collect();
        for (i, part) in parts.iter().enumerate() {
            if i > 0 { *s += &format!("{rtok}{IMAGE}{IMAGE_PAD}{END}"); }
            if !part.is_empty() || (parts.len() == 1 && n_images == 0) {
                *s += &format!("{rtok}{TEXT}{part}{END}");
            }
        }
    } else {
        for _ in 0..n_images {
            *s += &format!("{rtok}{IMAGE}{IMAGE_PAD}{END}");
        }
        if !content.is_empty() || n_images == 0 {
            *s += &format!("{rtok}{TEXT}{content}{END}");
        }
    }
}

/// Expand `<|unused_200054|>` (or `<image>` in raw prompts) into one `(soft_id, Some(&feats[row]))` per
/// projected vision token row, keeping `(id, None)` for text tokens.
pub fn prompt_seq<'a>(
    tok: &Tokenizer,
    prompt: &str,
    images: Option<(&[(usize, usize)], &'a [Vec<Vec<f32>>])>,
) -> anyhow::Result<Vec<(u32, Option<&'a [f32]>)>> {
    let (grids, embeddings) = images.unwrap_or((&[], &[]));
    ensure!(embeddings.len() == grids.len(), "image count mismatch");
    if grids.is_empty() {
        let ids = tok.encode(prompt)?;
        return Ok(ids.into_iter().map(|id| (id, None)).collect());
    }
    let soft = tok.token_id(IMAGE_PAD).context("tokenizer has no <|unused_200054|>")?;
    let expanded = if prompt.contains(IMAGE_PAD) {
        prompt.to_string()
    } else if prompt.contains("<image>") {
        let rep = format!("{USER}{IMAGE}{IMAGE_PAD}{END}");
        prompt.replace("<image>", &rep)
    } else {
        let prefix = format!("{USER}{IMAGE}{IMAGE_PAD}{END}").repeat(grids.len());
        format!("{prefix}{prompt}")
    };
    let ids = tok.encode(&expanded)?;
    let n_pad = ids.iter().filter(|&&id| id == soft).count();
    ensure!(n_pad == grids.len(), "prompt has {n_pad} image placeholder(s) for {} image(s)", grids.len());
    let mut seq = Vec::with_capacity(ids.len() + grids.iter().map(|(w, h)| w * h).sum::<usize>());
    let mut img = 0usize;
    for id in ids {
        if id == soft && img < grids.len() {
            let (nx, ny) = grids[img];
            let ntok = nx * ny;
            ensure!(embeddings[img].len() == ntok, "image {img}: {} rows for {ntok} tokens", embeddings[img].len());
            for row in 0..ntok {
                seq.push((soft, Some(&embeddings[img][row][..])));
            }
            img += 1;
        } else {
            seq.push((id, None));
        }
    }
    ensure!(img == grids.len(), "used {img} of {} image(s)", grids.len());
    Ok(seq)
}

/// What a piece of the reply belongs to.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Part { Text, Think, Call }

/// Name and JSON arguments of a call block body, `{"name":..,"args":{..}}`.
pub fn parse_call(body: &str) -> Option<(String, String)> {
    let v: serde_json::Value = serde_json::from_str(body.trim()).ok()?;
    Some((v["name"].as_str()?.to_string(), v.get("args").map_or("{}".into(), |a| a.to_string())))
}

/// Decode up to `n` tokens with `Sampler`, optional `thinking_budget` (0 = unlimited), and `stops` (`Hold`).
/// Returns `(tokens_generated, stopped_by_eos_or_stop_string)`.
pub fn decode_with(
    m: &mut Inkling2Model,
    tok: &Tokenizer,
    logits: Vec<f32>,
    n: usize,
    smp: &mut Sampler,
    thinking_budget: usize,
    stops: &[String],
    mut piece: impl FnMut(Part, bool, &str) -> bool,
) -> (usize, bool) {
    if n == 0 || logits.is_empty() { return (0, false); }
    let eos = tok.eos_ids();
    let id = |t: &str| tok.token_id(t);
    let (model, end) = (id(MODEL), id(END));
    let kinds = [(id(TEXT), Part::Text), (id(THINKING), Part::Think), (id(INVOKE), Part::Call)];
    let pure_greedy = smp.temp <= 0.0 && smp.repeat == 1.0 && smp.presence == 0.0 && smp.frequency == 0.0 && smp.dry_mult <= 0.0 && smp.limit.is_none();
    let pick_tok = |smp: &mut Sampler, l: &[f32]| -> u32 {
        if pure_greedy {
            crate::ops::argmax(l) as u32
        } else {
            smp.pick(l).unwrap_or_else(|_| crate::ops::argmax(l)) as u32
        }
    };
    let close_think_ids = (thinking_budget > 0)
        .then(|| tok.encode(&format!("{END}{MODEL}{TEXT}")).unwrap_or_default())
        .filter(|v| !v.is_empty());
    let mut hold = (!stops.is_empty()).then(|| Hold::new(false, stops.to_vec()));
    // Text outside a block is content. `None`: inside a block header (the tool name of a call).
    let (mut part, mut start, mut buf, mut shown) = (Some(Part::Text), true, vec![], 0);
    let mut think_tokens = 0usize;
    let mut next = pick_tok(smp, &logits);
    let mut pending = std::collections::VecDeque::<u32>::new();
    let mut i = 0usize;
    while i < n {
        if eos.contains(&next) {
            if let Some(h) = &mut hold && !h.stopped && !h.rest().is_empty() {
                let _ = piece(Part::Text, start, h.rest());
            }
            return (i, true);
        }
        let mark = if Some(next) == model {
            Some(None)
        } else if Some(next) == end {
            Some(Some(Part::Text))
        } else {
            kinds.iter().find(|k| k.0 == Some(next)).map(|k| Some(k.1))
        };
        if let Some(p) = mark {
            if part == Some(Part::Text) && let Some(h) = &mut hold {
                if !h.stopped && !h.rest().is_empty() {
                    let r = h.rest().to_string();
                    if !piece(Part::Text, start, &r) { return (i + 1, false); }
                }
                *h = Hold::new(false, stops.to_vec());
            }
            (part, start, shown) = (p, true, 0);
            buf.clear();
        } else if let Some(p) = part {
            if p == Part::Think { think_tokens += 1; }
            buf.push(next);
            let s = tok.decode(&buf).unwrap_or_default();
            if !s.ends_with('\u{FFFD}') && s.len() > shown && s.is_char_boundary(shown) {
                let delta = &s[shown..];
                shown = s.len();
                if p == Part::Text && let Some(h) = &mut hold {
                    let safe = h.push(delta);
                    if !safe.is_empty() {
                        if !piece(p, start, safe) { return (i + 1, false); }
                        start = false;
                    }
                    if h.stopped { return (i + 1, true); }
                } else {
                    if !piece(p, start, delta) { return (i + 1, false); }
                    start = false;
                }
            }
        }
        i += 1;
        crate::term::decoded(1);
        if part == Some(Part::Think)
            && thinking_budget > 0
            && think_tokens >= thinking_budget
            && i < n
            && let Some(close_ids) = &close_think_ids
            && m.toks.len() + 1 + close_ids.len() < m.max_ctx
        {
            pending.clear();
            let mut seq = Vec::with_capacity(1 + close_ids.len());
            seq.push(next);
            seq.extend_from_slice(close_ids);
            let l = m.prefill(&seq);
            (part, start, shown) = (Some(Part::Text), true, 0);
            buf.clear();
            next = pick_tok(smp, &l);
            continue;
        }
        if i < n {
            if let Some(t) = pending.pop_front() {
                next = t;
            } else if m.has_spec() && stops.is_empty() && smp.limit.is_none() {
                let budget_cap = if part == Some(Part::Think) && thinking_budget > 0 {
                    thinking_budget.saturating_sub(think_tokens)
                } else {
                    usize::MAX
                };
                let rem = (n - i).min(m.max_ctx.saturating_sub(m.toks.len() + 1)).min(budget_cap);
                let batch = m.spec_step(next, rem, &eos, smp);
                pending.extend(batch);
                next = pending.pop_front().unwrap();
            } else {
                next = if pure_greedy {
                    m.step_argmax(next)
                } else {
                    let l = m.step(next);
                    pick_tok(smp, &l)
                };
            }
        }
    }
    if let Some(h) = &mut hold && !h.stopped && !h.rest().is_empty() {
        let _ = piece(Part::Text, start, h.rest());
    }
    (n, false)
}

/// `rustingstrata generate`: raw prompt (no template), prints the continuation with its block tokens.
pub fn generate(g: &Gguf, prompt: &str, n: usize) -> anyhow::Result<()> {
    generate_opts(g, prompt, n, Inkling2Opts::default())
}

/// `generate` with an explicit context cap up to `cfg.ctx`.
pub fn generate_ctx(g: &Gguf, prompt: &str, n: usize, max_ctx: usize) -> anyhow::Result<()> {
    generate_opts(g, prompt, n, Inkling2Opts { max_ctx, ..Default::default() })
}

/// `generate` with full `Inkling2Opts` (`max_ctx`, `sampler`).
pub fn generate_opts(g: &Gguf, prompt: &str, n: usize, opts: Inkling2Opts) -> anyhow::Result<()> {
    generate_image_opts(g, prompt, None, n, opts)
}

/// `generate` with optional encoded image grids + embeddings.
pub fn generate_image_opts(
    g: &Gguf,
    prompt: &str,
    images: Option<(&[(usize, usize)], &[Vec<Vec<f32>>])>,
    n: usize,
    mut opts: Inkling2Opts,
) -> anyhow::Result<()> {
    gpu_defaults();
    let tok = Tokenizer::from_gguf(g)?;
    let mut m = opts.build_model(g, &tok)?;
    let seq = prompt_seq(&tok, prompt, images)?;
    ensure!(!seq.is_empty() && seq.len() < m.max_ctx, "prompt is {} tokens, the limit is {}", seq.len(), m.max_ctx);
    let trace = std::env::var_os("STRATA_DS_TRACE").is_some();
    if trace {
        let ids: Vec<u32> = seq.iter().map(|&(t, _)| t).collect();
        crate::say!("prompt tokens: {ids:?}");
    }
    let logits = m.prefill_embds(&seq);
    let (eos, mut out, mut shown) = (tok.eos_ids(), vec![], 0);
    let smp = &mut opts.sampler;
    let pure_greedy = smp.temp <= 0.0 && smp.repeat == 1.0 && smp.presence == 0.0 && smp.frequency == 0.0 && smp.dry_mult <= 0.0;
    let mut next = if pure_greedy { crate::ops::argmax(&logits) as u32 } else { smp.pick(&logits)? as u32 };
    let mut pending = std::collections::VecDeque::<u32>::new();
    let max_gen = n.min(m.max_ctx - seq.len());
    for i in 0..max_gen {
        if eos.contains(&next) { break; }
        out.push(next);
        crate::term::decoded(1);
        let s = tok.decode(&out).unwrap_or_default();
        if !s.ends_with('\u{FFFD}') && s.len() > shown && s.is_char_boundary(shown) {
            print!("{}", &s[shown..]);
            std::io::stdout().flush()?;
            shown = s.len();
        }
        if i + 1 < max_gen {
            if let Some(t) = pending.pop_front() {
                next = t;
            } else if m.has_spec() {
                let rem = (max_gen - (i + 1)).min(m.max_ctx.saturating_sub(m.toks.len() + 1));
                let batch = m.spec_step(next, rem, &eos, smp);
                pending.extend(batch);
                next = pending.pop_front().unwrap();
            } else {
                next = if pure_greedy {
                    m.step_argmax(next)
                } else {
                    let l = m.step(next);
                    smp.pick(&l)? as u32
                };
            }
        }
    }
    if m.drafts.0 > 0 {
        crate::say!("\nspeculative drafts accepted {}/{} ({:.1}%)", m.drafts.1, m.drafts.0, 100.0 * m.drafts.1 as f64 / m.drafts.0 as f64);
    }
    if trace { crate::say!("\noutput tokens: {out:?}"); }
    println!();
    Ok(())
}

pub fn chat(g: &Gguf, system: Option<String>, n: usize, think: bool) -> anyhow::Result<()> {
    chat_opts(g, system, n, Inkling2Opts { think_default: think, ..Default::default() })
}

/// Interactive chat with an explicit context cap up to `cfg.ctx`.
pub fn chat_ctx(g: &Gguf, system: Option<String>, n: usize, think: bool, max_ctx: usize) -> anyhow::Result<()> {
    chat_opts(g, system, n, Inkling2Opts { max_ctx, think_default: think, ..Default::default() })
}

/// Interactive chat with full `Inkling2Opts` (`max_ctx`, `sampler`, `thinking_budget`, `think_default`).
pub fn chat_opts(g: &Gguf, system: Option<String>, n: usize, opts: Inkling2Opts) -> anyhow::Result<()> {
    chat_image_opts(g, system, n, None, opts)
}

/// Interactive chat with optional first-turn image grids + embeddings.
pub fn chat_image_opts(
    g: &Gguf,
    system: Option<String>,
    n: usize,
    images: Option<(&[(usize, usize)], &[Vec<Vec<f32>>])>,
    opts: Inkling2Opts,
) -> anyhow::Result<()> {
    gpu_defaults();
    let tok = Tokenizer::from_gguf(g)?;
    let mut m = opts.build_model(g, &tok)?;
    let stdin = std::io::stdin();
    let mut ended = true;
    loop {
        print!("{}", crate::term::prompt());
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 { break; }
        let line = line.trim_end();
        if line.is_empty() { continue; }
        let first_turn = m.toks.is_empty();
        let text = if first_turn {
            let mut u = Msg::new("user", line);
            if let Some((grids, _)) = images && !line.contains("<image>") {
                u.images = vec![String::new(); grids.len()];
            }
            let msgs: Vec<Msg> = system.iter().map(|s| Msg::new("system", s)).chain([u]).collect();
            render(&msgs, &[], opts.think_default)
        } else {
            format!("{}{DONE}{USER}{TEXT}{line}{END}", if ended { "" } else { END })
        };
        let seq = prompt_seq(&tok, &text, if first_turn { images } else { None })?;
        if m.toks.len() + seq.len() >= m.max_ctx {
            println!("context is full ({} tokens); restart the chat", m.max_ctx);
            break;
        }
        let logits = m.prefill_embds(&seq);
        let left = m.max_ctx - m.toks.len();
        let mut last = Part::Text;
        let mut smp = opts.sampler.clone();
        (_, ended) = decode_with(&mut m, &tok, logits, n.min(left), &mut smp, opts.thinking_budget, &[], |p, start, s| {
            if start && p != last {
                let on = crate::term::out();
                print!("{}", match p { Part::Think if on => "\x1b[2m", Part::Think => "", _ if last == Part::Think && on => "\x1b[0m\n", _ => "\n" });
                last = p;
            }
            print!("{s}");
            std::io::stdout().flush().is_ok()
        });
        println!("{}", if last == Part::Think && crate::term::out() { "\x1b[0m" } else { "" });
    }
    Ok(())
}

/// The chat of `rustingstrata serve` for Inkling 2 (see `serve::Own`): one request at a time, JSON or SSE, with the
/// sampler, `reasoning_effort`, `thinking_budget`, `stop` strings, images (with a projector) and the drafts of `opts`.
/// Thinking is on unless the request turns it off (`chat_template_kwargs.enable_thinking: false`); it arrives as `reasoning_content`.
/// The state is reused when the new prompt extends the last sequence or the prompt of the last turn (a checkpoint),
/// so a client need not send `reasoning_content` back.
pub fn serve<'a>(g: &'a Gguf, tok: &'a Tokenizer, name: &'a str, opts: Inkling2Opts) -> anyhow::Result<crate::serve::Own<'a>> {
    use serde_json::json;
    gpu_defaults();
    let mut m = opts.build_model(g, tok)?;
    let projector = opts.mmproj.as_deref().map(crate::vision::VisionProjector::open).transpose()?;
    if let Some(p) = &projector { crate::vision::validate_projection(p.cfg(), m.cfg.n_embd)?; }
    let mut turn_ckpt: Option<(Checkpoint, Vec<f32>)> = None;
    Ok(Box::new(move |c: &mut std::net::TcpStream, q: &crate::serve::ChatRequest| {
        let effort = parse_effort(q.reasoning_effort.as_deref(), q.think, opts.think_default)?;
        let (mut grids, mut embs) = (vec![], vec![]);
        for src in q.messages.iter().flat_map(|msg| &msg.images) {
            let p = projector.as_ref().context("this server has no --mmproj")?;
            let (feats, nx, ny) = crate::vision::encode_image(p, &crate::vision::load_image(src)?)?;
            grids.push((nx, ny));
            embs.push(feats);
        }
        let has_images = !grids.is_empty();
        let seq = prompt_seq(tok, &render_with_effort(&q.messages, &q.tools, &effort), has_images.then_some((&grids[..], &embs[..])))?;
        let ids: Vec<u32> = seq.iter().map(|&(t, _)| t).collect();
        ensure!(!ids.is_empty(), "cannot encode prompt");
        ensure!(ids.len() < m.max_ctx, "prompt is {} tokens, the limit is {}", ids.len(), m.max_ctx);
        let t = std::time::Instant::now();
        if q.stream && !q.head_sent { let _ = write!(c, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n"); }
        let (keep, logits) = crate::serve::alive(c, q.stream, || if !has_images && m.toks.len() < ids.len() && ids.starts_with(&m.toks) {
            let k = m.toks.len();
            (k, m.prefill_embds(&seq[k..]))
        } else if !has_images && let Some((ck, ck_logits)) = &turn_ckpt && ck.toks.len() <= ids.len() && ids.starts_with(&ck.toks) {
            let k = ck.toks.len();
            m.restore(ck);
            (k, if k == ids.len() { ck_logits.clone() } else { m.prefill_embds(&seq[k..]) })
        } else {
            m.reset();
            (0, m.prefill_embds(&seq))
        });
        turn_ckpt = (!has_images).then(|| (m.checkpoint(), logits.clone()));
        let prefill_s = t.elapsed().as_secs_f64();

        let mut smp = opts.sampler.clone();
        if let Some(t) = q.temperature { smp.temp = t; }
        if let Some(k) = q.top_k { smp.top_k = k; }
        if let Some(p) = q.top_p { smp.top_p = p; }
        if let Some(p) = q.min_p { smp.min_p = p; }
        if let Some(p) = q.presence_penalty { smp.presence = p; }
        if let Some(f) = q.frequency_penalty { smp.frequency = f; }
        if let Some(rp) = q.repeat_penalty { smp.repeat = rp; }
        if let Some(ln) = q.penalty_last_n { smp.last_n = ln; }
        if let Some(dm) = q.dry_multiplier { smp.dry_mult = dm; }
        if let Some(db) = q.dry_base { smp.dry_base = db; }
        if let Some(da) = q.dry_allowed_length { smp.dry_allowed = da; }
        if let Some(s) = q.seed { smp.rng = s.max(1); }
        smp.limit = crate::grammar::Constraint::new(q, tok)?;

        let budget = q.thinking_budget.unwrap_or(opts.thinking_budget);
        let n = q.max_tokens.unwrap_or(usize::MAX).min(m.max_ctx - m.toks.len());
        let chunk = |d: serde_json::Value, fin: Option<&str>| json!({"object": "chat.completion.chunk", "model": name, "choices": [{"index": 0, "delta": d, "finish_reason": fin}]});
        let tools = q.tool_calls && !q.tools.is_empty();
        let (mut content, mut reasoning, mut bodies) = (String::new(), String::new(), Vec::<String>::new());
        let mut alive = std::time::Instant::now();
        let (count, stopped) = decode_with(&mut m, tok, logits, n, &mut smp, budget, &q.stop, |p, start, s| {
            let key = match p {
                Part::Call if tools => {
                    if start { bodies.push(String::new()); }
                    *bodies.last_mut().unwrap() += s;
                    return !q.stream || crate::serve::held(c, &mut alive);
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
        let usage = json!({"prompt_tokens": ids.len(), "completion_tokens": count, "total_tokens": ids.len() + count, "prompt_tokens_details": {"cached_tokens": keep}});
        if q.stream {
            if !calls.is_empty() { let _ = write!(c, "data: {}\n\n", chunk(json!({"tool_calls": calls}), None)); }
            let _ = write!(c, "data: {}\n\n", chunk(json!({}), Some(done)));
            if q.include_usage { let _ = write!(c, "data: {}\n\n", json!({"object": "chat.completion.chunk", "model": name, "choices": [], "usage": usage})); }
            let _ = write!(c, "data: [DONE]\n\n");
        } else {
            let mut message = json!({"role": "assistant", "content": content});
            if !reasoning.is_empty() { message["reasoning_content"] = reasoning.into(); }
            if !calls.is_empty() { message["tool_calls"] = calls.into(); }
            let b = json!({"object": "chat.completion", "model": name, "choices": [{"index": 0, "message": message, "finish_reason": done}], "usage": usage});
            crate::serve::send(c, "200 OK", "application/json", &b.to_string());
        }
        Ok(crate::serve::Done { prompt: ids.len(), cached: keep, completion: count, prefill_s, finish: done })
    }))
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
    for (inp, want) in [
        (Some("none"), "0"),
        (Some("minimal"), "0.1"),
        (Some("low"), "0.2"),
        (Some("medium"), "0.7"),
        (Some("high"), "0.9"),
        (Some("xhigh"), "0.99"),
        (Some("max"), "0.99"),
        (Some("0.45"), "0.45"),
        (Some("0"), "0"),
        (None, "0.9"),
    ] {
        assert_eq!(parse_effort(inp, None, true).unwrap(), want);
    }
    assert_eq!(parse_effort(None, Some(false), true).unwrap(), "0");
    assert!(parse_effort(Some("1.0"), None, true).is_err());
    assert!(parse_effort(Some("-0.1"), None, true).is_err());
    assert!(parse_effort(Some("turbo"), None, true).is_err());
    assert_eq!(
        render_with_effort(&[Msg::new("user", "hi")], &[], "0.7"),
        "<|message_system|><|content_text|>Thinking effort level: 0.7<|end_message|><|message_user|><|content_text|>hi<|end_message|>"
    );
}
