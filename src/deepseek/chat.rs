//! `strata chat` for DeepSeek-V4.1: greedy, CUDA for the matvecs and expert staging, KV kept across turns.
use super::forward::DsModel;
use crate::{chat::{Msg, Template}, gguf::Gguf, tokenizer::Tokenizer};
use std::io::{BufRead, Write};

pub fn chat(g: &Gguf, system: Option<String>, n: usize, think: bool) -> anyhow::Result<()> {
    // GPU on by default for the chat; VRAM expert cache 8 GB (rest of the experts stream through one staging slot).
    for (k, v) in [("STRATA_DS_GPU", "1"), ("STRATA_DS_GPU_EXPERTS", "8")] {
        if std::env::var_os(k).is_none() { unsafe { std::env::set_var(k, v) } }
    }
    let tok = Tokenizer::from_gguf(g)?;
    let mut m = DsModel::new(g)?;
    let eos = tok.eos_ids();
    let tpl = Template::Ds;
    let mut first = true;
    let stdin = std::io::stdin();
    loop {
        print!("> ");
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
        for t in tok.encode(&text)? { logits = m.step(t); }
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
