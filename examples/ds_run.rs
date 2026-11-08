//! Greedy CPU reference run of DeepSeek-V4.1: ds_run GGUF "prompt" [n_new]. Set STRATA_DS_TRACE=1 for per-tensor traces.
use rusting_strata::{deepseek::forward::DsModel, gguf::Gguf, tokenizer::Tokenizer};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let g = Gguf::open(std::path::Path::new(&a[1]))?;
    let tok = Tokenizer::from_gguf(&g)?;
    let ids = tok.encode(&a[2])?;
    let n_new: usize = a.get(3).map_or(1, |s| s.parse().unwrap());
    let mut m = DsModel::new(&g)?;
    let mut logits = vec![];
    for &t in &ids {
        let t0 = std::time::Instant::now();
        logits = m.step(t);
        eprintln!("prompt tok {t} {:?}", t0.elapsed());
    }
    let mut out = vec![];
    for _ in 0..n_new.max(1) {
        let mut top: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("top5 {:?}", &top[..5]);
        let next = top[0].0 as u32;
        out.push(next);
        if out.len() < n_new { logits = m.step(next); }
    }
    println!("ids {out:?}");
    println!("text {:?}", tok.decode(&out)?);
    Ok(())
}
