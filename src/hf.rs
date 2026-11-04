//! Model downloads from Hugging Face for the chat page of `strata serve`. `HF_TOKEN`, if set, is sent for gated
//! and private repos.

use crate::serve::{models, shard};
use anyhow::{Context, bail, ensure};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Mutex;
use std::time::Duration;

/// The running or last download, as the chat page shows it.
#[derive(Clone, Default, serde::Serialize)]
pub struct Progress {
    pub repo: String,
    pub file: String,
    pub done: u64,
    pub total: u64,
    pub running: bool,
    pub cancel: bool,
    pub error: Option<String>,
}

fn get(url: &str) -> ureq::RequestBuilder<ureq::typestate::WithoutBody> {
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_connect(Some(Duration::from_secs(30))).build().into();
    let r = agent.get(url);
    match std::env::var("HF_TOKEN") { Ok(t) if !t.is_empty() => r.header("Authorization", format!("Bearer {t}")), _ => r }
}

/// "owner/name" with the characters Hugging Face allows, so it cannot change the URL it goes into.
pub fn valid_repo(r: &str) -> bool {
    let ok = |s: &str| !s.is_empty() && s != "." && s != ".." && s.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b));
    r.split_once('/').is_some_and(|(a, b)| ok(a) && ok(b))
}

/// Every file of `repo` with its size in bytes.
fn files(repo: &str) -> anyhow::Result<Vec<(String, u64)>> {
    ensure!(valid_repo(repo), "{repo:?} is not a Hugging Face repo name (owner/name)");
    let mut r = get(&format!("https://huggingface.co/api/models/{repo}?blobs=true")).call()
        .with_context(|| format!("Hugging Face repo {repo} (check the name; a private or gated repo needs HF_TOKEN)"))?;
    let v: serde_json::Value = serde_json::from_str(&r.body_mut().read_to_string()?)?;
    Ok(v["siblings"].as_array().context("no file list")?.iter()
        .filter_map(|s| Some((s["rfilename"].as_str()?.to_string(), s["size"].as_u64().unwrap_or(0)))).collect())
}

/// The models of `repo` as (file, bytes over all shards); see `serve::models`.
pub fn list(repo: &str) -> anyhow::Result<Vec<(String, u64)>> { Ok(models(&files(repo)?)) }

/// Downloads `file` of `repo` (with the other shards of a split model) into `dir`, without the repo's folders.
/// Each file goes through a `.part` file that a later attempt resumes. `p` tracks the bytes; `p.cancel` stops it.
pub fn download(repo: &str, file: &str, dir: &Path, p: &Mutex<Progress>) -> anyhow::Result<()> {
    let all = files(repo)?;
    let want: Vec<_> = all.into_iter().filter(|(f, _)| f.ends_with(".gguf") && match shard(file) {
        Some((stem, _)) => shard(f).is_some_and(|(s, _)| s == stem),
        None => f == file,
    }).collect();
    ensure!(!want.is_empty(), "{repo} has no model file {file}");
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    p.lock().unwrap().total = want.iter().map(|x| x.1).sum();
    let mut before = 0;
    for (f, size) in &want {
        let name = f.rsplit('/').next().unwrap_or(f);
        let (dst, part) = (dir.join(name), dir.join(format!("{name}.part")));
        if !dst.exists() {
            let mut have = std::fs::metadata(&part).map_or(0, |m| m.len());
            if have < *size || *size == 0 {
                let url = format!("https://huggingface.co/{repo}/resolve/main/{f}");
                let r = get(&url).header("Range", format!("bytes={have}-")).call().with_context(|| format!("download {f}"))?;
                if r.status() != 206 { have = 0; }  // the range was ignored: start over
                let mut out = std::fs::OpenOptions::new().create(true).write(true).append(have > 0).truncate(have == 0).open(&part)?;
                let (mut rd, mut buf) = (r.into_body().into_reader(), vec![0; 1 << 20]);
                loop {
                    let n = rd.read(&mut buf).with_context(|| format!("download {f}"))?;
                    if n == 0 { break; }
                    out.write_all(&buf[..n]).with_context(|| format!("write {}", part.display()))?;
                    have += n as u64;
                    let mut g = p.lock().unwrap();
                    g.done = before + have;
                    if g.cancel { bail!("cancelled ({} kept, the next download of this file resumes it)", part.display()); }
                }
            }
            ensure!(*size == 0 || have == *size, "{f}: received {have} of {size} bytes");
            std::fs::rename(&part, &dst)?;
        }
        before += size;
        p.lock().unwrap().done = before;
    }
    Ok(())
}
