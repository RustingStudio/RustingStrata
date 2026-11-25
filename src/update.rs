//! `rustingstrata update`: replaces the running binary with the newest GitHub release, and the one-line notice
//! `serve` prints at start when a newer release is out. `STRATA_NO_UPDATE_CHECK=1` turns the notice off.

use anyhow::{Context, bail, ensure};
use std::io::{Read, Write};
use std::time::Duration;

const LATEST: &str = "https://api.github.com/repos/RustingStudio/RustingStrata/releases/latest";

/// The release asset for this platform, as the release workflow names it.
const ASSET: Option<&str> = if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
    Some("rustingstrata-linux-x86_64")
} else if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
    Some("rustingstrata-windows-x86_64.exe")
} else {
    None
};

fn agent(whole: Duration) -> ureq::Agent {
    let s = Some(Duration::from_secs(60));
    ureq::Agent::config_builder().timeout_connect(Some(Duration::from_secs(15))).timeout_recv_response(s)
        .timeout_per_read(s).timeout_global(Some(whole)).user_agent("rustingstrata-update").build().into()
}

/// "v1.2.10" or "1.2.10" as numbers, so 1.2.10 is newer than 1.2.9.
fn version(s: &str) -> Option<Vec<u64>> { s.trim_start_matches('v').split('.').map(|p| p.parse().ok()).collect() }

fn newer(tag: &str, have: &str) -> bool { matches!((version(tag), version(have)), (Some(a), Some(b)) if a > b) }

/// The latest release's tag, with the download URL and size of this platform's binary if it has one.
fn latest(whole: Duration) -> anyhow::Result<(String, Option<(String, u64)>)> {
    let mut r = agent(whole).get(LATEST).header("Accept", "application/vnd.github+json").call()
        .context("ask GitHub for the latest release")?;
    let v: serde_json::Value = serde_json::from_str(&r.body_mut().read_to_string()?)?;
    let tag = v["tag_name"].as_str().context("the release has no tag")?.to_string();
    let asset = v["assets"].as_array().into_iter().flatten().find(|a| Some(a["name"].as_str().unwrap_or("")) == ASSET)
        .and_then(|a| Some((a["browser_download_url"].as_str()?.to_string(), a["size"].as_u64()?)));
    Ok((tag, asset))
}

/// Prints one line if a newer release is out. It runs on its own thread and gives up after 5 s, so a machine
/// without internet starts as before.
pub fn notice() {
    if std::env::var("STRATA_NO_UPDATE_CHECK").is_ok_and(|v| v == "1") || ASSET.is_none() { return; }
    std::thread::spawn(|| {
        if let Ok((tag, Some(_))) = latest(Duration::from_secs(5)) && newer(&tag, env!("CARGO_PKG_VERSION")) {
            crate::say!("RustingStrata {tag} is out (this is {}): run `rustingstrata update` to install it", env!("CARGO_PKG_VERSION"));
        }
    });
}

/// Downloads the newest release next to the running binary and swaps it in. The old binary is kept as `<exe>.old`
/// until the next update: Windows cannot delete a running exe, but it can rename it.
pub fn run(check: bool) -> anyhow::Result<()> {
    let have = env!("CARGO_PKG_VERSION");
    let (tag, asset) = latest(Duration::from_secs(60))?;
    if !newer(&tag, have) {
        crate::say!("RustingStrata {have} is the newest release");
        return Ok(());
    }
    if check {
        crate::say!("RustingStrata {tag} is out (this is {have}): run `rustingstrata update` to install it");
        return Ok(());
    }
    let Some((url, size)) = asset else {
        bail!("release {tag} has no binary for this platform; build it from source or see \
               https://github.com/RustingStudio/RustingStrata/releases");
    };
    let exe = std::env::current_exe().context("find the running binary")?;
    let exe = exe.canonicalize().unwrap_or(exe);
    let (new, old) = (exe.with_extension("new"), exe.with_extension("old"));
    let _ = std::fs::remove_file(&old);
    crate::say!("downloading RustingStrata {tag} ({:.1} MB)", size as f64 / 1e6);
    let r = agent(Duration::from_secs(1800)).get(&url).call().with_context(|| format!("download {url}"))?;
    let mut out = std::fs::File::create(&new).with_context(|| format!("write {} (is the folder writable?)", new.display()))?;
    let (mut rd, mut buf, mut got) = (r.into_body().into_reader(), vec![0; 1 << 20], 0u64);
    loop {
        let n = rd.read(&mut buf).context("download")?;
        if n == 0 { break; }
        out.write_all(&buf[..n])?;
        got += n as u64;
    }
    drop(out);
    if got != size {
        let _ = std::fs::remove_file(&new);
        bail!("received {got} of {size} bytes; nothing was changed");
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&new, std::fs::Permissions::from_mode(0o755))?;
    }
    std::fs::rename(&exe, &old).with_context(|| format!("move {} aside", exe.display()))?;
    if let Err(e) = std::fs::rename(&new, &exe) {
        let _ = std::fs::rename(&old, &exe);
        return Err(e).with_context(|| format!("put the new binary at {}", exe.display()));
    }
    ensure!(exe.exists(), "{} is missing after the update", exe.display());
    crate::say!("updated {have} -> {tag}: {}", exe.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_compare_as_numbers() {
        assert!(newer("v1.3.10", "1.3.9"));
        assert!(newer("v1.4.0", "1.3.4"));
        assert!(!newer("v1.3.4", "1.3.4"));
        assert!(!newer("v1.3.3", "1.3.4"));
        assert!(!newer("nightly", "1.3.4"));
    }
}
