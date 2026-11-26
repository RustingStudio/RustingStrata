//! `tools/bench_2x3060.sh`: a row that fails or prints nothing must not end the sweep, the sweep must exit nonzero
//! when a row was lost, and the environment must reach `bench` (`gpu_defaults` sets `STRATA_DS_PLAIN_FIRST=1` only
//! when it is unset, so the script must not unset it).
#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

/// Run the sweep with `--pp 512,2048 --ctx 32768` against `body` as `BIN`; `out` adds `--out`. Returns the temp
/// dir and the output.
fn sweep(tag: &str, body: &str, out: bool) -> (PathBuf, std::process::Output) {
    let dir =
        std::env::temp_dir().join(format!("strata-bench-script-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // `BIN` is checked with `-f` after the script cds to the repo root, so it is a path, not a name on PATH.
    let bin = dir.join("fake");
    std::fs::write(&bin, body).unwrap();
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
    let json = dir.join("out").to_string_lossy().into_owned();
    let mut cmd = std::process::Command::new("bash");
    cmd.arg(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tools/bench_2x3060.sh"
    ))
    .args(["--pp", "512,2048", "--ctx", "32768"])
    .env("BIN", &bin)
    .env("STRATA_INKLING2_MODEL", "/dev/null")
    .env("FAKE_LOG", dir.join("calls"))
    .env("STRATA_DS_PLAIN_FIRST", "0");
    if out {
        cmd.args(["--out", &json]);
    }
    let out = cmd.output().unwrap();
    (dir, out)
}

fn report(out: &std::process::Output) -> String {
    format!(
        "status {}\nstdout:\n{}\nstderr:\n{}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

#[test]
fn a_failed_row_does_not_end_the_sweep() {
    let (dir, out) = sweep(
        "failed-row",
        r#"#!/bin/sh
if echo "$*" | grep -q -- "--pp 512"; then echo boom >&2; exit 1; fi
echo "pp 2048: prefill 100.0 (100.0-100.0) tok/s | tg 32: decode 50.0 (50.0-50.0) tok/s"
echo "${STRATA_DS_PLAIN_FIRST:-unset}" >> "$FAKE_LOG"
"#,
        false,
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !out.status.success(),
        "a sweep with a lost row must exit nonzero: {}",
        report(&out)
    );
    // The table row of the second prompt: ctx pp prefill decode.
    assert!(
        stdout
            .lines()
            .any(|l| l.split_whitespace().eq(["32768", "2048", "100.0", "50.0"])),
        "the row after the failure is missing: {}",
        report(&out)
    );
    assert!(
        stdout.lines().any(
            |l| l.contains("pp 512") && (l.contains("no result") || l.contains("bench failed"))
        ),
        "no diagnostic for the failed row: {}",
        report(&out)
    );
    let calls = std::fs::read_to_string(dir.join("calls")).unwrap_or_default();
    assert!(
        !calls.is_empty(),
        "the fake bench never ran: {}",
        report(&out)
    );
    for (i, line) in calls.lines().enumerate() {
        assert_eq!(
            line,
            "0",
            "call {} saw STRATA_DS_PLAIN_FIRST={line:?}, so the script unset it: {}",
            i + 1,
            report(&out)
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_row_with_no_output_does_not_end_the_sweep() {
    let (dir, out) = sweep("silent-row", "#!/bin/sh\nexit 0\n", false);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !out.status.success(),
        "a sweep with no rows must exit nonzero: {}",
        report(&out)
    );
    let missing = stdout
        .lines()
        .filter(|l| l.contains("no result") || l.contains("bench failed"))
        .count();
    assert_eq!(
        missing,
        2,
        "both rows must report no result: {}",
        report(&out)
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_successful_sweep_exits_zero_with_out() {
    let (dir, out) = sweep(
        "ok-out",
        r#"#!/bin/sh
pp=""
prev=""
for a in "$@"; do
  [ "$prev" = "--pp" ] && pp="$a"
  prev="$a"
done
echo "pp $pp: prefill 100.0 (100.0-100.0) tok/s | tg 32: decode 50.0 (50.0-50.0) tok/s"
"#,
        true,
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "a successful sweep must exit zero: {}",
        report(&out)
    );
    for pp in ["512", "2048"] {
        assert!(
            stdout
                .lines()
                .any(|l| l.split_whitespace().eq(["32768", pp, "100.0", "50.0"])),
            "the pp {pp} row is missing: {}",
            report(&out)
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
