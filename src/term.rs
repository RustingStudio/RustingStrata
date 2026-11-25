//! Terminal styling. It is off when the stream is not a terminal, when `NO_COLOR` is set and when `TERM` is `dumb`,
//! so a pipe, a log file and a script get the same bytes as without it.
use std::io::IsTerminal;
use std::sync::OnceLock;

/// Windows consoles before Windows Terminal print escape codes as text until the mode is set.
#[cfg(windows)]
fn vt(handle: u32) -> bool {
    unsafe extern "system" {
        fn GetStdHandle(n: u32) -> *mut core::ffi::c_void;
        fn GetConsoleMode(h: *mut core::ffi::c_void, mode: *mut u32) -> i32;
        fn SetConsoleMode(h: *mut core::ffi::c_void, mode: u32) -> i32;
    }
    let mut mode = 0;
    // 4 is ENABLE_VIRTUAL_TERMINAL_PROCESSING.
    unsafe {
        let h = GetStdHandle(handle);
        GetConsoleMode(h, &mut mode) != 0 && SetConsoleMode(h, mode | 4) != 0
    }
}
#[cfg(not(windows))]
fn vt(_: u32) -> bool { true }

fn wanted(tty: bool, handle: u32) -> bool {
    tty && std::env::var_os("NO_COLOR").is_none_or(|v| v.is_empty()) && std::env::var_os("TERM").is_none_or(|v| v != "dumb") && vt(handle)
}

/// Styling of stderr, where the status lines go.
pub fn err() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| wanted(std::io::stderr().is_terminal(), -12i32 as u32))
}

/// Styling of stdout, where the model's text and the prompts go.
pub fn out() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| wanted(std::io::stdout().is_terminal(), -11i32 as u32))
}

/// `s` in the SGR style `code` (`1` bold, `2` dim, `36` cyan) when `on`, else `s`.
pub fn paint(on: bool, code: &str, s: &str) -> String {
    if on { format!("\x1b[{code}m{s}\x1b[0m") } else { s.to_string() }
}

/// One status line for stderr: a leading `warning` or `error` in colour, any other line dim, so the model's text
/// stands out from the engine's notes.
pub fn status(s: &str) -> String {
    // Leading blank lines stay before the time.
    let t = s.trim_start_matches('\n');
    format!("{}{} {}", &s[..s.len() - t.len()], paint(err(), "2", &clock()), status_on(err(), t))
}

/// The local wall-clock time as `[HH:MM:SS]`, which starts each status line.
pub fn clock() -> String {
    let (h, m, s) = local_hms();
    format!("[{h:02}:{m:02}:{s:02}]")
}

#[cfg(unix)]
fn local_hms() -> (i32, i32, i32) {
    // `struct tm` starts with tm_sec, tm_min, tm_hour (ints) on every Unix; the buffer is larger than any `struct tm`.
    unsafe extern "C" { fn localtime_r(t: *const i64, tm: *mut [i32; 16]) -> *mut [i32; 16]; }
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64);
    let mut tm = [0i32; 16];
    if unsafe { localtime_r(&t, &mut tm) }.is_null() { return ((t / 3600 % 24) as i32, (t / 60 % 60) as i32, (t % 60) as i32); }
    (tm[2], tm[1], tm[0])
}

#[cfg(windows)]
fn local_hms() -> (i32, i32, i32) {
    // SYSTEMTIME: year, month, day of week, day, hour, minute, second, milliseconds.
    unsafe extern "system" { fn GetLocalTime(t: *mut [u16; 8]); }
    let mut t = [0u16; 8];
    unsafe { GetLocalTime(&mut t) };
    (t[4] as i32, t[5] as i32, t[6] as i32)
}

#[cfg(not(any(unix, windows)))]
fn local_hms() -> (i32, i32, i32) {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    ((t / 3600 % 24) as i32, (t / 60 % 60) as i32, (t % 60) as i32)
}

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use std::sync::Mutex;
use std::time::Instant;

/// Prompt and reply tokens run so far, for the speed line (`live`).
static PREFILLED: AtomicUsize = AtomicUsize::new(0);
static DECODED: AtomicUsize = AtomicUsize::new(0);
static LIVE_DECODE: AtomicBool = AtomicBool::new(false);
/// The [`Live`]s held, and while there is one: when the first started, the counters then, and when the last prompt
/// tokens ran.
static RUN: Mutex<(usize, Option<(Instant, usize, usize, Instant)>)> = Mutex::new((0, None));

/// Counts `n` prompt tokens that ran through the model.
pub fn prefilled(n: usize) {
    PREFILLED.fetch_add(n, Relaxed);
    if let Some(r) = RUN.lock().unwrap_or_else(|e| e.into_inner()).1.as_mut() { r.3 = Instant::now(); }
}

/// Counts `n` reply tokens.
pub fn decoded(n: usize) { DECODED.fetch_add(n, Relaxed); }

/// Turns on the speed line: every 2 s while a [`Live`] is held, one status line with the prefill speed (prompt
/// tokens so far over the time they took), or the decode speed over the last 2 s when `decode` (serve; the chat
/// prints its reply on the terminal, so it shows only the prefill). Nothing is printed while idle.
pub fn live(decode: bool) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    LIVE_DECODE.store(decode, Relaxed);
    ONCE.call_once(|| { std::thread::spawn(|| {
        let (mut last_p, mut last_d, mut at) = (0, 0, Instant::now());
        loop {
            std::thread::sleep(std::time::Duration::from_secs(2));
            let (p, d, now) = (PREFILLED.load(Relaxed), DECODED.load(Relaxed), Instant::now());
            let run = RUN.lock().unwrap_or_else(|e| e.into_inner()).1;
            if let Some((t0, p0, d0, tp)) = run {
                if let Some(l) = speed_line(p - p0, d - d0, p - last_p.max(p0), d - last_d.max(d0), (tp - t0).as_secs_f64(), (now - at).as_secs_f64(), LIVE_DECODE.load(Relaxed)) {
                    crate::say!("{l}");
                }
            }
            (last_p, last_d, at) = (p, d, now);
        }
    }); });
}

/// The speed line from the tokens of this run (`dp`, `dd`), those since the last line (`np`, `nd`), the seconds the
/// prompt tokens took and the seconds since the last line. None when nothing ran.
fn speed_line(dp: usize, dd: usize, np: usize, nd: usize, prefill_s: f64, window_s: f64, decode: bool) -> Option<String> {
    if dd > 0 {
        return (decode && nd > 0).then(|| format!("decode: {:.1} tok/s, {dd} tok", nd as f64 / window_s.max(1e-9)));
    }
    (np > 0).then(|| format!("prefill: {:.1} tok/s, {dp} tok", dp as f64 / prefill_s.max(1e-9)))
}

/// A request or chat turn that generates; the speed line (`live`) covers it until it is dropped. Requests that run
/// at once (`serve --parallel`) share one line.
pub struct Live;

impl Live {
    pub fn start() -> Live {
        let mut r = RUN.lock().unwrap_or_else(|e| e.into_inner());
        if r.0 == 0 { let now = Instant::now(); r.1 = Some((now, PREFILLED.load(Relaxed), DECODED.load(Relaxed), now)); }
        r.0 += 1;
        Live
    }
}

impl Drop for Live {
    fn drop(&mut self) {
        let mut r = RUN.lock().unwrap_or_else(|e| e.into_inner());
        r.0 -= 1;
        if r.0 == 0 { r.1 = None; }
    }
}

fn status_on(on: bool, s: &str) -> String {
    if !on { return s.to_string(); }
    for (label, code) in [("warning", "1;33"), ("error", "1;31")] {
        if let Some(rest) = s.strip_prefix(label) { return format!("\x1b[{code}m{label}\x1b[0m{rest}"); }
    }
    format!("\x1b[2m{s}\x1b[0m")
}

/// A part of a dim status line that the user acts on (an address), in bold.
pub fn hl(s: &str) -> String {
    if err() { format!("\x1b[0;1m{s}\x1b[0;2m") } else { s.to_string() }
}

/// The input prompt of the terminal chats.
pub fn prompt() -> String { paint(out(), "1;36", "> ") }

/// `eprintln!` for a status line, styled by [`term::status`](crate::term::status).
#[macro_export]
macro_rules! say {
    ($($t:tt)*) => { eprintln!("{}", $crate::term::status(&format!($($t)*))) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_styles_by_prefix_and_is_plain_when_off() {
        assert_eq!(status_on(false, "warning: x"), "warning: x");
        assert_eq!(status_on(true, "warning: x"), "\x1b[1;33mwarning\x1b[0m: x");
        assert_eq!(status_on(true, "error: x"), "\x1b[1;31merror\x1b[0m: x");
        assert_eq!(status_on(true, "loaded m"), "\x1b[2mloaded m\x1b[0m");
        assert_eq!(paint(false, "1", "a"), "a");
    }

    #[test]
    fn clock_and_speed_lines() {
        let c = clock();
        assert!(c.len() == 10 && c.starts_with('[') && c.ends_with(']') && c.as_bytes()[3] == b':', "{c}");
        assert_eq!(speed_line(0, 0, 0, 0, 0.0, 2.0, true), None);
        assert_eq!(speed_line(1000, 0, 500, 0, 4.0, 2.0, true).as_deref(), Some("prefill: 250.0 tok/s, 1000 tok"));
        // A prompt chunk still running: no new prompt tokens, no line.
        assert_eq!(speed_line(1000, 0, 0, 0, 4.0, 2.0, true), None);
        assert_eq!(speed_line(1000, 40, 0, 30, 4.0, 2.0, true).as_deref(), Some("decode: 15.0 tok/s, 40 tok"));
        assert_eq!(speed_line(1000, 40, 0, 30, 4.0, 2.0, false), None);
    }
}
