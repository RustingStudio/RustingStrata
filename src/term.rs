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
pub fn status(s: &str) -> String { status_on(err(), s) }

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
}
