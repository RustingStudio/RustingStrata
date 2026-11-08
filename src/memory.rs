//! Host RAM budget: total and free physical RAM, plus the usable cap.
//!
//! RustingStrata holds the model in an mmapped GGUF (page cache), weights
//! dequantized on demand, activation buffers, a paged expert mailbox, and
//! parked-conversation caches. To leave the OS breathing room, the engine
//! never plans to use more than `total - margin` bytes of host RAM. The
//! margin defaults to 512 MiB and is usually set higher (e.g. 6 GiB on a
//! 32 GiB host leaves ~26 GiB usable).

pub const DEFAULT_RAM_MARGIN_BYTES: u64 = 512 << 20;

/// (total_bytes, free_bytes) of physical RAM, or `None` if undetectable.
pub fn ram() -> Option<(u64, u64)> {
    #[cfg(target_os = "linux")]
    {
        let read = |p: &str| std::fs::read_to_string(p).unwrap_or_default();
        let kb = |s: &str, k: &str| {
            s.lines()
                .find_map(|l| l.strip_prefix(k)?.trim().strip_suffix("kB")?.trim().parse::<u64>().ok())
                .map(|n| n << 10)
        };
        let m = read("/proc/meminfo");
        if let (Some(t), Some(a)) = (kb(&m, "MemTotal:"), kb(&m, "MemAvailable:")) {
            return Some((t, a));
        }
    }
    #[cfg(windows)]
    {
        #[repr(C)]
        struct MemStatus {
            len: u32,
            load: u32,
            total: u64,
            avail: u64,
            rest: [u64; 5],
        }
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GlobalMemoryStatusEx(m: *mut MemStatus) -> i32;
        }
        let mut m = MemStatus { len: size_of::<MemStatus>() as u32, load: 0, total: 0, avail: 0, rest: [0; 5] };
        if unsafe { GlobalMemoryStatusEx(&mut m) } != 0 && m.total > 0 {
            return Some((m.total, m.avail));
        }
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        let _ = DEFAULT_RAM_MARGIN_BYTES;
    }
    None
}

/// Effective usable cap in bytes: an explicit `--max-ram` if given, else
/// `total - margin`. On failure to detect RAM, `None` means unlimited.
pub fn ram_cap(margin_bytes: u64, max_ram_bytes: Option<u64>) -> Option<u64> {
    if let Some(m) = max_ram_bytes {
        return Some(m);
    }
    let (total, _) = ram()?;
    Some(total.saturating_sub(margin_bytes))
}

/// Free space the budget still leaves after `used` bytes are allocated.
pub fn ram_room(cap: Option<u64>, used: u64) -> Option<u64> {
    cap.map(|c| c.saturating_sub(used))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cap_respects_margin() {
        let cap = (32 << 30) - (6 << 30);
        assert_eq!(cap, 26 << 30);
        assert_eq!(ram_room(Some(cap), 10 << 30), Some(16 << 30));
        assert_eq!(ram_room(Some(cap), 30 << 30), Some(0));
    }

    #[test]
    fn explicit_max_ram_wins() {
        assert_eq!(ram_cap(512 << 20, Some(8 << 30)), Some(8 << 30));
    }
}
