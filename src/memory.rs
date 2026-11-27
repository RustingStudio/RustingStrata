//! Host RAM cap: `total physical RAM - margin`, or an explicit ceiling.
//!
//! The cap bounds memory the process owns outright (baked tensors, LoRA factors, parked conversation states). The
//! mmapped GGUF is page cache the OS can evict, so a model larger than RAM still loads and streams.

/// Total physical RAM in bytes, or `None` if undetectable. On Linux the memory limit of the process's cgroup when it
/// is lower: a container sees the host's MemTotal (issue #132), and its page cache counts toward the limit.
pub fn total_ram() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let m = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
        let kb = m.lines().find_map(|l| l.strip_prefix("MemTotal:")?.trim().strip_suffix("kB")?.trim().parse::<u64>().ok());
        if let Some(k) = kb {
            return Some(cgroup_limit().map_or(k << 10, |c| c.min(k << 10)));
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
            return Some(m.total);
        }
    }
    None
}

/// The memory limit of this process's cgroup: v2 `memory.max` of its own cgroup or one above it (a container sees
/// its own as the root), else v1 `memory.limit_in_bytes`. `None` without a limit.
#[cfg(target_os = "linux")]
fn cgroup_limit() -> Option<u64> {
    let own = std::fs::read_to_string("/proc/self/cgroup").ok()
        .and_then(|c| c.lines().find_map(|l| l.strip_prefix("0::").map(str::to_owned))).unwrap_or_default();
    let mut dir = std::path::PathBuf::from(format!("/sys/fs/cgroup{}", own.trim_end_matches('/')));
    let read = |p: std::path::PathBuf| std::fs::read_to_string(p).ok()?.trim().parse::<u64>().ok();
    let mut lim: Option<u64> = None;
    loop {
        if let Some(v) = read(dir.join("memory.max")) { lim = Some(lim.map_or(v, |l| l.min(v))); }
        if !dir.pop() || !dir.starts_with("/sys/fs/cgroup") { break; }
    }
    lim.or_else(|| read("/sys/fs/cgroup/memory/memory.limit_in_bytes".into()))
}

/// Usable cap in bytes: `max_ram` if nonzero, else `total - margin`. `None` means unlimited (RAM undetectable).
pub fn ram_cap(margin: u64, max_ram: u64) -> Option<u64> {
    if max_ram > 0 {
        return Some(max_ram);
    }
    Some(total_ram()?.saturating_sub(margin))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_max_ram_wins() {
        assert_eq!(ram_cap(512 << 20, 8 << 30), Some(8 << 30));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cgroup_limit_caps_total() {
        if let (Some(t), Some(c)) = (total_ram(), cgroup_limit()) { assert!(t <= c); }
    }

    #[test]
    fn margin_is_subtracted() {
        if let Some(t) = total_ram() {
            assert_eq!(ram_cap(1 << 30, 0), Some(t.saturating_sub(1 << 30)));
        }
    }
}
