// Shared by integration tests via `mod common;`.
#![allow(dead_code)]

pub const DEFAULT_MODEL: &str = "/mnt/ollama/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf";

pub fn model_path() -> Option<std::path::PathBuf> {
    let p = std::path::PathBuf::from(std::env::var("STRATA_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into()));
    if p.exists() { Some(p) } else { eprintln!("SKIP: model file missing"); None }
}

/// Held by tests that load the whole model on the GPU: two at once do not fit in 12 GB.
pub fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}
