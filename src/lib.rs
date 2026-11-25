pub mod memory;
pub mod term;
pub mod calibrate;

/// `--ctx` when it was given on the command line, else 0. The models outside the main engine (Inkling 2, Hy4, GLM,
/// DeepSeek, Gemma 4) read it through [`ctx_limit`].
pub static CTX: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Prompt plus reply limit of a model outside the main engine: `--ctx`, else the trained context from the GGUF.
pub fn ctx_limit(trained: usize) -> usize {
    match CTX.load(std::sync::atomic::Ordering::Relaxed) { 0 => trained, n => n }
}
pub mod config;
pub mod gguf;
pub mod quant;
pub mod weights;
pub mod ops;
pub mod tokenizer;
pub mod model;
pub mod backend;
pub mod cuda;
pub mod vulkan;
pub mod hip;
pub mod gpu_model;
pub mod cpu_moe;
pub mod expert_cache;
pub mod cpu_worker;
pub mod serve;
pub mod chat;
pub mod vision;
pub mod hf;
pub mod mcp;
pub mod deepseek;
pub mod gemma4;
pub mod glm;
pub mod hy4;
pub mod kolibri;
pub mod inkling2;
pub mod ngram;
