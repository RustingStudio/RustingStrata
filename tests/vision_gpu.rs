//! The vision encoders on the GPU (`--gpu`) against the CPU code. Its own binary: `vision::use_gpu` holds for the
//! whole process. Each mmproj is skipped when absent, the test when no GPU opens.

use rusting_strata::vision::{VisionProjector, use_gpu};
use std::path::Path;

const MMPROJS: [(&str, &str); 4] = [
    ("STRATA_QWEN36_MMPROJ", "/home/vasylt/models/qwen36-mmproj/mmproj-BF16.gguf"),
    ("STRATA_INKLING2_MMPROJ", "/home/vasylt/models/inkling2/mmproj-F16.gguf"),
    ("STRATA_GEMMA4_MMPROJ", "/home/vasylt/models/gemma4-26b/mmproj-F16.gguf"),
    ("STRATA_GEMMA4UV_MMPROJ", "/home/vasylt/models/gemma4-uv-0.8b/gemma-4-unified-0.8B-tiny.mmproj-f16.gguf"),
];

#[test]
fn gpu_encode_matches_cpu() {
    // STRATA_TEST_BACKENDS=vulkan tests the Vulkan encoder; else CUDA.
    let kind = std::env::var("STRATA_TEST_BACKENDS").ok().filter(|k| k.as_str() == "vulkan");
    if let Err(e) = rusting_strata::backend::open(kind.as_deref(), 0) { eprintln!("skip: no GPU ({e:#})"); return; }
    let img = image::RgbImage::from_fn(448, 336, |x, y| image::Rgb([(x % 256) as u8, (y * 3 % 256) as u8, ((x ^ y) % 256) as u8]));
    let projs: Vec<_> = MMPROJS.iter().filter_map(|(var, def)| {
        let p = std::env::var(var).unwrap_or_else(|_| def.to_string());
        if !Path::new(&p).exists() { eprintln!("skip: {p} missing"); return None; }
        Some(VisionProjector::open(Path::new(&p)).expect("open mmproj"))
    }).collect();
    let pre: Vec<_> = projs.iter().map(|p| p.preprocess(&img).unwrap()).collect();
    let cpu: Vec<_> = projs.iter().zip(&pre).map(|(p, i)| p.encode(i).unwrap()).collect();
    use_gpu(kind, 0);
    for ((p, i), want) in projs.iter().zip(&pre).zip(&cpu) {
        let got = p.encode(i).unwrap();
        assert_eq!(got.len(), want.len());
        let worst = got.iter().zip(want).map(|(a, b)| {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
            let n = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            dot / (n(a) * n(b))
        }).fold(1f64, f64::min);
        eprintln!("{}: {} tokens, lowest cosine {worst:.6}", p.cfg().projector_type, got.len());
        assert!(worst > 0.999, "{}: GPU token cosine {worst} vs the CPU", p.cfg().projector_type);
    }
}
