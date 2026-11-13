//! Vision projector tests against the local Qwen3.5-2B mmproj. Skipped when the file is absent.

use rusting_strata::gguf::Gguf;
use rusting_strata::model::{Model, State};
use rusting_strata::ops;
use rusting_strata::tokenizer::Tokenizer;
use rusting_strata::vision::{PreprocessedImage, VisionProjector, expand_images};
use std::path::{Path, PathBuf};

const MMPROJ: &str = "/mnt/ollama/models/.studio_links/9a9ae26c56/qwen3.5-2b-mmproj.gguf";
const GEMMA4_MMPROJ: &str = "/home/vasylt/models/gemma4-26b/mmproj-F16.gguf";
const GEMMA4UV_MMPROJ: &str = "/home/vasylt/models/gemma4-uv-0.8b/gemma-4-unified-0.8B-tiny.mmproj-f16.gguf";
const GEMMA4UV_MODEL: &str = "/home/vasylt/models/gemma4-uv-0.8b/gemma-4-unified-0.8B-tiny.Q2_K.gguf";
const TEXT_MODEL: &str = "/mnt/ollama/models/.studio_links/9a9ae26c56/qwen3.5-2b.gguf";

fn text_model() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("STRATA_VISION_TEXT_MODEL").unwrap_or_else(|_| TEXT_MODEL.into()));
    if p.exists() { Some(p) } else { eprintln!("skip: {} missing", p.display()); None }
}

fn projector() -> Option<VisionProjector> {
    if !Path::new(MMPROJ).exists() { return None; }
    Some(VisionProjector::open(Path::new(MMPROJ)).expect("open mmproj"))
}

fn gemma4_projector() -> Option<VisionProjector> {
    let p = std::env::var("STRATA_GEMMA4_MMPROJ").unwrap_or_else(|_| GEMMA4_MMPROJ.into());
    if !Path::new(&p).exists() { eprintln!("skip: {p} missing"); return None; }
    Some(VisionProjector::open(Path::new(&p)).expect("open gemma4 mmproj"))
}

fn gemma4uv_projector() -> Option<VisionProjector> {
    let p = std::env::var("STRATA_GEMMA4UV_MMPROJ").unwrap_or_else(|_| GEMMA4UV_MMPROJ.into());
    if !Path::new(&p).exists() { eprintln!("skip: {p} missing"); return None; }
    Some(VisionProjector::open(Path::new(&p)).expect("open gemma4uv mmproj"))
}

#[test]
fn qwen35_2b_white_64x64_sum() {
    let Some(p) = projector() else { eprintln!("skip: {MMPROJ} missing"); return; };
    let img = image::RgbImage::from_pixel(64, 64, image::Rgb([255, 255, 255]));
    let c = p.cfg();
    let pre = PreprocessedImage::from_u8(64, 64, img.as_raw(), c.image_mean, c.image_std).unwrap();
    let feats = p.encode(&pre).unwrap();
    assert_eq!(feats.len(), 4, "64x64 -> 4x4 patches -> 2x2 merged tokens");
    assert!(feats.iter().all(|r| r.len() == c.projection_dim));
    let sum: f32 = feats.iter().flatten().sum();
    // llama-mtmd-debug -p encode -n 64 --image white: sum = -9.838478, first row
    // [0.1083, 0.1155, 0.0651, ...]. The remaining difference is F16 weight-dot accumulation order.
    assert!((sum - -9.838_478).abs() < 0.15, "sum {sum} vs llama.cpp -9.838478");
    assert!((feats[0][0] - 0.1083).abs() < 5e-3);
    assert!((feats[0][1] - 0.1155).abs() < 5e-3);
    assert!((feats[0][2] - 0.0651).abs() < 5e-3);
}

#[test]
fn qwen35_2b_preprocess_grid() {
    let Some(p) = projector() else { eprintln!("skip: {MMPROJ} missing"); return; };
    let img = image::RgbImage::from_pixel(64, 64, image::Rgb([128, 128, 128]));
    let pre = p.preprocess(&img).unwrap();
    // 64x64 is below 8 image tokens, so smart resize grows it to 96x96 (3x3 merged tokens).
    assert_eq!((pre.width, pre.height), (96, 96));
    assert_eq!(p.count_tokens(pre.width, pre.height).unwrap(), (3, 3));
    let feats = p.encode(&pre).unwrap();
    assert_eq!(feats.len(), 9);
}

#[test]
fn qwen35_2b_red_image_predicts_red() {
    let Some(p) = projector() else { return };
    let Some(model_path) = text_model() else { return };
    let g = Gguf::open(&model_path).unwrap();
    let m = Model::load(&g).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let img = image::RgbImage::from_pixel(128, 128, image::Rgb([200, 0, 0]));
    let pre = p.preprocess(&img).unwrap();
    let (nx, ny) = p.count_tokens(pre.width, pre.height).unwrap();
    let feats = p.encode(&pre).unwrap();
    assert_eq!(feats.len(), nx * ny);
    let tokens = "<|vision_start|><|image_pad|><|vision_end|>";
    let prompt = "<|im_start|>user\n<image>\nWhat is the main color of this image? Answer with one word.<|im_end|>\n<|im_start|>assistant\n".replace("<image>", tokens);
    let ids = tok.encode(&prompt).unwrap();
    let (start, end, pad) = (
        tok.token_id("<|vision_start|>").unwrap(),
        tok.token_id("<|vision_end|>").unwrap(),
        tok.token_id("<|image_pad|>").unwrap(),
    );
    let p = expand_images(&ids, start, end, pad, &[(nx, ny)]).unwrap();
    let span = p.images[0].clone();
    let mut st = State::new(&m.cfg);
    let mut logits = vec![];
    for (i, &id) in p.ids.iter().enumerate() {
        let x = if i >= span.start && i < span.start + span.n_tokens() { feats[i - span.start].clone() } else { m.embed(id) };
        logits = m.step_embed(&mut st, x, p.positions[i], id);
    }
    st.rope_pos = [p.next_position; 4];
    let mut text = String::new();
    for _ in 0..12 {
        let next = ops::argmax(&logits) as u32;
        text.push_str(&tok.decode(&[next]).unwrap());
        if text.contains("red") { break; }
        logits = m.step(&mut st, next);
    }
    eprintln!("generated = {text:?}");
    assert!(text.contains("red"), "generated {text:?}");
}


#[test]
fn gemma4v_preprocess_min_tokens() {
    let Some(p) = gemma4_projector() else { return };
    let img = image::RgbImage::from_pixel(64, 64, image::Rgb([128, 128, 128]));
    let pre = p.preprocess(&img).unwrap();
    // 64x64 is below the 40-token minimum, so smart resize grows it to 336x336 (7x7 merged tokens).
    assert_eq!((pre.width, pre.height), (336, 336));
    assert_eq!(p.count_tokens(pre.width, pre.height).unwrap(), (7, 7));
    let feats = p.encode(&pre).unwrap();
    assert_eq!(feats.len(), 49);
}

#[test]
fn gemma4v_white_96() {
    let Some(p) = gemma4_projector() else { return };
    let c = p.cfg();
    assert_eq!(c.projector_type, "gemma4v");
    let img = image::RgbImage::from_pixel(96, 96, image::Rgb([255, 255, 255]));
    let pre = PreprocessedImage::from_u8(96, 96, img.as_raw(), c.image_mean, c.image_std).unwrap();
    let feats = p.encode(&pre).unwrap();
    assert_eq!(feats.len(), 4, "96x96 -> 6x6 patches -> 2x2 pooled tokens");
    assert!(feats.iter().all(|r| r.len() == c.projection_dim));
    let sum: f32 = feats.iter().flatten().sum();
    // llama-mtmd-debug -p encode -n 96 --image white: sum = 24.886238. The total differs by ~1 from
    // F16-weight accumulation order; the first/last values of every token agree to ~1e-2.
    assert!((sum - 24.886_238).abs() < 2.0, "sum {sum} vs llama.cpp 24.886238");
    for (row, (first, last)) in feats.iter().zip([
        ([-0.0648, -0.6445, 2.9617], [1.4735, -0.4516, 1.8618]),
        ([-0.5023, -1.2571, 1.2229], [-0.2033, 1.7081, 1.7383]),
        ([-0.2900, -0.9796, 1.2749], [0.1459, -0.2637, 1.2786]),
        ([0.2409, -1.7028, 2.1579], [-1.9630, 0.6113, -0.7936]),
    ]) {
        let n = row.len();
        for i in 0..3 {
            assert!((row[i] - first[i]).abs() < 0.02, "{row:?} vs {first:?}");
            assert!((row[n - 3 + i] - last[i]).abs() < 0.02, "{row:?} vs {last:?}");
        }
    }
}

#[test]
fn gemma4v_red_96() {
    let Some(p) = gemma4_projector() else { return };
    let c = p.cfg();
    let img = image::RgbImage::from_pixel(96, 96, image::Rgb([255, 0, 0]));
    let pre = PreprocessedImage::from_u8(96, 96, img.as_raw(), c.image_mean, c.image_std).unwrap();
    let feats = p.encode(&pre).unwrap();
    assert_eq!(feats.len(), 4);
    let sum: f32 = feats.iter().flatten().sum();
    // llama-mtmd-debug -p encode -n 96 --image red: sum = -97.841537; first row
    // [-1.3750, 0.8956, -0.3048, ..., 0.5834, -0.4703, -0.7513].
    assert!((sum - -97.841_537).abs() < 2.0, "sum {sum} vs llama.cpp -97.841537");
    for i in 0..3 {
        assert!((feats[0][i] - [-1.3750, 0.8956, -0.3048][i]).abs() < 0.05, "{:?}", feats[0]);
        assert!((feats[0][c.projection_dim - 3 + i] - [0.5834, -0.4703, -0.7513][i]).abs() < 0.05);
    }
}


#[test]
fn gemma4uv_white_96() {
    let Some(p) = gemma4uv_projector() else { return };
    let c = p.cfg();
    assert_eq!(c.projector_type, "gemma4uv");
    assert_eq!((c.n_embd, c.projection_dim, c.patch_size, c.n_merge, c.n_layer), (3840, 2048, 48, 1, 0));
    let img = image::RgbImage::from_pixel(96, 96, image::Rgb([255, 255, 255]));
    let pre = PreprocessedImage::from_u8(96, 96, img.as_raw(), c.image_mean, c.image_std).unwrap();
    let feats = p.encode(&pre).unwrap();
    assert_eq!(feats.len(), 4, "96x96 with a 48x48 patch -> 2x2 tokens");
    assert!(feats.iter().all(|r| r.len() == c.projection_dim));
    let sum: f32 = feats.iter().flatten().sum();
    // llama-mtmd-debug -p encode -n 96 --image white: sum = 258.351807.
    assert!((sum - 258.351_807).abs() < 2.0, "sum {sum} vs llama.cpp 258.351807");
    for (row, (first, last)) in feats.iter().zip([
        ([1.8761, 1.2244, 1.7886], [-0.5965, 2.5001, 0.6498]),
        ([0.0049, 1.4813, 1.0087], [-0.5007, 1.5330, 0.3371]),
        ([3.2730, 1.1496, -0.0031], [-1.7632, 0.9163, -0.0889]),
        ([1.4552, 1.4222, -0.7808], [-1.6934, -0.0340, -0.4016]),
    ]) {
        let n = row.len();
        for i in 0..3 {
            assert!((row[i] - first[i]).abs() < 0.02, "{row:?} vs {first:?}");
            assert!((row[n - 3 + i] - last[i]).abs() < 0.02, "{row:?} vs {last:?}");
        }
    }
}

#[test]
fn gemma4uv_red_96() {
    let Some(p) = gemma4uv_projector() else { return };
    let c = p.cfg();
    let img = image::RgbImage::from_pixel(96, 96, image::Rgb([255, 0, 0]));
    let pre = PreprocessedImage::from_u8(96, 96, img.as_raw(), c.image_mean, c.image_std).unwrap();
    let feats = p.encode(&pre).unwrap();
    assert_eq!(feats.len(), 4);
    let sum: f32 = feats.iter().flatten().sum();
    // llama-mtmd-debug -p encode -n 96 --image red: sum = -382.583984.
    assert!((sum - -382.583_984).abs() < 2.0, "sum {sum} vs llama.cpp -382.583984");
    for (row, (first, last)) in feats.iter().zip([
        ([0.5507, 1.0628, 2.1264], [0.8727, -2.9371, 1.0751]),
        ([0.4980, 1.0699, 2.1040], [0.8749, -2.9633, 1.0660]),
        ([0.5918, 1.0607, 2.0749], [0.8385, -2.9798, 1.0540]),
        ([0.5390, 1.0676, 2.0522], [0.8409, -3.0064, 1.0448]),
    ]) {
        let n = row.len();
        for i in 0..3 {
            assert!((row[i] - first[i]).abs() < 0.02, "{row:?} vs {first:?}");
            assert!((row[n - 3 + i] - last[i]).abs() < 0.02, "{row:?} vs {last:?}");
        }
    }
}

#[test]
fn gemma4uv_preprocess_min_tokens() {
    let Some(p) = gemma4uv_projector() else { return };
    let img = image::RgbImage::from_pixel(64, 64, image::Rgb([128, 128, 128]));
    let pre = p.preprocess(&img).unwrap();
    assert_eq!((pre.width, pre.height), (336, 336));
    assert_eq!(p.count_tokens(pre.width, pre.height).unwrap(), (7, 7));
    let feats = p.encode(&pre).unwrap();
    assert_eq!(feats.len(), 49);
}

#[test]
fn gemma4uv_image_forward() {
    let Some(p) = gemma4uv_projector() else { return };
    let path = std::env::var("STRATA_GEMMA4UV_MODEL").unwrap_or_else(|_| GEMMA4UV_MODEL.into());
    if !Path::new(&path).exists() { eprintln!("skip: {path} missing"); return; }
    let g = Gguf::open(Path::new(&path)).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let mut m = rusting_strata::gemma4::forward::Gemma4Model::new(&g).unwrap();
    let img = image::RgbImage::from_pixel(64, 64, image::Rgb([200, 0, 0]));
    let pre = p.preprocess(&img).unwrap();
    let (nx, ny) = p.count_tokens(pre.width, pre.height).unwrap();
    let feats = p.encode(&pre).unwrap();
    assert_eq!(feats.len(), nx * ny);
    let (begin, soft, end) = (
        tok.token_id("<|image>").unwrap(),
        tok.token_id("<|image|>").unwrap(),
        tok.token_id("<image|>").unwrap(),
    );
    let mut ids = vec![2u32];
    ids.extend(tok.encode("<|turn>user\n").unwrap());
    ids.extend([begin, soft, end]);
    ids.extend(tok.encode("What is this?<turn|>\n<|turn>model\n").unwrap());
    let mut seq: Vec<(u32, Option<&[f32]>)> = vec![];
    let mut row = 0;
    for &id in &ids {
        if id == soft {
            for _ in 0..nx * ny {
                seq.push((id, Some(&feats[row])));
                row += 1;
            }
        } else {
            seq.push((id, None));
        }
    }
    let (head, last) = seq.split_at(seq.len() - 1);
    for &(id, emb) in head {
        match emb {
            Some(e) => { m.step_embed_opt(id, e, false); }
            None => { m.step_opt_pub(id, false); }
        }
    }
    let (id, emb) = last[0];
    let logits = match emb {
        Some(e) => m.step_embed_opt(id, e, true),
        None => m.step_opt_pub(id, true),
    };
    assert_eq!(logits.len(), m.cfg.n_vocab);
    assert!(logits.iter().all(|v| v.is_finite()), "non-finite logits");
}

/// End-to-end Gemma 4 vision: only runs when `STRATA_GEMMA4_MODEL` points at the 26B text GGUF.
#[test]
fn gemma4v_red_image_predicts_red() {
    let Some(p) = gemma4_projector() else { return };
    let Some(model_path) = std::env::var("STRATA_GEMMA4_MODEL").ok().map(PathBuf::from).filter(|p| p.exists()) else {
        eprintln!("skip: STRATA_GEMMA4_MODEL missing"); return;
    };
    let g = Gguf::open(&model_path).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let mut m = rusting_strata::gemma4::forward::Gemma4Model::new(&g).unwrap();
    let img = image::RgbImage::from_pixel(128, 128, image::Rgb([200, 0, 0]));
    let pre = p.preprocess(&img).unwrap();
    let (nx, ny) = p.count_tokens(pre.width, pre.height).unwrap();
    let feats = p.encode(&pre).unwrap();
    assert_eq!(feats.len(), nx * ny);
    let (begin, soft, end) = (
        tok.token_id("<|image>").unwrap(),
        tok.token_id("<|image|>").unwrap(),
        tok.token_id("<image|>").unwrap(),
    );
    let mut ids = vec![2u32];
    ids.extend(tok.encode("<|turn>user\n").unwrap());
    ids.extend([begin, soft, end]);
    ids.extend(tok.encode("What is the main color of this image? Answer with one word.<turn|>\n<|turn>model\n").unwrap());
    let mut seq: Vec<(u32, Option<&[f32]>)> = vec![];
    let mut row = 0;
    for &id in &ids {
        if id == soft {
            for _ in 0..nx * ny {
                seq.push((id, Some(&feats[row])));
                row += 1;
            }
        } else {
            seq.push((id, None));
        }
    }
    let (head, last) = seq.split_at(seq.len() - 1);
    for &(id, emb) in head {
        match emb {
            Some(e) => { m.step_embed_opt(id, e, false); }
            None => { m.step_opt_pub(id, false); }
        }
    }
    let (id, emb) = last[0];
    let mut logits = match emb {
        Some(e) => m.step_embed_opt(id, e, true),
        None => m.step_opt_pub(id, true),
    };
    let mut text = String::new();
    for _ in 0..12 {
        let next = ops::argmax(&logits) as u32;
        text.push_str(&tok.decode(&[next]).unwrap());
        if text.to_ascii_lowercase().contains("red") { break; }
        logits = m.step(next);
    }
    eprintln!("generated = {text:?}");
    assert!(text.to_ascii_lowercase().contains("red"), "generated {text:?}");
}
