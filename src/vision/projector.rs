//! Vision encoders: Qwen3-VL (`qwen3vl_merger`), Gemma 4 (`gemma4v`, `gemma4uv`) and the GLM-4V placeholder.
//!
//! All encoders return one row of `projection_dim` features per merged image token, in row-major token
//! order. `preprocess`/`count_tokens` are shared: smart resize preserves the aspect ratio, aligns to the
//! projector's merge factor and normalizes with the mmproj's mean/std.

use crate::ops;
use crate::vision::image::{PreprocessedImage, Rgb, resize_bilinear, resize_bilinear_antialias_f32, resize_bilinear_f32, smart_resize};
use crate::vision::mmproj::{ClipConfig, ClipModel, FfnKind, ProjectorKind};
use crate::backend::{Backend, GpuTensor};
use crate::quant::GgmlType;
use crate::weights::Tensor;
use anyhow::{Context, bail};
use rayon::prelude::*;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

fn dbg() -> bool { std::env::var_os("STRATA_VISION_DEBUG").is_some() }

pub struct VisionProjector {
    pub model: ClipModel,
}

impl VisionProjector {
    pub fn open(path: &Path) -> anyhow::Result<VisionProjector> {
        Ok(VisionProjector { model: ClipModel::open(path)? })
    }

    pub fn cfg(&self) -> &ClipConfig { &self.model.cfg }

    /// Smart resize and normalize an RGB image for this projector.
    pub fn preprocess(&self, img: &Rgb) -> anyhow::Result<PreprocessedImage> {
        let c = &self.model.cfg;
        if c.kind == ProjectorKind::Inkling2V {
            let align = c.merge_align().max(1);
            let (mut sw, mut sh) = (img.width() as usize, img.height() as usize);
            let aw = sw.max(1).div_ceil(align) * align;
            let ah = sh.max(1).div_ceil(align) * align;
            let too_large = c.image_max_pixels > 0 && aw.saturating_mul(ah) > c.image_max_pixels;
            let resized;
            let src = if too_large {
                (sw, sh) = smart_resize(sw, sh, align, 0, c.image_max_pixels);
                resized = resize_bilinear(img, sw, sh);
                &resized
            } else {
                img
            };
            let width = sw.max(1).div_ceil(align) * align;
            let height = sh.max(1).div_ceil(align) * align;
            let mut rgb = vec![-1.0f32; width * height * 3];
            let raw = src.as_raw();
            for y in 0..sh {
                let s_row = &raw[y * sw * 3..(y + 1) * sw * 3];
                let d_row = &mut rgb[y * width * 3..y * width * 3 + sw * 3];
                for (d, &b) in d_row.iter_mut().zip(s_row) { *d = b as f32; }
            }
            let mean = [c.image_mean[0] * 255.0, c.image_mean[1] * 255.0, c.image_mean[2] * 255.0];
            let std = [c.image_std[0] * 255.0, c.image_std[1] * 255.0, c.image_std[2] * 255.0];
            for (i, p) in rgb.iter_mut().enumerate() {
                let ch = i % 3;
                *p = (*p - mean[ch]) / std[ch];
            }
            return Ok(PreprocessedImage { width, height, rgb });
        }
        let (w, h) = smart_resize(img.width() as usize, img.height() as usize, c.merge_align(), c.image_min_pixels, c.image_max_pixels);
        let resized = resize_bilinear(img, w, h);
        PreprocessedImage::from_u8(w, h, resized.as_raw(), c.image_mean, c.image_std)
    }

    /// Number of merged image tokens and the token grid (`nx, ny`), without running the encoder.
    pub fn count_tokens(&self, width: usize, height: usize) -> anyhow::Result<(usize, usize)> {
        let c = &self.model.cfg;
        let align = c.merge_align().max(1);
        let (w, h) = if c.kind == ProjectorKind::Inkling2V {
            let aw = width.max(1).div_ceil(align) * align;
            let ah = height.max(1).div_ceil(align) * align;
            if c.image_max_pixels > 0 && aw.saturating_mul(ah) > c.image_max_pixels {
                smart_resize(width, height, align, 0, c.image_max_pixels)
            } else {
                (aw, ah)
            }
        } else {
            smart_resize(width, height, align, c.image_min_pixels, c.image_max_pixels)
        };
        let nx = w / align;
        let ny = h / align;
        anyhow::ensure!(nx > 0 && ny > 0, "image resized to {w}x{h}, too small for patch {align}");
        Ok((nx, ny))
    }

    /// Run the encoder. Returns `nx * ny` rows of `projection_dim` features in row-major token order.
    pub fn encode(&self, img: &PreprocessedImage) -> anyhow::Result<Vec<Vec<f32>>> {
        match self.model.cfg.kind {
            ProjectorKind::Qwen3Vl => encode_qwen3vl(self, img),
            ProjectorKind::Gemma4V => encode_gemma4v(self, img),
            ProjectorKind::Gemma4Uv => encode_gemma4uv(self, img),
            ProjectorKind::Inkling2V => encode_inkling2v(self, img),
            ProjectorKind::Glm4V => bail!("GLM-4V projectors are not implemented yet"),
        }
    }
}

fn merge_shape(c: &ClipConfig, img: &PreprocessedImage) -> anyhow::Result<(usize, usize, usize, usize)> {
    let align = c.patch_size * c.n_merge;
    anyhow::ensure!(img.width % align == 0 && img.height % align == 0,
        "image {}x{} is not aligned to {}", img.width, img.height, align);
    let nx = img.width / c.patch_size;
    let ny = img.height / c.patch_size;
    let (mx, my) = if c.n_merge > 1 { (nx / c.n_merge, ny / c.n_merge) } else { (nx, ny) };
    anyhow::ensure!(mx > 0 && my > 0, "image {}x{} is too small for the {}x{} merge", img.width, img.height, c.n_merge, c.n_merge);
    Ok((nx, ny, mx, my))
}

/// Qwen3-VL: two summed patch convs, resized learned position embedding, 2x2 patch reorder before the ViT,
/// then the `mm.0`/`mm.2` merger.
fn encode_qwen3vl(p: &VisionProjector, img: &PreprocessedImage) -> anyhow::Result<Vec<Vec<f32>>> {
    let c = &p.model.cfg;
    anyhow::ensure!(c.n_merge > 0, "clip: spatial_merge_size is 0");
    let m = c.n_merge;
    let (n_patches_x, n_patches_y, mx, my) = merge_shape(c, img)?;
    let n_patches = n_patches_x * n_patches_y;
    let n_merged = mx * my;
    let n_embd = c.n_embd;
    let patch_area = c.patch_size * c.patch_size * 3;

    // 1. patch convolution. `v.patch_embd.weight` is [kx, ky, c, out]; flatten it to [out][kx,ky,c].
    let w0 = rows_f32(&p.model.tensor_as("v.patch_embd.weight", Some(patch_area))?);
    let w1 = rows_f32(&p.model.tensor_as("v.patch_embd.weight.1", Some(patch_area))?);
    let bias = p.model.tensor("v.patch_embd.bias")?;
    let mut patch_bias = vec![0f32; n_embd];
    bias.row(0, &mut patch_bias);
    let mut x = vec![0f32; n_patches * n_embd];
    x.par_chunks_mut(n_embd).enumerate().for_each(|(q, xs)| {
        let px = q % n_patches_x;
        let py = q / n_patches_x;
        let mut patch = vec![0f32; patch_area];
        for ky in 0..c.patch_size {
            for kx in 0..c.patch_size {
                for ch in 0..3 {
                    let idx = ((py * c.patch_size + ky) * img.width + px * c.patch_size + kx) * 3 + ch;
                    patch[kx + c.patch_size * ky + c.patch_size * c.patch_size * ch] = img.rgb[idx];
                }
            }
        }
        for (o, (r0, r1)) in w0.chunks_exact(patch_area).zip(w1.chunks_exact(patch_area)).enumerate() {
            let mut v = r0.iter().zip(&patch).map(|(a, b)| a * b).sum::<f32>();
            v += r1.iter().zip(&patch).map(|(a, b)| a * b).sum::<f32>();
            xs[o] = v + patch_bias[o];
        }
    });

    // 2. learned 2-D position embedding, resized from 48x48 to the actual patch grid.
    let pos = p.model.to_f32("v.position_embd.weight")?;
    let n_per_side = ((pos.len() / n_embd) as f64).sqrt().round() as usize;
    anyhow::ensure!(n_per_side * n_per_side * n_embd == pos.len(), "position_embd is not square");
    let pos = resize_bilinear_antialias_f32(&pos, n_embd, n_per_side, n_per_side, n_patches_x, n_patches_y);
    x.par_chunks_mut(n_embd).enumerate().for_each(|(q, xs)| {
        for (o, v) in xs.iter_mut().enumerate() { *v += pos[o + n_embd * q]; }
    });

    // 2b. Reorder patches into 2x2 blocks before the transformer. llama.cpp's spatial merge does this
    // before attention, so the 4 patches of a block are consecutive tokens: (x0,y0), (x1,y0), (x0,y1),
    // (x1,y1). Blocks are row-major. This keeps the numerical accumulation order identical.
    let reordered: Vec<f32> = (0..n_patches).into_par_iter().flat_map_iter(|q| {
        let dx = q % m;
        let dy = (q / m) % m;
        let b = q / (m * m);
        let (w2, h2) = (b % mx, b / mx);
        let q = (h2 * m + dy) * n_patches_x + w2 * m + dx;
        x[q * n_embd..(q + 1) * n_embd].iter().copied()
    }).collect();
    let mut x = reordered;

    if std::env::var("STRATA_VISION_DEBUG").is_ok() {
        let sum: f32 = x.iter().sum();
        eprintln!("inp_pos_emb sum = {sum}");
        eprintln!("inp_pos_emb row0 first3 = {:?} last3 = {:?}", &x[..3], &x[n_embd - 3..n_embd]);
    }

    // 3. pre-layernorm (optional in qwen3vl).
    if p.model.has("v.pre_ln.weight") {
        let w = p.model.to_f32("v.pre_ln.weight")?;
        let b = p.model.to_f32("v.pre_ln.bias")?;
        x = layer_norm_all(&x, n_embd, &w, &b, c.eps);
    }

    // 4. transformer blocks.
    let base = 10000.0f32;
    let scale = 1.0 / (c.d_head as f32).sqrt();
    for il in 0..c.n_layer {
        let ln1w = p.model.to_f32(&format!("v.blk.{il}.ln1.weight"))?;
        let ln1b = p.model.to_f32(&format!("v.blk.{il}.ln1.bias"))?;
        let ln2w = p.model.to_f32(&format!("v.blk.{il}.ln2.weight"))?;
        let ln2b = p.model.to_f32(&format!("v.blk.{il}.ln2.bias"))?;

        let h = layer_norm_all(&x, n_embd, &ln1w, &ln1b, c.eps);
        let qkv_w = p.model.tensor(&format!("v.blk.{il}.attn_qkv.weight"))?;
        let qkv_b = p.model.tensor(&format!("v.blk.{il}.attn_qkv.bias"))?;
        let mut qkv = vec![0f32; n_patches * 3 * n_embd];
        matmul(&qkv_w, &h, n_patches, &mut qkv);
        add_bias_rows(&mut qkv, &qkv_b, 3 * n_embd);

        let mut q = vec![0f32; n_patches * n_embd];
        let mut k = vec![0f32; n_patches * n_embd];
        let mut v = vec![0f32; n_patches * n_embd];
        for t in 0..n_patches {
            let row = &qkv[t * 3 * n_embd..(t + 1) * 3 * n_embd];
            q[t * n_embd..(t + 1) * n_embd].copy_from_slice(&row[..n_embd]);
            k[t * n_embd..(t + 1) * n_embd].copy_from_slice(&row[n_embd..2 * n_embd]);
            v[t * n_embd..(t + 1) * n_embd].copy_from_slice(&row[2 * n_embd..]);
        }
        q.par_chunks_mut(n_embd).enumerate().for_each(|(t, row)| {
            let dx = t % m;
            let dy = (t / m) % m;
            let b = t / (m * m);
            let (w2, h2) = (b % mx, b / mx);
            let (px, py) = ((w2 * m + dx) as i32, (h2 * m + dy) as i32);
            for h in row.chunks_exact_mut(c.d_head) { ops::rope_vision(h, py, px, base); }
        });
        k.par_chunks_mut(n_embd).enumerate().for_each(|(t, row)| {
            let dx = t % m;
            let dy = (t / m) % m;
            let b = t / (m * m);
            let (w2, h2) = (b % mx, b / mx);
            let (px, py) = ((w2 * m + dx) as i32, (h2 * m + dy) as i32);
            for h in row.chunks_exact_mut(c.d_head) { ops::rope_vision(h, py, px, base); }
        });

        let attn = attention(&q, &k, &v, n_patches, c.n_head, c.d_head, scale);
        let wo_w = p.model.tensor(&format!("v.blk.{il}.attn_out.weight"))?;
        let wo_b = p.model.tensor(&format!("v.blk.{il}.attn_out.bias"))?;
        let mut attn_out = vec![0f32; n_patches * n_embd];
        matmul(&wo_w, &attn, n_patches, &mut attn_out);
        add_bias_rows(&mut attn_out, &wo_b, n_embd);
        for (a, b) in x.iter_mut().zip(&attn_out) { *a += *b; }

        let h = layer_norm_all(&x, n_embd, &ln2w, &ln2b, c.eps);
        let up_w = p.model.tensor(&format!("v.blk.{il}.ffn_up.weight"))?;
        let up_b = p.model.tensor(&format!("v.blk.{il}.ffn_up.bias"))?;
        let mut up = vec![0f32; n_patches * c.n_ff];
        matmul(&up_w, &h, n_patches, &mut up);
        add_bias_rows(&mut up, &up_b, c.n_ff);
        up.par_iter_mut().for_each(|v| *v = activate(c.ffn, *v));
        let down_w = p.model.tensor(&format!("v.blk.{il}.ffn_down.weight"))?;
        let down_b = p.model.tensor(&format!("v.blk.{il}.ffn_down.bias"))?;
        let mut down = vec![0f32; n_patches * n_embd];
        matmul(&down_w, &up, n_patches, &mut down);
        add_bias_rows(&mut down, &down_b, n_embd);
        for (a, b) in x.iter_mut().zip(&down) { *a += *b; }
    }

    // 5. post-layernorm.
    let post_w = p.model.to_f32("v.post_ln.weight")?;
    let post_b = p.model.to_f32("v.post_ln.bias")?;
    let x = layer_norm_all(&x, n_embd, &post_w, &post_b, c.eps);

    // 6. The patch sequence is already in 2x2 block order, so the merger just reinterprets four
    // consecutive n_embd rows as one 4*n_embd token.
    let feat = n_embd * m * m;
    anyhow::ensure!(x.len() == n_merged * feat, "merge shape mismatch");

    // 7. merger: mm.0 (4*n_embd -> 4*n_embd) + GELU + mm.2 (4*n_embd -> projection_dim).
    let mm0 = p.model.tensor("mm.0.weight")?;
    let mm0_b = p.model.tensor("mm.0.bias")?;
    let mut proj = vec![0f32; n_merged * feat];
    matmul(&mm0, &x, n_merged, &mut proj);
    add_bias_rows(&mut proj, &mm0_b, feat);
    proj.par_iter_mut().for_each(|v| *v = activate(c.ffn, *v));
    let mm2 = p.model.tensor("mm.2.weight")?;
    let mm2_b = p.model.tensor("mm.2.bias")?;
    let mut out = vec![0f32; n_merged * c.projection_dim];
    matmul(&mm2, &proj, n_merged, &mut out);
    add_bias_rows(&mut out, &mm2_b, c.projection_dim);

    rows(out, n_merged, c.projection_dim)
}

/// Gemma 4 vision tower: `2*x-1` patches, x/y learned position tables, RMS ViT with split NEOX RoPE,
/// 3x3 average pooling, std bias/scale, RMS norm and `mm.input_projection`.
fn encode_gemma4v(p: &VisionProjector, img: &PreprocessedImage) -> anyhow::Result<Vec<Vec<f32>>> {
    let c = &p.model.cfg;
    let (nx, ny, mx, my) = merge_shape(c, img)?;
    let n_patches = nx * ny;
    let n_embd = c.n_embd;
    let patch_area = c.patch_size * c.patch_size * 3;
    let w = rows_f32(&p.model.tensor_as("v.patch_embd.weight", Some(patch_area))?);
    let mut x = vec![0f32; n_patches * n_embd];
    x.par_chunks_mut(n_embd).enumerate().for_each(|(q, xs)| {
        let px = q % nx;
        let py = q / nx;
        let mut patch = vec![0f32; patch_area];
        for ky in 0..c.patch_size {
            for kx in 0..c.patch_size {
                for ch in 0..3 {
                    let idx = ((py * c.patch_size + ky) * img.width + px * c.patch_size + kx) * 3 + ch;
                    patch[kx + c.patch_size * ky + c.patch_size * c.patch_size * ch] = 2.0 * img.rgb[idx] - 1.0;
                }
            }
        }
        for (o, r) in w.chunks_exact(patch_area).enumerate() {
            xs[o] = r.iter().zip(&patch).map(|(a, b)| a * b).sum();
        }
    });

    if dbg() { eprintln!("g4 inp sum = {} row0 = {:?}", x.iter().sum::<f32>(), &x[..3]); }
    let pos = p.model.to_f32("v.position_embd.weight")?;
    anyhow::ensure!(pos.len() % (2 * n_embd) == 0, "position_embd size {} is not 2*n_embd*n_pos", pos.len());
    let pos_size = pos.len() / (2 * n_embd);
    x.par_chunks_mut(n_embd).enumerate().for_each(|(q, xs)| {
        let px = q % nx;
        let py = q / nx;
        for (o, v) in xs.iter_mut().enumerate() {
            *v += pos[o + n_embd * px] + pos[o + n_embd * (pos_size + py)];
        }
    });
    if dbg() { eprintln!("g4 pos sum = {} row0 = {:?}", x.iter().sum::<f32>(), &x[..3]); }

    let half = c.d_head / 2;
    for il in 0..c.n_layer {
        let ln1 = p.model.to_f32(&format!("v.blk.{il}.ln1.weight"))?;
        let h = rms_norm_all(&x, n_embd, &ln1, c.eps);
        if dbg() && il == 0 { eprintln!("g4 ln1 sum = {} row0 = {:?}", h.iter().sum::<f32>(), &h[..3]); }
        let q_w = p.model.tensor(&format!("v.blk.{il}.attn_q.weight"))?;
        let k_w = p.model.tensor(&format!("v.blk.{il}.attn_k.weight"))?;
        let v_w = p.model.tensor(&format!("v.blk.{il}.attn_v.weight"))?;
        let (mut q, mut k, mut v) = (vec![0f32; n_patches * n_embd], vec![0f32; n_patches * n_embd], vec![0f32; n_patches * n_embd]);
        matmul(&q_w, &h, n_patches, &mut q);
        matmul(&k_w, &h, n_patches, &mut k);
        matmul(&v_w, &h, n_patches, &mut v);

        let q_norm = p.model.to_f32(&format!("v.blk.{il}.attn_q_norm.weight"))?;
        let k_norm = p.model.to_f32(&format!("v.blk.{il}.attn_k_norm.weight"))?;
        q.par_chunks_mut(n_embd).enumerate().for_each(|(t, row)| {
            let px = t % nx;
            let py = t / nx;
            for head in row.chunks_exact_mut(c.d_head) { rms_norm_nw(head, &q_norm, c.eps); }
            for head in row.chunks_exact_mut(c.d_head) {
                ops::rope_neox(&mut head[..half], half, px, c.rope_theta);
                ops::rope_neox(&mut head[half..], half, py, c.rope_theta);
            }
        });
        k.par_chunks_mut(n_embd).enumerate().for_each(|(t, row)| {
            let px = t % nx;
            let py = t / nx;
            for head in row.chunks_exact_mut(c.d_head) { rms_norm_nw(head, &k_norm, c.eps); }
            for head in row.chunks_exact_mut(c.d_head) {
                ops::rope_neox(&mut head[..half], half, px, c.rope_theta);
                ops::rope_neox(&mut head[half..], half, py, c.rope_theta);
            }
        });
        v.par_chunks_mut(n_embd).for_each(|row| {
            for head in row.chunks_exact_mut(c.d_head) { rms_norm_nw(head, &[], c.eps); }
        });

        let attn = attention(&q, &k, &v, n_patches, c.n_head, c.d_head, 1.0);
        let o_w = p.model.tensor(&format!("v.blk.{il}.attn_out.weight"))?;
        let mut attn_out = vec![0f32; n_patches * n_embd];
        matmul(&o_w, &attn, n_patches, &mut attn_out);
        let apn = p.model.to_f32(&format!("v.blk.{il}.attn_post_norm.weight"))?;
        let attn_out = rms_norm_all(&attn_out, n_embd, &apn, c.eps);
        if dbg() && il == 0 { eprintln!("g4 attn_post sum = {} row0 = {:?}", attn_out.iter().sum::<f32>(), &attn_out[..3]); }
        for (a, b) in x.iter_mut().zip(&attn_out) { *a += *b; }

        let ln2 = p.model.to_f32(&format!("v.blk.{il}.ln2.weight"))?;
        let h = rms_norm_all(&x, n_embd, &ln2, c.eps);
        let gate_w = p.model.tensor(&format!("v.blk.{il}.ffn_gate.weight"))?;
        let up_w = p.model.tensor(&format!("v.blk.{il}.ffn_up.weight"))?;
        let down_w = p.model.tensor(&format!("v.blk.{il}.ffn_down.weight"))?;
        let (mut gate, mut up, mut down) = (vec![0f32; n_patches * c.n_ff], vec![0f32; n_patches * c.n_ff], vec![0f32; n_patches * n_embd]);
        matmul(&gate_w, &h, n_patches, &mut gate);
        matmul(&up_w, &h, n_patches, &mut up);
        for (u, g) in up.iter_mut().zip(&gate) { *u *= activate(c.ffn, *g); }
        matmul(&down_w, &up, n_patches, &mut down);
        let fpn = p.model.to_f32(&format!("v.blk.{il}.ffn_post_norm.weight"))?;
        let down = rms_norm_all(&down, n_embd, &fpn, c.eps);
        for (a, b) in x.iter_mut().zip(&down) { *a += *b; }
        if dbg() && il == 0 { eprintln!("g4 layer0 sum = {} row0 = {:?}", x.iter().sum::<f32>(), &x[..3]); }
    }
    if dbg() { eprintln!("g4 vit sum = {} row0 = {:?}", x.iter().sum::<f32>(), &x[..3]); }

    // 3x3 non-overlapping average pool, scaled by sqrt(n_embd).
    let (ox, oy) = (mx, my);
    let mut pooled = vec![0f32; ox * oy * n_embd];
    let scale = (n_embd as f32).sqrt();
    pooled.par_chunks_mut(n_embd).enumerate().for_each(|(t, out)| {
        let bx = t % ox;
        let by = t / ox;
        for (j, o) in out.iter_mut().enumerate() {
            let mut s = 0f32;
            for dy in 0..c.n_merge {
                for dx in 0..c.n_merge {
                    let q = (by * c.n_merge + dy) * nx + bx * c.n_merge + dx;
                    s += x[q * n_embd + j];
                }
            }
            *o = s / (c.n_merge * c.n_merge) as f32 * scale;
        }
    });
    if dbg() { eprintln!("g4 pooled sum = {} row0 = {:?}", pooled.iter().sum::<f32>(), &pooled[..3]); }
    let mut y = pooled;
    if p.model.has("v.std_bias") && p.model.has("v.std_scale") {
        let b = p.model.to_f32("v.std_bias")?;
        let s = p.model.to_f32("v.std_scale")?;
        y.par_chunks_mut(n_embd).for_each(|row| for j in 0..n_embd { row[j] = (row[j] - b[j]) * s[j]; });
    }
    if dbg() { eprintln!("g4 std sum = {} row0 = {:?}", y.iter().sum::<f32>(), &y[..3]); }
    let y = rms_norm_nw_all(&y, n_embd, c.eps);
    if dbg() { eprintln!("g4 preproj sum = {} row0 = {:?}", y.iter().sum::<f32>(), &y[..3]); }
    let proj = p.model.tensor("mm.input_projection.weight")?;
    let mut out = vec![0f32; ox * oy * c.projection_dim];
    matmul(&proj, &y, ox * oy, &mut out);
    if dbg() {
        eprintln!("g4 proj sum = {} row0 = {:?}", out.iter().sum::<f32>(), &out[..3]);
        for t in 0..(ox * oy) {
            let r = &out[t * c.projection_dim..(t + 1) * c.projection_dim];
            eprintln!("g4 row{t} first3 = {:?} last3 = {:?}", &r[..3], &r[c.projection_dim - 3..]);
        }
    }
    rows(out, ox * oy, c.projection_dim)
}

/// Gemma 4 unified vision embedder: im2col patches, two LayerNorms, x/y position tables, a third
/// LayerNorm, RMS norm and `mm.input_projection`. No ViT blocks and no pooling.
fn encode_gemma4uv(p: &VisionProjector, img: &PreprocessedImage) -> anyhow::Result<Vec<Vec<f32>>> {
    let c = &p.model.cfg;
    let (nx, ny, _, _) = merge_shape(c, img)?;
    let n_patches = nx * ny;
    let n_embd = c.n_embd;
    let patch_area = c.patch_size * c.patch_size * 3;
    let n1w = p.model.to_f32("v.patch_norm.1.weight")?;
    let n1b = p.model.to_f32("v.patch_norm.1.bias")?;
    let n2w = p.model.to_f32("v.patch_norm.2.weight")?;
    let n2b = p.model.to_f32("v.patch_norm.2.bias")?;
    let n3w = p.model.to_f32("v.patch_norm.3.weight")?;
    let n3b = p.model.to_f32("v.patch_norm.3.bias")?;
    let w = rows_f32(&p.model.tensor_as("v.patch_embd.weight", Some(patch_area))?);
    let bias = p.model.tensor("v.patch_embd.bias")?;
    let mut patch_bias = vec![0f32; n_embd];
    bias.row(0, &mut patch_bias);
    let eps = 1e-5f32;
    let mut x = vec![0f32; n_patches * n_embd];
    x.par_chunks_mut(n_embd).enumerate().for_each(|(q, xs)| {
        let px = q % nx;
        let py = q / nx;
        let mut patch = vec![0f32; patch_area];
        for ky in 0..c.patch_size {
            for kx in 0..c.patch_size {
                for ch in 0..3 {
                    let idx = ((py * c.patch_size + ky) * img.width + px * c.patch_size + kx) * 3 + ch;
                    patch[kx + c.patch_size * ky + c.patch_size * c.patch_size * ch] = img.rgb[idx];
                }
            }
        }
        let mut normed = vec![0f32; patch_area];
        ops::layer_norm(&patch, &n1w, &n1b, eps, &mut normed);
        let mut emb = vec![0f32; n_embd];
        for (o, r) in w.chunks_exact(patch_area).enumerate() {
            emb[o] = r.iter().zip(&normed).map(|(a, b)| a * b).sum::<f32>() + patch_bias[o];
        }
        ops::layer_norm(&emb, &n2w, &n2b, eps, xs);
    });

    let pos = p.model.to_f32("v.position_embd.weight")?;
    anyhow::ensure!(pos.len() % (2 * n_embd) == 0, "position_embd size {} is not 2*n_embd*n_pos", pos.len());
    let pos_size = pos.len() / (2 * n_embd);
    x.par_chunks_mut(n_embd).enumerate().for_each(|(q, xs)| {
        let px = q % nx;
        let py = q / nx;
        for (o, v) in xs.iter_mut().enumerate() {
            *v += pos[o + n_embd * px] + pos[o + n_embd * (pos_size + py)];
        }
    });
    let x = layer_norm_all(&x, n_embd, &n3w, &n3b, eps);
    let x = rms_norm_nw_all(&x, n_embd, c.eps);
    let proj = p.model.tensor("mm.input_projection.weight")?;
    let mut out = vec![0f32; n_patches * c.projection_dim];
    matmul(&proj, &x, n_patches, &mut out);
    rows(out, n_patches, c.projection_dim)
}

/// Inkling 2 vision tower (`inkling2v`): patch conv + bilinear-resized learned position embeddings,
/// prepended CLS token (`v.class_embd + v.class_position_embd.weight`) at index 0, `v.pre_ln`,
/// 23 LayerNorm ViT blocks with 1-indexed 2D RoPE and LayerScale (`ls1`, `ls2`), CLS drop,
/// 2x2 `build_patch_merge_permute`, and `mm.1` -> `GeluErf` -> `mm.2` projector.
fn encode_inkling2v(p: &VisionProjector, img: &PreprocessedImage) -> anyhow::Result<Vec<Vec<f32>>> {
    let c = &p.model.cfg;
    anyhow::ensure!(c.n_merge > 1, "inkling2v: spatial_merge_size must be > 1");
    let m = c.n_merge;
    let (nx, ny, mx, my) = merge_shape(c, img)?;
    let n_patches = nx * ny;
    let n_pos = n_patches + 1;
    let n_merged = mx * my;
    let n_embd = c.n_embd;
    let patch_area = c.patch_size * c.patch_size * 3;

    // 1. Patch convolution (`v.patch_embd.weight`, optional `v.patch_embd.bias`).
    let w0 = rows_f32(&p.model.tensor_as("v.patch_embd.weight", Some(patch_area))?);
    let patch_bias = if p.model.has("v.patch_embd.bias") {
        p.model.to_f32("v.patch_embd.bias")?
    } else {
        vec![0f32; n_embd]
    };
    let mut patches = vec![0f32; n_patches * n_embd];
    patches.par_chunks_mut(n_embd).enumerate().for_each(|(q, xs)| {
        let px = q % nx;
        let py = q / nx;
        let mut patch = vec![0f32; patch_area];
        for ky in 0..c.patch_size {
            for kx in 0..c.patch_size {
                for ch in 0..3 {
                    let idx = ((py * c.patch_size + ky) * img.width + px * c.patch_size + kx) * 3 + ch;
                    patch[kx + c.patch_size * ky + c.patch_size * c.patch_size * ch] = img.rgb[idx];
                }
            }
        }
        for (o, r) in w0.chunks_exact(patch_area).enumerate() {
            xs[o] = r.iter().zip(&patch).map(|(a, b)| a * b).sum::<f32>() + patch_bias[o];
        }
    });

    // 2. Bilinear-resized learned position embeddings (`GGML_SCALE_MODE_BILINEAR`).
    let pos = p.model.to_f32("v.position_embd.weight")?;
    let n_per_side = ((pos.len() / n_embd) as f64).sqrt().round() as usize;
    anyhow::ensure!(n_per_side * n_per_side * n_embd == pos.len(), "position_embd is not square");
    let pos = resize_bilinear_f32(&pos, n_embd, n_per_side, n_per_side, nx, ny);
    patches.par_chunks_mut(n_embd).enumerate().for_each(|(q, xs)| {
        for (o, v) in xs.iter_mut().enumerate() { *v += pos[o + n_embd * q]; }
    });

    // 3. Prepend CLS token (`v.class_embd + v.class_position_embd.weight`) at index 0.
    let cls_emb = p.model.to_f32("v.class_embd")?;
    let cls_pos = p.model.to_f32("v.class_position_embd.weight")?;
    anyhow::ensure!(cls_emb.len() == n_embd && cls_pos.len() == n_embd, "inkling2v: invalid class_embd length");
    let mut x = Vec::with_capacity(n_pos * n_embd);
    for i in 0..n_embd { x.push(cls_emb[i] + cls_pos[i]); }
    x.extend_from_slice(&patches);

    // 4. Pre-LayerNorm (`v.pre_ln`).
    let pre_w = p.model.to_f32("v.pre_ln.weight")?;
    let pre_b = p.model.to_f32("v.pre_ln.bias")?;
    x = layer_norm_all(&x, n_embd, &pre_w, &pre_b, c.eps);

    // 5. ViT transformer blocks with 2D RoPE (CLS at (0, 0), patches 1-indexed) and LayerScale.
    let scale = 1.0 / (c.d_head as f32).sqrt();
    let half = c.d_head / 2;
    let pairs = half / 2;
    let apply_rope_2d = |buf: &mut [f32]| {
        buf.par_chunks_mut(n_embd).enumerate().for_each(|(t, row)| {
            if t == 0 { return; }
            let q = t - 1;
            let pw = ((q % nx) + 1) as f32;
            let ph = ((q / nx) + 1) as f32;
            for head in row.chunks_exact_mut(c.d_head) {
                for i in 0..pairs {
                    let freq = c.rope_theta.powf(-2.0 * i as f32 / half as f32);
                    let (sw, cw) = (pw * freq).sin_cos();
                    let (x0, x1) = (head[2 * i], head[2 * i + 1]);
                    head[2 * i] = x0 * cw - x1 * sw;
                    head[2 * i + 1] = x0 * sw + x1 * cw;
                    let (sh, ch) = (ph * freq).sin_cos();
                    let (y0, y1) = (head[half + 2 * i], head[half + 2 * i + 1]);
                    head[half + 2 * i] = y0 * ch - y1 * sh;
                    head[half + 2 * i + 1] = y0 * sh + y1 * ch;
                }
            }
        });
    };

    for il in 0..c.n_layer {
        let ln1w = p.model.to_f32(&format!("v.blk.{il}.ln1.weight"))?;
        let ln1b = p.model.to_f32(&format!("v.blk.{il}.ln1.bias"))?;
        let h = layer_norm_all(&x, n_embd, &ln1w, &ln1b, c.eps);

        let (mut q, mut k, mut v) = (vec![0f32; n_pos * n_embd], vec![0f32; n_pos * n_embd], vec![0f32; n_pos * n_embd]);
        if p.model.has(&format!("v.blk.{il}.attn_qkv.weight")) {
            let qkv_w = p.model.tensor(&format!("v.blk.{il}.attn_qkv.weight"))?;
            let mut qkv = vec![0f32; n_pos * 3 * n_embd];
            matmul(&qkv_w, &h, n_pos, &mut qkv);
            if p.model.has(&format!("v.blk.{il}.attn_qkv.bias")) {
                let qkv_b = p.model.tensor(&format!("v.blk.{il}.attn_qkv.bias"))?;
                add_bias_rows(&mut qkv, &qkv_b, 3 * n_embd);
            }
            for t in 0..n_pos {
                let row = &qkv[t * 3 * n_embd..(t + 1) * 3 * n_embd];
                q[t * n_embd..(t + 1) * n_embd].copy_from_slice(&row[..n_embd]);
                k[t * n_embd..(t + 1) * n_embd].copy_from_slice(&row[n_embd..2 * n_embd]);
                v[t * n_embd..(t + 1) * n_embd].copy_from_slice(&row[2 * n_embd..]);
            }
        } else {
            let q_w = p.model.tensor(&format!("v.blk.{il}.attn_q.weight"))?;
            let k_w = p.model.tensor(&format!("v.blk.{il}.attn_k.weight"))?;
            let v_w = p.model.tensor(&format!("v.blk.{il}.attn_v.weight"))?;
            matmul(&q_w, &h, n_pos, &mut q);
            matmul(&k_w, &h, n_pos, &mut k);
            matmul(&v_w, &h, n_pos, &mut v);
            if p.model.has(&format!("v.blk.{il}.attn_q.bias")) {
                add_bias_rows(&mut q, &p.model.tensor(&format!("v.blk.{il}.attn_q.bias"))?, n_embd);
            }
            if p.model.has(&format!("v.blk.{il}.attn_k.bias")) {
                add_bias_rows(&mut k, &p.model.tensor(&format!("v.blk.{il}.attn_k.bias"))?, n_embd);
            }
            if p.model.has(&format!("v.blk.{il}.attn_v.bias")) {
                add_bias_rows(&mut v, &p.model.tensor(&format!("v.blk.{il}.attn_v.bias"))?, n_embd);
            }
        }

        apply_rope_2d(&mut q);
        apply_rope_2d(&mut k);

        let attn = attention(&q, &k, &v, n_pos, c.n_head, c.d_head, scale);
        let wo_w = p.model.tensor(&format!("v.blk.{il}.attn_out.weight"))?;
        let mut attn_out = vec![0f32; n_pos * n_embd];
        matmul(&wo_w, &attn, n_pos, &mut attn_out);
        if p.model.has(&format!("v.blk.{il}.attn_out.bias")) {
            add_bias_rows(&mut attn_out, &p.model.tensor(&format!("v.blk.{il}.attn_out.bias"))?, n_embd);
        }
        if p.model.has(&format!("v.blk.{il}.ls1.weight")) {
            let ls1 = p.model.to_f32(&format!("v.blk.{il}.ls1.weight"))?;
            attn_out.par_chunks_mut(n_embd).for_each(|row| {
                for (a, &s) in row.iter_mut().zip(&ls1) { *a *= s; }
            });
        }
        for (a, b) in x.iter_mut().zip(&attn_out) { *a += *b; }

        let ln2w = p.model.to_f32(&format!("v.blk.{il}.ln2.weight"))?;
        let ln2b = p.model.to_f32(&format!("v.blk.{il}.ln2.bias"))?;
        let h = layer_norm_all(&x, n_embd, &ln2w, &ln2b, c.eps);
        let up_w = p.model.tensor(&format!("v.blk.{il}.ffn_up.weight"))?;
        let mut up = vec![0f32; n_pos * c.n_ff];
        matmul(&up_w, &h, n_pos, &mut up);
        if p.model.has(&format!("v.blk.{il}.ffn_up.bias")) {
            add_bias_rows(&mut up, &p.model.tensor(&format!("v.blk.{il}.ffn_up.bias"))?, c.n_ff);
        }
        up.par_iter_mut().for_each(|v| *v = activate(c.ffn, *v));
        let down_w = p.model.tensor(&format!("v.blk.{il}.ffn_down.weight"))?;
        let mut down = vec![0f32; n_pos * n_embd];
        matmul(&down_w, &up, n_pos, &mut down);
        if p.model.has(&format!("v.blk.{il}.ffn_down.bias")) {
            add_bias_rows(&mut down, &p.model.tensor(&format!("v.blk.{il}.ffn_down.bias"))?, n_embd);
        }
        if p.model.has(&format!("v.blk.{il}.ls2.weight")) {
            let ls2 = p.model.to_f32(&format!("v.blk.{il}.ls2.weight"))?;
            down.par_chunks_mut(n_embd).for_each(|row| {
                for (a, &s) in row.iter_mut().zip(&ls2) { *a *= s; }
            });
        }
        for (a, b) in x.iter_mut().zip(&down) { *a += *b; }
    }

    // 6. Drop CLS token (row 0) and apply `build_patch_merge_permute` (`m x m` spatial blocks into `m*m*n_embd`).
    let patches = &x[n_embd..];
    let feat = n_embd * m * m;
    let mut merged = vec![0f32; n_merged * feat];
    merged.par_chunks_mut(feat).enumerate().for_each(|(t, dst)| {
        let bx = t % mx;
        let by = t / mx;
        let mut at = 0;
        for dy in 0..m {
            for dx in 0..m {
                let q = (by * m + dy) * nx + bx * m + dx;
                dst[at..at + n_embd].copy_from_slice(&patches[q * n_embd..(q + 1) * n_embd]);
                at += n_embd;
            }
        }
    });

    // 7. Projector MLP: `mm.1` -> `GeluErf` -> `mm.2`.
    let mm1 = p.model.tensor("mm.1.weight")?;
    let mut proj = vec![0f32; n_merged * mm1.rows];
    matmul(&mm1, &merged, n_merged, &mut proj);
    if p.model.has("mm.1.bias") {
        add_bias_rows(&mut proj, &p.model.tensor("mm.1.bias")?, mm1.rows);
    }
    proj.par_iter_mut().for_each(|v| *v = activate(FfnKind::GeluErf, *v));
    let mm2 = p.model.tensor("mm.2.weight")?;
    let mut out = vec![0f32; n_merged * c.projection_dim];
    matmul(&mm2, &proj, n_merged, &mut out);
    if p.model.has("mm.2.bias") {
        add_bias_rows(&mut out, &p.model.tensor("mm.2.bias")?, c.projection_dim);
    }
    rows(out, n_merged, c.projection_dim)
}

fn rows(out: Vec<f32>, n: usize, dim: usize) -> anyhow::Result<Vec<Vec<f32>>> {
    anyhow::ensure!(out.len() == n * dim, "encoder output {} is not {n}x{dim}", out.len());
    Ok((0..n).map(|t| out[t * dim..(t + 1) * dim].to_vec()).collect())
}

fn activate(f: FfnKind, x: f32) -> f32 {
    match f {
        FfnKind::Gelu => ops::gelu(x),
        FfnKind::GeluErf => 0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2)),
        FfnKind::GeluQuick => ops::gelu_quick(x),
        FfnKind::Silu => ops::silu(x),
    }
}

/// Abramowitz-Stegun 7.1.26, enough for the vision FFN.
fn erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let y = 1.0 - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_741) * t - 0.284_496_736) * t + 0.254_829_592) * t * (-x * x).exp();
    if x >= 0.0 { y } else { -y }
}

fn layer_norm_all(x: &[f32], n_embd: usize, w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    x.par_chunks(n_embd).flat_map_iter(|row| {
        let mut o = vec![0f32; n_embd];
        ops::layer_norm(row, w, b, eps, &mut o);
        o.into_iter()
    }).collect()
}

fn rms_norm_all(x: &[f32], n_embd: usize, w: &[f32], eps: f32) -> Vec<f32> {
    x.par_chunks(n_embd).flat_map_iter(|row| {
        let mut o = vec![0f32; n_embd];
        ops::rms_norm(row, w, eps, &mut o);
        o.into_iter()
    }).collect()
}

fn rms_norm_nw_all(x: &[f32], n_embd: usize, eps: f32) -> Vec<f32> {
    x.par_chunks(n_embd).flat_map_iter(|row| {
        let mut o = row.to_vec();
        rms_norm_nw(&mut o, &[], eps);
        o.into_iter()
    }).collect()
}

/// RMS norm over `x`, optionally multiplied by `w` (an empty weight skips the multiply).
fn rms_norm_nw(x: &mut [f32], w: &[f32], eps: f32) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let s = 1.0 / (ms + eps).sqrt();
    if w.is_empty() { for v in x { *v *= s; } }
    else { for (v, wv) in x.iter_mut().zip(w) { *v *= s * wv; } }
}

/// Add a `[cols]` bias row to every row of a flat `[tokens * cols]` tensor.
fn add_bias_rows(out: &mut [f32], bias: &Tensor, cols: usize) {
    let mut b = vec![0f32; cols];
    bias.row(0, &mut b);
    out.par_chunks_mut(cols).for_each(|row| {
        for (v, bv) in row.iter_mut().zip(&b) { *v += *bv; }
    });
}

/// The GPU the encoders run their matmuls and attention on (`--gpu`): backend kind and device. Unset: the CPU.
static GPU_AT: OnceLock<(Option<String>, usize)> = OnceLock::new();
/// The backend, opened at the first image (`Some(None)`: it did not open). Encodes take turns on it.
static GPU: Mutex<Option<Option<Box<dyn Backend>>>> = Mutex::new(None);

/// Run the encoders on GPU `device` of backend `kind` (`None`: CUDA, else Vulkan). `STRATA_VISION_CPU` keeps the CPU.
pub fn use_gpu(kind: Option<String>, device: usize) {
    if std::env::var_os("STRATA_VISION_CPU").is_none() { let _ = GPU_AT.set((kind, device)); }
}

/// `f` on the encoder's GPU; `None` without one, or when `f` fails, so the caller runs the CPU code. A failure is
/// said and sets `off`: that operation stays on the CPU from then on.
// ponytail: one failure (out of VRAM too) keeps the operation on the CPU for the rest of the run.
fn on_gpu<T>(off: &AtomicBool, f: impl FnOnce(&dyn Backend) -> anyhow::Result<T>) -> Option<T> {
    let (kind, device) = GPU_AT.get()?;
    if off.load(Relaxed) { return None; }
    let mut g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let b = g.get_or_insert_with(|| crate::backend::open(kind.as_deref(), *device)
        .inspect_err(|e| crate::say!("vision: no GPU ({e:#}), images encode on the CPU")).ok());
    f(b.as_deref()?).inspect_err(|e| {
        off.store(true, Relaxed);
        crate::say!("vision: {e:#}; that part runs on the CPU");
    }).ok()
}

/// The rows of `w` as f32, one after another.
fn rows_f32(w: &Tensor) -> Vec<f32> {
    let mut f = vec![0f32; w.rows * w.cols];
    f.par_chunks_mut(w.cols).enumerate().for_each(|(r, row)| w.row(r, row));
    f
}

fn f32_bytes(x: &[f32]) -> &[u8] {
    // SAFETY: any f32 is 4 plain bytes; u8 has no alignment.
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), 4 * x.len()) }
}

fn put(b: &dyn Backend, x: &[f32]) -> anyhow::Result<crate::backend::Buf> {
    let buf = b.alloc(4 * x.len())?;
    b.htod(f32_bytes(x), &buf, 0)?;
    Ok(buf)
}

fn get(b: &dyn Backend, buf: &crate::backend::Buf, out: &mut [f32]) -> anyhow::Result<()> {
    // SAFETY: as in f32_bytes; every byte pattern is a valid f32.
    b.dtoh(buf, 0, unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast(), 4 * out.len()) })
}

/// `out = w x` for `n` rows of x: on the GPU when there is one (the weight is uploaded for this call: each is used
/// once an image), else `Tensor::matmul`.
fn matmul(w: &Tensor, x: &[f32], n: usize, out: &mut [f32]) {
    // The GPU path does not bound-check like the CPU slices do: a mmproj with mismatched shapes must stop here.
    assert!(x.len() == n * w.cols && out.len() == n * w.rows, "vision matmul: shape mismatch");
    static OFF: AtomicBool = AtomicBool::new(false);
    let done = on_gpu(&OFF, |b| {
        // The GEMM takes columns in steps of 64; a weight with other widths (Qwen3-VL's ffn_down: 4304) goes up with
        // zero columns added (BF16 and F32 as they are, other types as F32), x with them too.
        let cols = w.cols.next_multiple_of(64);
        let (gw, cols) = if w.cols % 64 == 0 { (b.upload(w)?, w.cols) } else if matches!(w.ty, GgmlType::BF16 | GgmlType::F32) {
            let (rb, pb) = (w.data.len() / w.rows, w.data.len() / w.rows / w.cols * cols);
            let mut raw = vec![0u8; w.rows * pb];
            for (d, s) in raw.chunks_exact_mut(pb).zip(w.data.chunks_exact(rb)) { d[..rb].copy_from_slice(s); }
            let buf = b.alloc(raw.len())?;
            b.htod(&raw, &buf, 0)?;
            (GpuTensor::new(w.ty, w.rows, cols, buf).with_scale(w.scale), cols)
        } else {
            let mut f = vec![0f32; w.rows * cols];
            for (r, row) in f.chunks_exact_mut(cols).enumerate() { w.row(r, &mut row[..w.cols]); }
            (GpuTensor::new(GgmlType::F32, w.rows, cols, put(b, &f)?), cols)
        };
        let xb = if cols == w.cols { put(b, x)? } else {
            let mut p = vec![0f32; n * cols];
            for (d, s) in p.chunks_exact_mut(cols).zip(x.chunks_exact(w.cols)) { d[..w.cols].copy_from_slice(s); }
            put(b, &p)?
        };
        let y = b.alloc(4 * out.len())?;
        b.gemm_dense(&gw, n, xb.addr(), cols, y.addr(), w.rows, 0)?;
        get(b, &y, out)
    });
    if done.is_none() { w.matmul(x, n, out); }
}

/// Unmasked attention over `n` rows of `n_head` heads (`[n][n_head][d_head]`): on the GPU when there is one.
fn attention(q: &[f32], k: &[f32], v: &[f32], n: usize, n_head: usize, d_head: usize, scale: f32) -> Vec<f32> {
    let mut out = vec![0f32; n * n_head * d_head];
    assert!(q.len() == out.len() && k.len() == out.len() && v.len() == out.len(), "vision attention: shape mismatch");
    static OFF: AtomicBool = AtomicBool::new(false);
    let done = on_gpu(&OFF, |b| {
        let (qb, kb, vb, ob) = (put(b, q)?, put(b, k)?, put(b, v)?, b.alloc(4 * out.len())?);
        b.vit_attn(qb.addr(), kb.addr(), vb.addr(), ob.addr(), n, n_head, d_head, scale)?;
        get(b, &ob, &mut out)
    });
    if done.is_some() { return out; }
    out.par_chunks_mut(n_head * d_head).enumerate().for_each(|(t, o)| {
        let mut scores = vec![0f32; n];
        for h in 0..n_head {
            let qh = &q[t * n_head * d_head + h * d_head..t * n_head * d_head + (h + 1) * d_head];
            for s in 0..n {
                let kh = &k[s * n_head * d_head + h * d_head..s * n_head * d_head + (h + 1) * d_head];
                scores[s] = crate::weights::dot(qh, kh) * scale;
            }
            ops::softmax(&mut scores);
            // Token-outer: each V head row is read once, contiguously (the per-element sum order is unchanged).
            let oh = &mut o[h * d_head..(h + 1) * d_head];
            for s in 0..n {
                let vh = &v[s * n_head * d_head + h * d_head..][..d_head];
                for (a, b) in oh.iter_mut().zip(vh) { *a += scores[s] * b; }
            }
        }
    });
    out
}

/// Run the encoder for an image and return features plus the token grid (`nx, ny`).
pub fn encode_image(projector: &VisionProjector, img: &Rgb) -> anyhow::Result<(Vec<Vec<f32>>, usize, usize)> {
    let pre = projector.preprocess(img).context("preprocess image")?;
    let (nx, ny) = projector.count_tokens(pre.width, pre.height)?;
    let feats = projector.encode(&pre)?;
    anyhow::ensure!(feats.len() == nx * ny, "encoder returned {} tokens, expected {}x{}", feats.len(), nx, ny);
    Ok((feats, nx, ny))
}

