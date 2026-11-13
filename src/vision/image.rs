//! Image decoding, smart resize and normalization matching llama.cpp's mtmd preprocessing.

use anyhow::{Context, bail};
use base64::Engine as _;
use image::{DynamicImage, RgbImage};
use std::path::Path;
use std::time::Duration;

/// A decoded RGB image with rows top-to-bottom and interleaved channels (`R G B R G B ...`).
pub type Rgb = RgbImage;

/// A projector input: RGB pixels scaled to `[0,1]` and normalized with the projector's mean/std.
pub struct PreprocessedImage {
    pub width: usize,
    pub height: usize,
    /// `width * height * 3`, interleaved RGB.
    pub rgb: Vec<f32>,
}

impl PreprocessedImage {
    pub fn from_u8(width: usize, height: usize, bytes: &[u8], mean: [f32; 3], std: [f32; 3]) -> anyhow::Result<Self> {
        anyhow::ensure!(bytes.len() == width * height * 3, "image buffer has {} bytes for {width}x{height}", bytes.len());
        let mut rgb = vec![0.0; bytes.len()];
        for (o, &v) in rgb.iter_mut().zip(bytes) {
            // The channel is implied by the position in each RGB triple.
            *o = v as f32 / 255.0;
        }
        for (i, o) in rgb.iter_mut().enumerate() {
            let c = i % 3;
            *o = (*o - mean[c]) / std[c];
        }
        Ok(PreprocessedImage { width, height, rgb })
    }
}

/// Decode a PNG/JPEG/WebP byte stream to RGB.
pub fn decode_bytes(bytes: &[u8]) -> anyhow::Result<Rgb> {
    let img = image::load_from_memory(bytes).context("decode image")?;
    Ok(to_rgb(img))
}

/// Decode an image file to RGB. EXIF orientation is not applied, matching stb_image in llama.cpp.
pub fn decode_path(path: &Path) -> anyhow::Result<Rgb> {
    let img = image::ImageReader::open(path).with_context(|| format!("open image {}", path.display()))?.decode()
        .with_context(|| format!("decode image {}", path.display()))?;
    Ok(to_rgb(img))
}

fn to_rgb(img: DynamicImage) -> Rgb {
    // `to_rgb8` expands grayscale and drops alpha; the buffer stays row-major, top-to-bottom.
    img.to_rgb8()
}

/// Load an image from a file path, a `data:` URL, or an `http(s)` URL.
pub fn load_image(src: &str) -> anyhow::Result<Rgb> {
    if let Some(rest) = src.strip_prefix("data:") {
        return decode_bytes(&decode_data_url(rest)?);
    }
    if src.starts_with("http://") || src.starts_with("https://") {
        return decode_bytes(&fetch_url(src)?);
    }
    decode_path(Path::new(src))
}

fn decode_data_url(rest: &str) -> anyhow::Result<Vec<u8>> {
    let (meta, payload) = rest.split_once(',').context("data URL without a comma")?;
    anyhow::ensure!(meta.contains(";base64"), "only base64 data URLs are supported");
    base64::engine::general_purpose::STANDARD.decode(payload).context("decode data URL base64")
}

fn fetch_url(url: &str) -> anyhow::Result<Vec<u8>> {
    check_public_url(url)?;
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(10)))
        .timeout_global(Some(Duration::from_secs(30)))
        .build()
        .into();
    let mut r = agent.get(url).call().with_context(|| format!("fetch {url}"))?;
    let body = r.body_mut().read_to_vec().context("read image body")?;
    Ok(body)
}

/// Reject URLs whose host resolves to a loopback, private, link-local or otherwise non-global address.
/// `STRATA_ALLOW_PRIVATE_IMAGE_URLS=1` disables the check for tests and trusted local setups.
pub fn check_public_url(url: &str) -> anyhow::Result<()> {
    if std::env::var("STRATA_ALLOW_PRIVATE_IMAGE_URLS").is_ok_and(|v| v == "1") { return Ok(()); }
    let host = url_host(url).context("image URL has no host")?;
    use std::net::ToSocketAddrs;
    let addrs: Vec<_> = (host.as_str(), 0).to_socket_addrs().with_context(|| format!("resolve {host}"))?.collect();
    anyhow::ensure!(!addrs.is_empty(), "{host} did not resolve");
    for a in addrs {
        if !is_public_ip(a.ip()) { bail!("image URL {url} resolves to a private or loopback address"); }
    }
    Ok(())
}

fn url_host(url: &str) -> Option<String> {
    let after = url.split_once("://")?.1;
    let authority = after.split(['/', '?', '#']).next()?;
    let authority = authority.rsplit_once('@').map_or(authority, |x| x.1);
    if let Some(stripped) = authority.strip_prefix('[') { return stripped.split(']').next().map(Into::into); }
    Some(authority.rsplit_once(':').filter(|(_, p)| p.parse::<u16>().is_ok()).map_or(authority, |x| x.0).to_string())
}

fn is_public_ip(ip: std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v4) => !(v4.is_loopback() || v4.is_private() || v4.is_link_local() || v4.is_broadcast() || v4.is_unspecified() || v4.is_multicast()),
        std::net::IpAddr::V6(v6) => !(v6.is_loopback() || v6.is_unspecified() || v6.is_multicast() || v6.is_unicast_link_local() || v6.is_unique_local()),
    }
}

/// llama.cpp `calc_size_preserved_ratio`: round to `align`, keep `min_pixels <= w*h <= max_pixels`.
pub fn smart_resize(width: usize, height: usize, align: usize, min_pixels: usize, max_pixels: usize) -> (usize, usize) {
    if width == 0 || height == 0 { return (align.max(1), align.max(1)); }
    let align = align.max(1);
    let round_by = |x: f64| ((x / align as f64).round() as usize).max(1) * align;
    let ceil_by = |x: f64| ((x / align as f64).ceil() as usize).max(1) * align;
    let floor_by = |x: f64| ((x / align as f64).floor() as usize).max(1) * align;
    let mut h_bar = round_by(height as f64).max(align);
    let mut w_bar = round_by(width as f64).max(align);
    if h_bar.saturating_mul(w_bar) > max_pixels {
        let beta = ((height as f64 * width as f64) / max_pixels as f64).sqrt();
        h_bar = floor_by(height as f64 / beta).max(align);
        w_bar = floor_by(width as f64 / beta).max(align);
    } else if h_bar.saturating_mul(w_bar) < min_pixels {
        let beta = (min_pixels as f64 / (height as f64 * width as f64)).sqrt();
        h_bar = ceil_by(height as f64 * beta);
        w_bar = ceil_by(width as f64 * beta);
    }
    (w_bar, h_bar)
}

/// Bilinear resize with align-corners (`(src-1)/(dst-1)`), matching `mtmd_image.cpp::resize_bilinear`.
pub fn resize_bilinear(src: &Rgb, width: usize, height: usize) -> Rgb {
    let (sw, sh) = (src.width() as usize, src.height() as usize);
    if (sw, sh) == (width, height) { return src.clone(); }
    let mut dst = RgbImage::new(width as u32, height as u32);
    if width == 0 || height == 0 { return dst; }
    let x_ratio = if width > 1 { (sw - 1) as f32 / (width - 1) as f32 } else { 0.0 };
    let y_ratio = if height > 1 { (sh - 1) as f32 / (height - 1) as f32 } else { 0.0 };
    let s = src.as_raw();
    let d = dst.as_mut();
    for y in 0..height {
        let py = y as f32 * y_ratio;
        let y0 = (py as usize).min(sh - 1);
        let y1 = (y0 + 1).min(sh - 1);
        let yf = py - y0 as f32;
        for x in 0..width {
            let px = x as f32 * x_ratio;
            let x0 = (px as usize).min(sw - 1);
            let x1 = (x0 + 1).min(sw - 1);
            let xf = px - x0 as f32;
            for c in 0..3 {
                let i00 = (y0 * sw + x0) * 3 + c;
                let i10 = (y0 * sw + x1) * 3 + c;
                let i01 = (y1 * sw + x0) * 3 + c;
                let i11 = (y1 * sw + x1) * 3 + c;
                let top = s[i00] as f32 * (1.0 - xf) + s[i10] as f32 * xf;
                let bot = s[i01] as f32 * (1.0 - xf) + s[i11] as f32 * xf;
                let v = top * (1.0 - yf) + bot * yf;
                d[(y * width + x) * 3 + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    dst
}

/// Bilinear + antialias resize of a `[n_embd][w][h]` (feature-fastest) f32 plane, matching `ggml_interpolate`
/// with `GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS`.
pub fn resize_bilinear_antialias_f32(src: &[f32], n_embd: usize, sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; n_embd * dw * dh];
    if sw == dw && sh == dh { dst.copy_from_slice(src); return dst; }
    let sf0 = dw as f32 / sw as f32;
    let sf1 = dh as f32 / sh as f32;
    let pixel_offset = 0.5f32;
    let support0 = 1.0f32.max(1.0 / sf0);
    let invscale0 = 1.0 / support0;
    let support1 = 1.0f32.max(1.0 / sf1);
    let invscale1 = 1.0 / support1;
    let tri = |x: f32| (1.0 - x.abs()).max(0.0);
    for y in 0..dh {
        let yf = (y as f32 + pixel_offset) / sf1;
        let y_min = ((yf - support1 + pixel_offset) as i64).max(0) as usize;
        let y_max = ((yf + support1 + pixel_offset) as i64).min(sh as i64) as usize;
        for x in 0..dw {
            let xf = (x as f32 + pixel_offset) / sf0;
            let x_min = ((xf - support0 + pixel_offset) as i64).max(0) as usize;
            let x_max = ((xf + support0 + pixel_offset) as i64).min(sw as i64) as usize;
            let mut wsum = 0.0f32;
            let mut pixels: Vec<(usize, f32)> = Vec::new();
            for sy in y_min..y_max {
                let wy = tri((sy as f32 - yf + pixel_offset) * invscale1);
                for sx in x_min..x_max {
                    let wx = tri((sx as f32 - xf + pixel_offset) * invscale0);
                    let w = wx * wy;
                    if w > 0.0 { pixels.push((sx + sw * sy, w)); wsum += w; }
                }
            }
            let dst_off = x + dw * y;
            if wsum <= 0.0 { continue; }
            for c in 0..n_embd {
                let mut val = 0.0f32;
                for &(idx, w) in &pixels { val += src[c + n_embd * idx] * w; }
                dst[c + n_embd * dst_off] = val / wsum;
            }
        }
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_matches_llama() {
        // 1000x500, align 32, 8..4096 tokens * 1024 pixels.
        let (w, h) = smart_resize(1000, 500, 32, 8 * 1024, 4096 * 1024);
        assert_eq!((w, h), (992, 512));
        let (w, h) = smart_resize(10, 10, 32, 8 * 1024, 4096 * 1024);
        assert!(w >= 32 && h >= 32);
        assert!(w * h >= 8 * 1024);
    }

    #[test]
    fn bilinear_identity() {
        let mut img = RgbImage::new(2, 2);
        img.put_pixel(0, 0, image::Rgb([1, 2, 3]));
        img.put_pixel(1, 0, image::Rgb([4, 5, 6]));
        img.put_pixel(0, 1, image::Rgb([7, 8, 9]));
        img.put_pixel(1, 1, image::Rgb([10, 11, 12]));
        let same = resize_bilinear(&img, 2, 2);
        assert_eq!(same.as_raw(), img.as_raw());
    }
}
