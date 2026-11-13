//! Prompt assembly for image input: expand the projector's placeholder token and assign M-RoPE positions.
//!
//! For Qwen3-VL the template renders `<|vision_start|><|image_pad|><|vision_end|>`. The single image pad
//! token expands to `nx * ny` tokens. The pad's position is `p0`; image token `i` has `(t, y, x, z) =
//! (p0, p0 + i / nx, p0 + i % nx, 0)` (row before column, matching llama.cpp's M-RoPE sections).
//! Text after `<|vision_end|>` resumes at `p0 + max(nx, ny)`.

use crate::model::{Model, State};
use anyhow::{Context, bail};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageSpan {
    /// Index of the first image token in the expanded prompt.
    pub start: usize,
    pub width: usize,
    pub height: usize,
}

impl ImageSpan {
    pub fn n_tokens(&self) -> usize { self.width * self.height }
}

#[derive(Clone, Debug, Default)]
pub struct Prompt {
    pub ids: Vec<u32>,
    /// One `[t, y, x, z]` position per token. Text tokens use the same value in all four slots.
    pub positions: Vec<[i32; 4]>,
    pub images: Vec<ImageSpan>,
    /// M-RoPE position of the first token after the prompt.
    pub next_position: i32,
}

/// Expand `<|vision_start|> <|image_pad|> <|vision_end|>` into an image span and text positions.
/// `width`/`height` are the merged token grid (`nx`, `ny`).
pub fn expand_image(ids: &[u32], vision_start: u32, vision_end: u32, image_pad: u32, width: usize, height: usize) -> anyhow::Result<Prompt> {
    expand_images(ids, vision_start, vision_end, image_pad, &[(width, height)])
}

/// Expand one `<|vision_start|> <image_pad> <vision_end|>` group per grid, in prompt order.
pub fn expand_images(ids: &[u32], vision_start: u32, vision_end: u32, image_pad: u32, grids: &[(usize, usize)]) -> anyhow::Result<Prompt> {
    let mut p = 0i32;
    let mut out = Prompt::default();
    let mut next_image = 0;
    let mut i = 0;
    while i < ids.len() {
        let id = ids[i];
        if id == vision_start {
            let (width, height) = grids.get(next_image).copied().context("prompt has more images than --image")?;
            next_image += 1;
            anyhow::ensure!(width > 0 && height > 0, "image grid is empty");
            let n = width * height;
            out.ids.push(id);
            out.positions.push([p, p, p, p]);
            p += 1;
            i += 1;
            let next = ids.get(i).copied().context("vision_start without image_pad")?;
            if next != image_pad {
                bail!("vision_start is followed by {next}, expected image_pad {image_pad}");
            }
            let start = out.ids.len();
            let p0 = p;
            for k in 0..n {
                out.ids.push(image_pad);
                out.positions.push([p0, p0 + (k / width) as i32, p0 + (k % width) as i32, 0]);
            }
            out.images.push(ImageSpan { start, width, height });
            p = p0 + width.max(height) as i32;
            i += 1;
            // The matching vision_end is a text token at the first position after the image.
            if ids.get(i).copied() == Some(vision_end) {
                out.ids.push(vision_end);
                out.positions.push([p, p, p, p]);
                p += 1;
                i += 1;
            }
            continue;
        }
        out.ids.push(id);
        out.positions.push([p, p, p, p]);
        p += 1;
        i += 1;
    }
    anyhow::ensure!(next_image == grids.len(), "{} image grid(s) unused", grids.len() - next_image);
    out.next_position = p;
    Ok(out)
}

/// Position of the first text token after an image span, matching llama.cpp's decoder positions.
pub fn next_text_position(p0: i32, width: usize, height: usize) -> i32 { p0 + width.max(height) as i32 }

/// Feed the expanded prompt through the CPU model: text tokens use the model's embeddings, image tokens the
/// projector rows. Returns the logits after the last token and leaves `st.rope_pos` at the first text position.
pub fn feed_prompt(m: &Model, st: &mut State, p: &Prompt, embs: &[Vec<Vec<f32>>]) -> anyhow::Result<Vec<f32>> {
    let mut logits = Vec::new();
    let mut img = 0;
    let mut row = 0;
    for (i, &id) in p.ids.iter().enumerate() {
        let x = if let Some(span) = p.images.get(img) && i >= span.start && i < span.start + span.n_tokens() {
            let x = embs[img][row].clone();
            row += 1;
            if row == span.n_tokens() { img += 1; row = 0; }
            x
        } else {
            m.embed(id)
        };
        logits = m.step_embed(st, x, p.positions[i], id);
    }
    st.rope_pos = [p.next_position; 4];
    Ok(logits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_one_image() {
        let ids = [1, 10, 20, 30, 2];
        let p = expand_image(&ids, 10, 30, 20, 2, 2).unwrap();
        assert_eq!(p.ids, [1, 10, 20, 20, 20, 20, 30, 2]);
        assert_eq!(p.positions, [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 0], [2, 2, 3, 0], [2, 3, 2, 0], [2, 3, 3, 0], [4, 4, 4, 4], [5, 5, 5, 5]]);
        assert_eq!(p.images, [ImageSpan { start: 2, width: 2, height: 2 }]);
    }

    #[test]
    fn text_resumes_after_wider_side() {
        let ids = [10, 20, 30];
        let p = expand_image(&ids, 10, 30, 20, 3, 2).unwrap();
        assert_eq!(p.positions.last().copied(), Some([4, 4, 4, 4]));
    }

    #[test]
    fn image_positions_are_row_then_column() {
        let ids = [10, 20, 30];
        let p = expand_image(&ids, 10, 30, 20, 3, 2).unwrap();
        assert_eq!(p.positions[1..7], [
            [1, 1, 1, 0], [1, 1, 2, 0], [1, 1, 3, 0],
            [1, 2, 1, 0], [1, 2, 2, 0], [1, 2, 3, 0],
        ]);
    }
}
