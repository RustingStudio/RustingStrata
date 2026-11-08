//! Quantised matrix view over the GGUF mmap with an f32 matvec (slow reference path).
use crate::gguf::{Gguf, TensorInfo};
use crate::quant::{GgmlType, dequant_row};
use rayon::prelude::*;

/// Row-major weight: `rows` rows of `cols` elements (ggml dims[0] = cols, dims[1] = rows).
/// A 3-D expert tensor is addressed with `expert(e)`.
#[derive(Clone, Copy)]
pub struct Mat<'a> {
    pub data: &'a [u8],
    pub ty: GgmlType,
    pub rows: usize,
    pub cols: usize,
}

impl<'a> Mat<'a> {
    pub fn load(g: &'a Gguf, name: &str) -> anyhow::Result<Self> {
        let t: &TensorInfo = g.tensors.get(name).ok_or_else(|| anyhow::anyhow!("missing tensor {name}"))?;
        let ty = GgmlType::from_u32(t.ty)?;
        let cols = t.dims[0] as usize;
        let rows = t.dims[1..].iter().product::<u64>() as usize;
        Ok(Mat { data: g.bytes(t), ty, rows, cols })
    }

    /// Slice of a [cols, rows, n_expert] tensor for expert e.
    pub fn expert(&self, e: usize, n_expert: usize) -> Mat<'a> {
        let rows = self.rows / n_expert;
        let sz = rows * self.ty.row_bytes(self.cols as u64);
        Mat { data: &self.data[e * sz..(e + 1) * sz], ty: self.ty, rows, cols: self.cols }
    }

    pub fn row_into(&self, r: usize, out: &mut [f32]) {
        let rb = self.ty.row_bytes(self.cols as u64);
        dequant_row(self.ty, &self.data[r * rb..(r + 1) * rb], out);
    }

    /// y = W x
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(y.len(), self.rows);
        let rb = self.ty.row_bytes(self.cols as u64);
        y.par_iter_mut().enumerate().for_each_init(
            || vec![0f32; self.cols],
            |tmp, (r, o)| {
                dequant_row(self.ty, &self.data[r * rb..(r + 1) * rb], tmp);
                *o = tmp.iter().zip(x).map(|(a, b)| a * b).sum();
            },
        );
    }
}

/// Small F32/BF16 tensor fully dequantised.
pub fn vec_f32(g: &Gguf, name: &str) -> anyhow::Result<Vec<f32>> {
    let m = Mat::load(g, name)?;
    let mut v = vec![0f32; m.rows * m.cols];
    for r in 0..m.rows {
        m.row_into(r, &mut v[r * m.cols..(r + 1) * m.cols]);
    }
    Ok(v)
}
