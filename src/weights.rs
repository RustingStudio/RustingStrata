//! Named tensor views over the mmapped GGUF and a reference parallel matvec.

use crate::gguf::Gguf;
use crate::quant::{GgmlType, dequant_row};
use anyhow::Context;
use rayon::prelude::*;

#[derive(Clone, Copy)]
pub struct Tensor<'a> {
    pub ty: GgmlType,
    /// Elements per row (GGUF dims[0]).
    pub cols: usize,
    /// Product of the remaining dims; for 3-D expert tensors this is out_rows * n_expert.
    pub rows: usize,
    pub data: &'a [u8],
}

impl<'a> Tensor<'a> {
    fn row_bytes(&self) -> usize { self.ty.row_bytes(self.cols as u64) }

    pub fn row(&self, r: usize, out: &mut [f32]) {
        let rb = self.row_bytes();
        dequant_row(self.ty, &self.data[r * rb..(r + 1) * rb], out);
    }

    /// y[r] = row_r · x for all rows.
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!((x.len(), y.len()), (self.cols, self.rows));
        y.par_chunks_mut(16).enumerate().for_each_init(|| vec![0f32; self.cols], |buf, (ci, ys)| {
            for (k, yv) in ys.iter_mut().enumerate() {
                self.row(ci * 16 + k, buf);
                *yv = buf.iter().zip(x).map(|(a, b)| a * b).sum();
            }
        });
    }

    /// Rows belonging to expert `e` of an [cols, out, n_expert] tensor.
    pub fn expert(&self, e: usize, n_expert: usize) -> Tensor<'a> {
        let rows = self.rows / n_expert;
        let rb = self.row_bytes();
        Tensor { rows, data: &self.data[e * rows * rb..(e + 1) * rows * rb], ..*self }
    }

    pub fn to_f32(&self) -> Vec<f32> {
        let mut v = vec![0f32; self.rows * self.cols];
        for (r, out) in v.chunks_exact_mut(self.cols).enumerate() { self.row(r, out); }
        v
    }
}

pub struct Weights<'a> {
    pub g: &'a Gguf,
}

impl<'a> Weights<'a> {
    pub fn get(&self, name: &str) -> anyhow::Result<Tensor<'a>> {
        let t = self.g.tensors.get(name).with_context(|| format!("missing tensor {name}"))?;
        let ty = GgmlType::from_u32(t.ty).with_context(|| format!("tensor {name}"))?;
        Ok(Tensor {
            ty,
            cols: t.dims[0] as usize,
            rows: t.dims[1..].iter().product::<u64>() as usize,
            data: self.g.bytes(t),
        })
    }

    pub fn layer(&self, il: usize, suffix: &str) -> anyhow::Result<Tensor<'a>> { self.get(&format!("blk.{il}.{suffix}")) }

    /// Refuse files containing quant types this engine does not implement.
    pub fn validate(&self) -> anyhow::Result<()> {
        for t in self.g.tensors.values() {
            GgmlType::from_u32(t.ty).with_context(|| format!("tensor {}", t.name))?;
        }
        Ok(())
    }
}
