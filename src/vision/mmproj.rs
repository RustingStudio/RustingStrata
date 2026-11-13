//! CLIP/mmproj GGUF loading. A projector file is a separate GGUF with `general.architecture=clip`,
//! `general.type=mmproj`, and a `clip.projector_type` naming the graph.

use crate::gguf::{Gguf, Value};
use crate::quant::GgmlType;
use crate::weights::Tensor;
use anyhow::Context;
use std::path::Path;

/// The projector graph named by `clip.projector_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectorKind { Qwen3Vl, Gemma4V, Gemma4Uv, Glm4V }

impl ProjectorKind {
    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "qwen3vl_merger" => Self::Qwen3Vl,
            "gemma4v" => Self::Gemma4V,
            "gemma4uv" => Self::Gemma4Uv,
            "glm4v" => Self::Glm4V,
            _ => return None,
        })
    }
}

/// Activation of the vision FFN (`clip.use_gelu` / `clip.use_silu`, default quick GELU).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfnKind { Gelu, GeluErf, GeluQuick, Silu }

/// How patches become image tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MergeKind { SpatialReorder, AvgPool, PatchConv, None }

#[derive(Clone, Debug)]
pub struct ClipConfig {
    pub kind: ProjectorKind,
    pub ffn: FfnKind,
    pub norm_rms: bool,
    pub merge: MergeKind,
    pub rope_theta: f32,
    pub arch: String,
    pub projector_type: String,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub d_head: usize,
    pub n_layer: usize,
    pub n_ff: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub n_merge: usize,
    pub eps: f32,
    pub projection_dim: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub use_gelu: bool,
    pub image_min_pixels: usize,
    pub image_max_pixels: usize,
    pub deepstack: Vec<bool>,
}

impl ClipConfig {
    pub fn from_gguf(g: &Gguf) -> anyhow::Result<ClipConfig> {
        let arch = g.get("general.architecture")?.as_str().unwrap_or_default().to_string();
        let projector_type = g.get("clip.projector_type").ok().and_then(|v| v.as_str())
            .or_else(|| g.get("clip.vision.projector_type").ok().and_then(|v| v.as_str()))
            .unwrap_or_default().to_string();
        let kind = ProjectorKind::from_name(&projector_type)
            .with_context(|| format!("projector type '{projector_type}' is not implemented"))?;
        let u = |k: &str| -> anyhow::Result<usize> {
            g.get(k)?.as_u64().with_context(|| format!("{k} is not an integer")).map(|v| v as usize)
        };
        let opt_u = |k: &str| g.get(k).ok().and_then(|v| v.as_u64()).map(|v| v as usize);
        let opt_bool = |k: &str| match g.get(k) {
            Ok(Value::Bool(b)) => *b,
            Ok(v) => v.as_u64().is_some_and(|n| n != 0),
            Err(_) => false,
        };
        let n_embd = u("clip.vision.embedding_length")?;
        let n_head = u("clip.vision.attention.head_count")?;
        let n_head_kv = opt_u("clip.vision.attention.head_count_kv").unwrap_or(n_head);
        let n_layer = u("clip.vision.block_count")?;
        let n_ff = u("clip.vision.feed_forward_length")?;
        let image_size = u("clip.vision.image_size")?;
        let mut patch_size = u("clip.vision.patch_size")?;
        let mut n_merge = match kind {
            ProjectorKind::Qwen3Vl | ProjectorKind::Glm4V => opt_u("clip.vision.spatial_merge_size").unwrap_or(2),
            ProjectorKind::Gemma4V | ProjectorKind::Gemma4Uv => opt_u("clip.vision.projector.scale_factor").unwrap_or(3),
        };
        if kind == ProjectorKind::Gemma4Uv {
            // The unified variant folds the 3x3 merge into a larger conv patch and has no separate merge.
            patch_size *= n_merge;
            n_merge = 1;
        }
        let projection_dim = u("clip.vision.projection_dim")?;
        let eps = g.get("clip.vision.attention.layer_norm_epsilon")?.as_f32().context("clip.vision.attention.layer_norm_epsilon")?;
        let image_mean = arr3_f32(g, "clip.vision.image_mean")?;
        let image_std = arr3_f32(g, "clip.vision.image_std")?;
        let use_gelu = opt_bool("clip.use_gelu");
        let use_silu = opt_bool("clip.use_silu");
        let ffn = if use_gelu { FfnKind::Gelu } else if use_silu { FfnKind::Silu } else { FfnKind::GeluQuick };
        let (norm_rms, merge, rope_theta) = match kind {
            ProjectorKind::Qwen3Vl => (false, MergeKind::SpatialReorder, 10_000.0),
            ProjectorKind::Gemma4V => (true, MergeKind::AvgPool, 100.0),
            ProjectorKind::Gemma4Uv => (true, MergeKind::None, 100.0),
            ProjectorKind::Glm4V => (true, MergeKind::PatchConv, 10_000.0),
        };
        let patch_area = patch_size * patch_size * n_merge * n_merge;
        let (min_tok, max_tok) = match kind { ProjectorKind::Gemma4V | ProjectorKind::Gemma4Uv => (40, 280), _ => (8, 4096) };
        let image_min_pixels = opt_u("clip.vision.image_min_pixels").unwrap_or(min_tok * patch_area);
        let image_max_pixels = opt_u("clip.vision.image_max_pixels").unwrap_or(max_tok * patch_area);
        let deepstack = match g.get("clip.vision.is_deepstack_layers") {
            Ok(v) => v.as_arr().context("clip.vision.is_deepstack_layers")?.iter().map(|x| matches!(x, Value::Bool(true))).collect(),
            Err(_) => vec![],
        };
        let d_head = if n_head > 0 { n_embd / n_head } else { 0 };
        anyhow::ensure!(n_head == 0 || n_embd % n_head == 0, "clip: embedding_length {n_embd} is not divisible by head_count {n_head}");
        Ok(ClipConfig {
            kind,
            ffn,
            norm_rms,
            merge,
            rope_theta,
            arch,
            projector_type,
            n_embd,
            n_head,
            n_head_kv,
            d_head,
            n_layer,
            n_ff,
            image_size,
            patch_size,
            n_merge,
            eps,
            projection_dim,
            image_mean,
            image_std,
            use_gelu,
            image_min_pixels,
            image_max_pixels,
            deepstack,
        })
    }

    /// Pixels per side of the 2x2 spatial merge, i.e. the resize alignment.
    pub fn merge_align(&self) -> usize { self.patch_size * self.n_merge }
}

fn arr3_f32(g: &Gguf, key: &str) -> anyhow::Result<[f32; 3]> {
    let a = g.get(key)?.as_arr().with_context(|| format!("{key} is not an array"))?;
    anyhow::ensure!(a.len() >= 3, "{key} has {} entries", a.len());
    let mut out = [0.0; 3];
    for (o, v) in out.iter_mut().zip(a) { *o = v.as_f32().with_context(|| format!("{key} has a non-number"))?; }
    Ok(out)
}

pub struct ClipModel {
    pub g: Gguf,
    pub cfg: ClipConfig,
}

impl ClipModel {
    pub fn open(path: &Path) -> anyhow::Result<ClipModel> {
        let g = Gguf::open(path)?;
        let cfg = ClipConfig::from_gguf(&g)?;
        anyhow::ensure!(cfg.arch == "clip", "{}: not a CLIP projector (architecture {})", path.display(), cfg.arch);
        anyhow::ensure!(g.meta.get("general.type").and_then(Value::as_str) == Some("mmproj"),
            "{}: general.type is not mmproj", path.display());
        Ok(ClipModel { g, cfg })
    }

    pub fn has(&self, name: &str) -> bool { self.g.tensors.contains_key(name) }

    /// A tensor view whose first dim is `cols`; the remaining dims are flattened into rows.
    pub fn tensor(&self, name: &str) -> anyhow::Result<Tensor<'_>> {
        self.tensor_as(name, None)
    }

    /// Like `tensor`, but overriding the first dimension. Used for conv kernels stored as 4-D.
    pub fn tensor_as(&self, name: &str, cols: Option<usize>) -> anyhow::Result<Tensor<'_>> {
        let t = self.g.tensors.get(name).with_context(|| format!("missing tensor {name}"))?;
        let ty = GgmlType::from_u32(t.ty).with_context(|| format!("tensor {name}"))?;
        let cols = cols.unwrap_or(t.dims[0] as usize);
        let total: usize = t.dims.iter().map(|&d| d as usize).product();
        anyhow::ensure!(cols > 0 && total % cols == 0, "tensor {name}: {total} elements not divisible by cols {cols}");
        Ok(Tensor { ty, cols, rows: total / cols, data: self.g.bytes(t) })
    }

    /// F32 view of a tensor, copied out of the mmap.
    pub fn to_f32(&self, name: &str) -> anyhow::Result<Vec<f32>> { Ok(self.tensor(name)?.to_f32()) }

    pub fn meta_u64(&self, key: &str) -> Option<u64> { self.g.meta.get(key).and_then(Value::as_u64) }
}

/// Read a GGUF metadata array of strings (used by tests and future projector adapters).
pub fn meta_strs(g: &Gguf, key: &str) -> Vec<String> {
    g.meta.get(key).and_then(Value::as_arr).map(|a| a.iter().filter_map(|v| v.as_str().map(Into::into)).collect()).unwrap_or_default()
}

/// `true` when the file looks like an mmproj sidecar rather than a text model.
pub fn is_mmproj(path: &Path) -> bool {
    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
    name.starts_with("mmproj") || name.contains("-mmproj") || name.contains("mmproj-")
}

/// Find `mmproj*.gguf` next to a text model.
pub fn find_sibling(model: &Path) -> Option<std::path::PathBuf> {
    let dir = model.parent()?;
    let mut found = None;
    for e in std::fs::read_dir(dir).ok()? {
        let p = e.ok()?.path();
        if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("gguf")) && is_mmproj(&p) {
            found = Some(p);
            break;
        }
    }
    found
}

/// Validate that the projector's output width matches the text model's embedding width.
pub fn validate_projection(cfg: &ClipConfig, text_n_embd: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        cfg.projection_dim == text_n_embd,
        "projector {} produces {} features but the text model expects {}",
        cfg.projector_type, cfg.projection_dim, text_n_embd
    );
    Ok(())
}
