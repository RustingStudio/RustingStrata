//! GGUF v2/v3 parser over a memory map.

use anyhow::{Context, bail};
use memmap2::Mmap;
use std::collections::BTreeMap;
use std::path::Path;

use crate::quant::GgmlType;

#[derive(Clone, Debug)]
pub enum Value {
    U8(u8), I8(i8), U16(u16), I16(i16), U32(u32), I32(i32), U64(u64), I64(i64),
    F32(f32), F64(f64), Bool(bool), Str(String), Arr(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match *self {
            Value::U8(v) => v as u64, Value::U16(v) => v as u64, Value::U32(v) => v as u64, Value::U64(v) => v,
            Value::I8(v) if v >= 0 => v as u64, Value::I16(v) if v >= 0 => v as u64,
            Value::I32(v) if v >= 0 => v as u64, Value::I64(v) if v >= 0 => v as u64,
            Value::Bool(v) => v as u64,
            _ => return None,
        })
    }
    pub fn as_f32(&self) -> Option<f32> {
        match *self { Value::F32(v) => Some(v), Value::F64(v) => Some(v as f32), _ => self.as_u64().map(|v| v as f32) }
    }
    pub fn as_str(&self) -> Option<&str> { if let Value::Str(s) = self { Some(s) } else { None } }
    pub fn as_arr(&self) -> Option<&[Value]> { if let Value::Arr(a) = self { Some(a) } else { None } }
}

#[derive(Clone, Debug)]
pub struct TensorInfo { pub name: String, pub dims: Vec<u64>, pub ty: u32, pub offset: u64, /** Shard index. */ pub file: usize }

pub struct Gguf {
    pub meta: BTreeMap<String, Value>,
    pub tensors: BTreeMap<String, TensorInfo>,
    /// Each shard's map and data offset (one entry unless the model is split).
    files: Vec<(Map, usize)>,
}

/// A mapped model file. `weights::preload` counted its bytes; unmapping takes them off the count, so a file that was
/// opened only to read its metadata, or the model before a switch, does not make the next one look too large for RAM.
struct Map(Mmap);

impl Drop for Map {
    fn drop(&mut self) { crate::weights::unmapped(self.0.len()); }
}

struct Cur<'a> { b: &'a [u8], p: usize }

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> anyhow::Result<&'a [u8]> {
        let end = self.p.checked_add(n).context("GGUF truncated")?;
        let s = self.b.get(self.p..end).context("GGUF truncated")?;
        self.p = end;
        Ok(s)
    }
    fn u32(&mut self) -> anyhow::Result<u32> { Ok(u32::from_le_bytes(self.take(4)?.try_into()?)) }
    fn u64(&mut self) -> anyhow::Result<u64> { Ok(u64::from_le_bytes(self.take(8)?.try_into()?)) }
    fn str(&mut self) -> anyhow::Result<String> {
        let n = usize::try_from(self.u64()?).context("GGUF string length out of range")?;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn val(&mut self, t: u32) -> anyhow::Result<Value> {
        Ok(match t {
            0 => Value::U8(self.take(1)?[0]),
            1 => Value::I8(self.take(1)?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.take(2)?.try_into()?)),
            3 => Value::I16(i16::from_le_bytes(self.take(2)?.try_into()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.str()?),
            9 => {
                let (et, n) = (self.u32()?, usize::try_from(self.u64()?).context("GGUF array length out of range")?);
                if n > self.b.len().saturating_sub(self.p) { bail!("GGUF array length exceeds remaining data"); }
                let mut a = Vec::with_capacity(n.min(1024));
                for _ in 0..n { a.push(self.val(et)?); }
                Value::Arr(a)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            _ => bail!("unknown GGUF value type {t}"),
        })
    }
}

impl Gguf {
    /// Open a GGUF; for a split model (`split.count` > 1), pass the `-00001-of-0000N` shard and the others are
    /// opened from the same directory.
    pub fn open(path: &Path) -> anyhow::Result<Gguf> {
        let mut g = Self::open_one(path, 0, true)?;
        let n = g.meta.get("split.count").and_then(Value::as_u64).unwrap_or(1) as usize;
        if n > 1 {
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
            let tag = format!("-00001-of-{n:05}.gguf");
            let stem = name.strip_suffix(&tag).with_context(|| format!("{}: split model; open its first shard (*{tag})", path.display()))?;
            for i in 1..n {
                let mut s = Self::open_one(&path.with_file_name(format!("{stem}-{:05}-of-{n:05}.gguf", i + 1)), i, true)?;
                g.files.append(&mut s.files);
                for (k, t) in std::mem::take(&mut s.tensors) { anyhow::ensure!(g.tensors.insert(k.clone(), t).is_none(), "tensor {k} in two shards"); }
            }
        }
        // The reads are queued by `weights::preload`, one call per file. Said here because nothing else shows it: the
        // pages are not mapped into the process, so its resident size does not grow.
        #[cfg(target_os = "linux")]
        if crate::weights::fits_ram() {
            eprintln!("page cache: reading {} MB of {} in the background (it shows as buff/cache, not as used memory)",
                g.files.iter().map(|(m, _)| m.0.len()).sum::<usize>() >> 20, path.file_name().and_then(|s| s.to_str()).unwrap_or("the model"));
        }
        Ok(g)
    }

    /// `general.architecture` of a model, from its first shard and without starting to read the file into the page
    /// cache: for a caller that only chooses which server gets the model.
    pub fn arch(path: &Path) -> Option<String> {
        Self::open_one(path, 0, false).ok()?.meta.get("general.architecture")?.as_str().map(str::to_owned)
    }

    fn open_one(path: &Path, file: usize, read: bool) -> anyhow::Result<Gguf> {
        let f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&f)? };
        // Expert rows are 2 MB apart in a file the page cache reads at random: huge pages save TLB misses.
        #[cfg(target_os = "linux")]
        if std::env::var_os("STRATA_HUGEPAGE").is_some() {
            let _ = mmap.advise(memmap2::Advice::HugePage);
        }
        crate::weights::preload(&mmap, read);
        let mmap = Map(mmap);
        let mut c = Cur { b: &mmap.0, p: 0 };
        if c.take(4)? != b"GGUF" { bail!("{}: not a GGUF file", path.display()); }
        let ver = c.u32()?;
        if !(2..=3).contains(&ver) { bail!("unsupported GGUF version {ver}"); }
        let (nt, nkv) = (c.u64()?, c.u64()?);
        let mut meta = BTreeMap::new();
        for _ in 0..nkv {
            let k = c.str()?;
            let t = c.u32()?;
            meta.insert(k, c.val(t)?);
        }
        let mut tensors = BTreeMap::new();
        for _ in 0..nt {
            let name = c.str()?;
            let nd = c.u32()?;
            let dims = (0..nd).map(|_| c.u64()).collect::<anyhow::Result<Vec<_>>>()?;
            let (ty, offset) = (c.u32()?, c.u64()?);
            tensors.insert(name.clone(), TensorInfo { name, dims, ty, offset, file });
        }
        let align = usize::try_from(meta.get("general.alignment").and_then(Value::as_u64).unwrap_or(32))
            .context("GGUF alignment out of range")?;
        let data_offset = c.p.checked_next_multiple_of(align).context("invalid GGUF alignment")?;
        // A cut-off download keeps a valid header; catch it here, not as a slice panic in `bytes`.
        for t in tensors.values() {
            let Ok(ty) = GgmlType::from_u32(t.ty) else { continue };  // Weights::validate reports it
            let n = t.dims.iter().skip(1).try_fold(ty.row_bytes(t.dims.first().copied().unwrap_or(0)) as u64, |a, &d| a.checked_mul(d));
            if n.and_then(|n| n.checked_add(t.offset)).is_none_or(|e| e > (mmap.0.len() as u64).saturating_sub(data_offset as u64)) {
                bail!("{}: GGUF truncated (tensor {} lies past the end of the file; re-download it)", path.display(), t.name);
            }
        }
        Ok(Gguf { meta, tensors, files: vec![(mmap, data_offset)] })
    }

    pub fn get(&self, key: &str) -> anyhow::Result<&Value> {
        self.meta.get(key).with_context(|| format!("missing GGUF key {key}"))
    }

    /// Raw bytes of a tensor. Panics on an unsupported type; `Weights::validate` checks types first.
    pub fn bytes(&self, t: &TensorInfo) -> &[u8] {
        let ty = GgmlType::from_u32(t.ty).unwrap();
        let n = ty.row_bytes(t.dims[0]) * t.dims[1..].iter().product::<u64>() as usize;
        let (mmap, data_offset) = &self.files[t.file];
        let o = data_offset + t.offset as usize;
        &mmap.0[o..o + n]
    }

    /// Raw bytes for auxiliary metadata tensors whose GGML type is not in `GgmlType` (e.g. EAGLE3 `d2t` I32/I64).
    pub fn raw_bytes(&self, t: &TensorInfo, byte_len: usize) -> anyhow::Result<&[u8]> {
        let (mmap, data_offset) = &self.files[t.file];
        let o = data_offset + t.offset as usize;
        mmap.0.get(o..o + byte_len).context("GGUF tensor bytes out of range")
    }
}

const GGUF_ALIGNMENT: usize = 32;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_Q4_0: u32 = 2;
const GGML_TYPE_Q8_0: u32 = 8;
const GGML_TYPE_I32: u32 = 26;
const GGML_TYPE_BF16: u32 = 30;

struct StTensor<'a> {
    shape: Vec<u64>,
    dtype: String,
    raw: &'a [u8],
}

fn parse_safetensors<'a>(mmap: &'a [u8], st_path: &Path) -> anyhow::Result<BTreeMap<String, StTensor<'a>>> {
    anyhow::ensure!(mmap.len() >= 8, "{}: safetensors file truncated", st_path.display());
    let hdr_len = usize::try_from(u64::from_le_bytes(mmap[0..8].try_into()?)).context("safetensors header length out of range")?;
    let data_off = 8usize.checked_add(hdr_len).context("safetensors header overflow")?;
    anyhow::ensure!(data_off <= mmap.len(), "{}: safetensors header past EOF", st_path.display());
    let hdr: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&mmap[8..data_off]).with_context(|| format!("parse safetensors header {}", st_path.display()))?;
    let mut tensors = BTreeMap::new();
    for (name, val) in hdr {
        if name == "__metadata__" { continue; }
        let obj = val.as_object().with_context(|| format!("tensor {name} entry is not an object"))?;
        let dtype = obj.get("dtype").and_then(|v| v.as_str()).with_context(|| format!("tensor {name} missing dtype"))?.to_string();
        let shape = obj.get("shape").and_then(|v| v.as_array()).with_context(|| format!("tensor {name} missing shape"))?
            .iter().map(|d| d.as_u64().context("bad tensor dim")).collect::<anyhow::Result<Vec<_>>>()?;
        let offs = obj.get("data_offsets").and_then(|v| v.as_array()).with_context(|| format!("tensor {name} missing data_offsets"))?;
        anyhow::ensure!(offs.len() == 2, "tensor {name}: expected 2 data_offsets");
        let start = usize::try_from(offs[0].as_u64().context("bad start offset")?)?;
        let end = usize::try_from(offs[1].as_u64().context("bad end offset")?)?;
        let raw = mmap.get(data_off + start..data_off + end).with_context(|| format!("tensor {name} offsets out of range"))?;
        tensors.insert(name, StTensor { shape, dtype, raw });
    }
    Ok(tensors)
}

fn convert_st_tensor<'a>(st: &StTensor<'a>, name: &str, dst_ty: u32) -> anyhow::Result<std::borrow::Cow<'a, [u8]>> {
    use half::{bf16, f16};
    use rayon::prelude::*;
    use std::borrow::Cow;
    if dst_ty == GGML_TYPE_Q8_0 || dst_ty == GGML_TYPE_Q4_0 {
        let cols = *st.shape.last().context("tensor missing cols")? as usize;
        anyhow::ensure!(cols % 32 == 0, "{name}: Q8_0/Q4_0 requires cols % 32 == 0, got {cols}");
        let elem_bytes = match st.dtype.as_str() { "BF16" | "F16" => 2, "F32" => 4, dt => bail!("{name}: unsupported dtype {dt} for Q8_0/Q4_0") };
        let blk_in = 32 * elem_bytes;
        anyhow::ensure!(st.raw.len() % blk_in == 0, "{name}: raw byte length not a multiple of {blk_in}");
        let n_blocks = st.raw.len() / blk_in;
        let out_blk = if dst_ty == GGML_TYPE_Q4_0 { 18 } else { 34 };
        let mut out = vec![0u8; n_blocks * out_blk];
        out.par_chunks_exact_mut(out_blk).enumerate().for_each(|(bi, dst)| {
            let src = &st.raw[bi * blk_in..(bi + 1) * blk_in];
            let mut vals = [0f32; 32];
            let mut amax = 0f32;
            let mut max_v = 0f32;
            for j in 0..32 {
                let v = match st.dtype.as_str() {
                    "BF16" => bf16::from_le_bytes([src[2 * j], src[2 * j + 1]]).to_f32(),
                    "F16" => f16::from_le_bytes([src[2 * j], src[2 * j + 1]]).to_f32(),
                    _ => f32::from_le_bytes(src[4 * j..4 * j + 4].try_into().unwrap()),
                };
                vals[j] = v;
                let av = v.abs();
                if av > amax {
                    amax = av;
                    max_v = v;
                }
            }
            if dst_ty == GGML_TYPE_Q4_0 {
                let d = max_v / -8.0;
                let inv = if d != 0.0 { 1.0 / d } else { 0.0 };
                dst[..2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
                for j in 0..16 {
                    let q0 = ((vals[j] * inv).round() as i32 + 8).clamp(0, 15) as u8;
                    let q1 = ((vals[j + 16] * inv).round() as i32 + 8).clamp(0, 15) as u8;
                    dst[2 + j] = q0 | (q1 << 4);
                }
            } else {
                let d = amax / 127.0;
                let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
                dst[..2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
                for j in 0..32 {
                    dst[2 + j] = (vals[j] * inv).round().clamp(-127.0, 127.0) as i8 as u8;
                }
            }
        });
        return Ok(Cow::Owned(out));
    }
    Ok(match (dst_ty, st.dtype.as_str()) {
        (GGML_TYPE_BF16, "BF16") | (GGML_TYPE_F32, "F32") => Cow::Borrowed(st.raw),
        (GGML_TYPE_BF16, "F32") => Cow::Owned(
            st.raw.chunks_exact(4).flat_map(|b| bf16::from_f32(f32::from_le_bytes(b.try_into().unwrap())).to_le_bytes()).collect(),
        ),
        (GGML_TYPE_BF16, "F16") => Cow::Owned(
            st.raw.chunks_exact(2).flat_map(|b| bf16::from_f32(f16::from_le_bytes(b.try_into().unwrap()).to_f32()).to_le_bytes()).collect(),
        ),
        (GGML_TYPE_F32, "BF16") => Cow::Owned(
            st.raw.chunks_exact(2).flat_map(|b| bf16::from_le_bytes(b.try_into().unwrap()).to_f32().to_le_bytes()).collect(),
        ),
        (GGML_TYPE_F32, "F16") => Cow::Owned(
            st.raw.chunks_exact(2).flat_map(|b| f16::from_le_bytes(b.try_into().unwrap()).to_f32().to_le_bytes()).collect(),
        ),
        (_, dt) => bail!("{name}: unsupported dtype {dt} for GGUF target type {dst_ty}"),
    })
}

fn push_gguf_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn push_gguf_kv(buf: &mut Vec<u8>, n: &mut u64, k: &str, ty: u32, payload: &[u8]) {
    push_gguf_str(buf, k);
    buf.extend_from_slice(&ty.to_le_bytes());
    buf.extend_from_slice(payload);
    *n += 1;
}

fn write_gguf_file(out_gguf: &Path, kvs: &[u8], nkv: u64, prepared: &[(String, Vec<u64>, u32, std::borrow::Cow<'_, [u8]>)]) -> anyhow::Result<()> {
    use std::io::{BufWriter, Write};
    let mut tensor_infos = Vec::new();
    let mut cur_off = 0usize;
    for (name, dims, ty, data) in prepared {
        cur_off = cur_off.next_multiple_of(GGUF_ALIGNMENT);
        push_gguf_str(&mut tensor_infos, name);
        tensor_infos.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for &d in dims { tensor_infos.extend_from_slice(&d.to_le_bytes()); }
        tensor_infos.extend_from_slice(&ty.to_le_bytes());
        tensor_infos.extend_from_slice(&(cur_off as u64).to_le_bytes());
        cur_off += data.len();
    }
    let mut meta_block = Vec::with_capacity(24 + kvs.len() + tensor_infos.len());
    meta_block.extend_from_slice(b"GGUF");
    meta_block.extend_from_slice(&3u32.to_le_bytes());
    meta_block.extend_from_slice(&(prepared.len() as u64).to_le_bytes());
    meta_block.extend_from_slice(&nkv.to_le_bytes());
    meta_block.extend_from_slice(kvs);
    meta_block.extend_from_slice(&tensor_infos);
    let hdr_pad = (GGUF_ALIGNMENT - (meta_block.len() % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT;

    if let Some(parent) = out_gguf.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let tmp_out = out_gguf.with_extension("gguf.tmp");
    {
        let f = std::fs::File::create(&tmp_out).with_context(|| format!("create {}", tmp_out.display()))?;
        let mut w = BufWriter::new(f);
        w.write_all(&meta_block)?;
        let zeros = [0u8; GGUF_ALIGNMENT];
        if hdr_pad > 0 { w.write_all(&zeros[..hdr_pad])?; }
        let mut written = 0usize;
        for (_, _, _, data) in prepared {
            let tpad = (GGUF_ALIGNMENT - (written % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT;
            if tpad > 0 {
                w.write_all(&zeros[..tpad])?;
                written += tpad;
            }
            w.write_all(data)?;
            written += data.len();
        }
        w.flush()?;
    }
    std::fs::rename(&tmp_out, out_gguf).with_context(|| format!("rename {} -> {}", tmp_out.display(), out_gguf.display()))?;
    Ok(())
}

/// Returns true if `cfg_path` describes a `DFlashDraftModel` checkpoint.
pub fn is_dflash_config(cfg_path: &Path) -> bool {
    let Ok(s) = std::fs::read_to_string(cfg_path) else { return false };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) else { return false };
    v.get("dflash_config").is_some()
        || v.get("architectures").and_then(|a| a.as_array()).is_some_and(|a| a.iter().any(|x| x.as_str() == Some("DFlashDraftModel")))
}

/// Convert a HuggingFace EAGLE3 `.safetensors` checkpoint + `config.json` into a GGUF v3 draft model.
///
/// Mirrors `tools/convert_eagle3_to_gguf.py` natively in Rust: 2D linear weights stay in BF16 (or convert from
/// F32/F16), 1D RMS norms convert to F32, NeoX RoPE is preserved (`eagle3.rope.neox = true`), and `d2t` relative
/// offsets are stored as absolute target vocabulary IDs (`I32`).
pub fn convert_safetensors_eagle3_to_gguf(
    st_path: &Path,
    cfg_path: &Path,
    out_gguf: &Path,
    target_layers_default: usize,
) -> anyhow::Result<()> {
    use std::borrow::Cow;

    const TENSOR_MAP: [(&str, &str, u32); 13] = [
        ("fc.weight", "fc.weight", GGML_TYPE_BF16),
        ("midlayer.input_layernorm.weight", "blk.0.attn_norm.weight", GGML_TYPE_F32),
        ("midlayer.hidden_norm.weight", "blk.0.attn_norm_2.weight", GGML_TYPE_F32),
        ("midlayer.self_attn.q_proj.weight", "blk.0.attn_q.weight", GGML_TYPE_BF16),
        ("midlayer.self_attn.k_proj.weight", "blk.0.attn_k.weight", GGML_TYPE_BF16),
        ("midlayer.self_attn.v_proj.weight", "blk.0.attn_v.weight", GGML_TYPE_BF16),
        ("midlayer.self_attn.o_proj.weight", "blk.0.attn_output.weight", GGML_TYPE_BF16),
        ("midlayer.post_attention_layernorm.weight", "blk.0.ffn_norm.weight", GGML_TYPE_F32),
        ("midlayer.mlp.gate_proj.weight", "blk.0.ffn_gate.weight", GGML_TYPE_BF16),
        ("midlayer.mlp.up_proj.weight", "blk.0.ffn_up.weight", GGML_TYPE_BF16),
        ("midlayer.mlp.down_proj.weight", "blk.0.ffn_down.weight", GGML_TYPE_BF16),
        ("norm.weight", "output_norm.weight", GGML_TYPE_F32),
        ("lm_head.weight", "output.weight", GGML_TYPE_BF16),
    ];

    let cfg_str = std::fs::read_to_string(cfg_path).with_context(|| format!("read {}", cfg_path.display()))?;
    let cfg: serde_json::Value = serde_json::from_str(&cfg_str).with_context(|| format!("parse {}", cfg_path.display()))?;
    let layer_cfg = cfg.get("transformer_layer_config").unwrap_or(&cfg);

    let req_u64 = |k: &str| -> anyhow::Result<u64> {
        layer_cfg.get(k).or_else(|| cfg.get(k)).and_then(|v| v.as_u64())
            .with_context(|| format!("{}: missing integer key {k}", cfg_path.display()))
    };
    let n_embd = req_u64("hidden_size")? as u32;
    let n_ff = req_u64("intermediate_size")? as u32;
    let n_head = req_u64("num_attention_heads")? as u32;
    let n_head_kv = req_u64("num_key_value_heads")? as u32;
    let head_dim = layer_cfg.get("head_dim").or_else(|| cfg.get("head_dim")).and_then(|v| v.as_u64()).map(|v| v as u32).unwrap_or_else(|| n_embd / n_head.max(1));
    let eps = layer_cfg.get("rms_norm_eps").or_else(|| cfg.get("rms_norm_eps")).and_then(|v| v.as_f64()).unwrap_or(1e-6) as f32;
    let rope_theta = layer_cfg.get("rope_theta").or_else(|| cfg.get("rope_theta")).and_then(|v| v.as_f64()).unwrap_or(1_000_000.0) as f32;
    let norm_before_residual = cfg.get("norm_before_residual").and_then(|v| v.as_bool()).unwrap_or(false);

    let st_file = std::fs::File::open(st_path).with_context(|| format!("open {}", st_path.display()))?;
    let mmap = unsafe { Mmap::map(&st_file)? };
    let tensors = parse_safetensors(&mmap, st_path)?;

    let draft_vocab = cfg.get("draft_vocab_size").and_then(|v| v.as_u64()).map(|v| v as usize)
        .or_else(|| tensors.get("d2t").and_then(|t| t.shape.first().copied()).map(|v| v as usize))
        .unwrap_or(32000);
    let extract_layers: Vec<i32> = match cfg.get("eagle_aux_hidden_state_layer_ids").and_then(|v| v.as_array()) {
        Some(arr) if !arr.is_empty() => arr.iter().map(|v| v.as_i64().map(|x| x as i32).context("bad extract layer")).collect::<anyhow::Result<_>>()?,
        _ => {
            let n = target_layers_default as i32;
            vec![2, n / 2, (n - 3).max(0)]
        }
    };
    anyhow::ensure!(extract_layers.len() == 3, "expected 3 EAGLE3 extract_layers, got {extract_layers:?}");

    let mut prepared: Vec<(String, Vec<u64>, u32, Cow<'_, [u8]>)> = Vec::with_capacity(TENSOR_MAP.len() + 1);
    for &(hf_name, gguf_name, dst_ty) in &TENSOR_MAP {
        let st = tensors.get(hf_name)
            .or_else(|| hf_name.strip_prefix("midlayer.").and_then(|rest| tensors.get(&format!("layers.0.{rest}"))))
            .with_context(|| format!("{}: missing EAGLE3 tensor {hf_name}", st_path.display()))?;
        let gguf_dims: Vec<u64> = st.shape.iter().copied().rev().collect();
        prepared.push((gguf_name.to_string(), gguf_dims, dst_ty, convert_st_tensor(st, hf_name, dst_ty)?));
    }

    let d2t = tensors.get("d2t").with_context(|| format!("{}: missing EAGLE3 d2t tensor", st_path.display()))?;
    anyhow::ensure!(d2t.shape.len() == 1 && d2t.shape[0] as usize == draft_vocab, "unexpected d2t shape {:?} (draft_vocab={draft_vocab})", d2t.shape);
    let abs_d2t: Vec<u8> = match d2t.dtype.as_str() {
        "I64" => d2t.raw.chunks_exact(8).enumerate().flat_map(|(i, b)| ((i as i64 + i64::from_le_bytes(b.try_into().unwrap())) as i32).to_le_bytes()).collect(),
        "I32" => d2t.raw.chunks_exact(4).enumerate().flat_map(|(i, b)| (i as i32 + i32::from_le_bytes(b.try_into().unwrap())).to_le_bytes()).collect(),
        dt => bail!("unexpected d2t dtype {dt}"),
    };
    prepared.push(("d2t".to_string(), vec![draft_vocab as u64], GGML_TYPE_I32, Cow::Owned(abs_d2t)));

    let mut kvs = Vec::new();
    let mut nkv = 0u64;
    let push_s = |buf: &mut Vec<u8>, n: &mut u64, k: &str, v: &str| { let mut p = Vec::new(); push_gguf_str(&mut p, v); push_gguf_kv(buf, n, k, GGUF_TYPE_STRING, &p); };
    let push_u = |buf: &mut Vec<u8>, n: &mut u64, k: &str, v: u32| push_gguf_kv(buf, n, k, GGUF_TYPE_UINT32, &v.to_le_bytes());
    let push_f = |buf: &mut Vec<u8>, n: &mut u64, k: &str, v: f32| push_gguf_kv(buf, n, k, GGUF_TYPE_FLOAT32, &v.to_le_bytes());
    let push_b = |buf: &mut Vec<u8>, n: &mut u64, k: &str, v: bool| push_gguf_kv(buf, n, k, GGUF_TYPE_BOOL, &[u8::from(v)]);

    push_s(&mut kvs, &mut nkv, "general.architecture", "eagle3");
    push_u(&mut kvs, &mut nkv, "general.alignment", GGUF_ALIGNMENT as u32);
    push_u(&mut kvs, &mut nkv, "eagle3.block_count", 1);
    push_u(&mut kvs, &mut nkv, "eagle3.embedding_length", n_embd);
    push_u(&mut kvs, &mut nkv, "eagle3.feed_forward_length", n_ff);
    push_u(&mut kvs, &mut nkv, "eagle3.attention.head_count", n_head);
    push_u(&mut kvs, &mut nkv, "eagle3.attention.head_count_kv", n_head_kv);
    push_u(&mut kvs, &mut nkv, "eagle3.attention.key_length", head_dim);
    push_f(&mut kvs, &mut nkv, "eagle3.attention.layer_norm_rms_epsilon", eps);
    push_u(&mut kvs, &mut nkv, "eagle3.rope.dimension_count", head_dim);
    push_f(&mut kvs, &mut nkv, "eagle3.rope.freq_base", rope_theta);
    push_b(&mut kvs, &mut nkv, "eagle3.rope.neox", true);
    push_b(&mut kvs, &mut nkv, "eagle3.norm_before_residual", norm_before_residual);
    let mut tl_payload = Vec::with_capacity(12 + extract_layers.len() * 4);
    tl_payload.extend_from_slice(&GGUF_TYPE_INT32.to_le_bytes());
    tl_payload.extend_from_slice(&(extract_layers.len() as u64).to_le_bytes());
    for &v in &extract_layers { tl_payload.extend_from_slice(&v.to_le_bytes()); }
    push_gguf_kv(&mut kvs, &mut nkv, "eagle3.target_layers", GGUF_TYPE_ARRAY, &tl_payload);

    write_gguf_file(out_gguf, &kvs, nkv, &prepared)?;
    eprintln!(
        "converted EAGLE3 safetensors -> {} ({:.1} MiB, extract_layers={extract_layers:?}, heads={n_head}x{head_dim}, kv={n_head_kv}, ff={n_ff}, draft_vocab={draft_vocab})",
        out_gguf.display(),
        std::fs::metadata(out_gguf).map(|m| m.len() as f64 / 1048576.0).unwrap_or(0.0),
    );
    Ok(())
}

/// Convert a HuggingFace `DFlashDraftModel` `.safetensors` checkpoint + `config.json` into a Q8_0 GGUF v3 file.
///
/// 2D projections (`fc.weight`, per-layer `q/k/v/o_proj` and `gate/up/down_proj`) are quantized to `Q8_0` (`cols % 32 == 0`)
/// so the 6-layer block-diffusion model runs directly on `16x16x16` INT8 tensor cores (`gemmi_16` / `gemmi_s_16`),
/// while 1D RMS norms and `q/k_norm` weights are converted to `F32`.
pub fn convert_safetensors_dflash_to_gguf(
    st_path: &Path,
    cfg_path: &Path,
    out_gguf: &Path,
    target_layers_default: usize,
) -> anyhow::Result<()> {
    use std::borrow::Cow;

    let cfg_str = std::fs::read_to_string(cfg_path).with_context(|| format!("read {}", cfg_path.display()))?;
    let cfg: serde_json::Value = serde_json::from_str(&cfg_str).with_context(|| format!("parse {}", cfg_path.display()))?;
    let df_cfg = cfg.get("dflash_config").unwrap_or(&cfg);
    let req_u64 = |k: &str| -> anyhow::Result<u64> {
        cfg.get(k).and_then(|v| v.as_u64()).with_context(|| format!("{}: missing integer key {k}", cfg_path.display()))
    };
    let n_layer = req_u64("num_hidden_layers")? as u32;
    let n_embd = req_u64("hidden_size")? as u32;
    let n_ff = req_u64("intermediate_size")? as u32;
    let n_head = req_u64("num_attention_heads")? as u32;
    let n_head_kv = req_u64("num_key_value_heads")? as u32;
    let head_dim = cfg.get("head_dim").and_then(|v| v.as_u64()).map(|v| v as u32).unwrap_or_else(|| n_embd / n_head.max(1));
    let n_rot = cfg.get("rope_dimension_count").and_then(|v| v.as_u64()).map(|v| v as u32).unwrap_or(head_dim);
    let eps = cfg.get("rms_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-6) as f32;
    let rope_theta = cfg.get("rope_parameters").and_then(|r| r.get("rope_theta")).or_else(|| cfg.get("rope_theta")).and_then(|v| v.as_f64()).unwrap_or(10_000_000.0) as f32;
    let sliding_window = cfg.get("sliding_window").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let block_size = df_cfg.get("block_size").or_else(|| cfg.get("block_size")).and_then(|v| v.as_u64()).unwrap_or(16) as u32;
    let mask_token_id = df_cfg.get("mask_token_id").or_else(|| cfg.get("mask_token_id")).and_then(|v| v.as_u64()).context("missing dflash_config.mask_token_id")? as u32;
    let input_embedding_scale = df_cfg.get("input_embedding_scale").or_else(|| cfg.get("input_embedding_scale")).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
    let output_multiplier = df_cfg.get("output_multiplier").or_else(|| cfg.get("output_multiplier")).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
    let conv_kernel_size = df_cfg.get("conv_kernel_size").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let conv_group_size = df_cfg.get("conv_group_size").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let selector_rank = df_cfg.get("selector_rank").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let selector_top_k = df_cfg.get("selector_top_k").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

    let n_tgt = cfg.get("num_target_layers").and_then(|v| v.as_u64()).unwrap_or(target_layers_default as u64) as usize;
    let target_layers: Vec<i32> = match df_cfg.get("target_layer_ids").and_then(|v| v.as_array()) {
        Some(arr) if !arr.is_empty() => arr.iter().map(|v| v.as_i64().map(|x| x as i32).context("bad target_layer_ids")).collect::<anyhow::Result<_>>()?,
        _ => if n_layer == 1 { vec![(n_tgt / 2) as i32] } else {
            let (s, e) = (1usize, n_tgt.saturating_sub(3));
            (0..n_layer as usize).map(|i| (((s + (i * (e - s)) / (n_layer as usize - 1).max(1)) as f64).round()) as i32).collect()
        }
    };
    let layer_swa: Vec<u32> = match cfg.get("layer_types").and_then(|v| v.as_array()) {
        Some(arr) if arr.len() >= n_layer as usize => arr[..n_layer as usize].iter().map(|v| u32::from(v.as_str() == Some("sliding_attention"))).collect(),
        _ => vec![0; n_layer as usize],
    };

    let st_file = std::fs::File::open(st_path).with_context(|| format!("open {}", st_path.display()))?;
    let mmap = unsafe { Mmap::map(&st_file)? };
    let tensors = parse_safetensors(&mmap, st_path)?;

    let use_q4 = out_gguf.file_name().and_then(|s| s.to_str()).is_some_and(|s| s.contains("q4_0"))
        || std::env::var_os("STRATA_DFLASH_Q4").is_some_and(|v| v != "0");
    let blk_ty = if use_q4 { GGML_TYPE_Q4_0 } else { GGML_TYPE_Q8_0 };
    let mut prepared: Vec<(String, Vec<u64>, u32, Cow<'_, [u8]>)> = Vec::with_capacity(3 + 10 * n_layer as usize);
    let mut push_t = |hf_name: &str, gguf_name: String, dst_ty: u32| -> anyhow::Result<()> {
        let st = tensors.get(hf_name).with_context(|| format!("{}: missing DFlash tensor {hf_name}", st_path.display()))?;
        let gguf_dims: Vec<u64> = st.shape.iter().copied().rev().collect();
        let cols = *gguf_dims.first().unwrap_or(&0) as usize;
        let ty = if matches!(dst_ty, GGML_TYPE_Q8_0 | GGML_TYPE_Q4_0) && cols % 32 != 0 { GGML_TYPE_BF16 } else { dst_ty };
        prepared.push((gguf_name, gguf_dims, ty, convert_st_tensor(st, hf_name, ty)?));
        Ok(())
    };
    push_t("fc.weight", "dflash.fc.weight".into(), GGML_TYPE_Q8_0)?;
    push_t("hidden_norm.weight", "dflash.hidden_norm.weight".into(), GGML_TYPE_F32)?;
    push_t("norm.weight", "dflash.norm.weight".into(), GGML_TYPE_F32)?;
    for il in 0..n_layer {
        push_t(&format!("layers.{il}.input_layernorm.weight"), format!("dflash.blk.{il}.attn_norm.weight"), GGML_TYPE_F32)?;
        push_t(&format!("layers.{il}.post_attention_layernorm.weight"), format!("dflash.blk.{il}.ffn_norm.weight"), GGML_TYPE_F32)?;
        push_t(&format!("layers.{il}.self_attn.q_norm.weight"), format!("dflash.blk.{il}.attn_q_norm.weight"), GGML_TYPE_F32)?;
        push_t(&format!("layers.{il}.self_attn.k_norm.weight"), format!("dflash.blk.{il}.attn_k_norm.weight"), GGML_TYPE_F32)?;
        push_t(&format!("layers.{il}.self_attn.q_proj.weight"), format!("dflash.blk.{il}.attn_q.weight"), blk_ty)?;
        push_t(&format!("layers.{il}.self_attn.k_proj.weight"), format!("dflash.blk.{il}.attn_k.weight"), blk_ty)?;
        push_t(&format!("layers.{il}.self_attn.v_proj.weight"), format!("dflash.blk.{il}.attn_v.weight"), blk_ty)?;
        push_t(&format!("layers.{il}.self_attn.o_proj.weight"), format!("dflash.blk.{il}.attn_output.weight"), blk_ty)?;
        push_t(&format!("layers.{il}.mlp.gate_proj.weight"), format!("dflash.blk.{il}.ffn_gate.weight"), blk_ty)?;
        push_t(&format!("layers.{il}.mlp.up_proj.weight"), format!("dflash.blk.{il}.ffn_up.weight"), blk_ty)?;
        push_t(&format!("layers.{il}.mlp.down_proj.weight"), format!("dflash.blk.{il}.ffn_down.weight"), blk_ty)?;
        if conv_kernel_size > 0 {
            push_t(&format!("layers.{il}.attention_conv.base_kernel"), format!("dflash.blk.{il}.attn_conv_base.weight"), GGML_TYPE_F32)?;
            push_t(&format!("layers.{il}.attention_conv.kernel_projection.weight"), format!("dflash.blk.{il}.attn_conv_proj.weight"), blk_ty)?;
            push_t(&format!("layers.{il}.mlp_conv.base_kernel"), format!("dflash.blk.{il}.mlp_conv_base.weight"), GGML_TYPE_F32)?;
            push_t(&format!("layers.{il}.mlp_conv.kernel_projection.weight"), format!("dflash.blk.{il}.mlp_conv_proj.weight"), blk_ty)?;
        }
    }
    if selector_rank > 0 {
        push_t("candidate_selector.hidden_projection.weight", "dflash.selector.hidden_proj.weight".into(), GGML_TYPE_Q8_0)?;
        push_t("candidate_selector.predecessor_codebook", "dflash.selector.pred_codebook.weight".into(), GGML_TYPE_Q8_0)?;
        push_t("candidate_selector.successor_codebook", "dflash.selector.succ_codebook.weight".into(), GGML_TYPE_Q8_0)?;
    }

    let mut kvs = Vec::new();
    let mut nkv = 0u64;
    let push_s = |buf: &mut Vec<u8>, n: &mut u64, k: &str, v: &str| { let mut p = Vec::new(); push_gguf_str(&mut p, v); push_gguf_kv(buf, n, k, GGUF_TYPE_STRING, &p); };
    let push_u = |buf: &mut Vec<u8>, n: &mut u64, k: &str, v: u32| push_gguf_kv(buf, n, k, GGUF_TYPE_UINT32, &v.to_le_bytes());
    let push_f = |buf: &mut Vec<u8>, n: &mut u64, k: &str, v: f32| push_gguf_kv(buf, n, k, GGUF_TYPE_FLOAT32, &v.to_le_bytes());

    push_s(&mut kvs, &mut nkv, "general.architecture", "dflash");
    push_u(&mut kvs, &mut nkv, "general.alignment", GGUF_ALIGNMENT as u32);
    push_u(&mut kvs, &mut nkv, "dflash.block_count", n_layer);
    push_u(&mut kvs, &mut nkv, "dflash.block_size", block_size);
    push_u(&mut kvs, &mut nkv, "dflash.mask_token_id", mask_token_id);
    push_u(&mut kvs, &mut nkv, "dflash.embedding_length", n_embd);
    push_u(&mut kvs, &mut nkv, "dflash.feed_forward_length", n_ff);
    push_u(&mut kvs, &mut nkv, "dflash.attention.head_count", n_head);
    push_u(&mut kvs, &mut nkv, "dflash.attention.head_count_kv", n_head_kv);
    push_u(&mut kvs, &mut nkv, "dflash.attention.key_length", head_dim);
    push_f(&mut kvs, &mut nkv, "dflash.attention.layer_norm_rms_epsilon", eps);
    push_u(&mut kvs, &mut nkv, "dflash.attention.sliding_window", sliding_window);
    push_u(&mut kvs, &mut nkv, "dflash.rope.dimension_count", n_rot);
    push_f(&mut kvs, &mut nkv, "dflash.rope.freq_base", rope_theta);
    push_f(&mut kvs, &mut nkv, "dflash.input_embedding_scale", input_embedding_scale);
    push_f(&mut kvs, &mut nkv, "dflash.output_multiplier", output_multiplier);
    if conv_kernel_size > 0 {
        push_u(&mut kvs, &mut nkv, "dflash.conv_kernel_size", conv_kernel_size);
        push_u(&mut kvs, &mut nkv, "dflash.conv_group_size", conv_group_size);
    }
    if selector_rank > 0 {
        push_u(&mut kvs, &mut nkv, "dflash.selector_rank", selector_rank);
        push_u(&mut kvs, &mut nkv, "dflash.selector_top_k", selector_top_k);
    }
    let mut tl_payload = Vec::with_capacity(12 + target_layers.len() * 4);
    tl_payload.extend_from_slice(&GGUF_TYPE_INT32.to_le_bytes());
    tl_payload.extend_from_slice(&(target_layers.len() as u64).to_le_bytes());
    for &v in &target_layers { tl_payload.extend_from_slice(&v.to_le_bytes()); }
    push_gguf_kv(&mut kvs, &mut nkv, "dflash.target_layers", GGUF_TYPE_ARRAY, &tl_payload);
    let mut sw_payload = Vec::with_capacity(12 + layer_swa.len() * 4);
    sw_payload.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
    sw_payload.extend_from_slice(&(layer_swa.len() as u64).to_le_bytes());
    for &v in &layer_swa { sw_payload.extend_from_slice(&v.to_le_bytes()); }
    push_gguf_kv(&mut kvs, &mut nkv, "dflash.attention.sliding_window_pattern", GGUF_TYPE_ARRAY, &sw_payload);

    write_gguf_file(out_gguf, &kvs, nkv, &prepared)?;
    eprintln!(
        "converted DFlash safetensors -> {} ({:.1} MiB, layers={n_layer}, block={block_size}, mask={mask_token_id}, target_layers={target_layers:?}, heads={n_head}x{head_dim}, kv={n_head_kv}, ff={n_ff}, Q8_0)",
        out_gguf.display(),
        std::fs::metadata(out_gguf).map(|m| m.len() as f64 / 1048576.0).unwrap_or(0.0),
    );
    Ok(())
}
