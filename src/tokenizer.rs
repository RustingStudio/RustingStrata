//! GPT-2 style byte-level BPE built from GGUF tokenizer metadata.

use crate::gguf::Gguf;
use anyhow::{Context, anyhow, bail};
use std::collections::HashMap;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{AddedToken, PreTokenizerWrapper, SplitDelimiterBehavior};

/// llama.cpp `LLAMA_VOCAB_PRE_TYPE_QWEN35` pre-tokenizer regex.
const QWEN35_RE: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// llama.cpp `LLAMA_VOCAB_PRE_TYPE_QWEN2` pre-tokenizer regex (qwen3, qwen3moe, qwen3next).
const QWEN2_RE: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// llama.cpp `LLAMA_VOCAB_PRE_TYPE_CHATGLM4` (GLM-4/5): digits in groups of up to 3.
const GLM4_RE: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// llama.cpp `LLAMA_VOCAB_PRE_TYPE_JOYAI_LLM` (DeepSeek-V4): three splits applied one after another.
const JOYAI_RE: [&str; 3] = [
    r"\p{N}{1,3}",
    "[一-龥\u{3040}-\u{309f}\u{30a0}-\u{30ff}]+",
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
];

/// llama.cpp `LLAMA_VOCAB_PRE_TYPE_FLINT`: digits in groups of three, CJK runs, then Llama-3-style words.
/// GPT-4o (o200k), used by Inkling 2: case-aware words with an optional contraction, digits in groups of three.
const GPT4O_RE: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

const FLINT_RE: [&str; 3] = [
    r"\p{N}{1,3}",
    "[一-龥぀-ゟ゠-ヿ]+",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
];

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    eos: Vec<u32>,
    /// The BOS id when the GGUF asks for one at the start (`tokenizer.ggml.add_bos_token`).
    bos: Option<u32>,
    /// Control and user-defined tokens (types 3 and 4).
    special: Vec<bool>,
    /// Gemma 4 vocabulary: U+2581 for a space, `<0xXX>` for a byte.
    spm: bool,
    bytes: std::sync::OnceLock<Vec<Vec<u8>>>,
}

/// The BOS id when the GGUF asks for one at the start of a prompt.
fn add_bos(g: &Gguf) -> Option<u32> {
    let on = g.meta.get("tokenizer.ggml.add_bos_token").and_then(|v| v.as_u64()) == Some(1);
    on.then(|| g.meta.get("tokenizer.ggml.bos_token_id").and_then(|v| v.as_u64()).map(|v| v as u32)).flatten()
}

/// GPT-2 byte-level BPE writes byte `b` as one printable char: itself for the printable ones, 256 + n for the n-th
/// other byte. This is its inverse.
fn gpt2_byte(c: char) -> Option<u8> {
    let printable = |b: u32| (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
    let c = c as u32;
    if printable(c) { return Some(c as u8); }
    (0..256u32).filter(|&b| !printable(b)).nth(c.checked_sub(256)? as usize).map(|b| b as u8)
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> anyhow::Result<Tokenizer> {
        let model = g.get("tokenizer.ggml.model")?.as_str().unwrap_or("");
        if model == "gemma4" { return Self::gemma4(g); }
        if model != "gpt2" { bail!("unsupported tokenizer model '{model}'"); }
        let pre = g.get("tokenizer.ggml.pre")?.as_str().unwrap_or("");
        let res: &[&str] = match pre {
            "qwen35" => &[QWEN35_RE], "qwen2" => &[QWEN2_RE], "joyai-llm" | "hy_v4" => &JOYAI_RE,
            "flint" => &FLINT_RE,
            "glm4" | "glm5" | "chatglm-bpe" => &[GLM4_RE],
            "inkling2" => &[GPT4O_RE],
            _ => bail!("unsupported pre-tokenizer '{pre}'"),
        };
        let strs = |k: &str| -> anyhow::Result<Vec<String>> {
            g.get(k)?.as_arr().with_context(|| format!("{k}: not an array"))?
                .iter().map(|v| v.as_str().map(str::to_string).with_context(|| format!("{k}: not a string"))).collect()
        };
        let tokens = strs("tokenizer.ggml.tokens")?;
        let merges = strs("tokenizer.ggml.merges")?
            .into_iter()
            .map(|m| m.split_once(' ').map(|(a, b)| (a.to_string(), b.to_string())).context("bad merge"))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let vocab: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        let bpe = BPE::builder().vocab_and_merges(vocab, merges).build().map_err(|e| anyhow!("{e}"))?;

        let mut inner = tokenizers::Tokenizer::new(bpe);
        let mut pre_tok = Vec::new();
        for re in res {
            let split = Split::new(SplitPattern::Regex((*re).into()), SplitDelimiterBehavior::Isolated, false)
                .map_err(|e| anyhow!("{e}"))?;
            pre_tok.push(PreTokenizerWrapper::Split(split));
        }
        pre_tok.push(PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)));
        inner.with_pre_tokenizer(Some(Sequence::new(pre_tok)));
        inner.with_decoder(Some(ByteLevelDecoder::default()));

        let types = g.get("tokenizer.ggml.token_type")?.as_arr().context("token_type")?;
        let special: Vec<AddedToken> = types.iter().enumerate()
            .filter(|(_, t)| matches!(t.as_u64(), Some(3 | 4)))
            .map(|(i, _)| AddedToken::from(tokens[i].clone(), true))
            .collect();
        inner.add_special_tokens(&special);
        let special = types.iter().map(|t| matches!(t.as_u64(), Some(3 | 4))).collect();

        let mut eos: Vec<u32> = g.meta.get("tokenizer.ggml.eos_token_id").and_then(|v| v.as_u64()).map(|v| v as u32).into_iter().collect();
        if let Some(a) = g.meta.get("tokenizer.ggml.eog_token_ids").and_then(|v| v.as_arr()) {
            for v in a { if let Some(id) = v.as_u64().map(|x| x as u32) && !eos.contains(&id) { eos.push(id); } }
        }
        for t in ["<|im_end|>", "<|endoftext|>"] {
            if let Some(id) = inner.token_to_id(t) && !eos.contains(&id) { eos.push(id); }
        }
        if pre == "inkling2" && let Some(id) = inner.token_to_id("<|content_model_end_sampling|>") && !eos.contains(&id) { eos.push(id); }
        if pre == "flint" {
            // llama.cpp treats these turn markers as end of generation for Flint.
            for t in ["<|user|>", "<|observation|>"] {
                if let Some(id) = inner.token_to_id(t) && !eos.contains(&id) { eos.push(id); }
            }
        }
        Ok(Tokenizer { inner, eos, bos: add_bos(g), special, spm: false, bytes: Default::default() })
    }

    /// Gemma 4: SPM-style BPE. Spaces become U+2581, merges run over each newline-free run of raw UTF-8, unknown bytes fall back to <0xXX>.
    fn gemma4(g: &Gguf) -> anyhow::Result<Tokenizer> {
        use tokenizers::decoders::{byte_fallback::ByteFallback, fuse::Fuse, sequence::Sequence as DSeq, DecoderWrapper};
        use tokenizers::normalizers::Replace;
        let tokens: Vec<String> = g.get("tokenizer.ggml.tokens")?.as_arr().context("tokens")?
            .iter().map(|v| v.as_str().map(str::to_string).context("token")).collect::<anyhow::Result<_>>()?;
        let merges = g.get("tokenizer.ggml.merges")?.as_arr().context("merges")?.iter()
            .map(|v| v.as_str().and_then(|m| m.split_once(' ')).map(|(a, b)| (a.to_string(), b.to_string())).context("bad merge"))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let vocab: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        let bpe = BPE::builder().vocab_and_merges(vocab, merges).byte_fallback(true).unk_token("<unk>".into()).build().map_err(|e| anyhow!("{e}"))?;
        let mut inner = tokenizers::Tokenizer::new(bpe);
        inner.with_normalizer(Some(Replace::new(" ", "\u{2581}").map_err(|e| anyhow!("{e}"))?));
        let split = Split::new(SplitPattern::Regex("[^\\n]+|[\\n]+".into()), SplitDelimiterBehavior::Isolated, false).map_err(|e| anyhow!("{e}"))?;
        inner.with_pre_tokenizer(Some(split));
        inner.with_decoder(Some(DSeq::new(vec![
            DecoderWrapper::Replace(tokenizers::normalizers::replace::Replace::new("\u{2581}", " ").map_err(|e| anyhow!("{e}"))?),
            DecoderWrapper::ByteFallback(ByteFallback::new()),
            DecoderWrapper::Fuse(Fuse::new()),
        ])));
        let types = g.get("tokenizer.ggml.token_type")?.as_arr().context("token_type")?;
        let special: Vec<AddedToken> = types.iter().enumerate()
            .filter(|(_, t)| matches!(t.as_u64(), Some(3 | 4)))
            .map(|(i, _)| AddedToken::from(tokens[i].clone(), true))
            .collect();
        inner.add_special_tokens(&special);
        let special = types.iter().map(|t| matches!(t.as_u64(), Some(3 | 4))).collect();
        let mut eos: Vec<u32> = g.meta.get("tokenizer.ggml.eos_token_id").and_then(|v| v.as_u64()).map(|v| v as u32).into_iter().collect();
        for t in ["<turn|>", "<eos>"] {
            if let Some(id) = inner.token_to_id(t) && !eos.contains(&id) { eos.push(id); }
        }
        Ok(Tokenizer { inner, eos, bos: add_bos(g), special, spm: true, bytes: Default::default() })
    }

    pub fn token_id(&self, t: &str) -> Option<u32> { self.inner.token_to_id(t) }

    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        Ok(self.inner.encode(text, false).map_err(|e| anyhow!("{e}"))?.get_ids().to_vec())
    }

    /// `encode` with the BOS token in front when `add_special` and the model asks for one (llama.cpp `/tokenize`).
    pub fn encode_special(&self, text: &str, add_special: bool) -> anyhow::Result<Vec<u32>> {
        let mut ids = self.encode(text)?;
        if add_special && let Some(b) = self.bos { ids.insert(0, b); }
        Ok(ids)
    }

    /// Whether `id` is a control or user-defined token (`<|im_end|>`, `<think>`).
    pub fn is_special(&self, id: u32) -> bool { self.special.get(id as usize).copied().unwrap_or(false) }

    /// The bytes token `id` stands for: its text for a special token, nothing for an id outside the vocabulary.
    /// A token may hold part of a UTF-8 character.
    pub fn token_bytes(&self, id: u32) -> &[u8] {
        let all = self.bytes.get_or_init(|| (0..self.vocab_size() as u32).map(|i| self.bytes_of(i)).collect());
        all.get(id as usize).map_or(&[], |b| b.as_slice())
    }

    fn bytes_of(&self, id: u32) -> Vec<u8> {
        let Some(t) = self.inner.id_to_token(id) else { return vec![] };
        if self.is_special(id) { return t.into_bytes(); }
        if self.spm {
            if let Some(h) = t.strip_prefix("<0x").and_then(|h| h.strip_suffix('>')) && let Ok(b) = u8::from_str_radix(h, 16) { return vec![b]; }
            return t.replace('\u{2581}', " ").into_bytes();
        }
        // A char outside the byte map (an added non-special token) stands for itself.
        let mut out = vec![];
        for c in t.chars() {
            match gpt2_byte(c) { Some(b) => out.push(b), None => out.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes()) }
        }
        out
    }

    pub fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        self.inner.decode(ids, false).map_err(|e| anyhow!("{e}"))
    }

    pub fn eos_ids(&self) -> Vec<u32> { self.eos.clone() }

    pub fn vocab_size(&self) -> usize { self.inner.get_vocab_size(true) }
}

#[cfg(test)]
mod tests {
    #[test]
    fn gpt2_byte_map_is_the_inverse() {
        assert_eq!(super::gpt2_byte('a'), Some(b'a'));
        assert_eq!(super::gpt2_byte('\u{120}'), Some(b' '));  // Ġ
        assert_eq!(super::gpt2_byte('\u{10a}'), Some(b'\n'));  // Ċ
        assert_eq!(super::gpt2_byte('\u{100}'), Some(0));
        assert_eq!(super::gpt2_byte('\u{143}'), Some(173));  // the last of the 68 remapped bytes
        assert_eq!(super::gpt2_byte('\u{144}'), None);
    }
}
