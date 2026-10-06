mod common;
use rusting_strata::quant::{GgmlType, dequant_row};

fn check(name: &str, ty: GgmlType) {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/");
    let raw = std::fs::read(format!("{dir}{name}.raw")).unwrap();
    let want: Vec<f32> = std::fs::read(format!("{dir}{name}.f32")).unwrap()
        .chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(raw.len(), ty.row_bytes(want.len() as u64), "{name} size");
    let mut got = vec![0f32; want.len()];
    dequant_row(ty, &raw, &mut got);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!((g - w).abs() <= 1e-6 * w.abs().max(1.0), "{name}[{i}]: got {g} want {w}");
    }
}

#[test] fn f32_() { check("F32", GgmlType::F32) }
#[test] fn bf16() { check("BF16", GgmlType::BF16) }
#[test] fn q8_0() { check("Q8_0", GgmlType::Q8_0) }
#[test] fn q2_k() { check("Q2_K", GgmlType::Q2K) }
#[test] fn q3_k() { check("Q3_K", GgmlType::Q3K) }
#[test] fn q4_k() { check("Q4_K", GgmlType::Q4K) }
#[test] fn q5_k() { check("Q5_K", GgmlType::Q5K) }
#[test] fn q6_k() { check("Q6_K", GgmlType::Q6K) }
#[test] fn rejects_unknown() { assert!(GgmlType::from_u32(16).is_err()); } // IQ2_XXS is not in v1
