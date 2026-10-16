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
#[test] fn rejects_unknown() { assert!(GgmlType::from_u32(2).is_err()); } // Q4_0 is not supported

/// One IQ4_NL block: f16 scale, then 16 bytes holding elements j (low nibble) and j + 16 (high nibble).
#[test]
fn iq4_nl_block() {
    const K: [f32; 16] = [-127., -104., -83., -65., -49., -35., -22., -10., 1., 13., 25., 38., 53., 69., 89., 113.];
    let ty = GgmlType::from_u32(20).unwrap();
    assert_eq!((ty.name(), ty.row_bytes(32)), ("IQ4_NL", 18));
    let mut b = half::f16::from_f32(0.5).to_le_bytes().to_vec();
    b.extend((0..16u8).map(|j| j | (15 - j) << 4));
    let mut y = [0f32; 32];
    dequant_row(ty, &b, &mut y);
    for j in 0..16 { assert_eq!((y[j], y[j + 16]), (0.5 * K[j], 0.5 * K[15 - j]), "element {j}"); }
}
#[test] fn iq2_xs() { check("IQ2_XS", GgmlType::IQ2XS) }
#[test] fn iq3_xxs() { check("IQ3_XXS", GgmlType::IQ3XXS) }
#[test] fn iq4_xs() { check("IQ4_XS", GgmlType::IQ4XS) }

#[test]
fn matvec_matches_rowwise_dequant() {
    let Some(p) = common::model_path() else { return };
    let g = rusting_strata::gguf::Gguf::open(&p).unwrap();
    let w = rusting_strata::weights::Weights::new(&g).unwrap();
    w.validate().unwrap();
    let t = w.layer(0, "ffn_down_exps.weight").unwrap().expert(7, 256); // IQ3_XXS, 2048 rows x 512
    assert_eq!((t.rows, t.cols), (2048, 512));
    let x: Vec<f32> = (0..512).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
    let mut y = vec![0f32; 2048];
    t.matvec(&x, &mut y);
    let mut row = vec![0f32; 512];
    for r in [0, 1, 1000, 2047] {
        t.row(r, &mut row);
        let want: f32 = row.iter().zip(&x).map(|(a, b)| a * b).sum();
        assert!((y[r] - want).abs() < 1e-4, "row {r}");
    }
    // Expert slice must point at the right bytes: expert 7 row 0 == full tensor row 7*2048.
    let full = w.layer(0, "ffn_down_exps.weight").unwrap();
    let mut a = vec![0f32; 512];
    full.row(7 * 2048, &mut a);
    t.row(0, &mut row);
    assert_eq!(a, row);
}
