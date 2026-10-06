use rusting_strata::ops::*;

#[test]
fn rms() {
    let mut o = [0f32; 2];
    rms_norm(&[3.0, 4.0], &[1.0, 2.0], 0.0, &mut o);
    let s = 12.5f32.sqrt();
    assert!((o[0] - 3.0 / s).abs() < 1e-6 && (o[1] - 8.0 / s).abs() < 1e-6);
}
#[test]
fn l2() {
    let mut x = [3.0, 4.0];
    l2_norm(&mut x, 1e-6);
    assert!((x[0] - 0.6).abs() < 1e-6);
}
#[test]
fn softplus_values() {
    assert!((softplus(30.0) - 30.0).abs() < 1e-5);
    assert!((softplus(0.0) - 2f32.ln()).abs() < 1e-6);
}
#[test]
fn rope_pos0_identity() {
    let mut x: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let y = x.clone();
    rope_neox(&mut x, 4, 0, 1e7);
    assert_eq!(x, y);
}
#[test]
fn rope_rotates_pairs() {
    let mut x = vec![1.0, 0.0, 0.0, 0.0, 9.0]; // n_rot 4: pairs (0,2), (1,3)
    rope_neox(&mut x, 4, 1, 10000.0); // theta0 = 1 rad
    assert!((x[0] - 1f32.cos()).abs() < 1e-6 && (x[2] - 1f32.sin()).abs() < 1e-6 && x[4] == 9.0);
}
#[test]
fn topk() { assert_eq!(top_k(&[0.1, 0.5, 0.2, 0.5], 2), vec![1, 3]); }
#[test]
fn softmax_sums_to_one() {
    let mut x = [1.0, 2.0, 3.0];
    softmax(&mut x);
    assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6 && x[2] > x[1]);
}
