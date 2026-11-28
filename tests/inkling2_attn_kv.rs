//! Inkling 2 decode attention on a quantized KV cache (CUDA). Its own process: the kernel module is compiled per KV
//! format, and the kernel choice is read from the environment when the `Gpu` is made.
mod common;
use common::{down, up};
use rusting_strata::backend::Backend;
use rusting_strata::cuda::Gpu;

fn xs(n: usize) -> Vec<f32> { (0..n).map(|i| ((i as f32 * 0.7548777).fract() - 0.5) * 2.0).collect() }

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f32 = b.iter().map(|y| y * y).sum();
    (num / den.max(1e-30)).sqrt()
}

fn hash32(v: &[f32]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for x in v { for b in x.to_bits().to_le_bytes() { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); } }
    h
}

/// A `Gpu` whose K and V caches are both `kv`, with the fast Inkling kernels on or off and the scalar decode
/// attention kept or not.
fn open(kv: &str, fast: bool, old: bool) -> Option<Gpu> {
    unsafe {
        std::env::set_var("STRATA_KV_K", kv);
        std::env::set_var("STRATA_KV_V", kv);
        std::env::set_var("STRATA_KV", kv);
        if kv == "q8" { std::env::set_var("STRATA_KV_Q8", "1"); } else { std::env::remove_var("STRATA_KV_Q8"); }
        std::env::set_var("STRATA_DS_FAST_KERNELS", if fast { "1" } else { "0" });
        if old { std::env::set_var("STRATA_DS_ATTN_OLD", "1"); } else { std::env::remove_var("STRATA_DS_ATTN_OLD"); }
    }
    match Gpu::new(0) {
        Ok(g) => Some(g),
        Err(e) if format!("{e}") == "CUDA init" => { eprintln!("SKIP: no CUDA device: {e:#}"); None }
        Err(e) => panic!("{e:#}"),
    }
}

/// How a decode step names the rows it attends: `ink_attn` takes them as arguments, `ink_attn_pos` reads the
/// position from the device (the entry point a captured graph replays) and is launched over `max_rows`.
#[derive(Clone, Copy)]
enum Step { At(usize, usize), Pos(usize, usize) }

/// A ring of `ring` rows stored through the clip kernels, as the forward stores K and V, then one decode step for
/// each of `steps`: the outputs in order.
fn decode(g: &Gpu, nh: usize, nkv: usize, hd: usize, nr: usize, ring: usize, steps: &[Step]) -> Vec<f32> {
    let ext = 64usize;
    let g: &dyn Backend = g;
    let k_in = xs(ring * nkv * hd);
    let v_in: Vec<f32> = xs(ring * nkv * hd + 7)[7..].iter().map(|x| 0.5 * x).collect();
    let (dk_in, dv_in) = (up(g, &k_in), up(g, &v_in));
    let kc = g.zeros::<u16>(g.kv_words(nkv * hd * ring)).unwrap();
    let vc = g.zeros::<u16>(g.kv_words_v(nkv * hd * ring)).unwrap();
    g.ink_clip_n(dk_in.addr(), 0, hd, nkv, ring, 8.0, 0.0, 1.0, kc.addr(), 0, ring, 0, 0.0, 1.0).unwrap();
    g.ink_clip_n_v(dv_in.addr(), 0, hd, nkv, ring, 0.0, 1e-6, 1.0, vc.addr(), 0, ring, 0, 0.0, 1.0).unwrap();
    let (q, r, sink) = (xs(nh * hd + 3)[3..].to_vec(), xs(nh * nr + 5)[5..].to_vec(), xs(nh + 7)[7..].to_vec());
    let (gate, bias) = (xs(nh * hd + 11)[11..].to_vec(), xs(ext * nr + 19)[19..].to_vec());
    let (dq, dr, ds, dg, db) = (up(g, &q), up(g, &r), up(g, &sink), up(g, &gate), up(g, &bias));
    let sc = g.zeros::<f32>(nh * ring.div_ceil(64) * (hd + 2)).unwrap();
    let (dpos, out) = (g.zeros::<i32>(1).unwrap(), g.zeros::<f32>(nh * hd).unwrap());
    let mut all = Vec::with_capacity(steps.len() * nh * hd);
    for &step in steps {
        match step {
            Step::At(rows, at) => g.ink_attn(dq.addr(), dr.addr(), ds.addr(), dg.addr(), kc.addr(), vc.addr(), db.addr(),
                                             sc.addr(), out.addr(), rows, at, nh, nkv, hd, nr, ext).unwrap(),
            Step::Pos(pos, max_rows) => {
                g.put(&[pos as i32], &dpos, 0).unwrap();
                g.ink_attn_pos(dq.addr(), dr.addr(), ds.addr(), dg.addr(), kc.addr(), vc.addr(), db.addr(), sc.addr(),
                               out.addr(), max_rows, ring, dpos.addr(), nh, nkv, hd, nr, ext).unwrap();
            }
        }
        all.extend(down::<f32>(g, &out));
    }
    all
}

/// The decode output of the scalar kernel for f16/q8/q4/q2 and three (nh, nkv, hd) shapes, as FNV-1a over the
/// little-endian f32 of main @ 06b5ba0 on an RTX 3060 Ti (sm_86). Each line is `mode nh nkv hd hash`.
const MAIN_REFERENCE: &str = "
f16 6 2 64 0d1fd51b15e2905c
f16 6 2 128 e41e6d30e1e5c8ff
f16 24 8 128 6c36f406efbdd7d5
q8 6 2 64 4a08be494886397d
q8 6 2 128 b13de7ff0ec52979
q8 24 8 128 0eb3d0b98579a40d
q4 6 2 64 dabab8fd920d91dd
q4 6 2 128 f2b4e702575ec505
q4 24 8 128 d99eb35528c031dd
q2 6 2 64 35c1219022611f01
q2 6 2 128 dd9cc214fa9ed505
q2 24 8 128 3b97e86198270656
";

/// `ink_attn` reads a value by its cache row and its place in the row (`ldkv1_at`); it used to take `/ hd` and
/// `% hd` of a flat offset for every value. The values and the order of the sums are the same, so the output must
/// stay byte for byte what main gave: a full ring, a ring whose newest row is in the middle, a last chunk that is
/// not full, and a lone row.
#[test]
fn ink_attn_scalar_keeps_the_bytes_of_main() {
    let _gpu = common::gpu_lock();
    let shapes = [(6usize, 2usize, 64usize), (6, 2, 128), (24, 8, 128)];
    let steps = [Step::At(200, 199), Step::At(200, 37), Step::At(70, 69), Step::At(1, 0), Step::Pos(137, 192), Step::Pos(437, 200)];
    let mut diff = vec![];
    for kv in ["f16", "q8", "q4", "q2"] {
        let Some(g) = open(kv, false, false) else { return };
        let rows: Vec<_> = MAIN_REFERENCE.lines().filter(|l| l.starts_with(kv)).collect();
        assert_eq!(rows.len(), shapes.len(), "{kv}: one reference line a shape");
        for (line, (nh, nkv, hd)) in rows.iter().zip(shapes) {
            let got = format!("{kv} {nh} {nkv} {hd} {:016x}", hash32(&decode(&g, nh, nkv, hd, 16, 200, &steps)));
            eprintln!("{got}");
            if got != *line { diff.push(format!("{got} (main: {line})")); }
        }
    }
    assert!(diff.is_empty(), "ink_attn differs from main:\n{}", diff.join("\n"));
}

/// `ink_attn_128_q4` reads the values `ink_attn` reads (the f16 of nibble times block scale) and adds them in
/// another order, so the two agree to the rounding of the sums: at a full ring, a ring whose newest row is in the
/// middle, one block of 128 keys and one key more, a part of a block, a lone key, and through `ink_attn_pos` with
/// a last chunk that holds few keys or none. A relative-bias row wider than the kernel keeps in shared memory
/// (`nr > 32`) uses the scalar kernel on both sides.
#[test]
fn ink_attn_128_q4_matches_scalar() {
    let _gpu = common::gpu_lock();
    let Some(scalar) = open("q4", true, true) else { return };
    let Some(tile) = open("q4", true, false) else { return };
    let (hd, ring) = (128usize, 300usize);
    let steps = [Step::At(300, 299), Step::At(300, 37), Step::At(129, 128), Step::At(128, 127), Step::At(100, 99), Step::At(1, 0),
                 Step::Pos(137, 192), Step::Pos(128, 192), Step::Pos(10, 300), Step::Pos(437, 300)];
    for (nh, nkv) in [(6usize, 2usize), (24, 8)] {
        let (a, b) = (decode(&scalar, nh, nkv, hd, 16, ring, &steps), decode(&tile, nh, nkv, hd, 16, ring, &steps));
        for (i, (x, y)) in a.chunks(nh * hd).zip(b.chunks(nh * hd)).enumerate() {
            let e = rel_l2(y, x);
            eprintln!("nh {nh} nkv {nkv} step {i}: ink_attn_128_q4 vs ink_attn {e:.2e}");
            assert!(e < 1e-5, "ink_attn_128_q4 vs ink_attn nh {nh} nkv {nkv} step {i}: {e}");
        }
        assert_ne!(a, b, "nh {nh} nkv {nkv}: the 128-key kernel was not selected");
        let (a, b) = (decode(&scalar, nh, nkv, hd, 40, ring, &steps), decode(&tile, nh, nkv, hd, 40, ring, &steps));
        assert_eq!(a, b, "nh {nh} nkv {nkv}: nr 40 must use the scalar kernel");
    }
}
