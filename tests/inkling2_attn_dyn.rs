//! The `ink_*_dyn` kernels against the ones they mirror: a captured graph replays the `_dyn` pair, so with the same
//! position in device memory they must give bit-identical results. Everything else about them is the same code.
mod common;

use half::f16;

fn f16_bytes(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| f16::from_f32(*x).to_bits().to_le_bytes()).collect() }

fn pattern(n: usize, m: i32, step: f32) -> Vec<f32> { (0..n).map(|i| ((i % m as usize) as f32 - (m / 2) as f32) * step).collect() }

#[test]
fn dyn_kernels_match_the_direct_ones() {
    for (bk, g) in common::backends() {
        if !g.ink_pos_dyn() { eprintln!("{bk}: no kernels that read the position from device memory, skipping"); continue; }
        eprintln!("backend {bk}");
        let (nh, nkv, hd, nr, ext) = (4usize, 2usize, 64usize, 8usize, 4usize);
        let (kw, pos) = (nkv * hd, 100usize);
        let q = pattern(nh * hd, 17, 0.05);
        let r = pattern(nh * nr, 5, 0.1);
        let sink = pattern(nh, 3, 0.1);
        let gate = pattern(nh * hd, 7, 0.1);
        let bias = pattern(ext * nr, 3, 0.05);
        let kvf = pattern((pos + 1) * kw, 11, 0.01);

        let up = |v: &[f32]| -> anyhow::Result<rusting_strata::backend::Buf> {
            let b = g.alloc(v.len() * 2)?;
            g.htod(&f16_bytes(v), &b, 0)?;
            Ok(b)
        };
        let buf = |v: &[f32]| -> anyhow::Result<rusting_strata::backend::Buf> {
            let b = g.alloc(4 * v.len())?;
            g.put(v, &b, 0)?;
            Ok(b)
        };
        let (qa, ra, sa, ga, ba) = (buf(&q).unwrap(), buf(&r).unwrap(), buf(&sink).unwrap(), buf(&gate).unwrap(), buf(&bias).unwrap());
        // `n_swa` 0: full attention, the row is the position. 64: a ring of 64 rows, the row is the position modulo 64.
        for n_swa in [0usize, 64] {
            let rows = if n_swa == 0 { pos + 1 } else { (pos + 1).min(n_swa) };
            let at = if n_swa == 0 { pos } else { pos % n_swa };
            let sc = g.zeros::<f32>(nh * rows.div_ceil(64) * (hd + 2)).unwrap();
            let (o1, o2) = (g.zeros::<f32>(nh * hd).unwrap(), g.zeros::<f32>(nh * hd).unwrap());
            let (k1, v1, k2, v2) = (up(&kvf).unwrap(), up(&kvf).unwrap(), up(&kvf).unwrap(), up(&kvf).unwrap());
            let dynp = g.alloc(16).unwrap();
            g.put(&[pos as i32], &dynp, 0).unwrap();

            // The clips: the same rows, the same ring slot, from an argument and from device memory.
            let (x1, x2) = (buf(&kvf[..kw]).unwrap(), buf(&kvf[..kw]).unwrap());
            g.ink_clip_n(x1.addr(), 0, hd, nkv, 1, 2.5, 0.0, 1.0, k1.addr(), at, rows, 0, 0.0, 1.0).unwrap();
            g.ink_clip_dyn(x2.addr(), 0, hd, nkv, 1, 2.5, 0.0, 1.0, k2.addr(), rows, n_swa, dynp.addr(), 0.0, 1.0).unwrap();
            // The log-scaled clip: `pos0` reaches the kernel only through `dynp`.
            let (q1, q2) = (buf(&q).unwrap(), buf(&q).unwrap());
            g.ink_clip_n(q1.addr(), 0, hd, nh, 1, 2.5, 0.0, 1.0, 0, 0, 1, pos, 0.5, 16.0).unwrap();
            g.ink_clip_dyn(q2.addr(), 0, hd, nh, 1, 2.5, 0.0, 1.0, 0, 1, n_swa, dynp.addr(), 0.5, 16.0).unwrap();

            g.ink_attn(qa.addr(), ra.addr(), sa.addr(), ga.addr(), k1.addr(), v1.addr(), ba.addr(), sc.addr(), o1.addr(), rows, at, nh, nkv, hd, nr, ext).unwrap();
            g.ink_attn_dyn(qa.addr(), ra.addr(), sa.addr(), ga.addr(), k2.addr(), v2.addr(), ba.addr(), sc.addr(), o2.addr(), nh, nkv, hd, nr, ext, n_swa, dynp.addr()).unwrap();
            g.sync().unwrap();

            let (a, b) = (g.get::<f32>(&o1, 0, nh * hd).unwrap(), g.get::<f32>(&o2, 0, nh * hd).unwrap());
            assert_eq!(a, b, "n_swa {n_swa}: ink_attn_dyn differs from ink_attn");
            let (a, b) = (g.get::<f32>(&q1, 0, nh * hd).unwrap(), g.get::<f32>(&q2, 0, nh * hd).unwrap());
            assert_eq!(a, b, "n_swa {n_swa}: ink_clip_dyn differs from ink_clip_n on a log-scaled clip");
            assert!(a.iter().any(|x| *x != q[0]), "the clip changed nothing, so the comparison proves nothing");
            let words = rows * kw;
            let (a, b) = (g.get::<u16>(&k1, 0, words).unwrap(), g.get::<u16>(&k2, 0, words).unwrap());
            assert_eq!(a, b, "n_swa {n_swa}: ink_clip_dyn wrote another KV row than ink_clip_n");
            let (a, b) = (g.get::<f32>(&x1, 0, kw).unwrap(), g.get::<f32>(&x2, 0, kw).unwrap());
            assert_eq!(a, b, "n_swa {n_swa}: ink_clip_dyn scaled the rows differently");
        }
    }
}
