mod common;

#[test]
fn memory_ops_round_trip() {
    for (bk, g) in common::backends() {
        eprintln!("backend {bk}");
        let a = g.zeros::<f32>(8).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(g.get::<f32>(&a, 0, 8).unwrap(), [0.0; 8]);
        g.put(&[1.0f32, 2.0, 3.0], &a, 2).unwrap();
        assert_eq!(g.get::<f32>(&a, 0, 6).unwrap(), [0.0, 0.0, 1.0, 2.0, 3.0, 0.0]);
        let b = g.zeros::<f32>(4).unwrap();
        g.dtod(&a, 8, &b, 4, 12).unwrap();
        assert_eq!(g.get::<f32>(&b, 0, 4).unwrap(), [0.0, 1.0, 2.0, 3.0]);
        g.memset(&a).unwrap();
        assert_eq!(g.get::<f32>(&a, 0, 8).unwrap(), [0.0; 8]);
        assert!(g.put(&[0f32; 9], &a, 0).is_err(), "a copy past the end must fail");
        let (free, total) = g.mem_info().unwrap();
        assert!(free > 0 && free <= total);
    }
}

#[test]
fn large_copies_round_trip() {
    for (bk, g) in common::backends() {
        eprintln!("backend {bk}");
        let v: Vec<u32> = (0..20u32 << 20).map(|i| i.wrapping_mul(2654435761)).collect(); // 80 MB, past one staging chunk
        let a = g.zeros::<u32>(v.len() + 1).unwrap();
        g.put(&v, &a, 1).unwrap();
        let b = g.zeros::<u32>(v.len() + 1).unwrap();
        g.dtod(&a, 0, &b, 0, a.len()).unwrap();
        assert!(g.get::<u32>(&b, 1, v.len()).unwrap() == v);
    }
}

#[test]
fn graph_records_once_and_replays() {
    for (bk, g) in common::backends() {
        eprintln!("backend {bk}");
        if std::env::var_os("STRATA_NO_GRAPH").is_some() { return; }
        let (x, y) = (g.zeros::<f32>(4).unwrap(), g.zeros::<f32>(4).unwrap());
        g.put(&[1.0f32, 2.0, 3.0, 4.0], &y, 0).unwrap();
        let mut calls = 0;
        for _ in 0..3 { g.graph(5, &mut || { calls += 1; g.add(x.addr(), y.addr(), 4) }).unwrap(); }
        assert_eq!(calls, 1, "f runs only when recording");
        assert_eq!(g.get::<f32>(&x, 0, 4).unwrap(), [3.0, 6.0, 9.0, 12.0]);
    }
}

#[test]
fn host_mailbox_is_shared_with_kernels() {
    for (bk, g) in common::backends() {
        eprintln!("backend {bk}");
        let (mb, host) = g.host_mailbox(4).unwrap();
        let ones = g.zeros::<f32>(4).unwrap();
        g.put(&[1.0f32; 4], &ones, 0).unwrap();
        unsafe { host.cast::<f32>().write_volatile(2.0) };
        g.add(mb.addr(), ones.addr(), 4).unwrap(); // the kernel reads and writes host memory
        g.sync().unwrap();
        let got: Vec<f32> = (0..4).map(|i| unsafe { host.cast::<f32>().add(i).read_volatile() }).collect();
        assert_eq!(got, [3.0, 1.0, 1.0, 1.0]);
    }
}

/// The committed SPIR-V is what `tools/gen_spirv.sh` makes from the shaders now.
#[test]
fn spirv_is_fresh() {
    if std::process::Command::new("glslc").arg("--version").output().is_err() { return eprintln!("SKIP: no glslc"); }
    let tmp = std::env::temp_dir().join(format!("strata-spv-{}", std::process::id()));
    let st = std::process::Command::new("tools/gen_spirv.sh").arg(&tmp).status().unwrap();
    assert!(st.success());
    let names = |d: &std::path::Path| { let mut v: Vec<_> = std::fs::read_dir(d).unwrap().map(|e| e.unwrap().file_name()).collect(); v.sort(); v };
    let committed = std::path::Path::new("src/vulkan/spv");
    assert_eq!(names(&tmp), names(committed), "run tools/gen_spirv.sh");
    for n in names(&tmp) {
        assert!(std::fs::read(tmp.join(&n)).unwrap() == std::fs::read(committed.join(&n)).unwrap(), "{n:?} is stale: run tools/gen_spirv.sh");
    }
    std::fs::remove_dir_all(&tmp).unwrap();
}

/// `src/vulkan/shaders/tables.glsl` is generated from `src/quant/tables.rs`; `STRATA_WRITE_TABLES=1` rewrites it.
#[test]
fn glsl_tables_are_fresh() {
    use rusting_strata::quant::tables::*;
    fn arr<T: std::fmt::LowerHex>(ty: &str, name: &str, v: &[T], suf: &str) -> String {
        let body: Vec<_> = v.iter().map(|x| format!("{x:#x}{suf}")).collect();
        format!("const {ty} {name}[{}] = {{{}}};\n", v.len(), body.join(","))
    }
    let iq4: Vec<_> = KVALUES_IQ4NL.iter().map(|v| v.to_string()).collect();
    let want = format!(
        "// Generated from src/quant/tables.rs by tests/backend.rs glsl_tables_are_fresh (STRATA_WRITE_TABLES=1).\n{}{}{}{}{}{}{}const int kvalues_iq4nl[16] = {{{}}};\n",
        arr("uint64_t", "iq2xs_grid", &IQ2XS_GRID, "ul"), arr("uint64_t", "iq2xxs_grid", &IQ2XXS_GRID, "ul"),
        arr("uint64_t", "iq2s_grid", &IQ2S_GRID, "ul"), arr("uint64_t", "iq1s_grid", &IQ1S_GRID, "ul"),
        arr("uint", "iq3xxs_grid", &IQ3XXS_GRID, "u"), arr("uint", "iq3s_grid", &IQ3S_GRID, "u"),
        arr("uint", "ksigns_iq2xs", &KSIGNS_IQ2XS, "u"), iq4.join(","));
    let path = "src/vulkan/shaders/tables.glsl";
    if std::env::var_os("STRATA_WRITE_TABLES").is_some() { std::fs::write(path, &want).unwrap(); }
    assert!(std::fs::read_to_string(path).unwrap_or_default() == want, "{path} is stale: rerun with STRATA_WRITE_TABLES=1");
}
