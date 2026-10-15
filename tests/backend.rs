use rusting_strata::{backend::Backend, cuda::Gpu};

/// Skip only when there is no usable CUDA device; kernel compile errors fail the test.
fn backend() -> Option<Box<dyn Backend>> {
    match Gpu::new() {
        Ok(g) => Some(Box::new(g)),
        Err(e) if format!("{e}") == "CUDA init" => { eprintln!("SKIP: no GPU: {e:#}"); None }
        Err(e) => panic!("{e:#}"),
    }
}

#[test]
fn memory_ops_round_trip() {
    let Some(g) = backend() else { return };
    let a = g.zeros::<f32>(8).unwrap();
    assert_eq!(a.len, 32);
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

#[test]
fn graph_records_once_and_replays() {
    let Some(g) = backend() else { return };
    if std::env::var_os("STRATA_NO_GRAPH").is_some() { return; }
    let (x, y) = (g.zeros::<f32>(4).unwrap(), g.zeros::<f32>(4).unwrap());
    g.put(&[1.0f32, 2.0, 3.0, 4.0], &y, 0).unwrap();
    let mut calls = 0;
    for _ in 0..3 { g.graph(5, &mut || { calls += 1; g.add(x.addr, y.addr, 4) }).unwrap(); }
    assert_eq!(calls, 1, "f runs only when recording");
    assert_eq!(g.get::<f32>(&x, 0, 4).unwrap(), [3.0, 6.0, 9.0, 12.0]);
}

#[test]
fn host_mailbox_is_shared_with_kernels() {
    let Some(g) = backend() else { return };
    let (mb, host) = g.host_mailbox(4).unwrap();
    let ones = g.zeros::<f32>(4).unwrap();
    g.put(&[1.0f32; 4], &ones, 0).unwrap();
    unsafe { host.cast::<f32>().write_volatile(2.0) };
    g.add(mb.addr, ones.addr, 4).unwrap(); // the kernel reads and writes host memory
    g.sync().unwrap();
    let got: Vec<f32> = (0..4).map(|i| unsafe { host.cast::<f32>().add(i).read_volatile() }).collect();
    assert_eq!(got, [3.0, 1.0, 1.0, 1.0]);
}
