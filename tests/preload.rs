//! The preload decision counts a model file only while it is mapped (issue #49). One test in its own file: the count
//! is one number for the whole process.

use rusting_strata::{gguf::Gguf, memory::total_ram, weights::fits_ram};
use std::io::Write;

#[test]
fn a_closed_model_file_is_not_counted() {
    let Some(ram) = total_ram() else { return };
    if std::env::var_os("STRATA_PRELOAD").is_some() || ram < 8 << 30 { return; }
    // An empty GGUF (version 3, no tensors, no keys) in a sparse file of half the RAM: one fits, two do not.
    let p = std::env::temp_dir().join(format!("strata-preload-{}.gguf", std::process::id()));
    let mut f = std::fs::File::create(&p).unwrap();
    f.write_all(b"GGUF\x03\0\0\0").unwrap();
    f.write_all(&[0; 16]).unwrap();
    f.set_len(ram / 2).unwrap();
    for _ in 0..3 { assert_eq!(Gguf::arch(&p), None); }
    let fits = fits_ram();
    std::fs::remove_file(&p).unwrap();
    assert!(fits, "three opened and closed files of half the RAM still count as mapped");
}
