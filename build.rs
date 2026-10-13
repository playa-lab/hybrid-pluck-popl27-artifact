use std::env;
use std::path::{Path, PathBuf};

#[path = "build/program_tests.rs"]
mod program_tests;

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR not set"));

    println!("cargo:rerun-if-changed=build.rs");

    program_tests::generate(&manifest_dir, &out_dir);
}

fn rerun_if_changed(path: &Path) {
    println!("cargo:rerun-if-changed={}", path.display());
}
