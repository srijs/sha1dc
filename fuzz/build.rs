//! Compiles upstream's hasher, which the target checks this crate against,
//! from the submodule the generator reads its rules from.

fn main() {
    let lib = "../codegen/upstream/lib";
    println!("cargo::rerun-if-changed={lib}");
    println!("cargo::rerun-if-changed=oracle.c");
    if !std::path::Path::new(lib).join("sha1.c").exists() {
        panic!(
            "codegen/upstream/ is empty: it is a git submodule. \
             Run `git submodule update --init` and build again."
        );
    }
    cc::Build::new()
        .file(format!("{lib}/sha1.c"))
        .file(format!("{lib}/ubc_check.c"))
        .file("oracle.c")
        .include(lib)
        .warnings(false)
        .compile("sha1dc_oracle");
}
