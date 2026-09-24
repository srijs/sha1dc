//! Compiles upstream's UBC check, which the generator runs to read its rules
//! off. `upstream/` is a git submodule of sha1collisiondetection, pinned to
//! the commit the rules are read from.

fn main() {
    println!("cargo::rerun-if-changed=upstream/lib");
    if !std::path::Path::new("upstream/lib/ubc_check.c").exists() {
        panic!(
            "codegen/upstream/ is empty: it is a git submodule. \
             Run `git submodule update --init` and build again."
        );
    }
    cc::Build::new()
        .file("upstream/lib/ubc_check.c")
        .include("upstream/lib")
        .warnings(false)
        .compile("sha1dc_upstream");
}
