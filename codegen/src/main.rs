//! Generates the UBC check for the sha1dc crate.
//!
//! # Why this exists
//!
//! The check is most of the cost of detection, and it has a form per
//! instruction set as well as a scalar reference. Hand-written copies can go
//! out of step. The result is a mask that clears too few bits: digests stay correct,
//! tests pass, and throughput halves because more blocks recompress. Every
//! form comes from one table, in [`ubc`].
//!
//! [`solve`] turns that table into a plan. Per DV the published conditions
//! are one basis of a linear space, so an equivalent basis gives the same
//! mask; the solver picks the one that suits vector code. It splits the plan
//! into a prefix that runs on every block, and a tail that runs behind
//! guards.
//!
//! # Running it
//!
//! ```text
//! cargo run -p sha1dc-codegen              # writes into ../src/ubc_check/
//! cargo run -p sha1dc-codegen -- <dir>     # or somewhere else
//! cargo run -p sha1dc-codegen -- --report  # costs, over a range of budgets
//! ```
//!
//! `SHA1DC_PREFIX_GROUPS` overrides the budget, for measuring.
//!
//! Then run `cargo fmt -p sha1dc` and the crate tests.
//! `vectorized_prefix_matches_scalar` compares the vector forms against the
//! scalar one. `matches_c_reference` compares the whole check against the
//! original C. `codegen/check.sh` fails if the committed files are stale.

mod emit;
mod neon;
mod scalar;
mod solve;
mod tail;
mod ubc;

use std::path::PathBuf;

/// How many vector groups the prefix may spend, costed at eight lanes.
///
/// Every group runs on every block, so this trades unconditional vector work
/// against the guarded tail. Swept on a Xeon Platinum 8488C, in MiB/s:
///
/// ```text
/// 8     10     11     12     13     16     20     24     28
/// 1278  1293   1301   1289   1287   1262   1223   1165   1114
/// ```
///
/// 9 to 13 is a plateau. 11 is 0.6% above this value on that machine, but
/// 0.6% below it on an Apple M-series, so the middle of the plateau wins.
/// Past 16 the prefix does more work than the tail it removes.
const PREFIX_GROUPS: usize = 10;

/// The lane count the prefix is costed against.
///
/// Eight is AVX2. The four-lane forms cut the same families in half, so they
/// do not need a plan of their own. The reverse is not true: a plan costed at
/// four lanes fills only half of each AVX2 vector, and measures slower than
/// the published basis on x86.
const WIDTH: usize = 8;

/// Reads a tunable from the environment, for measuring.
fn tunable(name: &str, fallback: usize) -> usize {
    match std::env::var(name) {
        Ok(n) => n
            .parse()
            .unwrap_or_else(|_| panic!("{name} must be a number")),
        Err(_) => fallback,
    }
}

fn main() -> std::io::Result<()> {
    let arg = std::env::args().nth(1);
    if arg.as_deref() == Some("--report") {
        report();
        return Ok(());
    }

    let plan = solve::solve(
        tunable("SHA1DC_WIDTH", WIDTH),
        tunable("SHA1DC_PREFIX_GROUPS", PREFIX_GROUPS),
    );

    let out = match arg {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../src/ubc_check"),
    };
    std::fs::create_dir_all(out.join("prefix"))?;

    for (name, contents) in [
        ("prefix/scalar.rs", scalar::emit(&plan)),
        ("prefix/neon.rs", neon::emit(&plan)),
        ("tail.rs", tail::emit(&plan)),
    ] {
        let path = out.join(name);
        std::fs::write(&path, contents)?;
        println!("wrote {}", path.display());
    }
    Ok(())
}

/// What each budget costs, so the choice of [`PREFIX_GROUPS`] can be measured
/// rather than guessed.
fn report() {
    println!("groups  families  checks  g@8  g@4  tail  shared  per-DV");
    let width = tunable("SHA1DC_WIDTH", WIDTH);
    println!("(costed at {width} lanes)");
    for n in [8, 10, 12, 13, 14, 16, 18, 20, 24, 28, 32, 36] {
        let plan = solve::solve(width, n);
        let g8: usize = plan
            .families
            .iter()
            .map(|f| f.members.len().div_ceil(8))
            .sum();
        let g4: usize = plan
            .families
            .iter()
            .map(|f| f.members.len().div_ceil(4))
            .sum();
        let checks: usize = plan.families.iter().map(|f| f.members.len()).sum();
        let shared = plan.tail.iter().filter(|c| c.dvs.count_ones() > 1).count();
        println!(
            "{n:>6}  {:>8}  {checks:>6}  {g8:>3}  {g4:>3}  {:>4}  {shared:>6}  {:>6}",
            plan.families.len(),
            plan.tail.len(),
            plan.tail.len() - shared
        );
    }
}
