//! Generates the UBC check for the sha1dc crate.
//!
//! # Why this exists
//!
//! The check is most of the cost of detection, and it has four forms: one per
//! instruction set, plus a scalar reference. Hand-written copies can go out of
//! step. The result is a mask that clears too few bits: digests stay correct,
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

mod avx2;
mod conditions;
mod emit;
mod neon;
mod scalar;
mod solve;
mod sse2;
mod tail;
mod ubc;

use std::path::PathBuf;

/// What each target solves for.
///
/// `width` is the lane count a group is costed against, and `groups` is how
/// many groups the prefix may spend. A group runs on every block, so the
/// budget trades unconditional work against the guarded tail.
///
/// Every target has its own optimum, measured on an Apple M-series and a Xeon
/// Platinum 8488C. The scalar form has no lanes, so a group is one statement
/// and it wants far fewer of them. The two four-lane forms share a plan
/// because the lane count is all the solver sees.
const TARGETS: &[Target] = &[
    Target {
        name: "scalar",
        width: 1,
        groups: 40,
    },
    Target {
        name: "neon",
        width: 4,
        groups: 22,
    },
    Target {
        name: "sse2",
        width: 4,
        groups: 22,
    },
    Target {
        name: "avx2",
        width: 8,
        groups: 10,
    },
];

struct Target {
    name: &'static str,
    width: usize,
    groups: usize,
}

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

    let out = match arg {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../src/ubc_check"),
    };
    std::fs::create_dir_all(&out)?;

    for target in TARGETS {
        let plan = solve::solve(
            tunable(
                &format!("SHA1DC_{}_WIDTH", target.name.to_uppercase()),
                target.width,
            ),
            tunable(
                &format!("SHA1DC_{}_GROUPS", target.name.to_uppercase()),
                target.groups,
            ),
        );
        let prefix = match target.name {
            "scalar" => scalar::emit(&plan),
            "neon" => neon::emit(&plan),
            "sse2" => sse2::emit(&plan),
            "avx2" => avx2::emit(&plan),
            other => panic!("no emitter for {other}"),
        };
        let path = out.join(format!("{}.rs", target.name));
        std::fs::write(
            &path,
            emit::module(target.name, &prefix, &tail::emit(&plan)),
        )?;
        println!("wrote {}", path.display());
    }

    // Not a form of the check: the published conditions, which the tests
    // solve to reach the checks behind a chosen DV.
    let path = out.join("conditions.rs");
    std::fs::write(&path, conditions::emit())?;
    println!("wrote {}", path.display());

    Ok(())
}

/// What each budget costs a target, so the plans can be measured rather than
/// guessed.
fn report() {
    for target in TARGETS {
        println!(
            "\n{} — costed at {} lane(s), shipping {} groups",
            target.name, target.width, target.groups
        );
        println!("groups  families  checks  tail  shared  per-DV");
        for n in [8, 10, 13, 16, 20, 22, 26, 30, 40, 55] {
            let plan = solve::solve(target.width, n);
            let checks: usize = plan.families.iter().map(|f| f.members.len()).sum();
            let shared = plan.tail.iter().filter(|c| c.dvs.count_ones() > 1).count();
            println!(
                "{n:>6}  {:>8}  {checks:>6}  {:>4}  {shared:>6}  {:>6}",
                plan.families.len(),
                plan.tail.len(),
                plan.tail.len() - shared
            );
        }
    }
}
