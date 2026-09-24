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
//! cargo run -p sha1dc-codegen -- --check   # fails if those files are stale
//! cargo run -p sha1dc-codegen -- --report  # costs, over a range of budgets
//! ```
//!
//! The output goes through `rustfmt`, so it needs no `cargo fmt` afterwards
//! and `--check` compares like with like. CI runs `--check`, which reports a
//! file that changed, one that is missing, and one that this generator no
//! longer emits.
//!
//! `SHA1DC_<TARGET>_WIDTH`, `SHA1DC_<TARGET>_GROUPS` and
//! `SHA1DC_<TARGET>_SHAPE` (`strict` or `lanemask`) override a plan, for
//! measuring. `--check` ignores them, so that a shell with one set still
//! compares against the committed plan.
//!
//! It also runs upstream's C check, which `build.rs` compiles in, reads its
//! rules off it, checks them against it, and writes them to `upstream.rs`,
//! next to the forms. See [`upstream`].
//!
//! Then run the crate tests. The `every_form_matches_upstream*` tests compare
//! every form against those rules, and `forms_agree_on_arbitrary_words`
//! compares the forms against each other.

mod avx2;
mod emit;
mod neon;
mod padding;
mod scalar;
mod solve;
mod sse2;
mod tail;
mod ubc;
mod upstream;

use std::collections::BTreeSet;
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

use solve::Shape;

/// What each target solves for.
///
/// `width` is the lane count a group is costed against, `groups` how many
/// groups the prefix may spend, and `shape` what the lanes of one group
/// share. A group runs on every block, so the budget trades unconditional
/// work against the guarded tail.
///
/// Every target has its own optimum, measured with `bench/` on an Apple M4, a
/// Xeon Platinum 8488C and a Graviton4. Re-measure them when the solver or an
/// emitter changes.
const TARGETS: &[Target] = &[
    Target {
        name: "scalar",
        width: 1,
        groups: 70,
        shape: Shape::Strict,
    },
    Target {
        name: "neon",
        width: 4,
        groups: 20,
        shape: Shape::LaneMask,
    },
    Target {
        name: "sse2",
        width: 4,
        groups: 26,
        shape: Shape::LaneMask,
    },
    Target {
        name: "avx2",
        width: 8,
        groups: 14,
        shape: Shape::LaneMask,
    },
];

struct Target {
    name: &'static str,
    width: usize,
    groups: usize,
    shape: Shape,
}

/// Reads a tunable from the environment, for measuring.
fn tunable<T: std::str::FromStr>(name: &str, fallback: T) -> T
where
    T::Err: std::fmt::Display,
{
    match std::env::var(name) {
        Ok(v) => v.parse().unwrap_or_else(|e| panic!("{name}: {e}")),
        Err(_) => fallback,
    }
}

/// Where the generated files live when no directory is given.
fn default_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../src/ubc_check")
}

/// Formats generated source the way `cargo fmt` would.
///
/// Both modes run it, so what `--check` compares is what a write puts on
/// disk. `rustfmt` reads stdin, which keeps this to one process and no
/// temporary files.
fn rustfmt(source: &str) -> io::Result<String> {
    let mut child = Command::new("rustfmt")
        .args(["--edition", "2024"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| io::Error::new(e.kind(), format!("could not run rustfmt: {e}")))?;

    // On its own thread, so that a file larger than the pipe buffer cannot
    // deadlock the two processes against each other.
    let mut stdin = child.stdin.take().expect("stdin was piped");
    let source = source.to_owned();
    let feed = std::thread::spawn(move || stdin.write_all(source.as_bytes()));

    let out = child.wait_with_output()?;
    feed.join().expect("the feeding thread did not panic")?;
    if !out.status.success() {
        return Err(io::Error::other("rustfmt rejected the generated source"));
    }
    String::from_utf8(out.stdout).map_err(io::Error::other)
}

/// Every file this generator owns, as `(name, formatted source)`.
///
/// `tuned` takes the budgets from the environment. A write honours them so
/// that a plan can be measured; [`check`] does not, so that it always
/// compares against the committed plan.
fn generate(tuned: bool) -> io::Result<Vec<(String, String)>> {
    let mut files = Vec::with_capacity(TARGETS.len() + 1);

    for target in TARGETS {
        let name = target.name.to_uppercase();
        let (width, groups, shape) = if tuned {
            (
                tunable(&format!("SHA1DC_{name}_WIDTH"), target.width),
                tunable(&format!("SHA1DC_{name}_GROUPS"), target.groups),
                tunable(&format!("SHA1DC_{name}_SHAPE"), target.shape),
            )
        } else {
            (target.width, target.groups, target.shape)
        };

        files.push((
            format!("{}.rs", target.name),
            rustfmt(&form(target.name, width, groups, shape))?,
        ));
    }

    // Not a form of the check: upstream's rules, which the tests compare
    // every form against.
    files.push(("upstream.rs".to_owned(), rustfmt(&upstream::emit())?));

    Ok(files)
}

/// One form of the check, before formatting.
fn form(name: &str, width: usize, groups: usize, shape: Shape) -> String {
    let plan = solve::solve(width, groups, shape);
    let prefix = match name {
        "scalar" => scalar::emit(&plan),
        "neon" => neon::emit(&plan),
        "sse2" => sse2::emit(&plan),
        "avx2" => avx2::emit(&plan),
        other => panic!("no emitter for {other}"),
    };
    emit::module(name, &prefix, &tail::emit(&plan))
}

fn write(dir: &Path) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    for (name, source) in generate(true)? {
        let path = dir.join(name);
        std::fs::write(&path, source)?;
        println!("wrote {}", path.display());
    }
    Ok(())
}

/// Reports whether `dir` holds exactly what this generator emits.
///
/// Compares the directory rather than a list of names, so a file that is no
/// longer emitted is reported too, not only one that changed. Deliberately
/// does not consult git, so it behaves the same for a tracked, untracked or
/// dirty tree.
fn check(dir: &Path) -> io::Result<bool> {
    let mut leftover = BTreeSet::new();
    for entry in std::fs::read_dir(dir)? {
        let name = entry?.file_name();
        let name = name.to_string_lossy();
        if name.ends_with(".rs") {
            leftover.insert(name.into_owned());
        }
    }

    let mut stale = false;
    for (name, want) in generate(false)? {
        leftover.remove(&name);
        match std::fs::read_to_string(dir.join(&name)) {
            Ok(found) if found == want => {}
            Ok(found) => {
                report_difference(&name, &found, &want);
                stale = true;
            }
            Err(_) => {
                println!("{name} is missing");
                stale = true;
            }
        }
    }

    for name in leftover {
        println!("{name} is no longer generated");
        stale = true;
    }

    Ok(!stale)
}

/// Prints where the file on disk first departs from the generated source.
fn report_difference(name: &str, found: &str, want: &str) {
    let found: Vec<_> = found.lines().collect();
    let want: Vec<_> = want.lines().collect();
    let at = (0..found.len().max(want.len()))
        .find(|&i| found.get(i) != want.get(i))
        .unwrap_or(0);

    println!("{name} differs, first at line {}:", at + 1);
    for line in found.iter().take(at).skip(at.saturating_sub(2)) {
        println!("      {line}");
    }
    println!("    - {}", found.get(at).unwrap_or(&"(end of file)"));
    println!("    + {}", want.get(at).unwrap_or(&"(end of file)"));
}

fn main() -> ExitCode {
    let arg = std::env::args().nth(1);

    let result = match arg.as_deref() {
        Some("--report") => {
            report();
            return ExitCode::SUCCESS;
        }
        Some("--check") => match check(&default_dir()) {
            Ok(true) => {
                println!("src/ubc_check/ is up to date");
                return ExitCode::SUCCESS;
            }
            Ok(false) => {
                eprintln!();
                eprintln!("error: src/ubc_check/ is not what codegen/ produces.");
                eprintln!("Regenerate with: cargo run -p sha1dc-codegen");
                return ExitCode::FAILURE;
            }
            Err(e) => Err(e),
        },
        Some(dir) => write(&PathBuf::from(dir)),
        None => write(&default_dir()),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// What each budget costs a target, so the plans can be measured rather than
/// guessed.
fn report() {
    for target in TARGETS {
        println!(
            "\n{} — costed at {} lane(s), shipping {} groups",
            target.name, target.width, target.groups
        );
        println!(
            "groups  families  checks  tail  shared  per-DV  worst  P(tail)  last: enters  DVs"
        );
        // The shipped budget among them, wherever it falls.
        let mut budgets = vec![8, 10, 13, 16, 20, 22, 26, 30, 40, 55, target.groups];
        budgets.sort_unstable();
        budgets.dedup();
        for n in budgets {
            let plan = solve::solve(target.width, n, target.shape);
            let checks: usize = plan.families.iter().map(|f| f.members.len()).sum();
            let shared = plan.tail.iter().filter(|c| c.dvs.count_ones() > 1).count();
            // A DV the prefix covers to rank `r` reaches the tail with
            // probability `2^-r`, so this is how often the tail runs at all.
            let enters = 1.0
                - plan
                    .prefix_ranks
                    .iter()
                    .map(|&r| 1.0 - (-(r as f64) * std::f64::consts::LN_2).exp())
                    .product::<f64>();
            // The same for the last block of a message, which is mostly
            // padding and so passes or fails a condition every time.
            let (last, dvs) = padding::tail_of(&plan);
            println!(
                "{n:>6}  {:>8}  {checks:>6}  {:>4}  {shared:>6}  {:>6}  {:>5}  {enters:>7.4}  {last:>12.3}  {dvs:>4.2}",
                plan.families.len(),
                plan.tail.len(),
                plan.tail.len() - shared,
                plan.prefix_ranks.iter().min().unwrap(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every target emits source at the extremes of its budget: no prefix
    /// at all, and one so large that the tail is left with nothing.
    #[test]
    fn every_budget_emits_valid_source() {
        for target in TARGETS {
            for groups in [0, 1, 100] {
                rustfmt(&form(target.name, target.width, groups, target.shape))
                    .unwrap_or_else(|e| panic!("{} at {groups} groups: {e}", target.name));
            }
        }
    }
}
