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
//! `SHA1DC_<TARGET>_WIDTH` and `SHA1DC_<TARGET>_GROUPS` override a budget,
//! for measuring. `--check` ignores them, so that a shell with one set still
//! compares against the committed plan.
//!
//! Then run the crate tests. `every_form_matches_scalar` and
//! `every_form_matches_scalar_where_the_tail_runs` compare the forms against
//! each other, and `matches_c_reference` compares the whole check against the
//! original C.

mod avx2;
mod conditions;
mod emit;
mod neon;
mod scalar;
mod solve;
mod sse2;
mod tail;
mod ubc;

use std::collections::BTreeSet;
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

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
        let (width, groups) = if tuned {
            (
                tunable(&format!("SHA1DC_{name}_WIDTH"), target.width),
                tunable(&format!("SHA1DC_{name}_GROUPS"), target.groups),
            )
        } else {
            (target.width, target.groups)
        };

        let plan = solve::solve(width, groups);
        let prefix = match target.name {
            "scalar" => scalar::emit(&plan),
            "neon" => neon::emit(&plan),
            "sse2" => sse2::emit(&plan),
            "avx2" => avx2::emit(&plan),
            other => panic!("no emitter for {other}"),
        };
        let source = emit::module(target.name, &prefix, &tail::emit(&plan));
        files.push((format!("{}.rs", target.name), rustfmt(&source)?));
    }

    // Not a form of the check: the published conditions, which the tests
    // solve to reach the checks behind a chosen DV.
    files.push(("conditions.rs".to_owned(), rustfmt(&conditions::emit())?));

    Ok(files)
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
