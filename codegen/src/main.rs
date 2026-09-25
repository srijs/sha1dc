//! Generates the UBC check for the sha1dc crate.
//!
//! # Why this exists
//!
//! The check is most of the cost of detection, and has a form per instruction
//! set plus a scalar one. Hand-written copies drift apart, and a mask that
//! clears too few bits passes every test while halving throughput. So every
//! form comes from one table, [`ubc`], through a plan from [`solve`]: a
//! prefix that runs on every block, and a tail behind guards.
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
//! The search is seeded and uses only `+ - * /` on floats, so every machine
//! finds the same plans and `--check`, which CI runs, can search again. The
//! output goes through `rustfmt`.
//!
//! It also reads upstream's rules off its C check, which `build.rs` compiles
//! in, into `upstream.rs`; see [`upstream`]. The crate tests compare every
//! form against those rules and against each other.

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

use solve::{Ops, Params, Plan};

/// What each target's plan is searched for. A group runs on every block, so
/// the budget trades unconditional work against the tail. Each was measured
/// with `bench/` on an Apple M4, a Xeon Platinum 8488C and a Graviton4;
/// re-measure when the solver or an emitter changes.
const TARGETS: &[Target] = &[
    Target {
        name: "scalar",
        params: Params {
            width: 1,
            groups: 70,
            ops: NONE,
        },
    },
    Target {
        name: "neon",
        params: Params {
            width: 4,
            groups: 20,
            ops: LANE_BIT,
        },
    },
    Target {
        name: "sse2",
        params: Params {
            width: 4,
            groups: 26,
            ops: LANE_BIT,
        },
    },
    Target {
        name: "avx2",
        params: Params {
            width: 8,
            groups: 14,
            ops: LANE_BIT,
        },
    },
];

/// What a target with no per-lane operations can do: nothing but share.
const NONE: Ops = Ops { lane_bit: false };

/// A constant per lane, which every vector target can load.
const LANE_BIT: Ops = Ops { lane_bit: true };

struct Target {
    name: &'static str,
    params: Params,
}

/// Where the generated files live when no directory is given.
fn default_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../src/ubc_check")
}

/// Formats generated source the way `cargo fmt` would, through stdin.
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

/// `f` for every target, each on its own thread, in the order of [`TARGETS`]
/// whatever order they finish in.
fn per_target<T: Send>(f: impl Fn(&Target) -> T + Sync) -> Vec<T> {
    std::thread::scope(|scope| {
        let handles: Vec<_> = TARGETS
            .iter()
            .map(|target| scope.spawn(|| f(target)))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("a target's thread panicked"))
            .collect()
    })
}

/// The plan the search finds for `target`, which for a vector target must
/// not send every message of some length into the tail.
fn plan(target: &Target) -> io::Result<Plan> {
    let params = &target.params;
    let plan = solve::search(params, solve::STALL);
    let cliffs = padding::cliffs(&plan);
    if params.width > 1 && !cliffs.is_empty() {
        return Err(io::Error::other(format!(
            "{}: the search sends every message of {cliffs:?} bytes mod 64 into the tail",
            target.name
        )));
    }
    Ok(plan)
}

/// Every file this generator owns, as `(name, formatted source)`, with the
/// targets searched side by side.
fn generate() -> io::Result<Vec<(String, String)>> {
    // Not a form of the check: upstream's rules, which the tests compare
    // every form against. Read off while the targets search.
    let (forms, upstream) = std::thread::scope(|scope| {
        let upstream = scope.spawn(|| rustfmt(&upstream::emit()));
        let forms = per_target(|target| rustfmt(&form(target.name, &plan(target)?)));
        (forms, upstream.join().expect("upstream's thread panicked"))
    });
    let mut files = TARGETS
        .iter()
        .zip(forms)
        .map(|(target, form)| Ok((format!("{}.rs", target.name), form?)))
        .collect::<io::Result<Vec<_>>>()?;
    files.push(("upstream.rs".to_owned(), upstream?));

    Ok(files)
}

/// One form of the check, before formatting.
fn form(name: &str, plan: &Plan) -> String {
    let prefix = match name {
        "scalar" => scalar::emit(plan),
        "neon" => neon::emit(plan),
        "sse2" => sse2::emit(plan),
        "avx2" => avx2::emit(plan),
        other => panic!("no emitter for {other}"),
    };
    emit::module(name, &prefix, &tail::emit(plan))
}

fn write(dir: &Path) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    for (name, source) in generate()? {
        let path = dir.join(name);
        std::fs::write(&path, source)?;
        println!("wrote {}", path.display());
    }
    Ok(())
}

/// Reports whether `dir` holds exactly what this generator emits, no more
/// and no less, whatever git thinks of it.
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
    for (name, want) in generate()? {
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

/// What each budget costs a target, from a quick search, so the budgets can
/// be measured rather than guessed.
fn report() {
    for lines in per_target(report_target) {
        print!("{lines}");
    }
}

/// [`report`]'s lines for one target.
fn report_target(target: &Target) -> String {
    use std::fmt::Write as _;
    let Params { width, groups, ops } = target.params;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "\n{} — costed at {width} lane(s), shipped budget {groups} groups",
        target.name
    );
    let _ = writeln!(
        out,
        "groups  families  checks  tail  shared  per-DV  worst  P(tail)  last: enters  DVs"
    );
    // The shipped budget among them, wherever it falls.
    let mut budgets = vec![8, 10, 13, 16, 20, 22, 26, 30, 40, 55, groups];
    budgets.sort_unstable();
    budgets.dedup();
    for n in budgets {
        let plan = solve::search(
            &Params {
                width,
                groups: n,
                ops,
            },
            0,
        );
        let checks: usize = plan.families.iter().map(|f| f.members.len()).sum();
        let shared = plan.tail.iter().filter(|c| c.dvs.count_ones() > 1).count();
        // How often the tail runs, as each DV survives with `2^-r`.
        let enters = 1.0
            - plan
                .prefix_ranks
                .iter()
                .map(|&r| 1.0 - (-(r as f64) * std::f64::consts::LN_2).exp())
                .product::<f64>();
        // The same for the last block of a message.
        let (last, dvs) = padding::tail_of(&plan);
        let _ = writeln!(
            out,
            "{n:>6}  {:>8}  {checks:>6}  {:>4}  {shared:>6}  {:>6}  {:>5}  {enters:>7.4}  {last:>12.3}  {dvs:>4.2}",
            plan.families.len(),
            plan.tail.len(),
            plan.tail.len() - shared,
            plan.prefix_ranks.iter().min().unwrap(),
        );
    }
    out
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
                let plan = solve::search(
                    &Params {
                        groups,
                        ..target.params
                    },
                    0,
                );
                rustfmt(&form(target.name, &plan))
                    .unwrap_or_else(|e| panic!("{} at {groups} groups: {e}", target.name));
            }
        }
    }
}
