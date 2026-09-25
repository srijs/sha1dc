//! Emits the checks the prefix does not cover.
//!
//! The mask reaching the tail names the DVs still alive, and a flagged block
//! carries about one of them. So the tail runs only those DVs' checks, rather
//! than asking about every check to find the few that matter. It comes in
//! two shapes, and each target takes the one that measures faster there.

use std::fmt::Write as _;

use crate::emit::dv_expr;
use crate::solve::{Cond, Plan};

/// How the tail finds the checks of the DVs still alive.
#[derive(Clone, Copy)]
pub enum Shape {
    /// Walks the live DVs and reads each one's checks from a table, stopping
    /// at the first that fails. Compact; on x86 the guards measured flat on
    /// a Xeon and up to 3% slower on short messages on Zen 4.
    Table,
    /// A block per DV behind that DV's own bit, with its checks written out.
    /// Any one DV is rarely alive, so each guard predicts well where the
    /// table walk mispredicts, which pays on ARM. Kept out of line, which
    /// measured faster, and the DVs the prefix covers deeply share one guard.
    Guarded,
}

/// The checks of each DV, in DV order.
fn by_dv(plan: &Plan) -> Vec<Vec<&Cond>> {
    (0..32)
        .map(|d| {
            let mut v: Vec<&Cond> = plan.tail.iter().filter(|c| c.dvs >> d & 1 == 1).collect();
            // Checks wanting a 1 first: they fail at once on the mostly-zero
            // last block of a message; for random blocks order doesn't matter.
            v.sort_by_key(|c| (std::cmp::Reverse(c.c), c.i, c.a));
            v
        })
        .collect()
}

pub fn emit(plan: &Plan, shape: Shape) -> String {
    let mut checks: Vec<String> = Vec::new();
    let mut spans: Vec<String> = Vec::new();

    for list in by_dv(plan) {
        let start = checks.len();
        for c in &list {
            checks.push(format!("({}, {}, {}, {}, {})", c.i, c.a, c.j, c.b, c.c));
        }
        spans.push(format!("({start}, {})", list.len()));
    }

    // A budget large enough puts every check in the prefix, and then the
    // tables would be empty: nothing to read and nothing left to rule out.
    if checks.is_empty() {
        return r#"
/// The prefix checks everything, so the mask is already final.
fn tail(_: &Schedule, mask: u32) -> u32 {
    mask
}
"#
        .to_owned();
    }

    if let Shape::Guarded = shape {
        return guarded(plan);
    }

    let mut out = String::new();
    let _ = write!(
        out,
        r#"
/// Every tail check, grouped by the DV it belongs to.
///
/// `(i, a, j, b, c)` reads bit `a` of `w[i]` and bit `b` of `w[j]`; the DV
/// survives only where their XOR is `c`. A check that serves several DVs
/// appears under each of them, which costs a little table and saves asking
/// about DVs that are already dead.
static TAIL_CHECKS: [(u8, u8, u8, u8, u8); {}] = [
    {},
];

/// Where each DV's checks begin in [`TAIL_CHECKS`], and how many it has.
static TAIL_SPANS: [(u16, u8); 32] = [
    {},
];

/// The checks the prefix leaves. `mask` is never zero here.
#[inline]
fn tail(w: &Schedule, mask: u32) -> u32 {{
    let mut out = mask;
    let mut rest = mask;
    while rest != 0 {{
        let d = rest.trailing_zeros();
        rest &= rest - 1;
        let (start, len) = TAIL_SPANS[d as usize];
        let start = start as usize;
        let dead = TAIL_CHECKS[start..start + len as usize]
            .iter()
            .any(|&(i, a, j, b, c)| {{
                ((w[i as usize] >> a) ^ (w[j as usize] >> b)) & 1 != u32::from(c)
            }});
        if dead {{
            out &= !(1 << d);
        }}
    }}
    out
}}
"#,
        checks.len(),
        checks.join(",\n    "),
        spans.join(",\n    "),
    );
    out
}

/// The [`Shape::Guarded`] tail: for each DV with checks left, a block that
/// runs only when its bit is set, and then runs all of them without a branch.
fn guarded(plan: &Plan) -> String {
    let lists = by_dv(plan);
    // Most often alive first. A DV at prefix rank `r` is alive with
    // probability `2^-r`, so those two or more ranks above the lowest are at
    // most a quarter as likely, and go behind one shared guard instead of
    // costing a guard each on every call.
    let mut order: Vec<usize> = (0..32).filter(|&d| !lists[d].is_empty()).collect();
    order.sort_by_key(|&d| (plan.prefix_ranks[d], d));
    let Some(&first) = order.first() else {
        unreachable!("the caller emits the empty tail");
    };
    let rare_rank = plan.prefix_ranks[first] + 2;

    let mut common = String::new();
    let mut rare = String::new();
    let mut rare_mask = 0u32;
    for d in order {
        if plan.prefix_ranks[d] >= rare_rank {
            rare_mask |= 1 << d;
            rare.push_str(&block(d, &lists[d]));
        } else {
            common.push_str(&block(d, &lists[d]));
        }
    }

    let mut out = String::from(
        "\n/// The checks the prefix leaves, a block per DV. `mask` is never zero here.\n\
         #[cold]\n\
         #[inline(never)]\n\
         fn tail(w: &Schedule, mask: u32) -> u32 {\n    let mut out = mask;\n",
    );
    out.push_str(&common);
    if rare_mask != 0 {
        let _ = write!(
            out,
            "    // The DVs the prefix leaves rarely alive, behind one guard.\n    if mask & ({}) != 0 {{\n{rare}    }}\n",
            dv_expr(rare_mask)
        );
    }
    out.push_str("    out\n}\n");
    out
}

/// The block that runs one DV's checks when its bit is set.
fn block(d: usize, list: &[&Cond]) -> String {
    // Each term's low bit is set when its check fails; OR them all. A shift
    // or XOR by zero is left out, as clippy wants.
    let terms: Vec<String> = list.iter().map(term).collect();
    let clear = if d == 0 {
        "!fail".to_owned()
    } else {
        format!("!(fail << {d})")
    };
    format!(
        "    if mask & {} != 0 {{\n        let fail = ({}) & 1;\n        out &= {clear};\n    }}\n",
        dv_expr(1 << d),
        terms.join(" | "),
    )
}

/// One check as an expression whose low bit is set when the check fails.
fn term(c: &&Cond) -> String {
    let bit = |i: usize, a: u32| {
        if a == 0 {
            format!("w[{i}]")
        } else {
            format!("(w[{i}] >> {a})")
        }
    };
    let flip = if c.c == 1 { " ^ 1" } else { "" };
    format!("({} ^ {}{flip})", bit(c.i, c.a), bit(c.j, c.b))
}
