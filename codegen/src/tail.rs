//! Emits the checks the prefix does not cover.
//!
//! The mask reaching the tail names the DVs still alive, and a flagged block
//! carries about one of them. So the tail walks those DVs and runs only their
//! checks, read from a table, rather than asking about every check to find
//! the few that matter.

use std::fmt::Write as _;

use crate::solve::{Cond, Plan};

/// The checks of each DV, in DV order.
fn by_dv(plan: &Plan) -> Vec<Vec<&Cond>> {
    (0..32)
        .map(|d| {
            let mut v: Vec<&Cond> = plan.tail.iter().filter(|c| c.dvs >> d & 1 == 1).collect();
            v.sort_by_key(|c| (c.i, c.a));
            v
        })
        .collect()
}

pub fn emit(plan: &Plan) -> String {
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
