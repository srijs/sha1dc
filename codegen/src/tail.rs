//! Emits the checks the prefix does not cover.
//!
//! The prefix runs every check on every block. Down here the mask is already
//! sparse, so a guard that skips a check is worth more than the check costs.
//! Each statement only clears bits, so the order is free to suit that.
//!
//! A check that names one DV is written as a branch that tests the relations
//! in turn and stops at the first failure. A check that names several is
//! written as a mask update, because a branch per DV would cost more.

use std::fmt::Write as _;

use crate::emit::dv_expr;
use crate::solve::{Cond, Plan};

/// `w[i] >> n`, or `w[i]` when the shift is zero, which clippy rejects.
fn shifted(w: usize, n: u32) -> String {
    if n == 0 {
        format!("w[{w}]")
    } else {
        format!("(w[{w}] >> {n})")
    }
}

/// The XOR of the two bits, in the low bit.
fn tested(c: &Cond) -> String {
    format!("({} ^ {}) & 1", shifted(c.i, c.a), shifted(c.j, c.b))
}

/// True when the condition fails and its DVs must be cleared.
fn fails(c: &Cond) -> String {
    format!("{} {} 0", tested(c), if c.c == 1 { "==" } else { "!=" })
}

pub fn emit(plan: &Plan) -> String {
    let mut out = String::new();

    out.push_str(
        r#"
/// The checks the prefix leaves. `mask` is never zero here.
#[inline(always)]
fn tail(w: &[u32; 80], mut mask: u32) -> u32 {
"#,
    );

    let (mut shared, mut single): (Vec<&Cond>, Vec<&Cond>) =
        plan.tail.iter().partition(|c| c.dvs.count_ones() > 1);

    // The widest first: those clear the most, so the guards below them skip
    // more often.
    shared.sort_by_key(|c| (u32::MAX - c.dvs.count_ones(), c.i, c.a));
    single.sort_by_key(|c| (c.dvs.trailing_zeros(), c.i, c.a));

    for c in &shared {
        let dvs = dv_expr(c.dvs);
        let keep = if c.c == 1 {
            format!("(0u32).wrapping_sub({})", tested(c))
        } else {
            format!("({}).wrapping_sub(1)", tested(c))
        };
        let _ = write!(
            out,
            "    if mask & ({dvs}) != 0 {{\n        mask &= {keep} | !({dvs});\n    }}\n"
        );
    }

    // One test covers several DVs above, so the mask can already be empty.
    // Only clearing happens below, so that is the answer.
    if !shared.is_empty() && !single.is_empty() {
        out.push_str("\n    if mask == 0 {\n        return 0;\n    }\n");
    }

    // Group the per-DV branches: each DV's own guard is a single-bit test, so
    // evaluating them is free. It is the branches that cost, and few enough
    // bits survive this far that most groups are empty and skip in one.
    let mut by_dv: Vec<(u32, Vec<&Cond>)> = Vec::new();
    for c in single {
        let dv = c.dvs.trailing_zeros();
        match by_dv.last_mut() {
            Some((d, list)) if *d == dv => list.push(c),
            _ => by_dv.push((dv, vec![c])),
        }
    }

    for (n, chunk) in by_dv.chunks(3).enumerate() {
        if n > 0 && n % 2 == 0 {
            out.push_str("\n    if mask == 0 {\n        return 0;\n    }\n");
        }
        // The shared guard pays for itself only when it can skip several
        // DVs at once. For a chunk of one it is the same test as the branch
        // inside it, so that chunk is written flat.
        let shared = chunk.len() > 1;
        let pad = if shared { "        " } else { "    " };
        if shared {
            let guard = chunk.iter().fold(0, |a, (dv, _)| a | 1 << dv);
            let _ = write!(out, "\n    if mask & ({}) != 0 {{\n", dv_expr(guard));
        } else {
            out.push('\n');
        }
        for (dv, list) in chunk {
            let name = dv_expr(1 << dv);
            if list.len() == 1 {
                let _ = write!(
                    out,
                    "{pad}if mask & {name} != 0 && {} {{\n{pad}    mask &= !{name};\n{pad}}}\n",
                    fails(list[0])
                );
            } else {
                let any: Vec<String> = list.iter().map(|c| fails(c)).collect();
                let _ = write!(
                    out,
                    "{pad}if mask & {name} != 0\n{pad}    && ({})\n{pad}{{\n{pad}    mask &= !{name};\n{pad}}}\n",
                    any.join(&format!("\n{pad}        || "))
                );
            }
        }
        if shared {
            out.push_str("    }\n");
        }
    }

    out.push_str("\n    mask\n}\n");
    out
}
