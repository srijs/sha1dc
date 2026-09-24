//! Emits the scalar form. The vector forms must match it.
//!
//! One statement per check, directly from the table. It runs on a target that
//! has neither instruction set, and `forms_agree_on_arbitrary_words`
//! compares the vector forms against it. This form is generated and not
//! hand-written, because that comparison only has a meaning if all three
//! forms come from one table.

use std::fmt::Write as _;

use crate::emit::{align, dv_expr, test_bit};
use crate::solve::Plan;

pub fn emit(plan: &Plan) -> String {
    let mut out = String::new();

    out.push_str(
        r#"
/// The checks that run on every block.
#[inline(always)]
fn prefix(w: &Schedule) -> u32 {
    let mut mask: u32 = !0;
"#,
    );

    for f in &plan.families {
        out.push('\n');
        let (shift, _) = align(f);
        for &(i, a, dvs) in &f.members {
            let bit = test_bit(shift, a);
            let (near, far) = (i, i + f.offset);
            // A zero shift is not written out: `w[i] >> 0` is an identity that
            // clippy rejects.
            let sh = |w: usize, n: i32| {
                if n == 0 {
                    format!("w[{w}]")
                } else {
                    format!("(w[{w}] >> {n})")
                }
            };
            let x = if shift >= 0 {
                format!("{} ^ {}", sh(near, bit as i32), sh(far, bit as i32 + shift))
            } else {
                format!("{} ^ {}", sh(near, bit as i32 - shift), sh(far, bit as i32))
            };
            // `clears_on: 1` clears when the bit is set, so `b - 1` is all ones
            // when it is not; `clears_on: 0` is the other way round.
            let keep = if f.clears_on == 1 {
                "((X) & 1).wrapping_sub(1)"
            } else {
                "(0u32).wrapping_sub((X) & 1)"
            };
            let _ = write!(
                out,
                "    mask &= {}\n        | !({});\n",
                keep.replace("X", &x),
                dv_expr(dvs)
            );
        }
    }

    out.push_str("\n    mask\n}\n");
    out
}
