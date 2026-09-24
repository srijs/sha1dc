//! Emits the `aarch64` form.
//!
//! Four checks per vector. Load `w[i..i + 4]` and `w[i + offset..]`, align the
//! two bits, then test them. `vtstq_u32` sets a lane to all ones if the tested
//! bit is set. This replaces the shift, and, and negate of the scalar form
//! with one instruction. The DV bits of the lane are then masked in and added
//! to an accumulator.
//!
//! The mask is applied with `vminq_u32`, or `vqsubq_u32` where a clear bit is
//! what rules the DVs out, not with `vandq_u32` or `vbicq_u32`. Each lane of
//! the test is zero or all ones, so both give the DV bits or zero. LLVM
//! rewrites a `vtstq_u32` whose result meets an AND into `and`, `cmeq` and
//! `bic`, one instruction more per group and seventeen a block; it leaves
//! these two alone.
//!
//! There are two accumulators, so the final OR chain does not serialize the
//! groups. The target settles `neon`, so a `cfg` selects this form.

use std::fmt::Write as _;

use crate::emit::{align, all_groups, highest_read, lanes, test_const};
use crate::solve::Plan;

const W: usize = 4;

pub fn emit(plan: &Plan) -> String {
    let mut out = String::new();

    let preamble = r#"
/// The checks that run on every block. Requires `neon`.
///
/// The highest index read is {HIGH}, and every load proves its own bound.
#[inline]
#[target_feature(enable = "neon")]
fn prefix(w: &Schedule) -> u32 {
    let mut acc0 = vdupq_n_u32(0);
    let mut acc1 = vdupq_n_u32(0);
"#;
    out.push_str(&preamble.replace("{HIGH}", &highest_read(plan, W).to_string()));

    for (n, (f, g)) in all_groups(plan, W).iter().enumerate() {
        let base = g[0].0;
        let acc = format!("acc{}", n % 2);
        let bits: Vec<&str> = g.iter().map(|m| m.1.as_str()).collect();

        out.push_str("\n    {\n");
        let _ = writeln!(out, "        let near = load::<{base}>(w);");
        let _ = writeln!(out, "        let far = load::<{}>(w);", base + f.offset);
        let (shift, bit) = align(f);
        // `vshrq_n_u32` rejects a zero shift, and no shift is needed when the
        // bits are already aligned.
        if shift == 0 {
            out.push_str("        let x = veorq_u32(near, far);\n");
        } else if shift > 0 {
            let _ = writeln!(
                out,
                "        let x = veorq_u32(near, vshrq_n_u32(far, {shift}));"
            );
        } else {
            let _ = writeln!(
                out,
                "        let x = veorq_u32(vshrq_n_u32(near, {}), far);",
                -shift
            );
        }
        // With a bit per lane, the mask goes through memory like the DV bits.
        let mask = test_const(
            g,
            bit,
            W,
            "vdupq_n_u32",
            |l| format!("splat([{l}])"),
            |b| format!("1 << {b}"),
        );
        let _ = writeln!(out, "        let set = vtstq_u32(x, {mask});");
        let _ = writeln!(
            out,
            "        let bits = splat([{}]);",
            lanes(&bits, "        ", 4)
        );
        // The DVs are ruled out where the bit is set, or where it is clear.
        let cleared = if f.clears_on == 1 {
            "vminq_u32(set, bits)"
        } else {
            "vqsubq_u32(bits, set)"
        };
        let _ = write!(
            out,
            "        {acc} = vorrq_u32({acc}, {cleared});\n    }}\n"
        );
    }

    out.push_str(
        r#"
    let acc = vorrq_u32(acc0, acc1);
    let folded = vorr_u32(vget_low_u32(acc), vget_high_u32(acc));
    let cleared = vget_lane_u32(vorr_u32(folded, vdup_lane_u32(folded, 1)), 0);
    !cleared
}
"#,
    );
    out
}
