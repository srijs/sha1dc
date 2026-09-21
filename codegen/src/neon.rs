//! Emits the `aarch64` form.
//!
//! Four checks per vector. Load `w[i..i + 4]` and `w[i + offset..]`, align the
//! two bits, then test them. `vtstq_u32` sets a lane to all ones if the tested
//! bit is set. This replaces the shift, and, and negate of the scalar form
//! with one instruction. The DV bits of the lane are then masked in and added
//! to an accumulator.
//!
//! There are two accumulators, so the final OR chain does not serialize the
//! groups. The target settles `neon`, so a `cfg` selects this form.

use std::fmt::Write as _;

use crate::emit::{align, all_groups, highest_read, lanes};
use crate::solve::Plan;

const W: usize = 4;

pub fn emit(plan: &Plan) -> String {
    let mut out = String::new();

    let preamble = r#"
/// The checks that run on every block. Requires `neon`.
///
/// The highest index read is {HIGH}, and every load proves its own bound.
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
        if f.clears_on == 1 {
            let _ = writeln!(
                out,
                "        let hit = vtstq_u32(x, vdupq_n_u32(1 << {bit}));"
            );
        } else {
            let _ = writeln!(
                out,
                "        let hit = vceqq_u32(vandq_u32(x, vdupq_n_u32(1 << {bit})), vdupq_n_u32(0));"
            );
        }
        let _ = writeln!(
            out,
            "        let bits = splat([{}]);",
            lanes(&bits, "        ", 4)
        );
        let _ = write!(
            out,
            "        {acc} = vorrq_u32({acc}, vandq_u32(hit, bits));\n    }}\n"
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
