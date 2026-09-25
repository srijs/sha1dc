//! Emits the SSE2 form, used on both `x86` and `x86_64`.
//!
//! Like the NEON form, but SSE2 has no `vtst`: it compares the masked bit with
//! zero and keeps the lanes where it was *set* with `andnot`. `i586`, without
//! SSE2, uses the scalar form.

use std::fmt::Write as _;

use crate::emit::{align, all_groups, highest_read, lanes, test_const};
use crate::solve::Plan;

const W: usize = 4;

pub fn emit(plan: &Plan) -> String {
    let mut out = String::new();

    let preamble = r#"
/// The checks that run on every block. Requires `sse2`.
///
/// The highest index read is {HIGH}, and every load proves its own bound.
#[inline]
#[target_feature(enable = "sse2")]
fn prefix(w: &Schedule) -> u32 {
    let zero = _mm_setzero_si128();
    let mut acc0 = zero;
    let mut acc1 = zero;
"#;
    out.push_str(&preamble.replace("{HIGH}", &highest_read(plan, W).to_string()));

    for (n, (f, g)) in all_groups(plan, W).iter().enumerate() {
        let base = g[0].0;
        let acc = format!("acc{}", n % 2);
        let bits: Vec<&str> = g.iter().map(|m| m.1.as_str()).collect();

        out.push_str("\n    {\n");
        let _ = writeln!(out, "        let near = load::<{base}>(w);");
        let _ = writeln!(out, "        let far = load::<{}>(w);", base + f.offset);
        let shift = align(f);
        if shift == 0 {
            out.push_str("        let x = _mm_xor_si128(near, far);\n");
        } else if shift > 0 {
            let _ = writeln!(
                out,
                "        let x = _mm_xor_si128(near, _mm_srli_epi32(far, {shift}));"
            );
        } else {
            let _ = writeln!(
                out,
                "        let x = _mm_xor_si128(_mm_srli_epi32(near, {}), far);",
                -shift
            );
        }
        let mask = test_const(
            g,
            W,
            "_mm_set1_epi32",
            |l| format!("_mm_set_epi32({l})"),
            |b| format!("(1u32 << {b}) as i32"),
        );
        let _ = writeln!(out, "        let tested = _mm_and_si128(x, {mask});");
        out.push_str("        let miss = _mm_cmpeq_epi32(tested, zero);\n");
        // An empty lane is already an `i32`; wrapping it would only add
        // parentheses to the generated source.
        let cast: Vec<String> = bits
            .iter()
            .map(|b| {
                if *b == "0" {
                    (*b).to_owned()
                } else {
                    format!("({b}) as i32")
                }
            })
            .collect();
        let cast: Vec<&str> = cast.iter().map(String::as_str).collect();
        let _ = writeln!(
            out,
            // The lanes run backwards over a mirrored schedule, and `set` takes
            // its arguments the other way round from `setr`, so the same list
            // lands in the right places.
            "        let bits = _mm_set_epi32({});",
            lanes(&cast, "        ", 4)
        );
        let _ = write!(
            out,
            "        {acc} = _mm_or_si128({acc}, {});\n    }}\n",
            if f.clears_on == 1 {
                "_mm_andnot_si128(miss, bits)"
            } else {
                "_mm_and_si128(miss, bits)"
            }
        );
    }

    out.push_str(
        r#"
    let acc = _mm_or_si128(acc0, acc1);
    let acc = _mm_or_si128(acc, _mm_shuffle_epi32(acc, 0b01_00_11_10));
    let acc = _mm_or_si128(acc, _mm_shuffle_epi32(acc, 0b10_11_00_01));
    !(_mm_cvtsi128_si32(acc) as u32)
}
"#,
    );
    out
}
