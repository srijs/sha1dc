//! Emits the AVX2 form, used on `x86` and `x86_64` where the CPU has it.
//!
//! The same structure as the SSE2 form with twice the lanes: eight checks per
//! vector instead of four. AVX2 has no `vtst` either, so it compares the masked
//! bit against zero and uses `andnot` to keep the lanes where the bit was set.
//!
//! AVX2 is not baseline on any target, so unlike the other vector forms the
//! caller detects it at run time. The final fold drops to 128 bits first,
//! because that is where the horizontal shuffles are.

use std::fmt::Write as _;

use crate::emit::{align, all_groups, header, highest_read, lanes};
use crate::solve::Plan;

const W: usize = 8;

pub fn emit(plan: &Plan) -> String {
    let mut out = header("The unconditional UBC checks, AVX2 form.");

    let preamble = r#"
/// # Safety
///
/// Requires `avx2`. Every load stays in `w`. The highest index read is {HIGH}.
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn mask(w: &[u32; 80]) -> u32 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let p = w.as_ptr();
    let zero = _mm256_setzero_si256();
    let mut acc0 = zero;
    let mut acc1 = zero;
"#;
    out.push_str(&preamble.replace("{HIGH}", &highest_read(plan, W).to_string()));

    for (n, (f, g)) in all_groups(plan, W).iter().enumerate() {
        let base = g[0].0;
        let acc = format!("acc{}", n % 2);
        let bits: Vec<&str> = g.iter().map(|m| m.1.as_str()).collect();
        let (shift, bit) = align(f);

        out.push_str("\n    {\n");
        let _ = write!(
            out,
            "        let near = _mm256_loadu_si256(p.add({base}).cast());\n"
        );
        let _ = write!(
            out,
            "        let far = _mm256_loadu_si256(p.add({}).cast());\n",
            base + f.offset
        );
        if shift == 0 {
            out.push_str("        let x = _mm256_xor_si256(near, far);\n");
        } else if shift > 0 {
            let _ = write!(
                out,
                "        let x = _mm256_xor_si256(near, _mm256_srli_epi32(far, {shift}));\n"
            );
        } else {
            let _ = write!(
                out,
                "        let x = _mm256_xor_si256(_mm256_srli_epi32(near, {}), far);\n",
                -shift
            );
        }
        let _ = write!(
            out,
            "        let tested = _mm256_and_si256(x, _mm256_set1_epi32(1 << {bit}));\n"
        );
        out.push_str("        let miss = _mm256_cmpeq_epi32(tested, zero);\n");
        let cast: Vec<String> = bits.iter().map(|b| format!("({b}) as i32")).collect();
        let cast: Vec<&str> = cast.iter().map(String::as_str).collect();
        let _ = write!(
            out,
            "        let bits = _mm256_setr_epi32({});\n",
            lanes(&cast, "        ", W)
        );
        let _ = write!(
            out,
            "        {acc} = _mm256_or_si256({acc}, {});\n    }}\n",
            if f.clears_on == 1 {
                "_mm256_andnot_si256(miss, bits)"
            } else {
                "_mm256_and_si256(miss, bits)"
            }
        );
    }

    out.push_str(
        r#"
    let acc = _mm256_or_si256(acc0, acc1);
    let acc = _mm_or_si128(
        _mm256_castsi256_si128(acc),
        _mm256_extracti128_si256::<1>(acc),
    );
    let acc = _mm_or_si128(acc, _mm_shuffle_epi32(acc, 0b01_00_11_10));
    let acc = _mm_or_si128(acc, _mm_shuffle_epi32(acc, 0b10_11_00_01));
    !(_mm_cvtsi128_si32(acc) as u32)
}
"#,
    );
    out
}
