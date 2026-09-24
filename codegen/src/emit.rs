//! Shared helpers for the emitters.
//!
//! [`crate::solve`] decides which checks run in the prefix and how they group.
//! Everything here is about turning that decision into source: lining up the
//! two bits, cutting a family into vector-sized pieces, and ordering the
//! pieces.

use crate::solve::{Family, Plan};
use crate::ubc::DV_NAMES;

/// Words in the expanded message schedule. Every emitted form takes
/// `&[u32; SCHEDULE]`, and a read past it is out of bounds.
const SCHEDULE: usize = 80;

/// How the two words are lined up so that one bit test covers both. Returns
/// the shift to apply and the bit to test. A positive shift moves `far` down,
/// a negative shift moves `near` down.
pub fn align(f: &Family) -> (i32, u32) {
    if f.far_bit >= f.near_bit {
        ((f.far_bit - f.near_bit) as i32, f.near_bit)
    } else {
        (-((f.near_bit - f.far_bit) as i32), f.far_bit)
    }
}

/// The bit a member with near bit `a` is tested at, once [`align`] has lined
/// up its two words by `shift`.
pub fn test_bit(shift: i32, a: u32) -> u32 {
    (a as i32 + shift.min(0)) as u32
}

/// Whether every lane of `g` that checks anything tests `bit`, so that one
/// broadcast constant serves the whole group.
fn uniform(g: &Group, bit: u32) -> bool {
    g.iter().filter_map(|m| m.2).all(|b| b == bit)
}

/// The constant a group's lanes are tested against, as source: a broadcast
/// of `1 << bit` where every lane that checks anything tests `bit`, and
/// otherwise `per_lane` of each lane's own, `one(bit)` where the lane tests
/// `bit` and 0 where it tests nothing.
pub fn test_const(
    g: &Group,
    bit: u32,
    width: usize,
    broadcast: &str,
    per_lane: impl Fn(&str) -> String,
    one: impl Fn(u32) -> String,
) -> String {
    if uniform(g, bit) {
        format!("{broadcast}(1 << {bit})")
    } else {
        let masks: Vec<String> = g
            .iter()
            .map(|m| m.2.map_or_else(|| "0".to_owned(), &one))
            .collect();
        per_lane(&lanes(&masks, "        ", width))
    }
}

/// The DV bits as source, for example `DV_I_43_0_BIT | DV_I_45_0_BIT`.
pub fn dv_expr(dvs: u32) -> String {
    let names: Vec<&str> = (0..32)
        .filter(|n| dvs >> n & 1 == 1)
        .map(|n| DV_NAMES[n])
        .collect();
    names.join(" | ")
}

/// One vector group: the checks that share a pair of loads, as `(i, DV bits,
/// the bit tested)` per lane. A lane with no member tests nothing.
pub type Group = Vec<(usize, String, Option<u32>)>;

/// Where the groups of a family go: the start of each window and how many
/// of the members `is` it takes, in order.
///
/// One pair of loads covers `width` consecutive `i`, so a window takes every
/// member that falls in it. A lane the family has no member for takes no DV
/// bits and clears nothing, which is what the group would have cost anyway
/// had the window been shorter. A window starts at its first member, or
/// earlier where that would take its far load past the end of the schedule.
/// The solver counts its budget with this same walk.
pub fn windows(is: &[usize], width: usize, offset: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut rest = is;
    while let Some(&first) = rest.first() {
        let base = first.min(SCHEDULE - width - offset);
        let n = rest.iter().take_while(|&&i| i < base + width).count();
        out.push((base, n));
        rest = &rest[n..];
    }
    out
}

/// Only a continuous range can share one vector load.
pub fn lane_groups(f: &Family, width: usize) -> Vec<Group> {
    let (shift, _) = align(f);
    let is: Vec<usize> = f.members.iter().map(|m| m.0).collect();
    let mut out = Vec::new();
    let mut rest = f.members.as_slice();
    for (base, n) in windows(&is, width, f.offset) {
        let (window, tail) = rest.split_at(n);
        out.push(
            (0..width)
                .map(|k| {
                    let i = base + k;
                    let member = window.iter().find(|(m, _, _)| *m == i);
                    (
                        i,
                        member.map_or_else(|| "0".to_owned(), |&(_, _, d)| dv_expr(d)),
                        member.map(|&(_, a, _)| test_bit(shift, a)),
                    )
                })
                .collect(),
        );
        rest = tail;
    }
    out
}

/// Every lane group with its family. Both vector emitters read this list, so
/// they always agree.
pub fn all_groups(plan: &Plan, width: usize) -> Vec<(&Family, Group)> {
    let mut groups: Vec<_> = plan
        .families
        .iter()
        .flat_map(|f| lane_groups(f, width).into_iter().map(move |g| (f, g)))
        .collect();
    // Earliest-available first: a group's loads cannot issue until the
    // compression has produced the highest schedule word it reads, and every
    // later OR into the same accumulator waits behind it.
    groups.sort_by_key(|(f, g)| g[0].0 + f.offset + width - 1);
    groups
}

/// The highest `w` index any group reads, for the module comment.
///
/// `Schedule::window` proves each emitted load's bound at compile time, so
/// this is the second line of defence rather than the only one. It fails
/// here, naming the table, rather than inside a generated `const` block.
pub fn highest_read(plan: &Plan, width: usize) -> usize {
    let high = all_groups(plan, width)
        .iter()
        .map(|(f, g)| g[0].0 + f.offset + width - 1)
        .max()
        .unwrap_or(0);
    assert!(
        high < SCHEDULE,
        "a group reads w[{high}], past the {SCHEDULE}-word schedule"
    );
    high
}

/// Lane initializers. A short group gets a mask that clears no bits, so the
/// group still fills one vector.
pub fn lanes(bits: &[impl AsRef<str>], indent: &str, width: usize) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    for i in 0..width {
        let value = bits.get(i).map_or("0", AsRef::as_ref);
        let _ = write!(out, "\n{indent}    {value},");
    }
    let _ = write!(out, "\n{indent}");
    out
}

/// A whole generated module: the check for one target, then the two halves
/// it calls.
pub fn module(name: &str, prefix: &str, tail: &str) -> String {
    let (what, feature) = match name {
        "scalar" => ("scalar form, for a target with no vector unit", None),
        "neon" => ("`aarch64` NEON form", Some("neon")),
        "sse2" => ("SSE2 form", Some("sse2")),
        "avx2" => ("AVX2 form", Some("avx2")),
        other => panic!("no module shape for {other}"),
    };

    const BODY: &str = r#"    let mask = prefix(w);

    // Every check only clears bits, so an empty mask settles the answer.
    if mask == 0 {
        return 0;
    }

    tail(w, mask)
}
"#;

    // `#[target_feature]` on a safe function makes the body safe code: the
    // intrinsics need no `unsafe`, and a caller that cannot prove the feature
    // still has to write one. The loads go through `crate::mem`.
    //
    // `#[inline]` here, on `prefix` and on `tail` makes the whole check
    // available to the block loop in every codegen unit. Without it, whether
    // it is inlined depends on how the crate happens to be partitioned, and it
    // runs once per block. `#[inline(always)]` is not allowed alongside
    // `#[target_feature]`.
    let check = match feature {
        None => format!(
            "/// Runs the whole check.\n#[inline(always)]\npub(super) fn check(w: &Schedule) -> u32 {{\n{BODY}"
        ),
        Some(f) => format!(
            "/// Runs the whole check. Requires `{f}`, so a caller that cannot\n/// prove the feature needs an `unsafe` block.\n#[inline]\n#[target_feature(enable = \"{f}\")]\npub(super) fn check(w: &Schedule) -> u32 {{\n{BODY}"
        ),
    };

    format!(
        "//! The UBC check, {what}.\n\
         //!\n\
         //! @generated by `codegen/` — do not edit by hand. Edit the table in\n\
         //! `codegen/src/ubc.rs`, the solver in `codegen/src/solve.rs` or this\n\
         //! target's plan in `codegen/src/main.rs`, and re-run it.\n\
         \n\
         #![forbid(unsafe_code)]\n\
         \n\
         use crate::Schedule;\n\
         use crate::ubc_check::*;\n\
         {}\n\
         {check}\n{prefix}\n{tail}",
        preamble(name)
    )
}

/// The arch import and the loads, for one target.
///
/// Every load a generated body does goes through `load`, which takes its
/// window from `Schedule::window` and hands it to `crate::mem`. Neither
/// here nor in the hundreds of lines that follow is there any `unsafe`.
fn preamble(name: &str) -> String {
    match name {
        "scalar" => String::new(),
        "neon" => format!(
            "\nuse crate::mem::load_u32x4;\nuse core::arch::aarch64::*;\n{}{}",
            load("neon", 4, "uint32x4_t", "load_u32x4"),
            SPLAT
        ),
        "sse2" => format!(
            "\nuse crate::mem::load_u32x4;{X86_IMPORTS}{}",
            load("sse2", 4, "__m128i", "load_u32x4")
        ),
        "avx2" => format!(
            "\nuse crate::mem::load_u32x8;{X86_IMPORTS}{}",
            load("avx2", 8, "__m256i", "load_u32x8")
        ),
        other => panic!("no preamble for {other}"),
    }
}

const X86_IMPORTS: &str = r#"
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
"#;

/// NEON has no intrinsic that takes four lanes directly, so the DV bits go
/// through memory like the schedule words do.
const SPLAT: &str = r#"
/// The DV bits of a group, as a vector.
#[inline]
#[target_feature(enable = "neon")]
fn splat(bits: [u32; 4]) -> uint32x4_t {
    load_u32x4(&bits)
}
"#;

/// The one load a generated body uses.
///
/// `I` is a const parameter, so `Schedule::window` proves the bound at
/// compile time at the call site, and it holds for every plan this generator
/// can emit rather than only the one committed.
fn load(feature: &str, width: usize, ty: &str, helper: &str) -> String {
    format!(
        r#"
/// The {width} schedule words of steps `I..I + {width}`.
///
/// One load on either layout. A mirrored one hands back its lanes in the
/// other order, which each group's DV bits are emitted to match.
#[inline]
#[target_feature(enable = "{feature}")]
fn load<const I: usize>(w: &Schedule) -> {ty} {{
    {helper}(w.window::<I, {width}>())
}}
"#
    )
}
