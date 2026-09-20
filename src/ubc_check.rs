//! The unavoidable-bitconditions check: a cheap filter for whether a block
//! could be part of a collision attack.
//!
//! # What it does
//!
//! A SHA-1 collision attack must follow one of 32 known *disturbance vectors*
//! (DVs). Each DV makes certain bits of the expanded message relate in a fixed
//! way. [`ubc_check`] tests those relations and returns a mask. A set bit
//! means that the DV is still possible for this block.
//!
//! A zero mask rejects all known attacks. About 95% of blocks give one. A
//! non-zero mask sends the block to the recompression check in
//! [`crate::block`], which is much slower but gives a definite answer.
//! This check is a filter, not a decision. A false positive costs time. It
//! cannot cause a wrong result.
//!
//! # Why it looks the way it does
//!
//! Every check has the shape
//!
//! ```text
//! mask &= <one bit condition> | !(<the DVs that condition rules out>);
//! ```
//!
//! It tests one bit of `w[i]` against one bit of `w[j]` and clears the DVs
//! that the result rules out. A check only clears bits, so the order of the
//! checks does not matter.
//!
//! For one DV the published checks are one basis of a linear space, not a
//! fixed list. Any basis with the same span gives the same mask, so
//! `codegen/` is free to pick a different one. It solves once per target,
//! because each wants a different basis: a check costs one statement in
//! [`scalar`] but a quarter of a load pair in [`neon`] and [`sse2`], and an
//! eighth in [`avx2`]. Every target therefore gets its own module, holding
//! its own whole check.
//!
//! Inside a module the checks are in two parts. The prefix runs on every
//! block and is most of the cost. The tail runs the rest one at a time behind
//! guards, because by then the mask is sparse and a guard that skips a check
//! is worth more than the check costs. Where the split falls is what each
//! target solves for.
//!
//! # Provenance
//!
//! Both tables are the paper's. The conditions are its Table 4 and Table 5,
//! in `codegen/src/ubc.rs`, which `codegen/src/solve.rs` then rebases. The DV
//! table is not stored at all: [`SHA1_DVS`] is built from the two families of
//! section 4.4 and the local-collision sum of section 4.3, so the only data
//! here is one 16-word seed per family.
//!
//! Marc Stevens and Dan Shumow also wrote
//! [sha1collisiondetection](https://github.com/cr-marcstevens/sha1collisiondetection),
//! the implementation that accompanies the paper. It is licensed under MIT.
//!
//! Two tests compare this module against that original. `matches_c_reference`
//! runs 100,000 message schedules and compares the flagged count and a
//! checksum of the mask stream against values taken from the C.
//! `dv_table_matches_c` checksums the derived DV table.

const DV_I_43_0_BIT: u32 = 1 << 0;
const DV_I_44_0_BIT: u32 = 1 << 1;
const DV_I_45_0_BIT: u32 = 1 << 2;
const DV_I_46_0_BIT: u32 = 1 << 3;
const DV_I_46_2_BIT: u32 = 1 << 4;
const DV_I_47_0_BIT: u32 = 1 << 5;
const DV_I_47_2_BIT: u32 = 1 << 6;
const DV_I_48_0_BIT: u32 = 1 << 7;
const DV_I_48_2_BIT: u32 = 1 << 8;
const DV_I_49_0_BIT: u32 = 1 << 9;
const DV_I_49_2_BIT: u32 = 1 << 10;
const DV_I_50_0_BIT: u32 = 1 << 11;
const DV_I_50_2_BIT: u32 = 1 << 12;
const DV_I_51_0_BIT: u32 = 1 << 13;
const DV_I_51_2_BIT: u32 = 1 << 14;
const DV_I_52_0_BIT: u32 = 1 << 15;
const DV_II_45_0_BIT: u32 = 1 << 16;
const DV_II_46_0_BIT: u32 = 1 << 17;
const DV_II_46_2_BIT: u32 = 1 << 18;
const DV_II_47_0_BIT: u32 = 1 << 19;
const DV_II_48_0_BIT: u32 = 1 << 20;
const DV_II_49_0_BIT: u32 = 1 << 21;
const DV_II_49_2_BIT: u32 = 1 << 22;
const DV_II_50_0_BIT: u32 = 1 << 23;
const DV_II_50_2_BIT: u32 = 1 << 24;
const DV_II_51_0_BIT: u32 = 1 << 25;
const DV_II_51_2_BIT: u32 = 1 << 26;
const DV_II_52_0_BIT: u32 = 1 << 27;
const DV_II_53_0_BIT: u32 = 1 << 28;
const DV_II_54_0_BIT: u32 = 1 << 29;
const DV_II_55_0_BIT: u32 = 1 << 30;
const DV_II_56_0_BIT: u32 = 1 << 31;

/// Disturbance Vector (DV).
#[derive(Clone, Copy)]
pub(crate) struct Info {
    /// Which stored intermediate state this DV's recompression starts from.
    ///
    /// `testt` in the C original.
    pub(crate) recompress_from: RecompressFrom,
    /// This DV's bit in the mask [`ubc_check`] returns.
    ///
    /// `maskb` in the C original.
    pub(crate) mask_bit: i32,
    /// The expanded message block XOR-difference defined by the DV.
    pub(crate) dm: [u32; 80],
}

/// The compression step a DV's recompression starts from.
///
/// Detection runs the compression again from a stored intermediate state and
/// not from the start. The DV selects which state applies. The C original
/// calls this `testt` and uses these step numbers as its values.
#[derive(Copy, Clone)]
#[repr(u32)]
pub(crate) enum RecompressFrom {
    Step58 = 58,
    Step65 = 65,
}

/// Which family of disturbance vector a DV belongs to.
#[derive(Clone, Copy, PartialEq)]
enum DvType {
    I,
    II,
}

/// The two base message differences, one per DV family, as their first 16
/// words. [`expand`] derives the rest.
///
/// Each seed is the difference for the *highest* K in its family. The others
/// are the same sequence at a later offset, so one seed covers all sixteen.
const TYPE_I_SEED: [u32; 16] = [
    0x04000010, 0xe8000000, 0x0800000c, 0x18000000, 0xb800000a, 0xc8000010, 0x2c000010, 0xf4000014,
    0xb4000008, 0x08000000, 0x9800000c, 0xd8000010, 0x08000010, 0xb8000010, 0x98000000, 0x60000000,
];
const TYPE_II_SEED: [u32; 16] = [
    0x2600001a, 0x00000010, 0x0400001c, 0xcc000014, 0x0c000002, 0xc0000010, 0xb400001c, 0x3c000004,
    0xbc00001a, 0x20000010, 0x2400001c, 0xec000014, 0x0c000002, 0xc0000010, 0xb400001c, 0x2c000004,
];

/// The K each seed above is the difference for.
const TYPE_I_K: u32 = 52;
const TYPE_II_K: u32 = 56;

/// Runs SHA-1's message expansion over a seed.
///
/// A message difference obeys the same recurrence as the schedule, so only
/// the first 16 words are data. 96 words cover every offset in a family.
const fn expand(seed: [u32; 16]) -> [u32; 96] {
    let mut w = [0u32; 96];
    let mut t = 0;
    while t < 16 {
        w[t] = seed[t];
        t += 1;
    }
    while t < 96 {
        w[t] = (w[t - 3] ^ w[t - 8] ^ w[t - 14] ^ w[t - 16]).rotate_left(1);
        t += 1;
    }
    w
}

const TYPE_I_DM: [u32; 96] = expand(TYPE_I_SEED);
const TYPE_II_DM: [u32; 96] = expand(TYPE_II_SEED);

/// The message difference for one DV. This is the sequence of its family, read
/// from the offset that puts the disturbance at step `k` and rotated to bit
/// `b`.
const fn message_difference(family: DvType, k: u32, b: u32) -> [u32; 80] {
    let (base, base_k) = match family {
        DvType::I => (&TYPE_I_DM, TYPE_I_K),
        DvType::II => (&TYPE_II_DM, TYPE_II_K),
    };
    let offset = (base_k - k) as usize;

    let mut dm = [0u32; 80];
    let mut i = 0;
    while i < 80 {
        dm[i] = base[i + offset].rotate_left(b);
        i += 1;
    }
    dm
}

/// The DVs to check. Each row gives the family, the step K of the disturbance,
/// and the bit B in that step. The last column is the stored state that its
/// recompression starts from.
///
/// The C original writes out all 80 words of every difference. That is 500
/// lines of hex that a reader cannot verify. The two seeds above give the same
/// values, and `dv_table_matches_c` compares the result against them.
const DVS: [(DvType, u32, u32, RecompressFrom); 32] = [
    (DvType::I, 43, 0, RecompressFrom::Step58),
    (DvType::I, 44, 0, RecompressFrom::Step58),
    (DvType::I, 45, 0, RecompressFrom::Step58),
    (DvType::I, 46, 0, RecompressFrom::Step58),
    (DvType::I, 46, 2, RecompressFrom::Step58),
    (DvType::I, 47, 0, RecompressFrom::Step58),
    (DvType::I, 47, 2, RecompressFrom::Step58),
    (DvType::I, 48, 0, RecompressFrom::Step58),
    (DvType::I, 48, 2, RecompressFrom::Step58),
    (DvType::I, 49, 0, RecompressFrom::Step58),
    (DvType::I, 49, 2, RecompressFrom::Step58),
    (DvType::I, 50, 0, RecompressFrom::Step65),
    (DvType::I, 50, 2, RecompressFrom::Step65),
    (DvType::I, 51, 0, RecompressFrom::Step65),
    (DvType::I, 51, 2, RecompressFrom::Step65),
    (DvType::I, 52, 0, RecompressFrom::Step65),
    (DvType::II, 45, 0, RecompressFrom::Step58),
    (DvType::II, 46, 0, RecompressFrom::Step58),
    (DvType::II, 46, 2, RecompressFrom::Step58),
    (DvType::II, 47, 0, RecompressFrom::Step58),
    (DvType::II, 48, 0, RecompressFrom::Step58),
    (DvType::II, 49, 0, RecompressFrom::Step58),
    (DvType::II, 49, 2, RecompressFrom::Step58),
    (DvType::II, 50, 0, RecompressFrom::Step65),
    (DvType::II, 50, 2, RecompressFrom::Step65),
    (DvType::II, 51, 0, RecompressFrom::Step65),
    (DvType::II, 51, 2, RecompressFrom::Step65),
    (DvType::II, 52, 0, RecompressFrom::Step65),
    (DvType::II, 53, 0, RecompressFrom::Step65),
    (DvType::II, 54, 0, RecompressFrom::Step65),
    (DvType::II, 55, 0, RecompressFrom::Step65),
    (DvType::II, 56, 0, RecompressFrom::Step65),
];

/// The list of SHA-1 Disturbance Vectors (DV) to check.
pub(crate) const SHA1_DVS: [Info; 32] = build_dvs();

const fn build_dvs() -> [Info; 32] {
    let mut out = [Info {
        recompress_from: RecompressFrom::Step58,
        mask_bit: 0,
        dm: [0u32; 80],
    }; 32];

    let mut i = 0;
    while i < out.len() {
        let (family, k, b, recompress_from) = DVS[i];
        out[i] = Info {
            recompress_from,
            // Each DV uses the bit at its own position in this table.
            mask_bit: i as i32,
            dm: message_difference(family, k, b),
        };
        i += 1;
    }
    out
}
mod scalar;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "sse2"
))]
mod sse2;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "sse2",
    any(feature = "std", target_feature = "avx2")
))]
mod avx2;

/// Whether this CPU has AVX2, which no target guarantees.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "sse2",
    any(feature = "std", target_feature = "avx2")
))]
#[inline(always)]
fn has_avx2() -> bool {
    #[cfg(feature = "std")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(feature = "std"))]
    {
        cfg!(target_feature = "avx2")
    }
}

/// Checks the unavoidable bitconditions of every listed DV against an expanded
/// message block. Returns a mask. A set bit marks a DV that met all of its
/// conditions and still needs the recompression check.
///
/// `scalar_only` keeps to [`scalar`] on a machine that has a vector unit.
/// Tests and benchmarks use it to reach that path.
#[inline]
pub(crate) fn ubc_check(w: &[u32; 80], scalar_only: bool) -> u32 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    if !scalar_only {
        // SAFETY: the cfg guarantees `neon`. All reads stay in `w`.
        return unsafe { neon::check(w) };
    }

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
    ))]
    if !scalar_only {
        #[cfg(any(feature = "std", target_feature = "avx2"))]
        if has_avx2() {
            // SAFETY: just detected. All reads stay in `w`.
            return unsafe { avx2::check(w) };
        }
        // SAFETY: the cfg guarantees `sse2`. All reads stay in `w`.
        return unsafe { sse2::check(w) };
    }

    // On a target with neither there is nothing to turn off.
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "sse2"
        )
    )))]
    let _ = scalar_only;

    scalar::check(w)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic message schedules. The generator matches the one that
    /// produced the reference constants from the C implementation.
    fn schedules(n: usize, mut f: impl FnMut(&[u32; 80])) {
        let mut seed = 0x1234_5678_9abc_def0u64;
        for _ in 0..n {
            let mut w = [0u32; 80];
            for word in w.iter_mut().take(16) {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                *word = seed as u32;
            }
            for t in 16..80 {
                w[t] = (w[t - 3] ^ w[t - 8] ^ w[t - 14] ^ w[t - 16]).rotate_left(1);
            }
            f(&w);
        }
    }

    /// Every target solves for its own plan, so the forms share no code. They
    /// must still agree: a check that clears too few bits gives correct
    /// digests and only causes more recompressions, so no other test sees it.
    #[test]
    fn every_form_matches_scalar() {
        schedules(20_000, |w| {
            let want = scalar::check(w);
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            // SAFETY: the cfg guarantees `neon`.
            assert_eq!(unsafe { neon::check(w) }, want, "neon diverged");
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "sse2"
            ))]
            // SAFETY: the cfg guarantees `sse2`.
            assert_eq!(unsafe { sse2::check(w) }, want, "sse2 diverged");
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "sse2",
                any(feature = "std", target_feature = "avx2")
            ))]
            if has_avx2() {
                // SAFETY: just detected.
                assert_eq!(unsafe { avx2::check(w) }, want, "avx2 diverged");
            }
        });
    }

    /// Compares the derived DV table against the values in the C original.
    ///
    /// The checksum covers the written-out table that this crate had before
    /// `expand` replaced it. That table was a copy of the one in
    /// `ubc_check.c`. A wrong seed or recurrence stops the detection from
    /// matching upstream, and this test detects that.
    #[test]
    fn dv_table_matches_c() {
        const EXPECTED: u64 = 0xab2c_9b22_b0b0_b952;

        let mut checksum = 0xcbf2_9ce4_8422_2325u64;
        let mut feed = |v: u32| {
            for byte in v.to_le_bytes() {
                checksum ^= u64::from(byte);
                checksum = checksum.wrapping_mul(0x100_0000_01b3);
            }
        };
        for dv in &SHA1_DVS {
            feed(dv.recompress_from as u32);
            feed(dv.mask_bit as u32);
            for word in dv.dm {
                feed(word);
            }
        }
        assert_eq!(checksum, EXPECTED, "DV table diverged from the C original");
    }

    /// Compares `ubc_check` against the C original. Both numbers come from a
    /// run of upstream `lib/ubc_check.c` over this same schedule stream. The
    /// count is the more sensitive of the two. If a check clears too few bits,
    /// the number of flagged blocks increases immediately.
    #[test]
    fn matches_c_reference() {
        const C_NONZERO: u32 = 4767;
        const C_CHECKSUM: u64 = 0x8c03_7397_6647_17a3;

        let mut nonzero = 0u32;
        let mut checksum = 0xcbf2_9ce4_8422_2325u64;
        schedules(100_000, |w| {
            let mask = ubc_check(w, false);
            if mask != 0 {
                nonzero += 1;
            }
            for byte in mask.to_le_bytes() {
                checksum ^= u64::from(byte);
                checksum = checksum.wrapping_mul(0x100_0000_01b3);
            }
        });

        assert_eq!(nonzero, C_NONZERO, "flagged-block count diverged from C");
        assert_eq!(checksum, C_CHECKSUM, "mask stream diverged from C");
    }
}
