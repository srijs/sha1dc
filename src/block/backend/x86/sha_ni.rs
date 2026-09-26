//! SHA-1 `x86`/`x86_64` backend built on the SHA-NI instructions.
//!
//! Computes a block's digest and spills the expanded message schedule, which
//! the collision detection needs. The spill is the difference from an
//! ordinary SHA-1, which keeps the schedule in registers.
//!
//! `sha1rnds4` does four rounds at a time, so the work is in groups of four.
//! `sha1nexte` carries the fifth working word between groups, which is why
//! two of them alternate. The schedule runs four groups ahead of the rounds.
//! Words 16 to 31 come from SHA-NI's schedule instructions,
//! `sha1msg1`/`sha1msg2`. The rest come from plain SSE instructions, by
//! another form of the recurrence (see [`expand_rol2`]). Sapphire Rapids
//! microcodes `sha1msg2`; with four a block rather than sixteen, the
//! compression runs about a tenth faster there.

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
compile_error!("the sha_ni backend needs an x86 or x86_64 target");

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::{REVERSE, expand_rol2, spill};
use crate::Schedule;
use crate::block::rounds;
use crate::mem::{load_u8x16, load_u32x4, store_u32x4};
use crate::ubc_check::RecompressFrom;

/// The four message words starting at byte `I`, in native order.
///
/// `I` is a const parameter, so a load past the end of the block is a compile
/// error at the call site.
#[inline]
#[target_feature(enable = "sse2,ssse3")]
fn msg<const I: usize>(block: &[u8; 64], swap: __m128i) -> __m128i {
    const { assert!(I + 16 <= 64, "the load runs past the block") }
    let raw = load_u8x16(block[I..I + 16].try_into().unwrap());
    _mm_shuffle_epi8(raw, swap)
}

/// The four schedule words of steps `T..T + 4`, exclusive-ored with the
/// partner's difference, in the order the rounds want them.
///
/// The read side of [`spill`], with `T` const for the same reason. The spill is already reversed, being mirrored
/// for that reason; the difference table is in step order, so it is the one
/// that turns round.
#[inline]
#[target_feature(enable = "sse2,ssse3")]
fn words<const T: usize>(m1: &Schedule, dm: &[u32; 80]) -> __m128i {
    const { assert!(T + 4 <= 80, "the group runs past the schedule") }
    let spilled = load_u32x4(m1.window::<T, 4>());
    let diff = load_u32x4(dm[T..T + 4].try_into().unwrap());
    _mm_xor_si128(spilled, _mm_shuffle_epi32(diff, REVERSE))
}

/// The `abcd` half of the state. The fifth word is carried on its own.
#[inline]
#[target_feature(enable = "sse2")]
fn load_abcd(state: &[u32; 5]) -> __m128i {
    _mm_shuffle_epi32(load_u32x4(state.first_chunk().unwrap()), REVERSE)
}

/// Writes `abcd` back, leaving the fifth word alone.
#[inline]
#[target_feature(enable = "sse2")]
fn store_abcd(state: &mut [u32; 5], abcd: __m128i) {
    store_u32x4(
        state.first_chunk_mut().unwrap(),
        _mm_shuffle_epi32(abcd, REVERSE),
    );
}

/// Schedule words `4k..4k + 4` for `k` from 4 to 7, from the four groups
/// before, with the SHA-NI expansion instructions.
#[inline]
#[target_feature(enable = "sha,sse2")]
fn expand_ni(a: __m128i, b: __m128i, c: __m128i, d: __m128i) -> __m128i {
    _mm_sha1msg2_epu32(_mm_xor_si128(_mm_sha1msg1_epu32(a, b), c), d)
}

/// Whether this CPU has the instructions this module needs.
///
/// A `std` build asks the OS at run time. Run-time detection needs `cpuid`,
/// so a `no_std` build uses `target_feature` only, which needs the features
/// on the command line, for example `-C target-feature=+sha,+sse4.1`.
fn has_sha_ni() -> bool {
    #[cfg(feature = "std")]
    {
        std::arch::is_x86_feature_detected!("sha")
            && std::arch::is_x86_feature_detected!("sse2")
            && std::arch::is_x86_feature_detected!("ssse3")
            && std::arch::is_x86_feature_detected!("sse4.1")
    }
    #[cfg(not(feature = "std"))]
    {
        cfg!(all(
            target_feature = "sha",
            target_feature = "sse2",
            target_feature = "ssse3",
            target_feature = "sse4.1"
        ))
    }
}

/// Whether this CPU has AVX, which one with SHA-NI need not: the Atoms have
/// SHA-NI without it.
fn has_avx() -> bool {
    #[cfg(feature = "std")]
    {
        std::arch::is_x86_feature_detected!("avx")
    }
    #[cfg(not(feature = "std"))]
    {
        cfg!(target_feature = "avx")
    }
}

/// Proof that this CPU has what this module needs, and whether it has AVX
/// too. Only [`ShaNi::detect`] makes one, so holding one makes the
/// compression safe to call.
#[derive(Clone, Copy)]
pub(crate) struct ShaNi {
    avx: bool,
}

impl ShaNi {
    /// Checks this CPU. `None` without SHA-NI.
    pub(crate) fn detect() -> Option<Self> {
        has_sha_ni().then(|| Self { avx: has_avx() })
    }

    /// Compresses one block, built with AVX's three-operand
    /// forms where there is AVX, which the schedule needs no copies in.
    #[inline]
    pub(crate) fn compress_spill(
        self,
        state: &mut [u32; 5],
        block: &[u8; 64],
        w: &mut Schedule,
        at_60: &mut [u32; 5],
        at_64: &mut [u32; 5],
    ) {
        if self.avx {
            // SAFETY: `detect` found SHA-NI and AVX.
            unsafe { compress_spill_avx(state, block, w, at_60, at_64) };
        } else {
            // SAFETY: `detect` found SHA-NI.
            unsafe { compress_spill(state, block, w, at_60, at_64) };
        }
    }

    /// [`recompress`].
    #[inline]
    pub(crate) fn recompress(
        self,
        step: RecompressFrom,
        m1: &Schedule,
        dm: &[u32; 80],
        state: &[u32; 5],
        chaining_out: &[u32; 5],
    ) -> bool {
        // SAFETY: `detect` found SHA-NI.
        unsafe { recompress(step, m1, dm, state, chaining_out) }
    }
}

/// One group of four rounds, on the words of step `$t` on, spilling them to
/// `$w`. The working word arrives in `live` and the one for the next group is
/// put aside in `held`. With `=> $at`, also stores the state it starts from.
macro_rules! rounds {
    ($w:ident, $v:ident, $abcd:ident, $t:literal, $k:expr, $live:ident, $held:ident) => {{
        spill::<$t>($w, $v[$t / 4]);
        $live = _mm_sha1nexte_epu32($live, $v[$t / 4]);
        $held = $abcd;
        $abcd = _mm_sha1rnds4_epu32($abcd, $live, $k);
    }};
    ($w:ident, $v:ident, $abcd:ident, $t:literal, $k:expr, $live:ident, $held:ident => $at:ident) => {{
        spill::<$t>($w, $v[$t / 4]);
        $live = _mm_sha1nexte_epu32($live, $v[$t / 4]);
        // As held: `abcd` reversed in the first four words, and E + W in the
        // fifth, from the top of a store one word further on.
        store_u32x4((&mut $at[1..5]).try_into().unwrap(), $live);
        store_u32x4($at.first_chunk_mut().unwrap(), $abcd);
        $held = $abcd;
        $abcd = _mm_sha1rnds4_epu32($abcd, $live, $k);
    }};
}

/// Computes group `$k` of the schedule `$v` by the rol-2 recurrence.
macro_rules! rol2 {
    ($v:ident, $k:literal) => {
        $v[$k] = expand_rol2($v[$k - 8], $v[$k - 7], $v[$k - 4], $v[$k - 2], $v[$k - 1])
    };
}

/// A compression that spills the schedule, built with `$features`. Written
/// once and emitted for each build, so that each is compiled with its own
/// features rather than relying on one being inlined into the other.
macro_rules! compress_spill_fn {
    ($(#[$doc:meta])* $name:ident, $features:literal) => {
        $(#[$doc])*
        #[inline]
        #[target_feature(enable = $features)]
        fn $name(
            state: &mut [u32; 5],
            block: &[u8; 64],
            w: &mut Schedule,
            at_60: &mut [u32; 5],
            at_64: &mut [u32; 5],
        ) {
            // Turns the big-endian message into native order, reversing the
            // four words along with the bytes.
            let swap = _mm_set_epi64x(0x0001_0203_0405_0607, 0x0809_0A0B_0C0D_0E0F);

            let mut abcd = load_abcd(state);
            let abcd_in = abcd;
            let e_in = _mm_set_epi32(state[4] as i32, 0, 0, 0);

            // `v[k]` holds the words of steps `4k..4k + 4`, in the order the
            // rounds take them. Each group of rounds computes the one four
            // groups on.
            let mut v = [_mm_setzero_si128(); 20];
            v[0] = msg::<0>(block, swap);
            v[1] = msg::<16>(block, swap);
            v[2] = msg::<32>(block, swap);
            v[3] = msg::<48>(block, swap);

            spill::<0>(w, v[0]);
            let mut e0 = _mm_add_epi32(e_in, v[0]);
            let mut e1 = abcd;
            abcd = _mm_sha1rnds4_epu32(abcd, e0, 0);
            v[4] = expand_ni(v[0], v[1], v[2], v[3]);

            rounds!(w, v, abcd, 4, 0, e1, e0);
            v[5] = expand_ni(v[1], v[2], v[3], v[4]);
            rounds!(w, v, abcd, 8, 0, e0, e1);
            v[6] = expand_ni(v[2], v[3], v[4], v[5]);
            rounds!(w, v, abcd, 12, 0, e1, e0);
            v[7] = expand_ni(v[3], v[4], v[5], v[6]);
            rounds!(w, v, abcd, 16, 0, e0, e1);
            rol2!(v, 8);
            rounds!(w, v, abcd, 20, 1, e1, e0);
            rol2!(v, 9);
            rounds!(w, v, abcd, 24, 1, e0, e1);
            rol2!(v, 10);
            rounds!(w, v, abcd, 28, 1, e1, e0);
            rol2!(v, 11);
            rounds!(w, v, abcd, 32, 1, e0, e1);
            rol2!(v, 12);
            rounds!(w, v, abcd, 36, 1, e1, e0);
            rol2!(v, 13);
            rounds!(w, v, abcd, 40, 2, e0, e1);
            rol2!(v, 14);
            rounds!(w, v, abcd, 44, 2, e1, e0);
            rol2!(v, 15);
            rounds!(w, v, abcd, 48, 2, e0, e1);
            rol2!(v, 16);
            rounds!(w, v, abcd, 52, 2, e1, e0);
            rol2!(v, 17);
            rounds!(w, v, abcd, 56, 2, e0, e1);
            rol2!(v, 18);
            rounds!(w, v, abcd, 60, 3, e1, e0 => at_60);
            rol2!(v, 19);
            rounds!(w, v, abcd, 64, 3, e0, e1 => at_64);
            rounds!(w, v, abcd, 68, 3, e1, e0);
            rounds!(w, v, abcd, 72, 3, e0, e1);
            rounds!(w, v, abcd, 76, 3, e1, e0);

            // Feed-forward.
            e0 = _mm_sha1nexte_epu32(e0, e_in);
            abcd = _mm_add_epi32(abcd, abcd_in);

            store_abcd(state, abcd);
            state[4] = _mm_extract_epi32(e0, 3) as u32;
        }
    };
}

compress_spill_fn!(
    /// Compresses one block, spilling the message schedule into `w`.
    ///
    /// Requires `sha`, `sse2`, `ssse3` and `sse4.1`, so a caller that cannot
    /// prove the features needs an `unsafe` block.
    compress_spill,
    "sha,sse2,ssse3,sse4.1"
);

compress_spill_fn!(
    /// [`compress_spill`] built with AVX too, whose three-operand forms keep
    /// the schedule's inputs without copying them first.
    compress_spill_avx,
    "sha,sse2,ssse3,sse4.1,avx"
);

/// Whether this candidate is the attack it is a candidate for.
///
/// The `x86` half of [`Backend::is_attack`], which says why it runs this way
/// round. The chaining value it starts from is worked out here rather than
/// handed in, so it never leaves the vector registers.
///
/// [`Backend::is_attack`]: crate::block::Backend::is_attack
#[inline]
#[target_feature(enable = "sha,sse2,ssse3,sse4.1")]
fn recompress(
    step: RecompressFrom,
    m1: &Schedule,
    dm: &[u32; 80],
    state: &[u32; 5],
    chaining_out: &[u32; 5],
) -> bool {
    let target = rounds::partner_start(step, m1, dm, state, chaining_out);

    let mut abcd = load_abcd(&target);
    let abcd_in = abcd;
    let e_in = _mm_set_epi32(target[4] as i32, 0, 0, 0);

    // The first group has no round behind it to carry the working word in,
    // so it adds the chaining value's fifth word itself.
    let mut e0 = _mm_add_epi32(e_in, words::<0>(m1, dm));
    let mut e1 = abcd;
    abcd = _mm_sha1rnds4_epu32(abcd, e0, 0);

    /// Four rounds. The working word arrives in `live` and the one the next
    /// four need is put aside in `held`, which is why two alternate.
    ///
    /// `compress_spill`'s group carries the schedule too; here there is none
    /// to carry, only words to read.
    macro_rules! group {
        ($g:expr, $k:expr, $live:ident, $held:ident) => {{
            let ready = words::<{ 4 * $g }>(m1, dm);
            $live = _mm_sha1nexte_epu32($live, ready);
            $held = abcd;
            abcd = _mm_sha1rnds4_epu32(abcd, $live, $k);
        }};
    }

    group!(1, 0, e1, e0);
    group!(2, 0, e0, e1);
    group!(3, 0, e1, e0);
    group!(4, 0, e0, e1);

    group!(5, 1, e1, e0);
    group!(6, 1, e0, e1);
    group!(7, 1, e1, e0);
    group!(8, 1, e0, e1);
    group!(9, 1, e1, e0);

    group!(10, 2, e0, e1);
    group!(11, 2, e1, e0);
    group!(12, 2, e0, e1);
    group!(13, 2, e1, e0);
    group!(14, 2, e0, e1);

    group!(15, 3, e1, e0);
    group!(16, 3, e0, e1);
    group!(17, 3, e1, e0);
    group!(18, 3, e0, e1);
    group!(19, 3, e1, e0);

    // Feed-forward.
    e0 = _mm_sha1nexte_epu32(e0, e_in);
    abcd = _mm_add_epi32(abcd, abcd_in);

    let mut ends_on = [0u32; 5];
    store_abcd(&mut ends_on, abcd);
    ends_on[4] = _mm_extract_epi32(e0, 3) as u32;

    crate::block::xor(&ends_on, chaining_out) == 0
}
