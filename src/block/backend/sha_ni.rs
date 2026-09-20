//! SHA-1 `x86`/`x86_64` backend built on the SHA-NI instructions.
//!
//! Computes a block's digest and spills the expanded message schedule, which
//! the collision detection needs. The spill is the difference from an
//! ordinary SHA-1: `sha1msg1`/`sha1msg2` keep the schedule in registers, and
//! the check needs all 80 words in memory.
//!
//! `sha1rnds4` does four rounds at a time, so the work is in groups of four.
//! Each group consumes the four schedule words that are ready, finishes the
//! four that the next group needs, and takes the first two of the three steps
//! that will finish the group after that. `sha1nexte` carries the fifth
//! working word between groups, which is why two of them alternate.

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
compile_error!("the sha_ni backend needs an x86 or x86_64 target");

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// SHA-NI holds the four words of a group in the opposite order to the
/// schedule, so every crossing between the two reverses them.
const REVERSE: i32 = 0b00_01_10_11;

/// Writes the four words a group is about to use into the schedule.
macro_rules! spill {
    ($w:expr, $t:expr, $msg:expr) => {
        _mm_storeu_si128($w.add($t).cast(), _mm_shuffle_epi32($msg, REVERSE))
    };
}

/// One group of four rounds, once the schedule is under way.
///
/// `ready` holds the words these rounds use. `next` is finished here, and the
/// two groups behind it take the first and second of their three steps. The
/// working word arrives in `live` and the one for the next group is put aside
/// in `held`.
macro_rules! group {
    (
        $w:expr, $t:expr, $k:expr, $abcd:ident, $live:ident, $held:ident,
        $ready:ident, $next:ident, $second:ident, $first:ident
    ) => {{
        spill!($w, $t, $ready);
        $live = _mm_sha1nexte_epu32($live, $ready);
        $held = $abcd;
        $next = _mm_sha1msg2_epu32($next, $ready);
        $abcd = _mm_sha1rnds4_epu32($abcd, $live, $k);
        $first = _mm_sha1msg1_epu32($first, $ready);
        $second = _mm_xor_si128($second, $ready);
    }};
}

/// Compresses one block, spilling the message schedule into `w`.
#[target_feature(enable = "sha,sse2,ssse3,sse4.1")]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(clippy::too_many_lines)]
pub(crate) unsafe fn compress_spill(state: &mut [u32; 5], block: &[u8; 64], w: &mut [u32; 80]) {
    // Turns the big-endian message into native order, reversing the four
    // words along with the bytes.
    let swap = _mm_set_epi64x(0x0001_0203_0405_0607, 0x0809_0A0B_0C0D_0E0F);

    let wp = w.as_mut_ptr();
    let data: *const __m128i = block.as_ptr().cast();

    let mut abcd = _mm_shuffle_epi32(_mm_loadu_si128(state.as_ptr().cast()), REVERSE);
    let abcd_in = abcd;
    let e_in = _mm_set_epi32(state[4] as i32, 0, 0, 0);

    let mut msg0 = _mm_shuffle_epi8(_mm_loadu_si128(data.add(0)), swap);
    let mut msg1 = _mm_shuffle_epi8(_mm_loadu_si128(data.add(1)), swap);
    let mut msg2 = _mm_shuffle_epi8(_mm_loadu_si128(data.add(2)), swap);
    let mut msg3 = _mm_shuffle_epi8(_mm_loadu_si128(data.add(3)), swap);

    // The first sixteen words are the block itself. The expansion starts as
    // soon as enough of them are in, so these four groups build up to the
    // steady shape of `group!`.
    spill!(wp, 0, msg0);
    let mut e0 = _mm_add_epi32(e_in, msg0);
    let mut e1 = abcd;
    abcd = _mm_sha1rnds4_epu32(abcd, e0, 0);

    spill!(wp, 4, msg1);
    e1 = _mm_sha1nexte_epu32(e1, msg1);
    e0 = abcd;
    abcd = _mm_sha1rnds4_epu32(abcd, e1, 0);
    msg0 = _mm_sha1msg1_epu32(msg0, msg1);

    spill!(wp, 8, msg2);
    e0 = _mm_sha1nexte_epu32(e0, msg2);
    e1 = abcd;
    abcd = _mm_sha1rnds4_epu32(abcd, e0, 0);
    msg1 = _mm_sha1msg1_epu32(msg1, msg2);
    msg0 = _mm_xor_si128(msg0, msg2);

    spill!(wp, 12, msg3);
    e1 = _mm_sha1nexte_epu32(e1, msg3);
    e0 = abcd;
    msg0 = _mm_sha1msg2_epu32(msg0, msg3);
    abcd = _mm_sha1rnds4_epu32(abcd, e1, 0);
    msg2 = _mm_sha1msg1_epu32(msg2, msg3);
    msg1 = _mm_xor_si128(msg1, msg3);

    // Steady state. The four message registers take turns, and so do the two
    // working words.
    group!(wp, 16, 0, abcd, e0, e1, msg0, msg1, msg2, msg3);
    group!(wp, 20, 1, abcd, e1, e0, msg1, msg2, msg3, msg0);
    group!(wp, 24, 1, abcd, e0, e1, msg2, msg3, msg0, msg1);
    group!(wp, 28, 1, abcd, e1, e0, msg3, msg0, msg1, msg2);
    group!(wp, 32, 1, abcd, e0, e1, msg0, msg1, msg2, msg3);
    group!(wp, 36, 1, abcd, e1, e0, msg1, msg2, msg3, msg0);
    group!(wp, 40, 2, abcd, e0, e1, msg2, msg3, msg0, msg1);
    group!(wp, 44, 2, abcd, e1, e0, msg3, msg0, msg1, msg2);
    group!(wp, 48, 2, abcd, e0, e1, msg0, msg1, msg2, msg3);
    group!(wp, 52, 2, abcd, e1, e0, msg1, msg2, msg3, msg0);
    group!(wp, 56, 2, abcd, e0, e1, msg2, msg3, msg0, msg1);
    group!(wp, 60, 3, abcd, e1, e0, msg3, msg0, msg1, msg2);
    group!(wp, 64, 3, abcd, e0, e1, msg0, msg1, msg2, msg3);

    // The last words are already in hand, so the expansion stops one step at
    // a time.
    spill!(wp, 68, msg1);
    e1 = _mm_sha1nexte_epu32(e1, msg1);
    e0 = abcd;
    msg2 = _mm_sha1msg2_epu32(msg2, msg1);
    abcd = _mm_sha1rnds4_epu32(abcd, e1, 3);
    msg3 = _mm_xor_si128(msg3, msg1);

    spill!(wp, 72, msg2);
    e0 = _mm_sha1nexte_epu32(e0, msg2);
    e1 = abcd;
    msg3 = _mm_sha1msg2_epu32(msg3, msg2);
    abcd = _mm_sha1rnds4_epu32(abcd, e0, 3);

    spill!(wp, 76, msg3);
    e1 = _mm_sha1nexte_epu32(e1, msg3);
    e0 = abcd;
    abcd = _mm_sha1rnds4_epu32(abcd, e1, 3);

    // Feed-forward.
    e0 = _mm_sha1nexte_epu32(e0, e_in);
    abcd = _mm_add_epi32(abcd, abcd_in);

    _mm_storeu_si128(state.as_mut_ptr().cast(), _mm_shuffle_epi32(abcd, REVERSE));
    state[4] = _mm_extract_epi32(e0, 3) as u32;
}
