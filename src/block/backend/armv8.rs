//! SHA-1 `aarch64` backend built on the ARMv8 SHA-1 instructions.
//!
//! Computes a block's digest and spills the expanded message schedule, which
//! the collision detection needs. The spill is the difference from an
//! ordinary SHA-1: `vsha1su0q`/`vsha1su1q` keep the schedule in registers,
//! and the check needs all 80 words in memory.
//!
//! `vsha1cq`, `vsha1pq` and `vsha1mq` each do four rounds, so the work is in
//! groups of four. `vsha1h` carries the fifth working word between them, and
//! `vsha1su0q`/`vsha1su1q` take the schedule two steps at a time, far enough
//! ahead that the rounds do not wait for it.

#[cfg(not(target_arch = "aarch64"))]
compile_error!("the armv8 backend needs an aarch64 target");

use core::arch::aarch64::*;

use crate::block::rounds::K;

/// The four schedule words starting at `I`.
///
/// `I` is a const parameter, so a spill past the end of the schedule is a
/// compile error at the call site rather than a promise in a comment.
#[inline]
#[target_feature(enable = "sha2")]
fn spill<const I: usize>(w: &mut [u32; 80], v: uint32x4_t) {
    const { assert!(I + 4 <= 80, "the spill runs past the schedule") }
    // SAFETY: the const assert above proves `w[I..I + 4]` is in bounds,
    // which is the whole of what this writes.
    unsafe { vst1q_u32(w.as_mut_ptr().add(I), v) }
}

/// The four message words starting at byte `I`, in native order.
#[inline]
#[target_feature(enable = "sha2")]
fn msg<const I: usize>(block: &[u8; 64]) -> uint32x4_t {
    const { assert!(I + 16 <= 64, "the load runs past the block") }
    // SAFETY: the const assert above proves `block[I..I + 16]` is in bounds,
    // which is the whole of what this reads.
    let bytes = unsafe { vld1q_u8(block.as_ptr().add(I)) };
    vreinterpretq_u32_u8(vrev32q_u8(bytes))
}

/// The `abcd` half of the state. The fifth word is carried on its own.
#[inline]
#[target_feature(enable = "sha2")]
fn load_abcd(state: &[u32; 5]) -> uint32x4_t {
    // SAFETY: `vld1q_u32` reads four words, and `state` has five.
    unsafe { vld1q_u32(state.as_ptr()) }
}

/// Writes `abcd` back, leaving the fifth word alone.
#[inline]
#[target_feature(enable = "sha2")]
fn store_abcd(state: &mut [u32; 5], v: uint32x4_t) {
    // SAFETY: `vst1q_u32` writes four words, and `state` has five.
    unsafe { vst1q_u32(state.as_mut_ptr(), v) }
}

/// Compresses one block, spilling the message schedule into `w`.
///
/// Requires `sha2`, so a caller that cannot prove the feature needs an
/// `unsafe` block.
#[target_feature(enable = "sha2")]
#[expect(
    clippy::too_many_lines,
    reason = "the block compression is one unrolled body"
)]
pub(crate) fn compress_spill(state: &mut [u32; 5], block: &[u8; 64], w: &mut [u32; 80]) {
    let mut abcd = load_abcd(state);
    let mut e0 = state[4];
    let abcd_in = abcd;
    let e_in = e0;

    let k0 = vdupq_n_u32(K[0]);
    let k1 = vdupq_n_u32(K[1]);
    let k2 = vdupq_n_u32(K[2]);
    let k3 = vdupq_n_u32(K[3]);

    // The block, big-endian, one group of four words at a time.
    let mut msg0 = msg::<0>(block);
    let mut msg1 = msg::<16>(block);
    let mut msg2 = msg::<32>(block);
    let mut msg3 = msg::<48>(block);

    let mut e1;
    let mut tmp0;
    let mut tmp1;

    spill::<0>(w, msg0);
    spill::<4>(w, msg1);
    spill::<8>(w, msg2);
    spill::<12>(w, msg3);

    tmp0 = vaddq_u32(msg0, k0);
    tmp1 = vaddq_u32(msg1, k0);

    // Rounds 0-3
    e1 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1cq_u32(abcd, e0, tmp0);
    tmp0 = vaddq_u32(msg2, k0);
    msg0 = vsha1su0q_u32(msg0, msg1, msg2);

    // Rounds 4-7
    e0 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1cq_u32(abcd, e1, tmp1);
    tmp1 = vaddq_u32(msg3, k0);
    msg0 = vsha1su1q_u32(msg0, msg3);
    spill::<16>(w, msg0);
    msg1 = vsha1su0q_u32(msg1, msg2, msg3);

    // Rounds 8-11
    e1 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1cq_u32(abcd, e0, tmp0);
    tmp0 = vaddq_u32(msg0, k0);
    msg1 = vsha1su1q_u32(msg1, msg0);
    spill::<20>(w, msg1);
    msg2 = vsha1su0q_u32(msg2, msg3, msg0);

    // Rounds 12-15
    e0 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1cq_u32(abcd, e1, tmp1);
    tmp1 = vaddq_u32(msg1, k1);
    msg2 = vsha1su1q_u32(msg2, msg1);
    spill::<24>(w, msg2);
    msg3 = vsha1su0q_u32(msg3, msg0, msg1);

    // Rounds 16-19
    e1 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1cq_u32(abcd, e0, tmp0);
    tmp0 = vaddq_u32(msg2, k1);
    msg3 = vsha1su1q_u32(msg3, msg2);
    spill::<28>(w, msg3);
    msg0 = vsha1su0q_u32(msg0, msg1, msg2);

    // Rounds 20-23
    e0 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1pq_u32(abcd, e1, tmp1);
    tmp1 = vaddq_u32(msg3, k1);
    msg0 = vsha1su1q_u32(msg0, msg3);
    spill::<32>(w, msg0);
    msg1 = vsha1su0q_u32(msg1, msg2, msg3);

    // Rounds 24-27
    e1 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1pq_u32(abcd, e0, tmp0);
    tmp0 = vaddq_u32(msg0, k1);
    msg1 = vsha1su1q_u32(msg1, msg0);
    spill::<36>(w, msg1);
    msg2 = vsha1su0q_u32(msg2, msg3, msg0);

    // Rounds 28-31
    e0 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1pq_u32(abcd, e1, tmp1);
    tmp1 = vaddq_u32(msg1, k1);
    msg2 = vsha1su1q_u32(msg2, msg1);
    spill::<40>(w, msg2);
    msg3 = vsha1su0q_u32(msg3, msg0, msg1);

    // Rounds 32-35
    e1 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1pq_u32(abcd, e0, tmp0);
    tmp0 = vaddq_u32(msg2, k2);
    msg3 = vsha1su1q_u32(msg3, msg2);
    spill::<44>(w, msg3);
    msg0 = vsha1su0q_u32(msg0, msg1, msg2);

    // Rounds 36-39
    e0 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1pq_u32(abcd, e1, tmp1);
    tmp1 = vaddq_u32(msg3, k2);
    msg0 = vsha1su1q_u32(msg0, msg3);
    spill::<48>(w, msg0);
    msg1 = vsha1su0q_u32(msg1, msg2, msg3);

    // Rounds 40-43
    e1 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1mq_u32(abcd, e0, tmp0);
    tmp0 = vaddq_u32(msg0, k2);
    msg1 = vsha1su1q_u32(msg1, msg0);
    spill::<52>(w, msg1);
    msg2 = vsha1su0q_u32(msg2, msg3, msg0);

    // Rounds 44-47
    e0 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1mq_u32(abcd, e1, tmp1);
    tmp1 = vaddq_u32(msg1, k2);
    msg2 = vsha1su1q_u32(msg2, msg1);
    spill::<56>(w, msg2);
    msg3 = vsha1su0q_u32(msg3, msg0, msg1);

    // Rounds 48-51
    e1 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1mq_u32(abcd, e0, tmp0);
    tmp0 = vaddq_u32(msg2, k2);
    msg3 = vsha1su1q_u32(msg3, msg2);
    spill::<60>(w, msg3);
    msg0 = vsha1su0q_u32(msg0, msg1, msg2);

    // Rounds 52-55
    e0 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1mq_u32(abcd, e1, tmp1);
    tmp1 = vaddq_u32(msg3, k3);
    msg0 = vsha1su1q_u32(msg0, msg3);
    spill::<64>(w, msg0);
    msg1 = vsha1su0q_u32(msg1, msg2, msg3);

    // Rounds 56-59
    e1 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1mq_u32(abcd, e0, tmp0);
    tmp0 = vaddq_u32(msg0, k3);
    msg1 = vsha1su1q_u32(msg1, msg0);
    spill::<68>(w, msg1);
    msg2 = vsha1su0q_u32(msg2, msg3, msg0);

    // Rounds 60-63
    e0 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1pq_u32(abcd, e1, tmp1);
    tmp1 = vaddq_u32(msg1, k3);
    msg2 = vsha1su1q_u32(msg2, msg1);
    spill::<72>(w, msg2);
    msg3 = vsha1su0q_u32(msg3, msg0, msg1);

    // Rounds 64-67
    e1 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1pq_u32(abcd, e0, tmp0);
    tmp0 = vaddq_u32(msg2, k3);
    msg3 = vsha1su1q_u32(msg3, msg2);
    spill::<76>(w, msg3);

    // Rounds 68-71
    e0 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1pq_u32(abcd, e1, tmp1);
    tmp1 = vaddq_u32(msg3, k3);

    // Rounds 72-75
    e1 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1pq_u32(abcd, e0, tmp0);

    // Rounds 76-79
    e0 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1pq_u32(abcd, e1, tmp1);

    abcd = vaddq_u32(abcd_in, abcd);
    e0 = e0.wrapping_add(e_in);

    store_abcd(state, abcd);
    state[4] = e0;
}
