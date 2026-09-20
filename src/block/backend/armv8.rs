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

/// Compresses one block, spilling the message schedule into `w`.
#[target_feature(enable = "sha2")]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(clippy::too_many_lines)]
pub(crate) unsafe fn compress_spill(state: &mut [u32; 5], block: &[u8; 64], w: &mut [u32; 80]) {
    let mut abcd = vld1q_u32(state.as_ptr());
    let mut e0 = state[4];
    let abcd_in = abcd;
    let e_in = e0;

    let k0 = vdupq_n_u32(K[0]);
    let k1 = vdupq_n_u32(K[1]);
    let k2 = vdupq_n_u32(K[2]);
    let k3 = vdupq_n_u32(K[3]);

    let wp = w.as_mut_ptr();
    let data = block.as_ptr();

    // The block, big-endian, one group of four words at a time.
    let mut msg0 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(data)));
    let mut msg1 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(data.add(16))));
    let mut msg2 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(data.add(32))));
    let mut msg3 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(data.add(48))));

    let mut e1;
    let mut tmp0;
    let mut tmp1;

    vst1q_u32(wp, msg0);
    vst1q_u32(wp.add(4), msg1);
    vst1q_u32(wp.add(8), msg2);
    vst1q_u32(wp.add(12), msg3);

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
    vst1q_u32(wp.add(16), msg0);
    msg1 = vsha1su0q_u32(msg1, msg2, msg3);

    // Rounds 8-11
    e1 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1cq_u32(abcd, e0, tmp0);
    tmp0 = vaddq_u32(msg0, k0);
    msg1 = vsha1su1q_u32(msg1, msg0);
    vst1q_u32(wp.add(20), msg1);
    msg2 = vsha1su0q_u32(msg2, msg3, msg0);

    // Rounds 12-15
    e0 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1cq_u32(abcd, e1, tmp1);
    tmp1 = vaddq_u32(msg1, k1);
    msg2 = vsha1su1q_u32(msg2, msg1);
    vst1q_u32(wp.add(24), msg2);
    msg3 = vsha1su0q_u32(msg3, msg0, msg1);

    // Rounds 16-19
    e1 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1cq_u32(abcd, e0, tmp0);
    tmp0 = vaddq_u32(msg2, k1);
    msg3 = vsha1su1q_u32(msg3, msg2);
    vst1q_u32(wp.add(28), msg3);
    msg0 = vsha1su0q_u32(msg0, msg1, msg2);

    // Rounds 20-23
    e0 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1pq_u32(abcd, e1, tmp1);
    tmp1 = vaddq_u32(msg3, k1);
    msg0 = vsha1su1q_u32(msg0, msg3);
    vst1q_u32(wp.add(32), msg0);
    msg1 = vsha1su0q_u32(msg1, msg2, msg3);

    // Rounds 24-27
    e1 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1pq_u32(abcd, e0, tmp0);
    tmp0 = vaddq_u32(msg0, k1);
    msg1 = vsha1su1q_u32(msg1, msg0);
    vst1q_u32(wp.add(36), msg1);
    msg2 = vsha1su0q_u32(msg2, msg3, msg0);

    // Rounds 28-31
    e0 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1pq_u32(abcd, e1, tmp1);
    tmp1 = vaddq_u32(msg1, k1);
    msg2 = vsha1su1q_u32(msg2, msg1);
    vst1q_u32(wp.add(40), msg2);
    msg3 = vsha1su0q_u32(msg3, msg0, msg1);

    // Rounds 32-35
    e1 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1pq_u32(abcd, e0, tmp0);
    tmp0 = vaddq_u32(msg2, k2);
    msg3 = vsha1su1q_u32(msg3, msg2);
    vst1q_u32(wp.add(44), msg3);
    msg0 = vsha1su0q_u32(msg0, msg1, msg2);

    // Rounds 36-39
    e0 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1pq_u32(abcd, e1, tmp1);
    tmp1 = vaddq_u32(msg3, k2);
    msg0 = vsha1su1q_u32(msg0, msg3);
    vst1q_u32(wp.add(48), msg0);
    msg1 = vsha1su0q_u32(msg1, msg2, msg3);

    // Rounds 40-43
    e1 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1mq_u32(abcd, e0, tmp0);
    tmp0 = vaddq_u32(msg0, k2);
    msg1 = vsha1su1q_u32(msg1, msg0);
    vst1q_u32(wp.add(52), msg1);
    msg2 = vsha1su0q_u32(msg2, msg3, msg0);

    // Rounds 44-47
    e0 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1mq_u32(abcd, e1, tmp1);
    tmp1 = vaddq_u32(msg1, k2);
    msg2 = vsha1su1q_u32(msg2, msg1);
    vst1q_u32(wp.add(56), msg2);
    msg3 = vsha1su0q_u32(msg3, msg0, msg1);

    // Rounds 48-51
    e1 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1mq_u32(abcd, e0, tmp0);
    tmp0 = vaddq_u32(msg2, k2);
    msg3 = vsha1su1q_u32(msg3, msg2);
    vst1q_u32(wp.add(60), msg3);
    msg0 = vsha1su0q_u32(msg0, msg1, msg2);

    // Rounds 52-55
    e0 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1mq_u32(abcd, e1, tmp1);
    tmp1 = vaddq_u32(msg3, k3);
    msg0 = vsha1su1q_u32(msg0, msg3);
    vst1q_u32(wp.add(64), msg0);
    msg1 = vsha1su0q_u32(msg1, msg2, msg3);

    // Rounds 56-59
    e1 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1mq_u32(abcd, e0, tmp0);
    tmp0 = vaddq_u32(msg0, k3);
    msg1 = vsha1su1q_u32(msg1, msg0);
    vst1q_u32(wp.add(68), msg1);
    msg2 = vsha1su0q_u32(msg2, msg3, msg0);

    // Rounds 60-63
    e0 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1pq_u32(abcd, e1, tmp1);
    tmp1 = vaddq_u32(msg1, k3);
    msg2 = vsha1su1q_u32(msg2, msg1);
    vst1q_u32(wp.add(72), msg2);
    msg3 = vsha1su0q_u32(msg3, msg0, msg1);

    // Rounds 64-67
    e1 = vsha1h_u32(vgetq_lane_u32(abcd, 0));
    abcd = vsha1pq_u32(abcd, e0, tmp0);
    tmp0 = vaddq_u32(msg2, k3);
    msg3 = vsha1su1q_u32(msg3, msg2);
    vst1q_u32(wp.add(76), msg3);

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

    vst1q_u32(state.as_mut_ptr(), abcd);
    state[4] = e0;
}
