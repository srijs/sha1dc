//! Selects between the hardware and scalar SHA-1 compression implementations.
//!
//! The two are [`armv8`] and [`scalar`], one file each. Both leave the
//! expanded schedule behind, which the detection needs and an ordinary SHA-1
//! would not keep.
//!
//! [`Backend::scalar`] selects the scalar one. Tests use it to run that path
//! on a machine that has the instructions.

use crate::BLOCK_SIZE;
use crate::block::rounds;

#[cfg(target_arch = "aarch64")]
mod armv8;
mod scalar;

/// Whether this CPU has the SHA-1 instructions the hardware backend needs.
///
/// A `std` build asks the OS at run time. Run-time detection needs `cpuid` or
/// a system call, so a `no_std` build uses `target_feature` only. Such a build
/// needs the features on the command line, for example
/// `-C target-feature=+sha2`.
#[cfg(target_arch = "aarch64")]
fn has_sha1_instructions() -> bool {
    #[cfg(feature = "std")]
    {
        std::arch::is_aarch64_feature_detected!("sha2")
    }
    #[cfg(not(feature = "std"))]
    {
        cfg!(target_feature = "sha2")
    }
}

#[derive(Clone, Copy)]
enum Repr {
    #[cfg(target_arch = "aarch64")]
    Armv8,
    Scalar,
}

/// The implementation that computes the digest of a block. [`Backend::new`]
/// selects it once, from CPU feature detection.
#[derive(Clone, Copy)]
pub(crate) struct Backend(Repr);

impl Backend {
    pub(crate) fn new() -> Self {
        #[cfg(target_arch = "aarch64")]
        if has_sha1_instructions() {
            return Self(Repr::Armv8);
        }

        Self(Repr::Scalar)
    }

    /// The scalar implementation, on every CPU.
    pub(crate) fn scalar() -> Self {
        Self(Repr::Scalar)
    }

    /// Digests `block` into `state` and writes the input of the checker to
    /// `m1`.
    ///
    /// This can leave `state_58` and `state_65` unset. Call
    /// [`ensure_states`](Self::ensure_states) before you read them.
    pub(crate) fn compress_spill(
        &self,
        state: &mut [u32; 5],
        block: &[u8; BLOCK_SIZE],
        m1: &mut [u32; 80],
        state_58: &mut [u32; 5],
        state_65: &mut [u32; 5],
    ) {
        match self.0 {
            #[cfg(target_arch = "aarch64")]
            Repr::Armv8 => {
                // SAFETY: `Repr::Armv8` means the features were checked.
                unsafe { armv8::compress_spill(state, block, m1) };
            }
            // Only this arm decodes the block. The other byte-swaps in its
            // own loads, and the buffer stays here rather than moving
            // into `scalar` so that it is hoisted out of the loop over blocks
            // instead of being built again for each one. Moving it measures
            // 4% of the scalar backend.
            Repr::Scalar => {
                let mut block_u32 = [0u32; BLOCK_SIZE / 4];
                read_block(block, &mut block_u32);
                scalar::compress_spill(state, &block_u32, m1, state_58, state_65);
            }
        }
    }

    /// Updates `state_58` and `state_65` for this block.
    ///
    /// The hardware backends write out the message schedule but not the
    /// intermediate states. This recovers those states from the schedule and
    /// does not run the compression again. It runs only for a flagged block.
    pub(crate) fn ensure_states(
        &self,
        ihv_before: &[u32; 5],
        m1: &[u32; 80],
        state_58: &mut [u32; 5],
        state_65: &mut [u32; 5],
    ) {
        if matches!(self.0, Repr::Scalar) {
            return;
        }
        rounds::states_from_w(ihv_before, m1, state_58, state_65);
    }
}

/// Reads a block as big-endian `u32` words.
#[inline(always)]
pub(crate) fn read_block(block: &[u8; BLOCK_SIZE], out: &mut [u32; BLOCK_SIZE / 4]) {
    for (o, chunk) in out.iter_mut().zip(block.chunks_exact(4)) {
        *o = u32::from_be_bytes(chunk.try_into().unwrap());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    fn is_hardware(backend: &Backend) -> bool {
        match backend.0 {
            #[cfg(target_arch = "aarch64")]
            Repr::Armv8 => true,
            Repr::Scalar => false,
        }
    }

    /// Detects a silent fallback to scalar. If the compiler knows that the
    /// instructions are present, detection must agree.
    #[test]
    fn hardware_is_used_when_the_target_guarantees_it() {
        let guaranteed = cfg!(any(
            all(target_arch = "aarch64", target_feature = "sha2"),
            all(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "sha",
                target_feature = "sse2",
                target_feature = "ssse3",
                target_feature = "sse4.1"
            )
        ));
        if guaranteed {
            assert!(
                is_hardware(&Backend::new()),
                "cpu feature detection missed instructions the target guarantees"
            );
        }
    }

    #[test]
    fn hardware_agrees_with_scalar_compression() {
        let hardware = Backend::new();
        if !is_hardware(&hardware) {
            return;
        }

        let mut seed = 0x0BAD_C0DE_DEAD_BEEF;
        for _ in 0..2_000 {
            let block: [u8; BLOCK_SIZE] =
                core::array::from_fn(|_| (xorshift(&mut seed) >> 24) as u8);
            let ihv: [u32; 5] = core::array::from_fn(|_| xorshift(&mut seed) as u32);

            let mut block_u32 = [0u32; 16];
            read_block(&block, &mut block_u32);

            let (mut hw_state, mut hw_w) = (ihv, [0u32; 80]);
            let (mut hw_s58, mut hw_s65) = ([0u32; 5], [0u32; 5]);
            hardware.compress_spill(&mut hw_state, &block, &mut hw_w, &mut hw_s58, &mut hw_s65);

            let mut sc_state = ihv;
            let (mut sc_w, mut s58, mut s65) = ([0u32; 80], [0u32; 5], [0u32; 5]);
            Backend(Repr::Scalar).compress_spill(
                &mut sc_state,
                &block,
                &mut sc_w,
                &mut s58,
                &mut s65,
            );

            assert_eq!(hw_state, sc_state, "digest diverged");
            assert_eq!(hw_w, sc_w, "schedule diverged");

            // Independent of the scalar round macros, so a common fault
            // cannot hide.
            let mut want = [0u32; 80];
            want[..16].copy_from_slice(&block_u32);
            for t in 16..80 {
                want[t] = (want[t - 3] ^ want[t - 8] ^ want[t - 14] ^ want[t - 16]).rotate_left(1);
            }
            assert_eq!(hw_w, want, "schedule is not the standard expansion");
        }
    }

    /// `states_from_w` replaces a full scalar run on the hardware path, so it
    /// must give the same result.
    #[test]
    fn states_from_w_agrees_with_full_replay() {
        let mut seed = 0xC0FF_EE00_1234_5678;
        for _ in 0..2_000 {
            let m: [u32; 16] = core::array::from_fn(|_| xorshift(&mut seed) as u32);
            let ihv: [u32; 5] = core::array::from_fn(|_| xorshift(&mut seed) as u32);

            let mut replayed = ihv;
            let (mut w, mut want_58, mut want_65) = ([0u32; 80], [0u32; 5], [0u32; 5]);
            scalar::compress_spill(&mut replayed, &m, &mut w, &mut want_58, &mut want_65);

            let (mut got_58, mut got_65) = ([0u32; 5], [0u32; 5]);
            rounds::states_from_w(&ihv, &w, &mut got_58, &mut got_65);

            assert_eq!(got_58, want_58, "state_58 diverged");
            assert_eq!(got_65, want_65, "state_65 diverged");
        }
    }
}
