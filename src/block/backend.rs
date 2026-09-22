//! Selects between the hardware and scalar SHA-1 compression implementations.
//!
//! The three are [`sha_ni`], [`armv8`] and [`scalar`], one file each. All of
//! them leave the expanded schedule behind, which the detection needs and an
//! ordinary SHA-1 would not keep.
//!
//! [`Backend::scalar`] selects the scalar one. Tests use it to run that path
//! on a machine that has the instructions.

use crate::block::rounds;
use crate::{BLOCK_SIZE, Schedule};

#[cfg(target_arch = "aarch64")]
mod armv8;
mod scalar;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sha_ni;

/// Whether this CPU has the SHA-1 instructions the hardware backend needs.
///
/// A `std` build asks the OS at run time. Run-time detection needs `cpuid` or
/// a system call, so a `no_std` build uses `target_feature` only. Such a build
/// needs the features on the command line, for example
/// `-C target-feature=+sha2`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
fn has_sha1_instructions() -> bool {
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        std::arch::is_x86_feature_detected!("sha")
            && std::arch::is_x86_feature_detected!("sse2")
            && std::arch::is_x86_feature_detected!("ssse3")
            && std::arch::is_x86_feature_detected!("sse4.1")
    }
    #[cfg(all(feature = "std", target_arch = "aarch64"))]
    {
        std::arch::is_aarch64_feature_detected!("sha2")
    }
    #[cfg(all(not(feature = "std"), any(target_arch = "x86", target_arch = "x86_64")))]
    {
        cfg!(all(
            target_feature = "sha",
            target_feature = "sse2",
            target_feature = "ssse3",
            target_feature = "sse4.1"
        ))
    }
    #[cfg(all(not(feature = "std"), target_arch = "aarch64"))]
    {
        cfg!(target_feature = "sha2")
    }
}

#[derive(Clone, Copy)]
enum Repr {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    ShaNi,
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
        #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
        if has_sha1_instructions() {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            return Self(Repr::ShaNi);
            #[cfg(target_arch = "aarch64")]
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
        m1: &mut Schedule,
        state_58: &mut [u32; 5],
        state_65: &mut [u32; 5],
    ) {
        match self.0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Repr::ShaNi => {
                // SAFETY: `Repr::ShaNi` means the features were checked.
                unsafe { sha_ni::compress_spill(state, block, m1) };
            }
            #[cfg(target_arch = "aarch64")]
            Repr::Armv8 => {
                // SAFETY: `Repr::Armv8` means the features were checked.
                unsafe { armv8::compress_spill(state, block, m1) };
            }
            // Only this arm decodes the block. The other two byte-swap in
            // their own loads, and the buffer stays here rather than moving
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

    /// Updates the stored states this block's candidates will recompress
    /// from.
    ///
    /// The hardware backends write out the message schedule but not the
    /// intermediate states. This recovers them from the schedule and the two
    /// chaining values, running backwards from the end, and does not run the
    /// compression again. It runs only for a flagged block. `need_58` says
    /// whether any candidate recompresses from step 58.
    pub(crate) fn ensure_states(
        &self,
        ihv_before: &[u32; 5],
        ihv_after: &[u32; 5],
        m1: &Schedule,
        need_58: bool,
        state_58: &mut [u32; 5],
        state_65: &mut [u32; 5],
    ) {
        if matches!(self.0, Repr::Scalar) {
            return;
        }
        rounds::states_back_from_h(ihv_before, ihv_after, m1, need_58, state_58, state_65);
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
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Repr::ShaNi => true,
            #[cfg(target_arch = "aarch64")]
            Repr::Armv8 => true,
            Repr::Scalar => false,
        }
    }

    #[cfg(feature = "std")]
    fn name(backend: &Backend) -> &'static str {
        match backend.0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Repr::ShaNi => "sha-ni",
            #[cfg(target_arch = "aarch64")]
            Repr::Armv8 => "armv8",
            Repr::Scalar => "scalar",
        }
    }

    /// `SHA1DC_EXPECT_BLOCK` lets a job say which implementation it is there
    /// to cover. A runner without the SHA-1 instructions falls back to the
    /// scalar one and looks no different, so without this `sha_ni` can go a
    /// whole release without running.
    #[cfg(feature = "std")]
    #[test]
    fn the_expected_implementation_was_selected() {
        let expected = std::env::var("SHA1DC_EXPECT_BLOCK").unwrap_or_default();
        let expected = expected.trim();
        if expected.is_empty() {
            return; // a job that does not pin one leaves it empty
        }
        assert_eq!(
            name(&Backend::new()),
            expected,
            "this job was meant to exercise a different implementation of `block`"
        );
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

            let (mut hw_state, mut hw_w) = (ihv, Schedule::zeroed());
            let (mut hw_s58, mut hw_s65) = ([0u32; 5], [0u32; 5]);
            hardware.compress_spill(&mut hw_state, &block, &mut hw_w, &mut hw_s58, &mut hw_s65);

            let mut sc_state = ihv;
            let (mut sc_w, mut s58, mut s65) = (Schedule::zeroed(), [0u32; 5], [0u32; 5]);
            Backend(Repr::Scalar).compress_spill(
                &mut sc_state,
                &block,
                &mut sc_w,
                &mut s58,
                &mut s65,
            );

            assert_eq!(hw_state, sc_state, "digest diverged");
            assert_eq!(hw_w.words(), sc_w.words(), "schedule diverged");

            // Independent of the scalar round macros, so a common fault
            // cannot hide.
            let mut want = Schedule::zeroed();
            for (t, word) in block_u32.iter().enumerate() {
                want[t] = *word;
            }
            for t in 16..80 {
                want[t] = (want[t - 3] ^ want[t - 8] ^ want[t - 14] ^ want[t - 16]).rotate_left(1);
            }
            assert_eq!(
                hw_w.words(),
                want.words(),
                "schedule is not the standard expansion"
            );
        }
    }

    /// Property tests. The loops above run a fixed stream; these look for a
    /// disagreement anywhere and shrink a failure to a small case.
    ///
    /// `quickcheck` needs `std`, so a `no_std` build skips them.
    #[cfg(feature = "std")]
    mod properties {
        use super::*;
        use quickcheck::QuickCheck;

        /// The hardware backend and the scalar one must produce the same
        /// digest state and the same schedule for any block and any starting
        /// state, not only for a real message.
        #[test]
        fn hardware_agrees_with_scalar_on_any_block() {
            if !is_hardware(&Backend::new()) {
                return;
            }

            fn prop(block: [u8; BLOCK_SIZE], ihv: [u32; 5]) -> bool {
                let (mut hw_state, mut hw_w) = (ihv, Schedule::zeroed());
                let (mut hw_s58, mut hw_s65) = ([0u32; 5], [0u32; 5]);
                Backend::new().compress_spill(
                    &mut hw_state,
                    &block,
                    &mut hw_w,
                    &mut hw_s58,
                    &mut hw_s65,
                );

                let (mut sc_state, mut sc_w) = (ihv, Schedule::zeroed());
                let (mut s58, mut s65) = ([0u32; 5], [0u32; 5]);
                Backend(Repr::Scalar).compress_spill(
                    &mut sc_state,
                    &block,
                    &mut sc_w,
                    &mut s58,
                    &mut s65,
                );

                hw_state == sc_state && hw_w.words() == sc_w.words()
            }

            QuickCheck::new()
                .tests(1_000)
                .quickcheck(prop as fn([u8; BLOCK_SIZE], [u32; 5]) -> bool);
        }

        /// Both state recoveries must match a full scalar run for any
        /// schedule the hardware path can leave behind.
        #[test]
        fn state_recovery_agrees_on_any_block() {
            fn prop(m: [u32; 16], ihv: [u32; 5]) -> bool {
                let mut replayed = ihv;
                let (mut w, mut want_58, mut want_65) = (Schedule::zeroed(), [0u32; 5], [0u32; 5]);
                scalar::compress_spill(&mut replayed, &m, &mut w, &mut want_58, &mut want_65);

                let (mut got_58, mut got_65) = ([0u32; 5], [0u32; 5]);
                rounds::states_from_w(&ihv, &w, &mut got_58, &mut got_65);

                let (mut back_58, mut back_65) = ([0u32; 5], [0u32; 5]);
                rounds::states_back_from_h(&ihv, &replayed, &w, true, &mut back_58, &mut back_65);

                // Asking for step 65 alone must leave step 58 untouched and
                // still give step 65.
                let (mut skipped_58, mut only_65) = ([0xDEAD_BEEFu32; 5], [0u32; 5]);
                rounds::states_back_from_h(
                    &ihv,
                    &replayed,
                    &w,
                    false,
                    &mut skipped_58,
                    &mut only_65,
                );

                got_58 == want_58
                    && got_65 == want_65
                    && back_58 == want_58
                    && back_65 == want_65
                    && only_65 == want_65
                    && skipped_58 == [0xDEAD_BEEF; 5]
            }

            QuickCheck::new()
                .tests(1_000)
                .quickcheck(prop as fn([u32; 16], [u32; 5]) -> bool);
        }
    }

    /// State recovery replaces a full scalar run on the hardware path, so
    /// both forms must give the same result.
    #[test]
    fn state_recovery_agrees_with_full_replay() {
        let mut seed = 0xC0FF_EE00_1234_5678;
        for _ in 0..2_000 {
            let m: [u32; 16] = core::array::from_fn(|_| xorshift(&mut seed) as u32);
            let ihv: [u32; 5] = core::array::from_fn(|_| xorshift(&mut seed) as u32);

            let mut replayed = ihv;
            let (mut w, mut want_58, mut want_65) = (Schedule::zeroed(), [0u32; 5], [0u32; 5]);
            scalar::compress_spill(&mut replayed, &m, &mut w, &mut want_58, &mut want_65);

            let (mut got_58, mut got_65) = ([0u32; 5], [0u32; 5]);
            rounds::states_from_w(&ihv, &w, &mut got_58, &mut got_65);

            assert_eq!(got_58, want_58, "state_58 diverged");
            assert_eq!(got_65, want_65, "state_65 diverged");

            let (mut back_58, mut back_65) = ([0u32; 5], [0u32; 5]);
            rounds::states_back_from_h(&ihv, &replayed, &w, true, &mut back_58, &mut back_65);

            assert_eq!(back_58, want_58, "state_58 diverged running backwards");
            assert_eq!(back_65, want_65, "state_65 diverged running backwards");
        }
    }
}
