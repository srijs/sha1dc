//! The unconditional part of the UBC check.
//!
//! These checks run on every block and are most of the cost of detection.
//! They are the ones that pack into vector lanes: each family tests the same
//! bit pair at the same word gap over a continuous range of `i`, so one pair
//! of loads covers four checks or eight. `codegen/` chooses them, and the
//! rest go to [`tail`](super::tail).
//!
//! There is one form per instruction set. `codegen/` generates all of them
//! from one plan. Only this dispatcher is written by hand.
//!
//! A `cfg` selects the form. The SHA-1 backend in [`crate::block`] instead
//! asks the CPU at run time, because `sha` and `sha2` are optional extensions
//! that no target can promise. The target settles `sse2` and `neon`, so a
//! run-time test would cost a branch per block and can never fail.
//!
//! [`ubc_check`]: super::ubc_check

pub(super) mod scalar;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub(super) mod neon;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "sse2"
))]
pub(super) mod sse2;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "sse2",
    any(feature = "std", target_feature = "avx2")
))]
pub(super) mod avx2;

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

/// Runs the checks. Uses a vector form if the target has one.
#[inline(always)]
pub(super) fn mask(w: &[u32; 80]) -> u32 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: the cfg guarantees `neon`. All reads stay in `w`.
        unsafe { neon::mask(w) }
    }
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
    ))]
    {
        #[cfg(any(feature = "std", target_feature = "avx2"))]
        if has_avx2() {
            // SAFETY: just detected. All reads stay in `w`.
            return unsafe { avx2::mask(w) };
        }
        // SAFETY: the cfg guarantees `sse2`. All reads stay in `w`.
        unsafe { sse2::mask(w) }
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "sse2"
        )
    )))]
    {
        scalar::mask(w)
    }
}
