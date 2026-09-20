//! The unconditional part of the UBC check.
//!
//! These checks run on every block and are most of the cost of detection.
//! They are the ones that pack into vector lanes: each family tests the same
//! bit pair at the same word gap over a continuous range of `i`, so one pair
//! of loads covers four checks. `codegen/` chooses them, and the rest go to
//! [`tail`](super::tail).
//!
//! There is one form per instruction set, and a scalar one for the rest.
//! `codegen/` generates all of them from one plan. Only this dispatcher is
//! written by hand.
//!
//! A `cfg` selects the form, because the target settles `neon` and `sse2` and
//! a run-time test would cost a branch per block and can never fail.
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
