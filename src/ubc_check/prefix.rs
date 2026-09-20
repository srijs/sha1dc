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
//! A `cfg` selects the form, because the target settles `neon` and a run-time
//! test would cost a branch per block and can never fail.
//!
//! [`ubc_check`]: super::ubc_check

pub(super) mod scalar;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub(super) mod neon;

/// Runs the checks. Uses a vector form if the target has one.
#[inline(always)]
pub(super) fn mask(w: &[u32; 80]) -> u32 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: the cfg guarantees `neon`. All reads stay in `w`.
        unsafe { neon::mask(w) }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        scalar::mask(w)
    }
}
