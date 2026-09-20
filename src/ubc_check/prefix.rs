//! The unconditional part of the UBC check.
//!
//! These checks run on every block and are most of the cost of detection.
//! They are the ones that pack into vector lanes: each family tests the same
//! bit pair at the same word gap over a continuous range of `i`, so one pair
//! of loads covers several checks. `codegen/` chooses them, and the rest go
//! to [`tail`](super::tail).
//!
//! `codegen/` generates the form below. Only this dispatcher is written by
//! hand.
//!
//! [`ubc_check`]: super::ubc_check

pub(super) mod scalar;

/// Runs the checks.
#[inline(always)]
pub(super) fn mask(w: &[u32; 80]) -> u32 {
    scalar::mask(w)
}
