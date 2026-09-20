//! Block compression, with collision detection wrapped around it.
//!
//! This is Algorithm 1 of the [paper]. Hashing a block is the ordinary part.
//! The detection asks, for every attack the block could belong to, what the
//! other message of the colliding pair would have to be, and whether that
//! message would reach the same chaining value. If one does, the block is
//! part of an attack.
//!
//! Testing all 32 attacks on every block would cost 32 further compressions.
//! [`crate::ubc_check`] rules out nearly all of them first, so the work below
//! runs for about one block in twenty, and usually for one attack rather than
//! all of them.
//!
//! [paper]: https://marc-stevens.nl/research/papers/C13-S.pdf

use crate::{BLOCK_SIZE, Inner, ubc_check::RecompressFrom};

mod backend;
mod rounds;

pub(crate) use backend::Backend;
use rounds::{compression_w, recompression_step};

/// Compresses `blocks` into the hasher's chaining value, testing each one for
/// a collision attack.
#[inline]
pub(crate) fn compress(ctx: &mut Inner, blocks: &[[u8; BLOCK_SIZE]]) {
    let backend = ctx.backend;

    for block in blocks {
        ctx.ihv1 = ctx.h;

        let Inner {
            h,
            m1,
            state_58,
            state_65,
            ..
        } = ctx;
        backend.compress_spill(h, block, m1, state_58, state_65);

        // Without the filter every attack is a candidate, which is what the
        // `no ubc check` benchmark measures.
        let candidates = if ctx.ubc_check {
            crate::ubc_check::ubc_check(&ctx.m1)
        } else {
            !0
        };

        if candidates != 0 && attacked(backend, ctx.h, ctx, candidates) {
            ctx.found_collision = true;

            // Mitigation. Two more compressions of this block give a digest
            // that the other message of the pair does not share, at the cost
            // of one that no other SHA-1 implementation agrees with.
            if ctx.safe_hash {
                let Inner { h, m1, .. } = ctx;
                compression_w(h, m1);
                compression_w(h, m1);
            }
        }
    }
}

/// Zero when two chaining values are equal.
#[inline(always)]
fn xor(a: &[u32; 5], b: &[u32; 5]) -> u32 {
    a.iter().zip(b).fold(0, |differs, (x, y)| differs | (x ^ y))
}

/// Whether any candidate attack really describes this block.
///
/// For each one: the partner block differs from this one by a fixed
/// difference, so its state partway through is known too. Running the
/// compression function backwards from there gives the chaining value the
/// partner starts from, and running it forwards gives the one it ends on. A
/// collision attack needs that end to meet this block's.
fn attacked(backend: Backend, chaining_out: [u32; 5], ctx: &mut Inner, candidates: u32) -> bool {
    // The hardware backends give the schedule but not the states that
    // recompression starts from, so they are recovered here, once, and only
    // for a block that has a candidate at all.
    let Inner {
        ihv1,
        m1,
        state_58,
        state_65,
        ..
    } = ctx;
    backend.ensure_states(ihv1, m1, state_58, state_65);

    for dv in &crate::ubc_check::SHA1_DVS {
        if candidates & (1 << dv.mask_bit) == 0 {
            continue;
        }

        for (partner, (word, difference)) in ctx.m2.iter_mut().zip(ctx.m1.iter().zip(&dv.dm)) {
            *partner = word ^ difference;
        }

        let Inner {
            ihv2,
            m2,
            state_58,
            state_65,
            ..
        } = ctx;
        let mut ends_on = [0u32; 5];
        recompression_step(
            dv.recompress_from,
            ihv2,
            &mut ends_on,
            m2,
            match dv.recompress_from {
                RecompressFrom::Step58 => state_58,
                RecompressFrom::Step65 => state_65,
            },
        );

        // A collision on the way out is the attack. The option to accept one
        // on the way in is for the reduced-step test vectors, which are the
        // only real examples that exist for a shortened SHA-1.
        if xor(&ends_on, &chaining_out) == 0
            || (ctx.reduced_round_collision && xor(&ctx.ihv1, &ctx.ihv2) == 0)
        {
            return true;
        }
    }

    false
}
