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
        // The input chaining value. Only a flagged block goes on to use it,
        // and only within this iteration, so it belongs to the loop rather
        // than to the hasher.
        let ihv1 = ctx.h;

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
            crate::ubc_check::ubc_check(&ctx.m1, ctx.scalar_only)
        } else {
            !0
        };

        if candidates != 0 && attacked(backend, ihv1, ctx.h, ctx, candidates) {
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
///
/// Kept out of line: inlined, its frame joins [`compress`], which every
/// block traverses and one in twenty-one leaves for here. That trades short
/// messages for bulk, and short messages win — inlined, a Xeon 8488C gains
/// 0.6% on 16 KiB but loses 3.1% on 64 bytes, and git hashes far more small
/// objects than large ones.
#[inline(never)]
fn attacked(
    backend: Backend,
    ihv1: [u32; 5],
    chaining_out: [u32; 5],
    ctx: &mut Inner,
    candidates: u32,
) -> bool {
    // The hardware backends give the schedule but not the states that
    // recompression starts from, so they are recovered here, once, and only
    // for a block that has a candidate at all.
    let Inner {
        m1,
        state_58,
        state_65,
        ..
    } = ctx;
    backend.ensure_states(
        &ihv1,
        &chaining_out,
        m1,
        candidates & crate::ubc_check::STEP58_MASK != 0,
        state_58,
        state_65,
    );

    // The chaining value the partner starts from. It belongs to this call
    // and not to the hasher, which never reads it again, and which nineteen
    // blocks in twenty never get here to fill.
    let mut ihv2 = [0u32; 5];

    // Walking the set bits visits only the candidates. Reading `mask_bit`
    // out of all 32 entries instead would touch the whole table, which is
    // ten kilobytes, to find the one or two that are flagged.
    let mut remaining = candidates;
    while remaining != 0 {
        // The mask has a bit per entry, and masking the count keeps that
        // in range for the compiler as well as for the reader.
        let bit = (remaining.trailing_zeros() & 31) as usize;
        remaining &= remaining - 1;
        let dv = &crate::ubc_check::SHA1_DVS[bit];
        debug_assert_eq!(dv.mask_bit, bit as i32, "DV table is out of order");

        let mut ends_on = [0u32; 5];

        recompression_step(
            dv.recompress_from,
            &mut ihv2,
            &mut ends_on,
            &ctx.m1,
            &dv.dm,
            match dv.recompress_from {
                RecompressFrom::Step58 => &ctx.state_58,
                RecompressFrom::Step65 => &ctx.state_65,
            },
        );

        // A collision on the way out is the attack. The option to accept one
        // on the way in is for the reduced-step test vectors, which are the
        // only real examples that exist for a shortened SHA-1.
        if xor(&ends_on, &chaining_out) == 0
            || (ctx.reduced_round_collision && xor(&ihv1, &ihv2) == 0)
        {
            return true;
        }
    }

    false
}
