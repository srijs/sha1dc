//! Chooses the SHA-1 compression implementation.
//!
//! There is one, [`scalar`]. It leaves the expanded schedule behind, which
//! the detection needs and an ordinary SHA-1 would not keep.

use crate::BLOCK_SIZE;

mod scalar;

/// The implementation that computes the digest of a block.
#[derive(Clone, Copy)]
pub(crate) struct Backend;

impl Backend {
    pub(crate) fn new() -> Self {
        Self
    }

    /// The scalar implementation, on every CPU.
    pub(crate) fn scalar() -> Self {
        Self
    }

    /// Digests `block` into `state` and writes the input of the checker to
    /// `m1`, along with the two states that recompression starts from.
    pub(crate) fn compress_spill(
        &self,
        state: &mut [u32; 5],
        block: &[u8; BLOCK_SIZE],
        m1: &mut [u32; 80],
        state_58: &mut [u32; 5],
        state_65: &mut [u32; 5],
    ) {
        let mut block_u32 = [0u32; BLOCK_SIZE / 4];
        read_block(block, &mut block_u32);
        scalar::compress_spill(state, &block_u32, m1, state_58, state_65);
    }
}

/// Reads a block as big-endian `u32` words.
#[inline(always)]
pub(crate) fn read_block(block: &[u8; BLOCK_SIZE], out: &mut [u32; BLOCK_SIZE / 4]) {
    for (o, chunk) in out.iter_mut().zip(block.chunks_exact(4)) {
        *o = u32::from_be_bytes(chunk.try_into().unwrap());
    }
}
