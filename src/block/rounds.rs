//! SHA-1 block compression, and the recompression that detection needs.
//!
//! Everything here runs whatever backend is in use. [`states_back_from_h`]
//! recovers the states a hardware backend does not spill,
//! [`recompression_step`] is the detection itself, and [`compression_w`] is
//! the mitigation. The steps themselves are also the scalar backend, in
//! [`super::backend::scalar`].
//!
//! # The compression function
//!
//! Written from [FIPS 180-1]. For each step `t` of 80 the working state
//! `(A, B, C, D, E)` becomes
//!
//! ```text
//! (ROTL(A,5) + f_t(B,C,D) + E + K_t + W_t,  A,  ROTL(B,30),  C,  D)
//! ```
//!
//! Only the first and third words change. The rest move one place. So a step
//! does not move anything: it writes the new word over `E`, rotates `B`, and
//! the caller renames the five words one place round. After five steps the
//! names line up again, which is why the work here is grouped in fives.
//!
//! # Recompression
//!
//! Detection needs more than the digest. Algorithm 1 of the [paper] asks,
//! for each candidate attack, what the *other* message of the pair would
//! hash to. That message differs from this one by a fixed difference, so its
//! state at one step is known too. From there the compression function runs
//! backwards to the input chaining value and forwards to the output one.
//!
//! A step is reversible because only two words change: `ROTL(B,30)` undoes,
//! and the added word subtracts. [`recompression_step`] runs backwards from a
//! stored state to step 0, then forwards to step 80, and adds the two to get
//! the output chaining value the pair would share.
//!
//! Steps 58 and 65 are the two the DV table names. The scalar backend stores
//! the state there on the way past; a hardware backend does not, so
//! [`states_back_from_h`] recovers it afterwards by running the end of the
//! compression in reverse. A stored state is in the order `(A, B, C, D, E)`
//! of the step it belongs to.
//!
//! [FIPS 180-1]: https://csrc.nist.gov/pubs/fips/180-1/final
//! [paper]: https://marc-stevens.nl/research/papers/C13-S.pdf

use crate::Schedule;
use crate::ubc_check::RecompressFrom;

/// The round constants, one per 20 steps. FIPS 180-1, section 5.
pub(crate) const K: [u32; 4] = [0x5A82_7999, 0x6ED9_EBA1, 0x8F1B_BCDC, 0xCA62_C1D6];

/// `Ch`, the choice function of steps 0 to 19.
#[inline(always)]
pub(crate) fn ch(b: u32, c: u32, d: u32) -> u32 {
    d ^ (b & (c ^ d))
}

/// `Parity`, the XOR of steps 20 to 39 and 60 to 79.
#[inline(always)]
pub(crate) fn parity(b: u32, c: u32, d: u32) -> u32 {
    b ^ c ^ d
}

/// `Maj`, the majority function of steps 40 to 59.
#[inline(always)]
pub(crate) fn maj(b: u32, c: u32, d: u32) -> u32 {
    (b & c) | (d & (b | c))
}

/// One step. Writes over `$e` and rotates `$b`; the caller renames.
macro_rules! step {
    ($f:ident, $k:expr, $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $w:expr) => {{
        // The four terms that do not depend on `$e` are summed first. `$e`
        // carries the chain from one step to the next, so putting it last
        // keeps three of the four additions off that chain.
        $e = $e.wrapping_add(
            $a.rotate_left(5)
                .wrapping_add($f($b, $c, $d))
                .wrapping_add($k)
                .wrapping_add($w),
        );
        $b = $b.rotate_left(30);
    }};
}

/// One step backwards. Undoes the rotation first, because the word that
/// subtracts is a function of the unrotated value.
macro_rules! unstep {
    ($f:ident, $k:expr, $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $w:expr) => {{
        $c = $c.rotate_right(30);
        $a = $a.wrapping_sub(
            $b.rotate_left(5)
                .wrapping_add($f($c, $d, $e))
                .wrapping_add($k)
                .wrapping_add($w),
        );
    }};
}

/// Five steps, which is one turn of the names. The words go in and come out
/// under the names they started with.
macro_rules! five {
    ($f:ident, $k:expr, $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $w:expr, $t:expr) => {{
        step!($f, $k, $a, $b, $c, $d, $e, $w.at($t));
        step!($f, $k, $e, $a, $b, $c, $d, $w.at($t + 1));
        step!($f, $k, $d, $e, $a, $b, $c, $w.at($t + 2));
        step!($f, $k, $c, $d, $e, $a, $b, $w.at($t + 3));
        step!($f, $k, $b, $c, $d, $e, $a, $w.at($t + 4));
    }};
}

/// Five steps backwards: the same five in the other order.
macro_rules! unfive {
    ($f:ident, $k:expr, $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $w:expr, $t:expr) => {{
        unstep!($f, $k, $a, $b, $c, $d, $e, $w.at($t + 4));
        unstep!($f, $k, $b, $c, $d, $e, $a, $w.at($t + 3));
        unstep!($f, $k, $c, $d, $e, $a, $b, $w.at($t + 2));
        unstep!($f, $k, $d, $e, $a, $b, $c, $w.at($t + 1));
        unstep!($f, $k, $e, $a, $b, $c, $d, $w.at($t));
    }};
}

/// A source of schedule words for the rounds above: an array for the
/// compression, [`Xor`] for the recompression.
///
/// Every index is a constant at the call, so `at` is a constant-offset load
/// with no bounds check. [`Schedule`] knows where the spill actually put the
/// word; nothing here has to.
trait Words {
    fn at(&self, t: usize) -> u32;
}

impl Words for Schedule {
    #[inline(always)]
    fn at(&self, t: usize) -> u32 {
        self[t]
    }
}

/// The partner schedule of a candidate attack, as the two arrays it is the
/// XOR of rather than the array it would be.
///
/// The recompression reads every word exactly once, so storing them buys
/// nothing — and building the array is dearer than it looks. The compiler
/// vectorises the XOR, then has to move all eighty words back to the integer
/// registers the steps use: a hundred `fmov`/`mov.s` on `aarch64`, 134
/// `movd`/`movq` on `x86_64`. Forming each word where its step runs is 8% to
/// 16% faster per candidate.
struct Xor<'a> {
    m1: &'a Schedule,
    dm: &'a [u32; 80],
}

impl Words for Xor<'_> {
    #[inline(always)]
    fn at(&self, t: usize) -> u32 {
        // Only `m1` is a spill; the difference comes from the table, which
        // is in step order on every target.
        self.m1[t] ^ self.dm[t]
    }
}

/// One step, expanding the schedule word it needs first.
///
/// The word is kept for the step rather than read back from `w`, and the
/// expansion sits next to a chain of steps that cannot start early, so the
/// processor fills one with the other. Expanding the whole schedule up front
/// instead measures 14% slower.
///
/// The indices are constants, so the bounds are settled at compile time. A
/// loop over `16..80` costs four bounds checks a word, worth another 25%.
macro_rules! step_expand {
    ($f:ident, $k:expr, $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $w:expr, $t:expr) => {{
        let word = ($w[$t - 3] ^ $w[$t - 8] ^ $w[$t - 14] ^ $w[$t - 16]).rotate_left(1);
        $w[$t] = word;
        step!($f, $k, $a, $b, $c, $d, $e, word);
    }};
}

/// One step over a word of the block itself, which is stored as it is used.
/// Copying the block into `w` first and reading it back costs 16 loads a
/// block, worth about 4%.
macro_rules! step_load {
    ($f:ident, $k:expr, $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $w:expr, $m:expr, $t:expr) => {{
        let word = $m[$t];
        $w[$t] = word;
        step!($f, $k, $a, $b, $c, $d, $e, word);
    }};
}

/// Five steps over the block itself, which is one turn of the names.
macro_rules! five_load {
    ($f:ident, $k:expr, $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $w:expr, $m:expr, $t:expr) => {{
        step_load!($f, $k, $a, $b, $c, $d, $e, $w, $m, $t);
        step_load!($f, $k, $e, $a, $b, $c, $d, $w, $m, $t + 1);
        step_load!($f, $k, $d, $e, $a, $b, $c, $w, $m, $t + 2);
        step_load!($f, $k, $c, $d, $e, $a, $b, $w, $m, $t + 3);
        step_load!($f, $k, $b, $c, $d, $e, $a, $w, $m, $t + 4);
    }};
}

/// Five expanding steps, which is one turn of the names.
macro_rules! five_expand {
    ($f:ident, $k:expr, $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $w:expr, $t:expr) => {{
        step_expand!($f, $k, $a, $b, $c, $d, $e, $w, $t);
        step_expand!($f, $k, $e, $a, $b, $c, $d, $w, $t + 1);
        step_expand!($f, $k, $d, $e, $a, $b, $c, $w, $t + 2);
        step_expand!($f, $k, $c, $d, $e, $a, $b, $w, $t + 3);
        step_expand!($f, $k, $b, $c, $d, $e, $a, $w, $t + 4);
    }};
}

/// The round primitives are shared with [`super::backend::scalar`], which
/// runs them over a whole block.
pub(crate) use {five_expand, five_load, step, step_expand, step_load};

#[inline(always)]
pub(crate) fn add(left: &mut [u32; 5], right: [u32; 5]) {
    for (l, r) in left.iter_mut().zip(right) {
        *l = l.wrapping_add(r);
    }
}

/// All 80 steps over an expanded schedule, added into `ihv`.
pub(crate) fn compression_w(ihv: &mut [u32; 5], w: &Schedule) {
    let [mut a, mut b, mut c, mut d, mut e] = *ihv;

    five!(ch, K[0], a, b, c, d, e, w, 0);
    five!(ch, K[0], a, b, c, d, e, w, 5);
    five!(ch, K[0], a, b, c, d, e, w, 10);
    five!(ch, K[0], a, b, c, d, e, w, 15);

    five!(parity, K[1], a, b, c, d, e, w, 20);
    five!(parity, K[1], a, b, c, d, e, w, 25);
    five!(parity, K[1], a, b, c, d, e, w, 30);
    five!(parity, K[1], a, b, c, d, e, w, 35);

    five!(maj, K[2], a, b, c, d, e, w, 40);
    five!(maj, K[2], a, b, c, d, e, w, 45);
    five!(maj, K[2], a, b, c, d, e, w, 50);
    five!(maj, K[2], a, b, c, d, e, w, 55);

    five!(parity, K[3], a, b, c, d, e, w, 60);
    five!(parity, K[3], a, b, c, d, e, w, 65);
    five!(parity, K[3], a, b, c, d, e, w, 70);
    five!(parity, K[3], a, b, c, d, e, w, 75);

    add(ihv, [a, b, c, d, e]);
}

/// Recovers the stored states by running *backwards* from the output.
///
/// The hardware backends give the schedule but not the states. Both chaining
/// values are known by the time detection asks for them, and their difference
/// is the state at step 80, so the two states the DV table names are 15 and
/// 22 steps back from the end rather than 58 and 65 steps forward from the
/// start. `need_58` says whether the seven steps past step 65 are wanted;
/// a block whose candidates all recompress from step 65 does not need them.
pub(crate) fn states_back_from_h(
    ihv_before: &[u32; 5],
    ihv_after: &[u32; 5],
    w: &Schedule,
    need_58: bool,
    state_58: &mut [u32; 5],
    state_65: &mut [u32; 5],
) {
    // The state at step 80, which the final addition hid.
    let [mut a, mut b, mut c, mut d, mut e] = [
        ihv_after[0].wrapping_sub(ihv_before[0]),
        ihv_after[1].wrapping_sub(ihv_before[1]),
        ihv_after[2].wrapping_sub(ihv_before[2]),
        ihv_after[3].wrapping_sub(ihv_before[3]),
        ihv_after[4].wrapping_sub(ihv_before[4]),
    ];

    // Fifteen steps back is three whole turns of the names, so step 65 comes
    // out in order.
    unfive!(parity, K[3], a, b, c, d, e, w, 75);
    unfive!(parity, K[3], a, b, c, d, e, w, 70);
    unfive!(parity, K[3], a, b, c, d, e, w, 65);
    *state_65 = [a, b, c, d, e];

    if need_58 {
        unfive!(parity, K[3], a, b, c, d, e, w, 60);
        // Steps 59 and 58 are in the third round, and leave the names two
        // places round.
        unstep!(maj, K[2], a, b, c, d, e, w[59]);
        unstep!(maj, K[2], b, c, d, e, a, w[58]);
        *state_58 = [c, d, e, a, b];
    }
}

/// Recovers the two stored states from a schedule that is already expanded.
///
/// This runs the 65 steps that lead to them and no more. It is the forward
/// reference that [`states_back_from_h`] is checked against; the hardware
/// path uses the backward form, which is fewer steps.
#[cfg(test)]
pub(crate) fn states_from_w(
    ihv: &[u32; 5],
    w: &Schedule,
    state_58: &mut [u32; 5],
    state_65: &mut [u32; 5],
) {
    let [mut a, mut b, mut c, mut d, mut e] = *ihv;

    five!(ch, K[0], a, b, c, d, e, w, 0);
    five!(ch, K[0], a, b, c, d, e, w, 5);
    five!(ch, K[0], a, b, c, d, e, w, 10);
    five!(ch, K[0], a, b, c, d, e, w, 15);

    five!(parity, K[1], a, b, c, d, e, w, 20);
    five!(parity, K[1], a, b, c, d, e, w, 25);
    five!(parity, K[1], a, b, c, d, e, w, 30);
    five!(parity, K[1], a, b, c, d, e, w, 35);

    five!(maj, K[2], a, b, c, d, e, w, 40);
    five!(maj, K[2], a, b, c, d, e, w, 45);
    five!(maj, K[2], a, b, c, d, e, w, 50);

    // Step 58 falls three into a turn of the names, so that turn is written
    // out a step at a time. Step 65 falls on a turn, so the names are already
    // in order there.
    step!(maj, K[2], a, b, c, d, e, w[55]);
    step!(maj, K[2], e, a, b, c, d, w[56]);
    step!(maj, K[2], d, e, a, b, c, w[57]);
    *state_58 = [c, d, e, a, b];
    step!(maj, K[2], c, d, e, a, b, w[58]);
    step!(maj, K[2], b, c, d, e, a, w[59]);

    five!(parity, K[3], a, b, c, d, e, w, 60);
    *state_65 = [a, b, c, d, e];
}

/// Steps 39 down to 0, from the state at step 40. Each caller undoes its own
/// way back to there first, because the two stored states sit at different
/// places in the third round.
macro_rules! back_to_start {
    ($w:expr, $a:ident, $b:ident, $c:ident, $d:ident, $e:ident) => {{
        unfive!(parity, K[1], $a, $b, $c, $d, $e, $w, 35);
        unfive!(parity, K[1], $a, $b, $c, $d, $e, $w, 30);
        unfive!(parity, K[1], $a, $b, $c, $d, $e, $w, 25);
        unfive!(parity, K[1], $a, $b, $c, $d, $e, $w, 20);

        unfive!(ch, K[0], $a, $b, $c, $d, $e, $w, 15);
        unfive!(ch, K[0], $a, $b, $c, $d, $e, $w, 10);
        unfive!(ch, K[0], $a, $b, $c, $d, $e, $w, 5);
        unfive!(ch, K[0], $a, $b, $c, $d, $e, $w, 0);
    }};
}

/// The chaining values the partner of this block would give, from its state
/// at step 58 or at step 65.
///
/// The partner schedule is `m1` XORed with `dm`, formed a word at a time as
/// the steps run. `ihvin` gets the input chaining value, reached by running
/// backwards, and `ihvout` the output one, which is `ihvin` plus the state
/// at step 80.
#[inline(always)]
pub(crate) fn recompression_step(
    step: RecompressFrom,
    ihvin: &mut [u32; 5],
    ihvout: &mut [u32; 5],
    m1: &Schedule,
    dm: &[u32; 80],
    state: &[u32; 5],
) {
    let me2 = &Xor { m1, dm };
    let [mut a, mut b, mut c, mut d, mut e] = *state;

    match step {
        RecompressFrom::Step58 => {
            // Back over the three steps that are not part of a whole turn,
            // which leaves the names three places round.
            unstep!(maj, K[2], a, b, c, d, e, me2.at(57));
            unstep!(maj, K[2], b, c, d, e, a, me2.at(56));
            unstep!(maj, K[2], c, d, e, a, b, me2.at(55));
            unfive!(maj, K[2], d, e, a, b, c, me2, 50);
            unfive!(maj, K[2], d, e, a, b, c, me2, 45);
            unfive!(maj, K[2], d, e, a, b, c, me2, 40);
            back_to_start!(me2, d, e, a, b, c);
            *ihvin = [d, e, a, b, c];

            [a, b, c, d, e] = *state;
            step!(maj, K[2], a, b, c, d, e, me2.at(58));
            step!(maj, K[2], e, a, b, c, d, me2.at(59));
            five!(parity, K[3], d, e, a, b, c, me2, 60);
            five!(parity, K[3], d, e, a, b, c, me2, 65);
            five!(parity, K[3], d, e, a, b, c, me2, 70);
            five!(parity, K[3], d, e, a, b, c, me2, 75);
            [a, b, c, d, e] = [d, e, a, b, c];
        }
        RecompressFrom::Step65 => {
            unfive!(parity, K[3], a, b, c, d, e, me2, 60);
            unfive!(maj, K[2], a, b, c, d, e, me2, 55);
            unfive!(maj, K[2], a, b, c, d, e, me2, 50);
            unfive!(maj, K[2], a, b, c, d, e, me2, 45);
            unfive!(maj, K[2], a, b, c, d, e, me2, 40);
            back_to_start!(me2, a, b, c, d, e);
            *ihvin = [a, b, c, d, e];

            [a, b, c, d, e] = *state;
            five!(parity, K[3], a, b, c, d, e, me2, 65);
            five!(parity, K[3], a, b, c, d, e, me2, 70);
            five!(parity, K[3], a, b, c, d, e, me2, 75);
        }
    }

    *ihvout = *ihvin;
    add(ihvout, [a, b, c, d, e]);
}
