//! Checks the whole hasher against upstream's, on any message, cut into any
//! updates, under every option.
//!
//! The input is three header bytes, then the message:
//!
//! - byte 0 holds the options. Bit 0 turns the UBC filter off, bit 1 counts
//!   reduced-round collisions, bit 2 selects the scalar backend, and bit 3
//!   the mitigating hasher. Bits 4 to 6 put one of the [`ATTACKS`] in front
//!   of the message.
//! - bytes 1 and 2 give the lengths the message is cut into, alternately.
//!
//! Random messages almost never reach a detection. But a collision stays one
//! under any identical suffix, so every message behind an attack is an attack
//! too, and the fuzzer tries the detection with whatever follows it.

#![no_main]

use libfuzzer_sys::fuzz_target;
use sha1::Digest as _;
use sha1dc::{Builder, Digest, mitigate};

/// Messages known to be collision attacks, from the crate's tests.
///
/// The `shattered` PDFs collide in their fourth and fifth blocks and agree
/// after them, so only those five blocks are kept. The reduced-round
/// collision counts only when reduced-round collisions do.
const ATTACKS: [&[u8]; 5] = [
    include_bytes!("../../tests/data/shattered-1.pdf")
        .split_at(320)
        .0,
    include_bytes!("../../tests/data/shattered-2.pdf")
        .split_at(320)
        .0,
    include_bytes!("../../tests/data/sha-mbles-1.bin"),
    include_bytes!("../../tests/data/sha-mbles-2.bin"),
    include_bytes!("../../tests/data/sha1_reducedsha_coll.bin"),
];

unsafe extern "C" {
    fn sha1dc_oracle(
        data: *const u8,
        len: usize,
        safe_hash: i32,
        use_ubc: i32,
        reduced_round: i32,
        out: *mut [u8; 20],
    ) -> i32;
}

#[derive(Clone, Copy, Debug)]
struct Options {
    use_ubc: bool,
    reduced_round: bool,
    scalar: bool,
    mitigate: bool,
}

/// A digest, and whether an attack was detected.
type Outcome = ([u8; 20], bool);

/// What upstream makes of `message`.
fn upstream(message: &[u8], options: Options) -> Outcome {
    let mut out = [0; 20];
    // SAFETY: it reads `message.len()` bytes from `message` and writes 20 to
    // `out`, and keeps neither pointer.
    let found = unsafe {
        sha1dc_oracle(
            message.as_ptr(),
            message.len(),
            options.mitigate.into(),
            options.use_ubc.into(),
            options.reduced_round.into(),
            &mut out,
        )
    };
    (out, found != 0)
}

/// Either hasher, so that both run the same checks.
#[derive(Clone)]
enum Hasher {
    Plain(sha1dc::Hasher),
    Mitigating(mitigate::Hasher),
}

impl Hasher {
    fn new(options: Options) -> Self {
        let mut builder = Builder::new()
            .internal_use_ubc(options.use_ubc)
            .internal_reduced_round_collisions(options.reduced_round);
        if options.scalar {
            builder = builder.internal_scalar_backend();
        }
        if options.mitigate {
            Self::Mitigating(builder.build_mitigating())
        } else {
            Self::Plain(builder.build())
        }
    }

    fn update(&mut self, data: &[u8]) {
        match self {
            Self::Plain(h) => h.update(data),
            Self::Mitigating(h) => h.update(data),
        }
    }

    fn finalize(self) -> Outcome {
        match self {
            Self::Plain(h) => outcome(h.finalize().map_err(|c| c.digest())),
            Self::Mitigating(h) => outcome(h.finalize().map_err(|m| m.digest())),
        }
    }

    fn finalize_reset(&mut self) -> Outcome {
        match self {
            Self::Plain(h) => outcome(h.finalize_reset().map_err(|c| c.digest())),
            Self::Mitigating(h) => outcome(h.finalize_reset().map_err(|m| m.digest())),
        }
    }
}

fn outcome(result: Result<Digest, Digest>) -> Outcome {
    match result {
        Ok(digest) => (digest.into(), false),
        Err(digest) => (digest.into(), true),
    }
}

/// `message` cut into pieces of the two lengths in turn.
fn pieces(mut message: &[u8], lengths: [usize; 2]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    for &n in lengths.iter().cycle() {
        if message.is_empty() {
            break;
        }
        let (piece, rest) = message.split_at(n.min(message.len()));
        out.push(piece);
        message = rest;
    }
    out
}

fuzz_target!(|input: &[u8]| {
    let Some((&[flags, a, b], suffix)) = input.split_first_chunk::<3>() else {
        return;
    };
    let options = Options {
        use_ubc: flags & 1 == 0,
        reduced_round: flags & 2 != 0,
        scalar: flags & 4 != 0,
        mitigate: flags & 8 != 0,
    };
    let message = match ATTACKS.get(usize::from(flags >> 4 & 7)) {
        Some(attack) => [attack, suffix].concat(),
        None => suffix.to_vec(),
    };
    let message = &message[..];
    let pieces = pieces(message, [usize::from(a) + 1, usize::from(b) * 8 + 1]);

    let want = upstream(message, options);

    // In pieces, with a copy taken halfway that finishes on its own.
    let mut hasher = Hasher::new(options);
    let half = pieces.len() / 2;
    for piece in &pieces[..half] {
        hasher.update(piece);
    }
    let mut copy = hasher.clone();
    for piece in &pieces[half..] {
        hasher.update(piece);
        copy.update(piece);
    }
    let got = hasher.finalize_reset();
    assert_eq!(got, want, "differs from upstream under {options:?}");
    assert_eq!(copy.finalize(), got, "a copy differs under {options:?}");

    // Again after the reset, which must forget a detected attack.
    hasher.update(message);
    assert_eq!(
        hasher.finalize_reset(),
        got,
        "differs after a reset under {options:?}"
    );

    // Only a mitigated attack changes the digest from plain SHA-1.
    let (digest, attacked) = got;
    if !(options.mitigate && attacked) {
        let plain = sha1::Sha1::digest(message);
        assert_eq!(digest[..], plain[..], "not plain SHA-1 under {options:?}");
    }
});
