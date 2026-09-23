//! Throughput benchmark, runnable on stable: `cargo bench -p sha1dc-bench`.
//!
//! Two baselines. The `sha1` crate hashes without detecting anything, so the
//! distance to it is the cost of detection. The `sha1-checked` crate detects
//! the same collisions this crate does, in portable Rust, so the distance to
//! it is what the hardware backends buy.
//!
//! The cases pair up: `sha1dc` against `sha1-checked` is the default of each,
//! and `sha1dc/no-ubc` against `sha1-checked/no-ubc` is each with the
//! unavoidable-bitconditions filter turned off, which makes both run the full
//! recompression on every block.
//!
//! The `sha1dc/scalar` case takes no hardware at all: neither the SHA-1
//! instructions nor a vector form of the UBC check. To compare it against a
//! like-for-like baseline, run again with
//! `RUSTFLAGS='--cfg sha1_backend="soft"'`, which is the switch of the `sha1`
//! crate. In that run the `sha1dc` case still uses hardware.
//!
//! To compare two revisions, record the first with
//! `cargo bench -- --save-baseline before` and measure the second against it
//! with `cargo bench -- --baseline before`. That reports a confidence
//! interval for the change, which a single figure cannot give.

use std::hint::black_box;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use rand::rngs::SmallRng;
use rand::{RngExt as _, SeedableRng as _};
use sha1::Digest as _;

/// One iteration hashes this many bytes. Large enough that constructing the
/// hasher and padding the last block stay under a percent of the work.
const CHUNK: usize = 16 * 1024;

/// Fresh data for every iteration, made outside the timed part.
///
/// Hashing one buffer over and over lets the branch predictor learn which of
/// its blocks enter the UBC tail and which recompress, and those branches are
/// most of what the tail costs. On an Apple M4 that made `sha1dc` look 14%
/// faster than on data that does not repeat, and favoured a shorter prefix
/// than real data wants.
/// Cycling through a larger pool is not enough: the M4 still learns much of
/// a megabyte, and a pool large enough to defeat it no longer fits in L2. So
/// the data never repeats, and each input is still hashed from cache.
///
/// Seeded, so that a run is repeatable for a given lockfile. Only the
/// statistics of the data matter to the measurement, not its exact bytes.
fn fresh() -> impl FnMut() -> Vec<u8> {
    let mut rng = SmallRng::seed_from_u64(0x0123_4567_89ab_cdef);
    move || {
        let mut data = vec![0; CHUNK];
        rng.fill(&mut data[..]);
        data
    }
}

fn throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");
    group.throughput(Throughput::Bytes(CHUNK as u64));

    group.bench_function("sha1", |b| {
        b.iter_batched_ref(
            fresh(),
            |data| {
                let mut hasher = sha1::Sha1::new();
                hasher.update(black_box(&data[..]));
                black_box(hasher.finalize())
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("sha1dc", |b| {
        b.iter_batched_ref(
            fresh(),
            |data| {
                let mut hasher = sha1dc::Hasher::new();
                hasher.update(black_box(&data[..]));
                black_box(hasher.finalize().expect("no collision"))
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("sha1-checked", |b| {
        b.iter_batched_ref(
            fresh(),
            |data| {
                let mut hasher = sha1_checked::Sha1::new();
                hasher.update(black_box(&data[..]));
                black_box(hasher.try_finalize())
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("sha1dc/scalar", |b| {
        b.iter_batched_ref(
            fresh(),
            |data| {
                let mut hasher = sha1dc::Hasher::builder().internal_scalar_backend().build();
                hasher.update(black_box(&data[..]));
                black_box(hasher.finalize().expect("no collision"))
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("sha1dc/no-ubc", |b| {
        b.iter_batched_ref(
            fresh(),
            |data| {
                let mut hasher = sha1dc::Hasher::builder().internal_use_ubc(false).build();
                hasher.update(black_box(&data[..]));
                black_box(hasher.finalize().expect("no collision"))
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("sha1-checked/no-ubc", |b| {
        b.iter_batched_ref(
            fresh(),
            |data| {
                let mut hasher = sha1_checked::Sha1::builder().use_ubc(false).build();
                hasher.update(black_box(&data[..]));
                black_box(hasher.try_finalize())
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, throughput);
criterion_main!(benches);
