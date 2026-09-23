//! One-shot digests of short messages, where building and finishing the
//! hasher dominate; the throughput benchmark's 16 KiB chunks hide that.
//!
//! Each iteration hashes a batch of distinct, freshly generated messages, so
//! the branch predictor cannot learn the input. 60 bytes pads into a
//! mostly-zero second block, which takes the tail's slow path.

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::rngs::SmallRng;
use rand::{RngExt as _, SeedableRng as _};
use sha1::Digest as _;

/// Message lengths, in bytes.
const SIZES: [usize; 4] = [32, 60, 256, 1024];

/// Messages per timed iteration.
const BATCH: usize = 64;

/// A fresh batch of `BATCH` messages of `len` bytes each, back to back.
fn fresh(len: usize) -> impl FnMut() -> Vec<u8> {
    let mut rng = SmallRng::seed_from_u64(0x0123_4567_89ab_cdef);
    move || {
        let mut data = vec![0; len * BATCH];
        rng.fill(&mut data[..]);
        data
    }
}

fn small(c: &mut Criterion) {
    let mut group = c.benchmark_group("digest");

    for len in SIZES {
        group.throughput(Throughput::Elements(BATCH as u64));

        group.bench_with_input(BenchmarkId::new("sha1", len), &len, |b, &len| {
            b.iter_batched_ref(
                fresh(len),
                |data| {
                    for msg in data.chunks_exact(len) {
                        black_box(sha1::Sha1::digest(black_box(msg)));
                    }
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("sha1dc", len), &len, |b, &len| {
            b.iter_batched_ref(
                fresh(len),
                |data| {
                    for msg in data.chunks_exact(len) {
                        let mut hasher = sha1dc::Hasher::new();
                        hasher.update(black_box(msg));
                        black_box(hasher.finalize().expect("no collision"));
                    }
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("sha1-checked", len), &len, |b, &len| {
            b.iter_batched_ref(
                fresh(len),
                |data| {
                    for msg in data.chunks_exact(len) {
                        let mut hasher = sha1_checked::Sha1::new();
                        hasher.update(black_box(msg));
                        black_box(hasher.try_finalize());
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, small);
criterion_main!(benches);
