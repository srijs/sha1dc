//! One-shot digests of short messages, where building and finishing the
//! hasher dominate; the throughput benchmark's 16 KiB chunks hide that.
//!
//! Each iteration hashes a batch of distinct, freshly generated messages, so
//! the branch predictor cannot learn the input. Their lengths are drawn
//! afresh too, uniformly over a range that spans whole blocks, so every way a
//! message can end is as common as it is in real data: a last block of
//! padding alone, one with a few bytes of message, one that is nearly full.

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::rngs::SmallRng;
use rand::{RngExt as _, SeedableRng as _};
use sha1::Digest as _;

/// The lengths each case draws from, in bytes, and its name.
const CASES: [(&str, std::ops::Range<usize>); 3] = [
    ("1-64", 1..65),
    ("65-512", 65..513),
    ("513-4096", 513..4097),
];

/// Messages per timed iteration. Enough that the mean length of a batch
/// barely moves from one iteration to the next.
const BATCH: usize = 256;

/// A batch of messages, back to back in `data`, and where each one ends.
struct Batch {
    data: Vec<u8>,
    ends: Vec<usize>,
}

impl Batch {
    fn messages(&self) -> impl Iterator<Item = &[u8]> {
        let starts = std::iter::once(0).chain(self.ends.iter().copied());
        starts.zip(&self.ends).map(|(s, &e)| &self.data[s..e])
    }
}

/// A fresh batch of `BATCH` messages, each of a length drawn from `lens`.
fn fresh(lens: std::ops::Range<usize>) -> impl FnMut() -> Batch {
    let mut rng = SmallRng::seed_from_u64(0x0123_4567_89ab_cdef);
    move || {
        let mut end = 0;
        let ends: Vec<usize> = (0..BATCH)
            .map(|_| {
                end += rng.random_range(lens.clone());
                end
            })
            .collect();
        let mut data = vec![0; end];
        rng.fill(&mut data[..]);
        Batch { data, ends }
    }
}

fn small(c: &mut Criterion) {
    let mut group = c.benchmark_group("digest");

    for (name, lens) in CASES {
        group.throughput(Throughput::Elements(BATCH as u64));

        group.bench_function(BenchmarkId::new("sha1", name), |b| {
            b.iter_batched_ref(
                fresh(lens.clone()),
                |batch| {
                    for msg in batch.messages() {
                        black_box(sha1::Sha1::digest(black_box(msg)));
                    }
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_function(BenchmarkId::new("sha1dc", name), |b| {
            b.iter_batched_ref(
                fresh(lens.clone()),
                |batch| {
                    for msg in batch.messages() {
                        let mut hasher = sha1dc::Hasher::new();
                        hasher.update(black_box(msg));
                        black_box(hasher.finalize().expect("no collision"));
                    }
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_function(BenchmarkId::new("sha1-checked", name), |b| {
            b.iter_batched_ref(
                fresh(lens.clone()),
                |batch| {
                    for msg in batch.messages() {
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
