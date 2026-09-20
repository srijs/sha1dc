//! Throughput benchmark, runnable on stable: `cargo bench`.
//!
//! The baseline is the `sha1` crate. The percentages therefore give the cost
//! of detection and do not compare this crate against itself. Only the first
//! `sha1dc` row has a percentage. That row and the baseline both select the
//! best backend for the machine, so you can compare them.
//!
//! To compare the scalar paths, run again with
//! `RUSTFLAGS='--cfg sha1_backend="soft"'`, which is the switch of the `sha1`
//! crate, and read the `sha1dc (scalar)` row. In that run the percentage on
//! the plain row has no meaning, because this crate still uses hardware.

use std::hint::black_box;
use std::time::{Duration, Instant};

use sha1::Digest as _;

const CHUNK: usize = 16 * 1024;
const TARGET: Duration = Duration::from_millis(500);

fn main() {
    let data = pseudorandom(CHUNK);

    let baseline = bench("sha1 crate", &data, None, |data, iters| {
        let mut hasher = sha1::Sha1::new();
        for _ in 0..iters {
            hasher.update(black_box(data));
        }
        black_box(hasher.finalize());
    });

    bench("sha1dc", &data, Some(baseline), |data, iters| {
        let mut hasher = sha1dc::Hasher::new();
        for _ in 0..iters {
            hasher.update(black_box(data));
        }
        black_box(hasher.finalize().expect("no collision"));
    });

    // No ratio. If the baseline is not also set to soft, this compares the
    // scalar path against the hardware path of the `sha1` crate.
    bench("sha1dc (scalar)", &data, None, |data, iters| {
        let mut hasher = sha1dc::Hasher::builder().internal_scalar_backend().build();
        for _ in 0..iters {
            hasher.update(black_box(data));
        }
        black_box(hasher.finalize().expect("no collision"));
    });

    bench(
        "sha1dc, no ubc check",
        &data,
        Some(baseline),
        |data, iters| {
            let mut hasher = sha1dc::Hasher::builder().internal_use_ubc(false).build();
            for _ in 0..iters {
                hasher.update(black_box(data));
            }
            black_box(hasher.finalize().expect("no collision"));
        },
    );
}

/// Measures MiB/s and prints it with the fraction of `baseline` that it
/// reaches.
fn bench(name: &str, data: &[u8], baseline: Option<f64>, mut run: impl FnMut(&[u8], usize)) -> f64 {
    let mut iters = 64;
    loop {
        let start = Instant::now();
        run(data, iters);
        let elapsed = start.elapsed();

        if elapsed >= TARGET || iters >= 1 << 24 {
            let mib = (iters * data.len()) as f64 / (1024.0 * 1024.0);
            let rate = mib / elapsed.as_secs_f64();
            match baseline {
                Some(baseline) => {
                    println!(
                        "{name:<22} {rate:>8.0} MiB/s  {:>3.0}%",
                        100.0 * rate / baseline
                    )
                }
                None => println!("{name:<22} {rate:>8.0} MiB/s"),
            }
            return rate;
        }
        iters *= 2;
    }
}

fn pseudorandom(len: usize) -> Vec<u8> {
    let mut seed = 0x0123_4567_89ab_cdefu64;
    (0..len)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 24) as u8
        })
        .collect()
}
