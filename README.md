# sha1dc

[![CI](https://github.com/srijs/sha1dc/actions/workflows/ci.yml/badge.svg)](https://github.com/srijs/sha1dc/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/sha1dc.svg)](https://crates.io/crates/sha1dc)
[![docs.rs](https://docs.rs/sha1dc/badge.svg)](https://docs.rs/sha1dc)

SHA-1 is cryptographically broken, because chosen-prefix collisions against it
are practical. However, there are still cases where it is needed for
compatibility, such as in `git`'s object identifiers.

This security issue can be mitigated by detecting those manufactured collisions.
This crate follows the method of Marc Stevens and Dan Shumow, which finds the
message blocks that a collision attack produces and reports them ([paper]).

To implement filtering for known disturbance vectors, it follows a code
generation approach. Each condition for an attack is a linear equation over two
bits of the expanded message. A solver searches the space they span for a set of
equations that suits the target instruction set, and emits code for it. Current
targets are `neon`, `sse2` and `avx2`, as well as a scalar baseline.

Where available, the implementation also uses SHA-1 hardware instructions on
`x86_64` and `aarch64`. Detection still does more work per block than plain
SHA-1, and slows down hashing by 20% to 35%, depending on the machine.

## Usage

```rust
let digest = sha1dc::digest(b"hello world")?;
assert_eq!(digest.to_string(), "2aae6c35c94fcfb415dbe95f408b9ce91ee846ed");
```

Two modes are provided, as two separate `Hasher` structs. `Hasher` keeps the
standard digest, with output equivalent to a non-detecting SHA-1
implementation. `mitigate::Hasher` computes an alternative digest instead. Both
report a detected attack as an error. The [documentation] covers both.

## Performance

These figures compare the crate against two others: [`sha1`], which does no
detection at all, and [`sha1-checked`], which is detects the same collisions and
is a direct translation of the original C code to Rust. Each is at the best it
can do on the machine. The machines are an Apple M4 laptop, an EC2 c7i.xlarge
and an EC2 c8g.xlarge. Throughput is in MiB/s, and as a fraction of the [`sha1`]
row.

| implementation               |    Apple M4 | Xeon Platinum 8488C |   Graviton4 |
|------------------------------|------------:|--------------------:|------------:|
| [`sha1`] 0.11.0              | 2984 (100%) |         1916 (100%) | 1617 (100%) |
| `sha1dc` 0.1.0               |  2315 (78%) |          1284 (67%) |  1253 (77%) |
| [`sha1-checked`] 0.11.0-rc.0 |   851 (29%) |           537 (28%) |   458 (28%) |

[`sha1`] and `sha1dc` both take the SHA-1 instructions of the machine,
SHA-NI on the Xeon and the ARMv8 ones on the M4 and the Graviton4, and
differ in whether they detect. The `sha1dc` shortfall from 100% is therefore
what detection costs: 22% to 33%, depending on the machine.

[`sha1-checked`] being based on the original C code is portable Rust with no
hardware path to take, so its row is lower for the detection and the missing
instructions at once, and the instructions are the larger of the two. Built with
none of them, this crate runs at 918, 549 and 482 MiB/s on the three machines,
which is 2% to 8% ahead of [`sha1-checked`] rather than the 2.4x to 2.7x of the
table.

## Features

- `std` *(default)*: Enables run-time CPU feature detection for hardware
  acceleration and `std::io::Write` support for the hasher.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this project by you, as defined in the Apache-2.0 license,
shall be dual licensed as above, without any additional terms or conditions.

[paper]: https://marc-stevens.nl/research/papers/C13-S.pdf
[`sha1`]: https://crates.io/crates/sha1
[`sha1-checked`]: https://crates.io/crates/sha1-checked
[documentation]: https://docs.rs/sha1dc
