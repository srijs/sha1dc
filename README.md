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

These figures compare the crate against the [`sha1`] crate, which has no
collision detection. The machines are an Apple M4 laptop, an EC2 c7i.xlarge
and an EC2 c8g.xlarge.

| machine              | backend             | [`sha1`]  | `sha1dc`  | ratio |
|----------------------|---------------------|-----------|-----------|-------|
| Apple M4             | SHA-1 instructions  | 2977 MiB/s| 2296 MiB/s|   77% |
| Apple M4             | scalar              | 1332 MiB/s|  917 MiB/s|   69% |
| Xeon Platinum 8488C  | SHA-NI              | 1917 MiB/s| 1289 MiB/s|   67% |
| Xeon Platinum 8488C  | scalar              |  826 MiB/s|  544 MiB/s|   66% |
| Graviton4            | SHA-1 instructions  | 1616 MiB/s| 1251 MiB/s|   77% |
| Graviton4            | scalar              |  697 MiB/s|  479 MiB/s|   69% |

A scalar row builds the [`sha1`] crate with its own scalar backend, so that
both columns use the same class of instructions.

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
[documentation]: https://docs.rs/sha1dc
