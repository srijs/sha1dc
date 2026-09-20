#!/bin/sh
# Fails if src/ubc_check/ differs from what codegen/ currently emits.
#
# Run from anywhere; CI runs this too. Deliberately does not consult git, so it
# behaves the same for a tracked, untracked or dirty tree: it compares the
# files on disk against freshly generated output.
set -e
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

fresh=$(mktemp -d -t sha1dc-ubc.XXXXXX)
trap 'rm -rf "$fresh"' EXIT

env -u SHA1DC_SCALAR_GROUPS -u SHA1DC_NEON_GROUPS \
    -u SHA1DC_SSE2_GROUPS -u SHA1DC_AVX2_GROUPS \
    cargo run --quiet --manifest-path "$root/codegen/Cargo.toml" -- "$fresh" > /dev/null

stale=""
for f in scalar neon sse2 avx2 conditions; do
    rustfmt --edition 2024 "$fresh/$f.rs"
    if ! diff -u "$root/src/ubc_check/$f.rs" "$fresh/$f.rs"; then
        stale="$stale $f.rs"
    fi
done

if [ -n "$stale" ]; then
    echo >&2
    echo "error: stale generated file(s):$stale" >&2
    echo "The diff above shows what codegen/ produces. Regenerate with:" >&2
    echo "    cargo run -p sha1dc-codegen && cargo fmt -p sha1dc" >&2
    exit 1
fi
echo "src/ubc_check/ is up to date"
