#!/bin/sh
# Build the CPython extension into python/mojolearn/_mojolearn.so.
# Run from anywhere; requires pixi.
#
# Two include paths, both required. `-I .` is the mojolearn package root, so
# `cluster.estimator` and `neighbors.estimator` resolve. `-I bindings` is this
# directory: `_mojolearn.mojo` is the entry point, and if this project grows
# capability modules beside it the way mojotrees' bindings did, they resolve
# as top-level imports only when this directory is on the path. It costs
# nothing now and its absence would fail confusingly later.
#
# packaging/macos/build_release_wheel.sh runs this script rather than
# repeating the command, so the flags live in exactly one place.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$here"

mkdir -p python/mojolearn

# THE TARGET IS PINNED TO A PORTABLE BASELINE, NOT LEFT AT THE HOST.
#
# `mojo build` defaults --target-cpu and --target-features to WHATEVER CHIP
# RAN THE COMPILER. On this development machine that is:
#
#   --target-cpu apple-m4
#   --target-features ...,+bf16,+i8mm,+sme,+sme-f64f64,+sme-i16i64,+sme2
#
# +sme and +sme2 do not exist on M1, M2 or M3; +bf16 and +i8mm do not exist on
# M1. LLVM does not need to be ASKED to use an enabled feature -- it emits
# bfdot, smmla and bfmmla from ordinary loops once the bit is set -- so a
# host-built wheel SIGILLs on older Apple silicon, at the instruction, inside
# the extension, with no diagnostic a user can act on.
#
# Nothing downstream catches it either. arm64 Mach-O `cpusubtype` stays
# ARM64_ALL whatever -mcpu was, so a native-built wheel LOOKS portable to any
# header-reading check, and verifying it on the machine that built it proves
# nothing at all.
#
# apple-m1 is the oldest Apple silicon, and metal:1 likewise: measured
# 2026-08-20, a probe built at this baseline runs correctly on the M4, so the
# baseline is forward-compatible in the direction that matters.
#
# THE COST: a baseline artifact may be slower than a native one. mojotrees
# measured that difference by disassembly at 719,057 native instructions
# against 718,585 at apple-m1, of which 483 were `nop`. It has NOT been
# re-measured for mojolearn's kernels, and it should be before any published
# timing is attributed to this wheel rather than to a local build.
TARGET_FLAGS="--target-cpu apple-m1 --target-accelerator metal:1"

# --emit shared-lib, not an executable: CPython dlopens this and calls
# PyInit__mojolearn. The name of the file must match that symbol's suffix or
# the import fails with "dynamic module does not define module export
# function", which is the least helpful error in the toolchain.
# shellcheck disable=SC2086  # TARGET_FLAGS is deliberately word-split
pixi run mojo build --emit shared-lib \
    $TARGET_FLAGS \
    -I . -I bindings \
    bindings/_mojolearn.mojo \
    -o python/mojolearn/_mojolearn.so

echo "built python/mojolearn/_mojolearn.so"
