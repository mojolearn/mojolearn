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

# THE macOS FLOOR, AND IT DEFAULTED ABSURDLY HIGH.
#
# `mojo build` takes its deployment target from the host SDK. On this machine
# that produced an extension with `minos 26.0` -- a wheel installable only on
# macOS 26, released weeks ago, which is very nearly nobody.
#
# The MAX runtime dylibs this links are built at `minos 11.0`. Modular shipped
# them wide; only our own compile step was narrow. Measured 2026-08-20:
# MACOSX_DEPLOYMENT_TARGET of 11.0, 13.0 and 14.0 each produce exactly that
# minos, so the floor is ours to choose.
#
# 11.0 (Big Sur) is chosen because it is the FIRST macOS that runs on Apple
# silicon at all. There is no Apple silicon Mac that can run anything older,
# so this is the widest floor that means anything, and it matches the dylibs.
#
# setup.py's DEFAULT_MACOS_TARGET must equal this or the wheel TAG and the
# BINARY disagree, which is a published lie in one of two directions: too low
# and it installs where it cannot load, too high and it is refused by Macs
# that could have run it. packaging/macos/build_release_wheel.sh checks that
# they agree by reading the Mach-O header after the build.
MACOS_FLOOR="11.0"
export MACOSX_DEPLOYMENT_TARGET="$MACOS_FLOOR"

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
