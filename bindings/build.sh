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

# --emit shared-lib, not an executable: CPython dlopens this and calls
# PyInit__mojolearn. The name of the file must match that symbol's suffix or
# the import fails with "dynamic module does not define module export
# function", which is the least helpful error in the toolchain.
pixi run mojo build --emit shared-lib \
    -I . -I bindings \
    bindings/_mojolearn.mojo \
    -o python/mojolearn/_mojolearn.so

echo "built python/mojolearn/_mojolearn.so"
