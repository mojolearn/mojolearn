#!/bin/sh
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$here"

mkdir -p python/mojolearn
export MACOSX_DEPLOYMENT_TARGET="11.0"

pixi run mojo build --emit shared-lib \
    --target-cpu apple-m1 --target-accelerator metal:1 \
    -I . -I bindings \
    bindings/_mojolearn_estimators.mojo \
    -o python/mojolearn/_mojolearn_estimators.so

echo "built python/mojolearn/_mojolearn_estimators.so"
