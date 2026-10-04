#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Compile-only test binding for M2; no import/GPU gate, timing, or execution.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "${MOJOLEARN_BUILD_LOCK_HELD:-}" != 1 ]; then
    exec tools/with_build_lock.sh bash bindings/build_callpath_probe.sh "$@"
fi
[[ $(uname -s) == Darwin && $(uname -m) == arm64 ]] || { echo 'Apple silicon build required' >&2; exit 2; }
[[ ${MOJOLEARN_NUMERIC_MODE:-fast} == fast ]] || { echo 'callpath probe is FAST only' >&2; exit 2; }
unset MACOSX_DEPLOYMENT_TARGET
sdk=$(xcrun --sdk macosx --show-sdk-version)
mkdir -p python/mojolearn
out=python/mojolearn/_mojolearn_callpath_probe.so
# Word splitting is the existing compile_arms define-list interface (bash).
pixi run mojo build -j 1 --emit shared-lib ${MOJOLEARN_MOJO_BUILD_FLAGS:-} \
    --target-cpu apple-m1 --target-accelerator metal:1 \
    -Xlinker -platform_version -Xlinker macos -Xlinker 11.0 -Xlinker "$sdk" \
    -I . -I bindings bindings/_mojolearn_callpath_probe.mojo -o "$out"
otool -L "$out" > "$out.dependencies.txt"
otool -l "$out" > "$out.load-commands.txt"
echo "CALLPATH-BUILD output=$out mode=fast target=apple-m1 accelerator=metal:1 execution=NONE"
