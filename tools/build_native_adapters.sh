#!/usr/bin/env bash
# Optional adapter build; run on an owned Linux build/test machine only.
set -euo pipefail
out=${1:?absolute output directory}
arch=${2:-gfx942}
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
mkdir -p "$out"
git rev-parse HEAD > "$out/source-head.txt"
git diff --binary HEAD > "$out/source-changes.patch"
cp .git/devpod_synced "$out/synced-source.txt" 2>/dev/null || true
tar -czf "$out/adapter-source.tar.gz" adapters tools/build_native_adapters.sh tools/check_native_adapters.py
pixi run mojo --version > "$out/compiler.txt" 2>&1
set +e
timeout --signal=TERM --kill-after=30 1200 pixi run mojo build -j "${MOJOLEARN_COMPILE_JOBS:-2}" \
    --emit shared-lib --target-cpu x86-64-v3 --target-accelerator "$arch" \
    -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . -I bindings \
    adapters/native/c_api.mojo -o "$out/libmojolearn_adapter.so" > "$out/build.log" 2>&1
rc=$?
set -e
echo "$rc" > "$out/build.rc"
if [ "$rc" != 0 ]; then exit "$rc"; fi
sha256sum "$out/libmojolearn_adapter.so" > "$out/native.sha256"
ldd "$out/libmojolearn_adapter.so" > "$out/native-libraries.txt"
nm -D "$out/libmojolearn_adapter.so" > "$out/native-symbols.txt"
# An interpreter dependency is not admitted for this standalone native bridge.
if rg -q 'libpython|\bU Py[A-Za-z_]' "$out/native-libraries.txt" "$out/native-symbols.txt"; then
    echo 'adapter unexpectedly depends on CPython' >&2
    exit 1
fi
