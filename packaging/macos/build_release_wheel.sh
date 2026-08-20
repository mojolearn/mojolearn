#!/bin/sh
# Build the macOS arm64 wheel: extension, staged MAX runtime, re-signed, packed.
#
# WHY STAGING IS NOT OPTIONAL. `otool -L` on the freshly built extension shows
#
#     @rpath/libKGENCompilerRTShared.dylib
#     @rpath/libAsyncRTMojoBindings.dylib
#
# Those live in the pixi environment. A wheel that ships without them imports
# fine on THIS machine, where the rpath still resolves, and fails on every
# other with a dyld error naming a path the user has never heard of. That is
# the worst kind of packaging bug: invisible to the person who built it.
#
# So they are copied next to the extension, the rpath is repointed at
# @loader_path/.dylibs, and the result is re-signed -- macOS invalidates the
# signature the moment install_name_tool rewrites a load command, and an
# unsigned dylib is killed on load on Apple silicon rather than merely warned
# about.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$here"

ENV_LIB="$here/.pixi/envs/default/lib"
PKG="$here/python/mojolearn"
DYLIBS="$PKG/.dylibs"

# Refreshed every build. These are COPIES of the repository root's files, and
# a stale copy in a published wheel is a wrong LICENSE or a wrong README on
# PyPI. python/.gitignore keeps them out of the checkout.
cp "$here/LICENSE" "$here/NOTICE" "$here/README.md" "$here/python/"

./bindings/build.sh

rm -rf "$DYLIBS"
mkdir -p "$DYLIBS"

# Exactly the two the extension names. Discovered rather than hardcoded, so a
# toolchain that starts linking a third fails loudly here instead of shipping
# a wheel that cannot import.
NEEDED=$(otool -L "$PKG/_mojolearn.so" | awk '/@rpath\//{print $1}' | sed 's|@rpath/||')
for lib in $NEEDED; do
    if [ ! -f "$ENV_LIB/$lib" ]; then
        echo "ERROR: $lib is linked by the extension but not in $ENV_LIB" >&2
        exit 1
    fi
    cp "$ENV_LIB/$lib" "$DYLIBS/$lib"
    codesign --force --sign - "$DYLIBS/$lib"
done

# Point the extension at its own directory. add_rpath rather than replacing,
# because the original rpath is harmless once this one resolves first.
install_name_tool -add_rpath "@loader_path/.dylibs" "$PKG/_mojolearn.so" 2>/dev/null || true
codesign --force --sign - "$PKG/_mojolearn.so"

echo "staged: $NEEDED"

# THE TAG AND THE BINARY MUST AGREE, and nothing else checks this.
# The wheel filename is what pip compares before it tries to load anything, so
# a tag above the binary's floor turns installable Macs away and a tag below it
# installs onto Macs where the extension cannot load. Both are silent on the
# machine that built the wheel.
BIN_MINOS=$(otool -l "$PKG/_mojolearn.so" | awk '/LC_BUILD_VERSION/{f=1} f&&/minos/{print $2; exit}')
TAG_MINOS=$(grep -E '^DEFAULT_MACOS_TARGET' "$here/python/setup.py" | sed 's/[^0-9.]//g')
if [ "$BIN_MINOS" != "$TAG_MINOS" ]; then
    echo "ERROR: binary minos $BIN_MINOS but setup.py tags $TAG_MINOS" >&2
    echo "       bindings/build.sh MACOS_FLOOR and setup.py DEFAULT_MACOS_TARGET must match" >&2
    exit 1
fi
echo "macOS floor: binary minos $BIN_MINOS == wheel tag $TAG_MINOS"

cd "$here/python"
rm -rf dist build ./*.egg-info
pixi run -e pkg python -m build --wheel --no-isolation

echo "wheel:"
ls -la "$here/python/dist"/*.whl
