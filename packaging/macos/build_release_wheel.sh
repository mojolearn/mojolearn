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

# EVERY EXTENSION IN THE WHEEL IS BUILT HERE. `pyproject.toml`'s
# package-data globs `*.so`, so an extension that is NOT built here is not
# absent from the wheel -- it is shipped STALE, whatever happens to be sitting
# in the working tree from an earlier build. That is how `_mojolearn.so` came
# to ship an artifact predating `eval_x`/`eval_y`, where every `fit` raised
# "takes 6 positional arguments but 8 were given".
./bindings/build.sh
./bindings/build_gbdt.sh

# The FULL transitive closure, walked rather than sampled. See
# packaging/macos/stage_dylibs.py: reading only the extension's direct
# dependencies staged 2 dylibs when the real closure is 4, and shipped a wheel
# that failed on install with "Library not loaded:
# @rpath/libMSupportGlobals.dylib, referenced from .dylibs/libAsyncRT...".
# It also verifies statically that nothing is left unresolved, which is the
# only form of this check that means anything on the build machine.
# BOTH EXTENSIONS IN ONE CALL, because there is one `.dylibs` and the script
# wipes it before staging. Two calls would leave the first extension's closure
# deleted -- invisibly, because on THIS machine the original rpath still
# resolves into the pixi environment.
pixi run -e pkg python "$here/packaging/macos/stage_dylibs.py" \
    "$PKG/_mojolearn.so" "$PKG/_mojolearn_gbdt.so" "$ENV_LIB"



# THE TAG AND THE BINARY MUST AGREE, and nothing else checks this.
# The wheel filename is what pip compares before it tries to load anything, so
# a tag above the binary's floor turns installable Macs away and a tag below it
# installs onto Macs where the extension cannot load. Both are silent on the
# machine that built the wheel.
# CHECKED FOR EVERY EXTENSION, not just the first: one wheel carries one tag,
# and the tag is only honest if it is the floor of EVERYTHING inside.
TAG_MINOS=$(grep -E '^DEFAULT_MACOS_TARGET' "$here/python/setup.py" | sed 's/[^0-9.]//g')
for so in "$PKG/_mojolearn.so" "$PKG/_mojolearn_gbdt.so"; do
    BIN_MINOS=$(otool -l "$so" | awk '/LC_BUILD_VERSION/{f=1} f&&/minos/{print $2; exit}')
    if [ "$BIN_MINOS" != "$TAG_MINOS" ]; then
        echo "ERROR: $(basename "$so") minos $BIN_MINOS but setup.py tags $TAG_MINOS" >&2
        echo "       MACOS_FLOOR in bindings/build.sh and bindings/build_gbdt.sh" >&2
        echo "       and DEFAULT_MACOS_TARGET in python/setup.py must all match" >&2
        exit 1
    fi
    echo "macOS floor: $(basename "$so") minos $BIN_MINOS == wheel tag $TAG_MINOS"
done

# THE ISA BASELINE, WHICH NO HEADER CAN SEE. arm64 Mach-O cpusubtype stays
# ARM64_ALL whatever --target-cpu was, so this has to disassemble. Gates the
# wheel: a binary carrying bf16, i8mm or SME instructions SIGILLs on the Macs
# the macosx_11_0 tag invites in.
pixi run -e pkg python "$here/packaging/isa_baseline.py" \
    "$PKG/_mojolearn.so" "$PKG/_mojolearn_gbdt.so"

# NO GPU KERNELS, NO WHEEL. A build on a machine without a usable Apple GPU
# emits the host half and silently no Metal shader code, exits 0, and produces
# a wheel that imports and then dies on the first fit. That shipped once, as
# TestPyPI 0.1.0a2. See packaging/macos/check_gpu_embedded.py.
pixi run -e pkg python "$here/packaging/macos/check_gpu_embedded.py" \
    "$PKG/_mojolearn.so" "$PKG/_mojolearn_gbdt.so"

cd "$here/python"
rm -rf dist build ./*.egg-info
pixi run -e pkg python -m build --wheel --no-isolation

echo "wheel:"
ls -la "$here/python/dist"/*.whl
