#!/bin/sh
# Build the macOS arm64 wheel: ten extensions per numeric tier, staged MAX
# runtime, re-signed, packed.
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
STAMP=$(mktemp "${TMPDIR:-/tmp}/mojolearn-release-stamp.XXXXXX")
trap 'rm -f "$STAMP"' EXIT INT TERM

# Refreshed every build. These are COPIES of the repository root's files, and
# a stale copy in a published wheel is a wrong LICENSE or a wrong README on
# PyPI. python/.gitignore keeps them out of the checkout.
cp "$here/LICENSE" "$here/NOTICE" "$here/README.md" "$here/python/"
# CITATION.cff goes INSIDE the package, not beside pyproject.toml: it is
# shipped as package data so `pip install` carries the machine-readable
# citation. Attribution that lives only in the git repository does not
# travel with the artifact, and the third way people acquire this library
# is neither a clone nor a pip install but a copied file.
cp "$here/CITATION.cff" "$here/python/mojolearn/"

# EVERY EXTENSION IN THE WHEEL IS BUILT HERE. `pyproject.toml`'s
# package-data globs `*.so`, so an extension that is NOT built here is not
# absent from the wheel -- it is shipped STALE, whatever happens to be sitting
# in the working tree from an earlier build. That is how `_mojolearn.so` came
# to ship an artifact predating `eval_x`/`eval_y`, where every `fit` raised
# "takes 6 positional arguments but 8 were given", and how
# `_mojolearn_estimators.so` shipped as a ZERO-KERNEL artifact on 2026-08-22
# while this file said "every extension" and built two of three.
#
# AS OF 2026-08-24 THE LIST IS TEN EXTENSIONS IN TWO NUMERIC MODES. The
# FAST set lands at python/mojolearn/*.so and the IDENTICAL set at
# python/mojolearn/identical/*.so; python/mojolearn/_backend.py loads the
# identical set when MOJOLEARN_NUMERIC_MODE=identical is set at import
# (original/numerics.mojo reads the build define). Both sets ship in ONE
# wheel. The list below is THE list: bindings/build_*.sh that is not named
# here does not ship, and a name here with no script fails the build.
# 2026-08-24: five bindings were added at once (svm/isolation-forest,
# solver/hierarchy, metrics/spectral, holtwinters/tsa, and the linalg GEMM
# surface). They are listed here because a build script that is not named
# here does not ship, and python/mojolearn/_backend.py now knows all ten.
#
# DELIBERATELY NOT MIRRORED IN tools/e1_bootstrap.sh, AND THAT ASYMMETRY IS
# THE POINT. That script builds bindings on a RENTED GPU under a work bound.
# Phase 8 drives every lane gate through `mojo run` on the lane's own driver,
# not through a Python binding, so the five new extensions buy a leg nothing
# and would roughly double its binding-build time. A leg that spends its
# lease compiling and comes home with an empty lanes/ has bought nothing at
# all. Add a binding there only when a phase actually imports it.
BUILD_SCRIPTS="build.sh build_gbdt.sh build_estimators.sh build_rf.sh build_trees.sh build_svm.sh build_solver.sh build_metrics.sh build_tsa.sh build_linalg.sh"
EXT_NAMES="_mojolearn _mojolearn_gbdt _mojolearn_estimators _mojolearn_rf _mojolearn_trees _mojolearn_svm _mojolearn_solver _mojolearn_metrics _mojolearn_tsa _mojolearn_linalg"

# THE PER-SCRIPT GATES ARE OFF HERE, AND THE REASON IS A CLEAN CHECKOUT.
# Each bindings/build_*.sh ends by copying python/mojolearn/ aside and
# importing the package to fit on the binary it just built. The package
# __init__ imports EVERY binding, so that gate needs all five extensions to
# exist already; in the shared working tree they always did, and in a clean
# checkout of a tag (the only honest place to build a release from) the
# first gate fails on the fourth missing .so before anything runs. Found
# 2026-08-23 on the first clean-tree build. So this script builds all ten
# binaries gate-off and then runs THE release gate, verify_wheel.sh, which
# installs the finished wheel into a clean venv under every claimed
# interpreter and fits every estimator family in EVERY SHIPPED numeric mode
# (three since 2026-08-29; this line said BOTH when there were two). That
# is strictly more than the per-script gates check, and it runs on the
# artifact that ships rather than on a copy of the tree.
# THE TIERS THIS WHEEL CARRIES. `fast` lives at python/mojolearn/*.so and
# every other tier one directory down under its own name, which is the layout
# python/mojolearn/_backend.py loads from.
#
# THE DEFAULT IS THREE TIERS, 2026-08-29. It read "TWO TIERS AND NOT THREE"
# until the determinism lane closed the same day; that sentence is deleted
# rather than softened, because a stale default here silently ships a wheel
# whose deterministic tier raises from a missing-binary stub.
#
# WHAT CLOSED THE LANE, and it is not a pin count. A tier is a PROMISE, and
# the promise is measured by tools/repeat_run_stability.py, which runs one
# fit repeatedly in one process and compares RAW OUTPUT BYTES. Taken at one
# commit on all three vendors on 2026-08-29:
#
#     column                 fast                     deterministic
#     Apple M4 (Metal)       MOVED in 8 of 10 tries   STABLE 10/10
#     NVIDIA RTX 4090 (CUDA) MOVED, 24 answers in 24  STABLE, 1 in 12
#     AMD MI325X (HIP)       MOVED, 6 answers in 24   STABLE, 1 in 12
#
# bench/results/stability/RESULTS.md is the record. The pin side is 15 files
# keyed to PIN_DETERMINISM; the determinism class is small because this tree
# uses no float atomicAdd anywhere, which is why the middle tier is cheap.
#
# THE MIDDLE TIER IS NOT THE TOP ONE WEARING A HAT. Its hashes DIFFER across
# vendors on 10 of 12 comparable lanes, gemm-vendor among them -- it keeps
# the vendor matmul and its speed and buys no cross-vendor identity.
#
# To cut a two-tier wheel anyway (a hotfix, a bisect), name the tiers:
#     MOJOLEARN_RELEASE_MODES="fast identical" ./packaging/macos/build_release_wheel.sh
# and keep verify_wheel.sh's copy of the variable set the same for one release
# or the verifier fails the wheel for lacking a tier it was never asked to
# build. pyproject.toml's package-data glob carries all three directories.
MODES="${MOJOLEARN_RELEASE_MODES:-fast deterministic identical}"
echo "== numeric tiers in this wheel: $MODES"

for mode in $MODES; do
    for script in $BUILD_SCRIPTS; do
        echo "== $script ($mode)"
        MOJOLEARN_NUMERIC_MODE=$mode MOJOLEARN_SKIP_BUILD_GATE=1 ./bindings/$script
    done
done

# THE FILES THE REST OF THIS SCRIPT GATES, ten per tier. Built above or absent, never
# stale: every one is checked for existence and for being newer than this
# script's start, so a build script that silently left the old file in place
# fails here instead of shipping.
# One entry per extension per tier. `fast` is the package directory itself and
# every other tier is a subdirectory of the same name, so this loop does not
# need to know which tiers exist, only what MODES says.
ALL_SOS=""
for n in $EXT_NAMES; do
    for mode in $MODES; do
        if [ "$mode" = "fast" ]; then
            ALL_SOS="$ALL_SOS $PKG/$n.so"
        else
            ALL_SOS="$ALL_SOS $PKG/$mode/$n.so"
        fi
    done
done
for so in $ALL_SOS; do
    [ -f "$so" ] || { echo "ERROR: $so was not produced" >&2; exit 1; }
    [ "$so" -nt "$STAMP" ] || { echo "ERROR: $so predates this build (stale)" >&2; exit 1; }
done

# The FULL transitive closure, walked rather than sampled. See
# packaging/macos/stage_dylibs.py: reading only the extension's direct
# dependencies staged 2 dylibs when the real closure is 4, and shipped a wheel
# that failed on install with "Library not loaded:
# @rpath/libMSupportGlobals.dylib, referenced from .dylibs/libAsyncRT...".
# It also verifies statically that nothing is left unresolved, which is the
# only form of this check that means anything on the build machine.
# EVERY EXTENSION OF EVERY TIER IN ONE CALL, because there is one `.dylibs`
# and the
# script wipes it before staging. Separate calls would leave earlier closures
# deleted -- invisibly, because on THIS machine the original rpath still
# resolves into the pixi environment. The identical/ set sits one directory
# down and gets @loader_path/../.dylibs; the stager computes that per file.
# shellcheck disable=SC2086
pixi run -e pkg python "$here/packaging/macos/stage_dylibs.py" \
    $ALL_SOS "$ENV_LIB"



# THE TAG AND THE BINARY MUST AGREE, and nothing else checks this.
# The wheel filename is what pip compares before it tries to load anything, so
# a tag above the binary's floor turns installable Macs away and a tag below it
# installs onto Macs where the extension cannot load. Both are silent on the
# machine that built the wheel.
# CHECKED FOR EVERY EXTENSION, not just the first: one wheel carries one tag,
# and the tag is only honest if it is the floor of EVERYTHING inside.
TAG_MINOS=$(grep -E '^DEFAULT_MACOS_TARGET' "$here/python/setup.py" | sed 's/[^0-9.]//g')
for so in $ALL_SOS; do
    BIN_MINOS=$(otool -l "$so" | awk '/LC_BUILD_VERSION/{f=1} f&&/minos/{print $2; exit}')
    if [ "$BIN_MINOS" != "$TAG_MINOS" ]; then
        echo "ERROR: $(basename "$so") minos $BIN_MINOS but setup.py tags $TAG_MINOS" >&2
        echo "       MACOS_FLOOR in bindings/build.sh and bindings/build_gbdt.sh" >&2
        echo "       and DEFAULT_MACOS_TARGET in python/setup.py must all match" >&2
        exit 1
    fi
    echo "macOS floor: ${so#$PKG/} minos $BIN_MINOS == wheel tag $TAG_MINOS"
done

# THE ISA BASELINE, WHICH NO HEADER CAN SEE. arm64 Mach-O cpusubtype stays
# ARM64_ALL whatever --target-cpu was, so this has to disassemble. Gates the
# wheel: a binary carrying bf16, i8mm or SME instructions SIGILLs on the Macs
# the macosx_11_0 tag invites in.
# shellcheck disable=SC2086
pixi run -e pkg python "$here/packaging/isa_baseline.py" $ALL_SOS

# NO GPU KERNELS, NO WHEEL. A build on a machine without a usable Apple GPU
# emits the host half and silently no Metal shader code, exits 0, and produces
# a wheel that imports and then dies on the first fit. That shipped once, as
# TestPyPI 0.1.0a2. See packaging/macos/check_gpu_embedded.py.
# shellcheck disable=SC2086
pixi run -e pkg python "$here/packaging/macos/check_gpu_embedded.py" $ALL_SOS

cd "$here/python"
rm -rf dist build ./*.egg-info
pixi run -e pkg python -m build --wheel --no-isolation

echo "wheel:"
ls -la "$here/python/dist"/*.whl

# THE GATE. Not optional and not a separate step you may forget: a wheel
# that this script produced and that has not passed verify_wheel.sh is a
# wheel that imported on the build machine and nothing else, and that shape
# of artifact has shipped broken twice (TestPyPI 0.1.0a1, 0.1.0a2).
cd "$here"
./packaging/macos/verify_wheel.sh
