#!/bin/sh
# Build the DBSCAN / PCA / tSVD / OLS CPython extension into
# python/mojolearn/_mojolearn_estimators.so. Run from anywhere; requires pixi.
#
# FIXED 2026-08-22. THIS SCRIPT SHIPPED A ZERO-KERNEL ARTIFACT FOR HOURS AND
# NOTHING NOTICED, for two independent reasons, both fixed here.
#
# 1. IT SET `MACOSX_DEPLOYMENT_TARGET` IN THE ENVIRONMENT. That suppresses
#    ahead-of-time Metal compilation entirely: `mojo build` writes an empty
#    134-byte metallib per kernel and embeds nothing, so the extension imports
#    cleanly and dies at the first launch with "Failed to create Metal
#    function". Measured on a cold cache, one variable, same file and flags:
#
#        MACOSX_DEPLOYMENT_TARGET=11.0 set       0 AIR blobs
#        MACOSX_DEPLOYMENT_TARGET unset        141 AIR blobs
#
#    This script was the LAST ONE still exporting it, and its artifact was the
#    only one in the tree with zero blobs -- the natural experiment that
#    corroborated the cause. See bindings/build_gbdt.sh for the full write-up,
#    including why the cache made every earlier measurement disagree
#    (`.mojo_cache` is content-addressed and its key does NOT include the
#    deployment target, so one poisoned build serves empty metallibs to every
#    later build whatever ITS flags are).
#
#    The floor is passed to the LINKER instead, which stamps LC_BUILD_VERSION
#    exactly as the environment variable would while the Metal compile step
#    never sees it.
#
# 2. IT HAD NO GATE AT ALL. It built, printed "built ...", and exited 0
#    whatever came out. That is why a kernel-less artifact shipped silently.
#    `bindings/build.sh` learned this lesson twice already -- a blob floor is
#    only a pre-filter, and the real gate is LAUNCHING a kernel. Both are here
#    now, and this script refuses to install an artifact that fails either.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$here"

mkdir -p python/mojolearn

MACOS_FLOOR="11.0"

# NOT EXPORTED, DELIBERATELY -- see (1) above. Unset it if the caller had it
# set, because inheriting it from an outer shell reproduces the bug silently.
unset MACOSX_DEPLOYMENT_TARGET

MACOS_SDK=$(xcrun --sdk macosx --show-sdk-version)
LINK_FLAGS="-Xlinker -platform_version -Xlinker macos -Xlinker $MACOS_FLOOR -Xlinker $MACOS_SDK"

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-est.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT INT TERM
out="$tmpdir/_mojolearn_estimators.so"

# `--target-cpu apple-m1` is the oldest Apple silicon: `mojo build` otherwise
# defaults to whatever chip ran the compiler, and a host-built wheel SIGILLs on
# older Macs at the instruction, inside the extension, with no diagnostic a
# user can act on. arm64 Mach-O `cpusubtype` stays ARM64_ALL whatever -mcpu
# was, so such a wheel LOOKS portable to any header-reading check.
#
# `--target-accelerator metal:1` is kept from the original invocation.
# shellcheck disable=SC2086
pixi run mojo build --emit shared-lib \
    --target-cpu apple-m1 --target-accelerator metal:1 \
    $LINK_FLAGS \
    -I . -I bindings \
    bindings/_mojolearn_estimators.mojo \
    -o "$out"

air_blobs() {
    strings -a "$1" \
        | grep -oE '[0-9A-Za-z_]+_[0-9a-f]{16}air' \
        | sed -e 's/^_gpu_shared_mem//' -e 's/^0//' \
        | sort -u
}

# A FLOOR, NOT A PROOF. The failure this guards against is an artifact with
# zero kernels; the failure run_smoke guards against is an artifact that
# imports and dies at the first launch.
count=$(air_blobs "$out" | wc -l | tr -d ' ')
if [ "$count" -lt 10 ]; then
    printf 'FAILED: %s AIR blobs, want at least 10.\n' "$count" >&2
    printf 'If this is 0, check MACOSX_DEPLOYMENT_TARGET in the environment\n' >&2
    printf 'and then $MODULAR_HOME/cache/.mojo_cache for empty 134-byte\n' >&2
    printf 'metallibs -- one poisoned build serves them to every later one.\n' >&2
    exit 1
fi

got=$(otool -l "$out" | awk '/LC_BUILD_VERSION/{f=1} f && /minos/{print $2; exit}')
if [ "$got" != "$MACOS_FLOOR" ]; then
    printf 'FAILED: minos is %s, want %s.\n' "$got" "$MACOS_FLOOR" >&2
    exit 1
fi

# THE REAL GATE: import and LAUNCH one kernel from each estimator in this
# extension. Every broken build in this bug's history imported fine.
MOJOLEARN_SMOKE_SO="$out" python3 - <<'PY'
import os, shutil, sys, tempfile
tmp = tempfile.mkdtemp()
pkg = os.path.join(tmp, "mojolearn")
shutil.copytree("python/mojolearn", pkg,
                ignore=shutil.ignore_patterns("__pycache__"))
shutil.copyfile(os.environ["MOJOLEARN_SMOKE_SO"],
                os.path.join(pkg, "_mojolearn_estimators.so"))
sys.path.insert(0, tmp)
import numpy as np

# THROUGH THE PYTHON WRAPPERS, NOT THE RAW BINDINGS. The extension's entry
# points take bare `.ctypes.data` addresses plus a packed `params` tuple, so
# a hand-rolled call here would encode that ABI a second time and drift from
# it. `density`, `decomposition` and `linear_model` already encode it and are
# what a user reaches. Note they are NOT exported from `mojolearn/__init__.py`
# (they sit in `_NOT_YET`), so they must be imported as submodules.
from mojolearn import density, decomposition, linear_model

X = np.random.default_rng(0).random((128, 3), dtype=np.float32)
y = np.ascontiguousarray(X[:, 0])

# ONE LAUNCH PER ESTIMATOR IN THIS EXTENSION. Against the zero-kernel
# artifact this script used to produce, every one of these raises
# "Failed to create Metal function" -- which is exactly the failure a build
# with no gate shipped silently.
density.DBSCAN(eps=0.5, min_samples=3).fit(X)
decomposition.PCA(n_components=2).fit(X).transform(X)
decomposition.TruncatedSVD(n_components=2).fit(X).transform(X)
linear_model.LinearRegression().fit(X, y).predict(X)
print("  smoke: dbscan, pca and ols each launched")
PY

mv "$out" python/mojolearn/_mojolearn_estimators.so
echo "built python/mojolearn/_mojolearn_estimators.so ($count AIR blobs, minos $MACOS_FLOOR)"
