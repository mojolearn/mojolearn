#!/bin/sh
# Build the DBSCAN / PCA / tSVD / OLS / Ridge / logistic CPython extension into
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

if [ "$(uname)" = "Darwin" ]; then
    MACOS_SDK=$(xcrun --sdk macosx --show-sdk-version)
else
    MACOS_SDK=""  # linux arm (E1, 2026-08-22): no Mach-O, no Metal SDK
fi
LINK_FLAGS="-Xlinker -platform_version -Xlinker macos -Xlinker $MACOS_FLOOR -Xlinker $MACOS_SDK"
[ "$(uname)" = "Darwin" ] || LINK_FLAGS=""

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
# THE LINUX x86-64 CPU BASELINE. Without it `mojo build` targets WHATEVER
# CHIP THE BUILD BOX HAS, and on 2026-08-30 that shipped a 0.3.0 wheel whose
# HOST code used AVX-512. It died with SIGILL on an L40 whose host was an AMD
# EPYC 7773X, a Zen 3 part with no AVX-512, inside `cluster::estimator::
# kmeans_fit`, on
#
#     vandps 0x6d7ee(%rip){1to4},%xmm2,%xmm2
#
# where `{1to4}` is an EVEX embedded broadcast. There is no `cpuid` dispatch
# anywhere in these binaries, so it is unconditional. That breaks every AMD
# Zen 1, 2 and 3 host, most Intel consumer parts and every Xeon before
# Skylake-SP, which together are a large share of GPU servers. The GPU was
# never involved; the selector had correctly chosen sm_80 for that sm_89
# device.
#
# macOS has pinned `--target-cpu apple-m1` since 0.1.0 AND gates the result
# with `packaging/isa_baseline.py`. Linux had NEITHER. That is the whole bug.
#
# x86-64-v3 is AVX2, FMA, BMI2 and SSE4.2, so Haswell 2013 and Zen 1 2017
# onward, and it EXCLUDES AVX-512. Anything modern enough to hold a supported
# GPU clears it.
#
# aarch64 Linux keeps the empty flags it always had, because an x86 CPU name
# is meaningless there and its host CPU is not the variable that broke.
TARGET_FLAGS="--target-cpu apple-m1 --target-accelerator metal:1"
if [ "$(uname)" != "Darwin" ]; then
    case "$(uname -m)" in
        x86_64) TARGET_FLAGS="--target-cpu ${MOJOLEARN_LINUX_CPU:-x86-64-v3}" ;;
        *)      TARGET_FLAGS="" ;;   # linux arm: host cpu + its GPU
    esac
fi
# MOJOLEARN_GPU_ARCHS: ONE GPU architecture a LINUX set is compiled for
# (sm_80, gfx942, ...). Until 2026-08-30 ONLY bindings/build.sh read this
# variable, so a leg that set it built ONE binding for the asked-for
# architecture and every other binding for the build box's own device (the
# A40 leg's 27-sm_86 read-back, bench/results/wheels/LEGS_2026-08-30.md).
# The compiler takes EXACTLY ONE name (a comma list is rejected);
# packaging/linux/build_sets.sh reads the architecture back out of every
# binary and refuses a set that disagrees with what was asked.
if [ -n "${MOJOLEARN_GPU_ARCHS:-}" ] && [ "$(uname)" != "Darwin" ]; then
    case "$MOJOLEARN_GPU_ARCHS" in *,*)
        echo "MOJOLEARN_GPU_ARCHS='$MOJOLEARN_GPU_ARCHS': the compiler takes EXACTLY ONE architecture (measured 2026-08-30); run one build per architecture" >&2
        exit 2 ;;
    esac
    TARGET_FLAGS="$TARGET_FLAGS --target-accelerator $MOJOLEARN_GPU_ARCHS"
    echo "!! MOJOLEARN_GPU_ARCHS=$MOJOLEARN_GPU_ARCHS (--target-accelerator); read the architecture back out of the built .so"
fi
# Explicit kernel-matrix column: MOJOLEARN_TARGET_COLUMN=apple|nvidia|amd|amd_rdna
COLUMN_DEFINE=""
# THE NUMERIC MODE IS A BUILD DEFINE, NOT A SOURCE FLIP (2026-08-23).
# MOJOLEARN_NUMERIC_MODE=identical compiles with -D MOJOLEARN_NUMERIC_IDENTICAL=1
# (checks/numerics.mojo reads it through is_defined, the same shape as
# the column define) and lands the binary under python/mojolearn/identical/,
# where python/mojolearn/_backend.py picks it up when the env var
# MOJOLEARN_NUMERIC_MODE=identical is set at import. Default is fast and the
# default location. The build-time smoke gates import the FAST package, so
# they are skipped for an identical build; tools/e2_matrix_fit.py is that
# build's gate.
MODE_DEFINE=""
OUTDIR="python/mojolearn"
if [ "${MOJOLEARN_NUMERIC_MODE:-fast}" = "identical" ]; then
    MODE_DEFINE="-D MOJOLEARN_NUMERIC_IDENTICAL=1"
    OUTDIR="python/mojolearn/identical"
    mkdir -p "$OUTDIR"
    export MOJOLEARN_SKIP_BUILD_GATE=1
elif [ "${MOJOLEARN_NUMERIC_MODE:-fast}" = "deterministic" ]; then
    # The MIDDLE tier: reproducible run to run on ONE device, with no
    # promise about a second one. It gets its own directory because it
    # is its own binary -- PIN_DETERMINISM is comptime, so a
    # deterministic build is different code from both neighbours, not
    # the identical build with a flag turned down.
    MODE_DEFINE="-D MOJOLEARN_NUMERIC_DETERMINISTIC=1"
    OUTDIR="python/mojolearn/deterministic"
    mkdir -p "$OUTDIR"
    export MOJOLEARN_SKIP_BUILD_GATE=1
elif [ "${MOJOLEARN_NUMERIC_MODE:-fast}" != "fast" ]; then
    echo "MOJOLEARN_NUMERIC_MODE must be fast, deterministic or identical, got '$MOJOLEARN_NUMERIC_MODE'" >&2
    exit 2
fi
if [ -n "${MOJOLEARN_TARGET_COLUMN:-}" ]; then
    COLUMN_DEFINE="-D MOJOLEARN_COLUMN_$(printf %s "$MOJOLEARN_TARGET_COLUMN" | tr '[:lower:]' '[:upper:]')"
fi
# shellcheck disable=SC2086
pixi run mojo build -j "${MOJOLEARN_COMPILE_JOBS:-2}" --emit shared-lib \
    $TARGET_FLAGS $COLUMN_DEFINE $MODE_DEFINE \
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

# The gates below import the WHOLE python package, so during a
# from-scratch multi-binding build (a fresh linux box, E1) they fail on
# the siblings' not-yet-built .so files; and the AIR/otool checks are
# Mach-O-only. The caller that sets this owns end-to-end verification
# (the E1 bootstrap runs the traced-fit driver after all five builds).
if [ -n "${MOJOLEARN_SKIP_BUILD_GATE:-}" ] || [ "$(uname)" != "Darwin" ]; then
    mv "$out" "$OUTDIR/_mojolearn_estimators.so"
    echo "built $OUTDIR/_mojolearn_estimators.so (gate skipped: non-Darwin or MOJOLEARN_SKIP_BUILD_GATE)"
    exit 0
fi

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
# what a user reaches (exported from `mojolearn/__init__.py` since
# 2026-08-23; imported as submodules here so this gate does not depend on
# the package's export list).
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
linear_model.Ridge(alpha=1.0).fit(X, y).predict(X)
yb = (X[:, 0] > 0.5).astype(np.int32)
linear_model.LogisticRegression(max_iter=5).fit(X, yb).predict_proba(X)
print("  smoke: dbscan, pca, ols, ridge and logistic each launched")
PY

mv "$out" "$OUTDIR/_mojolearn_estimators.so"
echo "built $OUTDIR/_mojolearn_estimators.so ($count AIR blobs, minos $MACOS_FLOOR)"
