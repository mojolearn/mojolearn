#!/bin/sh
# Build the ExtraTrees CPython extension into
# python/mojolearn/_mojolearn_trees.so. Run from anywhere; requires pixi.
#
# This script is bindings/build_gbdt.sh's pattern applied to a new module,
# and the pattern is load-bearing -- read that file's header before changing
# anything here. The short form:
#
# 1. MACOSX_DEPLOYMENT_TARGET must NOT be in the environment. Set, it
#    suppresses ahead-of-time Metal compilation entirely: every kernel gets
#    an empty 134-byte metallib, the extension imports cleanly and dies at
#    the first launch with "Failed to create Metal function". The macOS
#    floor goes to the LINKER instead, which stamps LC_BUILD_VERSION without
#    the Metal compile step ever seeing it.
# 2. The compiler cache's key does NOT include the deployment target, so one
#    poisoned build serves empty metallibs to every later build. If the gate
#    below fails, suspect the cache first (the failure message says how).
# 3. The CPU target is pinned to apple-m1, the oldest Apple silicon, because
#    the host default (+sme on an M4) SIGILLs on older chips with no
#    diagnostic. No --target-accelerator flag, ever: passing it at all gives
#    0 AIR blobs (measured 2026-08-21).
# 4. Blob counting is a PRE-FILTER; the GATE is launching kernels through
#    the real wrapper (run_smoke below).
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$here"

mkdir -p python/mojolearn

MACOS_FLOOR="11.0"
unset MACOSX_DEPLOYMENT_TARGET

if [ "$(uname)" = "Darwin" ]; then
    MACOS_SDK=$(xcrun --sdk macosx --show-sdk-version)
else
    MACOS_SDK=""  # linux arm (E1, 2026-08-22): no Mach-O, no Metal SDK
fi
LINK_FLAGS="-Xlinker -platform_version -Xlinker macos -Xlinker $MACOS_FLOOR -Xlinker $MACOS_SDK"
[ "$(uname)" = "Darwin" ] || LINK_FLAGS=""
TARGET_FLAGS="--target-cpu apple-m1"
[ "$(uname)" = "Darwin" ] || TARGET_FLAGS=""
# Explicit kernel-matrix column: MOJOLEARN_TARGET_COLUMN=apple|nvidia|amd|amd_rdna
# becomes a -D define that overrides TARGET_COLUMN autodetection.
COLUMN_DEFINE=""
# THE NUMERIC MODE IS A BUILD DEFINE, NOT A SOURCE FLIP (2026-08-23).
# MOJOLEARN_NUMERIC_MODE=identical compiles with -D MOJOLEARN_NUMERIC_IDENTICAL=1
# (mojo_only/numerics.mojo reads it through is_defined, the same shape as
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
elif [ "${MOJOLEARN_NUMERIC_MODE:-fast}" != "fast" ]; then
    echo "MOJOLEARN_NUMERIC_MODE must be fast or identical, got '$MOJOLEARN_NUMERIC_MODE'" >&2
    exit 2
fi
if [ -n "${MOJOLEARN_TARGET_COLUMN:-}" ]; then
    COLUMN_DEFINE="-D MOJOLEARN_COLUMN_$(printf %s "$MOJOLEARN_TARGET_COLUMN" | tr '[:lower:]' '[:upper:]')"
fi

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-trees-build.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT INT TERM
out="$tmpdir/_mojolearn_trees.so"

# shellcheck disable=SC2086  # both flag strings are deliberately word-split
pixi run mojo build --emit shared-lib \
    $TARGET_FLAGS $COLUMN_DEFINE $MODE_DEFINE \
    $LINK_FLAGS \
    -I . -I bindings \
    bindings/_mojolearn_trees.mojo \
    -o "$out"

air_blobs() {
    strings -a "$1" \
        | grep -oE '[0-9A-Za-z_]+_[0-9a-f]{16}air' \
        | sed -e 's/^_gpu_shared_mem//' -e 's/^0//' \
        | sort -u
}

# The floor is measured on THIS module: a correct artifact built on
# 2026-08-21 carries exactly 11 blobs, every one prefixed
# `extratrees_ported_decisiontree` -- both objective instantiations of the
# builder's kernel set, deduplicated by content hash. A suppressed build
# carries 0. The floor is 10: below the measured count, far above any
# suppressed build, and run_smoke below is the gate that can say WHICH
# kernels are missing.
kernels_plausible() {
    [ "$(uname)" = "Darwin" ] || return 0  # linux gate is run_smoke
    _n=$(air_blobs "$1" | grep -c '^extratrees' || true)
    if [ "$_n" -lt 10 ]; then
        printf '  extratrees: %s AIR blobs, want at least 10\n' "$_n" >&2
        return 1
    fi
    return 0
}

minos_matches() {
    [ "$(uname)" = "Darwin" ] || return 0  # linux gate is run_smoke
    _got=$(otool -l "$1" \
        | awk '/LC_BUILD_VERSION/{f=1} f && /minos/{print $2; exit}')
    if [ "$_got" != "$MACOS_FLOOR" ]; then
        printf '  minos is %s, want %s\n' "$_got" "$MACOS_FLOOR" >&2
        return 1
    fi
    return 0
}

# THE REAL GATE: import the extension and LAUNCH kernels through both device
# fits. Every broken build in this family's history imported fine and died at
# the first launch. Kept tiny -- 512x6, three trees -- the GPU is shared.
run_smoke() {
    MOJOLEARN_SMOKE_SO="$1" python3 - <<'PY'
import os, shutil, sys, tempfile
tmp = tempfile.mkdtemp()
pkg = os.path.join(tmp, "mojolearn")
shutil.copytree("python/mojolearn", pkg,
                ignore=shutil.ignore_patterns("__pycache__"))
shutil.copyfile(os.environ["MOJOLEARN_SMOKE_SO"],
                os.path.join(pkg, "_mojolearn_trees.so"))
sys.path.insert(0, tmp)
import numpy as np
import mojolearn.extratrees as et

rng = np.random.default_rng(0)
X = rng.random((512, 6), dtype=np.float32)
yc = (X[:, 0] > 0.5).astype(np.int64)
yr = X[:, 1].astype(np.float32)

# CLASSIFIER, gpu arm vs cpu arm: the lane's device_forest_check proves the
# two forests bit-identical, so the wrapper must agree everywhere.
kw = dict(n_estimators=3, max_depth=6, random_state=7)
pg = et.ExtraTreesClassifier(device="gpu", **kw).fit(X, yc)
pc = et.ExtraTreesClassifier(device="cpu", **kw).fit(X, yc)
if not np.array_equal(pg.predict(X), pc.predict(X)):
    raise SystemExit("smoke: classifier gpu and cpu arms disagree")
if not np.array_equal(pg.predict_proba(X), pc.predict_proba(X)):
    raise SystemExit("smoke: classifier proba gpu/cpu arms disagree")
if (pg.predict(X) == yc).mean() < 0.9:
    raise SystemExit("smoke: classifier failed a separable fixture")

# REGRESSOR, gpu arm: structure-identical to cpu, leaves within one
# quantization step (deviation 135) -- at this scale, prediction agreement
# far tighter than any real signal.
rg = et.ExtraTreesRegressor(device="gpu", **kw).fit(X, yr)
rc = et.ExtraTreesRegressor(device="cpu", **kw).fit(X, yr)
d = np.abs(rg.predict(X) - rc.predict(X)).max()
if d > 1e-4:
    raise SystemExit(f"smoke: regressor gpu/cpu arms {d} apart")
if np.abs(rg.predict(X) - yr).mean() > 0.1:
    raise SystemExit("smoke: regressor failed to fit a copied column")
# The arms must NOT be bit-equal: the gpu arm's leaves are quantized means
# (the estimator's reach proof, deviation 188). Bit-equality here means the
# device arm silently served the host fit.
if d == 0.0:
    raise SystemExit("smoke: regressor arms bit-equal -- device arm did not"
                     " quantize, so it did not run on the device")

# Refusals must cross the boundary by name.
for bad in (dict(bootstrap=True), dict(warm_start=True),
            dict(ccp_alpha=0.1)):
    try:
        et.ExtraTreesClassifier(**bad).fit(X, yc)
    except Exception as e:
        if list(bad)[0] not in str(e):
            raise SystemExit(f"smoke: refusal for {bad} does not name it")
    else:
        raise SystemExit(f"smoke: {bad} was accepted")
try:
    et.ExtraTreesClassifier(criterion="entropy")
except NotImplementedError:
    pass
else:
    raise SystemExit("smoke: criterion='entropy' was accepted")
shutil.rmtree(tmp, ignore_errors=True)
PY
}

# The per-build gate imports the WHOLE python package, so during a
# from-scratch multi-binding build (a fresh linux box, E1) the first
# gates fail on the siblings' not-yet-built .so files. The caller that
# sets this owns end-to-end verification (the E1 bootstrap runs the
# traced-fit driver, which launches kernels through every lib).
if [ -n "${MOJOLEARN_SKIP_BUILD_GATE:-}" ]; then
    mv "$out" "$OUTDIR/_mojolearn_trees.so"
    echo "built $OUTDIR/_mojolearn_trees.so (gate SKIPPED by MOJOLEARN_SKIP_BUILD_GATE)"
    exit 0
fi

if ! kernels_plausible "$out" || ! minos_matches "$out" || ! run_smoke "$out"; then
    printf '%s\n' \
      "" \
      "FAILED: the ExtraTrees extension did not come out complete." \
      "THE FIRST THING TO SUSPECT IS THE COMPILER CACHE, not this source:" \
      "\$MODULAR_HOME/cache/.mojo_cache keys ignore the deployment target," \
      "so one build made with MACOSX_DEPLOYMENT_TARGET set poisons it with" \
      "empty 134-byte metallibs. See bindings/build_gbdt.sh's header for" \
      "the find command and the full write-up." >&2
    exit 1
fi

count=$(air_blobs "$out" | wc -l | tr -d ' ')
mv "$out" "$OUTDIR/_mojolearn_trees.so"
echo "built $OUTDIR/_mojolearn_trees.so ($count AIR blobs, minos $MACOS_FLOOR)"
