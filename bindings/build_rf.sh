#!/bin/sh
# Build the RandomForest CPython extension into
# python/mojolearn/_mojolearn_rf.so. Run from anywhere; requires pixi.
#
# This is bindings/build_trees.sh's pattern applied to the ensemble/ engine,
# and the pattern is load-bearing -- read build_gbdt.sh's header before
# changing anything. The short form: MACOSX_DEPLOYMENT_TARGET set in the
# environment SUPPRESSES Metal AOT (empty metallibs, import succeeds, first
# launch dies); the compiler cache does not key on it, so one poisoned build
# serves every later one; the macOS floor goes to the LINKER; the CPU target
# pins to apple-m1; blob counting is a pre-filter and run_smoke is the gate.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$here"

mkdir -p python/mojolearn

MACOS_FLOOR="11.0"
unset MACOSX_DEPLOYMENT_TARGET

MACOS_SDK=$(xcrun --sdk macosx --show-sdk-version)
LINK_FLAGS="-Xlinker -platform_version -Xlinker macos -Xlinker $MACOS_FLOOR -Xlinker $MACOS_SDK"
TARGET_FLAGS="--target-cpu apple-m1"

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-rf-build.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT INT TERM
out="$tmpdir/_mojolearn_rf.so"

# shellcheck disable=SC2086  # both flag strings are deliberately word-split
pixi run mojo build --emit shared-lib \
    $TARGET_FLAGS \
    $LINK_FLAGS \
    -I . -I bindings \
    bindings/_mojolearn_rf.mojo \
    -o "$out"

air_blobs() {
    strings -a "$1" \
        | grep -oE '[0-9A-Za-z_]+_[0-9a-f]{16}air' \
        | sed -e 's/^_gpu_shared_mem//' -e 's/^0//' \
        | sort -u
}

# The floor is measured on THIS module: the first correct artifact built on
# 2026-08-22 carries ensemble-prefixed blobs for both objective
# instantiations of the builder's kernel set. A suppressed build carries 0.
# The floor is 8: below the measured count, far above any suppressed build;
# run_smoke below is the gate that can say WHICH kernels are missing.
kernels_plausible() {
    _n=$(air_blobs "$1" | grep -c '^ensemble' || true)
    if [ "$_n" -lt 8 ]; then
        printf '  ensemble: %s AIR blobs, want at least 8\n' "$_n" >&2
        return 1
    fi
    return 0
}

minos_matches() {
    _got=$(otool -l "$1" \
        | awk '/LC_BUILD_VERSION/{f=1} f && /minos/{print $2; exit}')
    if [ "$_got" != "$MACOS_FLOOR" ]; then
        printf '  minos is %s, want %s\n' "$_got" "$MACOS_FLOOR" >&2
        return 1
    fi
    return 0
}

# THE REAL GATE: import the extension and LAUNCH kernels through both fits.
# Every broken build in this family's history imported fine and died at the
# first launch. Kept tiny -- the GPU is shared.
run_smoke() {
    MOJOLEARN_SMOKE_SO="$1" python3 - <<'PY'
import os, shutil, sys, tempfile
tmp = tempfile.mkdtemp()
pkg = os.path.join(tmp, "mojolearn")
shutil.copytree("python/mojolearn", pkg,
                ignore=shutil.ignore_patterns("__pycache__"))
shutil.copyfile(os.environ["MOJOLEARN_SMOKE_SO"],
                os.path.join(pkg, "_mojolearn_rf.so"))
sys.path.insert(0, tmp)
import numpy as np
import mojolearn.randomforest as rfm

rng = np.random.default_rng(0)
X = rng.random((512, 6), dtype=np.float32)
yc = (X[:, 0] > 0.5).astype(np.int64)
yr = X[:, 1].astype(np.float32)

kw = dict(n_estimators=5, max_depth=8, random_state=7)

# CLASSIFIER on a separable fixture, and reproducibility from the seed.
c1 = rfm.RandomForestClassifier(**kw).fit(X, yc)
if (c1.predict(X) == yc).mean() < 0.9:
    raise SystemExit("smoke: classifier failed a separable fixture")
p = c1.predict_proba(X)
if p.shape != (512, 2) or not np.allclose(p.sum(axis=1), 1.0, atol=1e-5):
    raise SystemExit("smoke: predict_proba rows do not sum to 1")
c2 = rfm.RandomForestClassifier(**kw).fit(X, yc)
if not np.array_equal(c1.predict_proba(X), c2.predict_proba(X)):
    raise SystemExit("smoke: same seed, different forest")

# THE BOOTSTRAP MUST DO SOMETHING: same seed, bootstrap on vs off, the
# forests must differ -- this extension exists to be the row sampler's
# first caller, and a sampler that changes nothing is not being reached.
c3 = rfm.RandomForestClassifier(bootstrap=False, **kw).fit(X, yc)
if np.array_equal(c1.predict_proba(X), c3.predict_proba(X)):
    raise SystemExit("smoke: bootstrap=True and False produced identical"
                     " forests -- the row sampler is not being reached")

# REGRESSOR fits a copied column.
r1 = rfm.RandomForestRegressor(**kw).fit(X, yr)
if np.abs(r1.predict(X) - yr).mean() > 0.1:
    raise SystemExit("smoke: regressor failed to fit a copied column")

# Refusals must fire by name.
for bad in (dict(oob_score=True), dict(warm_start=True),
            dict(ccp_alpha=0.1), dict(n_jobs=2)):
    try:
        rfm.RandomForestClassifier(**bad).fit(X, yc)
    except Exception as e:
        if list(bad)[0] not in str(e):
            raise SystemExit(f"smoke: refusal for {bad} does not name it")
    else:
        raise SystemExit(f"smoke: {bad} was accepted")
try:
    rfm.RandomForestClassifier(criterion="entropy")
except NotImplementedError:
    pass
else:
    raise SystemExit("smoke: criterion='entropy' was accepted")
shutil.rmtree(tmp, ignore_errors=True)
PY
}

if ! kernels_plausible "$out" || ! minos_matches "$out" || ! run_smoke "$out"; then
    printf '%s\n' \
      "" \
      "FAILED: the RandomForest extension did not come out complete." \
      "THE FIRST THING TO SUSPECT IS THE COMPILER CACHE, not this source:" \
      "\$MODULAR_HOME/cache/.mojo_cache keys ignore the deployment target," \
      "so one build made with MACOSX_DEPLOYMENT_TARGET set poisons it with" \
      "empty 134-byte metallibs. See bindings/build_gbdt.sh's header." >&2
    exit 1
fi

count=$(air_blobs "$out" | wc -l | tr -d ' ')
mv "$out" python/mojolearn/_mojolearn_rf.so
echo "built python/mojolearn/_mojolearn_rf.so ($count AIR blobs, minos $MACOS_FLOOR)"
