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

if [ "$(uname)" = "Darwin" ]; then
    MACOS_SDK=$(xcrun --sdk macosx --show-sdk-version)
else
    MACOS_SDK=""  # linux arm (E1, 2026-08-22): no Mach-O, no Metal SDK
fi
LINK_FLAGS="-Xlinker -platform_version -Xlinker macos -Xlinker $MACOS_FLOOR -Xlinker $MACOS_SDK"
[ "$(uname)" = "Darwin" ] || LINK_FLAGS=""
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
TARGET_FLAGS="--target-cpu apple-m1"
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

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-rf-build.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT INT TERM
out="$tmpdir/_mojolearn_rf.so"

# shellcheck disable=SC2086  # both flag strings are deliberately word-split
pixi run mojo build --emit shared-lib \
    $TARGET_FLAGS $COLUMN_DEFINE $MODE_DEFINE \
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
    [ "$(uname)" = "Darwin" ] || return 0  # linux gate is run_smoke
    _n=$(air_blobs "$1" | grep -c '^ensemble' || true)
    if [ "$_n" -lt 8 ]; then
        printf '  ensemble: %s AIR blobs, want at least 8\n' "$_n" >&2
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
# DEVIATION 407: THE CRITERION SELECTOR MUST REACH THE ENGINE. On a
# fixture with real structure gini and entropy choose different splits
# (criteria_check arm A); if the two forests are bit-equal the selector is
# a dead argument and the classifier is fitting gini under both names.
# Same kw, same seed, bootstrap off so only the criterion moves.
kwr = dict(n_estimators=3, max_depth=6, max_features=1.0, n_bins=16,
           bootstrap=False, random_state=2024)
yc3 = ((X[:, 0] * 7 + X[:, 1] * 13) % 3).astype(np.int64)
g = rfm.RandomForestClassifier(criterion="gini", **kwr).fit(X, yc3)
e = rfm.RandomForestClassifier(criterion="entropy", **kwr).fit(X, yc3)
if np.array_equal(g._quesval, e._quesval) and np.array_equal(
        g._colid, e._colid):
    raise SystemExit("smoke: gini and entropy fitted the IDENTICAL forest"
                     " -- the criterion is not reaching the engine")
l = rfm.RandomForestClassifier(criterion="log_loss", **kwr).fit(X, yc3)
if not (np.array_equal(l._quesval, e._quesval)
        and np.array_equal(l._colid, e._colid)):
    raise SystemExit("smoke: log_loss and entropy must be one criterion")
yp = X[:, 1].astype(np.float32) + 1.0
seen = {}
for crit in ("squared_error", "poisson", "gamma", "inverse_gaussian"):
    r = rfm.RandomForestRegressor(criterion=crit, **kwr).fit(X, yp)
    seen[crit] = (r._colid.tobytes(), r._quesval.tobytes())
if len(set(seen.values())) < 2:
    raise SystemExit("smoke: every regression criterion fitted the same"
                     " forest -- the criterion is not reaching the engine")
for bad in ("friedman_mse", "absolute_error"):
    try:
        rfm.RandomForestRegressor(criterion=bad)
    except NotImplementedError as ex:
        if bad not in str(ex):
            raise SystemExit(f"smoke: refusal for criterion={bad} does not"
                             " name it")
    else:
        raise SystemExit(f"smoke: criterion={bad!r} was accepted")
try:
    rfm.RandomForestRegressor(criterion="gamma", **kwr).fit(X, -yp)
except ValueError:
    pass
else:
    raise SystemExit("smoke: gamma accepted a non-positive target")
shutil.rmtree(tmp, ignore_errors=True)
PY
}

# The per-build gate imports the WHOLE python package, so during a
# from-scratch multi-binding build (a fresh linux box, E1) the first
# gates fail on the siblings' not-yet-built .so files. The caller that
# sets this owns end-to-end verification (the E1 bootstrap runs the
# traced-fit driver, which launches kernels through every lib).
if [ -n "${MOJOLEARN_SKIP_BUILD_GATE:-}" ]; then
    mv "$out" "$OUTDIR/_mojolearn_rf.so"
    echo "built $OUTDIR/_mojolearn_rf.so (gate SKIPPED by MOJOLEARN_SKIP_BUILD_GATE)"
    exit 0
fi

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
mv "$out" "$OUTDIR/_mojolearn_rf.so"
echo "built $OUTDIR/_mojolearn_rf.so ($count AIR blobs, minos $MACOS_FLOOR)"
