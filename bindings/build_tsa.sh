#!/bin/sh
# Build the time-series CPython extension (Holt-Winters + the KPSS test) into
# python/mojolearn/_mojolearn_tsa.so. Run from anywhere; requires pixi.
#
# NEW 2026-08-24 (DEVIATIONS 900-909). Structured after
# bindings/build_estimators.sh, which is the script that learned the two
# lessons below the hard way. They are restated rather than referenced,
# because the failure they describe is silent and someone editing THIS file
# needs to see it here.
#
# 1. DO NOT SET `MACOSX_DEPLOYMENT_TARGET` IN THE ENVIRONMENT. It suppresses
#    ahead-of-time Metal compilation entirely: `mojo build` writes an empty
#    134-byte metallib per kernel and embeds nothing, so the extension imports
#    cleanly and dies at the first launch with "Failed to create Metal
#    function". Measured by build_estimators.sh on a cold cache, one variable,
#    same file and flags:
#
#        MACOSX_DEPLOYMENT_TARGET=11.0 set       0 AIR blobs
#        MACOSX_DEPLOYMENT_TARGET unset        141 AIR blobs
#
#    The compiler cache is what made every earlier measurement of this
#    disagree: `.mojo_cache` is content-addressed and its key does NOT
#    include the deployment target, so ONE poisoned build serves empty
#    metallibs to every later build whatever ITS flags are. If this script
#    ever reports 0 blobs, clear $MODULAR_HOME/cache/.mojo_cache before
#    believing anything else.
#
#    The floor is passed to the LINKER instead, which stamps
#    LC_BUILD_VERSION exactly as the environment variable would while the
#    Metal compile step never sees it.
#
# 2. A BUILD WITHOUT A GATE SHIPS WHATEVER CAME OUT. build_estimators.sh
#    printed "built ..." and exited 0 for hours over a kernel-less artifact.
#    Two gates are here: an AIR-blob floor as a cheap pre-filter, and a real
#    LAUNCH of one kernel from each lane in this extension. Only the second
#    one proves anything; every broken build in that bug's history imported
#    fine.
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
    MACOS_SDK=""  # linux arm (E1): no Mach-O, no Metal SDK
fi
LINK_FLAGS="-Xlinker -platform_version -Xlinker macos -Xlinker $MACOS_FLOOR -Xlinker $MACOS_SDK"
[ "$(uname)" = "Darwin" ] || LINK_FLAGS=""

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-tsa.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT INT TERM
out="$tmpdir/_mojolearn_tsa.so"

# `--target-cpu apple-m1` is the oldest Apple silicon: `mojo build` otherwise
# defaults to whatever chip ran the compiler, and a host-built wheel SIGILLs on
# older Macs at the instruction, inside the extension, with no diagnostic a
# user can act on. arm64 Mach-O `cpusubtype` stays ARM64_ALL whatever -mcpu
# was, so such a wheel LOOKS portable to any header-reading check.
#
# `--target-accelerator metal:1` is kept from the sibling scripts.
# shellcheck disable=SC2086
TARGET_FLAGS="--target-cpu apple-m1 --target-accelerator metal:1"
[ "$(uname)" = "Darwin" ] || TARGET_FLAGS=""  # linux arm: host cpu + its GPU
# Explicit kernel-matrix column: MOJOLEARN_TARGET_COLUMN=apple|nvidia|amd|amd_rdna
COLUMN_DEFINE=""
# THE NUMERIC MODE IS A BUILD DEFINE, NOT A SOURCE FLIP.
# MOJOLEARN_NUMERIC_MODE=identical compiles with -D MOJOLEARN_NUMERIC_IDENTICAL=1
# (mojo_only/numerics.mojo reads it through is_defined, the same shape as the
# column define) and lands the binary under python/mojolearn/identical/.
#
# NOTE FOR WHOEVER WIRES THE PACKAGE. python/mojolearn/_backend.py's `_MODULES`
# tuple and `_build_script` map do NOT list `_mojolearn_tsa`; that file is not
# this work's to edit. Until it does, `_backend.select()` will not install the
# identical build of this extension under the canonical module name, so
# `python/mojolearn/_tsa_impl.py` carries its own mode-aware loader that reads
# `_backend.requested_mode()` and refuses BY NAME rather than silently falling
# back to the fast binary. Adding the two lines to `_backend.py` makes that
# fallback loader dead code, which is the right end state.
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
pixi run mojo build --emit shared-lib \
    $TARGET_FLAGS $COLUMN_DEFINE $MODE_DEFINE \
    $LINK_FLAGS \
    -I . -I bindings \
    bindings/_mojolearn_tsa.mojo \
    -o "$out"

air_blobs() {
    strings -a "$1" \
        | grep -oE '[0-9A-Za-z_]+_[0-9a-f]{16}air' \
        | sed -e 's/^_gpu_shared_mem//' -e 's/^0//' \
        | sort -u
}

# The gates below import the python package, so during a from-scratch
# multi-binding build (a fresh linux box, E1) they fail on the siblings'
# not-yet-built .so files; and the AIR/otool checks are Mach-O-only. The
# caller that sets this owns end-to-end verification.
if [ -n "${MOJOLEARN_SKIP_BUILD_GATE:-}" ] || [ "$(uname)" != "Darwin" ]; then
    mv "$out" "$OUTDIR/_mojolearn_tsa.so"
    echo "built $OUTDIR/_mojolearn_tsa.so (gate skipped: non-Darwin or MOJOLEARN_SKIP_BUILD_GATE)"
    exit 0
fi

# A FLOOR, NOT A PROOF. The failure this guards against is an artifact with
# zero kernels; the failure the smoke test guards against is an artifact that
# imports and dies at the first launch.
#
# WHY 10. The two lanes behind this extension define fifteen kernels that a
# reader can count in the source: eight in holtwinters/ported (hw_transpose,
# conv1d, season_residual, season_mean, batched_ls_solver,
# holtwinters_seasonal_forecast, holtwinters_eval_gpu_global,
# holtwinters_optim_gpu_global) and seven in tsa/ported (series_sum, center,
# s2B_accumulation, cumsum_by_series, kpss_stationarity_check, batched_diff,
# batched_second_diff), before anything core/ contributes and before a
# parametric kernel's instantiations are counted separately. Ten sits below
# that with room for a few to be inlined or specialized away without turning
# this red, and it is unmistakably far from the zero this gate exists to
# catch. It is also the number bindings/build_estimators.sh uses, so the two
# scripts do not disagree for no reason. RAISE IT once a successful build has
# printed its real count on the last line: a floor set from a measurement is
# worth more than one set from a source count, and this one has never been
# measured.
count=$(air_blobs "$out" | wc -l | tr -d ' ')
# MEASURED 2026-08-24, first cold build on the M4: 8 AIR blobs. The 10 that
# stood here was the source-count guess this comment block asked to have
# replaced, and it failed a build that was in fact complete. 5 is under the
# measured number with slack for an instantiation to be inlined away.
if [ "$count" -lt 5 ]; then
    printf 'FAILED: %s AIR blobs, want at least 5 (measured 8 on 2026-08-24).\n' "$count" >&2
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

# THE REAL GATE: import and LAUNCH a kernel from EACH LANE in this extension.
# These are THIS extension's kernels and nobody else's -- a copied smoke test
# from a sibling script would pass over a broken artifact here.
#
#   ExponentialSmoothing.fit  reaches hw_transpose, conv1d, season_mean,
#                             batched_ls_solver and the BFGS optimizer kernel
#   .forecast                 reaches holtwinters_seasonal_forecast
#   kpss_test                 reaches batched_diff, series_sum, center,
#                             s2B_accumulation, cumsum_by_series and
#                             kpss_stationarity_check
#   select_d                  reaches the same set, twice, through the host loop
MOJOLEARN_SMOKE_SO="$out" python3 - <<'PY'
import os, shutil, sys, tempfile
tmp = tempfile.mkdtemp()
pkg = os.path.join(tmp, "mojolearn")
shutil.copytree("python/mojolearn", pkg,
                ignore=shutil.ignore_patterns("__pycache__"))
shutil.copyfile(os.environ["MOJOLEARN_SMOKE_SO"],
                os.path.join(pkg, "_mojolearn_tsa.so"))
sys.path.insert(0, tmp)
import numpy as np

# THROUGH THE PYTHON WRAPPER, NOT THE RAW BINDING. The extension's entry
# points take bare addresses plus a packed `params` list, so a hand-rolled
# call here would encode that ABI a second time and drift from it.
# `_tsa_impl` is imported by module path rather than from the package's
# export list, because re-exporting these names is the operator's call and
# this gate must not depend on it having been made.
from mojolearn import _tsa_impl

rng = np.random.default_rng(0)

# Holt-Winters: n >= start_periods * seasonal_periods is their rule, so
# n = 24 with seasonal_periods = 4 and start_periods = 2 clears it.
t = np.arange(24, dtype=np.float32)
endog = np.stack([
    10.0 + 0.5 * t + 2.0 * np.sin(t * (np.pi / 2.0)),
    20.0 - 0.3 * t + 1.0 * np.cos(t * (np.pi / 2.0)),
]).astype(np.float32)
es = _tsa_impl.ExponentialSmoothing(
    endog, seasonal="additive", seasonal_periods=4, start_periods=2, ts_num=2
).fit()
fc = es.forecast(3)
# (h, ts_num), which is cuML's own orientation for the index=None,
# ts_num > 1 case (holtwinters.pyx:420 returns forecasted_points[:, :h].T).
assert fc.shape == (3, 2), fc.shape
assert np.isfinite(fc).all(), fc
assert es.level_.shape == (2, 24 - 4), es.level_.shape

# KPSS: an AR(1) around a mean is stationary, a random walk is not. The
# assertion is on the SHAPES and finiteness, not on the two verdicts:
# this is a build gate, and the verdicts belong to tsa/mojo_only/
# stationarity_check.mojo, which asserts them against an oracle.
n_obs = 120
ar = np.zeros(n_obs, dtype=np.float32)
for i in range(1, n_obs):
    ar[i] = np.float32(0.5) * ar[i - 1] + np.float32(rng.standard_normal())
rw = np.cumsum(rng.standard_normal(n_obs)).astype(np.float32)
y = np.stack([ar, rw], axis=1)          # (n_obs, batch), cuML's layout
flags = _tsa_impl.kpss_test(y, d=1)
assert flags.shape == (2,), flags.shape
d = _tsa_impl.select_d(y, D=0, s=0)
assert d.shape == (2,) and ((d >= 0) & (d <= 2)).all(), d

print("  smoke: holtwinters fit + forecast and kpss_test + select_d each launched")
PY

mv "$out" "$OUTDIR/_mojolearn_tsa.so"
echo "built $OUTDIR/_mojolearn_tsa.so ($count AIR blobs, minos $MACOS_FLOOR)"
