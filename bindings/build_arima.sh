#!/bin/sh
# Build the batched ARIMA CPython extension into
# python/mojolearn/_mojolearn_arima.so. Run from anywhere; requires pixi.
#
# THIS EXTENSION IS arima/ AND NOTHING ELSE. It is separate from
# `bindings/build_tsa.sh` (Holt-Winters, KPSS, select_d), which is the other
# time-series binding, for the reason the estimators binding's header gives:
# an independently changing binding must not become a merge point. The two
# lanes share `tsa/impl/timeSeries/arima_helpers.mojo::prepare_data` and
# nothing else. All eleven binaries land in one wheel.
#
# THE FLAGS BELOW ARE NOT ORNAMENTAL. Every one of them is a bug somebody
# already shipped. The full write-ups live in `bindings/build.sh` and
# `bindings/build_gbdt.sh`; the short version, because a reader who edits
# this file needs to know what not to touch:
#
#   * MACOSX_DEPLOYMENT_TARGET SET IN THE ENVIRONMENT suppresses
#     ahead-of-time Metal compilation entirely. `mojo build` writes an
#     empty 134-byte metallib per kernel, embeds nothing, and the
#     extension then imports cleanly and dies at the first launch with
#     "Failed to create Metal function". Measured cold, one variable:
#     set -> 0 AIR blobs, unset -> 141. The VALUE is innocent; SETTING IT
#     AT ALL is the bug. The macOS floor goes to the LINKER instead, which
#     stamps LC_BUILD_VERSION exactly the same way while the Metal compile
#     step never sees it.
#
#   * $MODULAR_HOME/cache/.mojo_cache IS CONTENT-ADDRESSED AND ITS KEY
#     DOES NOT INCLUDE THE DEPLOYMENT TARGET, so one poisoned build serves
#     empty metallibs to every later build whatever ITS flags are. Clear
#     the cache before any build whose kernel count you intend to believe,
#     or the number is fiction.
#
#   * --target-cpu apple-m1 IS THE PORTABLE BASELINE. `mojo build`
#     otherwise targets whatever chip ran the compiler (here apple-m4,
#     with +sme, +sme2, +bf16, +i8mm, none of which exist on M1), and LLVM
#     emits those instructions from ordinary loops once the bit is set. A
#     host-built wheel then SIGILLs on older Apple silicon, inside the
#     extension, with no diagnostic a user can act on, and arm64 Mach-O
#     cpusubtype stays ARM64_ALL whatever -mcpu was, so such a wheel looks
#     portable to any header-reading check.
#
#   * THERE IS NO --target-accelerator FLAG HERE. Measured with
#     --target-cpu held fixed: no flag gives every kernel, `metal:1` gives
#     0 and `apple-m1` gives 0. Passing the flag at all, right value or
#     wrong, suppresses ahead-of-time Metal compilation. (build_estimators.sh
#     still passes `metal:1` and still produces kernels; that is the
#     compiler cache being kind to it, not a contradiction. This script
#     does not rely on kindness.)
#
# MACOS_FLOOR MUST EQUAL setup.py's DEFAULT_MACOS_TARGET AND EVERY SIBLING
# BUILD SCRIPT'S. The .so files land in one wheel under one tag, and the
# tag is the lower bound of what is inside it.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$here"

mkdir -p python/mojolearn

MACOS_FLOOR="11.0"

# NOT EXPORTED, DELIBERATELY. Unset it if the caller had it set, because
# inheriting it from an outer shell reproduces the bug silently.
unset MACOSX_DEPLOYMENT_TARGET

if [ "$(uname)" = "Darwin" ]; then
    MACOS_SDK=$(xcrun --sdk macosx --show-sdk-version)
else
    MACOS_SDK=""  # linux arm (E1): no Mach-O, no Metal SDK
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
COLUMN_DEFINE=""

# THE NUMERIC MODE IS A BUILD DEFINE, NOT A SOURCE FLIP.
# MOJOLEARN_NUMERIC_MODE=identical compiles with -D MOJOLEARN_NUMERIC_IDENTICAL=1
# (checks/numerics.mojo reads it through is_defined) and lands the binary
# under python/mojolearn/identical/. The build-time smoke gate imports the
# FAST package, so it is skipped for an identical build.
#
# NOTE FOR THE OPERATOR: `_mojolearn_arima` IS listed in
# python/mojolearn/_backend.py's `_MODULES` and in its `_build_script` dict,
# both, from the day this script was written (DEVIATION 869 is what happens
# when only one of the two is filled in). `_arima_impl.py` therefore does
# NOT carry a private mode-aware loader; it uses `NumericModeMixin._bind`
# and cross-checks `arima_numeric_mode()` against the tier the package
# resolved, so an identical run cannot silently get the FAST binary.
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
    # is its own binary, PIN_DETERMINISM is comptime, so a
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

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-arima.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT INT TERM
out="$tmpdir/_mojolearn_arima.so"

# --emit shared-lib, not an executable: CPython dlopens this and calls
# PyInit__mojolearn_arima. The FILE NAME must match that symbol's suffix or
# the import fails with "dynamic module does not define module export
# function".
# shellcheck disable=SC2086  # the flag strings are deliberately word-split
pixi run mojo build -j "${MOJOLEARN_COMPILE_JOBS:-2}" --emit shared-lib \
    $TARGET_FLAGS $COLUMN_DEFINE $MODE_DEFINE \
    $LINK_FLAGS \
    -I . -I bindings \
    bindings/_mojolearn_arima.mojo \
    -o "$out"

air_blobs() {
    strings -a "$1" \
        | grep -oE '[0-9A-Za-z_]+_[0-9a-f]{16}air' \
        | sed -e 's/^_gpu_shared_mem//' -e 's/^0//' \
        | sort -u
}

# The gates below import the WHOLE python package, so during a from-scratch
# multi-binding build (a fresh linux box, E1) they fail on the siblings'
# not-yet-built .so files; and the AIR/otool checks are Mach-O only. The
# caller that sets MOJOLEARN_SKIP_BUILD_GATE owns end-to-end verification.
if [ -n "${MOJOLEARN_SKIP_BUILD_GATE:-}" ] || [ "$(uname)" != "Darwin" ]; then
    mv "$out" "$OUTDIR/_mojolearn_arima.so"
    echo "built $OUTDIR/_mojolearn_arima.so (gate skipped: non-Darwin or MOJOLEARN_SKIP_BUILD_GATE)"
    exit 0
fi

# ============================================================================
# A FLOOR, NOT A PROOF, AND THESE NUMBERS ARE PLACEHOLDERS SET TO 1.
# ============================================================================
#
# THE FLOORS BELOW ARE 1 AND 1 BECAUSE THIS SCRIPT HAS NEVER BEEN BUILT.
# THEY MUST BE RAISED, ON THE FIRST COLD BUILD, TO JUST UNDER WHAT THAT
# BUILD ACTUALLY MEASURES, and the measured number written into this
# comment. Set them at roughly two thirds of measured, the ratio
# bindings/build.sh uses against ITS measured counts (22 measured -> floor
# 15, 8 -> 3) and the ratio build_svm.sh adopted. The observed counts are
# printed unconditionally below, so the real value is one build away and a
# failure names it instead of hiding it.
#
# WHY A FLOOR OF 1 IS NOT ENOUGH AND MUST NOT BE LEFT HERE. build.sh learned
# twice that presence-of-one is not a filter: the build that lost GBDT kept
# exactly 1 of 85 gbdt_ blobs and passed a presence-of-one check. A floor of
# 1 catches the TOTAL Metal failure this gate was written for (the
# MACOSX_DEPLOYMENT_TARGET bug, 0 blobs) and catches nothing else.
# build_svm.sh's floors were written from a HAND COUNT of the source and one
# of them failed a perfectly good artifact on its first run; the honest
# starting point is therefore the smallest number that catches the known
# failure, plus this paragraph, rather than a guess dressed as a
# measurement.
#
# WHAT SHOULD BE IN HERE. `arima/` launches the batched Kalman kernels
# (init_batched_kalman_matrices_kernel, kalman_init_state_kernel,
# batched_kalman_loop_kernel), the pack/unpack pair, the Jones transform,
# the in-sample prediction and forecast-copy kernels, the finite-difference
# perturb/reset/grad kernels, and estimate_x0's least-squares kernels.
# `tsa/` should contribute exactly the differencing kernels behind
# `prepare_data`, which is the one function this lane imports from that one.
# The subsystem prefix is the top-level directory the kernel's module lives
# in, which is how build.sh's `cluster:15 neighbors:3 core:1` works. If that
# assumption is wrong for these two names the gate fails LOUDLY on the first
# build with both counts printed, which is the right way to find out.
#
# core/ and gemm/ blobs are printed but NOT floored: nothing in this lane
# calls a GEMM, and whether the identity helpers leave a `core` prefixed
# blob at all is not something to assert before it has been seen once.
_air=$(air_blobs "$out")
_total=$(printf '%s\n' "$_air" | grep -c . || true)
printf '  AIR blobs by subsystem (total %s):\n' "$_total"
for _sub in arima tsa core gemm; do
    printf '    %-18s %s\n' "$_sub" "$(printf '%s\n' "$_air" | grep -c "^${_sub}" || true)"
done

_failed=0
# MEASURED 2026-09-01 ON THE FIRST COLD BUILD: 12 AIR blobs, ALL under
# `arima`, and ZERO under `tsa`. The `tsa:1` floor guessed that
# `prepare_data` would emit its own device kernel from this binary; it does
# not, and the gate said so loudly with both counts printed, which is what
# it is for. The floor is now `arima:8`, two thirds of the measured 12, and
# `tsa` is dropped rather than set to 0 so a future zero under `arima` still
# fails.
for _pair in arima:8; do
    _s=${_pair%%:*}
    _min=${_pair#*:}
    _n=$(printf '%s\n' "$_air" | grep -c "^${_s}" || true)
    if [ "$_n" -lt "$_min" ]; then
        printf 'FAILED: %s has %s AIR blobs, want at least %s.\n' "$_s" "$_n" "$_min" >&2
        _failed=1
    fi
done
if [ "$_failed" -ne 0 ]; then
    printf 'If these are 0, check MACOSX_DEPLOYMENT_TARGET in the environment\n' >&2
    printf 'and then $MODULAR_HOME/cache/.mojo_cache for empty 134-byte\n' >&2
    printf 'metallibs -- one poisoned build serves them to every later one:\n' >&2
    printf '\n' >&2
    printf '  find "$MODULAR_HOME/cache/.mojo_cache" -type f -size -200c \\\n' >&2
    printf "    -exec sh -c 'head -c4 \"\$1\" | grep -q MTLB && echo \"\$1\"' _ {} \\;\n" >&2
    printf '\n' >&2
    printf 'If they are nonzero but under the floor, the floor may simply be\n' >&2
    printf 'wrong: it was never measured, it was set to 1 and left for the\n' >&2
    printf 'first build to replace.\n' >&2
    exit 1
fi

# THE MACH-O FLOOR IS READ BACK, NOT ASSUMED. A silently dropped -Xlinker
# would publish a wheel whose tag and binary disagree, which is exactly the
# failure the flag exists to prevent.
got=$(otool -l "$out" | awk '/LC_BUILD_VERSION/{f=1} f && /minos/{print $2; exit}')
if [ "$got" != "$MACOS_FLOOR" ]; then
    printf 'FAILED: minos is %s, want %s.\n' "$got" "$MACOS_FLOOR" >&2
    exit 1
fi

# ============================================================================
# THE REAL GATE: import and LAUNCH. Every broken build in this bug's history
# imported fine and died at the first kernel.
# ============================================================================
#
# THROUGH THE PYTHON WRAPPERS, NOT THE RAW BINDINGS, for the reason
# build_estimators.sh gives: the entry points take bare addresses plus a
# packed params list, and a hand-rolled call here would encode that ABI a
# second time and drift from it. The whole point of the thirteen-entry list
# being written out in two places is that there are two, not three.
#
# `_arima_impl` is imported as a submodule even though `ARIMA` IS exported
# from `mojolearn/__init__.py`, because the private name is stable and this
# gate should not break on a re-export.
#
# THE ARIMA ROWS ARE ARIMA'S. Copying a sibling script's smoke rows would be
# a gate that cannot fail, since none of those kernels are in this artifact.
#
#   ARIMA((1,0,0)).fit      -> estimate_x0's AR least squares (the
#                              Householder QR), test_invparams, the INVERSE
#                              Jones transform, the batched L-BFGS with its
#                              shared line search, the perturb/reset/grad
#                              finite-difference kernels, the whole Kalman
#                              loop once per evaluation, the FORWARD
#                              transform, pack and unpack
#   .predict                -> in_sample_prediction_kernel with dD == 0
#   .forecast               -> the forecast half of the Kalman loop and
#                              copy_forecast_kernel, neither of which any
#                              in-sample row reaches
#   ARIMA((1,1,1))          -> the DIFFERENCING path: prepare_data (the tsa
#                              blobs), the dD == 1 arm of the in-sample
#                              kernel, finalize_forecast, and the MA block
#                              of estimate_x0
#   the two refusals        -> exog and method='css', which cost no device
#                              work and prove the refusals are wired, not
#                              merely present
#
# Kept small on purpose, batch 3 and 96 observations, because this runs on
# every build, a fit is hundreds of Kalman passes, and the GPU is shared.
# The recovery assertion here is DELIBERATELY LOOSE (0.35 on a planted 0.6
# at n = 96) and is a smoke test, not the gate:
# python/mojolearn/tests/test_arima_surface.py is where recovery is asserted
# against a stated multiple of the standard error at the length the lane's
# own fixtures use.
MOJOLEARN_SMOKE_SO="$out" python3 - <<'PY'
import os, shutil, sys, tempfile
tmp = tempfile.mkdtemp()
pkg = os.path.join(tmp, "mojolearn")
shutil.copytree("python/mojolearn", pkg,
                ignore=shutil.ignore_patterns("__pycache__"))
shutil.copyfile(os.environ["MOJOLEARN_SMOKE_SO"],
                os.path.join(pkg, "_mojolearn_arima.so"))
sys.path.insert(0, tmp)
import numpy as np

from mojolearn import _arima_impl

# THREE SERIES FROM ONE KNOWN phi, with different innovations, so a fit that
# returns the same number for all three (a broadcast bug) is visible.
rng = np.random.default_rng(0)
n_obs, batch = 96, 3
e = rng.standard_normal((batch, n_obs))
y = np.zeros((batch, n_obs), dtype=np.float32)
acc = np.zeros(batch)
for t in range(n_obs):
    acc = 0.6 * acc + e[:, t]
    y[:, t] = acc

m = _arima_impl.ARIMA(order=(1, 0, 0), trend="n").fit(y)
assert m.params_.shape == (batch, 2), m.params_.shape      # ar, sigma2
ar = m.params_[:, 0]
assert np.all(np.abs(ar - 0.6) < 0.35), ar
assert len(set(np.round(ar, 6))) == batch, ar              # not broadcast
assert m.llf_.shape == (batch,) and np.all(np.isfinite(m.llf_)), m.llf_
assert m.aic_.shape == (batch,) and np.all(m.aic_ > -2.0 * m.llf_), m.aic_
assert m.bic_.shape == (batch,)

p = m.predict(0, n_obs)
assert p.shape == (batch, n_obs), p.shape
assert np.isfinite(p).all(), "d == 0, nothing should be NaN"

f = m.forecast(4)
assert f.shape == (batch, 4), f.shape
assert np.isfinite(f).all(), f
# a stationary AR(1) forecast decays toward the mean, so |f| cannot grow
assert np.all(np.abs(f[:, 3]) <= np.abs(f[:, 0]) + 1e-5), f

# THE DIFFERENCING PATH. A random walk with drift, d = 1.
w = np.cumsum(rng.standard_normal((batch, n_obs)) + 0.05, axis=1).astype(np.float32)
md = _arima_impl.ARIMA(order=(1, 1, 1), trend="c").fit(w)
assert md.params_.shape == (batch, 4), md.params_.shape    # mu, ar, ma, sigma2
pd_ = md.predict(0, n_obs)
assert np.isnan(pd_[:, 0]).all(), "step 0 is undefined when d == 1"
assert np.isfinite(pd_[:, 1:]).all(), pd_
fd = md.forecast(3)
assert fd.shape == (batch, 3) and np.isfinite(fd).all(), fd

# THE REFUSALS, made to fire. Neither reaches a kernel, and BOTH are raised
# in Mojo, so both are caught as a bare Exception: a Mojo `Error` crossing
# `def_function` is not a named Python type. That is what makes the message
# check the real assertion here.
try:
    _arima_impl.ARIMA(order=(1, 0, 0)).fit(y, exog=np.ones((batch, n_obs, 1)))
except Exception as exc:
    assert "exog" in str(exc), exc
else:
    raise AssertionError("exog was ACCEPTED")

try:
    _arima_impl.ARIMA(order=(1, 0, 0), method="css").fit(y)
except Exception as exc:
    assert "css" in str(exc), exc
else:
    raise AssertionError("method='css' was ACCEPTED")

# and one PYTHON-side refusal, which IS a named type
try:
    _arima_impl.ARIMA(order=(1, 0, 0), trend="ct")
except NotImplementedError as exc:
    assert "trend" in str(exc), exc
else:
    raise AssertionError("trend='ct' was ACCEPTED")

print("  smoke: ARIMA fit/predict/forecast on (1,0,0) and (1,1,1), "
      "exog and css refused")
shutil.rmtree(tmp, ignore_errors=True)
PY

mv "$out" "$OUTDIR/_mojolearn_arima.so"
echo "built $OUTDIR/_mojolearn_arima.so ($_total AIR blobs, minos $MACOS_FLOOR)"
