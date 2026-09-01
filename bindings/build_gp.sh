#!/bin/sh
# Build the Gaussian process regression CPython extension into
# python/mojolearn/_mojolearn_gp.so. Run from anywhere; requires pixi.
#
# THIS EXTENSION IS gaussian_process/ AND NOTHING ELSE. It is the
# THIRTEENTH binding, owed by commit 22a5b550 (the surface landed with the
# binding registered in _backend.py and not yet written), and it is
# separate from every sibling for the reason the estimators binding's
# header gives: an independently changing binding must not become a merge
# point. The lane reaches cholesky/ (factor, logdet, solve, trsm) and the
# identical GEMM through their own entry points; nothing else crosses. All
# thirteen binaries land in one wheel.
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
#   * THERE IS NO --target-accelerator FLAG HERE (on macOS). Measured with
#     --target-cpu held fixed: no flag gives every kernel, `metal:1` gives
#     0 and `apple-m1` gives 0. Passing the flag at all, right value or
#     wrong, suppresses ahead-of-time Metal compilation.
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
    MACOS_SDK=""  # linux (E1): no Mach-O, no Metal SDK
fi
LINK_FLAGS="-Xlinker -platform_version -Xlinker macos -Xlinker $MACOS_FLOOR -Xlinker $MACOS_SDK"
[ "$(uname)" = "Darwin" ] || LINK_FLAGS=""

# THE LINUX x86-64 CPU BASELINE. Without it `mojo build` targets WHATEVER
# CHIP THE BUILD BOX HAS; the 0.3.0 wheel shipped host AVX-512 with no cpuid
# dispatch that way and SIGILLed on every host without it (build_arima.sh
# carries the full write-up). x86-64-v3 is AVX2, FMA, BMI2 and SSE4.2 --
# Haswell 2013 and Zen 1 2017 onward -- and it EXCLUDES AVX-512. aarch64
# Linux keeps the empty flags it always had.
TARGET_FLAGS="--target-cpu apple-m1"
if [ "$(uname)" != "Darwin" ]; then
    case "$(uname -m)" in
        x86_64) TARGET_FLAGS="--target-cpu ${MOJOLEARN_LINUX_CPU:-x86-64-v3}" ;;
        *)      TARGET_FLAGS="" ;;   # linux arm: host cpu + its GPU
    esac
fi
# MOJOLEARN_GPU_ARCHS: ONE GPU architecture a LINUX set is compiled for
# (sm_80, gfx942, ...). The compiler takes EXACTLY ONE name (a comma list
# is rejected, measured 2026-08-30); packaging/linux/build_sets.sh reads
# the architecture back out of every binary and refuses a set that
# disagrees with what was asked.
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
# NOTE FOR THE OPERATOR: `_mojolearn_gp` IS listed in
# python/mojolearn/_backend.py's `_MODULES` and in its `_build_script`
# dict, both, SINCE BEFORE THIS SCRIPT EXISTED (commit 22a5b550 registered
# the name the day the surface landed, deliberately first, because
# DEVIATION 869 is what happens when only one of the two is filled in).
# `_gp_impl.py` therefore carries NO private mode-aware loader; it uses
# `NumericModeMixin._bind` and cross-checks `gp_numeric_mode()` against
# the tier the package resolved, so an identical run cannot silently get
# the FAST binary.
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

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-gp.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT INT TERM
out="$tmpdir/_mojolearn_gp.so"

# --emit shared-lib, not an executable: CPython dlopens this and calls
# PyInit__mojolearn_gp. The FILE NAME must match that symbol's suffix or
# the import fails with "dynamic module does not define module export
# function".
# shellcheck disable=SC2086  # the flag strings are deliberately word-split
pixi run mojo build --emit shared-lib \
    $TARGET_FLAGS $COLUMN_DEFINE $MODE_DEFINE \
    $LINK_FLAGS \
    -I . -I bindings \
    bindings/_mojolearn_gp.mojo \
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
    mv "$out" "$OUTDIR/_mojolearn_gp.so"
    echo "built $OUTDIR/_mojolearn_gp.so (gate skipped: non-Darwin or MOJOLEARN_SKIP_BUILD_GATE)"
    exit 0
fi

# ============================================================================
# A FLOOR, NOT A PROOF, AND THIS NUMBER IS A PLACEHOLDER SET TO 1.
# ============================================================================
#
# THE FLOOR BELOW IS 1 BECAUSE THIS SCRIPT HAS NEVER BEEN BUILT. IT MUST BE
# RAISED, ON THE FIRST COLD BUILD, TO TWO THIRDS OF WHAT THAT BUILD
# ACTUALLY MEASURES, and the measured number written into this comment --
# exactly what build_training.sh's history teaches: its floor was 1 until
# its first real build on 2026-09-01 measured 5 and it became 3, the ratio
# bindings/build.sh uses against ITS measured counts (22 measured -> floor
# 15, 8 -> 3) and the ratio build_svm.sh adopted after a hand-counted
# floor failed a perfectly good artifact on its first run. The observed
# counts are printed unconditionally below, so the real value is one build
# away and a failure names it instead of hiding it.
#
# WHY A FLOOR OF 1 IS NOT ENOUGH AND MUST NOT BE LEFT HERE. build.sh
# learned twice that presence-of-one is not a filter: the build that lost
# GBDT kept exactly 1 of 85 gbdt_ blobs and passed a presence-of-one
# check. A floor of 1 catches the TOTAL Metal failure this gate was
# written for (the MACOSX_DEPLOYMENT_TARGET bug, 0 blobs) and catches
# nothing else.
#
# WHAT SHOULD BE IN HERE. `gaussian_process/` launches the per-node kernel
# expression kernels behind `gp_kernel_matrix` (const/white/rbf/matern
# leaves and the sum/prod combiners), the predictive-variance kernel and
# the sabotage arms compiled into the lane. `cholesky/` should contribute
# the potrf panel kernels, the jitter, the logdet and the trsm solve;
# `gemm/` the identical GEMM behind the posterior mean. The subsystem
# prefix is the top-level directory the kernel's module lives in. If that
# assumption is wrong for any of these names the gate fails LOUDLY on the
# first build with every count printed, which is the right way to find
# out. cholesky/, gemm/, kde/ and core/ blobs are printed but NOT floored:
# whether a helper imported cross-lane (`l2_unexp_core` lives in kde/)
# leaves a blob under its own prefix from THIS binary is not something to
# assert before it has been seen once, and build_training.sh records that
# the identical GEMM's blobs can carry a non-gemm prefix.
_air=$(air_blobs "$out")
_total=$(printf '%s\n' "$_air" | grep -c . || true)
printf '  AIR blobs by subsystem (total %s):\n' "$_total"
for _sub in gaussian_process cholesky gemm kde core; do
    printf '    %-18s %s\n' "$_sub" "$(printf '%s\n' "$_air" | grep -c "^${_sub}" || true)"
done

_failed=0
for _pair in gaussian_process:1; do
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
# second time and drift from it. The whole point of the params list being
# written out in two places is that there are two, not three.
#
# `_gp_impl` is imported as a submodule even though
# `GaussianProcessRegressor` IS exported from `mojolearn/__init__.py`,
# because the private name is stable and this gate should not break on a
# re-export.
#
# THE GP ROWS ARE GP'S. Copying a sibling script's smoke rows would be a
# gate that cannot fail, since none of those kernels are in this artifact.
#
#   RBF fit                  -> the postfix spec rebuilt through the
#                               constructors, the RBF leaf kernel behind
#                               gp_kernel_matrix, the Cholesky factor with
#                               the ridge, cho_solve, logdet and the lml
#   .predict(return_std)     -> the cross-covariance kernel, the identical
#                               GEMM posterior mean, trsm_lower, the
#                               predictive-variance kernel and the zero
#                               clamp (DEVIATION 1760)
#   .predict() mean-only     -> the return_std=0 arm, whose var/std/clamp
#                               outputs must be skipped, not scribbled
#   Matern nu=0.7            -> the closed-forms refusal (DEVIATION 1765),
#                               raised in MOJO by gp_kernel_matern through
#                               the rebuild path, proving the constructor
#                               refusals are wired, not merely present
#   return_cov=True          -> the Python-side diagonal-only refusal
#                               (DEVIATION 1759)
#   duplicate rows, alpha=0  -> info_ != 0 as a RESULT (DEVIATION 1634)
#                               and predict on that fit refused BY NAME in
#                               Mojo, with info passed through the binding
#
# Kept small on purpose, 24 training points and 8 test points, because
# this runs on every build and the GPU is shared. The recovery assertion
# here is DELIBERATELY LOOSE (1e-2 where the lane's own bound is 2^-14)
# and is a smoke test, not the gate:
# python/mojolearn/tests/test_gp_surface.py is where recovery is asserted
# at the lane's bound.
MOJOLEARN_SMOKE_SO="$out" python3 - <<'PY'
import os, shutil, sys, tempfile
tmp = tempfile.mkdtemp()
pkg = os.path.join(tmp, "mojolearn")
shutil.copytree("python/mojolearn", pkg,
                ignore=shutil.ignore_patterns("__pycache__"))
shutil.copyfile(os.environ["MOJOLEARN_SMOKE_SO"],
                os.path.join(pkg, "_mojolearn_gp.so"))
sys.path.insert(0, tmp)
import numpy as np

from mojolearn import _gp_impl

# 24 points SPREAD over [-2, 2)^2 so a unit-length-scale RBF correlates
# them without driving K + 2^-20 I singular; points packed in [0, 1)^2
# would sit much closer to singular and turn this launch gate into a
# conditioning test, which it is not.
rng = np.random.default_rng(0)
n, d = 24, 2
x = (rng.random((n, d), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)
y = (np.sin(x[:, 0]) + 0.5 * x[:, 1]).astype(np.float32)

# alpha is the surface default 2^-20 spelled explicitly: the pinned ridge
# (DEVIATIONS 1751/1772), accepted on every tier.
m = _gp_impl.GaussianProcessRegressor(
    kernel=_gp_impl.RBF(1.0), alpha=2.0 ** -20).fit(x, y)
assert m.info_ == 0, f"info_={m.info_}: the factorization failed"
assert m.L_.shape == (n, n) and m.alpha_.shape == (n,)
assert np.isfinite(m.log_marginal_likelihood_value_)

mean, std = m.predict(x, return_std=True)
assert mean.shape == (n,) and std.shape == (n,)
worst = float(np.max(np.abs(mean - y)))
assert worst < 1e-2, f"posterior mean missed the training targets by {worst}"
assert (std >= 0.0).all(), "a negative std escaped the clamp"
assert m.clamped_.shape == (n,) and int(m.clamped_.sum()) == m.n_clamped_

alone = m.predict(x[:8])                    # the return_std=0 arm
assert alone.shape == (8,), alone.shape
assert np.isfinite(alone).all(), alone

# A COMPOSED kernel, so the sum/prod combiners and the length-scale table
# cross the boundary and are rebuilt through gp_kernel_sum/prod.
k2 = (_gp_impl.ConstantKernel(1.0) * _gp_impl.RBF([1.0, 1.0])
      + _gp_impl.WhiteKernel(0.0))
m2 = _gp_impl.GaussianProcessRegressor(kernel=k2, alpha=2.0 ** -20).fit(x, y)
assert m2.info_ == 0, m2.info_

# THE REFUSALS, made to fire. The Matern one is raised in MOJO, so it is
# caught as a bare Exception: a Mojo `Error` crossing `def_function` is
# not a named Python type. That is what makes the message check the real
# assertion here.
try:
    _gp_impl.GaussianProcessRegressor(
        kernel=_gp_impl.Matern(1.0, nu=0.7)).fit(x, y)
except Exception as exc:
    assert "CLOSED FORMS" in str(exc), exc
else:
    raise AssertionError("Matern nu=0.7 was ACCEPTED")

try:
    m.predict(x, return_cov=True)
except NotImplementedError as exc:
    assert "DIAGONAL" in str(exc), exc
else:
    raise AssertionError("return_cov=True was ACCEPTED")

# info IS A RESULT (DEVIATION 1634): duplicate rows with alpha=0 are
# exactly singular, and predict on that fit must be refused BY NAME in
# Mojo -- which only happens if the binding passed info through.
xdup = x.copy()
xdup[1] = xdup[0]
bad = _gp_impl.GaussianProcessRegressor(
    kernel=_gp_impl.RBF(1.0), alpha=0.0).fit(xdup, y)
assert bad.info_ != 0, "duplicate rows with alpha=0 factored cleanly?"
try:
    bad.predict(x)
except Exception as exc:
    assert "FAILED fit" in str(exc), exc
else:
    raise AssertionError("predict on a failed fit was ACCEPTED")

print("  smoke: GP fit/predict on RBF and a composed kernel, mean-only "
      "arm, Matern/return_cov refused, failed fit refused downstream")
shutil.rmtree(tmp, ignore_errors=True)
PY

mv "$out" "$OUTDIR/_mojolearn_gp.so"
echo "built $OUTDIR/_mojolearn_gp.so ($_total AIR blobs, minos $MACOS_FLOOR)"
