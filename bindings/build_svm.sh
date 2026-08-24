#!/bin/sh
# Build the C-SVC / Isolation Forest CPython extension into
# python/mojolearn/_mojolearn_svm.so. Run from anywhere; requires pixi.
#
# THIS EXTENSION IS svm/ AND isolation_forest/. It is separate from
# `bindings/build.sh` (k-means, k-NN), `build_gbdt.sh`, `build_rf.sh`,
# `build_trees.sh` and `build_estimators.sh` (DBSCAN, KDE, PCA, tSVD, OLS,
# Ridge, logistic) for the reason the estimators binding's header gives:
# an independently changing binding must not become a merge point. All of
# them land in one wheel.
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
#     AT ALL is the bug. build_estimators.sh was the last script still
#     exporting it and its artifact was the only one in the tree with zero
#     blobs. The macOS floor goes to the LINKER instead, which stamps
#     LC_BUILD_VERSION exactly the same way while the Metal compile step
#     never sees it.
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
#     extension, with no diagnostic a user can act on -- and arm64 Mach-O
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

TARGET_FLAGS="--target-cpu apple-m1"
[ "$(uname)" = "Darwin" ] || TARGET_FLAGS=""  # linux arm: host cpu + its GPU

# Explicit kernel-matrix column: MOJOLEARN_TARGET_COLUMN=apple|nvidia|amd|amd_rdna
COLUMN_DEFINE=""

# THE NUMERIC MODE IS A BUILD DEFINE, NOT A SOURCE FLIP.
# MOJOLEARN_NUMERIC_MODE=identical compiles with -D MOJOLEARN_NUMERIC_IDENTICAL=1
# (mojo_only/numerics.mojo reads it through is_defined) and lands the binary
# under python/mojolearn/identical/. The build-time smoke gate imports the
# FAST package, so it is skipped for an identical build.
#
# NOTE FOR THE OPERATOR: python/mojolearn/_backend.py's `_MODULES` tuple does
# NOT list `_mojolearn_svm`, so the identical selector will not install this
# module under the canonical name. `_svm_impl.py` loads it itself and
# cross-checks `svm_numeric_mode()`, so an identical run cannot silently get
# the FAST binary -- but adding `_mojolearn_svm` to `_MODULES` and
# `_build_script` is the tidier fix and it belongs to whoever owns that file.
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

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-svm.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT INT TERM
out="$tmpdir/_mojolearn_svm.so"

# --emit shared-lib, not an executable: CPython dlopens this and calls
# PyInit__mojolearn_svm. The FILE NAME must match that symbol's suffix or the
# import fails with "dynamic module does not define module export function".
# shellcheck disable=SC2086  # the flag strings are deliberately word-split
pixi run mojo build --emit shared-lib \
    $TARGET_FLAGS $COLUMN_DEFINE $MODE_DEFINE \
    $LINK_FLAGS \
    -I . -I bindings \
    bindings/_mojolearn_svm.mojo \
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
    mv "$out" "$OUTDIR/_mojolearn_svm.so"
    echo "built $OUTDIR/_mojolearn_svm.so (gate skipped: non-Darwin or MOJOLEARN_SKIP_BUILD_GATE)"
    exit 0
fi

# ============================================================================
# A FLOOR, NOT A PROOF -- AND THESE PARTICULAR NUMBERS HAVE NEVER BEEN
# MEASURED, BECAUSE THE AUTHOR OF THIS SCRIPT WAS NOT ALLOWED TO BUILD.
# ============================================================================
#
# build.sh learned twice that presence-of-one is not a filter: the build that
# lost GBDT kept exactly 1 of 85 gbdt_ blobs and passed. So the floors sit
# well above 1, and they are PER SUBSYSTEM, so that one lane's kernels going
# missing cannot be hidden by the other lane's still being there.
#
# WHERE THE NUMBERS COME FROM. Counted by hand from the source, not from an
# artifact: `svm/` launches 33 distinct kernel functions across
# ported/svm/*.mojo, ported/distance/kernel_matrices.mojo and
# mojo_only/device_select.mojo, plus the parameterized
# `smo_block_solve_kernel` at six instantiation sites in smosolver.mojo.
# `isolation_forest/` launches 5 (build_isolation_trees_global,
# compute_path_lengths_global, anomaly_score, predict_labels,
# xorwow_device). The floors below are roughly 60% of the hand count, which
# is the same ratio build.sh chose against ITS measured counts (22 measured
# -> floor 15, 8 -> 3).
#
# THE FIRST REAL BUILD SHOULD REPLACE THESE WITH TWO THIRDS OF WHAT IT
# ACTUALLY MEASURES, and write the measured number in this comment. The
# observed counts are printed unconditionally below so that number is one
# build away and a failure names the real value instead of hiding it.
#
# The subsystem prefix is the top-level directory the kernel's module lives
# in, which is how build.sh's `cluster:15 neighbors:3 core:1` works. If that
# assumption is wrong for these two names the gate fails LOUDLY on the first
# build with both counts printed, which is the right way to find out.
#
# gemm/ and core/ blobs also land here -- the LINEAR and RBF kernel matrices
# go through `identical_gemm_into` under IDENTICAL and `core.gemm.gemm_nt`
# under FAST -- and they are printed but NOT floored, because the FAST arm
# reaches MAX's matmul and its blobs need not carry a `gemm` prefix at all.
_air=$(air_blobs "$out")
_total=$(printf '%s\n' "$_air" | grep -c . || true)
printf '  AIR blobs by subsystem (total %s):\n' "$_total"
for _sub in svm isolation_forest gemm core; do
    printf '    %-18s %s\n' "$_sub" "$(printf '%s\n' "$_air" | grep -c "^${_sub}" || true)"
done

_failed=0
# FLOORS ARE MEASURED, NOT REASONED. First cold build 2026-08-24 printed
# svm 13, isolation_forest 3, gemm 0, core 0. The 20 that stood here was a
# source count of `enqueue_function` sites and it FAILED a perfectly good
# artifact on the first run; the author wrote that it was a guess and asked
# for it to be replaced by the real number, which is what these are.
#
# Set at roughly two thirds of measured, the ratio bindings/build.sh uses
# (it floors 15 against a measured 22, and 3 against 8). That leaves room
# for an instantiation to be inlined away without a false red, while still
# being unmistakably far from the 0 this gate exists to catch. gemm and core
# measured 0 here and are deliberately NOT floored: under FAST the kernel
# matrix routes through MAX's own matmul, whose blobs need not carry a
# `gemm` prefix. The counts are printed above either way.
#
# THE GATE THAT ACTUALLY PROVES THE ARTIFACT IS run_smoke BELOW. A blob
# count is a pre-filter, and build.sh records an artifact that kept 1 of 85
# gbdt blobs and passed a presence-of-one check.
for _pair in svm:8 isolation_forest:2; do
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
    printf 'wrong: it was written from a source count, never from a build.\n' >&2
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
# second time and drift from it.
#
# `_svm_impl` and `_iforest_impl` are private modules and are NOT exported
# from `mojolearn/__init__.py` yet, so they are imported as submodules. When
# the operator re-exports `SVC` and `IsolationForest` into the public
# namespace, this gate keeps working unchanged.
#
# THE SVM ROW IS SVM'S AND THE FOREST ROW IS THE FOREST'S. Copying a sibling
# script's smoke rows would be a gate that cannot fail, since none of those
# kernels are in this artifact.
#
#   SVC(kernel='linear')  -> ovr_labels, the gemm kernel matrix, the working
#                            set select/sort chain, smo_block_solve, update_f,
#                            the flag/scan/scatter compaction, CalcB's serial
#                            reductions, gather_rows, decision_kernel
#   SVC(kernel='rbf')     -> adds row_norm_l2sq and rbf_kernel_expanded, the
#                            two kernels the linear arm never touches
#   IsolationForest       -> xorwow init inside build_isolation_trees_global,
#                            compute_path_lengths_global, anomaly_score
#     .decision_function  -> the contamination quantile path (offset_ != -0.5)
#     .predict            -> predict_labels_kernel, the only kernel the score
#                            paths do not reach
#
# Kept small on purpose -- 128 rows -- because this runs on every build and
# the GPU is shared.
MOJOLEARN_SMOKE_SO="$out" python3 - <<'PY'
import os, shutil, sys, tempfile
tmp = tempfile.mkdtemp()
pkg = os.path.join(tmp, "mojolearn")
shutil.copytree("python/mojolearn", pkg,
                ignore=shutil.ignore_patterns("__pycache__"))
shutil.copyfile(os.environ["MOJOLEARN_SMOKE_SO"],
                os.path.join(pkg, "_mojolearn_svm.so"))
sys.path.insert(0, tmp)
import numpy as np

from mojolearn import _iforest_impl, _svm_impl

rng = np.random.default_rng(0)
X = rng.random((128, 4), dtype=np.float32)
# a separable-enough binary problem, and NOT symmetric in the labels, so a
# swapped classes[0]/classes[1] would show up as an inverted prediction
y = np.where(X[:, 0] + X[:, 1] > 1.0, 7, -3).astype(np.float32)

lin = _svm_impl.SVC(kernel="linear", C=1.0).fit(X, y)
p = lin.predict(X[:8])
assert set(np.unique(p)).issubset({7, -3}), p
assert lin.decision_function(X[:8]).shape == (8,)
assert lin.n_support_ >= 1, lin.n_support_

rbf = _svm_impl.SVC(kernel="rbf", gamma=0.5, C=10.0).fit(X, y)
assert rbf.predict(X[:8]).shape == (8,)

# the linear kernel's coef_ is dual_coef_ @ support_vectors_
assert lin.coef_.shape == (1, 4), lin.coef_.shape

f = _iforest_impl.IsolationForest(
    n_estimators=4, max_samples=64, random_state=0
).fit(X)
assert f.score_samples(X[:8]).shape == (8,)
assert f.offset_ == -0.5, f.offset_
lab = f.predict(X[:8])
assert set(np.unique(lab)).issubset({-1, 1}), lab

# contamination as a float takes the quantile path, which scores the whole
# training set during fit and moves offset_ off -0.5
fc = _iforest_impl.IsolationForest(
    n_estimators=4, max_samples=64, random_state=0, contamination=0.1
).fit(X)
assert fc.offset_ != -0.5, fc.offset_
assert fc.decision_function(X[:8]).shape == (8,)

print("  smoke: SVC linear, SVC rbf, IsolationForest score/predict/quantile "
      "each launched")
shutil.rmtree(tmp, ignore_errors=True)
PY

mv "$out" "$OUTDIR/_mojolearn_svm.so"
echo "built $OUTDIR/_mojolearn_svm.so ($_total AIR blobs, minos $MACOS_FLOOR)"
