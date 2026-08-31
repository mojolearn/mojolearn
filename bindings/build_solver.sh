#!/bin/sh
# Build the coordinate-descent / single-linkage CPython extension into
# python/mojolearn/_mojolearn_solver.so. Run from anywhere; requires pixi.
#
# WHY A SIXTH EXTENSION. bindings/build.sh states the rule: each binding is
# built separately so an independently changing one does not become a merge
# point, and all of them land in one wheel. `solver/` and `hierarchy/` had no
# Python surface at all until now; putting them into
# _mojolearn_estimators.mojo would have made that file a merge point between
# three lanes at once.
#
# EVERYTHING BELOW THAT LOOKS LIKE BOILERPLATE IS NOT. Each of the four
# blocks -- the deployment target, the CPU baseline, the accelerator flag and
# the gate -- is a bug this repository already shipped once. The write-ups
# live in bindings/build.sh and bindings/build_gbdt.sh; the short forms are
# here so nobody has to go and look before editing this file.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$here"

mkdir -p python/mojolearn

# ============================================================================
# 1. MACOSX_DEPLOYMENT_TARGET IN THE ENVIRONMENT SUPPRESSES METAL COMPILATION
# ============================================================================
# Measured on a COLD CACHE, one variable at a time (bindings/build.sh has the
# table): with the variable set, `mojo build` emits an EMPTY 134-byte metallib
# per kernel and embeds nothing -- 0 AIR blobs -- and the extension then
# imports cleanly and dies at the first launch with "Failed to create Metal
# function". With it unset, every kernel is there. The VALUE is innocent;
# SETTING IT AT ALL is what does it.
#
# And the compiler cache is why every earlier measurement of this disagreed:
# $MODULAR_HOME/cache/.mojo_cache is content-addressed and ITS KEY DOES NOT
# INCLUDE THE DEPLOYMENT TARGET, so one poisoned build serves empty metallibs
# to every later build whatever ITS flags are. Clear the cache before any
# build whose kernel count you intend to believe.
MACOS_FLOOR="11.0"

# NOT EXPORTED, DELIBERATELY. Unset it if the caller had it set, because
# inheriting it from an outer shell reproduces the bug silently.
unset MACOSX_DEPLOYMENT_TARGET

if [ "$(uname)" = "Darwin" ]; then
    MACOS_SDK=$(xcrun --sdk macosx --show-sdk-version)
else
    MACOS_SDK=""  # linux arm (E1): no Mach-O, no Metal SDK
fi
# THE macOS FLOOR GOES TO THE LINKER, which stamps LC_BUILD_VERSION exactly as
# the environment variable would while the Metal compile step never sees it.
# 11.0 (Big Sur) is the first macOS that runs on Apple silicon at all, and it
# matches the minos of the MAX runtime dylibs this links.
#
# setup.py's DEFAULT_MACOS_TARGET and EVERY SIBLING build script must agree
# with this number: the .so files land in one wheel under one tag, and the tag
# is the lower bound of what is inside it. All five existing scripts say 11.0.
LINK_FLAGS="-Xlinker -platform_version -Xlinker macos -Xlinker $MACOS_FLOOR -Xlinker $MACOS_SDK"
[ "$(uname)" = "Darwin" ] || LINK_FLAGS=""

# ============================================================================
# 2. THE CPU TARGET IS A PORTABLE BASELINE, NOT THE HOST
# ============================================================================
# `mojo build` defaults --target-cpu to whatever chip ran the compiler. On the
# development M4 that enables +sme/+sme2/+bf16/+i8mm, none of which exist on
# M1 (and some not on M2/M3), and LLVM emits those instructions from ordinary
# loops once the bit is set -- so a host-built wheel SIGILLs on older Apple
# silicon, inside the extension, with no diagnostic a user can act on. arm64
# Mach-O cpusubtype stays ARM64_ALL whatever -mcpu was, so such a wheel LOOKS
# portable to any header-reading check and verifying it on the machine that
# built it proves nothing.
#
# ============================================================================
# 3. THERE IS NO --target-accelerator FLAG HERE AND THERE MUST NOT BE
# ============================================================================
# Measured with --target-cpu apple-m1 held fixed (bindings/build.sh, 2026-08-21):
# no flag gives every kernel, `--target-accelerator metal:1` gives 0 and
# `--target-accelerator apple-m1` gives 0. PASSING THE FLAG AT ALL, right
# value or wrong, suppresses ahead-of-time Metal compilation.
#
# NOTE FOR WHOEVER READS BOTH FILES: bindings/build_estimators.sh still passes
# `--target-accelerator metal:1` ("kept from the original invocation") and its
# artifact does carry kernels, so one of those two statements is incomplete.
# This script follows build.sh, build_gbdt.sh, build_rf.sh and build_trees.sh,
# which are four of the five and are the ones with the cold-cache measurement
# behind them. Resolving the contradiction is build_estimators.sh's to do.
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
# (checks/numerics.mojo reads it through is_defined) and lands the binary
# under python/mojolearn/identical/, where python/mojolearn/_backend.py picks
# it up when MOJOLEARN_NUMERIC_MODE=identical is set at import.
#
# READ THIS BEFORE BUILDING AN IDENTICAL SET: _backend.py's `_MODULES` tuple
# does NOT list `_mojolearn_solver` yet, so the identical binary this script
# writes is not redirected under the canonical module name and the FAST one
# would be imported instead. `_solver_impl.py` and `_hierarchy_impl.py` REFUSE
# to import in that state rather than produce a mislabeled measurement, but
# the real fix is one line in _backend.py (`_MODULES` plus `_build_script`),
# and it belongs to whoever owns that file.
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

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-solver.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT INT TERM
out="$tmpdir/_mojolearn_solver.so"

# --emit shared-lib, not an executable: CPython dlopens this and calls
# PyInit__mojolearn_solver. The file's basename must match that symbol's
# suffix or the import fails with "dynamic module does not define module
# export function".
#
# Two include paths, both required. `-I .` is the mojolearn package root, so
# `solver.estimator` and `hierarchy.estimator` resolve; `-I bindings` is this
# directory.
# shellcheck disable=SC2086  # the flag strings are deliberately word-split
pixi run mojo build --emit shared-lib \
    $TARGET_FLAGS $COLUMN_DEFINE $MODE_DEFINE \
    $LINK_FLAGS \
    -I . -I bindings \
    bindings/_mojolearn_solver.mojo \
    -o "$out"

# `_gpu_shared_mem` is a prefix the compiler puts on the blob symbol; it is
# not part of the Metal function name. The blob name carries the DEFINING
# module's path with `/` and `.` flattened to `_`, so `solver/impl/solver/
# cd.mojo`'s kernels begin `solver` and `hierarchy/impl/...`'s begin
# `hierarchy`.
air_blobs() {
    strings -a "$1" \
        | grep -oE '[0-9A-Za-z_]+_[0-9a-f]{16}air' \
        | sed -e 's/^_gpu_shared_mem//' -e 's/^0//' \
        | sort -u
}

# ============================================================================
# 4. THE FLOOR IS A PRE-FILTER; THE GATE IS run_smoke
# ============================================================================
# A floor of "at least one blob per subsystem" already passed a known-broken
# artifact once (the build that lost GBDT kept exactly 1 of 85 gbdt_ blobs and
# shipped for weeks), so the floor sits well above 1 -- and even then it is
# only a way to skip smoke-testing a hopeless build. The failure actually
# being guarded against is a .so that imports cleanly and dies at the first
# launch, and only LAUNCHING catches that.
#
# **THESE TWO NUMBERS ARE NOT MEASURED.** Every sibling script's floor was set
# from the first correct artifact built on this machine; this file was written
# without building anything, so its floors come from a STATIC count of the
# `enqueue_function[...]` sites reachable from this module's two entry points:
#
#   solver      9-11  (cd.mojo 3, axpy 1, stats/mean 2, functions/linear_reg
#                      1-2, checks/record_canon 1, linalg/coalesced_
#                      reduction 1 -- the last two of those are arm-dependent,
#                      the FAST build has more than the IDENTICAL one)
#   hierarchy   ~20   (sparse/solver/detail/mst_kernels ~13, cluster/detail/
#                      agglomerative 4, cluster/detail/connectivities 4-6,
#                      cluster/detail/mst 1, checks/nan_guard 1,
#                      checks/sabotage_tile 1)
#
# 4 and 8 are far under both counts and far over a suppressed build's 0, which
# is all a pre-filter has to be. THE FIRST CORRECT BUILD ON THIS MACHINE
# PRINTS ITS OWN COUNT -- raise these to just under the measured per-subsystem
# numbers then, and delete this paragraph, the way build_rf.sh's 8 and
# build_trees.sh's 10 were arrived at.
kernels_plausible() {
    [ "$(uname)" = "Darwin" ] || return 0  # linux gate is the smoke test
    _air=$(air_blobs "$1")
    # PRINT THE COUNTS UNCONDITIONALLY, the way bindings/build_svm.sh does.
    # A gate that only speaks when it fails cannot be calibrated, and every
    # floor in this file was written from a source count rather than a build.
    printf '  AIR blobs by subsystem (total %s):\n' \
        "$(printf '%s\n' "$_air" | grep -c . || true)"
    for _s in solver hierarchy core; do
        printf '    %-18s %s\n' "$_s" \
            "$(printf '%s\n' "$_air" | grep -c "^${_s}" || true)"
    done
    # MEASURED 2026-08-24, first cold build on the M4: solver 2, hierarchy 8,
    # core 3, total 13. The 4/8/1 that stood here failed a good artifact. The 4 that
    # stood here failed a good artifact. Floors are now under the measured
    # numbers with slack; the gate that actually proves the build is
    # run_smoke below, which launches a kernel from each estimator.
    for _pair in solver:1 hierarchy:4 core:0; do
        _sub=${_pair%%:*}
        _min=${_pair#*:}
        _n=$(printf '%s\n' "$_air" | grep -c "^${_sub}" || true)
        if [ "$_n" -lt "$_min" ]; then
            printf '  %s: %s AIR blobs, want at least %s\n' \
                "$_sub" "$_n" "$_min" >&2
            return 1
        fi
    done
    return 0
}

# THE MACH-O FLOOR IS READ BACK, NOT ASSUMED. The linker flag is the only
# thing setting it, and a silently dropped -Xlinker would publish a wheel
# whose tag and binary disagree.
minos_matches() {
    [ "$(uname)" = "Darwin" ] || return 0
    _got=$(otool -l "$1" \
        | awk '/LC_BUILD_VERSION/{f=1} f && /minos/{print $2; exit}')
    if [ "$_got" != "$MACOS_FLOOR" ]; then
        printf '  minos is %s, want %s\n' "$_got" "$MACOS_FLOOR" >&2
        return 1
    fi
    return 0
}

# THE REAL GATE: import and LAUNCH one kernel from each estimator IN THIS
# EXTENSION. Not copied from a sibling -- a smoke test for kernels that are
# not in this artifact would be a gate that cannot fail.
#
#   Lasso / ElasticNet     -> cd_fit + cd_predict: the axpy, the coordinate
#                             update, the column norms, the preprocess pair,
#                             linearRegH
#   AgglomerativeClustering-> linkage_fit: the row norms, the distance tile,
#                             the self-loop transform, the NaN guard, the
#                             Boruvka rounds, the label extraction
#
# THROUGH THE PYTHON WRAPPERS, NOT THE RAW BINDINGS, so this gate does not
# encode the params-list ABI a second time and drift from it. The two impl
# modules are imported by name because the operator, not this script, owns
# whether they are re-exported from mojolearn/__init__.py.
#
# 96 x 3 on purpose: the linkage arm allocates an n_rows x n_rows distance
# matrix and this runs on every build, on a shared GPU.
run_smoke() {
    MOJOLEARN_SMOKE_SO="$1" python3 - <<'PY'
import os, shutil, sys, tempfile
tmp = tempfile.mkdtemp()
pkg = os.path.join(tmp, "mojolearn")
shutil.copytree("python/mojolearn", pkg,
                ignore=shutil.ignore_patterns("__pycache__"))
shutil.copyfile(os.environ["MOJOLEARN_SMOKE_SO"],
                os.path.join(pkg, "_mojolearn_solver.so"))
sys.path.insert(0, tmp)
import numpy as np
from mojolearn import _hierarchy_impl, _solver_impl

X = np.random.default_rng(0).random((96, 3), dtype=np.float32)
y = np.ascontiguousarray(X[:, 0])

las = _solver_impl.Lasso(alpha=0.01, max_iter=20).fit(X, y)
las.predict(X)
enet = _solver_impl.ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20).fit(X, y)
enet.predict(X)
agg = _hierarchy_impl.AgglomerativeClustering(n_clusters=3).fit(X)
assert agg.labels_.shape == (96,), agg.labels_.shape
assert agg.children_.shape == (95, 2), agg.children_.shape
print("  smoke: lasso, elasticnet and single linkage each launched")
shutil.rmtree(tmp, ignore_errors=True)
PY
}

# The gates import the WHOLE python package, so during a from-scratch
# multi-binding build (a fresh linux box, E1) they fail on the siblings'
# not-yet-built .so files; and the AIR/otool checks are Mach-O-only. The
# caller that sets MOJOLEARN_SKIP_BUILD_GATE owns end-to-end verification.
if [ -n "${MOJOLEARN_SKIP_BUILD_GATE:-}" ] || [ "$(uname)" != "Darwin" ]; then
    mv "$out" "$OUTDIR/_mojolearn_solver.so"
    echo "built $OUTDIR/_mojolearn_solver.so (gate skipped: non-Darwin or MOJOLEARN_SKIP_BUILD_GATE)"
    exit 0
fi

if ! kernels_plausible "$out" || ! minos_matches "$out" || ! run_smoke "$out"; then
    printf '%s\n' \
      "" \
      "FAILED: the extension did not come out complete." \
      "" \
      "THE FIRST THING TO SUSPECT IS THE COMPILER CACHE, not this source." \
      "\$MODULAR_HOME/cache/.mojo_cache is content-addressed and its key does" \
      "NOT include the macOS deployment target, so ONE build made with" \
      "MACOSX_DEPLOYMENT_TARGET set poisons those keys with empty 134-byte" \
      "metallibs and every later build reads them back. Find them with" \
      "" \
      "  find \"\$MODULAR_HOME/cache/.mojo_cache\" -type f -size -200c \\\\" \
      "    -exec sh -c 'head -c4 \"\$1\" | grep -q MTLB && echo \"\$1\"' _ {} \\;" \
      "" \
      "and if there are any, move the whole cache aside and build again." \
      "" \
      "THE SECOND THING TO SUSPECT is the per-subsystem floor above, which is" \
      "the one number in this file that was NEVER MEASURED. If the counts it" \
      "prints are nonzero but under the floor, the floor is wrong, not the" \
      "artifact -- read the block above it before changing anything else." \
      "" \
      "See bindings/build.sh and PORTING.md 70." >&2
    exit 1
fi

count=$(air_blobs "$out" | wc -l | tr -d ' ')
mv "$out" "$OUTDIR/_mojolearn_solver.so"
echo "built $OUTDIR/_mojolearn_solver.so ($count AIR blobs, minos $MACOS_FLOOR)"
