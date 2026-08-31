#!/bin/sh
# Build the bit-identical FP32 GEMM extension into
# python/mojolearn/_mojolearn_linalg.so. Run from anywhere; requires pixi.
#
# This is profile `mojolearn.identical.gemm.fp32.v1`
# (gemm/IDENTICAL_FP32_CONTRACT.md) reaching Python. It is its own extension
# for the reason bindings/build.sh gives for all of them -- an independently
# changing binding must not become a merge point -- and for one more that is
# particular to this module: it is the only thing in the package whose VALUE
# is a bit-level claim, so it must be possible to build, gate and ship it
# without touching anything that is not the claim.
#
# THE STRUCTURE BELOW IS bindings/build.sh's AND EVERY PART OF IT IS THERE FOR
# A MEASURED REASON. Read that file before changing anything here. The three
# that cost this repository the most:
#
#   1. MACOSX_DEPLOYMENT_TARGET IN THE ENVIRONMENT SUPPRESSES AHEAD-OF-TIME
#      METAL COMPILATION ENTIRELY. Set, `mojo build` emits an empty 134-byte
#      metallib per kernel and embeds nothing; the extension then imports
#      cleanly and dies at the first launch with "Failed to create Metal
#      function". Measured on a cold cache, one variable at a time: set = 0
#      AIR blobs, unset = 141. The macOS floor is passed to the LINKER
#      instead, which stamps LC_BUILD_VERSION exactly as the variable would
#      while the Metal compile step never sees it.
#
#   2. THE COMPILER CACHE IS WHY EVERY EARLIER MEASUREMENT DISAGREED.
#      $MODULAR_HOME/cache/.mojo_cache is content-addressed and ITS KEY DOES
#      NOT INCLUDE THE DEPLOYMENT TARGET, so one poisoned build serves empty
#      metallibs to every later build whatever ITS flags are. Clear the cache
#      before any build whose kernel count you intend to believe.
#
#   3. --target-cpu IS PINNED TO apple-m1, NOT LEFT AT THE HOST. `mojo build`
#      otherwise targets whatever chip ran the compiler; on this box that is
#      apple-m4 with +sme, +sme2, +bf16 and +i8mm, none of which exist on M1,
#      and LLVM emits those instructions from ordinary loops once the bit is
#      set. arm64 Mach-O cpusubtype stays ARM64_ALL whatever -mcpu was, so
#      such a wheel LOOKS portable to any header-reading check and SIGILLs
#      inside the extension on an older Mac.
#
# ONE DELIBERATE DIFFERENCE FROM bindings/build_estimators.sh: there is no
# `--target-accelerator metal:1` here. build_estimators.sh carries that flag
# with the comment "kept from the original invocation", and build.sh carries
# the MEASUREMENT that contradicts it -- 2026-08-21, --target-cpu apple-m1
# held fixed: no flag gives every kernel, `--target-accelerator metal:1` gives
# 0, `--target-accelerator apple-m1` gives 0, and PASSING THE FLAG AT ALL,
# right value or wrong, suppresses ahead-of-time Metal compilation. A
# measurement beats an inheritance, so this script follows build.sh. The two
# sibling scripts disagreeing is a real finding and it is in this lane's
# hand-off note; it is not resolved here, because build_estimators.sh is not
# this lane's file.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$here"

mkdir -p python/mojolearn

# setup.py's DEFAULT_MACOS_TARGET must equal this, and every bindings/build*.sh
# must agree with every other, because all the .so files land in ONE wheel
# under ONE tag and the tag is the lower bound of what is inside it.
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
if [ -n "${MOJOLEARN_TARGET_COLUMN:-}" ]; then
    COLUMN_DEFINE="-D MOJOLEARN_COLUMN_$(printf %s "$MOJOLEARN_TARGET_COLUMN" | tr '[:lower:]' '[:upper:]')"
fi

# THE NUMERIC MODE IS A BUILD DEFINE, NOT A SOURCE FLIP.
#
# AND FOR THIS EXTENSION IT IS THE PRODUCT. Every other binding is an
# estimator whose FAST build is a perfectly good estimator; this one's FAST
# build is a fast GEMM that makes NO identity claim at all, and
# python/mojolearn/_linalg_impl.py refuses to deliver the profile's guarantee
# unless the IDENTICAL binary is the one that loaded. So:
#
#     bash bindings/build_linalg.sh
#         -> python/mojolearn/_mojolearn_linalg.so           FAST
#     MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_linalg.sh
#         -> python/mojolearn/identical/_mojolearn_linalg.so IDENTICAL
#
# BUILD BOTH. A user who sets MOJOLEARN_NUMERIC_MODE=identical without the
# second build gets an ImportError naming this script, which is the right
# failure and still a failure.
MODE_DEFINE=""
OUTDIR="python/mojolearn"
if [ "${MOJOLEARN_NUMERIC_MODE:-fast}" = "identical" ]; then
    MODE_DEFINE="-D MOJOLEARN_NUMERIC_IDENTICAL=1"
    OUTDIR="python/mojolearn/identical"
    mkdir -p "$OUTDIR"
    # The smoke gate below imports the FAST package, so it is skipped for an
    # identical build. THAT BUILD'S GATE IS THE LANE'S OWN, and it is not
    # optional for this extension:
    #     tools/with_identical_mode.sh pixi run mojo run -I . \
    #         gemm/checks/gemm_device_check.mojo
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

# --emit shared-lib, not an executable: CPython dlopens this and calls
# PyInit__mojolearn_linalg. The name of the file must match that symbol's
# suffix or the import fails with "dynamic module does not define module
# export function", which is the least helpful error in the toolchain.
tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-linalg.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT INT TERM
out="$tmpdir/_mojolearn_linalg.so"

# `-I .` is the mojolearn package root, so `gemm.host_entry` and
# `original.numerics` resolve. `-I bindings` is this directory.
# shellcheck disable=SC2086  # the flag strings are deliberately word-split
pixi run mojo build --emit shared-lib \
    $TARGET_FLAGS $COLUMN_DEFINE $MODE_DEFINE \
    $LINK_FLAGS \
    -I . -I bindings \
    bindings/_mojolearn_linalg.mojo \
    -o "$out"

# `_gpu_shared_mem` is a prefix the compiler puts on the blob symbol; it is
# not part of the Metal function name.
air_blobs() {
    strings -a "$1" \
        | grep -oE '[0-9A-Za-z_]+_[0-9a-f]{16}air' \
        | sed -e 's/^_gpu_shared_mem//' -e 's/^0//' \
        | sort -u
}

if [ -n "${MOJOLEARN_SKIP_BUILD_GATE:-}" ] || [ "$(uname)" != "Darwin" ]; then
    mv "$out" "$OUTDIR/_mojolearn_linalg.so"
    echo "built $OUTDIR/_mojolearn_linalg.so (gate skipped: non-Darwin or MOJOLEARN_SKIP_BUILD_GATE)"
    exit 0
fi

# ============================================================================
# THE AIR-BLOB FLOOR: 8 gemm-prefixed blobs, AND IT IS A FILTER, NOT A PROOF.
# ============================================================================
#
# build.sh's lesson, learned twice: a "does it have at least one" floor was
# SATISFIED by a known-broken artifact -- the build that lost GBDT kept
# exactly 1 of 85 gbdt_ blobs. So the floor has to sit well above 1, and even
# then it only lets a hopeless build skip the smoke test.
#
# WHERE 8 COMES FROM. Ten distinct kernel instantiations are reachable from
# `identical_gemm_with_plan` in gemm/checks/gemm_identical.mojo:
#
#     identical_gemm_flat_kernel                            1
#     identical_gemm_tiled_kernel[TM, TN, KS], five of them:
#         16/16/32, 8/32/32, 32/8/16, 16/16/8, 4/4/32       5
#     identical_gemm_leaf_kernel        (SPLITK level 0)    1
#     identical_gemm_fold_kernel        (SPLITK fused fold) 1
#     identical_gemm_fold_level_kernel  (STAGED, one level) 1
#     identical_gemm_emit_kernel        (STAGED output)     1
#
# 8 is that count with two of slack, so a compiler that merges or drops one
# instantiation does not fail a good build, while a suppressed build (0) and a
# nearly-suppressed one both fail. It is deliberately NOT 10: a floor tuned to
# the exact count today becomes a false alarm the next time an execution plan
# is added or removed, and the execution plan is EXPLICITLY outside the
# profile version (contract preamble) so it is expected to move.
#
# The count is over blobs whose name starts `gemm`, not over all blobs, which
# is build.sh's per-subsystem shape rather than build_estimators.sh's bare
# total. A bare total can be met by blobs the MAX runtime brought along.
_air=$(air_blobs "$out")
_gemm=$(printf '%s\n' "$_air" | grep -c '^gemm' || true)
_total=$(printf '%s\n' "$_air" | grep -c . || true)
if [ "$_gemm" -lt 8 ]; then
    printf 'FAILED: %s gemm AIR blobs (of %s total), want at least 8.\n' \
        "$_gemm" "$_total" >&2
    printf '\nThe blobs that ARE in the artifact:\n' >&2
    printf '%s\n' "$_air" | sed 's/^/    /' >&2
    printf '\nIf this is 0, suspect the environment and the cache before the\n' >&2
    printf 'source: MACOSX_DEPLOYMENT_TARGET set anywhere in the environment,\n' >&2
    printf 'then empty 134-byte metallibs in $MODULAR_HOME/cache/.mojo_cache:\n' >&2
    printf '\n  find "$MODULAR_HOME/cache/.mojo_cache" -type f -size -200c \\\n' >&2
    printf "    -exec sh -c 'head -c4 \"\$1\" | grep -q MTLB && echo \"\$1\"' _ {} \;\n\n" >&2
    printf 'If it is nonzero but the names above do not start with "gemm",\n' >&2
    printf 'the prefix in this check is wrong and the fix is one line here.\n' >&2
    exit 1
fi

# THE MACH-O FLOOR IS READ BACK, NOT ASSUMED. The linker flag is the only
# thing setting it, and a silently dropped -Xlinker would publish a wheel
# whose tag and binary disagree.
got=$(otool -l "$out" | awk '/LC_BUILD_VERSION/{f=1} f && /minos/{print $2; exit}')
if [ "$got" != "$MACOS_FLOOR" ]; then
    printf 'FAILED: minos is %s, want %s.\n' "$got" "$MACOS_FLOOR" >&2
    exit 1
fi

# ============================================================================
# THE REAL GATE: LAUNCH THESE KERNELS, AND CHECK THAT THIS MODULE'S POLICY
# ACTUALLY FIRES.
# ============================================================================
#
# Every broken build in this bug's history imported fine and died at the first
# launch, so nothing short of launching proves anything. The shapes below
# reach EVERY EXECUTION PLAN `choose_gemm_plan` can return -- all five of them
# -- because a smoke test that only ever hits one plan prices the others at
# zero, and the execution plan is the part of this system that is EXPLICITLY
# free to change (contract preamble), so it is the part most likely to break
# quietly:
#
#     8 x 8 x 64      m,n < 16, n < 32, m < 32     -> PLAN_FLAT
#     64 x 64 x 64    m,n >= 16                    -> PLAN_TILE_16_16_32
#     32 x 32 x 1024  m*n <= 4096 and P = 8 >= 4   -> PLAN_SPLITK
#     8 x 64 x 64     m < 16, n >= 32              -> PLAN_TILE_8_32_32
#     16 x 16 x 300   m,n >= 16; P = 3 ragged      -> PLAN_TILE_16_16_32
#     32 x 1 x 24     the gemv below, m >= 32      -> PLAN_TILE_32_8_16
#
# `16 x 16 x 300` is the contract's own named ragged odd-P shape (section
# 12.1: "(k, L) = (300, 128), giving P = 3 with a ragged 44-element last
# leaf"), and `32 x 32 x 1024` is the only one that launches the leaf and fold
# kernels at all -- without it four of the ten blobs counted above are never
# executed.
#
# The three POLICY assertions at the end are reach checks, not output checks.
# DEVIATION 911's refusal, DEVIATION 913's refusal and the dtype refusal are
# each a branch that a passing build could contain and never take; a gate that
# does not make them fire cannot tell a live guard from an inert one.
#
# Sizes are tiny on purpose: this runs on every build and the GPU is shared.
MOJOLEARN_SMOKE_SO="$out" python3 - <<'PY'
import os, shutil, sys, tempfile
tmp = tempfile.mkdtemp()
pkg = os.path.join(tmp, "mojolearn")
shutil.copytree("python/mojolearn", pkg,
                ignore=shutil.ignore_patterns("__pycache__"))
shutil.copyfile(os.environ["MOJOLEARN_SMOKE_SO"],
                os.path.join(pkg, "_mojolearn_linalg.so"))
sys.path.insert(0, tmp)
import numpy as np

# Through the Python wrapper, not the raw binding: the extension's entry point
# takes bare addresses plus a packed params list, and a hand-rolled call here
# would encode that ABI a second time and drift from it. Imported as a
# submodule so this gate does not depend on the package's export list --
# `mojolearn.linalg` is the operator's re-export and may not be wired yet.
from mojolearn import _linalg_impl as linalg

assert linalg.numeric_mode() == "fast", linalg.numeric_mode()
assert linalg.profile()["identity_claimed"] is False
assert linalg.profile()["profile"] == "mojolearn.identical.gemm.fp32.v1"

rng = np.random.default_rng(0)
plans = 0
for (m, n, k, label) in (
    (8, 8, 64, "FLAT"),
    (64, 64, 64, "TILE_16_16_32"),
    (32, 32, 1024, "SPLITK, P=8"),
    (8, 64, 64, "TILE_8_32_32"),
    (16, 16, 300, "ragged P=3, 44-element last leaf"),
):
    a = rng.standard_normal((m, k), dtype=np.float32)
    b = rng.standard_normal((k, n), dtype=np.float32)
    # POISONED OUTPUT. A workspace sized for the wrong plan left whole
    # regions of C as +0.0 once in this lane's history and only a shape
    # change made it visible; NaN in every cell means any cell the device did
    # not write is loud instead of plausible.
    out = np.full((m, n), np.nan, dtype=np.float32)
    got = linalg.matmul(a, b, out=out, identical=False)
    assert got is out
    assert not np.isnan(out).any(), f"{label}: unwritten cells"
    # allclose and NOT array_equal: the profile is a different summation
    # order from numpy's and is not trying to match it (contract 7.4).
    assert np.allclose(out, a @ b, rtol=1e-3, atol=1e-3), label
    plans += 1

# The other two orientations, same values, so a wrong index expression shows
# up as a wrong number rather than as a shape error. Contract section 3
# requires all three to agree bit for bit on the same logical matrices.
a = rng.standard_normal((32, 24), dtype=np.float32)
b = rng.standard_normal((24, 16), dtype=np.float32)
nn = linalg.matmul(a, b, identical=False)
nt = linalg.matmul(a, np.ascontiguousarray(b.T), transpose_b=True,
                   identical=False)
tn = linalg.matmul(np.ascontiguousarray(a.T), b, transpose_a=True,
                   identical=False)
# allclose, NOT array_equal, AND THAT IS DELIBERATE. Contract section 3 makes
# bit-for-bit agreement of the three orientations a REQUIREMENT of the
# profile, and it is gated -- `check_orientations_agree` in
# gemm/checks/gemm_oracle_check.mojo, and `check_device_matches_oracle`
# across 62 shapes in gemm_device_check.mojo. But this build is FAST, and
# gemm_identical.mojo's own header says check_device_matches_oracle "asserts
# under IDENTICAL and reports under FAST". Nothing in this tree asserts
# cross-orientation bit equality under FAST, so this gate does not either.
# What it catches is what it is for: a wrong index expression in the wrapper's
# op mapping, which gives a completely different matrix and not a last bit.
assert np.allclose(nn, nt, rtol=1e-5, atol=1e-5), "OP_NN and OP_NT disagree"
assert np.allclose(nn, tn, rtol=1e-5, atol=1e-5), "OP_NN and OP_TN disagree"

# gemv: OP_NT at n == 1, and NOT a fourth operation (contract 0.1). The one
# shape where core/gemm.mojo measured 63 of 64 output rows left unwritten.
# It is also the fifth execution plan: m=32, n=1, k=24 gives P=1 and
# `choose_gemm_plan` falls to PLAN_TILE_32_8_16.
v = rng.standard_normal((1, 24), dtype=np.float32)
z = linalg.matmul(a, v, transpose_b=True, identical=False)
assert z.shape == (32, 1) and not np.isnan(z).any()

# POLICY REACH. Each of these is a branch that could be present and never
# taken; a gate that does not make it fire cannot see an inert guard.
try:
    linalg.matmul(a, b)                       # identical=True by default
except RuntimeError as e:
    assert "FAST build" in str(e), e
else:
    raise AssertionError(
        "DEVIATION 911 IS INERT: matmul returned a FAST product under its "
        "default identical=True")
try:
    linalg.matmul(a, b, transpose_a=True, transpose_b=True, identical=False)
except ValueError as e:
    assert "three operations" in str(e), e
else:
    raise AssertionError("DEVIATION 913 IS INERT: a.T @ b.T was accepted")
try:
    linalg.matmul(a.astype(np.float64), b.astype(np.float64), identical=False)
except TypeError as e:
    assert "float32" in str(e), e
else:
    raise AssertionError("the dtype refusal is inert: float64 was accepted")

print(f"  smoke: {plans} shapes over all five dispatchable execution plans, "
      "three orientations agree, gemv ran, all three refusals fired")
shutil.rmtree(tmp, ignore_errors=True)
PY

mv "$out" "$OUTDIR/_mojolearn_linalg.so"
echo "built $OUTDIR/_mojolearn_linalg.so ($_gemm gemm AIR blobs of $_total, minos $MACOS_FLOOR)"
