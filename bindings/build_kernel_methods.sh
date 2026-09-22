#!/bin/sh
# Build the kernel methods (KernelRidge, Nystroem, RBFSampler) CPython extension into
# python/mojolearn/_mojolearn_kernel_methods.so. Run from anywhere; requires pixi.
# Mirrors bindings/build_gp.sh line for line except where this family is named.
#
# MOJOLEARN_BUILD_EXTRA_DEFINES (optional, empty by default): extra flags
# appended verbatim, word-split, to the `mojo build` command, exactly as in
# bindings/build.sh. For trial builds only (for example
# `-D MOJOLEARN_GEMM_ARM_TRIAL=1`, tools/gemm_ksplit_classical_leg.sh); a
# release build leaves it unset.
#
# THIS EXTENSION IS kernel_methods/ AND NOTHING ELSE (workstream D,
# 2026-09-14). It reaches cholesky/, the identical GEMM, decomposition/'s
# Jacobi and sign flip and svm/'s KernelParams through their own entry
# points; nothing else crosses.
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
# ONE TIER (DEVIATION 2490, 2026-09-10): ONLY THE TREE LANES SHIP fast AND
# deterministic (build_gbdt.sh, build_rf.sh, build_trees.sh). Every other
# binding, this one included, builds IDENTICAL only. Cross-vendor bitwise
# identity is the product; a fast tier is shipped only where it has a
# measured win over the opponent's own CPU, and outside trees it has none
# (python/mojolearn/_backend.py, `_TIERED`, has the numbers). Refusing
# here, by name, is what keeps this an unshipped tier rather than an
# unchecked one (CONTRIBUTING.md (Numeric modes) and section 8).
[ "${MOJOLEARN_NUMERIC_MODE:-identical}" = identical ] || {
    echo 'build_kernel_methods.sh: only the tree lanes (gbdt, rf, trees) ship fast and deterministic; every other binding builds MOJOLEARN_NUMERIC_MODE=identical only (DEVIATION 2490, 0.8.0).' >&2
    exit 2; }
MODE_DEFINE=""
OUTDIR="python/mojolearn"
if [ "${MOJOLEARN_NUMERIC_MODE:-identical}" = "identical" ]; then
    MODE_DEFINE="-D MOJOLEARN_NUMERIC_IDENTICAL=1"
    OUTDIR="python/mojolearn/identical"
    mkdir -p "$OUTDIR"
    export MOJOLEARN_SKIP_BUILD_GATE=1
elif [ "${MOJOLEARN_NUMERIC_MODE:-identical}" = "deterministic" ]; then
    # The MIDDLE tier: reproducible run to run on ONE device, with no
    # promise about a second one. It gets its own directory because it
    # is its own binary, PIN_DETERMINISM is comptime, so a
    # deterministic build is different code from both neighbours, not
    # the identical build with a flag turned down.
    MODE_DEFINE="-D MOJOLEARN_NUMERIC_DETERMINISTIC=1"
    OUTDIR="python/mojolearn/deterministic"
    mkdir -p "$OUTDIR"
    export MOJOLEARN_SKIP_BUILD_GATE=1
elif [ "${MOJOLEARN_NUMERIC_MODE:-identical}" != "fast" ]; then
    echo "MOJOLEARN_NUMERIC_MODE must be fast, deterministic or identical, got '$MOJOLEARN_NUMERIC_MODE'" >&2
    exit 2
fi
if [ -n "${MOJOLEARN_TARGET_COLUMN:-}" ]; then
    COLUMN_DEFINE="-D MOJOLEARN_COLUMN_$(printf %s "$MOJOLEARN_TARGET_COLUMN" | tr '[:lower:]' '[:upper:]')"
fi

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-kernel_methods.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT INT TERM
out="$tmpdir/_mojolearn_kernel_methods.so"

# --emit shared-lib, not an executable: CPython dlopens this and calls
# PyInit__mojolearn_kernel_methods. The FILE NAME must match that symbol's suffix or
# the import fails with "dynamic module does not define module export
# function".
# shellcheck disable=SC2086  # the flag strings are deliberately word-split
pixi run mojo build -j "${MOJOLEARN_COMPILE_JOBS:-2}" --emit shared-lib \
    $TARGET_FLAGS $COLUMN_DEFINE $MODE_DEFINE \
    ${MOJOLEARN_BUILD_EXTRA_DEFINES:-} \
    $LINK_FLAGS \
    -I . -I bindings \
    bindings/_mojolearn_kernel_methods.mojo \
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
# THE BINARY CHECKS RUN EVEN WHEN THE SMOKE CANNOT (2026-09-19).
#
# This script used to `export MOJOLEARN_SKIP_BUILD_GATE=1` for itself on the
# identical and deterministic tiers, and the skip below then turned off ALL
# THREE checks. So the gate that exists because a ZERO-KERNEL ARTIFACT
# shipped for hours ran only on `fast` builds -- never on the identical ones
# whose cross-vendor claim is the entire product.
#
# Only the kernel-launch smoke needs the rest of the package (it imports
# mojolearn, which during a multi-binding build reaches siblings that are not
# built yet). The AIR blob floor is `strings` on the artifact and the minos
# check is `otool`; neither imports anything, both cost milliseconds, and
# both catch exactly the failure that shipped. They run here unconditionally
# on Darwin, and only the smoke is skipped.
if [ "$(uname)" = "Darwin" ]; then

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
    # WHAT SHOULD BE IN HERE. This family's own kernels under its top-level
    # directory prefix; helpers imported cross-lane are printed but NOT
    # floored. THE FLOOR IS 1 AND THE FIRST COLD BUILD ON A BOX MUST RAISE
    # IT to two thirds of what it measures (build_gp.sh's history says why 1
    # is not a gate). The compile check of 2026-09-14 on one Apple M4 went
    # through `mojo build` directly and measured no blob count.
    _air=$(air_blobs "$out")
    _total=$(printf '%s\n' "$_air" | grep -c . || true)
    printf '  AIR blobs by subsystem (total %s):\n' "$_total"
    for _sub in kernel_methods cholesky gemm decomposition svm core; do
        printf '    %-18s %s\n' "$_sub" "$(printf '%s\n' "$_air" | grep -c "^${_sub}" || true)"
    done

    _failed=0
    for _pair in kernel_methods:1; do
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
    # THE REAL GATE: import and LAUNCH, through the Python wrappers (the reason
    # build_gp.sh gives: a hand-rolled ABI here would be a third copy of it).
    # Kept small because this runs on every build and the GPU is shared; the
    # family's own Python surface test asserts the bits.
    # ============================================================================
fi

if [ -n "${MOJOLEARN_SKIP_BUILD_GATE:-}" ] || [ "$(uname)" != "Darwin" ]; then
    mv "$out" "$OUTDIR/_mojolearn_kernel_methods.so"
    echo "built $OUTDIR/_mojolearn_kernel_methods.so (kernel-launch smoke skipped; binary checks ran)"
    exit 0
fi

MOJOLEARN_SMOKE_SO="$out" python3 - <<'PY'
import os, shutil, sys, tempfile
tmp = tempfile.mkdtemp()
pkg = os.path.join(tmp, "mojolearn")
shutil.copytree("python/mojolearn", pkg,
                ignore=shutil.ignore_patterns("__pycache__"))
shutil.copyfile(os.environ["MOJOLEARN_SMOKE_SO"],
                os.path.join(pkg, "_mojolearn_kernel_methods.so"))
sys.path.insert(0, tmp)
import numpy as np

from mojolearn import kernel_methods as km
rng = np.random.default_rng(0)
x = (rng.random((24, 3), dtype=np.float32) * 2.0 - 1.0).astype(np.float32)
y = (x[:, 0] - 0.5 * x[:, 1]).astype(np.float32)
m = km.KernelRidge(alpha=0.1, kernel="rbf", gamma=0.5).fit(x, y)
assert m.info_ == 0 and m.dual_coef_.shape == (24,)
p = m.predict(x[:8]); assert p.shape == (8,) and np.isfinite(np.asarray(p)).all()
ny = km.Nystroem(kernel="rbf", gamma=0.5, n_components=6, random_state=7).fit(x)
t = ny.transform(x[:5]); assert t.shape == (5, 6) and np.isfinite(np.asarray(t)).all()
rf = km.RBFSampler(gamma=0.5, n_components=8, random_state=1).fit(x)
f = rf.transform(x[:5]); assert f.shape == (5, 8) and np.isfinite(np.asarray(f)).all()
try:
    km.KernelRidge(alpha=-1.0).fit(x, y)
except Exception as exc:
    assert "alpha" in str(exc), exc
else:
    raise AssertionError("alpha=-1 was ACCEPTED")
print("  smoke: KernelRidge fit/predict (rbf), Nystroem and RBFSampler fit/transform, negative alpha refused")
shutil.rmtree(tmp, ignore_errors=True)
PY

mv "$out" "$OUTDIR/_mojolearn_kernel_methods.so"
echo "built $OUTDIR/_mojolearn_kernel_methods.so ($_total AIR blobs, minos $MACOS_FLOOR)"
