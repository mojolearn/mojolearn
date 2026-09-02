#!/bin/sh
# Build the Mamba block CPython extension into
# python/mojolearn/_mojolearn_mamba.so. Run from anywhere; requires pixi.
#
# THIS EXTENSION IS mamba/ AND NOTHING ELSE. It is the FOURTEENTH binding,
# closing mamba/FEATURE_PARITY.md's "PyPI surface: NONE EXISTS" row, and it
# is separate from every sibling for the reason the estimators binding's
# header gives: an independently changing binding must not become a merge
# point. The lane reaches the identical GEMM and checks/numerics through
# their own entry points; nothing else crosses. All fourteen binaries land
# in one wheel.
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
# NO SABOTAGE DEFINE EVER BELONGS ON THIS COMMAND LINE. The
# MOJOLEARN_MAMBA_SABOTAGE_* / MOJOLEARN_MAMBA2_SABOTAGE_* /
# MOJOLEARN_MAMBA3_SABOTAGE_* defines are for the lane gates
# (mamba/checks/); the binding's PyInit aborts by name if one is compiled
# in, so a sabotaged build fails at import, loudly, rather than serving
# sabotaged bits under a green label.
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
# NOTE FOR THE OPERATOR: `_mojolearn_mamba` is listed in
# python/mojolearn/_backend.py's `_MODULES` and in its `_build_script`
# dict, both, in the same commit that added this script (DEVIATION 869 is
# what happens when only one of the two is filled in). `_mamba_impl.py`
# therefore carries NO private mode-aware loader; it uses
# `NumericModeMixin._bind` and cross-checks `mamba_numeric_mode()` against
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

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-mamba.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT INT TERM
out="$tmpdir/_mojolearn_mamba.so"

# --emit shared-lib, not an executable: CPython dlopens this and calls
# PyInit__mojolearn_mamba. The FILE NAME must match that symbol's suffix or
# the import fails with "dynamic module does not define module export
# function".
# shellcheck disable=SC2086  # the flag strings are deliberately word-split
pixi run mojo build --emit shared-lib \
    $TARGET_FLAGS $COLUMN_DEFINE $MODE_DEFINE \
    $LINK_FLAGS \
    -I . -I bindings \
    bindings/_mojolearn_mamba.mojo \
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
    mv "$out" "$OUTDIR/_mojolearn_mamba.so"
    echo "built $OUTDIR/_mojolearn_mamba.so (gate skipped: non-Darwin or MOJOLEARN_SKIP_BUILD_GATE)"
    exit 0
fi

# ============================================================================
# A FLOOR, NOT A PROOF. RAISED FROM THE PLACEHOLDER ON THE FIRST COLD BUILD.
# ============================================================================
#
# THE FIRST COLD BUILD (2026-09-01, fast tier) MEASURED 16 mamba-prefix AIR
# blobs (24 total with gemm's 8); the floor below is two thirds of that,
# rounded down: 10. THAT MEASUREMENT PREDATES THE MAMBA-3 ENTRY POINTS
# (added later on 2026-09-01: mamba3_forward / mamba3_decode_step pull in
# the m3_* kernel set of mamba3.mojo + mamba3_siso.mojo), so the observed
# count grew exactly as predicted: THE FIRST POST-MAMBA3 BUILD (2026-09-01
# evening, fast tier) MEASURED 24 mamba-prefix AIR blobs (32 total with
# gemm's 8), and the floor below is two thirds of that, rounded down: 16.
# The placeholder-then-raise discipline is --
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
# WHAT SHOULD BE IN HERE. `mamba/` launches the Mamba-1 kernels
# (mamba_rms_norm_kernel, causal_conv1d_fn_kernel + window,
# mamba_a_from_a_log_kernel, split_x_proj, softplus_delta, z_gate,
# residual_add, the selective scan) and the Mamba-2 set (m2_conv,
# m2_assemble_*, m2_dt, m2_buffer_update, m2_skip, m2_gate, and the SSD
# core's m2_discretize / m2_chunk_cumsum / m2_seg_l / m2_cb_g / m2_ydiag /
# m2_decay / m2_cstate / m2_statepass / m2_yoff_y) and, since the mamba3
# entry points landed, the Mamba-3 set (m3_a_dt, m3_bcnorm,
# m3_assemble_*, m3_buffer_update and the SISO core's m3_* kernels in
# mamba3_siso.mojo). The subsystem prefix
# is the top-level directory the kernel's module lives in, so `mamba` is
# the floored prefix. gemm/, core/ and checks/ blobs are printed but NOT
# floored: build_training.sh records that the identical GEMM's blobs can
# carry a non-gemm prefix, and whether a cross-lane helper leaves a blob
# under its own prefix is not something to assert before it has been seen
# once.
_air=$(air_blobs "$out")
_total=$(printf '%s\n' "$_air" | grep -c . || true)
printf '  AIR blobs by subsystem (total %s):\n' "$_total"
for _sub in mamba gemm core checks; do
    printf '    %-18s %s\n' "$_sub" "$(printf '%s\n' "$_air" | grep -c "^${_sub}" || true)"
done

_failed=0
for _pair in mamba:16; do
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
# second time and drift from it. The whole point of the addrs/params lists
# being written out in two places is that there are two, not three.
#
# THE MAMBA ROWS ARE MAMBA'S. Copying a sibling script's smoke rows would
# be a gate that cannot fail, since none of those kernels are in this
# artifact.
#
#   Mamba1 forward           -> RMSNorm, in_proj GEMM, the conv + window,
#                               x_proj/dt_proj GEMMs, softplus, the scan,
#                               gate, out_proj GEMM, residual
#   Mamba1 step x4           -> the same spelling at L = 1 with the state
#                               carried; compared LOOSELY to the prefill
#                               rows (the bitwise decode==prefill claim
#                               belongs to the lane gate and to
#                               test_mamba_surface.py under identical, not
#                               to a fast-tier smoke)
#   float64 x                -> the dtype refusal, BY NAME, in Python
#   d_model = 40             -> the multiple-of-32 refusal, by name (the
#                               wrapper's copy of Mamba2Dims.of's rule)
#   Mamba2 forward L=1       -> the whole M2 chain incl. one padded chunk;
#                               buf_len bookkeeping read back (1)
#   Mamba2 step              -> prefill resumption at L = 1; buf_len -> 2
#   Mamba3 forward L=1       -> norm, in_proj GEMM, A/dt, B/C norms, the
#                               SISO core (rotation, segsum, state pass),
#                               gate, out_proj GEMM, residual; buf_len 1
#   Mamba3 step              -> prefill resumption at L = 1; buf_len -> 2;
#                               the four reports' shapes; a pending
#                               Input_States continuation consumed
#
# Kept small on purpose (d_model 8 and 32, B <= 2, L <= 4) because this
# runs on every build and the GPU is shared. The recovery assertions here
# are DELIBERATELY LOOSE and are a smoke test, not the gate:
# python/mojolearn/tests/test_mamba_surface.py is where the corpus
# tolerances and the identical-tier bitwise arms are asserted.
MOJOLEARN_SMOKE_SO="$out" python3 - <<'PY'
import os, shutil, sys, tempfile
tmp = tempfile.mkdtemp()
pkg = os.path.join(tmp, "mojolearn")
shutil.copytree("python/mojolearn", pkg,
                ignore=shutil.ignore_patterns("__pycache__"))
shutil.copyfile(os.environ["MOJOLEARN_SMOKE_SO"],
                os.path.join(pkg, "_mojolearn_mamba.so"))
sys.path.insert(0, tmp)
import numpy as np

from mojolearn import _mamba_impl

rng = np.random.default_rng(0)

def u(shape, lo, hi):
    return (lo + (hi - lo) * rng.random(shape)).astype(np.float32)

# ---- Mamba-1: d_model 8 (di 16, dt_rank 1, xr 33), B 2, L 4.
dm, di, r = 8, 16, 1
w1 = {
    "norm.weight": u((dm,), 0.5, 1.5),
    "in_proj.weight": u((2 * di, dm), -0.25, 0.25),
    "conv1d.weight": u((di, 1, 4), -0.5, 0.5),
    "conv1d.bias": u((di,), -0.125, 0.125),
    "x_proj.weight": u((r + 32, di), -0.125, 0.125),
    "dt_proj.weight": u((di, r), -1.0, 1.0),
    "dt_proj.bias": u((di,), -7.0, -2.0),
    "A_log": u((di, 16), 0.0, 2.75),
    "D": u((di,), 0.5, 1.5),
    "out_proj.weight": u((dm, di), -0.125, 0.125),
}
b, l = 2, 4
x = u((b, l, dm), -2.0, 2.0)
blk = _mamba_impl.Mamba1Block(w1)
y = blk.forward(x)
assert y.shape == (b, l, dm), y.shape
assert np.isfinite(y).all()

st = blk.allocate_state(b)
worst = 0.0
for t in range(l):
    yt = blk.step(x[:, t:t + 1, :], st)
    worst = max(worst, float(np.max(np.abs(yt[:, 0, :] - y[:, t, :]))))
assert worst < 1e-4, ("decode drifted %g from prefill; one spelling "
                      "serves both paths, so even a loose smoke bound "
                      "should hold" % worst)

# ---- the dtype refusal, BY NAME, before any address is taken.
try:
    blk.forward(x.astype(np.float64))
except TypeError as exc:
    assert "float32" in str(exc) and "float64" in str(exc), exc
else:
    raise AssertionError("a float64 x was ACCEPTED")

# ---- the multiple-of-32 rule, refused by name. The wrapper refuses it
# (it cannot even size the state buffers otherwise); Mamba2Dims.of's own
# refusal stays the authority for raw-binding callers.
bad = {n: np.zeros(1, np.float32) for n in (
    "block_norm.weight", "in_proj.weight", "conv1d.weight", "conv1d.bias",
    "dt_bias", "A_log", "D", "norm.weight", "out_proj.weight")}
bad["block_norm.weight"] = np.ones(40, np.float32)
try:
    _mamba_impl.Mamba2Block(bad)
except ValueError as exc:
    assert "multiple of" in str(exc), exc
else:
    raise AssertionError("d_model = 40 was ACCEPTED")

# ---- Mamba-2: d_model 32 (di 64, H 1, CD 320), B 1.
dm2, di2, h2, cd2 = 32, 64, 1, 320
w2 = {
    "block_norm.weight": u((dm2,), 0.5, 1.5),
    "in_proj.weight": u((2 * di2 + 256 + h2, dm2), -0.09, 0.09),
    "conv1d.weight": u((cd2, 1, 4), -0.5, 0.5),
    "conv1d.bias": u((cd2,), -0.125, 0.125),
    "dt_bias": u((h2,), -7.0, -2.0),
    "A_log": u((h2,), 0.0, 2.75),
    "D": u((h2,), 0.5, 1.5),
    "norm.weight": u((di2,), 0.5, 1.5),
    "out_proj.weight": u((dm2, di2), -0.09, 0.09),
}
blk2 = _mamba_impl.Mamba2Block(w2)
st2 = blk2.allocate_state(1)
x2 = u((1, 1, dm2), -2.0, 2.0)
y2 = blk2.forward(x2, st2)
assert y2.shape == (1, 1, dm2), y2.shape
assert np.isfinite(y2).all()
assert st2.buffered_tokens == 1, st2.buffered_tokens
y3 = blk2.step(u((1, 1, dm2), -2.0, 2.0), st2)
assert y3.shape == (1, 1, dm2) and np.isfinite(y3).all()
assert st2.buffered_tokens == 2, st2.buffered_tokens
assert blk2.h_last_.shape == (1, h2, 64, 128)

# ---- Mamba-3: d_model 32 (di 64, H 1, N 128, P 64, R 32, dip 419), B 1.
dm3, di3, h3, dip3 = 32, 64, 1, 419
w3 = {
    "block_norm.weight": u((dm3,), 0.5, 1.5),
    "in_proj.weight": u((dip3, dm3), -0.18, 0.18),
    "dt_bias": u((h3,), -7.0, -2.0),
    "B_norm.weight": u((128,), 0.5, 1.5),
    "C_norm.weight": u((128,), 0.5, 1.5),
    "B_bias": u((h3, 128), 0.9, 1.1),
    "C_bias": u((h3, 128), 0.9, 1.1),
    "D": u((h3,), 0.5, 1.5),
    "out_proj.weight": u((dm3, di3), -0.125, 0.125),
}
blk3 = _mamba_impl.Mamba3Block(w3)
st3 = blk3.allocate_state(1)
x3 = u((1, 1, dm3), -2.0, 2.0)
y3f = blk3.forward(x3, st3)
assert y3f.shape == (1, 1, dm3), y3f.shape
assert np.isfinite(y3f).all()
assert st3.buffered_tokens == 1, st3.buffered_tokens
y3s = blk3.step(u((1, 1, dm3), -2.0, 2.0), st3)
assert y3s.shape == (1, 1, dm3) and np.isfinite(y3s).all()
assert st3.buffered_tokens == 2, st3.buffered_tokens
assert blk3.h_last_.shape == (1, h3, 64, 128)
assert blk3.k_last_.shape == (1, h3, 128)
assert blk3.v_last_.shape == (1, h3, 64)
assert blk3.theta_last_.shape == (1, h3, 32)
# a pending Input_States continuation on a FRESH state is consumed.
st3c = blk3.allocate_state(1)
st3c.set_input_states(
    u((1, h3, 32), 0.0, 6.25), u((1, h3, 64, 128), -0.5, 0.5),
    u((1, h3, 128), -0.5, 0.5), u((1, h3, 64), -0.5, 0.5))
assert st3c.pending is True
y3c = blk3.forward(x3, st3c)
assert np.isfinite(y3c).all() and st3c.pending is False

print("  smoke: Mamba1 forward + 4 decode steps (worst |step-prefill| "
      "%.2e), float64 refused by name, Mamba2 forward + step with "
      "buf_len 1 -> 2 and the h_last report, Mamba3 forward + step with "
      "buf_len 1 -> 2, the four reports and a consumed Input_States "
      "continuation" % worst)
shutil.rmtree(tmp, ignore_errors=True)
PY

mv "$out" "$OUTDIR/_mojolearn_mamba.so"
echo "built $OUTDIR/_mojolearn_mamba.so ($_total AIR blobs, minos $MACOS_FLOOR)"
