#!/bin/sh
# Build the transformer block CPython extension into
# python/mojolearn/_mojolearn_transformer.so. Run from anywhere; requires
# pixi.
#
# THIS EXTENSION IS transformer/ AND NOTHING ELSE. It is the FIFTEENTH
# binding, giving the certified transformer block (profile
# `mojolearn.identical.transformer.fp32.v1`) its first Python symbol, and it
# is separate from every sibling for the reason the estimators binding's
# header gives: an independently changing binding must not become a merge
# point. The lane reaches the identical GEMM, checks/numerics and the mamba
# lane's residual_add_kernel through their own entry points; nothing else
# crosses. All fifteen binaries land in one wheel.
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
# MOJOLEARN_TRANSFORMER_SABOTAGE_* / MOJOLEARN_BATCHINV_SABOTAGE_* defines
# are for the lane gates (transformer/checks/); the binding's PyInit aborts
# by name if one is compiled in, so a sabotaged build fails at import,
# loudly, rather than serving sabotaged bits under a green label.
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
# NOTE FOR THE OPERATOR: `_mojolearn_transformer` is listed in
# python/mojolearn/_backend.py's `_MODULES` and in its `_build_script`
# dict, both, in the same commit that added this script (DEVIATION 869 is
# what happens when only one of the two is filled in). `_transformer_impl.py`
# therefore carries NO private mode-aware loader; it uses
# `NumericModeMixin._bind` and cross-checks `transformer_numeric_mode()`
# against the tier the package resolved, so an identical run cannot
# silently get the FAST binary.
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

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-transformer.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT INT TERM
out="$tmpdir/_mojolearn_transformer.so"

# --emit shared-lib, not an executable: CPython dlopens this and calls
# PyInit__mojolearn_transformer. The FILE NAME must match that symbol's
# suffix or the import fails with "dynamic module does not define module
# export function".
# shellcheck disable=SC2086  # the flag strings are deliberately word-split
pixi run mojo build --emit shared-lib \
    $TARGET_FLAGS $COLUMN_DEFINE $MODE_DEFINE \
    $LINK_FLAGS \
    -I . -I bindings \
    bindings/_mojolearn_transformer.mojo \
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
    mv "$out" "$OUTDIR/_mojolearn_transformer.so"
    echo "built $OUTDIR/_mojolearn_transformer.so (gate skipped: non-Darwin or MOJOLEARN_SKIP_BUILD_GATE)"
    exit 0
fi

# ============================================================================
# A FLOOR, NOT A PROOF. THE FLOOR IS UNMEASURED, RUN OWED: THE FIRST BUILD
# MUST MEASURE THE BLOB COUNT AND PIN IT.
# ============================================================================
#
# THIS SCRIPT HAS NEVER RUN. The floor below is the PLACEHOLDER 1, the same
# placeholder build_training.sh carried until its first real build measured
# 5 and the floor became 3 (two thirds rounded down, the ratio
# bindings/build.sh and build_mamba.sh use against THEIR measured counts).
# floor UNMEASURED, RUN OWED: the first build must measure the blob count
# and pin it -- run `bash bindings/build_transformer.sh`, read the
# unconditional per-subsystem counts it prints below, and replace
# `transformer:1` with two thirds of the measured transformer-prefix count,
# in the same change that records the measurement.
#
# WHY A FLOOR OF 1 MUST NOT BE LEFT HERE. build.sh learned twice that
# presence-of-one is not a filter: the build that lost GBDT kept exactly 1
# of 85 gbdt_ blobs and passed a presence-of-one check. A floor of 1
# catches the TOTAL Metal failure this gate was written for (the
# MACOSX_DEPLOYMENT_TARGET bug, 0 blobs) and catches nothing else.
#
# WHAT SHOULD BE IN HERE. The blob names carry the kernel's MODULE PATH
# (measured on the built _mojolearn_mamba.so: `mamba_impl_mamba_ssm_...`,
# `gemm_checks_gemm_identical_...`), so the floored prefix for this binding
# is `transformer` (`transformer/impl/transformers/models/llama/
# modeling_llama.mojo`'s kernels: llama_rms_norm_kernel,
# llama_rope_table_kernel, apply_rotary_pos_emb_kernel, kv_append_kernel,
# the gather/scatter copies, the attn_* softmax chain, silu_kernel,
# mlp_gated_kernel). gemm/ blobs (identical_gemm), mamba/ blobs
# (residual_add_kernel is IMPORTED from the mamba lane, contract section
# 0), core/ and checks/ blobs are printed but NOT floored: whether a
# cross-lane helper leaves a blob under its own prefix is not something to
# assert before it has been seen once.
_air=$(air_blobs "$out")
_total=$(printf '%s\n' "$_air" | grep -c . || true)
printf '  AIR blobs by subsystem (total %s):\n' "$_total"
for _sub in transformer mamba gemm core checks; do
    printf '    %-18s %s\n' "$_sub" "$(printf '%s\n' "$_air" | grep -c "^${_sub}" || true)"
done

_failed=0
# floor UNMEASURED, RUN OWED (header above): raise transformer:1 to two
# thirds of the first cold build's measured count, same change as the run
# record.
for _pair in transformer:1; do
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
# THE TRANSFORMER ROWS ARE THE TRANSFORMER'S. Copying a sibling script's
# smoke rows would be a gate that cannot fail, since none of those kernels
# are in this artifact.
#
#   forward B2 L4            -> RMSNorm x2, the seven OP_NT GEMMs, the rope
#                               table + rotation, kv append, the eager
#                               softmax chain (scale, mask, max, exp,
#                               denom, weights, context), silu, gate,
#                               both residual adds; cached_tokens -> 4
#   4 decode steps           -> the same spelling at L = 1 with the cache
#                               carried; compared LOOSELY to the prefill
#                               rows (the bitwise decode==prefill claim
#                               belongs to the lane gate and to
#                               test_transformer_surface.py under
#                               identical, not to a fast-tier smoke);
#                               cached_tokens -> 4
#   float64 x                -> the dtype refusal, BY NAME, in Python
#   n_heads that does not    -> the d_model == n_heads*head_dim refusal
#     divide d_model            (the wrapper's copy of LlamaDims.validate's
#                               rule; the Mojo original stays the authority)
#   a step past max_tokens   -> the capacity refusal, raised IN MOJO by
#                               name, reached from Python
#
# Kept small on purpose (d_model 32, n_heads 2, n_kv 1 -- so GQA's n_rep=2
# index map is in the smoke -- B 2, L 4, capacity 8) because this runs on
# every build and the GPU is shared. The recovery assertions here are
# DELIBERATELY LOOSE and are a smoke test, not the gate:
# python/mojolearn/tests/test_transformer_surface.py is where the reference
# tolerances and the identical-tier bitwise arms are asserted.
MOJOLEARN_SMOKE_SO="$out" python3 - <<'PY'
import os, shutil, sys, tempfile
tmp = tempfile.mkdtemp()
pkg = os.path.join(tmp, "mojolearn")
shutil.copytree("python/mojolearn", pkg,
                ignore=shutil.ignore_patterns("__pycache__"))
shutil.copyfile(os.environ["MOJOLEARN_SMOKE_SO"],
                os.path.join(pkg, "_mojolearn_transformer.so"))
sys.path.insert(0, tmp)
import numpy as np

from mojolearn import _transformer_impl

rng = np.random.default_rng(0)

def u(shape, lo, hi):
    return (lo + (hi - lo) * rng.random(shape)).astype(np.float32)

# ---- d_model 32, n_heads 2, n_kv 1 (n_rep 2), head_dim 16, it 64.
dm, nh, nkv, hd, it = 32, 2, 1, 16, 64
s_w = float(dm) ** -0.5
w = {
    "input_layernorm.weight": u((dm,), 0.5, 1.5),
    "post_attention_layernorm.weight": u((dm,), 0.5, 1.5),
    "q_proj.weight": u((nh * hd, dm), -s_w, s_w),
    "k_proj.weight": u((nkv * hd, dm), -s_w, s_w),
    "v_proj.weight": u((nkv * hd, dm), -s_w, s_w),
    "o_proj.weight": u((dm, nh * hd), -s_w, s_w),
    "gate_proj.weight": u((it, dm), -s_w, s_w),
    "up_proj.weight": u((it, dm), -s_w, s_w),
    "down_proj.weight": u((dm, it), -0.125, 0.125),
}
b, l, cap = 2, 4, 8
x = u((b, l, dm), -2.0, 2.0)
blk = _transformer_impl.TransformerBlock(w, n_heads=nh, n_kv_heads=nkv)
st = blk.allocate_state(b, max_tokens=cap)
y = blk.forward(x, st)
assert y.shape == (b, l, dm), y.shape
assert np.isfinite(y).all()
assert st.cached_tokens == l, st.cached_tokens

st2 = blk.allocate_state(b, max_tokens=cap)
worst = 0.0
for t in range(l):
    yt = blk.step(x[:, t:t + 1, :], st2)
    worst = max(worst, float(np.max(np.abs(yt[:, 0, :] - y[:, t, :]))))
assert worst < 1e-4, ("decode drifted %g from prefill; one spelling "
                      "serves both paths, so even a loose smoke bound "
                      "should hold" % worst)
assert st2.cached_tokens == l, st2.cached_tokens

# ---- the dtype refusal, BY NAME, before any address is taken.
try:
    blk.forward(x.astype(np.float64))
except TypeError as exc:
    assert "float32" in str(exc) and "float64" in str(exc), exc
else:
    raise AssertionError("a float64 x was ACCEPTED")

# ---- d_model == n_heads*head_dim, refused by name. The wrapper refuses
# it (it cannot size the state buffers otherwise); LlamaDims.validate's
# own refusal stays the authority for raw-binding callers.
try:
    _transformer_impl.TransformerBlock(w, n_heads=3)
except ValueError as exc:
    assert "n_heads" in str(exc), exc
else:
    raise AssertionError("n_heads = 3 over d_model 32 was ACCEPTED")

# ---- capacity growth past max_tokens, refused IN MOJO by name, reached
# from Python (the smoke's one deliberate Mojo-refusal row).
st3 = blk.allocate_state(1, max_tokens=l)
blk.forward(x[:1], st3)  # fills the cache exactly
try:
    blk.step(x[:1, :1, :], st3)
except Exception as exc:
    assert "capacity" in str(exc), exc
else:
    raise AssertionError("a step past max_tokens was ACCEPTED")

print("  smoke: forward B2 L4 + 4 decode steps (worst |step-prefill| "
      "%.2e) with cached_tokens 4 both ways, float64 refused by name, "
      "n_heads*head_dim refused by name, a step past max_tokens refused "
      "in Mojo by name" % worst)
shutil.rmtree(tmp, ignore_errors=True)
PY

mv "$out" "$OUTDIR/_mojolearn_transformer.so"
echo "built $OUTDIR/_mojolearn_transformer.so ($_total AIR blobs, minos $MACOS_FLOOR)"
