#!/bin/sh
# Build the neural-training CPython extension into
# python/mojolearn/_mojolearn_training.so. Run from anywhere; requires pixi.
#
# THIS EXTENSION IS training/: the optimizer step (SGD, Adam, AdamW), the
# global-norm gradient clip and the cross-entropy loss. It is separate from
# `bindings/build.sh` (k-means, k-NN), `build_gbdt.sh`, `build_rf.sh`,
# `build_trees.sh`, `build_svm.sh` and `build_estimators.sh` for the reason
# the estimators binding's header gives: an independently changing binding
# must not become a merge point. All of them land in one wheel.
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
#     extension, with no diagnostic a user can act on -- and arm64 Mach-O
#     cpusubtype stays ARM64_ALL whatever -mcpu was, so such a wheel looks
#     portable to any header-reading check.
#
#   * THERE IS NO --target-accelerator FLAG HERE ON macOS. Measured with
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
# Skylake-SP, which together are a large share of GPU servers.
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
# (sm_80, gfx942, ...). The compiler takes EXACTLY ONE name (a comma list is
# rejected); packaging/linux/build_sets.sh reads the architecture back out of
# every binary and refuses a set that disagrees with what was asked.
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
# NOTE FOR THE OPERATOR: THE IDENTICAL BUILD IS THE ONE THIS LANE'S CLAIM IS
# ABOUT. Under FAST the pinned helpers in checks/numerics.mojo compile away
# and the loss and the optimizer are the same loops in whatever arithmetic
# the vendor's compiler chose; they train, and they promise nothing about
# bits. `python/mojolearn/tests/test_training_surface.py` asserts its bitwise
# arms only under identical and REPORTS them under fast, and says so in its
# own banner.
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

# DO NOT ADD -D MOJOLEARN_OPT_RECORD HERE. It turns on the optimizer's
# per-element recording of `adam.denom`, `adam.q` and `sgd.dir`, which the
# CARD build wants and a shipped wheel does not: `training/estimator.mojo`
# sizes `denom_out` and `q_out` off that same comptime flag, so a recording
# build allocates two extra buffers of N floats on every step. It is a gate
# build's flag, not a wheel's.
#
# DO NOT ADD -D MOJOLEARN_OPT_TRUST_INPUTS EITHER. It removes the
# device-side non-finite refusal that DEVIATION 1496 added after clause (f)
# MEASURED a NaN planted in a parameter reaching `param.out`. It is a
# deliberate downgrade of the profile, it is named so it shows up in a
# banner, and a wheel is not where a caller opts into it.

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-training.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT INT TERM
out="$tmpdir/_mojolearn_training.so"

# --emit shared-lib, not an executable: CPython dlopens this and calls
# PyInit__mojolearn_training. The FILE NAME must match that symbol's suffix
# or the import fails with "dynamic module does not define module export
# function".
# shellcheck disable=SC2086  # the flag strings are deliberately word-split
pixi run mojo build --emit shared-lib \
    $TARGET_FLAGS $COLUMN_DEFINE $MODE_DEFINE \
    $LINK_FLAGS \
    -I . -I bindings \
    bindings/_mojolearn_training.mojo \
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
    mv "$out" "$OUTDIR/_mojolearn_training.so"
    echo "built $OUTDIR/_mojolearn_training.so (gate skipped: non-Darwin or MOJOLEARN_SKIP_BUILD_GATE)"
    exit 0
fi

# ============================================================================
# A FLOOR, NOT A PROOF -- AND THESE PARTICULAR NUMBERS HAVE NEVER BEEN
# MEASURED, BECAUSE THE AUTHOR OF THIS SCRIPT WAS NOT ALLOWED TO BUILD.
# ============================================================================
#
# THE FLOOR WAS 1 UNTIL THE FIRST REAL BUILD, 2026-09-01, and it is now 3.
# `build.sh` learned twice that presence-of-one is not a filter: the build
# that lost GBDT kept exactly 1 of 85 gbdt_ blobs and passed. A floor of 1
# caught only the TOTAL loss this script's MACOSX_DEPLOYMENT_TARGET
# paragraph is about, and nothing subtler.
#
# MEASURED, Apple M4, FAST build, `MOJOLEARN_NUMERIC_MODE` unset:
#
#     training 5     gemm 8     core 0     total 13
#
# so the floor is two thirds of 5, which is 3. That is the ratio
# `bindings/build.sh` uses against ITS measured counts (22 measured -> floor
# 15, 8 -> 3). Setting a floor from a SOURCE count instead is what failed a
# perfectly good svm artifact on its first run, and it is what the paragraph
# below would have done: it counted 20 kernel FUNCTIONS in the source and
# five blobs is what the artifact actually carries.
#
# WHY 5 AND NOT 20, since the gap is large enough to look like a loss and is
# not one. `gemm` carries 8 blobs in the same artifact, and every reduction
# in both contracts is delegated to `identical_gemm_into`, so the loss and
# optimizer folds are compiled under that prefix rather than `training`.
# Add the sabotage-only kernels the compiler drops from a clean build, and
# 5 + 8 is the shape to expect. The gemm blobs stay UNFLOORED for the reason
# given below -- under FAST that route can reach MAX's own matmul and need
# not carry a `gemm` prefix at all -- so a floor over their sum would be a
# floor that changes meaning with the numeric mode.
#
# THE FLOOR IS NOT KNOWN TO BE CAPABLE OF FAILING, and that is stated rather
# than implied: showing it red would mean deliberately shipping a poisoned
# build. What IS verified is the arithmetic against the real artifact --
# 5 >= 3 -- and the total-loss case it is really aimed at, which
# `build.sh` has observed twice in the wild.
#
# What is in here, counted by hand from the source rather than from an
# artifact, so treat it as an expectation and not as the floor: `training/`
# launches 20 distinct kernel functions -- 13 in checks/loss.mojo
# (ce_row_max, ce_shift_exp, ce_logdenom, ce_nll, ce_logp, ce_smooth,
# ce_row_nll, ce_row_smooth, ce_divide, ce_weights, ce_dlogits,
# ce_serial_fold, plus the parameterized row-max instantiation) and 7 in
# checks/optimizer.mojo (adam_update, sgd_update, sqrt_vec, clip_finish,
# clip_scale, sab_chunk_sumsq, sab_combine). Several of those are
# SABOTAGE-ONLY and are unreachable in a clean build, so the compiler may
# drop them and the measured count is expected to be LOWER than 20.
#
# gemm/ and core/ blobs also land here -- every reduction in both contracts
# is delegated to `identical_gemm_into` -- and they are printed but NOT
# floored, because under FAST that route reaches MAX's own matmul and its
# blobs need not carry a `gemm` prefix at all.
#
# THE GATE THAT ACTUALLY PROVES THE ARTIFACT IS run_smoke BELOW. A blob
# count is a pre-filter.
_air=$(air_blobs "$out")
_total=$(printf '%s\n' "$_air" | grep -c . || true)
printf '  AIR blobs by subsystem (total %s):\n' "$_total"
for _sub in training gemm core; do
    printf '    %-18s %s\n' "$_sub" "$(printf '%s\n' "$_air" | grep -c "^${_sub}" || true)"
done

_failed=0
for _pair in training:3; do
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
# second time and drift from it. `_training_impl` is imported as a submodule
# because the private name is stable and this gate should not break on a
# re-export landing in `mojolearn/__init__.py`.
#
# EVERY ROW IS TRAINING'S. Copying a sibling script's smoke rows would be a
# gate that cannot fail, since none of those kernels are in this artifact.
#
#   Adam            -> adam_update_kernel, the whole host-scalar path
#   AdamW           -> the same kernel on its decoupled-decay arm, which is
#                      an ORDER and not a coefficient (contract 7.4)
#   SGD + momentum  -> sgd_update_kernel on BOTH arms of the per-tensor
#                      `buf_initialized` flag: step 1 COPIES the gradient
#                      into the buffer, step 2 runs the recurrence. One step
#                      only would leave the recurrence unlaunched.
#   clip_grad_norm_ -> sqrt_vec, clip_finish, clip_scale, plus the two
#                      delegated GEMM folds at m = n = 1
#   cross_entropy   -> ce_row_max, ce_shift_exp, ce_logdenom, ce_nll,
#                      ce_row_nll, ce_divide and the vocabulary fold
#     smoothing     -> ce_logp, ce_smooth, ce_row_smooth, the kernels the
#                      eps == 0 path never touches (contract 6.2(c))
#     with grad     -> ce_weights, ce_dlogits
#
# Kept small on purpose -- a 3-tensor 40-element model and a 6 x 5 loss --
# because this runs on every build and the GPU is shared. NOTHING HERE
# ASSERTS A BIT: it is a launch gate, and the bitwise arms live in
# `python/mojolearn/tests/test_training_surface.py`, which needs the
# IDENTICAL build this smoke test is skipped for.
MOJOLEARN_SMOKE_SO="$out" python3 - <<'PY'
import os, shutil, sys, tempfile
tmp = tempfile.mkdtemp()
pkg = os.path.join(tmp, "mojolearn")
shutil.copytree("python/mojolearn", pkg,
                ignore=shutil.ignore_patterns("__pycache__"))
shutil.copyfile(os.environ["MOJOLEARN_SMOKE_SO"],
                os.path.join(pkg, "_mojolearn_training.so"))
sys.path.insert(0, tmp)
import numpy as np

from mojolearn import _training_impl as T

rng = np.random.default_rng(0)
# THREE tensors, not one. The clip's cross-tensor fold is a tree over the
# per-tensor norms and at J = 1 it has no arithmetic node at all, so a
# one-tensor fixture launches the fold and evaluates none of it.
shapes = [(4, 3), (7,), (3, 7)]
params = [rng.standard_normal(s).astype(np.float32) for s in shapes]
grads = [rng.standard_normal(s).astype(np.float32) * 0.1 for s in shapes]

adam = T.Adam(params, lr=1e-2)
before = [p.copy() for p in params]
adam.step(grads)
assert all(not np.array_equal(p, b) for p, b in zip(params, before)), \
    "Adam moved no parameter"
assert adam.t == 1, adam.t

adamw = T.AdamW([p.copy() for p in params], lr=1e-2, weight_decay=0.1)
adamw.step(grads)

# TWO SGD STEPS, because step 1 takes the copy arm of the per-tensor
# `buf_initialized` flag and only step 2 runs the momentum recurrence.
sgd = T.SGD([p.copy() for p in params], lr=1e-2, momentum=0.9)
sgd.step(grads)
sgd.step(grads)
assert sgd.t == 2, sgd.t

# clip_grad_norm_ on its own, in place, and the reported norm must be the
# norm of what went IN, not of what came out.
g2 = [g.copy() for g in grads]
total = T.clip_grad_norm_(g2, max_norm=1e-4)
assert total > 1e-4, total
assert all(np.all(np.abs(a) <= np.abs(b) + 1e-6) for a, b in zip(g2, grads))

logits = rng.standard_normal((6, 5)).astype(np.float32)
targets = np.array([0, 4, 2, -100, 1, 3], dtype=np.int32)
loss = T.cross_entropy(logits, targets)
assert np.isfinite(loss) and loss > 0.0, loss
assert T.cross_entropy(logits, targets, reduction="none").shape == (6,)
# label_smoothing selects a DIFFERENT KERNEL and not a bit-inert branch
sm = T.cross_entropy(logits, targets, label_smoothing=0.1)
assert np.isfinite(sm), sm
lv, dl = T.cross_entropy(logits, targets, return_grad=True)
assert dl.shape == (6, 5) and np.all(np.isfinite(dl))
# the ignored row contributes no gradient
assert np.all(dl[3] == 0.0), dl[3]

print("  smoke: Adam, AdamW, SGD+momentum (2 steps), clip_grad_norm_, "
      "cross_entropy (mean/none/smoothed/backward) each launched")
shutil.rmtree(tmp, ignore_errors=True)
PY

mv "$out" "$OUTDIR/_mojolearn_training.so"
echo "built $OUTDIR/_mojolearn_training.so ($_total AIR blobs, minos $MACOS_FLOOR)"
