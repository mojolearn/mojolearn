#!/bin/sh
# Build the GBDT CPython extension into python/mojolearn/_mojolearn_gbdt.so.
# Run from anywhere; requires pixi.
#
# WHY GBDT HAS ITS OWN EXTENSION
# -------------------------------
# The same reason `bindings/_mojolearn_estimators.mojo` has one: an
# independently changing binding stops being a merge point. `_mojolearn.so`
# keeps k-means and k-NN, this keeps the ensemble, and nothing here imports
# `cluster.` or `neighbors.`.
#
# It is NOT here because of a size budget. That was the theory this split was
# commissioned under and the measurement below falsifies it; see the next
# comment, and PORTING.md 70, which has been corrected.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$here"

mkdir -p python/mojolearn

# ============================================================================
# MACOSX_DEPLOYMENT_TARGET IN THE ENVIRONMENT SUPPRESSES AHEAD-OF-TIME METAL
# COMPILATION ENTIRELY. THAT -- NOT THE ENTRY FILE'S BASENAME -- IS THE BUG.
# ============================================================================
#
# For weeks this build was treated as a lottery over the basename of the entry
# file: `bindings/build.sh` compiled a copy under a measured stem, checked the
# artifact, and retried over a list, because byte-identical sources under
# different names produced 113 / 29 / 0 compiled Metal functions. PORTING.md 70
# recorded that as an upstream defect keyed on the name.
#
# THE BASENAME IS INNOCENT. Measured 2026-08-21 on this module, one variable at
# a time, WITH THE COMPILER CACHE CLEARED BEFORE EACH BUILD -- which is the step
# every earlier experiment omitted, and the reason they all read wrong:
#
#   MACOSX_DEPLOYMENT_TARGET   --target-cpu    AIR blobs
#   11.0                       apple-m1            0
#   12.0                       apple-m1            0
#   (unset)                    apple-m1          141
#   11.0                       (host)              0
#   (unset)                    (host)            141
#
# `--target-cpu apple-m1` is innocent. The VALUE of the deployment target is
# innocent -- 12.0 fails exactly as 11.0 does. SETTING THE VARIABLE AT ALL is
# what does it. With it set, `mojo build` emits an EMPTY 134-byte metallib for
# every kernel and embeds nothing; the extension then imports cleanly and dies
# at the first launch with "Failed to create Metal function", which is the
# symptom this whole saga was about.
#
# Basenames were never the variable. Four stems built cold with the variable
# set -- `_mojolearn_gbdt`, `mojolearn_ext`, `qq`, `gbdtml` -- all give 0; the
# same four built cold with it unset all give 141. So this script compiles the
# entry file UNDER ITS OWN NAME, IN PLACE. There is no stem loop and there must
# not be one: a loop over a variable that does nothing is a gate that cannot
# fail, and a gate that cannot fail stops being read.
#
# WHY EVERY EARLIER MEASUREMENT DISAGREED: THE CACHE.
#
# `$MODULAR_HOME/cache/.mojo_cache` is content-addressed and ITS KEY DOES NOT
# INCLUDE THE DEPLOYMENT TARGET. So an empty metallib produced by a build with
# the variable set is served to every later build that hashes to the same key,
# whatever ITS flags are. When this was found, that cache held 40,772 files, of
# which 20,682 were 134-byte empty metallibs, the oldest stamped 05:58 that
# morning.
#
# That single fact explains the entire history. A "good" basename was one whose
# kernels happened to hash to keys still holding REAL metallibs from an older
# configuration; a "bad" one hashed to poisoned keys. The famous continuous
# decline --
#
#     shipped .so (older source)  gbdt 85   total 113
#     at 2cb82ac~1                gbdt 73   total 101
#     at 9ab10bc                  gbdt 58   total  86
#     at HEAD 2026-08-21          gbdt 56   total  84
#
# -- is not the module outgrowing a budget. It is CACHE ATTRITION: each source
# change invalidated more of the surviving real entries, each rebuild replaced
# them with empties, and nothing could ever refill them because every build ran
# with the variable set. The counts were archaeology, not compilation.
#
# THE COST OF LEARNING THIS THE SLOW WAY: with a warm cache, `12.0` measured
# 141 blobs and looked like the fix. It was reading back the artifact of the
# `(unset)` build that had run one minute earlier. **Clear the cache before any
# build whose kernel count you intend to believe**, or the number is fiction.
#
# ----------------------------------------------------------------------------
# THE macOS FLOOR IS SET AT THE LINKER INSTEAD, WHICH KEEPS BOTH PROPERTIES
# ----------------------------------------------------------------------------
#
# The deployment target still has to be low. `mojo build` takes it from the
# host SDK, which on this machine means `minos 26.0` -- a wheel installable
# only on macOS 26, which is very nearly nobody. The MAX runtime dylibs this
# links are built at `minos 11.0`; only our own compile step was narrow.
#
# 11.0 (Big Sur) is chosen because it is the FIRST macOS that runs on Apple
# silicon at all, so it is the widest floor that means anything, and it matches
# the dylibs.
#
# Passing it to the LINKER rather than through the environment gives both:
# `ld -platform_version macos 11.0 <sdk>` stamps LC_BUILD_VERSION exactly as
# the environment variable would, and the Metal compile step never sees it.
# Measured on a cold cache: 141 AIR blobs AND `minos 11.0`, together.
#
# setup.py's DEFAULT_MACOS_TARGET must equal MACOS_FLOOR below or the wheel TAG
# and the BINARY disagree, which is a published lie in one of two directions:
# too low and it installs where it cannot load, too high and it is refused by
# Macs that could have run it. packaging/macos/build_release_wheel.sh checks
# that they agree by reading the Mach-O header after the build.
MACOS_FLOOR="11.0"

# NOT EXPORTED, DELIBERATELY -- see above. Unset it if the caller had it set,
# because inheriting it from an outer shell silently reproduces the bug.
unset MACOSX_DEPLOYMENT_TARGET

if [ "$(uname)" = "Darwin" ]; then
    MACOS_SDK=$(xcrun --sdk macosx --show-sdk-version)
else
    MACOS_SDK=""  # linux arm (E1, 2026-08-22): no Mach-O, no Metal SDK
fi
LINK_FLAGS="-Xlinker -platform_version -Xlinker macos -Xlinker $MACOS_FLOOR -Xlinker $MACOS_SDK"
[ "$(uname)" = "Darwin" ] || LINK_FLAGS=""

# THE CPU TARGET IS PINNED TO A PORTABLE BASELINE, NOT LEFT AT THE HOST.
#
# `mojo build` defaults --target-cpu and --target-features to WHATEVER CHIP RAN
# THE COMPILER. On this machine that is `apple-m4` with +bf16, +i8mm, +sme,
# +sme2 and friends. +sme/+sme2 do not exist on M1, M2 or M3 and +bf16/+i8mm do
# not exist on M1. LLVM does not need to be ASKED to use an enabled feature --
# it emits bfdot, smmla and bfmmla from ordinary loops once the bit is set --
# so a host-built wheel SIGILLs on older Apple silicon, at the instruction,
# inside the extension, with no diagnostic a user can act on. arm64 Mach-O
# `cpusubtype` stays ARM64_ALL whatever -mcpu was, so a native-built wheel LOOKS
# portable to any header-reading check, and verifying it on the machine that
# built it proves nothing at all.
#
# apple-m1 is the oldest Apple silicon; measured 2026-08-20, a probe built at
# this baseline runs correctly on the M4. THE COST has not been re-measured for
# mojolearn's kernels and should be before any published timing is attributed
# to this wheel rather than to a local build.
#
# THERE IS NO --target-accelerator FLAG HERE AND THERE MUST NOT BE. Measured
# 2026-08-21: passing it at all, at a real value or a made-up one, gives 0 AIR
# blobs. (`bindings/build_estimators.sh` still passes it. That is not an
# endorsement; it is a file this script must not edit.)
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
# (checks/numerics.mojo reads it through is_defined, the same shape as
# the column define) and lands the binary under python/mojolearn/identical/,
# where python/mojolearn/_backend.py picks it up when the env var
# MOJOLEARN_NUMERIC_MODE=identical is set at import. Default is fast and the
# default location. The build-time smoke gates import the FAST package, so
# they are skipped for an identical build; tools/e2_matrix_fit.py is that
# build's gate.
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

# --emit shared-lib, not an executable: CPython dlopens this and calls
# PyInit__mojolearn_gbdt. The file's name must match that symbol's suffix or
# the import fails with "dynamic module does not define module export
# function", which is the least helpful error in the toolchain.
#
# Two include paths. `-I .` is the package root, so `gbdt.estimator` resolves.
# `-I bindings` is this directory, for capability modules placed beside the
# entry point.

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-gbdt-build.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT INT TERM
out="$tmpdir/_mojolearn_gbdt.so"

# MOJOLEARN_EXTRA_DEFINES: DIAGNOSTIC DEFINES, PASSED THROUGH VERBATIM, EMPTY
# BY DEFAULT (added 2026-09-03 for the DEVIATION 2043 accuracy A/B).
#
# The numeric mode and the column already have their own variables above
# because they are CONTRACT axes: each one lands its binary in its own
# directory and `python/mojolearn/_backend.py` picks it up by name. A
# per-deviation diagnostic define is neither -- it produces a FAST binary
# that belongs in the FAST directory and differs from the shipped one only
# in the arm one comptime row selects -- so it gets a pass-through rather
# than a directory.
#
# THE CALLER OWNS THE STAGING. This variable does NOT move OUTDIR, so a
# build made with it OVERWRITES the shipped fast artifact in place. That is
# deliberate: a define whose meaning changes per deviation cannot have a
# stable directory, and inventing one per deviation is how a tree grows
# directories nothing reads. `tools/gbdt_accuracy_ab.sh` stashes each .so
# under its own name immediately after the build and copies the one it wants
# back before each launch, which is the shape any caller of this variable
# has to take. A caller that sets it and then forgets to rebuild without it
# has left a diagnostic binary installed under the shipped name.
#
# shellcheck disable=SC2086  # both flag strings are deliberately word-split
pixi run mojo build --emit shared-lib \
    $TARGET_FLAGS $COLUMN_DEFINE $MODE_DEFINE ${MOJOLEARN_EXTRA_DEFINES:-} \
    $LINK_FLAGS \
    -I . -I bindings \
    bindings/_mojolearn_gbdt.mojo \
    -o "$out"

# COUNTING BLOBS IS A PRE-FILTER, NOT THE GATE.
#
# The first version of this check asked only whether each subsystem had at
# least one AIR blob, and a known-broken artifact PASSED IT: the build that
# lost GBDT kept exactly 1 of 85 `gbdt_` blobs. So the floor has to sit well
# above 1 -- and even then it only saves the time of smoke-testing a hopeless
# build. THE GATE IS run_smoke BELOW, because the failure being guarded against
# is precisely that a .so imports cleanly and dies at the first launch.
#
# `_gpu_shared_mem` is a prefix the compiler puts on the blob symbol; it is not
# part of the Metal function name.
air_blobs() {
    strings -a "$1" \
        | grep -oE '[0-9A-Za-z_]+_[0-9a-f]{16}air' \
        | sed -e 's/^_gpu_shared_mem//' -e 's/^0//' \
        | sort -u
}

# The floor is measured on THIS module, not copied from build.sh's. A correct
# gbdt-only artifact carries 141 `gbdt_` blobs; a suppressed one carries 0, and
# the cache-fed partials measured along the way carried 17 and 55. 120 sits
# above every broken build seen and far enough below 141 that an artifact
# missing a handful still reaches run_smoke, which is the check that can say
# WHICH ones. `cluster` and `neighbors` are deliberately absent: this module
# imports neither, and a floor on a subsystem that is not supposed to be here
# would fail every correct build.
kernels_plausible() {
    [ "$(uname)" = "Darwin" ] || return 0  # linux gate is run_smoke
    _air=$(air_blobs "$1")
    for _pair in gbdt:120; do
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

# THE MACH-O FLOOR IS READ BACK, NOT ASSUMED. The linker flag is the only thing
# setting it now, and a silently dropped `-Xlinker` would publish a wheel whose
# tag and binary disagree -- exactly the failure the flag exists to prevent.
minos_matches() {
    [ "$(uname)" = "Darwin" ] || return 0  # linux gate is run_smoke
    _got=$(otool -l "$1" \
        | awk '/LC_BUILD_VERSION/{f=1} f && /minos/{print $2; exit}')
    if [ "$_got" != "$MACOS_FLOOR" ]; then
        printf '  minos is %s, want %s\n' "$_got" "$MACOS_FLOOR" >&2
        return 1
    fi
    return 0
}

# THE REAL GATE: import the extension and LAUNCH a kernel from every loss
# family. Every broken build in this bug's history imported fine and raised
# "Failed to create Metal function" at the first launch, so nothing short of
# launching proves anything. Kept tiny on purpose -- 512x3, two trees --
# because this runs on every build and the GPU is shared.
run_smoke() {
    MOJOLEARN_SMOKE_SO="$1" python3 - <<'PY'
import os, shutil, sys, tempfile
tmp = tempfile.mkdtemp()
pkg = os.path.join(tmp, "mojolearn")
shutil.copytree("python/mojolearn", pkg,
                ignore=shutil.ignore_patterns("__pycache__"))
shutil.copyfile(os.environ["MOJOLEARN_SMOKE_SO"],
                os.path.join(pkg, "_mojolearn_gbdt.so"))
sys.path.insert(0, tmp)
import numpy as np
import mojolearn.ensemble as ens
X = np.random.default_rng(0).random((512, 3), dtype=np.float32)
m = ens.GradientBoosting(loss="RMSE", n_estimators=2, max_depth=3,
                         border_count=16).fit(X, X[:, 0])
m.predict(X)

# ONE LOSS PER KERNEL FAMILY, because a family the smoke test never fits is a
# family whose kernels can be missing from the artifact while this gate says
# PASS. That is not hypothetical: for one build this test fit RMSE only,
# `MultiClassOneVsAll`'s kernels were absent, every floor and the smoke test
# passed, and the failure surfaced in a user's fit.
#
#   RMSE                pointwise_target_kernel
#   Logloss             cross_entropy_kernel
#   MAE                 the EXACT estimator -- segmented sort, scan,
#                       need-weights, binary search
#   MultiClass          multilogit_val_and_first_der / second_der_row
#   MultiClassOneVsAll  one_vs_all_val_and_first_der / second_der
#
# Add a row here whenever a kernel family is added. Two trees each; the point
# is to LAUNCH them, not to fit anything.
yb = (X[:, 0] > 0.5).astype(np.float32)
ens.GradientBoosting(loss="Logloss", n_estimators=2, max_depth=3,
                     border_count=16).fit(X, yb).predict(X)
ens.GradientBoosting(loss="MAE", n_estimators=2, max_depth=3,
                     border_count=16).fit(X, X[:, 0]).predict(X)
yc = (X[:, 0] * 3).astype(np.int32).clip(0, 2).astype(np.float32)
mm = ens.GradientBoosting(loss="MultiClass", n_estimators=2, max_depth=3,
                          border_count=16).fit(X, yc)
mm.predict(X)
mm.predict_proba(X)

# MultiClassOneVsAll IS THE ROW THAT IS STILL REFUSED FROM PYTHON, and the
# reason it was refused NO LONGER HOLDS. `ensemble.py` says its kernels do not
# fit in the extension under any measured basename; the basename was never the
# variable, and this artifact carries every gbdt kernel the module contains.
# The assertion is kept only because reopening the loss means FITTING it under
# a check, not deleting a `try/except`. It is an OPEN item, named in the build
# script so it cannot be quietly forgotten.
try:
    ens.GradientBoosting(loss="MultiClassOneVsAll")
except NotImplementedError:
    pass
else:
    raise SystemExit("smoke: MultiClassOneVsAll was accepted from Python")

# THE HELD-OUT PATH, which crosses two addresses and five parameters nothing
# else here exercises. IT IS THE PATH THAT WAS BROKEN FOR USERS: the shipped
# `_mojolearn.so` predated `eval_x`/`eval_y`, so every `fit` raised "takes 6
# positional arguments but 8 were given". Their default `use_best_model` is ON
# with an eval set, so the fitted model must be SHORTER than the iteration
# count unless the last iteration happened to be the best.
Xe = np.random.default_rng(7).random((128, 3), dtype=np.float32)
me = ens.GradientBoosting(loss="RMSE", n_estimators=6, max_depth=3,
                          border_count=16, od_wait=2)
me.fit(X, X[:, 0], eval_set=(Xe, Xe[:, 0]))
if me.test_loss_curve_ is None or len(me.test_loss_curve_) == 0:
    raise SystemExit("smoke: eval_set produced no held-out curve")
if len(me.loss_curve_) != len(me.test_loss_curve_):
    raise SystemExit("smoke: the two loss curves have different lengths")
if not 0 <= me.best_iteration_ < len(me.test_loss_curve_):
    raise SystemExit("smoke: best_iteration_ is outside the curve")
me.predict(X)
# grow_policy (DEVIATION 259): the three policies are three models, the
# non-symmetric ones round-trip through the model text, and the pairs
# CatBoost refuses are refused by name
preds = {}
for gp in ("SymmetricTree", "Depthwise", "Lossguide"):
    g = ens.GradientBoosting(loss="RMSE", n_estimators=3, max_depth=4,
                             border_count=16, grow_policy=gp).fit(X, X[:, 0])
    preds[gp] = g.predict(X)
    if gp != "SymmetricTree":
        if "\nntree " not in g.model_:
            raise SystemExit(f"smoke: {gp} model text carries no ntree record")
        p = os.path.join(tmp, gp + ".npz")
        g.save(p)
        if not (ens.GradientBoosting.load(p).predict(X) == preds[gp]).all():
            raise SystemExit(f"smoke: {gp} save/load changed predictions")
if (preds["SymmetricTree"] == preds["Depthwise"]).all():
    raise SystemExit("smoke: Depthwise predictions equal SymmetricTree's")
if (preds["SymmetricTree"] == preds["Lossguide"]).all():
    raise SystemExit("smoke: Lossguide predictions equal SymmetricTree's")
for kw, exc in ((dict(grow_policy="Depthwise", use_pointwise_searcher=True),
                 ValueError),
                (dict(grow_policy="SymmetricTree", min_data_in_leaf=3),
                 ValueError),
                (dict(grow_policy="Depthwise", max_leaves=5), ValueError),
                (dict(grow_policy="Lossguide", loss="MultiClass"),
                 NotImplementedError),
                (dict(grow_policy="Region"), NotImplementedError)):
    try:
        ens.GradientBoosting(n_estimators=1, **kw)
    except exc:
        pass
    else:
        raise SystemExit(f"smoke: {kw} was accepted from Python")
shutil.rmtree(tmp, ignore_errors=True)
PY
}

# The per-build gate imports the WHOLE python package, so during a
# from-scratch multi-binding build (a fresh linux box, E1) the first
# gates fail on the siblings' not-yet-built .so files. The caller that
# sets this owns end-to-end verification (the E1 bootstrap runs the
# traced-fit driver, which launches kernels through every lib).
if [ -n "${MOJOLEARN_SKIP_BUILD_GATE:-}" ]; then
    mv "$out" "$OUTDIR/_mojolearn_gbdt.so"
    echo "built $OUTDIR/_mojolearn_gbdt.so (gate SKIPPED by MOJOLEARN_SKIP_BUILD_GATE)"
    exit 0
fi

if ! kernels_plausible "$out" || ! minos_matches "$out" || ! run_smoke "$out"; then
    printf '%s\n' \
      "" \
      "FAILED: the GBDT extension did not come out complete." \
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
      "The cache is regenerable; a poisoned one is not detectable any other" \
      "way, because the build succeeds and the artifact merely does not work." \
      "" \
      "See the comment at the top of this script and PORTING.md 70." >&2
    exit 1
fi

count=$(air_blobs "$out" | wc -l | tr -d ' ')
mv "$out" "$OUTDIR/_mojolearn_gbdt.so"
echo "built $OUTDIR/_mojolearn_gbdt.so ($count AIR blobs, minos $MACOS_FLOOR)"
