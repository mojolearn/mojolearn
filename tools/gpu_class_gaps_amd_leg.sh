#!/bin/sh
# tools/gpu_class_gaps_amd_leg.sh: the on-box body (MOJOLEARN_GEMM_LEG_EXTRA
# for tools/gemm_remote_leg.sh amd) of the AMD identity column that closes the
# other half of the "GPU column on fewer than three classes" gap, 2026-09-19.
#
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/gpu_class_gaps_amd_leg.sh \
#   sh tools/gemm_remote_leg.sh amd --rent \
#      --local-card <an apple.card from a previous leg>
#
# THE SIBLING, AND WHY THE LANE LISTS ARE NOT THE SAME LIST.
# tools/gpu_class_gaps_nvidia_leg.sh (985e57519) is this file's shape, its
# reasoning and its guards; only the lanes and the silicon differ, and the
# difference is not cosmetic. docs/VERIFICATION_MATRIX.md's 49 lanes split
# FOUR ways by the classes they already carry, not two:
#
#   apple + amd, missing nvidia  18   gbdt-query-rmse, the gmm, gp, gpc,
#                                     ivf-extend, kernel-ridge, nystroem and
#                                     svc-poly lanes. AN AMD COLUMN ADDS THEM
#                                     NOTHING -- they already have one. They
#                                     are the NVIDIA leg's REASON for phase A
#                                     and they are deliberately absent here.
#
# CORRECTION TO THIS FILE'S OWN COMMIT MESSAGE (10e299356), which said "NOT
# ONE LANE overlaps with the NVIDIA leg's phase A". THAT IS FALSE and the
# truth is the stronger claim. The NVIDIA leg's LANES_A is those 18 PLUS the
# four gbdt CTR and ranking lanes, which it carries because they are free once
# _mojolearn_gbdt is built -- and those four are in THIS file's LANES_A too.
# The overlap with this leg is not four lanes, it is EIGHTEEN, ALL of them, and
# it is the design: a lane that has apple alone needs an AMD column AND an
# NVIDIA column to reach three classes, so every one of the eighteen must
# appear in BOTH legs. Four of them ride in the NVIDIA leg's phase A and the
# other fourteen are its phase B. What does not overlap is the eighteen
# apple+amd lanes above, and those are the ones an AMD column cannot help.
#   apple ALONE                  18   the four gbdt CTR/ranking lanes and the
#                                     fourteen gemm and weight-format lanes.
#                                     THESE ARE THIS LEG'S WHOLE POINT: an AMD
#                                     column takes each to two classes, and
#                                     with the NVIDIA leg's phase A+B landing
#                                     the same day, to THREE.
#   par-* on nvidia              11   par-boosting-clf, par-boosting-reg,
#                                     par-cd-elasticnet, par-forest-et-clf,
#                                     par-forest-reg, par-gram-ols,
#                                     par-gram-pca, par-gram-tsvd,
#                                     par-queries-nn, par-scaler-minmax,
#                                     par-svm-svr
#   par-* on amd + nvidia         2   par-forest-pool, par-rbf-sampler
#
# NOTHING HERE ANSWERS A par-* LANE AND NOTHING HERE PRETENDS TO. Their claim
# is written in identity_break._par_devices and needs TWO devices; a column
# with par_devices set is REFUSED by python/mojolearn/_verify_reference.admit
# ("par_devices N"). They need a two-device leg on another lease.
#
# AND ESPECIALLY NOT ON THIS SILICON WITHOUT A DECISION FIRST. The standing
# reason AMD was left alone from 2026-09-15 was the MI300X SR-IOV peer-copy
# stale read: on 2x MI300X a device-1 kernel read a cross-device copy's
# DESTINATION before it was written, which broke Cholesky and the byte-LM
# pools. It was fixed by routing every AMD cross-device byte through
# transfer_bytes host staging (d36cd8cb4). This leg does not test that fix and
# does not depend on it: GPU_COUNT stays 1, every lane below is single-device,
# and no phase copies a byte between devices.
#
# TWO LANES RIDE ALONG FROM "No GPU column at all: 17", AND ONLY TWO.
# gbdt-adapter-score-weighted binds _mojolearn_gbdt, which phase A builds
# anyway; ordered-gradient-sum binds _mojolearn_training, which phase B builds
# anyway. Both are non-degenerate on the amd-1gpu column by
# tools/lane_applicability.py's own truth table. The other fifteen are NOT
# picked up: linalg-qr, linalg-eigh and linalg-svdvals look free (phase B
# builds linalg) but python/mojolearn/host_surface.py says of them "They take
# the HOST route on every box, a GPU box included", so an AMD column for them
# would time that box's CPU under an AMD label; cross-val-folds and
# language-model-config are DEGENERATE on amd-1gpu by name; and the rest need
# families no phase here builds.
#
# THE COLUMN MUST BE ADMISSIBLE OR IT IS NOT EVIDENCE. `admit` wants identical
# mode, a real commit, the default fixture size, one device, and a name with
# none of sabotage/partial/probe/unfixed/smoke in it. Hence: no --fixtures
# cap, no MOJOLEARN_IDENTITY_*_SABOTAGE anywhere in this file, the commit
# witness taken from the runner's own leg.txt rather than guessed, and file
# names chosen to clear the excluded tokens.
#
# EVERY PHASE IS BOUNDED, AND THE BOUND SHRINKS AS THE LEASE BURNS. The
# runner polls for a completion sentinel and fetches nothing from a body that
# never finishes. `cap` below never hands out more time than is left of this
# file's own budget, so a slow ROCm compile costs phase B some of its ceiling
# instead of costing the whole leg its fetch.
#
# POSIX sh: the pod's /bin/sh is dash. `set -u` and deliberately NOT `set -e`:
# a phase that fails is a FINDING and its log must come home.
set -u
cd /root/mojolearn || exit 9
OUT=/root/gemm_leg_out/gpu-class-gaps
mkdir -p "$OUT/logs"
G="$OUT/gate.txt"
JOBS="${MOJOLEARN_COMPILE_JOBS:-8}"
# rocminfo and rocm-smi live in /opt/rocm/bin, which the ROCm images do not
# always put on a non-login shell's PATH.
PATH="$PATH:/opt/rocm/bin"
export PATH

# PHASE A: the four gbdt lanes that have apple ALONE, plus the one lane from
# the no-GPU-column list that shares _mojolearn_gbdt with them.
LANES_A=gbdt-categorical-ctr-tables,gbdt-pair-logit,gbdt-tensor-ctr-tables,gbdt-yeti-rank,gbdt-adapter-score-weighted
# PHASE B: the fourteen apple-only gemm and weight-format lanes, plus the one
# lane from the no-GPU-column list that shares _mojolearn_training with them.
LANES_B=gemm-bf16,gemm-int8,mamba1-bf16w,mamba1-int8w,mamba2-bf16w,mamba2-int8w,mamba3-bf16w,mamba3-int8w,mlp-bf16w,mlp-int8w,samba-bf16w,samba-int8w,transformer-bf16w,transformer-int8w,ordered-gradient-sum

T0=$(date +%s)
BUDGET="${MOJOLEARN_AMD_LEG_BUDGET:-2700}"
# The runner's own work bound is MINUTES*60 - 600; 2700 s leaves it margin to
# poll, fetch and terminate after this body writes its sentinel.
cap() {
    _want=$1
    _left=$(( T0 + BUDGET - $(date +%s) ))
    [ "$_left" -lt 60 ] && _left=60
    if [ "$_want" -gt "$_left" ]; then echo "$_left"; else echo "$_want"; fi
}
say() { echo "$@" >> "$G"; }
run() {
    _n=$1; shift
    _t0=$(date +%s)
    "$@" > "$OUT/logs/$_n.log" 2>&1
    _e=$?
    echo "$_n	$_e	$(( $(date +%s) - _t0 ))" >> "$OUT/status.tsv"
    return "$_e"
}
say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
say "budget_seconds=$BUDGET"
say "lanes_a=$LANES_A"
say "lanes_b=$LANES_B"

# THE COMMIT WITNESS, OR identity_break REFUSES TO WRITE A JSON. The box has
# no .git (the runner ships `git archive` at a pinned sha) and the gemm
# payload writes no commit.txt for the extra body. The runner DOES record the
# commit in /root/gemm_leg_out/leg.txt. Take it from there, and never guess.
if [ ! -s /root/mojolearn/commit.txt ]; then
    _c=$(sed -n 's/^commit=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1 | awk '{print $1}')
    case "$_c" in
        [0-9a-f][0-9a-f]*) printf '%s\n' "$_c" > /root/mojolearn/commit.txt ;;
        *) say "NO COMMIT WITNESS: leg.txt gave '$_c'; every identity phase will refuse" ;;
    esac
fi
say "commit=$(head -1 /root/mojolearn/commit.txt 2>/dev/null)"

# ------------------------------------------------------------ the box itself
{ rocm-smi --showproductname; echo "-- driver --"; cat /sys/module/amdgpu/version 2>/dev/null;
  echo "-- agents --"; rocminfo 2>/dev/null | grep -E 'Name:|gfx' | head -20; } > "$OUT/logs/device.txt" 2>&1
say "device=$(rocm-smi --showproductname 2>/dev/null | tr '\n' ' ' | cut -c1-200)"
say "amdgpu_driver=$(cat /sys/module/amdgpu/version 2>/dev/null)"
# ONE DEVICE. The peer-copy history above is why this is asserted and printed
# rather than assumed: a second visible device would make every lane below a
# different claim than the one this column is recorded as.
say "visible_agents=$(rocminfo 2>/dev/null | grep -c -E '^ *Name: *gfx')"
if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
    MOJOLEARN_GPU_ARCHS=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-z]+')
    export MOJOLEARN_GPU_ARCHS
fi
say "gpu_archs=${MOJOLEARN_GPU_ARCHS:-unset}"
if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
    say "NO ARCHITECTURE: bindings/build_byte_lm.sh and its siblings refuse without one"
fi
# The marketing name, from rocm-smi's "Card series" field and, when that
# layout moves between ROCm releases, from rocminfo's Marketing Name.
_prod=$(rocm-smi --showproductname 2>/dev/null | sed -n 's/.*[Cc]ard [Ss]eries:[[:space:]]*//p' | head -1)
[ -z "$_prod" ] && _prod=$(rocminfo 2>/dev/null | sed -n 's/^ *Marketing Name: *//p' | grep -i -m1 'instinct\|radeon')
_prod=$(printf '%s' "$_prod" | tr 'A-Z ' 'a-z-' | tr -cd 'a-z0-9-' | cut -c1-24)
[ -z "$_prod" ] && _prod=gpu
LABEL="amd-$_prod-${MOJOLEARN_GPU_ARCHS:-gfx}"
# device_class() in python/mojolearn/_verify_reference.py reads the class out
# of this string; it must contain "amd" and it does, first.
say "vendor_label=$LABEL"

build() {
    run "$1" timeout "$(cap 420)" env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 \
        MOJOLEARN_COMPILE_JOBS="$JOBS" sh "bindings/$1.sh"
}

# ------------------------------------------- THE HOST MATH LIBRARY, OR NOTHING IMPORTS
# MEASURED ON THIS LEG'S FIRST POD, f7qt7xpx8f6raq, 2026-09-19, exactly as the
# NVIDIA sibling measured it on 9laka4vs9h2zli the same morning: phase A's
# column died ONE SECOND in, before any GPU work, at
#
#   OSError: python/mojolearn/.libs/libMojolearnMath.so: cannot open shared
#   object file: No such file or directory
#
# with both its bindings built and NOT ONE CELL recorded. The vendor read-back
# above carried the same traceback and was the first thing to say so.
#
# `python/mojolearn/_portable_math.py` dlopens that library, and
# `_training_impl.py`'s `def kaiming_uniform(self, shape, fan_in,
# a=math.sqrt(5.0))` evaluates it as a DEFAULT ARGUMENT at class definition
# time, so `import mojolearn` needs it unconditionally. It is not lazy and no
# lane can avoid it -- this is NOT a GPU, a vendor or a lane-list problem, and
# it fails identically on AMD and NVIDIA.
#
# Nothing under bindings/ builds it, `python/mojolearn/.libs/` is gitignored so
# `git archive` ships nothing, and the only thing in the tree that compiles it
# is `packaging/macos/build_release_wheel.sh`, which does not run on Linux. A
# developer Mac has it sitting in the checkout from some past wheel build and
# never notices; a freshly rented box cannot import the package at all.
#
# This calls the tree's OWN recipe, `packaging/portable_math/stage.py`'s
# build(), rather than retyping its compiler flags here -- those flags
# (-ffp-contract=off, -fno-fast-math, -nostdlib) are the arithmetic contract,
# and a second copy of them is a second answer to it.
run portable_math timeout "$(cap 300)" env PYTHONPATH=/root/mojolearn/packaging/portable_math \
    pixi run python -c "import pathlib, stage; stage.build(pathlib.Path('/root/mojolearn/python/mojolearn/.libs/libMojolearnMath.so'))"
say "portable_math_exit=$(awk -F'	' '$1=="portable_math"{print $2}' "$OUT/status.tsv")"
ls -l python/mojolearn/.libs/ >> "$G" 2>&1

# FAIL FAST, AND BEFORE THE EXPENSIVE HALF. If the package still cannot
# import, every phase below is a one-second traceback and the lease is spent
# discovering that twice. The probe is also the vendor witness: on the first
# pod it printed `IMPORT_OK hip`, which is the backend this column claims.
run import_probe timeout "$(cap 300)" env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
    pixi run python -c "import mojolearn; print('IMPORT_OK', mojolearn.vendor(), mojolearn.__version__)"
say "import_probe=$(tail -1 "$OUT/logs/import_probe.log" 2>/dev/null)"

# --------------------------------------------------- PHASE A builds, then the column
# bindings/build.sh is the shared kernels and fixtures every lane reaches;
# bindings/build_gbdt.sh is _mojolearn_gbdt, which every phase A lane binds
# (python/mojolearn/host_surface.py, family "gbdt").
for b in build build_gbdt; do
    build "$b"
done
sha256sum python/mojolearn/identical/*.so >> "$G" 2>/dev/null
say "vendor_readback=$(env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python pixi run python -c 'import mojolearn; print(mojolearn.vendor())' 2>&1 | tail -1)"

# THE DELIVERABLE. Default fixtures, default size, two repeats in one process,
# one device. No sabotage switch is set anywhere in this run.
run column-a timeout "$(cap 900)" env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
    pixi run python tools/identity_break.py --lanes "$LANES_A" --repeats 2 \
    --vendor "$LABEL" --json "$OUT/$LABEL.gbdt.json"
say "column_a_exit=$(awk -F'	' '$1=="column-a"{print $2}' "$OUT/status.tsv")"
grep -E '^cells=|^train:|^infer:|^model:|^batch:|MOVED|DIVERGENT|REFUSED|RELOAD' "$OUT/logs/column-a.log" | head -60 >> "$G"
cp "$OUT/logs/column-a.log" "$OUT/column-a.log" 2>/dev/null
say "elapsed_after_a=$(( $(date +%s) - T0 ))"

# --------------------------------------------------- PHASE B builds, then the column
# The weight-format lanes: ml.lowbit.pack goes through _mojolearn_linalg on a
# GPU column, the blocks through _mojolearn_mamba and _mojolearn_transformer,
# SmallMLPTrainer, SambaStack and ordered_sum_gradients through
# _mojolearn_training, and the infer part of each through the CPU-only
# _mojolearn_neural_host. `env -u MOJOLEARN_GPU_ARCHS` comes FIRST and not as
# decoration: this file exports that variable above and
# bindings/build_host_family.sh refuses a CPU build that carries one; `env
# FOO=1 -u BAR` does NOT unset BAR, because env stops parsing options at the
# first assignment.
for b in build_linalg build_mamba build_transformer build_training; do
    build "$b"
done
run build_neural_host timeout "$(cap 420)" env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_TARGET_COLUMN=cpu \
    MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_BUILD_JOBS="$JOBS" sh bindings/build_neural_host.sh
sha256sum python/mojolearn/identical/*.so >> "$G" 2>/dev/null

run column-b timeout "$(cap 900)" env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
    pixi run python tools/identity_break.py --lanes "$LANES_B" --repeats 2 \
    --vendor "$LABEL" --json "$OUT/$LABEL.lowbit.json"
say "column_b_exit=$(awk -F'	' '$1=="column-b"{print $2}' "$OUT/status.tsv")"
grep -E '^cells=|^train:|^infer:|^model:|^batch:|MOVED|DIVERGENT|REFUSED|RELOAD' "$OUT/logs/column-b.log" | head -60 >> "$G"
cp "$OUT/logs/column-b.log" "$OUT/column-b.log" 2>/dev/null
say "elapsed_after_b=$(( $(date +%s) - T0 ))"

# THE CROSS-COLUMN DIFF IS NOT RUN HERE. `git archive` ships the source only,
# so bench/results/ does not exist on this box and the Apple and NVIDIA
# columns are not here to diff against. That diff costs nothing at home,
# against the fetched column, and that is where it runs -- and where a
# DIVERGENT cell gets re-run SOLO before anyone calls it real.

# --------------------------------------------------------------- bring it home
cat "$OUT/status.tsv" >> "$G" 2>/dev/null
ls -l "$OUT" >> "$G" 2>/dev/null
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
