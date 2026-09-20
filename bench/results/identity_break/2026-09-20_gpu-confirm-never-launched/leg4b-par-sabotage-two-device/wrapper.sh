#!/bin/sh
# MOJOLEARN_GEMM_LEG_EXTRA for lane/gpu-confirm-never-launched, leg 4:
# A par-* SABOTAGE ARM WATCHED TO FAIL ON TWO REAL DEVICES.
#
# WHY THIS LEG HAD TO EXIST, AND WHY THE CPU SHARDS COULD NOT DO IT.
# Three RunPod CPU pods are running the same 43 lanes under a
# `-D MOJOLEARN_HOST_SABOTAGE=1` host build right now. That recipe is the one
# that earned 16 other par-* lanes their `seen(build)` on 2026-09-17. It
# cannot earn it for most of these, and the tree already says why in the
# harness's own words:
#
#   NotImplementedError: no CPU implementation of the cooperative multi-GPU
#   driver kmeans_fit yet: its shards are device row tiles, chunks or ranges
#   inside the GPU binding, which no host binding restates
#
# Measured over every committed CPU column: 19 par-* lanes have ever produced
# a STABLE cell on one, and 35 have only ever refused. The 19 are the
# `cooperative=False` drivers, whose shards are cut in Python
# (_parallel_pool.DevicePool, one worker masked to one device). The 35 are the
# `cooperative=True` drivers, whose shards live inside the GPU binding. A
# sabotage column whose cells REFUSE rather than move is not counted, and it
# should not be: a refusal is a build that did not run, not arithmetic that
# changed. So for those 35 the negative control is only stateable HERE.
#
# WHAT THIS LEG ACTUALLY ARMS, AND WHAT IT DOES NOT.
# Four parallel sabotage defines exist in the tree today. Each is the same
# idiom -- `if rank > 0, read one row early` -- inside a per-rank shard loop,
# so each is INERT ON ONE DEVICE BY CONSTRUCTION and says nothing until a
# second rank exists:
#
#   MOJOLEARN_CHOLESKY_PARALLEL_SABOTAGE   cholesky/multi_gpu.mojo:88
#   MOJOLEARN_HIERARCHY_PARALLEL_SABOTAGE  hierarchy/impl/cluster/detail/multi_gpu.mojo:37
#   MOJOLEARN_GMM_PARALLEL_SABOTAGE        mixture/multi_gpu.mojo:173
#   MOJOLEARN_RESAMPLE_PARALLEL_SABOTAGE   resample/estimator.mojo:703
#
# docs/lanes/SABOTAGE_AUDIT_2026-09-16.md records the limit this leg is
# testing: "their negative controls are device side and DEFEND THE MULTI-GPU
# CHECKS RATHER THAN THE LANE CELLS". GMM's and RESAMPLE's have only ever been
# run as `mojo run -D ...` against standalone check programs and have NEVER
# been compiled into a binding. Building them into the bindings and running
# the LANES under them is the new thing here.
#
# SO THIS LEG CAN COME HOME WITH EITHER ANSWER AND BOTH ARE RESULTS.
# If the cells move, six par-* arms are watched failing on the device axis for
# the first time. If they hold still, these four arms are REACHED BUT INERT at
# the lane level, which is a defect in the arms and has to be recorded as one
# rather than counted as coverage. What must not happen is the third outcome,
# a column that cannot fail: hence a clean two-device column from the SAME
# build on the SAME box, and a one-device column under the same defines to
# show the arms are inert there on purpose.
set -u
cd /root/mojolearn || exit 9
OUT=/root/gemm_leg_out/par-sabotage-two-device
mkdir -p "$OUT/logs"
G="$OUT/gate.txt"
JOBS="${MOJOLEARN_COMPILE_JOBS:-16}"
LANES=par-cholesky,par-kernel-ridge,par-hdbscan,par-graph-agglomerative,par-gmm,par-resample
DEFINES="-D MOJOLEARN_CHOLESKY_PARALLEL_SABOTAGE=1 -D MOJOLEARN_HIERARCHY_PARALLEL_SABOTAGE=1 -D MOJOLEARN_GMM_PARALLEL_SABOTAGE=1 -D MOJOLEARN_RESAMPLE_PARALLEL_SABOTAGE=1"
# The bindings that carry those four files' import graphs. Mojo compiles the
# whole graph per binding, so a -D reaches every transitively imported module.
SAB_BINDINGS="build_kernel_methods.sh build_hdbscan.sh build_solver.sh build_mixture.sh build_resample.sh build_estimators.sh"

T0=$(date +%s)
BUDGET="${MOJOLEARN_PARSAB_BUDGET:-2700}"
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
say "lanes=$LANES"
say "sabotage_defines=$DEFINES"

# The commit witness, or identity_break refuses to write a JSON. The box has
# no .git; the runner records the commit in leg.txt. Take it, never guess.
if [ ! -s /root/mojolearn/commit.txt ]; then
    _c=$(sed -n 's/^commit=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1 | awk '{print $1}')
    case "$_c" in
        [0-9a-f][0-9a-f]*) printf '%s\n' "$_c" > /root/mojolearn/commit.txt ;;
        *) say "NO COMMIT WITNESS: leg.txt gave '$_c'; every identity phase will refuse" ;;
    esac
fi
say "commit=$(head -1 /root/mojolearn/commit.txt 2>/dev/null)"

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    say "NO NVIDIA DEVICE: this body is NVIDIA-only"; exit 8
fi
VISIBLE=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')
say "visible_gpus=$VISIBLE"
# TWO DEVICES, COUNTED AND REFUSED BY NAME. Every arm here is `if rank > 0`.
# On one device there is no rank > 0, so a "the arms did not fire" line from a
# one-GPU box would be a verification that cannot fail.
if [ "$VISIBLE" -lt 2 ]; then
    say "REFUSED: the box reports $VISIBLE visible GPU(s). Every arm in this leg is gated on rank > 0 and is INERT on one device BY CONSTRUCTION. Nothing was run."
    exit 8
fi
_cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
case "$_cc" in
    9.0) MOJOLEARN_GPU_ARCHS=sm_90a ;; 8.9) MOJOLEARN_GPU_ARCHS=sm_89 ;;
    8.6) MOJOLEARN_GPU_ARCHS=sm_86 ;; 8.0) MOJOLEARN_GPU_ARCHS=sm_80 ;;
    12.0) MOJOLEARN_GPU_ARCHS=sm_120a ;;
    *) MOJOLEARN_GPU_ARCHS="sm_$(echo "$_cc" | tr -d .)" ;;
esac
export MOJOLEARN_GPU_ARCHS
_prod=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 \
        | tr 'A-Z ' 'a-z-' | tr -cd 'a-z0-9-' | cut -c1-24)
[ -z "$_prod" ] && _prod=gpu
LABEL="nvidia-$_prod-$MOJOLEARN_GPU_ARCHS"
say "vendor_label=$LABEL"
say "gpu_archs=$MOJOLEARN_GPU_ARCHS"

# THE HOST MATH LIBRARY, OR NOTHING IMPORTS. python/mojolearn/.libs is
# gitignored, so `git archive` ships nothing and a fresh box cannot `import
# mojolearn` at all. Call the tree's OWN recipe rather than retyping its
# arithmetic flags.
run portable_math timeout "$(cap 300)" env PYTHONPATH=/root/mojolearn/packaging/portable_math \
    pixi run python -c "import pathlib, stage; stage.build(pathlib.Path('/root/mojolearn/python/mojolearn/.libs/libMojolearnMath.so'))"
say "portable_math_exit=$(awk -F'	' '$1=="portable_math"{print $2}' "$OUT/status.tsv")"

# ----------------------------------------------------------- the clean build
# Every GPU family and every host family, DERIVED not typed: a hand-written
# family list rots, and two legs on 2026-09-19 each lost lanes to one in the
# same hour. These lanes are anchored cross-route, so a missing HOST binding
# makes the GPU cell refuse even with every GPU family present.
build() {
    run "$1" timeout "$(cap 420)" env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 \
        MOJOLEARN_COMPILE_JOBS="$JOBS" sh "bindings/$1.sh"
}
build_host() {
    run "$1" timeout "$(cap 420)" env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_TARGET_COLUMN=cpu \
        MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_BUILD_JOBS="$JOBS" sh "bindings/$1.sh"
}
# A TARGETED SET, AND THE FIRST ATTEMPT SHOWS WHY IT HAD TO BE.
# Building every family took ~2160 s of a 2700 s budget on pod q8j0udlmwgz0or
# and left the four columns to time out at 784, 60, 61 and 61 seconds, all
# exit 124, all `complete:false`. `cap()` shrinks every phase as the lease
# burns, so building indiscriminately spends the deliverable.
#
# This list is the families the SIX LANES reach plus the ones the four
# sabotage defines live in. It is a hand-written list and those rot -- two
# legs lost lanes to one on 2026-09-19 -- so the backstop below greps the
# column's own refusals for `bindings/build_*.sh`, builds whatever they name,
# and runs again. The refusal NAMES ITS OWN CURE verbatim in the cell, which
# is what makes a targeted list safe here and was not available to the
# derivation that produced this list's first draft.
GPU_SET="build build_estimators build_linalg build_preprocessing build_metrics build_kernel_methods build_hdbscan build_solver build_mixture build_resample"
HOST_SET="build_core_host build_estimators_host build_linalg_host build_preprocessing_host build_metrics_host build_kernel_methods_host build_hdbscan_host build_hdbscan_infer_host build_mixture_host build_mixture_infer_host build_resample_host build_solver_host"
failed=""
for n in $GPU_SET; do
    [ -f "bindings/$n.sh" ] || { say "no such GPU build: $n"; continue; }
    build "$n" || failed="$failed $n"
done
for n in $HOST_SET; do
    [ -f "bindings/$n.sh" ] || { say "no such host build: $n"; continue; }
    build_host "$n" || failed="$failed $n"
done
say "clean_build_failed=${failed:-none}"
say "elapsed_after_clean_build=$(( $(date +%s) - T0 ))"
sha256sum python/mojolearn/identical/*.so > "$OUT/so_sha256.clean.txt" 2>&1

run import_check timeout "$(cap 300)" env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
    pixi run python -c "import mojolearn; print('IMPORT_OK', mojolearn.vendor(), mojolearn.__version__)"
say "import_check=$(tail -1 "$OUT/logs/import_check.log" 2>/dev/null)"

# ------------------------------------------------------ the three clean columns
# TWO FIXTURES, NOT NINE, AND THE SABOTAGE QUESTION IS WHY.
# `clean-one` over nine fixtures took 784 s and still timed out on the first
# attempt. The question these four columns ask is "do the bytes MOVE under the
# arm", and `base,ties` answers it -- that is the pair the 2026-09-17 sabotage
# sweep used for every one of its arms, and `ties` is there on purpose: it is
# integer valued, so an order-permutation arm cannot move an exact sum, which
# is how the shared GEMM leaf arm was caught inert in the 2026-09-16 audit. A
# lane that moves on base and holds still on ties is reported as exactly that.
#
# THIS NARROWS THE CLEAN COLUMNS TOO, and they are deliberately NOT the
# deliverable here: the par-axis two-device column over the default fixtures
# is leg 3's job. These four exist to hold the sabotage arm to the clean arm
# on the same box, same build, same commit.
FIX=base,ties
col() {
    _name=$1; _devs=$2; _json=$3
    run "$_name" timeout "$(cap 900)" env MOJOLEARN_NUMERIC_MODE=identical \
        MOJOLEARN_PAR_DEVICES="$_devs" PYTHONPATH=/root/mojolearn/python \
        pixi run python tools/identity_break.py --lanes "$LANES" --repeats 2 \
        --fixtures "$FIX" --vendor "$LABEL" --json "$_json"
    say "${_name}_exit=$(awk -F'	' -v n="$_name" '$1==n{print $2}' "$OUT/status.tsv")"
    grep -E '^cells=|MOVED|DIVERGENT|REFUSED|RELOAD' "$OUT/logs/$_name.log" | head -60 >> "$G"
}
CLEAN1="$OUT/$LABEL.clean.one-device.json"
CLEAN2="$OUT/$LABEL.clean.two-device.json"
col clean-one 0   "$CLEAN1"
col clean-two 0,1 "$CLEAN2"
run clean_diff timeout "$(cap 300)" env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
    pixi run python tools/identity_break.py --diff "$CLEAN1" "$CLEAN2"
say "clean_diff_exit=$(awk -F'	' '$1=="clean_diff"{print $2}' "$OUT/status.tsv")"
grep -E 'summary|DIVERGENT|MOVED|REFUSED|NOT-COMPARED' "$OUT/logs/clean_diff.log" | head -40 >> "$G"
say "elapsed_after_clean_columns=$(( $(date +%s) - T0 ))"

# --------------------------------------------------------- the sabotage build
# MOJOLEARN_BINCACHE is not set anywhere in this body, so nothing here is
# served from or promoted into the production cache. tools/bincache.py refuses
# to serve a sabotage build as a production binding, and that refusal is a
# feature we are not going anywhere near.
say "sabotage_build_start_elapsed=$(( $(date +%s) - T0 ))"
sfailed=""
for n in $SAB_BINDINGS; do
    [ -f "bindings/$n" ] || { say "no such binding script: $n"; continue; }
    _b=$(basename "$n" .sh)
    run "sab-$_b" timeout "$(cap 420)" env MOJOLEARN_NUMERIC_MODE=identical \
        MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS="$JOBS" \
        MOJOLEARN_BUILD_EXTRA_DEFINES="$DEFINES" sh "bindings/$n" \
        || sfailed="$sfailed $_b"
done
say "sabotage_build_failed=${sfailed:-none}"
sha256sum python/mojolearn/identical/*.so > "$OUT/so_sha256.sabotage.txt" 2>&1
# THE TWO ARMS MUST BE DIFFERENT BINARIES. Same bytes = the define did nothing
# and every "it held still" line below would be about a build that never
# changed. Two builds of IDENTICAL source always differ anyway (mktemp paths),
# so this is a necessary condition and not a sufficient one -- which is why
# the one-device sabotage column below is also run.
diff "$OUT/so_sha256.clean.txt" "$OUT/so_sha256.sabotage.txt" > "$OUT/arm_bytes_differ.txt" 2>&1
say "so_digests_differ_lines=$(grep -c '^[<>]' "$OUT/arm_bytes_differ.txt" 2>/dev/null)"

# ------------------------------------------------------ the sabotage columns
# The file name carries "sabotage" on purpose: verification_matrix's
# `sabotage_signals` reads it, and `admit` REFUSES it by the same token, which
# is exactly right. This column is a negative control and must never be
# admitted as evidence of identity.
SAB1="$OUT/$LABEL.sabotage.one-device.json"
SAB2="$OUT/$LABEL.sabotage.two-device.json"
col sabotage-one 0   "$SAB1"
col sabotage-two 0,1 "$SAB2"

# THE DIFF THAT IS THE WHOLE POINT: clean two-device against sabotage
# two-device, same build set except the four defines, same box, same commit.
run sab_diff_two timeout "$(cap 300)" env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
    pixi run python tools/identity_break.py --diff "$CLEAN2" "$SAB2"
say "sab_diff_two_exit=$(awk -F'	' '$1=="sab_diff_two"{print $2}' "$OUT/status.tsv")"
grep -E 'summary|DIVERGENT|MOVED|REFUSED|NOT-COMPARED' "$OUT/logs/sab_diff_two.log" | head -60 >> "$G"

# AND THE CONTROL FOR THE CONTROL: on ONE device every arm here is gated on
# `rank > 0` and must be inert. A one-device column that MOVED would mean the
# define reached something other than the shard loop, and the two-device
# result would then be about that instead.
run sab_diff_one timeout "$(cap 300)" env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
    pixi run python tools/identity_break.py --diff "$CLEAN1" "$SAB1"
say "sab_diff_one_exit=$(awk -F'	' '$1=="sab_diff_one"{print $2}' "$OUT/status.tsv")"
grep -E 'summary|DIVERGENT|MOVED|REFUSED|NOT-COMPARED' "$OUT/logs/sab_diff_one.log" | head -60 >> "$G"

say "elapsed_total=$(( $(date +%s) - T0 ))"
cp "$OUT/logs/"*.log "$OUT/" 2>/dev/null
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
