#!/bin/sh
# tools/gemm_ksplit_classical_leg.sh -- the classical GEMM callers under the
# ksplit default (DEVIATION 2595) against the old plan, the A/B that
# ENGINEERING_RULES.md section 9 owes for a shared kernel's flip
# (docs/lanes/BRIEF_gemm_long_k_2026-09-11.md section 11, job 2).
#
# VENDOR-AGNOSTIC BODY, written for the RunPod H100. It runs ON THE BOX as the
# MOJOLEARN_GEMM_LEG_EXTRA body of tools/gemm_remote_leg.sh, after the leg's
# own IDENTICAL device check and card. That runner copies it to
# /root/gemm_leg_extra.sh and passes NO extra environment, so every knob below
# carries its leg value as its default; a value already exported wins.
#
#   MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key MOJOLEARN_GPU_ARCHS=sm_90a \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/gemm_ksplit_classical_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/<UTC stamp>-nvidia-h100-gemm-ksplit-classical \
#   sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 \
#       --gpu "NVIDIA H100 80GB HBM3" --local-card /tmp/gemm-ksplit-classical-apple.card
#
# WHAT RUNS, each step with its own exit code in status.tsv. A red step is a
# finding, never an abort: later steps still run (set -u, not set -e).
#
#   build-price   bench/gemm_step_price_main.mojo, IDENTICAL + trial, used
#                 ONLY in its host label mode: plans.tsv (arm -> PLANLABEL) and,
#                 after timing, dispatch.txt (one DISPATCH line per caller
#                 GEMM shape per arm: does the ksplit default take it, at which
#                 group size, which plan runs).
#   download-*    the section 9 datasets through the trees harness's own
#                 untimed step (`tools/speed_gbdt_arm.py --download taxi` and
#                 `--download istella`, under GBM_BENCH_DATA), in the
#                 BACKGROUND while the bindings build; a cache already on the
#                 box is reused.
#   build-binding-*  the three IDENTICAL bindings the callers import
#                 (bindings/build.sh, build_estimators.sh, build_gp.sh) with
#                 -D MOJOLEARN_GEMM_ARM_TRIAL=1 through their
#                 MOJOLEARN_BUILD_EXTRA_DEFINES hook, so MOJOLEARN_GEMM_ARM
#                 selects the plan per call.
#   smoke         tools/gemm_ksplit_classical_ab.py smoke: import and one tiny
#                 fit per estimator, so a missing binding is named early.
#   <lane>.<dataset>.<arm>.<block>
#                 tools/gemm_ksplit_classical_ab.py time, one process per
#                 lane, dataset, arm and block, blocks in ABBA order
#                 (tuned128 then default, then default then tuned128). The
#                 log of each is the FSPEED log tools/flip_verdict.py reads.
#                 Taxi first for every lane, then Istella-S; GP first in each.
#   dispatch      the FSPEED-GEMM caller shapes through the label mode.
#   verdicts      tools/gemm_ksplit_classical_ab.py verdict: flip_verdict per
#                 caller (before = tuned128, after = default), then one
#                 `caller=<lane> verdict=HOLDS|REGRESSES|UNMEASURED|IDENTITY-BREAK`
#                 line each and a `CLASSICAL KSPLIT A/B` summary.
#
# KNOBS (defaults are the leg):
#   MOJOLEARN_CLASSICAL_AB_LANES=gp,ols,pca
#   MOJOLEARN_CLASSICAL_AB_DATASETS=taxi,istella
#   MOJOLEARN_CLASSICAL_AB_ROUNDS=3        timed rounds per block (plus one warm-up)
#   MOJOLEARN_CLASSICAL_AB_BLOCKS=2        blocks per arm; 2 measures the noise band
#   MOJOLEARN_CLASSICAL_AB_ROWS=4000000    OLS and PCA rows (section 9); Istella-S has 2,043,304
#   MOJOLEARN_CLASSICAL_AB_GP_TRAIN=4000   GP training rows (a declared ladder rung)
#   MOJOLEARN_CLASSICAL_AB_GP_TEST=1000    GP prediction rows
#   MOJOLEARN_CLASSICAL_AB_PREP=standardize   or raw
#   MOJOLEARN_CLASSICAL_AB_DEADLINE=1200   seconds per timed process
#   MOJOLEARN_CLASSICAL_AB_OUT=/root/gemm_leg_out/ksplit-classical
#                                          comes home with the runner's fetch
#
# POSIX sh only: RunPod's Ubuntu images link /bin/sh to dash.
set -u
ROOT=${MOJOLEARN_GEMM_STEP_ROOT:-/root/mojolearn}
OUT=${MOJOLEARN_CLASSICAL_AB_OUT:-/root/gemm_leg_out/ksplit-classical}
LANES=${MOJOLEARN_CLASSICAL_AB_LANES:-gp,ols,pca}
DATASETS=${MOJOLEARN_CLASSICAL_AB_DATASETS:-taxi,istella}
ROUNDS=${MOJOLEARN_CLASSICAL_AB_ROUNDS:-3}
BLOCKS=${MOJOLEARN_CLASSICAL_AB_BLOCKS:-2}
ROWS=${MOJOLEARN_CLASSICAL_AB_ROWS:-4000000}
GP_TRAIN=${MOJOLEARN_CLASSICAL_AB_GP_TRAIN:-4000}
GP_TEST=${MOJOLEARN_CLASSICAL_AB_GP_TEST:-1000}
PREP=${MOJOLEARN_CLASSICAL_AB_PREP:-standardize}
DEADLINE=${MOJOLEARN_CLASSICAL_AB_DEADLINE:-1200}
JOBS=${MOJOLEARN_COMPILE_JOBS:-4}
GBM_BENCH_DATA=${GBM_BENCH_DATA:-$HOME/datasets/gbm-bench}
export GBM_BENCH_DATA
mkdir -p "$OUT/bin"
cd "$ROOT" || exit 9
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
unset MOJOLEARN_GEMM_ARM MOJOLEARN_GEMM_ARM_SABOTAGE

case "$LANES" in
    *[!a-z,]*) echo "MOJOLEARN_CLASSICAL_AB_LANES=$LANES: letters and commas only; nothing run" > "$OUT/gate.txt"; exit 9 ;;
esac
for lane in $(echo "$LANES" | tr ',' ' '); do
    case "$lane" in
        gp|ols|pca) ;;
        *) echo "lane $lane is not gp, ols or pca; nothing run" > "$OUT/gate.txt"; exit 9 ;;
    esac
done
for ds in $(echo "$DATASETS" | tr ',' ' '); do
    case "$ds" in
        taxi|istella) ;;
        *) echo "dataset $ds is not taxi or istella; nothing run" > "$OUT/gate.txt"; exit 9 ;;
    esac
done
case "$ROUNDS$BLOCKS$ROWS$GP_TRAIN$GP_TEST$DEADLINE$JOBS" in
    *[!0-9]*) echo "numeric knobs must be digits; nothing run" > "$OUT/gate.txt"; exit 9 ;;
esac
case "$PREP" in
    standardize|raw) ;;
    *) echo "MOJOLEARN_CLASSICAL_AB_PREP=$PREP is not standardize or raw; nothing run" > "$OUT/gate.txt"; exit 9 ;;
esac

# THE VENDOR AND THE ARCH (tools/gemm_step_leg.sh's reading, one mojo build
# is one GPU arch).
case "${MOJOLEARN_TARGET_COLUMN:-}" in
    amd|nvidia) VENDOR=$MOJOLEARN_TARGET_COLUMN ;;
    *) if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then
           VENDOR=nvidia
       elif [ -e /dev/kfd ] || command -v rocm-smi > /dev/null 2>&1; then
           VENDOR=amd
       else
           VENDOR=unknown
       fi ;;
esac
if [ "$VENDOR" = unknown ]; then
    echo "vendor=unknown: no working nvidia-smi, no /dev/kfd, no rocm-smi; nothing run" > "$OUT/gate.txt"
    exit 9
fi
export MOJOLEARN_TARGET_COLUMN="$VENDOR"
GPU_NAME=unknown
if [ "$VENDOR" = nvidia ]; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_')
    driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
    case "$driver_major" in
        ''|*[!0-9]*) ;;
        *) if [ "$driver_major" -lt 580 ] && [ -x /usr/local/cuda/bin/ptxas ]; then
               export MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-/usr/local/cuda/bin/ptxas}
           fi ;;
    esac
    if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
        cap=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
        case "$cap" in
            9.0) MOJOLEARN_GPU_ARCHS=sm_90a ;;
            [0-9].[0-9]) MOJOLEARN_GPU_ARCHS="sm_$(echo "$cap" | tr -d .)" ;;
            *) MOJOLEARN_GPU_ARCHS="" ;;
        esac
    fi
fi
if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
    echo "vendor=$VENDOR gpu_archs=MISSING: one mojo build is one GPU arch; set MOJOLEARN_GPU_ARCHS; nothing run" > "$OUT/gate.txt"
    exit 9
fi
export MOJOLEARN_GPU_ARCHS

gpu_snapshot() {  # <file>
    if [ "$VENDOR" = nvidia ]; then
        nvidia-smi --query-gpu=name,driver_version,uuid,clocks.sm,temperature.gpu,memory.used --format=csv > "$1" 2>&1
    else
        { rocm-smi --showproductname --showdriverversion --showuse --showmemuse --showtemp 2>&1 \
            || echo "rocm-smi did not answer"; } > "$1"
    fi
}

rc=0
run() {
    # run <name> <command...>: log to $OUT/<name>.log, record the exit code.
    # Knobs go through `env NAME=value` inside the command.
    _name=$1
    shift
    _t0=$(date +%s)
    "$@" > "$OUT/$_name.log" 2>&1
    _code=$?
    printf '%s\t%s\t%ss\n' "$_name" "$_code" "$(( $(date +%s) - _t0 ))" >> "$OUT/status.tsv"
    [ "$_code" -eq 0 ] || rc=1
    return "$_code"
}

{
    echo "brief=docs/lanes/BRIEF_gemm_long_k_2026-09-11.md section 11 (job 2)"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "root=$ROOT"
    echo "lanes=$LANES datasets=$DATASETS"
    echo "rounds=$ROUNDS blocks=$BLOCKS order=ABBA rows=$ROWS gp_train=$GP_TRAIN gp_test=$GP_TEST prep=$PREP deadline=$DEADLINE"
    echo "vendor=$VENDOR gpu=$GPU_NAME gpu_archs=$MOJOLEARN_GPU_ARCHS column=$MOJOLEARN_TARGET_COLUMN jobs=$JOBS"
    echo "data_root=$GBM_BENCH_DATA"
    [ -f /root/gemm_leg_out/leg.txt ] && grep -E '^(commit|vendor|provider|size)=' /root/gemm_leg_out/leg.txt
} > "$OUT/gate.txt"
gpu_snapshot "$OUT/gpu_before.txt"
pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1

IDENT="-D MOJOLEARN_NUMERIC_IDENTICAL=1"
TRIAL="-D MOJOLEARN_GEMM_ARM_TRIAL=1"
PY="pixi run python"

# ---- python prerequisites (numpy for the harness, pyarrow for the taxi decode)
if ! pixi run python -c 'import numpy' > "$OUT/numpy.log" 2>&1; then
    pixi run python -m pip install numpy >> "$OUT/numpy.log" 2>&1
fi
if ! pixi run python -c 'import pyarrow' > "$OUT/pyarrow.log" 2>&1; then
    pixi run python -m pip install pyarrow >> "$OUT/pyarrow.log" 2>&1
fi

# ---- the datasets, untimed, in the background --------------------------------
fetch() {
    for _ds in $(echo "$DATASETS" | tr ',' ' '); do
        case "$_ds" in
            taxi) _npz="$GBM_BENCH_DATA/taxi/taxi_speed.npz" ;;
            *) _npz="$GBM_BENCH_DATA/istella/istella_speed.npz" ;;
        esac
        if [ -s "$_npz" ]; then
            printf 'download-%s\t0\tcached\n' "$_ds" >> "$OUT/status.tsv"
            continue
        fi
        # shellcheck disable=SC2086
        run "download-$_ds" timeout -k 30 1800 $PY tools/speed_gbdt_arm.py --download "$_ds"
    done
}
fetch > "$OUT/fetch.console" 2>&1 &
FETCH_PID=$!

# ---- the label binary ----------------------------------------------------------
# shellcheck disable=SC2086
run build-price pixi run mojo build -j "$JOBS" -I . $IDENT $TRIAL \
    bench/gemm_step_price_main.mojo -o "$OUT/bin/step-price"
: > "$OUT/plans.tsv"
if [ -x "$OUT/bin/step-price" ]; then
    for arm in shipped tuned128; do
        env MOJOLEARN_GEMM_ARM="$arm" MOJOLEARN_GEMM_STEP_LABEL_ONLY=1 \
            timeout 120 "$OUT/bin/step-price" < /dev/null > "$OUT/label-$arm.txt" 2>&1
        plan_label=$(sed -n 's/^PLANLABEL arm=[^ ]* label=//p' "$OUT/label-$arm.txt" | head -1)
        printf '%s\t%s\n' "$arm" "${plan_label:-unlabeled}" >> "$OUT/plans.tsv"
    done
    grep -h '^DEFAULT' "$OUT/label-shipped.txt" >> "$OUT/gate.txt" 2>/dev/null
fi
awk -F '\t' '{ print "plan " $1 "=" $2 }' "$OUT/plans.tsv" >> "$OUT/gate.txt"

# ---- the bindings, with the trial hook ---------------------------------------
rm -f python/mojolearn/identical/_mojolearn.so python/mojolearn/identical/_mojolearn_estimators.so \
    python/mojolearn/identical/_mojolearn_gp.so
run build-binding-base env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 \
    MOJOLEARN_COMPILE_JOBS="$JOBS" MOJOLEARN_BUILD_EXTRA_DEFINES="$TRIAL" sh bindings/build.sh
run build-binding-estimators env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 \
    MOJOLEARN_COMPILE_JOBS="$JOBS" MOJOLEARN_BUILD_EXTRA_DEFINES="$TRIAL" sh bindings/build_estimators.sh
run build-binding-gp env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 \
    MOJOLEARN_COMPILE_JOBS="$JOBS" MOJOLEARN_BUILD_EXTRA_DEFINES="$TRIAL" sh bindings/build_gp.sh
for f in python/mojolearn/identical/_mojolearn.so python/mojolearn/identical/_mojolearn_estimators.so \
        python/mojolearn/identical/_mojolearn_gp.so; do
    [ -f "$f" ] && sha256sum "$f" >> "$OUT/bindings.sha256"
done
# shellcheck disable=SC2086
run smoke timeout -k 30 600 $PY tools/gemm_ksplit_classical_ab.py smoke

wait "$FETCH_PID"
cat "$OUT/fetch.console" >> "$OUT/gate.txt" 2>/dev/null

# ---- the timed A/B -------------------------------------------------------------
timed() {  # <lane> <dataset> <arm name> <block>
    _env_arm=""
    _plan_arm=shipped
    if [ "$3" = tuned128 ]; then
        _env_arm=tuned128
        _plan_arm=tuned128
    fi
    _label=$(awk -F '\t' -v a="$_plan_arm" '$1 == a { print $2; exit }' "$OUT/plans.tsv" 2>/dev/null)
    # shellcheck disable=SC2086
    run "$1.$2.$3.$4" env MOJOLEARN_GEMM_ARM="$_env_arm" MOJOLEARN_GEMM_PLAN_LABEL="${_label:-unlabeled}" \
        timeout -k 30 "$DEADLINE" $PY tools/gemm_ksplit_classical_ab.py time \
        --lane "$1" --dataset "$2" --arm-name "$3" --block "$4" --rounds "$ROUNDS" \
        --rows "$ROWS" --gp-train "$GP_TRAIN" --gp-test "$GP_TEST" --prep "$PREP" \
        --device "$GPU_NAME"
}

gpu_snapshot "$OUT/gpu_before_timing.txt"
for ds in $(echo "$DATASETS" | tr ',' ' '); do
    for lane in $(echo "$LANES" | tr ',' ' '); do
        b=1
        while [ "$b" -le "$BLOCKS" ]; do
            if [ $((b % 2)) -eq 1 ]; then
                order="tuned128 default"
            else
                order="default tuned128"
            fi
            for arm in $order; do
                timed "$lane" "$ds" "$arm" "$b"
            done
            b=$((b + 1))
        done
    done
done
gpu_snapshot "$OUT/gpu_after.txt"

# ---- which plan each caller GEMM ran (the Mojo dispatch, host only) ------------
grep -h '^FSPEED-GEMM ' "$OUT"/*.log 2>/dev/null \
    | sed -n 's/.* caller=\([^ ]*\) op=\([A-Z]*\) m=\([0-9]*\) n=\([0-9]*\) k=\([0-9]*\).*/\1 \2 \3 \4 \5/p' \
    | sort -u > "$OUT/gemm_shapes.txt"
: > "$OUT/dispatch.txt"
if [ -x "$OUT/bin/step-price" ]; then
    while read -r caller op m n k; do
        for arm in shipped tuned128; do
            env MOJOLEARN_GEMM_ARM="$arm" MOJOLEARN_GEMM_STEP_LABEL_ONLY=1 \
                MOJOLEARN_GEMM_STEP_LABEL_CALLER="$caller.$op" MOJOLEARN_GEMM_STEP_LABEL_M="$m" \
                MOJOLEARN_GEMM_STEP_LABEL_N="$n" MOJOLEARN_GEMM_STEP_LABEL_K="$k" \
                timeout 120 "$OUT/bin/step-price" < /dev/null 2>&1 | grep '^DISPATCH' >> "$OUT/dispatch.txt"
        done
    done < "$OUT/gemm_shapes.txt"
    printf 'dispatch\t0\t%s lines\n' "$(wc -l < "$OUT/dispatch.txt" | tr -d ' ')" >> "$OUT/status.tsv"
else
    printf 'dispatch\t9\tno label binary\n' >> "$OUT/status.tsv"
    rc=1
fi

# ---- the verdict ---------------------------------------------------------------
# shellcheck disable=SC2086
run verdicts $PY tools/gemm_ksplit_classical_ab.py verdict --out "$OUT" --lanes "$LANES"
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/gate.txt"
exit "$rc"
