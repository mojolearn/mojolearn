#!/usr/bin/env bash
# On-box source capture, launched only by an external guarded rental controller.
# Does not provision anything. Entire body (builds included) is deadline bounded.
# Usage: ordinary_holds_gpu_leg.sh cuda|hip VENDOR OUTPUT [BUDGET_SECONDS]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
[[ $# -ge 3 && $# -le 4 ]] || { echo 'Expected BACKEND VENDOR OUTPUT [BUDGET_SECONDS]' >&2; exit 2; }
case "$1" in cuda|hip) ;; *) echo 'Expected cuda or hip' >&2; exit 2 ;; esac
BACKEND=$1
VENDOR=$2
OUT=$(realpath -m "$3")
BUDGET=${4:-3600}
[[ "$BUDGET" =~ ^[1-9][0-9]*$ ]] || exit 2
mkdir -p "$OUT"
# A parent watchdog remains mandatory for rental lifetime. This inner deadline
# prevents a slow build from consuming the capture budget invisibly.
if [[ ${MOJOLEARN_ORDINARY_BOUNDED:-0} != 1 ]]; then
    exec timeout -k 15 "$BUDGET" env MOJOLEARN_ORDINARY_BOUNDED=1 \
        bash "$0" "$BACKEND" "$VENDOR" "$OUT" "$BUDGET"
fi
cd "$ROOT"
printf '1\n' > "$OUT/exit_code"
if [[ -s commit.txt ]]; then
    MOJOLEARN_COMMIT=$(cat commit.txt)
else
    MOJOLEARN_COMMIT=$(git rev-parse HEAD)
fi
[[ "$MOJOLEARN_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo 'Missing source witness' >&2; exit 2; }
export MOJOLEARN_COMMIT MOJOLEARN_GATE_COMMIT="$MOJOLEARN_COMMIT"
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_COMPILE_JOBS=1 MOJOLEARN_BUILD_JOBS=1
export MOJOLEARN_CPU_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 OMP_THREAD_LIMIT=1
: "${MOJOLEARN_GPU_ARCHS:?guard must set the exact target architecture}"
case "$BACKEND" in cuda) export MOJOLEARN_TARGET_COLUMN=NVIDIA ;; hip) export MOJOLEARN_TARGET_COLUMN=AMD ;; esac
printf 'commit=%s\nbackend=%s\narch=%s\nbudget=%s\n' \
    "$MOJOLEARN_COMMIT" "$BACKEND" "$MOJOLEARN_GPU_ARCHS" "$BUDGET" > "$OUT/context.txt"
start=$SECONDS
# Minimal family set for the 23 held routes; estimates are not qualification.
# Estimators-host supplies the independent six-kernel saved-model CPU replay.
for build in build build_training build_mamba build_transformer build_gbdt \
             build_mixture build_gp build_ivf build_svm build_kernel_methods \
             build_estimators_host; do
    remaining=$((BUDGET - (SECONDS - start)))
    (( remaining > 0 )) || exit 124
    stage_start=$SECONDS
    set +e
    if [[ "$build" = build_estimators_host ]]; then
        timeout -k 10 "$remaining" env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_TARGET_COLUMN=cpu \
            bash "bindings/$build.sh" > "$OUT/$build.log" 2>&1
    else
        timeout -k 10 "$remaining" bash "bindings/$build.sh" > "$OUT/$build.log" 2>&1
    fi
    code=$?
    set -e
    printf '%s\t%s\t%s\n' "$build" "$code" "$((SECONDS-stage_start))" >> "$OUT/build-status.tsv"
    (( code == 0 )) || exit "$code"
done
remaining=$((BUDGET - (SECONDS - start)))
(( remaining > 0 )) || exit 124
PY=$(pixi run python -c 'import sys; print(sys.executable)')
"$PY" tools/capture_ordinary_holds.py --source --python "$PY" \
    --backend "$BACKEND" --vendor "$VENDOR" --output "$OUT/captures" \
    --budget-seconds "$remaining" --stage-seconds 600
printf '0\n' > "$OUT/exit_code"
