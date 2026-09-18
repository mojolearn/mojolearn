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
BUDGET=${4:-2400}
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
PY=$(pixi run python -c 'import sys; print(sys.executable)')
SELECTED=${MOJOLEARN_ORDINARY_LANES:-}
# Validate before paying for any compilation. Explicit subsets enable short legs.
"$PY" - "$SELECTED" <<'CHECK'
import sys
sys.path.insert(0, 'tools')
from capture_ordinary_holds import LANES
selected = sys.argv[1].split(',') if sys.argv[1] else list(LANES)
if len(selected) != len(set(selected)) or set(selected) - set(LANES):
    raise SystemExit('Invalid MOJOLEARN_ORDINARY_LANES')
CHECK
select_lanes() {
    "$PY" - "$SELECTED" "$1" <<'SELECT'
import sys
wanted = set(sys.argv[1].split(',')) if sys.argv[1] else None
print(','.join(x for x in sys.argv[2].split(',') if wanted is None or x in wanted))
SELECT
}
build_binding() {
    local build=$1
    local remaining=$((BUDGET - (SECONDS - start)))
    (( remaining > 0 )) || return 124
    local stage_start=$SECONDS
    local code=0
    if [[ "$build" = *_host ]]; then
        timeout -k 10 "$remaining" env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_TARGET_COLUMN=cpu \
            bash "bindings/$build.sh" > "$OUT/$build.log" 2>&1 || code=$?
    else
        timeout -k 10 "$remaining" bash "bindings/$build.sh" > "$OUT/$build.log" 2>&1 || code=$?
    fi
    printf '%s\t%s\t%s\n' "$build" "$code" "$((SECONDS-stage_start))" >> "$OUT/build-status.tsv"
    return "$code"
}
# Finish a useful family before compiling the next. A late timeout keeps earlier
# model records and columns; a failed family does not erase unrelated results.
failed=0
group() {
    local name=$1
    local candidates=$2
    shift 2
    local lanes
    lanes=$(select_lanes "$candidates")
    [[ -n "$lanes" ]] || return 0
    local build
    for build in "$@"; do
        if ! build_binding "$build"; then failed=1; return 0; fi
    done
    local remaining=$((BUDGET - (SECONDS - start)))
    if (( remaining <= 0 )); then failed=1; return 0; fi
    if ! "$PY" tools/capture_ordinary_holds.py --source --python "$PY" \
        --backend "$BACKEND" --vendor "$VENDOR" --output "$OUT/captures-$name" \
        --lanes "$lanes" --budget-seconds "$remaining" --stage-seconds 600; then
        failed=1
    fi
}
build_binding build || exit $?
group kernels kernel-ridge-poly,kernel-ridge-sigmoid,kernel-ridge-laplacian,nystroem-poly,nystroem-sigmoid,nystroem-laplacian \
    build_kernel_methods build_estimators_host
group gp gp-normalize-y,gp-sample-y,gp-sample-y-normalize,gp-optimize,gp-optimize-restarts,gpc,gpc-multiclass build_gp
group mixture gmm-sample,gmm-random-init-sample build_mixture
group ivf ivf-extend build_ivf
group svm svc-poly build_svm
group gbdt gbdt-query-rmse build_gbdt
group neural mamba3,transformer,transformer-window,samba,samba-untied-dropout-accum \
    build_training build_mamba build_transformer
printf '%s\n' "$failed" > "$OUT/exit_code"
exit "$failed"
