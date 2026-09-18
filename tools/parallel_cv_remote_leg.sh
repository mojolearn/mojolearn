#!/bin/sh
# Re-exec because the guarded controller deliberately launches extras with sh.
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
# Extra body for a guarded TWO-GPU leg; wrapper owns provisioning and deletion.
# Every stage shares one deadline. Completed captures survive later failures.
set -uo pipefail
cd /root/mojolearn || exit 2
OUT=/root/gemm_leg_out/parallel-cv
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MOJOLEARN_CPU_THREADS=1
export MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python:/root/mojolearn
export MOJOLEARN_COMPILE_JOBS=2 MOJOLEARN_BUILD_JOBS=1
export MOJOLEARN_COMMIT=$(sed -n 's/^commit=//p' /root/gemm_leg_out/leg.txt)
export MOJOLEARN_GATE_COMMIT="$MOJOLEARN_COMMIT"
printf '%s\n' "$MOJOLEARN_COMMIT" > commit.txt
case "${MOJOLEARN_TARGET_COLUMN:-}" in
    amd|AMD) backend=hip ;;
    *) backend=cuda; export MOJOLEARN_TARGET_COLUMN=NVIDIA ;;
esac
# This source capsule is scheduled on L40S. The physical name is checked first.
nvidia-smi --query-gpu=name,uuid,pci.bus_id,driver_version --format=csv > "$OUT/devices.txt" 2>&1 || exit 2
if ! grep -q L40S "$OUT/devices.txt"; then
    echo 'This capsule expects the authorized L40S target' >&2; exit 2
fi
export MOJOLEARN_GPU_ARCHS=sm_89
started=$SECONDS
budget=2300
remaining() { echo $((budget - (SECONDS - started))); }
run_stage() {
    local label=$1; shift
    local left=$(remaining)
    (( left > 30 )) || return 124
    local begin=$SECONDS code=0
    timeout -k 10 "$left" "$@" > "$OUT/$label.log" 2>&1 || code=$?
    printf '%s\t%s\t%s\n' "$label" "$code" "$((SECONDS-begin))" >> "$OUT/stages.tsv"
    return "$code"
}
PY=$(pixi run python -c 'import sys; print(sys.executable)') || exit 2
profiler=$(command -v nsys || true)
if [[ -z "$profiler" ]]; then
    profiler=$(find /opt/nvidia /usr/local/cuda -type f -name nsys 2>/dev/null | head -1)
fi
if [[ -n "$profiler" ]]; then "$profiler" --version > "$OUT/profiler-version.txt" 2>&1; fi
capture() {
    local label=$1; shift
    if [[ -n "$profiler" ]]; then
        if run_stage "$label-profile" "$profiler" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
            --trace-fork-before-exec=true --wait=all --force-overwrite=true \
            -o "$OUT/$label-trace" "$@"; then
            run_stage "$label-export" "$profiler" export --type sqlite --force-overwrite=true \
                -o "$OUT/$label-trace.sqlite" "$OUT/$label-trace.nsys-rep" || true
            return 0
        fi
        # Preserve any partial capture; never overwrite it with the retry.
        echo 'Profiler/capture failed; inspect retained logs before interpreting results' >> "$OUT/profiler-findings.txt"
        return 1
    fi
    printf '%s\n' 'OWED: nsys not installed; PID/device inventory alone is not execution evidence' > "$OUT/physical-trace.txt"
    run_stage "$label" "$@"
}
failed=0
if run_stage build-gbdt bash bindings/build_gbdt.sh; then
    capture cv "$PY" tools/parallel_cross_val_check.py --require-backend "$backend" --devices 0,1 --out "$OUT/capture" || failed=1
else failed=1; fi
for family in arima tsa gp ivf; do
    run_stage "build-$family" bash "bindings/build_$family.sh" || failed=1
done
capture classical "$PY" tools/distributed_classical_check.py --devices 0,1 --out "$OUT/classical.json" || failed=1
# Existing freshly built GP/IVF/GBDT bindings can also pay down ordinary holds.
left=$(remaining)
if (( left > 180 )); then
    run_stage ordinary "$PY" tools/capture_ordinary_holds.py --source --python "$PY" \
        --backend cuda --vendor nvidia-L40S --output "$OUT/ordinary" --budget-seconds "$((left-20))" \
        --lanes gpc,gpc-multiclass,gp-normalize-y,gp-sample-y,gp-sample-y-normalize,gp-optimize,gp-optimize-restarts,ivf-extend,gbdt-query-rmse || failed=1
fi
printf '%s\n' "$failed" > "$OUT/exit_code"
exit "$failed"
