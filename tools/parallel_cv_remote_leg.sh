#!/bin/sh
# Extra body for a guarded TWO-GPU leg; wrapper owns provisioning and deletion.
set -eu
cd /root/mojolearn
OUT=/root/gemm_leg_out/parallel-cv
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python:/root/mojolearn
# The guarded wrapper writes this witness after checking archive provenance.
if [ -z "${MOJOLEARN_COMMIT:-}" ]; then
    MOJOLEARN_COMMIT=$(sed -n 's/^commit=//p' /root/gemm_leg_out/leg.txt)
    export MOJOLEARN_COMMIT
fi
case "${MOJOLEARN_TARGET_COLUMN:-}" in
    amd) backend=hip ;;
    nvidia|nv) backend=cuda ;;
    *)
        if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L > "$OUT/devices.txt"; then
            backend=cuda
        else
            echo 'explicit GPU target column required' >&2; exit 2
        fi ;;
esac
# Build and execution each have their own bound; the outer rental watchdog
# additionally bounds the whole leg including environment installation.
timeout 1200s sh bindings/build_gbdt.sh > "$OUT/build.log" 2>&1
timeout 1200s pixi run python tools/parallel_cross_val_check.py \
    --require-backend "$backend" --devices 0,1 --out "$OUT/capture" \
    > "$OUT/capture.log" 2>&1
