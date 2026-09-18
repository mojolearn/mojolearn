#!/usr/bin/env bash
set -euo pipefail
cd /root/mojolearn
export PATH=/root/.pixi/bin:$PATH
export MOJOLEARN_COMMIT=$(cat commit.txt)
export MOJOLEARN_GATE_COMMIT=$MOJOLEARN_COMMIT
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_GPU_ARCHS=gfx942 MOJOLEARN_TARGET_COLUMN=amd
export MOJOLEARN_COMPILE_JOBS=1 MOJOLEARN_BUILD_JOBS=1 MOJOLEARN_CPU_THREADS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
OUT=/root/gemm_leg_out/ordinary-holds/recovery-gp
mkdir -p "$OUT"
timeout -k 10 180 bash bindings/build_preprocessing.sh > "$OUT/build-preprocessing.log" 2>&1
PY=$(pixi run python -c 'import sys; print(sys.executable)')
"$PY" tools/capture_ordinary_holds.py --source --python "$PY" --backend hip --vendor amd-mi300x-gfx942 \
  --lanes gp-normalize-y,gp-sample-y-normalize --output "$OUT/capture" --budget-seconds 300 --stage-seconds 150
