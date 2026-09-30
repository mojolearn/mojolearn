#!/bin/sh
# tools/amd_step_time_leg6.sh -- lane/amd-step-time (2026-09-24): the MFMA
# probes (gemm/checks/amd_mfma_probe2.mojo: the 32x32x1 layout, the MODE
# field's effect on MFMA and on a multiply-by-one flush), then hold the box
# for the lane's session (the matrix-core GEMM kernel is built and tested
# interactively) until /root/amd_step_done.
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/amd-step-time
mkdir -p "$OUT/builds"
cd "$ROOT" || exit 9
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=amd
: "${MOJOLEARN_GPU_ARCHS:=gfx942}"; export MOJOLEARN_GPU_ARCHS
export PYTHONPATH="$ROOT/python:$ROOT"
BIN=/root/amd_bin; mkdir -p "$BIN"
ST="$OUT/session.txt"
say() { echo "$(date -u +%H:%M:%S) $*" >> "$ST"; }
say "leg6 started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
rocm-smi --showproductname --showuniqueid > "$OUT/gpu.txt" 2>&1
pixi run python -c 'import numpy' > /dev/null 2>&1 || pixi run python -m pip install numpy > "$OUT/numpy.log" 2>&1
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_AMD --target-accelerator gfx942 -I . \
    gemm/checks/amd_mfma_probe2.mojo -o "$BIN/mfma_probe2" > "$OUT/mfma2_build.log" 2>&1
say "mfma probe2 build exit=$?"
"$BIN/mfma_probe2" > "$OUT/mfma_probe2.log" 2>&1
say "mfma probe2 exit=$?: $(grep MFMA2_MODE "$OUT/mfma_probe2.log" | tr '\n' ' ' | cut -c1-600)"
( MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh > "$OUT/builds/base.log" 2>&1; say "base exit=$?" ) &
wait
touch /root/amd_step_ready
say "leg6 holding"
while [ ! -e /root/amd_step_done ]; do sleep 20; done
exit 0
