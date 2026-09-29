#!/bin/bash
# lane/linfit-speed: the tiled Gram lever (GLM). Bits: gram vs one-cell
# kernel vs tiny launches at 100k rows; then the board shape with the trace.
set -uo pipefail
T=${T:-/root/mojolearn-linfit-speed-dev}
cd $T
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_GPU_ARCHS=sm_89
sh bindings/build_x_linear.sh 2>&1 | grep -A8 " error:\|^built"
[ -f python/mojolearn/identical/_mojolearn.so ] || sh bindings/build.sh 2>&1 | grep -A8 " error:\|^built"
PY=".pixi/envs/default/bin/python"
O=bench/results/linfit_speed/gram-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $O
G="--data /root/linfit-data --lanes poisson --rows 100000 --max-iter 5"
$PY tools/linfit_speed.py $G --out $O/g-gram.json --env MOJOLEARN_X_LINEAR_GW_TRACE=1 | grep -v mbind
$PY tools/linfit_speed.py $G --out $O/g-cells.json --env MOJOLEARN_X_LINEAR_GW_GRAM=0 | grep -v mbind
$PY tools/linfit_speed.py $G --out $O/g-tiny.json --env MOJOLEARN_X_LINEAR_GW_STEPS=4096 | grep -v mbind
$PY tools/linfit_speed.py --data /root/linfit-data --lanes poisson --out $O/full-gram.json --env MOJOLEARN_X_LINEAR_GW_TRACE=1 | grep -v mbind
