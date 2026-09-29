#!/bin/bash
# lane/linfit-speed dev check: build, bits at 100k rows (against the base
# tree's words from gate job nvc1-0003), the SGD per-row probe, then the
# board-shape Poisson with the phase trace.
set -uo pipefail
T=${T:-/root/mojolearn-linfit-speed-dev}
cd $T
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_GPU_ARCHS=sm_89
sh bindings/build_x_linear.sh 2>&1 | grep -A8 " error:\|^built"
PY=".pixi/envs/default/bin/python"
O=bench/results/linfit_speed/dev-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $O
G="--data /root/linfit-data --rows 100000 --max-iter 5"
$PY tools/linfit_speed.py $G --lanes poisson,sgd-reg,sgd-clf --out $O/g.json | grep -v mbind
$PY tools/linfit_speed.py $G --lanes poisson --out $O/g-cells.json --env MOJOLEARN_X_LINEAR_GW_GRAM=0 | grep -v mbind
$PY tools/linfit_speed.py $G --lanes poisson --out $O/g-tiny.json --env MOJOLEARN_X_LINEAR_GW_STEPS=4096 | grep -v mbind
echo "base words 100k x 5: poisson taxi b67554ce99356987 istella dda00dc764f7d232; sgd-reg taxi f4690c8c2dc5f6fa istella 6f539256eac50e5f; sgd-clf taxi be52a6ba0f80954c istella 958638218ebef738"
T=$T bash bench/results/linfit_speed/nv_sgdprobe.sh
$PY tools/linfit_speed.py --data /root/linfit-data --lanes poisson --out $O/full-poisson.json --env MOJOLEARN_X_LINEAR_GW_TRACE=1 | grep -v mbind
echo "base words full: poisson taxi 4d7dede9ad8f88ca istella 6afc53f88796e11c"
