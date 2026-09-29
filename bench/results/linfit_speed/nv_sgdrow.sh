#!/bin/bash
# lane/linfit-speed: the SGD row-path lever on the dev tree. Bits at 100k rows
# x 5 epochs against the base tree's words (gate job nvc1-0003), then the
# per-row cost probe.
set -uo pipefail
T=${T:-/root/mojolearn-linfit-speed-dev}
cd $T
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_GPU_ARCHS=sm_89
sh bindings/build_x_linear.sh 2>&1 | grep -A8 " error:\|^built"
PY=".pixi/envs/default/bin/python"
O=bench/results/linfit_speed/sgdrow-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $O
$PY tools/linfit_speed.py --data /root/linfit-data --lanes sgd-reg,sgd-clf --rows 100000 --max-iter 5 --out $O/g.json | grep -v mbind
echo "base words: sgd-reg taxi f4690c8c2dc5f6fa istella 6f539256eac50e5f; sgd-clf taxi be52a6ba0f80954c istella 958638218ebef738"
T=$T bash bench/results/linfit_speed/nv_sgdprobe.sh
