#!/bin/bash
# lane/linfit-speed: the bounded SGD path (x_linear/sgd_bounded.mojo) forced on
# NVIDIA. Words at 100k rows x 5 epochs against the base tree's (gate job
# nvc1-0003), with default budgets, tiny budgets (mid-epoch resumes, many
# shuffle slices) and the one-thread form; then the SGD identity lanes with
# tiny budgets, --pass 1, the bounded-path sabotage seen DISAGREE.
set -uo pipefail
T=${T:-/root/mojolearn-linfit-speed-dev}
cd $T
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_GPU_ARCHS=${MOJOLEARN_GPU_ARCHS:-sm_89}
sh bindings/build_x_linear.sh 2>&1 | grep -A8 " error:\|^built"
PY=".pixi/envs/default/bin/python"
O=bench/results/linfit_speed/bounded-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $O
G="--data /root/linfit-data --rows 100000 --max-iter 5 --lanes sgd-reg,sgd-clf"
B=MOJOLEARN_X_LINEAR_SGD_BOUNDED=1
$PY tools/linfit_speed.py $G --out $O/b.json --env $B | grep -v mbind
$PY tools/linfit_speed.py $G --out $O/tiny.json --env $B --env MOJOLEARN_X_LINEAR_SGD_STEPS=300000 --env MOJOLEARN_X_LINEAR_SGD_SWAPS=7001 | grep -v mbind
$PY tools/linfit_speed.py $G --out $O/thread.json --env $B --env MOJOLEARN_X_LINEAR_SGD_THREAD=1 --env MOJOLEARN_X_LINEAR_SGD_STEPS=3000000 | grep -v mbind
echo "base words 100k x 5: sgd-reg taxi f4690c8c2dc5f6fa istella 6f539256eac50e5f; sgd-clf taxi be52a6ba0f80954c istella 958638218ebef738"
export $B MOJOLEARN_X_LINEAR_SGD_STEPS=20000 MOJOLEARN_X_LINEAR_SGD_SWAPS=300
L=x-sgd-clf,x-sgd-reg,x-sgd-clf-w,x-sgd-reg-sw,x-sgd-ocsvm,x-perceptron,x-pa-clf,x-pa-reg
sh tools/algos_lane_check.sh "$L" --pass 1 --sabotage x_linear/checks/sabotage/e2e_sgd_bounded.patch
echo "LANE-CHECK bounded rc=$?"
MOJOLEARN_X_LINEAR_SGD_THREAD=1 sh tools/algos_lane_check.sh "$L" --pass 1
echo "LANE-CHECK bounded thread form rc=$?"
