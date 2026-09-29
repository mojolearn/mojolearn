#!/bin/bash
# lane/linfit-speed identity on NVIDIA: the linear identity lanes the change
# touches (GLM, SGD family), --pass 2 (every seam driver of
# tools/identity_lanes/linear.checks with its sabotage arm seen failing),
# and the lane's own e2e sabotage seen DISAGREE then AGREE after reversal.
set -uo pipefail
cd ${T:-/root/mojolearn-linfit-speed}
git log --oneline -1 2>/dev/null || true
export MOJOLEARN_GPU_ARCHS=${MOJOLEARN_GPU_ARCHS:-sm_89}
L=x-glm-poisson,x-glm-gamma,x-glm-tweedie,x-glm-poisson-sw,x-sgd-clf,x-sgd-reg,x-sgd-clf-w,x-sgd-reg-sw
sh tools/algos_lane_check.sh "$L" --pass 2 --sabotage x_linear/checks/sabotage/e2e_linfit_wide.patch
echo "LANE-CHECK sabotaged set rc=$?"
sh tools/algos_lane_check.sh x-sgd-ocsvm,x-perceptron,x-pa-clf,x-pa-reg --pass 2
echo "LANE-CHECK clean set rc=$?"
