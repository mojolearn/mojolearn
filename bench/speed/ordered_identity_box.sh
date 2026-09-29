#!/bin/sh
# lane/ordered-speed: the GPU == CPU identity check of the Ordered speed work
# and the M2 8-bit histogram cap on one box (NVIDIA or AMD queue), with the
# GPU-only sabotage that must make every lane DISAGREE.
#
#   sh bench/speed/ordered_identity_box.sh [OUTDIR]
#
# gbdt-symmetric carries the 254-border fit (`borders254`, the fused 8-bit
# histogram arm).
set -u
cd "$(dirname "$0")/../.."
export MOJOLEARN_NUMERIC_MODE=identical
pixi install -e default > /dev/null 2>&1 || pixi install -e default
exec pixi run -e default python -u tools/algos_lane_check.py \
    gbdt-symmetric,gbdt-ordered,gbdt-ordered-rmse,gbdt-ordered-bayesian-noise \
    --sabotage gbdt/checks/sabotage/ordered_speed_gpu_only.patch --pass 1 \
    --out "${1:-$PWD/ord-idcheck}"
