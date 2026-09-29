#!/bin/bash
# lane/lowbit-blocks on the H100: rebuild at this commit, then the checked time.
cd "$(dirname "$0")/../.." || exit 9
LB_BOX=h100-nvc3 LB_HOST_BUILDS='' LB_RUNS="fp32_v1:auto:time fixed15_v1:auto:time fp32_v1:auto:time fixed15_v1:auto:time" \
    LB_OUT="$PWD/bench/results/lowbit_blocks/h100-nvc3/time_checked" bash tools/lowbit_blocks/model_job.sh
