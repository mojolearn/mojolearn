#!/bin/bash
# lane/lowbit-blocks on the H100 (nvc3, lane name lowbit-blocks-h100): the model job, identity only.
cd "$(dirname "$0")/../.." || exit 9
LB_BOX=h100-nvc3 LB_RUNS="fixed15_v1:auto:identity,decode,batch fp32_v1:auto:identity" \
    LB_OUT="$PWD/bench/results/lowbit_blocks/h100-nvc3/model" exec bash tools/lowbit_blocks/model_job.sh
