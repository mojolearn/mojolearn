#!/bin/bash
# lane/lowbit-blocks on the H100: the tuned-plan build, the model's full logits, then the time.
cd "$(dirname "$0")/../.." || exit 9
LB_BOX=h100-nvc3 LB_HOST_BUILDS='' LB_RUNS="fixed15_v1:auto:identity,decode,batch fp32_v1:auto:identity" \
    LB_OUT="$PWD/bench/results/lowbit_blocks/h100-nvc3/model_tuned" bash tools/lowbit_blocks/model_job.sh
LB_BOX=h100-nvc3 bash tools/lowbit_blocks/time_job.sh
