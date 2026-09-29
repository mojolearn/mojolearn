#!/bin/bash
# lane/lowbit-blocks: the lane's final code on the 4090: the block gate (with the default
# gate) and the model's full logits, both profiles, the GPU and the CPU host path.
cd "$(dirname "$0")/../.." || exit 9
bash tools/lowbit_blocks/gate_job.sh
LB_RUNS="fixed15_v1:auto:identity,decode,batch fp32_v1:auto:identity" LB_OUT="$PWD/bench/results/lowbit_blocks/final/model" bash tools/lowbit_blocks/model_job.sh
bash tools/lowbit_blocks/cpu_sab_job.sh
