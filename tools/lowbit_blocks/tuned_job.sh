#!/bin/bash
# lane/lowbit-blocks: ONE LEVER, the tuned unit plan on NVIDIA (int15_block.mojo,
# LLAMA_INT15_TUNED). The block gate (profile == host oracle, sabotage, default
# gate), the model's full logits (must equal the reference plan's d37c2ea8...),
# the whole-model sabotage arm with the tuned epilogue patched too, then the time.
# Submit with every GPU slot of the pod (the time runs alone).
cd "$(dirname "$0")/../.." || exit 9
bash tools/lowbit_blocks/gate_job.sh
LB_HOST_BUILDS='' LB_RUNS="fixed15_v1:auto:identity,decode,batch" LB_OUT="$PWD/bench/results/lowbit_blocks/${LB_BOX:-tuned}/model" bash tools/lowbit_blocks/model_job.sh
bash tools/lowbit_blocks/cpu_sab_job.sh
bash tools/lowbit_blocks/time_job.sh
