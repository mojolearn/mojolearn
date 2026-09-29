#!/bin/sh
# lane/lowbit-blocks: the model job on a steward box (m2pro, do-amd). Finds the
# staged SmolLM2-360M, then runs tools/lowbit_blocks/model_job.sh (which builds
# the bindings) with the runs named in LB_RUNS.
cd "$(dirname "$0")/../.." || exit 9
M=""
for d in "${LB_MODEL:-}" "$HOME/models/SmolLM2-360M" /root/models/SmolLM2-360M; do
    [ -n "$d" ] && [ -f "$d/model.safetensors" ] && { M=$d; break; }
done
if [ -z "$M" ]; then echo "MODEL NOT STAGED on $(hostname -s): no SmolLM2-360M/model.safetensors"; ls "$HOME/models" /root/models 2>&1 | head; exit 3; fi
echo "model $M"
git rev-parse HEAD > .lowbit_blocks_commit
LB_MODEL=$M LB_OUT="$PWD/bench/results/lowbit_blocks/${LB_BOX:-$(hostname -s)}/model" exec bash tools/lowbit_blocks/model_job.sh
