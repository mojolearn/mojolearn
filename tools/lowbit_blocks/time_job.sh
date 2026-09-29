#!/bin/bash
# lane/lowbit-blocks (f): whole-forward time, fixed15_v1 and fp32_v1, on this box, ALONE on it
# (submit with every GPU slot of the pod). The bindings are the ones the model job built.
cd "$(dirname "$0")/../.." || exit 9
export PATH="$HOME/.pixi/bin:$PATH"
M=""
for d in "${LB_MODEL:-}" "$HOME/models/SmolLM2-360M" /root/models/SmolLM2-360M; do
    [ -n "$d" ] && [ -f "$d/model.safetensors" ] && { M=$d; break; }
done
BOX=${LB_BOX:-$(hostname -s)}
OUT=$PWD/bench/results/lowbit_blocks/$BOX/time
mkdir -p "$OUT"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv
for p in fp32_v1 fixed15_v1 fp32_v1 fixed15_v1; do
    pixi run -e default python tools/lowbit_blocks/model_logits.py --model "$M" --profile $p --phases time --box "$BOX" --out "$OUT" 2>&1 | grep -v tcmalloc | grep -E "RESULT|rror|Traceback"
done
echo "out $OUT"
