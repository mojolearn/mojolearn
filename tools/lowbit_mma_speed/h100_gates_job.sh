#!/bin/bash
# The gates only, on the H100 (run 12): this lane's unit and four-product
# gates with their sabotage arms, and lane/lowbit-int15's gate on this tree.
# Times nothing.
cd "$(dirname "$0")/../.." || exit 9
export MOJOLEARN_LOWBIT_BOX=h100
export MOJOLEARN_LOWBIT_RESULTS=bench/results/lowbit_mma_speed
red=0
bash tools/lowbit_mma_speed/gate_job.sh unit || red=1
bash tools/lowbit_mma_speed/gate_job.sh pieces || red=1
OUT=bench/results/lowbit_mma_speed/h100/gate_int15
rm -rf "$OUT"; mkdir -p "$OUT"
export PATH="$HOME/.pixi/bin:$PATH"
pixi run check-gemm-int15-tuned > "$OUT/int15_tuned.log" 2>&1
rc=$?
echo "check-gemm-int15-tuned exit=$rc (lane/lowbit-int15's gate on this tree)" | tee "$OUT/verdict.txt"
grep -E "GATE FAILED|^ok |gates," "$OUT/int15_tuned.log" | tee -a "$OUT/verdict.txt"
echo "h100_gates_job: red=$red int15_gate=$rc"
exit "$red"
