#!/bin/bash
# TWO PAGES AND THE FUSED EPILOGUE on the H100, one queue job (run 8):
#   1. the four-product gate with its three sabotage arms (every plan, the
#      two-page plans and the FUSED form of every plan included);
#   2. the PTX counter (registers, local bytes, blocks per multiprocessor,
#      cp.async count of the new instantiations);
#   3. ONLY IF THE GATE IS GREEN, the timing: every row, fp32.v1, one tuned
#      product (the launcher's plan at the wide rows), four launches of it,
#      every four-product plan, the two-launch operation and the fused one.
cd "$(dirname "$0")/../.." || exit 9
export MOJOLEARN_LOWBIT_BOX=h100
export MOJOLEARN_LOWBIT_RESULTS=bench/results/lowbit_mma_speed
ARMS="fp32.v1,int8i32.v1.flat,int8i32.v1.mma"
ARMS="$ARMS,int8i32.v1.mma.staged.w32x32.b128x128.k64.l16,int8i32.v1.mma.staged.w16x16.b32x32.k64.l16"
ARMS="$ARMS,pieces.int8*,convert.int8.quantize.a.par"
ARMS="$ARMS,inference.int8i32.v1.tuned,inference.4x.int8i32.v1.tuned"
ARMS="$ARMS,inference.pieces.int8.tuned,inference.pieces.int8.fused"
export MOJOLEARN_LOWBIT_PRICE_ARMS=$ARMS
red=0
gate=0
bash tools/lowbit_mma_speed/gate_job.sh pieces || { red=1; gate=1; }
bash tools/lowbit_mma_speed/ptx_probe.sh || red=1
if [ "$gate" -eq 0 ]; then
    bash tools/lowbit_mma_speed/price_job.sh pipe || red=1
else
    echo "h100_pipe_job: the four-product gate is RED; NOTHING TIMED"
fi
echo "h100_pipe_job: red=$red gate_red=$gate"
exit "$red"
