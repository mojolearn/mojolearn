#!/bin/bash
# THE DECODE KERNEL on the H100, one queue job (run 9):
#   1. the unit gate with its sabotage arms (the decode kernel on codes at
#      every shape of m <= 16, the quantizer in its launch, and the new
#      quantizer arm that must fail it);
#   2. the PTX counter;
#   3. ONLY IF THE GATE IS GREEN, the decode rows (t1, t8): fp32.v1, the
#      launcher's staged plan there, quantize A + that plan, every decode
#      plan on codes and every decode plan with the quantizer in the launch.
cd "$(dirname "$0")/../.." || exit 9
export MOJOLEARN_LOWBIT_BOX=h100
export MOJOLEARN_LOWBIT_RESULTS=bench/results/lowbit_mma_speed
ARMS="fp32.v1,int8i32.v1.flat,int8i32.v1.mma"
ARMS="$ARMS,int8i32.v1.mma.staged.w16x16.b32x32.k64.l16,int8i32.v1.mma.decode*"
ARMS="$ARMS,convert.int8.quantize.a.par,inference.int8i32.v1.tuned,inference.int8i32.v1.quant-in-launch*"
export MOJOLEARN_LOWBIT_PRICE_ARMS=$ARMS
export MOJOLEARN_LOWBIT_PRICE_ONLY=t1,t8
red=0
gate=0
bash tools/lowbit_mma_speed/gate_job.sh unit || { red=1; gate=1; }
bash tools/lowbit_mma_speed/ptx_probe.sh || red=1
if [ "$gate" -eq 0 ]; then
    bash tools/lowbit_mma_speed/price_job.sh decode || red=1
else
    echo "h100_decode_job: the unit gate is RED; NOTHING TIMED"
fi
echo "h100_decode_job: red=$red gate_red=$gate"
exit "$red"
