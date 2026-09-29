#!/bin/bash
# THE TARGET on the H100 as ONE short queue job: the unit gate first (the
# launcher's plans changed, and a plan is timed only after its gate), then
# the complete operations on the tuned launcher's plans beside fp32.v1 and
# the reference unit plan. Built first, run with nothing else of the lane's
# on the box.
cd "$(dirname "$0")/../.." || exit 9
ARMS="fp32.v1,int8i32.v1.flat,int8i32.v1.mma"
ARMS="$ARMS,int8i32.v1.mma.staged.w32x32.b128x128.k64.l16,int8i32.v1.mma.staged.w16x16.b32x32.k64.l16"
ARMS="$ARMS,convert.int8.quantize.a.par,convert.int8.pack.b.par"
ARMS="$ARMS,inference.int8i32.v1.tuned,training.int8i32.v1.tuned"
ARMS="$ARMS,inference.4x.int8i32.v1.tuned,training.4x.int8i32.v1.tuned"
export MOJOLEARN_LOWBIT_PRICE_ARMS=$ARMS
export MOJOLEARN_LOWBIT_BOX=h100
export MOJOLEARN_LOWBIT_RESULTS=bench/results/lowbit_mma_speed
red=0
bash tools/lowbit_mma_speed/gate_job.sh unit || red=1
bash tools/lowbit_mma_speed/price_job.sh target || red=1
echo "h100_target_job: red=$red"
exit "$red"
