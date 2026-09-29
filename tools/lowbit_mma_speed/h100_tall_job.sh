#!/bin/bash
# The TALL blocks on the H100 as ONE short queue job: the two gates (new
# plans are timed only after their gate), the PTX counter with the
# runtime's blocks per multiprocessor, then the training rows only: fp32.v1,
# every one-product plan of sixteen warps and more, every four-product plan.
cd "$(dirname "$0")/../.." || exit 9
export MOJOLEARN_LOWBIT_BOX=h100
export MOJOLEARN_LOWBIT_RESULTS=bench/results/lowbit_mma_speed
ARMS="fp32.v1,int8i32.v1.flat,int8i32.v1.mma"
ARMS="$ARMS,int8i32.v1.mma.staged.w32x32.b128x128.k64.l16,int8i32.v1.mma.staged.w16x32.b128x128.k64.l16"
ARMS="$ARMS,int8i32.v1.mma.staged.w32x32.b64x64.k64.l16"
ARMS="$ARMS,int8i32.v1.mma.staged.w32x32.b256x64.k64.l16,int8i32.v1.mma.staged.w32x32.b512x32.k64.l16"
ARMS="$ARMS,int8i32.v1.mma.staged.w32x16.b512x16.k64.l16"
ARMS="$ARMS,pieces.int8*,convert.int8.quantize.a.par,convert.int8.pack.b.par"
ARMS="$ARMS,inference.int8i32.v1.tuned,inference.4x.int8i32.v1.tuned,training.4x.int8i32.v1.tuned"
ARMS="$ARMS,inference.pieces.int8.tuned,training.pieces.int8.tuned"
export MOJOLEARN_LOWBIT_PRICE_ARMS=$ARMS
export MOJOLEARN_LOWBIT_PRICE_ONLY=t512
red=0
bash tools/lowbit_mma_speed/gate_job.sh unit || red=1
bash tools/lowbit_mma_speed/gate_job.sh pieces || red=1
bash tools/lowbit_mma_speed/ptx_probe.sh || red=1
bash tools/lowbit_mma_speed/price_job.sh tall || red=1
echo "h100_tall_job: red=$red"
exit "$red"
