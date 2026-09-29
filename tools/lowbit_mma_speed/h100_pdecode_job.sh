#!/bin/bash
# Run 11 on the H100, one queue job: FOUR PRODUCTS AT THE DECODE ROWS.
#   1. the unit gate and the four-product gate with their sabotage arms (the
#      four-product decode plans in both forms at every shape of m <= 16;
#      the one-product launcher now on the decode kernel there);
#   2. lane/lowbit-int15's gate on this tree;
#   3. the PTX counter;
#   4. ONLY IF 1 IS GREEN, the decode rows (t1, t8): fp32.v1, the one-product
#      decode plans and the launcher, the four-product plans (the two-page
#      small block and the decode plans), the two-launch and the fused
#      operation on the launcher's plan.
cd "$(dirname "$0")/../.." || exit 9
export MOJOLEARN_LOWBIT_BOX=h100
export MOJOLEARN_LOWBIT_RESULTS=bench/results/lowbit_mma_speed
ARMS="fp32.v1,int8i32.v1.flat,int8i32.v1.mma,pieces.int8.flat"
ARMS="$ARMS,int8i32.v1.mma.staged.w16x16.b32x32.k64.l16,int8i32.v1.mma.decode*,convert.int8.quantize.a.par"
ARMS="$ARMS,inference.int8i32.v1.tuned,pieces.int8.mma.pipe2.w16x16.b32x32.k64.l16,pieces.int8.mma.decode*"
ARMS="$ARMS,inference.pieces.int8.tuned,inference.pieces.int8.fused"
export MOJOLEARN_LOWBIT_PRICE_ARMS=$ARMS
export MOJOLEARN_LOWBIT_PRICE_ONLY=t1,t8
red=0
gate=0
bash tools/lowbit_mma_speed/gate_job.sh unit || { red=1; gate=1; }
bash tools/lowbit_mma_speed/gate_job.sh pieces || { red=1; gate=1; }
OUT=bench/results/lowbit_mma_speed/h100/gate_int15
rm -rf "$OUT"; mkdir -p "$OUT"
export PATH="$HOME/.pixi/bin:$PATH"
pixi run check-gemm-int15-tuned > "$OUT/int15_tuned.log" 2>&1
rc=$?
echo "check-gemm-int15-tuned exit=$rc (lane/lowbit-int15's gate on this tree; expected 0)" | tee "$OUT/verdict.txt"
[ "$rc" -eq 0 ] || red=1
bash tools/lowbit_mma_speed/ptx_probe.sh || red=1
if [ "$gate" -eq 0 ]; then
    bash tools/lowbit_mma_speed/price_job.sh pdecode || red=1
else
    echo "h100_pdecode_job: a gate is RED; NOTHING TIMED"
fi
echo "h100_pdecode_job: red=$red gate_red=$gate"
exit "$red"
