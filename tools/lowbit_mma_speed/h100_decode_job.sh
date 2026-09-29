#!/bin/bash
# Run 9 on the H100, one queue job:
#   1. the unit gate with its sabotage arms (the DECODE KERNEL on codes at
#      every shape of m <= 16, the quantizer in its launch, and the
#      quantizer arm that must fail it);
#   2. the four-product gate with its arms: the FUSED form's first build on
#      lane/lowbit-int15's int15_store_cell, and the launcher on TWO PAGES;
#   3. lane/lowbit-int15's own gate (check-gemm-int15-tuned) on this tree:
#      its fifteen-bit product now runs on the two-page plans;
#   4. the PTX counter;
#   5. ONLY IF 1 AND 2 ARE GREEN, the timing at every row: fp32.v1, the
#      one-product plans, quantize A + one product, four launches of it, the
#      two-page four-product plans, the two-launch and the fused operation,
#      and (m <= 16 only) every decode plan on codes and with the quantizer
#      in the launch.
cd "$(dirname "$0")/../.." || exit 9
export MOJOLEARN_LOWBIT_BOX=h100
export MOJOLEARN_LOWBIT_RESULTS=bench/results/lowbit_mma_speed
ARMS="fp32.v1,int8i32.v1.flat,int8i32.v1.mma,pieces.int8.flat"
ARMS="$ARMS,int8i32.v1.mma.staged.w16x16.b32x32.k64.l16,int8i32.v1.mma.staged.w32x32.b128x128.k64.l16"
ARMS="$ARMS,int8i32.v1.mma.decode*,convert.int8.quantize.a.par"
ARMS="$ARMS,inference.int8i32.v1.tuned,inference.4x.int8i32.v1.tuned,inference.int8i32.v1.quant-in-launch*"
ARMS="$ARMS,pieces.int8.mma.pipe2*,pieces.int8.mma.staged.w16x32.b64x128.k64.l16"
ARMS="$ARMS,inference.pieces.int8.tuned,inference.pieces.int8.fused"
export MOJOLEARN_LOWBIT_PRICE_ARMS=$ARMS
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
    bash tools/lowbit_mma_speed/price_job.sh decode || red=1
else
    echo "h100_decode_job: a gate is RED; NOTHING TIMED"
fi
echo "h100_decode_job: red=$red gate_red=$gate"
exit "$red"
