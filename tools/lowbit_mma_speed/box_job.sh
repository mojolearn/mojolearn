#!/bin/bash
# tools/lowbit_mma_speed/box_job.sh -- lane/lowbit-mma-speed, phases of the
# lane on ONE box, as one job (one job at a time per lane on a shared box):
#
#   bash tools/lowbit_mma_speed/box_job.sh <box name> <phase> [<phase> ...]
#
#   quant-gate   tools/lowbit_mma_speed/gate_job.sh quant   (every column)
#   unit-gate    tools/lowbit_mma_speed/gate_job.sh unit    (NVIDIA, AMD)
#   pieces-gate  tools/lowbit_mma_speed/gate_job.sh pieces  (NVIDIA, AMD)
#   quant-price  tools/lowbit_mma_speed/price_job.sh quant: fp32.v1, the
#                int8 products the box has, the reference quantizer and the
#                parallel one, and the complete operations with each
#   unit-price   tools/lowbit_mma_speed/price_job.sh unit: fp32.v1, the
#                reference unit plan, every tuned plan, the direct kernel
#   target-price tools/lowbit_mma_speed/price_job.sh target: fp32.v1, the
#                tuned launcher's plans, FOUR PRODUCTS (four launches, and
#                one launch of one staging) and the conversions
#   full-price   tools/lowbit_mma_speed/price_job.sh full: the unit
#                phase's arms and the target phase's in ONE run, for a box
#                that is given one job (the MI325X)
#   amd-gate     tools/lowbit_mma_speed/gate_job.sh amd     (AMD; the plans
#                of gemm/checks/gemm_int8_mma_amd.mojo, lane/lowbit-amd-tuned)
#   amd-price    tools/lowbit_mma_speed/price_job.sh amd with the harness
#                bench/gemm_lowbit_amd_price_main.mojo: fp32.v1, the
#                reference unit plan, every tuned plan and every AMD plan,
#                one product and four, and the complete operation on every
#                four-product plan
#   amd-fault-repro  tools/lowbit_amd_tuned/fault_repro.sh (AMD): the
#                byte-path gate built four times and each run three times,
#                every launch named; whether job 1790657510941's fault comes
#                back, and whether it belongs to a build or to a run
#   ptx          tools/lowbit_mma_speed/ptx_probe.sh (NVIDIA): the kernels'
#                PTX, counted; launches nothing, times nothing
#   apple-gates  (Apple) the int8 flat plan's gate and its value arm, and the
#                four products' reference device plan against the host: both
#                plans now launch in slices bounded in work on Apple
#   flat-slices  (Apple) the int8 flat plan, sliced, at the three widest
#                512-token rows beside the Apple chunk probe: the harness
#                poisons every output and refuses a surviving poison, the two
#                digests must agree, cold run and run of record
#   quant-price-noflat  the quantizer phase WITHOUT the multi-second flat
#                products (the orchestrator, 2026-09-29)
#
# Every phase runs even when an earlier one failed (a red phase is a
# finding); the exit is non-zero when any phase's was. The box name is what
# the results are filed under (h100, mi325x, m3ultra, m2pro).
set -u
cd "$(dirname "$0")/../.." || exit 9
[ $# -ge 2 ] || { echo "box_job.sh <box> <phase> [<phase> ...]" >&2; exit 2; }
export MOJOLEARN_LOWBIT_BOX=$1
export MOJOLEARN_LOWBIT_RESULTS=${MOJOLEARN_LOWBIT_RESULTS:-bench/results/lowbit_mma_speed}
shift
#: The quantizer phase's arms: whole names (bench/gemm_lowbit_price_main.mojo).
QUANT_ARMS="fp32.v1,int8i32.v1.flat,int8i32.v1.mma,int8i32.v1.applechunk"
QUANT_ARMS="$QUANT_ARMS,convert.int8.quantize.a,convert.int8.pack.b"
QUANT_ARMS="$QUANT_ARMS,convert.int8.quantize.a.par,convert.int8.pack.b.par"
QUANT_ARMS="$QUANT_ARMS,inference.int8i32.v1,training.int8i32.v1"
QUANT_ARMS="$QUANT_ARMS,inference.int8i32.v1.parq,training.int8i32.v1.parq"
QUANT_ARMS="$QUANT_ARMS,inference.int8i32.v1.applechunk,training.int8i32.v1.applechunk"
QUANT_ARMS="$QUANT_ARMS,inference.int8i32.v1.applechunk.parq,training.int8i32.v1.applechunk.parq"
#: The unit phase's arms. A name ending in * is a prefix.
UNIT_ARMS="fp32.v1,int8i32.v1.flat,int8i32.v1.mma*,probe.int8.mma*"
UNIT_ARMS="$UNIT_ARMS,convert.int8.quantize.a.par,convert.int8.pack.b.par"
UNIT_ARMS="$UNIT_ARMS,inference.int8i32.v1.tuned,training.int8i32.v1.tuned"
UNIT_ARMS="$UNIT_ARMS,inference.4x.int8i32.v1.tuned,training.4x.int8i32.v1.tuned"
#: The target phase's arms.
TARGET_ARMS="fp32.v1,int8i32.v1.flat,int8i32.v1.mma"
TARGET_ARMS="$TARGET_ARMS,int8i32.v1.mma.staged.w32x32.b128x128.k64.l16,int8i32.v1.mma.staged.w16x16.b32x32.k64.l16"
TARGET_ARMS="$TARGET_ARMS,convert.int8.quantize.a.par,convert.int8.pack.b.par"
TARGET_ARMS="$TARGET_ARMS,inference.int8i32.v1.tuned,training.int8i32.v1.tuned"
TARGET_ARMS="$TARGET_ARMS,inference.4x.int8i32.v1.tuned,training.4x.int8i32.v1.tuned"
TARGET_ARMS="$TARGET_ARMS,pieces.int8*,inference.pieces.int8.tuned,training.pieces.int8.tuned"
#: The AMD phase's arms: every unit plan of either file (the prefix takes
#: the reference, the tuned and the AMD plans), the four-product plans, and
#: the complete operations.
AMD_ARMS="fp32.v1,int8i32.v1.flat,int8i32.v1.mma*,probe.amd*"
AMD_ARMS="$AMD_ARMS,convert.int8.quantize.a.par,convert.int8.pack.b.par"
AMD_ARMS="$AMD_ARMS,inference.int8i32.v1.tuned,training.int8i32.v1.tuned"
AMD_ARMS="$AMD_ARMS,inference.4x.int8i32.v1.tuned,training.4x.int8i32.v1.tuned"
AMD_ARMS="$AMD_ARMS,inference.int8i32.v1.amd,training.int8i32.v1.amd"
AMD_ARMS="$AMD_ARMS,inference.4x.int8i32.v1.amd,training.4x.int8i32.v1.amd"
AMD_ARMS="$AMD_ARMS,pieces.int8*,inference.pieces.int8*,training.pieces.int8*"
#: The quantizer phase without the flat products.
QUANT_NOFLAT_ARMS="fp32.v1,int8i32.v1.applechunk"
QUANT_NOFLAT_ARMS="$QUANT_NOFLAT_ARMS,convert.int8.quantize.a,convert.int8.pack.b"
QUANT_NOFLAT_ARMS="$QUANT_NOFLAT_ARMS,convert.int8.quantize.a.par,convert.int8.pack.b.par"
QUANT_NOFLAT_ARMS="$QUANT_NOFLAT_ARMS,inference.int8i32.v1.applechunk,training.int8i32.v1.applechunk"
QUANT_NOFLAT_ARMS="$QUANT_NOFLAT_ARMS,inference.int8i32.v1.applechunk.parq,training.int8i32.v1.applechunk.parq"
apple_gates() {
    local out="$MOJOLEARN_LOWBIT_RESULTS/$MOJOLEARN_LOWBIT_BOX/gate_apple"
    rm -rf "$out"; mkdir -p "$out"
    local r=0
    export PATH="$HOME/.pixi/bin:$PATH"
    pixi run check-gemm-lowbit > "$out/lowbit.log" 2>&1; local a=$?
    pixi run check-gemm-lowbit-sabotage > "$out/lowbit-sabotage.log" 2>&1; local b=$?
    pixi run check-gemm-int8-pieces-tuned > "$out/pieces.log" 2>&1; local c=$?
    {
        echo "check-gemm-lowbit exit=$a (expected 0)"
        echo "check-gemm-lowbit-sabotage exit=$b (expected non-zero)"
        echo "check-gemm-int8-pieces-tuned exit=$c (expected 0: the reference device plan against the host)"
    } | tee "$out/verdict.txt"
    [ "$a" -eq 0 ] || r=1
    [ "$b" -ne 0 ] || r=1
    [ "$c" -eq 0 ] || r=1
    tail -5 "$out/lowbit.log" "$out/pieces.log"
    return "$r"
}
red=0
summary=""
for phase in "$@"; do
    echo "######## phase $phase on $MOJOLEARN_LOWBIT_BOX, started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    case "$phase" in
        quant-gate) bash tools/lowbit_mma_speed/gate_job.sh quant ;;
        unit-gate) bash tools/lowbit_mma_speed/gate_job.sh unit ;;
        pieces-gate) bash tools/lowbit_mma_speed/gate_job.sh pieces ;;
        full-price) MOJOLEARN_LOWBIT_PRICE_ARMS=${MOJOLEARN_LOWBIT_PRICE_ARMS:-$UNIT_ARMS,pieces.int8*,inference.pieces.int8.tuned,training.pieces.int8.tuned} bash tools/lowbit_mma_speed/price_job.sh full ;;
        target-price) MOJOLEARN_LOWBIT_PRICE_ARMS=${MOJOLEARN_LOWBIT_PRICE_ARMS:-$TARGET_ARMS} bash tools/lowbit_mma_speed/price_job.sh target ;;
        quant-price) MOJOLEARN_LOWBIT_PRICE_ARMS=${MOJOLEARN_LOWBIT_PRICE_ARMS:-$QUANT_ARMS} bash tools/lowbit_mma_speed/price_job.sh quant ;;
        unit-price) MOJOLEARN_LOWBIT_PRICE_ARMS=${MOJOLEARN_LOWBIT_PRICE_ARMS:-$UNIT_ARMS} bash tools/lowbit_mma_speed/price_job.sh unit ;;
        amd-gate) bash tools/lowbit_mma_speed/gate_job.sh amd ;;
        amd-price) MOJOLEARN_LOWBIT_PRICE_ARMS=${MOJOLEARN_LOWBIT_PRICE_ARMS:-$AMD_ARMS} MOJOLEARN_LOWBIT_PRICE_MAIN=bench/gemm_lowbit_amd_price_main.mojo bash tools/lowbit_mma_speed/price_job.sh amd ;;
        amd-fault-repro) bash tools/lowbit_amd_tuned/fault_repro.sh 4 3 ;;
        amd-fault-stress) bash tools/lowbit_amd_tuned/fault_stress.sh 25 ;;
        ptx) bash tools/lowbit_mma_speed/ptx_probe.sh ;;
        apple-gates) apple_gates ;;
        flat-slices) MOJOLEARN_LOWBIT_PRICE_ARMS="fp32.v1,int8i32.v1.flat,int8i32.v1.applechunk" MOJOLEARN_LOWBIT_PRICE_ONLY="mlp_up.t512,mlp_down.t512,lm_head.t512" bash tools/lowbit_mma_speed/price_job.sh flatslices ;;
        quant-price-noflat) MOJOLEARN_LOWBIT_PRICE_ARMS=$QUANT_NOFLAT_ARMS bash tools/lowbit_mma_speed/price_job.sh quantnoflat ;;
        *) echo "box_job.sh: unknown phase $phase" >&2; exit 2 ;;
    esac
    rc=$?
    echo "######## phase $phase on $MOJOLEARN_LOWBIT_BOX exit=$rc, finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    summary="$summary $phase=$rc"
    [ "$rc" -eq 0 ] || red=1
done
echo "box_job: box=$MOJOLEARN_LOWBIT_BOX phases:$summary red=$red"
exit "$red"
