#!/bin/sh
# Qualify input-flush transport on the actual gather kernel, then ABBA prices.
set -eu
cd "${MOJOLEARN_GEMM_STEP_ROOT:-/root/mojolearn}"
OUT=/root/gemm_leg_out/gather-ftz
mkdir -p "$OUT"
export PATH="$HOME/.pixi/bin:$PATH"
if [ -e /dev/kfd ]; then
    export MOJOLEARN_GPU_ARCHS=gfx942
else
    export MOJOLEARN_GPU_ARCHS=sm_90a
fi
build() {
    pixi run mojo build -j 2 --target-accelerator "$MOJOLEARN_GPU_ARCHS" \
        -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . "$@"
}
build -D MOJOLEARN_GEMM_STAGE_FTZ=1 -D MOJOLEARN_GEMM_SABOTAGE_GATHER_FTZ=1 \
    gemm/checks/gemm_gather_ftz_check.mojo -o "$OUT/check" > "$OUT/build-negative.log" 2>&1
if "$OUT/check" > "$OUT/negative.log" 2>&1; then
    echo 'FAIL: omitted gather input flush was accepted'; exit 9
fi
grep 'got=.*flat=.*oracle=' "$OUT/negative.log"
for arm in base stage; do
    EXTRA=-D\ MOJOLEARN_GEMM_LEGACY_STAGE_FTZ=1
    [ "$arm" = base ] || EXTRA=-D\ MOJOLEARN_GEMM_STAGE_FTZ=1
    build $EXTRA gemm/checks/gemm_gather_ftz_check.mojo -o "$OUT/check" > "$OUT/build-check-$arm.log" 2>&1
    "$OUT/check" > "$OUT/check-$arm.log" 2>&1
    build $EXTRA -D MOJOLEARN_GEMM_ARM_TRIAL=1 bench/gemm_step_price_main.mojo \
        -o "$OUT/price-$arm" > "$OUT/build-price-$arm.log" 2>&1
done
if MOJOLEARN_GEMM_ARM=kpack_hg MOJOLEARN_GEMM_ARM_SABOTAGE=1 \
    MOJOLEARN_GEMM_STEP_CALLS=proj_fwd "$OUT/price-stage" > "$OUT/price-negative.log" 2>&1; then
    echo 'FAIL: sabotaged price arm accepted'; exit 9
fi
grep '^BITS .* MOVED' "$OUT/price-negative.log"
for entry in base-1 stage-1 stage-2 base-2; do
    arm=${entry%-*}
    MOJOLEARN_GEMM_ARM=shipped MOJOLEARN_GEMM_STEP_ROUNDS=11 \
        "$OUT/price-$arm" > "$OUT/price-$entry.log" 2>&1
done
python3 tools/gemm_class_price_summary.py "$OUT"/price-base-*.log "$OUT"/price-stage-*.log > "$OUT/summary.log"
rm "$OUT/check" "$OUT/price-base" "$OUT/price-stage"
echo COMPLETE
