#!/bin/sh
# Qualify reused grouped scratch, then ABBA prices on the fixed GEMM calls.
set -eu
cd "${MOJOLEARN_GEMM_STEP_ROOT:-/root/mojolearn}"
OUT=/root/gemm_leg_out/workspace-reuse
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
build -D MOJOLEARN_GEMM_REUSE_GROUP_WS=1 -D MOJOLEARN_GEMM_SABOTAGE_WS_FOLD=1 \
    gemm/checks/gemm_workspace_reuse_check.mojo -o "$OUT/check" > "$OUT/build-negative.log" 2>&1
if "$OUT/check" > "$OUT/negative.log" 2>&1; then
    echo 'FAIL: omitted workspace fold was accepted'; exit 9
fi
grep 'workspace output mismatch' "$OUT/negative.log"
for arm in base reuse; do
    EXTRA=
    [ "$arm" = base ] || EXTRA="-D MOJOLEARN_GEMM_REUSE_GROUP_WS=1"
    if [ "$arm" = reuse ]; then
        build $EXTRA gemm/checks/gemm_workspace_reuse_check.mojo -o "$OUT/check" > "$OUT/build-check-$arm.log" 2>&1
        "$OUT/check" > "$OUT/check-$arm.log" 2>&1
    fi
    build $EXTRA -D MOJOLEARN_GEMM_ARM_TRIAL=1 bench/gemm_step_price_main.mojo \
        -o "$OUT/price-$arm" > "$OUT/build-price-$arm.log" 2>&1
done
build -D MOJOLEARN_GEMM_REUSE_GROUP_WS=1 -D MOJOLEARN_GEMM_ARM_TRIAL=1 \
    -D MOJOLEARN_GEMM_PRICE_SABOTAGE=1 bench/gemm_step_price_main.mojo \
    -o "$OUT/price-negative" > "$OUT/build-price-negative.log" 2>&1
if MOJOLEARN_GEMM_ARM=kpack_hg \
    MOJOLEARN_GEMM_STEP_CALLS=proj_fwd "$OUT/price-negative" > "$OUT/price-negative.log" 2>&1; then
    echo 'FAIL: sabotaged price arm accepted'; exit 9
fi
grep '^BITS .* MOVED' "$OUT/price-negative.log"
for entry in base-1 reuse-1 reuse-2 base-2; do
    arm=${entry%-*}
    MOJOLEARN_GEMM_ARM=kpack_hg MOJOLEARN_GEMM_STEP_ROUNDS=11 \
        "$OUT/price-$arm" > "$OUT/price-$entry.log" 2>&1
done
python3 tools/gemm_class_price_summary.py "$OUT"/price-base-*.log "$OUT"/price-reuse-*.log > "$OUT/summary.log"
rm "$OUT/price-negative" "$OUT/check" "$OUT/price-base" "$OUT/price-reuse"
echo COMPLETE
