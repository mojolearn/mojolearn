#!/bin/sh
# Qualify the selected production seam, then compare to the legacy spelling.
set -eu
cd /root/mojolearn
[ -e /dev/kfd ] || { echo 'AMD device required'; exit 9; }
export MOJOLEARN_TARGET_COLUMN=amd MOJOLEARN_GPU_ARCHS=gfx942
export PATH="$HOME/.pixi/bin:$PATH"
OUT=/root/gemm_leg_out/class-default
mkdir -p "$OUT"
build() {
    pixi run mojo build -j 2 --target-accelerator gfx942 -D MOJOLEARN_COLUMN_AMD \
        -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . "$@"
}
build -D MOJOLEARN_CLASS_PROBE_SABOTAGE=1 gemm/checks/gemm_seam_probe.mojo \
    -o "$OUT/probe" > "$OUT/build-sabotage.log" 2>&1
"$OUT/probe" > "$OUT/sabotage.log" 2>&1
if python3 tools/gemm_class_gate.py "$OUT/sabotage.log" --require-shipped-class \
    > "$OUT/sabotage-gate.log" 2>&1; then exit 9; fi
grep '^SEAM_DIFF shipped/class' "$OUT/sabotage.log" > "$OUT/sabotage-matches.txt"
build gemm/checks/gemm_seam_probe.mojo -o "$OUT/probe" > "$OUT/build-clean.log" 2>&1
"$OUT/probe" > "$OUT/clean.log" 2>&1
python3 tools/gemm_class_gate.py "$OUT/clean.log" --require-shipped-class > "$OUT/clean-gate.log" 2>&1
for arm in legacy default; do
    EXTRA=
    [ "$arm" != legacy ] || EXTRA='-D MOJOLEARN_GEMM_LEGACY_CLASS_FLUSH=1'
    build $EXTRA -D MOJOLEARN_GEMM_ARM_TRIAL=1 bench/gemm_step_price_main.mojo \
        -o "$OUT/price" > "$OUT/build-price-$arm.log" 2>&1
    cp "$OUT/price" "$OUT/price-$arm"
done
for arm in legacy default default legacy; do
    n=$(find "$OUT" -name "price-$arm-*.log" | wc -l | tr -d ' ')
    MOJOLEARN_GEMM_ARM=shipped MOJOLEARN_GEMM_STEP_ROUNDS=11 \
        "$OUT/price-$arm" > "$OUT/price-$arm-$n.log" 2>&1
done
python3 tools/gemm_class_price_summary.py "$OUT"/price-*.log > "$OUT/cross-build-gate.log"
rm "$OUT/probe" "$OUT/price" "$OUT/price-legacy" "$OUT/price-default"
echo 'COMPLETE production-default seam and fixed-size GEMM gate'
