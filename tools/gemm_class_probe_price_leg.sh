#!/bin/sh
# One device proof followed by fixed-size GEMM prices, no short LM numbers.
set -eu
cd "${MOJOLEARN_GEMM_STEP_ROOT:-/root/mojolearn}"
OUT=/root/gemm_leg_out/class-flush
mkdir -p "$OUT"
export PATH="$HOME/.pixi/bin:$PATH"
if [ -e /dev/kfd ]; then
    export MOJOLEARN_TARGET_COLUMN=amd MOJOLEARN_GPU_ARCHS=gfx942
    COLUMN=MOJOLEARN_COLUMN_AMD
else
    export MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_GPU_ARCHS=sm_90a
    COLUMN=MOJOLEARN_COLUMN_NVIDIA
    if [ -x /usr/local/cuda/bin/ptxas ]; then
        export MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas
    fi
fi
build() {
    pixi run mojo build -j 2 --target-accelerator "$MOJOLEARN_GPU_ARCHS" \
        -D "$COLUMN" -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . "$@"
}
# Same directory and output name for both builds. Do not build the closed mode arm.
build -D MOJOLEARN_CLASS_PROBE_SABOTAGE=1 gemm/checks/gemm_seam_probe.mojo -o "$OUT/probe" > "$OUT/build-sabotage.log" 2>&1
"$OUT/probe" > "$OUT/sabotage.log" 2>&1
if python3 tools/gemm_class_gate.py "$OUT/sabotage.log" > "$OUT/sabotage-gate.log" 2>&1; then
    echo 'FAIL: deliberately wrong device output was accepted'; exit 9
fi
grep '^SEAM_DIFF shipped/class' "$OUT/sabotage.log" > "$OUT/sabotage-matches.txt"
build gemm/checks/gemm_seam_probe.mojo -o "$OUT/probe" > "$OUT/build-clean.log" 2>&1
"$OUT/probe" > "$OUT/clean.log" 2>&1
python3 tools/gemm_class_gate.py "$OUT/clean.log" > "$OUT/clean-gate.log" 2>&1
# Only after the device proof do we compare legacy and production spellings.
for arm in base class; do
    EXTRA=
    [ "$arm" != base ] || EXTRA='-D MOJOLEARN_GEMM_LEGACY_CLASS_FLUSH=1'
    build $EXTRA -D MOJOLEARN_GEMM_ARM_TRIAL=1 bench/gemm_step_price_main.mojo \
        -o "$OUT/price" > "$OUT/build-price-$arm.log" 2>&1
    cp "$OUT/price" "$OUT/price-$arm"
done
for arm in base class class base; do
    n=$(find "$OUT" -name "price-$arm-*.log" | wc -l | tr -d ' ')
    MOJOLEARN_GEMM_ARM=shipped MOJOLEARN_GEMM_STEP_ROUNDS=11 \
        "$OUT/price-$arm" > "$OUT/price-$arm-$n.log" 2>&1
done
rm "$OUT/probe" "$OUT/price" "$OUT/price-base" "$OUT/price-class"
echo 'COMPLETE fixed-size microbenchmarks; no step-level measurements'
