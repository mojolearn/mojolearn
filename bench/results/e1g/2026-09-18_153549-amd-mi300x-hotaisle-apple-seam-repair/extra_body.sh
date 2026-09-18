#!/bin/sh
# lane/apple-seam-repair (2026-09-18): the NVIDIA / AMD side of the Apple seam
# repair. Runs ON THE BOX as MOJOLEARN_GEMM_LEG_EXTRA of tools/gemm_remote_leg.sh
# (after its default payload, the GEMM identity card). No dataset.
#
# 1. gemm/checks/gemm_seam_probe.mojo: the shipped lane must still hash rtf
#    (62a6b5621e27c707) on this column; the new `nativefix` lane is the native
#    FMA with `rtf_fix`, the identity off Apple, so it reads `none` here.
# 2. gemm/checks/gemm_rtf_boundary_check.mojo: whole device GEMMs on the
#    adversarial words against the host oracle; its per-case device_fnv lines
#    must equal the Apple column's (bench/results/e1g/2026-09-18_apple-m4-gemm-rtf-boundary-check).
# POSIX sh (dash on RunPod images).
set -u
ROOT=${MOJOLEARN_GEMM_STEP_ROOT:-/root/mojolearn}
OUT=${MOJOLEARN_GEMM_STEP_LEG_OUT:-/root/gemm_leg_out/rtf}
mkdir -p "$OUT"
cd "$ROOT" || exit 9
export PATH="$HOME/.pixi/bin:$PATH"
if [ -e /dev/kfd ]; then
    export MOJOLEARN_TARGET_COLUMN=amd MOJOLEARN_GPU_ARCHS=${MOJOLEARN_GPU_ARCHS:-gfx942}
    COLUMN=MOJOLEARN_COLUMN_AMD
elif command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then
    export MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_GPU_ARCHS=${MOJOLEARN_GPU_ARCHS:-sm_90a}
    COLUMN=MOJOLEARN_COLUMN_NVIDIA
    if [ -x /usr/local/cuda/bin/ptxas ]; then
        export MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas
    fi
else
    echo "vendor=unknown; nothing built" > "$OUT/status.txt"; exit 9
fi
echo "column=$MOJOLEARN_TARGET_COLUMN arch=$MOJOLEARN_GPU_ARCHS started=$(date -u +%FT%TZ)" > "$OUT/status.txt"
for prog in gemm_seam_probe gemm_rtf_boundary_check; do
    if pixi run mojo build -j 2 --target-accelerator "$MOJOLEARN_GPU_ARCHS" -D "$COLUMN" \
        -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . "gemm/checks/$prog.mojo" -o "$OUT/$prog" \
        > "$OUT/build-$prog.log" 2>&1; then
        "$OUT/$prog" > "$OUT/$prog.log" 2>&1
        echo "$prog build=0 run=$?" >> "$OUT/status.txt"
    else
        echo "$prog build=$?" >> "$OUT/status.txt"
    fi
    rm -f "$OUT/$prog"
done
python3 tools/gemm_seam_probe_reference.py "$OUT/gemm_seam_probe.log" > "$OUT/reference.txt" 2>&1
echo "reference=$?" >> "$OUT/status.txt"
# 3. Optional (Hot Aisle, whose runner has no card payload): the GEMM
#    identity card of this commit, to diff HERE against the Apple card.
if [ "${MOJOLEARN_RTF_LEG_CARD:-0}" = 1 ]; then
    sh tools/gemm_card.sh device "$OUT/$MOJOLEARN_TARGET_COLUMN.card" > "$OUT/card.log" 2>&1
    echo "card=$?" >> "$OUT/status.txt"
fi
echo "finished=$(date -u +%FT%TZ)" >> "$OUT/status.txt"
