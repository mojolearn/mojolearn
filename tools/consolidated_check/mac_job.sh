#!/bin/bash
# One committed consolidated tree; output directories belong to one exact run.
set -euo pipefail
cd "$(dirname "$0")/../.."
SHARD=${1:-0/1}
OUT=${2:?output directory}
mkdir -p "$OUT"
PIXI=$(command -v pixi || echo "$HOME/.pixi/bin/pixi")
export MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-2}
P=("$PIXI" run -e default python -u tools/consolidated_check/check.py)
"$PIXI" install -e default > "$OUT/pixi_install.log" 2>&1
# Always regenerate the plan so changed LANES is caught by resume identity.
"${P[@]}" plan --out "$OUT" --lanes "${LANES:-}"
"${P[@]}" build --out "$OUT" --jobs "${BUILD_JOBS:-2}" 2>&1 | tee "$OUT/build.stdout"
failed=0
if [ "${RUN_RADIX:-0}" = 1 ]; then
    if ! "${P[@]}" radix --out "$OUT" --arm-timeout "${ARM_TIMEOUT:-120}"; then
        failed=1
    fi
fi
if ! "${P[@]}" clean --out "$OUT" --shard "$SHARD" --cpu-threads "${CPU_THREADS:-default}" --fixtures "${FIXTURES:-base}" --arm-timeout "${ARM_TIMEOUT:-120}" 2>&1 | tee "$OUT/clean.stdout"; then
    failed=1
fi
if [ "$failed" -ne 0 ]; then
    echo "CONSOLIDATED CHECK FAILED (see radix-public.log and clean.stdout)" >&2
    exit 1
fi
echo "CONSOLIDATED CHECK PASS $(date -u +%FT%TZ)"
