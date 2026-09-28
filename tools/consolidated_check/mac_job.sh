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
"${P[@]}" clean --out "$OUT" --shard "$SHARD" --cpu-threads "${CPU_THREADS:-default}" 2>&1 | tee "$OUT/clean.stdout"
echo "CONSOLIDATED CHECK PASS $(date -u +%FT%TZ)"
