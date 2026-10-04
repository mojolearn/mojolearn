#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# One fit/arm on the M3; save fitted quantities and timing from that same fit.
# Manager must first confirm branch base == current main (or merge main).
# Usage: bash tools/mcd_compat_ab.sh TAG DATASET [LANE] [ROWS]
# Prebuilt overrides: MCD_A_SO, MCD_B_SO. They must be default/current-main
# and this commit with MOJOLEARN_MCD_BATCH_COMPAT, respectively.
set -euo pipefail
TAG=$1
DATASET=$2
LANE=${3:-min-cov-det}
ROWS=${4:-}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
OUT="$HOME/afc-def/$TAG"
mkdir -p "$OUT"
PY="$HOME/board-0834/cache/venv/bin/python"
DATA="$HOME/board-0834/cache/algos-data/rows-full"
SO="$ROOT/python/mojolearn/_mojolearn_x_decomp.so"
for ARM in A B; do
  PREBUILT=${MCD_A_SO:-}
  DEFINES=
  if [ "$ARM" = B ]; then
    PREBUILT=${MCD_B_SO:-}
    DEFINES='-D MOJOLEARN_MCD_BATCH_COMPAT'
  fi
  if [ -n "$PREBUILT" ]; then
    cp "$PREBUILT" "$OUT/$ARM.so"
  else
    if MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_MOJO_BUILD_FLAGS="$DEFINES" \
       MOJOLEARN_COMPILE_JOBS=1 MOJOLEARN_SKIP_BUILD_GATE=1 \
       bash bindings/build_x_decomp.sh > "$OUT/build_$ARM.log" 2>&1; then
      cp "$SO" "$OUT/$ARM.so"
    else
      echo "MCDQ-BUILD arm=$ARM FAIL"
      grep -m 5 -B 2 -A 8 -i error "$OUT/build_$ARM.log" | cut -c1-300 || true
      exit 1
    fi
  fi
  echo "MCDQ-BUILD arm=$ARM head=$(git rev-parse HEAD) defines='$DEFINES'"
  cp "$OUT/$ARM.so" "$SO.tmp"
  mv "$SO.tmp" "$SO"
  ARGS=()
  [ -z "$ROWS" ] || ARGS=(--rows "$ROWS")
  MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_VENDOR=metal PYTHONPATH="$ROOT/python" \
    "$PY" tools/mcd_compat_quality.py fit "$DATA" "$DATASET" "$OUT/$ARM.npz" \
    --lane "$LANE" "${ARGS[@]}"
done
"$PY" tools/mcd_compat_quality.py compare "$OUT/A.npz" "$OUT/B.npz"
