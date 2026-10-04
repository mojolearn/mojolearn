#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# One fit/arm on the M3; save fitted quantities and timing from that same fit.
# Manager must first confirm branch base == current main (or merge main).
# Usage: bash tools/mcd_compat_ab.sh TAG DATASET [LANE] [ROWS]
# Prebuilt overrides: MCD_A_SO, MCD_B_SO. They must be default/current-main
# and this commit with MCD_DEFINE (default MOJOLEARN_MCD_BATCH_COMPAT).
# MCD_BASELINE_NPZ may reuse a saved A only after manager source review.
# Comparison still enforces exact dataset/lane/shape/input hash agreement.
set -euo pipefail
TAG=$1
DATASET=$2
LANE=${3:-min-cov-det}
ROWS=${4:-}
MCD_DEFINE=${MCD_DEFINE:-MOJOLEARN_MCD_BATCH_COMPAT}
# MCD_DEFINE may list several defines, space or comma separated (w2-mcd2: B =
# "MOJOLEARN_MCD_WIDE"; BMMA is default since its promotion, so
# "MOJOLEARN_MCD_BMMA_OFF" builds the pre-promotion path).
B_DEFINES=
for D in ${MCD_DEFINE//,/ }; do
  case "$D" in
    MOJOLEARN_MCD_BATCH_COMPAT|MOJOLEARN_MCD_BATCH_MMA|MOJOLEARN_MCD_BMMA_OFF|MOJOLEARN_MCD_WIDE) ;;
    MOJOLEARN_MCD_BMMA) echo "MCDQ MOJOLEARN_MCD_BMMA is default now; use MOJOLEARN_MCD_BMMA_OFF for the old arm"; exit 2 ;;
    *) echo "MCDQ unsupported define: $D"; exit 2 ;;
  esac
  B_DEFINES="$B_DEFINES -D $D"
done
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
OUT="$HOME/afc-def/$TAG"
if [ -e "$OUT/A.npz" ] || [ -e "$OUT/B.npz" ] || [ -e "$OUT/PASS" ]; then
  echo "MCDQ refusing to replay/overwrite existing fit artifacts: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"
PY="$HOME/board-0834/cache/venv/bin/python"
DATA="$HOME/board-0834/cache/algos-data/rows-full"
SO="$ROOT/python/mojolearn/_mojolearn_x_decomp.so"
for ARM in A B; do
  if [ "$ARM" = A ] && [ -n "${MCD_BASELINE_NPZ:-}" ]; then
    set --
    [ -z "$ROWS" ] || set -- --rows "$ROWS"
    PYTHONPATH="$ROOT/python" "$PY" tools/mcd_compat_quality.py verify-baseline       "$DATA" "$DATASET" "$MCD_BASELINE_NPZ" --lane "$LANE" "$@"
    cp "$MCD_BASELINE_NPZ" "$OUT/A.npz"
    echo "MCDQ-REUSE arm=A source=$MCD_BASELINE_NPZ (manager source review required)"
    continue
  fi
  PREBUILT=${MCD_A_SO:-}
  DEFINES=
  if [ "$ARM" = B ]; then
    PREBUILT=${MCD_B_SO:-}
    DEFINES="${B_DEFINES# }"
  fi
  # A manager-staged arm at $OUT/<arm>.so (as afc_ab_def.sh's AFC_SKIP_BUILD) is used as is.
  if [ -z "$PREBUILT" ] && [ -f "$OUT/$ARM.so" ]; then PREBUILT="$OUT/$ARM.so"; fi
  if [ -n "$PREBUILT" ]; then
    [ "$PREBUILT" = "$OUT/$ARM.so" ] || cp "$PREBUILT" "$OUT/$ARM.so"
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
  set --
  [ -z "$ROWS" ] || set -- --rows "$ROWS"
  MCD_FIT_ARM="$ARM" MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_VENDOR=apple PYTHONPATH="$ROOT/python" \
    "$PY" tools/mcd_compat_quality.py fit "$DATA" "$DATASET" "$OUT/$ARM.npz" \
    --lane "$LANE" "$@"
done
"$PY" tools/mcd_compat_quality.py compare "$OUT/A.npz" "$OUT/B.npz"
touch "$OUT/PASS"
echo "MCDQ-PAIR-PASS $TAG"
