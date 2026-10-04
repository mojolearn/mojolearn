#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# One diagnostic MinCovDet/EllipticEnvelope fit with -D MOJOLEARN_MCD_PROFILE
# (x_decomp/mcd_fast.mojo): prints the MCDPROF stage lines (per phase: steps,
# moments+covariance, log determinant, pinvh+precision, distance, final
# distance; per fit: buffers, finish). Not a scored timing (it syncs per stage).
# Usage: bash tools/mcd_profile.sh TAG DATASET [LANE] [ROWS]
# Prebuilt: a manager-staged $HOME/afc-def/$TAG/P.so is used as is; otherwise
# builds x_decomp with "-D MOJOLEARN_MCD_PROFILE $MCD_PROFILE_EXTRA".
# The installed x_decomp .so is restored afterwards.
set -euo pipefail
TAG=$1
DATASET=$2
LANE=${3:-min-cov-det}
ROWS=${4:-}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
OUT="$HOME/afc-def/$TAG"
mkdir -p "$OUT"
[ ! -e "$OUT/P.npz" ] || { echo "MCDPROF refusing to overwrite $OUT/P.npz" >&2; exit 2; }
PY="$HOME/board-0834/cache/venv/bin/python"
DATA="$HOME/board-0834/cache/algos-data/rows-full"
SO="$ROOT/python/mojolearn/_mojolearn_x_decomp.so"
[ ! -f "$SO" ] || cp "$SO" "$OUT/installed.so"
if [ ! -f "$OUT/P.so" ]; then
  if MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_MOJO_BUILD_FLAGS="-D MOJOLEARN_MCD_PROFILE ${MCD_PROFILE_EXTRA:-}" \
     MOJOLEARN_COMPILE_JOBS=1 MOJOLEARN_SKIP_BUILD_GATE=1 \
     bash bindings/build_x_decomp.sh > "$OUT/build_P.log" 2>&1; then
    cp "$SO" "$OUT/P.so"
  else
    echo "MCDPROF-BUILD FAIL"
    grep -m 5 -B 2 -A 8 -i error "$OUT/build_P.log" | cut -c1-300 || true
    exit 1
  fi
fi
cp "$OUT/P.so" "$SO.tmp" && mv "$SO.tmp" "$SO"
set --
[ -z "$ROWS" ] || set -- --rows "$ROWS"
rc=0
MCD_FIT_ARM=P MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_VENDOR=apple PYTHONPATH="$ROOT/python" \
  "$PY" tools/mcd_compat_quality.py fit "$DATA" "$DATASET" "$OUT/P.npz" --lane "$LANE" "$@" \
  > "$OUT/prof.log" 2>&1 || rc=$?
if [ -f "$OUT/installed.so" ]; then cp "$OUT/installed.so" "$SO.tmp" && mv "$SO.tmp" "$SO"; fi
grep -m 12 -E '^MCDPROF|^MCDQ-FIT' "$OUT/prof.log" | cut -c1-400 || true
[ $rc = 0 ] || { echo "MCDPROF-FIT rc=$rc"; tail -n 5 "$OUT/prof.log" | cut -c1-300; exit $rc; }
echo "MCDPROF-DONE $TAG"
