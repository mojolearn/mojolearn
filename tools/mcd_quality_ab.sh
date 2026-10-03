#!/bin/bash
# Quality-only A/B of MOJOLEARN_MCD_DEVICE_CSTEPS (x_decomp/mcd_fast.mojo) on an Apple Mac (the M2 via
# lq CMD). Builds x_decomp three ways in this tree: FAST + -D MOJOLEARN_MCD_DEVICE_CSTEPS_OFF ("off"),
# FAST ("fast", left installed last), IDENTICAL ("identical"); fits EllipticEnvelope and MinCovDet on
# the board's taxi arrays with each and prints tools/mcd_quality_ab.py's table. No timing.
#   MCDQ_DATA=<rows-full dir>  MCDQ_PY=<python with numpy/scipy>  (defaults: the board cache + venv)
set -u
cd "$(dirname "$0")/.."
export PATH=$HOME/.pixi/bin:$PATH PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE
B=""
for b in board-0834 board-0833; do [ -d $HOME/$b/cache/algos-data/rows-full ] && { B=$HOME/$b; break; }; done
DATA=${MCDQ_DATA:-$B/cache/algos-data/rows-full}
PY=${MCDQ_PY:-$B/cache/venv/bin/python}
[ -d "$DATA" ] && [ -x "$PY" ] || { echo "MCDQ FAIL no data ($DATA) or python ($PY)"; exit 2; }
OUT=$PWD/mcdq-out; rm -rf "$OUT"; mkdir -p "$OUT"
echo "tree $(git rev-parse --short HEAD) data $DATA py $PY"

arm() { # label mode extra-flags
  local L=$1 M=$2 F=$3 so
  MOJOLEARN_NUMERIC_MODE=$M MOJOLEARN_MOJO_BUILD_FLAGS="$F" pixi run bash bindings/build_x_decomp.sh > "$OUT/build-$L.log" 2>&1
  local rc=$?
  so=python/mojolearn/_mojolearn_x_decomp.so; [ "$M" = identical ] && so=python/mojolearn/identical/_mojolearn_x_decomp.so
  echo "BUILD $L rc=$rc $(tail -1 "$OUT/build-$L.log" | cut -c1-160) sha=$(shasum -a 256 $so 2>/dev/null | cut -c1-12)"
  [ $rc = 0 ] || { grep -m 5 -B 2 -A 8 error "$OUT/build-$L.log"; return 1; }
  (MOJOLEARN_NUMERIC_MODE=$M MOJOLEARN_BENCH_INSTALLED=0 PYTHONPATH=$PWD/python "$PY" tools/mcd_quality_ab.py fit \
      --data "$DATA" --out "$OUT/$L.npz" > "$OUT/fit-$L.log" 2>&1)
  rc=$?
  echo "FIT $L rc=$rc $(tail -1 "$OUT/fit-$L.log" | cut -c1-200)"
  [ $rc = 0 ] || tail -n 15 "$OUT/fit-$L.log"
}

arm off fast "-D MOJOLEARN_MCD_DEVICE_CSTEPS_OFF"
arm fast fast ""
arm identical identical ""
"$PY" tools/mcd_quality_ab.py compare --dir "$OUT" fast off identical
