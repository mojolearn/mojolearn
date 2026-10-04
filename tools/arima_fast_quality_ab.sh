#!/bin/bash
# One quality-only capture per afc-def arm; no build and no timing/opponents.
# Usage: bash tools/arima_fast_quality_ab.sh <afc-def-tag>
set -euo pipefail
arima_tag=${1:?provide afc-def tag}
[[ "$arima_tag" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 2
arima_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$arima_root"
arima_python=$HOME/board-0834/cache/venv/bin/python
arima_arms=$HOME/afc-def/$arima_tag
arima_out=$HOME/mq/out/arima-quality-$arima_tag
set --
[[ "${ARIMA_QUALITY_SMALL:-0}" != 1 ]] || set -- --small
# ARIMA_QUALITY_WIDE=1 adds the board-shaped 48-series fixture (seed61_n1392).
[[ "${ARIMA_QUALITY_WIDE:-0}" != 1 ]] || set -- "$@" --wide
arima_so=$arima_root/python/mojolearn/_mojolearn_arima.so
[[ -x "$arima_python" && -f "$arima_arms/A.so" && -f "$arima_arms/B.so" && -f "$arima_so" ]]
mkdir -p "$arima_out"
# Refuse accidental repeated captures and preserve the installed arm on error.
[[ ! -e "$arima_out/original.so" && ! -e "$arima_out/A.npz" && ! -e "$arima_out/B.npz" ]]
cp "$arima_so" "$arima_out/original.so"
restore_arima() {
  cp "$arima_out/original.so" "$arima_so.restore"
  mv -f "$arima_so.restore" "$arima_so"
  rm "$arima_out/original.so"
}
trap restore_arima EXIT
for arima_arm in A B; do
  cp "$arima_arms/$arima_arm.so" "$arima_so.next"
  mv -f "$arima_so.next" "$arima_so"
  PYTHONPATH="$arima_root/python" MOJOLEARN_VENDOR=apple MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_BENCH_INSTALLED=0 \
    "$arima_python" tools/arima_fast_fit_quality.py capture "$arima_out/$arima_arm.npz" "$@" \
    > "$arima_out/$arima_arm.log" 2>&1 || { tail -n 12 "$arima_out/$arima_arm.log"; exit 1; }
  grep '^ARIMA_FAST_CAPTURE ' "$arima_out/$arima_arm.log"
done
"$arima_python" tools/arima_fast_fit_quality.py compare "$arima_out/A.npz" "$arima_out/B.npz" \
  | tee "$arima_out/compare.log"
