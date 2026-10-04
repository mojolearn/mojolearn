#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Unscored AutoARIMA phase/RMSE diagnostic over staged afc-def arms (lane w2-ts).
# Usage: bash tools/arima_phase_diag.sh <afc-def-tag> <dataset> [search-maxiter for arm A]
# Arm A (main) runs the board call plus, if given, a converged-search variant;
# arm B runs the board call only. No build, no opponent; installed .so restored.
set -euo pipefail
tag=${1:?afc-def tag}; ds=${2:?dataset}; smax=${3:-0}
[[ "$tag" =~ ^[A-Za-z0-9_.-]+$ ]] && [[ "$ds" = synthetic || "$ds" = taxi-hourly ]]
root=$(cd "$(dirname "$0")/.." && pwd); cd "$root"
py=$HOME/board-0834/cache/venv/bin/python
arms=$HOME/afc-def/$tag; out=$HOME/mq/out/arima-diag-$tag-$ds
so=$root/python/mojolearn/_mojolearn_arima.so
[[ -f "$arms/A.so" && -f "$arms/B.so" && -f "$so" ]]
mkdir -p "$out"; [[ ! -e "$out/original.so" ]]
cp "$so" "$out/original.so"
restore() { cp "$out/original.so" "$so.restore"; mv -f "$so.restore" "$so"; rm "$out/original.so"; }
trap restore EXIT
for arm in A B; do
  cp "$arms/$arm.so" "$so.next"; mv -f "$so.next" "$so"
  extra=(); [[ $arm = A && "$smax" != 0 ]] && extra=(--search-maxiter "$smax")
  PYTHONPATH="$root/python" MOJOLEARN_VENDOR=apple MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_BENCH_INSTALLED=0 \
    "$py" tools/arima_phase_diag.py "$ds" --out "$out/$arm.json" ${extra[@]+"${extra[@]}"} > "$out/$arm.log" 2>&1 \
    || { tail -n 12 "$out/$arm.log"; exit 1; }
  grep '^ARIMA_DIAG ' "$out/$arm.log" | sed "s/^/arm=$arm /"
done
