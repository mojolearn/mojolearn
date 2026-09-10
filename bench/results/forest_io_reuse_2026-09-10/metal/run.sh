#!/bin/bash
set -euo pipefail
cd /Users/andrewhendel/CascadeProjects/mojolearn-trees-gpu
export BENCH_LOCK_PID=$$
tools/bench_lock.sh acquire forest-io-reuse-metal 'FAST RF HIGGS and ET Year buffer reuse' '5 minutes'
trap 'tools/bench_lock.sh release' EXIT INT TERM
export MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_SPEED_EXPECTED_VENDOR=metal
export DYLD_FALLBACK_LIBRARY_PATH="$PWD/.pixi/envs/default/lib"
export GBM_BENCH_DATA="$HOME/datasets/gbm-bench" PYTHONPATH=python
mkdir -p bench/results/forest_io_reuse_2026-09-10/metal
for cell in rf-higgs et-year; do
  lane=${cell%%-*}; dataset=${cell#*-}; rows=1000000
  if [ "$dataset" = year ]; then rows=500000; fi
  /tmp/mojolearn-b1-sklearn/bin/python bench/speed/forest_inference_ab.py --lane "$lane" --dataset "$dataset" --rows "$rows" --trees 100 --depth 16 --rounds 8 --reuse-io --output "bench/results/forest_io_reuse_2026-09-10/metal/$cell-calls1.json" > "bench/results/forest_io_reuse_2026-09-10/metal/$cell-calls1.log" 2>&1
done
