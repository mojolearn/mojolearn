#!/bin/bash
set -euo pipefail
cd /root/mojolearn
out=/root/trees_out/tuning
while [ ! -f "$out/completed.txt" ]; do sleep 10; done
export MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python CUDA_VISIBLE_DEVICES=0
unset RF_LAUNCH_LOG MOJOLEARN_SPEED_FORTRAN
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu --format=csv -l 1 > "$out/rf-retry-gpu-telemetry.csv" &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true' EXIT
set +e
timeout -k 20 1500 tools/with_build_lock.sh python3 -u bench/speed/rf_higgs_columns_ab.py --bindings build/rf-higgs-columns --rows 1000000 --rounds 6 --output "$out/rf-higgs-1m-single-prediction" > "$out/rf-higgs-1m-single-prediction.log" 2>&1
echo "$?" > "$out/rf-higgs-1m-single-prediction.exit"
date -u > "$out/retry-completed.txt"
