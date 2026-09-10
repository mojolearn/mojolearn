#!/bin/bash
set -euo pipefail
cd /root/mojolearn
out=/root/trees_out/tuning
while [ ! -f "$out/retry-completed.txt" ]; do sleep 10; done
# Keep time to preserve evidence and reap before the existing one-hour lease.
if [ "$(date -u +%s)" -ge "$(date -u -d '2026-09-10 14:01:00 UTC' +%s)" ]; then
  echo 'Skipped: less than the reserved comparison/evidence window before lease expiry' > "$out/cuml-skipped.txt"
  exit 0
fi
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SPEED_EXPECTED_VENDOR=cuda PYTHONPATH=python CUDA_VISIBLE_DEVICES=0
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu --format=csv -l 1 > "$out/cuml-gpu-telemetry.csv" &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true' EXIT
set +e
timeout -k 20 600 tools/with_build_lock.sh python3 -u bench/speed/nvidia_identical_trees.py --lane rf --dataset higgs --rows 1000000 --rounds 5 --output "$out/rf-cuml-higgs-1m.json" > "$out/rf-cuml-higgs-1m.log" 2>&1
echo "$?" > "$out/rf-cuml-higgs-1m.exit"
date -u > "$out/cuml-completed.txt"
