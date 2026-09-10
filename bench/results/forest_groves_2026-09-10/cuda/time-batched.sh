#!/bin/bash
set -euo pipefail
cd /root/mojolearn
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SPEED_EXPECTED_VENDOR=cuda PYTHONPATH=python GBM_BENCH_DATA=/root/datasets/gbm-bench
trap 'echo "$?" > /root/forest_out/timing-batched.exit' EXIT
for i in $(seq 1 60); do
    if test -f /root/forest_out/build-default.exit; then break; fi
    sleep 2
done
test "$(cat /root/forest_out/build-default.exit)" = 0
python3 checks/forest_inference_public.py --borrowed-buffers > /root/forest_out/public-default-borrowed-identical.run.log 2>&1
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,clocks.sm,power.draw,temperature.gpu --format=csv -l 1 > /root/forest_out/batched-telemetry.csv &
telemetry_pid=$!
trap 'code=$?; kill "$telemetry_pid" 2>/dev/null || true; echo "$code" > /root/forest_out/timing-batched.exit' EXIT
python3 bench/speed/forest_inference_ab.py --lane rf --dataset higgs --rows 1000000 --rounds 8 --calls-per-sample 8 --borrowed-buffers --staged-rounds 2 --cuml-context --output /root/forest_out/rf-higgs-batched.json > /root/forest_out/rf-higgs-batched.log 2>&1
python3 bench/speed/forest_inference_ab.py --lane et --dataset higgs --rows 1000000 --rounds 8 --calls-per-sample 8 --borrowed-buffers --staged-rounds 2 --output /root/forest_out/et-higgs-batched.json > /root/forest_out/et-higgs-batched.log 2>&1
python3 bench/speed/forest_inference_ab.py --lane et --dataset year --rows 500000 --rounds 8 --calls-per-sample 8 --borrowed-buffers --staged-rounds 2 --output /root/forest_out/et-year-batched.json > /root/forest_out/et-year-batched.log 2>&1
python3 bench/speed/forest_inference_ab.py --lane et --dataset covtype --rows 1000000 --rounds 8 --calls-per-sample 8 --borrowed-buffers --staged-rounds 2 --output /root/forest_out/et-covtype-batched.json > /root/forest_out/et-covtype-batched.log 2>&1
