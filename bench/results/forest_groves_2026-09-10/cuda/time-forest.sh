#!/bin/bash
set -euo pipefail
cd /root/mojolearn
export PATH=/root/.pixi/bin:$PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SPEED_EXPECTED_VENDOR=cuda PYTHONPATH=python GBM_BENCH_DATA=/root/datasets/gbm-bench
trap 'echo "$?" > /root/forest_out/timing-all.exit' EXIT
pixi run mojo build --target-accelerator sm_90 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_FOREST_VECTOR_GROVES=1 checks/forest_inference_model.mojo -o /root/forest_out/resident-into-check-final > /root/forest_out/resident-into-check-final.build.log 2>&1
/root/forest_out/resident-into-check-final > /root/forest_out/resident-into-check-final.run.log 2>&1
python3 checks/forest_inference_public.py --borrowed-buffers > /root/forest_out/public-borrowed-identical.run.log 2>&1
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,clocks.sm,power.draw,temperature.gpu --format=csv -l 1 > /root/forest_out/timing-telemetry.csv &
telemetry_pid=$!
trap 'code=$?; kill "$telemetry_pid" 2>/dev/null || true; echo "$code" > /root/forest_out/timing-all.exit' EXIT
/root/forest_out/grove-bench 1000000 100 16 2 6 > /root/forest_out/kernel-1m-depth16-output2.log 2>&1
/root/forest_out/grove-bench 1000000 100 16 7 6 > /root/forest_out/kernel-1m-depth16-output7.log 2>&1
python3 bench/speed/forest_inference_ab.py --lane rf --dataset higgs --rows 1000000 --rounds 8 --borrowed-buffers --staged-rounds 2 --cuml-context --output /root/forest_out/rf-higgs.json > /root/forest_out/rf-higgs.log 2>&1
python3 bench/speed/forest_inference_ab.py --lane et --dataset higgs --rows 1000000 --rounds 6 --borrowed-buffers --staged-rounds 2 --output /root/forest_out/et-higgs.json > /root/forest_out/et-higgs.log 2>&1
python3 bench/speed/forest_inference_ab.py --lane et --dataset year --rows 500000 --rounds 6 --borrowed-buffers --staged-rounds 2 --output /root/forest_out/et-year.json > /root/forest_out/et-year.log 2>&1
python3 bench/speed/forest_inference_ab.py --lane et --dataset covtype --rows 1000000 --rounds 6 --borrowed-buffers --staged-rounds 2 --output /root/forest_out/et-covtype.json > /root/forest_out/et-covtype.log 2>&1
