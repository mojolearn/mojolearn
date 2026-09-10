#!/bin/bash
set -euo pipefail
mkdir -p /root/forest_out
exec > /root/forest_out/campaign.log 2>&1
trap 'echo "$?" > /root/forest_out/campaign.exit' EXIT
cd /root/mojolearn
export PATH=/root/.pixi/bin:$PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_GPU_ARCHS=sm_90 MOJOLEARN_SKIP_BUILD_GATE=1
export MOJOLEARN_SPEED_EXPECTED_VENDOR=cuda PYTHONPATH=python GBM_BENCH_DATA=/root/datasets/gbm-bench
bash /root/data.sh
python3 /root/verify_higgs.py
pixi run mojo build --target-accelerator sm_90 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 checks/forest_inference_model.mojo -o /root/forest_out/resident-check > /root/forest_out/resident-check.build.log 2>&1
bash bindings/build_rf.sh > /root/forest_out/rf-identical.build.log 2>&1
bash bindings/build_trees.sh > /root/forest_out/et-identical.build.log 2>&1
/root/forest_out/resident-check > /root/forest_out/resident-check.run.log 2>&1
python3 checks/forest_inference_public.py --mode identical --vendor cuda > /root/forest_out/public-default.log 2>&1
python3 checks/forest_inference_public.py --mode identical --vendor cuda --reuse-io > /root/forest_out/public-reuse.log 2>&1
sha256sum python/mojolearn/identical/_mojolearn_{rf,trees}.so > /root/forest_out/binary-sha256.txt
python3 -m pip freeze > /root/forest_out/pip-freeze.txt
nvidia-smi -q > /root/forest_out/gpu-full.txt
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,clocks.sm,power.draw,temperature.gpu --format=csv -l 1 > /root/forest_out/telemetry.csv &
telemetry_pid=$!
trap 'code=$?; kill "$telemetry_pid" 2>/dev/null || true; echo "$code" > /root/forest_out/campaign.exit' EXIT
# Predeclared single-call grid and one throughput companion, no timing retries.
for calls in 1 8; do
  for cell in rf-higgs et-higgs et-year et-covtype; do
    lane=${cell%%-*}; dataset=${cell#*-}; rows=1000000
    extra=()
    if [ "$dataset" = year ]; then rows=500000; fi
    if [ "$lane" = rf ]; then extra+=(--cuml-context); fi
    python3 bench/speed/forest_inference_ab.py --lane "$lane" --dataset "$dataset" --rows "$rows" --trees 100 --depth 16 --rounds 8 --calls-per-sample "$calls" --reuse-io --staged-rounds 2 "${extra[@]}" --output "/root/forest_out/$cell-calls$calls.json" > "/root/forest_out/$cell-calls$calls.log" 2>&1
  done
done
echo CAMPAIGN_PASS
