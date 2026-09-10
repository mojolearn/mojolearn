#!/bin/bash
set -euo pipefail
cd /root/mojolearn
export PATH=/root/.pixi/bin:$PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1
export MOJOLEARN_GPU_ARCHS=sm_90 PYTHONPATH=python CUDA_VISIBLE_DEVICES=0
out=/root/trees_out/tuning
mkdir -p "$out"
while [ ! -f /root/trees_out/setup.done ]; do sleep 10; done
for name in pixi_install build_base build_gbdt build_rf build_trees pip_base pip_cuml download_higgs import_identical; do
  grep -q "^$name=0 " /root/trees_out/setup.txt || { echo "FAILED setup $name"; exit 1; }
done
mkdir -p build/rf-higgs-columns/{reference,columns2,columns4} build/et-shared/{baseline,candidate}
cp python/mojolearn/identical/_mojolearn_rf.so build/rf-higgs-columns/reference/
cp python/mojolearn/identical/_mojolearn_trees.so build/et-shared/baseline/
for arm in columns2 columns4; do
  if [ "$arm" = columns2 ]; then
    export MOJOLEARN_EXTRA_DEFINES='-D MOJOLEARN_RF_HIST_COLUMNS2=1'
  else
    export MOJOLEARN_EXTRA_DEFINES='-D MOJOLEARN_RF_HIST_COLUMNS4=1'
  fi
  timeout -k 20 600 tools/with_build_lock.sh bash bindings/build_rf.sh > "$out/build-rf-$arm.log" 2>&1
  cp python/mojolearn/identical/_mojolearn_rf.so "build/rf-higgs-columns/$arm/"
done
cp build/rf-higgs-columns/reference/_mojolearn_rf.so python/mojolearn/identical/
export MOJOLEARN_EXTRA_DEFINES='-D MOJOLEARN_ET_SHARED_CLASS_COUNTS=1'
timeout -k 20 600 tools/with_build_lock.sh bash bindings/build_trees.sh > "$out/build-et-shared.log" 2>&1
cp python/mojolearn/identical/_mojolearn_trees.so build/et-shared/candidate/
cp build/et-shared/baseline/_mojolearn_trees.so python/mojolearn/identical/
unset MOJOLEARN_EXTRA_DEFINES RF_LAUNCH_LOG MOJOLEARN_SPEED_FORTRAN
python3 -c 'from sklearn.datasets import fetch_covtype; d=fetch_covtype(); print(d.data.shape)' > "$out/covtype-load.log" 2>&1
nvidia-smi -q > "$out/gpu-before.txt"
lscpu > "$out/lscpu.txt"
sha256sum build/rf-higgs-columns/*/*.so build/et-shared/*/*.so > "$out/binary-sha256.txt"
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu --format=csv -l 1 > "$out/gpu-telemetry.csv" &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true' EXIT
set +e
timeout -k 20 1500 tools/with_build_lock.sh python3 -u bench/speed/rf_higgs_columns_ab.py --bindings build/rf-higgs-columns --rows 1000000 --rounds 6 --output "$out/rf-higgs-1m" > "$out/rf-higgs-1m.log" 2>&1
echo "$?" > "$out/rf-higgs-1m.exit"
timeout -k 20 180 tools/with_build_lock.sh python3 -u extratrees/bench/shared_score_ab.py "$out/et-smoke" --dataset covtype --rows 10000 --baseline-binding build/et-shared/baseline/_mojolearn_trees.so --candidate-binding build/et-shared/candidate/_mojolearn_trees.so --mode identical --vendor cuda --trees 3 --depth 4 --warmups 1 --repeats 1 > "$out/et-smoke.log" 2>&1
et_smoke_rc=$?
echo "$et_smoke_rc" > "$out/et-smoke.exit"
if [ "$et_smoke_rc" -eq 0 ]; then
timeout -k 20 1500 tools/with_build_lock.sh python3 -u extratrees/bench/shared_score_ab.py "$out/et-covtype" --dataset covtype --rows 581012 --baseline-binding build/et-shared/baseline/_mojolearn_trees.so --candidate-binding build/et-shared/candidate/_mojolearn_trees.so --mode identical --vendor cuda --trees 100 --depth 16 --warmups 1 --repeats 3 > "$out/et-covtype.log" 2>&1
echo "$?" > "$out/et-covtype.exit"
fi
nvidia-smi -q > "$out/gpu-after.txt"
date -u > "$out/completed.txt"
