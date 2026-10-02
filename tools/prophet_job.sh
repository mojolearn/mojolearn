#!/bin/bash
# PR #30 (prophet series as host tasks inside the GPU binding) on one GPU box: the board's garch cell
# (synthetic, taxi-hourly) on released 0.8.32 (device) vs this branch (host tasks) vs this branch with
# MOJOLEARN_SEQ_PROPHET_HOST_MAX=0 (device): digests must agree, walls compared.
set -uo pipefail
cd "$(dirname "$0")/.."
V=$1; O=/root/prophet-$V; rm -rf $O; mkdir -p $O
if [ $V = nvidia ]; then BK=cuda; AR=sm_89; else BK=hip; AR=gfx942; fi
export PATH=/root/.pixi/bin:/opt/rocm/bin:$PATH MOJOLEARN_NUMERIC_MODE=identical PYTHONUNBUFFERED=1 MOJOLEARN_COMPILE_JOBS=4; unset PYTHONPATH
rc() { echo "$(date -u +%T) $1 rc=$2" >> $O/rc.txt; }
git rev-parse HEAD > $O/head.txt; pixi install > $O/pixi.log 2>&1
MOJOLEARN_TARGET_COLUMN=$V MOJOLEARN_GPU_ARCHS=$AR bash bindings/build_x_sequence.sh > $O/build-x_sequence.log 2>&1; rc build-x_sequence $?
base=$(pixi run python3 -c 'import sys;print(sys.executable)' | tail -1)
for arm in before after; do
  $base -m venv $O/venv-$arm; P=$O/venv-$arm/bin/python
  $P -m pip -q install mojolearn==0.8.32 numpy==2.5.2 scipy==1.18.0 prophet > $O/pip-$arm.log 2>&1
  if [ $arm = after ]; then S=$($P -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')/mojolearn
    cp $S/_version.py /tmp/v.py; cp python/mojolearn/*.py $S/; cp /tmp/v.py $S/_version.py; find $S -name __pycache__ -exec rm -rf {} +
    so=$(ls -t python/mojolearn/identical/_mojolearn_x_sequence.so python/mojolearn/_mojolearn_x_sequence.so 2>/dev/null | head -1); cp $so $S/$BK/$AR/identical/; sha256sum $so >> $O/bindings.sha256; fi
done
timeout 3600 $O/venv-before/bin/python tools/bench_board_algos.py prep --data $O/data --lanes prophet --datasets synthetic,taxi-hourly > $O/prep.log 2>&1; rc prep $?
for run in before after after-device; do for ds in synthetic taxi-hourly; do
  arm=${run%%-*}; extra=""; [ $run = after-device ] && extra="MOJOLEARN_SEQ_PROPHET_HOST_MAX=0"
  env $extra MOJOLEARN_TARGET_COLUMN=$V MOJOLEARN_BENCH_INSTALLED=1 timeout 3600 $O/venv-$arm/bin/python tools/bench_board_algos.py race --lane prophet --dataset $ds --data $O/data --arms ours --rounds 3 --out $O/race-$run-$ds --work $O/work --ours-python $O/venv-$arm/bin/python > $O/race-$run-$ds.log 2>&1; rc race-$run-$ds $?
  echo "$run $ds $(grep -oE "ALGOS-ROUND.*" $O/race-$run-$ds.log | tail -1 | grep -oE "ms=[0-9.]+|digest=[0-9a-f]+" | tr "\n" " ")" >> $O/races.txt; done; done
echo done > $O/done
