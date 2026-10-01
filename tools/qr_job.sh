#!/bin/bash
# PR #24 (bit-inert geqrf/orgqr kernels): the qr and svd cells (taxi, istella) on released 0.8.32 vs this branch.
# qr and svd cells (taxi, istella) on the released 0.8.32 (before) vs 0.8.32 + this branch (after).
set -uo pipefail
cd "$(dirname "$0")/.."
V=$1; O=/root/qr-$V; rm -rf $O; mkdir -p $O
if [ $V = nvidia ]; then BK=cuda; AR=sm_89; else BK=hip; AR=gfx942; fi
export PATH=/root/.pixi/bin:/opt/rocm/bin:$PATH MOJOLEARN_NUMERIC_MODE=identical PYTHONUNBUFFERED=1 MOJOLEARN_COMPILE_JOBS=4; unset PYTHONPATH
rc() { echo "$(date -u +%T) $1 rc=$2" >> $O/rc.txt; }
git rev-parse HEAD > $O/head.txt; pixi install > $O/pixi.log 2>&1
for m in x_decomp linalg; do MOJOLEARN_TARGET_COLUMN=$V MOJOLEARN_GPU_ARCHS=$AR bash bindings/build_$m.sh > $O/build-$m.log 2>&1; rc build-$m $?; done
MOJOLEARN_TARGET_COLUMN=cpu bash bindings/build_x_decomp_host.sh > $O/build-x_decomp_host.log 2>&1; rc build-x_decomp_host $?
base=$(pixi run python3 -c 'import sys;print(sys.executable)' | tail -1)
for arm in before after; do
  $base -m venv $O/venv-$arm; P=$O/venv-$arm/bin/python
  $P -m pip -q install mojolearn==0.8.32 numpy==2.5.2 scipy==1.18.0 networkx > $O/pip-$arm.log 2>&1
  if [ $arm = after ]; then S=$($P -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')/mojolearn
    cp $S/_version.py /tmp/v.py; cp python/mojolearn/*.py $S/; cp /tmp/v.py $S/_version.py; find $S -name __pycache__ -exec rm -rf {} +
    for m in x_decomp linalg; do so=$(ls -t python/mojolearn/identical/_mojolearn_$m.so python/mojolearn/_mojolearn_$m.so 2>/dev/null | head -1); cp $so $S/$BK/$AR/identical/; sha256sum $so >> $O/bindings.sha256; done; fi
done
P=$O/venv-before/bin/python
timeout 3600 $P tools/bench_board_algos.py prep --data $O/data --lanes qr,svd --datasets taxi,istella > $O/prep.log 2>&1; rc prep $?
for lane in qr svd; do for arm in before after; do for ds in taxi istella; do
  MOJOLEARN_TARGET_COLUMN=$V MOJOLEARN_BENCH_INSTALLED=1 timeout 3600 $O/venv-$arm/bin/python tools/bench_board_algos.py race --lane $lane --dataset $ds --data $O/data --arms ours --rounds 3 --out $O/race-$lane-$arm-$ds --work $O/work --ours-python $O/venv-$arm/bin/python > $O/race-$lane-$arm-$ds.log 2>&1; rc race-$lane-$arm-$ds $?
  echo "$lane $arm $ds $(grep -oE "ALGOS-ROUND.*ms=[0-9.]+ .*digest=[0-9a-f]+" $O/race-$lane-$arm-$ds.log | tail -1 | grep -oE "ms=[0-9.]+|digest=[0-9a-f]+" | tr "\n" " ")" >> $O/races.txt; done; done; done
echo done > $O/done
