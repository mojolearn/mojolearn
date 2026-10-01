#!/bin/bash
# PR #20 (Louvain sparse walk) on one GPU box: the sparse check, graph_check, both bindings, and the board's
# louvain cell (taxi, istella) on the released 0.8.32 (before) vs 0.8.32 + this branch's Python and x_neighbors (after).
set -uo pipefail
cd "$(dirname "$0")/.."
V=$1; O=/root/louvain-$V; rm -rf $O; mkdir -p $O
if [ $V = nvidia ]; then BK=cuda; AR=sm_89; else BK=hip; AR=gfx942; fi
export PATH=/root/.pixi/bin:/opt/rocm/bin:$PATH MOJOLEARN_NUMERIC_MODE=identical PYTHONUNBUFFERED=1 MOJOLEARN_COMPILE_JOBS=4; unset PYTHONPATH
rc() { echo "$(date -u +%T) $1 rc=$2" >> $O/rc.txt; }
git rev-parse HEAD > $O/head.txt; pixi install > $O/pixi.log 2>&1
timeout 3600 pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . x_neighbors/checks/louvain_sparse_check.mojo > $O/sparse-check.log 2>&1; rc sparse-check $?
MOJOLEARN_TARGET_COLUMN=$V MOJOLEARN_GPU_ARCHS=$AR timeout 3600 pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . x_neighbors/checks/graph_check.mojo > $O/graph-check.log 2>&1; rc graph-check $?
MOJOLEARN_TARGET_COLUMN=$V MOJOLEARN_GPU_ARCHS=$AR bash bindings/build_x_neighbors.sh > $O/build-x_neighbors.log 2>&1; rc build-x_neighbors $?
MOJOLEARN_TARGET_COLUMN=cpu bash bindings/build_x_neighbors_host.sh > $O/build-x_neighbors_host.log 2>&1; rc build-x_neighbors_host $?
base=$(pixi run python3 -c 'import sys;print(sys.executable)' | tail -1)
for arm in before after; do
  $base -m venv $O/venv-$arm; P=$O/venv-$arm/bin/python
  $P -m pip -q install mojolearn==0.8.32 numpy==2.5.2 scipy==1.18.0 networkx > $O/pip-$arm.log 2>&1
  if [ $arm = after ]; then S=$($P -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')/mojolearn
    cp $S/_version.py /tmp/v.py; cp python/mojolearn/*.py $S/; cp /tmp/v.py $S/_version.py; find $S -name __pycache__ -exec rm -rf {} +
    so=$(ls -t python/mojolearn/identical/_mojolearn_x_neighbors.so python/mojolearn/_mojolearn_x_neighbors.so 2>/dev/null | head -1); cp $so $S/$BK/$AR/identical/; sha256sum $so >> $O/bindings.sha256; fi
done
P=$O/venv-before/bin/python
timeout 3600 $P tools/bench_board_algos.py prep --data $O/data --lanes louvain --datasets taxi,istella > $O/prep.log 2>&1; rc prep $?
for arm in before after; do for ds in taxi istella; do
  MOJOLEARN_TARGET_COLUMN=$V MOJOLEARN_BENCH_INSTALLED=1 timeout 3600 $O/venv-$arm/bin/python tools/bench_board_algos.py race --lane louvain --dataset $ds --data $O/data --arms ours --rounds 3 --out $O/race-$arm-$ds --work $O/work --ours-python $O/venv-$arm/bin/python > $O/race-$arm-$ds.log 2>&1; rc race-$arm-$ds $?
  echo "$arm $ds $(grep -oE "ALGOS-ROUND.*ms=[0-9.]+ .*digest=[0-9a-f]+" $O/race-$arm-$ds.log | tail -1 | grep -oE "ms=[0-9.]+|digest=[0-9a-f]+" | tr "\n" " ")" >> $O/races.txt; done; done
echo done > $O/done
