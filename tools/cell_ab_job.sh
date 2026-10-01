#!/bin/bash
# Generic GPU-box before/after: released 0.8.32 (before) vs 0.8.32 + this branch's Python and the named bindings
# (after), the named board cells (synthetic), plus optional pixi check tasks and an env-restore run.
# Usage: tools/cell_ab_job.sh <nvidia|amd> <tag> "<modules>" "<lanes>" "<pixi checks>" "<restore env or ->"
set -uo pipefail
cd "$(dirname "$0")/.."
V=$1; TAG=$2; MODS=$3; LANES=$4; CHECKS=$5; RESTORE=$6; O=/root/$TAG-$V; rm -rf $O; mkdir -p $O
if [ $V = nvidia ]; then BK=cuda; AR=sm_89; else BK=hip; AR=gfx942; fi
export PATH=/root/.pixi/bin:/opt/rocm/bin:$PATH MOJOLEARN_NUMERIC_MODE=identical PYTHONUNBUFFERED=1 MOJOLEARN_COMPILE_JOBS=4; unset PYTHONPATH
rc() { echo "$(date -u +%T) $1 rc=$2" >> $O/rc.txt; }
git rev-parse HEAD > $O/head.txt; pixi install > $O/pixi.log 2>&1
for m in $MODS; do MOJOLEARN_TARGET_COLUMN=$V MOJOLEARN_GPU_ARCHS=$AR bash bindings/build_$m.sh > $O/build-$m.log 2>&1; rc build-$m $?; done
for c in $CHECKS; do MOJOLEARN_TARGET_COLUMN=$V MOJOLEARN_GPU_ARCHS=$AR timeout 3600 pixi run $c > $O/check-$c.log 2>&1; rc check-$c $?; done
base=$(pixi run python3 -c 'import sys;print(sys.executable)' | tail -1)
for arm in before after; do $base -m venv $O/venv-$arm; P=$O/venv-$arm/bin/python
  $P -m pip -q install mojolearn==0.8.32 numpy==2.5.2 scipy==1.18.0 > $O/pip-$arm.log 2>&1
  if [ $arm = after ]; then S=$($P -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')/mojolearn
    cp $S/_version.py /tmp/v.py; cp python/mojolearn/*.py $S/; cp /tmp/v.py $S/_version.py; find $S -name __pycache__ -exec rm -rf {} +
    for m in $MODS; do so=$(ls -t python/mojolearn/identical/_mojolearn_$m.so python/mojolearn/_mojolearn_$m.so 2>/dev/null | head -1); [ -n "$so" ] && cp $so $S/$BK/$AR/identical/ && sha256sum $so >> $O/bindings.sha256; done; fi; done
timeout 3600 $O/venv-before/bin/python tools/bench_board_algos.py prep --data $O/data --lanes $(echo $LANES | tr " " ",") --datasets synthetic > $O/prep.log 2>&1; rc prep $?
for lane in $LANES; do for run in before after restore; do arm=${run}; extra=""; [ $run = restore ] && { [ "$RESTORE" = "-" ] && continue; arm=after; extra=$RESTORE; }
  env $extra MOJOLEARN_TARGET_COLUMN=$V MOJOLEARN_BENCH_INSTALLED=1 timeout 3600 $O/venv-$arm/bin/python tools/bench_board_algos.py race --lane $lane --dataset synthetic --data $O/data --arms ours --rounds 3 --out $O/race-$lane-$run --work $O/work --ours-python $O/venv-$arm/bin/python > $O/race-$lane-$run.log 2>&1; rc race-$lane-$run $?
  echo "$lane $run $(grep -oE "ALGOS-ROUND.*" $O/race-$lane-$run.log | tail -1 | grep -oE " ms=[0-9.]+|digest=[0-9a-f]+" | tr "\n" " ") $(grep -o "ALGOS-REFUSED.\{0,140\}" $O/race-$lane-$run.log | head -1)" >> $O/races.txt; done; done
echo done > $O/done
