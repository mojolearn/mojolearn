#!/bin/bash
# Host-CPU PR measurement (PR #10 token split, PR #11 host train step): the PR's own checks in its order,
# then the board cell with the change on and off. Usage: tools/hostpr_job.sh <pr10|pr11> <vendor nvidia|amd>
set -uo pipefail
cd "$(dirname "$0")/.."
PR=$1; V=${2:-amd}; O=/root/hostpr-$PR; rm -rf $O/rc.txt; mkdir -p $O
if [ $V = nvidia ]; then BK=cuda; AR=sm_89; else BK=hip; AR=gfx942; fi
export PATH=/root/.pixi/bin:/opt/rocm/bin:$PATH MOJOLEARN_TARGET_COLUMN=$V MOJOLEARN_GPU_ARCHS=$AR \
  MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_COMPILE_JOBS=2 MOJOLEARN_BENCH_INSTALLED=1 PYTHONUNBUFFERED=1
unset PYTHONPATH
st() { echo "{\"phase\":\"$1\",\"utc\":\"$(date -u +%FT%TZ)\"}" > $O/status.json; }
rc() { echo "$(date -u +%T) $1 rc=$2" >> $O/rc.txt; }
git rev-parse HEAD > $O/head.txt; nproc > $O/nproc.txt; grep -m1 "model name" /proc/cpuinfo > $O/cpu.txt
st pixi; pixi install > $O/pixi.log 2>&1
st build-host; rm -f python/mojolearn/_mojolearn_byte_lm_host.so python/mojolearn/*/_mojolearn_byte_lm_host.so
bash bindings/build_byte_lm_host.sh > $O/build-host.log 2>&1; rc build-host $?
PYP=$(pixi run python3 -c 'import sys;print(sys.executable)' | tail -1)
if [ $PR = pr11 ]; then
  st gate; timeout 3600 pixi run python tools/byte_lm_cpu_train_gate.py cpu --steps all > $O/train-gate.log 2>&1; rc train-gate $?
  st step-check-board; timeout 3600 pixi run python tools/byte_lm_host_step_check.py --steps 4 > $O/step-check-board.log 2>&1; rc step-check-board $?
  st step-check-small; timeout 3600 pixi run python tools/byte_lm_host_step_check.py --small --steps 16 > $O/step-check-small.log 2>&1; rc step-check-small $?
  st profile; timeout 3600 pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_CPU -I . -I bindings tools/byte_lm_host_step_profile.mojo > $O/profile.log 2>&1; rc profile $?
  for r in 1 0; do st cell-rows$r; MOJOLEARN_BYTE_LM_HOST_STEP_ROWS=$r MOJOLEARN_BENCH_INSTALLED=0 PYTHONPATH=$PWD/python timeout 3600 pixi run python tools/bench_board_neural.py race --lane lm-host-train-step --shape full --arms ours --rounds 3 --out $O/cell-rows$r --work $O/work --ours-python $PYP > $O/cell-rows$r.log 2>&1; rc cell-rows$r $?; done
else
  st gate; timeout 3600 pixi run python tools/byte_lm_host_gate.py > $O/host-gate.log 2>&1; rc host-gate $?
  st sweep; timeout 3600 pixi run python tools/byte_lm_host_path_sweep.py > $O/path-sweep.log 2>&1; rc path-sweep $?
  for r in 1 0; do st cell-split$r; MOJOLEARN_BYTE_LM_HOST_TOKEN_SPLIT=$r MOJOLEARN_BENCH_INSTALLED=0 PYTHONPATH=$PWD/python timeout 3600 pixi run python tools/bench_board_neural.py race --lane lm-infer --shape full --arms ours --rounds 3 --out $O/cell-split$r --work $O/work --ours-python $PYP > $O/cell-split$r.log 2>&1; rc cell-split$r $?; done
fi
st done; echo done > $O/done
