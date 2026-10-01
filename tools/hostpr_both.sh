#!/bin/bash
# Both host PRs' gates on this box's CPU (main through PR #10/#11): train gate, step check, inference gate and sweep.
set -uo pipefail
cd "$(dirname "$0")/.."
O=/root/hostpr-both; rm -rf $O; mkdir -p $O
export PATH=/root/.pixi/bin:$PATH MOJOLEARN_NUMERIC_MODE=identical PYTHONUNBUFFERED=1; unset PYTHONPATH
rc() { echo "$(date -u +%T) $1 rc=$2" >> $O/rc.txt; }
grep -m1 "model name" /proc/cpuinfo > $O/cpu.txt; nproc > $O/nproc.txt; git rev-parse HEAD > $O/head.txt
pixi install > $O/pixi.log 2>&1
MOJOLEARN_TARGET_COLUMN=cpu bash bindings/build_byte_lm_host.sh > $O/build-host.log 2>&1; rc build-host $?
MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_GPU_ARCHS=sm_89 MOJOLEARN_COMPILE_JOBS=4 bash bindings/build.sh > $O/build-base.log 2>&1; rc build-base $?
MOJOLEARN_TARGET_COLUMN=cpu MOJOLEARN_VENDOR=cpu timeout 3600 pixi run python tools/byte_lm_cpu_train_gate.py cpu --steps all > $O/train-gate.log 2>&1; rc train-gate $?
MOJOLEARN_TARGET_COLUMN=cpu MOJOLEARN_VENDOR=cpu timeout 3600 pixi run python tools/byte_lm_host_step_check.py --steps 4 > $O/step-check.log 2>&1; rc step-check $?
for r in 1 0; do MOJOLEARN_BYTE_LM_HOST_TOKEN_SPLIT=$r timeout 3600 pixi run python tools/byte_lm_host_gate.py > $O/infer-gate-$r.log 2>&1; rc infer-gate-$r $?; done
timeout 3600 pixi run python tools/byte_lm_host_path_sweep.py > $O/sweep.log 2>&1; rc sweep $?
echo done > $O/done
