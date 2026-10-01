#!/bin/bash
# PR #16 (masked keys skipped in the host attention) / PR #17 (the same in lm-infer's kernels): host checks,
# the transformer GPU device gates (PR #16), and the board cells on the CPU column.
# Usage: tools/hostgpu_job.sh <pr16|pr17> <nvidia|amd>
set -uo pipefail
cd "$(dirname "$0")/.."
PR=$1; V=$2; O=/root/$PR; rm -rf $O; mkdir -p $O
if [ $V = nvidia ]; then AR=sm_89; else AR=gfx942; fi
export PATH=/root/.pixi/bin:/opt/rocm/bin:$PATH MOJOLEARN_NUMERIC_MODE=identical PYTHONUNBUFFERED=1; unset PYTHONPATH
rc() { echo "$(date -u +%T) $1 rc=$2" >> $O/rc.txt; }
grep -m1 "model name" /proc/cpuinfo > $O/cpu.txt; git rev-parse HEAD > $O/head.txt
pixi install > $O/pixi.log 2>&1
TV=/root/torchvenv; [ -x $TV/bin/python ] || { $(pixi run python3 -c 'import sys;print(sys.executable)' | tail -1) -m venv --system-site-packages $TV; $TV/bin/python -m ensurepip -q >/dev/null 2>&1; $TV/bin/python -m pip -q install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu >/dev/null 2>&1; }
cpu() { MOJOLEARN_TARGET_COLUMN=cpu MOJOLEARN_VENDOR=cpu "$@"; }
cell() { cpu env MOJOLEARN_BENCH_INSTALLED=0 PYTHONPATH=$PWD/python timeout 3600 $TV/bin/python tools/bench_board_neural.py race --lane $1 --shape full --arms ours --rounds 3 --out $O/cell-$1 --work $O/work --ours-python $TV/bin/python > $O/cell-$1.log 2>&1; rc cell-$1 $?; }
if [ $PR = pr16 ]; then
  for b in neural_host mamba_host byte_lm_host; do cpu bash bindings/build_$b.sh > $O/build-$b.log 2>&1; rc build-$b $?; done
  cpu timeout 3600 pixi run python tools/host_threads_ab_check.py --model transformer --calls 2 > $O/ab-transformer.log 2>&1; rc ab-transformer $?
  cpu timeout 3600 pixi run python tools/host_threads_ab_check.py --model mamba3 --calls 2 > $O/ab-mamba3.log 2>&1; rc ab-mamba3 $?
  cpu timeout 3600 pixi run python tools/byte_lm_cpu_train_gate.py cpu --steps all > $O/train-gate.log 2>&1; rc train-gate $?
  export MOJOLEARN_TARGET_COLUMN=$V MOJOLEARN_GPU_ARCHS=$AR
  timeout 3600 pixi run check-transformer > $O/gpu-check-transformer.log 2>&1; rc gpu-check-transformer $?
  timeout 3600 pixi run check-transformer-backward > $O/gpu-check-transformer-backward.log 2>&1; rc gpu-check-transformer-backward $?
  for c in forward_readback_check backward_readback_check; do timeout 3600 pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . transformer/checks/$c.mojo > $O/gpu-$c.log 2>&1; rc gpu-$c $?; done
  unset MOJOLEARN_TARGET_COLUMN MOJOLEARN_GPU_ARCHS
  for l in transformer-infer lm-host-train-step samba-infer; do cell $l; done
else
  for b in byte_lm_host core_host; do cpu bash bindings/build_$b.sh > $O/build-$b.log 2>&1; rc build-$b $?; done
  MOJOLEARN_TARGET_COLUMN=$V MOJOLEARN_GPU_ARCHS=$AR MOJOLEARN_COMPILE_JOBS=4 bash bindings/build.sh > $O/build-base.log 2>&1; rc build-base $?
  timeout 3600 pixi run python tools/byte_lm_host_gate.py > $O/host-gate.log 2>&1; rc host-gate $?
  timeout 3600 pixi run python tools/byte_lm_host_path_sweep.py --threads 1,8,64 --max-batch 2 --states initial,final > $O/sweep.log 2>&1; rc sweep $?
  cell lm-infer
fi
echo done > $O/done
