#!/bin/bash
# PR #15 (Mamba-3 host oracle over host tasks, host GEMM packs uninitialized, keep-alive fix) on one host CPU.
set -uo pipefail
cd "$(dirname "$0")/.."
O=${PR18_OUT:-$HOME/pr18}; rm -rf $O; mkdir -p $O
export PATH=$HOME/.pixi/bin:/root/.pixi/bin:$PATH MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=cpu MOJOLEARN_VENDOR=cpu PYTHONUNBUFFERED=1
unset PYTHONPATH
rc() { echo "$(date -u +%T) $1 rc=$2" >> $O/rc.txt; }
(grep -m1 "model name" /proc/cpuinfo 2>/dev/null || sysctl -n machdep.cpu.brand_string) > $O/cpu.txt
git rev-parse HEAD > $O/head.txt
pixi install > $O/pixi.log 2>&1
for b in neural_host mamba_host byte_lm_host; do rm -f python/mojolearn/host/_mojolearn_$b.so python/mojolearn/_mojolearn_$b.so; bash bindings/build_$b.sh > $O/build-$b.log 2>&1; rc build-$b $?; done
timeout 3600 pixi run check-gemm-host-rows > $O/check-gemm-host-rows.log 2>&1; rc check-gemm-host-rows $?
timeout 3600 pixi run python tools/host_threads_ab_check.py --model mamba3 --calls 3 --timing > $O/ab-mamba3.log 2>&1; rc ab-mamba3 $?
timeout 3600 pixi run python tools/host_threads_ab_check.py --model transformer --calls 2 > $O/ab-transformer.log 2>&1; rc ab-transformer $?
timeout 3600 pixi run python tools/byte_lm_cpu_train_gate.py cpu --steps all > $O/train-gate.log 2>&1; rc train-gate $?

MOJOLEARN_CPU_THREADS=1 timeout 3600 pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_CPU -I . -I bindings tools/gemm_host_rows_bench.mojo > $O/gemm-bench-1thread.log 2>&1; rc gemm-bench-1 $?
timeout 3600 pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_CPU -I . -I bindings tools/gemm_host_rows_bench.mojo > $O/gemm-bench-policy.log 2>&1; rc gemm-bench-policy $?
PYP=/root/torchvenv/bin/python
for lane in mamba3-infer samba-infer transformer-infer lm-host-train-step; do
  MOJOLEARN_BENCH_INSTALLED=0 PYTHONPATH=$PWD/python timeout 3600 pixi run python tools/bench_board_neural.py race --lane $lane --shape full --arms ours --rounds 3 --out $O/cell-$lane --work $O/work --ours-python $PYP > $O/cell-$lane.log 2>&1; rc cell-$lane $?; done
echo done > $O/done
