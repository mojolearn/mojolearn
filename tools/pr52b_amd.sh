#!/bin/bash
# PR #52 on the MI325X: MOJOLEARN_GEMM_SPLIT_CELLS sweep (unset = 128K, 512K, 1M, 2M) on lm-forward and lm-train-step
# (released 0.8.32 + this branch's Python and neural bindings), then the gemm device check at 2M.
set -uo pipefail
while [ ! -f /root/amd-toggle-sweep/done ]; do sleep 60; done
cd "$(dirname "$0")/.."
O=/root/pr52b-amd; rm -rf $O; mkdir -p $O
export PATH=/root/.pixi/bin:/opt/rocm/bin:$PATH MOJOLEARN_NUMERIC_MODE=identical PYTHONUNBUFFERED=1 MOJOLEARN_COMPILE_JOBS=4 MOJOLEARN_TARGET_COLUMN=amd MOJOLEARN_GPU_ARCHS=gfx942 MOJOLEARN_BENCH_INSTALLED=1; unset PYTHONPATH
rc() { echo "$(date -u +%T) $1 rc=$2" >> $O/rc.txt; }
git rev-parse HEAD > $O/head.txt; pixi install > $O/pixi.log 2>&1
MODS="transformer byte_lm training mamba"
for m in $MODS; do rm -f python/mojolearn/identical/_mojolearn_$m.so python/mojolearn/_mojolearn_$m.so; bash bindings/build_$m.sh > $O/build-$m.log 2>&1; rc build-$m $?; done
base=$(pixi run python3 -c 'import sys;print(sys.executable)' | tail -1); $base -m venv $O/venv; P=$O/venv/bin/python
$P -m pip -q install mojolearn==0.8.32 numpy==2.5.2 scipy==1.18.0 > $O/pip.log 2>&1; $P -m pip -q install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu >> $O/pip.log 2>&1
S=$($P -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')/mojolearn
cp $S/_version.py /tmp/v52.py; cp python/mojolearn/*.py $S/; cp /tmp/v52.py $S/_version.py; find $S -name __pycache__ -exec rm -rf {} +
for m in $MODS; do so=$(ls -t python/mojolearn/identical/_mojolearn_$m.so python/mojolearn/_mojolearn_$m.so 2>/dev/null | head -1); [ -n "$so" ] && cp $so $S/hip/gfx942/identical/ && sha256sum $so >> $O/bindings.sha256; done
for cap in unset unset2 524288; do ex=""; [ $cap = 524288 ] && ex="MOJOLEARN_GEMM_SPLIT_CELLS=$cap"
  env $ex timeout 3600 $P tools/neural_stage_timing.py --lane lm-forward --lane lm-train-step --calls 6 > $O/stage-$cap.log 2>&1; rc stage-$cap $?; done
timeout 3600 pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_device_check.mojo > $O/gemm-check-default.log 2>&1; rc gemm-check-default $?
echo done > $O/done
