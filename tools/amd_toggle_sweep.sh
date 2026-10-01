#!/bin/bash
# The Sep 30 neural toggle sweep on the MI325X (peer request, 2026-10-01): released 0.8.32 + main's Python and neural
# bindings; --set nvidia over every lane, the GEMM plan arms on transformer-forward, then an attention-phase-timer
# byte_lm build for lm-train-step's backward kernels. Waits for the PR #52 job to finish first.
set -uo pipefail
cd "$(dirname "$0")/.."
while [ ! -f /root/pr52-amd/done ]; do sleep 60; done
O=/root/amd-toggle-sweep; rm -rf $O; mkdir -p $O
export PATH=/root/.pixi/bin:/opt/rocm/bin:$PATH MOJOLEARN_NUMERIC_MODE=identical PYTHONUNBUFFERED=1 MOJOLEARN_COMPILE_JOBS=4 MOJOLEARN_TARGET_COLUMN=amd MOJOLEARN_GPU_ARCHS=gfx942 MOJOLEARN_BENCH_INSTALLED=1; unset PYTHONPATH
rc() { echo "$(date -u +%T) $1 rc=$2" >> $O/rc.txt; }
git rev-parse HEAD > $O/head.txt; pixi install > $O/pixi.log 2>&1
MODS="transformer byte_lm training mamba"
for m in $MODS; do rm -f python/mojolearn/identical/_mojolearn_$m.so python/mojolearn/_mojolearn_$m.so; bash bindings/build_$m.sh > $O/build-$m.log 2>&1; rc build-$m $?; done
base=$(pixi run python3 -c 'import sys;print(sys.executable)' | tail -1); $base -m venv $O/venv; P=$O/venv/bin/python
$P -m pip -q install mojolearn==0.8.32 numpy==2.5.2 scipy==1.18.0 > $O/pip.log 2>&1; $P -m pip -q install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu >> $O/pip.log 2>&1
S=$($P -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')/mojolearn; T=$S/hip/gfx942/identical
cp $S/_version.py /tmp/vts.py; cp python/mojolearn/*.py $S/; cp /tmp/vts.py $S/_version.py; find $S -name __pycache__ -exec rm -rf {} +
for m in $MODS; do so=$(ls -t python/mojolearn/identical/_mojolearn_$m.so python/mojolearn/_mojolearn_$m.so 2>/dev/null | head -1); [ -n "$so" ] && cp $so $T/ && cp $so $O/$m.so && sha256sum $so >> $O/bindings.sha256; done
timeout 7200 $P tools/neural_experiments.py --set nvidia --calls 10 --json $O/set-nvidia.json > $O/set-nvidia.log 2>&1; rc set-nvidia $?
timeout 3600 $P tools/neural_experiments.py --only baseline --gemm-arms tuned128,half,quarter,kpack,kfoldv --lane transformer-forward --lane lm-forward --calls 10 --json $O/gemm-arms.json > $O/gemm-arms.log 2>&1; rc gemm-arms $?
rm -f python/mojolearn/identical/_mojolearn_byte_lm.so python/mojolearn/_mojolearn_byte_lm.so
MOJOLEARN_MOJO_BUILD_FLAGS="-D MOJOLEARN_ATTN_PHASE_TIMERS=1" bash bindings/build_byte_lm.sh > $O/build-byte_lm-attntimers.log 2>&1; rc build-attntimers $?
so=$(ls -t python/mojolearn/identical/_mojolearn_byte_lm.so python/mojolearn/_mojolearn_byte_lm.so 2>/dev/null | head -1); [ -n "$so" ] && cp $so $T/_mojolearn_byte_lm.so
timeout 3600 $P tools/neural_stage_timing.py --lane lm-train-step --lane lm-forward --calls 4 > $O/attn-phase.log 2>&1; rc attn-phase $?
cp $O/byte_lm.so $T/_mojolearn_byte_lm.so
echo done > $O/done
