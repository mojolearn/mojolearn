#!/bin/bash
# Per-stage timing of lm-forward, lm-train-step and transformer-forward on the MI325X (peer request, 2026-10-01):
# released 0.8.32 + main's Python and neural bindings rebuilt from main; tools/neural_stage_timing.py --calls 6.
set -uo pipefail
cd "$(dirname "$0")/.."
O=/root/stage-diag-amd; rm -rf $O; mkdir -p $O
export PATH=/root/.pixi/bin:/opt/rocm/bin:$PATH MOJOLEARN_NUMERIC_MODE=identical PYTHONUNBUFFERED=1 MOJOLEARN_COMPILE_JOBS=4 MOJOLEARN_TARGET_COLUMN=amd MOJOLEARN_GPU_ARCHS=gfx942 MOJOLEARN_BENCH_INSTALLED=1; unset PYTHONPATH
rc() { echo "$(date -u +%T) $1 rc=$2" >> $O/rc.txt; }
git rev-parse HEAD > $O/head.txt; pixi install > $O/pixi.log 2>&1
MODS="transformer byte_lm training mamba"
for m in $MODS; do rm -f python/mojolearn/identical/_mojolearn_$m.so python/mojolearn/_mojolearn_$m.so; bash bindings/build_$m.sh > $O/build-$m.log 2>&1; rc build-$m $?; done
base=$(pixi run python3 -c 'import sys;print(sys.executable)' | tail -1); $base -m venv $O/venv; P=$O/venv/bin/python
$P -m pip -q install mojolearn==0.8.32 numpy==2.5.2 scipy==1.18.0 > $O/pip.log 2>&1; $P -m pip -q install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu >> $O/pip.log 2>&1
S=$($P -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')/mojolearn
cp $S/_version.py /tmp/vsd.py; cp python/mojolearn/*.py $S/; cp /tmp/vsd.py $S/_version.py; find $S -name __pycache__ -exec rm -rf {} +
for m in $MODS; do so=$(ls -t python/mojolearn/identical/_mojolearn_$m.so python/mojolearn/_mojolearn_$m.so 2>/dev/null | head -1); [ -n "$so" ] && cp $so $S/hip/gfx942/identical/ && sha256sum $so >> $O/bindings.sha256; done
timeout 3600 $P tools/neural_stage_timing.py --lane lm-forward --lane lm-train-step --lane transformer-forward --calls 6 > $O/stage-timing.log 2>&1; rc stage-timing $?
echo done > $O/done
