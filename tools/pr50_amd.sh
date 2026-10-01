#!/bin/bash
# PR #50 (regs2h as the AMD S16 default) on the MI325X: released 0.8.32 + this branch's Python and neural bindings.
# Fixture digests (must stay md5 26abf7d3), backward stage walls and samba-train-step, default vs the regs2 restore.
set -uo pipefail
cd "$(dirname "$0")/.."
O=/root/pr50-amd; rm -rf $O; mkdir -p $O
export PATH=/root/.pixi/bin:/opt/rocm/bin:$PATH MOJOLEARN_NUMERIC_MODE=identical PYTHONUNBUFFERED=1 MOJOLEARN_COMPILE_JOBS=4 MOJOLEARN_TARGET_COLUMN=amd MOJOLEARN_GPU_ARCHS=gfx942 MOJOLEARN_BENCH_INSTALLED=1; unset PYTHONPATH
rc() { echo "$(date -u +%T) $1 rc=$2" >> $O/rc.txt; }
git rev-parse HEAD > $O/head.txt; pixi install > $O/pixi.log 2>&1
MODS="mamba transformer byte_lm training"
for m in $MODS; do rm -f python/mojolearn/identical/_mojolearn_$m.so python/mojolearn/_mojolearn_$m.so; bash bindings/build_$m.sh > $O/build-$m.log 2>&1; rc build-$m $?; done
base=$(pixi run python3 -c 'import sys;print(sys.executable)' | tail -1); $base -m venv $O/venv; P=$O/venv/bin/python
$P -m pip -q install mojolearn==0.8.32 numpy==2.5.2 scipy==1.18.0 > $O/pip.log 2>&1; $P -m pip -q install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu >> $O/pip.log 2>&1
S=$($P -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')/mojolearn
cp $S/_version.py /tmp/v50.py; cp python/mojolearn/*.py $S/; cp /tmp/v50.py $S/_version.py; find $S -name __pycache__ -exec rm -rf {} +
for m in $MODS; do so=$(ls -t python/mojolearn/identical/_mojolearn_$m.so python/mojolearn/_mojolearn_$m.so 2>/dev/null | head -1); [ -n "$so" ] && cp $so $S/hip/gfx942/identical/ && sha256sum $so >> $O/bindings.sha256; done
for arm in default regs2; do ex=""; [ $arm = regs2 ] && ex="MOJOLEARN_MAMBA3_S16_QK_ARM=regs2"
  env $ex timeout 1800 $P tools/strides_digest.py 2 512 384 > $O/digest-$arm.json 2> $O/digest-$arm.err; rc digest-$arm $?
  env $ex MOJOLEARN_MAMBA_TIMING=1 timeout 1800 $P tools/mamba3_backward_timing.py --batch 2 --length 512 --d-model 384 --calls 4 > $O/timing-$arm.log 2>&1; rc timing-$arm $?
  env $ex timeout 3600 $P tools/neural_stage_timing.py --lane samba-train-step --calls 8 > $O/samba-$arm.log 2>&1; rc samba-$arm $?; done
echo done > $O/done
