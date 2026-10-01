#!/bin/bash
# PR #44 on one GPU box: released 0.8.32 + this branch's Python and mamba binding; per S16 arm the
# Mamba3Block fixture digests (2x512x384) and the backward stage walls. Digests are the reference for the M3.
set -uo pipefail
cd "$(dirname "$0")/.."
V=$1; O=/root/pr44-$V; rm -rf $O; mkdir -p $O
if [ $V = nvidia ]; then BK=cuda; AR=sm_89; else BK=hip; AR=gfx942; fi
export PATH=/root/.pixi/bin:/opt/rocm/bin:$PATH MOJOLEARN_NUMERIC_MODE=identical PYTHONUNBUFFERED=1 MOJOLEARN_COMPILE_JOBS=4 MOJOLEARN_TARGET_COLUMN=$V MOJOLEARN_GPU_ARCHS=$AR MOJOLEARN_BENCH_INSTALLED=1; unset PYTHONPATH
rc() { echo "$(date -u +%T) $1 rc=$2" >> $O/rc.txt; }
git rev-parse HEAD > $O/head.txt; pixi install > $O/pixi.log 2>&1
rm -f python/mojolearn/identical/_mojolearn_mamba.so python/mojolearn/_mojolearn_mamba.so
bash bindings/build_mamba.sh > $O/build-mamba.log 2>&1; rc build-mamba $?
base=$(pixi run python3 -c 'import sys;print(sys.executable)' | tail -1); $base -m venv $O/venv; P=$O/venv/bin/python
$P -m pip -q install mojolearn==0.8.32 numpy==2.5.2 scipy==1.18.0 > $O/pip.log 2>&1; $P -m pip -q install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu >> $O/pip.log 2>&1
S=$($P -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')/mojolearn
cp $S/_version.py /tmp/v44.py; cp python/mojolearn/*.py $S/; cp /tmp/v44.py $S/_version.py; find $S -name __pycache__ -exec rm -rf {} +
so=$(ls -t python/mojolearn/identical/_mojolearn_mamba.so python/mojolearn/_mojolearn_mamba.so 2>/dev/null | head -1); cp $so $S/$BK/$AR/identical/; sha256sum $so > $O/bindings.sha256
for arm in default naive regs2h regsh; do ex=""; [ $arm != default ] && ex="MOJOLEARN_MAMBA3_S16_QK_ARM=$arm"
  env $ex timeout 1800 $P tools/strides_digest.py 2 512 384 > $O/digest-$arm.json 2> $O/digest-$arm.err; rc digest-$arm $?
  env $ex MOJOLEARN_MAMBA_TIMING=1 timeout 1800 $P tools/mamba3_backward_timing.py --batch 2 --length 512 --d-model 384 --calls 4 > $O/timing-$arm.log 2>&1; rc timing-$arm $?; done
echo done > $O/done
