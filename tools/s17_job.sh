#!/bin/bash
# PR #9 (S17 tail pipe kernel) on one NVIDIA box: timing and digests, pipe (default) vs shared.
set -uo pipefail
cd "$(dirname "$0")/.."
O=/root/s17; rm -rf $O/rc.txt; mkdir -p $O
export PATH=/root/.pixi/bin:$PATH MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_GPU_ARCHS=sm_89 \
  MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_COMPILE_JOBS=2 MOJOLEARN_BENCH_INSTALLED=1 PYTHONUNBUFFERED=1
unset PYTHONPATH
st() { echo "{\"phase\":\"$1\",\"utc\":\"$(date -u +%FT%TZ)\"}" > $O/status.json; }
rc() { echo "$(date -u +%T) $1 rc=$2" >> $O/rc.txt; }
git rev-parse HEAD > $O/head.txt
PY=/root/pass4/venv/bin/python   # the pass-4 overlay venv on this pod (0.8.31 + main's bindings); mamba replaced below
SITE=$($PY -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')/mojolearn; T=$SITE/cuda/sm_89/identical
st build-mamba; rm -f python/mojolearn/identical/_mojolearn_mamba.so python/mojolearn/_mojolearn_mamba.so
bash bindings/build_mamba.sh > $O/build-mamba.log 2>&1; rc build-mamba $?
so=$(ls -t python/mojolearn/identical/_mojolearn_mamba.so python/mojolearn/_mojolearn_mamba.so 2>/dev/null | head -1); cp $so $T/_mojolearn_mamba.so
cp -r python/mojolearn/*.py $SITE/; find $SITE -name __pycache__ -exec rm -rf {} +
export MOJOLEARN_REPO_COMMIT=$(cat $O/head.txt)
for arm in pipe shared; do st m3-$arm
  if [ $arm = pipe ]; then unset MOJOLEARN_MAMBA3_S17_TAIL_ARM; else export MOJOLEARN_MAMBA3_S17_TAIL_ARM=shared; fi
  timeout 1200 $PY tools/mamba3_backward_timing.py --batch 2 --length 512 --d-model 384 --calls 4 > $O/m3bwd-board-$arm.log 2>&1; rc m3bwd-board-$arm $?
  timeout 1200 $PY tools/mamba3_backward_timing.py --batch 8 --length 512 --d-model 768 --calls 3 > $O/m3bwd-default-$arm.log 2>&1; rc m3bwd-default-$arm $?
  timeout 1200 $PY tools/strides_digest.py 2 512 384 > $O/digest-board-$arm.json 2>$O/digest-board-$arm.err; rc digest-board-$arm $?
  timeout 1800 $PY tools/strides_digest.py 8 512 768 > $O/digest-default-$arm.json 2>$O/digest-default-$arm.err; rc digest-default-$arm $?
done; unset MOJOLEARN_MAMBA3_S17_TAIL_ARM
st set-s17; timeout 3600 $PY tools/neural_experiments.py --set s17 --lane mamba3-forward --lane samba-train-step --calls 10 --json $O/set-s17.json > $O/set-s17.log 2>&1; rc set-s17 $?
st done; echo done > $O/done
