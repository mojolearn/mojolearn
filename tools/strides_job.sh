#!/bin/bash
# The strides pass on one NVIDIA box: each change against its restore.
set -uo pipefail
cd "$(dirname "$0")/.."
O=/root/strides; mkdir -p $O
export PATH=/root/.pixi/bin:$PATH MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_GPU_ARCHS=sm_89 \
  MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_COMPILE_JOBS=2 MOJOLEARN_BENCH_INSTALLED=1 PYTHONUNBUFFERED=1
unset PYTHONPATH
st() { echo "{\"phase\":\"$1\",\"utc\":\"$(date -u +%FT%TZ)\"}" > $O/status.json; }
rc() { echo "$1 rc=$2" >> $O/rc.txt; }
git rev-parse HEAD > $O/head.txt; git diff --binary HEAD > $O/source.patch
st pixi; pixi install > $O/pixi.log 2>&1
base=$(pixi run python3 -c 'import sys;print(sys.executable)' | tail -1)
[ -x $O/venv/bin/python ] || $base -m venv $O/venv
PY=$O/venv/bin/python
$PY -m pip -q install mojolearn==0.8.31 numpy==2.5.2 scipy==1.18.0 > $O/pip.log 2>&1
$PY -m pip -q install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu >> $O/pip.log 2>&1
SITE=$($PY -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')/mojolearn
T=$SITE/cuda/sm_89/identical
N=$O/native; mkdir -p $N
build() {  # module tag [flags]
  st build-$2; rm -f python/mojolearn/identical/_mojolearn_$1.so python/mojolearn/_mojolearn_$1.so
  MOJOLEARN_MOJO_BUILD_FLAGS="${3:-}" bash bindings/build_$1.sh > $O/build-$2.log 2>&1; rc build-$2 $?
  so=$(ls -t python/mojolearn/identical/_mojolearn_$1.so python/mojolearn/_mojolearn_$1.so 2>/dev/null | head -1)
  [ -n "$so" ] && cp $so $N/$2.so && sha256sum $N/$2.so >> $O/bindings.sha256
}
build mamba mamba-naive "-D MOJOLEARN_MAMBA3_S16_QK_NAIVE=1 -D MOJOLEARN_MAMBA3_S17_OPERANDS_NAIVE=1"
build mamba mamba-new
build linalg linalg-new
cp -r python/mojolearn/*.py $SITE/; find $SITE -name __pycache__ -exec rm -rf {} +
cp $N/linalg-new.so $T/_mojolearn_linalg.so
# 1. Mamba-3 backward: stage timing and digests, naive kernels vs new
for arm in naive new; do
  cp $N/mamba-$arm.so $T/_mojolearn_mamba.so
  st m3-$arm
  timeout 900 $PY tools/mamba3_backward_timing.py --batch 2 --length 512 --d-model 384 --calls 4 > $O/m3bwd-board-$arm.log 2>&1; rc m3bwd-board-$arm $?
  timeout 900 $PY tools/mamba3_backward_timing.py --batch 8 --length 512 --d-model 768 --calls 3 > $O/m3bwd-default-$arm.log 2>&1; rc m3bwd-default-$arm $?
  timeout 900 $PY tools/strides_digest.py 2 512 384 > $O/digest-board-$arm.json 2>$O/digest-board-$arm.err; rc digest-board-$arm $?
  timeout 900 $PY tools/strides_digest.py 8 512 768 > $O/digest-default-$arm.json 2>$O/digest-default-$arm.err; rc digest-default-$arm $?
done
# 2. board cells (ours only), new bindings
for lane in samba-train-step gemm-int8; do
  st race-$lane
  timeout 3600 $PY tools/bench_board_neural.py race --lane $lane --shape full --arms ours --rounds 5 \
    --out $O/race-$lane --work $O/work --ours-python $PY > $O/race-$lane.log 2>&1; rc race-$lane $?
done
# 3. fixed15 price harness: transposed tile A/B and the plan table A/B (digests per row)
st build-int15
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 --target-accelerator sm_89 -I . bench/gemm_int15_price_main.mojo -o $O/int15_price > $O/build-int15.log 2>&1; rc build-int15 $?
if [ -x $O/int15_price ]; then
  for tile in 1 0; do st tile-$tile
    MOJOLEARN_INT15_TRANSPOSED_TILE=$tile timeout 1800 $O/int15_price > $O/price-tile-$tile.log 2>&1; rc price-tile-$tile $?; done
  st box-high
  MOJOLEARN_INT15_BOX=high timeout 1800 $O/int15_price > $O/price-box-high.log 2>&1; rc price-box-high $?
fi
st done; echo done > $O/done
