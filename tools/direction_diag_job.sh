#!/bin/bash
# The direction pass's diagnostics on one NVIDIA box (EXPERIMENTS.md "The direction pass").
set -uo pipefail
cd "$(dirname "$0")/.."
O=/root/direction; mkdir -p $O
export PATH=/root/.pixi/bin:$PATH MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_GPU_ARCHS=sm_89 \
  MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_COMPILE_JOBS=2 MOJOLEARN_BENCH_INSTALLED=1 PYTHONUNBUFFERED=1
unset PYTHONPATH
st() { echo "{\"phase\":\"$1\",\"utc\":\"$(date -u +%FT%TZ)\"}" > $O/status.json; }
git rev-parse HEAD > $O/head.txt; git diff --binary HEAD > $O/source.patch
st pixi; pixi install > $O/pixi.log 2>&1
base=$(pixi run python3 -c 'import sys;print(sys.executable)' | tail -1)
[ -x $O/venv/bin/python ] || $base -m venv $O/venv
PY=$O/venv/bin/python
$PY -m pip -q install mojolearn==0.8.31 numpy==2.5.2 scipy==1.18.0 > $O/pip.log 2>&1
$PY -m pip -q install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu >> $O/pip.log 2>&1
SITE=$($PY -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')/mojolearn
T=$SITE/cuda/sm_89/identical
for m in linalg mamba; do
  st build-$m; rm -f python/mojolearn/identical/_mojolearn_$m.so python/mojolearn/_mojolearn_$m.so
  bash bindings/build_$m.sh > $O/build-$m.log 2>&1 || { echo "build $m failed" >> $O/rc.txt; continue; }
  so=$(ls -t python/mojolearn/identical/_mojolearn_$m.so python/mojolearn/_mojolearn_$m.so 2>/dev/null | head -1)
  cp $so $T/; sha256sum $so >> $O/bindings.sha256
done
cp -r python/mojolearn/*.py $SITE/; find $SITE -name __pycache__ -exec rm -rf {} +
st int8-profile; MOJOLEARN_LOWBIT_TIMING=1 timeout 900 $PY tools/int8_profile.py --calls 5 > $O/int8-profile.log 2>&1; echo "int8 rc=$?" >> $O/rc.txt
st mamba-board; timeout 900 $PY tools/mamba3_backward_timing.py --batch 2 --length 512 --d-model 384 --calls 4 > $O/m3bwd-board-shape.log 2>&1; echo "m3 board rc=$?" >> $O/rc.txt
st mamba-default; timeout 900 $PY tools/mamba3_backward_timing.py --batch 8 --length 512 --d-model 768 --calls 3 > $O/m3bwd-default.log 2>&1; echo "m3 default rc=$?" >> $O/rc.txt
st build-int15; pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 --target-accelerator sm_89 -I . bench/gemm_int15_price_main.mojo -o $O/int15_price > $O/build-int15.log 2>&1 || echo "build int15 failed" >> $O/rc.txt
if [ -x $O/int15_price ]; then
  for p in 0 1 2 3 4 5 6 7 8 9 10; do st plan-$p
    MOJOLEARN_INT15_PRICE_ONLY=mlp_up.t512 MOJOLEARN_INT15_PLAN=$p timeout 900 $O/int15_price > $O/plan-$p-mlp_up.log 2>&1; echo "plan $p rc=$?" >> $O/rc.txt; done
  for rb in 1 0; do for row in qkv.t512 mlp_up.t512; do st rowblock-$rb-$row
    MOJOLEARN_INT15_PRICE_ONLY=$row MOJOLEARN_INT15_ROW_BLOCK=$rb timeout 900 $O/int15_price > $O/rowblock-$rb-$row.log 2>&1; echo "rowblock $rb $row rc=$?" >> $O/rc.txt; done; done
fi
st done; echo done > $O/done
