#!/bin/bash
# Everything merged to main on 2026-09-30, on one Hot Aisle MI300X (gfx942):
#  1. Kernel PCA Lanczos (256/1000/10000 rows, output sha vs NVIDIA/Apple/CPU),
#     strides pass (Mamba-3 S16/S17 new vs naive: digests + stage timing; gemm-int8
#     board cell), fixed15 price harness (transposed tile 1 vs 0; digests vs NVIDIA)
#  2. classical pass (tools/classical_pass_run.py amd: old vs new LU/SGD/LARS/IVF + toggle sweep)
#  3. priority pass (tools/priority_pass_run.py amd: session checks, neural priority set,
#     gemm-int8 tile vs reference, board algos rows new vs old)
set -uo pipefail
cd "$(dirname "$0")/.."
O=/root/amd-verify; mkdir -p $O
export PATH=/root/.pixi/bin:/opt/rocm/bin:$PATH MOJOLEARN_TARGET_COLUMN=amd MOJOLEARN_GPU_ARCHS=gfx942 \
  MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_COMPILE_JOBS=2 MOJOLEARN_BENCH_INSTALLED=1 PYTHONUNBUFFERED=1
unset PYTHONPATH
st() { echo "{\"phase\":\"$1\",\"utc\":\"$(date -u +%FT%TZ)\"}" > $O/status.json; }
rc() { echo "$(date -u +%T) $1 rc=$2" >> $O/rc.txt; }
git rev-parse HEAD > $O/head.txt; git diff --binary HEAD > $O/source.patch
st pixi; pixi install > $O/pixi.log 2>&1
base=$(pixi run python3 -c 'import sys;print(sys.executable)' | tail -1)
[ -x $O/venv/bin/python ] || $base -m venv $O/venv
PY=$O/venv/bin/python
$PY -m pip -q install mojolearn==0.8.31 numpy==2.5.2 scipy==1.18.0 > $O/pip.log 2>&1
$PY -m pip -q install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu >> $O/pip.log 2>&1
$PY -m pip freeze > $O/packages.txt
SITE=$($PY -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')/mojolearn
T=$SITE/hip/gfx942/identical
N=$O/native; mkdir -p $N
build() {  # module tag [flags]
  st build-$2; rm -f python/mojolearn/identical/_mojolearn_$1.so python/mojolearn/_mojolearn_$1.so
  MOJOLEARN_MOJO_BUILD_FLAGS="${3:-}" bash bindings/build_$1.sh > $O/build-$2.log 2>&1; rc build-$2 $?
  so=$(ls -t python/mojolearn/identical/_mojolearn_$1.so python/mojolearn/_mojolearn_$1.so 2>/dev/null | head -1)
  [ -n "$so" ] && cp $so $N/$2.so && sha256sum $N/$2.so >> $O/bindings.sha256
}
build mamba mamba-anglenaive "-D MOJOLEARN_MAMBA3_ANGLE_DT_NAIVE=1"
build mamba mamba-new
for m in linalg x_neighbors x_decomp; do build $m $m; done
cp -r python/mojolearn/*.py $SITE/; find $SITE -name __pycache__ -exec rm -rf {} +
for m in linalg x_neighbors x_decomp; do [ -f $N/$m.so ] && cp $N/$m.so $T/_mojolearn_$m.so; done
# 1a. Kernel PCA (the NVIDIA/Apple/CPU runs used this input and this check)
for r in 256 1000 10000; do st kpca-$r
  timeout 1800 $PY tools/kernel_pca_trial_check.py --data /root/taxi-board-10000.npy --rows $r --out $O/kpca-rows-$r.json > $O/kpca-rows-$r.log 2>&1; rc kpca-$r $?; done
# 1b. Mamba-3 backward: the five S16 arms (PR #6) and the angle-naive build; digests must equal each other and the NVIDIA ones
cp $N/mamba-new.so $T/_mojolearn_mamba.so
for arm in regs2 regs shared smem48 naive; do st m3-$arm; export MOJOLEARN_MAMBA3_S16_QK_ARM=$arm
  timeout 1200 $PY tools/mamba3_backward_timing.py --batch 2 --length 512 --d-model 384 --calls 4 > $O/m3bwd-board-$arm.log 2>&1; rc m3bwd-board-$arm $?
  timeout 1200 $PY tools/mamba3_backward_timing.py --batch 8 --length 512 --d-model 768 --calls 3 > $O/m3bwd-default-$arm.log 2>&1; rc m3bwd-default-$arm $?
  timeout 1200 $PY tools/strides_digest.py 2 512 384 > $O/digest-board-$arm.json 2>$O/digest-board-$arm.err; rc digest-board-$arm $?
  timeout 1800 $PY tools/strides_digest.py 8 512 768 > $O/digest-default-$arm.json 2>$O/digest-default-$arm.err; rc digest-default-$arm $?
done; unset MOJOLEARN_MAMBA3_S16_QK_ARM
cp $N/mamba-anglenaive.so $T/_mojolearn_mamba.so; st m3-anglenaive
timeout 1200 $PY tools/mamba3_backward_timing.py --batch 2 --length 512 --d-model 384 --calls 4 > $O/m3bwd-board-anglenaive.log 2>&1; rc m3bwd-board-anglenaive $?
timeout 1200 $PY tools/strides_digest.py 2 512 384 > $O/digest-board-anglenaive.json 2>$O/digest-board-anglenaive.err; rc digest-board-anglenaive $?
cp $N/mamba-new.so $T/_mojolearn_mamba.so
# 1c. board cells, new bindings
for lane in samba-train-step gemm-int8; do st race-$lane
  timeout 3600 $PY tools/bench_board_neural.py race --lane $lane --shape full --arms ours --rounds 5 \
    --out $O/race-$lane --work $O/work --ours-python $PY > $O/race-$lane.log 2>&1; rc race-$lane $?; done
# 1c2. fused small-MLP step (PR #7): both paths byte for byte; matmul digests (compare with NVIDIA)
build training training
cp $N/training.so $T/_mojolearn_training.so
cp -r python/mojolearn/*.py $SITE/; find $SITE -name __pycache__ -exec rm -rf {} +
st mlp-check; timeout 1800 $PY tools/mlp_step_check.py --json $O/mlp-check-256.json > $O/mlp-check-256.log 2>&1; rc mlp-check-256 $?
timeout 1800 $PY tools/mlp_step_check.py --rows 32 --json $O/mlp-check-32.json > $O/mlp-check-32.log 2>&1; rc mlp-check-32 $?
timeout 900 $PY tools/matmul_digest.py > $O/matmul.json 2>$O/matmul.err; rc matmul $?
# 1d. fixed15 price harness (exact integer sums: digests must equal the L40S's)
st build-int15
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 --target-accelerator gfx942 -I . bench/gemm_int15_price_main.mojo -o $O/int15_price > $O/build-int15.log 2>&1; rc build-int15 $?
if [ -x $O/int15_price ]; then for tile in 1 0; do st tile-$tile
  MOJOLEARN_INT15_TRANSPOSED_TILE=$tile timeout 2400 $O/int15_price > $O/price-tile-$tile.log 2>&1; rc price-tile-$tile $?; done; fi
# 2. classical pass, 3. priority pass (each its own venv and builds)
st classical; timeout 21600 bash tools/classical_pass_run.sh amd; rc classical $?
st priority; timeout 21600 bash tools/priority_pass_run.sh amd; rc priority $?
st done; echo done > $O/done
