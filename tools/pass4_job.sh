#!/bin/bash
# PR #8 (resident optimizer moments, zero-copy pack, S16 default regs, fixed15 k<=1024 rule) on one NVIDIA box.
set -uo pipefail
cd "$(dirname "$0")/.."
O=/root/pass4; rm -rf $O/native $O/rc.txt; mkdir -p $O
export PATH=/root/.pixi/bin:$PATH MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_GPU_ARCHS=sm_89 \
  MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_COMPILE_JOBS=2 MOJOLEARN_BENCH_INSTALLED=1 PYTHONUNBUFFERED=1
unset PYTHONPATH
st() { echo "{\"phase\":\"$1\",\"utc\":\"$(date -u +%FT%TZ)\"}" > $O/status.json; }
rc() { echo "$(date -u +%T) $1 rc=$2" >> $O/rc.txt; }
nvidia-smi --query-gpu=name --format=csv,noheader > $O/gpu.txt; git rev-parse HEAD > $O/head.txt
st pixi; pixi install > $O/pixi.log 2>&1
base=$(pixi run python3 -c 'import sys;print(sys.executable)' | tail -1)
[ -x $O/venv/bin/python ] || $base -m venv $O/venv
PY=$O/venv/bin/python
$PY -m pip -q install mojolearn==0.8.31 numpy==2.5.2 scipy==1.18.0 pytest > $O/pip.log 2>&1
$PY -m pip -q install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu >> $O/pip.log 2>&1
SITE=$($PY -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')/mojolearn
T=$SITE/cuda/sm_89/identical; N=$O/native; mkdir -p $N
build() { st build-$1; rm -f python/mojolearn/identical/_mojolearn_$1.so python/mojolearn/_mojolearn_$1.so
  bash bindings/build_$1.sh > $O/build-$1.log 2>&1; rc build-$1 $?
  so=$(ls -t python/mojolearn/identical/_mojolearn_$1.so python/mojolearn/_mojolearn_$1.so 2>/dev/null | head -1)
  [ -n "$so" ] && cp $so $N/$1.so && cp $so $T/_mojolearn_$1.so && sha256sum $so >> $O/bindings.sha256; }
for m in training mamba linalg transformer byte_lm; do build $m; done
cp -r python/mojolearn/*.py $SITE/; cp -r python/mojolearn/tests $SITE/; find $SITE -name __pycache__ -exec rm -rf {} +
export MOJOLEARN_REPO_COMMIT=$(cat $O/head.txt)
# 1. resident optimizer: both paths byte for byte
for k in adamw adam sgd; do st opt-check-$k
  timeout 1800 $PY tools/optimizer_resident_check.py --kind $k --json $O/opt-check-$k.json > $O/opt-check-$k.log 2>&1; rc opt-check-$k $?; done
# 2. the board cells with the resident optimizer on and off (digests must agree)
st set-optimizer; timeout 3600 $PY tools/neural_experiments.py --set optimizer --lane samba-train-step --lane lm-train-step --calls 10 --json $O/set-optimizer.json > $O/set-optimizer.log 2>&1; rc set-optimizer $?
st samba-profile; timeout 1800 $PY tools/samba_step_profile.py --calls 5 > $O/samba-step-profile.log 2>&1; rc samba-profile $?
# 3. S16 default regs: digests must equal the strides/S16 evidence; timing regs vs regs2
for arm in default regs2; do st m3-$arm
  if [ $arm = default ]; then unset MOJOLEARN_MAMBA3_S16_QK_ARM; else export MOJOLEARN_MAMBA3_S16_QK_ARM=$arm; fi
  timeout 1200 $PY tools/mamba3_backward_timing.py --batch 2 --length 512 --d-model 384 --calls 4 > $O/m3bwd-board-$arm.log 2>&1; rc m3bwd-board-$arm $?
  timeout 1200 $PY tools/strides_digest.py 2 512 384 > $O/digest-board-$arm.json 2>$O/digest-board-$arm.err; rc digest-board-$arm $?
  timeout 1800 $PY tools/strides_digest.py 8 512 768 > $O/digest-default-$arm.json 2>$O/digest-default-$arm.err; rc digest-default-$arm $?
done; unset MOJOLEARN_MAMBA3_S16_QK_ARM
# 4. the small-MLP GPU gate and matmul digests (the optimizer path is shared)
for f in 1 0; do st mlp-tests-fused$f; (cd /tmp && MOJOLEARN_RUN_SMALL_MLP_GPU=1 MOJOLEARN_MLP_FUSED=$f timeout 1800 $PY -m pytest -q -rs -p no:cacheprovider $SITE/tests/test_small_mlp_surface.py $SITE/tests/test_small_mlp_numerical_edges.py) > $O/mlp-tests-fused$f.log 2>&1; rc mlp-tests-fused$f $?; done
for r in 1 0; do st mlp-check-resident$r; MOJOLEARN_OPTIMIZER_RESIDENT=$r timeout 1800 $PY tools/mlp_step_check.py > $O/mlp-check-resident$r.log 2>&1; rc mlp-check-resident$r $?; done
st matmul; timeout 900 $PY tools/matmul_digest.py > $O/matmul.json 2>$O/matmul.err; rc matmul $?
# 5. fixed15: the k <= 1024 rule (table vs H100 choice)
st build-int15
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 --target-accelerator sm_89 -I . bench/gemm_int15_price_main.mojo -o $O/int15_price > $O/build-int15.log 2>&1; rc build-int15 $?
if [ -x $O/int15_price ]; then st price-table; timeout 1800 $O/int15_price > $O/price-table.log 2>&1; rc price-table $?
  st price-high; MOJOLEARN_INT15_BOX=high timeout 1800 $O/int15_price > $O/price-high.log 2>&1; rc price-high $?; fi
st done; echo done > $O/done
