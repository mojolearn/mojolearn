#!/bin/bash
# PR #6 (the S16 pass) on one NVIDIA box: ftz hardware-flush probe, S16 arms, angle d_dt,
# ftz A/B across the neural board, classical and Kernel PCA digests under the new flush.
set -uo pipefail
cd "$(dirname "$0")/.."
O=/root/s16; mkdir -p $O
export PATH=/root/.pixi/bin:$PATH MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_GPU_ARCHS=${ARCH:-sm_89} \
  MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_COMPILE_JOBS=2 MOJOLEARN_BENCH_INSTALLED=1 PYTHONUNBUFFERED=1
unset PYTHONPATH
st() { echo "{\"phase\":\"$1\",\"utc\":\"$(date -u +%FT%TZ)\"}" > $O/status.json; }
rc() { echo "$(date -u +%T) $1 rc=$2" >> $O/rc.txt; }
nvidia-smi --query-gpu=name --format=csv,noheader > $O/gpu.txt
git rev-parse HEAD > $O/head.txt; git diff --binary HEAD > $O/source.patch
st pixi; pixi install > $O/pixi.log 2>&1
st probe; timeout 3600 pixi run mojo run -I . tools/probe_ftz_hw.mojo > $O/probe-ftz-hw.log 2>&1; rc probe $?
base=$(pixi run python3 -c 'import sys;print(sys.executable)' | tail -1)
[ -x $O/venv/bin/python ] || $base -m venv $O/venv
PY=$O/venv/bin/python
$PY -m pip -q install mojolearn==0.8.31 numpy==2.5.2 scipy==1.18.0 > $O/pip.log 2>&1
$PY -m pip -q install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu >> $O/pip.log 2>&1
$PY -m pip freeze > $O/packages.txt
SITE=$($PY -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')/mojolearn
T=$SITE/cuda/$MOJOLEARN_GPU_ARCHS/identical
N=$O/native; mkdir -p $N
build() {  # module tag [flags]
  st build-$2; rm -f python/mojolearn/identical/_mojolearn_$1.so python/mojolearn/_mojolearn_$1.so
  MOJOLEARN_MOJO_BUILD_FLAGS="${3:-}" bash bindings/build_$1.sh > $O/build-$2.log 2>&1; rc build-$2 $?
  so=$(ls -t python/mojolearn/identical/_mojolearn_$1.so python/mojolearn/_mojolearn_$1.so 2>/dev/null | head -1)
  [ -n "$so" ] && cp $so $N/$2.so && sha256sum $N/$2.so >> $O/bindings.sha256
}
OFF="-D MOJOLEARN_FTZ_HW_OFF=1"
build mamba mamba-anglenaive "-D MOJOLEARN_MAMBA3_ANGLE_DT_NAIVE=1"
for m in mamba transformer byte_lm; do build $m $m-ftzoff "$OFF"; done
for m in mamba transformer byte_lm x_decomp x_linear ivf x_neighbors linalg; do build $m $m; done
cp -r python/mojolearn/*.py $SITE/; find $SITE -name __pycache__ -exec rm -rf {} +
use() { for m in "$@"; do cp $N/$m.so $T/_mojolearn_${m%-*}.so 2>/dev/null || cp $N/$m.so $T/_mojolearn_$m.so; done; }
inst() { cp $N/$1.so $T/_mojolearn_$2.so; }
for m in mamba transformer byte_lm x_decomp x_linear ivf x_neighbors linalg; do inst $m $m; done
export MOJOLEARN_REPO_COMMIT=$(cat $O/head.txt)
# 1. S16 arms (new flush): stage timing and digests
for arm in regs2 regs shared smem48 naive; do st s16-$arm
  MOJOLEARN_MAMBA3_S16_QK_ARM=$arm timeout 1200 $PY tools/mamba3_backward_timing.py --batch 2 --length 512 --d-model 384 --calls 4 > $O/m3bwd-board-$arm.log 2>&1; rc m3bwd-board-$arm $?
  MOJOLEARN_MAMBA3_S16_QK_ARM=$arm timeout 1200 $PY tools/mamba3_backward_timing.py --batch 8 --length 512 --d-model 768 --calls 3 > $O/m3bwd-default-$arm.log 2>&1; rc m3bwd-default-$arm $?
  MOJOLEARN_MAMBA3_S16_QK_ARM=$arm timeout 1200 $PY tools/strides_digest.py 2 512 384 > $O/digest-board-$arm.json 2>$O/digest-board-$arm.err; rc digest-board-$arm $?
  MOJOLEARN_MAMBA3_S16_QK_ARM=$arm timeout 1800 $PY tools/strides_digest.py 8 512 768 > $O/digest-default-$arm.json 2>$O/digest-default-$arm.err; rc digest-default-$arm $?
done
# 2. angle d_dt naive build, and the flush off (mamba), default arm
for b in mamba-anglenaive mamba-ftzoff; do inst $b mamba; st $b
  timeout 1200 $PY tools/mamba3_backward_timing.py --batch 2 --length 512 --d-model 384 --calls 4 > $O/m3bwd-board-$b.log 2>&1; rc m3bwd-board-$b $?
  timeout 1200 $PY tools/mamba3_backward_timing.py --batch 8 --length 512 --d-model 768 --calls 3 > $O/m3bwd-default-$b.log 2>&1; rc m3bwd-default-$b $?
  timeout 1200 $PY tools/strides_digest.py 2 512 384 > $O/digest-board-$b.json 2>$O/digest-board-$b.err; rc digest-board-$b $?
  timeout 1800 $PY tools/strides_digest.py 8 512 768 > $O/digest-default-$b.json 2>$O/digest-default-$b.err; rc digest-default-$b $?
done
inst mamba mamba
# 3. session checks, the s16 set, the whole neural board (new flush), then the flush off
for group in reuse refusals lifetime budget; do st session-$group
  timeout 600 $PY tools/transformer_session_check.py --binding $T/_mojolearn_transformer.so --backend cuda --group $group --out $O/session-$group.json > $O/session-$group.log 2>&1; rc session-$group $?; done
st set-s16; timeout 3600 $PY tools/neural_experiments.py --set s16 --lane mamba3-forward --lane samba-train-step --calls 10 --json $O/set-s16.json > $O/set-s16.log 2>&1; rc set-s16 $?
st set-nvidia; timeout 7200 $PY tools/neural_experiments.py --set nvidia --calls 10 --json $O/set-nvidia.json > $O/set-nvidia.log 2>&1; rc set-nvidia $?
for m in mamba transformer byte_lm; do inst $m-ftzoff $m; done
st baseline-ftzoff; timeout 7200 $PY tools/neural_experiments.py --set nvidia --only baseline --calls 10 --json $O/baseline-ftzoff.json > $O/baseline-ftzoff.log 2>&1; rc baseline-ftzoff $?
for m in mamba transformer byte_lm; do inst $m $m; done
st samba-profile; timeout 1800 $PY tools/samba_step_profile.py --calls 5 > $O/samba-step-profile.log 2>&1; rc samba-profile $?
# 4. classical and Kernel PCA digests under the new flush (compare with the classical-pass and kernel-pca evidence)
mkdir -p $O/classical
for c in "lu 1024" "sgd-reg 20000" "sgd-clf 20000" "lars 200000" "ivf 40000"; do set -- $c; st classical-$1
  timeout 3600 $PY tools/classical_pass_ab.py case $1 $2 --out $O/classical/$1-new-$2.json > $O/classical/$1-new-$2.log 2>&1; rc classical-$1 $?; done
if [ -f /root/taxi-board-10000.npy ]; then for r in 256 1000 10000; do st kpca-$r
  timeout 1800 $PY tools/kernel_pca_trial_check.py --data /root/taxi-board-10000.npy --rows $r --out $O/kpca-rows-$r.json > $O/kpca-rows-$r.log 2>&1; rc kpca-$r $?; done; fi
st race-gemm-int8; timeout 3600 $PY tools/bench_board_neural.py race --lane gemm-int8 --shape full --arms ours --rounds 5 --out $O/race-gemm-int8 --work $O/work --ours-python $PY > $O/race-gemm-int8.log 2>&1; rc race-gemm-int8 $?
# 5. fixed15 price harness: the dW rule change (table vs H100 choice)
st build-int15
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 --target-accelerator $MOJOLEARN_GPU_ARCHS -I . bench/gemm_int15_price_main.mojo -o $O/int15_price > $O/build-int15.log 2>&1; rc build-int15 $?
if [ -x $O/int15_price ]; then
  st price-table; timeout 1800 $O/int15_price > $O/price-table.log 2>&1; rc price-table $?
  st price-high; MOJOLEARN_INT15_BOX=high timeout 1800 $O/int15_price > $O/price-high.log 2>&1; rc price-high $?
fi
st done; echo done > $O/done
