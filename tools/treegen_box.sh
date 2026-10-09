#!/bin/bash
# Tree generalization race, attempt 2 (owner request 2026-10-09): the opponents attempt 1 could not race.
# Runs ON an NVIDIA lq box as a CMD job (lq add nv2 CMD lane/treegen-2 <tag> '... bash tools/treegen_box.sh'),
# in the branch tree with our bindings already built by box_job.sh (BUILDS= on the lq line, IDENTICAL, sm_89).
#   rf vs cuml-rf-gpu, et vs lightgbm-cuda (boosting=rf + extra_trees), gbdt-depthwise vs xgboost-gpu;
#   on covtype (581k x 54; gbdt lanes run binary covtype2), higgs --rows 1000000, higgs (11M x 28).
# Owner rule (2026-10-09): scikit-learn is installed ONLY as an import dependency of cuML / xgboost / lightgbm wrappers.
# No CPU arm, no scikit-learn arm: every opponent arm is a GPU arm (--devices gpu). CatBoost stays out (attempt 1 has it).
# Data: presigned R2 GET URLs in TG_HIGGS_URL / TG_COVTYPE_URL (env, never committed); skipped when the npz is present.
set -u
T=$(pwd); PY=$T/.pixi/envs/default/bin/python; V=/root/opp-tg; P=$V/bin
export PYTHONUNBUFFERED=1
LOGD=/root/tg2-logs/$(date -u +%Y%m%dT%H%M%S); mkdir -p $LOGD; echo "TG2-LOGDIR $LOGD"   # full race output stays on the box
echo "TG2-HEAD $(git -C $T log --oneline -1 | cut -c1-80)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | sed 's/^/TG2-GPU /'
mkdir -p /root/gbm/higgs /root/gbm/covtype
for n in higgs covtype; do
  f=/root/gbm/$n/${n}_speed.npz
  if [ -s $f ]; then echo "TG2-DATA $n present"; continue; fi
  u=TG_${n^^}_URL; [ -n "${!u:-}" ] || { echo "TG2-DATA $n MISSING and no $u"; exit 3; }
  curl -sSf -o $f.part "${!u}" && mv $f.part $f; echo "TG2-DATA $n fetched rc=$? bytes=$(stat -c %s $f 2>/dev/null)"
done
# opponent venv on top of the tree's pixi python (numpy and the Mojo runtime come from there); rebuilt when its base is gone
$P/python -c 1 2>/dev/null || { rm -rf $V; $PY -m venv --system-site-packages $V; }
$P/pip -q install catboost==1.2.10 xgboost==3.2.0 cmake scikit-build-core ninja scikit-learn 2>&1 | tail -2
$P/pip show cuml-cu12 > /dev/null 2>&1 || $P/pip -q install --extra-index-url=https://pypi.nvidia.com "cuml-cu12==26.8.*" 2>&1 | tail -2
# LightGBM with CUDA: attempt 1 needed CUDACXX set ("No CMAKE_CUDA_COMPILER" without it)
# probe: one tiny round with device_type=cuda (fails on a CPU-only LightGBM build)
LGBPROBE='import numpy as np, lightgbm as lgb; lgb.train({"device_type": "cuda", "verbose": -1}, lgb.Dataset(np.random.rand(256, 3), np.random.rand(256)), 1)'
if ! $P/python -c "$LGBPROBE" > /dev/null 2>&1; then
  CUDACXX=/usr/local/cuda/bin/nvcc PATH=/usr/local/cuda/bin:$PATH $P/pip install --force-reinstall --no-deps --no-binary lightgbm \
    --config-settings=cmake.define.USE_CUDA=ON --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=89 lightgbm > $LOGD/lgb-build.log 2>&1
  echo "TG2-LGB-BUILD rc=$? $(tail -1 $LOGD/lgb-build.log | cut -c1-160)"
  $P/python -c "$LGBPROBE" > /dev/null 2>&1; echo "TG2-LGB-CUDA-PROBE rc=$?"
fi
for m in catboost xgboost cuml lightgbm sklearn; do $P/python -c "import $m;print(\"OPPVER\",\"$m\",$m.__version__)" 2>&1 | tail -1; done
$P/pip list 2>/dev/null | grep -iE "scikit|cuml|xgboost|lightgbm" | sed 's/^/TG2-PIP /'
export GBM_BENCH_DATA=/root/gbm PYTHONPATH=$T/python MOJOLEARN_SPEED_ROUNDS=1 MOJOLEARN_SPEED_SIZE=shipped \
  MOJOLEARN_SPEED_EXPECTED_VENDOR=cuda MOJOLEARN_SPEED_BUDGET_S=5400 MOJOLEARN_SPEED_DEADLINE_S=21600
for SHAPE in "covtype" "higgs --rows 1000000" "higgs"; do
  for L in "rf cuml-rf-gpu" "et lightgbm-cuda" "gbdt-depthwise xgboost-gpu"; do
    set -- $L; echo "TG-RACE lane=$1 shape=$SHAPE"
    f=$LOGD/race-$1-${SHAPE// /_}.log
    $P/python -u bench/speed/forest_speed_arm.py --lane $1 --dataset $SHAPE --devices gpu --arms $2 --mem > $f 2>&1; rc=$?
    grep -E '^(FSPEED|OPPVER)' $f; echo "TG-RACE-RC lane=$1 shape=$SHAPE rc=$rc"
    [ $rc = 0 ] || grep -m 3 -E 'Error|Traceback' $f | cut -c1-240
  done
done
echo TG-DONE
