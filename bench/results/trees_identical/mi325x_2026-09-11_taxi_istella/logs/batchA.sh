#!/bin/sh
# MI325X leg 1, 2026-09-11 (lane/amd-trees-leg), DigitalOcean tor1, the first
# trees leg under ENGINEERING_RULES.md section 10. IDENTICAL tier, default
# build (DEVIATION 2502 pure-node leaf ON), never below 1M rows. Opponents on
# this box: CatBoost, scikit-learn RF/ET on the CPU (no AMD GPU path);
# XGBoost on the GPU when AMD's ROCm build passed the setup probe, else CPU;
# LightGBM on OpenCL when it built, else CPU. Every full cell interleaves ours
# with the opponents round by round, 1 warm-up + 5 rounds, opponents imported
# BEFORE our binding in the process (--opponents-first).
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
OUT=/root/trees_out
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
if mountpoint -q /mnt/mojolearn-data; then export GBM_BENCH_DATA=/mnt/mojolearn-data/gbm-bench; fi
mark() { echo "$1 $(date -u +%T)" | tee -a $OUT/ab.txt; : > "$OUT/phase.$1"; }
while [ ! -f $OUT/setup.done ]; do sleep 15; done
echo "batchA start $(date -u +%T)"; cat $OUT/setup.txt
grep -q '^import_identical=0' $OUT/setup.txt || { echo "import_identical failed; see logs/import_identical.log"; exit 3; }

XGB=xgboost-cpu; LGB=lightgbm-cpu; PY=python3
grep -q '^xgb_rocm_works=yes' $OUT/setup.txt && { XGB=xgboost-gpu; PY=/root/venv-gpu/bin/python; }
grep -q '^lightgbm_opencl_works=yes' $OUT/setup.txt && { LGB=lightgbm-opencl; PY=/root/venv-gpu/bin/python; }
export MOJOLEARN_SPEED_PY=$PY MOJOLEARN_SPEED_DEVICES=cpu,gpu,opencl MOJOLEARN_SPEED_OPPONENTS_FIRST=1
echo "arms: xgboost=$XGB lightgbm=$LGB python=$PY" | tee -a $OUT/ab.txt
$PY -c "import catboost, sklearn, xgboost, lightgbm, numpy, sys; print('python', sys.version.split()[0]); print('catboost', catboost.__version__); print('sklearn', sklearn.__version__); print('xgboost', xgboost.__version__, xgboost.__file__); print('lightgbm', lightgbm.__version__, lightgbm.__file__); print('numpy', numpy.__version__)" > $OUT/versions_used.txt 2>&1
cat $OUT/versions_used.txt

full() {  # <lane> <dataset> <rows> <arms>
    MOJOLEARN_SPEED_ARMS="$4" $AB speed baseline "$1" "$2" "$3" 5 full
}

# ---------- PHASE_T: taxi, every lane at 1M, then RF 2M.
full rf taxi 1000000 sklearn-rf-cpu
full gbdt-symmetric taxi 1000000 catboost-cpu
full gbdt-depthwise taxi 1000000 "catboost-cpu,$XGB"
full gbdt-lossguide taxi 1000000 "catboost-cpu,$XGB,$LGB"
full et taxi 1000000 sklearn-et-cpu
mark PHASE_T1_DONE
full rf taxi 2000000 sklearn-rf-cpu
mark PHASE_T2_DONE

# ---------- PHASE_I: Istella-S boosting cells (the forest cells go to leg 2).
full gbdt-symmetric istella 1000000 catboost-cpu
full gbdt-depthwise istella 1000000 "catboost-cpu,$XGB"
full gbdt-lossguide istella 1000000 "catboost-cpu,$XGB,$LGB"
mark PHASE_I_DONE
echo "batchA end $(date -u +%T)"
