#!/bin/sh
# Leg 1, replacing batchA/batchB after the taxi fetch 403 (logs/fetch_taxi.sh).
# While the Istella-S download and its single-core decode run, only ours-only
# work that tolerates one busy core runs here: the symmetric
# use_pointwise_searcher A/B (both arms alternate in one process, so a
# shared slowdown cancels in the ratio) and the taxi stage ledgers (splits,
# not timings). The opponent cells, whose CPU arms take all 20 cores, wait
# for setup.done, i.e. for the decode to finish.
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
OUT=/root/trees_out
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
export GBM_BENCH_DATA=/mnt/mojolearn-data/gbm-bench
mark() { echo "$1 $(date -u +%T)" | tee -a $OUT/ab.txt; : > "$OUT/phase.$1"; }
while [ ! -f $OUT/taxi.ok ]; do sleep 5; done
while [ ! -f $OUT/track_import.done ] || [ ! -f $OUT/track_gpu_opponents.done ]; do sleep 10; done
echo "batchT start $(date -u +%T)"
grep -q '^import_identical=0' $OUT/setup.txt || { echo "import_identical failed"; exit 3; }

# ---------- PHASE_TS: ours only, beside the Istella-S decode.
echo "load $(cat /proc/loadavg) $(date -u +%T)" >> $OUT/ab.txt
MOJOLEARN_SPEED_OURS_AB=use_pointwise_searcher=True MOJOLEARN_SPEED_TAG=pointwise \
    $AB speed baseline gbdt-symmetric taxi 1000000 5 ours
$AB speed baseline gbdt-symmetric taxi 1000000 1 stage
$AB speed baseline gbdt-depthwise taxi 1000000 1 stage
$AB speed baseline gbdt-lossguide taxi 1000000 1 stage
mark PHASE_TS_DONE

# ---------- PHASE_T: the opponent cells on a quiet CPU.
while [ ! -f $OUT/setup.done ]; do sleep 10; done
echo "load $(cat /proc/loadavg) $(date -u +%T)" >> $OUT/ab.txt
XGB=xgboost-cpu; LGB=lightgbm-cpu; PY=python3
grep -q '^xgb_rocm_works=yes' $OUT/setup.txt && { XGB=xgboost-gpu; PY=/root/venv-gpu/bin/python; }
grep -q '^lightgbm_opencl_works=yes' $OUT/setup.txt && { LGB=lightgbm-opencl; PY=/root/venv-gpu/bin/python; }
export MOJOLEARN_SPEED_PY=$PY MOJOLEARN_SPEED_DEVICES=cpu,gpu,opencl MOJOLEARN_SPEED_OPPONENTS_FIRST=1
echo "arms: xgboost=$XGB lightgbm=$LGB python=$PY" | tee -a $OUT/ab.txt
$PY -c "import catboost, sklearn, xgboost, lightgbm, numpy, sys; print('python', sys.version.split()[0]); print('catboost', catboost.__version__); print('sklearn', sklearn.__version__); print('xgboost', xgboost.__version__, xgboost.__file__); print('lightgbm', lightgbm.__version__, lightgbm.__file__); print('numpy', numpy.__version__)" > $OUT/versions_used.txt 2>&1
full() {  # <lane> <dataset> <rows> <arms>
    MOJOLEARN_SPEED_ARMS="$4" $AB speed baseline "$1" "$2" "$3" 5 full
}
full gbdt-symmetric taxi 1000000 catboost-cpu
full rf taxi 1000000 sklearn-rf-cpu
full gbdt-depthwise taxi 1000000 "catboost-cpu,$XGB"
full gbdt-lossguide taxi 1000000 "catboost-cpu,$XGB,$LGB"
full et taxi 1000000 sklearn-et-cpu
mark PHASE_T1_DONE
full rf taxi 2000000 sklearn-rf-cpu
mark PHASE_T2_DONE
full gbdt-symmetric istella 1000000 catboost-cpu
full gbdt-depthwise istella 1000000 "catboost-cpu,$XGB"
full gbdt-lossguide istella 1000000 "catboost-cpu,$XGB,$LGB"
mark PHASE_I_DONE
echo "batchT end $(date -u +%T)"
