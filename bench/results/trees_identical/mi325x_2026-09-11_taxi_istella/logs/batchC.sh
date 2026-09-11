#!/bin/sh
# MI325X leg 2, 2026-09-11 (lane/amd-trees-leg), a fresh droplet on the same
# tor1 volume (taxi decoded there in leg 1; the Istella-S tar or cache too).
# IDENTICAL tier, default build. Priority order:
#  (1) Istella-S opponent cells, interleaved round by round, 1 warm-up + 5
#      rounds, opponents imported before our binding: symmetric vs CatBoost
#      CPU; depthwise and lossguide vs CatBoost CPU and XGBoost (AMD ROCm
#      build on the GPU when the setup probe passes, else CPU) and, for
#      lossguide, LightGBM CPU (PyPI wheel); RF 1M, ET 1M, RF 2M vs
#      scikit-learn CPU.
#  (2) ours only: the symmetric use_pointwise_searcher A/B and the stage
#      ledgers (DEVIATION 2510) for symmetric, depthwise and lossguide; then
#      the pointwise flip verdict against leg 1's taxi A/B log (pushed to
#      $OUT/leg1/).
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
OUT=/root/trees_out
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
export GBM_BENCH_DATA=/mnt/mojolearn-data/gbm-bench
mark() { echo "$1 $(date -u +%T)" | tee -a $OUT/ab.txt; : > "$OUT/phase.$1"; }
while [ ! -f $OUT/setup.done ]; do sleep 10; done
echo "batchC start $(date -u +%T)"; cat $OUT/setup.txt
grep -q '^import_identical=0' $OUT/setup.txt || { echo "import_identical failed"; exit 3; }
[ -s $GBM_BENCH_DATA/istella/istella_speed.npz ] || { echo "no Istella-S cache on the volume; refusing (the loader would fall back to a synthetic fixture)"; exit 4; }
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

# ---------- (1) Istella-S opponent cells.
full gbdt-symmetric istella 1000000 catboost-cpu
full gbdt-depthwise istella 1000000 "catboost-cpu,$XGB"
full gbdt-lossguide istella 1000000 "catboost-cpu,$XGB,$LGB"
mark PHASE_IG_DONE
full rf istella 1000000 sklearn-rf-cpu
full et istella 1000000 sklearn-et-cpu
mark PHASE_IF1_DONE
full rf istella 2000000 sklearn-rf-cpu
mark PHASE_IF2_DONE

# ---------- (2) ours only on Istella-S.
unset MOJOLEARN_SPEED_OPPONENTS_FIRST
MOJOLEARN_SPEED_OURS_AB=use_pointwise_searcher=True MOJOLEARN_SPEED_TAG=pointwise \
    $AB speed baseline gbdt-symmetric istella 1000000 5 ours
$AB speed baseline gbdt-symmetric istella 1000000 1 stage
$AB speed baseline gbdt-depthwise istella 1000000 1 stage
$AB speed baseline gbdt-lossguide istella 1000000 1 stage
mark PHASE_IS_DONE

for ds in taxi istella; do
    f=$OUT/speed/baseline.gbdt-symmetric.$ds.r1000000.ours.pointwise.log
    [ -f "$f" ] || f=$OUT/leg1/baseline.gbdt-symmetric.$ds.r1000000.ours.pointwise.log
    [ -f "$f" ] || continue
    grep -v 'arm=ours-ab' "$f" > $OUT/speed/ab_pointwise.$ds.before.log
    grep -v 'arm=ours ' "$f" | sed 's/arm=ours-ab/arm=ours/' > $OUT/speed/ab_pointwise.$ds.after.log
done
python3 tools/flip_verdict.py --lane gbdt-symmetric \
    --taxi-before $OUT/speed/ab_pointwise.taxi.before.log --taxi-after $OUT/speed/ab_pointwise.taxi.after.log \
    --istella-before $OUT/speed/ab_pointwise.istella.before.log --istella-after $OUT/speed/ab_pointwise.istella.after.log \
    > $OUT/speed/flip_verdict.pointwise.txt 2>&1
echo "flip_verdict_pointwise=$? $(date -u +%T)" | tee -a $OUT/ab.txt
cat $OUT/speed/flip_verdict.pointwise.txt
python3 bench/results/trees_identical/mi325x_2026-09-11_taxi_istella/logs/summarize_stage.py $OUT/speed/*.stage.log $OUT/leg1/*.stage.log > $OUT/speed/STAGES.md 2>&1
echo "batchC end $(date -u +%T)"
