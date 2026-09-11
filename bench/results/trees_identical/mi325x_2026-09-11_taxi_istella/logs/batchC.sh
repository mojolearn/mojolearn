#!/bin/sh
# MI325X leg 2, 2026-09-11 (lane/amd-trees-leg), a fresh droplet on the same
# tor1 volume (taxi and Istella-S decoded there in leg 1). IDENTICAL tier,
# default build. Priority order:
#  (1) Istella-S opponent cells, interleaved round by round, 1 warm-up + 5
#      rounds, opponents imported before our binding: symmetric vs CatBoost
#      CPU; depthwise and lossguide vs CatBoost CPU and XGBoost (AMD ROCm
#      build on the GPU when the setup probe passes, else CPU) and, for
#      lossguide, LightGBM CPU; RF 1M, ET 1M, RF 2M vs scikit-learn CPU.
#  (2) ours only: Istella-S stage ledgers (DEVIATION 2510) for symmetric,
#      depthwise and lossguide, and the symmetric use_pointwise_searcher A/B.
#  (3) DEVIATION 2512 off (set no2512: HEAD plus patches/dev2512_off_*.patch)
#      against on, ours only, A B A per lane per dataset.
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
OUT=/root/trees_out
EVD=bench/results/trees_identical/mi325x_2026-09-11_taxi_istella
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
unset MOJOLEARN_SPEED_ARMS MOJOLEARN_SPEED_OPPONENTS_FIRST
MOJOLEARN_SPEED_OURS_AB=use_pointwise_searcher=True MOJOLEARN_SPEED_TAG=pointwise \
    $AB speed baseline gbdt-symmetric istella 1000000 5 ours
$AB speed baseline gbdt-symmetric istella 1000000 1 stage
$AB speed baseline gbdt-depthwise istella 1000000 1 stage
$AB speed baseline gbdt-lossguide istella 1000000 1 stage
mark PHASE_IS_DONE

# ---------- (3) DEVIATION 2512 off vs on.
patch -p1 --forward < $EVD/patches/dev2512_off_rf.patch > $OUT/logs/patch_2512_rf.log 2>&1
echo "patch_2512_rf=$? $(date -u +%T)" | tee -a $OUT/ab.txt
patch -p1 --forward < $EVD/patches/dev2512_off_gbdt.patch > $OUT/logs/patch_2512_gbdt.log 2>&1
echo "patch_2512_gbdt=$? $(date -u +%T)" | tee -a $OUT/ab.txt
$AB build no2512 rf
$AB build no2512 gbdt
patch -p1 -R < $EVD/patches/dev2512_off_gbdt.patch > $OUT/logs/unpatch_2512_gbdt.log 2>&1
patch -p1 -R < $EVD/patches/dev2512_off_rf.patch > $OUT/logs/unpatch_2512_rf.log 2>&1
sha256sum /root/bins/baseline/*.so /root/bins/no2512/*.so | tee $OUT/bins_sha256.txt
mark PHASE_Z0_DONE
ab3() {  # <lane> <dataset>: baseline, no2512, baseline again
    $AB speed baseline "$1" "$2" 1000000 5 ours
    $AB speed no2512 "$1" "$2" 1000000 5 ours
    MOJOLEARN_SPEED_TAG=pass2 $AB speed baseline "$1" "$2" 1000000 5 ours
}
ab3 rf taxi
ab3 gbdt-symmetric taxi
mark PHASE_Z1_DONE
ab3 rf istella
ab3 gbdt-symmetric istella
mark PHASE_Z2_DONE
for L in rf gbdt-symmetric; do
    python3 tools/flip_verdict.py --lane $L --arm ours \
        --taxi-before $OUT/speed/no2512.$L.taxi.r1000000.ours.log \
        --taxi-after $OUT/speed/baseline.$L.taxi.r1000000.ours.log $OUT/speed/baseline.$L.taxi.r1000000.ours.pass2.log \
        --istella-before $OUT/speed/no2512.$L.istella.r1000000.ours.log \
        --istella-after $OUT/speed/baseline.$L.istella.r1000000.ours.log $OUT/speed/baseline.$L.istella.r1000000.ours.pass2.log \
        > $OUT/speed/flip_verdict.2512.$L.txt 2>&1
    echo "flip_verdict_2512_$L=$? $(date -u +%T)" | tee -a $OUT/ab.txt
done
echo "batchC end $(date -u +%T)"
