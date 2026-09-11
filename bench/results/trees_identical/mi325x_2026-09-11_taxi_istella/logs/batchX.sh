#!/bin/sh
# Leg 1, after batchT's taxi opponent cells (PHASE_T2_DONE 13:15:36), while
# the Istella-S tar re-downloads (the byte-range resume got HTTP 504; batchI
# decodes it when it lands). Taxi only:
#  (1) LightGBM CPU row. The venv's lightgbm was the USE_GPU source build
#      (its OpenCL probe failed with "Check failed: best_split_info.left_count
#      > 0", and its CPU learner refused the taxi lossguide warm-up with the
#      same check); it is uninstalled so the venv sees the PyPI wheel, and
#      the lossguide cell re-runs with CatBoost CPU, XGBoost ROCm GPU and
#      LightGBM CPU.
#  (2) FAST tier beside IDENTICAL (ENGINEERING_RULES.md section 10): FAST
#      builds of gbdt, rf and trees (base ships IDENTICAL only, DEVIATION
#      2490), then arm ours (IDENTICAL) and arm ours-ab (numeric_mode='fast')
#      interleaved round by round, 1 warm-up + 5 rounds, all five lanes.
#  (3) DEVIATION 2512 off (set no2512 = HEAD plus patches/dev2512_off_*.patch)
#      against on, ours only, baseline / no2512 / baseline per lane, RF and
#      symmetric.
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
OUT=/root/trees_out
EVD=bench/results/trees_identical/mi325x_2026-09-11_taxi_istella
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
export GBM_BENCH_DATA=/mnt/mojolearn-data/gbm-bench
mark() { echo "$1 $(date -u +%T)" | tee -a $OUT/ab.txt; : > "$OUT/phase.$1"; }
echo "batchX start $(date -u +%T) load $(cat /proc/loadavg)" | tee -a $OUT/ab.txt

# ---------- (1) LightGBM CPU from the PyPI wheel.
/root/venv-gpu/bin/python -m pip uninstall -y lightgbm > $OUT/logs/venv_uninstall_lightgbm.log 2>&1
/root/venv-gpu/bin/python -c "import lightgbm, xgboost, catboost; print('after uninstall: lightgbm', lightgbm.__version__, lightgbm.__file__, 'xgboost', xgboost.__version__, 'catboost', catboost.__version__)" >> $OUT/versions_used.txt 2>&1
tail -1 $OUT/versions_used.txt
MOJOLEARN_SPEED_PY=/root/venv-gpu/bin/python MOJOLEARN_SPEED_DEVICES=cpu,gpu,opencl MOJOLEARN_SPEED_OPPONENTS_FIRST=1 \
MOJOLEARN_SPEED_ARMS=catboost-cpu,xgboost-gpu,lightgbm-cpu MOJOLEARN_SPEED_TAG=lgbmwheel \
    $AB speed baseline gbdt-lossguide taxi 1000000 5 full
mark PHASE_X1_DONE

# ---------- (2) FAST builds, then FAST beside IDENTICAL on taxi.
for b in gbdt rf trees; do
    MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=12 \
        timeout -k 30 1200 bash bindings/build_$b.sh > $OUT/logs/build.fast.$b.log 2>&1
    echo "build_fast_$b=$? $(date -u +%T)" | tee -a $OUT/ab.txt
done
MOJOLEARN_NUMERIC_MODE=fast timeout -k 10 120 bash bindings/build.sh > $OUT/logs/build.fast.base.log 2>&1
echo "build_fast_base=$? (the base binding refuses fast by name, DEVIATION 2490) $(date -u +%T)" | tee -a $OUT/ab.txt
sha256sum python/mojolearn/*.so python/mojolearn/identical/*.so > $OUT/fast_bins_sha256.txt 2>&1
( cd python && python3 -c "
import mojolearn
for cls in (mojolearn.GradientBoosting, mojolearn.RandomForestClassifier, mojolearn.ExtraTreesClassifier):
    m = cls(numeric_mode='fast')
    print(cls.__name__, 'fast ->', m.numeric_mode_used(), m.vendor_used())
" ) > $OUT/logs/import_fast.log 2>&1
echo "import_fast=$? $(date -u +%T)" | tee -a $OUT/ab.txt
export MOJOLEARN_SPEED_PY=python3
for L in gbdt-symmetric gbdt-depthwise gbdt-lossguide rf et; do
    MOJOLEARN_SPEED_OURS_AB="numeric_mode='fast'" MOJOLEARN_SPEED_TAG=fastab \
        $AB speed baseline $L taxi 1000000 5 ours
done
mark PHASE_X2_DONE

# ---------- (3) DEVIATION 2512 off vs on, taxi.
patch -p1 --forward < $EVD/patches/dev2512_off_rf.patch > $OUT/logs/patch_2512_rf.log 2>&1
echo "patch_2512_rf=$? $(date -u +%T)" | tee -a $OUT/ab.txt
patch -p1 --forward < $EVD/patches/dev2512_off_gbdt.patch > $OUT/logs/patch_2512_gbdt.log 2>&1
echo "patch_2512_gbdt=$? $(date -u +%T)" | tee -a $OUT/ab.txt
$AB build no2512 rf
$AB build no2512 gbdt
patch -p1 -R < $EVD/patches/dev2512_off_gbdt.patch > $OUT/logs/unpatch_2512_gbdt.log 2>&1
patch -p1 -R < $EVD/patches/dev2512_off_rf.patch > $OUT/logs/unpatch_2512_rf.log 2>&1
echo "unpatch_2512 rf=$(grep -c FAILED $OUT/logs/unpatch_2512_rf.log) gbdt=$(grep -c FAILED $OUT/logs/unpatch_2512_gbdt.log) $(date -u +%T)" | tee -a $OUT/ab.txt
sha256sum /root/bins/baseline/*.so /root/bins/no2512/*.so > $OUT/bins_sha256.txt 2>&1
for L in rf gbdt-symmetric; do
    $AB speed baseline $L taxi 1000000 5 ours
    $AB speed no2512 $L taxi 1000000 5 ours
    MOJOLEARN_SPEED_TAG=pass2 $AB speed baseline $L taxi 1000000 5 ours
done
mark PHASE_X3_DONE
echo "batchX end $(date -u +%T)"
