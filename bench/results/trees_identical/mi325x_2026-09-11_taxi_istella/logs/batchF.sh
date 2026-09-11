#!/bin/sh
# MI325X FAST-tier leg (ENGINEERING_RULES.md section 10: the tree FAST tier
# is tuned on AMD). FAST builds of gbdt, rf and trees (the base binding ships
# IDENTICAL only, DEVIATION 2490, and its build refuses fast by name; the
# fast tree tiers resolve the shared host helpers from it). Then ours only,
# 1M rows, taxi and Istella-S, every lane: arm `ours` = IDENTICAL and arm
# `ours-ab` = the same estimator with numeric_mode='fast', interleaved round
# by round in one process, 1 warm-up + 5 rounds, accuracy and prediction
# hash per arm. Opponent rows are the ones this leg's IDENTICAL cells measured.
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
OUT=/root/trees_out
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
if mountpoint -q /mnt/mojolearn-data; then export GBM_BENCH_DATA=/mnt/mojolearn-data/gbm-bench; fi
mark() { echo "$1 $(date -u +%T)" | tee -a $OUT/ab.txt; : > "$OUT/phase.$1"; }
while [ ! -f $OUT/track_mojo.done ] || [ ! -f $OUT/track_import.done ] || [ ! -f $OUT/track_pip.done ]; do sleep 10; done
echo "batchF start $(date -u +%T)"
grep -q '^import_identical=0' $OUT/setup.txt || { echo "import_identical failed"; exit 3; }

# ---------- PHASE_FB: FAST builds (python/mojolearn/*.so, beside identical/).
MOJOLEARN_NUMERIC_MODE=fast bash bindings/build.sh > $OUT/logs/build.fast.base.log 2>&1
echo "build_fast_base=$? (expected 2: identical-only binding) $(date -u +%T)" | tee -a $OUT/ab.txt
for b in gbdt rf trees; do
    MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=8 \
        timeout -k 30 1500 bash bindings/build_$b.sh > $OUT/logs/build.fast.$b.log 2>&1
    echo "build_fast_$b=$? $(date -u +%T)" | tee -a $OUT/ab.txt
done
ls -la python/mojolearn/*.so python/mojolearn/identical/*.so > $OUT/fast_bindings_listing.txt 2>&1
sha256sum python/mojolearn/*.so python/mojolearn/identical/*.so > $OUT/fast_bins_sha256.txt 2>&1
( cd python && MOJOLEARN_NUMERIC_MODE=identical python3 -c "
import mojolearn
for cls in (mojolearn.GradientBoosting, mojolearn.RandomForestClassifier, mojolearn.ExtraTreesClassifier):
    m = cls(numeric_mode='fast')
    print(cls.__name__, 'fast ->', m.numeric_mode_used(), m.vendor_used())
" ) > $OUT/logs/import_fast.log 2>&1
echo "import_fast=$? $(date -u +%T)" | tee -a $OUT/ab.txt
mark PHASE_FB_DONE

fastab() {  # <lane> <dataset>
    MOJOLEARN_SPEED_OURS_AB="numeric_mode='fast'" MOJOLEARN_SPEED_TAG=fastab \
        $AB speed baseline "$1" "$2" 1000000 5 ours
}
for ds in taxi istella; do
    fastab gbdt-symmetric $ds
    fastab gbdt-depthwise $ds
    fastab gbdt-lossguide $ds
    fastab rf $ds
    fastab et $ds
    mark PHASE_FA_${ds}_DONE
done
echo "batchF end $(date -u +%T)"
