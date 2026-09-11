#!/bin/sh
# MI325X leg 2, 2026-09-11 (lane/amd-trees-leg), a fresh droplet on the same
# tor1 volume (datasets decoded once in leg 1). IDENTICAL tier, default build.
# (1) the Istella-S forest cells against scikit-learn on all CPU cores,
#     interleaved round by round, 1 warm-up + 5 rounds, opponents imported
#     first; (2) DEVIATION 2512 on (baseline) vs off (set no2512, HEAD with
#     patches/dev2512_off_{rf,gbdt}.patch applied, the pre-2512 memsets),
#     ours only, A B A per lane per dataset so drift shows.
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
OUT=/root/trees_out
EVD=bench/results/trees_identical/mi325x_2026-09-11_taxi_istella
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
if mountpoint -q /mnt/mojolearn-data; then export GBM_BENCH_DATA=/mnt/mojolearn-data/gbm-bench; fi
mark() { echo "$1 $(date -u +%T)" | tee -a $OUT/ab.txt; : > "$OUT/phase.$1"; }
while [ ! -f $OUT/track_mojo.done ] || [ ! -f $OUT/track_pip.done ] || [ ! -f $OUT/track_import.done ]; do sleep 15; done
echo "batchC start $(date -u +%T)"; cat $OUT/setup.txt
grep -q '^import_identical=0' $OUT/setup.txt || { echo "import_identical failed"; exit 3; }
export MOJOLEARN_SPEED_PY=python3 MOJOLEARN_SPEED_DEVICES=cpu,gpu,opencl MOJOLEARN_SPEED_OPPONENTS_FIRST=1
python3 -c "import sklearn, numpy, sys; print('python', sys.version.split()[0]); print('sklearn', sklearn.__version__); print('numpy', numpy.__version__)" > $OUT/versions_used.txt 2>&1

full() {  # <lane> <dataset> <rows> <arms>
    MOJOLEARN_SPEED_ARMS="$4" $AB speed baseline "$1" "$2" "$3" 5 full
}

# ---------- PHASE_F: Istella-S forests.
full rf istella 1000000 sklearn-rf-cpu
full et istella 1000000 sklearn-et-cpu
mark PHASE_F1_DONE
full rf istella 2000000 sklearn-rf-cpu
mark PHASE_F2_DONE

# ---------- PHASE_Z: DEVIATION 2512 off, built from HEAD plus the two patches.
patch -p1 --forward < $EVD/patches/dev2512_off_rf.patch > $OUT/logs/patch_2512_rf.log 2>&1
echo "patch_2512_rf=$? $(date -u +%T)" | tee -a $OUT/ab.txt
patch -p1 --forward < $EVD/patches/dev2512_off_gbdt.patch > $OUT/logs/patch_2512_gbdt.log 2>&1
echo "patch_2512_gbdt=$? $(date -u +%T)" | tee -a $OUT/ab.txt
$AB build no2512 rf
$AB build no2512 gbdt
patch -p1 -R < $EVD/patches/dev2512_off_gbdt.patch > $OUT/logs/unpatch_2512_gbdt.log 2>&1
patch -p1 -R < $EVD/patches/dev2512_off_rf.patch > $OUT/logs/unpatch_2512_rf.log 2>&1
echo "unpatch_2512=$(grep -c FAILED $OUT/logs/unpatch_2512_*.log | tr '\n' ' ') $(date -u +%T)" | tee -a $OUT/ab.txt
sha256sum /root/bins/baseline/*.so /root/bins/no2512/*.so | tee $OUT/bins_sha256.txt
mark PHASE_Z0_DONE
ab3() {  # <lane> <dataset>: baseline, no2512, baseline (pass2)
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
