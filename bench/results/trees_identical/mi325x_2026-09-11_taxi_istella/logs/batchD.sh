#!/bin/sh
# MI325X leg 3, 2026-09-11 (lane/amd-trees-leg): ours only, Istella-S, the
# halves whose taxi side ran in leg 1 (batchX); leg 1's taxi logs are pushed
# to $OUT/leg1/ for the verdicts.
#  (1) FAST tier beside IDENTICAL: FAST builds of gbdt, rf and trees, then
#      arm ours (IDENTICAL) and arm ours-ab (numeric_mode='fast') interleaved
#      round by round, 1 warm-up + 5 rounds, all five lanes.
#  (2) DEVIATION 2512 off (set no2512 = HEAD plus patches/dev2512_off_*.patch)
#      against on, baseline / no2512 / baseline, RF and symmetric; then
#      tools/flip_verdict.py per lane over both datasets.
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
OUT=/root/trees_out
EVD=bench/results/trees_identical/mi325x_2026-09-11_taxi_istella
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
export GBM_BENCH_DATA=/mnt/mojolearn-data/gbm-bench
export MOJOLEARN_SPEED_PY=python3
mark() { echo "$1 $(date -u +%T)" | tee -a $OUT/ab.txt; : > "$OUT/phase.$1"; }
while [ ! -f $OUT/track_mojo.done ] || [ ! -f $OUT/track_import.done ]; do sleep 10; done
# In the same lease as batchC (leg 2), it runs after batchC; the hold cuts what does not fit.
sleep 20; while pgrep -f "[s]h /root/batchC.sh" > /dev/null; do sleep 10; done
echo "batchD start $(date -u +%T) load $(cat /proc/loadavg)" | tee -a $OUT/ab.txt
grep -q '^import_identical=0' $OUT/setup.txt || { echo "import_identical failed"; exit 3; }
[ -s $GBM_BENCH_DATA/istella/istella_speed.npz ] || { echo "no Istella-S cache on the volume; refusing"; exit 4; }

# ---------- (1) FAST beside IDENTICAL, Istella-S.
for b in gbdt rf trees; do
    MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=12 \
        timeout -k 30 1200 bash bindings/build_$b.sh > $OUT/logs/build.fast.$b.log 2>&1
    echo "build_fast_$b=$? $(date -u +%T)" | tee -a $OUT/ab.txt
done
sha256sum python/mojolearn/*.so python/mojolearn/identical/*.so > $OUT/fast_bins_sha256.txt 2>&1
for L in gbdt-symmetric gbdt-depthwise gbdt-lossguide rf et; do
    MOJOLEARN_SPEED_OURS_AB="numeric_mode='fast'" MOJOLEARN_SPEED_TAG=fastab \
        $AB speed baseline $L istella 1000000 5 ours
done
mark PHASE_D1_DONE

# ---------- (2) DEVIATION 2512 off vs on, Istella-S, then the verdicts.
patch -p1 --forward < $EVD/patches/dev2512_off_rf.patch > $OUT/logs/patch_2512_rf.log 2>&1
echo "patch_2512_rf=$? $(date -u +%T)" | tee -a $OUT/ab.txt
patch -p1 --forward < $EVD/patches/dev2512_off_gbdt.patch > $OUT/logs/patch_2512_gbdt.log 2>&1
echo "patch_2512_gbdt=$? $(date -u +%T)" | tee -a $OUT/ab.txt
$AB build no2512 rf
$AB build no2512 gbdt
patch -p1 -R < $EVD/patches/dev2512_off_gbdt.patch > $OUT/logs/unpatch_2512_gbdt.log 2>&1
patch -p1 -R < $EVD/patches/dev2512_off_rf.patch > $OUT/logs/unpatch_2512_rf.log 2>&1
sha256sum /root/bins/baseline/*.so /root/bins/no2512/*.so > $OUT/bins_sha256.txt 2>&1
for L in rf gbdt-symmetric; do
    $AB speed baseline $L istella 1000000 5 ours
    $AB speed no2512 $L istella 1000000 5 ours
    MOJOLEARN_SPEED_TAG=pass2 $AB speed baseline $L istella 1000000 5 ours
done
mark PHASE_D2_DONE
S=$OUT/speed; P=$OUT/leg1
for L in rf gbdt-symmetric; do
    python3 tools/flip_verdict.py --lane $L --arm ours \
        --taxi-before $P/no2512.$L.taxi.r1000000.ours.log \
        --taxi-after $P/baseline.$L.taxi.r1000000.ours.log $P/baseline.$L.taxi.r1000000.ours.pass2.log \
        --istella-before $S/no2512.$L.istella.r1000000.ours.log \
        --istella-after $S/baseline.$L.istella.r1000000.ours.log $S/baseline.$L.istella.r1000000.ours.pass2.log \
        > $S/flip_verdict.2512.$L.txt 2>&1
    echo "flip_verdict_2512_$L=$? $(date -u +%T)" | tee -a $OUT/ab.txt
    tail -1 $S/flip_verdict.2512.$L.txt
done
echo "batchD end $(date -u +%T)"
