#!/bin/sh
# H100 confirmation leg 2026-09-11 (lane/nvidia-identical-trees-0911): main at
# 352d9781 (DEVIATION 2502 pure-node leaf, 2512 device_zero, 2510 stamps) on
# the same GPU model and image as the 2026-09-10 night leg (7cebeecf).
# IDENTICAL tier only, ours alone, no opponent, never below 1M rows.
# Order of the timing cells is the brief's and is interleaved on purpose so
# drift is visible (RF 1M first and last; symmetric 1M twice).
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
OUT=/root/trees_out
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
while [ ! -f $OUT/setup.done ]; do sleep 15; done
echo "batchP start $(date -u +%T)"; cat $OUT/setup.txt
mark() { echo "$1 $(date -u +%T)" | tee -a $OUT/ab.txt; : > "$OUT/phase.$1"; }

# ---------- PHASE_A: baseline set = the four .so the setup built, then the
# brief's explicit rebuild of each binding into the same set (sha256 ledger).
mkdir -p /root/bins/baseline && cp python/mojolearn/identical/*.so /root/bins/baseline/
sha256sum /root/bins/baseline/*.so | tee $OUT/setup_so_sha256.txt
for b in base gbdt rf trees; do $AB build baseline $b; done
sha256sum /root/bins/baseline/*.so | tee $OUT/baseline_so_sha256.txt
mark PHASE_A_DONE

# ---------- PHASE_B: fingerprints, nine lanes; diff vs Sep 10 night H100 and
# vs the Apple M4 rf-2502 set.
$AB ib baseline
$AB diff sep10b_baseline baseline
$AB diff apple_rf2502 baseline
mark PHASE_B_DONE

# ---------- PHASE_C: timing, ours only, in the brief's order.
$AB speed baseline rf higgs 1000000 5 ours
$AB speed baseline gbdt-symmetric higgs 1000000 7 ours
$AB speed baseline rf higgs 2000000 5 ours
cp $OUT/speed/baseline.gbdt-symmetric.higgs.r1000000.ours.log $OUT/speed/baseline.gbdt-symmetric.higgs.r1000000.ours.pass1.log
$AB speed baseline gbdt-symmetric higgs 1000000 7 ours
mv $OUT/speed/baseline.gbdt-symmetric.higgs.r1000000.ours.log $OUT/speed/baseline.gbdt-symmetric.higgs.r1000000.ours.pass2.log
$AB speed baseline gbdt-depthwise higgs 1000000 7 ours
$AB speed baseline gbdt-lossguide higgs 1000000 7 ours
$AB speed baseline et higgs 1000000 5 ours
cp $OUT/speed/baseline.rf.higgs.r1000000.ours.log $OUT/speed/baseline.rf.higgs.r1000000.ours.pass1.log
$AB speed baseline rf higgs 1000000 5 ours
mv $OUT/speed/baseline.rf.higgs.r1000000.ours.log $OUT/speed/baseline.rf.higgs.r1000000.ours.pass2.log
mark PHASE_C_DONE
$AB speed baseline rf higgs 1000000 1 stage
mark PHASE_D_DONE
echo "batchP end $(date -u +%T)"
