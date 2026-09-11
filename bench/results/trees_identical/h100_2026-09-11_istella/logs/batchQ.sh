#!/bin/sh
# H100 leg 2026-09-11 istella (lane/nvidia-istella-0911): main at a6d25306,
# IDENTICAL tier only, never below 1M rows. Second dataset kind (Istella-S,
# 3.4M x 220) beside HIGGS; the istella opponents run ONCE here (mode full);
# HIGGS opponents are never re-run (mode ours). DEVIATION 2502 default (OFF)
# vs opt-in (-D MOJOLEARN_2502_PURE_LEAF=1) on both datasets.
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
OUT=/root/trees_out
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
# Builds and fingerprints need only the mojo track; the istella download is slow.
while [ ! -f $OUT/track_mojo.done ]; do sleep 15; done
echo "batchQ start $(date -u +%T)"; cat $OUT/setup.txt
mark() { echo "$1 $(date -u +%T)" | tee -a $OUT/ab.txt; : > "$OUT/phase.$1"; }

# ---------- PHASE_A: baseline set = the four .so the setup built, then the
# brief's explicit rebuild of each binding (sha256 ledger); pureleaf rf.
mkdir -p /root/bins/baseline && cp python/mojolearn/identical/*.so /root/bins/baseline/
sha256sum /root/bins/baseline/*.so | tee $OUT/setup_so_sha256.txt
for b in base gbdt rf trees; do $AB build baseline $b; done
sha256sum /root/bins/baseline/*.so | tee $OUT/baseline_so_sha256.txt
$AB build pureleaf rf -D MOJOLEARN_2502_PURE_LEAF=1
sha256sum /root/bins/pureleaf/*.so | tee $OUT/pureleaf_so_sha256.txt
mark PHASE_A_DONE

# ---------- PHASE_B: fingerprints vs the Sep 10 night H100 set (pre-2502 forest).
$AB ib baseline
$AB diff sep10b_baseline baseline
mark PHASE_B_DONE

# ---------- PHASE_C: timing in the brief's order. Waits for the operator's
# istella decode check (sentinel written from the Mac after step 3).
while [ ! -f $OUT/setup.done ]; do sleep 15; done
cat $OUT/setup.txt
while [ ! -f $OUT/istella.ok ]; do sleep 10; done
$AB speed baseline rf istella 1000000 5 full
$AB speed baseline gbdt-symmetric istella 1000000 7 full
$AB speed baseline rf higgs 1000000 5 ours
$AB speed pureleaf rf istella 1000000 5 ours
$AB speed pureleaf rf higgs 1000000 5 ours
$AB speed baseline gbdt-depthwise istella 1000000 7 full
$AB speed baseline gbdt-lossguide istella 1000000 7 full
$AB speed baseline et istella 1000000 5 ours
mark PHASE_C_DONE
$AB speed baseline rf istella 2000000 5 full
$AB speed pureleaf rf istella 2000000 5 ours
mark PHASE_D_DONE
$AB speed baseline rf istella 1000000 1 stage
$AB speed pureleaf rf istella 1000000 1 stage
mark PHASE_E_DONE
echo "batchQ end $(date -u +%T)"
