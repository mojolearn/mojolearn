#!/bin/sh
# H100 leg 2026-09-10 night (lane/nvidia-identical-trees-0910b): fingerprints
# of main at 7cebeecf against the retained Sep 10 set, then ours-only IDENTICAL
# timing at 1M/2M for rf, gbdt-depthwise, gbdt-lossguide, gbdt-symmetric, et,
# with stage splits early so attribution starts while the rest times.
# No opponent is run. Never below 1M rows.
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
OUT=/root/trees_out
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
while [ ! -f $OUT/setup.done ]; do sleep 15; done
echo "batchK start $(date -u +%T)"; cat $OUT/setup.txt
mark() { echo "$1 $(date -u +%T)" | tee -a $OUT/ab.txt; : > "$OUT/phase.$1"; }

# ---------- PHASE_A: baseline snapshot, fingerprints, diff vs the Sep 10 H100 set
mkdir -p /root/bins/baseline && cp python/mojolearn/identical/*.so /root/bins/baseline/
sha256sum /root/bins/baseline/*.so | tee $OUT/baseline_so_sha256.txt
$AB ib baseline
$AB diff sep10_baseline baseline
$AB diff sep10_flip2011 baseline
mark PHASE_A_DONE

# ---------- PHASE_B: ours-only timing + stage splits (1M first, stage early)
$AB speed baseline rf higgs 1000000 5 ours
$AB speed baseline rf higgs 1000000 1 stage
$AB speed baseline gbdt-depthwise higgs 1000000 1 stage
$AB speed baseline gbdt-lossguide higgs 1000000 1 stage
mark PHASE_B1_DONE
$AB speed baseline gbdt-depthwise higgs 1000000 7 ours
$AB speed baseline gbdt-lossguide higgs 1000000 7 ours
$AB speed baseline rf higgs 2000000 5 ours
$AB speed baseline gbdt-depthwise higgs 2000000 5 ours
$AB speed baseline gbdt-lossguide higgs 2000000 5 ours
$AB speed baseline gbdt-symmetric higgs 1000000 7 ours
$AB speed baseline et higgs 1000000 5 ours
mark PHASE_B_DONE
