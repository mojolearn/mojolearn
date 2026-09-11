#!/bin/sh
# Remaining baseline cells (old Python), then the DEVIATION 2511 A/B.
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
OUT=/root/trees_out
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
mark() { echo "$1 $(date -u +%T)" | tee -a $OUT/ab.txt; : > "$OUT/phase.$1"; }
mkdir -p /root/bins/baseline2 /root/bins/exp2511
cp /root/bins/baseline/*.so /root/bins/baseline2/; cp /root/bins/baseline/*.so /root/bins/exp2511/
# ---------- PHASE_B2: baseline, old Python
$AB speed baseline gbdt-lossguide higgs 1000000 7 ours
$AB speed baseline gbdt-depthwise higgs 2000000 5 ours
$AB speed baseline gbdt-lossguide higgs 2000000 5 ours
$AB speed baseline gbdt-symmetric higgs 1000000 7 ours
$AB speed baseline et higgs 1000000 5 ours
$AB speed baseline rf higgs 2000000 5 ours
$AB speed baseline2 rf higgs 1000000 5 ours
mark PHASE_B_DONE
# ---------- PHASE_D: DEVIATION 2511 (Python-only: raw-malloc export destinations)
cp /root/exp2511/_forest_protocol.py python/mojolearn/_forest_protocol.py
$AB ib exp2511 rf-clf,rf-reg,et-clf,et-reg
$AB diff baseline exp2511
$AB speed exp2511 rf higgs 1000000 5 ours
$AB speed exp2511 rf higgs 2000000 5 ours
$AB use exp2511
MOJOLEARN_SPEED_SIZE=shipped PYTHONPATH=/root/mojolearn/python timeout 600 python3 -u /root/rf_export_profile.py > $OUT/logs/rf_export_profile.exp2511.log 2>&1
echo "export_profile_exit=$? $(date -u +%T)" | tee -a $OUT/ab.txt
mark PHASE_D_DONE
