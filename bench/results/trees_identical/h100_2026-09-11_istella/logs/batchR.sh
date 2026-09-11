#!/bin/sh
# Resume of batchQ.sh after the _buffer.py fix (treelite/ctypes argtypes
# clash refused our RF arm in the same process as cuML). Cells (b) and (c)
# of the brief already ran clean and are kept; (a) is re-run in full mode
# because its first pass has only the cuML rows (kept as *.pass0.log).
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
OUT=/root/trees_out
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
mark() { echo "$1 $(date -u +%T)" | tee -a $OUT/ab.txt; : > "$OUT/phase.$1"; }
while pgrep -f "forest_speed_arm.py --lane rf --dataset higgs" > /dev/null; do sleep 5; done
echo "batchR start $(date -u +%T)"
mv $OUT/speed/baseline.rf.istella.r1000000.full.log $OUT/speed/baseline.rf.istella.r1000000.full.pass0.log
$AB speed baseline rf istella 1000000 5 full
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
echo "batchR end $(date -u +%T)"
