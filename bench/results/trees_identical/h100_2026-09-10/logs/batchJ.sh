#!/bin/sh
# Confirmation of the rf2011 flip (HIST_ITEMS_PER_THREAD default 4): the
# flipped SOURCE (no define) must reproduce rf2011's fingerprints and timing.
# A baseline 1M re-run follows as the drift control for the sequential cells.
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
OUT=/root/trees_out
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
while [ ! -f $OUT/batchI.done ]; do sleep 20; done
echo "batchJ start $(date -u +%T)" | tee -a $OUT/ab.txt
grep -c 'HIST_ITEMS1' ensemble/decisiontree/batched_levelalgo/builder.mojo | tee -a $OUT/ab.txt
$AB build flip2011 rf
$AB ib flip2011 rf-clf,rf-reg
$AB diff baseline flip2011
$AB diff rf2011 flip2011
$AB speed flip2011 rf higgs 1000000 5 ours
$AB speed baseline rf higgs 1000000 5 ours
mv $OUT/speed/baseline.rf.higgs.r1000000.ours.log $OUT/speed/baseline.rf.higgs.r1000000.ours.driftcheck.log 2>/dev/null
$AB rfgate flip2011src
$AB rfgate flip2011optout "-D MOJOLEARN_2011_HIST_ITEMS1=1"
echo "BATCHJ_DONE $(date -u +%T)" | tee -a $OUT/ab.txt
: > $OUT/batchJ.done
