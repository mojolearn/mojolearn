#!/bin/sh
# gap work: RF Python-residual profile on the baseline set, then the stamped
# (DEVIATION 2510) rf binding into set `stamps` and its stage run.
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
OUT=/root/trees_out
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SPEED_SIZE=shipped
$AB use baseline
echo "profile start $(date -u +%T)"
PYTHONPATH=/root/mojolearn/python timeout -k 30 900 python3 -u /root/rf_residual_profile.py 1000000 3 > $OUT/logs/rf_residual_profile.baseline.log 2>&1
echo "profile_exit=$? $(date -u +%T)" | tee -a $OUT/ab.txt
# stamped sources into the checkout (source-only change; baseline .so untouched)
cp /root/stamps_src/bindings/_mojolearn_rf.mojo bindings/_mojolearn_rf.mojo
cp /root/stamps_src/ensemble/randomforest.mojo ensemble/randomforest.mojo
cp /root/stamps_src/ensemble/decisiontree/batched_levelalgo/builder.mojo ensemble/decisiontree/batched_levelalgo/builder.mojo
$AB build stamps rf
$AB speed stamps rf higgs 1000000 1 stage
echo "GAP_L_DONE $(date -u +%T)" | tee -a $OUT/ab.txt
