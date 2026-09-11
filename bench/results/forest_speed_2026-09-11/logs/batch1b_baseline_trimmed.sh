#!/bin/sh
# Lane forest-speed, H100 leg 2026-09-11 night, batch 1b: batch 1 trimmed to
# the same-pod speed cells at 3 rounds on the wind-down order (host splits and
# stage replicates dropped). Baseline set = source 36ca51fd as built by setup
# plus the svm extension; harness edits of lane/forest-speed. Taxi cells first
# (already on disk), Istella-S after the setup sentinel.
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
OUT=/root/trees_out
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
mark() { echo "$1 $(date -u +%T)" | tee -a $OUT/ab.txt; : > "$OUT/phase.$1"; }
echo "batch1b start $(date -u +%T)"
$AB speed baseline rf taxi 1000000 3 full
$AB speed baseline iforest taxi 1000000 3 full
MOJOLEARN_SPEED_DEVICES=cpu MOJOLEARN_SPEED_ARMS=sklearn-et-cpu $AB speed baseline et taxi 1000000 3 full
mark B1B_TAXI_DONE
while [ ! -f $OUT/setup.done ]; do sleep 5; done
$AB speed baseline rf istella 1000000 3 full
$AB speed baseline iforest istella 1000000 3 full
MOJOLEARN_SPEED_DEVICES=cpu MOJOLEARN_SPEED_ARMS=sklearn-et-cpu $AB speed baseline et istella 1000000 3 full
mark B1B_ISTELLA_DONE
echo "batch1b end $(date -u +%T)"
