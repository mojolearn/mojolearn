#!/bin/sh
# Leg 1, after batchX. batchX's FAST cells timed arm ours alone: the
# droplet still ran the shipped bench/speed/forest_speed_arm.py (92b4bf9b),
# whose --ours-ab reached the GBDT lanes only and verified every arm against
# the environment's tier, so arm ours-ab (numeric_mode='fast') was REFUSED
# ("mode/vendor mismatch ... resolved=fast"). The harness of aca4c256 was
# pushed at 13:23 UTC; this re-runs the five taxi FAST cells with it.
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
OUT=/root/trees_out
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
export GBM_BENCH_DATA=/mnt/mojolearn-data/gbm-bench
export MOJOLEARN_SPEED_PY=python3
while pgrep -f "[s]h /root/batchX.sh" > /dev/null; do sleep 5; done
echo "batchY start $(date -u +%T) load $(cat /proc/loadavg)" | tee -a $OUT/ab.txt
for L in gbdt-symmetric gbdt-depthwise gbdt-lossguide rf et; do
    MOJOLEARN_SPEED_OURS_AB="numeric_mode='fast'" MOJOLEARN_SPEED_TAG=fastab2 \
        $AB speed baseline $L taxi 1000000 5 ours
done
echo "PHASE_Y_DONE $(date -u +%T)" | tee -a $OUT/ab.txt
