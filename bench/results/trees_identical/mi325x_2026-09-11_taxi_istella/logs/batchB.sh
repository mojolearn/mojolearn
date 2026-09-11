#!/bin/sh
# MI325X, after batchA on the same droplet: ours only, both datasets.
# (a) symmetric stage ledgers (DEVIATION 2510 stamps, MOJOLEARN_STAGE_TIMES=1,
#     one untimed replicate after the warm-up; a split, not a timing),
#     then the depthwise and lossguide ledgers;
# (b) symmetric use_pointwise_searcher=True (arm ours-ab) against the default
#     False (arm ours), interleaved round by round in one process, 1 warm-up
#     + 5 rounds, logloss/AUC and prediction hash per side; the verdict comes
#     from tools/flip_verdict.py on logs split per side.
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
OUT=/root/trees_out
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
if mountpoint -q /mnt/mojolearn-data; then export GBM_BENCH_DATA=/mnt/mojolearn-data/gbm-bench; fi
mark() { echo "$1 $(date -u +%T)" | tee -a $OUT/ab.txt; : > "$OUT/phase.$1"; }
while pgrep -f "[s]h /root/batchA.sh" > /dev/null; do sleep 10; done
echo "batchB start $(date -u +%T)"

# (b) first on taxi, then the ledgers, then (b) on Istella-S.
MOJOLEARN_SPEED_OURS_AB=use_pointwise_searcher=True MOJOLEARN_SPEED_TAG=pointwise \
    $AB speed baseline gbdt-symmetric taxi 1000000 5 ours
$AB speed baseline gbdt-symmetric taxi 1000000 1 stage
$AB speed baseline gbdt-symmetric istella 1000000 1 stage
mark PHASE_B1_DONE
MOJOLEARN_SPEED_OURS_AB=use_pointwise_searcher=True MOJOLEARN_SPEED_TAG=pointwise \
    $AB speed baseline gbdt-symmetric istella 1000000 5 ours
mark PHASE_B2_DONE
$AB speed baseline gbdt-depthwise taxi 1000000 1 stage
$AB speed baseline gbdt-lossguide taxi 1000000 1 stage
$AB speed baseline gbdt-depthwise istella 1000000 1 stage
$AB speed baseline gbdt-lossguide istella 1000000 1 stage
mark PHASE_B3_DONE

# Split the A/B logs per side so flip_verdict reads arm=ours on both.
for ds in taxi istella; do
    f=$OUT/speed/baseline.gbdt-symmetric.$ds.r1000000.ours.pointwise.log
    [ -f "$f" ] || continue
    grep -v 'arm=ours-ab' "$f" > $OUT/speed/ab_pointwise.$ds.before.log
    grep -v 'arm=ours ' "$f" | grep -v 'arm=ours$' | sed 's/arm=ours-ab/arm=ours/' > $OUT/speed/ab_pointwise.$ds.after.log
done
python3 tools/flip_verdict.py --lane gbdt-symmetric \
    --taxi-before $OUT/speed/ab_pointwise.taxi.before.log --taxi-after $OUT/speed/ab_pointwise.taxi.after.log \
    --istella-before $OUT/speed/ab_pointwise.istella.before.log --istella-after $OUT/speed/ab_pointwise.istella.after.log \
    > $OUT/speed/flip_verdict.pointwise.txt 2>&1
echo "flip_verdict_exit=$? $(date -u +%T)" | tee -a $OUT/ab.txt
python3 bench/results/trees_identical/mi325x_2026-09-11_taxi_istella/logs/summarize_stage.py $OUT/speed/*.stage.log > $OUT/speed/STAGES.md 2>&1
echo "batchB end $(date -u +%T)"
