#!/bin/sh
# gbdt-finish lane: the opponent rows for this pod's tuple (RUNS ON THE POD,
# after body_ab.sh). A new GPU model and driver is a new tuple under
# ENGINEERING_RULES 9, so CatBoost GPU (all three policies) and XGBoost GPU
# (depthwise, lossguide) are measured once here, interleaved in one process
# with our IDENTICAL arm from set $1 (default all), taxi and Istella-S at 1M
# rows. Waits for the A/B so no opponent cell overlaps a timed A/B cell.
set -u
cd /root/mojolearn || exit 9
OUT=/root/trees_out; AB="sh tools/trees_identical_ab.sh"
export PATH="$HOME/.pixi/bin:$PATH" MOJOLEARN_NUMERIC_MODE=identical
SET="${1:-all}"; ROUNDS="${2:-5}"
mark() { echo "$1 $(date -u +%T)" | tee -a $OUT/progress.txt; }
while ! grep -qE "^ab_done|REFUSED|_build_failed_|_missing_set_" $OUT/progress.txt 2>/dev/null; do sleep 15; done
# the A/B refused (driver, or a set whose own build failed): time nothing
if ! grep -q "^ab_done" $OUT/progress.txt; then mark opponents_skipped_ab_not_done; exit 1; fi
# the opponent cells take our arm from $SET, so $SET's own build must have run
if [ "$SET" != baseline ] && ! grep -q "^build_exit $SET.gbdt=0 " $OUT/ab.txt; then
  mark "opponents_build_failed_$SET"; exit 1
fi
mark opponents_start
for ds in taxi istella; do
  MOJOLEARN_SPEED_ARMS=catboost-gpu $AB speed $SET gbdt-symmetric $ds 1000000 $ROUNDS full
  MOJOLEARN_SPEED_ARMS=catboost-gpu,xgboost-gpu $AB speed $SET gbdt-depthwise $ds 1000000 $ROUNDS full
  MOJOLEARN_SPEED_ARMS=catboost-gpu,xgboost-gpu $AB speed $SET gbdt-lossguide $ds 1000000 $ROUNDS full
done
mark opponents_done
