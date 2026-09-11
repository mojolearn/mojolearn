#!/bin/sh
# gbdt-finish lane, phase 2 (RUNS ON THE POD, after body_ab.sh). Everything
# here is serialized against timing: a compile during a timed cell would
# corrupt it, so the build runs first, then identity, then the A/B, then the
# opponent cells.
#
#   1. set `a2661` = the `all` source plus DEVIATION 2661 compiled in
#      (-D MOJOLEARN_2661_NONSYM_GROUP_WIDTH=1). The swapped
#      greedy_search_helper_depthwise.mojo compiles bit-identically to `all`
#      without that define, so the four built sets stay valid.
#   2. identity_break on a2661, its diff against `all`, the sub-byte check.
#   3. speed A/B `all` vs `a2661`, alternating processes, 3 rounds, three
#      grow policies, taxi and Istella-S at 1M rows, our IDENTICAL arm only.
#   4. the opponent cells for this pod's tuple, our arm from `all`.
set -u
cd /root/mojolearn || exit 9
OUT=/root/trees_out; AB="sh tools/trees_identical_ab.sh"
export PATH="$HOME/.pixi/bin:$PATH" MOJOLEARN_NUMERIC_MODE=identical
ROUNDS="${1:-3}"; OPP_ROUNDS="${2:-5}"
mark() { echo "$1 $(date -u +%T)" | tee -a $OUT/progress.txt; }
while ! grep -qE "^ab_done|REFUSED|_build_failed_|_missing_set_" $OUT/progress.txt 2>/dev/null; do sleep 15; done
grep -q "^ab_done" $OUT/progress.txt || { mark phase2_skipped_ab_not_done; exit 1; }

mark phase2_build_start
$AB build a2661 gbdt -D MOJOLEARN_2661_NONSYM_GROUP_WIDTH=1
if ! grep -q "^build_exit a2661.gbdt=0 " $OUT/ab.txt; then
  mark phase2_build_failed_a2661
else
  mark phase2_identity_start
  LANES=gbdt-symmetric,gbdt-depthwise,gbdt-lossguide,gbdt-rmse
  $AB ib a2661 "$LANES"
  $AB diff all a2661
  $AB use a2661 > /dev/null
  PYTHONPATH=/root/mojolearn/python timeout -k 30 1800 python3 -u checks/gbdt_sub_byte_identity_check.py \
    --json $OUT/check/a2661.json > $OUT/check/a2661.txt 2>&1
  echo "subbyte_check a2661 exit=$? $(tail -1 $OUT/check/a2661.txt) $(date -u +%T)" | tee -a $OUT/ab.txt
  mark phase2_identity_done
  r=1
  while [ $r -le $ROUNDS ]; do
    for ds in taxi istella; do
      for lane in gbdt-symmetric gbdt-depthwise gbdt-lossguide; do
        if [ $((r % 2)) -eq 1 ]; then order="all a2661"; else order="a2661 all"; fi
        for s in $order; do
          MOJOLEARN_SPEED_TAG=p2.r$r $AB speed $s $lane $ds 1000000 5 ours
        done
      done
    done
    mark "phase2_round$r"
    r=$((r + 1))
  done
  mark phase2_ab_done
fi

# the opponent rows for this tuple, our arm from the shipped-default set
mark opponents_start
for ds in taxi istella; do
  MOJOLEARN_SPEED_ARMS=catboost-gpu $AB speed all gbdt-symmetric $ds 1000000 $OPP_ROUNDS full
  MOJOLEARN_SPEED_ARMS=catboost-gpu,xgboost-gpu $AB speed all gbdt-depthwise $ds 1000000 $OPP_ROUNDS full
  MOJOLEARN_SPEED_ARMS=catboost-gpu,xgboost-gpu $AB speed all gbdt-lossguide $ds 1000000 $OPP_ROUNDS full
done
mark opponents_done
mark phase2_done
