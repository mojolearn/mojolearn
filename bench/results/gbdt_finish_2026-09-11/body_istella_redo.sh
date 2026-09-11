#!/bin/sh
# gbdt-finish lane: the ab phase's Istella-S cells, redone (RUNS ON THE POD).
#
# WHY: the lane's Istella-S fetch timed out at 40 minutes with 410 of 472 MB
# on disk, and body_setup.sh wrote `data.done` regardless, so body_ab.sh was
# released and every Istella-S cell of all three rounds ran with no dataset.
# The taxi cells of that phase are unaffected and stand. This redoes the
# Istella-S half only, same sets, same rotation, same tag, once the fetch has
# finished and phase 2 is done, so no two timed runs overlap.
set -u
cd /root/mojolearn || exit 9
OUT=/root/trees_out; AB="sh tools/trees_identical_ab.sh"
export PATH="$HOME/.pixi/bin:$PATH" MOJOLEARN_NUMERIC_MODE=identical
SETS="${1:-baseline a2634 both all}"; ROUNDS="${2:-3}"; TAG="${3:-ab}"
mark() { echo "$1 $(date -u +%T)" | tee -a $OUT/progress.txt; }
# the real dataset, and nothing else timing
while [ ! -f $OUT/istella_real.done ]; do sleep 20; done
while ! grep -qE "^phase2_done|^phase2_skipped|^phase2_build_failed" $OUT/progress.txt; do sleep 20; done
mark istella_redo_start
r=1
while [ $r -le $ROUNDS ]; do
  order=$(echo $SETS | awk -v r=$r '{n=NF; s=""; for(i=0;i<n;i++){k=(i+r-1)%n+1; s=s" "$k}; print s}')
  for lane in gbdt-symmetric gbdt-depthwise gbdt-lossguide; do
    for s in $order; do
      MOJOLEARN_SPEED_TAG=$TAG.r$r $AB speed $s $lane istella 1000000 5 ours
    done
  done
  mark "istella_redo_round$r"
  r=$((r + 1))
done
mark istella_redo_done
