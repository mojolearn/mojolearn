#!/bin/sh
# gbdt-speed lane, DEVIATION 2634 A/B on the H100 (RUNS ON THE POD from
# /root/mojolearn). Set `baseline` is origin/main 36ca51fd; set `a2634` is the
# same tree plus the CTR target prep gate. Identity first (identity_break
# fingerprints of the four GBDT lanes on both sets and their diff, and the
# sub-byte identity check on both), then timing: two .so sets cannot share a
# process, so rounds alternate PROCESSES, baseline then a2634, three rounds,
# our IDENTICAL arm only, 5 timed fits per process, all three grow policies,
# taxi and Istella-S at 1M rows. Waits for the a2634 build and for the
# Istella-S baseline cells so nothing here overlaps a timed opponent cell.
set -u
cd /root/mojolearn || exit 9
OUT=/root/trees_out; AB="sh tools/trees_identical_ab.sh"
export PATH="$HOME/.pixi/bin:$PATH" MOJOLEARN_NUMERIC_MODE=identical
mark() { echo "$1 $(date -u +%T)" | tee -a $OUT/progress.txt; }
while ! grep -q "build_exit a2634.gbdt=" $OUT/ab.txt 2>/dev/null; do sleep 10; done
grep -q "build_exit a2634.gbdt=0" $OUT/ab.txt || { mark ab2634_build_failed; exit 1; }
while ! grep -q "full_done" $OUT/progress.txt; do sleep 10; done
mark ab2634_start
LANES=gbdt-symmetric,gbdt-depthwise,gbdt-lossguide,gbdt-rmse
$AB ib baseline "$LANES"
$AB ib a2634 "$LANES"
$AB diff baseline a2634
mkdir -p $OUT/check
for s in baseline a2634; do
  $AB use $s > /dev/null
  PYTHONPATH=/root/mojolearn/python timeout -k 30 1800 python3 -u checks/gbdt_sub_byte_identity_check.py \
    --json $OUT/check/$s.json > $OUT/check/$s.txt 2>&1
  echo "subbyte_check $s exit=$? $(tail -1 $OUT/check/$s.txt) $(date -u +%T)" | tee -a $OUT/ab.txt
done
mark ab2634_identity_done
for r in 1 2 3; do
  for ds in taxi istella; do
    for lane in gbdt-symmetric gbdt-depthwise gbdt-lossguide; do
      for s in baseline a2634; do
        MOJOLEARN_SPEED_TAG=r$r $AB speed $s $lane $ds 1000000 5 ours
      done
    done
  done
  mark "ab2634_round$r"
done
mark ab2634_done
