#!/bin/sh
# Lane forest-finish, batch C, queued behind batch B on the same pod.
# DEVIATION 2663 trial: the ExtraTrees frontier batch width. Every set is the
# rowmajor set with ONLY _mojolearn_trees.so rebuilt from this source:
#   ctl    no define (4096, the shipped width; the A/B control)
#   stats  -D MOJOLEARN_ET_CYCLE_STATS=1 (prints cycles, nodes, surveys)
#   bw16k  -D MOJOLEARN_ET_DEVICE_BATCH_16384=1
#   bw32k  -D MOJOLEARN_ET_DEVICE_BATCH_32768=1
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
OUT=/root/trees_out
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
mark() { echo "$1 $(date -u +%T)" | tee -a $OUT/ab.txt; : > "$OUT/phase.$1"; }
LANES=rf-clf,rf-reg,et-clf,et-reg,iforest
while [ ! -f $OUT/phase.B2_SPLIT_DONE ]; do sleep 15; done
echo "batchC start $(date -u +%T)"

for set in ctl stats bw16k bw32k; do
  rm -rf /root/bins/$set; mkdir -p /root/bins/$set; cp /root/bins/rowmajor/*.so /root/bins/$set/
done
$AB build ctl trees || { rm -f /root/bins/ctl/_mojolearn_trees.so; echo "CTL BUILD FAILED" | tee -a $OUT/ab.txt; }
$AB build stats trees -D MOJOLEARN_ET_CYCLE_STATS=1 || { rm -f /root/bins/stats/_mojolearn_trees.so; echo "STATS BUILD FAILED" | tee -a $OUT/ab.txt; }
$AB build bw16k trees -D MOJOLEARN_ET_DEVICE_BATCH_16384=1 || { rm -f /root/bins/bw16k/_mojolearn_trees.so; echo "BW16K BUILD FAILED" | tee -a $OUT/ab.txt; }
$AB build bw32k trees -D MOJOLEARN_ET_DEVICE_BATCH_32768=1 || { rm -f /root/bins/bw32k/_mojolearn_trees.so; echo "BW32K BUILD FAILED" | tee -a $OUT/ab.txt; }
mark C1_BUILD_DONE

# ---- identity: fingerprints and diffs against the rowmajor set.
for set in ctl bw16k bw32k; do
  $AB ib $set $LANES
  $AB diff rowmajor $set
done
timeout -k 30 1500 pixi run mojo run -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 extratrees/checks/device_batched_check.mojo \
  > $OUT/logs/device_batched_check.log 2>&1
echo "device_batched_check=$? $(date -u +%T)" | tee -a $OUT/ab.txt
tail -4 $OUT/logs/device_batched_check.log
mark C2_IDENTITY_DONE

# ---- what the level loop does at 4096: one untimed replicate per dataset.
for ds in istella taxi; do
  $AB speed stats et $ds 1000000 1 ours
  grep ET_CYCLE_STATS $OUT/speed/stats.et.$ds.r1000000.ours.log | tail -4
done
mark C3_STATS_DONE

# ---- the A/B: ours-only, two passes in rotated order per dataset.
for ds in taxi istella; do
  for set in ctl bw16k bw32k; do MOJOLEARN_SPEED_TAG=p1 $AB speed $set et $ds 1000000 3 ours; done
  for set in bw32k bw16k ctl; do MOJOLEARN_SPEED_TAG=p2 $AB speed $set et $ds 1000000 3 ours; done
done
mark C4_AB_DONE
echo "batchC end $(date -u +%T)"
