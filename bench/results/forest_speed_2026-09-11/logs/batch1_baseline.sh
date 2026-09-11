#!/bin/sh
# Lane forest-speed, H100 leg 2026-09-11 night, batch 1: the same-pod baseline.
# Source 36ca51fd (main) plus the harness edits of lane/forest-speed (svm
# binding in the A/B helper, iforest proxy labels, tools/forest_host_split.py).
# IDENTICAL tier only, 1M training rows, taxi and Istella-S.
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
OUT=/root/trees_out
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
mark() { echo "$1 $(date -u +%T)" | tee -a $OUT/ab.txt; : > "$OUT/phase.$1"; }
echo "batch1 start $(date -u +%T)"

# ---- the baseline set: the four .so the setup built plus the svm extension.
mkdir -p /root/bins/baseline $OUT/split
cp python/mojolearn/identical/*.so /root/bins/baseline/
$AB build baseline svm
sha256sum /root/bins/baseline/*.so | tee $OUT/baseline_so_sha256.txt
mark B1_BUILD_DONE

# ---- fingerprints (forest lanes and the isolation forest).
$AB ib baseline rf-clf,rf-reg,et-clf,et-reg,iforest
$AB diff sep11c_baseline baseline
mark B1_IB_DONE

while [ ! -f $OUT/setup.done ]; do sleep 10; done
cat $OUT/setup.txt

# ---- Python-side host split, untimed diagnostics, one warm-up plus 3 reps.
$AB use baseline
for lane in rf et iforest; do
  for ds in taxi istella; do
    PYTHONPATH=/root/mojolearn/python timeout -k 30 900 python3 -u tools/forest_host_split.py \
      --lane $lane --dataset $ds --rows 1000000 --reps 3 > $OUT/split/baseline.$lane.$ds.log 2>&1
    echo "split_exit baseline.$lane.$ds=$? $(date -u +%T)" | tee -a $OUT/ab.txt
  done
done
mark B1_SPLIT_DONE

# ---- same-pod speed cells: ours IDENTICAL against the opponent's FAST arm.
$AB speed baseline rf taxi 1000000 5 full
$AB speed baseline rf istella 1000000 5 full
$AB speed baseline iforest taxi 1000000 5 full
$AB speed baseline iforest istella 1000000 5 full
MOJOLEARN_SPEED_DEVICES=cpu MOJOLEARN_SPEED_ARMS=sklearn-et-cpu $AB speed baseline et taxi 1000000 5 full
MOJOLEARN_SPEED_DEVICES=cpu MOJOLEARN_SPEED_ARMS=sklearn-et-cpu $AB speed baseline et istella 1000000 5 full
mark B1_SPEED_DONE

# ---- one untimed stage replicate per forest cell.
$AB speed baseline rf taxi 1000000 1 stage
$AB speed baseline rf istella 1000000 1 stage
$AB speed baseline et taxi 1000000 1 stage
$AB speed baseline et istella 1000000 1 stage
mark B1_STAGE_DONE
echo "batch1 end $(date -u +%T)"
