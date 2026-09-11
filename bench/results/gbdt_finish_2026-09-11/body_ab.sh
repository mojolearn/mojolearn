#!/bin/sh
# gbdt-finish lane, DEVIATIONS 2634 and 2635 A/B on the H100 (RUNS ON THE POD
# from /root/mojolearn, after body_setup.sh). Identity first: identity_break
# fingerprints of the four GBDT lanes on baseline, a2634 and both, their
# diffs, and the sub-byte identity check on each set. Then timing: two .so
# sets cannot share a process, so rounds alternate PROCESSES over the three
# sets (the order rotates each round), our IDENTICAL arm only, 5 timed fits
# per process, the three grow policies, taxi and Istella-S at 1M rows.
# Usage: sh body_ab.sh [sets, default "baseline a2634 both"] [rounds, default 3]
set -u
cd /root/mojolearn || exit 9
OUT=/root/trees_out; AB="sh tools/trees_identical_ab.sh"
export PATH="$HOME/.pixi/bin:$PATH" MOJOLEARN_NUMERIC_MODE=identical
# Mojo 1.0.0's GPU runtime refuses NVIDIA drivers below 580 (pod 1 of this
# lane, 570.195.03: every fit refused; the image's CUDA 12.4 ptxas handed in
# through MODULAR_NVPTX_COMPILER_PATH gave CUDA_ERROR_INVALID_IMAGE). Refuse
# before timing anything.
_drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
if [ -n "$_drv" ] && [ "$_drv" -lt 580 ]; then
  echo "REFUSED driver $_drv < 580 $(date -u +%T)" | tee -a $OUT/progress.txt
  exit 3
fi
SETS="${1:-baseline a2634 both}"; ROUNDS="${2:-3}"; TAG="${3:-ab}"
mark() { echo "$1 $(date -u +%T)" | tee -a $OUT/progress.txt; }
while [ ! -f $OUT/builds.done ]; do sleep 10; done
if [ "${SKIP_IDENTITY:-0}" != 1 ]; then
  mark ${TAG}_identity_start
  LANES=gbdt-symmetric,gbdt-depthwise,gbdt-lossguide,gbdt-rmse
  first=""
  for s in $SETS; do
    [ -f /root/bins/$s/_mojolearn_gbdt.so ] || { mark "${TAG}_missing_set_$s"; exit 1; }
    $AB ib $s "$LANES"
    [ -n "$first" ] && $AB diff $first $s
    [ -z "$first" ] && first=$s
  done
  mkdir -p $OUT/check
  for s in $SETS; do
    $AB use $s > /dev/null
    PYTHONPATH=/root/mojolearn/python timeout -k 30 1800 python3 -u checks/gbdt_sub_byte_identity_check.py \
      --json $OUT/check/$s.json > $OUT/check/$s.txt 2>&1
    echo "subbyte_check $s exit=$? $(tail -1 $OUT/check/$s.txt) $(date -u +%T)" | tee -a $OUT/ab.txt
  done
  mark ${TAG}_identity_done
fi
while [ ! -f $OUT/data.done ]; do sleep 10; done
mark ${TAG}_speed_start
r=1
while [ $r -le $ROUNDS ]; do
  # rotate the set order by the round number
  order=$(echo $SETS | awk -v r=$r '{n=NF; s=""; for(i=0;i<n;i++){k=(i+r-1)%n+1; s=s" "$k}; print s}')
  for ds in taxi istella; do
    for lane in gbdt-symmetric gbdt-depthwise gbdt-lossguide; do
      for s in $order; do
        MOJOLEARN_SPEED_TAG=$TAG.r$r $AB speed $s $lane $ds 1000000 5 ours
      done
    done
  done
  mark "${TAG}_round$r"
  r=$((r + 1))
done
mark ${TAG}_done
