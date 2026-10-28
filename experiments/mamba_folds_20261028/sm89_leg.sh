#!/bin/bash
# RunPod CPU leg for lane mamba-folds: the NVIDIA sm_89 (L40S / Ada) device roots, every arm.
# Runs inside the shipped repo on the pod; results land in $LEG_OUT.
set -u
NPROC=$(nproc)
echo "nproc=$NPROC" > "$LEG_OUT/host.txt"
MAIN=$PWD COLUMN_DEFS="" EXTRA="--target-accelerator sm_89" JOBS=$NPROC \
  bash experiments/mamba_folds_20261028/compile.sh "$LEG_OUT/sm89" device training
rc=$?
echo "LEG rc=$rc" | tee -a "$LEG_OUT/host.txt"
for f in "$LEG_OUT"/sm89/*.so; do echo "$(basename $f) sm_89_strings=$(strings "$f" | grep -c sm_89)"; done >> "$LEG_OUT/host.txt" 2>/dev/null
rm -f "$LEG_OUT"/sm89/*.so
exit $rc
