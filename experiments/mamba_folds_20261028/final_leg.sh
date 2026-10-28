#!/bin/bash
# Pod leg for lane mamba-folds after the chunked-scan deletion: host (CPU column) and device (sm_89) mamba roots, arms off/bwd.
set -u
NPROC=$(nproc); echo "nproc=$NPROC" > "$LEG_OUT/host.txt"
export DEFSETS="off bwd" MAIN=$PWD JOBS=$NPROC
bash experiments/mamba_folds_20261028/compile.sh "$LEG_OUT/cpu" host; rc1=$?
COLUMN_DEFS="-D MOJOLEARN_COLUMN_NVIDIA" EXTRA="--target-accelerator sm_89" bash experiments/mamba_folds_20261028/compile.sh "$LEG_OUT/sm89" device; rc2=$?
for f in "$LEG_OUT"/sm89/*.so; do echo "$(basename $f) sm_89_strings=$(strings "$f" | grep -c sm_89)"; done >> "$LEG_OUT/host.txt" 2>/dev/null
rm -f "$LEG_OUT"/cpu/*.so "$LEG_OUT"/sm89/*.so
echo "LEG cpu_rc=$rc1 sm89_rc=$rc2" | tee -a "$LEG_OUT/host.txt"
[ $rc1 = 0 ] && [ $rc2 = 0 ]
