#!/bin/bash
# Cross-compile the GEMM-family consumers for sm_90a and gfx942 at origin/main's
# versions of the changed files and at the branch's, in the SAME worktree path
# and SAME output names, and digest the emitted assembly.
EV=~/mojolearn-evidence/apple-seam-repair-2026-09-18
. $EV/env.sh
WT=~/mojolearn-wt/apple-seam-xasm
cd $WT
BASE=${BASE:-c7442abed}
FILES="checks/kernel_matrix.mojo core/gemm.mojo core/gram_multi_gpu.mojo gemm/checks/gemm_identical.mojo gemm/checks/gemm_lowbit.mojo"
PROGS="bench/gemm_step_price_main.mojo gemm/checks/gemm_device_check.mojo bench/lanes_price_main.mojo gemm/checks/gemm_lowbit_check.mojo training/checks/gram_outputs_parallel_check.mojo"
mkdir -p $EV/xasm/branchcopy
for f in $FILES; do mkdir -p $EV/xasm/branchcopy/$(dirname $f); cp $f $EV/xasm/branchcopy/$f; done
restore() { for f in $FILES; do cp $EV/xasm/branchcopy/$f $WT/$f; done; }
trap restore EXIT
for arm in main branch; do
  if [ $arm = main ]; then for f in $FILES; do git show $BASE:$f > $f; done; else restore; fi
  git diff --stat > $EV/xasm/tree-$arm.txt
  for col in nvidia:sm_90a:MOJOLEARN_COLUMN_NVIDIA amd:gfx942:MOJOLEARN_COLUMN_AMD apple:apple-m4:MOJOLEARN_COLUMN_APPLE; do
    IFS=: read name arch def <<< "$col"
    TRIPLE="--target-triple x86_64-unknown-linux-gnu --target-cpu x86-64-v3"
    [ $name = apple ] && TRIPLE=""
    P=$PROGS; [ $name = apple ] && P="gemm/checks/gemm_device_check.mojo"
    for p in $P; do
      tag=$(basename $p .mojo)
      nice -n 19 $MOJO build -j 1 --emit asm $TRIPLE --target-accelerator $arch -D $def -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . $p -o $EV/xasm/out.s > $EV/xasm/build-$arm-$name-$tag.log 2>&1
      rc=$?
      if [ $rc = 0 ]; then d=$(shasum -a 256 $EV/xasm/out.s | cut -c1-16); sz=$(wc -c < $EV/xasm/out.s); gzip -c $EV/xasm/out.s > $EV/xasm/$arm-$name-$tag.s.gz; else d=FAIL; sz=0; fi
      echo "$arm $name $tag rc=$rc sha256=$d bytes=$sz" | tee -a $EV/xasm/digests.txt
    done
  done
done
