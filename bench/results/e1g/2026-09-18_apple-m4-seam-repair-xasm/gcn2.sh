#!/bin/bash
EV=~/mojolearn-evidence/apple-seam-repair-2026-09-18
. $EV/env.sh
WT=~/mojolearn-wt/apple-seam-xasm; cd $WT
FILES="checks/kernel_matrix.mojo core/gemm.mojo core/gram_multi_gpu.mojo gemm/checks/gemm_identical.mojo gemm/checks/gemm_lowbit.mojo"
mkdir -p $EV/xasm/bc; for f in $FILES; do mkdir -p $EV/xasm/bc/$(dirname $f); cp $f $EV/xasm/bc/$f; done
restore() { for f in $FILES; do cp $EV/xasm/bc/$f $WT/$f; done; }
trap restore EXIT
for arm in main branch; do
  if [ $arm = main ]; then for f in $FILES; do git show c7442abed:$f > $f; done; else restore; fi
  rm -rf $EV/xasm/gcn-$arm; mkdir -p $EV/xasm/gcn-$arm
  nice -n 19 $MOJO build -j 1 --emit asm --target-triple x86_64-unknown-linux-gnu --target-cpu x86-64-v3 --target-accelerator gfx942 -D MOJOLEARN_COLUMN_AMD -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_ARM_TRIAL=1 -D MOJOLEARN_XASM_NONCE_7=1 -I . gemm/checks/gemm_device_check.mojo -o $EV/xasm/gcn-$arm/out.s > $EV/xasm/gcn-$arm/build.log 2>&1
  echo "$arm rc=$? n=$(ls $EV/xasm/gcn-$arm/*.amdgcn 2>/dev/null | wc -l)" >> $EV/xasm/gcn2_rc.txt
done
