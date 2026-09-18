#!/bin/bash
# Post-merge: origin/main vs merged branch, same path, gfx942 + sm_90a, gemm_device_check + gemm_step_price_main
EV=~/mojolearn-evidence/apple-seam-repair-2026-09-18
. $EV/env.sh
WT=~/mojolearn-wt/apple-seam-xasm2; cd $WT
MAIN=$(git -C ~/mojolearn-wt/apple-seam rev-parse origin/main)
FILES="checks/kernel_matrix.mojo core/gemm.mojo core/gram_multi_gpu.mojo gemm/checks/gemm_identical.mojo gemm/checks/gemm_lowbit.mojo"
mkdir -p $EV/xasm/bc3; for f in $FILES; do mkdir -p $EV/xasm/bc3/$(dirname $f); cp $f $EV/xasm/bc3/$f; done
trap 'for f in $FILES; do cp $EV/xasm/bc3/$f $WT/$f; done' EXIT
for arm in main branch; do
  if [ $arm = main ]; then for f in $FILES; do git show $MAIN:$f > $f; done; else for f in $FILES; do cp $EV/xasm/bc3/$f $f; done; fi
  for col in nvidia:sm_90a:MOJOLEARN_COLUMN_NVIDIA amd:gfx942:MOJOLEARN_COLUMN_AMD; do
    IFS=: read name arch def <<< "$col"
    for p in gemm/checks/gemm_device_check.mojo bench/gemm_step_price_main.mojo; do
      tag=$(basename $p .mojo)
      nice -n 19 $MOJO build -j 1 --emit asm --target-triple x86_64-unknown-linux-gnu --target-cpu x86-64-v3 --target-accelerator $arch -D $def -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . $p -o $EV/xasm/o3.s > $EV/xasm/b3-$arm-$name-$tag.log 2>&1
      echo "$arm $name $tag rc=$?" >> $EV/xasm/xcheck3_rc.txt
      gzip -c $EV/xasm/o3.s > $EV/xasm/m3-$arm-$name-$tag.s.gz
    done
  done
done
