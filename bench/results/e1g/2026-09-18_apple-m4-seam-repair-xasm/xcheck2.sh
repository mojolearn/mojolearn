#!/bin/bash
EV=~/mojolearn-evidence/apple-seam-repair-2026-09-18
. $EV/env.sh
WT=~/mojolearn-wt/apple-seam-xasm; cd $WT
F=gemm/checks/gemm_identical.mojo
trap 'cp $EV/xasm/bc/$F $WT/$F' EXIT
cp ~/mojolearn-wt/apple-seam/$F $F
for p in bench/gemm_step_price_main.mojo gemm/checks/gemm_device_check.mojo bench/lanes_price_main.mojo gemm/checks/gemm_lowbit_check.mojo training/checks/gram_outputs_parallel_check.mojo; do
  tag=$(basename $p .mojo)
  nice -n 19 $MOJO build -j 1 --emit asm --target-triple x86_64-unknown-linux-gnu --target-cpu x86-64-v3 --target-accelerator gfx942 -D MOJOLEARN_COLUMN_AMD -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . $p -o $EV/xasm/out2.s > $EV/xasm/build2-$tag.log 2>&1
  echo "branch2 amd $tag rc=$?" >> $EV/xasm/xcheck2_rc.txt
  gzip -c $EV/xasm/out2.s > $EV/xasm/branch2-amd-$tag.s.gz
done
