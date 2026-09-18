#!/bin/bash
EV=~/mojolearn-evidence/apple-seam-repair-2026-09-18
. $EV/env.sh
WT=~/mojolearn-wt/apple-seam-xasm; cd $WT
F=gemm/checks/gemm_identical.mojo
trap 'cp $EV/xasm/bc/$F $WT/$F' EXIT
cp ~/mojolearn-wt/apple-seam/$F $F
rm -rf $EV/xasm/gcn-v7; mkdir -p $EV/xasm/gcn-v7
nice -n 19 $MOJO build -j 1 --emit asm --target-triple x86_64-unknown-linux-gnu --target-cpu x86-64-v3 --target-accelerator gfx942 -D MOJOLEARN_COLUMN_AMD -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_ARM_TRIAL=1 -D MOJOLEARN_XASM_NONCE_7=1 -I . gemm/checks/gemm_device_check.mojo -o $EV/xasm/gcn-v7/out.s > $EV/xasm/gcn-v7/build.log 2>&1
echo "v7 amd rc=$?" >> $EV/xasm/gcn4_rc.txt
nice -n 19 $MOJO build -j 1 --emit asm --target-triple x86_64-unknown-linux-gnu --target-cpu x86-64-v3 --target-accelerator sm_90a -D MOJOLEARN_COLUMN_NVIDIA -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . gemm/checks/gemm_device_check.mojo -o $EV/xasm/v7-nvidia.s > $EV/xasm/v7-nvidia.log 2>&1
echo "v7 nvidia rc=$?" >> $EV/xasm/gcn4_rc.txt
nice -n 19 $MOJO build -j 1 --emit asm --target-accelerator apple-m4 -D MOJOLEARN_COLUMN_APPLE -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . gemm/checks/gemm_device_check.mojo -o $EV/xasm/v7-apple.s > $EV/xasm/v7-apple.log 2>&1
echo "v7 apple rc=$?" >> $EV/xasm/gcn4_rc.txt
