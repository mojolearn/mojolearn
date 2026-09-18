#!/bin/bash
cd ~/mojolearn-evidence/apple-seam-repair-2026-09-18/price
i=8
for arm in norepair admit admit norepair norepair admit admit norepair; do
  i=$((i+1))
  MOJOLEARN_GEMM_ARM=shipped MOJOLEARN_GEMM_STEP_ROUNDS=7 ./price-$arm > price-$arm-$i.log 2>&1
  echo "$arm $i rc=$?" >> run_rc.txt
done
