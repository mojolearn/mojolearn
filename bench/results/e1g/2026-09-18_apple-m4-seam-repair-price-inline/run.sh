#!/bin/bash
cd ~/mojolearn-evidence/apple-seam-repair-2026-09-18/price
i=0
for arm in norepair repair repair norepair; do
  i=$((i+1))
  MOJOLEARN_GEMM_ARM=shipped MOJOLEARN_GEMM_STEP_ROUNDS=11 ./price-$arm > price-$arm-$i.log 2>&1
  echo "$arm $i rc=$?" >> run_rc.txt
done
