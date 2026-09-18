#!/bin/bash
EV=~/mojolearn-evidence/apple-seam-repair-2026-09-18
. $EV/env.sh
cd ~/mojolearn-wt/apple-seam
for arm in norepair repair; do
  EXTRA=""; [ $arm = norepair ] && EXTRA="-D MOJOLEARN_NO_ZERO_FMA_REPAIR=1"
  # same directory, same output name for both arms, then copied
  nice -n 19 $MOJO build -j 1 -D MOJOLEARN_COLUMN_APPLE -D MOJOLEARN_NUMERIC_IDENTICAL=1 $EXTRA -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . bench/gemm_step_price_main.mojo -o $EV/price/price > $EV/price/build-$arm.log 2>&1
  echo "$arm build rc=$?" >> $EV/price/build_rc.txt
  cp $EV/price/price $EV/price/price-$arm
done
