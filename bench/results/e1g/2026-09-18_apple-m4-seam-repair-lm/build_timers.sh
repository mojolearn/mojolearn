#!/bin/bash
EV=~/mojolearn-evidence/apple-seam-repair-2026-09-18
. $EV/env.sh
export PATH=$EV/bind/shim:$PATH
export MOJOLEARN_COMPILE_JOBS=1
cd ~/mojolearn-wt/apple-seam
for arm in norepair admit; do
  X="-D MOJOLEARN_STEP_PHASE_TIMERS=1"; [ $arm = norepair ] && X="$X -D MOJOLEARN_NO_ZERO_FMA_REPAIR=1"
  rm -rf $EV/bind/timers-$arm
  MOJOLEARN_BUILD_EXTRA_DEFINES="$X" MOJOLEARN_BYTE_LM_OUTDIR=$EV/bind/timers-$arm nice -n 19 sh bindings/build_byte_lm.sh > $EV/bind/build-timers-$arm.log 2>&1
  echo "timers $arm rc=$? $(date)" >> $EV/bind/build_rc.txt
  rm -rf $EV/pkgt-$arm; cp -R $EV/pkg-$arm $EV/pkgt-$arm
  cp $EV/bind/timers-$arm/_mojolearn_byte_lm.so $EV/pkgt-$arm/mojolearn/identical/_mojolearn_byte_lm.so
done
