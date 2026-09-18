#!/bin/bash
cd ~/mojolearn-evidence/apple-seam-repair-2026-09-18/rtfgemm
for arm in norepair admit never inline; do
  ./chk-$arm > run_$arm.log 2>&1; echo "$arm rc=$?" >> run_rc.txt
done
