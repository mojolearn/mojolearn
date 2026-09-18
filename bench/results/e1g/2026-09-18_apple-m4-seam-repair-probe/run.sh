#!/bin/bash
EV=~/mojolearn-evidence/apple-seam-repair-2026-09-18
cd $EV/probe
for arm in norepair repaired; do
  ./seam_$arm > seam_$arm.log 2>&1; echo "$arm rc=$?" >> run_rc.txt
done
