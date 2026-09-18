#!/bin/bash
EV=~/mojolearn-evidence/apple-seam-repair-2026-09-18
PY=/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/default/bin/python
cd ~/mojolearn-wt/apple-seam
export MOJOLEARN_NUMERIC_MODE=identical
i=0
for arm in norepair admit admit norepair; do
  i=$((i+1)); out=$EV/lm/timers-enwik8-$arm-$i; rm -rf $out
  PYTHONPATH=$EV/pkgt-$arm $PY tools/lm_step_memory_probe.py --out $out --target --resident-lean --witness-every-step --steps 1 --component-timing --component-timing-steps 2 --budget-seconds 1500 --corpus $EV/corpus/enwik8/input.txt > $out.log 2>&1
  echo "timers $arm $i rc=$? $(date +%T)" >> $EV/metal4_rc.txt
done
