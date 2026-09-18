#!/bin/bash
EV=~/mojolearn-evidence/apple-seam-repair-2026-09-18
PY=/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/default/bin/python
cd ~/mojolearn-wt/apple-seam
log() { echo "$* $(date +%T)" >> $EV/metal2_rc.txt; }
# 1. Apple GEMM identity cards, both arms
for arm in norepair admit; do
  MOJOLEARN_GEMM_CARD_ARM=device MOJOLEARN_IDENTITY_TRACE=$EV/card/apple-$arm.card $EV/card/c-$arm > $EV/card/run-$arm.log 2>&1
  log "card $arm rc=$?"
done
# 2. pinned kernels price, interleaved
i=0
for arm in norepair admit admit norepair; do
  i=$((i+1)); MOJOLEARN_RTF_PRICE_ROUNDS=7 $EV/pinned/p-$arm > $EV/pinned/run-$arm-$i.log 2>&1
  log "pinned $arm $i rc=$?"
done
# 3. LM lean steps, target shape, both corpora, interleaved
for corpus in enwik8 pile_github; do
  i=0
  for arm in norepair admit admit norepair; do
    i=$((i+1)); out=$EV/lm/$corpus-$arm-$i; rm -rf $out
    PYTHONPATH=$EV/pkg-$arm MOJOLEARN_THREADS=1 OMP_NUM_THREADS=1 $PY tools/lm_step_memory_probe.py --out $out --target --resident-lean --witness-every-step --steps ${LM_STEPS:-5} --budget-seconds 1500 --corpus $EV/corpus/$corpus/input.txt > $out.log 2>&1
    log "lm $corpus $arm $i rc=$?"
  done
done
