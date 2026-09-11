#!/bin/sh
# Lane forest-finish, batch D: the flip verdicts of ENGINEERING_RULES.md
# section 9, run ON THE POD behind batch C. One block per switch.
cd /root/mojolearn || exit 9
OUT=/root/trees_out
S=$OUT/speed
V=$OUT/verdicts.txt
PATH="$HOME/.pixi/bin:$PATH"; export PATH
while [ ! -f $OUT/phase.C4_AB_DONE ]; do sleep 15; done
: > $V
say() { echo "$@" | tee -a $V; }

# DEVIATIONS 2637 and 2638: main (baseline) against this lane (rowmajor).
for lane in rf et iforest; do
  say "=== flip_verdict lane=$lane baseline -> rowmajor (DEVIATIONS 2637/2638)"
  python3 tools/flip_verdict.py --lane $lane --arm ours --rows 1000000 \
    --taxi-before $S/baseline.$lane.taxi.r1000000.full.log $S/baseline.$lane.taxi.r1000000.ours.log \
    --taxi-after $S/rowmajor.$lane.taxi.r1000000.full.log $S/rowmajor.$lane.taxi.r1000000.ours.log \
    --istella-before $S/baseline.$lane.istella.r1000000.full.log $S/baseline.$lane.istella.r1000000.ours.log \
    --istella-after $S/rowmajor.$lane.istella.r1000000.full.log $S/rowmajor.$lane.istella.r1000000.ours.log \
    >> $V 2>&1
  say "verdict_exit $lane=$?"
done

# DEVIATION 2663: the ExtraTrees frontier batch width, ctl (4096) against each
# trial width, ours-only logs, two passes per side.
for w in bw16k bw32k; do
  say "=== flip_verdict lane=et ctl -> $w (DEVIATION 2663)"
  python3 tools/flip_verdict.py --lane et --arm ours --rows 1000000 \
    --taxi-before $S/ctl.et.taxi.r1000000.ours.p1.log $S/ctl.et.taxi.r1000000.ours.p2.log \
    --taxi-after $S/$w.et.taxi.r1000000.ours.p1.log $S/$w.et.taxi.r1000000.ours.p2.log \
    --istella-before $S/ctl.et.istella.r1000000.ours.p1.log $S/ctl.et.istella.r1000000.ours.p2.log \
    --istella-after $S/$w.et.istella.r1000000.ours.p1.log $S/$w.et.istella.r1000000.ours.p2.log \
    >> $V 2>&1
  say "verdict_exit et.$w=$?"
done
echo "D1_VERDICTS_DONE $(date -u +%T)" | tee -a $OUT/ab.txt
: > $OUT/phase.D1_VERDICTS_DONE
echo "batchD end $(date -u +%T)"
