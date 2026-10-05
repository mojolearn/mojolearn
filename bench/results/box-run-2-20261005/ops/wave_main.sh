#!/bin/bash
# wave_main.sh <vendor> <arch> [phases] (box-run-2): wave at origin/main head in its own worktree /root/mojolearn-main,
# so a running wave from /root/mojolearn is untouched. Output /root/lq/br2-wave-<sha9>, logs /root/lq/br2-ctl-<sha9>/.
V=$1; A=$2; PHASES=${3:-prepare quality identity}
R=/root/mojolearn-main; PY=/root/mojolearn/.pixi/envs/default/bin/python; DATA=/root/board-0833/cache/algos-data/rows-small
git -C /root/mojolearn fetch -q origin main || exit 2
F=$(git -C /root/mojolearn rev-parse FETCH_HEAD)  # FETCH_HEAD is per worktree: resolve it in the main tree
if [ ! -d $R ]; then git -C /root/mojolearn worktree add -q --detach $R $F || exit 2; else git -C $R checkout -q --detach $F || exit 2; fi
SHA=$(git -C $R rev-parse HEAD); W=/root/lq/br2-wave-${SHA:0:9}; C=/root/lq/br2-ctl-${SHA:0:9}; mkdir -p $C
ln -sfn /root/mojolearn/.pixi $R/.pixi
for ph in $PHASES; do
  echo "$(date -u +%FT%TZ) start $ph" >> $C/wave-$V.log
  extra=""; [ $ph = prepare ] && extra="--prepare-jobs 4"
  $PY $R/tools/identical_wave_runner.py $ph --plan $R/tools/identical_wave_plan.json --sha $SHA --vendor $V --gpu-arch $A \
     --repo $R --out $W --python $PY --data $DATA $extra > $C/wave-$V-$ph.log 2>&1
  rc=$?; echo $rc > $C/wave-$V-$ph.rc; echo "$(date -u +%FT%TZ) end $ph rc=$rc" >> $C/wave-$V.log
done
