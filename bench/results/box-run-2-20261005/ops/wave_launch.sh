#!/bin/bash
# wave_launch.sh <vendor> <gpu-arch> [phases]  (box-run-2): wave phases for the consolidated head on this box.
# Output /root/lq/br2-wave-<sha9>; each phase: full log + rc file; later phases run even if one fails.
V=$1; A=$2; PHASES=${3:-prepare quality identity}
R=/root/mojolearn; SHA=$(git -C $R rev-parse HEAD); W=/root/lq/br2-wave-${SHA:0:9}; mkdir -p /root/lq/br2-ctl
PY=$R/.pixi/envs/default/bin/python; DATA=/root/board-0833/cache/algos-data/rows-small
for ph in $PHASES; do
  echo "$(date -u +%FT%TZ) start $ph" >> /root/lq/br2-ctl/wave-$V.log
  extra=""; [ $ph = prepare ] && extra="--prepare-jobs 4"
  $PY $R/tools/identical_wave_runner.py $ph --plan $R/tools/identical_wave_plan.json --sha $SHA --vendor $V --gpu-arch $A \
     --repo $R --out $W --python $PY --data $DATA $extra > /root/lq/br2-ctl/wave-$V-$ph.log 2>&1
  rc=$?; echo $rc > /root/lq/br2-ctl/wave-$V-$ph.rc
  echo "$(date -u +%FT%TZ) end $ph rc=$rc" >> /root/lq/br2-ctl/wave-$V.log
done
