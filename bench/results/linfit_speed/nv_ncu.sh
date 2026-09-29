#!/bin/bash
# lane/linfit-speed: Nsight Compute on our SGD fit kernel (one pass, 100k rows)
# and the wide GLM's cell kernel: stall reasons and the hottest SASS.
set -uo pipefail
cd /root/mojolearn-linfit-speed
export MOJOLEARN_NUMERIC_MODE=identical
NCU=/usr/local/cuda/bin/ncu
O=bench/results/linfit_speed/ncu-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p $O
PY=".pixi/envs/default/bin/python"
for spec in "sgd-reg taxi fit_kernel" "sgd-reg istella fit_kernel" "poisson istella gw_cells"; do
  set -- $spec
  $NCU -k regex:$3 -c 1 --section WarpStateStats --section SourceCounters --section SpeedOfLight \
      --section LaunchStats --section Occupancy --import-source no -f -o $O/$1-$2 \
      $PY tools/linfit_speed.py --data /root/linfit-data --lanes $1 --datasets $2 --rows 100000 \
      --max-iter 1 --no-warm --out $O/$1-$2.json > $O/$1-$2.log 2>&1
  $NCU --import $O/$1-$2.ncu-rep --print-details all > $O/$1-$2.details.txt 2>&1
  $NCU --import $O/$1-$2.ncu-rep --page source --csv --print-source sass > $O/$1-$2.sass.csv 2>&1
  grep -i "stall\|Warp Cycles\|Duration\|Issued" $O/$1-$2.details.txt | head -40
done
ls -la $O
