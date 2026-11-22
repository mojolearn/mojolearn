#!/bin/bash
# M2 serial build queue: processes ~/m2-arms/bq/*.json in name order, one at a time; done files move to bq/done/.
cd ~/m2-arms; mkdir -p bq/done
while true; do
  f=$(ls bq/*.json 2>/dev/null | head -1)
  if [ -z "$f" ]; then sleep 60; continue; fi
  while pgrep -f "m2_build_ab.py m2_jobs" >/dev/null; do sleep 60; done
  n=$(basename $f .json); python3 m2_build_ab.py $f > bq/done/$n.log 2>&1; mv $f bq/done/
done
