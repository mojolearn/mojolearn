#!/bin/bash
# Keeps harvester.py alive: restarts it 10 s after any exit until harvester.stop exists.
D=$HOME/mojolearn-evidence/box-run-2
echo $$ > $D/harvester_loop.pid
while [ ! -f $D/harvester.stop ]; do
  /usr/bin/python3 $D/harvester.py $D/harvester-config.json >> $D/harvester.log 2>&1
  echo "$(date -u +%FT%TZ) harvester exited rc=$?; restarting" >> $D/harvester.log
  sleep 10
done
