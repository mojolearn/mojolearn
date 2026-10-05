#!/bin/bash
# wave_start.sh <vendor> <arch>: stop any compile matrix, move /root/mojolearn to lane/box-run-2 head, start wave phases.
for p in $(ps -eo pid,args | grep -E '[c]f_run.sh|[c]f_driver.py' | awk '{print $1}'); do kill -TERM $p 2>/dev/null; done
for g in $(ps -eo pgid,args | grep -E '[c]ompile_slot.sh|[m]odular-crashpad' | awk '{print $1}' | sort -u); do kill -TERM -- -$g 2>/dev/null; done
sleep 3; rm -rf /root/mojolearn-evidence/compile-slots/slot*
cd /root/mojolearn && git fetch -q origin lane/box-run-2 && git checkout -q --detach FETCH_HEAD || exit 2
SHA=$(git rev-parse HEAD); [ -e /root/lq/br2-ctl ] && mv /root/lq/br2-ctl /root/lq/br2-ctl.prev-$(date +%s)
mkdir -p /root/lq/br2-ctl
(setsid nohup bash /root/lq/br2-wave_launch.sh $1 $2 > /root/lq/br2-ctl/launch-$1.out 2>&1 < /dev/null &)
echo "wave started $1 $2 at ${SHA:0:9}"
