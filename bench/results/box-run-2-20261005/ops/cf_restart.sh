#!/bin/bash
# cf_restart.sh <sha> <jobs.tsv>: stop this box's running compile matrix (by process group), then start a new one.
for p in $(ps -eo pid,args | grep -E '[c]f_run.sh|[c]f_driver.py' | awk '{print $1}'); do kill -TERM $p 2>/dev/null; done
for g in $(ps -eo pgid,args | grep -E '[c]ompile_slot.sh|[m]odular-crashpad' | awk '{print $1}' | sort -u); do kill -TERM -- -$g 2>/dev/null; done
sleep 3; rm -rf /root/mojolearn-evidence/compile-slots/slot*
echo "mojo builds left: $(ps -eo args | grep -cE '[m]ojo build')"
(setsid nohup bash /root/lq/br2-cf/cf_run.sh $1 $2 > /dev/null 2>&1 < /dev/null &)
echo relaunched
