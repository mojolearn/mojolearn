#!/bin/bash
# stop_wave.sh <sha9> (box-run-2): stop the wave launcher + runner/workers for one superseded wave (by process group).
for g in $(ps -eo pgid,args | grep -E "[b]r2-wave_main.sh|[b]r2-wave-$1" | awk '{print $1}' | sort -u); do kill -TERM -- -$g 2>/dev/null; done
sleep 3; echo "left: $(ps -eo args | grep -cE "[b]r2-wave-$1|[b]r2-wave_main.sh")"
echo "stopped $(date -u +%FT%TZ): superseded by a newer main freeze" > /root/lq/br2-ctl-$1/STOPPED
