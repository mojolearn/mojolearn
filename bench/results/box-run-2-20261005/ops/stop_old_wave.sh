#!/bin/bash
# stop_old_wave.sh (box-run-2): stop the superseded 5e284c314 wave launcher and its runner/workers (by process group).
for g in $(ps -eo pgid,args | grep -E '[b]r2-wave_launch.sh|[b]r2-wave-5e284c314' | awk '{print $1}' | sort -u); do kill -TERM -- -$g 2>/dev/null; done
sleep 3; echo "left: $(ps -eo args | grep -cE '[b]r2-wave-5e284c314|[b]r2-wave_launch.sh')"
echo "stopped $(date -u +%FT%TZ): superseded by main 368fc68ba" > /root/lq/br2-ctl/STOPPED
