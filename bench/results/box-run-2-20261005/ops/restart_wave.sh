#!/bin/bash
# restart_wave.sh <old sha9> <vendor> <arch> (box-run-2): stop the old wave, start wave_main at origin/main head.
bash /root/lq/stop_wave.sh $1
cp /root/lq/wave_main.sh /root/lq/br2-wave_main.sh
(setsid nohup bash /root/lq/br2-wave_main.sh $2 $3 > /root/lq/br2-wave_main-$2.out 2>&1 < /dev/null &)
echo "$2 launched"
