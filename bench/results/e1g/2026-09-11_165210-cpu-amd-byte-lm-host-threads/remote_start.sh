#!/bin/sh
# Written by tools/do_extra_leg.sh. Starts the remote body DETACHED under the
# work bound and prints the wrapper pid; the leg polls that pid and the
# sentinel. body_exit=124 is the bound firing.
set -u
rm -f /root/gemm_leg.done /root/gemm_leg_console.log
if command -v timeout > /dev/null 2>&1; then
    nohup sh -c 'timeout -k 30 2230 sh /root/gemm_leg.sh; rc=$?; mkdir -p /root/gemm_leg_out; echo "body_exit=$rc" >> /root/gemm_leg_out/leg.txt' \
        > /root/gemm_leg_console.log 2>&1 < /dev/null &
else
    nohup sh -c 'sh /root/gemm_leg.sh; rc=$?; mkdir -p /root/gemm_leg_out; echo "body_exit=$rc (NO timeout(1): unbounded)" >> /root/gemm_leg_out/leg.txt' \
        > /root/gemm_leg_console.log 2>&1 < /dev/null &
fi
echo "REMOTE_PID=$!"
