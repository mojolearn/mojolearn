#!/bin/sh
set -eu
while [ ! -f /root/jobs/replay-neural.done ] || [ ! -f /root/jobs/replay-classical.done ]; do sleep 5; done
python3 /root/compare-replay.py > /root/replay-out/h100-comparison.log 2>&1
