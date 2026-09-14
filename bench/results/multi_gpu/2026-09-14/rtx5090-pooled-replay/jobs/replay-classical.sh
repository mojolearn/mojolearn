#!/bin/sh
set -eu
# The neural build job is already running. Keep builds serial on this pod.
while [ ! -f /root/jobs/replay-neural.done ]; do sleep 5; done
[ "$(cat /root/jobs/replay-neural.rc)" = 0 ]
bash /root/jobs/replay-classical-body.sh
