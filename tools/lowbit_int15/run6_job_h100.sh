#!/bin/bash
# Run 6: the epilogue fold, one lever. THE GATE BEFORE THE CLOCK: the tuned
# plan's gate (fused and two-launch must print the oracle's digest; every arm
# seen failing), and only if it is GREEN the timing, with the fused arms beside
# the two-launch arms, on the shared NVIDIA pod.
cd "$(dirname "$0")/../.." || exit 9
export MOJOLEARN_LOWBIT_BOX=h100
bash tools/lowbit_int15/box_job.sh h100 tuned_gate
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "run6: the tuned gate is RED (exit $rc); the timing is NOT run"
    exit "$rc"
fi
exec bash tools/lowbit_int15/box_job.sh h100 price
