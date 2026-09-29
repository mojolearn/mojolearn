#!/bin/bash
# Run 6: the epilogue fold, one lever. The tuned plan's gate (fused and
# two-launch must print one digest; every arm seen failing), then the timing
# with the fused arms beside the two-launch arms, on the shared NVIDIA pod.
exec bash "$(dirname "$0")/box_job.sh" h100 tuned_gate price
