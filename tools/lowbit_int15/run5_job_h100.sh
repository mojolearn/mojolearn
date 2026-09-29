#!/bin/bash
# The gate at the commit with the large product, the tuned plan's gate, then the timing (run 5), on the shared NVIDIA pod.
exec bash "$(dirname "$0")/box_job.sh" h100 gate tuned_gate price
