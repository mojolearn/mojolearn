#!/bin/bash
# The tuned plan: its gate, then the timing with it, on the shared NVIDIA pod.
exec bash "$(dirname "$0")/box_job.sh" h100 tuned_gate price
