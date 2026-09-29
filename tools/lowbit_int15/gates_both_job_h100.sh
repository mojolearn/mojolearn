#!/bin/bash
# The reference gate (with the row-scales case and the exponent arm) and the
# tuned gate, on the shared NVIDIA pod. No timing.
exec bash "$(dirname "$0")/box_job.sh" h100 gate tuned_gate
