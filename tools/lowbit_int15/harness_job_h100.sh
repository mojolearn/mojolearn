#!/bin/bash
# The harness job with the box named, for the shared NVIDIA pod's queue.
export MOJOLEARN_LOWBIT_BOX=h100
exec bash "$(dirname "$0")/harness_job.sh"
