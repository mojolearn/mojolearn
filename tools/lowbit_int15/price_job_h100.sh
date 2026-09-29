#!/bin/bash
# The timing job with the box named, for the shared NVIDIA pod's queue.
export MOJOLEARN_LOWBIT_BOX=h100
exec bash "$(dirname "$0")/price_job.sh"
