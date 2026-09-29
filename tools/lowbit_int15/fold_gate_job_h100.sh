#!/bin/bash
# The epilogue fold: the tuned plan's gate alone (fused and two-launch arms, the
# exponent arm), on the shared NVIDIA pod. While INT15_FUSED_IS_STUB it proves
# this lane's half builds and its arms fail as they must; nothing of the fold.
exec bash "$(dirname "$0")/box_job.sh" h100 tuned_gate
