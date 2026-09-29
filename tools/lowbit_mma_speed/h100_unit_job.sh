#!/bin/bash
# The tuned unit plans on the H100 as ONE queue job (a queue job takes one
# script path and no arguments): THE GATE BEFORE THE CLOCK (every plan
# against the reference unit plan, the flat plan and the oracle, with the
# arms that must fail), then the PTX counter, then the timing, built first
# and run with nothing else of the lane's on the box.
cd "$(dirname "$0")/../.." || exit 9
exec bash tools/lowbit_mma_speed/box_job.sh h100 unit-gate ptx unit-price
