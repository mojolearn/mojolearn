#!/bin/bash
# The H100's timed phases as ONE queue job, so they run with nothing else
# of the lane's on the box (the brief's review point 4): the harness and its
# sabotage arm, then the vendor comparison. Filed under h100/.
cd "$(dirname "$0")/../.." || exit 9
exec bash tools/lowbit_units/box_job.sh h100 price vendor
