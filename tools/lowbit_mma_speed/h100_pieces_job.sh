#!/bin/bash
# FOUR PRODUCTS, ONE STAGING on the H100 as ONE queue job: its gate with the
# three arms that must fail, the one-product gate (the staging helper and
# the launches changed under it), then the target's timing: four separate
# tuned products against one launch of one staging, with the conversions.
cd "$(dirname "$0")/../.." || exit 9
exec bash tools/lowbit_mma_speed/box_job.sh h100 pieces-gate unit-gate target-price
