#!/bin/bash
# The parallel quantizer on the H100 as ONE queue job (a queue job takes one
# script path and no arguments): its gate with the arms that must fail, then
# its timing, built first and run with nothing else of the lane's on the box.
cd "$(dirname "$0")/../.." || exit 9
exec bash tools/lowbit_mma_speed/box_job.sh h100 quant-gate quant-price
