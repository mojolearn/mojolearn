#!/bin/bash
# tools/lowbit_mma_speed/baseline_job_h100.sh -- lane/lowbit-mma-speed, the
# starting point on the H100, in this lane's own tree: lane/lowbit-units'
# gate (the reference unit plan, the flat plan, the host oracle and the
# sabotage arms) and then its timing harness, as ONE queue job so the timed
# phase runs with nothing else of the lane's on the box. Nothing here is
# tuned; it is what every lever is read against. Filed under
# bench/results/lowbit_mma_speed/h100/gate and .../h100/baseline_price.
cd "$(dirname "$0")/../.." || exit 9
export MOJOLEARN_LOWBIT_RESULTS=bench/results/lowbit_mma_speed
export MOJOLEARN_LOWBIT_PRICE_DIR=baseline_price
exec bash tools/lowbit_units/box_job.sh h100 gate price
