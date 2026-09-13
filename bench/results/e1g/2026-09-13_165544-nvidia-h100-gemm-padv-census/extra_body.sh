#!/bin/sh
# tools/gemm_census_then_kernel_leg.sh -- one NVIDIA lease, two bodies in
# sequence: the register census (DEVIATION 2702, ~1 minute, launches nothing)
# and then the kernel arms leg (DEVIATIONS 2700 and 2703, price + LM probe).
# Each writes its own /root/gemm_leg_out/<lane> directory; the runner's fetch
# brings both home. POSIX sh only.
set -u
ROOT=${MOJOLEARN_GEMM_STEP_ROOT:-/root/mojolearn}
cd "$ROOT" || exit 9
sh tools/gemm_kernel_census_leg.sh
echo "census_exit=$?" > /root/gemm_leg_out/census_then_kernel.txt
exec sh tools/gemm_kernel_leg.sh
