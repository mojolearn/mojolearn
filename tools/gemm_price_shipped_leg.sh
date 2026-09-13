#!/bin/sh
# tools/gemm_price_shipped_leg.sh -- the price-shipped noise control alone
# (DEVIATION 2707's gate had it exit 1 on a harness PHASE routing bug, fixed in
# bench/gemm_step_price_main.mojo). Builds and runs the step leg with only the
# `shipped` price arm and no LM probe; the check still runs. POSIX sh.
set -u
MOJOLEARN_GEMM_STEP_LEG_ARMS=shipped
MOJOLEARN_GEMM_STEP_LEG_LM_ARMS=kpack_hg
MOJOLEARN_GEMM_STEP_LEG_SKIP_LM=1
MOJOLEARN_GEMM_STEP_LEG_CHECK_ARMS=shipped,kpack_hg
MOJOLEARN_GEMM_STEP_LEG_OUT=${MOJOLEARN_GEMM_STEP_LEG_OUT:-/root/gemm_leg_out/gemm-price-shipped}
export MOJOLEARN_GEMM_STEP_LEG_ARMS MOJOLEARN_GEMM_STEP_LEG_LM_ARMS MOJOLEARN_GEMM_STEP_LEG_SKIP_LM \
    MOJOLEARN_GEMM_STEP_LEG_CHECK_ARMS MOJOLEARN_GEMM_STEP_LEG_OUT
cd "${MOJOLEARN_GEMM_STEP_ROOT:-/root/mojolearn}" || exit 9
exec sh tools/gemm_step_leg.sh
