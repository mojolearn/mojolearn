#!/bin/sh
# The neural pass's two owed on-box items, one H100 lease, main 5b7e1e41.
#
# 1. THE FORWARD REGISTER READBACK (DEVIATIONS 2655/2656). The regs lane
#    merged the six forward RESOURCES rows at 18bd7da7 but never ran them on a
#    box, and Metal refuses the query, so the forward kernel's register count
#    has never been read anywhere. Phase 1 only (MOJOLEARN_ATTN_REGS_FULL=0):
#    no corpus, no binding, nothing timed, about four minutes. It runs FIRST so
#    the number comes home even if the lease dies later.
#
# 2. THE OPPONENT CONFIRMATION. bench/OPPONENT_REFERENCE.md still quotes
#    0.2919 / 0.2906 for our IDENTICAL row, measured before the estash flip
#    (DEVIATION 2657) moved the NVIDIA default. This re-measures OUR shipped
#    default and every torch column on the SAME pod and the same heat window,
#    which is the only way that table's ratios are allowed to be quoted.
set -u
cd /root/mojolearn || exit 9
OUTROOT=/root/gemm_leg_out
NEW_DEFAULT=stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32

MOJOLEARN_ATTN_REGS_FULL=0 \
MOJOLEARN_ATTN_REGS_OUT=$OUTROOT/attention-regs \
MOJOLEARN_ATTN_BASELINE=$NEW_DEFAULT \
MOJOLEARN_ATTN_LEG_ARMS=$NEW_DEFAULT \
MOJOLEARN_ATTN_LEG_LM_ARMS=$NEW_DEFAULT \
MOJOLEARN_COMPILE_JOBS=8 \
    sh tools/attention_regs_leg.sh
r=$?
echo "regs_leg_exit=$r" >> $OUTROOT/leg.txt
# The readback is phase 1 of its own leg and stands alone: a failure here must
# not cost the opponent rows, so the body carries on and the gate reads both.

MOJOLEARN_ATTN_BASELINE=$NEW_DEFAULT \
MOJOLEARN_ATTN_LEG_ARMS=$NEW_DEFAULT \
MOJOLEARN_ATTN_LEG_LM_ARMS=$NEW_DEFAULT \
MOJOLEARN_ATTN_LEG_SHIPPED_CHECK=1 \
MOJOLEARN_ATTN_LEG_SKIP_TIMERS=1 \
MOJOLEARN_COMPILE_JOBS=8 \
    sh tools/attention_step_leg.sh
a=$?
echo "attention_leg_exit=$a" >> $OUTROOT/leg.txt

sh tools/torch_lm_step_opponent_leg.sh
t=$?
echo "torch_leg_exit=$t" >> $OUTROOT/leg.txt

echo "regs=$r attention=$a torch=$t" >> $OUTROOT/leg.txt
[ "$a" = 0 ] && [ "$t" = 0 ]
