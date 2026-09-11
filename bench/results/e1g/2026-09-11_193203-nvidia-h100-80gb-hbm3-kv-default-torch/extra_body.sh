#!/bin/sh
# Neural H100 leg body: confirm the NVIDIA attention default flip to
# stash_tiled_fgrid_r32_qres_pf_kvgrid_r32 (DEVIATION 2597 dk/dv on the round 3
# arm) and re-measure every torch column on the SAME pod.
#
# 1. tools/attention_step_leg.sh with the previous NVIDIA default as the
#    reference: the no-trial shipped fused check (gate.txt must name
#    DEFAULT column=nvidia arm=stash_tiled_fgrid_r32_qres_pf_kvgrid_r32), one
#    real-activation price, and the lean LM step of the old and new defaults on
#    enwik8 and Pile GitHub (the new one must read arm_is_default=True).
# 2. tools/torch_lm_step_opponent_leg.sh: all six torch columns, same corpora.
set -u
cd /root/mojolearn || exit 9
MOJOLEARN_ATTN_BASELINE=stash_tiled_fgrid_r32_qres_pf \
MOJOLEARN_ATTN_LEG_ARMS=stash_tiled_fgrid_r32_qres_pf_kvgrid_r32 \
MOJOLEARN_ATTN_LEG_LM_ARMS=stash_tiled_fgrid_r32_qres_pf,stash_tiled_fgrid_r32_qres_pf_kvgrid_r32 \
MOJOLEARN_ATTN_LEG_SHIPPED_CHECK=1 \
MOJOLEARN_ATTN_LEG_SKIP_TIMERS=1 \
MOJOLEARN_COMPILE_JOBS=8 \
    sh tools/attention_step_leg.sh
a=$?
echo "attention_leg_exit=$a" >> /root/gemm_leg_out/leg.txt
sh tools/torch_lm_step_opponent_leg.sh
t=$?
echo "torch_leg_exit=$t" >> /root/gemm_leg_out/leg.txt
[ "$a" = 0 ] && [ "$t" = 0 ]
