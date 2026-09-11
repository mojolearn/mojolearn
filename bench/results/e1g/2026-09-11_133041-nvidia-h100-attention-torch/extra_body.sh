#!/bin/sh
# Neural NVIDIA leg body (MOJOLEARN_GEMM_LEG_EXTRA for tools/gemm_remote_leg.sh,
# which has no extra-env plumbing, so the settings live here and this file is
# copied into the evidence as extra_body.sh).
#
# 1. tools/attention_step_leg.sh: price stash_tiled (the shipped default) and
#    DEVIATION 2528 at 64 and 32 rows against baseline; LM step for baseline,
#    stash_tiled and stash_tiled_ztiled_r64 on enwik8 and Pile GitHub.
# 2. tools/torch_lm_step_opponent_leg.sh on the SAME box, so the torch row and
#    our lean step share one GPU.
set -u
cd /root/mojolearn || exit 9
MOJOLEARN_ATTN_LEG_ARMS=stash_tiled,stash_tiled_ztiled_r64,stash_tiled_ztiled_r32 \
MOJOLEARN_ATTN_LEG_LM_ARMS=baseline,stash_tiled,stash_tiled_ztiled_r64 \
MOJOLEARN_ATTN_LEG_SKIP_TIMERS=1 \
MOJOLEARN_COMPILE_JOBS=8 \
    sh tools/attention_step_leg.sh
a=$?
echo "attention_leg_exit=$a" >> /root/gemm_leg_out/leg.txt
sh tools/torch_lm_step_opponent_leg.sh
t=$?
echo "torch_leg_exit=$t" >> /root/gemm_leg_out/leg.txt
[ "$a" = 0 ] && [ "$t" = 0 ]
