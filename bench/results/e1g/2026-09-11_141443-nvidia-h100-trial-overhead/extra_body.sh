#!/bin/sh
# Neural H100 leg body: one box, two trial bindings, the shipped arm in each.
# Attributes the 0.457 s (GEMM trial binding, shipped arm) against 0.383 s
# (attention trial binding, stash_tiled) seen on two different pods on
# 2026-09-11: same box here, so a remaining gap belongs to the build.
set -u
cd /root/mojolearn || exit 9
MOJOLEARN_ATTN_LEG_ARMS=stash_tiled \
MOJOLEARN_ATTN_LEG_LM_ARMS=stash_tiled \
MOJOLEARN_ATTN_LEG_SKIP_TIMERS=1 \
MOJOLEARN_COMPILE_JOBS=8 \
    sh tools/attention_step_leg.sh
a=$?
echo "attention_leg_exit=$a" >> /root/gemm_leg_out/leg.txt
MOJOLEARN_GEMM_STEP_LEG_ARMS=shipped \
MOJOLEARN_GEMM_STEP_LEG_LM_ARMS=shipped \
MOJOLEARN_GEMM_STEP_LEG_SKIP_RESOURCES=1 \
MOJOLEARN_COMPILE_JOBS=8 \
    sh tools/gemm_step_leg.sh
g=$?
echo "gemm_leg_exit=$g" >> /root/gemm_leg_out/leg.txt
[ "$a" = 0 ] && [ "$g" = 0 ]
