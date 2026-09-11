#!/bin/sh
# Neural H100 leg body, second try: one box, two trial bindings, the shipped
# step in each. The first try (e1g/2026-09-11_141443) never built the GEMM
# binding because tools/gemm_step_leg.sh drops `shipped` from LM_ARMS (it is
# always the bracket), so LM_ARMS=shipped resolved to none. `quarter` forces the
# build and the shipped bracket rides along. Component timing on both.
set -u
cd /root/mojolearn || exit 9
MOJOLEARN_ATTN_LEG_ARMS=stash_tiled \
MOJOLEARN_ATTN_LEG_LM_ARMS=stash_tiled \
MOJOLEARN_ATTN_LEG_SKIP_TIMERS=1 \
MOJOLEARN_COMPILE_JOBS=8 \
    sh tools/attention_step_leg.sh
a=$?
echo "attention_leg_exit=$a" >> /root/gemm_leg_out/leg.txt
MOJOLEARN_GEMM_STEP_LEG_ARMS=shipped,quarter \
MOJOLEARN_GEMM_STEP_LEG_LM_ARMS=quarter \
MOJOLEARN_GEMM_STEP_LEG_LMTIMING=1 \
MOJOLEARN_GEMM_STEP_LEG_SKIP_RESOURCES=1 \
MOJOLEARN_COMPILE_JOBS=8 \
    sh tools/gemm_step_leg.sh
g=$?
echo "gemm_leg_exit=$g" >> /root/gemm_leg_out/leg.txt
[ "$a" = 0 ] && [ "$g" = 0 ]
