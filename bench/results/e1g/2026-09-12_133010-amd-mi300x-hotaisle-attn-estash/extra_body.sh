#!/bin/sh
# AMD CONFIRMATION FOR DEVIATION 2657 (the attention estash flip), MI300X.
#
# 2657 flipped the NVIDIA column's attention default to
# stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32 on an H100 leg. The
# AMD column's default was NOT touched: it is still
# stash_tiled_fgrid_r32_qres_pf_kvgrid_r32 (kernel matrix
# `attn_default_arm_for`, brief section 18). So this leg answers two separate
# questions and must not blur them:
#
#   a. DID THE MERGE DISTURB AMD? The shipped (no trial define)
#      transformer_fused_check must still pass on this board and its DEFAULT
#      line must still name the AMD default, unchanged. That is a BITS
#      question and it is the one that matters.
#   b. SHOULD AMD FLIP TOO? The estash arm is priced against the AMD default
#      on both corpora. The flip rule (ENGINEERING_RULES section 9) wants the
#      geometric mean of the two lean step ratios below 1 with witnesses
#      equal. Nothing here flips anything by itself; the number decides and a
#      losing number is a result, not a failure.
#
# The AMD arm name is the AMD default plus the estash tokens, in the order
# fused_attention.mojo's parser requires (`_estash` or `_estash_dres`, then
# the kv tokens), which spells the same word the NVIDIA default carries.
#
# A MOJOLEARN_GEMM_LEG_EXTRA body for tools/hotaisle_leg.sh.
# POSIX sh only.
set -u
cd /root/mojolearn || exit 9
OUTROOT=/root/gemm_leg_out

AMD_DEFAULT=stash_tiled_fgrid_r32_qres_pf_kvgrid_r32
AMD_ESTASH=stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32

MOJOLEARN_ATTN_BASELINE=$AMD_DEFAULT \
MOJOLEARN_ATTN_LEG_ARMS=$AMD_ESTASH \
MOJOLEARN_ATTN_LEG_LM_ARMS=$AMD_DEFAULT,$AMD_ESTASH \
MOJOLEARN_ATTN_LEG_SHIPPED_CHECK=1 \
MOJOLEARN_ATTN_LEG_SKIP_TIMERS=1 \
MOJOLEARN_COMPILE_JOBS=8 \
    sh tools/attention_step_leg.sh
a=$?
echo "attention_leg_exit=$a" >> $OUTROOT/leg.txt
echo "amd_default=$AMD_DEFAULT amd_arm=$AMD_ESTASH" >> $OUTROOT/leg.txt
exit "$a"
