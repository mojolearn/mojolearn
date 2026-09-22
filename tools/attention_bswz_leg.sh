#!/bin/sh
# tools/attention_bswz_leg.sh -- the attention-speed lane's on-box body:
# DEVIATION 2900 (`_bswz`), the causal block-index map of the four kernels the
# shipped estash arm runs, priced against that arm on the same pod and the
# same heat window.
#
# The arm changes NO arithmetic: `_blk_map` hands the same set of
# (tile, head, batch) triples to a different `block_idx.x`, so the heaviest
# blocks are dispatched first instead of last. Every chain's terms, their
# order, the visibility tests and every staged operand are the shipped ones,
# so the gate is bit equality on both corpora plus reach by sabotage, and a
# ratio is the only thing that can move.
#
# A MOJOLEARN_GEMM_LEG_EXTRA body. tools/gemm_remote_leg.sh has no extra-env
# plumbing, so the settings live here (that leg copies this file into the
# evidence as extra_body.sh); on a runner that passes an environment every
# setting below can be overridden by name.
#
# NVIDIA (RunPod), from a `git worktree add --detach` checkout at the lane's
# commit:
#
#   MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
#   MOJOLEARN_GPU_ARCHS=sm_90a \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/attention_bswz_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-nvidia-h100-attention-bswz \
#   sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 75 \
#       --gpu "NVIDIA H100 80GB HBM3"
#
# What it runs (tools/attention_step_leg.sh): the arms check (25 arms, the two
# `_bswz` arms with REACH_E), one smoke, one real-activation price against the
# shipped default with the RESOURCES readback, the shipped fused check (no
# trial define, the control that names the column default), and the lean LM
# step of the default and the arm on enwik8 and Pile GitHub.
#
# Gates: brief section 22. Flip (CONTRIBUTING.md (Performance claims)): the geometric mean of
# the enwik8 and pilegithub lean step ratios (arm over the default,
# lm_summary.tsv steady medians, same pod) below 1, with
# witnesses_equal_baseline=True for every step on both corpora, and every
# BITS line MATCH on both corpora's real activations.
#
# POSIX sh only: RunPod's Ubuntu images link /bin/sh to dash.
set -u
cd /root/mojolearn || exit 9
DEF=stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32
MOJOLEARN_ATTN_BASELINE=${MOJOLEARN_ATTN_BASELINE:-$DEF}
MOJOLEARN_ATTN_LEG_ARMS=${MOJOLEARN_ATTN_LEG_ARMS:-${DEF}_bswz}
MOJOLEARN_ATTN_LEG_LM_ARMS=${MOJOLEARN_ATTN_LEG_LM_ARMS:-$DEF,${DEF}_bswz}
MOJOLEARN_ATTN_LEG_SKIP_TIMERS=${MOJOLEARN_ATTN_LEG_SKIP_TIMERS:-1}
MOJOLEARN_ATTN_LEG_SHIPPED_CHECK=${MOJOLEARN_ATTN_LEG_SHIPPED_CHECK:-1}
MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-8}
export MOJOLEARN_ATTN_BASELINE MOJOLEARN_ATTN_LEG_ARMS MOJOLEARN_ATTN_LEG_LM_ARMS
export MOJOLEARN_ATTN_LEG_SKIP_TIMERS MOJOLEARN_ATTN_LEG_SHIPPED_CHECK
export MOJOLEARN_COMPILE_JOBS
# The baseline arm's LM run is the witness reference lm_summary.tsv compares
# every other arm with; without it no witness verdict can be printed.
case ",$MOJOLEARN_ATTN_LEG_LM_ARMS," in
    *",$MOJOLEARN_ATTN_BASELINE,"*) ;;
    *) echo "attention_bswz_leg: MOJOLEARN_ATTN_LEG_LM_ARMS ($MOJOLEARN_ATTN_LEG_LM_ARMS) must include the baseline arm $MOJOLEARN_ATTN_BASELINE" >&2
       exit 9 ;;
esac
sh tools/attention_step_leg.sh
a=$?
if [ -f /root/gemm_leg_out/leg.txt ]; then
    echo "attention_bswz_leg_exit=$a" >> /root/gemm_leg_out/leg.txt
fi
exit "$a"
