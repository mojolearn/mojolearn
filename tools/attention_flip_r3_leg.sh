#!/bin/sh
# tools/attention_flip_r3_leg.sh -- the on-box body of the confirmation leg
# for DEVIATION 2534: `stash_tiled_fgrid_r32_qres_pf` is NVIDIA's shipped
# attention default (kernel matrix `attn_default_arm_for`), priced against
# `stash_tiled`, with the no-trial fused check proving the shipped path runs
# the default and is bit-identical to eager (brief
# docs/lanes/BRIEF_attention_step_2026-09-11.md section 15).
#
# A MOJOLEARN_GEMM_LEG_EXTRA body. tools/gemm_remote_leg.sh has no extra-env
# plumbing, so the settings live here (that leg copies this file into the
# evidence as extra_body.sh); on a runner that passes an environment every
# setting below can be overridden by name.
#
# NVIDIA (RunPod), from a `git worktree add --detach` checkout at the merge:
#
#   MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
#   MOJOLEARN_GPU_ARCHS=sm_90a \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/attention_flip_r3_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-nvidia-h100-attention-flip-r3 \
#   sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 \
#       --gpu "NVIDIA H100 80GB HBM3"
#
# What it runs (tools/attention_step_leg.sh): the arms check, one smoke and
# one real-activation price of the new default against stash_tiled, the
# shipped (no trial define) transformer_fused_check, whose DEFAULT line and
# PASS line land in gate.txt, and the lean LM step of stash_tiled and the new
# default on enwik8 and Pile GitHub. Every lm result.json names the arm the
# binding ran (`attention_arm`), the column default and the raw request.
#
# Gates: brief section 6 with stash_tiled in place of baseline, plus
# shipped-fused-check exit 0 with `DEFAULT column=nvidia
# arm=stash_tiled_fgrid_r32_qres_pf` and `resolved_hd64` the same name. The
# flip holds when the geometric mean of the enwik8 and pilegithub lean step
# ratios (default over stash_tiled) stays below 1 with
# witnesses_equal_baseline=True on both corpora.
#
# POSIX sh only: RunPod's Ubuntu images link /bin/sh to dash.
set -u
cd /root/mojolearn || exit 9
MOJOLEARN_ATTN_BASELINE=${MOJOLEARN_ATTN_BASELINE:-stash_tiled}
MOJOLEARN_ATTN_LEG_ARMS=${MOJOLEARN_ATTN_LEG_ARMS:-stash_tiled_fgrid_r32_qres_pf}
MOJOLEARN_ATTN_LEG_LM_ARMS=${MOJOLEARN_ATTN_LEG_LM_ARMS:-stash_tiled,stash_tiled_fgrid_r32_qres_pf}
MOJOLEARN_ATTN_LEG_SKIP_TIMERS=${MOJOLEARN_ATTN_LEG_SKIP_TIMERS:-1}
MOJOLEARN_ATTN_LEG_SHIPPED_CHECK=${MOJOLEARN_ATTN_LEG_SHIPPED_CHECK:-1}
MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-8}
export MOJOLEARN_ATTN_BASELINE MOJOLEARN_ATTN_LEG_ARMS MOJOLEARN_ATTN_LEG_LM_ARMS
export MOJOLEARN_ATTN_LEG_SKIP_TIMERS MOJOLEARN_ATTN_LEG_SHIPPED_CHECK MOJOLEARN_COMPILE_JOBS
# The baseline arm's LM run is the witness reference lm_summary.tsv compares
# every other arm with; without it no witness verdict can be printed.
case ",$MOJOLEARN_ATTN_LEG_LM_ARMS," in
    *",$MOJOLEARN_ATTN_BASELINE,"*) ;;
    *) echo "attention_flip_r3_leg: MOJOLEARN_ATTN_LEG_LM_ARMS ($MOJOLEARN_ATTN_LEG_LM_ARMS) must include the baseline arm $MOJOLEARN_ATTN_BASELINE" >&2
       exit 9 ;;
esac
sh tools/attention_step_leg.sh
a=$?
if [ -f /root/gemm_leg_out/leg.txt ]; then
    echo "attention_flip_r3_leg_exit=$a" >> /root/gemm_leg_out/leg.txt
fi
exit "$a"
