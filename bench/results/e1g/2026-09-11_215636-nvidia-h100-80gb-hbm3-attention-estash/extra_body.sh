#!/bin/sh
# tools/attention_final_leg.sh -- the attention final H100 lane's on-box body:
# DEVIATIONS 2650 (`_estash`: the backward's zdot kernel reads the exp stash
# the forward kept, eight rows per block, no K staging, no score chain, no
# exp) and 2651 (`_estash_dres`: the same with the block's dctx rows in the
# shared page), on the shipped NVIDIA default
# stash_tiled_fgrid_r32_qres_pf_kvgrid_r32 and priced against it (brief
# docs/lanes/BRIEF_attention_step_2026-09-11.md section 20). The kept stash is
# DEVIATION 2652 (one host field on LlamaDeviceStages, trial builds only).
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
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/attention_final_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-nvidia-h100-attention-final \
#   sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 \
#       --gpu "NVIDIA H100 80GB HBM3" \
#       --local-card bench/results/e1/2026-08-28_131651-runpod-nvidia/lanes/gemm.identical.card
#
# What it runs (tools/attention_step_leg.sh): the arms check (23 arms, the
# estash arms with REACH_E), two smokes, two real-activation prices against
# the default (the first with the RESOURCES readback, both with REACH_ES and
# REACH_E lines), and the lean LM step of the default and both arms on
# enwik8 and Pile GitHub (12 probes).
#
# Gates: brief section 20.9. Flip (brief 20.7, ENGINEERING_RULES 9): the
# geometric mean of the enwik8 and pilegithub lean step ratios (arm over the
# default, lm_summary.tsv steady medians, same pod) below 1, with
# witnesses_equal_baseline=True for every step on both corpora. A flip changes
# only the NVIDIA row, after the lane adds the shipped branch.
#
# POSIX sh only: RunPod's Ubuntu images link /bin/sh to dash.
set -u
cd /root/mojolearn || exit 9
MOJOLEARN_ATTN_BASELINE=${MOJOLEARN_ATTN_BASELINE:-stash_tiled_fgrid_r32_qres_pf_kvgrid_r32}
MOJOLEARN_ATTN_LEG_ARMS=${MOJOLEARN_ATTN_LEG_ARMS:-stash_tiled_fgrid_r32_qres_pf_estash_kvgrid_r32,stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32}
MOJOLEARN_ATTN_LEG_LM_ARMS=${MOJOLEARN_ATTN_LEG_LM_ARMS:-stash_tiled_fgrid_r32_qres_pf_kvgrid_r32,stash_tiled_fgrid_r32_qres_pf_estash_kvgrid_r32,stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32}
MOJOLEARN_ATTN_LEG_SKIP_TIMERS=${MOJOLEARN_ATTN_LEG_SKIP_TIMERS:-1}
MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-8}
export MOJOLEARN_ATTN_BASELINE MOJOLEARN_ATTN_LEG_ARMS MOJOLEARN_ATTN_LEG_LM_ARMS
export MOJOLEARN_ATTN_LEG_SKIP_TIMERS MOJOLEARN_COMPILE_JOBS
# The baseline arm's LM run is the witness reference lm_summary.tsv compares
# every other arm with; without it no witness verdict can be printed.
case ",$MOJOLEARN_ATTN_LEG_LM_ARMS," in
    *",$MOJOLEARN_ATTN_BASELINE,"*) ;;
    *) echo "attention_final_leg: MOJOLEARN_ATTN_LEG_LM_ARMS ($MOJOLEARN_ATTN_LEG_LM_ARMS) must include the baseline arm $MOJOLEARN_ATTN_BASELINE" >&2
       exit 9 ;;
esac
sh tools/attention_step_leg.sh
a=$?
if [ -f /root/gemm_leg_out/leg.txt ]; then
    echo "attention_final_leg_exit=$a" >> /root/gemm_leg_out/leg.txt
fi
exit "$a"
