#!/bin/sh
# tools/attention_round3_leg.sh -- the third attention round's on-box body:
# DEVIATIONS 2533 (preflushed seams, `_pf`), 2531 (forward grid,
# `_fgrid_r32` / `_fgrid_r64`) and 2530 (forward Q residency, `_qres`),
# priced against the shipped default stash_tiled (brief
# docs/lanes/BRIEF_attention_step_2026-09-11.md section 14).
#
# A MOJOLEARN_GEMM_LEG_EXTRA body. tools/gemm_remote_leg.sh has no extra-env
# plumbing, so the settings live here (that leg copies this file into the
# evidence as extra_body.sh). On a runner that does pass an environment
# (tools/do_extra_leg.sh MOJOLEARN_DO_EXTRA_ENV, tools/hotaisle_leg.sh
# MOJOLEARN_HOTAISLE_EXTRA_ENV) every setting below can be overridden by
# name; an unset one takes the default written here.
#
# NVIDIA (RunPod, the measurement column while ENGINEERING_RULES 10's NVIDIA
# clause is in force):
#
#   MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
#   MOJOLEARN_GPU_ARCHS=sm_90a \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/attention_round3_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-nvidia-h100-attention-round3 \
#   sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 \
#       --gpu "NVIDIA H100 80GB HBM3"
#
# AMD (Hot Aisle MI300X, once tools/hotaisle_leg.sh is built): brief
# section 14.7.
#
# Gates: brief section 6 with stash_tiled in place of baseline. Flip:
# ENGINEERING_RULES 9, the geometric mean of the enwik8 and pilegithub lean
# step ratios (arm over stash_tiled, lm_summary.tsv steady medians) below 1,
# and witnesses_equal_baseline=True for every arm and corpus.
#
# POSIX sh only: RunPod's Ubuntu images link /bin/sh to dash.
set -u
cd /root/mojolearn || exit 9
MOJOLEARN_ATTN_BASELINE=${MOJOLEARN_ATTN_BASELINE:-stash_tiled}
MOJOLEARN_ATTN_LEG_ARMS=${MOJOLEARN_ATTN_LEG_ARMS:-stash_tiled_pf,stash_tiled_fgrid_r64,stash_tiled_fgrid_r32,stash_tiled_fgrid_r32_qres,stash_tiled_fgrid_r32_qres_pf}
MOJOLEARN_ATTN_LEG_LM_ARMS=${MOJOLEARN_ATTN_LEG_LM_ARMS:-stash_tiled,stash_tiled_pf,stash_tiled_fgrid_r32,stash_tiled_fgrid_r32_qres_pf}
MOJOLEARN_ATTN_LEG_SKIP_TIMERS=${MOJOLEARN_ATTN_LEG_SKIP_TIMERS:-1}
MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-8}
export MOJOLEARN_ATTN_BASELINE MOJOLEARN_ATTN_LEG_ARMS MOJOLEARN_ATTN_LEG_LM_ARMS
export MOJOLEARN_ATTN_LEG_SKIP_TIMERS MOJOLEARN_COMPILE_JOBS
# The baseline arm's LM run is the witness reference lm_summary.tsv compares
# every other arm with; without it no witness verdict can be printed.
case ",$MOJOLEARN_ATTN_LEG_LM_ARMS," in
    *",$MOJOLEARN_ATTN_BASELINE,"*) ;;
    *) echo "attention_round3_leg: MOJOLEARN_ATTN_LEG_LM_ARMS ($MOJOLEARN_ATTN_LEG_LM_ARMS) must include the baseline arm $MOJOLEARN_ATTN_BASELINE" >&2
       exit 9 ;;
esac
sh tools/attention_step_leg.sh
a=$?
if [ -f /root/gemm_leg_out/leg.txt ]; then
    echo "attention_round3_leg_exit=$a" >> /root/gemm_leg_out/leg.txt
fi
exit "$a"
