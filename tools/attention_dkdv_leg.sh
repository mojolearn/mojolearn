#!/bin/sh
# tools/attention_dkdv_leg.sh -- the attention dk/dv AMD lane's on-box body:
# DEVIATION 2596 (`_kvrecompute`: the preflushed zdot and dq, the stashes
# freed, dk/dv by the shipped recompute kernel) and DEVIATION 2597
# (`_kvgrid_r32` / `_kvgrid_r64`: the tiled dk/dv fold's keys per block;
# `_kvsplit`: that fold as two kernels), all on the round 3 arm
# stash_tiled_fgrid_r32_qres_pf, priced against `baseline`, the shipped AMD
# default (brief docs/lanes/BRIEF_attention_step_2026-09-11.md section 16).
# AMD is the deciding column (ENGINEERING_RULES 10).
#
# A MOJOLEARN_GEMM_LEG_EXTRA body for tools/hotaisle_leg.sh,
# tools/do_extra_leg.sh or tools/gemm_remote_leg.sh. Every setting below is a
# default taken only when unset, so MOJOLEARN_HOTAISLE_EXTRA_ENV or
# MOJOLEARN_DO_EXTRA_ENV can override any of them by name
# (MOJOLEARN_ATTN_LEG_SKIP_TIMERS=0 adds the standalone per-kernel breakdown,
# the direct read of the step-versus-harness gap, brief 16.3).
# tools/gemm_remote_leg.sh passes no environment and does not export the
# arch, so MOJOLEARN_GPU_ARCHS defaults to gfx942 (the MI300X and MI325X)
# here; the Hot Aisle and DigitalOcean runners export their own, which wins.
#
# First `tools/pick_box.sh --need amd`; for `hotaisle`:
#
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/attention_dkdv_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-amd-mi300x-hotaisle-attention-dkdv \
#   bash tools/hotaisle_leg.sh amd --rent --minutes 60 --skip-gates
#
# `do` and `runpod-amd`: brief section 16.8.
#
# Gates: brief section 6 against `baseline`, plus REACH_KV proven for every
# kv arm. Flip: ENGINEERING_RULES 9, the geometric mean of the enwik8 and
# pilegithub lean step ratios (arm over baseline, lm_summary.tsv steady
# medians) below 1, and witnesses_equal_baseline=True for every arm and
# corpus.
#
# POSIX sh only: RunPod's Ubuntu images link /bin/sh to dash.
set -u
cd /root/mojolearn || exit 9
MOJOLEARN_ATTN_BASELINE=${MOJOLEARN_ATTN_BASELINE:-baseline}
MOJOLEARN_ATTN_LEG_ARMS=${MOJOLEARN_ATTN_LEG_ARMS:-stash_tiled_fgrid_r32_qres_pf_kvrecompute,stash_tiled_fgrid_r32_qres_pf,stash_tiled_fgrid_r32_qres_pf_kvsplit,stash_tiled_fgrid_r32_qres_pf_kvgrid_r32,stash_tiled_fgrid_r32_qres_pf_kvgrid_r64,stash_tiled_fgrid_r32_qres_pf_kvgrid_r32_kvsplit}
MOJOLEARN_ATTN_LEG_LM_ARMS=${MOJOLEARN_ATTN_LEG_LM_ARMS:-baseline,stash_tiled_fgrid_r32_qres_pf_kvrecompute,stash_tiled_fgrid_r32_qres_pf,stash_tiled_fgrid_r32_qres_pf_kvsplit,stash_tiled_fgrid_r32_qres_pf_kvgrid_r32}
MOJOLEARN_ATTN_LEG_SKIP_TIMERS=${MOJOLEARN_ATTN_LEG_SKIP_TIMERS:-1}
MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-8}
MOJOLEARN_GPU_ARCHS=${MOJOLEARN_GPU_ARCHS:-gfx942}
export MOJOLEARN_ATTN_BASELINE MOJOLEARN_ATTN_LEG_ARMS MOJOLEARN_ATTN_LEG_LM_ARMS
export MOJOLEARN_ATTN_LEG_SKIP_TIMERS MOJOLEARN_COMPILE_JOBS MOJOLEARN_GPU_ARCHS
# The baseline arm's LM run is the witness reference lm_summary.tsv compares
# every other arm with; without it no witness verdict can be printed.
case ",$MOJOLEARN_ATTN_LEG_LM_ARMS," in
    *",$MOJOLEARN_ATTN_BASELINE,"*) ;;
    *) echo "attention_dkdv_leg: MOJOLEARN_ATTN_LEG_LM_ARMS ($MOJOLEARN_ATTN_LEG_LM_ARMS) must include the baseline arm $MOJOLEARN_ATTN_BASELINE" >&2
       exit 9 ;;
esac
sh tools/attention_step_leg.sh
a=$?
if [ -f /root/gemm_leg_out/leg.txt ]; then
    echo "attention_dkdv_leg_exit=$a" >> /root/gemm_leg_out/leg.txt
fi
exit "$a"
