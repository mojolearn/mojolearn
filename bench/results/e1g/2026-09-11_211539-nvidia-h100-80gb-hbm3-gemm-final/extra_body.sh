#!/bin/sh
# tools/gemm_final_leg.sh -- DEVIATIONS 2640 to 2642, the GEMM final lane's
# on-box work (docs/lanes/BRIEF_gemm_final_2026-09-11.md). A thin wrapper, the
# pattern of tools/gemm_kernel_leg.sh: it names the arms `kfoldv` and
# `kfoldv_leaf` (with `ksplit_leaf` as a price-only control that separates the
# rule from the fold), turns on the per-component LM timing, then runs
# tools/gemm_step_leg.sh (DEVIATION 2544): the step arms check, the resources
# instrument, the price runs (PHASE lines with phase_of=arm_kfold give the
# lane fold's fold_ms), the trial bindings and the lean LM step on enwik8 and
# the Pile GitHub component, bracketed by the shipped default.
#
# VENDOR-AGNOSTIC. It runs ON THE BOX as the MOJOLEARN_GEMM_LEG_EXTRA body of
# tools/gemm_remote_leg.sh (RunPod), which copies it to /root/gemm_leg_extra.sh,
# so it changes into the source root (MOJOLEARN_GEMM_STEP_ROOT, default
# /root/mojolearn) itself. RunPod passes no extra environment to the body, so
# the defaults below are the H100 leg.
#
# NVIDIA H100 (RunPod), from a `git worktree add --detach` checkout of the
# commit that carries DEVIATIONS 2640 to 2642, after the brief's RUN OWED is
# green:
#
#   tools/gemm_card.sh device /tmp/gemm-final-apple.card
#   MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key MOJOLEARN_GPU_ARCHS=sm_90a \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/gemm_final_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-nvidia-h100-gemm-final \
#   sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 \
#       --gpu "NVIDIA H100 80GB HBM3" --local-card /tmp/gemm-final-apple.card
#
# WHAT IT EXPORTS. A value the runner already exported wins.
#   MOJOLEARN_GEMM_STEP_LEG_ARMS=shipped,ksplit_leaf,kfoldv,kfoldv_leaf
#   MOJOLEARN_GEMM_STEP_LEG_LM_ARMS=kfoldv,kfoldv_leaf
#   MOJOLEARN_GEMM_STEP_LEG_LMTIMING=1
#   MOJOLEARN_GEMM_STEP_LEG_CHECK_ARMS=shipped,kfoldv,kfoldv_leaf
#   MOJOLEARN_GEMM_STEP_LEG_OUT=/root/gemm_leg_out/gemm-final
#
# `shipped` is always the LM bracket and must not be the only LM arm.
# The flip rule is the step leg's (ENGINEERING_RULES 9): the geometric mean of
# the enwik8 and pilegithub lean step ratios against the shipped default below
# 1 on the same pod, with every step witness equal to shipped on both corpora.
#
# POSIX sh only: RunPod's Ubuntu images link /bin/sh to dash.
set -u
MOJOLEARN_GEMM_STEP_LEG_ARMS=${MOJOLEARN_GEMM_STEP_LEG_ARMS:-shipped,ksplit_leaf,kfoldv,kfoldv_leaf}
MOJOLEARN_GEMM_STEP_LEG_LM_ARMS=${MOJOLEARN_GEMM_STEP_LEG_LM_ARMS:-kfoldv,kfoldv_leaf}
MOJOLEARN_GEMM_STEP_LEG_LMTIMING=${MOJOLEARN_GEMM_STEP_LEG_LMTIMING:-1}
MOJOLEARN_GEMM_STEP_LEG_CHECK_ARMS=${MOJOLEARN_GEMM_STEP_LEG_CHECK_ARMS:-shipped,kfoldv,kfoldv_leaf}
MOJOLEARN_GEMM_STEP_LEG_OUT=${MOJOLEARN_GEMM_STEP_LEG_OUT:-/root/gemm_leg_out/gemm-final}
export MOJOLEARN_GEMM_STEP_LEG_ARMS MOJOLEARN_GEMM_STEP_LEG_LM_ARMS \
    MOJOLEARN_GEMM_STEP_LEG_LMTIMING MOJOLEARN_GEMM_STEP_LEG_CHECK_ARMS \
    MOJOLEARN_GEMM_STEP_LEG_OUT

ROOT=${MOJOLEARN_GEMM_STEP_ROOT:-/root/mojolearn}
mkdir -p "$MOJOLEARN_GEMM_STEP_LEG_OUT"

lm_left=$(echo "$MOJOLEARN_GEMM_STEP_LEG_LM_ARMS" | tr ',' '\n' | grep -v '^shipped$' | grep -v '^$' | paste -sd, -)
if [ -z "$lm_left" ] && [ "$MOJOLEARN_GEMM_STEP_LEG_LM_ARMS" != auto ]; then
    echo "MOJOLEARN_GEMM_STEP_LEG_LM_ARMS=$MOJOLEARN_GEMM_STEP_LEG_LM_ARMS names no arm but shipped, which is always the bracket; nothing run" \
        > "$MOJOLEARN_GEMM_STEP_LEG_OUT/final.txt"
    exit 9
fi

{
    echo "deviations=2640-2642"
    echo "brief=docs/lanes/BRIEF_gemm_final_2026-09-11.md"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "root=$ROOT"
    echo "arms=$MOJOLEARN_GEMM_STEP_LEG_ARMS"
    echo "lm_arms=$MOJOLEARN_GEMM_STEP_LEG_LM_ARMS"
    echo "lmtiming=$MOJOLEARN_GEMM_STEP_LEG_LMTIMING"
    echo "check_arms=$MOJOLEARN_GEMM_STEP_LEG_CHECK_ARMS"
    echo "out=$MOJOLEARN_GEMM_STEP_LEG_OUT"
} > "$MOJOLEARN_GEMM_STEP_LEG_OUT/final.txt"

cd "$ROOT" || exit 9
[ -f tools/gemm_step_leg.sh ] || { echo "no tools/gemm_step_leg.sh under $ROOT; nothing run" >> "$MOJOLEARN_GEMM_STEP_LEG_OUT/final.txt"; exit 9; }
exec sh tools/gemm_step_leg.sh
