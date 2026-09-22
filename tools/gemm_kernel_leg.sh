#!/bin/sh
# tools/gemm_kernel_leg.sh -- DEVIATION 2599, the GEMM kernel lane's on-box work.
# A thin wrapper, the pattern of tools/gemm_longk_leg.sh: it names the arms
# `kpack` and `kpack_wide`, turns on the per-component LM timing, then runs
# tools/gemm_step_leg.sh (DEVIATION 2544), which builds and runs the step arms check, the resources instrument,
# the price runs, the trial bindings and the lean LM step on enwik8 and the
# Pile GitHub component, bracketed by the shipped default.
#
# VENDOR-AGNOSTIC. It runs ON THE BOX as the MOJOLEARN_GEMM_LEG_EXTRA body of
# tools/gemm_remote_leg.sh (RunPod), which copies it to /root/gemm_leg_extra.sh,
# so it changes into the source root (MOJOLEARN_GEMM_STEP_ROOT, default
# /root/mojolearn) itself. RunPod passes no extra environment to the body, so
# the defaults below are the H100 leg.
#
# NVIDIA H100 (RunPod), from a `git worktree add --detach` checkout of the
# commit that carries DEVIATION 2599, after the brief's RUN OWED is green:
#
#   tools/gemm_card.sh device /tmp/gemm-kernel-apple.card
#   MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key MOJOLEARN_GPU_ARCHS=sm_90a \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/gemm_kernel_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-nvidia-h100-gemm-kernel \
#   sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 \
#       --gpu "NVIDIA H100 80GB HBM3" --local-card /tmp/gemm-kernel-apple.card
#
# WHAT IT EXPORTS. A value the runner already exported wins, so the arms can be
# narrowed without editing this file.
#   MOJOLEARN_GEMM_STEP_LEG_ARMS=shipped,kpack,kpack_wide   price runs (shipped
#                                  is the shipped-against-shipped noise control)
#   MOJOLEARN_GEMM_STEP_LEG_LM_ARMS=kpack,kpack_wide        LM probe arms
#   MOJOLEARN_GEMM_STEP_LEG_LMTIMING=1                      lmtiming-* probes
#   MOJOLEARN_GEMM_STEP_LEG_CHECK_ARMS=shipped,kpack,kpack_wide
#                                  the check's LM section (fits the lease; the
#                                  ragged part still forces every geometry)
#   MOJOLEARN_GEMM_STEP_LEG_OUT=/root/gemm_leg_out/gemm-kernel comes home with
#                                  the runner's fetch of /root/gemm_leg_out
#
# `shipped` IS ALWAYS THE LM BRACKET AND MUST NOT BE THE ONLY LM ARM.
# tools/gemm_step_leg.sh brackets every corpus with lm-shipped-<corpus> and
# lm-shippedclose-<corpus> itself and DROPS `shipped` from the LM arm list, so
# an LM arm list of `shipped` alone resolves to none and no LM probe runs. This
# wrapper refuses that list before anything is built.
#
# The flip rule is the step leg's (CONTRIBUTING.md (Performance claims)): the geometric mean of
# the enwik8 and pilegithub lean step ratios against the shipped default below
# 1 on the same pod, with every step witness equal to shipped on both corpora
# (lm_summary.tsv verdict lines). A flip changes only the NVIDIA row.
#
# POSIX sh only: RunPod's Ubuntu images link /bin/sh to dash.
set -u
# DEVIATION 2700 (brief section 13): `kpack_pad` against `shipped`, with `kpack`
# priced on the same pod as the unpadded CONTROL (its 2599 reading was on
# another pod). The LM probe runs both, the pair being the same-pod A/B.
# DEVIATION 2703 (brief section 15): `kpack_padv` (aligned page, vector loads)
# against `kpack_pad` (the same page, scalar loads) and `shipped`, same pod.
# DEVIATION 2706 (brief section 17): the hardware fold flush (`kpack_hf`), the
# gather staging (`kpack_gs`) and both (`kpack_hg`) against `kpack_padv` (their
# base) and `shipped`, same pod.
# DEVIATION 2707 (brief section 18), the SHIP gate: `shipped` is now the kpack_hg
# body on NVIDIA; `ksplit` is exactly the 2595 default it replaced (the same-pod
# OLD reference, expected above 1), and `kpack_hg` is the arm it was flipped
# from (expected 1.00: the proof the shipped path IS that body).
MOJOLEARN_GEMM_STEP_LEG_ARMS=${MOJOLEARN_GEMM_STEP_LEG_ARMS:-shipped,ksplit,kpack_hg}
MOJOLEARN_GEMM_STEP_LEG_LM_ARMS=${MOJOLEARN_GEMM_STEP_LEG_LM_ARMS:-ksplit,kpack_hg}
MOJOLEARN_GEMM_STEP_LEG_LMTIMING=${MOJOLEARN_GEMM_STEP_LEG_LMTIMING:-1}
MOJOLEARN_GEMM_STEP_LEG_CHECK_ARMS=${MOJOLEARN_GEMM_STEP_LEG_CHECK_ARMS:-shipped,ksplit,kpack_hg}
MOJOLEARN_GEMM_STEP_LEG_OUT=${MOJOLEARN_GEMM_STEP_LEG_OUT:-/root/gemm_leg_out/gemm-kernel}
export MOJOLEARN_GEMM_STEP_LEG_ARMS MOJOLEARN_GEMM_STEP_LEG_LM_ARMS \
    MOJOLEARN_GEMM_STEP_LEG_LMTIMING MOJOLEARN_GEMM_STEP_LEG_CHECK_ARMS \
    MOJOLEARN_GEMM_STEP_LEG_OUT

ROOT=${MOJOLEARN_GEMM_STEP_ROOT:-/root/mojolearn}
mkdir -p "$MOJOLEARN_GEMM_STEP_LEG_OUT"

lm_left=$(echo "$MOJOLEARN_GEMM_STEP_LEG_LM_ARMS" | tr ',' '\n' | grep -v '^shipped$' | grep -v '^$' | paste -sd, -)
if [ -z "$lm_left" ] && [ "$MOJOLEARN_GEMM_STEP_LEG_LM_ARMS" != auto ]; then
    echo "MOJOLEARN_GEMM_STEP_LEG_LM_ARMS=$MOJOLEARN_GEMM_STEP_LEG_LM_ARMS names no arm but shipped, which is always the bracket; nothing run" \
        > "$MOJOLEARN_GEMM_STEP_LEG_OUT/kernel.txt"
    exit 9
fi

{
    echo "deviations=2599,2700,2703,2706,2707"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "root=$ROOT"
    echo "arms=$MOJOLEARN_GEMM_STEP_LEG_ARMS"
    echo "lm_arms=$MOJOLEARN_GEMM_STEP_LEG_LM_ARMS"
    echo "lmtiming=$MOJOLEARN_GEMM_STEP_LEG_LMTIMING"
    echo "check_arms=$MOJOLEARN_GEMM_STEP_LEG_CHECK_ARMS"
    echo "out=$MOJOLEARN_GEMM_STEP_LEG_OUT"
} > "$MOJOLEARN_GEMM_STEP_LEG_OUT/kernel.txt"

cd "$ROOT" || exit 9
[ -f tools/gemm_step_leg.sh ] || { echo "no tools/gemm_step_leg.sh under $ROOT; nothing run" >> "$MOJOLEARN_GEMM_STEP_LEG_OUT/kernel.txt"; exit 9; }
exec sh tools/gemm_step_leg.sh
