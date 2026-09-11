#!/bin/sh
# tools/gemm_longk_leg.sh -- DEVIATION 2594, the GEMM long-k lane's on-box work
# (docs/lanes/BRIEF_gemm_long_k_2026-09-11.md). A thin wrapper: it names the
# arms, turns on the CONTROL and PHASE lines of DEVIATION 2593 and the
# per-component LM timing, then runs tools/gemm_step_leg.sh (DEVIATION 2544),
# which builds and runs the check, the resources instrument, the price runs,
# the trial bindings and the lean LM step on enwik8 and the Pile GitHub
# component.
#
# DEVIATION 2595 (brief section 10). `ksplit` is now the SHIPPED DEFAULT on a
# column whose block parallelism row is above 0 (NVIDIA 132); AMD stays on the
# old TUNED 128x128 plan until the MI300X leg reads it. `shipped` in every
# line below is that default, and `tuned128` is the old plan as a trial arm.
# The defaults are the NVIDIA CONFIRMATION leg (LM `tuned128` against the new
# default). The AMD leg narrows them through the runner's extra-env knob
# (LM `ksplit` against `shipped`, which is still the old plan there).
#
# VENDOR-AGNOSTIC. It runs ON THE BOX as the MOJOLEARN_GEMM_LEG_EXTRA body of
# tools/gemm_remote_leg.sh (RunPod) or of a runner with the
# tools/do_extra_leg.sh body interface (tools/hotaisle_leg.sh). Those runners
# copy it to /root/gemm_leg_extra.sh, so it cannot find the source tree from
# its own path: it changes into the source root (MOJOLEARN_GEMM_STEP_ROOT,
# default /root/mojolearn) itself.
#
# NVIDIA H100 (RunPod), from a checkout of the commit that carries the flip
# (RunPod passes no extra environment to the body, so these defaults are
# the leg):
#
#   MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key MOJOLEARN_GPU_ARCHS=sm_90a \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/gemm_longk_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/<UTC stamp>-nvidia-h100-gemm-ksplit-default \
#   sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 \
#       --gpu "NVIDIA H100 80GB HBM3" --local-card <card>
#
# AMD MI300X (Hot Aisle):
#
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/gemm_longk_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/<UTC stamp>-amd-mi300x-hotaisle-gemm-ksplit-default \
#   MOJOLEARN_HOTAISLE_EXTRA_ENV="MOJOLEARN_GEMM_STEP_LEG_ARMS=shipped,ksplit,ksplit_leaf MOJOLEARN_GEMM_STEP_LEG_LM_ARMS=ksplit" \
#   bash tools/hotaisle_leg.sh amd --rent --minutes 60 --skip-gates
#
# WHAT IT EXPORTS. A value the runner already exported (for example through
# its extra-env knob) wins, so the arms can be narrowed without editing this
# file.
#   MOJOLEARN_GEMM_STEP_LEG_ARMS=shipped,tuned128,ksplit,ksplit_leaf
#                                  price runs (shipped is the
#                                  shipped-against-shipped noise control;
#                                  tuned128 prices the old plan against the
#                                  default)
#   MOJOLEARN_GEMM_STEP_LEG_LM_ARMS=tuned128                  LM probe arms
#   MOJOLEARN_GEMM_STEP_LEG_LMTIMING=1                        lmtiming-* probes
#   MOJOLEARN_GEMM_STEP_LEG_CHECK_ARMS=shipped,tuned128,ksplit,ksplit_leaf
#                                  the check's LM section (fits the lease)
#   MOJOLEARN_GEMM_STEP_CONTROLS=1                            CONTROL lines in
#                                  every price run (inherited by its env)
#   MOJOLEARN_GEMM_STEP_LEG_OUT=/root/gemm_leg_out/gemm-longk comes home with
#                                  the runner's fetch of /root/gemm_leg_out
#
# `shipped` IS ALWAYS THE LM BRACKET AND MUST NOT BE THE ONLY LM ARM.
# tools/gemm_step_leg.sh brackets every corpus with lm-shipped-<corpus> and
# lm-shippedclose-<corpus> itself and DROPS `shipped` from the LM arm list,
# so MOJOLEARN_GEMM_STEP_LEG_LM_ARMS=shipped alone resolves to none and no LM
# probe runs. This wrapper refuses that list before anything is built.
#
# The flip rule is the step leg's (ENGINEERING_RULES 9): the geometric mean
# of the enwik8 and pilegithub lean step ratios below 1, with every step
# witness equal to shipped on both corpora (lm_summary.tsv verdict lines,
# which since 2595 also name the arm's plan and the shipped plan). On the
# NVIDIA confirmation leg the expected reading is the mirror: `tuned128`
# NO FLIP with a geomean above 1.
#
# POSIX sh only: RunPod's Ubuntu images link /bin/sh to dash.
set -u
MOJOLEARN_GEMM_STEP_LEG_ARMS=${MOJOLEARN_GEMM_STEP_LEG_ARMS:-shipped,tuned128,ksplit,ksplit_leaf}
MOJOLEARN_GEMM_STEP_LEG_LM_ARMS=${MOJOLEARN_GEMM_STEP_LEG_LM_ARMS:-tuned128}
MOJOLEARN_GEMM_STEP_LEG_LMTIMING=${MOJOLEARN_GEMM_STEP_LEG_LMTIMING:-1}
MOJOLEARN_GEMM_STEP_LEG_CHECK_ARMS=${MOJOLEARN_GEMM_STEP_LEG_CHECK_ARMS:-shipped,tuned128,ksplit,ksplit_leaf}
MOJOLEARN_GEMM_STEP_CONTROLS=${MOJOLEARN_GEMM_STEP_CONTROLS:-1}
MOJOLEARN_GEMM_STEP_LEG_OUT=${MOJOLEARN_GEMM_STEP_LEG_OUT:-/root/gemm_leg_out/gemm-longk}
export MOJOLEARN_GEMM_STEP_LEG_ARMS MOJOLEARN_GEMM_STEP_LEG_LM_ARMS \
    MOJOLEARN_GEMM_STEP_LEG_LMTIMING MOJOLEARN_GEMM_STEP_LEG_CHECK_ARMS \
    MOJOLEARN_GEMM_STEP_CONTROLS MOJOLEARN_GEMM_STEP_LEG_OUT

ROOT=${MOJOLEARN_GEMM_STEP_ROOT:-/root/mojolearn}
mkdir -p "$MOJOLEARN_GEMM_STEP_LEG_OUT"

lm_left=$(echo "$MOJOLEARN_GEMM_STEP_LEG_LM_ARMS" | tr ',' '\n' | grep -v '^shipped$' | grep -v '^$' | paste -sd, -)
if [ -z "$lm_left" ] && [ "$MOJOLEARN_GEMM_STEP_LEG_LM_ARMS" != auto ]; then
    echo "MOJOLEARN_GEMM_STEP_LEG_LM_ARMS=$MOJOLEARN_GEMM_STEP_LEG_LM_ARMS names no arm but shipped, which is always the bracket; nothing run" \
        > "$MOJOLEARN_GEMM_STEP_LEG_OUT/longk.txt"
    exit 9
fi

{
    echo "deviations=2590-2595"
    echo "brief=docs/lanes/BRIEF_gemm_long_k_2026-09-11.md"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "root=$ROOT"
    echo "arms=$MOJOLEARN_GEMM_STEP_LEG_ARMS"
    echo "lm_arms=$MOJOLEARN_GEMM_STEP_LEG_LM_ARMS"
    echo "lmtiming=$MOJOLEARN_GEMM_STEP_LEG_LMTIMING"
    echo "check_arms=$MOJOLEARN_GEMM_STEP_LEG_CHECK_ARMS"
    echo "controls=$MOJOLEARN_GEMM_STEP_CONTROLS"
    echo "out=$MOJOLEARN_GEMM_STEP_LEG_OUT"
} > "$MOJOLEARN_GEMM_STEP_LEG_OUT/longk.txt"

cd "$ROOT" || exit 9
[ -f tools/gemm_step_leg.sh ] || { echo "no tools/gemm_step_leg.sh under $ROOT; nothing run" >> "$MOJOLEARN_GEMM_STEP_LEG_OUT/longk.txt"; exit 9; }
exec sh tools/gemm_step_leg.sh
