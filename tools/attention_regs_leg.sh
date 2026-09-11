#!/bin/sh
# tools/attention_regs_leg.sh -- the attention REGISTER PRESSURE lane's
# on-box body (DEVIATIONS 2653 and 2654; brief
# docs/lanes/BRIEF_attention_regs_2026-09-11.md).
#
# WHAT THIS LANE FOUND, so the body is read for what it is. Section 20.2 of
# docs/lanes/BRIEF_attention_step_2026-09-11.md reads the H100 as
# `floor(256 / pad8(regs))` 256-thread blocks per SM, and every attention
# kernel of the step has a register readback EXCEPT the forward. The backward
# dq kernel is already at two blocks per SM (118 registers) and brief section
# 3 proves no mechanism can take it to three, so this lane writes NO ARM. It
# writes the missing number instead: six forward RESOURCES rows (DEVIATION
# 2653), which launch nothing and compile no kernel a trial build does not
# already compile. Brief section 4's table says what each value licenses the
# next lane to build.
#
# A MOJOLEARN_GEMM_LEG_EXTRA body, the shape tools/attention_final_leg.sh
# has. tools/gemm_remote_leg.sh has no extra-env plumbing, so the settings
# live here (that leg copies this file into the evidence as extra_body.sh);
# on a runner that passes an environment every setting below can be
# overridden by name.
#
# NVIDIA (RunPod), from a `git worktree add --detach` checkout at the merge:
#
#   MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
#   MOJOLEARN_GPU_ARCHS=sm_90a \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/attention_regs_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-nvidia-h100-attention-regs \
#   sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 \
#       --gpu "NVIDIA H100 80GB HBM3" \
#       --local-card bench/results/e1/2026-08-28_131651-runpod-nvidia/lanes/gemm.identical.card
#
# TWO PHASES.
#
#   1. THE READBACK, standalone and FIRST, so the one number this lane needs
#      comes home even if the lease dies later. One build of
#      bench/attention_step_price_main.mojo under IDENTICAL with the trial
#      define, then one run at L 512 with MOJOLEARN_ATTN_RESOURCES=1,
#      MOJOLEARN_ATTN_TIMING=0, MOJOLEARN_ATTN_ORACLE=0 and
#      MOJOLEARN_ATTN_REACH=0, candidate and baseline both the NVIDIA default
#      by name. Nothing is timed, no corpus is fetched, no binding is built:
#      about four minutes. The RESOURCES lines land in
#      $OUT/resources.txt and are echoed at the end.
#
#   2. THE FULL BODY (MOJOLEARN_ATTN_REGS_FULL=1, the default):
#      tools/attention_step_leg.sh with the NVIDIA default
#      stash_tiled_fgrid_r32_qres_pf_kvgrid_r32 as the baseline, the only
#      priced arm and the only LM arm, both corpora, and THE TIMERS ON
#      (MOJOLEARN_ATTN_LEG_SKIP_TIMERS=0). It brings home the per-kernel
#      breakdown (timers_summary.tsv, the lmtiming-* component lines) beside
#      the register rows, so attn.fwd_r2_kernel at 20.7 ms and
#      attn.bwd_dq_tiled_pf at 12.5 ms are read on the same pod as the counts
#      that explain them. There is no candidate arm on this branch, so the
#      price compares the default with itself and its ratio is 1.0 by
#      construction; RESOURCES, timers and lmtiming are what it is for.
#      MOJOLEARN_ATTN_REGS_FULL=0 runs phase 1 alone, which fits inside any
#      NVIDIA lease that is already renting for something else.
#
# GATES (brief section 8). Phase 1 exits 0; resources.txt carries a
# RESOURCES_BEGIN and a `regs=` and `blocks_per_sm_256=` line for each of
# fwd_r2_r32_qres_pf, fwd_r2_r32_qres, fwd_r2_r32_pf, fwd_r2_r32,
# fwd_r2_r64_pf and fwd_r2_r64 (a vendor that refuses an attribute prints
# RESOURCES_ERROR for that row and the rest stand); the run ends
# `attention_step_price: PASS`. Phase 2 is section 6 of the attention step
# brief with the NVIDIA default in place of `baseline`.
#
# THERE IS NO FLIP. This lane changes no kernel and no default, so
# ENGINEERING_RULES 9 has nothing to decide here; the product is the six
# register rows and brief section 4's reading of them.
#
# POSIX sh only: RunPod's Ubuntu images link /bin/sh to dash.
set -u
ROOT=${MOJOLEARN_ATTN_ROOT:-/root/mojolearn}
cd "$ROOT" || exit 9
OUT=${MOJOLEARN_ATTN_REGS_OUT:-/root/gemm_leg_out/attention-regs}
JOBS=${MOJOLEARN_COMPILE_JOBS:-8}
DEADLINE=${MOJOLEARN_ATTN_REGS_DEADLINE:-300}
FULL=${MOJOLEARN_ATTN_REGS_FULL:-1}
DEFAULT_ARM=stash_tiled_fgrid_r32_qres_pf_kvgrid_r32

MOJOLEARN_ATTN_BASELINE=${MOJOLEARN_ATTN_BASELINE:-$DEFAULT_ARM}
MOJOLEARN_ATTN_LEG_ARMS=${MOJOLEARN_ATTN_LEG_ARMS:-$DEFAULT_ARM}
MOJOLEARN_ATTN_LEG_LM_ARMS=${MOJOLEARN_ATTN_LEG_LM_ARMS:-$DEFAULT_ARM}
MOJOLEARN_ATTN_LEG_SKIP_TIMERS=${MOJOLEARN_ATTN_LEG_SKIP_TIMERS:-0}
MOJOLEARN_COMPILE_JOBS=$JOBS
export MOJOLEARN_ATTN_BASELINE MOJOLEARN_ATTN_LEG_ARMS MOJOLEARN_ATTN_LEG_LM_ARMS
export MOJOLEARN_ATTN_LEG_SKIP_TIMERS MOJOLEARN_COMPILE_JOBS

mkdir -p "$OUT/bin"
export PATH="$HOME/.pixi/bin:$PATH"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export MOJOLEARN_NUMERIC_MODE=identical

# The baseline arm's LM run is the witness reference lm_summary.tsv compares
# every other arm with; without it no witness verdict can be printed.
case ",$MOJOLEARN_ATTN_LEG_LM_ARMS," in
    *",$MOJOLEARN_ATTN_BASELINE,"*) ;;
    *) echo "attention_regs_leg: MOJOLEARN_ATTN_LEG_LM_ARMS ($MOJOLEARN_ATTN_LEG_LM_ARMS) must include the baseline arm $MOJOLEARN_ATTN_BASELINE" >&2
       exit 9 ;;
esac

# THE VENDOR AND THE ARCH, for phase 1 only (phase 2 derives its own, and
# this body must not fight it). One mojo build is one GPU arch: the runner's
# MOJOLEARN_GPU_ARCHS wins, and on NVIDIA the device can say it (9.0 is
# spelled sm_90a, DEVIATION 2293). /dev/dri alone is not AMD evidence.
if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ] \
   && command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then
    cap=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    case "$cap" in
        9.0) MOJOLEARN_GPU_ARCHS=sm_90a ;;
        [0-9].[0-9]) MOJOLEARN_GPU_ARCHS="sm_$(echo "$cap" | tr -d .)" ;;
        *) MOJOLEARN_GPU_ARCHS="" ;;
    esac
    export MOJOLEARN_GPU_ARCHS
fi
# MAX's bundled CUDA 13 assembler needs driver 580. Older-driver pods use
# their installed assembler at BOTH build and runtime.
if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then
    driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
    case "$driver_major" in
        ''|*[!0-9]*) ;;
        *) if [ "$driver_major" -lt 580 ] && [ -x /usr/local/cuda/bin/ptxas ]; then
               export MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-/usr/local/cuda/bin/ptxas}
           fi ;;
    esac
fi

rc=0
run() {
    # run <name> <command...>: log to $OUT/<name>.log, record the exit code.
    _name=$1
    shift
    _t0=$(date +%s)
    "$@" > "$OUT/$_name.log" 2>&1
    _code=$?
    printf '%s\t%s\t%ss\n' "$_name" "$_code" "$(( $(date +%s) - _t0 ))" >> "$OUT/status.tsv"
    [ "$_code" -eq 0 ] || rc=1
    return "$_code"
}

{
    echo "deviations=2653,2654"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "root=$ROOT"
    echo "default_arm=$DEFAULT_ARM baseline_arm=$MOJOLEARN_ATTN_BASELINE"
    echo "arms=$MOJOLEARN_ATTN_LEG_ARMS lm_arms=$MOJOLEARN_ATTN_LEG_LM_ARMS"
    echo "skip_timers=$MOJOLEARN_ATTN_LEG_SKIP_TIMERS full=$FULL jobs=$JOBS"
    echo "gpu_archs=${MOJOLEARN_GPU_ARCHS:-UNSET}"
    echo "product=six forward RESOURCES rows (DEVIATION 2653); no arm, no flip"
    [ -f /root/gemm_leg_out/leg.txt ] && grep -E '^(commit|vendor|provider|size)=' /root/gemm_leg_out/leg.txt
} > "$OUT/gate.txt"

# ---- phase 1: the readback (no launch of a timed kernel, no corpus) --------
IDENT="-D MOJOLEARN_NUMERIC_IDENTICAL=1"
TRIAL="-D MOJOLEARN_ATTN_ARM_TRIAL=1"
if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
    echo "phase1=SKIPPED gpu_archs=MISSING (one mojo build is one GPU arch)" >> "$OUT/gate.txt"
    printf 'build-regs-price\t9\t0s\n' >> "$OUT/status.tsv"
    rc=1
else
    # shellcheck disable=SC2086
    run build-regs-price pixi run mojo build -j "$JOBS" -I . $IDENT $TRIAL \
        bench/attention_step_price_main.mojo -o "$OUT/bin/attn-price"
    if [ -x "$OUT/bin/attn-price" ]; then
        MOJOLEARN_ATTN_ARM="$MOJOLEARN_ATTN_BASELINE" \
        MOJOLEARN_ATTN_BASELINE="$MOJOLEARN_ATTN_BASELINE" \
        MOJOLEARN_ATTN_KINDS=hashed MOJOLEARN_ATTN_TIMING=0 \
        MOJOLEARN_ATTN_ORACLE=0 MOJOLEARN_ATTN_REACH=0 \
        MOJOLEARN_ATTN_RESOURCES=1 \
        MOJOLEARN_ATTN_L=512 MOJOLEARN_ATTN_NH=4 MOJOLEARN_ATTN_NKV=2 \
        run regs-readback timeout "$DEADLINE" "$OUT/bin/attn-price"
        grep -h '^RESOURCES' "$OUT/regs-readback.log" > "$OUT/resources.txt" 2>/dev/null
        grep -h '^DEFAULT\|^PATH \|attention_step_price:' "$OUT/regs-readback.log" \
            | sed 's/^/readback: /' >> "$OUT/gate.txt" 2>/dev/null
    fi
fi

# ---- phase 2: the full body, timers on, both corpora -----------------------
if [ "$FULL" = "1" ]; then
    sh tools/attention_step_leg.sh
    a=$?
    echo "attention_step_leg_exit=$a" >> "$OUT/gate.txt"
    [ "$a" -eq 0 ] || rc=1
else
    echo "attention_step_leg=NOT RUN (MOJOLEARN_ATTN_REGS_FULL=0)" >> "$OUT/gate.txt"
fi

# Binaries stay on the box: they are not evidence and the blob fences refuse them.
rm -rf "${OUT:?}/bin"
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/gate.txt"
[ -f "$OUT/resources.txt" ] && cat "$OUT/resources.txt"
[ -f "$OUT/status.tsv" ] && cat "$OUT/status.tsv"
if [ -f /root/gemm_leg_out/leg.txt ]; then
    echo "attention_regs_leg_exit=$rc" >> /root/gemm_leg_out/leg.txt
fi
exit "$rc"
