#!/bin/sh
# The NVIDIA half of the three unmeasured items, one H100 lease, main 00cd9a5f.
#
# 1. THE OPPONENT TABLE, ONE FLIP BEHIND. bench/OPPONENT_REFERENCE.md's
#    "Same pod, after the NVIDIA attention estash flip" rows were measured on
#    a pod running 5b7e1e41, and DEVIATION 2649 (the step glue flip) merged
#    AFTER that pod died. The table says so itself. This re-measures OUR
#    shipped cell and every torch column on the SAME pod in the same heat
#    window, at a commit that carries both flips, which is the only way that
#    table's ratios are allowed to be quoted.
#
# 2. THE SAMBA RMSNORM PRICE. BRIEF_step_glue section 11's WHAT IS NOT
#    CLAIMED: 2649 moves llama_rms_norm and bwd_rms_norm to 16 threads per
#    block for EVERY caller on an NVIDIA build, including
#    training/samba_ops.mojo, which the leg never timed. bench/
#    samba_rms_price_main.mojo is that measurement. ONE binary with the trial
#    define, run alternately under `shipped` (the 128-thread geometry) and
#    `optskip_noshadow_rows16`, because the M4 lesson about process
#    alternation applies to a rented board too.
#
# Order is deliberate. The opponent pair runs FIRST and together, because it
# is the item whose value depends on one heat window; the samba price is an
# A/B against itself and survives running later on a warm board.
#
# POSIX sh only: RunPod's Ubuntu images link /bin/sh to dash.
set -u
cd /root/mojolearn || exit 9
OUTROOT=/root/gemm_leg_out
NV_DEFAULT=stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32

# ---- 1. our shipped cell, then every torch column, same pod ----------------
MOJOLEARN_ATTN_BASELINE=$NV_DEFAULT \
MOJOLEARN_ATTN_LEG_ARMS=$NV_DEFAULT \
MOJOLEARN_ATTN_LEG_LM_ARMS=$NV_DEFAULT \
MOJOLEARN_ATTN_LEG_SHIPPED_CHECK=1 \
MOJOLEARN_ATTN_LEG_SKIP_TIMERS=1 \
MOJOLEARN_COMPILE_JOBS=8 \
    sh tools/attention_step_leg.sh
a=$?
echo "attention_leg_exit=$a" >> $OUTROOT/leg.txt

sh tools/torch_lm_step_opponent_leg.sh
t=$?
echo "torch_leg_exit=$t" >> $OUTROOT/leg.txt

# ---- 2. the samba RMSNorm price under the two geometries -------------------
SOUT=$OUTROOT/samba-rms
mkdir -p "$SOUT"
JOBS=${MOJOLEARN_COMPILE_JOBS:-8}

pixi run mojo build -j "$JOBS" -I . \
    -D MOJOLEARN_NUMERIC_IDENTICAL=1 \
    -D MOJOLEARN_STEP_GLUE_TRIAL=1 \
    bench/samba_rms_price_main.mojo -o "$SOUT/samba-rms-price" \
    > "$SOUT/build.log" 2>&1
b=$?
echo "samba_build_exit=$b" >> $OUTROOT/leg.txt

s=0
if [ "$b" = 0 ] && [ -x "$SOUT/samba-rms-price" ]; then
    # ALTERNATE WHOLE PROCESSES. The two arms are one binary, but a board
    # drifts, so `shipped` and the winner take turns and each gets three
    # rounds. A round's medians are compared round by round, not pooled.
    r=1
    while [ "$r" -le 3 ]; do
        for arm in shipped optskip_noshadow_rows16; do
            MOJOLEARN_STEP_GLUE_ARM="$arm" timeout 900 \
                "$SOUT/samba-rms-price" > "$SOUT/round$r-$arm.log" 2>&1
            e=$?
            echo "round=$r arm=$arm exit=$e" >> "$SOUT/status.tsv"
            [ "$e" = 0 ] || s=1
        done
        r=$((r + 1))
    done
    # The reach line of every run, gathered where a reader will find it.
    grep -h '^ARM' "$SOUT"/round*.log >> "$SOUT/arms.txt" 2>/dev/null
    cat "$SOUT"/round*.log > "$SOUT/all.log" 2>/dev/null
else
    s=1
fi
echo "samba_runs_exit=$s" >> $OUTROOT/leg.txt

echo "attention=$a torch=$t samba_build=$b samba_runs=$s" >> $OUTROOT/leg.txt
# The opponent pair is the item that must not be lost. A red samba phase is a
# finding and is reported, but it does not fail the leg on its own.
[ "$a" = 0 ] && [ "$t" = 0 ]
