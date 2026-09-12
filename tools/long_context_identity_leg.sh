#!/bin/sh
# tools/long_context_identity_leg.sh -- DOES ATTENTION'S BITWISE IDENTITY
# SURVIVE A REAL SEQUENCE LENGTH? On-box body, vendor-agnostic, CORRECTNESS
# ONLY: no timing, no price, no flip, no default changed.
#
# WHY. Every byte LM identity check in the tree runs the default profile,
# whose `length` is 32, and the attention arms gate
# (transformer/checks/transformer_attention_arms_check.mojo) says of itself
# "This is a SMALL-SHAPE gate (L <= 700)". Attention's softmax is a
# REDUCTION whose length IS the sequence length: at L=32 the denominator is
# a 32-term fold, at L=8192 an 8192-term fold. Float addition is not
# associative, so if a launch geometry ever regrouped that fold, the bits
# would move -- and the whole IDENTICAL tier claims they do not. Nothing had
# ever been run between 700 and the 8192 ceiling.
#
# WHAT IS VARIED, at every length, on the SAME inputs:
#   * LAUNCH GEOMETRY, through the arm names: the forward's query rows per
#     block (`_fgrid_r32` / `_fgrid_r64`), the zdot stash kernel's rows
#     (`_ztiled_r32` / `_ztiled_r64`), the dk/dv fold's keys per block
#     (`_kvgrid_r32` / `_kvgrid_r64`) and that fold split into two launches
#     (`_kvsplit`), plus Q residency (`_qres`) and preflushed seams (`_pf`).
#     Each is a different thread-block and tile shape over the same
#     arithmetic; the kernel matrix calls them schedules, never numeric
#     terms, and THAT is the claim under test here.
#   * POISONED SCRATCH: the harness's own `clear_outputs` refills all seven
#     output buffers with NaN between every arm, so an arm that wrote
#     nothing cannot pass by inheriting the previous arm's bits.
#   * RAGGED LENGTHS (2047, 8191) beside the round ones, so the tile seam
#     and the masked tail are exercised at long context and not only at a
#     length that divides the block.
#   * REPEAT RUNS: the first arm at each length runs twice in separate
#     processes; the DIGEST lines must match, which is run-to-run
#     determinism at that length.
#
# THE ORACLE is the harness's own: the eager stage kernels
# (MOJOLEARN_ATTN_ORACLE=1) computed on the same inputs, compared cell by
# cell against the fused path on ctx, amax, denom, zdot, dq, dk and dv. It
# runs once per length (the first arm), because it allocates FOUR [B,nh,L,S]
# buffers -- 12.9 GB at L=8192, nh=12 -- and that is the memory wall this
# sweep meets, not an identity one.
#
# POSIX sh only: RunPod's Ubuntu images link /bin/sh to dash.
set -u
ROOT=${MOJOLEARN_LONGCTX_ROOT:-/root/mojolearn}
OUT=${MOJOLEARN_LONGCTX_OUT:-/root/longctx_out}
LENGTHS=${MOJOLEARN_LONGCTX_LENGTHS:-32 128 512 2048 8192}
RAGGED=${MOJOLEARN_LONGCTX_RAGGED:-8191}
DEADLINE=${MOJOLEARN_LONGCTX_DEADLINE:-900}
JOBS=${MOJOLEARN_COMPILE_JOBS:-8}
# The geometries. Every one of these must produce the SAME BITS as the
# shipped `baseline` kernels at every length, or the tier's claim is wrong.
# Four maximally separated schedules rather than all twenty-odd, so the
# sweep fits one lease: the stash geometry alone, the forward's rows halved,
# the zdot stash kernel's rows halved, and the shipped NVIDIA default which
# carries every token at once (forward rows, Q residency, preflushed seams
# and the 32-key dk/dv fold).
ARMS=${MOJOLEARN_LONGCTX_ARMS:-stash_tiled stash_tiled_fgrid_r32 stash_tiled_ztiled_r32 stash_tiled_fgrid_r32_qres_pf_kvgrid_r32}
KINDS=${MOJOLEARN_LONGCTX_KINDS:-hashed,heavytail}
mkdir -p "$OUT"
cd "$ROOT" || exit 9
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical
# CORRECTNESS LANE: never a timing input, so the timer is off everywhere.
export MOJOLEARN_ATTN_TIMING=0
export MOJOLEARN_ATTN_RESOURCES=0
export MOJOLEARN_ATTN_B=1 MOJOLEARN_ATTN_NH=12 MOJOLEARN_ATTN_NKV=12
export MOJOLEARN_ATTN_HD=64 MOJOLEARN_ATTN_WINDOW=0
export MOJOLEARN_ATTN_BASELINE=baseline
export MOJOLEARN_ATTN_KINDS="$KINDS"

if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then
    VENDOR=nvidia
elif [ -e /dev/kfd ] || command -v rocm-smi > /dev/null 2>&1; then
    VENDOR=amd
else
    echo "vendor=unknown: nothing run" > "$OUT/gate.txt"; exit 9
fi
export MOJOLEARN_TARGET_COLUMN="$VENDOR"
if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ] && [ "$VENDOR" = nvidia ]; then
    cap=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    case "$cap" in
        9.0) MOJOLEARN_GPU_ARCHS=sm_90a ;;
        [0-9].[0-9]) MOJOLEARN_GPU_ARCHS="sm_$(echo "$cap" | tr -d .)" ;;
    esac
fi
[ -n "${MOJOLEARN_GPU_ARCHS:-}" ] || { echo "gpu_archs=MISSING" > "$OUT/gate.txt"; exit 9; }
export MOJOLEARN_GPU_ARCHS

{
    echo "lane=long-context-identity"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "vendor=$VENDOR arch=$MOJOLEARN_GPU_ARCHS"
    echo "lengths=$LENGTHS ragged=$RAGGED"
    echo "arms=$ARMS"
    echo "kinds=$KINDS timing=off"
    [ -f "$ROOT/SHIPPED_COMMIT.txt" ] && echo "commit=$(cat "$ROOT/SHIPPED_COMMIT.txt")"
} > "$OUT/gate.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUT/gpu.txt" 2>&1

IDENT="-D MOJOLEARN_NUMERIC_IDENTICAL=1"
TRIAL="-D MOJOLEARN_ATTN_ARM_TRIAL=1"

rc=0
record() {  # record <name> <exit> <seconds>
    printf '%s\t%s\t%ss\n' "$1" "$2" "$3" >> "$OUT/status.tsv"
    [ "$2" -eq 0 ] || rc=1
}

# ---- builds ---------------------------------------------------------------
t0=$(date +%s)
# shellcheck disable=SC2086
pixi run mojo build -j "$JOBS" -I . $IDENT $TRIAL \
    bench/attention_step_price_main.mojo -o "$OUT/attn-price" > "$OUT/build-price.log" 2>&1
record build-price $? $(( $(date +%s) - t0 ))
t0=$(date +%s)
# shellcheck disable=SC2086
pixi run mojo build -j "$JOBS" -I . $IDENT $TRIAL \
    transformer/checks/transformer_attention_arms_check.mojo -o "$OUT/arms-check" > "$OUT/build-arms.log" 2>&1
record build-arms $? $(( $(date +%s) - t0 ))
t0=$(date +%s)
pixi run mojo build -j "$JOBS" -I . $IDENT \
    training/checks/byte_lm_length_sweep_check.mojo -o "$OUT/len-sweep" > "$OUT/build-lensweep.log" 2>&1
record build-lensweep $? $(( $(date +%s) - t0 ))

# ---- the host-only shape boundary (where the guards bind) -----------------
if [ -x "$OUT/len-sweep" ]; then
    t0=$(date +%s); "$OUT/len-sweep" > "$OUT/length-sweep.log" 2>&1
    record length-sweep $? $(( $(date +%s) - t0 ))
fi

# ---- the existing small-shape control (L <= 700) --------------------------
if [ -x "$OUT/arms-check" ]; then
    t0=$(date +%s); timeout "$DEADLINE" "$OUT/arms-check" > "$OUT/arms-check.log" 2>&1
    record arms-check $? $(( $(date +%s) - t0 ))
fi

# ---- THE SWEEP ------------------------------------------------------------
[ -x "$OUT/attn-price" ] || { echo "no harness binary; sweep NOT RUN" >> "$OUT/gate.txt"; exit 9; }
printf 'length\tarm\toracle\texit\tseconds\tverdict\n' > "$OUT/sweep.tsv"
for L in $LENGTHS $RAGGED; do
    first=1
    for arm in $ARMS; do
        if [ "$first" = 1 ]; then oracle=1; else oracle=0; fi
        name="L${L}-${arm}"
        t0=$(date +%s)
        MOJOLEARN_ATTN_L="$L" MOJOLEARN_ATTN_ARM="$arm" \
            MOJOLEARN_ATTN_ORACLE="$oracle" MOJOLEARN_ATTN_REACH=1 \
            timeout "$DEADLINE" "$OUT/attn-price" > "$OUT/$name.log" 2>&1
        code=$?
        secs=$(( $(date +%s) - t0 ))
        if [ "$code" -eq 0 ]; then
            v=PASS
        elif grep -q '^FAIL ' "$OUT/$name.log" 2>/dev/null; then
            v=IDENTITY_FAIL
        elif [ "$code" -eq 124 ]; then
            v=TIMEOUT
        else
            v=ERROR
        fi
        printf '%s\t%s\t%s\t%s\t%ss\t%s\n' "$L" "$arm" "$oracle" "$code" "$secs" "$v" >> "$OUT/sweep.tsv"
        [ "$code" -eq 0 ] || rc=1
        # The repeat run: same length, same arm, a separate process. Its
        # DIGEST lines must equal the first run's, byte for byte.
        if [ "$first" = 1 ]; then
            t0=$(date +%s)
            MOJOLEARN_ATTN_L="$L" MOJOLEARN_ATTN_ARM="$arm" \
                MOJOLEARN_ATTN_ORACLE="$oracle" MOJOLEARN_ATTN_REACH=1 \
                timeout "$DEADLINE" "$OUT/attn-price" > "$OUT/$name.repeat.log" 2>&1
            code=$?
            secs=$(( $(date +%s) - t0 ))
            if grep '^DIGEST ' "$OUT/$name.log" > "$OUT/$name.d1" 2>/dev/null &&
               grep '^DIGEST ' "$OUT/$name.repeat.log" > "$OUT/$name.d2" 2>/dev/null &&
               [ -s "$OUT/$name.d1" ] && cmp -s "$OUT/$name.d1" "$OUT/$name.d2"; then
                v=REPEAT_IDENTICAL
            else
                v=REPEAT_DIFFERS
                rc=1
            fi
            printf '%s\t%s\t%s\t%s\t%ss\t%s\n' "$L" "$arm (repeat)" "$oracle" "$code" "$secs" "$v" >> "$OUT/sweep.tsv"
        fi
        first=0
    done
done

# ---- the summary ----------------------------------------------------------
{
    echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "overall_exit=$rc"
    echo "--- per-length verdicts"
    awk -F'\t' 'NR>1 {n[$1]++; if ($6!="PASS" && $6!="REPEAT_IDENTICAL") bad[$1]=bad[$1] " " $2 "=" $6}
        END {for (l in n) printf "length=%s runs=%s %s\n", l, n[l], (bad[l]=="" ? "ALL IDENTICAL" : "NOT IDENTICAL:" bad[l])}' \
        "$OUT/sweep.tsv" | sort -t= -k2 -n
    echo "--- FAIL lines, if any"
    grep -h '^FAIL ' "$OUT"/L*.log 2>/dev/null | sort | uniq -c | head -40
} >> "$OUT/gate.txt"
cat "$OUT/gate.txt"
exit "$rc"
