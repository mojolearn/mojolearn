#!/bin/sh
# tools/gbdt_per_round_h100_leg.sh -- lane gbdt-per-round-h100, DEVIATIONS 2550
# (X read in place) and 2551 (device leaf partition) on a RunPod H100, the
# on-box body of tools/gbdt_per_round_leg.sh without the droplet parts.
# RUNS ON THE POD (tools/trees_leg.sh rent/ssh), from /root/mojolearn, after
# tools/trees_identical_remote.sh was started in the background (the pod
# setup: pixi, IDENTICAL default bindings, pyarrow, taxi and Istella-S).
#
#   GPR_LEG=1 GPR_BODY_MINUTES=58 nohup sh tools/gbdt_per_round_h100_leg.sh > /root/gpr_out/console.log 2>&1 &
#
# GPR_LEG=1  IDENTICAL: builds (default from the setup, a2550, a2551,
#            a2550_2551), the four named checks (MODEL_HASH equal across
#            builds), identity_break per build diffed against the retained
#            H100 set, timing, stage split.
# GPR_LEG=2  FAST: the four builds, timing (depthwise, symmetric, then
#            lossguide), stage split.
# GPR_SETS_EXTRA / GPR_DEFINES_<set>: a further arm (2553) on either leg.
#
# TIMING SHAPE. Two builds cannot share a process (one module name), so a
# round is one process per build: warm-up fit, one timed fit, logloss and
# AUC. Round r runs every build of the cell, order rotated by r; five rounds.
# The per-build logs are concatenated for tools/flip_verdict.py. Every log
# carries BENCH_BINDING and BENCH_PATHS (ENGINEERING_RULES 8).
#
# POSIX sh (dash), `set -u` not `set -e`; a red phase does not stop the next.
set -u
ROOT=/root/mojolearn
SETUP=/root/trees_out
G=/root/gpr_out
BINS=/root/gpr_bins
LEG="${GPR_LEG:-1}"
BODY_MIN="${GPR_BODY_MINUTES:-55}"
PY=python3
RETAINED="$ROOT/bench/results/trees_identical/h100_2026-09-11_istella/ib/baseline.json"
mkdir -p "$G/logs" "$G/speed" "$G/ib" "$G/checks" "$BINS"
cd "$ROOT" || exit 9
PATH="$HOME/.pixi/bin:$PATH"
export PATH
export MOJOLEARN_GPU_ARCHS="${MOJOLEARN_GPU_ARCHS:-sm_90a}"
export MOJOLEARN_SPEED_EXPECTED_VENDOR=cuda
export GBM_BENCH_DATA="${GBM_BENCH_DATA:-/root/datasets/gbm-bench}"
DATA="$GBM_BENCH_DATA"
JOBS=$(nproc 2>/dev/null || echo 8)
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')"
DEADLINE=$(( $(date +%s) + BODY_MIN * 60 ))
echo "leg=$LEG deadline_epoch=$DEADLINE body_min=$BODY_MIN jobs=$JOBS gpu=$GPU_NAME commit=$(cat SHIPPED_COMMIT.txt 2>/dev/null)" >> "$G/deadline.txt"

left() { echo $((DEADLINE - $(date +%s))); }
status() { printf '%s\t%s\t%s\tleft=%s\n' "$1" "$2" "$(date -u +%H:%M:%S)" "$(left)" | tee -a "$G/status.tsv"; }
status "body_start_leg$LEG" 0

D2550="-D MOJOLEARN_2550_BORROW_X=1"
D2551="-D MOJOLEARN_2551_DEVICE_PARTITION=1"
defines_of() {  # <set>
    case "$1" in
        default) echo "" ;;
        a2550) echo "$D2550" ;;
        a2551) echo "$D2551" ;;
        a2550_2551) echo "$D2550 $D2551" ;;
        *) eval "echo \"\${GPR_DEFINES_$1:-}\"" ;;
    esac
}

# ---------------------------------------------------------------- setup wait
while [ ! -f "$SETUP/track_mojo.done" ] && [ "$(left)" -gt 60 ]; do sleep 10; done
status setup_track_mojo "$(grep -c '=0 ' "$SETUP/setup.txt" 2>/dev/null) zero-exit steps"

# ---------------------------------------------------------------- builds
tier_dir() { if [ "$1" = identical ]; then echo python/mojolearn/identical; else echo python/mojolearn; fi; }
build() {  # <mode> <set>
    _mode=$1; _set=$2; _defs=$(defines_of "$2")
    _dir=$(tier_dir "$_mode")
    mkdir -p "$BINS/$_mode/$_set" "$_dir"
    rm -f "$_dir/_mojolearn_gbdt.so"
    MOJOLEARN_NUMERIC_MODE=$_mode MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=$JOBS \
        MOJOLEARN_EXTRA_DEFINES="$_defs" \
        timeout -k 30 1500 bash bindings/build_gbdt.sh > "$G/logs/build.$_mode.$_set.log" 2>&1
    _rc=$?
    if [ "$_rc" = 0 ] && [ -f "$_dir/_mojolearn_gbdt.so" ]; then
        cp "$_dir/_mojolearn_gbdt.so" "$BINS/$_mode/$_set/_mojolearn_gbdt.so"
        sha256sum "$BINS/$_mode/$_set/_mojolearn_gbdt.so" >> "$G/so_sha256.txt"
    else
        grep -m 12 -i 'error' "$G/logs/build.$_mode.$_set.log" >> "$G/build_errors.txt"
    fi
    status "build.$_mode.$_set defines='$_defs'" "$_rc"
}
use() {  # <mode> <set>
    cp "$BINS/$1/$2/_mojolearn_gbdt.so" "$(tier_dir "$1")/_mojolearn_gbdt.so"
}
imp() {  # <mode> <set>
    use "$1" "$2"
    MOJOLEARN_NUMERIC_MODE=$1 PYTHONPATH="$ROOT/python" timeout -k 10 180 $PY -c \
        "import mojolearn; m = mojolearn.GradientBoosting(); b = m._bind(); print('import OK', b.__file__, b.gbdt_per_round_paths())" \
        > "$G/logs/import.$1.$2.log" 2>&1
    _rc=$?
    status "import.$1.$2 $(tail -1 "$G/logs/import.$1.$2.log")" "$_rc"
    return $_rc
}

if [ "$LEG" = 1 ]; then
    MODE=identical
    SETS4="default a2550 a2551 a2550_2551 ${GPR_SETS_EXTRA:-}"
    mkdir -p "$BINS/identical/default"
    if [ -f python/mojolearn/identical/_mojolearn_gbdt.so ]; then
        cp python/mojolearn/identical/_mojolearn_gbdt.so "$BINS/identical/default/"
        sha256sum "$BINS/identical/default/_mojolearn_gbdt.so" >> "$G/so_sha256.txt"
        status build.identical.default.from_setup 0
    else
        build identical default
    fi
    # the four named checks, both sides of both switches and their sum
    ( timeout -k 30 1800 pixi run check-gbdt-per-round > "$G/checks/off.log" 2>&1; echo $? > "$G/checks/off.exit" ) &
    ( timeout -k 30 1800 pixi run check-gbdt-per-round-2550 > "$G/checks/2550.log" 2>&1; echo $? > "$G/checks/2550.exit" ) &
    ( timeout -k 30 1800 pixi run check-gbdt-per-round-2551 > "$G/checks/2551.log" 2>&1; echo $? > "$G/checks/2551.exit" ) &
    ( timeout -k 30 1800 pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 $D2550 $D2551 -I . checks/gbdt_per_round_check.mojo \
        > "$G/checks/2550_2551.log" 2>&1; echo $? > "$G/checks/2550_2551.exit" ) &
    status checks_started 0
    for s in $SETS4; do [ "$s" = default ] || build identical "$s"; done
else
    MODE=fast
    SETS4="default a2550 a2551 a2550_2551 ${GPR_SETS_EXTRA:-}"
    for s in $SETS4; do build fast "$s"; done
fi

if ! imp "$MODE" default; then
    for b in rf trees; do
        MOJOLEARN_NUMERIC_MODE=$MODE MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=$JOBS \
            timeout -k 30 900 bash "bindings/build_$b.sh" > "$G/logs/build.$MODE.$b.log" 2>&1
        status "build.$MODE.$b" $?
    done
    imp "$MODE" default
fi
for s in $SETS4; do [ "$s" = default ] || imp "$MODE" "$s"; done

# ---------------------------------------------------------------- checks + identity (leg 1)
if [ "$LEG" = 1 ]; then
    for c in off 2550 2551 2550_2551; do
        while [ ! -f "$G/checks/$c.exit" ] && [ "$(left)" -gt 600 ]; do sleep 10; done
        grep '^MODEL_HASH\|^GBDT_PER_ROUND\|^CLAIM1\|FAIL' "$G/checks/$c.log" > "$G/checks/$c.summary" 2>/dev/null
        status "check.$c $(grep -m1 GBDT_PER_ROUND_PATH "$G/checks/$c.log")" "$(cat "$G/checks/$c.exit" 2>/dev/null || echo UNFINISHED)"
        grep '^MODEL_HASH' "$G/checks/$c.log" > "$G/checks/$c.hash" 2>/dev/null
    done
    _eq=0
    for c in 2550 2551 2550_2551; do
        [ -s "$G/checks/off.hash" ] && cmp -s "$G/checks/off.hash" "$G/checks/$c.hash" || _eq=1
    done
    status model_hashes_equal_across_four_builds "$_eq"

    for s in $SETS4; do
        use identical "$s"
        MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$ROOT/python" timeout -k 30 900 $PY -u tools/identity_break.py \
            --lanes gbdt-symmetric,gbdt-depthwise,gbdt-lossguide,gbdt-rmse --repeats 2 \
            --vendor "nvidia-$GPU_NAME" --json "$G/ib/$s.json" > "$G/ib/$s.txt" 2>&1
        status "ib.$s" $?
    done
    if [ -f "$RETAINED" ]; then
        PYTHONPATH="$ROOT/python" $PY tools/identity_break.py --diff "$RETAINED" "$G/ib/default.json" > "$G/ib/diff.retained.default.txt" 2>&1
        status diff.retained_h100_0911_istella.default $?
    fi
    for s in $SETS4; do
        [ "$s" = default ] && continue
        PYTHONPATH="$ROOT/python" $PY tools/identity_break.py --diff "$G/ib/default.json" "$G/ib/$s.json" > "$G/ib/diff.default.$s.txt" 2>&1
        status "diff.default.$s" $?
    done
fi

# ---------------------------------------------------------------- timing
data_ready() {  # <dataset>
    if [ "$1" = istella ]; then
        while [ ! -f "$SETUP/setup.done" ] && [ "$(left)" -gt 300 ]; do sleep 10; done
    fi
    while [ ! -f "$DATA/$1/$1_speed.npz" ] && [ ! -f "$SETUP/track_pip.done" ] && [ "$(left)" -gt 300 ]; do sleep 10; done
    [ -f "$DATA/$1/$1_speed.npz" ]
}
run1() {  # <kind timed|stage> <mode> <set> <lane> <dataset> <round>
    _k=$1; _m=$2; _s=$3; _l=$4; _d=$5; _r=$6
    use "$_m" "$_s"
    _st=0
    [ "$_k" = stage ] && _st=1
    MOJOLEARN_NUMERIC_MODE=$_m MOJOLEARN_SPEED_SIZE=shipped MOJOLEARN_SPEED_BUDGET_S=900 \
        MOJOLEARN_SPEED_DEADLINE_S=1200 MOJOLEARN_SPEED_DEVICE="$GPU_NAME" \
        MOJOLEARN_SPEED_ROUNDS=1 MOJOLEARN_STAGE_TIMES=$_st PYTHONPATH="$ROOT/python" \
        timeout -k 20 600 $PY -u bench/speed/forest_speed_arm.py \
        --lane "$_l" --dataset "$_d" --rows 1000000 --ours-only \
        > "$G/speed/$_k.$_m.$_s.$_l.$_d.r$_r.log" 2>&1
    _rc=$?
    [ "$_rc" = 0 ] || status "run.$_k.$_m.$_s.$_l.$_d.r$_r" "$_rc"
}
rotate() {  # <r> <items...>
    _r=$1; shift; _n=$#; _i=0; _out=""
    while [ "$_i" -lt "$_n" ]; do
        _idx=$(( (_i + _r) % _n + 1 ))
        eval "_v=\${$_idx}"
        _out="$_out $_v"
        _i=$((_i + 1))
    done
    echo $_out
}
cell() {  # <mode> <lane> <dataset> <sets...>
    _cm=$1; _cl=$2; _cd=$3; shift 3
    _n=$#
    if ! data_ready "$_cd"; then status "cell.$_cm.$_cl.$_cd" NO_DATA; return 0; fi
    _per=45
    [ "$_cd" = istella ] && _per=75
    _done=0
    for _r in 1 2 3 4 5; do
        if [ "$(left)" -lt $((_n * _per)) ]; then status "cell.$_cm.$_cl.$_cd rounds_done=$_done" SKIPPED_TIME; break; fi
        # shellcheck disable=SC2046
        for _s in $(rotate "$_r" "$@"); do run1 timed "$_cm" "$_s" "$_cl" "$_cd" "$_r"; done
        _done=$_r
    done
    for _s in "$@"; do
        cat "$G/speed/timed.$_cm.$_s.$_cl.$_cd".r*.log > "$G/speed/timed.$_cm.$_s.$_cl.$_cd.log" 2>/dev/null
    done
    status "cell.$_cm.$_cl.$_cd rounds=$_done" 0
}
stage() {  # <mode> <lane> <dataset> <sets...>
    _sm=$1; _sl=$2; _sd=$3; shift 3
    if ! data_ready "$_sd"; then status "stage.$_sm.$_sl.$_sd" NO_DATA; return 0; fi
    for _s in "$@"; do
        if [ "$(left)" -lt 90 ]; then status "stage.$_sm.$_s.$_sl.$_sd" SKIPPED_TIME; continue; fi
        run1 stage "$_sm" "$_s" "$_sl" "$_sd" 1
    done
}

# symmetric never reaches the 2551 partition (DEVIATION 90 is the
# non-symmetric estimator's); its cell times the 2550 side only
SYM="default a2550 ${GPR_SETS_EXTRA:-}"
# wait for the checks' compiles to leave the CPU before any timed fit
for c in off 2550 2551 2550_2551; do
    [ "$LEG" = 1 ] || break
    while [ ! -f "$G/checks/$c.exit" ] && [ "$(left)" -gt 600 ]; do sleep 10; done
done
status timing_start 0
# shellcheck disable=SC2086
if [ "$LEG" = 1 ]; then
    cell identical gbdt-depthwise taxi $SETS4
    cell identical gbdt-symmetric taxi $SYM
    cell identical gbdt-lossguide taxi $SETS4
    cell identical gbdt-depthwise istella $SETS4
    cell identical gbdt-symmetric istella $SYM
    cell identical gbdt-lossguide istella $SETS4
    stage identical gbdt-depthwise istella default a2550_2551
    stage identical gbdt-lossguide istella default a2550_2551
    stage identical gbdt-symmetric istella default a2550
    stage identical gbdt-depthwise taxi default a2550_2551
    stage identical gbdt-lossguide taxi default a2550_2551
    stage identical gbdt-symmetric taxi default a2550
else
    cell fast gbdt-depthwise taxi $SETS4
    cell fast gbdt-symmetric taxi $SYM
    cell fast gbdt-depthwise istella $SETS4
    cell fast gbdt-symmetric istella $SYM
    cell fast gbdt-lossguide taxi $SETS4
    cell fast gbdt-lossguide istella $SETS4
    stage fast gbdt-depthwise istella default a2550_2551
    stage fast gbdt-symmetric istella default a2550
    stage fast gbdt-depthwise taxi default a2550_2551
fi
: > "$G/summary_fspeed.txt"
for f in "$G"/speed/timed.*.log; do
    case "$f" in *.r[0-9].log) continue ;; esac
    grep -H '^FSPEED \|^FSPEED-ACC\|^BENCH_PATHS\|^FSPEED-REFUSED' "$f" >> "$G/summary_fspeed.txt" 2>/dev/null
done
status body_done 0
: > "$G/body.done"
exit 0
