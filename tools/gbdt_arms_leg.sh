#!/bin/sh
# tools/gbdt_arms_leg.sh -- lane gbdt-arms-hotaisle: DEVIATIONS 2550 (X read
# in place), 2551 (device leaf partition), 2580 (level quantize) and 2581
# (per-group bit width) compiled, checked and A/B'd on one AMD box (gfx942) in
# IDENTICAL and FAST. Runs ON THE BOX as the MOJOLEARN_GEMM_LEG_EXTRA body of
# either AMD runner (the tools/do_extra_leg.sh body contract: cwd
# /root/mojolearn, pixi on PATH, pixi install done, output under
# /root/gemm_leg_out/gah/, fetched home with the leg). Every number is
# labeled with the box it ran on; before and after always share one box.
#
#   DigitalOcean MI325X (native; one GPU droplet at a time, so one group per leg):
#   MOJOLEARN_GPU_ARCHS=gfx942 \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/gbdt_arms_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=$HOME/mojolearn-evidence/gbdt-arms-hotaisle/<stamp>-amd-mi325x-<group> \
#   MOJOLEARN_DO_EXTRA_ENV='MOJOLEARN_GBDT_ARMS_GROUP=<perround|sym>' \
#   bash tools/do_extra_leg.sh amd --minutes 60 --skip-gates
#
#   Hot Aisle MI300X (in the ROCm container):
#   MOJOLEARN_HOTAISLE_SPEC=13core MOJOLEARN_GEMM_LEG_EXTRA=tools/gbdt_arms_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=... MOJOLEARN_HOTAISLE_EXTRA_ENV='MOJOLEARN_GBDT_ARMS_GROUP=<group>' \
#   bash tools/hotaisle_leg.sh amd --rent --minutes 60 --skip-gates
#
# ENV (values limited to the runners' [A-Za-z0-9_.,:/=-])
#   MOJOLEARN_GBDT_ARMS_GROUP   perround: sets default a2550 a2551, combo
#                               c2550_2551, lanes depthwise lossguide symmetric
#                               sym: sets default a2580 a2581, combo
#                               c2580_2581, lane symmetric (the only lane the
#                               two arms reach)
#   MOJOLEARN_GBDT_ARMS_TIERS   identical,fast (default)
#   MOJOLEARN_GBDT_ARMS_COMBOS  1 (default) or 0; combos build and time last
#                               and are what goes first when time is short
#   MOJOLEARN_GBDT_ARMS_BUDGET_S  work bound when the start wrapper's is unreadable
#
# PHASES (each a row in status.tsv; a red phase does not stop the next)
#   deps, data (background; taxi falls back to curl with an explicit agent),
#   builds (identical base, then per tier default and single arms; the body
#   exits at once when no default gbdt build compiled, so the runner ends the
#   bill; combos last), checks (every named check of the group, both sides,
#   in parallel after the builds, so no compile shares the box with a timed
#   round), import (prints the compiled path of all four switches), ib
#   (identity_break per IDENTICAL set), speed (1M rows, per cell one process
#   per set, warm-up plus 5 rounds, set order rotated per cell; IDENTICAL
#   cells before FAST cells), stage (one MOJOLEARN_STAGE_TIMES=1 replicate
#   per set, Istella-S first), combos (default re-timed beside each combo in
#   the same window), verdicts (tools/flip_verdict.py per tier per lane per
#   arm, rerun on the Mac).
#
# Which lane an arm reaches decides its cells: 2551 is the non-symmetric
# estimator only; 2580 and 2581 are the symmetric searcher only; 2550 is
# every lane (train's column staging). A lane an arm cannot reach is covered
# by identity_break's hashes, not by a timing.
#
# POSIX sh only (dash), `set -u` not `set -e`.
set -u
ROOT=/root/mojolearn
G=/root/gemm_leg_out/gah
BINS=/root/gah_bins
DATA="${GBM_BENCH_DATA:-/root/datasets/gbm-bench}"
GBM_BENCH_DATA="$DATA"
export GBM_BENCH_DATA
export DEBIAN_FRONTEND=noninteractive
PY=/usr/bin/python3
[ -x "$PY" ] || PY=python3
mkdir -p "$G/logs" "$G/speed" "$G/stage" "$G/ib" "$G/checks" "$G/verdicts" "$BINS"
cd "$ROOT" || exit 9
PATH="$HOME/.pixi/bin:$PATH"
export PATH
JOBS=$(nproc 2>/dev/null || echo 8)
BODY_START=$(date +%s)
GROUP="${MOJOLEARN_GBDT_ARMS_GROUP:-perround}"
TIERS=$(echo "${MOJOLEARN_GBDT_ARMS_TIERS:-identical,fast}" | tr ',' ' ')
COMBOS="${MOJOLEARN_GBDT_ARMS_COMBOS:-1}"

# THE DEADLINE: the start wrapper's timeout bound, from its argv and the
# remote body's started= line; the budget env or a conservative bound else.
# Read from /proc first: in the Hot Aisle ROCm container procps is not
# guaranteed; `ps` is the fallback.
WORK=$(for _f in /proc/[0-9]*/cmdline; do tr '\000' ' ' < "$_f" 2>/dev/null; echo; done \
    | sed -n 's/^timeout -k [0-9]* \([0-9][0-9]*\) sh \/root\/gemm_leg\.sh.*$/\1/p' | head -n 1)
[ -n "$WORK" ] || WORK=$(ps -eo args 2>/dev/null | sed -n 's/^timeout -k [0-9]* \([0-9][0-9]*\) sh \/root\/gemm_leg\.sh.*$/\1/p' | head -n 1)
STARTED=$(sed -n 's/^started=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -n 1)
ST_EPOCH=$(date -u -d "$STARTED" +%s 2>/dev/null || echo "")
if [ -n "$WORK" ] && [ -n "$ST_EPOCH" ]; then
    DEADLINE=$((ST_EPOCH + WORK - 90))
else
    DEADLINE=$((BODY_START + ${MOJOLEARN_GBDT_ARMS_BUDGET_S:-2400}))
fi
echo "deadline_epoch=$DEADLINE work=$WORK started=$STARTED body_start=$BODY_START jobs=$JOBS group=$GROUP tiers=$TIERS combos=$COMBOS" > "$G/deadline.txt"

left() { echo $((DEADLINE - $(date +%s))); }
status() { printf '%s\t%s\t%s\tleft=%s\n' "$1" "$2" "$(date -u +%H:%M:%S)" "$(left)" >> "$G/status.tsv"; }
status body_start 0

case "$GROUP" in
    perround)
        SETS="default a2550 a2551"
        COMBO=c2550_2551
        LANES="gbdt-depthwise gbdt-lossguide gbdt-symmetric"
        ;;
    sym)
        SETS="default a2580 a2581"
        COMBO=c2580_2581
        LANES="gbdt-symmetric"
        ;;
    *)
        status bad_group 2
        exit 2
        ;;
esac
defines_for() {
    case "$1" in
        default) echo "" ;;
        a2550) echo "-D MOJOLEARN_2550_BORROW_X=1" ;;
        a2551) echo "-D MOJOLEARN_2551_DEVICE_PARTITION=1" ;;
        c2550_2551) echo "-D MOJOLEARN_2550_BORROW_X=1 -D MOJOLEARN_2551_DEVICE_PARTITION=1" ;;
        a2580) echo "-D MOJOLEARN_2580_LEVEL_QUANT=1" ;;
        a2581) echo "-D MOJOLEARN_2581_GROUP_WIDTH=1" ;;
        c2580_2581) echo "-D MOJOLEARN_2580_LEVEL_QUANT=1 -D MOJOLEARN_2581_GROUP_WIDTH=1" ;;
    esac
}
# sets_for <lane> <sets...>: the sets that reach the lane, default first
sets_for() {
    _lane=$1; shift
    _o=""
    for _s in "$@"; do
        case "$_lane.$_s" in
            gbdt-symmetric.a2551) ;;
            gbdt-symmetric.c2550_2551) ;;
            gbdt-depthwise.a258*|gbdt-lossguide.a258*|gbdt-depthwise.c2580_2581|gbdt-lossguide.c2580_2581) ;;
            *) _o="$_o $_s" ;;
        esac
    done
    echo $_o
}

# ---------------------------------------------------------------- deps
if ! $PY -c 'import pip' > /dev/null 2>&1; then
    timeout -k 10 600 sh -c 'apt-get -o DPkg::Lock::Timeout=180 update -qq; apt-get -o DPkg::Lock::Timeout=180 install -y --no-install-recommends python3-pip' \
        > "$G/logs/apt.log" 2>&1
    status apt_pip $?
fi
timeout -k 10 600 $PY -m pip install --break-system-packages --disable-pip-version-check \
    --no-input numpy pyarrow pandas > "$G/logs/pip.log" 2>&1
_rc=$?
if [ "$_rc" != 0 ]; then
    timeout -k 10 600 $PY -m pip install --disable-pip-version-check \
        --no-input numpy pyarrow pandas >> "$G/logs/pip.log" 2>&1
    _rc=$?
fi
status deps "$_rc"
$PY -c "import numpy, pyarrow, pandas; print('numpy', numpy.__version__, 'pyarrow', pyarrow.__version__, 'pandas', pandas.__version__)" > "$G/versions.txt" 2>&1
GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | sed -n 's/.*Card Series:[[:space:]]*//p' | head -n 1 | tr ' ' '_')
[ -n "$GPU_NAME" ] || GPU_NAME=AMD_unknown
IB_LABEL="amd-$(echo "$GPU_NAME" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9\n' '-')"
echo "gpu_name=$GPU_NAME ib_label=$IB_LABEL" >> "$G/deadline.txt"
{ rocminfo 2>/dev/null | grep -m 4 -E 'Marketing Name|Name:[[:space:]]+gfx'; nproc; free -g; } > "$G/box.txt" 2>&1

# ---------------------------------------------------------------- data (background)
(
    timeout -k 10 1200 $PY tools/speed_gbdt_arm.py --download taxi > "$G/logs/download_taxi.log" 2>&1
    _t=$?
    if [ "$_t" != 0 ]; then
        # the TLC CloudFront has refused a default agent before
        # (bench/results/trees_identical/mi325x_2026-09-11_taxi_istella/logs/fetch_taxi.sh)
        mkdir -p "$DATA/taxi"
        for m in 2024-01 2024-02; do
            curl -fL --retry 3 -A "Mozilla/5.0 (X11; Linux x86_64) mojolearn-bench" \
                -o "$DATA/taxi/yellow_tripdata_$m.parquet.part" \
                "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_$m.parquet" \
                && mv "$DATA/taxi/yellow_tripdata_$m.parquet.part" "$DATA/taxi/yellow_tripdata_$m.parquet"
        done >> "$G/logs/download_taxi.curl.log" 2>&1
        timeout -k 10 1200 $PY tools/speed_gbdt_arm.py --download taxi >> "$G/logs/download_taxi.curl.log" 2>&1
        _t=$?
    fi
    echo "download_taxi=$_t $(date -u +%H:%M:%S)" >> "$G/data.txt"
    timeout -k 10 2400 $PY tools/speed_gbdt_arm.py --download istella > "$G/logs/download_istella.log" 2>&1
    echo "download_istella=$? $(date -u +%H:%M:%S)" >> "$G/data.txt"
    : > "$G/data.done"
) &

# ---------------------------------------------------------------- builds
tier_dir() { if [ "$1" = identical ]; then echo python/mojolearn/identical; else echo python/mojolearn; fi; }
build() {  # <mode> <set> <script> <so> <defines>
    _mode=$1; _set=$2; _script=$3; _so=$4; _defs=$5
    if [ "$(left)" -lt 600 ]; then status "build.$_mode.$_set.$_so" SKIPPED_TIME; return 0; fi
    mkdir -p "$BINS/$_mode/$_set"
    _dir=$(tier_dir "$_mode")
    mkdir -p "$_dir"
    rm -f "$_dir/$_so"
    _t0=$(date +%s)
    MOJOLEARN_NUMERIC_MODE=$_mode MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=$JOBS \
        MOJOLEARN_EXTRA_DEFINES="$_defs" \
        timeout -k 30 1200 bash "$_script" > "$G/logs/build.$_mode.$_set.$_so.log" 2>&1
    _rc=$?
    if [ "$_rc" = 0 ] && [ -f "$_dir/$_so" ]; then
        cp "$_dir/$_so" "$BINS/$_mode/$_set/$_so"
        sha256sum "$BINS/$_mode/$_set/$_so" >> "$G/so_sha256.txt"
    else
        echo "=== $_mode.$_set.$_so" >> "$G/build_errors.txt"
        grep -m 12 -i -A3 'error' "$G/logs/build.$_mode.$_set.$_so.log" >> "$G/build_errors.txt"
    fi
    status "build.$_mode.$_set.$_so" "$_rc s=$(( $(date +%s) - _t0 ))"
}
use() {  # <mode> <set>
    cp "$BINS/$1/$2/_mojolearn_gbdt.so" "$(tier_dir "$1")/_mojolearn_gbdt.so"
}

build identical base bindings/build.sh _mojolearn.so ""
for _t in $TIERS; do
    for _s in $SETS; do
        build "$_t" "$_s" bindings/build_gbdt.sh _mojolearn_gbdt.so "$(defines_for "$_s")"
    done
done
# No default gbdt build in any tier: nothing below can measure, so end the
# body now and let the runner destroy the box (the bill runs until then).
_any_default=0
for _t in $TIERS; do
    [ -f "$BINS/$_t/default/_mojolearn_gbdt.so" ] && _any_default=1
done
if [ "$_any_default" = 0 ]; then
    status no_default_gbdt_build 3
    exit 3
fi
if [ "$COMBOS" = 1 ]; then
    for _t in $TIERS; do
        build "$_t" "$COMBO" bindings/build_gbdt.sh _mojolearn_gbdt.so "$(defines_for "$COMBO")"
    done
fi

# ---------------------------------------------------------------- checks (parallel, then wait)
check() {  # <name> <mojo run args...>
    _n=$1; shift
    ( timeout -k 30 1200 pixi run mojo run -I . "$@" > "$G/checks/$_n.log" 2>&1; echo "$?" > "$G/checks/$_n.exit" ) &
}
DI="-D MOJOLEARN_NUMERIC_IDENTICAL=1"
if [ "$GROUP" = perround ]; then
    # shellcheck disable=SC2086
    check check-gbdt-per-round $DI checks/gbdt_per_round_check.mojo
    check check-gbdt-per-round-2550 $DI -D MOJOLEARN_2550_BORROW_X=1 checks/gbdt_per_round_check.mojo
    check check-gbdt-per-round-2551 $DI -D MOJOLEARN_2551_DEVICE_PARTITION=1 checks/gbdt_per_round_check.mojo
    check check-gbdt-per-round-2550-2551 $DI -D MOJOLEARN_2550_BORROW_X=1 -D MOJOLEARN_2551_DEVICE_PARTITION=1 checks/gbdt_per_round_check.mojo
    CHECKS="check-gbdt-per-round check-gbdt-per-round-2550 check-gbdt-per-round-2551 check-gbdt-per-round-2550-2551"
else
    # the check instantiates every side itself; one run per tier
    check check-sym-arms-identical $DI checks/sym_arms_check.mojo
    check check-sym-arms checks/sym_arms_check.mojo
    CHECKS="check-sym-arms-identical check-sym-arms"
fi
status checks_started 0
for c in $CHECKS; do
    while [ ! -f "$G/checks/$c.exit" ] && [ "$(left)" -gt 900 ]; do sleep 10; done
    status "$c" "$(cat "$G/checks/$c.exit" 2>/dev/null || echo UNFINISHED)"
    grep -E '^MODEL_HASH|^GBDT_PER_ROUND_PATH|PASS|FAIL' "$G/checks/$c.log" > "$G/checks/$c.summary" 2>/dev/null
    grep '^MODEL_HASH' "$G/checks/$c.log" > "$G/checks/$c.hash" 2>/dev/null
done
if [ "$GROUP" = perround ]; then
    _eq=0
    [ -s "$G/checks/check-gbdt-per-round.hash" ] || _eq=1
    for c in $CHECKS; do
        cmp -s "$G/checks/check-gbdt-per-round.hash" "$G/checks/$c.hash" || _eq=1
    done
    status model_hashes_equal_across_builds "$_eq"
fi

# ---------------------------------------------------------------- import
imp() {  # <mode>
    MOJOLEARN_NUMERIC_MODE=$1 PYTHONPATH="$ROOT/python" timeout -k 10 180 $PY -c \
        "import mojolearn; m = mojolearn.GradientBoosting(); b = m._bind(); print('import OK', m.numeric_mode_used(), m.vendor_used(), b.gbdt_per_round_paths())" \
        > "$G/logs/import.$1.log" 2>&1
}
_ri=0
_rf=0
for _t in $TIERS; do
    [ -f "$BINS/$_t/default/_mojolearn_gbdt.so" ] && use "$_t" default
    imp "$_t"
    _rc=$?
    if [ "$_t" = identical ]; then _ri=$_rc; else _rf=$_rc; fi
done
if [ "$_ri" != 0 ] || [ "$_rf" != 0 ]; then
    status import_first_try "identical=$_ri fast=$_rf"
    for _t in $TIERS; do
        for b in rf trees; do
            MOJOLEARN_NUMERIC_MODE=$_t MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=$JOBS \
                timeout -k 30 900 bash "bindings/build_$b.sh" > "$G/logs/build.$_t.$b.log" 2>&1
        done
        imp "$_t"
        _rc=$?
        if [ "$_t" = identical ]; then _ri=$_rc; else _rf=$_rc; fi
    done
fi
status import "identical=$_ri fast=$_rf"

# ---------------------------------------------------------------- identity_break (IDENTICAL sets)
ib() {  # <set> <repeats>
    if [ ! -f "$BINS/identical/$1/_mojolearn_gbdt.so" ]; then status "ib.$1" NO_BINARY; return 0; fi
    if [ "$(left)" -lt 900 ]; then status "ib.$1" SKIPPED_TIME; return 0; fi
    use identical "$1"
    MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$ROOT/python" timeout -k 30 600 $PY -u tools/identity_break.py \
        --lanes gbdt-symmetric,gbdt-depthwise,gbdt-lossguide --repeats "$2" --vendor "$IB_LABEL-$1" \
        --json "$G/ib/$1.json" > "$G/ib/$1.txt" 2>&1
    status "ib.$1" $?
}
case " $TIERS " in
    *" identical "*)
        ib default 2
        for _s in $SETS; do [ "$_s" = default ] || ib "$_s" 1; done
        ;;
esac

# ---------------------------------------------------------------- speed
data_ready() {  # <dataset>: wait for its npz, bounded
    while [ ! -f "$DATA/$1/$1_speed.npz" ] && [ ! -f "$G/data.done" ] && [ "$(left)" -gt 300 ]; do sleep 10; done
    [ -f "$DATA/$1/$1_speed.npz" ]
}
speed() {  # <pass> <mode> <set> <lane> <dataset> <timed|stage>
    _pass=$1; _mode=$2; _set=$3; _lane=$4; _ds=$5; _kind=$6
    _tag="$_pass.$_mode.$_set.$_lane.$_ds"
    _need=150
    [ "$_ds" = istella ] && _need=260
    [ "$_kind" = stage ] && _need=$((_need / 2))
    if [ "$(left)" -lt "$_need" ]; then status "$_kind.$_tag" SKIPPED_TIME; return 0; fi
    if [ ! -f "$BINS/$_mode/$_set/_mojolearn_gbdt.so" ]; then status "$_kind.$_tag" NO_BINARY; return 0; fi
    use "$_mode" "$_set"
    _rounds=5; _st=0; _dir="$G/speed"; _gp=0
    [ "$_kind" = stage ] && { _rounds=1; _st=1; _dir="$G/stage"; _gp=1; }
    MOJOLEARN_NUMERIC_MODE=$_mode MOJOLEARN_SPEED_EXPECTED_VENDOR=hip MOJOLEARN_SPEED_SIZE=shipped \
        MOJOLEARN_SPEED_BUDGET_S=900 MOJOLEARN_SPEED_DEADLINE_S=1200 MOJOLEARN_SPEED_DEVICE="$GPU_NAME" \
        MOJOLEARN_SPEED_ROUNDS=$_rounds MOJOLEARN_STAGE_TIMES=$_st MOJOLEARN_GBDT_PATH=$_gp \
        PYTHONPATH="$ROOT/python" \
        timeout -k 20 $(( $(left) - 20 )) $PY -u bench/speed/forest_speed_arm.py \
        --lane "$_lane" --dataset "$_ds" --rows 1000000 --ours-only > "$_dir/$_tag.log" 2>&1
    status "$_kind.$_tag" $?
}
rotate() {  # <r> <items...>: the items rotated left by r
    _r=$1; shift; _n=$#; _k=0; _out=""
    while [ "$_k" -lt "$_n" ]; do
        _idx=$(( (_k + _r) % _n + 1 ))
        eval "_v=\${$_idx}"
        _out="$_out $_v"
        _k=$((_k + 1))
    done
    echo $_out
}
ROT=0
cell() {  # <pass> <mode> <kind> <lane> <dataset> <sets...>
    _cp=$1; _cm=$2; _ck=$3; _cl=$4; _cd=$5; shift 5
    if ! data_ready "$_cd"; then status "cell.$_cp.$_ck.$_cm.$_cl.$_cd" NO_DATA; return 0; fi
    for _s in $(rotate "$ROT" "$@"); do
        speed "$_cp" "$_cm" "$_s" "$_cl" "$_cd" "$_ck"
    done
    ROT=$((ROT + 1))
}
# main pass: IDENTICAL cells, then FAST cells, taxi before Istella-S per lane
for _t in $TIERS; do
    for _l in $LANES; do
        for _d in taxi istella; do
            # shellcheck disable=SC2046
            cell main "$_t" timed "$_l" "$_d" $(sets_for "$_l" $SETS)
        done
    done
done
# stage split, Istella-S then taxi, IDENTICAL first
for _d in istella taxi; do
    for _t in $TIERS; do
        for _l in $LANES; do
            # shellcheck disable=SC2046
            cell stage "$_t" stage "$_l" "$_d" $(sets_for "$_l" $SETS)
        done
    done
done
# combos: default re-timed beside each combo in the same window
if [ "$COMBOS" = 1 ]; then
    for _t in $TIERS; do
        for _l in $LANES; do
            _cs=$(sets_for "$_l" default "$COMBO")
            [ "$_cs" = default ] && continue
            for _d in taxi istella; do
                # shellcheck disable=SC2086
                cell combo "$_t" timed "$_l" "$_d" $_cs
            done
        done
    done
    case " $TIERS " in *" identical "*) ib "$COMBO" 1 ;; esac
fi

# ---------------------------------------------------------------- verdicts (rerun on the Mac)
for _pass in main combo; do
    for _t in $TIERS; do
        for _l in $LANES; do
            for _s in $SETS $COMBO; do
                [ "$_s" = default ] && continue
                _b="$G/speed/$_pass.$_t.default.$_l"
                _a="$G/speed/$_pass.$_t.$_s.$_l"
                [ -f "$_a.taxi.log" ] || [ -f "$_a.istella.log" ] || continue
                $PY tools/flip_verdict.py --lane "$_l" --arm ours --rows 1000000 \
                    --taxi-before "$_b.taxi.log" --taxi-after "$_a.taxi.log" \
                    --istella-before "$_b.istella.log" --istella-after "$_a.istella.log" \
                    > "$G/verdicts/$_pass.$_t.$_s.$_l.txt" 2>&1
                echo "$_pass.$_t.$_s.$_l $(tail -n 1 "$G/verdicts/$_pass.$_t.$_s.$_l.txt")" >> "$G/verdicts.txt"
            done
        done
    done
done
grep -h '^FSPEED \|^FSPEED-ACC\|^BENCH_PATHS\|^BENCH_BINDING' "$G"/speed/*.log > "$G/summary_fspeed.txt" 2>/dev/null
status body_done 0
exit 0
