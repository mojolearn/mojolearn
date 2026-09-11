#!/bin/sh
# tools/gbdt_per_round_leg.sh -- lane gbdt-per-round, DEVIATIONS 2550 (X read
# in place) and 2551 (device leaf partition): the MI325X A/B. Runs ON THE
# DROPLET as tools/do_extra_leg.sh's MOJOLEARN_GEMM_LEG_EXTRA body, from
# /root/mojolearn with pixi on PATH and MOJOLEARN_GPU_ARCHS /
# MOJOLEARN_TARGET_COLUMN exported. Everything lands under
# /root/gemm_leg_out/gpr/ and comes home with the leg's fetch.
#
#   bash tools/gbdt_per_round_do.sh     (the shared lock, its order, then:)
#   MOJOLEARN_DO_TOKEN_FILE=$HOME/.mojolearn_do_token MOJOLEARN_GPU_ARCHS=gfx942 \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/gbdt_per_round_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/<stamp>-amd-mi325x-gbdt-per-round \
#   bash tools/do_extra_leg.sh amd --minutes 60 --skip-gates
#
# PHASES, each with a row in status.tsv; a red phase does not stop the next:
#   deps        system python3 + pip: numpy pyarrow pandas (the ROCm image's
#               python has none; the bindings are version agnostic)
#   data        background: speed_gbdt_arm.py --download taxi, then istella
#   build       IDENTICAL: base, gbdt x {default, a2550, a2551}; FAST: gbdt x
#               the same three (the base binding is identical only, 2490)
#   checks      pixi run check-gbdt-per-round{,-2550,-2551} in parallel;
#               MODEL_HASH lines must be equal across the three (rule 8)
#   ib          identity_break, gbdt lanes, IDENTICAL, one json per set
#               (diffed on the Mac against the retained fingerprint set)
#   speed       per cell (lane x dataset) one process per set, warm-up + 5
#               rounds, set order ROTATED per cell (not per round), 1M rows
#   stage       one MOJOLEARN_STAGE_TIMES=1 replicate per set
#   ORDER       IDENTICAL speed (taxi, then istella) > IDENTICAL stage istella
#               > FAST speed depthwise + symmetric > stage taxi > FAST
#               lossguide (dropped first when time runs short)
#
# POSIX sh only (dash), `set -u` not `set -e`.
set -u
ROOT=/root/mojolearn
G=/root/gemm_leg_out/gpr
BINS=/root/gpr_bins
DATA="${GBM_BENCH_DATA:-/root/datasets/gbm-bench}"
GBM_BENCH_DATA="$DATA"
export GBM_BENCH_DATA
export DEBIAN_FRONTEND=noninteractive
PY=/usr/bin/python3
mkdir -p "$G/logs" "$G/speed" "$G/ib" "$G/checks" "$BINS"
cd "$ROOT" || exit 9
PATH="$HOME/.pixi/bin:$PATH"
export PATH
JOBS=$(nproc 2>/dev/null || echo 8)
BODY_START=$(date +%s)

# THE DEADLINE: the start wrapper's timeout bound, from its argv and the
# remote body's started= line; a conservative bound when either is missing.
WORK=$(ps -eo args | sed -n 's/^timeout -k 30 \([0-9][0-9]*\) sh \/root\/gemm_leg\.sh.*$/\1/p' | head -n 1)
STARTED=$(sed -n 's/^started=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -n 1)
ST_EPOCH=$(date -u -d "$STARTED" +%s 2>/dev/null || echo "")
if [ -n "$WORK" ] && [ -n "$ST_EPOCH" ]; then
    DEADLINE=$((ST_EPOCH + WORK - 90))
else
    DEADLINE=$((BODY_START + 2400))
fi
echo "deadline_epoch=$DEADLINE work=$WORK started=$STARTED body_start=$BODY_START jobs=$JOBS" > "$G/deadline.txt"

left() { echo $((DEADLINE - $(date +%s))); }
status() { printf '%s\t%s\t%s\tleft=%s\n' "$1" "$2" "$(date -u +%H:%M:%S)" "$(left)" >> "$G/status.tsv"; }
status body_start 0

# ---------------------------------------------------------------- deps
# A fresh droplet's unattended upgrade holds the dpkg lock at boot; wait on it
# (tools/trees_amd_remote.sh, 92b4bf9b) rather than fail the install.
if ! $PY -c 'import pip' > /dev/null 2>&1; then
    timeout -k 10 600 sh -c 'apt-get -o DPkg::Lock::Timeout=180 update -qq; apt-get -o DPkg::Lock::Timeout=180 install -y --no-install-recommends python3-pip' \
        > "$G/logs/apt.log" 2>&1
    status apt_pip $?
fi
timeout -k 10 600 $PY -m pip install --break-system-packages --disable-pip-version-check \
    --no-input numpy pyarrow pandas > "$G/logs/pip.log" 2>&1
status deps $?
$PY -c "import numpy, pyarrow, pandas; print('numpy', numpy.__version__, 'pyarrow', pyarrow.__version__, 'pandas', pandas.__version__)" > "$G/versions.txt" 2>&1

# ---------------------------------------------------------------- data (background)
(
    timeout -k 10 900 $PY tools/speed_gbdt_arm.py --download taxi > "$G/logs/download_taxi.log" 2>&1
    echo "download_taxi=$?" >> "$G/data.txt"
    timeout -k 10 1800 $PY tools/speed_gbdt_arm.py --download istella > "$G/logs/download_istella.log" 2>&1
    echo "download_istella=$?" >> "$G/data.txt"
    : > "$G/data.done"
) &

# ---------------------------------------------------------------- builds
build() {  # <mode> <set> <script> <so> <defines>
    _mode=$1; _set=$2; _script=$3; _so=$4; _defs=$5
    mkdir -p "$BINS/$_mode/$_set"
    _dir=python/mojolearn
    [ "$_mode" = identical ] && _dir=python/mojolearn/identical
    mkdir -p "$_dir"
    rm -f "$_dir/$_so"
    MOJOLEARN_NUMERIC_MODE=$_mode MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=$JOBS \
        MOJOLEARN_EXTRA_DEFINES="$_defs" \
        timeout -k 30 1200 bash "$_script" > "$G/logs/build.$_mode.$_set.$_so.log" 2>&1
    _rc=$?
    if [ "$_rc" = 0 ] && [ -f "$_dir/$_so" ]; then
        cp "$_dir/$_so" "$BINS/$_mode/$_set/$_so"
        sha256sum "$BINS/$_mode/$_set/$_so" >> "$G/so_sha256.txt"
    else
        grep -m 8 -i 'error' "$G/logs/build.$_mode.$_set.$_so.log" >> "$G/build_errors.txt"
    fi
    status "build.$_mode.$_set.$_so" "$_rc"
}
use() {  # <mode> <set>
    _dir=python/mojolearn
    [ "$1" = identical ] && _dir=python/mojolearn/identical
    cp "$BINS/$1/$2/_mojolearn_gbdt.so" "$_dir/_mojolearn_gbdt.so"
}

D2550="-D MOJOLEARN_2550_BORROW_X=1"
D2551="-D MOJOLEARN_2551_DEVICE_PARTITION=1"
build identical base bindings/build.sh _mojolearn.so ""
cp python/mojolearn/identical/_mojolearn.so "$BINS/identical_base_mojolearn.so" 2>/dev/null
build identical default bindings/build_gbdt.sh _mojolearn_gbdt.so ""
build identical a2550 bindings/build_gbdt.sh _mojolearn_gbdt.so "$D2550"
build identical a2551 bindings/build_gbdt.sh _mojolearn_gbdt.so "$D2551"

# ---------------------------------------------------------------- checks (background)
for c in check-gbdt-per-round check-gbdt-per-round-2550 check-gbdt-per-round-2551; do
    ( timeout -k 30 1500 pixi run "$c" > "$G/checks/$c.log" 2>&1; echo "$?" > "$G/checks/$c.exit" ) &
done
status checks_started 0

build fast default bindings/build_gbdt.sh _mojolearn_gbdt.so ""
build fast a2550 bindings/build_gbdt.sh _mojolearn_gbdt.so "$D2550"
build fast a2551 bindings/build_gbdt.sh _mojolearn_gbdt.so "$D2551"

# ---------------------------------------------------------------- import
imp() {  # <mode>
    MOJOLEARN_NUMERIC_MODE=$1 PYTHONPATH="$ROOT/python" timeout -k 10 180 $PY -c \
        "import mojolearn; m = mojolearn.GradientBoosting(); b = m._bind(); print('import OK', m.numeric_mode_used(), m.vendor_used(), b.gbdt_per_round_paths())" \
        > "$G/logs/import.$1.log" 2>&1
}
use identical default
use fast default
imp identical; _ri=$?
imp fast; _rf=$?
if [ "$_ri" != 0 ] || [ "$_rf" != 0 ]; then
    status import_first_try "identical=$_ri fast=$_rf"
    for b in rf trees; do
        MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=$JOBS \
            timeout -k 30 900 bash "bindings/build_$b.sh" > "$G/logs/build.identical.$b.log" 2>&1
        MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=$JOBS \
            timeout -k 30 900 bash "bindings/build_$b.sh" > "$G/logs/build.fast.$b.log" 2>&1
    done
    imp identical; _ri=$?
    imp fast; _rf=$?
fi
status import "identical=$_ri fast=$_rf"

# ---------------------------------------------------------------- checks: wait and diff
for c in check-gbdt-per-round check-gbdt-per-round-2550 check-gbdt-per-round-2551; do
    while [ ! -f "$G/checks/$c.exit" ] && [ "$(left)" -gt 900 ]; do sleep 10; done
    status "$c" "$(cat "$G/checks/$c.exit" 2>/dev/null || echo UNFINISHED)"
    grep '^MODEL_HASH' "$G/checks/$c.log" > "$G/checks/$c.hash" 2>/dev/null
done
if [ -s "$G/checks/check-gbdt-per-round.hash" ] \
    && cmp -s "$G/checks/check-gbdt-per-round.hash" "$G/checks/check-gbdt-per-round-2550.hash" \
    && cmp -s "$G/checks/check-gbdt-per-round.hash" "$G/checks/check-gbdt-per-round-2551.hash"; then
    status model_hashes_equal_across_builds 0
else
    status model_hashes_equal_across_builds 1
fi

# ---------------------------------------------------------------- identity_break
ib() {  # <set> <repeats>
    use identical "$1"
    MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$ROOT/python" timeout -k 30 600 $PY -u tools/identity_break.py \
        --lanes gbdt-symmetric,gbdt-depthwise,gbdt-lossguide --repeats "$2" --vendor amd-mi325x \
        --json "$G/ib/$1.json" > "$G/ib/$1.txt" 2>&1
    status "ib.$1" $?
}
ib default 2
ib a2550 1
ib a2551 1

# ---------------------------------------------------------------- speed
data_ready() {  # <dataset>: wait for its npz, bounded
    while [ ! -f "$DATA/$1/$1_speed.npz" ] && [ ! -f "$G/data.done" ] && [ "$(left)" -gt 300 ]; do sleep 10; done
    [ -f "$DATA/$1/$1_speed.npz" ]
}
speed() {  # <mode> <set> <lane> <dataset> <timed|stage>
    _mode=$1; _set=$2; _lane=$3; _ds=$4; _kind=$5
    _tag="$_kind.$_mode.$_set.$_lane.$_ds"
    _need=150
    [ "$_ds" = istella ] && _need=260
    [ "$_kind" = stage ] && _need=$((_need / 2))
    if [ "$(left)" -lt "$_need" ]; then status "$_tag" SKIPPED_TIME; return 0; fi
    if [ ! -f "$BINS/$_mode/$_set/_mojolearn_gbdt.so" ]; then status "$_tag" NO_BINARY; return 0; fi
    use "$_mode" "$_set"
    _rounds=5; _st=0
    [ "$_kind" = stage ] && { _rounds=1; _st=1; }
    MOJOLEARN_NUMERIC_MODE=$_mode MOJOLEARN_SPEED_EXPECTED_VENDOR=hip MOJOLEARN_SPEED_SIZE=shipped \
        MOJOLEARN_SPEED_BUDGET_S=900 MOJOLEARN_SPEED_DEADLINE_S=1200 MOJOLEARN_SPEED_DEVICE=AMD_MI325X \
        MOJOLEARN_SPEED_ROUNDS=$_rounds MOJOLEARN_STAGE_TIMES=$_st PYTHONPATH="$ROOT/python" \
        timeout -k 20 $(( $(left) - 20 )) $PY -u bench/speed/forest_speed_arm.py \
        --lane "$_lane" --dataset "$_ds" --rows 1000000 --ours-only > "$G/speed/$_tag.log" 2>&1
    status "$_tag" $?
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
cell() {  # <mode> <kind> <lane> <dataset> <rotation> <sets...>
    _cm=$1; _ck=$2; _cl=$3; _cd=$4; _cr=$5; shift 5
    if ! data_ready "$_cd"; then status "cell.$_ck.$_cm.$_cl.$_cd" NO_DATA; return 0; fi
    for _s in $(rotate "$_cr" "$@"); do
        speed "$_cm" "$_s" "$_cl" "$_cd" "$_ck"
    done
}
ALL3="default a2550 a2551"
SYM2="default a2550"
# shellcheck disable=SC2086
{
    cell identical timed gbdt-depthwise taxi 0 $ALL3
    cell identical timed gbdt-lossguide taxi 1 $ALL3
    cell identical timed gbdt-symmetric taxi 0 $SYM2
    cell identical timed gbdt-depthwise istella 2 $ALL3
    cell identical timed gbdt-lossguide istella 0 $ALL3
    cell identical timed gbdt-symmetric istella 1 $SYM2
    cell identical stage gbdt-depthwise istella 0 $ALL3
    cell identical stage gbdt-lossguide istella 0 $ALL3
    cell fast timed gbdt-depthwise taxi 1 $ALL3
    cell fast timed gbdt-symmetric taxi 1 $SYM2
    cell fast timed gbdt-depthwise istella 0 $ALL3
    cell fast timed gbdt-symmetric istella 0 $SYM2
    cell identical stage gbdt-depthwise taxi 0 $ALL3
    cell identical stage gbdt-lossguide taxi 0 $ALL3
    cell identical stage gbdt-symmetric istella 0 $SYM2
    cell fast timed gbdt-lossguide taxi 2 $ALL3
    cell fast timed gbdt-lossguide istella 1 $ALL3
}

grep -h '^FSPEED\|^BENCH_PATHS\|^BENCH_BINDING' "$G"/speed/*.log > "$G/summary_fspeed.txt" 2>/dev/null
status body_done 0
exit 0
