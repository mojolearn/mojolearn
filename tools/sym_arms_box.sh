#!/usr/bin/env bash
# RUNS ON THE MI325X. The gbdt-symmetric-arms lane's payload, driven by
# tools/sym_arms_do_leg.sh (which ships the commit, arms the watchdogs and
# destroys the droplet). Every phase respects SYMARMS_DEADLINE (epoch
# seconds) and never starts work it cannot finish before it.
#
#   SYMARMS_DEADLINE=<epoch> SYMARMS_TIERS="identical fast" bash tools/sym_arms_box.sh all
#   phases: setup builds checks identity speed stage verdict all
#
# Arms: default, a2580 (-D MOJOLEARN_2580_LEVEL_QUANT=1), b2581
# (-D MOJOLEARN_2581_GROUP_WIDTH=1), ab (both). One gbdt binary per
# (tier, arm) under /root/symarms_bins, swapped into the tier directory
# before each process, so every timed round is its own process with its
# own warm-up and the arms interleave round by round.
set -uo pipefail
ROOT=/root/mojolearn
OUT=/root/symarms_out
BINS=/root/symarms_bins
mkdir -p "$OUT/logs" "$OUT/speed" "$OUT/ib" "$OUT/stage" "$BINS"
cd "$ROOT" || exit 9
export PATH="/root/.pixi/bin:$PATH"
# The trees AMD leg's proven choices (tools/trees_amd_remote.sh): the
# harness on the image's python3 with pip packages, and the datasets and
# caches on the persistent tor1 volume when it is mounted.
PY=python3
if mountpoint -q /mnt/mojolearn-data 2>/dev/null; then
    export GBM_BENCH_DATA=/mnt/mojolearn-data/gbm-bench
    export PIP_CACHE_DIR=/mnt/mojolearn-data/pip-cache
    export RATTLER_CACHE_DIR=/mnt/mojolearn-data/rattler-cache
    mkdir -p "$GBM_BENCH_DATA"
fi
export DEBIAN_FRONTEND=noninteractive
BPATH="$ROOT/.pixi/envs/default/bin:$PATH"
ARMS="default a2580 b2581 ab"
TIERS="${SYMARMS_TIERS:-identical fast}"
ROWS="${SYMARMS_ROWS:-1000000}"
ROUNDS="${SYMARMS_ROUNDS:-5}"
DEADLINE="${SYMARMS_DEADLINE:-$(( $(date +%s) + 3000 ))}"
JOBS="$(nproc 2>/dev/null || echo 8)"

note() { echo "$* $(date -u +%H:%M:%S)" | tee -a "$OUT/ab.txt"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
defines_for() {
    case "$1" in
        default) echo "" ;;
        a2580) echo "-D MOJOLEARN_2580_LEVEL_QUANT=1" ;;
        b2581) echo "-D MOJOLEARN_2581_GROUP_WIDTH=1" ;;
        ab) echo "-D MOJOLEARN_2580_LEVEL_QUANT=1 -D MOJOLEARN_2581_GROUP_WIDTH=1" ;;
    esac
}
tier_dir() { if [ "$1" = fast ]; then echo python/mojolearn; else echo "python/mojolearn/$1"; fi; }
use_bin() { cp "$BINS/$1-$2/_mojolearn_gbdt.so" "$(tier_dir "$1")/_mojolearn_gbdt.so"; }

phase_setup() {
    command -v pixi >/dev/null 2>&1 || curl -fsSL https://pixi.sh/install.sh | bash > "$OUT/logs/pixi_bootstrap.log" 2>&1
    export PATH="/root/.pixi/bin:$PATH"
    timeout -k 30 1500 pixi install > "$OUT/logs/pixi_install.log" 2>&1
    note "pixi_install_exit=$?"
    if ! python3 -c 'import pip, venv, ensurepip' > /dev/null 2>&1; then
        timeout -k 30 600 sh -c 'apt-get -o DPkg::Lock::Timeout=180 update -qq; apt-get -o DPkg::Lock::Timeout=180 install -y --no-install-recommends python3-pip python3-venv' \
            > "$OUT/logs/apt_pip.log" 2>&1
        note "apt_pip_exit=$?"
    fi
    timeout -k 30 900 python3 -m pip install --break-system-packages --no-input \
        --disable-pip-version-check numpy pandas pyarrow scikit-learn \
        > "$OUT/logs/pip_base.log" 2>&1
    note "pip_base_exit=$?"
    echo "GBM_BENCH_DATA=${GBM_BENCH_DATA:-default}" >> "$OUT/ab.txt"
    ls -la "${GBM_BENCH_DATA:-/root/datasets/gbm-bench}"/taxi "${GBM_BENCH_DATA:-/root/datasets/gbm-bench}"/istella \
        > "$OUT/datasets_before.txt" 2>&1
    rocm-smi --showproductname > "$OUT/gpu.txt" 2>&1
    pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1
    nproc > "$OUT/nproc.txt"
    # the datasets download beside the builds; the speed phase waits for it
    (
        for ds in taxi istella; do
            timeout -k 30 2400 "$PY" tools/speed_gbdt_arm.py --download "$ds" \
                > "$OUT/logs/download_$ds.log" 2>&1
            echo "download_${ds}_exit=$? $(date -u +%H:%M:%S)" >> "$OUT/ab.txt"
        done
        touch "$OUT/DONE.download"
    ) > /dev/null 2>&1 < /dev/null &
}

build_one() {
    # build_one <script> <tier> <label> <defines>
    _script="$1"; _tier="$2"; _label="$3"; _defs="$4"
    _so="_mojolearn_gbdt.so"
    [ "$_script" = build.sh ] && _so="_mojolearn.so"
    [ "$_script" = build_rf.sh ] && _so="_mojolearn_rf.so"
    [ "$_script" = build_trees.sh ] && _so="_mojolearn_trees.so"
    if [ "$(left)" -lt 240 ]; then note "build_skipped_deadline $_tier.$_label"; return 3; fi
    mkdir -p "$(tier_dir "$_tier")"
    rm -f "$(tier_dir "$_tier")/$_so"
    PATH="$BPATH" MOJOLEARN_NUMERIC_MODE="$_tier" MOJOLEARN_SKIP_BUILD_GATE=1 \
        MOJOLEARN_COMPILE_JOBS="$JOBS" MOJOLEARN_EXTRA_DEFINES="$_defs" \
        timeout -k 30 1500 bash "bindings/$_script" \
        > "$OUT/logs/$_script.$_tier.$_label.log" 2>&1
    _rc=$?
    note "build_exit $_script.$_tier.$_label=$_rc"
    if [ "$_rc" = 0 ] && [ -f "$(tier_dir "$_tier")/$_so" ]; then
        mkdir -p "$BINS/$_tier-$_label"
        cp "$(tier_dir "$_tier")/$_so" "$BINS/$_tier-$_label/$_so"
        sha256sum "$BINS/$_tier-$_label/$_so" | tee -a "$OUT/ab.txt"
        return 0
    fi
    grep -m8 -iE 'error|failed' "$OUT/logs/$_script.$_tier.$_label.log" | tee -a "$OUT/ab.txt"
    return 1
}

phase_builds() {
    # shared identical bindings the package imports beside gbdt
    for s in build.sh build_rf.sh build_trees.sh; do
        build_one "$s" identical shared ""
    done
    for tier in $TIERS; do
        for arm in default ab a2580 b2581; do
            build_one build_gbdt.sh "$tier" "$arm" "$(defines_for "$arm")"
        done
    done
}

phase_checks() {
    for tier in $TIERS; do
        if [ "$(left)" -lt 300 ]; then note "check_skipped_deadline $tier"; continue; fi
        _def=""
        [ "$tier" = identical ] && _def="-D MOJOLEARN_NUMERIC_IDENTICAL=1"
        # shellcheck disable=SC2086
        PATH="$BPATH" timeout -k 30 900 pixi run mojo run -I . $_def \
            checks/sym_arms_check.mojo > "$OUT/logs/check_sym_arms.$tier.log" 2>&1
        note "check_sym_arms_exit $tier=$?"
        tail -3 "$OUT/logs/check_sym_arms.$tier.log" | tee -a "$OUT/ab.txt"
    done
}

phase_identity() {
    case " $TIERS " in *" identical "*) : ;; *) note "identity_skipped_no_identical_tier"; return 0 ;; esac
    for arm in default ab a2580 b2581; do
        [ -f "$BINS/identical-$arm/_mojolearn_gbdt.so" ] || { note "ib_missing identical.$arm"; continue; }
        if [ "$(left)" -lt 300 ]; then note "ib_skipped_deadline $arm"; continue; fi
        use_bin identical "$arm"
        MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$ROOT/python" timeout -k 30 900 \
            "$PY" -u tools/identity_break.py \
            --lanes gbdt-symmetric,gbdt-depthwise,gbdt-lossguide,gbdt-rmse \
            --vendor "amd-mi325x-$arm" --json "$OUT/ib/identical.$arm.json" \
            > "$OUT/ib/identical.$arm.txt" 2>&1
        note "ib_exit identical.$arm=$?"
    done
    _sets=""
    for arm in default ab a2580 b2581; do
        [ -f "$OUT/ib/identical.$arm.json" ] && _sets="$_sets $OUT/ib/identical.$arm.json"
    done
    # shellcheck disable=SC2086
    "$PY" tools/identity_break.py --diff /root/retained/amd-mi325x.identical.json \
        /root/retained/apple-m4.identical.json $_sets > "$OUT/ib/diff.retained.txt" 2>&1
    note "ib_diff_retained_exit=$?"
    grep -E '^\| gbdt-|gbdt-' "$OUT/ib/diff.retained.txt" | head -60 > "$OUT/ib/diff.gbdt.txt"
}

speed_one() {
    # speed_one <tier> <arm> <dataset> <round> [stage]
    _t="$1"; _a="$2"; _d="$3"; _r="$4"; _mode="${5:-timed}"
    [ -f "$BINS/$_t-$_a/_mojolearn_gbdt.so" ] || { note "speed_missing $_t.$_a"; return 3; }
    use_bin "$_t" "$_a"
    _dir="$OUT/speed"; _extra=""
    if [ "$_mode" = stage ]; then _dir="$OUT/stage"; _extra="MOJOLEARN_STAGE_TIMES=1"; fi
    echo "SYMARMS-ROUND tier=$_t arm=$_a dataset=$_d round=$_r $(date -u +%H:%M:%S)" >> "$_dir/$_t.$_a.$_d.log"
    env $_extra MOJOLEARN_NUMERIC_MODE="$_t" MOJOLEARN_SPEED_EXPECTED_VENDOR=hip \
        MOJOLEARN_SPEED_SIZE=shipped MOJOLEARN_SPEED_BUDGET_S=900 \
        MOJOLEARN_SPEED_DEADLINE_S=1200 MOJOLEARN_SPEED_ROUNDS=1 \
        MOJOLEARN_GBDT_PATH=1 PYTHONPATH="$ROOT/python" \
        timeout -k 30 900 "$PY" -u bench/speed/forest_speed_arm.py \
        --lane gbdt-symmetric --dataset "$_d" --rows "$ROWS" --ours-only \
        >> "$_dir/$_t.$_a.$_d.log" 2>&1
    note "speed_exit $_mode.$_t.$_a.$_d.r$_r=$?"
}

rotate() {
    # rotate <n> <items...>: the items rotated left n times
    _n="$1"; shift
    set -- "$@"
    _i=0
    while [ "$_i" -lt "$_n" ]; do _f="$1"; shift; set -- "$@" "$_f"; _i=$((_i + 1)); done
    echo "$@"
}

phase_speed() {
    _w=0
    while [ ! -f "$OUT/DONE.download" ] && [ "$(left)" -gt 600 ]; do sleep 10; _w=$((_w + 10)); done
    note "download_wait_s=$_w"
    _tiers="$TIERS"
    _r=1
    while [ "$_r" -le "$ROUNDS" ]; do
        _t0=$(date +%s)
        for ds in taxi istella; do
            for tier in $_tiers; do
                for arm in $(rotate $(( (_r - 1) % 4 )) default a2580 b2581 ab); do
                    if [ "$(left)" -lt 120 ]; then note "speed_stopped_deadline r$_r"; return 0; fi
                    speed_one "$tier" "$arm" "$ds" "$_r"
                done
            done
        done
        _dt=$(( $(date +%s) - _t0 ))
        note "speed_round_s r$_r=$_dt tiers=$(echo $_tiers | tr ' ' ',')"
        if [ "$_r" = 1 ]; then
            _need=$(( _dt * (ROUNDS - 1) + 300 ))
            case " $_tiers " in
                *" fast "*)
                    if [ "$_need" -gt "$(left)" ] && [ "$_tiers" != fast ]; then
                        _tiers="identical"
                        note "speed_fast_deferred_to_second_leg need_s=$_need left_s=$(left)"
                    fi ;;
            esac
        fi
        _r=$((_r + 1))
    done
}

phase_stage() {
    for ds in taxi istella; do
        for tier in $TIERS; do
            for arm in default ab a2580 b2581; do
                if [ "$(left)" -lt 150 ]; then note "stage_stopped_deadline"; return 0; fi
                speed_one "$tier" "$arm" "$ds" 1 stage
            done
        done
    done
}

phase_verdict() {
    for tier in $TIERS; do
        for arm in a2580 b2581 ab; do
            _b="$OUT/speed/$tier.default"; _a="$OUT/speed/$tier.$arm"
            [ -f "$_b.taxi.log" ] && [ -f "$_a.taxi.log" ] || continue
            python3 tools/flip_verdict.py --lane gbdt-symmetric --arm ours --rows "$ROWS" \
                --taxi-before "$_b.taxi.log" --taxi-after "$_a.taxi.log" \
                --istella-before "$_b.istella.log" --istella-after "$_a.istella.log" \
                > "$OUT/verdict.$tier.$arm.txt" 2>&1
            note "verdict $tier.$arm: $(tail -1 "$OUT/verdict.$tier.$arm.txt")"
        done
    done
    grep -h '^FSPEED ' "$OUT"/speed/*.log | sed 's/ ms=[^ ]*//; s/ round=[^ ]*//' \
        | sort | uniq -c > "$OUT/hashes.txt" 2>/dev/null
    for f in "$OUT"/speed/*.log; do
        echo "$(basename "$f") $(grep -h '^GBDT-SYM-PATH' "$f" | sort -u | head -2)"
    done > "$OUT/paths.txt"
}

case "${1:-}" in
    setup) phase_setup ;;
    builds) phase_builds ;;
    checks) phase_checks ;;
    identity) phase_identity ;;
    speed) phase_speed ;;
    stage) phase_stage ;;
    verdict) phase_verdict ;;
    all)
        phase_setup; touch "$OUT/DONE.setup"
        phase_builds; touch "$OUT/DONE.builds"
        phase_checks; touch "$OUT/DONE.checks"
        phase_identity; touch "$OUT/DONE.identity"
        phase_speed; touch "$OUT/DONE.speed"
        phase_stage; touch "$OUT/DONE.stage"
        phase_verdict; touch "$OUT/DONE.verdict"
        touch "$OUT/DONE.all" ;;
    *) sed -n '2,16p' "$0"; exit 2 ;;
esac
