#!/bin/bash
# Guarded on-box body for the default-off exact GBDT device winner fold.
# It runs exactly Taxi and Istella-S, each with Depthwise and Lossguide.
set -euo pipefail

[ "${MOJOLEARN_GBDT_DEVICE_WINNER_RUN_GUARD:-}" = "R2_TAXI_ISTELLA" ] || {
    echo "refusing: set MOJOLEARN_GBDT_DEVICE_WINNER_RUN_GUARD=R2_TAXI_ISTELLA" >&2
    exit 64
}

R=${MOJOLEARN_TRIAL_ROOT:-/root/mojolearn}
B=${MOJOLEARN_GBDT_WINNER_BASELINE_ROOT:-/root/mojolearn-gdw-baseline}
S=${MOJOLEARN_GBDT_WINNER_SABOTAGE_ROOT:-/root/mojolearn-gdw-sabotage}
OUT=${MOJOLEARN_GBDT_WINNER_OUT:-/root/gbdt_device_winner_out}
DATA=${GBM_BENCH_DATA:-/root/datasets/gbm-bench}
ROWS=${MOJOLEARN_GBDT_WINNER_ROWS:-1000000}
ROUNDS=${MOJOLEARN_GBDT_WINNER_RETAINED:-5}
OUTERS=${MOJOLEARN_GBDT_WINNER_OUTERS:-3}
PY=${PYTHON:-$R/.pixi/envs/default/bin/python}
mkdir -p "$OUT/logs" "$OUT/json"
export GBM_BENCH_DATA="$DATA"
export MOJOLEARN_NUMERIC_MODE=identical
export PYTHONUNBUFFERED=1

say() { printf '[%s gdw] %s\n' "$(date +%T)" "$*"; }
commit_of() {
    if [ -f "$1/SHIPPED_COMMIT.txt" ]; then cat "$1/SHIPPED_COMMIT.txt"
    elif [ -f "$1/MOJOLEARN_COMMIT" ]; then cat "$1/MOJOLEARN_COMMIT"
    else git -C "$1" rev-parse HEAD
    fi
}
source_path() {
    case "$1" in
        taxi) printf '%s/taxi/taxi_speed.npz\n' "$DATA" ;;
        istella) printf '%s/istella/istella_speed.npz\n' "$DATA" ;;
        *) return 2 ;;
    esac
}
expected_sha() {
    case "$1" in
        taxi) echo 10d5d35f376a5b2ad5c66fabe624ee2caafb58bc5e7516824f801de9aab6cc15 ;;
        istella) echo 31f042376c0b819fe169cbbd998e840567c23dae25c14cb9054b460292f6ffef ;;
        *) return 2 ;;
    esac
}
verify_data() {
    for ds in taxi istella; do
        p=$(source_path "$ds")
        [ -f "$p" ] || { echo "missing R2-staged input $p" >&2; return 1; }
        got=$(sha256sum "$p" | awk '{print $1}')
        want=$(expected_sha "$ds")
        [ "$got" = "$want" ] || { echo "$ds sha256 $got, expected $want" >&2; return 1; }
        printf '%s\t%s\t%s\n' "$ds" "$(stat -c %s "$p")" "$got" >> "$OUT/input_manifest.tsv"
    done
}
build_one() {
    tree=$1; label=$2; defines=${3:-}
    rm -f "$tree/python/mojolearn/identical/_mojolearn_gbdt.so"
    (cd "$tree" && MOJOLEARN_EXTRA_DEFINES="$defines" bash bindings/build_gbdt.sh) \
        > "$OUT/logs/build_${label}.log" 2>&1
    sha256sum "$tree/python/mojolearn/identical/_mojolearn_gbdt.so" \
        > "$OUT/${label}_binding.sha256"
}
clone_tree() {
    src=$1; dst=$2
    mkdir -p "$dst"
    (cd "$src" && tar --exclude=.git --exclude=.pixi -cf - .) | (cd "$dst" && tar -xf -)
    ln -s "$R/.pixi" "$dst/.pixi"
}

phase_build() {
    rm -rf "$B" "$S"
    cd "$R" || return 1
    [ -x .pixi/envs/default/bin/python ] || pixi install
    [ -f python/mojolearn/identical/_mojolearn.so ] || bash bindings/build.sh
    build_one "$R" baseline ""
    clone_tree "$R" "$B"
    build_one "$R" candidate "-D MOJOLEARN_EXPERIMENTAL_IDENTICAL_DEVICE_WINNER_FOLD=1"
    clone_tree "$R" "$S"
    build_one "$S" sabotage "-D MOJOLEARN_EXPERIMENTAL_IDENTICAL_DEVICE_WINNER_FOLD=1 -D MOJOLEARN_SABOTAGE_IDENTICAL_DEVICE_WINNER_FOLD=1"
    {
        echo "source_commit=$(commit_of "$R")"
        "$R/.pixi/envs/default/bin/mojo" --version
        uname -a
        command -v rocm-smi >/dev/null && rocm-smi --showproductname --showdriverversion
        command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    } > "$OUT/build_provenance.txt" 2>&1
}

run_fit() {
    tree=$1; arm=$2; ds=$3; policy=$4; outer=$5; position=$6; rounds=$7; warmup=$8
    p=$(source_path "$ds"); sha=$(expected_sha "$ds")
    tag="${ds}_${policy}_${arm}_${outer}"
    (cd "$tree" && PYTHONPATH=python "$PY" tools/gbdt_device_winner_probe.py fit \
        --dataset "$ds" --policy "$policy" --arm "$arm" --outer "$outer" \
        --launch-position "$position" \
        --rounds "$rounds" --rows "$ROWS" --warmup "$warmup" \
        --source-path "$p" --source-sha256 "$sha" --commit "$(commit_of "$R")" \
        --json "$OUT/json/$tag.json") > "$OUT/logs/$tag.log" 2>&1
    rc=$?
    say "$tag rc=$rc"
    return "$rc"
}

phase_run() {
    : > "$OUT/input_manifest.tsv"
    : > "$OUT/run_order.tsv"
    verify_data || return 1
    [ "$OUTERS" -eq 3 ] || { echo "gate requires exactly 3 fresh-process outers" >&2; return 2; }
    [ "$ROUNDS" -ge 5 ] || { echo "gate requires at least 5 retained fits" >&2; return 2; }
    for ds in taxi istella; do
        for policy in Depthwise Lossguide; do
            for outer in 1 2 3; do
                if [ $((outer % 2)) -eq 1 ]; then
                    printf '%s\t%s\t%d\tbaseline,candidate\n' "$ds" "$policy" "$outer" >> "$OUT/run_order.tsv"
                    run_fit "$B" baseline "$ds" "$policy" "$outer" 0 "$ROUNDS" 1 || return 1
                    run_fit "$R" candidate "$ds" "$policy" "$outer" 1 "$ROUNDS" 1 || return 1
                else
                    printf '%s\t%s\t%d\tcandidate,baseline\n' "$ds" "$policy" "$outer" >> "$OUT/run_order.tsv"
                    run_fit "$R" candidate "$ds" "$policy" "$outer" 0 "$ROUNDS" 1 || return 1
                    run_fit "$B" baseline "$ds" "$policy" "$outer" 1 "$ROUNDS" 1 || return 1
                fi
            done
            run_fit "$S" sabotage "$ds" "$policy" 1 0 1 0 || return 1
        done
    done
    (cd "$R" && PYTHONPATH=python "$PY" tools/gbdt_device_winner_probe.py summarize \
        "$OUT/json/*.json" --json "$OUT/summary.json") | tee "$OUT/summary.txt"
}

case "${1:-}" in
    build) phase_build ;;
    run) phase_run ;;
    all) phase_build && phase_run ;;
    *) echo "usage: $0 build|run|all" >&2; exit 64 ;;
esac
