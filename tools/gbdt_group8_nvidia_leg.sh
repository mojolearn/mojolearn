#!/bin/sh
# On-box RunPod body for the IDENTICAL symmetric-GBDT resident-apply group8
# experiment.  The caller stages exactly the taxi and Istella-S NPZ files
# from Cloudflare R2 before invoking this script.  This body never downloads
# data and never rents or reaps a machine.
#
# It builds two packages from the same source tree: the shipped group4 path
# and a group8 candidate selected only by a compile define.  One baseline
# build prepares each model once.  The two packages then run in alternating
# processes over the same saved model and prediction bytes.  Each process has
# one excluded warmup and at least five timed calls; three process pairs are
# used by default.  Both predict and predict_proba are measured on both of the
# project's pinned medium/large datasets, with full output/input/model hashes
# and quality in every JSON record.
set -u

R=${MOJOLEARN_BOX_REPO:-/root/mojolearn}
OUT=${GROUP8_OUT:-/root/leg_out/group8}
AB=${GROUP8_AB:-/root/group8_ab}
ARMS=${GROUP8_ARMS:-/root/group8_arms}
ROWS=${GROUP8_ROWS:-1000000}
TRAIN_ROWS=${GROUP8_TRAIN_ROWS:-1000000}
TREES=${GROUP8_TREES:-256}
ROUNDS=${GROUP8_ROUNDS:-5}
PASSES=${GROUP8_PASSES:-3}
DATA=${GBM_BENCH_DATA:-/root/datasets/gbm-bench}
export GBM_BENCH_DATA="$DATA"
export MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_COMPILE_JOBS="${MOJOLEARN_COMPILE_JOBS:-16}"
export MOJOLEARN_BUILD_JOBS="${MOJOLEARN_BUILD_JOBS:-16}"
export PYTHONUNBUFFERED=1
export PATH="$HOME/.pixi/bin:$PATH"

say() { printf '[%s group8] %s\n' "$(date +%T)" "$*"; }
run() {
    _name=$1
    shift
    say "$_name: $*"
    "$@" >"$OUT/$_name.log" 2>&1
    _rc=$?
    say "$_name rc=$_rc"
    return "$_rc"
}

require_data() {
    _missing=0
    for _p in "$DATA/taxi/taxi_speed.npz" "$DATA/istella/istella_speed.npz"; do
        if [ ! -s "$_p" ]; then
            echo "REFUSED: R2-staged dataset missing: $_p" >&2
            _missing=1
        fi
    done
    [ "$_missing" -eq 0 ]
}

build_arm() {
    _label=$1
    _defines=$2
    cd "$R" || return 1
    rm -f python/mojolearn/identical/_mojolearn_gbdt.so
    run "build_$_label" env MOJOLEARN_EXTRA_DEFINES="$_defines" sh bindings/build_gbdt.sh || return 1
    rm -rf "${ARMS:?}/$_label"
    mkdir -p "$ARMS/$_label"
    cp -R python/mojolearn "$ARMS/$_label/"
    sha256sum "$ARMS/$_label/mojolearn/identical/_mojolearn_gbdt.so" \
        >"$OUT/${_label}_binding.sha256"
}

phase_setup() {
    mkdir -p "$OUT" "$AB" "$ARMS"
    require_data || return 1
    cd "$R" || return 1
    if [ ! -x "$HOME/.pixi/bin/pixi" ]; then
        curl -fsSL https://pixi.sh/install.sh | sh >"$OUT/pixi_install.log" 2>&1 || return 1
    fi
    run pixi_install pixi install || return 1
    pixi run mojo --version >"$OUT/mojo_version.txt" 2>&1
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >"$OUT/gpu.txt" 2>&1
    build_arm base "" || return 1
    build_arm group8 "-D MOJOLEARN_GBDT_GROUP8=1" || return 1
    if cmp -s "$ARMS/base/mojolearn/identical/_mojolearn_gbdt.so" \
              "$ARMS/group8/mojolearn/identical/_mojolearn_gbdt.so"; then
        echo "REFUSED: base and group8 bindings are byte-identical; compile define did not select candidate" >&2
        return 1
    fi
    cd "$R" || return 1
    run prepare env PYTHONPATH="$ARMS/base" pixi run python3 bench/speed/infer_speed_trees_ab.py prepare \
        --out "$AB" --rows "$ROWS" --train-rows "$TRAIN_ROWS" \
        --gbdt-iterations "$TREES" --gbdt-datasets taxi,istella \
        --gbdt-only || return 1
    sha256sum "$AB"/*.npz "$AB"/*.npy >"$OUT/prepared.sha256"
}

time_one() {
    _arm=$1
    _dataset=$2
    _path=$3
    _pass=$4
    case "$_dataset" in
        taxi) _model="$AB/gbdt-logloss-$TREES.npz" ;;
        istella) _model="$AB/gbdt-istella-logloss-$TREES.npz" ;;
        *) return 2 ;;
    esac
    _json="$OUT/json/${_dataset}.${_path}.${_arm}.${_pass}.json"
    env PYTHONPATH="$ARMS/$_arm" pixi run python3 bench/speed/infer_speed_trees_ab.py time \
        --model "$_model" --x "$AB/x_${_dataset}.npy" --y "$AB/y_${_dataset}.npy" \
        --dataset "$_dataset" --task binary --kind gbdt --path "gpu-$_path" \
        --rounds "$ROUNDS" --label "$_arm" --json "$_json"
}

phase_run() {
    require_data || return 1
    mkdir -p "$OUT/json" "$OUT/summary"
    cd "$R" || return 1
    _pass=1
    while [ "$_pass" -le "$PASSES" ]; do
        for _dataset in taxi istella; do
            for _path in predict proba; do
                # Reverse every other pair so thermal/order drift is balanced.
                if [ $((_pass % 2)) -eq 1 ]; then
                    _order="base group8"
                else
                    _order="group8 base"
                fi
                for _arm in $_order; do
                    run "time.${_dataset}.${_path}.${_arm}.${_pass}" \
                        time_one "$_arm" "$_dataset" "$_path" "$_pass" || return 1
                done
            done
        done
        _pass=$((_pass + 1))
    done
    for _dataset in taxi istella; do
        for _path in predict proba; do
            env PYTHONPATH="$ARMS/base" pixi run python3 bench/speed/infer_speed_trees_ab.py summarize \
                "$OUT/json/${_dataset}.${_path}."'*.json' --before base --after group8 \
                --out "$OUT/summary/${_dataset}.${_path}.json" \
                >"$OUT/summary/${_dataset}.${_path}.txt" || return 1
        done
    done
    # Identity and quality are hard gates.  Timing instability remains in the
    # evidence as an unqualified cell and can be rerun without hiding results.
    env GROUP8_RESULT_DIR="$OUT" pixi run python3 - <<'PY'
import glob, json, os, sys
root = os.environ["GROUP8_RESULT_DIR"]
bad = []
for path in sorted(glob.glob(root + "/summary/*.json")):
    rows = json.load(open(path))
    if len(rows) != 1:
        bad.append((path, "expected one summary row"))
        continue
    row = rows[0]
    if not row.get("hashes_equal_across_arms"):
        bad.append((path, "prediction bits differ"))
    if not row.get("quality_equal_across_arms"):
        bad.append((path, "quality differs"))
if bad:
    for item in bad:
        print("IDENTITY-FAIL", *item)
    sys.exit(1)
print("IDENTITY-PASS: taxi and Istella-S predict/proba hashes and quality match")
PY
}

case "${1:-all}" in
    setup) phase_setup ;;
    run) phase_run ;;
    all) phase_setup && phase_run ;;
    *) echo "usage: $0 [setup|run|all]" >&2; exit 2 ;;
esac
