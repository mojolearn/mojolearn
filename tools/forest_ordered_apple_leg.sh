#!/bin/sh
# Local Apple Metal A/B for the ordered resident IDENTICAL forest route.
set -eu
SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH='' cd -- "$SCRIPT_DIR/.." && pwd)
OUT=${MOJOLEARN_FOREST_APPLE_OUT:-/tmp/mojolearn-forest-ordered-apple}
BINS="$OUT/bins"
MODELS="$OUT/models"
TIER="$ROOT/python/mojolearn/identical"
PY="$ROOT/.pixi/envs/default/bin/python3"
OUTERS=${MOJOLEARN_FOREST_ORDERED_OUTERS:-3}
ROUNDS=${MOJOLEARN_FOREST_ORDERED_ROUNDS:-5}
ROWS=${MOJOLEARN_FOREST_ORDERED_ROWS:-250000}
TRAIN_ROWS=${MOJOLEARN_FOREST_ORDERED_TRAIN_ROWS:-250000}
mkdir -p "$OUT" "$BINS/sequential" "$BINS/ordered" "$MODELS"
cd "$ROOT"
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=apple
export MOJOLEARN_SKIP_BUILD_GATE=1 GBM_BENCH_DATA="${GBM_BENCH_DATA:-$HOME/datasets/gbm-bench}"

build() {
    _name=$1; _defines=$2
    MOJOLEARN_EXTRA_DEFINES="$_defines" tools/with_build_lock.sh sh bindings/build_rf.sh \
        > "$OUT/build.$_name.rf.log" 2>&1
    cp "$TIER/_mojolearn_rf.so" "$BINS/$_name/"
    MOJOLEARN_EXTRA_DEFINES="$_defines" tools/with_build_lock.sh sh bindings/build_trees.sh \
        > "$OUT/build.$_name.trees.log" 2>&1
    cp "$TIER/_mojolearn_trees.so" "$BINS/$_name/"
}
use() {
    cp "$BINS/$1/_mojolearn_rf.so" "$TIER/"
    cp "$BINS/$1/_mojolearn_trees.so" "$TIER/"
}

tools/with_build_lock.sh sh bindings/build.sh > "$OUT/build.base.log" 2>&1
build sequential ""
build ordered "-D MOJOLEARN_FOREST_ORDERED_RESIDENT=1"
use sequential
PYTHONPATH=python "$PY" -u bench/speed/forest_ordered_resident_ab.py prepare \
    --out "$MODELS" --train-rows "$TRAIN_ROWS" > "$OUT/prepare.log" 2>&1

outer=1
while [ "$outer" -le "$OUTERS" ]; do
    if [ $((outer % 2)) -eq 1 ]; then order="sequential ordered"; else order="ordered sequential"; fi
    for arm in $order; do
        use "$arm"
        for dataset in taxi istella; do
            PYTHONPATH=python "$PY" -u bench/speed/forest_ordered_resident_ab.py run \
                --dataset "$dataset" --arm "$arm" --models "$MODELS" --outer "$outer" \
                --rounds "$ROUNDS" --rows "$ROWS" --json "$OUT/$dataset.$arm.o$outer.json" \
                > "$OUT/run.$outer.$arm.$dataset.log" 2>&1
        done
    done
    outer=$((outer + 1))
done
PYTHONPATH=python "$PY" bench/speed/forest_ordered_resident_ab.py summarize \
    "$OUT"/'*.o*.json' --outers "$OUTERS" --out "$OUT/verdict.json" > "$OUT/summary.log"
use sequential
cat "$OUT/verdict.json"
