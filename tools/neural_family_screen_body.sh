#!/bin/sh
# Guarded small-rung Transformer/Mamba family profiler; never provisions a box.
set -eu
[ "${MOJOLEARN_NEURAL_SCREEN_GUARD:-}" = R2_TAXI_ISTELLA ] || { echo 'set MOJOLEARN_NEURAL_SCREEN_GUARD=R2_TAXI_ISTELLA' >&2; exit 64; }
ROOT=${MOJOLEARN_NEURAL_SCREEN_ROOT:-/root/mojolearn}; OUT=${MOJOLEARN_NEURAL_SCREEN_OUT:-/root/neural-family-screen}; DATA=${GBM_BENCH_DATA:-$HOME/datasets/gbm-bench}; PROCESSES=${MOJOLEARN_NEURAL_SCREEN_PROCESSES:-1}
case "${MOJOLEARN_TARGET_COLUMN:-}" in nvidia|amd) ;; *) echo 'MOJOLEARN_TARGET_COLUMN must be nvidia or amd' >&2; exit 2;; esac
cd "$ROOT"; export PATH="$HOME/.pixi/bin:$PATH" PYTHONPATH="$ROOT/python:$ROOT" MOJOLEARN_NUMERIC_MODE=identical
if [ -n "${MOJOLEARN_COMMIT:-}" ]; then COMMIT=$MOJOLEARN_COMMIT
elif [ -f SHIPPED_COMMIT.txt ]; then COMMIT=$(cat SHIPPED_COMMIT.txt)
elif [ -f MOJOLEARN_COMMIT ]; then COMMIT=$(cat MOJOLEARN_COMMIT)
else COMMIT=$(git rev-parse HEAD); fi
mkdir -p "$OUT/logs"
build() { name=$1; shift; "$@" >"$OUT/logs/build-$name.log" 2>&1; }
build base sh bindings/build.sh
build transformer sh bindings/build_transformer.sh
build mamba sh bindings/build_mamba.sh
sha256sum python/mojolearn/identical/*.so > "$OUT/bindings.sha256"
for row in "taxi:$DATA/taxi/taxi_speed.npz:gbm-bench/taxi/taxi_speed.npz:10d5d35f376a5b2ad5c66fabe624ee2caafb58bc5e7516824f801de9aab6cc15" "istella:$DATA/istella/istella_speed.npz:gbm-bench/istella/istella_speed.npz:31f042376c0b819fe169cbbd998e840567c23dae25c14cb9054b460292f6ffef"; do
    ds=${row%%:*}; rest=${row#*:}; path=${rest%%:*}; rest=${rest#*:}; key=${rest%%:*}; sha=${rest#*:}
    sh tools/dataset_store.sh verify "$key" "$path" >"$OUT/logs/verify-$ds.log" 2>&1
    for family in transformer mamba1 mamba2 mamba3; do
        p=0
        while [ "$p" -lt "$PROCESSES" ]; do
            dest="$OUT/$ds/$family/process$p"; mkdir -p "$dest"
            pixi run python tools/neural_family_screen.py --family "$family" --dataset "$path" --dataset-key "$key" --dataset-sha256 "$sha" --out "$dest/result.json" --commit "$COMMIT" --process "$p" --target-column "$MOJOLEARN_TARGET_COLUMN" >"$dest/probe.log" 2>&1
            p=$((p+1))
        done
    done
done
pixi run python tools/neural_family_screen_summary.py --root "$OUT" --out "$OUT/summary.json" | tee "$OUT/summary.txt"
