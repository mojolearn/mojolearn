#!/bin/sh
# Guarded default-off StandardScaler.fit_transform trial on the two R2-staged
# classical datasets. Intended for one warm GPU lease after local review.
set -u
R=/root/mojolearn
O=/root/gemm_leg_out/standard-fit-transform
DATA=/root/ctd-data
BINS=/root/standard-fit-transform-bins
P=$R/.pixi/envs/default/bin/python3
D_FUSED='-D MOJOLEARN_EXPERIMENTAL_STANDARD_FIT_TRANSFORM=1'

cd "$R" || exit 9
mkdir -p "$O/logs" "$BINS"
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1
export MOJOLEARN_COMPILE_JOBS="${MOJOLEARN_COMPILE_JOBS:-13}"
MOJOLEARN_COMMIT=${MOJOLEARN_COMMIT:-$(sed -n 's/^commit=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1)}
[ -n "$MOJOLEARN_COMMIT" ] || MOJOLEARN_COMMIT=$(git rev-parse HEAD 2>/dev/null)
[ -n "$MOJOLEARN_COMMIT" ] || { echo "missing commit witness" >&2; exit 10; }
export MOJOLEARN_COMMIT

note() { printf '%s %s\n' "$*" "$(date -u +%H:%M:%S)" | tee -a "$O/progress.txt"; }
step() {
    _name=$1; _cap=$2; shift 2; _start=$(date +%s)
    timeout -k 30 "$_cap" "$@" > "$O/logs/$_name.log" 2>&1
    _rc=$?
    printf '%s\t%s\t%s\n' "$_name" "$_rc" "$(( $(date +%s) - _start ))" >> "$O/status.tsv"
    note "$_name=$_rc"
    return "$_rc"
}
require_step() { step "$@" || exit 30; }

build_arm() {
    _arm=$1; _defs=$2
    require_step "build_$_arm" 1800 env MOJOLEARN_BUILD_EXTRA_DEFINES="$_defs" \
      sh "$R/bindings/build_preprocessing.sh"
    mkdir -p "$BINS/$_arm"
    cp "$R/python/mojolearn/identical/_mojolearn_preprocessing.so" "$BINS/$_arm/"
    printf '%s\n' "$_defs" > "$O/$_arm.defines.txt"
    sha256sum "$BINS/$_arm/_mojolearn_preprocessing.so" > "$O/$_arm.so.sha256"
    rm -rf "/root/standard-fit-transform-$_arm"
    mkdir -p "/root/standard-fit-transform-$_arm/python"
    cp -R "$R/python/mojolearn" "/root/standard-fit-transform-$_arm/python/"
    cp "$BINS/$_arm/_mojolearn_preprocessing.so" \
      "/root/standard-fit-transform-$_arm/python/mojolearn/identical/"
}

note start
{ date -u; uname -a; nproc; } > "$O/box.txt" 2>&1
for _src in /root/datasets/gbm-bench/taxi/taxi_speed.npz \
            /root/datasets/gbm-bench/istella/istella_speed.npz; do
    test -s "$_src" || { note "R2_STAGING_MISSED $_src"; exit 31; }
    sha256sum "$_src" >> "$O/r2-source.sha256"
done
note R2_STAGED
require_step prep_r2_blocks 1800 pixi run python3 tools/classical_two_datasets.py prep \
  --data "$DATA" --lanes pca --datasets taxi,istella
for _ds in taxi istella; do
    test -s "$DATA/big-$_ds.npz" || exit 32
    sha256sum "$DATA/big-$_ds.npz" >> "$O/data.sha256"
done

require_step build_base 2400 sh bindings/build.sh
build_arm off ''
build_arm fused "$D_FUSED"
for _outer in 0 1 2; do
  _first=off; [ $(( _outer % 2 )) -eq 0 ] || _first=fused
  for _ds in taxi istella; do
    require_step "race_${_ds}_${_outer}" 5400 "$P" "$R/bench/speed/standard_fit_transform_ab.py" race \
      --block "$DATA/big-$_ds.npz" --rounds 5 --timeout 1800 --first-arm "$_first" \
      --spread-gate 1.10 --min-speedup 0.0 --output "$O/${_ds}.${_outer}.json" \
      --arm off "$P" /root/standard-fit-transform-off \
      --arm fused "$P" /root/standard-fit-transform-fused
  done
done

"$P" - "$O" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
datasets = ("taxi", "istella")
cells = {name: [json.loads((root / f"{name}.{outer}.json").read_text())
                for outer in range(3)] for name in datasets}
summaries = []
ok = True
for name in datasets:
    rows = cells[name]
    ready = [row["ready"] for row in rows]
    inputs = {(tuple(r[arm]["shape"]), r[arm]["dtype"], r[arm]["input_sha256"])
              for r in ready for arm in ("off", "fused")}
    stats_hashes = {row["stats_sha256"] for row in rows}
    output_hashes = {row["output_sha256"] for row in rows}
    qualities = {json.dumps(row["quality"], sort_keys=True) for row in rows}
    conservative = (min(row["arms"]["off"]["median_ms"] for row in rows) /
                    max(row["arms"]["fused"]["median_ms"] for row in rows))
    stable = all(row["arms"][arm]["spread"] <= 1.10
                 for row in rows for arm in ("off", "fused"))
    exact = len(inputs) == len(stats_hashes) == len(output_hashes) == len(qualities) == 1
    passed = exact and stable and conservative >= 1.02
    ok = ok and passed
    shape, dtype, input_sha = next(iter(inputs))
    summaries.append({"dataset": name, "shape": list(shape), "dtype": dtype,
                      "input_sha256": input_sha,
                      "stats_sha256": next(iter(stats_hashes)),
                      "output_sha256": next(iter(output_hashes)),
                      "quality": rows[0]["quality"],
                      "outer_speedups": [row["speedup"] for row in rows],
                      "conservative_speedup": conservative,
                      "stable": stable, "pass": passed})
summary = {"datasets": list(datasets), "exact_dataset_set": list(cells) == list(datasets),
           "fresh_process_outers": 3, "retained_rounds_per_arm_per_outer": 5,
           "pass": ok and list(cells) == list(datasets), "cells": summaries}
(root / "verdict.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
raise SystemExit(0 if summary["pass"] else 1)
PY
test $? -eq 0 || exit 35
git rev-parse HEAD > "$O/commit.txt"
note finished
