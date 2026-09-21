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
export MOJOLEARN_COMMIT="${MOJOLEARN_COMMIT:-$(git rev-parse HEAD)}"

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

build_arm off ''
build_arm fused "$D_FUSED"
for _ds in taxi istella; do
    require_step "race_$_ds" 5400 "$P" "$R/bench/speed/standard_fit_transform_ab.py" race \
      --block "$DATA/big-$_ds.npz" --rounds 5 --timeout 1800 \
      --spread-gate 1.10 --min-speedup 1.02 --output "$O/$_ds.json" \
      --arm off "$P" /root/standard-fit-transform-off \
      --arm fused "$P" /root/standard-fit-transform-fused
done

"$P" - "$O" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
cells = [json.loads((root / (name + ".json")).read_text())
         for name in ("taxi", "istella")]
ok = all(cell.get("promotion_eligible") is True for cell in cells)
summary = {"pass": ok, "cells": [
    {"dataset": pathlib.Path(cell["block"]).stem,
     "speedup": cell["speedup"],
     "spread": {arm: cell["arms"][arm]["spread"] for arm in ("off", "fused")},
     "stats_sha256": cell["stats_sha256"],
     "output_sha256": cell["output_sha256"],
     "quality": cell["quality"], "verdict": cell["verdict"]}
    for cell in cells]}
(root / "verdict.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
raise SystemExit(0 if ok else 1)
PY
test $? -eq 0 || exit 35
git rev-parse HEAD > "$O/commit.txt"
note finished
