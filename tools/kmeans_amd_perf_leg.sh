#!/bin/sh
# Default-off AMD trial for the two exact KMeans changes already selectable by
# experimental defines.  Intended as MOJOLEARN_GEMM_LEG_EXTRA on one warm
# MI300X.  It consumes only the R2-staged Taxi and Istella-S datasets, builds
# isolated arms, races repeated public fit/transform calls, and makes identity
# plus sabotage reachability hard gates.  It never changes a shipping default.
set -u
R=/root/mojolearn
O=/root/gemm_leg_out/kmeans-amd-perf
DATA=/root/ctd-data
BINS=/root/kmeans-amd-bins
P=$R/.pixi/envs/default/bin/python3
LANES=kmeans,kmeans-random,kmeans-array,kmeans-weighted,kmeans-sqrt,kmeans-classic-pp
FIXTURES=base,ties,hashed,wide,denormal,denormal_ftz,dupes,odd,negative
D_BLOCK='-D MOJOLEARN_EXPERIMENTAL_KMEANS_BLOCK_ACC=1'
D_SCALE='-D MOJOLEARN_EXPERIMENTAL_KMEANS_DEVICE_SCALE=1'
S_BLOCK='-D MOJOLEARN_KMEANS_BLOCK_ACC_SABOTAGE=1'
S_SCALE='-D MOJOLEARN_KMEANS_DEVICE_SCALE_SABOTAGE=1'

cd "$R" || exit 9
mkdir -p "$O/logs" "$BINS"
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1
export MOJOLEARN_TARGET_COLUMN=amd
export MOJOLEARN_GPU_ARCHS="${MOJOLEARN_GPU_ARCHS:-gfx942}"
export MOJOLEARN_COMPILE_JOBS="${MOJOLEARN_COMPILE_JOBS:-13}"
export MOJOLEARN_BUILD_JOBS="${MOJOLEARN_BUILD_JOBS:-13}"
MOJOLEARN_COMMIT=$(sed -n 's/^commit=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1)
[ -n "$MOJOLEARN_COMMIT" ] || MOJOLEARN_COMMIT=$(cat "$R/SHIPPED_COMMIT.txt" 2>/dev/null)
[ -n "$MOJOLEARN_COMMIT" ] || { echo "no shipped commit witness" >&2; exit 10; }
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
    rm -f "$R/python/mojolearn/identical/_mojolearn.so"
    require_step "build_$_arm" 2400 env MOJOLEARN_BUILD_EXTRA_DEFINES="$_defs" sh "$R/bindings/build.sh"
    mkdir -p "$BINS/$_arm"
    cp "$R/python/mojolearn/identical/_mojolearn.so" "$BINS/$_arm/"
    printf '%s\n' "$_defs" > "$O/$_arm.defines.txt"
    sha256sum "$BINS/$_arm/_mojolearn.so" > "$O/$_arm.so.sha256"
    rm -rf "/root/kmeans-amd-$_arm"; mkdir -p "/root/kmeans-amd-$_arm"
    (cd "$R" && tar cf - --exclude=.pixi --exclude='*.so' .) | \
        (cd "/root/kmeans-amd-$_arm" && tar xf -)
    ln -s "$R/.pixi" "/root/kmeans-amd-$_arm/.pixi"
    mkdir -p "/root/kmeans-amd-$_arm/python/mojolearn/identical"
    ln -s "$R/python/mojolearn/.libs" "/root/kmeans-amd-$_arm/python/mojolearn/.libs"
    cp "$BINS/$_arm/_mojolearn.so" "/root/kmeans-amd-$_arm/python/mojolearn/identical/"
}

identity_arm() {
    _arm=$1
    require_step "identity_$_arm" 3600 env \
      PYTHONPATH="/root/kmeans-amd-$_arm/python:/root/kmeans-amd-$_arm/tools" \
      "$P" "/root/kmeans-amd-$_arm/tools/identity_break.py" \
      --require-backend hip --vendor amd-mi300x --lanes "$LANES" \
      --fixtures "$FIXTURES" --repeats 2 --json "$O/identity.$_arm.json"
}

note start
{ date -u; uname -a; nproc; rocminfo 2>&1 | grep -E 'Marketing Name|Name: +gfx|Compute Unit' | head -20; } \
  > "$O/box.txt" 2>&1

# The provider runner stages these decoded caches from Cloudflare R2. Refuse
# the experiment if either is absent: falling back to public downloads would
# conceal an R2 staging failure and waste a warm lease.
for _src in /root/datasets/gbm-bench/taxi/taxi_speed.npz \
            /root/datasets/gbm-bench/istella/istella_speed.npz; do
    test -s "$_src" || { note "R2_STAGING_MISSED $_src"; exit 31; }
    sha256sum "$_src" >> "$O/r2-source.sha256"
done
note R2_STAGED
require_step prep_r2_blocks 1800 pixi run python3 tools/classical_two_datasets.py prep \
  --data "$DATA" --lanes kmeans --datasets taxi,istella
for _ds in taxi istella; do
    test -s "$DATA/big-$_ds.npz" || { note "missing R2-derived block big-$_ds.npz"; exit 32; }
    sha256sum "$DATA/big-$_ds.npz" >> "$O/data.sha256"
done

# OFF is explicit even though AMD currently defaults off. BOTH is experimental
# and therefore cannot silently become the shipping selection in this body.
build_arm off '-D MOJOLEARN_KMEANS_BLOCK_ACC_OFF=1 -D MOJOLEARN_KMEANS_DEVICE_SCALE_OFF=1'
build_arm both "$D_BLOCK $D_SCALE"
build_arm sabotage_block "$D_BLOCK $D_SCALE $S_BLOCK"
build_arm sabotage_scale "$D_BLOCK $D_SCALE $S_SCALE"

# All nine standard fixtures gate repeated-run identity before the expensive
# data races. Candidate and off must compare cleanly.
for _arm in off both sabotage_block sabotage_scale; do identity_arm "$_arm"; done
require_step diff_off_both 300 "$P" "$R/tools/identity_break.py" --diff \
  "$O/identity.off.json" "$O/identity.both.json" \
  --require-columns 2 --lanes "$LANES"

# A sabotage comparison is expected to return nonzero. Require an actual
# DIVERGENT cell and reject a refusal, so a dead experimental arm cannot pass.
for _sab in sabotage_block sabotage_scale; do
    "$P" "$R/tools/identity_break.py" --diff \
      "$O/identity.both.json" "$O/identity.$_sab.json" \
      --require-columns 2 --lanes "$LANES" \
      > "$O/logs/diff_$_sab.log" 2>&1
    _diff_rc=$?
    printf 'diff_%s\t%s\t0\n' "$_sab" "$_diff_rc" >> "$O/status.tsv"
    grep -q 'DIVERGENT' "$O/logs/diff_$_sab.log" || exit 33
    ! grep -q 'REFUSED' "$O/logs/diff_$_sab.log" || exit 34
done

# Prove both experimental stages execute on each full timed workload. A
# small-fixture divergence cannot establish that the certified scale did not
# refuse and fall back for Taxi or Istella-S.
for _ds in taxi istella; do
    require_step "reach_$_ds" 3600 "$P" "$R/bench/speed/kmeans_amd_ab.py" reach \
      --block "$DATA/big-$_ds.npz" --timeout 1200 --output "$O/reach-$_ds.json" \
      --arm both "$P" /root/kmeans-amd-both \
      --arm sabotage_block "$P" /root/kmeans-amd-sabotage_block \
      --arm sabotage_scale "$P" /root/kmeans-amd-sabotage_scale
done

# Five alternating fit samples after one warmup; seven alternating transform
# samples on 100k held-out rows. The Python harness hashes every output byte,
# records fit quality, checks transform minima against public predict, and
# rejects max/min spread above 1.10, fit speedup below 1.02, or a material
# transform regression (speedup below 0.95).
for _ds in taxi istella; do
    require_step "race_$_ds" 5400 "$P" "$R/bench/speed/kmeans_amd_ab.py" race \
      --block "$DATA/big-$_ds.npz" --rounds 5 --transform-rounds 7 \
      --transform-rows 100000 --timeout 1200 --output "$O/$_ds.json" \
      --arm off "$P" /root/kmeans-amd-off \
      --arm both "$P" /root/kmeans-amd-both
done

"$P" - "$O" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
cells = [json.loads((root / (name + ".json")).read_text()) for name in ("taxi", "istella")]
reach = [json.loads((root / ("reach-" + name + ".json")).read_text())
         for name in ("taxi", "istella")]
ok = (all(c.get("promotion_eligible") is True for c in cells) and
      all(c.get("verdict") == "PROMOTION_ELIGIBLE_BITWISE_AND_QUALITY_EQUAL" for c in cells) and
      all(c.get("verdict") == "BOTH_FULL_DATA_STAGES_REACHED" for c in reach))
summary = {"pass": ok, "cells": [{"dataset": pathlib.Path(c["block"]).stem,
            "fit_speedup": c["fit_speedup"],
            "transform_speedup": c["transform_speedup"],
            "fit_spread": {a: c["arms"][a]["fit_spread"] for a in ("off", "both")},
            "transform_spread": {a: c["arms"][a]["transform_spread"] for a in ("off", "both")},
            "verdict": c["verdict"]} for c in cells],
           "reach": [{"dataset": pathlib.Path(c["block"]).stem,
                       "block_accumulator_reached": c["block_accumulator_reached"],
                       "device_scale_reached": c["device_scale_reached"],
                       "verdict": c["verdict"]} for c in reach]}
(root / "verdict.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
raise SystemExit(0 if ok else 1)
PY
test $? -eq 0 || exit 35
note finished
