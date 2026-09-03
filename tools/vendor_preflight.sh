#!/bin/sh
# Run tools/vendor_preflight.py under EVERY interpreter a vendor leg uses, and
# report what each one has, in ONE pass.
#
# The two interpreters are separate ON PURPOSE (pixi.toml, "THE VENDOR ARMS
# ARE NOT TASKS HERE"): cuml/cuvs/cudf are not on conda-forge for osx-arm64,
# so adding them would re-solve pixi.lock for every environment and move every
# timing ever recorded under it. They live on the pod's system python instead.
#
#   gbmbench   OUR arms plus catboost/lightgbm -- numpy, pandas, the extension
#   system     THE RAPIDS arms -- cuml, cuvs, torch
#
# Exit is non-zero if a REQUIRED module is missing anywhere. A missing
# OPPONENT is reported and does not fail the probe: which opponents a box has
# is the thing being measured.
set -u

OUT="${MOJOLEARN_PREFLIGHT_OUT:-bench/results/vendor_preflight/$(date +%Y-%m-%d_%H%M%S)}"
mkdir -p "$OUT"
rc=0

echo "== vendor preflight -> $OUT =="
{
  echo "date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host $(hostname)"
  command -v nvidia-smi > /dev/null 2>&1 \
    && echo "device $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  command -v rocm-smi > /dev/null 2>&1 && echo "device $(rocm-smi --showproductname 2>/dev/null | head -3 | tail -1)"
} > "$OUT/env.txt" 2>&1
cat "$OUT/env.txt"

echo
echo "-- gbmbench (our arms, catboost, lightgbm) --"
PREFLIGHT_TAG=gbmbench pixi run -e gbmbench python tools/vendor_preflight.py \
    2>&1 | tee "$OUT/gbmbench.log"
[ "$(tail -1 "$OUT/gbmbench.log" | grep -c FAILED)" -gt 0 ] && rc=1

echo
echo "-- system python3 (cuml, cuvs, torch) --"
PREFLIGHT_TAG=system python3 tools/vendor_preflight.py \
    2>&1 | tee "$OUT/system.log"
# The system interpreter is NOT required to carry our extension -- it exists
# for the RAPIDS arms, which import nothing from mojolearn. A failure there is
# reported and does not set rc; what matters is WHICH opponents it found.

echo
echo "== what this box can run =="
grep -h "^PREFLIGHT" "$OUT"/*.log \
  | awk '$4=="OK"||$4=="MISSING"||$4=="ERROR" {printf "  %-10s %-12s %-8s %s\n",$2,$3,$4,$6}' \
  | sort -u
echo
echo "results in $OUT"
[ "$rc" != "0" ] && echo "!! preflight FAILED: a REQUIRED module is missing"
exit $rc
