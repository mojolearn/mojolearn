#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# lane/gap-lm (2026-10-02): where the byte LM's GPU step goes on a Linux box,
# run as an lq CMD from the branch tree (pixi env, GPU archs set by the job):
#
#   bash tools/gap_lm_profile.sh [label] [DEFINES...]
#
# 1. builds bindings/build_byte_lm.sh (with MOJOLEARN_BUILD_EXTRA_DEFINES when
#    DEFINES are given, e.g. -D MOJOLEARN_X=1),
# 2. plain wall timing of lm-train-step and lm-forward (no stage timers, so
#    no timer waits), 8 calls each, with the digest and loss trace,
# 3. when nsys is on the box, a kernel summary of 4 lm-train-step calls:
#    per-kernel GPU time, CUDA API time (synchronize, copies) and memops.
# Output lines start with GAPLM so `lq log <box> <id> GAPLM` reads them.
set -u
LABEL=${1:-base}; shift || true
DEFS="$*"
OUT=${GAPLM_OUT:-$PWD/gaplm-out}/$LABEL; mkdir -p "$OUT"
say() { echo "GAPLM $LABEL $*"; }
if [ -n "$DEFS" ]; then export MOJOLEARN_BUILD_EXTRA_DEFINES="$DEFS"; fi
rm -f python/mojolearn/identical/_mojolearn_byte_lm.so
bash bindings/build_byte_lm.sh > "$OUT/build.log" 2>&1; rc=$?
say "build rc=$rc defs=[$DEFS]"
[ $rc = 0 ] || { grep -m 5 -B 2 -A 6 'error' "$OUT/build.log" | sed "s/^/GAPLM $LABEL BUILDERR /"; exit 1; }
MOJOLEARN_TRANSFORMER_TIMING= pixi run python tools/neural_stage_timing.py --lane lm-train-step --lane lm-forward \
    --calls 8 > "$OUT/plain.log" 2>&1
say "plain rc=$?"
grep -E '^(SUMMARY|DIGEST|LOSSES)' "$OUT/plain.log" | cut -c1-300 | sed "s/^/GAPLM $LABEL /"
grep -m 3 -E 'Error|error|Traceback' "$OUT/plain.log" | sed "s/^/GAPLM $LABEL ERR /"
NSYS=""
for c in nsys /usr/local/cuda/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys /usr/local/cuda-*/bin/nsys; do
    if command -v "$c" > /dev/null 2>&1; then NSYS=$(command -v "$c"); break; fi
    [ -x "$c" ] && { NSYS=$c; break; }
done
if [ -n "$NSYS" ] && [ "${GAPLM_NSYS:-1}" = 1 ]; then
    MOJOLEARN_TRANSFORMER_TIMING= "$NSYS" profile -t cuda -s none --force-overwrite true -o "$OUT/step" \
        pixi run python tools/neural_stage_timing.py --lane lm-train-step --calls 5 > "$OUT/nsys.log" 2>&1
    say "nsys rc=$?"
    "$NSYS" stats -q --report cuda_gpu_kern_sum --format csv -o "$OUT/k" "$OUT/step.nsys-rep" > /dev/null 2>&1
    "$NSYS" stats -q --report cuda_api_sum --format csv -o "$OUT/a" "$OUT/step.nsys-rep" > /dev/null 2>&1
    "$NSYS" stats -q --report cuda_gpu_mem_time_sum --format csv -o "$OUT/m" "$OUT/step.nsys-rep" > /dev/null 2>&1
    for f in "$OUT"/k*.csv; do [ -f "$f" ] && python3 - "$f" "$LABEL KERN" 30 <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
for r in rows[: int(sys.argv[3])]:
    name = r.get("Name", "")[:70]
    print("GAPLM %s %5s%% %10.3fms n=%6s avg=%9.1fus %s" % (sys.argv[2], r.get("Time (%)", "?"),
          float(r.get("Total Time (ns)", 0)) / 1e6, r.get("Instances", "?"),
          float(r.get("Avg (ns)", 0)) / 1e3, name))
EOF
    done
    for f in "$OUT"/a*.csv "$OUT"/m*.csv; do [ -f "$f" ] && python3 - "$f" "$LABEL API" 12 <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
for r in rows[: int(sys.argv[3])]:
    name = (r.get("Name") or r.get("Operation") or "")[:60]
    print("GAPLM %s %5s%% %10.3fms n=%6s %s" % (sys.argv[2], r.get("Time (%)", "?"),
          float(r.get("Total Time (ns)", 0)) / 1e6, r.get("Count", r.get("Num Calls", "?")), name))
EOF
    done
else
    say "nsys absent"
fi
