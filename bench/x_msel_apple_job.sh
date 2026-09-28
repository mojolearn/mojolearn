#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Lane metrics-apple3: the model selection PHASE job for the Apple steward
# (measurement tooling). Builds the named bindings from this worktree in the
# named modes, then times bench/x_msel_speed.py (phases, cProfile) per mode,
# the metrics board and the epilogue before/after table.
#
#   sh bench/x_msel_apple_job.sh "<bindings>" "<modes>" [rows] [cprofile] [only] [extras 0|1]
#
# bindings: names after bindings/build_ (base = bindings/build.sh), or "".
set -u
binds=$1; modes=$2; rows=${3:-1000000}; prof=${4:-0}; only=${5:-}; extras=${6:-1}
tmp=$(mktemp -d "${TMPDIR:-/tmp}/xmsel.XXXXXX")
trap 'rm -rf "$tmp"' EXIT INT TERM
echo "XMSELJOB head $(git rev-parse --short HEAD) host $(hostname) $(sysctl -n machdep.cpu.brand_string 2>/dev/null) $(date -u +%FT%TZ)"
for m in $modes; do
    for b in $binds; do
        f=bindings/build_$b.sh; [ "$b" = base ] && f=bindings/build.sh
        t0=$(date +%s)
        if MOJOLEARN_NUMERIC_MODE=$m pixi run -e default sh $f >"$tmp/build_${b}_$m.log" 2>&1; then
            echo "XMSELJOB built $b $m $(( $(date +%s) - t0 ))s"
        else
            echo "XMSELJOB BUILD_FAIL $b $m"; tail -15 "$tmp/build_${b}_$m.log"
        fi
    done
done
(cd python/mojolearn && ls -l *.so identical/*.so 2>/dev/null | awk '{print "XMSELJOB so", $5, $6, $7, $8, $9}')
oflag=""; [ -n "$only" ] && oflag="--only $only"
for m in $modes; do
    # shellcheck disable=SC2086
    MOJOLEARN_NUMERIC_MODE=$m pixi run -e default python -u bench/x_msel_speed.py --rows "$rows" \
        --cprofile "$prof" $oflag 2>&1 | sed "s/^/ARM head-$m /"
done
if [ "$extras" = 1 ]; then
    for m in $modes; do
        MOJOLEARN_NUMERIC_MODE=$m pixi run -e default python -u bench/x_metrics_speed.py --reps 2 2>&1 \
            | sed "s/^/BOARD head-$m /"
        MOJOLEARN_NUMERIC_MODE=$m PYTHONPATH=python pixi run -e default python -u tools/py_misc/metrics_time.py gpu 2>&1 \
            | sed "s/^/EPI head-$m /"
        MOJOLEARN_NUMERIC_MODE=$m PYTHONPATH=python pixi run -e default python -u tools/py_misc_msel/check.py time 2>&1 \
            | sed "s/^/MSEL head-$m /"
    done
fi
echo "XMSELJOB END $(date -u +%FT%TZ)"
