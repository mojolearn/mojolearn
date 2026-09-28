#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# The neighbors lane's Apple A/B of build-define arms (lane neighbors-apple).
#
#   sh bench/x_neighbors_ab.sh <build script> <cases> <reps> <arm> [<arm> ...]
#
# An arm is a quoted define list ("-D X -D Y") or "base" (no define). Each arm
# rebuilds <build script> under MOJOLEARN_NUMERIC_MODE (default identical) with
# MOJOLEARN_BUILD_EXTRA_DEFINES and runs bench/x_neighbors_apple_speed.py --only
# <cases>; the lines are prefixed `ARM <n>`. The arms run in order, then
# again in reverse order (a drift check), and the script ends by rebuilding
# the default so no arm binary is left in the tree for a later job.
set -u
build=$1; cases=$2; reps=$3; shift 3
run_arm() {
    tag=$1; defs=$2
    if [ "$defs" = base ]; then defs=""; fi
    if ! MOJOLEARN_BUILD_EXTRA_DEFINES="$defs" pixi run -e default sh "$build" >/tmp/xn_ab_build.log 2>&1; then
        echo "ARM $tag BUILD_FAIL [$defs]"; tail -5 /tmp/xn_ab_build.log; return
    fi
    pixi run -e default python -u bench/x_neighbors_apple_speed.py --only "$cases" --reps "$reps" --no-quality 2>&1 \
        | sed "s/^/ARM $tag [$defs] /"
}
i=0
for a in "$@"; do run_arm "$i" "$a"; i=$((i + 1)); done
n=$#
while [ "$n" -gt 0 ]; do
    eval "a=\${$n}"
    run_arm "r$((n - 1))" "$a"
    n=$((n - 1))
done
MOJOLEARN_BUILD_EXTRA_DEFINES="" pixi run -e default sh "$build" >/dev/null 2>&1 || echo "DEFAULT_REBUILD_FAIL $build"
