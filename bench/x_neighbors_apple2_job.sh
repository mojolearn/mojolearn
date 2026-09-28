#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Lane neighbors-apple2: one batched Apple speed job, before and after arms on
# one Mac in one job.
#
#   sh bench/x_neighbors_apple2_job.sh <modes> <cases> <reps> <arm> [<arm> ...]
#
# <modes> is "identical", "fast" or "identical,fast". An arm is
# "NAME|<build defines>|<env assignments>" (either part may be empty); each
# arm rebuilds bindings/build.sh, build_metrics.sh and build_x_neighbors.sh
# with MOJOLEARN_BUILD_EXTRA_DEFINES=<build defines>, then runs
# bench/x_neighbors_apple2_speed.py --only <cases> under the environment
# assignments. Lines are prefixed `ARM <mode> <NAME>`. The arms run in order,
# then in reverse order (a drift check); the default is rebuilt at the end.
set -u
modes=$1; cases=$2; reps=$3; shift 3
BUILDS="${XN2_BUILDS:-build.sh build_metrics.sh build_x_neighbors.sh}"
run_arm() {
    mode=$1; spec=$2; tag=$3
    name=$(printf %s "$spec" | cut -d'|' -f1)
    defs=$(printf %s "$spec" | cut -d'|' -f2)
    envs=$(printf %s "$spec" | cut -d'|' -f3)
    for b in $BUILDS; do
        if ! MOJOLEARN_NUMERIC_MODE=$mode MOJOLEARN_BUILD_EXTRA_DEFINES="$defs" pixi run -e default sh "bindings/$b" >/tmp/xn2_job_build.log 2>&1; then
            echo "ARM $mode $name$tag BUILD_FAIL $b [$defs]"; tail -12 /tmp/xn2_job_build.log; return
        fi
    done
    env $envs MOJOLEARN_NUMERIC_MODE=$mode MOJOLEARN_STAGE_TIMES=${XN2_STAGE_TIMES:-0} \
        pixi run -e default python -u bench/x_neighbors_apple2_speed.py --only "$cases" --reps "$reps" 2>&1 \
        | sed "s/^/ARM $mode $name$tag /"
}
for mode in $(echo "$modes" | tr , ' '); do
    for a in "$@"; do run_arm "$mode" "$a" ""; done
    n=$#
    while [ "$n" -gt 0 ]; do
        eval "a=\${$n}"
        run_arm "$mode" "$a" "-r"
        n=$((n - 1))
    done
done
for b in $BUILDS; do
    MOJOLEARN_BUILD_EXTRA_DEFINES="" pixi run -e default sh "bindings/$b" >/dev/null 2>&1 || echo "DEFAULT_REBUILD_FAIL $b"
done
