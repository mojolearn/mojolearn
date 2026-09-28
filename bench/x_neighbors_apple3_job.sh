#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Lane neighbors-apple3: one batched Apple speed job, before and after arms on
# one Mac in one job, over either board of the family.
#
#   sh bench/x_neighbors_apple3_job.sh <board> <modes> <builds> <cases> <reps> <arm> [<arm> ...]
#
# <board> is 1 (bench/x_neighbors_apple_speed.py: k-NN, SVM, KernelRidge, GP,
# Nystroem, RBFSampler) or 2 (bench/x_neighbors_apple2_speed.py: the
# x_neighbors expansion). <modes> is "identical", "fast" or "identical,fast".
# <builds> is a comma separated list of bindings/build_*.sh names, rebuilt
# for every arm. An arm is "NAME|<build defines>|<env assignments>" (either
# part may be empty). Lines are prefixed `ARM <mode> <NAME>`.
#
# The arms run in the order given. With XN3_REVERSE=1 they then run again in
# reverse order (a drift check), tagged `-r`. The default is rebuilt at the
# end, so no arm binary is left in the tree for a later job.
set -u
board=$1; modes=$2; builds=$3; cases=$4; reps=$5; shift 5
if [ "$board" = 1 ]; then script=bench/x_neighbors_apple_speed.py; else script=bench/x_neighbors_apple2_speed.py; fi
run_arm() {
    mode=$1; spec=$2; tag=$3
    name=$(printf %s "$spec" | cut -d'|' -f1)
    defs=$(printf %s "$spec" | cut -d'|' -f2)
    envs=$(printf %s "$spec" | cut -d'|' -f3)
    for b in $(echo "$builds" | tr , ' '); do
        if ! MOJOLEARN_NUMERIC_MODE=$mode MOJOLEARN_BUILD_EXTRA_DEFINES="$defs" pixi run -e default sh "bindings/$b" >/tmp/xn3_job_build.log 2>&1; then
            echo "ARM $mode $name$tag BUILD_FAIL $b [$defs]"; grep -n -i -B2 -A12 "error" /tmp/xn3_job_build.log | head -60; tail -5 /tmp/xn3_job_build.log; return
        fi
    done
    env $envs MOJOLEARN_NUMERIC_MODE=$mode MOJOLEARN_STAGE_TIMES=${XN3_STAGE_TIMES:-0} \
        pixi run -e default python -u "$script" --only "$cases" --reps "$reps" ${XN3_BOARD_FLAGS:-} 2>&1 \
        | sed "s/^/ARM $mode $name$tag /"
}
for mode in $(echo "$modes" | tr , ' '); do
    for a in "$@"; do run_arm "$mode" "$a" ""; done
    if [ "${XN3_REVERSE:-0}" = 1 ]; then
        n=$#
        while [ "$n" -gt 0 ]; do
            eval "a=\${$n}"
            run_arm "$mode" "$a" "-r"
            n=$((n - 1))
        done
    fi
    for b in $(echo "$builds" | tr , ' '); do
        MOJOLEARN_NUMERIC_MODE=$mode MOJOLEARN_BUILD_EXTRA_DEFINES="" pixi run -e default sh "bindings/$b" >/dev/null 2>&1 || echo "DEFAULT_REBUILD_FAIL $mode $b"
    done
done
