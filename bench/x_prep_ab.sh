#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# The prep lane's Apple A/B in ONE job (lane prep-apple2): a base commit and
# the tree, same Mac, same run.
#
#   sh bench/x_prep_ab.sh <base commit> <mode> <reps> <cases|all> <arm> [<arm> ...]
#
# An arm is "name|defines|env[|bench args]": name "base..." runs <base commit> (checked
# out beside the tree, its own x_prep build), any other name runs this tree;
# defines are extra -D flags for bindings/build_x_prep.sh
# (MOJOLEARN_MOJO_BUILD_FLAGS); env is NAME=VALUE[,NAME=VALUE] for the timing
# (bench/x_prep_speed.py under MOJOLEARN_NUMERIC_MODE=<mode>); bench args
# are extra flags for it (e.g. --profile, or --only to narrow). Lines are
# prefixed `ARM <name>`; each XPSPEED line carries its digest, so IDENTICAL
# arms compare by eye. The tree's default build is restored at the end.
set -u
base=$1; mode=$2; reps=$3; cases=$4; shift 4
export MOJOLEARN_NUMERIC_MODE="$mode"
root=$(pwd)
bt="${TMPDIR:-/tmp}/xprep_ab_base_$$"
only=""
[ "$cases" = all ] || only="--only $cases"
built=""
basebuilt=""
build_in() {
    # $1 dir, $2 defines
    pixi run --manifest-path "$root/pixi.toml" -e default sh -c \
        "cd '$1' && MOJOLEARN_MOJO_BUILD_FLAGS='$2' sh bindings/build_x_prep.sh" >"$bt.build.log" 2>&1
}
run_arm() {
    spec=$1
    name=${spec%%|*}; rest=${spec#*|}; defs=${rest%%|*}; envs=${rest#*|}
    [ "$envs" = "$rest" ] && envs=""
    args=""
    case "$envs" in *"|"*) args=${envs#*|}; envs=${envs%%|*} ;; esac
    aonly=$only
    case "$args" in *--only*) aonly="" ;; esac
    case "$name" in
        base*)
            dir=$bt
            if [ -z "$basebuilt" ]; then
                git worktree add --detach "$bt" "$base" >/dev/null 2>&1 || { echo "ARM $name BASE_CHECKOUT_FAIL $base"; return; }
                build_in "$bt" "" || { echo "ARM $name BUILD_FAIL"; tail -5 "$bt.build.log"; return; }
                basebuilt=1
            fi ;;
        *)
            dir=$root
            if [ "$built" != "x$defs" ]; then
                build_in "$root" "$defs" || { echo "ARM $name BUILD_FAIL [$defs]"; tail -5 "$bt.build.log"; return; }
                built="x$defs"
            fi ;;
    esac
    echo "ARM $name XPINFO defines=[$defs] env=[$envs] args=[$args] dir=$dir"
    env $(echo "$envs" | tr , ' ') pixi run --manifest-path "$root/pixi.toml" -e default \
        python -u "$dir/bench/x_prep_speed.py" --reps "$reps" $aonly $args 2>&1 | sed "s/^/ARM $name /"
}
for a in "$@"; do run_arm "$a"; done
if [ "$built" != "x" ]; then build_in "$root" "" || echo "DEFAULT_REBUILD_FAIL"; fi
[ -z "$basebuilt" ] || git worktree remove --force "$bt" >/dev/null 2>&1 || true
rm -f "$bt.build.log"
