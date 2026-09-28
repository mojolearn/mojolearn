#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# The prep lane's Apple A/B in ONE job (lane prep-apple2): a base commit and
# the tree, same Mac, same run.
#
#   sh bench/x_prep_ab.sh <base commit> <mode> <reps> <cases|all> <arm> [<arm> ...]
#
# An arm is "name|defines|env[|bench args]": name "base..." runs <base commit> (checked
# out beside the tree, its own x_prep build), name "at-<commit>-<label>" runs that
# commit the same way (lane prep-apple3: a job whose tree does not build still
# times its earlier commits), any other name runs this tree;
# defines are extra -D flags for bindings/build_x_prep.sh
# (MOJOLEARN_MOJO_BUILD_FLAGS); env is NAME=VALUE[,NAME=VALUE] for the timing (a value
# never holds a comma: MOJOLEARN_XPREP_R3_ON / _OFF take names joined by "+")
# (bench/x_prep_speed.py under MOJOLEARN_NUMERIC_MODE=<mode>); bench args
# are extra flags for it (e.g. --profile, or --only to narrow). Lines are
# prefixed `ARM <name>`; each XPSPEED line carries its digest, so IDENTICAL
# arms compare by eye. The tree's default build is restored at the end.
# XPREP_AB_SCRIPT=<path in the tree> runs that script with the arm's bench args alone
# (bench/x_prep_quality.py: the paired quality check) in place of the speed board.
set -u
base=$1; mode=$2; reps=$3; cases=$4; shift 4
export MOJOLEARN_NUMERIC_MODE="$mode"
root=$(pwd)
bt="${TMPDIR:-/tmp}/xprep_ab_base_$$"
only=""
[ "$cases" = all ] || only="--only $cases"
built=""
sides=""
bad=""
side_dir() {
    # $1 commit: its checkout beside the tree, built once; prints the directory
    sd="$bt.$1"
    case " $bad " in *" $1 "*) return 2 ;; esac
    case " $sides " in
        *" $1 "*) ;;
        *)
            git worktree add --detach "$sd" "$1" >/dev/null 2>&1 || return 1
            sides="$sides $1"
            # the tree's other built bindings (the label encoder's, ...), so every arm
            # of the job runs beside the same ones; x_prep is built here
            (cd "$root/python/mojolearn" && find . -name '_mojolearn*.so' ! -name '_mojolearn_x_prep*') |
                while read -r so; do
                    mkdir -p "$sd/python/mojolearn/$(dirname "$so")"
                    ln -sf "$root/python/mojolearn/$so" "$sd/python/mojolearn/$so"
                done
            build_in "$sd" "" || { bad="$bad $1"; return 2; } ;;
    esac
    echo "$sd"
}
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
        base*|at-*)
            commit=$base
            case "$name" in at-*) commit=${name#at-}; commit=${commit%%-*} ;; esac
            # the function runs in this shell (not a subshell), so `sides` is kept
            side_dir "$commit" >"$bt.side" 2>/dev/null
            rc=$?
            [ $rc -eq 1 ] && { echo "ARM $name BASE_CHECKOUT_FAIL $commit"; return; }
            [ $rc -eq 2 ] && { echo "ARM $name BUILD_FAIL $commit"; tail -40 "$bt.build.log"; return; }
            dir=$(cat "$bt.side") ;;
        *)
            dir=$root
            if [ "$built" != "x$defs" ]; then
                build_in "$root" "$defs" || { echo "ARM $name BUILD_FAIL [$defs]"; tail -40 "$bt.build.log"; return; }
                built="x$defs"
            fi ;;
    esac
    echo "ARM $name XPINFO defines=[$defs] env=[$envs] args=[$args] dir=$dir"
    if [ -n "${XPREP_AB_SCRIPT:-}" ]; then
        env $(echo "$envs" | tr , ' ') pixi run --manifest-path "$root/pixi.toml" -e default \
            python -u "$dir/$XPREP_AB_SCRIPT" $args 2>&1 | sed "s/^/ARM $name /"
        return
    fi
    env $(echo "$envs" | tr , ' ') pixi run --manifest-path "$root/pixi.toml" -e default \
        python -u "$dir/bench/x_prep_speed.py" --reps "$reps" $aonly $args 2>&1 | sed "s/^/ARM $name /"
}
for a in "$@"; do run_arm "$a"; done
if [ "$built" != "x" ] && [ -n "$built" ]; then build_in "$root" "" || echo "DEFAULT_REBUILD_FAIL"; fi
for c in $sides; do git worktree remove --force "$bt.$c" >/dev/null 2>&1 || true; done
rm -f "$bt.build.log" "$bt.side"
