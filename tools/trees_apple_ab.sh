#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Before and after arms of ONE trees change in ONE steward speed job (same
# commit, same Mac): each arm rebuilds the named bindings with its own
# MOJOLEARN_EXTRA_DEFINES, then runs tools/trees_apple_speed.sh on
# TAP_CELLS. The arms interleave per round (A B A B ...) so drift on the
# Mac lands on both.
#
#   TAB_BUILDS="rf" TAB_ARMS="before=-D MOJOLEARN_RF_BINS_COLUMN_MAJOR;after=" \
#   TAB_ROUNDS=2 TAP_CELLS="rf:taxireg xt:dt:taxi" sh tools/trees_apple_ab.sh
#
# TAB_ARMS: ';' separated name=defines[|VAR=value ...]. The LAST arm's
# build is left in place. Words after '|' are exported for that arm's
# timing only (a Python-side toggle); the build sees the defines alone.
set -u
BUILDS="${TAB_BUILDS:-rf}"
ROUNDS="${TAB_ROUNDS:-2}"
ARMS="${TAB_ARMS:?TAB_ARMS=name=defines;name=defines}"
echo "TABRUN commit=$(git rev-parse --short HEAD) mode=${MOJOLEARN_NUMERIC_MODE:-unset} builds=$BUILDS rounds=$ROUNDS arms=$ARMS"
r=0
while [ "$r" -lt "$ROUNDS" ]; do
    oldifs=$IFS; IFS=';'
    for arm in $ARMS; do
        IFS=$oldifs
        name=${arm%%=*}; defs=${arm#*=}
        envs=""
        case "$defs" in *"|"*) envs=${defs#*|}; defs=${defs%%|*} ;; esac
        echo "##### ARM $name round=$r defines=[$defs] env=[$envs]"
        for b in $BUILDS; do
            MOJOLEARN_EXTRA_DEFINES="$defs" pixi run -e default sh "bindings/build_$b.sh" >"/tmp/tab_build_$b.log" 2>&1 \
                || { echo "ARM $name build_$b FAILED"; tail -n 40 "/tmp/tab_build_$b.log"; exit 1; }
        done
        env $envs TAP_ROUNDS=1 sh tools/trees_apple_speed.sh 2>&1 | sed "s/^/[$name] /"
        IFS=';'
    done
    IFS=$oldifs
    r=$((r + 1))
done
