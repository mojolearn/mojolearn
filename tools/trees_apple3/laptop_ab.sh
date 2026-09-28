#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# trees-apple3 on the LAPTOP M4 (brief update 21:25Z Sep 28): the arms of
# tools/trees_apple_ab.sh under the laptop's rules.
#   - every build goes through `tools/mac_slot.py run`, every timing through
#     `tools/mac_slot.py metal` (MAC_SLOTS=2): nice 19, one thread, one Metal
#     job at a time, FIFO with the other lanes;
#   - an arm is BUILT ONCE and its bindings kept under build/trees_apple3_arms/
#     <mode>/<arm>/; a timing round copies the arm's bindings into place, so
#     the arms alternate per round without rebuilding;
#   - no build starts with less than 15 GB free; `LAB_CLEAN=1` deletes the
#     kept bindings at the end.
#
#   MOJOLEARN_NUMERIC_MODE=fast TAB_BUILDS="rf gbdt" TAB_ROUNDS=2 \
#   TAB_ARMS="before=|X=0;after=-D SOMETHING|X=1" TAP_CELLS="rf:taxi" \
#       sh tools/trees_apple3/laptop_ab.sh
set -u
export MAC_SLOTS=2
MODE="${MOJOLEARN_NUMERIC_MODE:-fast}"
export MOJOLEARN_NUMERIC_MODE="$MODE"
BUILDS="${TAB_BUILDS:-rf}"
ROUNDS="${TAB_ROUNDS:-2}"
ARMS="${TAB_ARMS:?TAB_ARMS=name=defines;name=defines}"
SLOT="python3 tools/mac_slot.py"
KEEP="build/trees_apple3_arms/$MODE"
OUTDIR="python/mojolearn"
[ "$MODE" = "fast" ] || OUTDIR="python/mojolearn/$MODE"
free_gb() { df -g "$HOME" | awk 'NR==2 {print $4}'; }
echo "LABRUN machine=laptop-M4 commit=$(git rev-parse --short HEAD) mode=$MODE builds=$BUILDS rounds=$ROUNDS rows=${TAP_ROWS:-1000000} arms=$ARMS"

oldifs=$IFS; IFS=';'
for arm in $ARMS; do
    IFS=$oldifs
    name=${arm%%=*}; defs=${arm#*=}
    case "$defs" in *"|"*) defs=${defs%%|*} ;; esac
    # arms with equal defines share one build
    key=$(printf '%s' "$defs" | cksum | cut -d' ' -f1)
    if [ ! -f "$KEEP/$key/.built" ]; then
        g=$(free_gb)
        [ "$g" -ge 15 ] || { echo "LAB REFUSED: $g GB free, under 15"; exit 3; }
        mkdir -p "$KEEP/$key"
        for b in $BUILDS; do
            t0=$(date +%s)
            MOJOLEARN_EXTRA_DEFINES="$defs" MOJOLEARN_COMPILE_JOBS=1 MOJOLEARN_BUILD_JOBS=1 \
                $SLOT run pixi run -e default sh "bindings/build_$b.sh" > "$KEEP/$key/build_$b.log" 2>&1 \
                || { echo "ARM $name build_$b FAILED"; grep -v '^\s*$' "$KEEP/$key/build_$b.log" | tail -n 40; exit 1; }
            cp "$OUTDIR/_mojolearn_$b.so" "$KEEP/$key/" || exit 1
            echo "LAB BUILT arm=$name build=$b defines=[$defs] s=$(( $(date +%s) - t0 ))"
        done
        touch "$KEEP/$key/.built"
    fi
    IFS=';'
done
IFS=$oldifs

r=0
while [ "$r" -lt "$ROUNDS" ]; do
    oldifs=$IFS; IFS=';'
    for arm in $ARMS; do
        IFS=$oldifs
        name=${arm%%=*}; defs=${arm#*=}
        envs=""
        case "$defs" in *"|"*) envs=${defs#*|}; defs=${defs%%|*} ;; esac
        key=$(printf '%s' "$defs" | cksum | cut -d' ' -f1)
        echo "##### ARM $name round=$r defines=[$defs] env=[$envs]"
        for b in $BUILDS; do
            # a NEW file, never a rewrite in place (a binding rewritten in
            # place is killed at load, exit 137)
            rm -f "$OUTDIR/_mojolearn_$b.so"
            cp "$KEEP/$key/_mojolearn_$b.so" "$OUTDIR/_mojolearn_$b.so" || exit 1
        done
        env $envs TAP_ROUNDS=1 $SLOT metal sh tools/trees_apple_speed.sh 2>&1 \
            | grep -v '^mac_slot:' | sed "s/^/[$name] /"
        IFS=';'
    done
    IFS=$oldifs
    r=$((r + 1))
done
if [ "${LAB_CLEAN:-0}" = "1" ]; then
    rm -rf "$KEEP"
    echo "LAB CLEANED $KEEP"
fi
