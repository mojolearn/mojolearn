#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# The metrics family's Apple A/B (lane metrics-apple2): the speed board
# (bench/x_metrics_speed.py) at a BASE commit and at this commit (HEAD), on
# the same Mac, in one steward job.
#
#   sh bench/x_metrics_apple_ab.sh <base full sha> [reps] [profile 0|1] [only] [tests 0|1]
#
# The base is checked out in a temporary git worktree that shares this
# worktree's pixi environment (a symlink: the pixi files are the same) and
# its other prebuilt bindings (copied); x_metrics is built in both trees
# and both numeric modes. The arms run base, head for IDENTICAL then FAST,
# then again in reverse order (a drift check). Lines are prefixed
# `ARM <tree>-<mode>-<pass>`. With profile=1 each tree also runs once more
# under MOJOLEARN_XMETRICS_PROFILE=1 with the board's --cprofile 25.
# With tests=1 (the default) the job first runs tools/apple_speed_metrics/:
# eq_cases.py in both trees under both modes (every `EQ` line must be
# equal: XMAB-EQ SAME or DIFF) and, in the head tree, the Mojo word tests
# (*_words.mojo, built IDENTICAL).
set -u
base=$1; reps=${2:-2}; prof=${3:-0}; only=${4:-}; tests=${5:-1}
wt=$(pwd)
echo "XMAB head $(git rev-parse --short HEAD) base $(printf %s "$base" | cut -c1-12) host $(hostname) $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
git cat-file -e "$base^{commit}" 2>/dev/null || git fetch -q origin "$base" || exit 3
tmp=$(mktemp -d "${TMPDIR:-/tmp}/xmab.XXXXXX")
bdir=$tmp/base
git worktree add -q --detach "$bdir" "$base" || exit 3
cleanup() { cd "$wt" && git worktree remove --force "$bdir" >/dev/null 2>&1; rm -rf "$tmp"; }
trap cleanup EXIT INT TERM
ln -s "$wt/.pixi" "$bdir/.pixi"
(cd python/mojolearn && find . -name '*.so' ! -name '_mojolearn_x_metrics.so') | while read -r f; do
    mkdir -p "$bdir/python/mojolearn/$(dirname "$f")"
    cp "python/mojolearn/$f" "$bdir/python/mojolearn/$f"
done
for t in head base; do
    d=$wt; [ "$t" = base ] && d=$bdir
    for m in identical fast; do
        if ! (cd "$d" && MOJOLEARN_NUMERIC_MODE=$m pixi run -e default sh bindings/build_x_metrics.sh >"$tmp/build.log" 2>&1); then
            echo "XMAB BUILD_FAIL $t $m"; tail -20 "$tmp/build.log"; exit 4
        fi
        echo "XMAB built $t $m"
    done
done
if [ "$tests" = 1 ]; then
    for m in identical fast; do
        for t in base head; do
            d=$wt; [ "$t" = base ] && d=$bdir
            if [ -f "$wt/tools/apple_speed_metrics/eq_cases.py" ]; then
                (cd "$d" && MOJOLEARN_NUMERIC_MODE=$m pixi run -e default python -u "$wt/tools/apple_speed_metrics/eq_cases.py" \
                    --tree "$d" 2>/dev/null) | grep '^EQ ' >"$tmp/eq_${t}_$m.txt"
            fi
        done
        nb=$(wc -l <"$tmp/eq_base_$m.txt"); nh=$(wc -l <"$tmp/eq_head_$m.txt")
        if cmp -s "$tmp/eq_base_$m.txt" "$tmp/eq_head_$m.txt" && [ "$nb" -gt 0 ]; then
            echo "XMAB-EQ $m SAME $nh cases"
        else
            echo "XMAB-EQ $m DIFF base $nb head $nh cases"
            diff "$tmp/eq_base_$m.txt" "$tmp/eq_head_$m.txt" | head -40
        fi
    done
    for f in "$wt"/tools/apple_speed_metrics/*_words.mojo; do
        [ -f "$f" ] || continue
        b=$tmp/$(basename "$f" .mojo)
        if pixi run mojo build -j 2 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . "$f" -o "$b" >"$tmp/wb.log" 2>&1; then
            "$b" 2>&1 | tail -60 | sed "s/^/WORDS /"
        else
            echo "WORDS BUILD_FAIL $f"; tail -30 "$tmp/wb.log"
        fi
    done
fi
oflag=""; [ -n "$only" ] && oflag="--only $only"
arm() {  # tree mode pass
    d=$wt; [ "$1" = base ] && d=$bdir
    # shellcheck disable=SC2086
    (cd "$d" && MOJOLEARN_NUMERIC_MODE=$2 pixi run -e default python -u bench/x_metrics_speed.py --reps "$reps" $oflag 2>&1) \
        | sed "s/^/ARM $1-$2-$3 /"
}
for m in identical fast; do arm base $m 1; arm head $m 1; done
for m in fast identical; do arm head $m 2; arm base $m 2; done
if [ "$prof" = 1 ]; then
    for t in base head; do
        d=$wt; [ "$t" = base ] && d=$bdir
        # shellcheck disable=SC2086
        (cd "$d" && MOJOLEARN_XMETRICS_PROFILE=1 MOJOLEARN_NUMERIC_MODE=identical pixi run -e default python -u \
            bench/x_metrics_speed.py --reps 1 --cprofile 25 $oflag 2>&1) | sed "s/^/PROF $t /"
    done
fi
