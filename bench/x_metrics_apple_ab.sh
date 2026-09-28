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
#
# lane metrics-apple3 additions, each off unless its variable is set:
#   XMAB_BUILDS="base estimators ..."  build these bindings in the head tree
#       first, in both modes (base = bindings/build.sh); the base tree gets
#       copies, so both arms run the same estimators
#   XMAB_MSEL=1      model selection: tools/apple_speed_metrics/msel_eq.py in
#       the base tree, the head tree under MOJOLEARN_MSEL3_BEFORE=1 and the
#       head tree as shipped (XMAB-MEQ SAME or DIFF), then
#       bench/x_msel_speed.py before / after / after / before per mode
#       (XMAB_MSEL_PROF=N adds its cProfile top N in FAST)
#   XMAB_XTRA=1      bench/x_metrics_speed.py --extra 2 (cases outside the
#       board's 29) against the base and the head package, both modes
#   XMAB_EXTRAS=1    the epilogue python-vs-native table
#       (tools/py_misc/metrics_time.py), the whole-arena arm of the board
#       (MOJOLEARN_ARENA_RANGES=0 against 1) and tools/py_misc_msel/check.py
#   XMAB_KEEP_GOING=1  a head x_metrics build failure prints its log and the
#       head tree runs on the base tree's x_metrics binary (the Python falls
#       back where an export is missing), so the Python arms still run
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
for m in identical fast; do
    for b in ${XMAB_BUILDS:-}; do
        f=bindings/build_$b.sh; [ "$b" = base ] && f=bindings/build.sh
        t0=$(date +%s)
        if MOJOLEARN_NUMERIC_MODE=$m pixi run -e default sh $f >"$tmp/build_${b}_$m.log" 2>&1; then
            echo "XMAB built head $b $m $(( $(date +%s) - t0 ))s"
        else
            echo "XMAB BUILD_FAIL head $b $m"; tail -25 "$tmp/build_${b}_$m.log"
        fi
    done
done
(cd python/mojolearn && find . \( -name '*.so' -o -name '*.dylib' \) ! -name '_mojolearn_x_metrics.so') | while read -r f; do
    mkdir -p "$bdir/python/mojolearn/$(dirname "$f")"
    cp "python/mojolearn/$f" "$bdir/python/mojolearn/$f"
done
headfail=0
for t in base head; do
    d=$wt; [ "$t" = base ] && d=$bdir
    for m in identical fast; do
        if ! (cd "$d" && MOJOLEARN_NUMERIC_MODE=$m pixi run -e default sh bindings/build_x_metrics.sh >"$tmp/build.log" 2>&1); then
            echo "XMAB BUILD_FAIL $t $m"; tail -60 "$tmp/build.log"
            [ "$t" = head ] && [ "${XMAB_KEEP_GOING:-0}" = 1 ] || exit 4
            headfail=1
            sub=""; [ "$m" = identical ] && sub=identical/
            cp "$bdir/python/mojolearn/${sub}_mojolearn_x_metrics.so" "$wt/python/mojolearn/${sub}_mojolearn_x_metrics.so"
            echo "XMAB head $m runs on the BASE x_metrics binary"
            continue
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
    if [ "${XMAB_MSEL:-0}" = 1 ]; then
        for m in identical fast; do
            (cd "$bdir" && MOJOLEARN_NUMERIC_MODE=$m pixi run -e default python -u "$wt/tools/apple_speed_metrics/msel_eq.py" \
                --tree "$bdir" 2>/dev/null) | grep '^MEQ ' >"$tmp/meq_base_$m.txt"
            (cd "$wt" && MOJOLEARN_MSEL3_BEFORE=1 MOJOLEARN_NUMERIC_MODE=$m pixi run -e default python -u \
                "$wt/tools/apple_speed_metrics/msel_eq.py" --tree "$wt" 2>/dev/null) | grep '^MEQ ' >"$tmp/meq_before_$m.txt"
            (cd "$wt" && MOJOLEARN_NUMERIC_MODE=$m pixi run -e default python -u \
                "$wt/tools/apple_speed_metrics/msel_eq.py" --tree "$wt" 2>/dev/null) | grep '^MEQ ' >"$tmp/meq_head_$m.txt"
            nh=$(wc -l <"$tmp/meq_head_$m.txt")
            nr=$(grep -c ' RAISED ' "$tmp/meq_head_$m.txt")
            if cmp -s "$tmp/meq_base_$m.txt" "$tmp/meq_head_$m.txt" && cmp -s "$tmp/meq_before_$m.txt" "$tmp/meq_head_$m.txt" \
                    && [ "$nh" -gt 0 ]; then
                echo "XMAB-MEQ $m SAME $nh cases ($nr raised in every arm)"
            else
                echo "XMAB-MEQ $m DIFF base $(wc -l <"$tmp/meq_base_$m.txt") before $(wc -l <"$tmp/meq_before_$m.txt") head $nh cases"
                diff "$tmp/meq_base_$m.txt" "$tmp/meq_head_$m.txt" | head -40 | sed 's/^/MEQ-DIFF base-head /'
                diff "$tmp/meq_before_$m.txt" "$tmp/meq_head_$m.txt" | head -40 | sed 's/^/MEQ-DIFF before-head /'
            fi
            grep ' RAISED ' "$tmp/meq_head_$m.txt" | head -20 | sed "s/^/MEQ-RAISED $m /"
        done
    fi
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
if [ "${XMAB_XTRA:-0}" = 1 ]; then
    # the extra cases (outside the board's total), this tree's bench file
    # against each tree's package
    for m in fast identical; do
        for t in base head head base; do
            d=$wt; [ "$t" = base ] && d=$bdir
            (cd "$d" && MOJOLEARN_NUMERIC_MODE=$m pixi run -e default python -u "$wt/bench/x_metrics_speed.py" \
                --tree "$d" --reps "$reps" --extra 2 2>&1) | grep -v 'XMSPEED-INPUT' | sed "s/^/XARM $t-$m /"
        done
    done
    (cd "$wt" && MOJOLEARN_NUMERIC_MODE=fast pixi run -e default python -u "$wt/bench/x_metrics_speed.py" \
        --tree "$wt" --reps 1 --extra 2 --cprofile 14 2>&1) | grep 'XMPROFILE' | sed "s/^/XPROF head-fast /"
fi
marm() {  # before|after mode pass profile
    b=""; [ "$1" = before ] && b=1
    (cd "$wt" && MOJOLEARN_MSEL3_BEFORE=$b MOJOLEARN_NUMERIC_MODE=$2 pixi run -e default python -u bench/x_msel_speed.py \
        --reps 1 --fits "${5:-0}" --cprofile "$4" 2>&1) | sed "s/^/MSEL $1-$2-$3 /"
}
if [ "${XMAB_MSEL:-0}" = 1 ]; then
    marm before fast 1 0 1; marm after fast 1 0
    marm after identical 1 0; marm before identical 1 0
    marm after fast 2 "${XMAB_MSEL_PROF:-0}"; marm before fast 2 0
fi
if [ "${XMAB_EXTRAS:-0}" = 1 ]; then
    for m in fast identical; do
        (cd "$wt" && MOJOLEARN_NUMERIC_MODE=$m PYTHONPATH=python pixi run -e default python -u tools/py_misc/metrics_time.py gpu 2>&1) \
            | sed "s/^/EPI head-$m /"
        for r in 0 1; do
            # shellcheck disable=SC2086
            (cd "$wt" && MOJOLEARN_ARENA_RANGES=$r MOJOLEARN_NUMERIC_MODE=$m pixi run -e default python -u \
                bench/x_metrics_speed.py --reps "$reps" $oflag 2>&1) | sed "s/^/ARM ranges$r-$m-1 /"
        done
    done
    (cd "$wt" && MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python pixi run -e default python -u tools/py_misc_msel/check.py equal 2>&1) \
        | tail -5 | sed "s/^/MSELCHK equal /"
    (cd "$wt" && MOJOLEARN_NUMERIC_MODE=fast PYTHONPATH=python pixi run -e default python -u tools/py_misc_msel/check.py time 2>&1) \
        | tail -24 | sed "s/^/MSELCHK time /"
fi
echo "XMAB headfail $headfail"
if [ "$prof" = 1 ]; then
    for t in base head; do
        d=$wt; [ "$t" = base ] && d=$bdir
        # shellcheck disable=SC2086
        (cd "$d" && MOJOLEARN_XMETRICS_PROFILE=1 MOJOLEARN_NUMERIC_MODE=identical pixi run -e default python -u \
            bench/x_metrics_speed.py --reps 1 --cprofile 25 $oflag 2>&1) | sed "s/^/PROF $t /"
    done
fi
