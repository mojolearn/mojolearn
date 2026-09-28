#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# lane linear-apple3: before and after arms of the x_linear FAST change in
# ONE steward speed job (same commit, same Mac). An arm is a set of build
# defines for bindings/build_x_linear.sh; the arm named "after" is built
# FIRST, so a source that does not build fails the job at once.
#
#   L3_AFTER="-D MOJOLEARN_X_LINEAR_BLOCKS=1" L3_BEFORE="" \
#   L3_CASES=poisson,gamma,tweedie,huber,quantile,logistic-cv \
#   L3_ROWS="100000 1000000" L3_BEFORE_ROWS="100000 1000000" \
#   L3_QUAL=1 L3_IDENT=1 sh tools/linear_apple3_ab.sh
#
# L3_IDENT=1: the IDENTICAL binding at HEAD and with L3_BASE's copy of
# L3_IDENT_FILES, the x_linear board's digests printed for both (they must
# be equal line by line).
set -u
export PYTHONUNBUFFERED=1
P="pixi run -e default"
AFTER="${L3_AFTER--D MOJOLEARN_X_LINEAR_BLOCKS=1}"
BEFORE="${L3_BEFORE-}"
CASES="${L3_CASES:-poisson,gamma,tweedie,huber,quantile,logistic-cv}"
ROWS="${L3_ROWS:-100000 1000000}"
BEFORE_ROWS="${L3_BEFORE_ROWS:-$ROWS}"
QUAL="${L3_QUAL:-1}"
QUAL_CASES="${L3_QUAL_CASES:-}"
IDENT="${L3_IDENT:-1}"
BASE="${L3_BASE:-origin/lane/apple3-merged}"
IDENT_FILES="${L3_IDENT_FILES:-x_linear/device.mojo}"
IDENT_CASES="${L3_IDENT_CASES:-sgd-clf,sgd-reg,poisson,tweedie,huber,quantile,bayes-ridge,ard,lars,lasso-lars,ridge-clf,ridge-cv,lasso-cv,enet-cv,logistic-cv,isotonic}"
echo "L3AB commit=$(git rev-parse --short HEAD) after=[$AFTER] before=[$BEFORE] cases=$CASES rows=[$ROWS] $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"

build() {  # mode, defines, binding
    MOJOLEARN_MOJO_BUILD_FLAGS="$2" MOJOLEARN_NUMERIC_MODE=$1 $P sh "bindings/build_$3.sh" > "/tmp/l3_$3.log" 2>&1 \
        || { echo "BUILD FAIL mode=$1 defines=[$2] $3"; grep -v "ld: warning" "/tmp/l3_$3.log" | tail -n 60; return 1; }
}

time_arm() {  # label, rows list
    for r in $2; do
        echo "=== $1 fast rows=$r"
        MOJOLEARN_NUMERIC_MODE=fast $P python bench/x_linear_speed.py --rows "$r" --column gpu --only "$CASES" 2>&1 \
            | sed "s/^/[$1 $r] /"
    done
}

qual_arm() {  # label, mode, extra args
    [ "$QUAL" = 1 ] || return 0
    echo "=== $1 quality mode=$2"
    MOJOLEARN_NUMERIC_MODE=$2 $P python bench/linear_apple3_quality.py --arm "$1" --cases "$QUAL_CASES" $3 2>&1 | grep -v "^$"
}

build identical "" x_linear_host || echo "host binding not rebuilt; the reference column uses the one in place"

build fast "$AFTER" x_linear || exit 1
echo "=== after smoke (2k rows: the one-block path below the row floor; 20k rows: blocks)"
MOJOLEARN_NUMERIC_MODE=fast $P python bench/x_linear_speed.py --rows 2000 --column gpu --only "$CASES" 2>&1 | sed "s/^/[after 2000] /"
MOJOLEARN_NUMERIC_MODE=fast $P python bench/x_linear_speed.py --rows 20000 --column both --only "$CASES" 2>&1 | sed "s/^/[after 20000] /"
time_arm after "$ROWS"
qual_arm after fast ""

build fast "$BEFORE" x_linear || exit 1
MOJOLEARN_NUMERIC_MODE=fast $P python bench/x_linear_speed.py --rows 2000 --column gpu --only "$CASES" > /dev/null 2>&1
time_arm before "$BEFORE_ROWS"
qual_arm before fast "--no-host"

if [ "$IDENT" = 1 ]; then
    build identical "" x_linear || exit 1
    echo "=== identical HEAD digests"
    MOJOLEARN_NUMERIC_MODE=identical $P python bench/x_linear_speed.py --rows 100000 --column gpu --only "$IDENT_CASES" 2>&1 | sed "s/^/[ident-head] /"
    qual_arm identical identical "--no-host"
    git checkout "$BASE" -- $IDENT_FILES
    build identical "" x_linear || { git checkout HEAD -- $IDENT_FILES; exit 1; }
    echo "=== identical BASE digests ($IDENT_FILES of $BASE)"
    MOJOLEARN_NUMERIC_MODE=identical $P python bench/x_linear_speed.py --rows 100000 --column gpu --only "$IDENT_CASES" 2>&1 | sed "s/^/[ident-base] /"
    git checkout HEAD -- $IDENT_FILES
    git status --short | grep -v "^??" | head
fi
# leave the default build of HEAD in place
build fast "" x_linear
echo JOBDONE
