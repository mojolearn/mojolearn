#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# lane linear-apple3: the arms of x_linear FAST changes in ONE steward speed
# job (same commit, same Mac). An arm is a set of build defines for
# bindings/build_x_linear.sh, so every arm is one commit's source. The
# FIRST arm is built first: a source that does not build fails the job at
# once.
#
#   L3_ARMS="blocks=-D MOJOLEARN_X_LINEAR_BLOCKS=1;before=" \
#   L3_CASES=poisson,gamma,tweedie,huber,quantile,logistic-cv \
#   L3_ROWS="100000 1000000" L3_QUAL=1 L3_IDENT=1 sh tools/linear_apple3_ab.sh
#
# L3_ARMS: ';' separated name=defines. L3_ROWS_<name> overrides L3_ROWS for
# that arm. L3_QUAL=1: bench/linear_apple3_quality.py per arm (the first arm
# also fits the CPU host reference column) and under IDENTICAL.
# L3_IDENT=1: the IDENTICAL binding at HEAD and with L3_BASE's copy of
# L3_IDENT_FILES; the x_linear board's digests are printed for both and
# compared line by line (IDENTSAME / IDENTMOVED).
set -u
export PYTHONUNBUFFERED=1
P="pixi run -e default"
ARMS="${L3_ARMS:-blocks=-D MOJOLEARN_X_LINEAR_BLOCKS=1;before=}"
CASES="${L3_CASES:-poisson,gamma,tweedie,huber,quantile,logistic-cv}"
ROWS="${L3_ROWS:-100000 1000000}"
QUAL="${L3_QUAL:-1}"
QUAL_CASES="${L3_QUAL_CASES:-$CASES}"
QUAL_SEEDS="${L3_QUAL_SEEDS:-0,1,2,3,4}"
IDENT="${L3_IDENT:-1}"
BASE="${L3_BASE:-origin/lane/apple3-merged}"
IDENT_FILES="${L3_IDENT_FILES:-x_linear/device.mojo x_linear/ops.mojo x_linear/sgd.mojo x_linear/tops.mojo}"
IDENT_CASES="${L3_IDENT_CASES:-sgd-clf,sgd-reg,perceptron,pa-clf,pa-reg,sgd-ocsvm,poisson,tweedie,huber,quantile,bayes-ridge,ard,lars,lasso-lars,ridge-clf,ridge-cv,lasso-cv,enet-cv,logistic-cv,isotonic}"
echo "L3AB commit=$(git rev-parse --short HEAD) arms=[$ARMS] cases=$CASES rows=[$ROWS] $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"

build() {  # mode, defines, binding
    MOJOLEARN_MOJO_BUILD_FLAGS="$2" MOJOLEARN_NUMERIC_MODE=$1 $P sh "bindings/build_$3.sh" > "/tmp/l3_$3.log" 2>&1 \
        || { echo "BUILD FAIL mode=$1 defines=[$2] $3"; grep -v "ld: warning" "/tmp/l3_$3.log" | tail -n 80; return 1; }
}

[ "$QUAL" = 1 ] && { build identical "" x_linear_host || echo "host binding not rebuilt; the reference column uses the one in place"; }

first=1
oldifs=$IFS; IFS=';'
for arm in $ARMS; do
    IFS=$oldifs
    name=${arm%%=*}; defs=${arm#*=}
    echo "##### ARM $name defines=[$defs]"
    build fast "$defs" x_linear || exit 1
    # the first fit of a process builds the pipelines; 20k rows also checks the GPU column against the host's
    MOJOLEARN_NUMERIC_MODE=fast $P python bench/x_linear_speed.py --rows 20000 --column both --only "$CASES" 2>&1 | sed "s/^/[$name 20000] /"
    eval "rows=\${L3_ROWS_$name:-\$ROWS}"
    for r in $rows; do
        MOJOLEARN_NUMERIC_MODE=fast $P python bench/x_linear_speed.py --rows "$r" --column gpu --only "$CASES" 2>&1 | sed "s/^/[$name $r] /"
    done
    if [ "$QUAL" = 1 ]; then
        hostarg="--no-host"; [ "$first" = 1 ] && hostarg=""
        MOJOLEARN_NUMERIC_MODE=fast $P python bench/linear_apple3_quality.py --arm "$name" --cases "$QUAL_CASES" --seeds "$QUAL_SEEDS" $hostarg 2>&1 | grep -v "^$"
    fi
    first=0
    IFS=';'
done
IFS=$oldifs

if [ "$IDENT" = 1 ]; then
    build identical "" x_linear || exit 1
    MOJOLEARN_NUMERIC_MODE=identical $P python bench/x_linear_speed.py --rows 100000 --column gpu --only "$IDENT_CASES" > /tmp/l3_ident_head.txt 2>&1
    sed "s/^/[ident-head] /" /tmp/l3_ident_head.txt
    [ "$QUAL" = 1 ] && MOJOLEARN_NUMERIC_MODE=identical $P python bench/linear_apple3_quality.py --arm identical --cases "$QUAL_CASES" --seeds "$QUAL_SEEDS" --no-host 2>&1 | grep -v "^$"
    git checkout "$BASE" -- $IDENT_FILES
    build identical "" x_linear || { git checkout HEAD -- $IDENT_FILES; exit 1; }
    MOJOLEARN_NUMERIC_MODE=identical $P python bench/x_linear_speed.py --rows 100000 --column gpu --only "$IDENT_CASES" > /tmp/l3_ident_base.txt 2>&1
    sed "s/^/[ident-base] /" /tmp/l3_ident_base.txt
    git checkout HEAD -- $IDENT_FILES
    git status --short | grep -v "^??" | head
    for c in $(echo "$IDENT_CASES" | tr ',' ' '); do
        a=$(grep "^XLSPEED $c gpu" /tmp/l3_ident_head.txt | awk '{print $NF}')
        b=$(grep "^XLSPEED $c gpu" /tmp/l3_ident_base.txt | awk '{print $NF}')
        if [ -n "$a" ] && [ "$a" = "$b" ]; then echo "IDENTSAME $c $a"; else echo "IDENTMOVED $c head=[$a] base=[$b]"; fi
    done
fi
# leave the default FAST build of HEAD in place
build fast "" x_linear
echo JOBDONE
