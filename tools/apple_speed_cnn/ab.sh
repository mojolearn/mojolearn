#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# lane/cnn-apple2 (2026-09-28): before and after arms of ONE x_cnn change in
# ONE steward speed job (same commit, same Mac). Each arm builds the x_cnn
# binding once with its own defines (MOJOLEARN_MOJO_BUILD_FLAGS) and keeps
# the .so; the timing then alternates the arms per round (A B A B ...) so
# drift on the Mac lands on both. Every arm prints its XCNN-DIGEST lines:
# IDENTICAL arms must print the same digests.
#
#   CAB_ARMS="before=-D MOJOLEARN_XCNN_NO_MMA_SPLIT;after=" CAB_ROUNDS=2 \
#   CAB_PARTS="speed entries" CAB_PLANS="t_b2_dw,c64_dw" sh tools/apple_speed_cnn/ab.sh
#
# CAB_PY_<name>: extra profile.py words for that arm (e.g. legacy).
# CAB_PLANS: rows of gemm_plans.mojo to sweep once at the end (empty: none).
set -u
ARMS="${CAB_ARMS:?CAB_ARMS=name=defines;name=defines}"
ROUNDS="${CAB_ROUNDS:-2}"
PARTS="${CAB_PARTS:-speed}"
MODE=${MOJOLEARN_NUMERIC_MODE:-identical}
sub=identical
[ "$MODE" = fast ] && sub=.
so=python/mojolearn/$sub/_mojolearn_x_cnn.so
[ "$MODE" = fast ] && so=python/mojolearn/_mojolearn_x_cnn.so
keep=$(mktemp -d "${TMPDIR:-/tmp}/cnn-ab.XXXXXX")
echo "CABRUN commit=$(git rev-parse --short HEAD) host=$(sysctl -n machdep.cpu.brand_string 2>/dev/null) mode=$MODE rounds=$ROUNDS arms=$ARMS"
oldifs=$IFS; IFS=';'
for arm in $ARMS; do
    IFS=$oldifs
    name=${arm%%=*}; defs=${arm#*=}
    echo "##### BUILD $name defines=[$defs]"
    MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_MOJO_BUILD_FLAGS="$defs" sh bindings/build_x_cnn.sh >"$keep/build_$name.log" 2>&1 \
        || { echo "ARM $name build FAILED"; tail -n 40 "$keep/build_$name.log"; exit 1; }
    cp "$so" "$keep/$name.so"
    IFS=';'
done
IFS=$oldifs
r=0
while [ "$r" -lt "$ROUNDS" ]; do
    IFS=';'
    for arm in $ARMS; do
        IFS=$oldifs
        name=${arm%%=*}
        cp "$keep/$name.so" "$so"
        extra=$(eval "printf %s \"\${CAB_PY_$name:-}\"")
        echo "##### ARM $name round=$r"
        # shellcheck disable=SC2086
        pixi run -e default python tools/apple_speed_cnn/profile.py "$MODE" . $PARTS $extra 2>&1 | sed "s/^/[$name] /"
        IFS=';'
    done
    IFS=$oldifs
    r=$((r + 1))
done
if [ -n "${CAB_FASTQ:-}" ]; then
    # the FAST paired quality set, once per arm
    IFS=';'
    for arm in $ARMS; do
        IFS=$oldifs
        name=${arm%%=*}
        cp "$keep/$name.so" "$so"
        extra=$(eval "printf %s \"\${CAB_PY_$name:-}\"")
        echo "##### FASTQ $name"
        # shellcheck disable=SC2086
        CNN_FASTQ_EXTRA="$extra" pixi run -e default python tools/apple_speed_cnn/fastq.py "$MODE" . 2>&1 | sed "s/^/[$name] /"
        IFS=';'
    done
    IFS=$oldifs
fi
if [ -n "${CAB_STAGES:-}" ]; then
    # every launch of the conv blocks alone, per arm (stages.mojo)
    mf="-D MOJOLEARN_NUMERIC_IDENTICAL=1"
    [ "$MODE" = fast ] && mf=""
    IFS=';'
    for arm in $ARMS; do
        IFS=$oldifs
        name=${arm%%=*}; defs=${arm#*=}
        echo "##### STAGES $name"
        # shellcheck disable=SC2086
        pixi run mojo build -j 2 $mf $defs -I . tools/apple_speed_cnn/stages.mojo -o "$keep/stages_$name" \
            && "$keep/stages_$name" 2>&1 | sed "s/^/[$name] /"
        IFS=';'
    done
    IFS=$oldifs
fi
if [ -n "${CAB_PLANS:-}" ]; then
    mf="-D MOJOLEARN_NUMERIC_IDENTICAL=1"
    [ "$MODE" = fast ] && mf=""
    # shellcheck disable=SC2086
    pixi run mojo build -j 2 $mf -I . tools/apple_speed_cnn/gemm_plans.mojo -o "$keep/plans" \
        && CNN_PLANS_ONLY="$CAB_PLANS" "$keep/plans"
fi
rm -rf "$keep"
