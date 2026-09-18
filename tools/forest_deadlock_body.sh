#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
#
# DEVIATION 3010's on-pod body: the release-then-prepare hang, watched failing
# on the tree WITHOUT the drain and passing on the tree WITH it.
#
# Runs from inside a mojolearn checkout on one CUDA pod. Every command writes
# under $OUT (default /root/fd_out) and prints its own rc, so a hang names its
# phase. Nothing here stages a dataset: the repro fits its own rows.
#
#   sh tools/forest_deadlock_body.sh setup            pixi env + rf/trees/gbdt bindings
#   sh tools/forest_deadlock_body.sh stacklib         build the SIGUSR2 backtrace library
#   sh tools/forest_deadlock_body.sh repro LABEL MODE the repro, mode release|keep
#   sh tools/forest_deadlock_body.sh lanes LABEL      the two -parallel identity_break lanes
#   sh tools/forest_deadlock_body.sh gate  LABEL      the forest lanes, bitwise, to JSON
set -u
OUT=${OUT:-/root/fd_out}
DEADLINE=${DEADLINE:-180}
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONPATH=python
export PYTHONUNBUFFERED=1
export MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-12}
export MOJOLEARN_BUILD_JOBS=${MOJOLEARN_BUILD_JOBS:-12}
mkdir -p "$OUT/logs"

say() { printf '[%s %s] %s\n' "$(date -u +%T)" "$(basename "$PWD")" "$*"; }
step() { _n=$1; shift; say "$_n: $*"; "$@" > "$OUT/logs/$_n.log" 2>&1; _r=$?; say "$_n rc=$_r"; return $_r; }

case "${1:-}" in
setup)
    [ -x "$HOME/.pixi/bin/pixi" ] || curl -fsSL --max-time 300 https://pixi.sh/install.sh | sh
    step pixi_install-"$(basename "$PWD")" pixi install || exit 1
    for b in rf trees gbdt; do
        step build_"$b"-"$(basename "$PWD")" pixi run sh bindings/build_"$b".sh || exit 1
    done
    say "setup done"
    ;;
stacklib)
    cc -shared -fPIC -O1 -g -o /root/native_stack_dump.so tools/native_stack_dump.c \
        && say "native_stack_dump.so built" || say "native_stack_dump.so FAILED (stacks will be Python only)"
    ;;
repro)
    L=${2:?label}; M=${3:?mode}
    export MOJOLEARN_NATIVE_STACK_FILE="$OUT/$L-$M.nativestack.txt"
    [ -f /root/native_stack_dump.so ] && export LD_PRELOAD="/root/native_stack_dump.so${LD_PRELOAD:+:$LD_PRELOAD}"
    say "repro $L mode=$M deadline=${DEADLINE}s"
    pixi run python3 tools/forest_release_prepare_repro.py --mode "$M" --deadline "$DEADLINE" \
        > "$OUT/$L-$M.log" 2>&1
    rc=$?
    say "repro $L mode=$M rc=$rc  (124 = THE HANG, 0 with DONE = pass)"
    echo "repro	$L	$M	$rc" >> "$OUT/verdicts.tsv"
    tail -25 "$OUT/$L-$M.log"
    ;;
lanes)
    L=${2:?label}
    say "identity_break rf-clf-balanced-parallel,et-reg-bootstrap-parallel ($L)"
    pixi run python3 tools/identity_break.py \
        --lanes rf-clf-balanced-parallel,et-reg-bootstrap-parallel \
        --fixtures base --repeats 2 --vendor cuda-4090 \
        --json "$OUT/$L-parallel-lanes.json" > "$OUT/$L-parallel-lanes.log" 2>&1
    rc=$?
    say "lanes $L rc=$rc"
    echo "lanes	$L	-	$rc" >> "$OUT/verdicts.tsv"
    tail -30 "$OUT/$L-parallel-lanes.log"
    ;;
gate)
    L=${2:?label}
    LANES=${LANES:-rf-clf,rf-reg,et-clf,et-reg,rf-clf-entropy-log2-noboot,rf-reg-poisson,rf-reg-gamma-ig,et-clf-entropy-bestfirst,rf-score-weighted,rf-clf-balanced-parallel,et-reg-bootstrap-parallel}
    say "identity_break gate ($L): $LANES"
    pixi run python3 tools/identity_break.py --lanes "$LANES" \
        --fixtures base,ties,odd,dupes,wide --repeats 2 --vendor cuda-4090 \
        --json "$OUT/$L-gate.json" > "$OUT/$L-gate.log" 2>&1
    rc=$?
    say "gate $L rc=$rc"
    echo "gate	$L	-	$rc" >> "$OUT/verdicts.tsv"
    tail -30 "$OUT/$L-gate.log"
    ;;
*)
    echo "usage: $0 setup|stacklib|repro LABEL MODE|lanes LABEL|gate LABEL" >&2
    exit 2
    ;;
esac
