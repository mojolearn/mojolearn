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
# A pod whose host driver is older than 580 (CUDA 13.0) refuses to load the
# module unless the runtime is pointed at a system ptxas; the kNN lane's pod
# harness does the same. Unset when the driver is new enough.
if [ -z "${MODULAR_NVPTX_COMPILER_PATH:-}" ] && [ -x /usr/local/cuda/bin/ptxas ]; then
    _drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
    [ -n "$_drv" ] && [ "$_drv" -lt 580 ] 2>/dev/null && export MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas
fi
export MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-12}
export MOJOLEARN_BUILD_JOBS=${MOJOLEARN_BUILD_JOBS:-12}
mkdir -p "$OUT/logs"

say() { printf '[%s %s] %s\n' "$(date -u +%T)" "$(basename "$PWD")" "$*"; }
step() { _n=$1; shift; say "$_n: $*"; "$@" > "$OUT/logs/$_n.log" 2>&1; _r=$?; say "$_n rc=$_r"; return $_r; }

case "${1:-}" in
setup)
    [ -x "$HOME/.pixi/bin/pixi" ] || curl -fsSL --max-time 300 https://pixi.sh/install.sh | sh
    step pixi_install-"$(basename "$PWD")" pixi install || exit 1
    # the base extension carries encode_labels_i32, which every classifier fit
    # reaches before it ever touches a forest kernel
    for b in "" rf trees gbdt; do
        _s="bindings/build${b:+_$b}.sh"
        step build_"${b:-base}"-"$(basename "$PWD")" pixi run sh "$_s" || exit 1
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
    # $2 label, $3 the lane list (default: the two that hang a one-GPU box
    # TOGETHER; each alone is the baseline both trees can produce)
    L=${2:?label}
    LN=${3:-rf-clf-balanced-parallel,et-reg-bootstrap-parallel}
    say "identity_break $LN ($L)"
    pixi run python3 tools/identity_break.py \
        --lanes "$LN" --fixtures base --repeats 2 --vendor cuda-4090 \
        --json "$OUT/$L.json" > "$OUT/$L.log" 2>&1
    rc=$?
    say "lanes $L rc=$rc"
    echo "lanes	$L	$LN	$rc" >> "$OUT/verdicts.tsv"
    tail -30 "$OUT/$L.log"
    ;;
gate)
    L=${2:?label}
    # The two -parallel lanes are LEFT OUT here on purpose: the BEFORE tree
    # cannot finish them in one process, so a column that contained them
    # could not be compared. They get their own before/after arms.
    LANES=${LANES:-rf-clf,rf-reg,et-clf,et-reg,rf-clf-entropy-log2-noboot,rf-reg-poisson,rf-reg-gamma-ig,et-clf-entropy-bestfirst,rf-score-weighted,gbdt-symmetric,gbdt-depthwise,gbdt-rmse}
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
