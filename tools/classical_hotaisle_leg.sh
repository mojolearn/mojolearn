#!/bin/sh
# tools/classical_hotaisle_leg.sh -- DEVIATION 2572, the classical opponent
# rows (DEVIATION 2570) on a Hot Aisle AMD MI300X VM (ENGINEERING_RULES.md
# sections 9 and 10; MI300X rows are a new tuple). A tools/hotaisle_leg.sh
# body: runs ON THE BOX, inside rocm/dev-ubuntu-22.04:6.4.1-complete, from
# /root/mojolearn after pixi install. Everything it writes is under
# /root/gemm_leg_out/classical-hotaisle, which the runner fetches to
# <MOJOLEARN_GEMM_LEG_OUT>/remote/classical-hotaisle.
#
#   MOJOLEARN_HOTAISLE_SPEC=13core MOJOLEARN_HOTAISLE_LANE=classical-hotaisle \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/classical_hotaisle_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=$HOME/mojolearn-evidence/classical-hotaisle/leg1 \
#   MOJOLEARN_HOTAISLE_EXTRA_ENV='MOJOLEARN_CTD_HOTAISLE_STAGE=all' \
#   bash tools/hotaisle_leg.sh amd --rent --minutes 60 --skip-gates
#
# A second lease runs only the cells the first skipped, in the order given:
#   MOJOLEARN_HOTAISLE_EXTRA_ENV='MOJOLEARN_CTD_HOTAISLE_STAGE=rest MOJOLEARN_CTD_REST_CELLS=kde:istella,svc:istella'
#
# ONE DEADLINE (MOJOLEARN_CTD_BODY_SECONDS from the body's start, capped 150 s
# inside the runner's own work bound, read from the container's PID 1
# `timeout -k 30 <seconds> sh /root/gemm_leg.sh` and leg.txt's started=). A race
# that cannot fit is SKIPPED_DEADLINE in status.tsv.
#
# STAGE all, one dataset at a time (the Istella fetch and decode run in the
# background from t=0, so taxi races while it decodes):
#   setup          env record, fetches, the opponents' venv (torch ROCm,
#                  scikit-learn), decodes, every IDENTICAL binding six lanes need
#   smoke-taxi     prep + races of all six lanes at 20,000 rows, 1 round
#   prep-taxi      the full blocks of all six lanes
#   leg1-taxi      kmeans, pca, ols, knn
#   leg2-taxi      kde, svc (here when Istella has not decoded yet, else later)
#   smoke-istella, prep-istella, leg1-istella, [leg2-taxi], leg2-istella
# STAGE rest: setup, prep and races of MOJOLEARN_CTD_REST_CELLS (lane:dataset,
# commas; default the four kde and svc cells). STAGE leg2 is rest's default.
#
# Arms on AMD: ours IDENTICAL; scikit-learn on every CPU core (CPU); torch ROCm
# (GPU) for kmeans, pca, ols (lstsq and eigh) and knn. cuML has no ROCm path.
#
# Env values reach the box through MOJOLEARN_HOTAISLE_EXTRA_ENV, which allows
# no spaces: lists are commas. POSIX sh only.
set -u
L=/root/mojolearn/tools/classical_two_datasets_leg.sh
STAGE=${MOJOLEARN_CTD_HOTAISLE_STAGE:-all}
MOJOLEARN_CTD_BODY_START=$(date +%s)
MOJOLEARN_CTD_BODY_SECONDS=${MOJOLEARN_CTD_BODY_SECONDS:-3000}
MOJOLEARN_CTD_OUT=${MOJOLEARN_CTD_OUT:-/root/gemm_leg_out/classical-hotaisle}
MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-$(nproc)}
ALL=kmeans,pca,ols,knn,kde,svc
LEG1=kmeans,pca,ols,knn
LEG2=kde,svc
mkdir -p "$MOJOLEARN_CTD_OUT"
B="$MOJOLEARN_CTD_OUT/hotaisle_body.txt"
[ "$(id -u)" = 0 ] || { echo "the body needs root (/root paths); id -u is $(id -u)" >> "$B"; exit 5; }

# The runner's work bound: the body must end inside it, or the fetch sees a
# killed body. Unreadable (a native runtime) leaves the default.
_work=$(tr '\0' ' ' < /proc/1/cmdline 2>/dev/null | sed -n 's|^timeout -k [0-9]* \([0-9][0-9]*\) sh /root/gemm_leg.sh.*|\1|p')
_started=$(sed -n 's/^started=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1)
_s0=$(date -d "$_started" +%s 2>/dev/null)
if [ -n "$_work" ] && [ -n "$_s0" ]; then
    _cap=$(( _s0 + _work - 150 - MOJOLEARN_CTD_BODY_START ))
    [ "$_cap" -lt "$MOJOLEARN_CTD_BODY_SECONDS" ] && MOJOLEARN_CTD_BODY_SECONDS=$_cap
fi
export MOJOLEARN_CTD_BODY_START MOJOLEARN_CTD_BODY_SECONDS MOJOLEARN_CTD_OUT MOJOLEARN_COMPILE_JOBS

# The IDENTICAL bindings build for the box's gfx target: the runner exports
# MOJOLEARN_GPU_ARCHS (read from rocminfo's Name: field); this is the fallback.
if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
    MOJOLEARN_GPU_ARCHS=$(rocminfo 2>/dev/null | awk '$1 == "Name:" && $2 ~ /^gfx[0-9a-f]+$/ {print $2; exit}')
    [ -n "$MOJOLEARN_GPU_ARCHS" ] && export MOJOLEARN_GPU_ARCHS
fi
echo "stage=$STAGE body_seconds=$MOJOLEARN_CTD_BODY_SECONDS runner_work_seconds=${_work:-unread} runner_started=${_started:-unread} gpu_archs=${MOJOLEARN_GPU_ARCHS:-NONE} nproc=$(nproc) start=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$B"

step() {  # <label> <env words>...: one call of the leg file, console kept
    _label=$1
    shift
    echo "step=$_label begin=$(date -u +%H:%M:%S) left=$(( MOJOLEARN_CTD_BODY_START + MOJOLEARN_CTD_BODY_SECONDS - $(date +%s) ))s" >> "$B"
    env "$@" sh "$L" > "$MOJOLEARN_CTD_OUT/console-$_label.log" 2>&1
    echo "step=$_label rc=$? end=$(date -u +%H:%M:%S)" >> "$B"
}
SMOKE="MOJOLEARN_CTD_SMOKE_ROWS=20000 MOJOLEARN_CTD_MIN_RACE=60 MOJOLEARN_CTD_RACE_SECONDS=240"

case "$STAGE" in
all)
    step setup MOJOLEARN_CTD_LANES=$ALL MOJOLEARN_CTD_PHASES=setup MOJOLEARN_CTD_TAXI_WAIT="${MOJOLEARN_CTD_TAXI_WAIT:-0}"
    # shellcheck disable=SC2086
    step smoke-taxi MOJOLEARN_CTD_LANES=$ALL MOJOLEARN_CTD_PHASES=prep,races MOJOLEARN_CTD_DATASETS=taxi $SMOKE
    step prep-taxi MOJOLEARN_CTD_LANES=$ALL MOJOLEARN_CTD_PHASES=prep MOJOLEARN_CTD_DATASETS=taxi
    step leg1-taxi MOJOLEARN_CTD_LANES=$LEG1 MOJOLEARN_CTD_PHASES=races MOJOLEARN_CTD_DATASETS=taxi
    L2T=0
    if [ ! -f "$MOJOLEARN_CTD_OUT/istella.done" ]; then
        step leg2-taxi MOJOLEARN_CTD_LANES=$LEG2 MOJOLEARN_CTD_PHASES=races MOJOLEARN_CTD_DATASETS=taxi
        L2T=1
    fi
    # shellcheck disable=SC2086
    step smoke-istella MOJOLEARN_CTD_LANES=$ALL MOJOLEARN_CTD_PHASES=prep,races MOJOLEARN_CTD_DATASETS=istella $SMOKE
    step prep-istella MOJOLEARN_CTD_LANES=$ALL MOJOLEARN_CTD_PHASES=prep MOJOLEARN_CTD_DATASETS=istella
    step leg1-istella MOJOLEARN_CTD_LANES=$LEG1 MOJOLEARN_CTD_PHASES=races MOJOLEARN_CTD_DATASETS=istella
    [ "$L2T" = 1 ] || step leg2-taxi MOJOLEARN_CTD_LANES=$LEG2 MOJOLEARN_CTD_PHASES=races MOJOLEARN_CTD_DATASETS=taxi
    step leg2-istella MOJOLEARN_CTD_LANES=$LEG2 MOJOLEARN_CTD_PHASES=races MOJOLEARN_CTD_DATASETS=istella
    ;;
rest|leg2)
    CELLS=${MOJOLEARN_CTD_REST_CELLS:-kde:taxi,svc:taxi,kde:istella,svc:istella}
    RL=$(echo "$CELLS" | tr ',' '\n' | cut -d: -f1 | awk 'NF && !s[$0]++' | paste -sd, -)
    echo "cells=$CELLS lanes=$RL" >> "$B"
    step setup MOJOLEARN_CTD_LANES="$RL" MOJOLEARN_CTD_PHASES=setup MOJOLEARN_CTD_TAXI_WAIT="${MOJOLEARN_CTD_TAXI_WAIT:-0}"
    for ds in taxi istella; do
        DL=$(echo "$CELLS" | tr ',' '\n' | awk -F: -v d="$ds" '$2 == d && !s[$1]++ {print $1}' | paste -sd, -)
        [ -n "$DL" ] && step "prep-$ds" MOJOLEARN_CTD_LANES="$DL" MOJOLEARN_CTD_PHASES=prep MOJOLEARN_CTD_DATASETS="$ds"
    done
    for c in $(echo "$CELLS" | tr ',' ' '); do
        step "race-${c%%:*}-${c#*:}" MOJOLEARN_CTD_LANES="${c%%:*}" MOJOLEARN_CTD_PHASES=races MOJOLEARN_CTD_DATASETS="${c#*:}"
    done
    ;;
*)
    echo "MOJOLEARN_CTD_HOTAISLE_STAGE=$STAGE: all, rest or leg2" >> "$B"
    exit 2
    ;;
esac
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$B"
exit 0
