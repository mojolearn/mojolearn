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
# RunPod AMD MI300X (tools/pick_box.sh printed runpod-amd; same image, the
# body runs natively in the pod after the gemm payload's device check and card;
# no env passing, so STAGE is its default `all`). Rows from a RunPod pod go in
# their own OPPONENT_REFERENCE subsection with that pod's CPU model and cores:
#   MOJOLEARN_RUNPOD_KEY_FILE=~/.mojolearn_runpod_key MOJOLEARN_GPU_ARCHS=gfx942 \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/classical_hotaisle_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=$HOME/mojolearn-evidence/classical-runpod-amd/leg1 \
#   sh tools/gemm_remote_leg.sh amd --payload gemm --rent --minutes 60
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
# killed body. Both runners write started= in /root/gemm_leg_out/leg.txt first.
#   Hot Aisle (tools/hotaisle_leg.sh): the container's PID 1 is
#     `timeout -k 30 <work seconds> sh /root/gemm_leg.sh`; end 150 s inside it.
#   RunPod (tools/gemm_remote_leg.sh --payload gemm): no timeout on the body;
#     the Mac polls min(WORK_TIMEOUT 3000 + 240 s from started=, lease deadline
#     - 300 s), and the lease deadline on the pod is the mtime of
#     /tmp/mojolearn-lease.pid plus the `sleep N` of /tmp/mojolearn-lease.sh
#     (tools/runpod_guard.sh). End 180 s inside the smaller.
#   DigitalOcean (tools/do_extra_leg.sh): the same `timeout -k 30 <seconds> sh
#     /root/gemm_leg.sh` wrapper, natively (not PID 1), so every pid is read.
_work=""
for _c in /proc/1/cmdline /proc/[0-9]*/cmdline; do
    _work=$(tr '\0' ' ' < "$_c" 2>/dev/null | sed -n 's|^timeout -k [0-9]* \([0-9][0-9]*\) sh /root/gemm_leg.sh.*|\1|p')
    [ -n "$_work" ] && break
done
_started=$(sed -n 's/^started=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1)
_s0=$(date -d "$_started" +%s 2>/dev/null)
_provider=unknown
_cap=""
if [ -n "$_work" ] && [ -n "$_s0" ]; then
    _provider=$(sed -n 's/^provider=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1)
    [ -n "$_provider" ] || _provider=digitalocean
    _cap=$(( _s0 + _work - 150 - MOJOLEARN_CTD_BODY_START ))
elif [ -f /tmp/mojolearn-lease.sh ] && [ -f /tmp/mojolearn-lease.pid ] && [ -n "$_s0" ]; then
    _provider=runpod
    _lsecs=$(sed -n 's/^sleep \([0-9][0-9]*\)$/\1/p' /tmp/mojolearn-lease.sh | head -1)
    _larmed=$(stat -c %Y /tmp/mojolearn-lease.pid 2>/dev/null)
    _cap=$(( _s0 + ${MOJOLEARN_CTD_RUNNER_WORK:-3000} + 240 - 180 - MOJOLEARN_CTD_BODY_START ))
    if [ -n "$_lsecs" ] && [ -n "$_larmed" ]; then
        _lcap=$(( _larmed + _lsecs - 300 - 180 - MOJOLEARN_CTD_BODY_START ))
        [ "$_lcap" -lt "$_cap" ] && _cap=$_lcap
    fi
    _work="runpod lease=${_lsecs:-unread}s armed=${_larmed:-unread}"
fi
if [ -n "$_cap" ] && [ "$_cap" -lt "$MOJOLEARN_CTD_BODY_SECONDS" ]; then
    MOJOLEARN_CTD_BODY_SECONDS=$_cap
fi
MOJOLEARN_CTD_PROVIDER=$_provider
# A CFS quota below the visible CPU count (a RunPod pod saw 192 CPUs and was
# allowed 20.4): scikit-learn's pools size to 192, so the capped arm
# sklearn-cpu-quota runs beside the uncapped one and both are reported.
_qcpus=""
if [ -r /sys/fs/cgroup/cpu.max ]; then
    _qcpus=$(awk '$1 != "max" && $2 > 0 {print int($1 / $2)}' /sys/fs/cgroup/cpu.max)
elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
    _qcpus=$(awk -v p="$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null)" '$1 > 0 && p > 0 {print int($1 / p)}' /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
fi
if [ -n "$_qcpus" ] && [ "$_qcpus" -lt "$(nproc)" ] && [ -z "${MOJOLEARN_CTD_EXTRA_ARMS:-}" ]; then
    MOJOLEARN_CTD_EXTRA_ARMS=sklearn-cpu-quota
    export MOJOLEARN_CTD_EXTRA_ARMS
fi
export MOJOLEARN_CTD_BODY_START MOJOLEARN_CTD_BODY_SECONDS MOJOLEARN_CTD_OUT MOJOLEARN_COMPILE_JOBS MOJOLEARN_CTD_PROVIDER

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
