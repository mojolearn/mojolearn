#!/bin/bash
# tools/gpu_queue_box.sh -- THE FIFO GPU JOB QUEUE ON A SHARED BOX (AMD and NVIDIA).
#
# One script, two installs:
#   - the central AMD box: /root/amd-queue/bin/amdq, run by the systemd
#     service amd-queue.service (tools/amd_central.sh queue-install);
#   - each shared NVIDIA RunPod pod: /root/gpu-queue/bin/nvq, a plain
#     setsid daemon (a RunPod pod is a container with no systemd), started and
#     restarted by `nvq ensure` (tools/nvidia_central.sh install, and every
#     nvidia_central.sh command).
# Lanes never call this directly; they use tools/amd_central.sh or
# tools/nvidia_central.sh submit/queue/status/log/cancel.
#
#   q submit <lane> <gpus> <cap-min> <script> [note]   enqueue; prints the job id
#                                                      (GQ_FRONT=1: a FRONT job, started before every
#                                                      plain queued job, after the running ones; front
#                                                      jobs keep submission order among themselves.
#                                                      lq add --front, the release smoke, 2026-10-08)
#   q queue                                            every unfinished job + the last finished ones
#   q status <id>                                      one job: its record, state, exit, log tail
#   q cancel <lane> <id>                               cancel a job the lane itself submitted
#   q ensure                                           (plain mode) start the dispatcher / lease backstop if down
#   q hold <command...>                                run a no-GPU command that keeps the pod's lease busy
#   q lease-check                                      one line for the Mac watcher: LEASE OK|BAD ...
#   q daemon                                           (the service) the dispatcher loop
#   q runner <id>                                      (the dispatcher, per job) run one job
#   q backstop                                         (plain mode, lease) delete the pod if the dispatcher stops
#
# CONFIG. Files in the queue dir, written at install: slots (GPU count),
# vendor (amd|nvidia), mode (systemd|plain), service (the systemd unit name),
# default_cap and max_cap (minutes; max_cap 0 = none). With no files, the
# values are the central AMD box's (2 slots, amd, systemd, 720, none), so the
# AMD install behaves as it did before this script was generalized.
#
# JOBS. <queue dir>/<id>/ holds job.env (lane, gpus, script, cap, submit
# time), status (queued|starting|running|done|failed|timeout|cancelled|
# interrupted), log (the script's stdout+stderr), exit (its exit code),
# gpus, started, ended. Ids are a zero-padded counter: submission order.
#
# ORDER. Strict FIFO: the dispatcher starts the OLDEST queued job as soon as
# that job's --gpus slots are free, and starts nothing behind it until it has
# started (no backfill, so a two-GPU job is never starved by one-GPU jobs).
#
# SLOTS. flock slots <lock dir>/gpu<g>.lock, the same locks as the client's
# `run`, so a queued job never shares a GPU with a `run` job, and `run` does
# not take a free slot while jobs are queued (<queue dir>/pending).
#
# A JOB runs as root in /root/mojolearn-<lane> (the lane's own tree and pixi
# env), seeing only its GPUs: AMD ROCR_VISIBLE_DEVICES=<slots> (HIP devices
# 0..N-1); NVIDIA CUDA_VISIBLE_DEVICES=<slots> with CUDA_DEVICE_ORDER=PCI_BUS_ID
# (CUDA devices 0..N-1). The job is over when EVERY process it started has
# exited (background work keeps its slot). At the wall-clock cap, or on a
# cancel, every process of the job is killed (TERM, then KILL).
#   systemd mode: the job is its own transient unit <unit>-<id>.service and its
#     processes are that unit's cgroup (setsid does not escape it).
#   plain mode: the job is its own setsid runner, and its processes are the
#     ones whose environment carries the job's random GQ_JOB_TOKEN (every child
#     inherits it; a process that wipes its environment escapes, so do not).
# The dispatcher restarting never touches a running job. A job that was
# running when the dispatcher came up without it (reboot, container restart)
# is put back at the head of the queue once (RESTARTS=1), and marked
# interrupted if it is cut short again.
#
# LEASE (plain mode, when <queue dir>/lease.conf exists: the NVIDIA pods).
# The queue IS the pod's lease. While anything is queued, starting or running,
# a GPU slot is held (a `run`), or a `hold` command runs, the pod is BUSY and
# lease.last_busy is now. The deadline is last_busy + IDLE_MIN (60). At the
# deadline the dispatcher marks the queue closing (submit refuses from then
# on) and DELETEs the pod through the RunPod API (the key is the 0600
# /tmp/mojolearn-lease.curlrc tools/runpod_guard.sh put on the pod). The
# backstop, a separate process, deletes the pod GRACE_MIN (15) after the
# deadline if the dispatcher did not (the dispatcher died), and the Mac
# watcher (tools/nvidia_central.sh) deletes it when the backstop is not alive
# or the pod stops answering. So a pod ends by itself whoever goes away.
set -uo pipefail
SELF=$(readlink -f "$0")
# the queue dir is the one this script is installed in (<dir>/bin/<name>): /root/amd-queue, /root/gpu-queue
Q=${GQ_DIR:-${AMDQ_DIR:-$(dirname "$(dirname "$SELF")")}}
LOCKD=${GQ_LOCKD:-${AMDQ_LOCKD:-/var/lock/mojolearn-central}}
U=${GQ_UNIT:-${AMDQ_UNIT:-$(basename "$SELF")}}
cfg() { local v; v=$(cat "$Q/$1" 2>/dev/null) && [ -n "$v" ] && printf '%s' "$v" || printf '%s' "$2"; }
SLOTS=$(cfg slots 2)
VENDOR=$(cfg vendor amd)
MODE=$(cfg mode systemd)
SERVICE=$(cfg service amd-queue.service)
DEFAULT_CAP_MIN=$(cfg default_cap 720)
MAX_CAP_MIN=$(cfg max_cap 0)
export PATH=/root/.pixi/bin:/opt/rocm/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

now() { date -u +%FT%TZ; }
die() { echo "gpuq: $*" >&2; exit 1; }
qlock() { exec 8>$Q/.lock; flock 8; }
qunlock() { flock -u 8; exec 8>&-; }
st() { cat "$Q/$1/status" 2>/dev/null || echo missing; }
setst() { echo "$2" > "$Q/$1/status.tmp" && mv -f "$Q/$1/status.tmp" "$Q/$1/status"; }
putf() { echo "$2" > "$1.tmp" && mv -f "$1.tmp" "$1"; }
ids() { ls -1 $Q 2>/dev/null | grep -E '^[0-9]{4,}$' | sort; }
field() { (. "$Q/$1/job.env"; eval "printf '%s' \"\${$2:-}\""); }
slot_free() { flock -n "$LOCKD/gpu$1.lock" true 2>/dev/null; }
lock_free() { flock -n "$1" true 2>/dev/null; }
lane_ok() { [[ "${1:-}" =~ ^[a-z0-9-]{1,32}$ ]] || die "lane must be [a-z0-9-]{1,32}"; }
id_ok() { [[ "${1:-}" =~ ^[0-9]{4,}$ ]] && [ -d "$Q/$1" ] || die "no job ${1:-?}"; }
# the job's runner is alive
unit_active() {
    if [ "$MODE" = systemd ]; then systemctl is-active --quiet "$U-$1.service"; return; fi
    local p; p=$(cat "$Q/$1/runner.pid" 2>/dev/null) || return 1
    [ -n "$p" ] && kill -0 "$p" 2>/dev/null && tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q " runner $1"
}
dispatcher_active() {
    if [ "$MODE" = systemd ]; then systemctl is-active --quiet "$SERVICE"; else ! lock_free $Q/daemon.lock; fi
}
leased() { [ "$MODE" = plain ] && [ -f $Q/lease.conf ]; }
lease_conf() { IDLE_MIN=60; GRACE_MIN=15; POD_ID=''; CURLRC=/tmp/mojolearn-lease.curlrc; FAKE_DELETE=0
    DELETE_URLS=''; IDLE_SEC=''; GRACE_SEC=''; [ -f $Q/lease.conf ] && . $Q/lease.conf
    IDLE_SEC=${IDLE_SEC:-$((IDLE_MIN * 60))}; GRACE_SEC=${GRACE_SEC:-$((GRACE_MIN * 60))}   # *_SEC: tests only
    [ -n "$DELETE_URLS" ] || DELETE_URLS="https://rest.runpod.io/v1/pods/$POD_ID https://api.runpod.io/v2/pods/$POD_ID"; }
# DELETE this pod through the RunPod API (lease.conf's key file); 0 when the API took it
lease_delete() {  # <who>
    lease_conf
    echo "$(now) $1: DELETING pod $POD_ID (idle since $(date -u -d @"$(cat $Q/lease.last_busy 2>/dev/null || echo 0)" +%FT%TZ 2>/dev/null))" >> $Q/lease.log
    if [ "$FAKE_DELETE" = 1 ]; then echo "$(now) FAKE DELETE $POD_ID by $1" >> $Q/lease.deleted; return 0; fi
    [ -s "$CURLRC" ] || { echo "$(now) $1: NO KEY FILE $CURLRC; cannot delete" >> $Q/lease.log; return 1; }
    local u c
    for u in $DELETE_URLS; do
        c=$(curl -s -m 60 -o /dev/null -w '%{http_code}' -X DELETE "$u" -K "$CURLRC" 2>>$Q/lease.log)
        echo "$(now) $1: DELETE $u -> $c" >> $Q/lease.log
        case "$c" in 2*|404) return 0 ;; esac
    done
    return 1
}
# plain mode: start the dispatcher (and the lease backstop) unless one is running.
# Every lock fd is closed for the child so it holds nothing but its own lock.
start_plain() {  # <subcommand> <lock file> <log>
    lock_free "$2" || return 0
    GQ_DIR=$Q GQ_LOCKD=$LOCKD GQ_UNIT=$U setsid nohup "$SELF" "$1" 7>&- 8>&- 9>&- >> "$3" 2>&1 < /dev/null &
    local i=0; while [ $i -lt 20 ] && lock_free "$2"; do sleep 0.25; i=$((i + 1)); done
    lock_free "$2" && die "the $1 did not come up (see $3)"
    return 0
}

cmd="${1:-}"; shift || true
mkdir -p $Q $LOCKD
case "$cmd" in
submit)
    lane="${1:-}"; gpus="${2:-}"; cap="${3:-}"; script="${4:-}"; note="${5:-}"
    lane_ok "$lane"
    [[ "$gpus" =~ ^[1-9][0-9]*$ ]] && [ "$gpus" -le "$SLOTS" ] || die "--gpus must be 1..$SLOTS"
    [ -n "$cap" ] || cap=$DEFAULT_CAP_MIN
    [[ "$cap" =~ ^[1-9][0-9]*$ ]] || die "--cap must be whole minutes"
    [ "$MAX_CAP_MIN" = 0 ] || [ "$cap" -le "$MAX_CAP_MIN" ] || die "--cap $cap is over this queue's $MAX_CAP_MIN-min maximum"
    tree=/root/mojolearn-$lane
    [ -d "$tree" ] || die "no lane tree $tree on the box (run sync first)"
    case "$script" in /*) ;; "") die "submit needs a script on the box" ;; *) script="$tree/$script" ;; esac
    [ -f "$script" ] || die "no script $script on the box"
    qlock
    [ ! -e $Q/closing ] || { qunlock; die "this pod is going down (idle past its lease: $(cat $Q/closing)); nothing was queued"; }
    n=$(( $(cat $Q/.seq 2>/dev/null || echo 0) + 1 )); echo $n > $Q/.seq
    id=$(printf '%04d' $n); d=$Q/$id; mkdir -p "$d"
    {
        printf 'ID=%q\nLANE=%q\nGPUS=%q\nCAP_MIN=%q\nSCRIPT=%q\nTREE=%q\n' "$id" "$lane" "$gpus" "$cap" "$script" "$tree"
        printf 'SCRIPT_SHA256=%q\nSUBMITTED=%q\nNOTE=%q\nRESTARTS=0\n' "$(sha256sum "$script" | cut -c1-64)" "$(now)" "$note"
    } > "$d/job.env"
    [ "${GQ_FRONT:-0}" != 1 ] || : > "$d/front"
    : > "$d/log"; setst $id queued
    ! leased || date +%s > $Q/lease.last_busy
    qunlock
    echo "$id"
    ;;
queue)
    printf '%-5s %-13s %-4s %-11s %-20s %-20s %s\n' ID LANE GPUS STATE SUBMITTED STARTED SCRIPT
    done_ids=""
    for id in $(ids); do
        s=$(st $id)
        case "$s" in queued|starting|running) ;; *) done_ids="$done_ids $id"; continue ;; esac
        printf '%-5s %-13s %-4s %-11s %-20s %-20s %s\n' $id "$(field $id LANE)" "$(field $id GPUS)" "$s$( [ $s = running ] && printf '@%s' "$(cat $Q/$id/gpus 2>/dev/null)")" \
            "$(field $id SUBMITTED)" "$(cat $Q/$id/started 2>/dev/null || echo -)" "$(field $id SCRIPT)"
    done
    last=$(echo $done_ids | tr ' ' '\n' | tail -8)
    [ -z "$last" ] || echo "-- last finished:"
    for id in $last; do
        printf '%-5s %-13s %-4s %-11s %-20s %-20s %s\n' $id "$(field $id LANE)" "$(field $id GPUS)" "$(st $id) $(cat $Q/$id/exit 2>/dev/null)" \
            "$(field $id SUBMITTED)" "$(cat $Q/$id/ended 2>/dev/null || echo -)" "$(field $id SCRIPT)"
    done
    for g in $(seq 0 $((SLOTS - 1))); do
        if slot_free $g; then echo "slot $g: free"; else echo "slot $g: BUSY $(cat $LOCKD/gpu$g.owner 2>/dev/null)"; fi
    done
    dispatcher_active && echo "dispatcher: active" || echo "dispatcher: NOT RUNNING"
    if leased; then
        dl=$(cat $Q/lease.deadline 2>/dev/null || echo 0)
        echo "lease: idle deadline $(date -u -d @"$dl" +%FT%TZ) ($(( (dl - $(date +%s)) / 60 )) min; pushed forward while busy)$( [ -e $Q/closing ] && echo '; CLOSING')"
    fi
    ;;
status)
    id="${1:-}"; id_ok "$id"; d=$Q/$id
    echo "job $id: $(st $id)"
    cat "$d/job.env"
    for f in gpus started ended exit; do [ -f "$d/$f" ] && echo "$f: $(cat $d/$f)"; done
    echo "log: $d/log ($(wc -c < $d/log) bytes); last lines:"
    tail -15 "$d/log"
    ;;
cancel)
    lane="${1:-}"; id="${2:-}"; lane_ok "$lane"; id_ok "$id"
    owner=$(field $id LANE)
    [ "$owner" = "$lane" ] || die "job $id belongs to lane $owner, not $lane: a lane cancels only its own job"
    qlock
    s=$(st $id)
    case "$s" in
        queued) setst $id cancelled; now > $Q/$id/ended; echo "gpuq: job $id cancelled (it had not started)" ;;
        starting|running) now > $Q/$id/cancel; echo "gpuq: job $id is $s; cancel requested (its processes are killed within seconds)" ;;
        *) echo "gpuq: job $id is already $s" ;;
    esac
    qunlock
    ;;
ensure)
    if [ "$MODE" = systemd ]; then dispatcher_active || systemctl restart "$SERVICE"; exit; fi
    [ ! -e $Q/closing ] || die "this pod is going down (idle past its lease: $(cat $Q/closing))"
    start_plain daemon $Q/daemon.lock $Q/dispatcher.log
    ! leased || start_plain backstop $Q/backstop.lock $Q/lease.log
    echo "gpuq: dispatcher up$(leased && echo ', lease backstop up')"
    ;;
hold)
    [ $# -gt 0 ] || die "hold needs a command"
    [ ! -e $Q/closing ] || die "this pod is going down (idle past its lease: $(cat $Q/closing))"
    exec 9>$Q/activity.lock; flock -s 9
    ! leased || date +%s > $Q/lease.last_busy
    bash -c "$*" 9>&-; rc=$?
    ! leased || date +%s > $Q/lease.last_busy
    exit $rc
    ;;
lease-check)
    leased || { echo "LEASE BAD no lease.conf (not a leased queue)"; exit 0; }
    lease_conf
    dl=$(cat $Q/lease.deadline 2>/dev/null || echo 0); t=$(date +%s)
    bs=dead; lock_free $Q/backstop.lock || bs=alive
    dp=dead; lock_free $Q/daemon.lock || dp=alive
    ok=OK; [ $bs = alive ] || ok=BAD; [ "$t" -le $(( dl + GRACE_SEC )) ] || ok=BAD
    echo "LEASE $ok deadline=$dl now=$t grace=$GRACE_MIN backstop=$bs dispatcher=$dp closing=$( [ -e $Q/closing ] && echo yes || echo no)"
    ;;
backstop)
    exec 7>$Q/backstop.lock; flock -n 7 || die "a backstop is already running"
    lease_conf; t0=$(date +%s)
    echo "$(now) backstop up: deletes pod $POD_ID ${GRACE_MIN} min past the idle deadline if the dispatcher has not"
    while :; do
        lease_conf
        dl=$(cat $Q/lease.deadline 2>/dev/null || echo $(( t0 + IDLE_SEC )))
        if [ "$(date +%s)" -ge $(( dl + GRACE_SEC )) ]; then
            echo "$(now) backstop: $GRACE_MIN min past the deadline $(date -u -d @"$dl" +%FT%TZ) and the pod is still here"
            touch $Q/closing
            lease_delete backstop && { echo "$(now) backstop: pod deleted"; sleep 3600 7>&-; exit 0; }
        fi
        sleep "${GQ_BACKSTOP_POLL:-60}" 7>&-   # a child never holds the backstop lock
    done
    ;;
runner)
    id="${1:-}"; id_ok "$id"; d=$Q/$id
    [ "$MODE" = systemd ] || echo $$ > $d/runner.pid
    # shellcheck disable=SC1090
    . "$d/job.env"
    if [ -e "$d/cancel" ]; then qlock; setst $id cancelled; now > $d/ended; qunlock; exit 0; fi
    got=''; n=0
    for g in $(seq 0 $((SLOTS - 1))); do
        exec {fd}>"$LOCKD/gpu$g.lock"
        if flock -n $fd; then got="$got $g"; n=$((n + 1)); [ $n -ge "$GPUS" ] && break; else exec {fd}>&-; fi
    done
    if [ $n -lt "$GPUS" ]; then qlock; setst $id queued; qunlock; exit 0; fi   # lost a race for a slot: stay at the head
    got=$(echo $got | tr ' ' ,)
    for g in ${got//,/ }; do
        printf 'lane=%s pid=%s utc=%s cmd=queue job %s: %s\n' "$LANE" $$ "$(now)" "$id" "$SCRIPT" > "$LOCKD/gpu$g.owner"
    done
    echo "$got" > $d/gpus; now > $d/started; rm -f $d/exit $d/ended
    qlock; setst $id running; qunlock
    start=$(date +%s); deadline=$(( start + CAP_MIN * ${GQ_MIN_SECONDS:-60} ))   # GQ_MIN_SECONDS: tests only
    token="$id-$(od -An -N8 -tx8 /dev/urandom | tr -d ' \n')"
    # OTHERS = every process of the job but the runner
    if [ "$MODE" = systemd ]; then
        cg=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)
        # builtins only, so no helper process is counted
        others() { local p; local -a all=(); OTHERS=''; mapfile -t all < "$cg/cgroup.procs" 2>/dev/null
            for p in "${all[@]}"; do [ "$p" = "$$" ] || OTHERS="$OTHERS $p"; done; }
    else
        # the runner's own environment has no token, so neither do its helpers (grep)
        others() { local e p; OTHERS=''
            for e in /proc/[0-9]*/environ; do
                p=${e#/proc/}; p=${p%/environ}
                [ "$p" = "$$" ] && continue
                grep -qzx "GQ_JOB_TOKEN=$token" "$e" 2>/dev/null && OTHERS="$OTHERS $p"
            done; }
    fi
    killall_job() {
        for sig in TERM TERM KILL; do
            others; [ -n "$OTHERS" ] || return 0
            kill -$sig $OTHERS 2>/dev/null; sleep "${GQ_KILL_WAIT:-20}"
        done
    }
    {
        echo "gpuq: job $id lane=$LANE gpus=$got cap=${CAP_MIN}min started $(now)"
        echo "gpuq: script $SCRIPT sha256 at submit $SCRIPT_SHA256, now $(sha256sum "$SCRIPT" | cut -c1-64)"
    } >> $d/log
    (
        cd "$TREE" || exit 97
        export HOME=/root USER=root LOGNAME=root LANG=C.UTF-8 SHELL=/bin/bash
        export GQ_JOB_ID=$id GQ_LANE=$LANE GQ_JOB_TOKEN=$token
        if [ "$VENDOR" = nvidia ]; then
            export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$got NVQ_JOB_ID=$id NVQ_LANE=$LANE
        else
            export ROCR_VISIBLE_DEVICES=$got HIP_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS - 1))) AMDQ_JOB_ID=$id AMDQ_LANE=$LANE
        fi
        exec bash "$SCRIPT" < /dev/null
    ) >> $d/log 2>&1 &
    cpid=$!; reason=''; rc=''
    while :; do
        if [ -z "$rc" ] && ! kill -0 $cpid 2>/dev/null; then wait $cpid; rc=$?; fi
        if [ -n "$rc" ]; then others; [ -z "$OTHERS" ] && break; fi                       # the script AND everything it started are done
        if [ -e $d/cancel ]; then reason=cancelled; killall_job; break; fi
        if [ "$(date +%s)" -ge $deadline ]; then reason=timeout; killall_job; break; fi
        sleep "${GQ_RUNNER_POLL:-5}"
    done
    if [ -z "$rc" ]; then kill -0 $cpid 2>/dev/null && kill -KILL $cpid 2>/dev/null; wait $cpid; rc=$?; fi
    echo "$rc" > $d/exit; now > $d/ended
    case "$reason" in
        cancelled) s=cancelled ;; timeout) s=timeout ;;
        *) [ "$rc" = 0 ] && s=done || s=failed ;;
    esac
    echo "gpuq: job $id $s exit=$rc at $(now) after $(( $(date +%s) - start ))s" >> $d/log
    qlock; setst $id $s; qunlock
    ;;
daemon)
    if [ "$MODE" = plain ]; then exec 7>$Q/daemon.lock; flock -n 7 || die "a dispatcher is already running"; fi
    echo "gpuq dispatcher up $(now), $SLOTS $VENDOR slots, $MODE mode$(leased && echo ', leased')"
    # a job that was running when the box (or systemd, or the container) went down: back to the head once
    qlock
    for id in $(ids); do
        case "$(st $id)" in
        starting|running)
            unit_active $id && continue
            r=$(field $id RESTARTS)
            if [ "${r:-0}" -ge 1 ]; then setst $id interrupted; now > $Q/$id/ended
                echo "gpuq: job $id cut short twice; interrupted" | tee -a $Q/$id/log
            else sed -i 's/^RESTARTS=.*/RESTARTS=1/' $Q/$id/job.env; setst $id queued
                echo "gpuq: job $id was running when the dispatcher came up without it (reboot?); requeued at the head $(now)" | tee -a $Q/$id/log
            fi ;;
        esac
    done
    qunlock
    if leased; then lease_conf; [ -s $Q/lease.last_busy ] || date +%s > $Q/lease.last_busy; fi
    starting_since=0
    while :; do
        qlock
        head=''; fhead=''; npend=0; busy=0; active=0
        for id in $(ids); do
            s=$(st $id)
            case "$s" in
            queued) npend=$((npend + 1)); [ -n "$head" ] || head=$id
                [ -n "$fhead" ] || [ ! -e $Q/$id/front ] || fhead=$id ;;
            starting) busy=1; active=1
                if ! unit_active $id && [ $(( $(date +%s) - starting_since )) -gt 60 ]; then
                    setst $id queued; echo "gpuq: runner for $id never came up; requeued $(now)" >> $Q/$id/log
                fi ;;
            running) active=1
                if ! unit_active $id; then   # the runner died without recording an end
                    echo "gpuq: runner for $id vanished $(now)" >> $Q/$id/log; setst $id interrupted; now > $Q/$id/ended
                fi ;;
            esac
        done
        echo $npend > $Q/pending
        [ -z "$fhead" ] || head=$fhead   # a front job (GQ_FRONT=1 at submit) goes first
        if [ -n "$head" ] && [ $busy = 0 ]; then
            need=$(field $head GPUS); free=0
            for g in $(seq 0 $((SLOTS - 1))); do slot_free $g && free=$((free + 1)); done
            if [ $free -ge "$need" ]; then
                setst $head starting; starting_since=$(date +%s)
                if [ "$MODE" = systemd ]; then
                    systemctl reset-failed "$U-$head.service" 2>/dev/null
                    if systemd-run --quiet --unit="$U-$head" --collect -p KillMode=control-group \
                            -E GQ_DIR=$Q -E GQ_LOCKD=$LOCKD -E GQ_UNIT=$U \
                            -p Description="gpu-queue job $head ($(field $head LANE))" "$SELF" runner $head; then
                        echo "gpuq: started $head ($(field $head LANE), $need GPU) $(now)"
                    else
                        setst $head queued; echo "gpuq: systemd-run failed for $head $(now)"
                    fi
                else
                    rm -f $Q/$head/runner.pid
                    GQ_DIR=$Q GQ_LOCKD=$LOCKD GQ_UNIT=$U setsid nohup "$SELF" runner $head 7>&- 8>&- > /dev/null 2>&1 < /dev/null &
                    echo "gpuq: started $head ($(field $head LANE), $need GPU) $(now)"
                fi
            fi
        fi
        if leased; then
            lease_conf   # re-read each pass: an IDLE_MIN edited in lease.conf applies to a live pod
            t=$(date +%s)
            held=0; for g in $(seq 0 $((SLOTS - 1))); do slot_free $g || held=1; done
            lock_free $Q/activity.lock || held=1
            if [ $npend -gt 0 ] || [ $active = 1 ] || [ $held = 1 ]; then echo $t > $Q/lease.last_busy; fi
            dl=$(( $(cat $Q/lease.last_busy 2>/dev/null || echo $t) + IDLE_SEC ))
            putf $Q/lease.deadline $dl
            if [ "$t" -ge "$dl" ] && [ ! -e $Q/closing ]; then
                echo "$(now) idle $((IDLE_SEC / 60)) min: nothing queued, running or holding a slot" > $Q/closing
                echo "gpuq: $(cat $Q/closing); deleting pod $POD_ID"
            fi
            if [ -e $Q/closing ] && [ ! -e $Q/lease.deleted_ok ]; then
                qunlock
                if lease_delete dispatcher; then touch $Q/lease.deleted_ok; echo "gpuq: pod $POD_ID deleted $(now)"; fi
                sleep "${GQ_DAEMON_POLL:-5}" 7>&-; continue
            fi
        fi
        qunlock
        sleep "${GQ_DAEMON_POLL:-5}" 7>&-
    done
    ;;
*) sed -n 2,26p "$0"; exit 2 ;;
esac
