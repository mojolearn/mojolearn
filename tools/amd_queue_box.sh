#!/bin/bash
# tools/amd_queue_box.sh -- THE FIFO JOB QUEUE ON THE CENTRAL AMD BOX.
#
# Installed on the box as /root/amd-queue/bin/amdq by
# `tools/amd_central.sh queue-install`, and run there by the systemd service
# amd-queue.service (enabled: it starts again after a reboot). Lanes never call
# this directly; they use tools/amd_central.sh submit/queue/status/cancel.
#
#   amdq submit <lane> <gpus> <cap-min> <script> [note]   enqueue; prints the job id
#   amdq queue                                            every unfinished job + the last finished ones
#   amdq status <id>                                      one job: its record, state, exit, log tail
#   amdq cancel <lane> <id>                               cancel a job the lane itself submitted
#   amdq daemon                                           (the service) the dispatcher loop
#   amdq runner <id>                                      (the dispatcher, per job) run one job
#
# JOBS. /root/amd-queue/<id>/ holds job.env (lane, gpus, script, cap, submit
# time), status (queued|starting|running|done|failed|timeout|cancelled|
# interrupted), log (the script's stdout+stderr), exit (its exit code),
# gpus, started, ended. Ids are a zero-padded counter: submission order.
#
# ORDER. Strict FIFO: the dispatcher starts the OLDEST queued job as soon as
# that job's --gpus slots are free, and starts nothing behind it until it has
# started (no backfill, so a two-GPU job is never starved by one-GPU jobs).
#
# SLOTS. The same flock slots as `amd_central.sh run`
# (/var/lock/mojolearn-central/gpu<g>.lock), so a queued job never shares a GPU
# with a `run` job, and `run` does not take a free slot while jobs are queued.
#
# A JOB runs as its own transient systemd unit amdq-<id>.service, as root, in
# /root/mojolearn-<lane> (the lane's own tree and pixi env), seeing only its
# GPUs (ROCR_VISIBLE_DEVICES=<slots>, HIP devices 0..N-1). The job is over when
# EVERY process it started has exited (a script that backgrounds work keeps its
# slot until the background work ends too). At the wall-clock cap, or on a
# cancel, every process in the job's unit is killed (TERM, then KILL). The
# dispatcher restarting never touches a running job. After a reboot a job that
# was running is put back at the head of the queue once (restarts=1), and
# marked interrupted if it is cut short again.
set -uo pipefail
Q=${AMDQ_DIR:-/root/amd-queue}                    # overridable only to test the queue beside the real one
LOCKD=${AMDQ_LOCKD:-/var/lock/mojolearn-central}
U=${AMDQ_UNIT:-amdq}
SLOTS=$(cat $Q/slots 2>/dev/null || echo 2)
DEFAULT_CAP_MIN=720
export PATH=/root/.pixi/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

now() { date -u +%FT%TZ; }
die() { echo "amdq: $*" >&2; exit 1; }
qlock() { exec 8>$Q/.lock; flock 8; }
qunlock() { flock -u 8; exec 8>&-; }
st() { cat "$Q/$1/status" 2>/dev/null || echo missing; }
setst() { echo "$2" > "$Q/$1/status.tmp" && mv -f "$Q/$1/status.tmp" "$Q/$1/status"; }
ids() { ls -1 $Q 2>/dev/null | grep -E '^[0-9]{4,}$' | sort; }
field() { (. "$Q/$1/job.env"; eval "printf '%s' \"\${$2:-}\""); }
unit_active() { systemctl is-active --quiet "$U-$1.service"; }
slot_free() { flock -n "$LOCKD/gpu$1.lock" true 2>/dev/null; }
lane_ok() { [[ "${1:-}" =~ ^[a-z0-9-]{1,32}$ ]] || die "lane must be [a-z0-9-]{1,32}"; }
id_ok() { [[ "${1:-}" =~ ^[0-9]{4,}$ ]] && [ -d "$Q/$1" ] || die "no job ${1:-?}"; }

cmd="${1:-}"; shift || true
mkdir -p $Q $LOCKD
case "$cmd" in
submit)
    lane="${1:-}"; gpus="${2:-}"; cap="${3:-}"; script="${4:-}"; note="${5:-}"
    lane_ok "$lane"
    [[ "$gpus" =~ ^[1-9][0-9]*$ ]] && [ "$gpus" -le "$SLOTS" ] || die "--gpus must be 1..$SLOTS"
    [ -n "$cap" ] || cap=$DEFAULT_CAP_MIN
    [[ "$cap" =~ ^[1-9][0-9]*$ ]] || die "--cap must be whole minutes"
    tree=/root/mojolearn-$lane
    [ -d "$tree" ] || die "no lane tree $tree on the box (run amd_central.sh sync first)"
    case "$script" in /*) ;; "") die "submit needs a script on the box" ;; *) script="$tree/$script" ;; esac
    [ -f "$script" ] || die "no script $script on the box"
    qlock
    n=$(( $(cat $Q/.seq 2>/dev/null || echo 0) + 1 )); echo $n > $Q/.seq
    id=$(printf '%04d' $n); d=$Q/$id; mkdir -p "$d"
    {
        printf 'ID=%q\nLANE=%q\nGPUS=%q\nCAP_MIN=%q\nSCRIPT=%q\nTREE=%q\n' "$id" "$lane" "$gpus" "$cap" "$script" "$tree"
        printf 'SCRIPT_SHA256=%q\nSUBMITTED=%q\nNOTE=%q\nRESTARTS=0\n' "$(sha256sum "$script" | cut -c1-64)" "$(now)" "$note"
    } > "$d/job.env"
    : > "$d/log"; setst $id queued
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
    systemctl is-active --quiet amd-queue.service && echo "dispatcher: active" || echo "dispatcher: NOT RUNNING"
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
        queued) setst $id cancelled; now > $Q/$id/ended; echo "amdq: job $id cancelled (it had not started)" ;;
        starting|running) now > $Q/$id/cancel; echo "amdq: job $id is $s; cancel requested (its processes are killed within seconds)" ;;
        *) echo "amdq: job $id is already $s" ;;
    esac
    qunlock
    ;;
runner)
    id="${1:-}"; id_ok "$id"; d=$Q/$id
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
    start=$(date +%s); deadline=$(( start + CAP_MIN * 60 ))
    cg=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)
    # OTHERS = every process in this job's unit but the runner; builtins only, so no helper process is counted
    others() { local p; local -a all=(); OTHERS=''; mapfile -t all < "$cg/cgroup.procs" 2>/dev/null
        for p in "${all[@]}"; do [ "$p" = "$$" ] || OTHERS="$OTHERS $p"; done; }
    killall_job() {
        for sig in TERM TERM KILL; do
            others; [ -n "$OTHERS" ] || return 0
            kill -$sig $OTHERS 2>/dev/null; sleep 20
        done
    }
    {
        echo "amdq: job $id lane=$LANE gpus=$got cap=${CAP_MIN}min started $(now)"
        echo "amdq: script $SCRIPT sha256 at submit $SCRIPT_SHA256, now $(sha256sum "$SCRIPT" | cut -c1-64)"
    } >> $d/log
    (
        cd "$TREE" || exit 97
        export HOME=/root USER=root LOGNAME=root LANG=C.UTF-8 SHELL=/bin/bash
        export ROCR_VISIBLE_DEVICES=$got HIP_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS - 1)))
        export AMDQ_JOB_ID=$id AMDQ_LANE=$LANE
        exec bash "$SCRIPT" < /dev/null
    ) >> $d/log 2>&1 &
    cpid=$!; reason=''; rc=''
    while :; do
        if [ -z "$rc" ] && ! kill -0 $cpid 2>/dev/null; then wait $cpid; rc=$?; fi
        if [ -n "$rc" ]; then others; [ -z "$OTHERS" ] && break; fi                       # the script AND everything it started are done
        if [ -e $d/cancel ]; then reason=cancelled; killall_job; break; fi
        if [ "$(date +%s)" -ge $deadline ]; then reason=timeout; killall_job; break; fi
        sleep 5
    done
    if [ -z "$rc" ]; then kill -0 $cpid 2>/dev/null && kill -KILL $cpid 2>/dev/null; wait $cpid; rc=$?; fi
    echo "$rc" > $d/exit; now > $d/ended
    case "$reason" in
        cancelled) s=cancelled ;; timeout) s=timeout ;;
        *) [ "$rc" = 0 ] && s=done || s=failed ;;
    esac
    echo "amdq: job $id $s exit=$rc at $(now) after $(( $(date +%s) - start ))s" >> $d/log
    qlock; setst $id $s; qunlock
    ;;
daemon)
    echo "amdq dispatcher up $(now), $SLOTS slots"
    # a job that was running when the box (or systemd) went down: back to the head once
    qlock
    for id in $(ids); do
        case "$(st $id)" in
        starting|running)
            unit_active $id && continue
            r=$(field $id RESTARTS)
            if [ "${r:-0}" -ge 1 ]; then setst $id interrupted; now > $Q/$id/ended
                echo "amdq: job $id cut short twice; interrupted" | tee -a $Q/$id/log
            else sed -i 's/^RESTARTS=.*/RESTARTS=1/' $Q/$id/job.env; setst $id queued
                echo "amdq: job $id was running when the dispatcher came up without it (reboot?); requeued at the head $(now)" | tee -a $Q/$id/log
            fi ;;
        esac
    done
    qunlock
    starting_since=0
    while :; do
        qlock
        head=''; npend=0; busy=0
        for id in $(ids); do
            s=$(st $id)
            case "$s" in
            queued) npend=$((npend + 1)); [ -n "$head" ] || head=$id ;;
            starting) busy=1
                if ! unit_active $id && [ $(( $(date +%s) - starting_since )) -gt 60 ]; then
                    setst $id queued; echo "amdq: runner for $id never came up; requeued $(now)" >> $Q/$id/log
                fi ;;
            running)
                if ! unit_active $id; then   # the runner died without recording an end
                    echo "amdq: runner for $id vanished $(now)" >> $Q/$id/log; setst $id interrupted; now > $Q/$id/ended
                fi ;;
            esac
        done
        echo $npend > $Q/pending
        if [ -n "$head" ] && [ $busy = 0 ]; then
            need=$(field $head GPUS); free=0
            for g in $(seq 0 $((SLOTS - 1))); do slot_free $g && free=$((free + 1)); done
            if [ $free -ge "$need" ]; then
                setst $head starting; starting_since=$(date +%s)
                systemctl reset-failed "$U-$head.service" 2>/dev/null
                if systemd-run --quiet --unit="$U-$head" --collect -p KillMode=control-group \
                        -E AMDQ_DIR=$Q -E AMDQ_LOCKD=$LOCKD -E AMDQ_UNIT=$U \
                        -p Description="amd-queue job $head ($(field $head LANE))" $Q/bin/amdq runner $head; then
                    echo "amdq: started $head ($(field $head LANE), $need GPU) $(now)"
                else
                    setst $head queued; echo "amdq: systemd-run failed for $head $(now)"
                fi
            fi
        fi
        qunlock
        sleep 5
    done
    ;;
*) sed -n 2,40p "$0"; exit 2 ;;
esac
