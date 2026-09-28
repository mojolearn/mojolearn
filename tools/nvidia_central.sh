#!/bin/bash
# tools/nvidia_central.sh -- THE SHARED NVIDIA PODS (Andrew, 2026-09-28: "share
# a runpod or 2 runpods and not create 12 of them").
#
# THE ONLY WAY A LANE USES NVIDIA. At most NVC_MAX_PODS (3, the account cap in
# tools/dev_pod.sh) shared RunPod pods, each with several cheap GPUs (4x or 2x
# RTX 4090 where RunPod has them; a 1-GPU pod is the fallback), keyed nvc1,
# nvc2, nvc3 in the dev_pod state. Every lane's GPU work goes through the
# pod's FIFO queue (tools/gpu_queue_box.sh, the same queue as the central AMD
# box), one job per GPU slot, in the lane's own tree /root/mojolearn-<lane>.
# A lane never runs `tools/dev_pod.sh up` for NVIDIA (dev_pod refuses it).
# ONLY THE ORCHESTRATOR PROVISIONS MACHINES (Andrew, 2026-09-28): `up`,
# `install` and `down` refuse without MOJOLEARN_ORCHESTRATOR=1; a lane that
# finds no pod up says so to the orchestrator and waits.
#
#   tools/nvidia_central.sh up [N]                     ORCHESTRATOR ONLY (MOJOLEARN_ORCHESTRATOR=1). RENTS.
#                                                      Brings the shared set to N pods (default NVC_PODS=2, at
#                                                      most 3). Idempotent: never more than N.
#   tools/nvidia_central.sh sync   <lane> <worktree>   patch-sync to /root/mojolearn-<lane> on the lane's pod
#                                                      (a lane is assigned to one pod at its first sync)
#   tools/nvidia_central.sh submit <lane> [--gpus N] [--cap MIN] [--note TEXT] <script-on-box>
#                                                      THE WAY TO RUN A GATE: enqueue; prints the job id
#                                                      (<pod>-<n>, e.g. nvc1-0003). Default and max cap 240 min.
#   tools/nvidia_central.sh queue                      every pod's queued/running jobs, last finished, slots, lease
#   tools/nvidia_central.sh status [<id>]              no id: pods, lanes, queues; id: that job
#   tools/nvidia_central.sh log    <id> [N]            last N (default 40) lines of the job's log
#   tools/nvidia_central.sh cancel <lane> <id>         cancel the lane's OWN job (queued or running)
#   tools/nvidia_central.sh run    <lane> [--gpus N] [--wait MIN] <command>
#                                                      SHORT interactive GPU command on N free slots
#                                                      (yields to queued jobs; exit 75 if no slot came)
#   tools/nvidia_central.sh sh     <lane> <command>    NO GPU (CUDA_VISIBLE_DEVICES=-1): builds, pixi, tail, git
#   tools/nvidia_central.sh fetch  <lane> <path> <local dir>   tar a box path back to the Mac
#   tools/nvidia_central.sh install <pod>              ORCHESTRATOR ONLY (up does this) install/upgrade queue + lease
#   tools/nvidia_central.sh down   <pod>|--all [--force]   ORCHESTRATOR ONLY: delete shared pods (refuses while
#                                                      jobs are queued/running; they go down idle by themselves)
#   tools/nvidia_central.sh watch  <pod>               (internal) (re)start the Mac watcher of a pod
#
# THE LEASE IS THE QUEUE. A pod is busy while any job is queued, starting or
# running, a `run` holds a slot, or an `sh` command runs. It is DELETED by
# itself 30 idle minutes after it was last busy (IDLE_MIN), through the RunPod
# API from the pod; a backstop process on the pod deletes it 15 minutes later
# if the dispatcher did not; and the Mac watcher (below) deletes it when the
# pod's backstop is dead or the pod stops answering. So nothing idles for
# hours and nothing depends on an agent session staying alive. The fixed
# runpod_guard.sh watchdog dev_pod armed at the create is retired once this
# lease is proven alive; `dev_pod.sh extend/down nvc<N>` refuse (the queue owns
# the lease).
#
# THE MAC WATCHER replaces dev_pod's Mac dead-man for a shared pod: a nohup
# loop in the pod's state dir that asks the pod `lease-check` every
# NVC_WATCH_SECONDS (300). Pod gone per the API: it retires the state and
# exits. Backstop dead: it runs `ensure` once, then deletes the pod if the
# lease is still not OK. Three unanswered checks in a row: it deletes the pod.
#
# LANES AND PODS. A lane's tree, pixi env and Mojo cache live on ONE pod (the
# live pod with the fewest lanes per GPU when it first syncs); its jobs run
# there. ~/mojolearn-evidence/nvidia_central/lanes/<lane>/pod names it. When a
# pod goes down idle, the lane's next sync picks a live pod (and its first
# `pixi install` there links from that pod's package cache).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATE_ROOT="${MOJOLEARN_DEVPOD_STATE:-$HOME/mojolearn-evidence/devpods}"
NVC_HOME="${MOJOLEARN_NVC_HOME:-$HOME/mojolearn-evidence/nvidia_central}"
CONF="${MOJOLEARN_NVC_CONF:-$HOME/mojolearn-evidence/nvidia_central.env}"
# shellcheck disable=SC1090
[ ! -f "$CONF" ] || . "$CONF"
NVC_MAX_PODS=3                                        # the account cap (tools/dev_pod.sh MAX_RUNPOD_PODS)
NVC_PODS=${NVC_PODS:-2}                               # `up` with no N brings the set to this many
NVC_GPU_COUNTS=${NVC_GPU_COUNTS:-"4 2 1"}             # GPUs per pod, tried in order (1 = the plain dev_pod fallback)
NVC_GPUS=${NVC_GPUS:-"NVIDIA GeForce RTX 4090,NVIDIA RTX A6000,NVIDIA A40"}
NVC_STOCK_WAIT=${NVC_STOCK_WAIT:-5}                   # minutes of out-of-stock retry per multi-GPU count
NVC_STOCK_WAIT_1=${NVC_STOCK_WAIT_1:-30}              # ... and for the 1-GPU fallback
NVC_MIN_BALANCE=${NVC_MIN_BALANCE:-10}                # `up` refuses below this RunPod balance (USD)
NVC_IDLE_MIN=${NVC_IDLE_MIN:-30}
NVC_GRACE_MIN=${NVC_GRACE_MIN:-15}
NVC_CAP_MIN=${NVC_CAP_MIN:-240}                       # a job's default AND maximum wall-clock cap
NVC_WATCH_SECONDS=${NVC_WATCH_SECONDS:-300}
DEVPOD="${MOJOLEARN_NVC_DEVPOD:-$ROOT/tools/dev_pod.sh}"
NO_API="${MOJOLEARN_NVC_NO_API:-0}"                   # tests only: no RunPod API calls (reconcile, balance)
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=8 -o BatchMode=yes"
QD=/root/gpu-queue
NVQ=$QD/bin/nvq
LOCKD=/var/lock/mojolearn-central
BOX_PATH='export PATH=/root/.pixi/bin:/usr/local/cuda/bin:$PATH'

die() { echo "nvidia_central: $*" >&2; exit 1; }
say() { echo "nvidia_central: $*"; }
sq() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }
TMPD="$(mktemp -d "${TMPDIR:-/tmp}/nvc.XXXXXX")"; trap 'rm -rf "$TMPD"' EXIT
CURLRC="$TMPD/curlrc"
. "$ROOT/tools/runpod_pod_lib.sh"
mkdir -p "$NVC_HOME/lanes"

pod_keys() { local i; for i in $(seq 1 $NVC_MAX_PODS); do echo nvc$i; done; }
live_pods() { local k; for k in $(pod_keys); do [ ! -f "$STATE_ROOT/$k/state.env" ] || echo $k; done; }
pod_env() {  # <key>: SSH_TARGET POD_ID GPU GPU_COUNT COST_HR
    [ -f "$STATE_ROOT/$1/state.env" ] || die "shared pod $1 is not up"
    SSH_TARGET=$(. "$STATE_ROOT/$1/state.env"; printf '%s' "$SSH_TARGET")
    POD_ID=$(. "$STATE_ROOT/$1/state.env"; printf '%s' "$POD_ID")
    GPU=$(. "$STATE_ROOT/$1/state.env"; printf '%s' "$GPU")
    GPU_COUNT=$(. "$STATE_ROOT/$1/state.env"; printf '%s' "${GPU_COUNT:-1}")
    COST_HR=$(. "$STATE_ROOT/$1/state.env"; printf '%s' "${COST_HR:-?}")
}
# shellcheck disable=SC2086
box() { with_timeout "${BOX_TIMEOUT:-900}" ssh $SSH_OPTS $SSH_TARGET "$1"; }
lane_ok() { [[ "${1:-}" =~ ^[a-z0-9-]{1,24}$ ]] || die "lane must be [a-z0-9-]{1,24}"; }
lane_pod() {  # <lane>: its live pod, or nothing
    local k; k=$(cat "$NVC_HOME/lanes/$1/pod" 2>/dev/null || true)
    [ -n "$k" ] && [ -f "$STATE_ROOT/$k/state.env" ] && echo "$k" || true
}
need_lane_pod() {
    LANE_POD=$(lane_pod "$1")
    [ -n "$LANE_POD" ] || die "lane $1 has no live shared pod; run: tools/nvidia_central.sh sync $1 <worktree>"
    pod_env "$LANE_POD"
}
split_id() {  # <pod>-<n> | <n> -> JOB_POD JOB_N
    case "$1" in
        nvc[0-9]-[0-9]*) JOB_POD=${1%%-*}; JOB_N=${1#*-} ;;
        [0-9]*) JOB_N=$1; JOB_POD=$(live_pods)
                [ "$(echo "$JOB_POD" | grep -c .)" = 1 ] || die "several pods are up: name the job as <pod>-$1 (e.g. nvc1-$1)" ;;
        *) die "a job id is <pod>-<n>, e.g. nvc1-0003" ;;
    esac
    [[ "$JOB_N" =~ ^[0-9]{4,}$ ]] || die "a job id is <pod>-<n>, e.g. nvc1-0003"
}

# Pods the API says are gone: the watcher stopped, the state retired, lanes unassigned.
reconcile() {
    [ "$NO_API" != 1 ] || return 0
    load_key || return 0
    local k
    for k in $(live_pods); do
        pod_env $k
        rp_call GET "$RP/pods/$POD_ID"
        if [ "$RP_CODE" = 404 ] || [ "$(rp_py status)" = TERMINATED ]; then retire $k "gone per the RunPod API"; fi
    done
}
retire() {  # <key> <why>
    local d="$STATE_ROOT/$1" p
    p=$(cat "$d/deadman.pid" 2>/dev/null || true)
    [ -z "$p" ] || { pkill -P "$p" 2>/dev/null || true; kill "$p" 2>/dev/null || true; }
    mv "$d" "$d.down-$(date -u +%Y%m%dT%H%M%SZ)"
    say "shared pod $1 retired ($2)"
}
balance() {
    if [ -n "${MOJOLEARN_NVC_TEST_BALANCE:-}" ]; then echo "$MOJOLEARN_NVC_TEST_BALANCE"; return; fi
    curl -s -m 20 -K "$CURLRC" -H 'Content-Type: application/json' https://api.runpod.io/graphql \
        -d '{"query":"query { myself { clientBalance } }"}' \
        | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"]["myself"]["clientBalance"])' 2>/dev/null
}
up_lock() {
    local t0; t0=$(date +%s)
    until mkdir "$NVC_HOME/up.lock" 2>/dev/null; do
        [ $(( $(date +%s) - $(stat -f %m "$NVC_HOME/up.lock" 2>/dev/null || stat -c %Y "$NVC_HOME/up.lock" 2>/dev/null || echo 0) )) -lt 7200 ] \
            || { rm -rf "$NVC_HOME/up.lock"; continue; }
        [ $(( $(date +%s) - t0 )) -lt 60 ] || die "another nvidia_central.sh up holds $NVC_HOME/up.lock; it is bringing the pods up (wait for it, then status)"
        sleep 2
    done
    trap 'rm -rf "$NVC_HOME/up.lock" "$TMPD"' EXIT
}

# The Mac watcher: a self-contained script in the pod's state dir (a running
# script is never rewritten under it), started with nohup; its pid replaces
# dev_pod's Mac dead-man in deadman.pid (dev_pod down stops it).
start_watch() {  # <key>
    local k=$1 d="$STATE_ROOT/$1" w old pid
    pod_env $k
    w="$d/watch-$(date +%s)"; mkdir -p "$w"
    ( umask 077; if [ -s "$d/deadman/curlrc" ]; then cp "$d/deadman/curlrc" "$w/curlrc"; elif [ -s "$CURLRC" ]; then cp "$CURLRC" "$w/curlrc"; else : > "$w/curlrc"; fi )
    cat > "$w/watch.sh" <<'WATCH'
#!/bin/bash
# tools/nvidia_central.sh's Mac watcher for one shared NVIDIA pod.
set -u
W="$(cd "$(dirname "$0")" && pwd)"
log() { echo "$(date -u +%FT%TZ) $*" >> "$W/watch.log"; }
tmo() { perl -e 'alarm shift @ARGV; exec @ARGV or exit 127' "$@"; }
api_gone() {
    [ "@NO_API@" = 1 ] && return 1
    local c; c=$(curl -s -m 30 -K "$W/curlrc" -o "$W/pod.json" -w '%{http_code}' "@RP@/pods/@POD@")
    [ "$c" = 404 ] && return 0
    [ "$c" = 200 ] && grep -q '"desiredStatus": *"TERMINATED"' "$W/pod.json" && return 0
    return 1
}
retire() { [ -d "@D@" ] && mv "@D@" "@D@.down-$(date -u +%Y%m%dT%H%M%SZ)"; log "state retired: $1"; exit 0; }
delete_pod() {
    log "DELETING @POD@: $1"
    if [ "@FAKE@" = 1 ]; then log "FAKE DELETE @POD@"; exit 0; fi
    for u in "@RP@/pods/@POD@" "@RPV2@/pods/@POD@"; do
        c=$(curl -s -m 60 -K "$W/curlrc" -o /dev/null -w '%{http_code}' -X DELETE "$u"); log "DELETE $u -> $c"
        case "$c" in 2*|404) break ;; esac
    done
    for i in 1 2 3 4 5 6 7 8; do api_gone && retire "deleted by the watcher ($1)"; sleep 10; done
    log "@POD@ NOT CONFIRMED GONE after the delete; retrying next round"
}
check() { tmo 90 ssh @SSH_OPTS@ @TARGET@ "@NVQ@ lease-check" 2>> "$W/watch.log" | tail -1; }
log "watcher up for @POD@ (@KEY@), every @SECS@s"
fails=0
while :; do
    sleep @SECS@
    api_gone && retire "gone per the RunPod API"
    out=$(check)
    case "$out" in
    "LEASE OK"*) fails=0 ;;
    "LEASE BAD"*)
        log "$out; running ensure once"
        tmo 90 ssh @SSH_OPTS@ @TARGET@ "@NVQ@ ensure" >> "$W/watch.log" 2>&1
        out=$(check)
        case "$out" in "LEASE OK"*) fails=0; log "lease OK after ensure" ;; *) delete_pod "lease not OK: $out" ;; esac ;;
    *)
        fails=$((fails + 1)); log "no lease answer ($fails/3): ${out:-nothing}"
        [ $fails -lt 3 ] || delete_pod "the pod did not answer 3 checks in a row" ;;
    esac
done
WATCH
    sed -i.bak -e "s|@RP@|$RP|g" -e "s|@RPV2@|$RP_V2|g" -e "s|@POD@|$POD_ID|g" -e "s|@KEY@|$k|g" -e "s|@D@|$d|g" \
        -e "s|@SECS@|$NVC_WATCH_SECONDS|g" -e "s|@NVQ@|$NVQ|g" -e "s|@SSH_OPTS@|$SSH_OPTS|g" -e "s|@TARGET@|$SSH_TARGET|g" \
        -e "s|@NO_API@|$NO_API|g" -e "s|@FAKE@|${MOJOLEARN_NVC_WATCH_FAKE_DELETE:-0}|g" "$w/watch.sh"
    rm -f "$w/watch.sh.bak"
    ! grep -q '@[A-Z_]*@' "$w/watch.sh" && bash -n "$w/watch.sh" || die "the watcher did not compose ($w/watch.sh)"
    nohup bash -c 'trap "" HUP INT; exec bash "$0"' "$w/watch.sh" > /dev/null 2>&1 < /dev/null &
    pid=$!; sleep 1; kill -0 $pid 2>/dev/null || die "the Mac watcher of $k did not start; the old dead-man stays armed"
    old=$(cat "$d/deadman.pid" 2>/dev/null || true)
    echo $pid > "$d/deadman.pid"
    if [ -n "$old" ] && [ "$old" != "$pid" ]; then pkill -P "$old" 2>/dev/null || true; kill "$old" 2>/dev/null || true; fi
    say "$k: Mac watcher pid $pid ($w/watch.log) replaces dead-man ${old:-none}"
}

# Install or upgrade the queue and its lease on one pod, retire the fixed
# runpod_guard watchdog once the lease is proven alive, start the Mac watcher.
install_pod() {  # <key>
    local k=$1 n out
    pod_env $k
    n=$(BOX_TIMEOUT=120 box "nvidia-smi -L 2>/dev/null | grep -c '^GPU '" < /dev/null | tr -dc 0-9)
    [ -n "$n" ] && [ "$n" -ge 1 ] || die "$k: nvidia-smi shows no GPU"
    [ "$n" = "$GPU_COUNT" ] || say "$k: nvidia-smi shows $n GPUs (state says $GPU_COUNT); the queue uses $n"
    BOX_TIMEOUT=120 box "set -e; command -v flock setsid > /dev/null
[ -s /tmp/mojolearn-lease.curlrc ] || { echo 'no RunPod key on the pod (/tmp/mojolearn-lease.curlrc); the queue lease could not delete it' >&2; exit 3; }
mkdir -p $QD/bin $LOCKD && cat > $NVQ.new && chmod 755 $NVQ.new && mv -f $NVQ.new $NVQ
cd $QD && echo $n > slots && echo nvidia > vendor && echo plain > mode && echo - > service && echo $NVC_CAP_MIN > default_cap && echo $NVC_CAP_MIN > max_cap
printf 'POD_ID=%s\nIDLE_MIN=%s\nGRACE_MIN=%s\nCURLRC=/tmp/mojolearn-lease.curlrc\n' $(sq "$POD_ID") $NVC_IDLE_MIN $NVC_GRACE_MIN > lease.conf
[ -s lease.last_busy ] || date +%s > lease.last_busy
export GQ_DIR=$QD GQ_LOCKD=$LOCKD GQ_UNIT=nvq; $NVQ ensure; sleep 7; $NVQ lease-check" < "$ROOT/tools/gpu_queue_box.sh" > "$TMPD/install.out" 2>&1 \
        || { cat "$TMPD/install.out" >&2; die "$k: queue install failed"; }
    cat "$TMPD/install.out"
    out=$(tail -1 "$TMPD/install.out")
    case "$out" in "LEASE OK"*) ;; *) die "$k: the queue lease is not OK after install ($out); the fixed watchdog stays armed" ;; esac
    # the fixed-deadline runpod_guard watchdog would delete the pod under running jobs; the queue lease replaces it
    BOX_TIMEOUT=60 box "p=\$(cat /tmp/mojolearn-lease.pid 2>/dev/null); [ -z \"\$p\" ] || kill \$p 2>/dev/null; rm -f /tmp/mojolearn-lease.pid; echo 'retired by nvidia_central.sh install: the queue lease ($NVQ lease-check) owns this pod' >> /tmp/mojolearn-lease.out; true" < /dev/null
    echo "the queue lease ($NVQ, idle $NVC_IDLE_MIN min) owns this pod since $(date -u +%FT%TZ)" > "$STATE_ROOT/$k/QUEUE_LEASE"
    start_watch $k
    say "$k: queue up, $n GPU slots, pod $POD_ID deletes itself $NVC_IDLE_MIN idle minutes after its last job"
}

pick_pod() {  # the live pod with the fewest assigned lanes per GPU
    local k best='' bestv='' c l v
    for k in $(live_pods); do
        pod_env $k; c=$GPU_COUNT; l=0
        for f in "$NVC_HOME"/lanes/*/pod; do [ -f "$f" ] && [ "$(cat "$f")" = $k ] && l=$((l + 1)); done
        v=$(( l * 1000 / c ))
        if [ -z "$best" ] || [ $v -lt $bestv ]; then best=$k; bestv=$v; fi
    done
    echo "$best"
}

cmd="${1:-}"; shift || true
case "$cmd" in up|install|down)
    [ "${MOJOLEARN_ORCHESTRATOR:-0}" = 1 ] || die "only the orchestrator provisions machines (Andrew, 2026-09-28): '$cmd' needs MOJOLEARN_ORCHESTRATOR=1. A lane reports that it needs NVIDIA and waits." ;;
esac
case "$cmd" in
up)
    want=${1:-$NVC_PODS}
    [[ "$want" =~ ^[1-9]$ ]] && [ "$want" -le $NVC_MAX_PODS ] || die "up [N]: N is 1..$NVC_MAX_PODS"
    [ "$NO_API" = 1 ] || load_key || die "no RunPod key (~/.mojolearn_runpod_key, mode 600)"
    up_lock
    reconcile
    have=$(live_pods | grep -c . || true)
    [ "$have" -lt "$want" ] || { say "$have shared pod(s) already up ($(live_pods | tr '\n' ' ')); nothing rented"; exit 0; }
    bal=$(balance); [ -n "$bal" ] || die "could not read the RunPod balance; nothing rented"
    awk -v b="$bal" -v m="$NVC_MIN_BALANCE" 'BEGIN{exit !(b >= m)}' \
        || die "RunPod balance is \$$bal, under \$$NVC_MIN_BALANCE (NVC_MIN_BALANCE); nothing rented. Fund the account first."
    say "RunPod balance \$$bal; bringing the shared set from $have to $want pod(s)"
    for k in $(pod_keys); do
        [ "$(live_pods | grep -c . || true)" -lt "$want" ] || break
        [ ! -f "$STATE_ROOT/$k/state.env" ] || continue
        made=0
        for c in $NVC_GPU_COUNTS; do
            w=$NVC_STOCK_WAIT; [ "$c" != 1 ] || w=$NVC_STOCK_WAIT_1
            say "$k: asking RunPod for ${c}x of: $NVC_GPUS (out-of-stock retry $w min)"
            if MOJOLEARN_DEVPOD_VIA_CENTRAL=1 MOJOLEARN_DEVPOD_GPU_COUNT=$c MOJOLEARN_DEVPOD_GPUS="$NVC_GPUS" \
                    MOJOLEARN_DEVPOD_RETRY_MINUTES=$w "$DEVPOD" up $k 60 2>&1 | tee "$TMPD/up-$k-$c.log"; then
                made=1; break
            fi
            grep -q 'out of stock' "$TMPD/up-$k-$c.log" || die "$k: dev_pod up failed, and not for stock (log above); stopping"
        done
        [ $made = 1 ] || die "$k: RunPod had none of $NVC_GPUS at any of ($NVC_GPU_COUNTS) GPUs; nothing more rented"
        ( install_pod $k ) || { say "$k: install failed; deleting it"; MOJOLEARN_NVC_DOWN=1 "$DEVPOD" down $k; exit 1; }
    done
    say "shared NVIDIA pods up: $(live_pods | tr '\n' ' ')"
    ;;
install) k="${1:?install <pod>}"; [ "$NO_API" = 1 ] || load_key || true; install_pod "$k" ;;
watch) k="${1:?watch <pod>}"; [ "$NO_API" = 1 ] || load_key || true; start_watch "$k" ;;
sync)
    lane="${1:-}"; lane_ok "$lane"; wt="${2:?sync <lane> <worktree>}"
    reconcile
    k=$(lane_pod "$lane"); [ -n "$k" ] || k=$(pick_pod)
    [ -n "$k" ] || die "no shared NVIDIA pod is up. Only the orchestrator brings them up (tools/nvidia_central.sh up); report it and wait"
    pod_env $k
    BOX_TIMEOUT=120 box "$NVQ ensure" < /dev/null || die "$k's queue is not accepting work (above); tools/nvidia_central.sh status"
    mkdir -p "$NVC_HOME/lanes/$lane/$lane"
    prev=$(cat "$NVC_HOME/lanes/$lane/pod" 2>/dev/null || true); echo $k > "$NVC_HOME/lanes/$lane/pod"
    [ -z "$prev" ] || [ "$prev" = "$k" ] || say "lane $lane moves from $prev (down) to $k; run 'sh $lane \"pixi install\"' there first"
    # dev_pod.sh's patch sync, pointed at /root/mojolearn-<lane> on the lane's pod
    grep -v '^BOX_DIR=\|^BOX_ENV=\|^PROVIDER=' "$STATE_ROOT/$k/state.env" > "$NVC_HOME/lanes/$lane/$lane/state.env"
    printf 'PROVIDER=central\nBOX_DIR=/root/mojolearn-%s\nBOX_ENV=\n' "$lane" >> "$NVC_HOME/lanes/$lane/$lane/state.env"
    MOJOLEARN_DEVPOD_STATE="$NVC_HOME/lanes/$lane" MOJOLEARN_DEVPOD_ALLOW_SELF=1 "$ROOT/tools/dev_pod.sh" sync "$lane" "$wt"
    BOX_TIMEOUT=60 box "$NVQ hold true" < /dev/null || true
    say "lane $lane is on $k ($GPU_COUNT GPU slots, $GPU); submit with: tools/nvidia_central.sh submit $lane <script>"
    ;;
submit)
    lane="${1:-}"; lane_ok "$lane"; shift; n=1; cap=$NVC_CAP_MIN; note=''
    while [ $# -gt 0 ]; do
        case "$1" in
            --gpus) n="${2:?--gpus N}"; shift 2 ;;
            --cap) cap="${2:?--cap MIN}"; shift 2 ;;
            --note) note="${2:?--note TEXT}"; shift 2 ;;
            *) break ;;
        esac
    done
    [ $# -eq 1 ] || die "submit <lane> [--gpus N] [--cap MIN] [--note TEXT] <script-on-box> (one script path; put arguments inside the script)"
    need_lane_pod "$lane"
    id=$(box "$NVQ ensure > /dev/null && $NVQ submit $(sq "$lane") $(sq "$n") $(sq "$cap") $(sq "$1") $(sq "$note")" < /dev/null) || exit 1
    echo "$LANE_POD-$id"
    ;;
queue)
    for k in $(live_pods); do
        pod_env $k; echo "== $k: pod $POD_ID, $GPU_COUNT GPU ($GPU), \$$COST_HR/hr"
        BOX_TIMEOUT=120 box "$NVQ queue" < /dev/null || echo "(no answer from $k)"
    done
    [ -n "$(live_pods)" ] || echo "no shared NVIDIA pod is up (only the orchestrator brings them up: tools/nvidia_central.sh up)"
    ;;
status)
    if [ -n "${1:-}" ]; then split_id "$1"; pod_env $JOB_POD; box "$NVQ status $(sq "$JOB_N")" < /dev/null; exit; fi
    reconcile
    for k in $(live_pods); do
        pod_env $k
        echo "== $k: pod $POD_ID, $GPU_COUNT GPU ($GPU), \$$COST_HR/hr, ssh ${SSH_TARGET}"
        echo "lanes: $(for f in "$NVC_HOME"/lanes/*/pod; do [ -f "$f" ] && [ "$(cat "$f")" = $k ] && basename "$(dirname "$f")"; done | tr '\n' ' ')"
        echo "watcher: $(tail -1 "$(ls -d "$STATE_ROOT/$k"/watch-* 2>/dev/null | tail -1)/watch.log" 2>/dev/null || echo none)"
        BOX_TIMEOUT=120 box "$NVQ lease-check; nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null; echo; $NVQ queue" < /dev/null || echo "(no answer from $k)"
    done
    [ -n "$(live_pods)" ] || echo "no shared NVIDIA pod is up (only the orchestrator brings them up: tools/nvidia_central.sh up)"
    ;;
log)
    split_id "${1:?log <id> [N]}"; nl="${2:-40}"; [[ "$nl" =~ ^[0-9]+$ ]] || die "N is a line count"
    pod_env $JOB_POD; box "tail -n $nl $QD/$JOB_N/log" < /dev/null
    ;;
cancel)
    lane="${1:-}"; lane_ok "$lane"; split_id "${2:?cancel <lane> <id>}"
    pod_env $JOB_POD; box "$NVQ cancel $(sq "$lane") $(sq "$JOB_N")" < /dev/null
    ;;
run)
    lane="${1:-}"; lane_ok "$lane"; shift; n=1; wait_min=120
    while [ $# -gt 0 ]; do
        case "$1" in
            --gpus) n="${2:?--gpus N}"; shift 2 ;;
            --wait) wait_min="${2:?--wait MIN}"; shift 2 ;;
            *) break ;;
        esac
    done
    [ $# -gt 0 ] || die "run needs a command"
    need_lane_pod "$lane"
    [[ "$n" =~ ^[1-9][0-9]*$ ]] && [ "$n" -le "$GPU_COUNT" ] || die "--gpus must be 1..$GPU_COUNT on $LANE_POD"
    user_cmd="$*"
    remote="set -u; $NVQ ensure > /dev/null || exit 1; mkdir -p $LOCKD; N=$n; T=\$(cat $QD/slots); deadline=\$(( \$(date +%s) + $wait_min * 60 )); said=0
while :; do
    got=''; fds=''
    pend=\$(cat $QD/pending 2>/dev/null || echo 0)
    [ \"\$pend\" -gt 0 ] 2>/dev/null && T0=0 || T0=\$T
    for g in \$(seq 0 \$((T0 - 1))); do
        exec {fd}>$LOCKD/gpu\$g.lock
        if flock -n \$fd; then got=\"\$got \$g\"; fds=\"\$fds \$fd\"; [ \$(echo \$got | wc -w) -ge \$N ] && break
        else exec {fd}>&-; fi
    done
    [ \$(echo \$got | wc -w) -ge \$N ] && break
    for f in \$fds; do exec {f}>&-; done
    [ \$(date +%s) -lt \$deadline ] || { echo 'nvidia_central: no free GPU slot within $wait_min min' >&2; exit 75; }
    [ \$said = 1 ] || { echo 'nvidia_central: waiting for '\$N' free GPU slot(s) (queued jobs go first; use submit to hold a place in line)...' >&2; said=1; }
    sleep 5
done
for g in \$got; do printf 'lane=%s pid=%s utc=%s cmd=%s\n' $(sq "$lane") \$\$ \$(date -u +%FT%TZ) $(sq "${user_cmd:0:160}") > $LOCKD/gpu\$g.owner; done
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=\$(echo \$got | tr ' ' ,)
echo \"nvidia_central: $lane on GPU(s) \$CUDA_VISIBLE_DEVICES of $LANE_POD ($POD_ID)\" >&2
$BOX_PATH; cd /root/mojolearn-$lane && bash -c $(sq "$user_cmd")"
    BOX_TIMEOUT=$(( (wait_min + 240) * 60 )) box "$remote"
    ;;
sh)
    lane="${1:-}"; lane_ok "$lane"; shift; [ $# -gt 0 ] || die "sh needs a command"
    need_lane_pod "$lane"
    # every GPU hidden (an invalid index shows no device); `hold` keeps the pod's lease busy while it runs
    inner="mkdir -p /root/mojolearn-$lane; export CUDA_VISIBLE_DEVICES=-1; $BOX_PATH; cd /root/mojolearn-$lane && bash -c $(sq "$*")"
    BOX_TIMEOUT=14400 box "$NVQ hold $(sq "$inner")"
    ;;
fetch)
    lane="${1:-}"; lane_ok "$lane"; p="${2:?fetch <lane> <path> <local dir>}"; out="${3:?local dir}"
    need_lane_pod "$lane"; mkdir -p "$out"
    case "$p" in /*) dir=$(dirname "$p"); base=$(basename "$p") ;; *) dir=/root/mojolearn-$lane/$(dirname "$p"); base=$(basename "$p") ;; esac
    box "tar -C $(sq "$dir") -czf - $(sq "$base")" < /dev/null | tar -C "$out" -xzf - || die "fetch of $p failed"
    say "fetched $p -> $out/$base"
    ;;
down)
    [ "$NO_API" = 1 ] || load_key || die "no RunPod key"
    force=0; keys=''
    for a in "$@"; do case "$a" in --force) force=1 ;; --all) keys=$(live_pods) ;; nvc[0-9]) keys="$keys $a" ;; *) die "down <pod>|--all [--force]" ;; esac; done
    [ -n "$keys" ] || die "down <pod>|--all [--force]"
    for k in $keys; do
        pod_env $k
        busy=$(BOX_TIMEOUT=60 box "cat $QD/pending 2>/dev/null; grep -lE '^(starting|running)\$' $QD/[0-9]*/status 2>/dev/null | wc -l" < /dev/null | tr '\n' ' ' || true)
        set -- $busy
        if [ "${1:-0}" != 0 ] || [ "${2:-0}" != 0 ]; then
            [ $force = 1 ] || die "$k has ${1:-0} queued and ${2:-0} running job(s); it goes down by itself $NVC_IDLE_MIN min after they finish (or pass --force)"
        fi
        MOJOLEARN_NVC_DOWN=1 "$DEVPOD" down $k
        for f in "$NVC_HOME"/lanes/*/pod; do [ -f "$f" ] && [ "$(cat "$f")" = $k ] && rm -f "$f"; done
    done
    ;;
*) sed -n 2,33p "$0"; exit 2 ;;
esac
