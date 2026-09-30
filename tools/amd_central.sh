#!/bin/bash
# tools/amd_central.sh -- THE CENTRAL AMD BOX (2026-09-28): one shared
# Hot Aisle AMD VM, kept up for 24 hours, that every lane submits to.
#
# ONE Hot Aisle MI300X (gfx942) box that every lane shares for AMD identity
# checks and AMD speed work. No lane rents its own AMD box any more.
#
#   tools/amd_central.sh sync   <lane> <worktree>        patch-sync the worktree to /root/mojolearn-<lane>
#   tools/amd_central.sh run    <lane> [--gpus N] [--wait MIN] <command>
#                                                         one job on N GPU slots (default 1), in
#                                                         /root/mojolearn-<lane>, as root; waits for a free
#                                                         slot (default 120 min, exit 75 if none came)
#   tools/amd_central.sh sh     <lane> <command>          NO GPU (every GPU hidden): tail logs, ls, git
#   tools/amd_central.sh fetch  <lane> <path> <local dir> tar a box path (relative to the lane tree, or
#                                                         absolute) back into <local dir>
#   tools/amd_central.sh submit <lane> [--gpus N] [--cap MIN] [--note TEXT] <script-on-box>
#                                                         THE WAY TO RUN A GATE: enqueue the script in the box's
#                                                         FIFO queue; prints the job id. Absolute path, or relative
#                                                         to /root/mojolearn-<lane>. Default cap 720 min.
#   tools/amd_central.sh queue                            every queued/running job + the last finished ones
#   tools/amd_central.sh status [<id>]                    no id: box, lease, slots, queue; id: that job
#   tools/amd_central.sh cancel <lane> <id>               cancel the lane's OWN job (queued or running)
#   tools/amd_central.sh log    <id> [N]                  last N (default 40) lines of the job's log
#   tools/amd_central.sh queue-install                    (once per box) install/upgrade the queue service
#   tools/amd_central.sh keep                             (the allocator loop only) pin the lease to LEASE_END
#
# WHICH BOX. ~/mojolearn-evidence/amd_central.env (outside the repo) names the
# dev_pod state key of the central box (CENTRAL_KEY), its slot count (SLOTS)
# and the lease end (LEASE_END, epoch). The box itself is an ordinary
# tools/dev_pod.sh Hot Aisle box: the same Mac dead-man and on-box watchdog.
# dev_pod.sh refuses to shorten its lease or take it down (see there).
#
# SLOTS. A slot is one GPU. `run` takes N free GPUs with flock(1) on the box
# (/var/lock/mojolearn-central/gpu<g>.lock), so the lock is correct across
# every Mac process and agent, and it is released by the kernel when the job
# exits however it exits. The job sees ONLY its GPUs (ROCR_VISIBLE_DEVICES=
# <its GPUs>, HIP devices 0..N-1). A job started in the background from `run`
# (`setsid nohup ... &`) inherits the lock and keeps its slot until it exits.
#
# QUEUE. tools/gpu_queue_box.sh (the same queue as the shared NVIDIA pods of
# tools/nvidia_central.sh; its defaults are this box's) is installed on the box
# as /root/amd-queue/bin/amdq and runs as the systemd service amd-queue.service
# (survives agent sessions and reboots). `submit` enqueues; the service starts jobs in submission order as
# their GPU slots free (the same flock slots as `run`), writes
# /root/amd-queue/<id>/{status,log,exit}, and kills a job at its wall-clock cap.
# While any job is queued, `run` does not take a free slot (it waits, or exits
# 75 at --wait 0), so the queue keeps its order. Prefer `submit` over a waiting
# `run`: a `run` waiter dies with the agent session and holds no place in line.
#
# BUILDS. Each lane has its own tree and its own pixi env (and so its own
# warm Mojo cache: AMD codegen is only stable from a warm cache, so a lane's
# cache is never wiped or shared with another lane's source). The conda
# package cache (/root/.cache/rattler) IS shared, so a lane's first
# `pixi install` links packages instead of downloading them.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONF="${MOJOLEARN_AMD_CENTRAL_CONF:-$HOME/mojolearn-evidence/amd_central.env}"
STATE_ROOT="${MOJOLEARN_DEVPOD_STATE:-$HOME/mojolearn-evidence/devpods}"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=8 -o BatchMode=yes"
LOCKD=/var/lock/mojolearn-central
BOX_PATH='export PATH=/root/.pixi/bin:/opt/rocm/bin:$PATH'

die() { echo "amd_central: $*" >&2; exit 1; }
say() { echo "amd_central: $*"; }
sq() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }

[ -f "$CONF" ] || die "no central AMD box is configured ($CONF missing)"
. "$CONF"
: "${CENTRAL_KEY:?}" "${SLOTS:?}" "${LEASE_END:?}"
ST="$STATE_ROOT/$CENTRAL_KEY/state.env"
[ -f "$ST" ] || die "the central box's state $ST is gone (lease ended or torn down)"
SSH_TARGET=$(. "$ST"; printf '%s' "$SSH_TARGET")
POD_ID=$(. "$ST"; printf '%s' "$POD_ID")
GPU=$(. "$ST"; printf '%s' "$GPU")
# shellcheck disable=SC2086
box() { ssh $SSH_OPTS $SSH_TARGET "sudo -n -H bash -c $(sq "$1")"; }

lane_ok() { [[ "${1:-}" =~ ^[a-z0-9-]{1,32}$ ]] || die "lane must be [a-z0-9-]{1,32}"; }
cmd="${1:-}"; shift || true
case "$cmd" in
sync)
    lane="${1:-}"; lane_ok "$lane"; wt="${2:?sync <lane> <worktree>}"
    # dev_pod.sh's patch sync, pointed at /root/mojolearn-<lane> on the central box
    LS="$HOME/mojolearn-evidence/amd_central/lanes"; mkdir -p "$LS/$lane-amd"
    grep -v '^BOX_DIR=\|^BOX_ENV=\|^PROVIDER=\|^HA_' "$ST" > "$LS/$lane-amd/state.env"
    printf 'PROVIDER=central\nBOX_DIR=/root/mojolearn-%s\nBOX_ENV=\n' "$lane" >> "$LS/$lane-amd/state.env"
    MOJOLEARN_DEVPOD_STATE="$LS" MOJOLEARN_DEVPOD_ALLOW_SELF=1 "$ROOT/tools/dev_pod.sh" sync "$lane-amd" "$wt"
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
    [[ "$n" =~ ^[1-9][0-9]*$ ]] && [ "$n" -le "$SLOTS" ] || die "--gpus must be 1..$SLOTS"
    user_cmd="$*"
    remote="set -u; mkdir -p $LOCKD /root/mojolearn-$lane; N=$n; T=$SLOTS; deadline=\$(( \$(date +%s) + $wait_min * 60 )); said=0
while :; do
    got=''; fds=''
    pend=\$(cat /root/amd-queue/pending 2>/dev/null || echo 0)
    [ \"\$pend\" -gt 0 ] 2>/dev/null && T0=0 || T0=\$T
    for g in \$(seq 0 \$((T0 - 1))); do
        exec {fd}>$LOCKD/gpu\$g.lock
        if flock -n \$fd; then got=\"\$got \$g\"; fds=\"\$fds \$fd\"; [ \$(echo \$got | wc -w) -ge \$N ] && break
        else exec {fd}>&-; fi
    done
    [ \$(echo \$got | wc -w) -ge \$N ] && break
    for f in \$fds; do exec {f}>&-; done
    [ \$(date +%s) -lt \$deadline ] || { echo 'amd_central: no free GPU slot within $wait_min min' >&2; exit 75; }
    [ \$said = 1 ] || { echo 'amd_central: waiting for '\$N' free GPU slot(s) (queued jobs go first; use submit to hold a place in line)...' >&2; said=1; }
    sleep 5
done
for g in \$got; do printf 'lane=%s pid=%s utc=%s cmd=%s\n' $(sq "$lane") \$\$ \$(date -u +%FT%TZ) $(sq "${user_cmd:0:160}") > $LOCKD/gpu\$g.owner; done
export ROCR_VISIBLE_DEVICES=\$(echo \$got | tr ' ' ,) HIP_VISIBLE_DEVICES=\$(seq -s, 0 \$((N - 1)))
echo \"amd_central: $lane on GPU(s) \$ROCR_VISIBLE_DEVICES of $POD_ID\" >&2
$BOX_PATH; cd /root/mojolearn-$lane && bash -c $(sq "$user_cmd")"
    box "$remote"
    ;;
sh)
    lane="${1:-}"; lane_ok "$lane"; shift; [ $# -gt 0 ] || die "sh needs a command"
    # every GPU hidden: ROCr ignores an index that does not exist and shows no agent
    box "mkdir -p /root/mojolearn-$lane; export ROCR_VISIBLE_DEVICES=999 HIP_VISIBLE_DEVICES=; $BOX_PATH; cd /root/mojolearn-$lane && bash -c $(sq "$*")"
    ;;
fetch)
    lane="${1:-}"; lane_ok "$lane"; p="${2:?fetch <lane> <path> <local dir>}"; out="${3:?local dir}"
    mkdir -p "$out"
    case "$p" in /*) dir=$(dirname "$p"); base=$(basename "$p") ;; *) dir=/root/mojolearn-$lane/$(dirname "$p"); base=$(basename "$p") ;; esac
    box "tar -C $(sq "$dir") -czf - $(sq "$base")" < /dev/null | tar -C "$out" -xzf - || die "fetch of $p failed"
    say "fetched $p -> $out/$base"
    ;;
submit)
    lane="${1:-}"; lane_ok "$lane"; shift; n=1; cap=720; note=''
    while [ $# -gt 0 ]; do
        case "$1" in
            --gpus) n="${2:?--gpus N}"; shift 2 ;;
            --cap) cap="${2:?--cap MIN}"; shift 2 ;;
            --note) note="${2:?--note TEXT}"; shift 2 ;;
            *) break ;;
        esac
    done
    [ $# -eq 1 ] || die "submit <lane> [--gpus N] [--cap MIN] [--note TEXT] <script-on-box> (one script path; put arguments inside the script)"
    box "/root/amd-queue/bin/amdq submit $(sq "$lane") $(sq "$n") $(sq "$cap") $(sq "$1") $(sq "$note")" < /dev/null
    ;;
queue)
    box "/root/amd-queue/bin/amdq queue" < /dev/null
    ;;
cancel)
    lane="${1:-}"; lane_ok "$lane"; id="${2:?cancel <lane> <id>}"
    box "/root/amd-queue/bin/amdq cancel $(sq "$lane") $(sq "$id")" < /dev/null
    ;;
log)
    id="${1:?log <id> [N]}"; [[ "$id" =~ ^[0-9]{4,}$ ]] || die "job id is digits"; nl="${2:-40}"
    [[ "$nl" =~ ^[0-9]+$ ]] || die "N is a line count"
    box "tail -n $nl /root/amd-queue/$id/log" < /dev/null
    ;;
queue-install)
    box "mkdir -p /root/amd-queue/bin && cat > /root/amd-queue/bin/amdq.new && chmod 755 /root/amd-queue/bin/amdq.new && mv -f /root/amd-queue/bin/amdq.new /root/amd-queue/bin/amdq && echo $SLOTS > /root/amd-queue/slots
cat > /etc/systemd/system/amd-queue.service <<'UNIT'
[Unit]
Description=mojolearn central AMD FIFO job queue (tools/gpu_queue_box.sh)
After=network.target

[Service]
ExecStart=/root/amd-queue/bin/amdq daemon
Restart=always
RestartSec=5
KillMode=process
StandardOutput=append:/root/amd-queue/dispatcher.log
StandardError=append:/root/amd-queue/dispatcher.log

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload && systemctl enable amd-queue.service >/dev/null 2>&1 && systemctl restart amd-queue.service && sleep 2 && systemctl is-active amd-queue.service" < "$ROOT/tools/gpu_queue_box.sh"
    ;;
status)
    if [ -n "${1:-}" ]; then
        [[ "$1" =~ ^[0-9]{4,}$ ]] || die "status <id>: job id is digits"
        box "/root/amd-queue/bin/amdq status $(sq "$1")" < /dev/null; exit
    fi
    echo "central AMD box: $CENTRAL_KEY $POD_ID ($GPU), $SLOTS GPU slots"
    echo "lease end: $(date -u -r "$LEASE_END" +%FT%TZ 2>/dev/null || date -u -d "@$LEASE_END" +%FT%TZ)"
    box "mkdir -p $LOCKD; for g in \$(seq 0 $((SLOTS - 1))); do
  if flock -n $LOCKD/gpu\$g.lock true 2>/dev/null; then echo \"slot \$g: free\"; else echo \"slot \$g: BUSY \$(cat $LOCKD/gpu\$g.owner 2>/dev/null)\"; fi; done
tail -1 /root/.mojolearn-devpod-guard/watchdog.out 2>/dev/null; /opt/rocm/bin/rocm-smi --showuse 2>/dev/null | grep 'GPU\\['
echo; [ -x /root/amd-queue/bin/amdq ] && /root/amd-queue/bin/amdq queue || echo 'queue: not installed (tools/amd_central.sh queue-install)'" < /dev/null
    ;;
keep)
    left=$(( (LEASE_END - $(date +%s)) / 60 ))
    [ "$left" -gt 5 ] || die "the central lease has ended (LEASE_END passed); nothing extended"
    "$ROOT/tools/dev_pod.sh" extend "$CENTRAL_KEY" "$left"
    ;;
*) sed -n 2,28p "$0"; exit 2 ;;
esac
