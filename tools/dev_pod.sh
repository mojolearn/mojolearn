#!/bin/bash
# tools/dev_pod.sh -- ONE long-lived NVIDIA development pod per lane, for the
# algorithm expansion (docs/lanes/ALGORITHM_EXPANSION_PLAN.md). A lane's agent
# edits in its Mac worktree, then syncs, builds, verifies and times on its pod.
#
#   tools/dev_pod.sh up     <lane> [minutes]   RENTS. Dead-man, create (retried while RunPod
#                                              is out of stock, MOJOLEARN_DEVPOD_RETRY_MINUTES,
#                                              default 60), ssh, lease, pixi.
#   tools/dev_pod.sh sync   <lane> <worktree>  tracked + modified files -> /root/mojolearn
#   tools/dev_pod.sh run    <lane> <command>   runs in /root/mojolearn on the pod
#   tools/dev_pod.sh extend <lane> [minutes]   the heartbeat: on-pod lease AND Mac dead-man
#   tools/dev_pod.sh down   <lane>             delete, verify gone, disarm the Mac dead-man
#   tools/dev_pod.sh list                      every lane's pod, from the state files
#
# The safety path is tools/runpod_pod_lib.sh + tools/runpod_guard.sh, the same
# as tools/release_wheel_smoke.sh: the Mac dead-man is armed BEFORE the create,
# the on-pod watchdog is armed before any work, and `down` must see the pod
# gone from the API. A pod whose lane stops calling `extend` ends by itself.
# A lane's worktree is its own, so the sync is a tar of that worktree's
# tracked and modified files (never the shared checkout; never a stash).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATE_ROOT="${MOJOLEARN_DEVPOD_STATE:-$HOME/mojolearn-evidence/devpods}"
# Comma-separated; RunPod places the pod on whichever of these has stock.
# Identity needs any NVIDIA; speed is judged before/after on the SAME pod.
GPUS="${MOJOLEARN_DEVPOD_GPUS:-NVIDIA GeForce RTX 4090,NVIDIA L40S,NVIDIA RTX 6000 Ada Generation,NVIDIA RTX A6000,NVIDIA A40,NVIDIA H100 PCIe,NVIDIA H100 80GB HBM3}"
IMAGE="${MOJOLEARN_DEVPOD_IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
DISK_GB="${MOJOLEARN_DEVPOD_DISK_GB:-80}"
READY_TIMEOUT=900
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o BatchMode=yes"

die() { echo "dev_pod: $*" >&2; exit 1; }
say() { echo "dev_pod: $*"; }
now() { date +%s; }

TMPD="$(mktemp -d "${TMPDIR:-/tmp}/devpod.XXXXXX")"; trap 'rm -rf "$TMPD"' EXIT
CURLRC="$TMPD/curlrc"
. "$ROOT/tools/runpod_pod_lib.sh"

cmd="${1:-}"; lane="${2:-}"
case "$cmd" in list) ;; up|sync|run|extend|down) [[ "$lane" =~ ^[a-z0-9-]{1,24}$ ]] || die "lane must be [a-z0-9-]{1,24}" ;; *) sed -n 2,20p "$0"; exit 2 ;; esac
D="$STATE_ROOT/$lane"
load_state() { [ -f "$D/state.env" ] || die "no pod for lane $lane (run: $0 up $lane)"; . "$D/state.env"; }
bx() { with_timeout "$1" ssh $SSH_OPTS $SSH_TARGET "$2"; }

arm_mac_deadman() {  # seconds; replaces any earlier dead-man for this lane
    if [ -s "$D/deadman.pid" ]; then
        _p=$(cat "$D/deadman.pid"); pkill -P "$_p" 2>/dev/null || true; kill "$_p" 2>/dev/null || true
    fi
    write_deadman "$D/deadman" "$1" || die "the dead-man did not compose"
    [ -z "${POD_ID:-}" ] || printf '%s\n' "$POD_ID" > "$D/deadman/pod_id.txt"
    nohup sh -c 'trap "" HUP INT; exec sh "$0"' "$D/deadman/deadman.sh" > /dev/null 2>&1 < /dev/null &
    echo $! > "$D/deadman.pid"
    sleep 1; kill -0 "$(cat "$D/deadman.pid")" 2>/dev/null || die "the Mac dead-man did not start"
}

case "$cmd" in
up)
    minutes="${3:-240}"
    [ ! -f "$D/state.env" ] || die "lane $lane already has a pod ($D/state.env); down it first"
    mkdir -p "$D"
    load_key || die "no RunPod key (~/.mojolearn_runpod_key, mode 600)"
    POD_NAME="mojolearn-dev-$lane-$(date -u +%m%d%H%M)"
    rp_call GET "$RP/pods"; case "$RP_CODE" in 2*) ;; *) die "pod listing HTTP $RP_CODE" ;; esac
    python3 - "$TMPD/create.json" "$POD_NAME" "$IMAGE" "$GPUS" "$DISK_GB" <<'PY'
import json, sys
out, name, image, gpus, disk = sys.argv[1:]
json.dump({"name": name, "imageName": image, "gpuTypeIds": [g.strip() for g in gpus.split(",") if g.strip()], "gpuCount": 1,
           "cloudType": "SECURE", "containerDiskInGb": int(disk), "volumeInGb": 0,
           "ports": ["22/tcp"], "supportPublicIp": True, "interruptible": False},
          open(out, "w"), indent=2)
PY
    # OUT OF STOCK IS RETRIED, NOTHING ELSE IS (2026-09-27, FINAL DECISIONS in
    # docs/lanes/ALGORITHM_EXPANSION_PLAN.md). Each attempt arms its own
    # dead-man before the create and disarms it when nothing was created, as
    # the failed-create path always did; a create that fails for any reason but
    # stock dies at once. state.env is written only after ssh answers, so its
    # appearing is the lane's report that it has a pod. Every attempt is
    # logged to $STATE_ROOT/<lane>.attempts.log.
    retry_until=$(( $(now) + ${MOJOLEARN_DEVPOD_RETRY_MINUTES:-60} * 60 ))
    attempts_log="$STATE_ROOT/$lane.attempts.log"
    attempt=0
    while :; do
        attempt=$(( attempt + 1 ))
        POD_ID=""
        arm_mac_deadman $(( READY_TIMEOUT + minutes * 60 + 600 ))
        say "creating $POD_NAME (any of: $GPUS), attempt $attempt. THE BILL STARTS HERE."
        rp_call POST "$RP/pods" "$TMPD/create.json"
        cp "$TMPD/rp.body" "$D/create_response.json"
        POD_ID=$(rp_py id)
        [ -z "$POD_ID" ] || break
        _body=$(head -c 300 "$TMPD/rp.body"); rp_call GET "$RP/pods"
        POD_ID=$(rp_py byname "$POD_NAME" | awk '{print $1}')
        if [ -n "$POD_ID" ]; then
            echo "$POD_ID" > "$D/deadman/pod_id.txt"; delete_pod "$POD_ID"
            die "create response unparsed but $POD_ID exists by name; deleted, dead-man stays armed"
        fi
        _p=$(cat "$D/deadman.pid"); pkill -P "$_p" 2>/dev/null || true; kill "$_p" 2>/dev/null || true
        rm -rf "$D/deadman" "$D/deadman.pid"
        printf '%s attempt %s: nothing created: %s\n' "$(date -u +%FT%TZ)" "$attempt" "$_body" >> "$attempts_log"
        if ! printf '%s' "$_body" | grep -qiE 'no (gpu )?instances|currently available|out of stock|insufficient|not enough|no available|unavailable'; then
            rm -rf "$D"
            die "create failed, nothing was created, and not for stock: $_body"
        fi
        if [ "$(now)" -ge "$retry_until" ]; then
            rm -rf "$D"
            die "out of stock for ${MOJOLEARN_DEVPOD_RETRY_MINUTES:-60} min ($attempt attempts, $attempts_log); nothing was created"
        fi
        _wait=$(( 60 + attempt * 15 )); [ "$_wait" -le 120 ] || _wait=120
        say "out of stock (attempt $attempt); retrying in ${_wait}s (log: $attempts_log)"
        sleep "$_wait"
    done
    printf '%s attempt %s: created %s\n' "$(date -u +%FT%TZ)" "$attempt" "$POD_ID" >> "$attempts_log"
    echo "$POD_ID" > "$D/deadman/pod_id.txt"
    COST_HR=$(rp_py cost)
    GPU=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); m=d.get("machine") or {}; print(m.get("gpuTypeId") or (d.get("gpu") or {}).get("id") or "?")' "$D/create_response.json" 2>/dev/null || echo "?")
    _deadline=$(( $(now) + READY_TIMEOUT )); SSH_TARGET=""
    while [ "$(now)" -lt "$_deadline" ]; do
        rp_call GET "$RP/pods/$POD_ID"; SSH_TARGET=$(rp_py ssh)
        if [ -n "$SSH_TARGET" ] && with_timeout 40 ssh $SSH_OPTS $SSH_TARGET 'echo SSH-OK' 2>/dev/null | grep -q SSH-OK; then break; fi
        SSH_TARGET=""; sleep 10
    done
    [ -n "$SSH_TARGET" ] || { delete_pod "$POD_ID"; die "no ssh after ${READY_TIMEOUT}s; deleted"; }
    printf 'POD_ID=%q\nPOD_NAME=%q\nSSH_TARGET=%q\nCOST_HR=%q\nGPU=%q\n' "$POD_ID" "$POD_NAME" "$SSH_TARGET" "$COST_HR" "$GPU" > "$D/state.env"
    MOJOLEARN_LEASE_DIR="$D/lease" with_timeout 180 sh "$ROOT/tools/runpod_guard.sh" arm "$POD_ID" "$SSH_TARGET" "$minutes" > "$D/arm.log" 2>&1 \
        || { cat "$D/arm.log"; delete_pod "$POD_ID"; die "on-pod lease REFUSED; deleted"; }
    say "pod $POD_ID up (\$${COST_HR:-?}/hr), lease ${minutes} min; installing pixi"
    bx 600 'command -v pixi >/dev/null || curl -fsSL https://pixi.sh/install.sh | bash; mkdir -p /root/mojolearn' > "$D/bootstrap.log" 2>&1 \
        || say "pixi bootstrap failed; see $D/bootstrap.log"
    say "ready: $0 sync $lane <worktree>"
    ;;
sync)
    load_state; wt="${3:?worktree}"; [ -d "$wt/.git" ] || [ -f "$wt/.git" ] || die "$wt is not a git worktree"
    [ "$(cd "$wt" && git rev-parse --show-toplevel)" != "$(cd "$ROOT" && git rev-parse --show-toplevel)" ] || [ "${MOJOLEARN_DEVPOD_ALLOW_SELF:-0}" = 1 ] \
        || die "sync a lane's OWN worktree, not the checkout this tool runs from"
    ( cd "$wt" && git ls-files -z -c -o --exclude-standard | grep -zvE '\.(so|dylib|metallib)$' \
        | COPYFILE_DISABLE=1 tar --null -czf - -T - ) | bx 900 'cd /root/mojolearn && tar xzf -' || die "sync failed"
    say "synced $(cd "$wt" && git rev-parse --short HEAD)+worktree -> $POD_ID:/root/mojolearn"
    ;;
run)
    load_state; shift 2; [ $# -gt 0 ] || die "run needs a command"
    ssh $SSH_OPTS $SSH_TARGET "export PATH=/root/.pixi/bin:\$PATH; cd /root/mojolearn && $*"
    ;;
extend)
    load_state; minutes="${3:-120}"; load_key || die "no RunPod key"
    MOJOLEARN_LEASE_DIR="$D/lease" with_timeout 180 sh "$ROOT/tools/runpod_guard.sh" extend "$POD_ID" "$SSH_TARGET" "$minutes" \
        || die "extend REFUSED; the pod keeps its old lease"
    arm_mac_deadman $(( minutes * 60 + 600 ))
    say "extended $POD_ID by ${minutes} min (on-pod lease and Mac dead-man)"
    ;;
down)
    load_state; load_key || die "no RunPod key"
    delete_pod "$POD_ID"
    verify_gone "$POD_ID" || die "$POD_ID NOT CONFIRMED GONE; the Mac dead-man stays armed"
    _p=$(cat "$D/deadman.pid" 2>/dev/null || true)
    [ -z "$_p" ] || { pkill -P "$_p" 2>/dev/null || true; kill "$_p" 2>/dev/null || true; }
    mv "$D" "$D.down-$(date -u +%Y%m%dT%H%M%SZ)"
    say "lane $lane pod $POD_ID down"
    ;;
list)
    for s in "$STATE_ROOT"/*/state.env; do
        [ -f "$s" ] || continue
        ( . "$s"; printf '%-12s %-16s %-24s $%s/hr\n' "$(basename "$(dirname "$s")")" "$POD_ID" "$GPU" "${COST_HR:-?}" )
    done
    ;;
esac
