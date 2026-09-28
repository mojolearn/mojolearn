#!/bin/bash
# tools/dev_pod.sh -- ONE long-lived development box per lane and vendor, for
# the algorithm expansion (docs/lanes/ALGORITHM_EXPANSION_PLAN.md). A lane's
# agent edits in its Mac worktree, then syncs, builds, verifies and times on
# its box. A lane may hold an NVIDIA box AND an AMD box at once.
#
#   tools/dev_pod.sh up     <lane> [minutes] [--vendor nvidia|amd] [--base <commit>]
#                                              RENTS. Dead-man, create (retried while out of
#                                              stock), ssh, lease, pixi, git tree seeded.
#   tools/dev_pod.sh sync   <lane> [--vendor amd] <worktree>
#                                              brings the box's git HEAD to the worktree's merge
#                                              base with origin/main, then tracked + modified
#                                              files -> /root/mojolearn
#   tools/dev_pod.sh run    <lane> [--vendor amd] <command>   in /root/mojolearn, as root
#   tools/dev_pod.sh extend <lane> [--vendor amd] [minutes]   the heartbeat: on-box lease AND Mac dead-man
#   tools/dev_pod.sh down   <lane> [--vendor amd]             delete, verify gone, disarm the Mac dead-man
#   tools/dev_pod.sh list                                     every box, from the state files
#   tools/dev_pod.sh host up [minutes] | extend [minutes] | status | down [--force]
#                                              THE SHARED AMD HOST (below)
#
# THE SHARED AMD HOST (2026-09-27). `host up` rents ONE Hot Aisle bare-metal
# 8x MI300X server (tools/hotaisle_vm_lib.sh with HA_RES=bare_metal: the same
# Mac dead-man and on-box watchdog as a VM; 8-hour minimum billed up front),
# state key `amdhost`. While it is up, `up <lane> --vendor amd` takes a free
# GPU SLOT on it instead of renting a VM: the lane's state key is still
# `<lane>-amd`, but its tree on the host is /root/mojolearn-<lane> (never
# /root/mojolearn), and every `run` exports ROCR_VISIBLE_DEVICES=<slot>
# HIP_VISIBLE_DEVICES=0 (ROCr hides every other GPU from the process, so the
# one GPU it sees is HIP device 0; a lane never touches another's GPU).
# `sync` targets that dir; `extend` from ANY lane on the host renews the HOST
# lease (serialized by a Mac lock; the re-armed watchdog kills every older
# one); `down <lane>` frees the slot only. The host is torn down by
# `host down`, which refuses while slots are held unless --force.
# Slots are $STATE_ROOT/amdhost/slots/<n> directories (mkdir is the lock).
#
# THE STATE KEY. An NVIDIA box is keyed `<lane>`, an AMD box `<lane>-amd`
# (both under $MOJOLEARN_DEVPOD_STATE). `--vendor amd` right after the lane, or
# the key itself (`run linear-amd ...`), names the AMD box.
#
# NVIDIA: a RunPod pod (MOJOLEARN_DEVPOD_GPUS; out of stock is retried for
# MOJOLEARN_DEVPOD_RETRY_MINUTES, default 60). ONLY tools/nvidia_central.sh
# rents one now (the shared pods nvc1..nvc3, MOJOLEARN_DEVPOD_GPU_COUNT GPUs
# each); `up` for NVIDIA refuses any other caller (2026-09-28).
# AMD: a RunPod AMD Instinct MI300X first (rocm/dev-ubuntu-22.04:6.4.1-complete
# with tools/runpod_ssh_bootstrap.sh as its dockerStartCmd, exactly as
# tools/release_wheel_smoke.sh --vendor hip), out of stock retried for
# MOJOLEARN_DEVPOD_AMD_RETRY_MINUTES (default 15); then a Hot Aisle MI300X VM
# through tools/hotaisle_vm_lib.sh (its own Mac dead-man, on-box watchdog that
# DELETEs the VM at the lease, described mojolearn:devpod-<key>:<utc>; cap
# MOJOLEARN_DEVPOD_HA_CAP_USD, default 150, for the whole lease; the team
# balance tops up by itself). On a 2x MI300X VM every command is pinned to
# GPU 0. On Hot Aisle the login is `hotaisle`; every box command runs as root
# through `sudo -n -H bash -c`, so /root/mojolearn is the tree on every box.
# DigitalOcean is never used here.
#
# AMD BITS ARE NOT REPRODUCIBLE FROM A COLD CACHE. Mojo's gfx942 codegen varies
# run to run when its cache is cold (register numbers, swapped instructions;
# NVIDIA and CPU do not); a warm cache replays the first compile, and the R2
# binding cache (tools/bincache.py) is what freezes a RELEASED AMD binary. So
# on an AMD box: never wipe .pixi/envs/default/share/max/cache/.mojo_cache,
# build each binding once and let every later build of the same source
# (the lane check's restored arm, a re-run) replay it from that warm cache,
# and compare NUMBERS (tools/algos_lane_check.sh: CPU == AMD, AGREE), never
# .so digests. A lane check on a fresh AMD box is a cold build: its verdict is
# about the numbers, and the AMD bytes that ship come from the release's cache.
#
# The safety path is tools/runpod_pod_lib.sh + tools/runpod_guard.sh (RunPod)
# or tools/hotaisle_vm_lib.sh (Hot Aisle): the Mac dead-man is armed BEFORE the
# create, the on-box watchdog is armed before any work, and `down` must see
# the box gone from the API. A box whose lane stops calling `extend` ends by
# itself. A lane's worktree is its own, so the sync is a tar of that
# worktree's tracked and modified files (never the shared checkout; never a
# stash). The box's /root/mojolearn is a git tree whose HEAD is the lane's
# base (fetched shallow from the public GitHub repo), so `git diff`,
# `git apply` and `git apply -R` on the box act against the same base as the
# laptop.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATE_ROOT="${MOJOLEARN_DEVPOD_STATE:-$HOME/mojolearn-evidence/devpods}"
# Comma-separated; RunPod places the pod on whichever of these has stock.
# Identity needs any NVIDIA; speed is judged before/after on the SAME pod.
# Andrew 2026-09-28: 13 H100s at $3.49/h emptied the account in four hours. Identity work
# needs no H100: the default list stops at the cheap cards, and an H100/H200/A100/B200 in
# MOJOLEARN_DEVPOD_GPUS is refused unless MOJOLEARN_DEVPOD_ALLOW_BIG_GPU=1 (Andrew's OK only).
NV_GPUS="${MOJOLEARN_DEVPOD_GPUS:-NVIDIA GeForce RTX 4090,NVIDIA L40S,NVIDIA RTX 6000 Ada Generation,NVIDIA RTX A6000,NVIDIA A40}"
case "$NV_GPUS" in *H100*|*H200*|*A100*|*B200*)
    [ "${MOJOLEARN_DEVPOD_ALLOW_BIG_GPU:-0}" = 1 ] || { echo "dev_pod: $NV_GPUS includes an H100/H200/A100/B200; refused without Andrew's OK (MOJOLEARN_DEVPOD_ALLOW_BIG_GPU=1)" >&2; exit 2; } ;;
esac
# At most this many live RunPod pods on the account, counted from the API at every up.
MAX_RUNPOD_PODS="${MOJOLEARN_DEVPOD_MAX_PODS:-3}"
# No lease longer than this many minutes; a lane that needs more extends, visibly.
MAX_LEASE_MIN="${MOJOLEARN_DEVPOD_MAX_LEASE_MIN:-240}"
# GPUs per NVIDIA pod: 1, except the shared pods tools/nvidia_central.sh brings up (4x/2x).
NV_GPU_COUNT="${MOJOLEARN_DEVPOD_GPU_COUNT:-1}"
[[ "$NV_GPU_COUNT" =~ ^[1-8]$ ]] || { echo "dev_pod: MOJOLEARN_DEVPOD_GPU_COUNT must be 1..8" >&2; exit 2; }
NV_IMAGE="${MOJOLEARN_DEVPOD_IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
AMD_GPUS="${MOJOLEARN_DEVPOD_AMD_GPUS:-AMD Instinct MI300X OAM}"
AMD_IMAGE="${MOJOLEARN_DEVPOD_AMD_IMAGE:-rocm/dev-ubuntu-22.04:6.4.1-complete}"
DISK_GB="${MOJOLEARN_DEVPOD_DISK_GB:-80}"
READY_TIMEOUT=900
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o BatchMode=yes"
REPO_URL="${MOJOLEARN_DEVPOD_REPO_URL:-https://github.com/mojolearn/mojolearn.git}"
BOX_PATH='export PATH=/root/.pixi/bin:/opt/rocm/bin:$PATH'
BOX_DIR=/root/mojolearn   # a slot on the shared AMD host: /root/mojolearn-<lane>
HOST_KEY=amdhost

die() { echo "dev_pod: $*" >&2; exit 1; }
say() { echo "dev_pod: $*"; }
now() { date +%s; }

TMPD="$(mktemp -d "${TMPDIR:-/tmp}/devpod.XXXXXX")"
STATE_WRITTEN=1          # set to 0 while an `up` owns a box that has no state file yet
PROVIDER=""
on_exit() {
    if [ "$STATE_WRITTEN" = 0 ] && [ "$PROVIDER" = hotaisle ]; then
        say "the Hot Aisle box never reached a state file; tearing it down"
        ha_teardown || true
    fi
    rm -rf "$TMPD"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
CURLRC="$TMPD/curlrc"
. "$ROOT/tools/runpod_pod_lib.sh"
# Hot Aisle: a dev box waits for stock and a slot (the release default is 0).
HA_STOCK_WAIT_MINUTES=${MOJOLEARN_HOTAISLE_STOCK_WAIT_MINUTES:-30}
HA_SLOT_WAIT_MINUTES=${MOJOLEARN_HOTAISLE_SLOT_WAIT_MINUTES:-10}
. "$ROOT/tools/hotaisle_vm_lib.sh"

cmd="${1:-}"; lane="${2:-}"; lane_arg=""; VENDOR=nvidia
case "$cmd" in
list) ;;
host)
    hcmd="${2:-}"; shift $(( $# < 2 ? $# : 2 ))
    case "$hcmd" in up|extend|status|down) ;; *) die "host up [minutes] | extend [minutes] | status | down [--force]" ;; esac
    VENDOR=amd; lane_arg=$HOST_KEY
    ;;
up|sync|run|extend|down)
    shift $(( $# < 2 ? $# : 2 ))
    case "$lane" in *-amd) VENDOR=amd ;; esac
    lane_arg=${lane%-amd}
    [[ "$lane_arg" =~ ^[a-z0-9-]{1,24}$ ]] || die "lane must be [a-z0-9-]{1,24}"
    if [ "$cmd" != up ] && [ "${1:-}" = --vendor ]; then VENDOR="${2:?--vendor nvidia|amd}"; shift 2; fi
    ;;
*) sed -n 2,18p "$0"; exit 2 ;;
esac
if [ "$cmd" = up ]; then
    minutes=240; BASE_REF=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --vendor) VENDOR="${2:?--vendor nvidia|amd}"; shift 2 ;;
            --base) BASE_REF="${2:?--base <commit>}"; shift 2 ;;
            [0-9]*) minutes="$1"; shift; [ "$minutes" -le "$MAX_LEASE_MIN" ] || { echo "dev_pod: lease $minutes min is over the $MAX_LEASE_MIN-min cap (MOJOLEARN_DEVPOD_MAX_LEASE_MIN)" >&2; exit 2; } ;;
            *) die "up: unknown argument $1" ;;
        esac
    done
fi
case "$VENDOR" in nvidia|amd) ;; *) die "--vendor must be nvidia or amd" ;; esac
KEY="$lane_arg"; [ "$VENDOR" = amd ] && KEY="$lane_arg-amd"
[ "$cmd" = host ] && KEY=$HOST_KEY
D="$STATE_ROOT/$KEY"
BOX_SUDO=0; BOX_ENV=""
# THE CENTRAL AMD BOX (tools/amd_central.sh, 2026-09-28): the box named by
# CENTRAL_KEY in ~/mojolearn-evidence/amd_central.env is shared by every lane.
# `extend` on it never shortens its lease below LEASE_END, `down` refuses
# (MOJOLEARN_CENTRAL_DOWN=1 overrides), and `run` on it takes GPU 0's slot lock.
CENTRAL_KEY=""; CENTRAL_LEASE_END=0
_cc="${MOJOLEARN_AMD_CENTRAL_CONF:-$HOME/mojolearn-evidence/amd_central.env}"
if [ -f "$_cc" ]; then
    CENTRAL_KEY=$(sed -n 's/^CENTRAL_KEY=//p' "$_cc" | head -1)
    CENTRAL_LEASE_END=$(sed -n 's/^LEASE_END=//p' "$_cc" | head -1); CENTRAL_LEASE_END=${CENTRAL_LEASE_END:-0}
fi
# THE SHARED NVIDIA PODS (tools/nvidia_central.sh, 2026-09-28): keys nvc1..nvc3.
# Their lease is the pod's job queue, so `extend` refuses, and `down`, `run` and
# `sync` refuse unless tools/nvidia_central.sh is the caller.
case "$KEY" in nvc[0-9])
    case "$cmd" in
    extend) die "$KEY is a shared NVIDIA pod: its job queue keeps its lease (tools/nvidia_central.sh); nothing extended" ;;
    down) [ "${MOJOLEARN_NVC_DOWN:-0}" = 1 ] || die "$KEY is a shared NVIDIA pod every lane uses; tools/nvidia_central.sh down $KEY (it refuses while jobs are queued)" ;;
    run|sync) die "$KEY is a shared NVIDIA pod: use tools/nvidia_central.sh sync/submit/run/sh" ;;
    esac ;;
esac
load_state() {
    [ -f "$D/state.env" ] || die "no box for $KEY (run: $0 up $lane_arg${VENDOR:+ --vendor $VENDOR})"
    . "$D/state.env"
    PROVIDER=${PROVIDER:-runpod}; BOX_SUDO=${BOX_SUDO:-0}; BOX_ENV=${BOX_ENV:-}; BOX_DIR=${BOX_DIR:-/root/mojolearn}
    HA_RES=${HA_RES:-virtual_machines}
}
box_cmd() {  # the command string ssh runs: as root on every provider
    if [ "$BOX_SUDO" = 1 ]; then ha_root_cmd "$1"; else printf '%s' "$1"; fi
}
# shellcheck disable=SC2086
bx() { with_timeout "$1" ssh $SSH_OPTS $SSH_TARGET "$(box_cmd "$2")"; }

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

# The Hot Aisle Mac dead-man, re-armed with a new deadline (extend). The old
# one is stopped only after the new one is alive.
ha_rearm_mac_deadman() {  # seconds from now
    _old=$(cat "$D/deadman.pid" 2>/dev/null || true); _olddir=${HA_DEADMAN_DIR:-}
    _new="$D/ha-deadman-$(now)"
    ha_write_deadman "$_new" $(( $(now) + $1 )) "$D/hotaisle.record" || die "the Hot Aisle dead-man did not compose"
    printf '%s\n' "$HA_VMREF" > "$_new/vm_ref.txt"; printf '%s\n' "$HA_DESC" > "$_new/desc.txt"
    nohup sh -c 'trap "" HUP INT; exec sh "$0"' "$_new/deadman.sh" > /dev/null 2>&1 < /dev/null &
    _pid=$!; sleep 1; kill -0 "$_pid" 2>/dev/null || die "the new Hot Aisle dead-man did not start; the old one stays armed"
    echo "$_pid" > "$D/deadman.pid"
    if [ -n "$_old" ] && [ "$_old" != "$_pid" ]; then pkill -P "$_old" 2>/dev/null || true; kill "$_old" 2>/dev/null || true; fi
    [ -z "$_olddir" ] || [ "$_olddir" = "$_new" ] || rm -rf "$_olddir"
    HA_DEADMAN_DIR=$_new; HA_DEADMAN_PID=$_pid
}

# The Hot Aisle on-box watchdog, re-armed: a NEW script file (a running sh
# reads its script as it goes, so the old file is never rewritten), started
# and seen alive from a second session, and only then is the old one killed.
ha_rearm_watchdog() {  # seconds from now
    _f="$TMPD/wd.sh"; _name="watchdog-$(now).sh"
    ha_write_watchdog "$_f" "$HA_GUARD" "$1" "$HA_VMREF" || die "the watchdog did not compose"
    _oldw=$(bx 60 "cat $HA_GUARD/watchdog.pid 2>/dev/null" < /dev/null | tr -dc 0-9 || true)
    bx 60 "umask 077; cat > $HA_GUARD/$_name && chmod 700 $HA_GUARD/$_name" < "$_f" || die "could not deliver the new watchdog"
    bx 60 "rm -f $HA_GUARD/watchdog.pid; if command -v setsid > /dev/null 2>&1; then setsid nohup sh $HA_GUARD/$_name > $HA_GUARD/watchdog.log 2>&1 < /dev/null &
else nohup sh $HA_GUARD/$_name > $HA_GUARD/watchdog.log 2>&1 < /dev/null & fi
i=0; while [ \$i -lt 15 ] && [ ! -s $HA_GUARD/watchdog.pid ]; do sleep 1; i=\$((i + 1)); done; cat $HA_GUARD/watchdog.pid" < /dev/null > "$TMPD/wd.pid" 2>&1 || true
    _neww=$(tr -dc 0-9 < "$TMPD/wd.pid")
    [ -n "$_neww" ] && bx 60 "kill -0 $_neww && echo ALIVE" < /dev/null | grep -q ALIVE \
        || die "the new on-box watchdog is not alive; the old one (pid ${_oldw:-?}) stays armed"
    # every older watchdog, not only the one in watchdog.pid: two extends that
    # raced (two lanes on the shared host) must not leave a stale timer running
    bx 60 "for p in \$(pgrep -f '^sh $HA_GUARD/watchdog' 2>/dev/null); do [ \"\$p\" = $_neww ] || kill \"\$p\" 2>/dev/null; done; true" < /dev/null || true
    say "on-box watchdog re-armed: pid $_neww fires in ${1}s (old pid ${_oldw:-none} stopped)"
}

# R6: the box's /root/mojolearn is a git tree whose HEAD is <sha>. HEAD and the
# index move to <sha> (fetched shallow when absent); the files are NOT touched,
# so a synced lane diff survives; `git status` on the box is the lane's diff.
seed_git() {  # <full sha>
    _out=$(bx 900 "set -e; command -v git > /dev/null || { export DEBIAN_FRONTEND=noninteractive; apt-get update -qq > /dev/null && apt-get install -y -qq git > /dev/null; }
mkdir -p $BOX_DIR && cd $BOX_DIR
[ -d .git ] || { git init -q . && git remote add origin $REPO_URL; }
git config --global --add safe.directory $BOX_DIR 2>/dev/null || true
grep -qx '.devpod_manifest' .git/info/exclude 2>/dev/null || printf '.devpod_manifest\n.devpod_manifest.prev\n' >> .git/info/exclude
if [ \"\$(git rev-parse -q --verify HEAD 2>/dev/null)\" != $1 ]; then
    git cat-file -e $1^{commit} 2>/dev/null || git fetch -q --depth=1 $REPO_URL $1
    git update-ref HEAD $1
    git read-tree HEAD
    git update-index -q --refresh > /dev/null || true
fi
echo HEAD=\$(git rev-parse HEAD)" < /dev/null 2>&1) || { printf '%s\n' "$_out" >&2; die "could not seed the box's git tree at $1"; }
    printf '%s\n' "$_out" | grep -qx "HEAD=$1" || { printf '%s\n' "$_out" >&2; die "the box's HEAD is not $1 after seeding"; }
}

# One RunPod create loop: out of stock is retried until $1 minutes pass, any
# other refusal dies. 0 with POD_ID set, 1 when stock never came (nothing
# was created, the per-attempt dead-man disarmed).
runpod_create() {  # <retry minutes> <gpus> <image> <amd 0|1>
    POD_NAME="mojolearn-dev-$KEY-$(date -u +%m%d%H%M)"
    rp_call GET "$RP/pods"; case "$RP_CODE" in 2*) ;; *) die "pod listing HTTP $RP_CODE" ;; esac
    python3 - "$TMPD/create.json" "$POD_NAME" "$3" "$2" "$DISK_GB" "$4" "$ROOT/tools/runpod_ssh_bootstrap.sh" "$( [ "$4" = 1 ] && echo 1 || echo "$NV_GPU_COUNT")" <<'PY'
import json, sys
from pathlib import Path
out, name, image, gpus, disk, amd, bootstrap, count = sys.argv[1:]
req = {"name": name, "imageName": image, "gpuTypeIds": [g.strip() for g in gpus.split(",") if g.strip()], "gpuCount": int(count),
       "cloudType": "SECURE", "containerDiskInGb": int(disk), "volumeInGb": 0,
       "ports": ["22/tcp"], "supportPublicIp": True, "interruptible": False}
if amd != "1":
    # THE PINNED MAX NEEDS AN NVIDIA DRIVER >= 580 (CUDA 13.0). On an older
    # host a GPU binding built without a named arch fails in the pass manager
    # and a loaded one refuses the driver (lane/algos-ann's H100 pod, 570.211,
    # 2026-09-27); RunPod filters hosts before provisioning on this field
    # (tools/kmeans_host_recording_nvidia_leg.sh, gemm_remote_leg.sh).
    req["allowedCudaVersions"] = ["13.0"]
if amd == "1" and image.startswith("rocm/"):
    # plain ROCm images have no ssh; the repo's bootstrap, as release_wheel_smoke.sh --vendor hip
    req["dockerEntrypoint"] = ["/bin/bash", "-lc"]
    req["dockerStartCmd"] = [Path(bootstrap).read_text()]
json.dump(req, open(out, "w"), indent=2)
PY
    # OUT OF STOCK IS RETRIED, NOTHING ELSE IS (2026-09-27, FINAL DECISIONS in
    # docs/lanes/ALGORITHM_EXPANSION_PLAN.md). Each attempt arms its own
    # dead-man before the create and disarms it when nothing was created; a
    # create that fails for any reason but stock dies at once. Every attempt is
    # logged to $STATE_ROOT/<key>.attempts.log.
    retry_until=$(( $(now) + $1 * 60 ))
    attempts_log="$STATE_ROOT/$KEY.attempts.log"
    attempt=0
    while :; do
        attempt=$(( attempt + 1 ))
        POD_ID=""
        arm_mac_deadman $(( READY_TIMEOUT + minutes * 60 + 600 ))
        say "creating $POD_NAME on RunPod (any of: $2), attempt $attempt. THE BILL STARTS HERE."
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
            say "RunPod out of stock for $1 min ($attempt attempts, $attempts_log); nothing was created"
            return 1
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
    if [ "$4" = 1 ]; then
        # the watchdog below needs curl; the ROCm image may lack it
        bx 600 'command -v curl > /dev/null || { export DEBIAN_FRONTEND=noninteractive; apt-get update -qq > /dev/null && apt-get install -y -qq curl ca-certificates > /dev/null; }; command -v curl' \
            < /dev/null > "$D/curl.log" 2>&1 || { cat "$D/curl.log"; delete_pod "$POD_ID"; die "no curl on the AMD pod for its lease; deleted"; }
    fi
    write_state
    MOJOLEARN_LEASE_DIR="$D/lease" with_timeout 180 sh "$ROOT/tools/runpod_guard.sh" arm "$POD_ID" "$SSH_TARGET" "$minutes" > "$D/arm.log" 2>&1 \
        || { cat "$D/arm.log"; delete_pod "$POD_ID"; rm -f "$D/state.env"; die "on-pod lease REFUSED; deleted"; }
    return 0
}

write_state() {
    {
        printf 'PROVIDER=%q\nVENDOR=%q\nPOD_ID=%q\nPOD_NAME=%q\nSSH_TARGET=%q\nCOST_HR=%q\nGPU=%q\nBOX_SUDO=%q\nBOX_ENV=%q\nBOX_DIR=%q\n' \
            "$PROVIDER" "$VENDOR" "$POD_ID" "$POD_NAME" "$SSH_TARGET" "$COST_HR" "$GPU" "$BOX_SUDO" "$BOX_ENV" "$BOX_DIR"
        [ "$PROVIDER" != runpod ] || [ "$VENDOR" != nvidia ] || printf 'GPU_COUNT=%q\n' "$NV_GPU_COUNT"
        [ "$PROVIDER" != amdhost ] || printf 'SLOT=%q\n' "$SLOT"
        [ -z "${HOST_SLOTS:-}" ] || printf 'HOST_SLOTS=%q\n' "$HOST_SLOTS"
        if [ "$PROVIDER" = hotaisle ]; then
            printf 'HA_VMREF=%q\nHA_VMNAME=%q\nHA_DESC=%q\nHA_SLOT=%q\nHA_NONCE=%q\nHA_GUARD=%q\nHA_DEADMAN_DIR=%q\nHA_T_CREATE=%q\nHA_PRICE=%q\nHA_MINRES=%q\nHA_SPEC_USED=%q\nHA_RES=%q\n' \
                "$HA_VMREF" "$HA_VMNAME" "$HA_DESC" "$HA_SLOT" "$HA_NONCE" "$HA_GUARD" "$HA_DEADMAN_DIR" "$HA_T_CREATE" "$HA_PRICE" "$HA_MINRES" "$HA_SPEC_USED" "$HA_RES"
        fi
    } > "$D/state.env"
}

# ---------------------------------------------------------------- the shared AMD host
HD="$STATE_ROOT/$HOST_KEY"
host_lock() {  # serializes host extends and slot takes on this Mac
    _t0=$(now)
    until mkdir "$HD/.lock" 2>/dev/null; do
        _m=$(stat -f %m "$HD/.lock" 2>/dev/null || stat -c %Y "$HD/.lock" 2>/dev/null || echo 0)
        [ $(( $(now) - _m )) -lt 900 ] || { rm -rf "$HD/.lock"; continue; }
        [ $(( $(now) - _t0 )) -lt 900 ] || die "the host lock $HD/.lock stayed held for 15 minutes"
        sleep 2
    done
    echo "$$" > "$HD/.lock/pid"
}
host_unlock() { rm -rf "$HD/.lock"; }
# take_slot <lane key>: prints the slot number; 1 when the host is full
take_slot() {
    . "$HD/state.env"
    for _n in $(seq 0 $(( ${HOST_SLOTS:-8} - 1 ))); do
        if mkdir "$HD/slots/$_n" 2>/dev/null; then
            { echo "key=$1"; echo "utc=$(date -u +%FT%TZ)"; } > "$HD/slots/$_n/owner"
            echo "$_n"; return 0
        fi
    done
    return 1
}
host_extend() {  # minutes; the host's watchdog and Mac dead-man, under the Mac lock
    ( D=$HD; load_state; ha_load_key || die "no Hot Aisle key ($HA_KEYFILE)"
      host_lock; trap 'host_unlock' EXIT
      ha_rearm_watchdog $(( $1 * 60 ))
      ha_rearm_mac_deadman $(( $1 * 60 + 600 ))
      write_state )
}

case "$cmd" in
host)
    case "$hcmd" in
    up)
        minutes=${1:-480}
        [ ! -f "$D/state.env" ] || die "the shared AMD host is already up ($D/state.env)"
        mkdir -p "$D/slots"
        BASE_SHA=$(git -C "$ROOT" rev-parse --verify "origin/main^{commit}")
        PROVIDER=hotaisle; STATE_WRITTEN=0; HA_RES=bare_metal; HA_SPEC_WANT=8gpu
        HA_READY_SECONDS=${MOJOLEARN_HOTAISLE_BM_READY_SECONDS:-5400}
        HA_STOCK_WAIT_MINUTES=${MOJOLEARN_HOTAISLE_BM_STOCK_WAIT_MINUTES:-0}
        _cap=$(( ${MOJOLEARN_DEVPOD_HA_BM_CAP_USD:-2000} * 100 ))
        HA_GUARD=/root/.mojolearn-devpod-guard
        ha_rent "devpod-$HOST_KEY" "$minutes" "$_cap" "$D/hotaisle.record" "$HA_GUARD" "$D" \
            || { rm -rf "$D"; die "Hot Aisle created no bare-metal server: $HA_REFUSED"; }
        SSH_TARGET="$HA_TARGET"; BOX_SUDO=1; POD_ID="$HA_VMREF"; POD_NAME="$HA_VMNAME"; BOX_ENV=""
        COST_HR=$(awk -v c="$HA_PRICE" 'BEGIN{printf "%.2f", c/100}')
        HOST_SLOTS=$(sed -n 's/^GPU_AGENTS=//p' "$D/hotaisle_device.txt" | head -1); HOST_SLOTS=${HOST_SLOTS:-8}
        GPU="${HOST_SLOTS}x MI300X hotaisle bare metal $HA_GFX"
        echo "$HA_DEADMAN_PID" > "$D/deadman.pid"
        write_state; STATE_WRITTEN=1
        say "host $POD_ID up (\$${COST_HR}/hr, $GPU), lease ${minutes} min; installing git, pixi"
        bx 900 'export DEBIAN_FRONTEND=noninteractive; command -v git > /dev/null && command -v curl > /dev/null || { apt-get -o DPkg::Lock::Timeout=300 update -qq && apt-get -o DPkg::Lock::Timeout=300 install -y -qq git curl ca-certificates; } > /dev/null
[ -x /root/.pixi/bin/pixi ] || curl -fsSL https://pixi.sh/install.sh | bash > /dev/null 2>&1; /root/.pixi/bin/pixi --version' < /dev/null > "$D/bootstrap.log" 2>&1 \
            || say "bootstrap failed; see $D/bootstrap.log"
        say "ready: $HOST_SLOTS GPU slots; lanes take one with: $0 up <lane> --vendor amd"
        ;;
    extend) load_state; host_extend "${1:-120}"; say "extended the shared AMD host $POD_ID by ${1:-120} min" ;;
    status)
        load_state
        echo "host $POD_ID ($GPU, \$$COST_HR/hr) ssh ${SSH_TARGET##* }"
        for _s in "$D"/slots/*/owner; do [ -f "$_s" ] && echo "slot $(basename "$(dirname "$_s")"): $(tr '\n' ' ' < "$_s")"; done
        bx 60 "cat $HA_GUARD/watchdog.out 2>/dev/null | tail -2; pgrep -af '^sh $HA_GUARD/watchdog'" < /dev/null || true
        ;;
    down)
        load_state
        _held=$(ls "$D/slots" 2>/dev/null | wc -l | tr -d ' ')
        [ "$_held" = 0 ] || [ "${1:-}" = --force ] || die "$_held slot(s) are held ($(ls "$D/slots" | tr '\n' ' ')); free them or pass --force"
        [ -n "${HA_VMREF:-}" ] || die "the host state has no HA_VMREF; refusing a teardown that would adopt by listing"
        ha_load_key || die "no Hot Aisle key ($HA_KEYFILE)"
        HA_RECORD="$D/hotaisle.record"; HA_CREATE_ATTEMPTED=1; HA_GONE=0
        HA_DEADMAN_PID=$(cat "$D/deadman.pid" 2>/dev/null || true)
        ha_teardown || die "$POD_ID NOT CONFIRMED GONE; the Mac dead-man and the on-box watchdog stay armed"
        for _s in "$D"/slots/*/owner; do
            [ -f "$_s" ] || continue
            _k=$(sed -n 's/^key=//p' "$_s"); [ -z "$_k" ] || [ ! -f "$STATE_ROOT/$_k/state.env" ] \
                || mv "$STATE_ROOT/$_k" "$STATE_ROOT/$_k.down-$(date -u +%Y%m%dT%H%M%SZ)"
        done
        mv "$D" "$D.down-$(date -u +%Y%m%dT%H%M%SZ)"
        say "the shared AMD host $POD_ID is down"
        ;;
    esac
    ;;
up)
    [ ! -f "$D/state.env" ] || die "$KEY already has a box ($D/state.env); down it first"
    if [ "$VENDOR" != amd ]; then
        # Andrew 2026-09-28: "share a runpod or 2 runpods and not create 12 of them". A lane
        # never rents its own NVIDIA pod; every lane submits to the shared pods.
        [ "${MOJOLEARN_DEVPOD_VIA_CENTRAL:-0}" = 1 ] || [ "${MOJOLEARN_DEVPOD_OWN_NVIDIA:-0}" = 1 ] \
            || die "lanes do not rent NVIDIA pods: use the shared pods (tools/nvidia_central.sh sync/submit; the pods come up with tools/nvidia_central.sh up). MOJOLEARN_DEVPOD_OWN_NVIDIA=1 only with Andrew's OK"
        _live=$(curl -s -m 20 -H "Content-Type: application/json" -H "Authorization: Bearer $(cat "$HOME/.mojolearn_runpod_key" 2>/dev/null)" https://api.runpod.io/graphql \
            -d '{"query":"query { myself { pods { id } } }"}' | python3 -c 'import sys,json; print(len(json.load(sys.stdin)["data"]["myself"]["pods"]))' 2>/dev/null)
        [ -n "$_live" ] || die "could not count live RunPod pods; refusing to rent (cap $MAX_RUNPOD_PODS)"
        [ "$_live" -lt "$MAX_RUNPOD_PODS" ] || die "$_live RunPod pods are live and the cap is $MAX_RUNPOD_PODS (MOJOLEARN_DEVPOD_MAX_PODS); share a pod or bring one down"
    fi
    if [ "$VENDOR" = amd ] && [ -f "$HD/state.env" ] && [ "${MOJOLEARN_DEVPOD_NO_HOST:-0}" != 1 ]; then
        mkdir -p "$HD/slots"
        if SLOT=$(take_slot "$KEY"); then
            BASE_SHA=$(git -C "$ROOT" rev-parse --verify "${BASE_REF:-origin/main}^{commit}") || { rm -rf "$HD/slots/$SLOT"; die "no commit ${BASE_REF:-origin/main}"; }
            ( . "$HD/state.env"; printf 'SSH_TARGET=%q\nBOX_SUDO=%q\nPOD_ID=%q\nPOD_NAME=%q\nCOST_HR=%q\n' "$SSH_TARGET" "$BOX_SUDO" "$POD_ID" "$POD_NAME" "$COST_HR" ) > "$TMPD/host.env"
            . "$TMPD/host.env"
            PROVIDER=amdhost; BOX_DIR=/root/mojolearn-$lane_arg
            BOX_ENV="ROCR_VISIBLE_DEVICES=$SLOT HIP_VISIBLE_DEVICES=0"
            GPU="MI300X slot $SLOT of the shared host"; COST_HR="shared"
            mkdir -p "$D"; write_state
            # the pin, proven: exactly one gfx942 agent under the slot's env
            _n=$(bx 120 "export PATH=/opt/rocm/bin:\$PATH $BOX_ENV; rocminfo | awk '\$1 == \"Name:\" && \$2 ~ /^gfx/' | wc -l" < /dev/null | tr -dc 0-9)
            [ "$_n" = 1 ] || { rm -rf "$D" "$HD/slots/$SLOT"; die "slot $SLOT's pin shows $_n GPU agents, not 1; slot freed"; }
            seed_git "$BASE_SHA"
            say "$KEY: GPU slot $SLOT on the shared AMD host $POD_ID, tree $BOX_DIR (1 GPU visible); ready: $0 sync $KEY <worktree>"
            exit 0
        fi
        say "the shared AMD host has no free GPU slot; renting a box of its own"
    fi
    mkdir -p "$D"
    BASE_SHA=$(git -C "$ROOT" rev-parse --verify "${BASE_REF:-origin/main}^{commit}") || die "no commit ${BASE_REF:-origin/main}"
    POD_ID=""; COST_HR=""; GPU=""
    if [ "$VENDOR" = nvidia ]; then
        load_key || die "no RunPod key (~/.mojolearn_runpod_key, mode 600)"
        PROVIDER=runpod
        runpod_create "${MOJOLEARN_DEVPOD_RETRY_MINUTES:-60}" "$NV_GPUS" "$NV_IMAGE" 0 \
            || { rm -rf "$D"; die "nothing was created"; }
    else
        _rp=1
        if load_key; then
            PROVIDER=runpod
            runpod_create "${MOJOLEARN_DEVPOD_AMD_RETRY_MINUTES:-15}" "$AMD_GPUS" "$AMD_IMAGE" 1 && _rp=0
        else
            say "no RunPod key; going to Hot Aisle"
        fi
        if [ "$_rp" = 1 ]; then
            PROVIDER=hotaisle; STATE_WRITTEN=0
            say "falling back to a Hot Aisle MI300X VM"
            _cap=$(( ${MOJOLEARN_DEVPOD_HA_CAP_USD:-150} * 100 ))
            HA_GUARD=/root/.mojolearn-devpod-guard
            ha_rent "devpod-$KEY" "$minutes" "$_cap" "$D/hotaisle.record" "$HA_GUARD" "$D" \
                || { rm -rf "$D"; die "Hot Aisle created nothing: $HA_REFUSED"; }
            SSH_TARGET="$HA_TARGET"; BOX_SUDO=1; POD_ID="$HA_VMREF"; POD_NAME="$HA_VMNAME"
            COST_HR=$(awk -v c="$HA_PRICE" 'BEGIN{printf "%.2f", c/100}'); GPU="MI300X hotaisle $HA_SPEC_USED $HA_GFX"
            [ "$HA_SPEC_USED" = 2gpu ] && BOX_ENV="ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0"
            echo "$HA_DEADMAN_PID" > "$D/deadman.pid"
            write_state; STATE_WRITTEN=1
        fi
    fi
    say "box $POD_ID up on $PROVIDER (\$${COST_HR:-?}/hr, $GPU), lease ${minutes} min; installing pixi"
    bx 600 'command -v curl > /dev/null || { export DEBIAN_FRONTEND=noninteractive; apt-get update -qq > /dev/null && apt-get install -y -qq curl ca-certificates > /dev/null; }
[ -x /root/.pixi/bin/pixi ] || command -v pixi > /dev/null || curl -fsSL https://pixi.sh/install.sh | bash; mkdir -p /root/mojolearn' < /dev/null > "$D/bootstrap.log" 2>&1 \
        || say "pixi bootstrap failed; see $D/bootstrap.log"
    seed_git "$BASE_SHA"
    say "git tree seeded at $(git -C "$ROOT" rev-parse --short "$BASE_SHA") (sync moves it to the worktree's merge base)"
    say "ready: $0 sync $KEY <worktree>"
    ;;
sync)
    load_state; wt="${1:?worktree}"; [ -d "$wt/.git" ] || [ -f "$wt/.git" ] || die "$wt is not a git worktree"
    [ "$(cd "$wt" && git rev-parse --show-toplevel)" != "$(cd "$ROOT" && git rev-parse --show-toplevel)" ] || [ "${MOJOLEARN_DEVPOD_ALLOW_SELF:-0}" = 1 ] \
        || die "sync a lane's OWN worktree, not the checkout this tool runs from"
    # R6: the box's HEAD is the worktree's merge base with origin/main, or the
    # sync refuses. seed_git brings it there (a lane that merged main moves it).
    base=$(git -C "$wt" merge-base HEAD origin/main) || die "no merge base of $wt with origin/main"
    # A worktree mid-merge (unmerged paths) would ship main's new files as
    # this patch's own additions; refuse it.
    [ -z "$(git -C "$wt" diff --name-only --diff-filter=U)" ] || die "$wt has unmerged paths; finish the merge before a sync"
    seed_git "$base"
    if [ "${MOJOLEARN_DEVPOD_FULL_SYNC:-0}" != 1 ]; then
        # PATCH SYNC (default, 2026-09-27; the metrics lane's psync idea): ship
        # only `git diff --binary <merge base>` of the worktree, not the whole
        # tree (the tar was ~800 MB). New files are marked intent-to-add in a
        # COPY of the index, so the lane's real index is never touched. On the
        # box: reset to the base, remove the files the previous patch added
        # (so a dropped file never lingers) and this patch's added files, then
        # apply. Untracked build outputs (.so, .pixi) are left alone. A file
        # the previous patch added that the NEW base tracks (the lane merged
        # to main and its merge base moved past it) is kept: removing it after
        # the reset deleted a tracked source (lane/trees-cpu, 2026-09-28:
        # gbdt/host/gbdt_oracle_ctr.mojo vanished from the box).
        _idx="$TMPD/sync.index"; cp "$(git -C "$wt" rev-parse --path-format=absolute --git-path index)" "$_idx"
        ( cd "$wt" && git ls-files -z -o --exclude-standard | { grep -zvE '\.(so|dylib|metallib)$' || true; } \
            | GIT_INDEX_FILE="$_idx" xargs -0 -r git add -N -- ) || die "could not mark new files"
        ( cd "$wt" && GIT_INDEX_FILE="$_idx" git diff --binary "$base" ) > "$TMPD/sync.patch" || die "diff failed"
        ( cd "$wt" && GIT_INDEX_FILE="$_idx" git diff --name-only --diff-filter=A "$base" ) > "$TMPD/sync.added"
        bx 60 "mkdir -p $BOX_DIR && cat > $BOX_DIR/.git/devpod_added.new" < "$TMPD/sync.added" || die "added-list upload failed"
        bx 900 "cd $BOX_DIR && git reset -q --hard $base && { [ ! -f .devpod_manifest ] || { git ls-files -o --exclude-standard | grep -Fxf .devpod_manifest | grep -vE '\\.(so|dylib|metallib)\$' | xargs -r rm -f --; rm -f .devpod_manifest .devpod_manifest.prev; }; } && { [ ! -f .git/devpod_added ] || { git ls-files > /tmp/devpod_tracked && { grep -vxFf /tmp/devpod_tracked .git/devpod_added || true; } | xargs -r rm -f --; }; } && xargs -r rm -f -- < .git/devpod_added.new && cat > /tmp/devpod_sync.patch && { [ ! -s /tmp/devpod_sync.patch ] || git apply --whitespace=nowarn /tmp/devpod_sync.patch; } && mv .git/devpod_added.new .git/devpod_added" \
            < "$TMPD/sync.patch" || die "patch sync failed (retry with MOJOLEARN_DEVPOD_FULL_SYNC=1)"
        _head=$(bx 60 "cd $BOX_DIR && git rev-parse HEAD" < /dev/null | tr -d '\r')
        [ "$_head" = "$base" ] || die "the box's HEAD ($_head) is not the worktree's merge base ($base)"
        say "patch-synced $(cd "$wt" && git rev-parse --short HEAD)+worktree -> $POD_ID:$BOX_DIR ($(wc -c < "$TMPD/sync.patch" | tr -d ' ') bytes over merge base $(git -C "$wt" rev-parse --short "$base"))"
        exit 0
    fi
    # FULL SYNC (MOJOLEARN_DEVPOD_FULL_SYNC=1). A file that was in the LAST sync's manifest but not in this one is deleted
    # on the box, so a moved or deleted source never lingers there to mask a result.
    ( cd "$wt" && git ls-files -c -o --exclude-standard | grep -vE '\.(so|dylib|metallib)$' ) > "$TMPD/manifest" || die "no file list"
    bx 60 "mkdir -p $BOX_DIR && cd $BOX_DIR && { [ ! -f .devpod_manifest ] || mv .devpod_manifest .devpod_manifest.prev; } && cat > .devpod_manifest" < "$TMPD/manifest" \
        || die "manifest upload failed"
    ( cd "$wt" && tr '\n' '\0' < "$TMPD/manifest" | COPYFILE_DISABLE=1 tar --null -czf - -T - ) \
        | bx 900 "cd $BOX_DIR && tar xzf - --no-same-owner" || die "sync failed"
    bx 120 'cd '"$BOX_DIR"' && if [ -f .devpod_manifest.prev ]; then sort .devpod_manifest.prev > /tmp/m.prev; sort .devpod_manifest > /tmp/m.now;
            comm -23 /tmp/m.prev /tmp/m.now | while IFS= read -r f; do [ -n "$f" ] && rm -f -- "$f" && echo "dev_pod: removed stale $f"; done; fi' < /dev/null \
        || die "stale-file cleanup failed"
    _head=$(bx 60 "cd $BOX_DIR && git rev-parse HEAD" < /dev/null | tr -d '\r')
    [ "$_head" = "$base" ] || die "the box's HEAD ($_head) is not the worktree's merge base ($base)"
    say "synced $(cd "$wt" && git rev-parse --short HEAD)+worktree -> $POD_ID:$BOX_DIR (box HEAD = merge base $(git -C "$wt" rev-parse --short "$base"))"
    ;;
run)
    load_state; [ $# -gt 0 ] || die "run needs a command"
    # shellcheck disable=SC2086
    _lk=""; [ -z "$CENTRAL_KEY" ] || [ "$KEY" != "$CENTRAL_KEY" ] \
        || _lk="mkdir -p /var/lock/mojolearn-central; exec 9>/var/lock/mojolearn-central/gpu0.lock; flock 9; echo 'lane=$KEY (dev_pod run)' > /var/lock/mojolearn-central/gpu0.owner; "
    ssh $SSH_OPTS $SSH_TARGET "$(box_cmd "$_lk$BOX_PATH; ${BOX_ENV:+export $BOX_ENV; }cd $BOX_DIR && $*")"
    ;;
extend)
    load_state; minutes="${1:-120}"
    if [ -n "$CENTRAL_KEY" ] && [ "$KEY" = "$CENTRAL_KEY" ]; then
        _left=$(( (CENTRAL_LEASE_END - $(now)) / 60 ))
        [ "$minutes" -ge "$_left" ] || { say "$KEY is the central AMD box: extending to its lease end ($_left min), not $minutes"; minutes=$_left; }
    fi
    if [ "$PROVIDER" = amdhost ]; then
        [ -f "$HD/state.env" ] || die "the shared AMD host is gone; $KEY's slot is stale (down $KEY)"
        host_extend "$minutes"
        say "extended the shared AMD host (for $KEY, slot $SLOT) by ${minutes} min"; exit 0
    fi
    if [ "$PROVIDER" = hotaisle ]; then
        ha_load_key || die "no Hot Aisle key ($HA_KEYFILE)"
        ha_rearm_watchdog $(( minutes * 60 ))
        ha_rearm_mac_deadman $(( minutes * 60 + 600 ))
        write_state
    else
        load_key || die "no RunPod key"
        MOJOLEARN_LEASE_DIR="$D/lease" with_timeout 180 sh "$ROOT/tools/runpod_guard.sh" extend "$POD_ID" "$SSH_TARGET" "$minutes" \
            || die "extend REFUSED; the pod keeps its old lease"
        arm_mac_deadman $(( minutes * 60 + 600 ))
    fi
    say "extended $POD_ID by ${minutes} min (on-box lease and Mac dead-man)"
    ;;
down)
    load_state
    [ -z "$CENTRAL_KEY" ] || [ "$KEY" != "$CENTRAL_KEY" ] || [ "${MOJOLEARN_CENTRAL_DOWN:-0}" = 1 ] \
        || die "$KEY is the central AMD box every lane shares (tools/amd_central.sh); refusing (MOJOLEARN_CENTRAL_DOWN=1 overrides)"
    if [ "$PROVIDER" = amdhost ]; then
        grep -qx "key=$KEY" "$HD/slots/$SLOT/owner" 2>/dev/null && rm -rf "$HD/slots/$SLOT"
        mv "$D" "$D.down-$(date -u +%Y%m%dT%H%M%SZ)"
        say "$KEY: slot $SLOT freed (its tree $BOX_DIR stays on the host; the host stays up)"; exit 0
    fi
    if [ "$PROVIDER" = hotaisle ]; then
        ha_load_key || die "no Hot Aisle key ($HA_KEYFILE)"
        HA_RECORD="$D/hotaisle.record"; HA_CREATE_ATTEMPTED=1; HA_GONE=0
        HA_DEADMAN_PID=$(cat "$D/deadman.pid" 2>/dev/null || true)
        ha_teardown || die "$POD_ID NOT CONFIRMED GONE; the Mac dead-man and the on-box watchdog stay armed"
    else
        load_key || die "no RunPod key"
        delete_pod "$POD_ID"
        verify_gone "$POD_ID" || die "$POD_ID NOT CONFIRMED GONE; the Mac dead-man stays armed"
        _p=$(cat "$D/deadman.pid" 2>/dev/null || true)
        [ -z "$_p" ] || { pkill -P "$_p" 2>/dev/null || true; kill "$_p" 2>/dev/null || true; }
    fi
    mv "$D" "$D.down-$(date -u +%Y%m%dT%H%M%SZ)"
    say "$KEY box $POD_ID down"
    ;;
list)
    for s in "$STATE_ROOT"/*/state.env; do
        [ -f "$s" ] || continue
        case "$s" in *.down-*) continue ;; esac
        ( . "$s"; printf '%-16s %-9s %-22s %-34s $%s/hr\n' "$(basename "$(dirname "$s")")" "${PROVIDER:-runpod}" "$POD_ID" "$GPU" "${COST_HR:-?}" )
    done
    ;;
esac
