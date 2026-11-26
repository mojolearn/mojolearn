#!/bin/bash
# tools/dev_pod.sh -- ONE long-lived development box per lane and vendor, for
# the algorithm expansion. A lane's
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
#   tools/dev_pod.sh run    <lane> [--vendor amd] [--results DIR] [--no-persist] <command>
#                                              in /root/mojolearn, as root, under tools/box_persist_run.sh:
#                                              DIR (default box-out/<tag>) persists to R2 from the box
#   tools/dev_pod.sh extend <lane> [--vendor amd] [minutes]   the heartbeat: on-box lease AND Mac dead-man
#   tools/dev_pod.sh down   <lane> [--vendor amd]             delete, verify gone, disarm the Mac dead-man
#   tools/dev_pod.sh list                                     every box, from the state files
#   tools/dev_pod.sh seed   <key> <commit> <box dir> [--checkout]
#                                              that commit into the git repo at <box dir> on the box (fetched from
#                                              GitHub, shipped from this checkout when GitHub refuses the box);
#                                              --checkout also moves the working tree to it (tools/cpu_central.sh)
#
# THE CPU CHECK BOX (`up <key> --vendor cpu`, lane fx-cpu-box-checks, 2026-11-19): a RunPod CPU pod (no GPU) that
# only tools/cpu_central.sh rents (keys cpc1, cpc2), with the CPU flavors and vCPU count it picks
# (MOJOLEARN_DEVPOD_CPU_FLAVORS, MOJOLEARN_DEVPOD_CPU_VCPU) on runpod/base (MOJOLEARN_DEVPOD_CPU_IMAGE); created
# through the v1 CPU schema as tools/runpod_cpu_leg.sh does, its status read through v2. Its lease is the queue's
# (tools/cpu_central.sh), as for the shared NVIDIA pods.
#
# THE SHARED AMD HOST (`host up`, a Hot Aisle bare-metal 8x MI300X with GPU
# slots per lane) and the Hot Aisle VM fallback were removed 2026-11-18 by
# lane fx-dead-small: AMD is the DigitalOcean MI300X (AGENTS.md "Boxes");
# recoverable at 7c1aad6561. A state file left with PROVIDER=hotaisle or
# amdhost is refused by name.
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
# MOJOLEARN_DEVPOD_AMD_RETRY_MINUTES (default 15); nothing else. /root/mojolearn
# is the tree on every box. DigitalOcean is never used here.
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
# The safety path is tools/runpod_pod_lib.sh + tools/runpod_guard.sh: the Mac dead-man is armed BEFORE the
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
# Comma-separated; v2 tries one GPU type at a time after confirmed stock refusals.
# Identity needs any NVIDIA; speed is judged before/after on the SAME pod.
# 2026-09-28: 13 H100s at $3.49/h cost too much in four hours. Identity work
# needs no H100: the default list stops at the cheap cards, and an H100/H200/A100/B200 in
# MOJOLEARN_DEVPOD_GPUS is refused unless MOJOLEARN_DEVPOD_ALLOW_BIG_GPU=1 (explicit approval only).
NV_GPUS="${MOJOLEARN_DEVPOD_GPUS:-NVIDIA GeForce RTX 4090,NVIDIA L40S,NVIDIA RTX 6000 Ada Generation,NVIDIA RTX A6000,NVIDIA A40}"
case "$NV_GPUS" in *H100*|*H200*|*A100*|*B200*)
    [ "${MOJOLEARN_DEVPOD_ALLOW_BIG_GPU:-0}" = 1 ] || { echo "dev_pod: $NV_GPUS includes an H100/H200/A100/B200; refused without explicit approval (MOJOLEARN_DEVPOD_ALLOW_BIG_GPU=1)" >&2; exit 2; } ;;
esac
# At most this many live mojolearn-* RunPod pods (NVIDIA), and at most MOJOLEARN_RUNPOD_CAP (8) pods on the whole
# account, counted from the API by tools/rent_guard.sh before every create.
MAX_RUNPOD_PODS="${MOJOLEARN_DEVPOD_MAX_PODS:-3}"
# No lease longer than this many minutes; a lane that needs more extends, visibly.
MAX_LEASE_MIN="${MOJOLEARN_DEVPOD_MAX_LEASE_MIN:-240}"
# GPUs per NVIDIA pod: 1, except the shared pods tools/nvidia_central.sh brings up (4x/2x).
NV_GPU_COUNT="${MOJOLEARN_DEVPOD_GPU_COUNT:-1}"
[[ "$NV_GPU_COUNT" =~ ^[1-8]$ ]] || { echo "dev_pod: MOJOLEARN_DEVPOD_GPU_COUNT must be 1..8" >&2; exit 2; }
# NVIDIA uses v2's >=13.0 CUDA floor, required by pinned MAX. An explicit
# MOJOLEARN_DEVPOD_ALLOWED_CUDA_VERSIONS=13.0,13.2 selects an exact set, but
# empty/malformed values or ANY version below 13.0 refuse before a create.
NV_IMAGE="${MOJOLEARN_DEVPOD_IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
AMD_GPUS="${MOJOLEARN_DEVPOD_AMD_GPUS:-AMD Instinct MI300X OAM}"
AMD_IMAGE="${MOJOLEARN_DEVPOD_AMD_IMAGE:-rocm/dev-ubuntu-22.04:6.4.1-complete}"
# Container disk: sized from the release gate's disk plan (tools/ops/gate_disk_plan.py pod-gb: the gate job's measured
# footprint + 20 GB headroom, at least the former fixed 80 GB; 0.8.66 run t filled an 80 GB pod). The gate's work is
# under /root, the container disk, so no network volume is requested (volumeInGb stays 0).
DISK_GB="${MOJOLEARN_DEVPOD_DISK_GB:-$(python3 "$ROOT/tools/ops/gate_disk_plan.py" pod-gb 2>/dev/null || echo 80)}"
[[ "$DISK_GB" =~ ^[1-9][0-9]*$ ]] || { echo "dev_pod: the container disk size must be a whole number of GB (got '$DISK_GB')" >&2; exit 2; }
READY_TIMEOUT=900
# Slow provider SSH handshakes may need a longer bounded connection attempt.
SSH_CONNECT_TIMEOUT="${MOJOLEARN_RUNPOD_SSH_CONNECT_TIMEOUT:-20}"
case "$SSH_CONNECT_TIMEOUT" in ''|*[!0-9]*) echo "SSH connect timeout must be an integer from 1 to 300" >&2; exit 2;; esac
[ "$SSH_CONNECT_TIMEOUT" -ge 1 ] && [ "$SSH_CONNECT_TIMEOUT" -le 300 ] || { echo "SSH connect timeout must be an integer from 1 to 300" >&2; exit 2; }
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=$SSH_CONNECT_TIMEOUT -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o BatchMode=yes"
REPO_URL="${MOJOLEARN_DEVPOD_REPO_URL:-https://github.com/mojolearn/mojolearn.git}"
BOX_PATH='export PATH=/root/.pixi/bin:/opt/rocm/bin:$PATH'
BOX_DIR=/root/mojolearn

die() { echo "dev_pod: $*" >&2; exit 1; }
say() { echo "dev_pod: $*"; }
now() { date +%s; }

TMPD="$(mktemp -d "${TMPDIR:-/tmp}/devpod.XXXXXX")"
PROVIDER=""
on_exit() {
    rm -rf "$TMPD"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
CURLRC="$TMPD/curlrc"
. "$ROOT/tools/runpod_pod_lib.sh"
. "$ROOT/tools/box_persist_lib.sh"

cmd="${1:-}"; lane="${2:-}"; lane_arg=""; VENDOR=nvidia
case "$cmd" in
list) ;;
up|sync|run|extend|down|seed)
    shift $(( $# < 2 ? $# : 2 ))
    case "$lane" in *-amd) VENDOR=amd ;; esac
    lane_arg=${lane%-amd}
    [[ "$lane_arg" =~ ^[a-z0-9-]{1,24}$ ]] || die "lane must be [a-z0-9-]{1,24}"
    if [ "$cmd" != up ] && [ "${1:-}" = --vendor ]; then VENDOR="${2:?--vendor nvidia|amd}"; shift 2; fi
    ;;
*) sed -n 2,24p "$0"; exit 2 ;;
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
case "$VENDOR" in nvidia|amd|cpu) ;; *) die "--vendor must be nvidia, amd or cpu" ;; esac
# the CPU check box (tools/cpu_central.sh picks the flavors and vCPUs; runpod_gpu_request.py cpu validates them)
CPU_IMAGE="${MOJOLEARN_DEVPOD_CPU_IMAGE:-runpod/base:1.3.1-ubuntu2204}"
CPU_FLAVORS="${MOJOLEARN_DEVPOD_CPU_FLAVORS:-cpu5c,cpu3c}"
CPU_VCPU="${MOJOLEARN_DEVPOD_CPU_VCPU:-32}"
KEY="$lane_arg"; [ "$VENDOR" = amd ] && KEY="$lane_arg-amd"
D="$STATE_ROOT/$KEY"
BOX_ENV=""
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
case "$KEY" in cpc[0-9])
    case "$cmd" in
    extend) die "$KEY is the shared CPU check box: its job queue keeps its lease (tools/cpu_central.sh); nothing extended" ;;
    down) [ "${MOJOLEARN_NVC_DOWN:-0}" = 1 ] || die "$KEY is the shared CPU check box; tools/cpu_central.sh down" ;;
    run|sync) die "$KEY is the shared CPU check box: use tools/cpu_central.sh check/sh" ;;
    esac ;;
esac
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
    PROVIDER=${PROVIDER:-runpod}; BOX_ENV=${BOX_ENV:-}; BOX_DIR=${BOX_DIR:-/root/mojolearn}
    case "$PROVIDER" in hotaisle|amdhost)
        die "$KEY is a Hot Aisle box (PROVIDER=$PROVIDER); Hot Aisle support was removed 2026-11-18 (recoverable at 7c1aad6561): tear it down there" ;;
    esac
    if [ "$PROVIDER" = runpod ]; then
        # Older state files keep their v1 lifecycle; new NVIDIA rentals persist v2.
        rp_select_api "${RUNPOD_API_VERSION:-1}" || die "invalid RunPod state API"
    fi
}
box_cmd() {  # the command string ssh runs: as root on every provider
    printf '%s' "$1"
}
# shellcheck disable=SC2086
bx() { with_timeout "$1" ssh $SSH_OPTS $SSH_TARGET "$(box_cmd "$2")"; }

arm_mac_deadman() {  # seconds; replaces any earlier dead-man for this lane
    if [ -s "$D/deadman.pid" ]; then
        _p=$(cat "$D/deadman.pid"); pkill -P "$_p" 2>/dev/null || true; kill "$_p" 2>/dev/null || true
    fi
    write_deadman "$D/deadman" "$1" || die "the dead-man did not compose"
    [ -z "${POD_ID:-}" ] || printf '%s\n' "$POD_ID" > "$D/deadman/pod_id.txt"
    python3 "$ROOT/tools/deadman_session.py" "$D/deadman/deadman.sh" > "$D/deadman.pid" \
        || die "the Mac dead-man did not start"
    sleep 1; kill -0 "$(cat "$D/deadman.pid")" 2>/dev/null || die "the Mac dead-man did not start"
}

ship_commit() {  # <full sha>: the Mac hands the box the commit GitHub would not
    # GitHub refuses an anonymous fetch from some rented addresses ("could not
    # read Username", 2026-09-29, nvc1) although the repository is public. The
    # commit and its tree then travel from this checkout as one pack over the
    # ssh the box already trusts, less whatever the box's own HEAD holds, and
    # the commit is recorded as a shallow root, as `fetch --depth=1` records it.
    git -C "$ROOT" cat-file -e "$1^{commit}" 2>/dev/null || die "this checkout does not hold $1 either"
    _have=$(bx 120 "cd $BOX_DIR 2>/dev/null && git rev-parse -q --verify HEAD 2>/dev/null || true" < /dev/null 2>/dev/null | tr -d '\r' | grep -E '^[0-9a-f]{40}$' | head -1 || true)
    # What travels is the commit and every object of its tree that the tree of
    # the box's HEAD does not already hold. A set difference of the two trees,
    # never `--not <HEAD>`: when the wanted commit is an ANCESTOR of the box's
    # HEAD that excludes the commit itself and ships nothing (seen 2026-09-29).
    git -C "$ROOT" rev-list --objects --no-walk "$1" | cut -d' ' -f1 | LC_ALL=C sort -u > "$TMPD/ship.want"
    : > "$TMPD/ship.have"
    if [ -n "$_have" ] && git -C "$ROOT" cat-file -e "$_have^{commit}" 2>/dev/null; then
        git -C "$ROOT" rev-list --objects --no-walk "$_have" | cut -d' ' -f1 | LC_ALL=C sort -u > "$TMPD/ship.have"
    else
        _have=""
    fi
    LC_ALL=C comm -23 "$TMPD/ship.want" "$TMPD/ship.have" > "$TMPD/ship.send"
    grep -qx "$1" "$TMPD/ship.send" || die "the pack for $1 would not hold the commit itself"
    say "GitHub refused the box; shipping $1 from this checkout: $(wc -l < "$TMPD/ship.send" | tr -d ' ') objects (less the tree of the HEAD the box holds: ${_have:-none})"
    # The box records the shallow root only AFTER every object of the tree is
    # seen present, so a short pack never leaves a root that names nothing.
    git -C "$ROOT" pack-objects --stdout -q < "$TMPD/ship.send" \
        | bx 1800 "set -e; mkdir -p $BOX_DIR && cd $BOX_DIR
[ -d .git ] || { git init -q . && git remote add origin $REPO_URL; }
git unpack-objects -q
git cat-file -e $1^{commit}
[ \"\$(git ls-tree -r -t $1 | awk '{print \$3}' | git cat-file --batch-check | grep -c ' missing\$' || true)\" = 0 ]
grep -qx $1 .git/shallow 2>/dev/null || echo $1 >> .git/shallow" || die "could not ship $1 to the box"
}

seed_git() {  # <full sha>
    TMPD_SEED="$TMPD/seed.err"
    _seed_git "$1" 2> "$TMPD_SEED" && return 0
    grep -q SEED_FETCH_FAILED "$TMPD_SEED" || grep -q 'could not read Username' "$TMPD_SEED" || { cat "$TMPD_SEED" >&2; die "could not seed the box's git tree at $1"; }
    ship_commit "$1"
    _seed_git "$1" 2> "$TMPD_SEED" || { cat "$TMPD_SEED" >&2; die "could not seed the box's git tree at $1 after shipping it"; }
}

_seed_git() {  # <full sha>; returns 1 with the reason on stderr
    _out=$(bx 900 "set -e; command -v git > /dev/null || { export DEBIAN_FRONTEND=noninteractive; apt-get update -qq > /dev/null && apt-get install -y -qq git > /dev/null; }
mkdir -p $BOX_DIR && cd $BOX_DIR
[ -d .git ] || { git init -q . && git remote add origin $REPO_URL; }
git config --global --add safe.directory $BOX_DIR 2>/dev/null || true
grep -qx '.devpod_manifest' .git/info/exclude 2>/dev/null || printf '.devpod_manifest\n.devpod_manifest.prev\n' >> .git/info/exclude
if [ \"\$(git rev-parse -q --verify HEAD 2>/dev/null)\" != $1 ]; then
    git cat-file -e $1^{commit} 2>/dev/null || git fetch -q --depth=1 $REPO_URL $1 || { echo SEED_FETCH_FAILED >&2; exit 1; }
    git update-ref HEAD $1
    git read-tree HEAD
    git update-index -q --refresh > /dev/null || true
fi
echo HEAD=\$(git rev-parse HEAD)" < /dev/null 2>&1) || { printf '%s\n' "$_out" >&2; return 1; }
    printf '%s\n' "$_out" | grep -qx "HEAD=$1" || { printf '%s\n' "$_out" >&2; die "the box's HEAD is not $1 after seeding"; }
}

# One RunPod create loop: out of stock is retried until $1 minutes pass, any
# other refusal dies. 0 with POD_ID set, 1 when stock never came (nothing
# was created, the per-attempt dead-man disarmed).
runpod_create() {  # <retry minutes> <gpus | cpu flavors> <image> <kind: 0 nvidia, 1 amd, cpu>
    POD_NAME="mojolearn-dev-$KEY-$(date -u +%m%d%H%M)"
    # NVIDIA and the CPU box create, read and delete through v2 (api.runpod.io); only the AMD kind keeps v1, whose host
    # rest.runpod.io no longer resolves (2026-11-20, lane fx-cpu-pod-v2: the AMD kind needs the same v2 move)
    if [ "$4" = 1 ]; then rp_select_api 1; else rp_select_api 2; fi
    # Keep complete request/response evidence outside D: a confirmed failed
    # create may remove D, but must never erase the provider's diagnosis.
    _request_evidence="$STATE_ROOT/$KEY.requests-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    ( umask 077; mkdir -p "$_request_evidence" ) || die "cannot retain rental evidence"
    rp_call GET "$RP/pods"; case "$RP_CODE" in 2*) ;; *) die "pod listing HTTP $RP_CODE" ;; esac
    cp "$TMPD/rp.body" "$_request_evidence/list-before.json" || die "cannot retain pre-create listing"
    [ "$(rp_py listingvalid)" = yes ] || die "pod listing incomplete or invalid; no create attempted"
    if [ "$4" = cpu ]; then
        python3 "$ROOT/tools/runpod_gpu_request.py" cpu "$TMPD/create.json" "$POD_NAME" "$3" "$2" "$CPU_VCPU" "$DISK_GB" 0 \
            || die "CPU request validation failed; no create attempted"
    elif [ "$4" = 1 ]; then
        python3 "$ROOT/tools/runpod_gpu_request.py" "$TMPD/create.json" "$POD_NAME" "$3" "$2" "$DISK_GB" 1 \
            "$ROOT/tools/runpod_ssh_bootstrap.sh" 1 "" || die "AMD placement validation failed; no create attempted"
    else
        python3 "$ROOT/tools/runpod_gpu_request.py" v2 "$TMPD/create.json" "$POD_NAME" "$3" "$2" "$DISK_GB" "$NV_GPU_COUNT" 0 \
            || die "CUDA/placement validation failed; no create attempted"
    fi
    cp "$TMPD/create.json" "$D/create_request.json" || die "cannot retain validated create request"
    # OUT OF STOCK IS RETRIED, NOTHING ELSE IS (2026-09-27). Each attempt arms its own
    # dead-man before the create and disarms it when nothing was created; a
    # create that fails for any reason but stock dies at once. Every attempt is
    # logged to $STATE_ROOT/<key>.attempts.log.
    retry_until=$(( $(now) + $1 * 60 ))
    attempts_log="$STATE_ROOT/$KEY.attempts.log"
    attempt=0
    while :; do
        attempt=$(( attempt + 1 ))
        POD_ID=""
        if [ "$4" = 0 ]; then
            python3 "$ROOT/tools/runpod_gpu_request.py" v2 "$TMPD/create.json" "$POD_NAME" "$3" "$2" "$DISK_GB" "$NV_GPU_COUNT" "$((attempt - 1))" \
                || die "CUDA/placement validation failed; no create attempted"
        elif [ "$4" = cpu ]; then
            # v2 takes one CPU flavor per create: attempt N asks for flavor N of our order (runpod_gpu_request.py cpu)
            python3 "$ROOT/tools/runpod_gpu_request.py" cpu "$TMPD/create.json" "$POD_NAME" "$3" "$2" "$CPU_VCPU" "$DISK_GB" "$((attempt - 1))" \
                || die "CPU request validation failed; no create attempted"
        fi
        cp "$TMPD/create.json" "$_request_evidence/request-$attempt.json" || die "cannot retain request; no create attempted"
        cp "$TMPD/create.json" "$D/create_request.json" || die "cannot retain request; no create attempted"
        # THE ACCOUNT CAP (2026-11-13): tools/rent_guard.sh counts every live pod on the shared account through the
        # API and refuses at or above MOJOLEARN_RUNPOD_CAP (12; the Mambik pods count); NVIDIA pods also stay under our own
        # MAX_RUNPOD_PODS (mojolearn-*). Checked here, before every attempt, so no caller (tools/nvidia_central.sh
        # up, the AMD RunPod path, a retry an hour later) creates past it. Nothing was created or armed yet.
        _own_cap="--own-cap=$MAX_RUNPOD_PODS"; [ "$4" = 0 ] || _own_cap=""
        bash "$ROOT/tools/rent_guard.sh" $_own_cap || die "tools/rent_guard.sh refused the create (above); nothing was created"
        arm_mac_deadman $(( READY_TIMEOUT + minutes * 60 + 600 ))
        say "creating $POD_NAME on RunPod API v$RP_API_VERSION (requested types: $2), attempt $attempt. THE BILL STARTS HERE."
        rp_call POST "$RP/pods" "$TMPD/create.json"
        _create_code=$RP_CODE
        cp "$TMPD/rp.body" "$_request_evidence/response-$attempt.json" || die "cannot retain create response; dead-man stays armed"
        printf '%s\n' "$_create_code" > "$_request_evidence/http-$attempt.txt" || die "cannot retain create status; dead-man stays armed"
        cp "$TMPD/rp.body" "$D/create_response.json" || die "cannot retain create response; dead-man stays armed"
        POD_ID=$(rp_py id)
        # v2 answers 202 only for a QUEUED create (no pod yet, one may appear later); we never send `queue`, so a 202
        # is unexpected: stop with the dead-man armed (it deletes by name), never a second create
        [ "$RP_API_VERSION" != 2 ] || [ "$_create_code" != 202 ] \
            || die "create HTTP 202 (queued); dead-man stays armed; evidence $_request_evidence"
        [ -z "$POD_ID" ] || break
        _body=$(head -c 300 "$TMPD/rp.body"); rp_call GET "$RP/pods"
        cp "$TMPD/rp.body" "$_request_evidence/list-after-$attempt.json" || die "cannot retain create recovery listing; dead-man stays armed"
        case "$RP_CODE" in 2*) ;; *) die "create unresolved; listing HTTP $RP_CODE; dead-man stays armed; evidence $_request_evidence" ;; esac
        [ "$(rp_py listingvalid)" = yes ] || die "create unresolved; listing incomplete; dead-man stays armed; evidence $_request_evidence"
        POD_ID=$(rp_py byname "$POD_NAME" | awk '{print $1}')
        if [ -n "$POD_ID" ]; then
            echo "$POD_ID" > "$D/deadman/pod_id.txt"; delete_pod "$POD_ID"
            die "create response unparsed but $POD_ID exists by name; deleted, dead-man stays armed"
        fi
        if [ "$RP_API_VERSION" = 2 ] || [ "$4" = cpu ]; then
            # A timeout, 5xx, or malformed successful response is not proof
            # that no pod will appear. Never create a second pod on uncertainty.
            case "$_create_code" in 4??) ;; *) die "create HTTP $_create_code unresolved; dead-man stays armed; evidence $_request_evidence" ;; esac
        fi
        _p=$(cat "$D/deadman.pid"); pkill -P "$_p" 2>/dev/null || true; kill "$_p" 2>/dev/null || true
        rm -rf "$D/deadman" "$D/deadman.pid"
        printf '%s attempt %s: nothing created: %s\n' "$(date -u +%FT%TZ)" "$attempt" "$_body" >> "$attempts_log"
        if ! printf '%s' "$_body" | grep -qiE 'no (gpu )?instances|no longer any instances available|currently available|out of stock|insufficient|not enough|no available|unavailable'; then
            rm -rf "$D"
            die "create failed, nothing was created, and not for stock: $_body (full response: $_request_evidence)"
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
    if [ "$4" = cpu ]; then
        GPU=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); m=d.get("machine") or {}
c=d.get("cpu") if isinstance(d.get("cpu"), dict) else {}
print("cpu %s %s vCPU %s GB" % (c.get("id") or "?", c.get("vcpuCount") or "?", c.get("memory") or "?"))' "$D/create_response.json" 2>/dev/null || echo "cpu ?")
    fi
    _deadline=$(( $(now) + READY_TIMEOUT )); SSH_TARGET=""; _ready_attempt=0
    while [ "$(now)" -lt "$_deadline" ]; do
        _ready_attempt=$((_ready_attempt + 1))
        rp_call GET "$RP/pods/$POD_ID"
        cp "$TMPD/rp.body" "$_request_evidence/readiness-$_ready_attempt.json"
        SSH_TARGET=$(rp_py ssh)
        printf '%s\t%s\tHTTP%s\n' "$(date -u +%FT%TZ)" "$_ready_attempt" "$RP_CODE" >> "$_request_evidence/readiness.tsv"
        if [ -n "$SSH_TARGET" ]; then
            # Keep the actual connection failure alongside its provider
            # status. A timeout is not evidence that the image or job failed.
            _ssh_log="$_request_evidence/ssh-$_ready_attempt.log"
            if with_timeout "$((10#$SSH_CONNECT_TIMEOUT + 20))" ssh $SSH_OPTS $SSH_TARGET 'echo SSH-OK' > "$_ssh_log" 2>&1; then
                if grep -qx SSH-OK "$_ssh_log"; then break; fi
            fi
        fi
        SSH_TARGET=""; sleep 10
    done
    [ -n "$SSH_TARGET" ] || { delete_pod "$POD_ID"; die "no ssh after ${READY_TIMEOUT}s; deleted"; }
    if [ "$4" != 0 ]; then
        # the watchdog below needs curl; the ROCm image (and a plain CPU image) may lack it
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
        printf 'PROVIDER=%q\nVENDOR=%q\nPOD_ID=%q\nPOD_NAME=%q\nSSH_TARGET=%q\nCOST_HR=%q\nGPU=%q\nBOX_ENV=%q\nBOX_DIR=%q\n' \
            "$PROVIDER" "$VENDOR" "$POD_ID" "$POD_NAME" "$SSH_TARGET" "$COST_HR" "$GPU" "$BOX_ENV" "$BOX_DIR"
        [ "$PROVIDER" != runpod ] || [ "$VENDOR" != nvidia ] || printf 'GPU_COUNT=%q\n' "$NV_GPU_COUNT"
        [ "$PROVIDER" != runpod ] || printf 'RUNPOD_API_VERSION=%q\n' "$RP_API_VERSION"
    } > "$D/state.env"
}

case "$cmd" in
up)
    [ ! -f "$D/state.env" ] || die "$KEY already has a box ($D/state.env); down it first"
    [ ! -s "$D/deadman.pid" ] || die "$KEY has an unresolved rental/dead-man; recover its retained evidence before another create"
    if [ "$VENDOR" = cpu ]; then
        [ "${MOJOLEARN_DEVPOD_VIA_CENTRAL:-0}" = 1 ] || die "only tools/cpu_central.sh rents the CPU check box (orchestrator only)"
    elif [ "$VENDOR" != amd ]; then
        # 2026-09-28: one or two shared RunPod pods, never one per lane. A lane
        # never rents its own NVIDIA pod; every lane submits to the shared pods.
        [ "${MOJOLEARN_DEVPOD_VIA_CENTRAL:-0}" = 1 ] || [ "${MOJOLEARN_DEVPOD_OWN_NVIDIA:-0}" = 1 ] \
            || die "lanes do not rent NVIDIA pods: use the shared pods (tools/nvidia_central.sh sync/submit; the pods come up with tools/nvidia_central.sh up). MOJOLEARN_DEVPOD_OWN_NVIDIA=1 only with explicit approval"
        # The pod caps (the account's MOJOLEARN_RUNPOD_CAP and our own MAX_RUNPOD_PODS) are checked by
        # tools/rent_guard.sh in runpod_create, before every create attempt.
    fi
    mkdir -p "$D"
    BASE_SHA=$(git -C "$ROOT" rev-parse --verify "${BASE_REF:-origin/main}^{commit}") || die "no commit ${BASE_REF:-origin/main}"
    POD_ID=""; COST_HR=""; GPU=""
    if [ "$VENDOR" = nvidia ]; then
        load_key || die "no RunPod key (~/.mojolearn_runpod_key, mode 600)"
        PROVIDER=runpod
        runpod_create "${MOJOLEARN_DEVPOD_RETRY_MINUTES:-60}" "$NV_GPUS" "$NV_IMAGE" 0 \
            || { rm -rf "$D"; die "nothing was created"; }
    elif [ "$VENDOR" = cpu ]; then
        load_key || { rm -rf "$D"; die "no RunPod key (~/.mojolearn_runpod_key, mode 600)"; }
        PROVIDER=runpod
        runpod_create "${MOJOLEARN_DEVPOD_RETRY_MINUTES:-15}" "$CPU_FLAVORS" "$CPU_IMAGE" cpu \
            || { rm -rf "$D"; die "nothing was created"; }
    else
        load_key || { rm -rf "$D"; die "no RunPod key (~/.mojolearn_runpod_key, mode 600)"; }
        PROVIDER=runpod
        runpod_create "${MOJOLEARN_DEVPOD_AMD_RETRY_MINUTES:-15}" "$AMD_GPUS" "$AMD_IMAGE" 1 \
            || { rm -rf "$D"; die "nothing was created"; }
    fi
    if [ "${MOJOLEARN_DEVPOD_PROVISION_ONLY:-0}" = 1 ]; then
        say "box $POD_ID up on $PROVIDER, lease ${minutes} min; bootstrap deferred for harvester admission"
        exit 0
    fi
    say "box $POD_ID up on $PROVIDER (\$${COST_HR:-?}/hr, $GPU), lease ${minutes} min; installing pixi"
    bx 600 'command -v curl > /dev/null || { export DEBIAN_FRONTEND=noninteractive; apt-get update -qq > /dev/null && apt-get install -y -qq curl ca-certificates > /dev/null; }
[ -x /root/.pixi/bin/pixi ] || command -v pixi > /dev/null || curl -fsSL https://pixi.sh/install.sh | bash; mkdir -p /root/mojolearn' < /dev/null > "$D/bootstrap.log" 2>&1 \
        || say "pixi bootstrap failed; see $D/bootstrap.log"
    # an NVIDIA pod takes torch and its CUDA wheels from the PyTorch CDN (runpod_pod_lib.sh pod_pip_conf_script:
    # the extra index only adds a source; PyPI stays the index and every pin is unchanged; no nvidia-smi, no file)
    bx 60 "$(pod_pip_conf_script)" < /dev/null >> "$D/bootstrap.log" 2>&1 \
        || say "pip.conf not written; see $D/bootstrap.log"
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
        # PROVENANCE (2026-09-29): the box's HEAD is the merge base, so its own
        # `git rev-parse HEAD` names code that did not run. Record the synced
        # commit, whether the worktree had uncommitted changes, the base and the
        # patch's sha256 in .git/devpod_synced (outside the tree), which
        # tools/bench_board.py reads (repo_sync).
        _psha=$( (sha256sum 2>/dev/null || shasum -a 256) < "$TMPD/sync.patch" | cut -d' ' -f1)
        # worktree_dirty: MODIFIED TRACKED files only (the code is not exactly the commit); untracked files (a box-local/
        # script run by name) are shipped too but leave the commit's tracked bytes exact: counted apart, worktree_untracked
        _dirty=0; [ -z "$(git -C "$wt" status --porcelain --untracked-files=no 2>/dev/null | grep -vE '\.(so|dylib|metallib)$')" ] || _dirty=1
        _untracked=$(git -C "$wt" ls-files -o --exclude-standard 2>/dev/null | grep -cvE '\.(so|dylib|metallib)$' || true)
        printf 'commit=%s\nworktree_dirty=%s\nworktree_untracked=%s\nbase=%s\npatch_sha256=%s\npatch_bytes=%s\nsynced=%s\n' \
            "$(git -C "$wt" rev-parse HEAD)" "$_dirty" "$_untracked" "$base" "$_psha" "$(wc -c < "$TMPD/sync.patch" | tr -d ' ')" \
            "$(date -u +%FT%TZ)" | bx 60 "cat > $BOX_DIR/.git/devpod_synced" || die "provenance upload failed"
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
    # The full sync records the synced commit too, as the patch sync does: without it a record left by an earlier
    # patch sync would name the wrong commit to every job that reads it (tools/ops/tree_commit.sh).
    # worktree_dirty: MODIFIED TRACKED files only (the code is not exactly the commit); untracked files (a box-local/
    # script run by name) are shipped too but leave the commit's tracked bytes exact: counted apart, worktree_untracked
    _dirty=0; [ -z "$(git -C "$wt" status --porcelain --untracked-files=no 2>/dev/null | grep -vE '\.(so|dylib|metallib)$')" ] || _dirty=1
    _untracked=$(git -C "$wt" ls-files -o --exclude-standard 2>/dev/null | grep -cvE '\.(so|dylib|metallib)$' || true)
    printf 'commit=%s\nworktree_dirty=%s\nworktree_untracked=%s\nbase=%s\npatch_sha256=none-full-sync\npatch_bytes=0\nsynced=%s\n' \
        "$(git -C "$wt" rev-parse HEAD)" "$_dirty" "$_untracked" "$base" "$(date -u +%FT%TZ)" | bx 60 "cat > $BOX_DIR/.git/devpod_synced" \
        || die "provenance upload failed"
    say "synced $(cd "$wt" && git rev-parse --short HEAD)+worktree -> $POD_ID:$BOX_DIR (box HEAD = merge base $(git -C "$wt" rev-parse --short "$base"))"
    ;;
run)
    load_state
    _res=''; _persist=1
    while :; do
        case "${1:-}" in
            --results) _res="${2:?--results DIR}"; shift 2 ;;
            --no-persist) _persist=0; shift ;;
            *) break ;;
        esac
    done
    [ $# -gt 0 ] || die "run needs a command"
    # every run persists its results dir to R2 from the box (tools/box_persist_run.sh): default box-out/<tag> in
    # the tree, the command sees it as $MOJOLEARN_RESULTS_DIR; an ssh drop leaves the job running and uploading
    _cmd="$*"
    if [ "$_persist" = 1 ]; then
        bp_prepare "dp-$KEY" dev_pod "$TMPD/persist.sh" --slots 2 --every-min 30 --days 3 || die "$BP_MINT_FAILED"
        _staged=$(bx 120 "bash -s" < "$TMPD/persist.sh") || true
        printf '%s\n' "$_staged" | grep -q '^BOX_PERSIST_STAGED ' || die "could not stage the job's persistence on $KEY; nothing ran"
        _cmd=$(bp_wrap "${_res:-box-out/$BP_TAG}" "$_cmd")
    else
        bp_opt_out dev_pod "run $KEY $*" || exit 1
    fi
    # shellcheck disable=SC2086
    _lk=""; [ -z "$CENTRAL_KEY" ] || [ "$KEY" != "$CENTRAL_KEY" ] \
        || _lk="mkdir -p /var/lock/mojolearn-central; exec 9>/var/lock/mojolearn-central/gpu0.lock; flock 9; echo 'lane=$KEY (dev_pod run)' > /var/lock/mojolearn-central/gpu0.owner; "
    ssh $SSH_OPTS $SSH_TARGET "$(box_cmd "$_lk$BOX_PATH; ${BOX_ENV:+export $BOX_ENV; }cd $BOX_DIR && $_cmd")"
    ;;
extend)
    load_state; minutes="${1:-120}"
    if [ -n "$CENTRAL_KEY" ] && [ "$KEY" = "$CENTRAL_KEY" ]; then
        _left=$(( (CENTRAL_LEASE_END - $(now)) / 60 ))
        [ "$minutes" -ge "$_left" ] || { say "$KEY is the central AMD box: extending to its lease end ($_left min), not $minutes"; minutes=$_left; }
    fi
    load_key || die "no RunPod key"
    MOJOLEARN_LEASE_DIR="$D/lease" with_timeout 180 sh "$ROOT/tools/runpod_guard.sh" extend "$POD_ID" "$SSH_TARGET" "$minutes" \
        || die "extend REFUSED; the pod keeps its old lease"
    arm_mac_deadman $(( minutes * 60 + 600 ))
    say "extended $POD_ID by ${minutes} min (on-box lease and Mac dead-man)"
    ;;
down)
    load_state
    [ -z "$CENTRAL_KEY" ] || [ "$KEY" != "$CENTRAL_KEY" ] || [ "${MOJOLEARN_CENTRAL_DOWN:-0}" = 1 ] \
        || die "$KEY is the central AMD box every lane shares (tools/amd_central.sh); refusing (MOJOLEARN_CENTRAL_DOWN=1 overrides)"
    # logs and results go home and to R2 before the delete; a reachable box that fails to archive is not deleted
    # (MOJOLEARN_DOWN_NO_ARCHIVE=1 skips, for a box whose results are already home)
    if [ "${MOJOLEARN_DOWN_NO_ARCHIVE:-0}" != 1 ]; then
        # shellcheck disable=SC2086
        _arc=0; bash "$ROOT/tools/box_archive.sh" "$KEY" $SSH_TARGET || _arc=$?
        [ "$_arc" = 0 ] || [ "$_arc" = 3 ] || die "$KEY: archive failed (box_archive.sh rc=$_arc); not deleting. MOJOLEARN_DOWN_NO_ARCHIVE=1 overrides"
    fi
    load_key || die "no RunPod key"
    delete_pod "$POD_ID"
    verify_gone "$POD_ID" || die "$POD_ID NOT CONFIRMED GONE; the Mac dead-man stays armed"
    _p=$(cat "$D/deadman.pid" 2>/dev/null || true)
    [ -z "$_p" ] || { pkill -P "$_p" 2>/dev/null || true; kill "$_p" 2>/dev/null || true; }
    mv "$D" "$D.down-$(date -u +%Y%m%dT%H%M%SZ)"
    say "$KEY box $POD_ID down"
    ;;
seed)
    # seed <key> <commit> <box dir> [--checkout]: the commit into the repo at <box dir> (seed_git: GitHub first, else
    # shipped from this checkout); --checkout also moves its working tree there. tools/cpu_central.sh check calls it.
    load_state
    _c="${1:?seed <key> <commit> <box dir> [--checkout]}"; BOX_DIR="${2:?seed <key> <commit> <box dir> [--checkout]}"
    case "$BOX_DIR" in /root/*) ;; *) die "seed: the box dir must be under /root" ;; esac
    _sha=$(git -C "$ROOT" rev-parse --verify "$_c^{commit}") || die "no commit $_c in this checkout"
    seed_git "$_sha"
    if [ "${3:-}" = --checkout ]; then
        bx 900 "cd $BOX_DIR && git reset -q --hard $_sha && git rev-parse HEAD" < /dev/null | grep -qx "$_sha" \
            || die "could not check out $_sha in $BOX_DIR"
    fi
    say "seeded $(git -C "$ROOT" rev-parse --short "$_sha") into $KEY:$BOX_DIR${3:+ (checked out)}"
    echo "SEEDED $_sha"
    ;;
list)
    for s in "$STATE_ROOT"/*/state.env; do
        [ -f "$s" ] || continue
        case "$s" in *.down-*) continue ;; esac
        ( . "$s"; printf '%-16s %-9s %-22s %-34s $%s/hr\n' "$(basename "$(dirname "$s")")" "${PROVIDER:-runpod}" "$POD_ID" "$GPU" "${COST_HR:-?}" )
    done
    ;;
esac
