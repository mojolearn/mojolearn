#!/bin/sh
# tools/trees_leg.sh with one change: TREES_LEG_GPU_COUNT rents a node with
# more than one GPU, which the NCCL determinism probe needs and the trees lane
# never did. Everything else -- the watchdog armed before any work, the pod
# named by TREES_LEG_NAME, the reap that verifies HTTP 404, the key never in
# an argv -- is trees_leg.sh's and is unchanged. Kept as a separate file so a
# trees lane running out of this same checkout is never edited underneath.
#
# The trees lane's RunPod session: one guarded pod that STAYS UP between
# builds, iterated over ssh/rsync instead of re-shipped archives.
#
#   MOJOLEARN_RUNPOD_KEY_FILE=~/.mojolearn_runpod_key tools/trees_leg.sh rent [--gpu "NVIDIA L40S"] [--minutes 60]
#   tools/trees_leg.sh pods                 list this account's pods (name, id, gpu, status)
#   tools/trees_leg.sh ssh <cmd...>         run on the current pod
#   tools/trees_leg.sh push <path...>       rsync working-tree paths onto the pod's checkout
#   tools/trees_leg.sh pull <remote> <local>
#   tools/trees_leg.sh extend [minutes]     re-arm the on-pod watchdog (default 60)
#   tools/trees_leg.sh lease                minutes left on the current lease
#   tools/trees_leg.sh reap                 terminate the current pod and VERIFY it is gone
#
# Safety is tools/runpod_guard.sh's: the on-pod watchdog is armed BEFORE any
# work and terminates the pod through the API at the deadline; `extend` is
# the only way to keep the box. The key is read from
# MOJOLEARN_RUNPOD_KEY_FILE inside this script, handed to curl through a 0600
# config file and to the guard through the environment; it is never printed
# and never in an argv. Pods named samba-* belong to another program and are
# never touched: every action here is by the pod id this script created.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 9
STATE_DIR="${TREES_LEG_STATE:-$REPO/bench/results/trees_identical/pod}"
export MOJOLEARN_LEASE_DIR="${MOJOLEARN_LEASE_DIR:-$STATE_DIR/leases}"
mkdir -p "$STATE_DIR" "$MOJOLEARN_LEASE_DIR"
RP_HOST="https://rest.runpod.io"
IMAGE="${TREES_LEG_IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
TMPD=$(mktemp -d "${TMPDIR:-/tmp}/trees_leg.XXXXXX")
CURLRC="$TMPD/curlrc"
trap 'rm -rf "$TMPD"' EXIT INT TERM

die() { printf '%s\n' "$*" >&2; exit 1; }
say() { printf '[%s trees-leg] %s\n' "$(date +%T)" "$*"; }

load_key() {
    _kf="${MOJOLEARN_RUNPOD_KEY_FILE:-$HOME/.mojolearn_runpod_key}"
    [ -f "$_kf" ] || die "key file $_kf does not exist"
    _perm=$(stat -f '%OLp' "$_kf" 2>/dev/null || stat -c '%a' "$_kf" 2>/dev/null || echo "?")
    [ "$_perm" = "600" ] || die "key file $_kf is mode $_perm, must be 600"
    RUNPOD_API_KEY=$(cat "$_kf")
    [ -n "$RUNPOD_API_KEY" ] || die "key file is empty"
    export RUNPOD_API_KEY
    ( umask 077
      printf 'header = "Authorization: Bearer %s"\nsilent\nshow-error\n' \
          "$RUNPOD_API_KEY" > "$CURLRC" )
}

rp_call() {
    _m="$1"; _p="$2"; _d="${3:-}"
    RP_BODY="$TMPD/rp.body"; : > "$RP_BODY"
    if [ -n "$_d" ]; then
        RP_CODE=$(curl -K "$CURLRC" -o "$RP_BODY" -w '%{http_code}' -X "$_m" \
            -H 'Content-Type: application/json' --data-binary "@$_d" \
            "$RP_HOST$_p" 2>>"$TMPD/curl.err") || RP_CODE=000
    else
        RP_CODE=$(curl -K "$CURLRC" -o "$RP_BODY" -w '%{http_code}' -X "$_m" \
            "$RP_HOST$_p" 2>>"$TMPD/curl.err") || RP_CODE=000
    fi
    return 0
}

rp_json() {
    python3 - "$RP_BODY" "$1" <<'PYEOF' 2>/dev/null || true
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
if isinstance(d, list):
    d = {"items": d}
try:
    v = eval(sys.argv[2], {"__builtins__": {"str": str, "len": len}}, {"d": d})
except Exception:
    sys.exit(0)
sys.stdout.write("" if v is None else str(v))
PYEOF
}

current_pod() {
    [ -s "$STATE_DIR/pod_id.txt" ] || die "no current pod (no $STATE_DIR/pod_id.txt)"
    POD_ID=$(cat "$STATE_DIR/pod_id.txt")
    SSH_TARGET=$(cat "$STATE_DIR/ssh_target.txt")
}

pod_ssh() {
    # shellcheck disable=SC2086
    ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 \
        -o ServerAliveInterval=30 -o BatchMode=yes $SSH_TARGET "$@"
}

cmd_pods() {
    load_key
    rp_call GET /v1/pods
    [ "$RP_CODE" = "200" ] || die "GET /v1/pods -> HTTP $RP_CODE"
    rp_json "'\n'.join('%s %s %s %s' % (p.get('name'), p.get('id'), (p.get('machine') or {}).get('gpuTypeId') or p.get('gpuTypeId') or '?', p.get('desiredStatus')) for p in (d.get('items') or d.get('pods') or d.get('data') or []))"
    echo
}

cmd_rent() {
    GPU="NVIDIA L40S"; MINUTES=60
    while [ $# -gt 0 ]; do
        case "$1" in
            --gpu) shift; GPU="$1" ;;
            --minutes) shift; MINUTES="$1" ;;
            *) die "unknown rent option $1" ;;
        esac
        shift
    done
    load_key
    [ -s "$STATE_DIR/pod_id.txt" ] && die "a pod is already recorded in $STATE_DIR ($(cat "$STATE_DIR/pod_id.txt")); reap it first"
    rp_call GET /v1/pods
    [ "$RP_CODE" = "200" ] || die "pre-flight GET /v1/pods -> HTTP $RP_CODE; not renting"
    # TREES_LEG_NAME names this lane's pods (default mojolearn-trees), so two
    # trees lanes renting at once each check only their own prefix.
    # TREES_LEG_NAME_PREFIX (the symmetric-arms lane's spelling, a full
    # prefix with its trailing dash) still wins when set.
    _prefix="${TREES_LEG_NAME_PREFIX:-${TREES_LEG_NAME:-mojolearn-trees}-}"
    _mine=$(rp_json "','.join(str(p.get('id','')) for p in (d.get('items') or d.get('pods') or d.get('data') or []) if str(p.get('name','')).startswith('$_prefix'))")
    [ -z "$_mine" ] || die "this lane already has pod(s) up: $_mine; reap first"
    STAMP=$(date -u +%Y-%m-%d_%H%M%S)
    POD_NAME="$_prefix$STAMP"
    # TREES_LEG_CUDA_VERSIONS (comma list) narrows the host CUDA versions.
    # Mojo 1.0.0's GPU runtime refuses NVIDIA drivers below 580 (CUDA 13.0),
    # so a leg that fits on the device passes TREES_LEG_CUDA_VERSIONS=13.0.
    python3 - "$TMPD/create.json" "$POD_NAME" "$IMAGE" "$GPU" "${TREES_LEG_CUDA_VERSIONS:-12.4,12.5,12.6,12.7,12.8,12.9,13.0}" "${TREES_LEG_GPU_COUNT:-1}" <<'PY'
import json, sys
out, name, image, gpu, cudas, gpucount = sys.argv[1:]
req = {"name": name, "imageName": image, "gpuTypeIds": [gpu], "gpuCount": int(gpucount),
       "cloudType": "SECURE", "containerDiskInGb": 80, "volumeInGb": 0,
       "ports": ["22/tcp"], "supportPublicIp": True, "interruptible": False,
       "allowedCudaVersions": [c.strip() for c in cudas.split(",") if c.strip()]}
open(out, "w").write(json.dumps(req, indent=2) + "\n")
PY
    say "creating $POD_NAME (${TREES_LEG_GPU_COUNT:-1}x $GPU, $IMAGE); THE BILL STARTS HERE"
    rp_call POST /v1/pods "$TMPD/create.json"
    cp "$RP_BODY" "$STATE_DIR/create_response.json" 2>/dev/null || true
    POD_ID=$(rp_json "d.get('id') or (d.get('pod') or {}).get('id') or ''")
    if [ -z "$POD_ID" ]; then
        rp_call GET /v1/pods
        POD_ID=$(rp_json "([str(p.get('id','')) for p in (d.get('items') or d.get('pods') or d.get('data') or []) if p.get('name')=='$POD_NAME'] or [''])[0]")
        [ -n "$POD_ID" ] || die "create FAILED (HTTP $RP_CODE, see $STATE_DIR/create_response.json); no pod named $POD_NAME exists. CHECK THE CONSOLE."
        say "create response unparsed but pod $POD_ID exists by name; adopting it"
    fi
    echo "$POD_ID" > "$STATE_DIR/pod_id.txt"
    echo "$POD_NAME" > "$STATE_DIR/pod_name.txt"
    echo "$GPU" > "$STATE_DIR/gpu.txt"
    say "pod $POD_ID created; waiting for ssh (600 s cap, then terminate)"
    _deadline=$(( $(date -u +%s) + 600 ))
    SSH_TARGET=""
    while [ "$(date -u +%s)" -lt "$_deadline" ]; do
        rp_call GET "/v1/pods/$POD_ID"
        _ip=$(rp_json "d.get('publicIp') or ''")
        _port=$(rp_json "str((d.get('portMappings') or {}).get('22') or '')")
        if [ -n "$_ip" ] && [ -n "$_port" ]; then
            SSH_TARGET="-p $_port root@$_ip"
            # shellcheck disable=SC2086
            if ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o BatchMode=yes \
                   $SSH_TARGET 'echo SSH-OK' 2>/dev/null | grep -q SSH-OK; then
                break
            fi
        fi
        sleep 10
    done
    if [ -z "$SSH_TARGET" ] || ! pod_ssh 'echo SSH-OK' 2>/dev/null | grep -q SSH-OK; then
        say "no ssh within the cap; terminating $POD_ID"
        rp_call DELETE "/v1/pods/$POD_ID"
        rm -f "$STATE_DIR/pod_id.txt"
        die "READY TIMEOUT; pod deleted (HTTP $RP_CODE)"
    fi
    echo "$SSH_TARGET" > "$STATE_DIR/ssh_target.txt"
    say "ssh target: $SSH_TARGET"
    say "arming the $MINUTES-minute on-pod watchdog BEFORE any work"
    if ! tools/runpod_guard.sh arm "$POD_ID" "$SSH_TARGET" "$MINUTES" > "$STATE_DIR/arm.log" 2>&1; then
        cat "$STATE_DIR/arm.log"
        rp_call DELETE "/v1/pods/$POD_ID"
        rm -f "$STATE_DIR/pod_id.txt"
        die "ARM REFUSED; pod deleted (HTTP $RP_CODE)"
    fi
    tail -3 "$STATE_DIR/arm.log"
    pod_ssh 'nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; uname -r; nvcc --version 2>/dev/null | tail -1' | tee "$STATE_DIR/box.txt"
    cmd_ship
}

cmd_ship() {
    current_pod
    say "shipping git archive of HEAD ($(git rev-parse --short HEAD)), source only"
    git archive --format=tar HEAD -- . ':!bench/results' ':!mamba/corpus' ':!bench/oracle*' ':!archive' ':!upstream' ':!docs' ':!paper' \
        | gzip > "$TMPD/src.tgz"
    say "  $(wc -c < "$TMPD/src.tgz" | tr -d ' ') bytes compressed"
    pod_ssh 'rm -rf /root/mojolearn && mkdir -p /root/mojolearn' > /dev/null
    pod_ssh 'cd /root/mojolearn && tar xzf -' < "$TMPD/src.tgz"
    git rev-parse HEAD | pod_ssh 'cat > /root/mojolearn/SHIPPED_COMMIT.txt'
    say "shipped"
}

cmd_ssh() { current_pod; pod_ssh "$@"; }

cmd_push() {
    current_pod
    [ $# -gt 0 ] || die "push needs paths"
    # shellcheck disable=SC2086
    rsync -az --relative -e "ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes $(echo "$SSH_TARGET" | sed 's/ root@.*//')" \
        "$@" "root@$(echo "$SSH_TARGET" | sed 's/.*root@//'):/root/mojolearn/"
}

cmd_pull() {
    current_pod
    [ $# -eq 2 ] || die "pull <remote> <local>"
    mkdir -p "$2"
    rsync -az --exclude '*.so' --exclude '*.npz' --exclude '*.gz' --exclude '*.csv' \
        -e "ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes $(echo "$SSH_TARGET" | sed 's/ root@.*//')" \
        "root@$(echo "$SSH_TARGET" | sed 's/.*root@//'):$1" "$2"
}

cmd_extend() {
    load_key; current_pod
    tools/runpod_guard.sh extend "$POD_ID" "$SSH_TARGET" "${1:-60}"
}

cmd_lease() {
    current_pod
    tools/runpod_guard.sh check "$POD_ID" 2>&1 || true
    pod_ssh 'echo "watchdog pid alive: $(kill -0 $(cat /tmp/mojolearn-lease.pid) 2>/dev/null && echo yes || echo NO)"' 2>/dev/null || echo "(pod unreachable)"
}

cmd_reap() {
    load_key; current_pod
    say "terminating $POD_ID"
    rp_call DELETE "/v1/pods/$POD_ID"
    say "  DELETE -> HTTP $RP_CODE"
    rm -f "$MOJOLEARN_LEASE_DIR/$POD_ID.lease"
    _i=1
    while [ "$_i" -le 8 ]; do
        rp_call GET "/v1/pods/$POD_ID"
        if [ "$RP_CODE" = "404" ]; then
            say "VERIFIED: $POD_ID is gone (HTTP 404)"
            mv "$STATE_DIR/pod_id.txt" "$STATE_DIR/pod_id.reaped.$(date -u +%Y%m%dT%H%M%SZ).txt"
            return 0
        fi
        _st=$(rp_json "d.get('desiredStatus') or ''")
        case "$_st" in TERMINATED|EXITED) say "VERIFIED: status $_st"; mv "$STATE_DIR/pod_id.txt" "$STATE_DIR/pod_id.reaped.txt"; return 0 ;; esac
        say "  still '$_st' (HTTP $RP_CODE), attempt $_i/8"
        sleep 10; _i=$((_i + 1))
    done
    die "THE POD MAY STILL BE BILLING: $POD_ID. Check https://console.runpod.io/pods"
}

case "${1:-}" in
    rent)   shift; cmd_rent "$@" ;;
    pods)   cmd_pods ;;
    ship)   cmd_ship ;;
    ssh)    shift; cmd_ssh "$@" ;;
    push)   shift; cmd_push "$@" ;;
    pull)   shift; cmd_pull "$@" ;;
    extend) shift; cmd_extend "$@" ;;
    lease)  cmd_lease ;;
    reap)   cmd_reap ;;
    *) sed -n '2,20p' "$0"; exit 2 ;;
esac
