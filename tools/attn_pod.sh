#!/bin/sh
# ONE RunPod box for the fused-attention lane: create, arm the one-hour
# self-kill, ship a commit, run jobs detached, fetch, terminate and VERIFY.
#
#   MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
#       sh tools/attn_pod.sh up [gpu-name]        create + arm + bootstrap pixi
#   sh tools/attn_pod.sh ship <sha> <remote-dir>  git archive a commit onto the box
#   sh tools/attn_pod.sh put <local> <remote>     scp a file up
#   sh tools/attn_pod.sh ssh '<command>'          run a command (attached)
#   sh tools/attn_pod.sh run <name> <script.sh>   run a script DETACHED; writes
#                                                 /root/jobs/<name>.{log,done,rc}
#   sh tools/attn_pod.sh wait <name> [seconds]    block until the done-file exists
#   sh tools/attn_pod.sh fetch <remote> <local>   scp -r a directory down
#   sh tools/attn_pod.sh extend [minutes]         re-arm the lease (a decision)
#   sh tools/attn_pod.sh down                     terminate and verify it is gone
#
# The safety story is `tools/gemm_remote_leg.sh`'s, reduced: the key is read
# from a 0600 file, reaches curl through `-K` and the guard through the
# environment of this process only, and is never in an argv or printed;
# `tools/runpod_guard.sh arm` installs the on-pod watchdog BEFORE any work
# and refuses without a key; `down` asks the API after the DELETE and prints
# what it got. The pod's state lives in $MOJOLEARN_ATTN_POD_STATE (default
# /tmp/mojolearn-attn-pod.env) so a later shell, or a later agent, can find
# the box. This script never touches a pod it did not create by name.
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
STATE="${MOJOLEARN_ATTN_POD_STATE:-/tmp/mojolearn-attn-pod.env}"
RP_HOST="https://rest.runpod.io"
IMAGE="${MOJOLEARN_ATTN_POD_IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
MINUTES="${MOJOLEARN_ATTN_POD_MINUTES:-60}"
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -o ServerAliveInterval=30 -o BatchMode=yes"

die() { printf '%s\n' "$*" >&2; exit 1; }
say() { printf '[%s attn-pod] %s\n' "$(date +%T)" "$*"; }

load_key() {
    _kf="${MOJOLEARN_RUNPOD_KEY_FILE:-$HOME/.mojolearn_runpod_key}"
    [ -f "$_kf" ] || die "key file $_kf does not exist"
    RUNPOD_API_KEY=$(cat "$_kf")
    [ -n "$RUNPOD_API_KEY" ] || die "key file is empty"
    export RUNPOD_API_KEY
    CURLRC=$(mktemp "${TMPDIR:-/tmp}/attn-pod.XXXXXX")
    chmod 600 "$CURLRC"
    printf 'header = "Authorization: Bearer %s"\nsilent\nshow-error\n' "$RUNPOD_API_KEY" > "$CURLRC"
    trap 'rm -f "$CURLRC" "$RP_BODY"' EXIT INT TERM
}

RP_BODY="${TMPDIR:-/tmp}/attn-pod-body.$$"
rp() {  # method path [json-file] -> body in $RP_BODY, code in RP_CODE
    _m="$1"; _p="$2"; _d="${3:-}"
    : > "$RP_BODY"
    if [ -n "$_d" ]; then
        RP_CODE=$(curl -K "$CURLRC" -o "$RP_BODY" -w '%{http_code}' -X "$_m" \
            -H 'Content-Type: application/json' --data-binary "@$_d" "$RP_HOST$_p") || RP_CODE=000
    else
        RP_CODE=$(curl -K "$CURLRC" -o "$RP_BODY" -w '%{http_code}' -X "$_m" "$RP_HOST$_p") || RP_CODE=000
    fi
}

load_state() {
    [ -f "$STATE" ] || die "no pod state at $STATE (run 'up' first)"
    # shellcheck disable=SC1090
    . "$STATE"
    [ -n "${POD_ID:-}" ] && [ -n "${SSH_PORT:-}" ] && [ -n "${SSH_HOST:-}" ] || die "state file $STATE is incomplete"
    SSH_TARGET="-p $SSH_PORT root@$SSH_HOST"
}

pod_scp_up() {   # local remote
    # shellcheck disable=SC2086
    scp $SSH_OPTS -P "$SSH_PORT" "$1" "root@$SSH_HOST:$2" >/dev/null
}

pod_scp_down() { # remote local
    # shellcheck disable=SC2086
    scp -r $SSH_OPTS -P "$SSH_PORT" "root@$SSH_HOST:$1" "$2" >/dev/null
}

pod_ssh() {
    # shellcheck disable=SC2086
    ssh $SSH_OPTS $SSH_TARGET "$@"
}

cmd_up() {
    load_key
    [ -f "$STATE" ] && die "a pod state already exists at $STATE; run 'down' first (one pod at a time)"
    GPU="${1:-NVIDIA L40S}"
    STAMP=$(date +%Y%m%d_%H%M%S)
    NAME="mojolearn-attn-$STAMP"
    REQ=$(mktemp "${TMPDIR:-/tmp}/attn-pod-req.XXXXXX")
    python3 - "$REQ" "$NAME" "$IMAGE" "$GPU" <<'PY'
import json, sys
out, name, image, gpu = sys.argv[1:]
req = {"name": name, "imageName": image, "gpuTypeIds": [gpu], "gpuCount": 1,
       "cloudType": "SECURE", "containerDiskInGb": 60, "volumeInGb": 0,
       "ports": ["22/tcp"], "supportPublicIp": True, "interruptible": False,
       "allowedCudaVersions": ["13.0"]}
open(out, "w").write(json.dumps(req))
PY
    say "creating $NAME ($GPU, $IMAGE). THE BILL STARTS HERE."
    rp POST /v1/pods "$REQ"; rm -f "$REQ"
    POD_ID=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("id") or "")' "$RP_BODY" 2>/dev/null || true)
    if [ -z "$POD_ID" ]; then
        head -c 2000 "$RP_BODY" >&2
        die "create failed (HTTP $RP_CODE). CHECK https://console.runpod.io/pods for a pod named $NAME."
    fi
    printf 'POD_ID=%s\nPOD_NAME=%s\n' "$POD_ID" "$NAME" > "$STATE"
    say "pod $POD_ID created; waiting for ssh (10 min cap)"
    _deadline=$(( $(date +%s) + 600 ))
    SSH_PORT=""; SSH_HOST=""; _ok=0
    while [ "$(date +%s)" -lt "$_deadline" ]; do
        rp GET "/v1/pods/$POD_ID"
        _tgt=$(python3 -c '
import json,sys
d=json.load(open(sys.argv[1]))
ip=d.get("publicIp") or ""
pm=d.get("portMappings") or {}
port=""
if isinstance(pm, dict):
    port=str(pm.get("22") or pm.get(22) or "")
else:
    for m in pm:
        if str(m.get("privatePort",""))=="22": port=str(m.get("publicPort",""))
print("%s %s" % (port, ip) if ip and port else "")' "$RP_BODY" 2>/dev/null || true)
        if [ -n "$_tgt" ]; then
            SSH_PORT=${_tgt%% *}; SSH_HOST=${_tgt##* }
            SSH_TARGET="-p $SSH_PORT root@$SSH_HOST"
            # shellcheck disable=SC2086
            if ssh $SSH_OPTS $SSH_TARGET 'echo SSH-OK' 2>/dev/null | grep -q SSH-OK; then
                _ok=1; break
            fi
        fi
        sleep 10
    done
    if [ "$_ok" != 1 ]; then
        say "ssh never answered; terminating"
        rp DELETE "/v1/pods/$POD_ID" || true
        rm -f "$STATE"
        die "READY TIMEOUT; the pod was deleted (HTTP $RP_CODE). Check the console."
    fi
    printf 'SSH_PORT=%s\nSSH_HOST=%s\n' "$SSH_PORT" "$SSH_HOST" >> "$STATE"
    say "ssh target: $SSH_TARGET"
    say "ARMING THE LEASE ($MINUTES minutes) BEFORE ANY WORK"
    if ! "$HERE/tools/runpod_guard.sh" arm "$POD_ID" "$SSH_TARGET" "$MINUTES"; then
        say "ARM REFUSED: terminating the box"
        rp DELETE "/v1/pods/$POD_ID" || true
        rm -f "$STATE"
        die "arm refused; pod deleted (HTTP $RP_CODE)"
    fi
    say "bootstrapping pixi on the box"
    pod_ssh 'mkdir -p /root/jobs; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > /root/jobs/gpu.txt 2>&1; cat /root/jobs/gpu.txt;
        if [ ! -x "$HOME/.pixi/bin/pixi" ]; then curl -fsSL --max-time 300 https://pixi.sh/install.sh | sh > /root/jobs/pixi_install.log 2>&1 || echo PIXI_INSTALL_FAILED; fi;
        "$HOME/.pixi/bin/pixi" --version'
    say "up. State in $STATE"
}

archive_paths() {  # every top-level entry except the heavy ones, plus the light bench/ and mamba/ parts
    _sha="$1"
    git -C "$HERE" ls-tree --name-only "$_sha" | grep -v -E '^(bench|mamba|archive|upstream|parked|docs)$'
    git -C "$HERE" ls-tree --name-only "$_sha" bench/ | grep -v -E '^bench/(results|oracle)'
    git -C "$HERE" ls-tree --name-only "$_sha" mamba/ | grep -v -E '^mamba/corpus$'
}

cmd_ship() {
    load_state
    _sha="$1"; _dst="$2"
    _tgz=$(mktemp "${TMPDIR:-/tmp}/attn-src.XXXXXX")
    # shellcheck disable=SC2046
    git -C "$HERE" archive --format=tar.gz -o "$_tgz" "$_sha" -- $(archive_paths "$_sha")
    say "archive $(du -h "$_tgz" | cut -f1) of $_sha -> $_dst"
    pod_scp_up "$_tgz" /root/src.tgz
    pod_ssh "mkdir -p $_dst && tar xzf /root/src.tgz -C $_dst && rm -f /root/src.tgz && printf '%s\n' '$_sha' > $_dst/commit.txt && ls $_dst | head -3"
    rm -f "$_tgz"
}

cmd_put() {
    load_state
    pod_scp_up "$1" "$2"
}

cmd_run() {
    load_state
    _name="$1"; _script="$2"
    cmd_put "$_script" "/root/jobs/$_name.sh"
    pod_ssh "cd /root/jobs && rm -f $_name.done $_name.rc && nohup sh -c 'bash /root/jobs/$_name.sh > /root/jobs/$_name.log 2>&1; echo \$? > /root/jobs/$_name.rc; touch /root/jobs/$_name.done' > /dev/null 2>&1 &"
    say "job $_name started detached"
}

cmd_wait() {
    load_state
    _name="$1"; _secs="${2:-3000}"
    _deadline=$(( $(date +%s) + _secs ))
    while [ "$(date +%s)" -lt "$_deadline" ]; do
        if pod_ssh "test -f /root/jobs/$_name.done" 2>/dev/null; then
            say "job $_name done, rc $(pod_ssh "cat /root/jobs/$_name.rc" 2>/dev/null)"
            return 0
        fi
        sleep 20
    done
    say "job $_name NOT done after $_secs s"
    return 1
}

cmd_fetch() {
    load_state
    mkdir -p "$2"
    pod_scp_down "$1" "$2"
    say "fetched $1 -> $2"
}

cmd_extend() {
    load_key; load_state
    "$HERE/tools/runpod_guard.sh" extend "$POD_ID" "$SSH_TARGET" "${1:-60}"
}

cmd_down() {
    load_key; load_state
    say "terminating $POD_ID"
    rp DELETE "/v1/pods/$POD_ID" || true
    say "DELETE -> HTTP $RP_CODE"
    sleep 5
    rp GET "/v1/pods/$POD_ID"
    if [ "$RP_CODE" = 404 ] || grep -qi 'not found' "$RP_BODY"; then
        say "VERIFIED GONE (GET -> HTTP $RP_CODE)"
        rm -f "$STATE"
        return 0
    fi
    _st=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("desiredStatus") or d.get("status") or "?")' "$RP_BODY" 2>/dev/null || echo "?")
    say "GET after DELETE -> HTTP $RP_CODE status $_st"
    case "$_st" in
        TERMINATED|EXITED) say "VERIFIED terminated"; rm -f "$STATE"; return 0 ;;
    esac
    die "COULD NOT VERIFY the terminate; the on-pod watchdog still fires at the lease deadline. Check https://console.runpod.io/pods and re-run 'down'."
}

case "${1:-}" in
    up)     shift; cmd_up "$@" ;;
    ship)   shift; cmd_ship "$@" ;;
    put)    shift; cmd_put "$@" ;;
    ssh)    shift; load_state; pod_ssh "$@" ;;
    run)    shift; cmd_run "$@" ;;
    wait)   shift; cmd_wait "$@" ;;
    fetch)  shift; cmd_fetch "$@" ;;
    extend) shift; cmd_extend "$@" ;;
    down)   shift; cmd_down "$@" ;;
    *)      sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
