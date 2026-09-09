#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# One guarded RunPod pod for the Samba training lane, driven step by step:
#
#   tools/samba_training_leg.sh rent [--gpu "NVIDIA L40S"] [--minutes 60]
#   tools/samba_training_leg.sh ssh  <cmd...>          run a command on the pod
#   tools/samba_training_leg.sh ship <sha>             git archive of a commit -> /root/mojolearn
#   tools/samba_training_leg.sh sync                   rsync the worktree's changed files
#   tools/samba_training_leg.sh fetch <remote> <local> scp -r back
#   tools/samba_training_leg.sh extend [minutes]       re-arm the on-pod watchdog
#   tools/samba_training_leg.sh terminate              DELETE the pod and verify
#
# The key is read from MOJOLEARN_RUNPOD_KEY_FILE (mode 600) into this
# process only, goes to curl through a 0600 config file, and reaches the pod
# on ssh stdin through tools/runpod_guard.sh arm. It is never printed and
# never in an argv. The pod is named samba-training-* and this script only
# ever touches the pod id recorded in its own state file; pods named
# samba-* or samba-gate-* belong to another program and are never listed
# for action here.
set -eu
REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$REPO"
STATE="${MOJOLEARN_SAMBA_LEG_STATE:-$REPO/bench/results/runpod_leases/samba-training.state}"
RP_HOST="https://rest.runpod.io"
IMAGE="${MOJOLEARN_SAMBA_LEG_IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
PY=python3

die() { echo "samba_training_leg: $*" >&2; exit 1; }

load_key() {
    _kf="${MOJOLEARN_RUNPOD_KEY_FILE:-}"
    [ -n "$_kf" ] || die "set MOJOLEARN_RUNPOD_KEY_FILE"
    [ -f "$_kf" ] || die "$_kf does not exist"
    _perm=$(stat -f '%OLp' "$_kf" 2>/dev/null || stat -c '%a' "$_kf" 2>/dev/null || echo "?")
    [ "$_perm" = "600" ] || die "key file must be mode 600"
    RUNPOD_API_KEY=$(cat "$_kf")
    export RUNPOD_API_KEY
    CURLRC="${TMPDIR:-/tmp}/samba-leg-$$.curlrc"
    ( umask 077; printf 'header = "Authorization: Bearer %s"\nsilent\nshow-error\n' "$RUNPOD_API_KEY" > "$CURLRC" )
    trap 'rm -f "$CURLRC"' EXIT INT TERM
}

rp() {
    _m="$1"; _p="$2"; _d="${3:-}"
    RP_BODY="${TMPDIR:-/tmp}/samba-leg-$$.body"
    if [ -n "$_d" ]; then
        RP_CODE=$(curl -K "$CURLRC" -o "$RP_BODY" -w '%{http_code}' -X "$_m" \
            -H 'Content-Type: application/json' --data-binary "@$_d" "$RP_HOST$_p") || RP_CODE=000
    else
        RP_CODE=$(curl -K "$CURLRC" -o "$RP_BODY" -w '%{http_code}' -X "$_m" "$RP_HOST$_p") || RP_CODE=000
    fi
}

rp_json() {
    "$PY" - "$RP_BODY" "$1" <<'PYEOF' 2>/dev/null || true
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
try:
    v = eval(sys.argv[2], {"__builtins__": {"str": str}}, {"d": d})
    if v is not None:
        print(v)
except Exception:
    pass
PYEOF
}

load_state() {
    [ -f "$STATE" ] || die "no state file $STATE; rent first"
    # shellcheck disable=SC1090
    . "$STATE"
}

pod_ssh() {
    # shellcheck disable=SC2086
    ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -o ServerAliveInterval=30 \
        -o BatchMode=yes $SSH_TARGET "$@"
}

cmd_rent() {
    GPU="NVIDIA L40S"; MINUTES=60
    while [ $# -gt 0 ]; do
        case "$1" in
            --gpu) shift; GPU="$1" ;;
            --minutes) shift; MINUTES="$1" ;;
            *) die "unknown option $1" ;;
        esac
        shift
    done
    [ "$MINUTES" -le 60 ] || die "one hour is the cap"
    [ ! -f "$STATE" ] || die "state file exists ($STATE); terminate first"
    load_key
    mkdir -p "$(dirname "$STATE")"
    STAMP=$(date -u +%Y%m%dT%H%M%SZ)
    NAME="samba-training-$STAMP"
    REQ="${TMPDIR:-/tmp}/samba-leg-$$.create.json"
    "$PY" - "$REQ" "$NAME" "$IMAGE" "$GPU" <<'PYEOF'
import json, sys
out, name, image, gpu = sys.argv[1:]
req = {"name": name, "imageName": image, "gpuTypeIds": [gpu], "gpuCount": 1,
       "cloudType": "SECURE", "containerDiskInGb": 60, "volumeInGb": 0,
       "ports": ["22/tcp"], "supportPublicIp": True, "interruptible": False,
       "allowedCudaVersions": ["12.4", "12.5", "12.6", "12.7", "12.8", "12.9", "13.0"]}
open(out, "w").write(json.dumps(req))
PYEOF
    echo "creating $NAME ($GPU, $IMAGE); THE BILL STARTS HERE"
    rp POST /v1/pods "$REQ"
    rm -f "$REQ"
    POD_ID=$(rp_json "d.get('id') or ''")
    if [ -z "$POD_ID" ]; then
        rp GET /v1/pods
        POD_ID=$(rp_json "([str(p.get('id','')) for p in (d if isinstance(d, list) else (d.get('items') or d.get('pods') or [])) if p.get('name')=='$NAME'] or [''])[0]")
        [ -n "$POD_ID" ] || die "create failed (HTTP $RP_CODE): $(cat "$RP_BODY")"
    fi
    echo "pod $POD_ID"
    printf "POD_ID='%s'\nPOD_NAME='%s'\n" "$POD_ID" "$NAME" > "$STATE"
    deadline=$(( $(date +%s) + 600 ))
    SSH_TARGET=""
    while [ "$(date +%s)" -lt "$deadline" ]; do
        rp GET "/v1/pods/$POD_ID"
        ip=$(rp_json "d.get('publicIp') or ''")
        port=$(rp_json "str((d.get('portMappings') or {}).get('22') or '')")
        if [ -n "$ip" ] && [ -n "$port" ]; then
            SSH_TARGET="-p $port root@$ip"
            # shellcheck disable=SC2086
            if ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o BatchMode=yes $SSH_TARGET 'echo SSH-OK' 2>/dev/null | grep -q SSH-OK; then
                break
            fi
            SSH_TARGET=""
        fi
        sleep 10
    done
    if [ -z "$SSH_TARGET" ]; then
        echo "no ssh within 600 s; terminating"
        cmd_terminate
        exit 1
    fi
    printf "SSH_TARGET='%s'\n" "$SSH_TARGET" >> "$STATE"
    echo "ssh target: $SSH_TARGET"
    if ! tools/runpod_guard.sh arm "$POD_ID" "$SSH_TARGET" "$MINUTES"; then
        echo "ARM REFUSED; terminating"
        cmd_terminate
        exit 1
    fi
    tools/runpod_guard.sh check "$POD_ID"
}

cmd_ssh() { load_state; pod_ssh "$@"; }

cmd_ship() {
    load_state
    sha="$1"
    git archive --format=tar "$sha" -- . ':!bench/results' ':!mamba/corpus' ':!bench/oracle*' \
        ':!archive' ':!docs' ':!paper' ':!training/corpus' \
        | gzip -6 > "${TMPDIR:-/tmp}/samba-leg-$$.tgz"
    ls -l "${TMPDIR:-/tmp}/samba-leg-$$.tgz"
    pod_ssh 'mkdir -p /root/mojolearn && cat > /root/mojolearn/src.tgz' < "${TMPDIR:-/tmp}/samba-leg-$$.tgz"
    rm -f "${TMPDIR:-/tmp}/samba-leg-$$.tgz"
    pod_ssh "cd /root/mojolearn && tar xzf src.tgz && rm src.tgz && echo '$sha' > commit.txt && ls | head -50"
}

cmd_sync() {
    load_state
    # shellcheck disable=SC2086
    rsync -az -e "ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes $(echo $SSH_TARGET | sed 's/root@.*//')" \
        --exclude .git --exclude .pixi --exclude bench/results --exclude mamba/corpus \
        --exclude 'bench/oracle*' --exclude archive --exclude docs --exclude paper \
        --exclude '*.so' --exclude __pycache__ --exclude training/corpus \
        ./ "root@$(echo $SSH_TARGET | sed 's/.*root@//'):/root/mojolearn/"
}

cmd_fetch() {
    load_state
    # shellcheck disable=SC2086
    scp -o StrictHostKeyChecking=accept-new -o BatchMode=yes -r \
        -P "$(echo $SSH_TARGET | sed 's/-p \([0-9]*\).*/\1/')" \
        "root@$(echo $SSH_TARGET | sed 's/.*root@//'):$1" "$2"
}

cmd_extend() {
    load_state; load_key
    tools/runpod_guard.sh extend "$POD_ID" "$SSH_TARGET" "${1:-60}"
}

cmd_terminate() {
    [ -n "${CURLRC:-}" ] || load_key
    [ -n "${POD_ID:-}" ] || load_state
    rp DELETE "/v1/pods/$POD_ID"
    echo "DELETE $POD_ID -> HTTP $RP_CODE"
    sleep 5
    rp GET "/v1/pods/$POD_ID"
    echo "GET after delete -> HTTP $RP_CODE (404 means gone)"
    rm -f "$STATE" "$REPO/bench/results/runpod_leases/$POD_ID.lease"
}

cmd="${1:-}"; [ $# -gt 0 ] && shift
case "$cmd" in
    rent) cmd_rent "$@" ;;
    ssh) cmd_ssh "$@" ;;
    ship) cmd_ship "$@" ;;
    sync) cmd_sync ;;
    fetch) cmd_fetch "$@" ;;
    extend) cmd_extend "$@" ;;
    terminate) cmd_terminate ;;
    *) sed -n '3,20p' "$0"; exit 2 ;;
esac
