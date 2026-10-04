#!/bin/bash
# Internal transport for nvidia_baseline_gpu_batch.py. Do not invoke directly.
# All cloud operations reuse the established lease primitives. Experimental only.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
STAGE=${1:?}; OUT=${2:?}; GPU=${3:?}
# Optional fallback plan (tools/nvidia_ptx_fallback_stage.py): a second body on
# the same pod, inside the same lease and the same 6300-second work cap.
FALLBACK=${4:-}; WORK_SECONDS=6300; FALLBACK_SECONDS=2200
[ -z "$FALLBACK" ] || [ -f "$FALLBACK" ] || exit 2
case "$GPU" in 'NVIDIA GeForce RTX 4090'|'NVIDIA H100 80GB HBM3'|'NVIDIA A100 80GB PCIe') ;; *) exit 2 ;; esac
[ -f "$STAGE/SHA256SUMS" ] && [ -d "$OUT" ] || exit 2
TMPD=$(mktemp -d); CURLRC="$TMPD/curlrc"
POD_NAME="mojolearn-ptx-$(date +%s)-$$"
die() { echo "REFUSED: $*" >&2; exit 1; }
source "$ROOT/tools/runpod_pod_lib.sh"
POD_ID=''; DEADMAN_PID=''; DEADMAN_DIR=''; CREATE_ATTEMPTED=0; SSH_TARGET=''; FETCH_READY=0
SSH_OPTS='-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=3'
fetch_results() {
    mkdir -p "$OUT/remote"
    with_timeout 180 ssh $SSH_OPTS $SSH_TARGET 'cd /root/ptx-batch && tar czf - results SHA256SUMS' > "$OUT/results.tgz" || return 1
    tar xzf "$OUT/results.tgz" -C "$OUT/remote"
}
cleanup() {
    rc=$?
    trap - EXIT INT TERM
    set +e
    if [ "$FETCH_READY" = 1 ]; then
        fetch_results || rc=1
    fi
    safe=0
    if [ "$CREATE_ATTEMPTED" = 1 ] && [ -z "$POD_ID" ]; then
        rp_call GET "$RP/pods"
        # A missing name immediately after an ambiguous POST is not proof of
        # failed creation. Leave the durable deadman armed if no ID resolves.
        case "$RP_CODE" in 2*) POD_ID=$(rp_py byname "$POD_NAME") ;; esac
    fi
    if [ -n "$POD_ID" ]; then
        safe=1
        for pod in $POD_ID; do
            delete_pod "$pod"
            verify_gone "$pod" >> "$OUT/teardown.txt" 2>&1 || safe=0
        done
    elif [ "$CREATE_ATTEMPTED" = 0 ]; then safe=1; fi
    echo "terminated_verified=$safe" >> "$OUT/teardown.txt"
    if [ "$safe" = 1 ]; then
        if [ -n "$DEADMAN_PID" ]; then
            pkill -P "$DEADMAN_PID" 2>/dev/null
            kill "$DEADMAN_PID" 2>/dev/null
        fi
        [ -z "$DEADMAN_DIR" ] || rm -rf "$DEADMAN_DIR"
    else
        echo 'TEARDOWN UNCONFIRMED: local deadman remains armed' >&2
        rc=1
    fi
    rm -rf "$TMPD"
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
load_key || die 'No RunPod key'
rp_call GET "$RP/pods"
case "$RP_CODE" in 2*) ;; *) die 'Cannot list pods before rental' ;; esac
DEADMAN_DIR=$(mktemp -d /tmp/mojolearn-ptx-deadman.XXXXXX)
write_deadman "$DEADMAN_DIR" 8400 || die 'Cannot compose deadman'
nohup sh -c 'trap "" HUP INT; exec sh "$0"' "$DEADMAN_DIR/deadman.sh" >/dev/null 2>&1 </dev/null &
DEADMAN_PID=$!
sleep 1
kill -0 "$DEADMAN_PID" || die 'Deadman failed before create'
python3 - "$POD_NAME" "$GPU" "$TMPD/create.json" <<'PY'
import json,sys
json.dump(dict(name=sys.argv[1],gpuTypeIds=[sys.argv[2]],gpuCount=1,cloudType='SECURE',
 imageName='runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04',
 containerDiskInGb=30,volumeInGb=0,ports=['22/tcp'],supportPublicIp=True,
 interruptible=False,allowedCudaVersions=['13.0']),open(sys.argv[3],'w'))
PY
cp "$TMPD/create.json" "$OUT/create_request.json"
CREATE_ATTEMPTED=1
rp_call POST "$RP/pods" "$TMPD/create.json"
cp "$TMPD/rp.body" "$OUT/create_response.json"
POD_ID=$(rp_py id)
[ -n "$POD_ID" ] || die 'No parsed pod ID; cleanup resolves name'
printf '%s\n' "$POD_ID" > "$DEADMAN_DIR/pod_id.txt"
printf '%s\n' "$POD_ID" > "$OUT/pod_id.txt"
COST=$(rp_py cost)
python3 - "$COST" "$GPU" <<'PY'
import sys
cost=float(sys.argv[1]); cap=.74 if '4090' in sys.argv[2] else 1.99 if 'A100' in sys.argv[2] else 3.49
if not 0 < cost <= cap + .001: raise SystemExit('Unexpected hourly price: stop and tear down')
PY
end=$(( $(date +%s) + 600 ))
while [ "$(date +%s)" -lt "$end" ]; do
    rp_call GET "$RP/pods/$POD_ID"
    SSH_TARGET=$(rp_py ssh)
    if [ -n "$SSH_TARGET" ] && with_timeout 40 ssh $SSH_OPTS $SSH_TARGET 'true'; then break; fi
    SSH_TARGET=''
    sleep 10
done
[ -n "$SSH_TARGET" ] || die 'SSH ready timeout'
MOJOLEARN_LEASE_DIR="$OUT/lease" with_timeout 180 sh "$ROOT/tools/runpod_guard.sh" arm "$POD_ID" "$SSH_TARGET" 120 > "$OUT/arm.log" 2>&1 || die 'On-pod watchdog refused'
with_timeout 60 ssh $SSH_OPTS $SSH_TARGET "p=\$(cat /tmp/mojolearn-lease.pid); kill -0 \"\$p\" && echo WATCHDOG_ALIVE; curl -s --max-time 20 -o /dev/null -w 'TOKEN_GET_%{http_code}' -K /tmp/mojolearn-lease.curlrc $RP/pods/$POD_ID" > "$OUT/watchdog.txt"
grep -q WATCHDOG_ALIVE "$OUT/watchdog.txt" && grep -q TOKEN_GET_200 "$OUT/watchdog.txt" || die 'Watchdog/token verification failed'
tar czf "$TMPD/stage.tgz" -C "$STAGE" .
with_timeout 300 ssh $SSH_OPTS $SSH_TARGET 'mkdir -p /root/ptx-batch && tar xzf - -C /root/ptx-batch' < "$TMPD/stage.tgz" || die 'Upload failed'
FETCH_READY=1
with_timeout 60 ssh $SSH_OPTS $SSH_TARGET 'cd /root/ptx-batch && sha256sum -c SHA256SUMS' > "$OUT/upload-check.txt" || die 'Staged bytes differ'
# Remote timeout survives a disconnected SSH client; both delete watchdogs remain.
WORK_START=$(date +%s)
with_timeout 6420 ssh $SSH_OPTS $SSH_TARGET 'cd /root/ptx-batch && timeout -k 30 6300 bash body.sh' > "$OUT/ssh.log" 2>&1
[ -n "$FALLBACK" ] || exit 0
# Fallback stage. This machine generates the admission from the receipt and
# witness just collected and packs the vendor wheel; the pod then installs it
# unforced. Any refusal here still tears the pod down through cleanup.
fetch_results || die 'Cannot fetch collection results for the fallback stage'
with_timeout 1500 python3 "$ROOT/tools/nvidia_ptx_fallback_stage.py" prepare --plan "$FALLBACK" \
    --results "$OUT/remote/results" --stage "$OUT/fallback-stage" > "$OUT/fallback-prepare.log" 2>&1 || die 'Fallback admission or packing refused'
tar czf "$TMPD/fallback.tgz" -C "$OUT/fallback-stage" .
with_timeout 300 ssh $SSH_OPTS $SSH_TARGET 'mkdir -p /root/ptx-batch/fallback && tar xzf - -C /root/ptx-batch/fallback' < "$TMPD/fallback.tgz" || die 'Fallback upload failed'
with_timeout 60 ssh $SSH_OPTS $SSH_TARGET 'cd /root/ptx-batch/fallback && sha256sum -c SHA256SUMS' > "$OUT/fallback-upload-check.txt" || die 'Fallback staged bytes differ'
LEFT=$(( WORK_SECONDS - ($(date +%s) - WORK_START) ))
[ "$LEFT" -ge "$FALLBACK_SECONDS" ] || die "Work cap leaves $LEFT seconds; the fallback stage needs $FALLBACK_SECONDS"
with_timeout $(( LEFT + 120 )) ssh $SSH_OPTS $SSH_TARGET "cd /root/ptx-batch && timeout -k 30 $LEFT bash fallback/body.sh" > "$OUT/ssh-fallback.log" 2>&1
