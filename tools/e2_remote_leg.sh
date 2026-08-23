#!/usr/bin/env bash
# One supervised GPU-droplet leg: create -> bootstrap -> fetch -> DESTROY.
#
#   bash tools/e2_remote_leg.sh amd|nv <token_file>
#
# The whole point of this file is the teardown. The E1 AMD droplet ran
# 10h48m for a 90-second job because the destroy call lived in an
# interactive session that got interrupted (2026-08-22/23, ~$41). Here
# the destroy is (a) an EXIT trap on this script, so any failure path
# runs it, and (b) a DETACHED dead-man timer that destroys the droplet
# after DEADMAN_SECONDS no matter what happened to this script or the
# session that launched it; the timer is cancelled only by a clean
# teardown. Powering off does not stop DigitalOcean billing; only
# destroy does. Run this under nohup/setsid from a machine that stays
# awake and plugged in.
#
# Artifacts land in bench/results/e1/<stamp>-<host>/ beside the Mac's.
set -uo pipefail

VENDOR="${1:?amd|nv}"
TOKFILE="${2:?token file}"
TOK="$(cat "$TOKFILE")"
DEADMAN_SECONDS="${DEADMAN_SECONDS:-7200}"   # 2 h hard cap per leg
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API=https://api.digitalocean.com/v2
SSH_KEY_FP="df:f7:6b:0c:56:da:48:a5:6f:6d:ae:44:af:de:f3:0b"
STATE="$REPO/bench/results/e1/.leg-$VENDOR.state"

case "$VENDOR" in
  amd) NAME=mojolearn-e2-amd; REGION=tor1; SIZE=gpu-mi325x1-256gb; IMAGE=188571990 ;;
  nv)  NAME=mojolearn-e2-nv;  REGION=nyc2; SIZE=gpu-h100x1-80gb;   IMAGE=236925144 ;;
  *) echo "vendor must be amd or nv"; exit 2 ;;
esac

api() { curl -s -H "Authorization: Bearer $TOK" "$@"; }
log() { echo "[$(date +%T) $VENDOR] $*"; }

DROPLET_ID=""
DEADMAN_PID=""

destroy() {
  [ -z "$DROPLET_ID" ] && return 0
  for i in 1 2 3 4 5 6; do
    code=$(curl -s -o /dev/null -w '%{http_code}' -X DELETE \
      -H "Authorization: Bearer $TOK" "$API/droplets/$DROPLET_ID")
    log "DELETE droplet $DROPLET_ID -> HTTP $code"
    case "$code" in 204|404) break ;; esac
    sleep 10
  done
  sleep 5
  left=$(api "$API/droplets/$DROPLET_ID" | python3 -c \
    "import json,sys; d=json.load(sys.stdin); print(d.get('droplet',{}).get('status','gone'))" 2>/dev/null)
  log "post-destroy status: ${left:-gone}"
  echo "destroyed $(date -u +%FT%TZ)" >> "$STATE"
}

teardown() {
  rc=$?
  log "teardown (rc=$rc)"
  destroy
  [ -n "$DEADMAN_PID" ] && kill "$DEADMAN_PID" 2>/dev/null && log "dead-man timer cancelled"
  exit $rc
}
trap teardown EXIT

mkdir -p "$(dirname "$STATE")"
log "creating $NAME ($SIZE, $REGION)"
DROPLET_ID=$(api -X POST -H "Content-Type: application/json" \
  -d "{\"name\":\"$NAME\",\"region\":\"$REGION\",\"size\":\"$SIZE\",\"image\":$IMAGE,\"ssh_keys\":[\"$SSH_KEY_FP\"],\"tags\":[\"e2\"]}" \
  "$API/droplets" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['droplet']['id'] if 'droplet' in d else '')")
if [ -z "$DROPLET_ID" ]; then
  log "create FAILED"; exit 3
fi
echo "droplet $DROPLET_ID created $(date -u +%FT%TZ)" > "$STATE"
log "droplet id $DROPLET_ID"

# THE DEAD-MAN: a detached process that destroys the droplet after the cap.
# It survives this script and the session that launched it.
nohup bash -c "sleep $DEADMAN_SECONDS; curl -s -o /dev/null -w 'deadman DELETE -> %{http_code}\n' -X DELETE -H 'Authorization: Bearer $TOK' '$API/droplets/$DROPLET_ID' >> '$STATE'" \
  >/dev/null 2>&1 < /dev/null &
DEADMAN_PID=$!
disown "$DEADMAN_PID" 2>/dev/null || true
log "dead-man timer pid $DEADMAN_PID ($DEADMAN_SECONDS s)"

IP=""
for i in $(seq 1 90); do
  read -r status IP < <(api "$API/droplets/$DROPLET_ID" | python3 -c "
import json,sys
d=json.load(sys.stdin)['droplet']
ips=[n['ip_address'] for n in d['networks'].get('v4',[]) if n['type']=='public']
print(d['status'], ips[0] if ips else '')")
  [ "$status" = "active" ] && [ -n "$IP" ] && break
  sleep 10
done
if [ -z "$IP" ]; then log "never became active"; exit 4; fi
log "active at $IP"
echo "ip $IP" >> "$STATE"

SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o ServerAliveInterval=30 root@$IP"
ok=0
for i in $(seq 1 30); do
  if $SSH 'echo SSH-OK' 2>/dev/null | grep -q SSH-OK; then ok=1; break; fi
  sleep 15
done
[ $ok = 1 ] || { log "ssh never came up"; exit 5; }
$SSH 'nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || rocm-smi --showproductname 2>/dev/null | grep -i "card series\|name" | head -2' | sed "s/^/[$VENDOR gpu] /"

# SHIP THE COMMIT, NOT THE WORKING TREE. The first E2 legs rsync'd a
# worktree whose numerics.mojo the Mac's own bootstrap had ALREADY flipped
# to IDENTICAL (the flip is session-local and the two sessions shared the
# tree), and whose .git was a worktree pointer file; the remote bootstrap
# refused the flip and git had no repository. A bundle of HEAD is exact
# by construction: the remote clones it, checks out the hash, and the
# commit recorded in every artifact is the commit that ran.
COMMIT="$(git -C "$REPO" rev-parse "${E2_COMMIT:-HEAD}")"   # pin to the Mac reference run's commit
BUNDLE="/tmp/mojolearn-e2-$COMMIT.bundle"
# a bundle needs a REF (a bare sha is "Refusing to create empty bundle")
git -C "$REPO" branch -f "e2-run" "$COMMIT" >/dev/null 2>&1
log "bundle $COMMIT (branch e2-run)"
git -C "$REPO" bundle create "$BUNDLE" e2-run >/dev/null 2>&1 || { log "bundle failed"; exit 6; }
scp -q -o StrictHostKeyChecking=accept-new "$BUNDLE" "root@$IP:/root/e2.bundle" || { log "scp failed"; exit 6; }
$SSH "rm -rf /root/mojolearn && git clone -q -b e2-run /root/e2.bundle /root/mojolearn && cd /root/mojolearn && git rev-parse HEAD && git status --short | head -3 && grep -c '^comptime GLOBAL_NUMERIC_MODE = NUMERIC_FAST' mojo_only/numerics.mojo" \
  || { log "remote clone/checkout failed"; exit 6; }

log "bootstrap (phases 0-4) -- this is the long step"
$SSH 'export PATH=/root/.pixi/bin:$PATH; cd /root/mojolearn && bash tools/e1_bootstrap.sh > /root/e2_run.log 2>&1; echo "BOOTSTRAP-EXIT=$?"; tail -20 /root/e2_run.log'

# TRAIN-HERE-INFER-THERE, the Mac -> box direction: the Mac reference
# run's models (MAC_REF_DIR, optional) are loaded on the box with the
# IDENTICAL .so the bootstrap just built and predicted there. The box ->
# Mac direction runs on the Mac after the fetch.
if [ -n "${MAC_REF_DIR:-}" ] && [ -d "$MAC_REF_DIR" ]; then
  log "cross-infer: Mac models on the box"
  rsync -az --include '*.model.npz' --include '*.json' --exclude '*' "$MAC_REF_DIR/" "root@$IP:/root/mac_ref/" \
    && $SSH 'export PATH=/root/.pixi/bin:$PATH; cd /root/mojolearn && OUT=$(ls -td bench/results/e1/*/ | head -1) && PYTHONPATH=python pixi run -e gbmbench python3 tools/e1_cross_infer.py /root/mac_ref "$OUT/cross_infer_mac_models_on_box.json" 2>&1 | tail -8' \
    || log "cross-infer on box FAILED (see above)"
fi

log "fetch artifacts"
mkdir -p "$REPO/bench/results/e1"
rsync -az "root@$IP:/root/mojolearn/bench/results/e1/" "$REPO/bench/results/e1/" \
  && log "fetched: $(ls -t "$REPO/bench/results/e1" | head -1)" \
  || log "FETCH FAILED (the remote log is /root/e2_run.log; droplet is being destroyed regardless)"
rsync -az "root@$IP:/root/e2_run.log" "$REPO/bench/results/e1/e2_run_$VENDOR.log" 2>/dev/null || true

log "leg done; destroying"
# teardown runs via the EXIT trap
exit 0
