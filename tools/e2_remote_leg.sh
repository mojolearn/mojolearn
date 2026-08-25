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
DEADMAN_SECONDS="${DEADMAN_SECONDS:-3600}"   # ONE HOUR hard cap per leg (Andrew, 2026-08-23: a rented GPU expires on its own after an hour, in code, not in memory)
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
  if [ -z "$DROPLET_ID" ]; then
    # no id known: sweep by name so an unreadable create response cannot
    # leave a droplet behind (the leg-10 orphan)
    for id in $(api "$API/droplets?tag_name=e2&per_page=50" | python3 -c "import json,sys
d=json.load(sys.stdin); print(' '.join(str(x['id']) for x in d.get('droplets',[]) if x['name']=='$NAME'))" 2>/dev/null); do
      code=$(curl -s -o /dev/null -w '%{http_code}' -X DELETE -H "Authorization: Bearer $TOK" "$API/droplets/$id")
      log "DELETE by-name droplet $id -> HTTP $code"
    done
    return 0
  fi
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
  # kill the wrapper AND its sleep child: killing only the wrapper leaves an
  # orphan `sleep` (harmless -- the DELETE after it dies with the wrapper --
  # but seven of them were found after the first E2 day)
  if [ -n "$DEADMAN_PID" ]; then
    pkill -P "$DEADMAN_PID" 2>/dev/null
    kill "$DEADMAN_PID" 2>/dev/null && log "dead-man timer cancelled"
  fi
  exit $rc
}
trap teardown EXIT

mkdir -p "$(dirname "$STATE")"
LEG_START=$(date +%s)   # every bound below is measured from here, not from the step that uses it

# THE DEAD-MAN, ARMED BEFORE THE CREATE CALL AND KEYED BY NAME, NOT ID.
# Leg 10 (2026-08-23 12:11): the create call made the droplet but returned
# a non-JSON body, this script parsed no id, printed "create FAILED", and
# exited through a teardown that had nothing to destroy and a dead-man that
# had never been armed -- an ORPHAN MI325X, found by hand 19 minutes later
# through the account listing. The id-keyed timer protected every path
# except the one where the id is unknown. This one lists every droplet
# tagged e2 whose name is $NAME and deletes each, so it needs nothing this
# script learns later. It survives this script and the session that
# launched it, and is cancelled only by a clean teardown.
nohup bash -c "sleep $DEADMAN_SECONDS; for id in \$(curl -s -H 'Authorization: Bearer $TOK' '$API/droplets?tag_name=e2&per_page=50' | python3 -c \"import json,sys; d=json.load(sys.stdin); print(' '.join(str(x['id']) for x in d.get('droplets',[]) if x['name']=='$NAME'))\"); do curl -s -o /dev/null -w \"deadman DELETE \$id -> %{http_code}\\n\" -X DELETE -H 'Authorization: Bearer $TOK' \"$API/droplets/\$id\" >> '$STATE'; done" \
  >/dev/null 2>&1 < /dev/null &
DEADMAN_PID=$!
disown "$DEADMAN_PID" 2>/dev/null || true
log "dead-man timer pid $DEADMAN_PID ($DEADMAN_SECONDS s, keyed by tag e2 + name $NAME)"

log "creating $NAME ($SIZE, $REGION)"
CREATE_BODY="$(api -X POST -H "Content-Type: application/json" \
  -d "{\"name\":\"$NAME\",\"region\":\"$REGION\",\"size\":\"$SIZE\",\"image\":$IMAGE,\"ssh_keys\":[\"$SSH_KEY_FP\"],\"tags\":[\"e2\"]}" \
  "$API/droplets")"
DROPLET_ID=$(printf '%s' "$CREATE_BODY" | python3 -c "import json,sys
try:
    d=json.load(sys.stdin); print(d['droplet']['id'] if 'droplet' in d else '')
except Exception:
    print('')")
if [ -z "$DROPLET_ID" ]; then
  # The create may have SUCCEEDED with an unreadable body (leg 10). Look
  # the droplet up by name before concluding anything, and ADOPT it so the
  # id-keyed teardown below owns it.
  log "create returned no id; body: $(printf '%s' "$CREATE_BODY" | head -c 200)"
  sleep 5
  DROPLET_ID=$(api "$API/droplets?tag_name=e2&per_page=50" | python3 -c "import json,sys
d=json.load(sys.stdin); ids=[x['id'] for x in d.get('droplets',[]) if x['name']=='$NAME']
print(ids[0] if ids else '')")
  if [ -n "$DROPLET_ID" ]; then
    log "ADOPTED droplet $DROPLET_ID found by name after an unreadable create response"
  else
    log "create FAILED (no droplet by that name either)"; exit 3
  fi
fi
echo "droplet $DROPLET_ID created $(date -u +%FT%TZ)" > "$STATE"
log "droplet id $DROPLET_ID"

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
$SSH "rm -rf /root/mojolearn && git clone -q -b e2-run /root/e2.bundle /root/mojolearn && cd /root/mojolearn && git rev-parse HEAD && git status --short | head -3 && grep -c 'is_defined\\[\"MOJOLEARN_NUMERIC_IDENTICAL\"\\]' mojo_only/numerics.mojo" \
  || { log "remote clone/checkout failed"; exit 6; }

# THE WORK BOUND, COMPUTED FROM THE LEASE AND NOT FROM A GUESS.
# DEVIATION 1090, 2026-08-25. This step used to be an unbounded `$SSH` that
# ran the whole bootstrap. If the payload hung -- a pixi solve that never
# finishes, a device that never answers -- the ONLY thing that ended the leg
# was the dead-man, which destroys the droplet WITH THE ARTIFACTS STILL ON IT
# and the fetch below never runs. A leg that dies that way costs the full hour
# and comes home with nothing, which is the worst of the two failures.
#
# So the remote work gets its own `timeout`, and its budget is what is LEFT of
# the lease after provisioning, minus a fetch reserve. It is absolute, not a
# fixed number: a slow provision shortens the work, it does not push the fetch
# past the dead-man. `timeout` is coreutils on the Ubuntu image; it does not
# exist on the launching Mac and is deliberately not used there.
FETCH_RESERVE="${FETCH_RESERVE:-420}"
# WAVES: THE LANE THAT MATTERS MOST RUNS FIRST, IN ITS OWN BOOTSTRAP.
# E2_LANE_WAVES="mamba;gemm;cd,kde,linkage,svm,metrics" runs three bootstraps
# on the one droplet, each with its own bound, each writing its own stamped
# directory. Unset means a single wave whose lanes are MOJOLEARN_E1_LANES.
#
# WHY NOT ONE CALL WITH ALL SEVEN. Phase 8's loop order is fixed (gemm first,
# mamba sixth) and `MOJOLEARN_E1_LANES` selects without reordering, so on a
# rented box a slow gemm compile decides that mamba does not run at all --
# exactly what happened to leg 12 (identical_cards=2, five lanes missing). A
# wave is the only way to say "this one first" without editing phase 8's order
# for everybody. The waves share the box's pixi env and mojo cache, so only
# the first pays the setup.
WAVES="${E2_LANE_WAVES:-${MOJOLEARN_E1_LANES:-}}"
WAVE_N=0
IFS=';' read -r -a WAVE_ARR <<< "$WAVES"
[ ${#WAVE_ARR[@]} -eq 0 ] && WAVE_ARR=("")
for WAVE in "${WAVE_ARR[@]}"; do
  WAVE_N=$((WAVE_N+1))
  NOW=$(date +%s)
  WORK_SECONDS=$(( LEG_START + DEADMAN_SECONDS - NOW - FETCH_RESERVE ))
  if [ "$WORK_SECONDS" -lt 120 ]; then
    log "wave $WAVE_N (${WAVE:-all}) SKIPPED -- only ${WORK_SECONDS}s of lease left"
    continue
  fi
  log "wave $WAVE_N: phases=${MOJOLEARN_E1_PHASES:-all} lanes=${WAVE:-all}, bound ${WORK_SECONDS}s"
  $SSH "export PATH=/root/.pixi/bin:\$PATH; cd /root/mojolearn && MOJOLEARN_E1_PHASES='${MOJOLEARN_E1_PHASES:-}' MOJOLEARN_E1_LANES='$WAVE' timeout -k 30 $WORK_SECONDS bash tools/e1_bootstrap.sh > /root/e2_run_w$WAVE_N.log 2>&1; echo \"WAVE-$WAVE_N-EXIT=\$?  (124 = hit the work bound)\"; tail -30 /root/e2_run_w$WAVE_N.log"
done

# EXTRA CHECKS: things phase 8 does not know about yet, run only when asked.
# Named explicitly rather than swept, so this leg's payload is readable from
# this file alone. Each gets its own slice of what is left, so one hang cannot
# eat the others.
if [ -n "${E2_EXTRA_CHECKS:-}" ]; then
  for chk in $E2_EXTRA_CHECKS; do
    # WRAP=1 runs the command under tools/with_identical_mode.sh, which injects
    # -D MOJOLEARN_NUMERIC_IDENTICAL=1. WRAP=0 IS NOT A CONVENIENCE: a check
    # that alternates the modes ITSELF must not be forced into one of them, or
    # both of its arms are the same binary and its ratio is 1.0 by
    # construction. That is DEVIATION 1091 in a different costume.
    WRAP=1
    case "$chk" in
      gemm-backward) CMD='pixi run mojo run -I . gemm/mojo_only/gemm_backward_check.mojo' ;;
      # THE SPEED LANE. Both sides run under ONE MAC cap so a row is either
      # measured on both arms or skipped on both; a full-shape vendor number
      # beside a capped one of ours would be a ratio between two different
      # questions.
      #
      # FIVE ROUNDS, NOT ONE. DEVIATION 1094, 2026-08-25. Leg 15 and leg 16
      # both ran ROUNDS=1 on an H100 an hour apart and our own v1 arm measured
      # 4.90 ms and 8.799 ms at llama8b.qkv.t512 -- the SAME shape, the SAME
      # commit, a 1.8x spread. A single round is a sample, not a measurement,
      # and a 1.8x band swallows most of the effects this lane is trying to
      # report. tools/gemm_price.sh already takes the median over rounds; it
      # was being told to take the median of one.
      gemm-price)    CMD="env MOJOLEARN_GEMM_PRICE_ROUNDS=${GEMM_PRICE_ROUNDS:-5} MOJOLEARN_GEMM_PRICE_ARM=device MOJOLEARN_GEMM_PRICE_DEV_MAC_BUDGET=${GEMM_PRICE_DEV_BUDGET:-50000000000} sh tools/gemm_price.sh"; WRAP=0 ;;
      # THE ONE-VARIABLE PRICE OF THE FOLD PIN. Three arms in ONE binary
      # alternating call by call, so the governor drift cancels at the arm
      # level instead of being averaged over rounds. Measured 1.52x-1.55x on
      # an M4; this is the leg that asks whether that number is Apple's or
      # the profile's. WRAP=0: the driver builds its own arms and forcing it
      # under one mode would make both of them the same binary.
      unpinned-price) CMD='pixi run mojo run -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_UNPINNED_ARM=1 gemm/mojo_only/gemm_unpinned_price.mojo'; WRAP=0 ;;
      # The vendor LIBRARY (cuBLAS / hipBLASLt), not MAX's linalg.matmul.
      # torch goes into a THROWAWAY VENV, never into the pinned pixi
      # environment every recorded timing in this repository was taken under.
      # The script refuses on a CPU torch, so a wheel that resolves to the
      # wrong backend fails loudly instead of timing the host.
      vendor-price)  CMD='bash tools/remote_vendor_torch.sh'; WRAP=0 ;;
      # THE REACH EVIDENCE. A phase-8-only leg proves the CARDS agree; it does
      # not prove the kernels ran on the rented GPU rather than falling back.
      # These two print what the build targeted and what the device answered,
      # which is the only thing that separates a cross-vendor result from a
      # pair of host runs on two machines.
      column)        CMD='pixi run mojo run -I . matrix_main.mojo' ;;
      gpu-probe)     CMD='pixi run mojo run -I . probe_main.mojo' ;;
      # (no `transformer` case: the device spelling in transformer/ported/ has
      #  no check driver yet, so there is nothing here to run. Add the case in
      #  the same commit that adds the driver.)
      *) log "unknown extra check '$chk' -- skipped"; continue ;;
    esac
    NOW=$(date +%s)
    LEFT=$(( LEG_START + DEADMAN_SECONDS - NOW - FETCH_RESERVE ))
    [ "$LEFT" -lt 60 ] && { log "extra check $chk SKIPPED (${LEFT}s left)"; continue; }
    log "extra check: $chk (bound ${LEFT}s)"
    PREFIX="bash tools/with_identical_mode.sh"
    [ "$WRAP" = 0 ] && PREFIX=""
    $SSH "export PATH=/root/.pixi/bin:\$PATH; cd /root/mojolearn && OUT=\$(ls -td bench/results/e1/*/ | head -1) && mkdir -p \"\$OUT/lanes\" && MOJOLEARN_IDENTITY_TRACE=\"\$OUT/lanes/$chk.identical.card\" VENDOR_PRICE_OUT=\"\$OUT/lanes/vendor_price.json\" MOJOLEARN_GEMM_PRICE_OUT=\"\$OUT/lanes/gemm_price_medians.txt\" timeout -k 30 $LEFT $PREFIX $CMD > \"\$OUT/lanes/$chk.identical.check.log\" 2>&1; echo \"EXTRA-$chk-EXIT=\$?\"; tail -8 \"\$OUT/lanes/$chk.identical.check.log\""
  done
fi

# TRAIN-HERE-INFER-THERE, the Mac -> box direction: the Mac reference
# run's models (MAC_REF_DIR, optional) are loaded on the box with the
# IDENTICAL .so the bootstrap just built and predicted there. The box ->
# Mac direction runs on the Mac after the fetch.
# MAC_REF_GLOB resolves LATE (the Mac reference run may still be writing
# when this leg starts; its stamped directory name is unknown up front)
if [ -n "${MAC_REF_GLOB:-}" ]; then
  MAC_REF_DIR="$(ls -td $MAC_REF_GLOB 2>/dev/null | head -1)"
fi
if [ -n "${MAC_REF_DIR:-}" ] && [ -d "$MAC_REF_DIR" ]; then
  log "cross-infer: Mac models on the box"
  rsync -az --include '*.model.npz' --include '*.json' --exclude '*' "$MAC_REF_DIR/" "root@$IP:/root/mac_ref/" \
    && $SSH 'export PATH=/root/.pixi/bin:$PATH; export MOJOLEARN_NUMERIC_MODE=identical; cd /root/mojolearn && OUT=$(ls -td bench/results/e1/*/ | head -1) && PYTHONPATH=python pixi run -e gbmbench python3 tools/e1_cross_infer.py /root/mac_ref "$OUT/cross_infer_mac_models_on_box.json" 2>&1 | tail -8' \
    || log "cross-infer on box FAILED (see above)"
fi

log "fetch artifacts"
mkdir -p "$REPO/bench/results/e1"
rsync -az "root@$IP:/root/mojolearn/bench/results/e1/" "$REPO/bench/results/e1/" \
  && log "fetched: $(ls -t "$REPO/bench/results/e1" | head -1)" \
  || log "FETCH FAILED (the remote log is /root/e2_run.log; droplet is being destroyed regardless)"
for wl in $($SSH "ls /root/e2_run*.log 2>/dev/null"); do
  rsync -az "root@$IP:$wl" "$REPO/bench/results/e1/${VENDOR}_$(basename "$wl")" 2>/dev/null || true
done

log "leg done; destroying"
# teardown runs via the EXIT trap
exit 0
