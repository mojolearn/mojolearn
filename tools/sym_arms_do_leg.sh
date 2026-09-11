#!/usr/bin/env bash
# ONE GUARDED DIGITALOCEAN MI325X LEG FOR THE gbdt-symmetric-arms LANE.
#
#   SYMARMS_TIERS="identical fast" bash tools/sym_arms_do_leg.sh
#
# Order, and every step is a guard for the one after it:
#   1. the shared GPU lock (/tmp/mojolearn-do-gpu.lock), taken in the order
#      the coordinator set: after an owner containing "extra:" has held and
#      released it following an amd-trees-leg owner, or after it has sat
#      free for 300 s with no taker
#   2. a Mac-side dead-man keyed by tag AND name, armed BEFORE the create
#   3. create, then an ON-BOX watchdog that deletes its own droplet before
#      the hour, verified alive, with the droplet id baked in and the token
#      proven by a GET returning 200 (the Mac can go away; the box cannot)
#   4. ship `git archive` of HEAD (results trees excluded), nohup the
#      payload (tools/sym_arms_box.sh all), poll and fetch partial results
#   5. destroy on EXIT, verify with GET /v2/droplets that none of ours is
#      left, cancel the dead-man, release the lock
#
# The token is read from ~/.mojolearn_do_token into a 0600 curl config and a
# 0600 header file on the box; it is never printed and never in an argv.
# Evidence lands in ~/mojolearn-evidence/gbdt-symmetric-arms/<stamp>/.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOKFILE="${MOJOLEARN_DO_TOKEN_FILE:-$HOME/.mojolearn_do_token}"
API=https://api.digitalocean.com/v2
NAME=mojolearn-symarms-amd
TAG=symarms
REGION=tor1
SIZE=gpu-mi325x1-256gb
IMAGE=188571990
# the persistent tor1 volume the trees AMD leg created: decoded datasets and
# pip/rattler caches survive the lease (tools/trees_amd_leg.sh)
VOLUME=mojolearn-data-tor1
VOL_MOUNT=/mnt/mojolearn-data
SSH_KEY_FP="df:f7:6b:0c:56:da:48:a5:6f:6d:ae:44:af:de:f3:0b"
LEASE="${SYMARMS_LEASE_S:-3600}"
FETCH_RESERVE="${SYMARMS_FETCH_RESERVE_S:-420}"
TIERS="${SYMARMS_TIERS:-identical fast}"
LOCK=/tmp/mojolearn-do-gpu.lock
STAMP="$(date -u +%Y-%m-%d_%H%M%S)"
EVID="$HOME/mojolearn-evidence/gbdt-symmetric-arms/$STAMP"
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
mkdir -p "$EVID"
STATE="$EVID/leg.state"
LOG="$EVID/leg.log"
log() { echo "[$(date -u +%H:%M:%S) symarms] $*" | tee -a "$LOG"; }

[ -f "$TOKFILE" ] || { echo "no token file"; exit 2; }
CURLRC="$EVID/.curlrc"
( umask 077; printf 'header = "Authorization: Bearer %s"\nsilent\nshow-error\n' "$(cat "$TOKFILE")" > "$CURLRC" )
api() { curl -K "$CURLRC" "$@"; }

HAVE_LOCK=0; DROPLET_ID=""; DEADMAN_PID=""; T_CREATE=""
{ echo "commit=$COMMIT"; echo "tiers=$TIERS"; echo "stamp=$STAMP"; } > "$STATE"

# ---- 1. the lock, in the agreed order -------------------------------------
wait_for_turn() {
    local seen_trees=0 seen_extra=0 free_since="" owner=""
    while :; do
        if [ -d "$LOCK" ]; then
            owner="$(cat "$LOCK/owner" 2>/dev/null || true)"
            free_since=""
            case "$owner" in *amd-trees-leg*) seen_trees=1 ;; esac
            case "$owner" in *extra:*) [ "$seen_trees" = 1 ] && seen_extra=1 ;; esac
            local age=$(( $(date +%s) - $(stat -f %m "$LOCK" 2>/dev/null || stat -c %Y "$LOCK") ))
            if [ "$age" -gt 6000 ]; then
                local live
                live="$(api "$API/droplets?per_page=100" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(sum(1 for x in d.get("droplets",[]) if x["size_slug"].startswith("gpu")))' 2>/dev/null)"
                [ "$live" = 0 ] && log "lock held ${age}s by '$owner' with zero GPU droplets live: it may be broken; not breaking it"
            fi
            log "lock held by '$owner' (seen_trees=$seen_trees seen_extra=$seen_extra)"
            sleep 60
            continue
        fi
        # Coordinator, 2026-09-11 13:30Z: the neural session left the DO
        # queue, so the lock is taken the moment it is free.
        if mkdir "$LOCK" 2>/dev/null; then
            echo "gbdt-symmetric-arms $(date -u +%FT%TZ)" > "$LOCK/owner"
            HAVE_LOCK=1
            log "lock taken"
            return 0
        fi
        sleep 20
    done
}

destroy_and_verify() {
    if [ -n "$DROPLET_ID" ]; then
        for i in 1 2 3 4 5 6; do
            code=$(api -o /dev/null -w '%{http_code}' -X DELETE "$API/droplets/$DROPLET_ID")
            log "DELETE droplet $DROPLET_ID -> HTTP $code"
            case "$code" in 204|404) break ;; esac
            sleep 10
        done
    fi
    # by name too, so a create whose id was never parsed cannot orphan
    for id in $(api "$API/droplets?tag_name=$TAG&per_page=50" | python3 -c "import json,sys
d=json.load(sys.stdin); print(' '.join(str(x['id']) for x in d.get('droplets',[]) if x['name']=='$NAME'))" 2>/dev/null); do
        code=$(api -o /dev/null -w '%{http_code}' -X DELETE "$API/droplets/$id")
        log "DELETE by-name droplet $id -> HTTP $code"
    done
    local mine="unknown"
    for i in $(seq 1 12); do
        sleep 10
        mine=$(api "$API/droplets?per_page=100" | python3 -c "import json,sys
d=json.load(sys.stdin); print(len([x for x in d.get('droplets',[]) if x['name']=='$NAME']))" 2>/dev/null)
        [ "$mine" = 0 ] && break
    done
    log "GET /v2/droplets: droplets named $NAME left = $mine"
    echo "destroy_verified=$mine $(date -u +%FT%TZ)" >> "$STATE"
    [ "$mine" = 0 ]
}

teardown() {
    rc=$?
    log "teardown (rc=$rc)"
    if [ -n "${IP:-}" ] && [ -n "${SSH:-}" ]; then
        # the shared volume is unmounted cleanly before the droplet goes
        $SSH "sync; umount $VOL_MOUNT 2>/dev/null && echo VOLUME_UNMOUNTED || echo VOLUME_NOT_MOUNTED; sync" 2>&1 | tail -1 | tee -a "$LOG"
    fi
    if [ -n "$DROPLET_ID" ] || [ -n "$T_CREATE" ]; then
        if destroy_and_verify; then
            if [ -n "$DEADMAN_PID" ]; then
                pkill -P "$DEADMAN_PID" 2>/dev/null
                kill "$DEADMAN_PID" 2>/dev/null && log "dead-man cancelled"
            fi
            if [ "$HAVE_LOCK" = 1 ]; then
                rm -rf "$LOCK" && log "lock released"
            fi
            rm -f "$CURLRC"
        else
            log "DESTROY NOT VERIFIED: dead-man and lock left in place"
        fi
    else
        if [ "$HAVE_LOCK" = 1 ]; then
            rm -rf "$LOCK" && log "lock released (nothing created)"
        fi
        [ -n "$DEADMAN_PID" ] && kill "$DEADMAN_PID" 2>/dev/null
        rm -f "$CURLRC"
    fi
    exit $rc
}
trap teardown EXIT
trap 'exit 130' INT TERM

# The whole run is one function, parsed before it executes, so an edit to
# this file while a leg waits on the lock cannot shift bash's read offset.
main() {
wait_for_turn

# ---- 2. the Mac-side dead-man, before the create ---------------------------
cat > "$EVID/deadman.sh" <<EOF
sleep $LEASE
for id in \$(curl -K '$CURLRC' '$API/droplets?tag_name=$TAG&per_page=50' | python3 -c "import json,sys
d=json.load(sys.stdin); print(' '.join(str(x['id']) for x in d.get('droplets',[]) if x['name']=='$NAME'))"); do
  curl -K '$CURLRC' -o /dev/null -w "deadman DELETE \$id -> %{http_code}\n" -X DELETE "$API/droplets/\$id" >> '$STATE'
done
EOF
nohup bash "$EVID/deadman.sh" > /dev/null 2>&1 < /dev/null &
DEADMAN_PID=$!
disown "$DEADMAN_PID" 2>/dev/null || true
log "dead-man pid $DEADMAN_PID armed for ${LEASE}s (tag $TAG, name $NAME)"

# ---- 3. create -------------------------------------------------------------
VOL_ID=$(api "$API/volumes?region=$REGION&per_page=200" | python3 -c "import json,sys
d=json.load(sys.stdin); v=[x for x in d.get('volumes',[]) if x['name']=='$VOLUME']
print(v[0]['id'] if v and not v[0].get('droplet_ids') else '')" 2>/dev/null)
VOL_ARG=""
if [ -n "$VOL_ID" ]; then
    VOL_ARG=",\"volumes\":[\"$VOL_ID\"]"
    log "attaching volume $VOLUME ($VOL_ID)"
else
    log "volume $VOLUME absent or attached elsewhere; datasets download to the droplet"
fi
T_CREATE=$(date +%s)
BODY=$(api -X POST -H 'Content-Type: application/json' \
    -d "{\"name\":\"$NAME\",\"region\":\"$REGION\",\"size\":\"$SIZE\",\"image\":$IMAGE,\"ssh_keys\":[\"$SSH_KEY_FP\"],\"tags\":[\"$TAG\"]$VOL_ARG}" \
    "$API/droplets")
DROPLET_ID=$(printf '%s' "$BODY" | python3 -c "import json,sys
try:
    d=json.load(sys.stdin); print(d['droplet']['id'] if 'droplet' in d else '')
except Exception: print('')")
if [ -z "$DROPLET_ID" ]; then
    log "create returned no id: $(printf '%s' "$BODY" | head -c 200)"
    exit 3
fi
echo "droplet=$DROPLET_ID created=$(date -u +%FT%TZ)" >> "$STATE"
log "droplet $DROPLET_ID"

IP=""
for i in $(seq 1 90); do
    read -r status IP < <(api "$API/droplets/$DROPLET_ID" | python3 -c "
import json,sys
d=json.load(sys.stdin)['droplet']
ips=[n['ip_address'] for n in d['networks'].get('v4',[]) if n['type']=='public']
print(d['status'], ips[0] if ips else '')")
    [ "$status" = active ] && [ -n "$IP" ] && break
    sleep 10
done
[ -n "$IP" ] || { log "never became active"; exit 4; }
log "active at $IP"
SSH="ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$EVID/known_hosts -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30 root@$IP"
ok=0
for i in $(seq 1 30); do
    $SSH 'echo SSH-OK' 2>/dev/null | grep -q SSH-OK && { ok=1; break; }
    sleep 15
done
[ "$ok" = 1 ] || { log "ssh never came up"; exit 5; }

BOX_TTL=$(( LEASE - ($(date +%s) - T_CREATE) - 120 ))
printf 'Authorization: Bearer %s\n' "$(cat "$TOKFILE")" | $SSH 'umask 077; cat > /root/.do_auth_header'
$SSH "nohup bash -c 'sleep $BOX_TTL; curl -s -X DELETE -H @/root/.do_auth_header https://api.digitalocean.com/v2/droplets/$DROPLET_ID > /root/watchdog.fired 2>&1' > /dev/null 2>&1 < /dev/null & echo \$! > /root/watchdog.pid"
WD=$($SSH "pid=\$(cat /root/watchdog.pid); kill -0 \$pid 2>/dev/null && echo alive; ps -o args= -p \$pid | grep -c 'droplets/$DROPLET_ID'; curl -s -o /dev/null -w 'get%{http_code}' -H @/root/.do_auth_header https://api.digitalocean.com/v2/droplets/$DROPLET_ID" | tr '\n' ' ')
log "on-box watchdog: $WD (ttl ${BOX_TTL}s)"
echo "watchdog=$WD ttl=$BOX_TTL" >> "$STATE"
case "$WD" in *alive*1*get200*) : ;; *) log "watchdog NOT verified; aborting"; exit 6 ;; esac

# ---- 4. ship and run -------------------------------------------------------
git -C "$REPO" archive --format=tar "$COMMIT" -- . ':!bench/results' ':!mamba/corpus' \
    ':!bench/oracle*' ':!bench/minentropy_oracle.txt' ':!*.bin' ':!archive' ':!upstream' \
    ':!docs' ':!paper' | gzip -1 > "$EVID/src.tgz"
log "archive $(du -h "$EVID/src.tgz" | cut -f1)"
SCP="scp -q -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$EVID/known_hosts -o BatchMode=yes"
$SCP "$EVID/src.tgz" "root@$IP:/root/src.tgz" || { log "scp failed"; exit 7; }
$SSH "mkdir -p /root/retained" && $SCP "$REPO/bench/results/identity_break/amd-mi325x.identical.json" \
    "$REPO/bench/results/identity_break/apple-m4.identical.json" "root@$IP:/root/retained/"
if [ -n "$VOL_ID" ]; then
    $SSH "dev=/dev/disk/by-id/scsi-0DO_Volume_$VOLUME; for i in \$(seq 1 30); do [ -e \$dev ] && break; sleep 2; done
          if [ -e \$dev ]; then mkdir -p $VOL_MOUNT; mountpoint -q $VOL_MOUNT || mount -o discard,defaults \$dev $VOL_MOUNT;
            echo VOLUME_MOUNTED; ls -la $VOL_MOUNT/gbm-bench/*/*.npz 2>/dev/null; else echo VOLUME_DEVICE_MISSING; fi" 2>&1 | tee -a "$LOG"
fi
$SSH "rm -rf /root/mojolearn && mkdir -p /root/mojolearn /root/symarms_out && tar xzf /root/src.tgz -C /root/mojolearn && echo $COMMIT > /root/symarms_out/commit.txt" \
    || { log "unpack failed"; exit 7; }
BOX_DEADLINE=$(( T_CREATE + LEASE - FETCH_RESERVE - 180 ))
$SSH "cd /root/mojolearn && SYMARMS_DEADLINE=$BOX_DEADLINE SYMARMS_TIERS='$TIERS' nohup bash tools/sym_arms_box.sh all > /root/symarms_out/payload.log 2>&1 < /dev/null & echo started"
log "payload started; box deadline $(date -u -r "$BOX_DEADLINE" +%H:%M:%S 2>/dev/null || echo "$BOX_DEADLINE")"

SSH_RSYNC="ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$EVID/known_hosts -o BatchMode=yes -o ConnectTimeout=10"
WORK_END=$(( T_CREATE + LEASE - FETCH_RESERVE ))
PATCHES=0
while [ "$(date +%s)" -lt "$WORK_END" ]; do
    sleep 30
    # THE PATCH LOOP. A compile error found on the box should cost minutes,
    # not a lease: when every gbdt build failed the droplet is kept, and a
    # committed fix named in $EVID/patch.list (paths relative to the repo)
    # is rsynced over and the payload restarts under the same deadline.
    if [ -f "$EVID/patch.list" ]; then
        PATCHES=$((PATCHES + 1))
        log "patch $PATCHES at $(git -C "$REPO" rev-parse --short HEAD): $(tr '\n' ' ' < "$EVID/patch.list")"
        $SSH 'pkill -f sym_arms_box.sh; pkill -f forest_speed_arm.py; pkill -f "mojo build"; pkill -f "mojo run"; true' 2>/dev/null
        ( cd "$REPO" && rsync -az --relative -e "$SSH_RSYNC" $(cat "$EVID/patch.list") "root@$IP:/root/mojolearn/" ) \
            && log "patch $PATCHES shipped" || log "patch $PATCHES rsync FAILED"
        $SSH "echo patch$PATCHES $(git -C "$REPO" rev-parse HEAD) >> /root/symarms_out/commit.txt; rm -f /root/symarms_out/DONE.*; cd /root/mojolearn && SYMARMS_DEADLINE=$BOX_DEADLINE SYMARMS_TIERS='$TIERS' nohup bash tools/sym_arms_box.sh all >> /root/symarms_out/payload.log 2>&1 < /dev/null & echo restarted" | tee -a "$LOG"
        mv "$EVID/patch.list" "$EVID/patch.$PATCHES.applied"
        continue
    fi
    if $SSH 'test -f /root/symarms_out/DONE.all' 2>/dev/null; then
        rsync -az -e "$SSH_RSYNC" "root@$IP:/root/symarms_out/" "$EVID/out/" 2>/dev/null
        if ! grep -q 'build_exit build_gbdt.sh.*=0' "$EVID/out/ab.txt" 2>/dev/null; then
            log "BUILDS FAILED: every gbdt build failed; holding the droplet for $EVID/patch.list"
            $SSH 'rm -f /root/symarms_out/DONE.all'
            continue
        fi
        log "payload DONE.all"
        break
    fi
    last=$($SSH 'tail -1 /root/symarms_out/ab.txt 2>/dev/null' 2>/dev/null)
    log "progress: $last"
    rsync -az -e "$SSH_RSYNC" "root@$IP:/root/symarms_out/" "$EVID/out/" 2>/dev/null
done
$SSH 'pkill -f sym_arms_box.sh; pkill -f forest_speed_arm.py' 2>/dev/null
rsync -az -e "$SSH_RSYNC" "root@$IP:/root/symarms_out/" "$EVID/out/" && log "fetched out/" || log "FETCH FAILED"
log "done; teardown follows"
}

main "$@"
