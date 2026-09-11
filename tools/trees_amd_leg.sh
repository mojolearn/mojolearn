#!/usr/bin/env bash
# The trees lane's DigitalOcean AMD session (ENGINEERING_RULES.md section 10):
# one MI325X that stays up while batches run over ssh, then is fetched,
# flushed, destroyed and VERIFIED gone by the same process that created it.
#
#   (take the shared lock first:
#    until mkdir /tmp/mojolearn-do-gpu.lock 2>/dev/null; do sleep 120; done
#    echo "amd-trees-leg $(date -u +%FT%TZ)" > /tmp/mojolearn-do-gpu.lock/owner)
#   nohup bash tools/trees_amd_leg.sh up --label leg1 [--minutes 52] > leg1.log 2>&1 &
#   tools/trees_amd_leg.sh ssh <cmd...>        run on the droplet
#   tools/trees_amd_leg.sh push <path...>      rsync working-tree paths into /root/mojolearn
#   tools/trees_amd_leg.sh pull <remote> <local>
#   tools/trees_amd_leg.sh status
#   tools/trees_amd_leg.sh release             end the hold now: fetch, flush, destroy, verify
#
# `up` holds the box until `release` or --minutes after the create (55 at
# most; a second leg, never an extension). Its EXIT trap fetches
# /root/trees_out, syncs and unmounts the data volume, DELETEs the droplet,
# proves it gone (GET /droplets/<id> 404 and a droplet listing without the
# name), cancels the local dead-man and only then removes the shared lock.
# Guards, in the order they are armed: the lock must be held by this lane; a
# preflight listing refuses while any GPU droplet exists; a detached LOCAL
# dead-man keyed by tag and name is armed BEFORE the create; an ON-DROPLET
# watchdog DELETEs its own droplet at 60 minutes after the create and is
# verified (process alive, id baked in, the token proven by a GET that
# returns 200) or the droplet is destroyed on the spot. The token lives in
# 0600 curl config files and reaches the droplet on ssh stdin, never argv.
# The source is `git archive HEAD` without results, corpora and oracles,
# checked by sha256 on both ends. Datasets live on the persistent tor1 volume
# mojolearn-data-tor1 so a second leg does not download Istella-S again.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 9
STATE="${TREES_AMD_STATE:-$HOME/mojolearn-evidence/trees_amd_leg/state}"
PULL_ROOT="${TREES_AMD_PULL:-$HOME/mojolearn-evidence/mi325x_2026-09-11_taxi_istella}"
API=https://api.digitalocean.com/v2
SSH_KEY_FP="df:f7:6b:0c:56:da:48:a5:6f:6d:ae:44:af:de:f3:0b"
NAME=mojolearn-trees-amd; TAG=trees; REGION=tor1; SIZE=gpu-mi325x1-256gb; IMAGE=188571990
VOLUME=mojolearn-data-tor1; VOL_MOUNT=/mnt/mojolearn-data
LOCK=/tmp/mojolearn-do-gpu.lock
WATCHDOG_SECONDS=3600
TOKFILE="${MOJOLEARN_DO_TOKEN_FILE:-$HOME/.mojolearn_do_token}"
mkdir -p "$STATE"
RSH="ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$STATE/known_hosts -o ConnectTimeout=15 -o ServerAliveInterval=30 -o BatchMode=yes"

log() { printf '[%s trees-amd] %s\n' "$(date +%T)" "$*"; }
die() { printf '[%s trees-amd] %s\n' "$(date +%T)" "$*" >&2; exit "${2:-1}"; }
sha256_of() { shasum -a 256 "$1" 2>/dev/null | awk '{print $1}' || sha256sum "$1" | awk '{print $1}'; }
bounded() {  # <seconds> <cmd...>: macOS has no timeout(1)
    local s=$1; shift
    "$@" & local p=$!
    ( sleep "$s"; kill "$p" 2>/dev/null ) & local w=$!
    wait "$p"; local rc=$?
    kill "$w" 2>/dev/null; wait "$w" 2>/dev/null
    return $rc
}
box() { # shellcheck disable=SC2086
    $RSH "root@$(cat "$STATE/ip")" "$@"; }

TMPD=""; CURLRC=""
make_curlrc() {
    local perm tok
    [ -f "$TOKFILE" ] || die "token file $TOKFILE missing" 2
    perm=$(stat -f '%OLp' "$TOKFILE" 2>/dev/null || stat -c '%a' "$TOKFILE" 2>/dev/null)
    [ "$perm" = 600 ] || die "token file $TOKFILE is mode $perm, must be 600" 2
    TMPD="$(mktemp -d "${TMPDIR:-/tmp}/trees-amd.XXXXXX")"; chmod 700 "$TMPD"
    CURLRC="$TMPD/curlrc"
    IFS= read -r tok < "$TOKFILE"; tok="${tok//[$'\t\r\n ']/}"
    [ -n "$tok" ] || die "token file is empty" 2
    ( umask 077; printf 'header = "Authorization: Bearer %s"\nsilent\nshow-error\n' "$tok" > "$CURLRC" )
    TOK="$tok"
}
http() {  # <method> <url> <body-out> [json-data]
    local c
    if [ -n "${4:-}" ]; then
        c=$(curl -K "$CURLRC" --max-time 60 -o "$3" -w '%{http_code}' -X "$1" -H 'Content-Type: application/json' -d "$4" "$2") || c=000
    else
        c=$(curl -K "$CURLRC" --max-time 60 -o "$3" -w '%{http_code}' -X "$1" "$2") || c=000
    fi
    printf '%s' "$c"
}
jget() { python3 -c "import json,sys
try:
    d=json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
$2" "$1" "${3:-}"; }

# ------------------------------------------------------------------ up
CREATE_ATTEMPTED=0; DROPLET_ID=""; DESTROY_CONFIRMED=0; DEADMAN_PID=""; PULL_DIR=""

cancel_deadman() {
    [ -n "$DEADMAN_PID" ] || return 0
    pkill -P "$DEADMAN_PID" 2>/dev/null; kill "$DEADMAN_PID" 2>/dev/null
    rm -f "$STATE/deadman/curlrc"
    log "local dead-man cancelled (pid $DEADMAN_PID)"
}

teardown() {
    local rc=$? i c ids
    trap - EXIT INT TERM
    if [ "$CREATE_ATTEMPTED" = 1 ]; then
        log "teardown (exit $rc)"
        if [ -n "$DROPLET_ID" ] && [ -s "$STATE/ip" ]; then
            # [x]yz patterns so pkill -f cannot match this very remote shell.
            box 'pkill -f "[s]h /root/batch"; pkill -f "[f]orest_speed_arm.py"; pkill -f "[t]rees_amd_remote.sh"; pkill -f "[p]ip install"; sleep 2; echo stopped' 2>&1 | tail -1
            mkdir -p "$PULL_DIR"
            if bounded 300 rsync -az --exclude '*.so' --exclude '*.npz' --exclude '*.gz' --exclude '*.whl' \
                    -e "$RSH" "root@$(cat "$STATE/ip"):/root/trees_out/" "$PULL_DIR/"; then
                log "fetched /root/trees_out -> $PULL_DIR"
            else
                log "!! FETCH FAILED or timed out; whatever arrived is in $PULL_DIR"
            fi
            box "cat /root/selfkill.out 2>/dev/null; sync; umount $VOL_MOUNT && echo VOLUME_UNMOUNTED || echo VOLUME_UNMOUNT_FAILED; sync" 2>&1 | tail -3
        fi
        if [ -z "$DROPLET_ID" ]; then
            http GET "$API/droplets?tag_name=$TAG&per_page=200" "$TMPD/list.json" > /dev/null
            DROPLET_ID=$(jget "$TMPD/list.json" "print(' '.join(str(x['id']) for x in d.get('droplets',[]) if x['name']=='$NAME'))")
            [ -n "$DROPLET_ID" ] && log "found by name: $DROPLET_ID"
        fi
        local all_gone=1
        for id in $DROPLET_ID; do
            for i in 1 2 3 4 5 6; do
                c=$(http DELETE "$API/droplets/$id" /dev/null); log "DELETE droplet $id -> HTTP $c"
                case "$c" in 204|404) break ;; esac; sleep 10
            done
            local gone=0
            for i in 1 2 3 4 5 6 7 8 9 10; do
                c=$(http GET "$API/droplets/$id" /dev/null); log "GET droplet $id -> HTTP $c"
                [ "$c" = 404 ] && { gone=1; break; }; sleep 6
            done
            [ "$gone" = 1 ] || all_gone=0
        done
        c=$(http GET "$API/droplets?per_page=200" "$TMPD/final.json")
        ids=$(jget "$TMPD/final.json" "print(' '.join(str(x['id']) for x in d.get('droplets',[]) if x['name']=='$NAME'))")
        log "final GET /v2/droplets -> HTTP $c; droplets named $NAME: '${ids}'"
        if [ "$all_gone" = 1 ] && [ "$c" = 200 ] && [ -z "$ids" ]; then
            DESTROY_CONFIRMED=1
            echo "destroyed $DROPLET_ID verified $(date -u +%FT%TZ) (GET id 404, listing HTTP 200 without $NAME)" >> "$STATE/ledger.txt"
            log "DESTROY VERIFIED: $DROPLET_ID"
            mv "$STATE/droplet_id" "$STATE/droplet_id.reaped.$DROPLET_ID" 2>/dev/null
            rm -f "$STATE/ip" "$STATE/release" "$STATE/ready"
        fi
    fi
    if [ "$CREATE_ATTEMPTED" = 0 ] || [ "$DESTROY_CONFIRMED" = 1 ]; then
        cancel_deadman
        if [ "$CREATE_ATTEMPTED" = 1 ] && grep -q '^amd-trees-leg ' "$LOCK/owner" 2>/dev/null; then
            rm -rf "$LOCK" && log "shared lock released"
        fi
    else
        log "############ DROPLET ${DROPLET_ID:-unknown} MAY STILL BE BILLING: destruction NOT verified."
        log "############ Both dead-men and the lock are LEFT in place. Destroy it at https://cloud.digitalocean.com/droplets"
        [ "$rc" = 0 ] && rc=1
    fi
    [ -n "$TMPD" ] && rm -rf "$TMPD"
    exit "$rc"
}

cmd_up() {
    local minutes=52 label="" i c
    while [ $# -gt 0 ]; do
        case "$1" in
            --minutes) shift; minutes="$1" ;;
            --label) shift; label="$1" ;;
            *) die "unknown option $1" 2 ;;
        esac; shift
    done
    [ -n "$label" ] || die "--label is required (names the pull directory)" 2
    [ "$minutes" -le 55 ] || die "--minutes $minutes refused: 55 is the cap; split into a second leg" 2
    grep -q '^amd-trees-leg ' "$LOCK/owner" 2>/dev/null || die "the shared lock $LOCK is not held by amd-trees-leg; take it first" 2
    [ -s "$STATE/droplet_id" ] && die "a droplet is recorded in $STATE/droplet_id; release it first" 2
    PULL_DIR="$PULL_ROOT/$label"
    make_curlrc
    trap teardown EXIT; trap 'exit 130' INT; trap 'exit 143' TERM

    c=$(http GET "$API/droplets?per_page=200" "$TMPD/pre.json")
    [ "$c" = 200 ] || die "preflight GET /v2/droplets -> HTTP $c; not renting"
    local busy
    busy=$(jget "$TMPD/pre.json" "print(' '.join('%s(%s)' % (x['name'], x['size_slug']) for x in d.get('droplets',[]) if x['size_slug'].startswith('gpu-') or x['name'].startswith('mojolearn')))")
    [ -z "$busy" ] || die "a GPU or mojolearn droplet exists: $busy; one GPU droplet at a time"

    local commit; commit=$(git rev-parse HEAD)
    git archive --format=tar HEAD -- . ':!bench/results' ':!mamba/corpus' ':!bench/oracle*' \
        ':!bench/minentropy_oracle.txt' ':!*.bin' ':!archive' ':!upstream' ':!docs' ':!paper' \
        | gzip -9 > "$TMPD/src.tgz"
    local bytes sha; bytes=$(wc -c < "$TMPD/src.tgz" | tr -d ' '); sha=$(sha256_of "$TMPD/src.tgz")
    log "source $commit: $bytes bytes gzipped, sha256 ${sha:0:16}"
    [ "$bytes" -le 15000000 ] || die "archive $bytes bytes is over the 15 MB cap"

    for h in https://pypi.org/ https://github.com/ https://www.google.com/; do
        curl -s -o /dev/null --max-time 8 "$h" || die "uplink probe failed on $h; nothing created"
    done

    # LOCAL DEAD-MAN, armed before the create, keyed by tag + name (+ id once known).
    mkdir -p "$STATE/deadman"; chmod 700 "$STATE/deadman"
    cp "$CURLRC" "$STATE/deadman/curlrc"; rm -f "$STATE/deadman/droplet_id.txt"
    cat > "$STATE/deadman/deadman.sh" <<DEADMAN
#!/bin/sh
sleep $WATCHDOG_SECONDS
D="$STATE/deadman"
ids="\$(cat "\$D/droplet_id.txt" 2>/dev/null)"
curl -K "\$D/curlrc" --max-time 30 -o "\$D/list.json" "$API/droplets?tag_name=$TAG&per_page=200"
ids="\$ids \$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(' '.join(str(x['id']) for x in d.get('droplets',[]) if x['name']=='$NAME'))" "\$D/list.json" 2>/dev/null)"
for id in \$ids; do
  c=\$(curl -K "\$D/curlrc" --max-time 30 -o /dev/null -w '%{http_code}' -X DELETE "$API/droplets/\$id")
  echo "\$(date -u +%FT%TZ) local dead-man DELETE \$id -> \$c" >> "$STATE/ledger.txt"
done
rm -f "\$D/curlrc"
DEADMAN
    nohup sh "$STATE/deadman/deadman.sh" > /dev/null 2>&1 < /dev/null &
    DEADMAN_PID=$!; disown "$DEADMAN_PID" 2>/dev/null || true
    kill -0 "$DEADMAN_PID" || die "local dead-man did not start"
    log "local dead-man pid $DEADMAN_PID armed (${WATCHDOG_SECONDS}s, tag $TAG, name $NAME)"

    # THE VOLUME (datasets and caches survive the lease).
    http GET "$API/volumes?region=$REGION&per_page=200" "$TMPD/vols.json" > /dev/null
    local vol_id; vol_id=$(jget "$TMPD/vols.json" "v=[x['id'] for x in d.get('volumes',[]) if x['name']=='$VOLUME']; print(v[0] if v else '')")
    if [ -z "$vol_id" ]; then
        c=$(http POST "$API/volumes" "$TMPD/vol.json" "{\"name\":\"$VOLUME\",\"region\":\"$REGION\",\"size_gigabytes\":20,\"filesystem_type\":\"ext4\"}")
        vol_id=$(jget "$TMPD/vol.json" "print(d.get('volume',{}).get('id',''))")
        log "created volume $VOLUME -> HTTP $c id ${vol_id:-none}"
    fi
    local vol_arg=""; [ -n "$vol_id" ] && vol_arg=",\"volumes\":[\"$vol_id\"]"

    CREATE_ATTEMPTED=1
    log "creating $NAME ($SIZE, $REGION, image $IMAGE); THE BILL STARTS HERE"
    c=$(http POST "$API/droplets" "$TMPD/create.json" \
        "{\"name\":\"$NAME\",\"region\":\"$REGION\",\"size\":\"$SIZE\",\"image\":$IMAGE,\"ssh_keys\":[\"$SSH_KEY_FP\"],\"tags\":[\"$TAG\"]$vol_arg}")
    DROPLET_ID=$(jget "$TMPD/create.json" "print((d.get('droplet') or {}).get('id',''))")
    local created; created=$(date +%s)
    if [ -z "$DROPLET_ID" ]; then
        log "create -> HTTP $c, no id: $(head -c 300 "$TMPD/create.json")"
        sleep 8
        http GET "$API/droplets?tag_name=$TAG&per_page=200" "$TMPD/list.json" > /dev/null
        DROPLET_ID=$(jget "$TMPD/list.json" "v=[str(x['id']) for x in d.get('droplets',[]) if x['name']=='$NAME']; print(v[0] if v else '')")
        [ -n "$DROPLET_ID" ] || die "create FAILED and no droplet named $NAME exists" 3
        log "adopted droplet $DROPLET_ID by name"
    fi
    echo "$DROPLET_ID" > "$STATE/droplet_id"; echo "$DROPLET_ID" > "$STATE/deadman/droplet_id.txt"
    echo "$created" > "$STATE/created_epoch"
    echo "created $DROPLET_ID $(date -u +%FT%TZ) label=$label commit=$commit" >> "$STATE/ledger.txt"
    log "droplet $DROPLET_ID"

    local ip="" st
    for i in $(seq 1 90); do
        http GET "$API/droplets/$DROPLET_ID" "$TMPD/d.json" > /dev/null
        st=$(jget "$TMPD/d.json" "print(d['droplet']['status'])")
        ip=$(jget "$TMPD/d.json" "v=[n['ip_address'] for n in d['droplet']['networks'].get('v4',[]) if n['type']=='public']; print(v[0] if v else '')")
        [ "$st" = active ] && [ -n "$ip" ] && break
        sleep 10
    done
    [ -n "$ip" ] || die "never became active" 4
    echo "$ip" > "$STATE/ip"; : > "$STATE/known_hosts"
    log "active at $ip"
    local ok=0
    for i in $(seq 1 60); do
        if box true 2>/dev/null; then ok=$((ok + 1)); [ "$ok" -ge 3 ] && break; else ok=0; fi
        sleep 5
    done
    [ "$ok" -ge 3 ] || die "ssh never settled" 5

    # ON-DROPLET WATCHDOG at 60 minutes after the create, verified.
    printf 'header = "Authorization: Bearer %s"\nsilent\n' "$TOK" | box 'umask 077; cat > /root/.mojolearn-do.curlrc'
    local left=$(( WATCHDOG_SECONDS - ($(date +%s) - created) ))
    sed -e "s|@SECS@|$left|" -e "s|@ID@|$DROPLET_ID|g" > "$TMPD/selfkill.sh" <<'SELFKILL'
#!/bin/sh
sleep @SECS@
for a in 1 2 3; do
  c=$(curl -K /root/.mojolearn-do.curlrc --max-time 30 -o /root/selfkill.body -w '%{http_code}' -X DELETE https://api.digitalocean.com/v2/droplets/@ID@)
  echo "$(date -u +%FT%TZ) DELETE @ID@ attempt $a -> $c" >> /root/selfkill.out
  case "$c" in 2*|404) break ;; esac
  sleep 10
done
SELFKILL
    box 'cat > /root/mojolearn-selfkill.sh && chmod 700 /root/mojolearn-selfkill.sh' < "$TMPD/selfkill.sh"
    box 'setsid nohup sh /root/mojolearn-selfkill.sh > /dev/null 2>&1 < /dev/null & echo $! > /root/selfkill.pid; sleep 1; echo started'
    local verify
    verify=$(box "pid=\$(cat /root/selfkill.pid); kill -0 \$pid 2>/dev/null && echo ALIVE=\$pid; grep -c 'droplets/$DROPLET_ID)' /root/mojolearn-selfkill.sh | sed 's/^/ID_BAKED=/'; echo GET=\$(curl -K /root/.mojolearn-do.curlrc --max-time 30 -o /dev/null -w '%{http_code}' https://api.digitalocean.com/v2/droplets/$DROPLET_ID)")
    log "on-droplet watchdog ($left s): $(echo "$verify" | tr '\n' ' ')"
    echo "$verify" | grep -q '^ALIVE=' && echo "$verify" | grep -q '^ID_BAKED=[1-9]' && echo "$verify" | grep -q '^GET=200$' \
        || die "on-droplet watchdog NOT verified; destroying" 6
    echo "watchdog $DROPLET_ID verified $(date -u +%FT%TZ): $(echo "$verify" | tr '\n' ' ')" >> "$STATE/ledger.txt"

    # SOURCE.
    bounded 900 scp -q -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$STATE/known_hosts" -o BatchMode=yes \
        "$TMPD/src.tgz" "root@$ip:/root/src.tgz" || die "scp failed" 7
    box "set -e; got=\$(sha256sum /root/src.tgz | awk '{print \$1}'); [ \"\$got\" = $sha ] || { echo SHA MISMATCH; exit 9; }
         rm -rf /root/mojolearn && mkdir -p /root/mojolearn /root/trees_out && tar xzf /root/src.tgz -C /root/mojolearn && rm /root/src.tgz
         echo $commit > /root/mojolearn/SHIPPED_COMMIT.txt; echo SOURCE-OK \$(find /root/mojolearn -type f | wc -l) files" || die "source unpack failed" 7

    # VOLUME MOUNT.
    box "dev=/dev/disk/by-id/scsi-0DO_Volume_$VOLUME; for i in \$(seq 1 30); do [ -e \$dev ] && break; sleep 2; done
         if [ -e \$dev ]; then mkdir -p $VOL_MOUNT; mountpoint -q $VOL_MOUNT || mount -o discard,defaults \$dev $VOL_MOUNT;
           mkdir -p $VOL_MOUNT/gbm-bench; echo VOLUME_MOUNTED; du -sh $VOL_MOUNT/gbm-bench/* 2>/dev/null; ls -la $VOL_MOUNT/gbm-bench/*/*.npz 2>/dev/null;
         else echo VOLUME_DEVICE_MISSING; fi" 2>&1 | sed 's/^/[volume] /'

    # SETUP, detached.
    box "cd /root/mojolearn && export MOJOLEARN_TREES_SKIP_XGB_ROCM=${MOJOLEARN_TREES_SKIP_XGB_ROCM:-0} MOJOLEARN_TREES_SKIP_LGBM_OPENCL=${MOJOLEARN_TREES_SKIP_LGBM_OPENCL:-0};
         if mountpoint -q $VOL_MOUNT; then export GBM_BENCH_DATA=$VOL_MOUNT/gbm-bench PIP_CACHE_DIR=$VOL_MOUNT/pip-cache RATTLER_CACHE_DIR=$VOL_MOUNT/rattler-cache; fi;
         setsid nohup sh tools/trees_amd_remote.sh > /root/trees_out/setup_console.log 2>&1 < /dev/null & echo SETUP-STARTED"
    : > "$STATE/ready"
    log "READY: ssh via 'tools/trees_amd_leg.sh ssh ...'; hold until release or $minutes min after create"

    local deadline=$(( created + minutes * 60 )) beat=0
    while [ "$(date +%s)" -lt "$deadline" ] && [ ! -f "$STATE/release" ]; do
        sleep 20; beat=$((beat + 1))
        if [ $((beat % 15)) = 0 ]; then
            log "hold: $(( (deadline - $(date +%s)) / 60 )) min left; $(box 'tail -1 /root/trees_out/ab.txt 2>/dev/null; tail -1 /root/trees_out/setup.txt' 2>/dev/null | tr '\n' ' ')"
        fi
    done
    [ -f "$STATE/release" ] && log "release requested" || log "hold deadline reached"
    exit 0
}

cmd_ssh() { [ -s "$STATE/ip" ] || die "no droplet ip in $STATE"; box "$@"; }
cmd_push() {
    [ -s "$STATE/ip" ] || die "no droplet ip in $STATE"
    [ $# -gt 0 ] || die "push needs paths"
    rsync -az --relative -e "$RSH" "$@" "root@$(cat "$STATE/ip"):/root/mojolearn/"
}
cmd_pull() {
    [ -s "$STATE/ip" ] || die "no droplet ip in $STATE"
    [ $# -eq 2 ] || die "pull <remote> <local>"
    mkdir -p "$2"
    rsync -az --exclude '*.so' --exclude '*.npz' --exclude '*.gz' --exclude '*.whl' -e "$RSH" "root@$(cat "$STATE/ip"):$1" "$2"
}
cmd_status() {
    echo "state $STATE"; for f in droplet_id ip created_epoch ready release; do [ -e "$STATE/$f" ] && echo "$f: $(cat "$STATE/$f")"; done
    [ -s "$STATE/created_epoch" ] && echo "minutes since create: $(( ($(date +%s) - $(cat "$STATE/created_epoch")) / 60 ))"
    cat "$LOCK/owner" 2>/dev/null | sed 's/^/lock: /'
    tail -5 "$STATE/ledger.txt" 2>/dev/null
}
cmd_release() { : > "$STATE/release"; log "release requested; the up process tears down within 20 s"; }

case "${1:-}" in
    up) shift; cmd_up "$@" ;;
    ssh) shift; cmd_ssh "$@" ;;
    push) shift; cmd_push "$@" ;;
    pull) shift; cmd_pull "$@" ;;
    status) cmd_status ;;
    release) cmd_release ;;
    *) sed -n '2,30p' "$0"; exit 2 ;;
esac
