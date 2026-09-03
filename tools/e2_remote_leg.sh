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

# FLUSH THE VOLUME BEFORE THE DEVICE GOES AWAY.
#
# DigitalOcean detaches the block device when the droplet is destroyed, and
# nothing on the host syncs it first. On 2026-09-03 the seed leg reported
# "higgs decoded to .../higgs_speed.npz" and was torn down about a minute
# later; the next leg found a TRUNCATED gzip -- "No data left in file", 2.7G
# of a 2.8G download -- and no .npz at all, because both writes were still in
# page cache. The decode had genuinely succeeded and the bytes were still
# lost.
#
# A dirty page that never reached the disk is the worst kind of cache: the
# volume readback says the directory is there, so the next leg believes it is
# warm and fails on the contents instead of on the absence. Sync and unmount
# before every teardown, and say whether it worked.
flush_volume() {
  [ -z "${E2_VOLUME:-}" ] && return 0
  [ -z "${IP:-}" ] && return 0
  [ -z "${SSH:-}" ] && return 0   # the leg died before SSH was composed
  log "syncing and unmounting $VOL_MOUNT before teardown"
  $SSH "sync; umount $VOL_MOUNT 2>/dev/null && echo VOLUME_UNMOUNTED || echo VOLUME_UNMOUNT_FAILED; sync" \
    2>&1 | sed "s/^/[$VENDOR vol] /" || log "!! volume flush failed; its contents may be incomplete"
}

destroy() {
  flush_volume
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

# A PERSISTENT DATA VOLUME, BECAUSE THE BIG DATASET OUTLIVES THE LEASE.
#
# `tools/speed_gbdt_arm.py:load_higgs` says it plainly: the download is
# 2.6 GB, the decode is another 1.3 GB of float32, and it is "a SEPARATE,
# EXPLICITLY NAMED STEP, never something a timed run does on its own" --
# the orchestrator budgets a lease around it. On 2026-09-03 a leg folded
# the fetch into the measurement lease anyway and the whole hour went to
# the gzip-CSV decode: builds finished in five minutes, then forty-six
# minutes of pandas and the work bound killed it (EXIT=124).
#
# A destroyed droplet takes its cache with it, so paying that once per leg
# is paying it forever. E2_VOLUME names a DigitalOcean block volume that is
# attached at create and mounted at /mnt/mojolearn-data; the dataset store
# lives there and survives teardown. Block storage is about $0.10/GB/month,
# so ten gigabytes costs a dollar a month and removes forty minutes from
# every leg that reads HIGGS -- and five of them are queued.
#
# VOLUMES ARE REGION-LOCKED. This one is created in $REGION on first use, so
# an nyc2 volume serves the NVIDIA legs and an AMD leg in tor1 would need
# its own. That is correct rather than limiting: the vendor comparison is an
# NVIDIA question.
VOLUME_ARG=""
VOL_MOUNT="/mnt/mojolearn-data"
if [ -n "${E2_VOLUME:-}" ]; then
  VOL_ID=$(api "$API/volumes?region=$REGION&per_page=100" | python3 -c "import json,sys
d=json.load(sys.stdin); v=[x['id'] for x in d.get('volumes',[]) if x['name']=='${E2_VOLUME}']
print(v[0] if v else '')")
  if [ -z "$VOL_ID" ]; then
    log "volume ${E2_VOLUME} not found in $REGION; creating ${E2_VOLUME_GB:-20} GB"
    VOL_ID=$(api -X POST -H "Content-Type: application/json" \
      -d "{\"name\":\"${E2_VOLUME}\",\"region\":\"$REGION\",\"size_gigabytes\":${E2_VOLUME_GB:-20},\"filesystem_type\":\"ext4\"}" \
      "$API/volumes" | python3 -c "import json,sys
try:
    d=json.load(sys.stdin); print(d.get('volume',{}).get('id',''))
except Exception:
    print('')")
  fi
  if [ -n "$VOL_ID" ]; then
    log "attaching volume ${E2_VOLUME} ($VOL_ID) at create"
    VOLUME_ARG=",\"volumes\":[\"$VOL_ID\"]"
  else
    log "!! could not resolve or create volume ${E2_VOLUME}; continuing WITHOUT it"
  fi
fi

log "creating $NAME ($SIZE, $REGION)"
CREATE_BODY="$(api -X POST -H "Content-Type: application/json" \
  -d "{\"name\":\"$NAME\",\"region\":\"$REGION\",\"size\":\"$SIZE\",\"image\":$IMAGE,\"ssh_keys\":[\"$SSH_KEY_FP\"],\"tags\":[\"e2\"]$VOLUME_ARG}" \
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

# A SECOND DEAD-MAN, ON THE DROPLET, BECAUSE THE FIRST ONE DIES WITH THIS MAC.
#
# The dead-man armed before the create is `nohup bash -c "sleep N; curl
# DELETE"` RUNNING LOCALLY. It does not survive the laptop sleeping, losing
# its network, or being shut down, and an MI325X left running is a real bill.
# RunPod has never had this exposure: `tools/runpod_guard.sh` installs its
# watchdog ON THE POD, so the pod kills itself whatever happens here. This
# gives the droplet the same property.
#
# Both timers are kept. The local one is the one that can be CANCELLED on a
# clean teardown; this one is the one that cannot be lost. A droplet that is
# already destroyed makes this DELETE a harmless 404.
log "arming a second dead-man ON THE DROPLET (${DEADMAN_SECONDS}s), which the local one cannot guarantee"
$SSH "umask 077
cat > /tmp/mojolearn-selfkill.sh <<'SELFKILL'
#!/bin/sh
sleep __SECS__
code=\$(curl -s -o /tmp/selfkill.body -w '%{http_code}' -X DELETE \
  -H 'Authorization: Bearer __TOK__' \
  'https://api.digitalocean.com/v2/droplets/__ID__')
echo \"\$(date -u +%FT%TZ) DELETE __ID__ -> \$code\" >> /tmp/selfkill.out
SELFKILL
sed -i 's|__SECS__|$DEADMAN_SECONDS|; s|__TOK__|$TOK|; s|__ID__|$DROPLET_ID|' /tmp/mojolearn-selfkill.sh
chmod 700 /tmp/mojolearn-selfkill.sh
nohup /tmp/mojolearn-selfkill.sh >/tmp/selfkill.log 2>&1 &
echo \$! > /tmp/selfkill.pid
sleep 1
kill -0 \$(cat /tmp/selfkill.pid) 2>/dev/null && echo ON_DROPLET_DEADMAN_ARMED || echo ON_DROPLET_DEADMAN_FAILED" \
  2>&1 | tail -1 | sed "s/^/[$VENDOR] /"
$SSH 'nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || rocm-smi --showproductname 2>/dev/null | grep -i "card series\|name" | head -2' | sed "s/^/[$VENDOR gpu] /"

# SHIP THE COMMIT, NOT THE WORKING TREE. The first E2 legs rsync'd a
# worktree whose numerics.mojo the Mac's own bootstrap had ALREADY flipped
# to IDENTICAL (the flip is session-local and the two sessions shared the
# tree), and whose .git was a worktree pointer file; the remote bootstrap
# refused the flip and git had no repository. A bundle of HEAD is exact
# by construction: the remote clones it, checks out the hash, and the
# commit recorded in every artifact is the commit that ran.
# MEASURED COST, AND THE OBVIOUS FIX DOES NOT EXIST. 2026-08-31: the bundle
# is 236 MB and the scp takes ELEVEN MINUTES of a sixty-minute lease, a fifth
# of the lease spent moving bytes no one on the box reads. Twenty-seven of
# these were sitting in /tmp, 5.6 GB.
#
# `--depth=1` WAS TRIED AND IS NOT A `git bundle` ARGUMENT (git 2.50.1:
# "error: unrecognized argument: --depth=1"). A bundle of a truncated history
# records a PREREQUISITE the receiving box does not have, so there is no
# shallow bundle to make here.
#
# THE ROUTE THAT WORKS is `git archive` at the pinned sha with
# `':!bench/results'` excluded, which `tools/gemm_remote_leg.sh:3270` already
# does and is exactly why the RunPod leg never pays this. Adopting it here
# costs the one property the bundle gives for free: the remote `git clone`
# checks out by hash and so cannot silently receive the wrong tree, whereas
# an archive has to be checksummed separately (the RunPod leg does that, with
# `source_sha256_local.txt`). That is the swap, it is not hard, and it wants
# its own leg to verify rather than riding along on one that is measuring
# something else.
#
# Untracking bench/'s build artifacts (2026-08-31, 235 files, 333 MB) shrank
# the WORKING TREE but not this bundle, because history still carries the
# blobs. Only the transport change fixes the transport.
COMMIT="$(git -C "$REPO" rev-parse "${E2_COMMIT:-HEAD}")"   # pin to the Mac reference run's commit
BUNDLE="/tmp/mojolearn-e2-$COMMIT.tgz"
# `git archive`, NOT `git bundle`, SINCE 2026-08-31, AND IT IS A BLOCKER FIX
# RATHER THAN AN OPTIMISATION.
#
# A bundle carries the whole history, and this history is mostly
# `bench/results`. Measured: 236 MB against 20 MB for the archive below.
# The 2026-08-31 20:51 lanes leg spent its ENTIRE SIXTY-MINUTE LEASE on the
# scp, the on-droplet dead-man destroyed the box mid-transfer at 3600s, the
# scp then failed five times against a box that no longer existed, and the
# leg tore down having run zero lanes and fetched nothing. One lease, one
# droplet, no result. At that size no AMD leg can complete at all.
#
# THE ONE PROPERTY THE BUNDLE GAVE FREE IS REPLACED, NOT DROPPED. A remote
# `git clone` checks out by hash and so cannot silently receive the wrong
# tree; a tarball cannot. So the sha256 of the archive is computed on BOTH
# SIDES and compared before anything runs, which is what
# `tools/gemm_remote_leg.sh` already does and why the RunPod leg never paid
# this cost. `bench/results` is excluded because nothing on the box reads it.
# THE MAC'S UPLINK IS NOT IN THIS PATH ANY MORE (2026-09-02).
#
# The archive route below is correct and it is SLOW FOR ONE REASON THAT IS
# NOT ITS FAULT: this machine uploads at about 68 KB/s, measured, so an 80 MB
# archive is fifty-five minutes of a lease that is only two hours long. Three
# RunPod legs died of exactly this the same morning, one of them after 22
# minutes at 24 KB/s with the tar still running.
#
# THE REPOSITORY IS PUBLIC AND THE BOX HAS A DATACENTER LINK, so the box
# fetches the commit itself: measured 3.2 SECONDS on an MI325X droplet for
# this same commit, against fifty-five minutes the other way. `--depth 1` of
# ONE SHA (GitHub serves a by-sha fetch for a public repo), sparse-checkout
# excluding bench/results exactly as the archive did, and then the check that
# replaces the archive's sha256: the box reports `git rev-parse HEAD` and it
# must equal the commit this leg was told to ship. That is a STRONGER
# provenance statement than the tarball's, because it is git's own hash of
# the tree rather than a hash of one transfer of it.
#
# The archive path stays as the fallback for a private mirror, an offline
# commit, or a GitHub outage: set MOJOLEARN_LEG_NO_CLONE=1 to force it.
CLONE_URL="${MOJOLEARN_LEG_CLONE_URL:-https://github.com/mojolearn/mojolearn.git}"
CLONE_OK=0

# WAIT FOR SSH TO BE STABLE BEFORE SPENDING THE CHEAP PATH ON IT. A droplet
# reports `active` before sshd is reliably accepting, and the clone below is
# ONE attempt: a single "Connection closed by <ip> port 22" during boot sends
# the leg down the 81 MB archive fallback, which on this uplink is about
# twenty minutes of a sixty-minute lease -- the exact cost the clone exists
# to avoid. Measured 2026-09-03: droplet active at 03:16:59, clone attempted
# at 03:18:02, connection closed, archive path taken.
#
# Three CONSECUTIVE successes, because the failure mode is a port that
# accepts once and then resets while sshd finishes starting.
ssh_settle() {
  _ok=0
  for _i in $(seq 1 40); do
    if $SSH true 2>/dev/null; then
      _ok=$((_ok + 1))
      [ "$_ok" -ge 3 ] && { log "ssh settled after $_i probe(s)"; return 0; }
    else
      _ok=0
    fi
    sleep 5
  done
  log "ssh NEVER settled after 40 probes; continuing anyway"
  return 1
}
ssh_settle

# MOUNT THE PERSISTENT VOLUME BEFORE ANYTHING READS A DATASET.
#
# DigitalOcean attaches the volume as a device and mounts NOTHING; an
# unmounted volume looks exactly like an empty cache, so the fetch would run
# again and the whole point would be lost silently. This mounts by the stable
# by-id path, creates the dataset store on it, and READS BACK what is already
# there so the log says whether this leg will pay for the decode or skip it.
if [ -n "${E2_VOLUME:-}" ]; then
  log "mounting volume ${E2_VOLUME} at $VOL_MOUNT"
  $SSH "set -e
    dev=/dev/disk/by-id/scsi-0DO_Volume_${E2_VOLUME}
    if [ ! -e \"\$dev\" ]; then echo VOLUME_DEVICE_MISSING; exit 0; fi
    mkdir -p $VOL_MOUNT
    mountpoint -q $VOL_MOUNT || mount -o discard,defaults \"\$dev\" $VOL_MOUNT
    mkdir -p $VOL_MOUNT/gbm-bench
    echo VOLUME_MOUNTED \$(df -h $VOL_MOUNT | tail -1 | awk '{print \$2\" total, \"\$4\" free\"}')
    echo VOLUME_HOLDS: \$(ls -1 $VOL_MOUNT/gbm-bench 2>/dev/null | tr '\n' ' ')
    # A DIRECTORY NAME IS NOT A DATASET. The 2026-09-03 leg saw
    # \"VOLUME_HOLDS: higgs\" over a truncated gzip and no decoded array, so
    # the readback names the file the loader actually opens and its size.
    for f in $VOL_MOUNT/gbm-bench/higgs/higgs_speed.npz $VOL_MOUNT/gbm-bench/higgs/HIGGS.csv.gz; do
      [ -f \"\$f\" ] && echo VOLUME_FILE \$(basename \$f) \$(stat -c %s \"\$f\") bytes \
                     || echo VOLUME_FILE \$(basename \$f) ABSENT
    done
    du -sh $VOL_MOUNT/gbm-bench 2>/dev/null | awk '{print \"VOLUME_USED \"\$1}'" \
    2>&1 | sed "s/^/[$VENDOR vol] /" || log "!! volume mount step failed; the fetch will run as if cold"
fi

if [ "${MOJOLEARN_LEG_NO_CLONE:-0}" != "1" ] && [ -n "$CLONE_URL" ]; then
  # RETRY THE CLONE. A commit mismatch (exit 9) is a real failure and must
  # not be retried; a transport failure is worth another try before paying
  # for the archive.
  for _try in 1 2 3; do
  log "cloning $COMMIT ON THE BOX from $CLONE_URL (bench/results excluded), attempt $_try"
  if $SSH "set -e
      rm -rf /root/mojolearn && mkdir -p /root/mojolearn
      cd /root/mojolearn
      git init -q
      git remote add origin '$CLONE_URL'
      git -c protocol.version=2 fetch -q --depth 1 origin '$COMMIT'
      git sparse-checkout set --no-cone '/*' '!bench/results' >/dev/null 2>&1 || true
      git checkout -q --detach FETCH_HEAD
      got=\$(git rev-parse HEAD)
      if [ \"\$got\" != '$COMMIT' ]; then
        echo \"CLONE COMMIT MISMATCH: want $COMMIT got \$got\"; exit 9
      fi
      printf '%s\\n' '$COMMIT' > commit.txt
      echo CLONE-COMMIT-OK \$got
      grep -c 'is_defined\\[\"MOJOLEARN_NUMERIC_IDENTICAL\"\\]' checks/numerics.mojo"; then
    CLONE_OK=1
    break
  else
    _rc=$?
    if [ "$_rc" = 9 ]; then
      log "clone reported a COMMIT MISMATCH (exit 9); not retrying"
      break
    fi
    log "clone attempt $_try failed (rc $_rc)"
    [ "$_try" -lt 3 ] && sleep 10
  fi
  done
  [ "$CLONE_OK" = 1 ] || log "the clone path failed 3 times; falling back to the archive upload"
fi

if [ "$CLONE_OK" != 1 ]; then
  log "archive $COMMIT (git archive, bench/results excluded)"
  git -C "$REPO" archive --format=tar "$COMMIT" -- . ':!bench/results' \
    | gzip > "$BUNDLE" || { log "archive failed"; exit 6; }
  SRC_SHA=$( { shasum -a 256 "$BUNDLE" 2>/dev/null || sha256sum "$BUNDLE"; } | awk '{print $1}' )
  log "  archive $(du -h "$BUNDLE" | awk '{print $1}'), sha256 ${SRC_SHA:0:16}"
  # THE BUNDLE COPY RETRIES. A box answers `echo SSH-OK` before its sshd is
  # settled, and the very next scp can still come back "Connection closed" --
  # which is how the 2026-08-28 13:21 AMD identity leg died thirty seconds
  # after a healthy `rocm-smi`, at a cost of one droplet, one lease and the
  # whole column. The readiness probe above is necessary and it is not
  # sufficient; a transient here is worth five tries, not a leg.
  _scp_ok=0
  for _try in 1 2 3 4 5; do
    if scp -q -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 \
          "$BUNDLE" "root@$IP:/root/e2.tgz"; then _scp_ok=1; break; fi
    log "scp attempt $_try failed; retrying in 15s"
    sleep 15
  done
  [ "$_scp_ok" = 1 ] || { log "scp failed after 5 attempts"; exit 6; }
  # THE INTEGRITY CHECK THAT REPLACES `git clone`'s. The box recomputes the
  # sha256 of what it received and refuses if it differs from what was sent.
  # `commit.txt` is written from HERE, because an archive has no .git and the
  # box therefore cannot tell you which commit it is running; every artifact
  # the leg files is attributed from that file.
  $SSH "set -e
        cd /root
        got=\$( sha256sum e2.tgz | awk '{print \$1}' )
        if [ \"\$got\" != \"$SRC_SHA\" ]; then
          echo \"ARCHIVE SHA MISMATCH: sent $SRC_SHA got \$got\"; exit 9
        fi
        echo ARCHIVE-SHA-OK
        rm -rf /root/mojolearn && mkdir -p /root/mojolearn
        tar -xzf e2.tgz -C /root/mojolearn
        cd /root/mojolearn
        printf '%s\\n' '$COMMIT' > commit.txt
        cat commit.txt
        grep -c 'is_defined\\[\"MOJOLEARN_NUMERIC_IDENTICAL\"\\]' checks/numerics.mojo" \
    || { log "remote unpack/verify failed"; exit 6; }
fi

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
# THE PHASE-9 DIAG KNOBS PASS THROUGH THIS LINE TOO (2026-08-29):
# MOJOLEARN_P9_DIAG (a script path in the commit, run on the box after phase
# 9's builds), MOJOLEARN_P9_ONLY_DIAG (1 = run nothing in phase 9 but that
# script), MOJOLEARN_P9_DIAG_TIMEOUT, and MOJOLEARN_WHEEL_VERSION / _INDEX for
# packaging/linux/leg_diag_install.sh. tools/e1_bootstrap.sh honours them;
# until this line forwarded them an exported MOJOLEARN_P9_DIAG never reached
# the box, because ssh starts a fresh environment.
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
  $SSH "export PATH=/root/.pixi/bin:\$PATH; cd /root/mojolearn && MOJOLEARN_COMMIT='$COMMIT' MOJOLEARN_E1_PHASES='${MOJOLEARN_E1_PHASES:-}' MOJOLEARN_E1_LANES='$WAVE' MOJOLEARN_P9_BINDINGS='${MOJOLEARN_P9_BINDINGS:-}' MOJOLEARN_P9_LANES='${MOJOLEARN_P9_LANES:-}' ${MOJOLEARN_P9_TIERS:+MOJOLEARN_P9_TIERS='$MOJOLEARN_P9_TIERS'} MOJOLEARN_P9_BREAK='${MOJOLEARN_P9_BREAK:-0}' MOJOLEARN_P9_VENDOR='amd-mi325x' ${MOJOLEARN_P9_DIAG:+MOJOLEARN_P9_DIAG='$MOJOLEARN_P9_DIAG'} MOJOLEARN_P9_ONLY_DIAG='${MOJOLEARN_P9_ONLY_DIAG:-0}' MOJOLEARN_P9_DIAG_TIMEOUT='${MOJOLEARN_P9_DIAG_TIMEOUT:-2400}' ${MOJOLEARN_WHEEL_VERSION:+MOJOLEARN_WHEEL_VERSION='$MOJOLEARN_WHEEL_VERSION'} MOJOLEARN_WHEEL_INDEX='${MOJOLEARN_WHEEL_INDEX:-testpypi}' ${MOJOLEARN_GPU_ARCHS:+MOJOLEARN_GPU_ARCHS='$MOJOLEARN_GPU_ARCHS'} timeout -k 30 $WORK_SECONDS bash tools/e1_bootstrap.sh > /root/e2_run_w$WAVE_N.log 2>&1; echo \"WAVE-$WAVE_N-EXIT=\$?  (124 = hit the work bound)\"; tail -30 /root/e2_run_w$WAVE_N.log"
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
      gemm-backward) CMD='pixi run mojo run -I . gemm/checks/gemm_backward_check.mojo' ;;
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
      # THE TUNED ARM: does it hold the bits on silicon that is not Apple,
      # and is the 3.5x Apple's or the kernel's. Both arms in ONE binary,
      # output poisoned before each, poison counted, so a kernel that never
      # wrote cannot agree by accident.
      tuned-probe)   CMD='pixi run mojo run -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_TUNED_ARM=1 gemm/checks/gemm_tuned_probe.mojo'; WRAP=0 ;;
      # THE ONE-VARIABLE PRICE OF THE FOLD PIN. Three arms in ONE binary
      # alternating call by call, so the governor drift cancels at the arm
      # level instead of being averaged over rounds. Measured 1.52x-1.55x on
      # an M4; this is the leg that asks whether that number is Apple's or
      # the profile's. WRAP=0: the driver builds its own arms and forcing it
      # under one mode would make both of them the same binary.
      unpinned-price) CMD='pixi run mojo run -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_UNPINNED_ARM=1 gemm/checks/gemm_unpinned_price.mojo'; WRAP=0 ;;
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
      # THE EXTRATREES BREAKDOWN ON THIS VENDOR. Section 2.3 of the 2026-08-28
      # board has ET 4.4x slower on the MI325X than on the H100 with the same
      # source, while rf and iforest are faster there; nobody had profiled
      # it. This runs fit_once's PhaseClock at the higgs speed-arm shape, at
      # DEVICE_TPB 128/256/512, under rocprofv3 where the image has it.
      # WRAP=0: it is a FAST-tier measurement, and the script builds its own
      # arms.
      # Every ET_PROFILE_* knob crosses the ssh line; leg 3 on 2026-08-29
      # forwarded only ARMS, so the OLD-commit arm (DEVIATION 1945) never ran.
      et-profile)    CMD="env ET_PROFILE_ARMS='${ET_PROFILE_ARMS:-shipped 128}' ET_PROFILE_OLD_COMMIT='${ET_PROFILE_OLD_COMMIT:-}' ET_PROFILE_ROWS2='${ET_PROFILE_ROWS2:-}' ET_PROFILE_SKIP_BREAK='${ET_PROFILE_SKIP_BREAK:-}' ET_PROFILE_ROWS='${ET_PROFILE_ROWS:-}' ET_PROFILE_TREES='${ET_PROFILE_TREES:-}' bash tools/et_profile_leg.sh"; WRAP=0 ;;
      # THE IDENTITY TAX, MEASURED WHERE IT CAN BE BELIEVED. Every priced
      # number this repository has published came off ONE Apple M4 -- a
      # 16 GB laptop that shares its memory between the page cache and the
      # GPU. On 2026-09-03 that box was carrying 7.5 GB of swap on an 8 GB
      # device, and the SAME SOURCE that measured the Gram product at
      # 5.27 ms in August measured it at 26.33 ms there. The failure is not
      # that the box is slower. A fixed per-launch cost lands in BOTH arms
      # and drags every ratio toward 1.0, so a loaded box UNDERSTATES what
      # identity costs and does it worst on the smallest arms -- which is
      # how `nt` came to read 0.94x on the very code that read 5.69x.
      # A rented single-tenant GPU has no laptop governor, no swap and no
      # Chrome, so this is the column where the tax is a measurement
      # rather than an artifact of the machine.
      #
      # WRAP=0 IS LOAD-BEARING HERE, for the reason stated at the top of
      # this loop: lanes_price.sh builds BOTH arms itself. Forcing it under
      # with_identical_mode.sh would make its FAST arm a second IDENTICAL
      # binary and every ratio 1.0 by construction.
      #
      # The output goes under the phase-8 run directory because the fetch
      # at the bottom of this file rsyncs bench/results/e1/ and nothing
      # else; a price written to the harness's own default path would be
      # destroyed with the droplet.
      lanes-price)   CMD="env MOJOLEARN_LANES_PRICE_OUT=\"\$OUT/lanes_price\" MOJOLEARN_LANES_PRICE_LANES='${LANES_PRICE_LANES:-kmeans knn dbscan gram nt gemv gbdt rf et}' MOJOLEARN_LANES_PRICE_ROUNDS=${LANES_PRICE_ROUNDS:-5} sh tools/lanes_price.sh"; WRAP=0 ;;
      # THE SIZE SWEEP, and why it is a separate check rather than more rounds.
      # The first priced columns (MI325X and H100, 2026-09-03) returned three
      # results that a single fixture cannot explain: `nt` came back 0.69x on
      # AMD and 1.31x on NVIDIA -- opposite signs -- and on AMD the two modes
      # returned BIT-IDENTICAL output while taking different times, which is
      # the fast path walking a slower route to the same answer rather than
      # identity being free. `kmeans` on the H100 did the same thing at 1.04x.
      #
      # A ratio that inverts between vendors at ONE fixture is not yet a
      # finding about either vendor. The question it raises is whether the
      # ratio is a property of the CONTRACT or of the SHAPE, and only a sweep
      # answers that: if identity's cost grows with the fixture, the small
      # readings were launch overhead; if it holds, the sign is real.
      #
      # Three size points per lane, from the harness's own documented steps
      # (bench/lanes_price_main.mojo's knob table). The tree lanes keep the
      # 1M-row floor at their smallest point, because a tree measured below
      # 1M rows is not admissible here at all.
      # THE SIZE SWEEP. Three fixture points per lane, so a ratio can be
      # attributed to the CONTRACT or to the SHAPE. See the header of
      # tools/lanes_price_sweep.sh for why more sizes and not more rounds.
      # The body lives in that script rather than inline here: the first
      # attempt escaped a `bash -c` through the ssh line and exited 127.
      lanes-price-sweep)
        CMD="env MOJOLEARN_SWEEP_OUT=\"\$OUT/sweep\" MOJOLEARN_SWEEP_ROUNDS=${SWEEP_ROUNDS:-3} sh tools/lanes_price_sweep.sh"; WRAP=0 ;;
      # DEVIATION 2040. Does capping FAST's histogram replication restore
      # the expected sign? IDENTICAL beat FAST on the gbdt lane at every
      # fixture on both devices -- 0.731 on a 304-CU MI325X, 0.938 on a
      # 132-SM H100 -- and identity is not supposed to be free, so the FAST
      # arm is the thing under suspicion. The effect tracking the core count
      # is the signature: `replication_for` multiplies it.
      #
      # BOTH ARMS ARE FAST, so the numeric contract is held constant and
      # only the grid changes. The harness refuses to report a ratio on any
      # row whose two output hashes differ, because a replication change
      # that moved bits would be a tradeoff and not a win.
      # ONE ARM PER DEFINE, SEQUENTIALLY, NEVER COMBINED. Combining them
      # would make a null result unattributable, which is the whole value of
      # the 2040 run: it was inert, and because it was the only thing
      # changed, "inert" means something.
      fast-replication-ab)
        CMD="env MOJOLEARN_AB_DEFINES=\"${AB_DEFINES:-MOJOLEARN_2041_FAST_CHUNKS_PIN MOJOLEARN_2042_FAST_NO_LOOKBACK}\" MOJOLEARN_AB_OUT_BASE=\"\$OUT/fast_replication_ab\" MOJOLEARN_AB_ROUNDS=${AB_ROUNDS:-3} MOJOLEARN_AB_ROWS=${AB_ROWS:-1000000} sh tools/fast_replication_ab_all.sh"; WRAP=0 ;;
      # THE CHECKPOINT FILE, WRITTEN ON THIS BOX SO IT CAN BE COMPARED TO
      # ANOTHER BOX'S BYTES. The gate's own closing line is the reason this
      # check exists: "ALL CLAUSES GREEN on ONE device. That is a serializer
      # result and not a cross-vendor one... TWO BOXES WRITING THE SAME BYTES
      # is the claim, and this gate cannot manufacture the second file."
      #
      # The file lands under the run directory so the fetch brings it back;
      # comparing it is a `cmp` on the orchestrator's machine, not on the
      # droplet, because a comparison run on one of the two boxes that made
      # the files is a weaker arrangement than one run on neither.
      train-checkpoint)
        CMD="env MOJOLEARN_TRAIN_STEPS=${TRAIN_STEPS:-8} MOJOLEARN_TRAIN_CKPT_FILE=\"\$OUT/lanes/checkpoint.ckptbin\" pixi run mojo run -I . training/checks/checkpoint_check.mojo" ;;
      column)        CMD='pixi run mojo run -I . matrix_main.mojo' ;;
      gpu-probe)     CMD='pixi run mojo run -I . probe_main.mojo' ;;
      # THE TRANSFORMER BACKWARD. The comment here said "no check driver
      # yet, so there is nothing to run" -- that was true until 2026-09-03,
      # when the driver compiled for the first time and passed all six
      # clauses on Apple under IDENTICAL (37 stages, 313,308 cells, plus the
      # batch and length halves with their negative controls, the chunked
      # arm, the row-39 refusal and clause (f)'s exact-integer correctness
      # check). It is the cheapest cross-vendor result available: the code
      # exists, it passes, and the only thing between it and a second column
      # is a leg.
      #
      # ALL SIX CLAUSES, because the default run skips (b) through (f) and a
      # column that ran only (a) would be a weaker column than Apple's.
      transformer-backward)
        CMD='env MOJOLEARN_TFB_CHECK_CLAUSE_B=1 MOJOLEARN_TFB_CHECK_CLAUSE_C=1 MOJOLEARN_TFB_CHECK_CLAUSE_D=1 MOJOLEARN_TFB_CHECK_CLAUSE_E=1 MOJOLEARN_TFB_CHECK_CLAUSE_F=1 pixi run mojo run -I . transformer/checks/transformer_backward_check.mojo' ;;
      # THE LOSSGUIDE RUNG DEVIATION 1902 ASKED FOR. `fast_replication_ab`
      # runs lanes gbdt/rf/et out of lanes_price_main, which has no
      # lossguide and no depthwise lane -- so the half of 1902's own stated
      # gate that could VINDICATE the row has never been runnable by the
      # harness that keeps ruling on it. Its saving is proportional to the
      # sequential splits a LOSSGUIDE tree runs, and the symmetric lanes
      # collect none of it by construction.
      grow-policy-ab)
        CMD="env GBM_BENCH_DATA=$VOL_MOUNT/gbm-bench MOJOLEARN_GP_OUT=\"\$OUT/grow_policy_ab\" MOJOLEARN_GP_DEFINES=\"${GP_DEFINES:-MOJOLEARN_2044_FAST_NO_RIDX_ONLY MOJOLEARN_2045_FAST_NO_QUANT_HIST}\" MOJOLEARN_GP_LANES=\"${GP_LANES:-gbdt-lossguide gbdt-depthwise}\" MOJOLEARN_GP_ROWS=${GP_ROWS:-1000000} sh tools/grow_policy_ab.sh"; WRAP=0 ;;
      # OUR TREES vs THE VENDOR LIBRARY, BOTH TIERS, ON NVIDIA. The FAST
      # process interleaves ours with the opponents in one thermal window;
      # the IDENTICAL process runs ours only, because the opponents do not
      # move with our build define. Nothing here compares BITS against a
      # vendor: our identity claim is across OUR OWN backends, and against
      # theirs the only comparison is speed and held-out accuracy.
      vendor-trees)
        CMD="env GBM_BENCH_DATA=$VOL_MOUNT/gbm-bench MOJOLEARN_VT_OUT=\"\$OUT/vendor_trees\" MOJOLEARN_VT_LANE=${VT_LANE:-gbdt-symmetric} MOJOLEARN_VT_ROWS=${VT_ROWS:-1000000} sh tools/vendor_trees_leg.sh"; WRAP=0 ;;
      # FILL THE VOLUME AND MEASURE NOTHING. `load_higgs` says the download
      # is a separate, explicitly named step the orchestrator budgets a lease
      # around; this IS that step. It fetches and decodes onto the persistent
      # volume so every later leg starts warm. Run once per volume.
      higgs-seed)
        CMD="env GBM_BENCH_DATA=$VOL_MOUNT/gbm-bench pixi run -e gbmbench python tools/speed_gbdt_arm.py --download higgs"; WRAP=0 ;;
      # DEVIATION 2043's ACCURACY HALF. The speed half is already measured
      # (0.936 under FAST with the fused one-byte route); what was never
      # measured is whether held-out quality moves with it, and that -- not
      # the hashes -- is the gate a FAST default turns on.
      #
      # The harness backgrounds its own 2.6 GB higgs fetch beside the three
      # binding builds because the lease is an hour and serially it does not
      # fit. It leaves the checkout holding the BASE binary, so a diagnostic
      # arm is never left installed under the shipped name.
      gbdt-accuracy-ab)
        CMD="env GBM_BENCH_DATA=$VOL_MOUNT/gbm-bench MOJOLEARN_GACC_SKIP_FETCH=${GACC_SKIP_FETCH:-0} MOJOLEARN_GACC_OUT=\"\$OUT/gbdt_accuracy_ab\" MOJOLEARN_GACC_ROUNDS=${GACC_ROUNDS:-3} MOJOLEARN_GACC_ROWS=${GACC_ROWS:-1000000} sh tools/gbdt_accuracy_ab.sh"; WRAP=0 ;;
      # WHAT THIS BOX IS MISSING, IN ONE PASS. Four leases on 2026-09-03 died
      # one import at a time -- no numpy, then no pandas, then an environment
      # that had to exist before the job importing from it. Each run only ever
      # reported the FIRST thing wrong. The vendor legs have a much larger
      # install surface (cuml, cuvs, catboost, lightgbm, torch, mamba-ssm), so
      # this probes every interpreter for every opponent and reports them all.
      # A missing OPPONENT is the answer, not a crash; only a missing REQUIRED
      # module fails it.
      vendor-preflight)
        CMD="env MOJOLEARN_PREFLIGHT_OUT=\"\$OUT/vendor_preflight\" sh tools/vendor_preflight.sh"; WRAP=0 ;;
      *) log "unknown extra check '$chk' -- skipped"; continue ;;
    esac
    NOW=$(date +%s)
    LEFT=$(( LEG_START + DEADMAN_SECONDS - NOW - FETCH_RESERVE ))
    [ "$LEFT" -lt 60 ] && { log "extra check $chk SKIPPED (${LEFT}s left)"; continue; }
    log "extra check: $chk (bound ${LEFT}s)"
    PREFIX="bash tools/with_identical_mode.sh"
    [ "$WRAP" = 0 ] && PREFIX=""
    $SSH "export PATH=/root/.pixi/bin:\$PATH; cd /root/mojolearn && OUT=\$(ls -td bench/results/e1/*/ 2>/dev/null | head -1); [ -z \"\$OUT\" ] && { OUT=bench/results/e1/\$(date +%Y-%m-%d_%H%M%S)-\$(hostname); mkdir -p \"\$OUT\"; }; mkdir -p \"\$OUT/lanes\" && MOJOLEARN_IDENTITY_TRACE=\"\$OUT/lanes/$chk.identical.card\" VENDOR_PRICE_OUT=\"\$OUT/lanes/vendor_price.json\" MOJOLEARN_GEMM_PRICE_OUT=\"\$OUT/lanes/gemm_price_medians.txt\" timeout -k 30 $LEFT $PREFIX $CMD > \"\$OUT/lanes/$chk.identical.check.log\" 2>&1; echo \"EXTRA-$chk-EXIT=\$?\"; tail -8 \"\$OUT/lanes/$chk.identical.check.log\""
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
