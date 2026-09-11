#!/usr/bin/env bash
# ONE GUARDED DIGITALOCEAN LEG THAT BUILDS THE 0.6.1 HIP gfx942 SET:
# create an MI325X -> prepare the host -> tools/release061_remote_build.sh
# hip gfx942 -> fetch the evidence -> DESTROY (absence verified by GET 404).
#
#   bash tools/do_release061_leg.sh <frozen-40-hex-commit> <token_file>          DRY RUN, rents nothing
#   bash tools/do_release061_leg.sh <frozen-40-hex-commit> <token_file> --rent   spends money
#
# DEVIATION 2268. The 0.6.1 wheel needs three per-architecture build proofs
# (docs/RELEASE_0_6_1_EXECUTION_PLAN.md). The two CUDA proofs come from
# tools/gemm_remote_leg.sh on RunPod (MOJOLEARN_NVIDIA_CAMPAIGN=7); the only
# AMD silicon this account can rent is a DigitalOcean MI325X, so this file is
# the third leg. It produces the SAME evidence tree the NVIDIA legs keep under
# OUT/remote/release-build, here under
#   bench/results/releases/2026-09-08-linux-0.7.0/hip-gfx942/release-build/
# so the packer reads it as
#   --set .../release-build/build/sets/hip
#   --build-proof .../release-build/build/build-provenance.json
# A successful leg is BUILT_NOT_INSTALLED, never wheel admission.
#
# THE GUARDS ARE tools/do_speed_leg.sh's AND tools/e2_remote_leg.sh's, COPIED
# RATHER THAN FACTORED OUT, for their reason: DigitalOcean bills until a droplet
# is DESTROYED, so the destroy is an EXIT trap, a detached local dead-man keyed
# by TAG AND NAME (an unparseable create response cannot orphan a box), and a
# second dead-man ON THE DROPLET that survives this desk sleeping. Only a GET
# 404 after the DELETE counts as destroyed. Image, size, region and key
# fingerprint are the ones every prior AMD leg used.
set -uo pipefail

COMMIT="${1:?frozen 40-hex commit}"
TOKFILE="${2:?token file (~/.mojolearn_do_token)}"
RENT=0
case "$#:${3:-}" in 2:|3:--rent) [ "${3:-}" = "--rent" ] && RENT=1 ;; *) echo "usage: $0 <commit> <token_file> [--rent]"; exit 2 ;; esac
DEADMAN_SECONDS="${DEADMAN_SECONDS:-3600}"
# DEVIATION 2294: the same guarded rental, doing the OTHER half of the release.
# tools/linux_surface_qualification.sh runs ON the device, and until now nothing
# carried a finished wheel to a device and brought the verdict home -- the
# runbook says so itself, "authored but unexecuted". Everything this leg already
# owns is what that job needs: a dead-man before the create, a second dead-man
# ON the droplet, an uplink probe before the bill starts, a detached-and-polled
# work phase, and a destroy verified by GET 404. So it is a MODE here, not a
# second file that would drift from these guards.
#   MOJOLEARN_LEG_MODE=qualify
#   MOJOLEARN_QUALIFY_WHEEL=/abs/path/mojolearn-X.Y.Z-...manylinux....whl
#   MOJOLEARN_QUALIFY_PROOFS=/abs/dir holding cuda-sm_89.json, cuda-sm_90a.json,
#                            hip-gfx942.json  (the packer's three, unchanged)
LEG_MODE="${MOJOLEARN_LEG_MODE:-build}"
case "$LEG_MODE" in build|qualify) ;; *) echo "MOJOLEARN_LEG_MODE must be build or qualify" >&2; exit 2 ;; esac
if [ "$LEG_MODE" = qualify ]; then
  QUAL_WHEEL="${MOJOLEARN_QUALIFY_WHEEL:?qualify mode needs the repaired wheel}"
  QUAL_PROOFS="${MOJOLEARN_QUALIFY_PROOFS:?qualify mode needs the proof directory}"
  [ -f "$QUAL_WHEEL" ] || { echo "no wheel at $QUAL_WHEEL" >&2; exit 2; }
  [ -d "$QUAL_PROOFS" ] || { echo "no proof directory at $QUAL_PROOFS" >&2; exit 2; }
  case "$QUAL_WHEEL" in *manylinux*) ;; *) echo "REFUSING: qualify the REPAIRED wheel; $QUAL_WHEEL is not manylinux-tagged" >&2; exit 2 ;; esac
fi
FETCH_RESERVE="${FETCH_RESERVE:-900}"     # three tiers + MAX runtime closure come home in this
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API=https://api.digitalocean.com/v2
SSH_KEY_FP="df:f7:6b:0c:56:da:48:a5:6f:6d:ae:44:af:de:f3:0b"
SSH_KEY_FILE="$HOME/.ssh/id_ed25519"
# DEVIATION 2294: the CUDA columns need the same guarded rental, so the box is
# a knob. Defaults are the MI325X this leg was written for; MOJOLEARN_LEG_GPU=h100
# switches to DigitalOcean's H100 for the Hopper column. The vendor and
# architecture strings travel with it, because a leg that rented an H100 and
# then told the driver "hip gfx942" would be qualifying a lie.
case "${MOJOLEARN_LEG_GPU:-mi325x}" in
  mi325x) NAME=mojolearn-rel061-amd; REGION=tor1; SIZE=gpu-mi325x1-256gb; IMAGE=188571990
          LEG_VENDOR=hip;  LEG_ARCH=gfx942; LEG_GUARD=tools/amd_serial_guard.py; GPU_PROBE='rocm-smi --showproductname 2>/dev/null | grep -i "card series\|name" | head -2' ;;
  h100)   NAME=mojolearn-rel061-nv;  REGION=nyc2; SIZE=gpu-h100x1-80gb;   IMAGE=236925144
          LEG_VENDOR=cuda; LEG_ARCH=sm_90a; LEG_GUARD=tools/nvidia_serial_guard.py; GPU_PROBE='nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -2' ;;
  *) echo "MOJOLEARN_LEG_GPU must be mi325x or h100" >&2; exit 2 ;;
esac
TAG=rel061
REMOTE_PY=/usr/bin/python3                 # the image's stdlib 3.12 (tools/do_byte_lm_setup.sh)
REMOTE_OUT=/root/rel061-build; REMOTE_LOG=/root/rel061-build.log
RELEASE_ROOT="${MOJOLEARN_RELEASE_RESULTS_ROOT:-$REPO/bench/results/releases/2026-09-08-linux-0.7.0}"
OUT="$RELEASE_ROOT/$LEG_VENDOR-$LEG_ARCH"
# DEVIATION 2294: a qualification is not a build proof and must not land where
# one lives; the packer reads that path and would find a directory of the
# wrong shape. Its own destination, stamped, so two qualification runs of the
# same wheel do not overwrite each other either.
[ "$LEG_MODE" = qualify ] && OUT="$RELEASE_ROOT/qualification/$LEG_VENDOR-$LEG_ARCH-$(date -u +%Y%m%dT%H%M%SZ)"
STATE="$OUT/leg.txt"
TMPD="$(mktemp -d "${TMPDIR:-/tmp}/rel061.XXXXXX")"

log() { echo "[$(date +%T) amd/rel061] $*"; }
die() { log "REFUSING: $*"; exit "${2:-2}"; }

# WHICH SIDE OF THE WIRE DIED (DEVIATION 2292). HTTP 000 is curl saying it
# could not connect, and that is as true of a dead uplink here as of a dead
# API there. On 2026-09-08 this leg lost the Mac's network one minute after
# the droplet came up ("Can't assign requested address"), could not start the
# build, and then logged HTTP 000 against every DELETE and GET for two and a
# half hours. Read cold, that log accuses DigitalOcean. It was this desk.
# Neutral hosts, none of them a vendor API, settle which end is silent.
uplink_down() {
  local h
  for h in https://pypi.org/ https://github.com/ https://www.google.com/; do
    curl -s -o /dev/null --max-time 8 "$h" 2>/dev/null && return 1
  done
  return 0
}

# Three rounds before the bill starts. A flapping link fails one of them, and
# a leg that never creates a droplet cannot strand one.
uplink_stable() {
  local r=1
  while [ "$r" -le 3 ]; do
    if uplink_down; then log "uplink probe $r/3: NO neutral host answered"; return 1; fi
    [ "$r" -lt 3 ] && sleep 7
    r=$((r + 1))
  done
  return 0
}
sha256_of() { { shasum -a 256 "$1" 2>/dev/null || sha256sum "$1"; } | cut -d' ' -f1; }
mode_of() { stat -f %Lp "$1" 2>/dev/null || stat -c %a "$1"; }

# ------------------------------------------------------------ preflight (both modes)
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "commit must be the full 40-hex SHA, got '$COMMIT'"
git -C "$REPO" cat-file -e "$COMMIT^{commit}" 2>/dev/null || die "commit $COMMIT is not in $REPO"
[ -f "$TOKFILE" ] || die "token file $TOKFILE missing"
[ "$(mode_of "$TOKFILE")" = 600 ] || die "token file $TOKFILE is mode $(mode_of "$TOKFILE"), want 600"
TOK="$(tr -d '[:space:]' < "$TOKFILE")"; [ -n "$TOK" ] || die "token file is empty"
[ -f "$SSH_KEY_FILE" ] || die "ssh key $SSH_KEY_FILE missing"
[ ! -e "$OUT/release-build" ] || die "$OUT/release-build already exists; nothing local is ever overwritten"
api() { curl -s --max-time 20 -H "Authorization: Bearer $TOK" "$@"; }

# SHIP THE COMMIT, NOT THE WORKING TREE: the whole tree at $COMMIT minus the
# result, corpus and oracle data, which no native-inventory file lives under.
log "archiving $COMMIT"
git -C "$REPO" archive --format=tar "$COMMIT" -- . ':!bench/results' ':!mamba/corpus' ':!bench/oracle*' \
  | gzip > "$TMPD/src.tgz" || die "git archive failed"
ARCHIVE_BYTES=$(wc -c < "$TMPD/src.tgz" | tr -d ' ')
[ "$ARCHIVE_BYTES" -lt $((15 * 1024 * 1024)) ] || die "archive is $ARCHIVE_BYTES bytes gzipped, cap 15 MiB"
ARCHIVE_SHA=$(sha256_of "$TMPD/src.tgz")
mkdir "$TMPD/archive" && tar -xzf "$TMPD/src.tgz" -C "$TMPD/archive" || die "archive does not unpack"
# The packer compares every proof's inventory against THIS checkout, so the
# commit and the working tree must agree on every native-inventory file now.
python3 - "$TMPD/archive" "$REPO" > "$TMPD/source_inventory_local.json" <<'PY' || die "native inventory differs between $COMMIT and the working tree; freeze first"
import json, pathlib, sys
archive, local = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(archive / 'tools'))
from check_linux_release_qualification import native_inventory
a, l = native_inventory(archive), native_inventory(local)
if a != l:
    names = sorted({p for p, _ in set(map(tuple, a)) ^ set(map(tuple, l))})
    raise SystemExit('differs: ' + ' '.join(names[:8]))
print(json.dumps(a, separators=(',', ':')))
PY
log "archive $ARCHIVE_BYTES bytes, sha256 $ARCHIVE_SHA, native inventory matches working tree"

# A GET that costs nothing: proves the token and shows any droplet already
# wearing our name or any GPU droplet at all (one lease at a time, no orphans).
LISTING="$(api -w '\nHTTP %{http_code}' "$API/droplets?per_page=200")"
[ "$(printf '%s' "$LISTING" | tail -1)" = "HTTP 200" ] || die "API not reachable: $(printf '%s' "$LISTING" | tail -1)"
GPU_LIVE="$(printf '%s' "$LISTING" | sed '$d' | python3 -c "import json,sys
d=json.load(sys.stdin); print(' '.join('%s:%s:%s'%(x['id'],x['name'],x.get('size_slug','')) for x in d.get('droplets',[]) if x['name']=='$NAME' or str(x.get('size_slug','')).startswith('gpu-')))")"
[ -z "$GPU_LIVE" ] || die "GPU droplet(s) already live, destroy or adopt them first: $GPU_LIVE"
log "API reachable, no GPU droplet live, no droplet named $NAME"

# THIS DESK'S UPLINK IS PART OF THE RENTAL (DEVIATION 2292). The build start,
# the fetch and the destroy that stops the bill all run over it.
uplink_stable || die "this machine could not reach ANY neutral host. Nothing
  was created, so nothing is billing. Fix this desk's network, then re-run." 2
log "uplink up on all three probes"

if [ $RENT = 0 ]; then
if [ "$LEG_MODE" = qualify ]; then
  WOULD_RUN="  upload   $QUAL_WHEEL
           + $(ls "$QUAL_PROOFS"/*.json 2>/dev/null | wc -l | tr -d ' ') build proofs -> /root/proofs/
  qualify  bash tools/linux_surface_qualification.sh qualify-release-linux3 \\
             /root/$(basename "$QUAL_WHEEL") <sha256> $LEG_VENDOR \$REMOTE_OUT /root/proofs $LEG_ARCH"
else
  WOULD_RUN="  build    MOJOLEARN_COMMIT=$COMMIT MOJOLEARN_PYTHON=$REMOTE_PY MOJOLEARN_RELEASE_BUILD_SECONDS=<=2400
           bash tools/release061_remote_build.sh $LEG_VENDOR $LEG_ARCH \$REMOTE_OUT > \$REMOTE_LOG"
fi
  cat <<EOF
DRY RUN -- nothing rented. With --rent this leg would:
  create   $NAME  image=$IMAGE size=$SIZE region=$REGION key=$SSH_KEY_FP tag=$TAG
  dead-man local ${DEADMAN_SECONDS}s (tag+name) and on-droplet ${DEADMAN_SECONDS}s (id)
  upload   $TMPD/src.tgz ($ARCHIVE_BYTES bytes) -> /root/mojolearn + commit.txt=$COMMIT
  prepare  apt-get patchelf/binutils if absent; pixi; pixi install --locked --environment default (guarded)
  mode     $LEG_MODE
$WOULD_RUN
  fetch    $REMOTE_OUT -> $OUT/release-build/ ; logs and leg.txt beside it
  destroy  DELETE then GET until 404; never deletes anything local
EOF
  exit 0
fi

# ------------------------------------------------------------------ --rent
mkdir -p "$OUT"
DROPLET_ID=""; DEADMAN_PID=""; DESTROY_CONFIRMED=0; IP=""
SSH_OPTS="-i $SSH_KEY_FILE -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o ServerAliveInterval=30"

destroy() {
  if [ -z "$DROPLET_ID" ]; then
    for id in $(api "$API/droplets?tag_name=$TAG&per_page=50" | python3 -c "import json,sys
d=json.load(sys.stdin); print(' '.join(str(x['id']) for x in d.get('droplets',[]) if x['name']=='$NAME'))" 2>/dev/null); do
      code=$(api -o /dev/null -w '%{http_code}' -X DELETE "$API/droplets/$id")
      log "DELETE by-name droplet $id -> HTTP $code"
    done
    return 0
  fi
  UPLINK_FAULT=0
  for i in 1 2 3 4 5 6; do
    code=$(api -o /dev/null -w '%{http_code}' -X DELETE "$API/droplets/$DROPLET_ID")
    log "DELETE droplet $DROPLET_ID -> HTTP $code"
    case "$code" in 204|404) break ;; esac
    # DEVIATION 2292: say which end is silent, once, rather than leaving a
    # column of HTTP 000 that reads as a vendor outage.
    if [ "$code" = 000 ] && [ "$UPLINK_FAULT" = 0 ] && uplink_down; then
      UPLINK_FAULT=1
      log "THIS MACHINE HAS NO UPLINK: no neutral host answered either, so the"
      log "  HTTP 000 above is this desk, not DigitalOcean. The droplet's own"
      log "  dead-man is the layer that can still end it. Still retrying."
      echo "uplink_fault=1" >> "$STATE"
    fi
    sleep 10
  done
  # DELETE 204 acknowledges an asynchronous request. Only GET 404 proves absence.
  for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
    code=$(api -o /dev/null -w '%{http_code}' "$API/droplets/$DROPLET_ID")
    log "post-destroy GET droplet $DROPLET_ID -> HTTP $code"
    if [ "$code" = 404 ]; then
      DESTROY_CONFIRMED=1
      echo "destroyed=$(date -u +%FT%TZ) verified_http_404" >> "$STATE"
      break
    fi
    sleep 5
  done
  [ "$DESTROY_CONFIRMED" = 1 ] || { log "destruction UNCONFIRMED; dead-men stay armed"; echo "destroyed=UNCONFIRMED" >> "$STATE"; }
}
teardown() {
  rc=$?
  log "teardown (rc=$rc)"
  destroy
  if [ -n "$DEADMAN_PID" ] && [ "$DESTROY_CONFIRMED" = 1 ]; then
    pkill -P "$DEADMAN_PID" 2>/dev/null
    kill "$DEADMAN_PID" 2>/dev/null && log "dead-man timer cancelled"
  fi
  echo "finished=$(date -u +%FT%TZ) rc=$rc" >> "$STATE"
  exit $rc
}
trap teardown EXIT

LEG_START=$(date +%s)
cp "$TMPD/source_inventory_local.json" "$OUT/source_inventory_local.json"
{
  echo "vendor=amd"; echo "arch=hip/gfx942"; echo "image=$IMAGE"; echo "size=$SIZE"; echo "region=$REGION"
  echo "commit=$COMMIT"; echo "archive_sha256=$ARCHIVE_SHA"; echo "archive_bytes=$ARCHIVE_BYTES"
  echo "deadman_seconds=$DEADMAN_SECONDS"; echo "fetch_reserve=$FETCH_RESERVE"; echo "started=$(date -u +%FT%TZ)"
} > "$STATE"

nohup bash -c "sleep $DEADMAN_SECONDS; for id in \$(curl -s -H 'Authorization: Bearer $TOK' '$API/droplets?tag_name=$TAG&per_page=50' | python3 -c \"import json,sys; d=json.load(sys.stdin); print(' '.join(str(x['id']) for x in d.get('droplets',[]) if x['name']=='$NAME'))\"); do curl -s -o /dev/null -w \"deadman DELETE \$id -> %{http_code}\\n\" -X DELETE -H 'Authorization: Bearer $TOK' \"$API/droplets/\$id\" >> '$STATE'; done" \
  >/dev/null 2>&1 < /dev/null &
DEADMAN_PID=$!
disown "$DEADMAN_PID" 2>/dev/null || true
log "dead-man pid $DEADMAN_PID ($DEADMAN_SECONDS s, tag $TAG + name $NAME)"

log "creating $NAME ($SIZE, $REGION, image $IMAGE)"
CREATE_BODY="$(api -X POST -H "Content-Type: application/json" \
  -d "{\"name\":\"$NAME\",\"region\":\"$REGION\",\"size\":\"$SIZE\",\"image\":$IMAGE,\"ssh_keys\":[\"$SSH_KEY_FP\"],\"tags\":[\"$TAG\"]}" \
  "$API/droplets")"
DROPLET_ID=$(printf '%s' "$CREATE_BODY" | python3 -c "import json,sys
try:
    d=json.load(sys.stdin); print(d['droplet']['id'] if 'droplet' in d else '')
except Exception: print('')")
if [ -z "$DROPLET_ID" ]; then
  log "create returned no id; body: $(printf '%s' "$CREATE_BODY" | head -c 300)"
  sleep 5
  DROPLET_ID=$(api "$API/droplets?tag_name=$TAG&per_page=50" | python3 -c "import json,sys
d=json.load(sys.stdin); ids=[x['id'] for x in d.get('droplets',[]) if x['name']=='$NAME']
print(ids[0] if ids else '')")
  [ -n "$DROPLET_ID" ] && log "ADOPTED droplet $DROPLET_ID found by name" || { log "create FAILED"; exit 3; }
fi
log "droplet id $DROPLET_ID"
{ echo "droplet=$DROPLET_ID"; echo "created=$(date -u +%FT%TZ)"; } >> "$STATE"

for i in $(seq 1 90); do
  read -r status IP < <(api "$API/droplets/$DROPLET_ID" | python3 -c "
import json,sys
d=json.load(sys.stdin)['droplet']
ips=[n['ip_address'] for n in d['networks'].get('v4',[]) if n['type']=='public']
print(d['status'], ips[0] if ips else '')")
  [ "$status" = "active" ] && [ -n "$IP" ] && break
  sleep 10
done
[ -n "$IP" ] || { log "never became active"; exit 4; }
log "active at $IP"; echo "ip=$IP" >> "$STATE"
SSH="ssh $SSH_OPTS root@$IP"
ok=0
for i in $(seq 1 30); do
  $SSH 'echo SSH-OK' 2>/dev/null | grep -q SSH-OK && { ok=1; break; }
  sleep 15
done
[ $ok = 1 ] || { log "ssh never came up"; exit 5; }

# Second dead-man ON THE DROPLET (tools/e2_remote_leg.sh): keyed by id, survives this desk.
$SSH "umask 077; printf '%s\n' '#!/bin/sh' 'sleep $DEADMAN_SECONDS' \
  \"curl -s -o /dev/null -w '%{http_code}\\n' -X DELETE -H 'Authorization: Bearer $TOK' '$API/droplets/$DROPLET_ID' >> /tmp/selfkill.out\" \
  > /tmp/mojolearn-selfkill.sh; chmod 700 /tmp/mojolearn-selfkill.sh
nohup /tmp/mojolearn-selfkill.sh >/tmp/selfkill.log 2>&1 & echo \$! > /tmp/selfkill.pid; sleep 1
kill -0 \$(cat /tmp/selfkill.pid) 2>/dev/null && echo ON_DROPLET_DEADMAN_ARMED || echo ON_DROPLET_DEADMAN_FAILED" \
  2>&1 | tail -1 | tee -a "$STATE" | sed 's/^/[amd] /'
$SSH "$GPU_PROBE" | tee "$OUT/device.txt" | sed 's/^/[gpu] /'

log "shipping $ARCHIVE_BYTES bytes"
scp -q $SSH_OPTS "$TMPD/src.tgz" "root@$IP:/root/src.tgz" || { log "scp failed"; exit 6; }
REMOTE_SHA=$($SSH 'sha256sum /root/src.tgz' | cut -d' ' -f1)
[ "$REMOTE_SHA" = "$ARCHIVE_SHA" ] || { log "archive sha mismatch after transfer ($REMOTE_SHA)"; exit 6; }
$SSH "test ! -e /root/mojolearn && mkdir /root/mojolearn && tar -xzf /root/src.tgz -C /root/mojolearn \
  && printf '%s\n' '$COMMIT' > /root/mojolearn/commit.txt && test -x $REMOTE_PY" || { log "remote unpack failed"; exit 6; }
log "shipped $COMMIT"

if [ "$LEG_MODE" = qualify ]; then
  # THE BYTES THAT GET QUALIFIED ARE THE BYTES THAT GET PUBLISHED. The sha256
  # is compared on both ends, because a wheel that arrived corrupted would
  # qualify a file nobody will ever install.
  QUAL_SHA=$(sha256_of "$QUAL_WHEEL")
  QUAL_BASE=$(basename "$QUAL_WHEEL")
  log "shipping the repaired wheel ($(wc -c < "$QUAL_WHEEL" | tr -d ' ') bytes, sha256 $QUAL_SHA)"
  scp -q $SSH_OPTS "$QUAL_WHEEL" "root@$IP:/root/$QUAL_BASE" || { log "wheel scp failed"; exit 6; }
  RSHA=$($SSH "sha256sum /root/$QUAL_BASE" | cut -d' ' -f1)
  [ "$RSHA" = "$QUAL_SHA" ] || { log "wheel sha mismatch after transfer ($RSHA)"; exit 6; }
  $SSH 'mkdir -p /root/proofs'
  scp -q $SSH_OPTS "$QUAL_PROOFS"/*.json "root@$IP:/root/proofs/" || { log "proof scp failed"; exit 6; }
  # THE CORPORA, WHICH THE BUILD ARCHIVE DELIBERATELY EXCLUDES. `git archive`
  # drops mamba/corpus because a build never reads it and it is 63 MB; the
  # INSTALLED qualification does read it, and refused this leg's first run with
  # "Missing installed Mamba corpus: base_b2_l4_d8". Only the three cases
  # CORPUS_CASES names are shipped, 4.6 MB, not the whole directory.
  CORPUS_CASES=$(MOJOLEARN_QUIET=1 python3 -c "import sys;sys.path.insert(0,'$REPO/tools');from verify_linux_surface_qualification import CORPUS_CASES;print(' '.join(CORPUS_CASES))")
  [ -n "$CORPUS_CASES" ] || { log "could not read CORPUS_CASES"; exit 6; }
  ( cd "$REPO" && tar czf "$TMPD/corpus.tgz" $(for c in $CORPUS_CASES; do echo "mamba/corpus/$c"; done) ) \
    || { log "corpus tar failed"; exit 6; }
  CORPUS_SHA=$(sha256_of "$TMPD/corpus.tgz")
  log "shipping the qualification corpora ($(wc -c < "$TMPD/corpus.tgz" | tr -d ' ') bytes, $(echo $CORPUS_CASES | wc -w | tr -d ' ') cases)"
  scp -q $SSH_OPTS "$TMPD/corpus.tgz" "root@$IP:/root/corpus.tgz" || { log "corpus scp failed"; exit 6; }
  RCS=$($SSH 'sha256sum /root/corpus.tgz' | cut -d' ' -f1)
  [ "$RCS" = "$CORPUS_SHA" ] || { log "corpus sha mismatch after transfer ($RCS)"; exit 6; }
  $SSH 'tar -xzf /root/corpus.tgz -C /root/mojolearn' || { log "corpus unpack failed"; exit 6; }
  for c in $CORPUS_CASES; do
    $SSH "test -f /root/mojolearn/mamba/corpus/$c/x.f32" || { log "corpus case $c did not land"; exit 6; }
  done
  echo "qualify_corpus_cases=$CORPUS_CASES" >> "$STATE"
  log "shipped wheel, $(ls "$QUAL_PROOFS"/*.json | wc -l | tr -d ' ') proofs and the corpora"
  echo "qualify_wheel=$QUAL_BASE" >> "$STATE"
  echo "qualify_wheel_sha256=$QUAL_SHA" >> "$STATE"
fi

# HOST PREPARATION. release061_remote_build.sh refuses a missing patchelf or an
# unprepared pixi default environment on purpose, so both are prepared HERE,
# as named steps with their own exit codes. The locked install runs under the
# AMD serial guard exactly as gemm_remote_leg.sh's profile 7 does.
PREP_SECONDS=$(( LEG_START + DEADMAN_SECONDS - $(date +%s) - FETCH_RESERVE - 600 ))
[ "$PREP_SECONDS" -gt 900 ] && PREP_SECONDS=900
[ "$PREP_SECONDS" -ge 120 ] || { log "only ${PREP_SECONDS}s for host prep; skipping"; exit 7; }
$SSH "set -u; export DEBIAN_FRONTEND=noninteractive
need=''; command -v patchelf >/dev/null || need=\"\$need patchelf\"
{ command -v objdump && command -v strings; } >/dev/null || need=\"\$need binutils\"
# DEVIATION 2294: the INSTALLED qualification builds a venv and pip-installs
# the wheel into it. This image's python has no ensurepip, so python3 -m venv
# failed with 'ensurepip is not available' and the driver never ran a single
# job. A build never needs this, which is why nothing had noticed.
$REMOTE_PY -c 'import ensurepip' 2>/dev/null || need=\"\$need python3-venv python3-pip\"
if [ -n \"\$need\" ]; then
  timeout -k 10 180 apt-get -qq -o Acquire::Retries=1 -o Acquire::http::Timeout=30 update > /root/apt.log 2>&1
  timeout -k 10 300 apt-get -qq -o DPkg::Lock::Timeout=120 -o Acquire::Retries=1 install -y --no-install-recommends \$need >> /root/apt.log 2>&1; echo APT_EXIT=\$? need=\$need
fi
export PATH=/root/.pixi/bin:\$PATH
command -v pixi >/dev/null || timeout -k 10 120 sh -c 'curl -fsSL --max-time 30 https://pixi.sh/install.sh | sh' > /root/pixi_bootstrap.log 2>&1
cd /root/mojolearn && $REMOTE_PY $LEG_GUARD --seconds $PREP_SECONDS --rss-gib 12 -- \
  pixi install --locked --environment default > /root/pixi_install.log 2>&1; echo PIXI_INSTALL_EXIT=\$?
# Match the NVIDIA release builder: a private pinned wheel provides patchelf
# when the image apt repositories fail. Keep it outside the locked Pixi env.
if ! command -v patchelf >/dev/null; then
  tail -40 /root/apt.log
  .pixi/envs/default/bin/python -m venv /root/release-tools &&
  timeout -k 10 120 /root/release-tools/bin/python -m pip install --disable-pip-version-check --only-binary=:all: --retries 1 --timeout 20 patchelf==0.17.2.4
fi
export PATH=/root/release-tools/bin:\$PATH
command -v patchelf >/dev/null && patchelf --version
for t in taskset objdump patchelf pixi; do command -v \$t >/dev/null || echo MISSING_\$t; done
test -x .pixi/envs/default/bin/mojo && test -x .pixi/envs/default/bin/python && echo PIXI_ENV_OK || echo PIXI_ENV_MISSING" \
  2>&1 | tee "$OUT/prep-console.log" | sed 's/^/[amd prep] /'
if ! grep -q '^PIXI_INSTALL_EXIT=0$' "$OUT/prep-console.log" || grep -q 'MISSING_\|PIXI_ENV_MISSING' "$OUT/prep-console.log"; then
  echo "build_exit=NOT_STARTED_HOST_PREP_FAILED" >> "$STATE"; log "host prep failed"; exit 8
fi

# THE BUILD, DETACHED AND POLLED, so a dropped ssh cannot kill a 30-minute compile.
WORK_SECONDS=$(( LEG_START + DEADMAN_SECONDS - $(date +%s) - FETCH_RESERVE ))
[ "$WORK_SECONDS" -gt 2400 ] && WORK_SECONDS=2400
[ "$WORK_SECONDS" -ge 300 ] || { echo "build_exit=NOT_STARTED_${WORK_SECONDS}s_LEFT" >> "$STATE"; log "only ${WORK_SECONDS}s left; skipping the build"; exit 7; }
echo "work_seconds=$WORK_SECONDS" >> "$STATE"; log "build bound ${WORK_SECONDS}s"
if [ "$LEG_MODE" = qualify ]; then
  # 25 installed jobs on THIS device, from the wheel's own bytes. The driver
  # refuses an architecture override and records the device it actually found,
  # so it cannot be talked into agreeing with us.
  $SSH "cd /root/mojolearn && nohup bash -c 'export PATH=/root/release-tools/bin:/root/.pixi/bin:\$PATH; \
    MOJOLEARN_EXPECT_VENDOR=$LEG_VENDOR \
    timeout -k 20 $((WORK_SECONDS + 40)) bash tools/linux_surface_qualification.sh qualify-release-linux3 \
      /root/$QUAL_BASE $QUAL_SHA $LEG_VENDOR $REMOTE_OUT /root/proofs $LEG_ARCH > $REMOTE_LOG 2>&1; \
    echo \$? > /root/rel061.exit' > /dev/null 2>&1 < /dev/null &" || { log "could not start the qualification"; exit 9; }
else
$SSH "cd /root/mojolearn && nohup bash -c 'export PATH=/root/release-tools/bin:/root/.pixi/bin:\$PATH; \
  MOJOLEARN_COMMIT=$COMMIT MOJOLEARN_PYTHON=$REMOTE_PY MOJOLEARN_RELEASE_BUILD_SECONDS=$WORK_SECONDS \
  MOJOLEARN_BUILD_JOBS=${MOJOLEARN_BUILD_JOBS:-4} \
  timeout -k 20 $((WORK_SECONDS + 40)) bash tools/release061_remote_build.sh $LEG_VENDOR $LEG_ARCH $REMOTE_OUT > $REMOTE_LOG 2>&1; \
  echo \$? > /root/rel061.exit' > /dev/null 2>&1 < /dev/null &" || { log "could not start the build"; exit 9; }
fi
BUILD_EXIT=""
while [ $(date +%s) -lt $(( LEG_START + DEADMAN_SECONDS - FETCH_RESERVE + 60 )) ]; do
  BUILD_EXIT=$($SSH 'cat /root/rel061.exit 2>/dev/null' 2>/dev/null | tr -d '[:space:]')
  [ -n "$BUILD_EXIT" ] && break
  sleep 30
done
echo "build_exit=${BUILD_EXIT:-NO_EXIT_BEFORE_FETCH_RESERVE}" >> "$STATE"
log "build exit ${BUILD_EXIT:-none}; $($SSH "tail -3 $REMOTE_LOG" 2>/dev/null | tr '\n' '|')"

log "fetch evidence"
rsync -az -e "ssh $SSH_OPTS" "root@$IP:$REMOTE_OUT/" "$OUT/release-build/" && log "fetched release-build/" || log "FETCH FAILED (release-build/)"
for f in rel061-build.log pixi_install.log pixi_bootstrap.log apt.log rel061.exit; do
  rsync -az -e "ssh $SSH_OPTS" "root@$IP:/root/$f" "$OUT/$f" 2>/dev/null || true
done
if [ "$LEG_MODE" = qualify ]; then
  # THE VERDICT IS THE DRIVER'S, NOT THIS SCRIPT'S. All this checks is that a
  # verdict came home, that it is about the wheel we shipped, and that the
  # device it names is the one we rented. check_linux_release_qualification.py
  # is what admits; it runs on the Mac, over all three architectures at once,
  # and this leg supplies exactly one of its three columns.
  python3 - "$OUT/release-build" "$QUAL_SHA" "$LEG_ARCH" <<'PYQ' 2>&1 | tee -a "$STATE"
import json, pathlib, sys
out, sha, want_arch = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
try:
    q = json.loads((out / 'qualification.json').read_text())
    audit = json.loads((out / 'wheel-audit.json').read_text())
    assert audit.get('sha256') == sha, 'audit is about a different wheel'
    arch = audit.get('runtime_architecture')
    assert arch == want_arch, 'runtime architecture is %r, not %r' % (arch, want_arch)
    # The driver emits mojolearn.linux.installed-surfaces.v1: a single status
    # plus installed_records, one per job. It has no 'jobs' or 'rows' key, and
    # reading for one printed RED over a PASSED run.
    records = q.get('installed_records') or {}
    status = str(q.get('status', '')).upper()
    print('qualify_arch=%s' % arch)
    print('qualify_vendor=%s' % q.get('vendor'))
    print('qualify_jobs=%d' % len(records))
    print('qualify_wheel_sha256=%s' % q.get('wheel_sha256'))
    assert q.get('wheel_sha256') == sha, 'qualification is about a different wheel'
    print('qualify_admission=%s' % ('GREEN' if status == 'PASSED' and records else 'RED'))
    if status != 'PASSED':
        print('  driver status=%s reason=%s' % (status, q.get('reason')))
except Exception as exc:
    print('qualify_admission=RED')
    print('qualify_error=%s' % exc)
PYQ
else
python3 - "$OUT/release-build" "$COMMIT" "$OUT/source_inventory_local.json" "$REPO/tools" <<'PY' 2>&1 | tee -a "$STATE"
import hashlib, json, pathlib, sys
out, commit = pathlib.Path(sys.argv[1]), sys.argv[2]
local = json.loads(pathlib.Path(sys.argv[3]).read_text())
try:
    assert (out / 'exit_code').read_text().strip() == '0', 'campaign exit_code != 0'
    proof = json.loads((out / 'build/build-provenance.json').read_text())
    pre = json.loads((out / 'preflight.json').read_text())
    assert proof.get('complete') is True and proof.get('build_exit') == 0 and proof.get('source_commit') == commit, 'proof incomplete'
    assert pre.get('vendor') == 'hip' and pre.get('device_architecture') == 'gfx942', 'physical witness is not hip/gfx942'
    assert pre['source_inventory'] == proof['source_inventory'] == local, 'local/preflight/proof inventories differ'
    # DEVIATION 2490: the count is the verifier's (23 with the byte LM), never a literal here.
    sys.path.insert(0, sys.argv[4])
    from verify_linux_surface_qualification import MODES, expected_bindings
    expected = sum(len(expected_bindings(mode, True)) for mode in MODES)
    assert len(proof['extensions']) == expected, 'expected %d extensions, proof has %d' % (expected, len(proof['extensions']))
    for name, digest in proof['extensions'].items():
        assert name.startswith('mojolearn/hip/gfx942/'), 'wrong architecture path ' + name
        assert hashlib.sha256((out / 'build/sets' / name[len('mojolearn/'):]).read_bytes()).hexdigest() == digest, 'fetched byte mismatch ' + name
    print('admission=BUILT_NOT_INSTALLED hip/gfx942 %d extensions, fetched bytes match proof' % expected)
except Exception as exc:
    print('admission=REFUSED ' + str(exc))
PY
fi
log "done -- $OUT (leg.txt has the verdict; the droplet is destroyed by the EXIT trap next)"
