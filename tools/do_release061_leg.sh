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
FETCH_RESERVE="${FETCH_RESERVE:-900}"     # three tiers + MAX runtime closure come home in this
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API=https://api.digitalocean.com/v2
SSH_KEY_FP="df:f7:6b:0c:56:da:48:a5:6f:6d:ae:44:af:de:f3:0b"
SSH_KEY_FILE="$HOME/.ssh/id_ed25519"
NAME=mojolearn-rel061-amd; TAG=rel061; REGION=tor1; SIZE=gpu-mi325x1-256gb; IMAGE=188571990
REMOTE_PY=/usr/bin/python3                 # the image's stdlib 3.12 (tools/do_byte_lm_setup.sh)
REMOTE_OUT=/root/rel061-build; REMOTE_LOG=/root/rel061-build.log
OUT="$REPO/bench/results/releases/2026-09-08-linux-0.7.0/hip-gfx942"
STATE="$OUT/leg.txt"
TMPD="$(mktemp -d "${TMPDIR:-/tmp}/rel061.XXXXXX")"

log() { echo "[$(date +%T) amd/rel061] $*"; }
die() { log "REFUSING: $*"; exit "${2:-2}"; }
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

if [ $RENT = 0 ]; then
  cat <<EOF
DRY RUN -- nothing rented. With --rent this leg would:
  create   $NAME  image=$IMAGE size=$SIZE region=$REGION key=$SSH_KEY_FP tag=$TAG
  dead-man local ${DEADMAN_SECONDS}s (tag+name) and on-droplet ${DEADMAN_SECONDS}s (id)
  upload   $TMPD/src.tgz ($ARCHIVE_BYTES bytes) -> /root/mojolearn + commit.txt=$COMMIT
  prepare  apt-get patchelf/binutils if absent; pixi; pixi install --locked --environment default (guarded)
  build    MOJOLEARN_COMMIT=$COMMIT MOJOLEARN_PYTHON=$REMOTE_PY MOJOLEARN_RELEASE_BUILD_SECONDS=<=2400
           bash tools/release061_remote_build.sh hip gfx942 $REMOTE_OUT > $REMOTE_LOG
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
  for i in 1 2 3 4 5 6; do
    code=$(api -o /dev/null -w '%{http_code}' -X DELETE "$API/droplets/$DROPLET_ID")
    log "DELETE droplet $DROPLET_ID -> HTTP $code"
    case "$code" in 204|404) break ;; esac
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
$SSH 'rocm-smi --showproductname 2>/dev/null | grep -i "card series\|name" | head -2' | tee "$OUT/device.txt" | sed 's/^/[amd gpu] /'

log "shipping $ARCHIVE_BYTES bytes"
scp -q $SSH_OPTS "$TMPD/src.tgz" "root@$IP:/root/src.tgz" || { log "scp failed"; exit 6; }
REMOTE_SHA=$($SSH 'sha256sum /root/src.tgz' | cut -d' ' -f1)
[ "$REMOTE_SHA" = "$ARCHIVE_SHA" ] || { log "archive sha mismatch after transfer ($REMOTE_SHA)"; exit 6; }
$SSH "test ! -e /root/mojolearn && mkdir /root/mojolearn && tar -xzf /root/src.tgz -C /root/mojolearn \
  && printf '%s\n' '$COMMIT' > /root/mojolearn/commit.txt && test -x $REMOTE_PY" || { log "remote unpack failed"; exit 6; }
log "shipped $COMMIT"

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
if [ -n \"\$need\" ]; then
  timeout -k 10 180 apt-get -qq -o Acquire::Retries=1 -o Acquire::http::Timeout=30 update > /root/apt.log 2>&1
  timeout -k 10 300 apt-get -qq -o Acquire::Retries=1 install -y --no-install-recommends \$need >> /root/apt.log 2>&1; echo APT_EXIT=\$? need=\$need
fi
export PATH=/root/.pixi/bin:\$PATH
command -v pixi >/dev/null || timeout -k 10 120 sh -c 'curl -fsSL --max-time 30 https://pixi.sh/install.sh | sh' > /root/pixi_bootstrap.log 2>&1
cd /root/mojolearn && $REMOTE_PY tools/amd_serial_guard.py --seconds $PREP_SECONDS --rss-gib 12 -- \
  pixi install --locked --environment default > /root/pixi_install.log 2>&1; echo PIXI_INSTALL_EXIT=\$?
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
$SSH "cd /root/mojolearn && nohup bash -c 'export PATH=/root/.pixi/bin:\$PATH; \
  MOJOLEARN_COMMIT=$COMMIT MOJOLEARN_PYTHON=$REMOTE_PY MOJOLEARN_RELEASE_BUILD_SECONDS=$WORK_SECONDS \
  timeout -k 20 $((WORK_SECONDS + 40)) bash tools/release061_remote_build.sh hip gfx942 $REMOTE_OUT > $REMOTE_LOG 2>&1; \
  echo \$? > /root/rel061.exit' > /dev/null 2>&1 < /dev/null &" || { log "could not start the build"; exit 9; }
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
python3 - "$OUT/release-build" "$COMMIT" "$OUT/source_inventory_local.json" <<'PY' 2>&1 | tee -a "$STATE"
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
    assert len(proof['extensions']) == 46, 'expected 46 extensions'
    for name, digest in proof['extensions'].items():
        assert name.startswith('mojolearn/hip/gfx942/'), 'wrong architecture path ' + name
        assert hashlib.sha256((out / 'build/sets' / name[len('mojolearn/'):]).read_bytes()).hexdigest() == digest, 'fetched byte mismatch ' + name
    print('admission=BUILT_NOT_INSTALLED hip/gfx942 46 extensions, fetched bytes match proof')
except Exception as exc:
    print('admission=REFUSED ' + str(exc))
PY
log "done -- $OUT (leg.txt has the verdict; the droplet is destroyed by the EXIT trap next)"
