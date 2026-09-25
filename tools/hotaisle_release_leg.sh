#!/usr/bin/env bash
# THE AMD RELEASE BUILD LEG ON HOT AISLE (2026-09-25): the same gfx942 build as
# tools/do_release061_leg.sh, on a Hot Aisle 1x MI300X VM when DigitalOcean is
# unavailable (its one GPU droplet held, as by the GPT-3 Small run).
#
#   bash tools/hotaisle_release_leg.sh <frozen-40-hex-commit>                     DRY RUN, rents nothing
#   bash tools/hotaisle_release_leg.sh <frozen-40-hex-commit> --rent              spends money
#   ... [--rent] [--expect-from <NVIDIA leg release-build dir>] [--lease MIN] [--cap USD]
#
# WHAT IS THE SAME AS THE DIGITALOCEAN LEG, BY CONSTRUCTION. The frozen commit's
# `git archive` (same excludes, same 24 MiB cap, the same native-inventory
# compare against this checkout), unpacked at /root/mojolearn with commit.txt;
# the same host preparation (patchelf, docker, python3-venv, pixi, `pixi install
# --locked --environment default` under tools/amd_serial_guard.py, patchelf
# 0.17.2.4 in /root/release-tools); the same pinned Ubuntu 22.04 ROCm container
# (tools/release_ubuntu22_build.sh prepare, then run hip gfx942
# /root/rel061-build, its bytes copied beside the evidence and sha256-checked on
# the box) running tools/release061_remote_build.sh: every gfx942 binding in the
# IDENTICAL, deterministic and fast tiers plus the host set, the read-backs
# (readback.txt, arch_readback.txt) and build/build-provenance.json. The same
# paths on the box, so the compiler witnesses are the same. The same output
# tree: $MOJOLEARN_RELEASE_RESULTS_ROOT/hip-gfx942/release-build/ (default
# ~/mojolearn-evidence/releases/<commit>/hip-gfx942/), which is exactly where
# tools/release.py's linux-wait and the packer read it, with leg.txt,
# source_inventory_local.json, the logs and the admission line beside it. The
# same exit codes: 0 BUILT_NOT_INSTALLED, 10 the remote work or the admission
# failed, 2 refused before anything was created.
#
# WHAT DIFFERS. The box: a Hot Aisle MI300X VM (gfx942, the architecture the
# wheel ships; the MI325X is gfx942 too) whose login is the `hotaisle` user, so
# every box command runs as root through `sudo -n bash -c`, and files come home
# as a tar stream rather than rsync. The guards are tools/hotaisle_vm_lib.sh's:
# a slot shared with tools/hotaisle_leg.sh; the whole horizon priced live and
# refused above --cap (default $10) or when the balance cannot hold it plus
# $5; a Mac dead-man armed before the create; the description PATCHed; an
# on-box watchdog verified from two ssh sessions; DELETE ?force=true then GET
# 404 or absent from the listing, and only then the dead-man cancelled and the
# slot released. ONLY THE 1x VM: tools/amd_serial_guard.py (the host pixi
# install and every step of the build) requires exactly one visible AMD render
# GPU, and the 2x VM shows two, so MOJOLEARN_HOTAISLE_RELEASE_SPEC is 1gpu here.
#
# THE LEASE. 0.8.18's DigitalOcean leg took 11 minutes from create to verified
# destroy, the build itself 469 s at four jobs on eight cores
# (~/mojolearn-evidence/release/0.8.18/f2293183c729/legs/hip-gfx942/). The
# default lease is 60 minutes, the maximum this leg takes: the on-box watchdog
# fires at the lease, the build's own bound is the lease less the fetch
# reserve (600 s), never above release061_remote_build.sh's 2400 s.
set -uo pipefail

COMMIT="${1:?frozen 40-hex commit}"
shift
RENT=0; EXPECT_FROM=${MOJOLEARN_EXPECT_CORE_HOST_FROM:-}; LEASE=60; CAP_USD=10
USAGE="usage: $0 <commit> [--rent] [--expect-from <nvidia release-build dir>] [--lease MIN] [--cap USD]"
while [ $# -gt 0 ]; do
  case "$1" in
    --rent) RENT=1 ;;
    --expect-from) [ $# -ge 2 ] || { echo "$USAGE" >&2; exit 2; }; EXPECT_FROM=$2; shift ;;
    --lease) [ $# -ge 2 ] || { echo "$USAGE" >&2; exit 2; }; LEASE=$2; shift ;;
    --cap) [ $# -ge 2 ] || { echo "$USAGE" >&2; exit 2; }; CAP_USD=$2; shift ;;
    *) echo "$USAGE" >&2; exit 2 ;;
  esac
  shift
done
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT=$REPO
log() { echo "[$(date +%T) amd/hotaisle] $*"; }
say() { log "$@"; }
die() { log "REFUSING: $1"; exit "${2:-2}"; }
with_timeout() { _secs=$1; shift; perl -e 'alarm shift @ARGV; exec @ARGV or exit 127' "$_secs" "$@"; }
sha256_of() { { shasum -a 256 "$1" 2>/dev/null || sha256sum "$1"; } | cut -d' ' -f1; }

case "$LEASE" in ''|*[!0-9]*) die "--lease must be whole minutes" ;; esac
[ "$LEASE" -ge 30 ] && [ "$LEASE" -le 60 ] || die "--lease must be 30..60 minutes (a longer build is a second leg, never an extension)"
printf '%s' "$CAP_USD" | grep -Eq '^[0-9]+(\.[0-9]{1,2})?$' || die "--cap must be a dollar figure like 10 or 7.50"
CAP_CENTS=$(awk -v d="$CAP_USD" 'BEGIN { printf "%d", d * 100 + 0.5 }')
case "${MOJOLEARN_HOTAISLE_RELEASE_SPEC:-1gpu}" in
  1gpu) ;;
  *) die "the build leg needs the 1x MI300X VM: tools/amd_serial_guard.py requires exactly one visible AMD render GPU and the 2x VM shows two (MOJOLEARN_HOTAISLE_RELEASE_SPEC=${MOJOLEARN_HOTAISLE_RELEASE_SPEC})" ;;
esac
TMPD="$(mktemp -d "${TMPDIR:-/tmp}/hotaisle-release.XXXXXX")"
chmod 700 "$TMPD"
# shellcheck source=tools/hotaisle_vm_lib.sh
. "$REPO/tools/hotaisle_vm_lib.sh"
HA_SPEC_WANT=1gpu
: "${HA_STOCK_WAIT_MINUTES:=0}"

# The box's paths. MOJOLEARN_HOTAISLE_BOX_ROOT exists only so the shim test can
# run the box on this machine; a real leg is /root, the path the DigitalOcean
# leg builds under (the compiler witnesses and the container helper need it).
BR=${MOJOLEARN_HOTAISLE_BOX_ROOT:-/root}
printf '%s' "$BR" | grep -Eq '^(/root|/[A-Za-z0-9/_.-]*/root)$' || die "bad MOJOLEARN_HOTAISLE_BOX_ROOT"
BOX_GUARD=/var/lib/mojolearn-hotaisle-release
[ "$BR" = /root ] || BOX_GUARD="${BR}-guard"
REMOTE_PY=${MOJOLEARN_HOTAISLE_REMOTE_PY:-/usr/bin/python3}   # the VM's stdlib 3.12, as the droplet's
REMOTE_OUT=$BR/rel061-build; REMOTE_LOG=$BR/rel061-build.log
FETCH_RESERVE=${FETCH_RESERVE:-600}
BUILD_JOBS=${MOJOLEARN_BUILD_JOBS:-4}
[[ "$BUILD_JOBS" =~ ^[1-9][0-9]?$ && "$BUILD_JOBS" -le 16 ]] || die 'MOJOLEARN_BUILD_JOBS must be 1..16'
LEG_EVIDENCE_ROOT="${MOJOLEARN_EVIDENCE_ROOT:-$HOME/mojolearn-evidence}"
RELEASE_ROOT="${MOJOLEARN_RELEASE_RESULTS_ROOT:-$LEG_EVIDENCE_ROOT/releases/$COMMIT}"
OUT="$RELEASE_ROOT/hip-gfx942"
STATE="$OUT/leg.txt"

# CORE HOST PROBE, exactly tools/do_release061_leg.sh's (read its header): skip
# unless --expect-from names an NVIDIA leg of THIS commit, whose STAGED set copy
# gives the digest. pack_wheel.py compares every host binding across legs anyway.
CORE_HOST_SHA=${MOJOLEARN_EXPECT_CORE_HOST_SHA256:-}
CORE_HOST_SOURCE=${CORE_HOST_SHA:+by-hand}
if [ -n "$EXPECT_FROM" ]; then
  derived=$(python3 - "$EXPECT_FROM" "$COMMIT" <<'PYX'
import hashlib, json, pathlib, sys
root, commit = pathlib.Path(sys.argv[1]).expanduser(), sys.argv[2]
cands = [root, root / 'release-build', root / 'remote' / 'release-build']
base = next((c for c in cands if (c / 'build' / 'build-provenance.json').is_file()), None)
if base is None:
    sys.exit('no build/build-provenance.json under ' + str(root) + ' (give an NVIDIA leg release-build directory)')
proof = json.loads((base / 'build' / 'build-provenance.json').read_text())
if proof.get('source_commit') != commit:
    sys.exit('the NVIDIA build is of %s, not %s' % (proof.get('source_commit'), commit))
if proof.get('complete') is not True or proof.get('build_exit') != 0:
    sys.exit('the NVIDIA build proof is not complete with build_exit 0')
sos = sorted((base / 'build' / 'sets' / 'cuda').glob('*/host/_mojolearn_core_host.so'))
if not sos:
    sys.exit('no staged build/sets/cuda/<arch>/host/_mojolearn_core_host.so under ' + str(base))
digests = {hashlib.sha256(p.read_bytes()).hexdigest() for p in sos}
if len(digests) != 1:
    sys.exit('the staged core host copies disagree: ' + ' '.join(sorted(digests)))
print(digests.pop(), sos[0])
PYX
) || die "--expect-from $EXPECT_FROM: refused"
  set -- $derived
  if [ -n "$CORE_HOST_SHA" ] && [ "$CORE_HOST_SHA" != "$1" ]; then
    die "MOJOLEARN_EXPECT_CORE_HOST_SHA256=$CORE_HOST_SHA disagrees with the staged copy $2 ($1)"
  fi
  CORE_HOST_SHA=$1; CORE_HOST_SOURCE=$2
fi
[ -n "$CORE_HOST_SHA" ] || { CORE_HOST_SHA=skip; CORE_HOST_SOURCE=none; }
[[ "$CORE_HOST_SHA" =~ ^([0-9a-f]{64}|skip)$ ]] || die 'MOJOLEARN_EXPECT_CORE_HOST_SHA256 must be 64 lowercase hex (the STAGED NVIDIA set copy) or skip'

# ------------------------------------------------------------ preflight (as the DigitalOcean leg)
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "commit must be the full 40-hex SHA, got '$COMMIT'"
git -C "$REPO" cat-file -e "$COMMIT^{commit}" 2>/dev/null || die "commit $COMMIT is not in $REPO"
[ ! -e "$OUT/release-build" ] || die "$OUT/release-build already exists; nothing local is ever overwritten"
log "archiving $COMMIT"
git -C "$REPO" archive --format=tar "$COMMIT" -- . ':!bench/results' ':!bench/evidence' ':!mamba/corpus' ':!bench/oracle*' \
  | gzip > "$TMPD/src.tgz" || die "git archive failed"
ARCHIVE_BYTES=$(wc -c < "$TMPD/src.tgz" | tr -d ' ')
[ "$ARCHIVE_BYTES" -lt $((24 * 1024 * 1024)) ] || die "archive is $ARCHIVE_BYTES bytes gzipped, cap 24 MiB"
ARCHIVE_SHA=$(sha256_of "$TMPD/src.tgz")
mkdir "$TMPD/archive" && tar -xzf "$TMPD/src.tgz" -C "$TMPD/archive" || die "archive does not unpack"
python3 -B - "$TMPD/archive" "$REPO" > "$TMPD/source_inventory_local.json" <<'PY' || die "native inventory differs between $COMMIT and the working tree; freeze first"
import json, pathlib, subprocess, sys
archive, local = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(archive / 'tools'))
from check_linux_release_qualification import native_inventory
tracked = set(subprocess.run(['git', '-C', str(local), 'ls-files', '-z'], check=True,
                             capture_output=True, timeout=60).stdout.decode().split('\0'))
a = native_inventory(archive)
l = [entry for entry in native_inventory(local) if entry[0] in tracked]
if a != l:
    names = sorted({p for p, _ in set(map(tuple, a)) ^ set(map(tuple, l))})
    raise SystemExit('differs: ' + ' '.join(names[:8]))
print(json.dumps(a, separators=(',', ':')))
PY
log "archive $ARCHIVE_BYTES bytes, sha256 $ARCHIVE_SHA, native inventory matches working tree"
HELPER="$REPO/tools/release_ubuntu22_build.sh"
[ -f "$HELPER" ] || die "no $HELPER"

# The host preparation, tools/do_release061_leg.sh's with MOJOLEARN_RELEASE_UBUNTU22=1,
# as a file run as root on the VM. @..@ are filled below.
cat > "$TMPD/host_prep.sh" <<'PREP'
# tools/hotaisle_release_leg.sh's host preparation: tools/do_release061_leg.sh's,
# for the pinned Ubuntu 22.04 container build. Runs as root on the VM.
set -u; export DEBIAN_FRONTEND=noninteractive
BR='@BR@'; REMOTE_PY='@REMOTE_PY@'; PREP_SECONDS='@PREP_SECONDS@'
export HOME="$BR"   # root's home, as on the droplet: pixi and its caches live under /root
need=''; command -v patchelf >/dev/null || need="$need patchelf"
command -v docker >/dev/null || need="$need docker.io"
{ command -v objdump && command -v strings; } >/dev/null || need="$need binutils"
"$REMOTE_PY" -c 'import ensurepip' 2>/dev/null || need="$need python3-venv python3-pip"
if [ -n "$need" ]; then
  update_end=$(( $(date +%s) + 180 )); updated=0
  : > "$BR/apt.log"
  while [ $(date +%s) -lt $update_end ]; do
    timeout -k 10 60 apt-get -qq -o Acquire::Retries=1 -o Acquire::http::Timeout=20 update >> "$BR/apt.log" 2>&1 && { updated=1; break; }
    sleep 5
  done
  if [ $updated = 1 ]; then
    timeout -k 10 300 apt-get -qq -o DPkg::Lock::Timeout=120 -o Acquire::Retries=1 install -y --no-install-recommends $need >> "$BR/apt.log" 2>&1; echo APT_EXIT=$? need=$need
  else
    echo APT_EXIT=124 need=$need
  fi
fi
export PATH=$BR/.pixi/bin:$PATH
command -v pixi >/dev/null || timeout -k 10 120 sh -c 'curl -fsSL --max-time 30 https://pixi.sh/install.sh | sh' > "$BR/pixi_bootstrap.log" 2>&1
cd "$BR/mojolearn" && "$REMOTE_PY" tools/amd_serial_guard.py --seconds "$PREP_SECONDS" --rss-gib 12 -- \
  pixi install --locked --environment default > "$BR/pixi_install.log" 2>&1; echo PIXI_INSTALL_EXIT=$?
tail -40 "$BR/apt.log" 2>/dev/null
.pixi/envs/default/bin/python -m venv "$BR/release-tools" &&
timeout -k 10 120 "$BR/release-tools/bin/python" -m pip install --disable-pip-version-check --only-binary=:all: --retries 1 --timeout 20 patchelf==0.17.2.4
export PATH=$BR/release-tools/bin:$PATH
command -v patchelf >/dev/null && patchelf --version
for t in taskset objdump patchelf pixi docker; do command -v $t >/dev/null || echo MISSING_$t; done
test -x .pixi/envs/default/bin/mojo && test -x .pixi/envs/default/bin/python && echo PIXI_ENV_OK || echo PIXI_ENV_MISSING
PREP

if [ $RENT = 0 ]; then
  cat <<EOF
DRY RUN -- nothing rented. With --rent this leg would:
  rent     a Hot Aisle 1x MI300X VM (tools/hotaisle_vm_lib.sh), team $HA_TEAM, lease ${LEASE}m, horizon $(( LEASE + HA_READY_SECONDS / 60 + 10 ))m priced live, cap \$$CAP_USD
  guards   slot ($HA_SLOT_PREFIX.N), balance >= lease + \$5, Mac dead-man before the create, description PATCH,
           on-box watchdog in $BOX_GUARD verified from two sessions, DELETE ?force=true then GET 404 or absent
  upload   $TMPD/src.tgz ($ARCHIVE_BYTES bytes, sha256 $ARCHIVE_SHA) -> $BR/mojolearn + commit.txt=$COMMIT
  prepare  as root: patchelf/docker/python3-venv if absent; pixi; pixi install --locked --environment default (guarded); patchelf 0.17.2.4
  build    MOJOLEARN_COMMIT=$COMMIT MOJOLEARN_RELEASE_BUILD_SECONDS=<=2400 MOJOLEARN_BUILD_JOBS=$BUILD_JOBS
           bash $BR/release_ubuntu22_build.sh run hip gfx942 $REMOTE_OUT > $REMOTE_LOG
           (pinned Ubuntu 22.04 container $(sed -n 's/^IMAGE=//p' "$HELPER"), core host probe: $CORE_HOST_SHA from $CORE_HOST_SOURCE)
  fetch    $REMOTE_OUT -> $OUT/release-build/ ; logs and leg.txt beside it
  destroy  DELETE ?force=true then GET 404 or absent from the listing; never deletes anything local
EOF
  mkdir -p "$TMPD/dry"
  ha_write_deadman "$TMPD/dry/dm" "$(( $(date +%s) + 60 ))" /dev/null || die "the Mac dead-man did not compose" 1
  ha_write_watchdog "$TMPD/dry/watchdog.sh" "$BOX_GUARD" 60 DRYRUN_REF || die "the on-box watchdog did not compose" 1
  sed -e "s|@BR@|$BR|g" -e "s|@REMOTE_PY@|$REMOTE_PY|g" -e "s|@PREP_SECONDS@|900|g" "$TMPD/host_prep.sh" > "$TMPD/dry/prep.sh"
  bash -n "$TMPD/dry/prep.sh" || die "the host preparation is not valid bash" 1
  echo "  checks   the Mac dead-man, the on-box watchdog and the host preparation compose (sh -n, bash -n)"
  if _why=$(ha_key_hygiene) && ha_load_key; then
    ha_call GET "teams/$HA_TEAM/balance/"; echo "  balance  HTTP $HA_CODE; $(ha_dollars "$(ha_py balance)")"
    ha_call GET "teams/$HA_TEAM/virtual_machines/available/"
    read -r _f _spec _q _p _m _c <<< "$(ha_py pick 1gpu "$TMPD/dry/pick.json")"
    echo "  stock    1x MI300X: $_f quantity $_q ${_p} cents/h (all offerings: $(ha_py offers | tr '\n' ';'))"
  else
    echo "  no Hot Aisle key here (${_why:-empty key file}); a rent would refuse"
  fi
  rm -rf "$TMPD"
  exit 0
fi

# ------------------------------------------------------------------ --rent
mkdir -p "$OUT" || die "cannot create $OUT"
teardown() {
  rc=$?
  trap - EXIT INT TERM
  log "teardown (rc=$rc)"
  HA_RECORD="$OUT/hotaisle.txt"
  if ha_teardown; then
    [ "$HA_CREATE_ATTEMPTED" = 1 ] && echo "destroyed=$(date -u +%FT%TZ) verified_gone ${HA_GONE_LINE:-}" >> "$STATE"
  else
    echo "destroyed=UNCONFIRMED" >> "$STATE"; [ "$rc" = 0 ] && rc=1
  fi
  if [ -n "$HA_CURLRC" ]; then ha_call GET "teams/$HA_TEAM/balance/"; echo "balance_after_cents=$(ha_py balance)" >> "$STATE"; fi
  ha_spend >> "$STATE"
  echo "finished=$(date -u +%FT%TZ) rc=$rc" >> "$STATE"
  rm -rf "$TMPD"
  exit $rc
}
trap teardown EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

LEG_START=$(date +%s)
cp "$TMPD/source_inventory_local.json" "$OUT/source_inventory_local.json"
{
  echo "vendor=amd"; echo "arch=hip/gfx942"; echo "provider=hotaisle"; echo "commit=$COMMIT"
  echo "archive_sha256=$ARCHIVE_SHA"; echo "archive_bytes=$ARCHIVE_BYTES"
  echo "lease_minutes=$LEASE"; echo "cap_cents=$CAP_CENTS"; echo "fetch_reserve=$FETCH_RESERVE"; echo "started=$(date -u +%FT%TZ)"
  echo "expect_core_host_sha256=$CORE_HOST_SHA source=$CORE_HOST_SOURCE"
} > "$STATE"

ha_rent "rel-amd-build" "$LEASE" "$CAP_CENTS" "$OUT/hotaisle.txt" "$BOX_GUARD" "$OUT" \
  || die "Hot Aisle created nothing: $HA_REFUSED" 2
{ echo "vm=$HA_VMREF"; echo "vm_name=$HA_VMNAME"; echo "spec=$HA_SPEC_USED"; echo "price_cents_per_hour=$HA_PRICE"
  echo "created=$(date -u -r "$HA_T_CREATE" +%FT%TZ 2>/dev/null || date -u -d "@$HA_T_CREATE" +%FT%TZ)"; echo "gfx=$HA_GFX"; } >> "$STATE"
cp "$OUT/hotaisle_device.txt" "$OUT/device.txt"

log "shipping $ARCHIVE_BYTES bytes"
ha_ssh 300 "cat > $BR/src.tgz" < "$TMPD/src.tgz" || { log "upload failed"; exit 6; }
REMOTE_SHA=$(ha_ssh 60 "sha256sum $BR/src.tgz" < /dev/null | cut -d' ' -f1)
[ "$REMOTE_SHA" = "$ARCHIVE_SHA" ] || { log "archive sha mismatch after transfer ($REMOTE_SHA)"; exit 6; }
ha_ssh 120 "test ! -e $BR/mojolearn && mkdir $BR/mojolearn && tar -xzf $BR/src.tgz -C $BR/mojolearn \
  && printf '%s\n' '$COMMIT' > $BR/mojolearn/commit.txt && test -x $REMOTE_PY && echo UNPACKED" < /dev/null | grep -q UNPACKED \
  || { log "remote unpack failed"; exit 6; }
log "shipped $COMMIT"

PREP_SECONDS=$(( HA_T_CREATE + LEASE * 60 - $(date +%s) - FETCH_RESERVE - 600 ))
[ "$PREP_SECONDS" -gt 900 ] && PREP_SECONDS=900
[ "$PREP_SECONDS" -ge 120 ] || { log "only ${PREP_SECONDS}s for host prep; skipping"; exit 7; }
sed -e "s|@BR@|$BR|g" -e "s|@REMOTE_PY@|$REMOTE_PY|g" -e "s|@PREP_SECONDS@|$PREP_SECONDS|g" "$TMPD/host_prep.sh" > "$OUT/host_prep.sh"
ha_ssh 60 "cat > $BR/hotaisle_host_prep.sh" < "$OUT/host_prep.sh" || { log "could not deliver the host preparation"; exit 8; }
ha_ssh $(( PREP_SECONDS + 600 )) "bash $BR/hotaisle_host_prep.sh" < /dev/null 2>&1 | tee "$OUT/prep-console.log" | sed 's/^/[amd prep] /'
if ! grep -q '^PIXI_INSTALL_EXIT=0$' "$OUT/prep-console.log" || grep -q 'MISSING_\|PIXI_ENV_MISSING' "$OUT/prep-console.log" \
   || ! grep -qx 'patchelf 0.17.2' "$OUT/prep-console.log"; then
  echo "build_exit=NOT_STARTED_HOST_PREP_FAILED" >> "$STATE"; log "host prep failed"; exit 8
fi
# Environment setup is controller-owned, outside the frozen source archive.
# Retain its exact bytes beside the build evidence and pin the transferred copy.
cp "$HELPER" "$OUT/release_ubuntu22_build.sh" || exit 8
HELPER_SHA=$(sha256_of "$OUT/release_ubuntu22_build.sh")
ha_ssh 60 "cat > $BR/release_ubuntu22_build.sh" < "$OUT/release_ubuntu22_build.sh" || exit 8
[ "$(ha_ssh 60 "sha256sum $BR/release_ubuntu22_build.sh" < /dev/null | cut -d' ' -f1)" = "$HELPER_SHA" ] || { log "helper sha mismatch on the box"; exit 8; }
echo "container_helper_sha256=$HELPER_SHA" >> "$STATE"
ha_ssh 600 "bash $BR/release_ubuntu22_build.sh prepare" < /dev/null > "$OUT/container-prepare.log" 2>&1 \
  || { log 'Ubuntu 22.04 build image preparation failed'; exit 8; }
echo 'build_environment=ROCm 6.4.1 Ubuntu 22.04 pinned container' >> "$STATE"

# THE BUILD, DETACHED AND POLLED, so a dropped ssh cannot kill it.
WORK_SECONDS=$(( HA_T_CREATE + LEASE * 60 - $(date +%s) - FETCH_RESERVE ))
[ "$WORK_SECONDS" -gt 2400 ] && WORK_SECONDS=2400
[ "$WORK_SECONDS" -ge 300 ] || { echo "build_exit=NOT_STARTED_${WORK_SECONDS}s_LEFT" >> "$STATE"; log "only ${WORK_SECONDS}s left; skipping the build"; exit 7; }
echo "work_seconds=$WORK_SECONDS" >> "$STATE"; log "build bound ${WORK_SECONDS}s"
BUILD_T0=$(date +%s)
ha_ssh 60 "cd $BR/mojolearn && rm -f $BR/rel061.exit && if command -v setsid > /dev/null 2>&1; then S=setsid; else S=; fi; \$S nohup bash -c 'export HOME=$BR PATH=$BR/release-tools/bin:$BR/.pixi/bin:\$PATH; \
  MOJOLEARN_COMMIT=$COMMIT MOJOLEARN_PYTHON=$REMOTE_PY MOJOLEARN_RELEASE_BUILD_SECONDS=$WORK_SECONDS \
  MOJOLEARN_EXPECT_CORE_HOST_SHA256=$CORE_HOST_SHA MOJOLEARN_BUILD_JOBS=$BUILD_JOBS \
  timeout -k 20 $((WORK_SECONDS + 40)) bash $BR/release_ubuntu22_build.sh run hip gfx942 $REMOTE_OUT > $REMOTE_LOG 2>&1; \
  echo \$? > $BR/rel061.exit' > /dev/null 2>&1 < /dev/null & echo STARTED" < /dev/null | grep -q STARTED \
  || { log "could not start the build"; exit 9; }
BUILD_EXIT=""
while [ "$(date +%s)" -lt $(( HA_T_CREATE + LEASE * 60 - FETCH_RESERVE + 60 )) ]; do
  BUILD_EXIT=$(ha_ssh 45 "cat $BR/rel061.exit 2>/dev/null; true" < /dev/null 2>/dev/null | tr -d '[:space:]')
  [ -n "$BUILD_EXIT" ] && break
  sleep "${MOJOLEARN_HOTAISLE_BUILD_POLL_SECONDS:-30}"
done
BUILD_SECONDS=$(( $(date +%s) - BUILD_T0 ))
echo "build_exit=${BUILD_EXIT:-NO_EXIT_BEFORE_FETCH_RESERVE}" >> "$STATE"
echo "build_seconds=$BUILD_SECONDS" >> "$STATE"
log "build exit ${BUILD_EXIT:-none} after ${BUILD_SECONDS}s; $(ha_ssh 30 "tail -3 $REMOTE_LOG" < /dev/null 2>/dev/null | tr '\n' '|')"

log "fetch evidence"
mkdir -p "$OUT/release-build"
if ha_ssh 600 "cd $REMOTE_OUT && tar czf - ." < /dev/null | tar xzf - -C "$OUT/release-build"; then log "fetched release-build/"; else log "FETCH FAILED (release-build/)"; fi
if [ "$CORE_HOST_SHA" != skip ]; then
  mkdir -p "$OUT/toolchain-probe"
  ha_ssh 120 "cd $BR/release-toolchain-probe && tar czf - ." < /dev/null | tar xzf - -C "$OUT/toolchain-probe" || log 'FETCH FAILED (toolchain-probe/)'
fi
AMD_CORE_HOST="$OUT/release-build/build/sets/hip/gfx942/host/_mojolearn_core_host.so"
if [ -f "$AMD_CORE_HOST" ]; then
  AMD_CORE_SHA=$(sha256_of "$AMD_CORE_HOST")
  echo "staged_core_host_sha256=$AMD_CORE_SHA" >> "$STATE"
  if [[ "$CORE_HOST_SHA" =~ ^[0-9a-f]{64}$ ]] && [ "$CORE_HOST_SHA" != "$AMD_CORE_SHA" ]; then
    log "WARNING: staged AMD core host $AMD_CORE_SHA differs from the NVIDIA copy $CORE_HOST_SHA; pack_wheel.py will refuse"
  fi
fi
ha_ssh 120 "cd $BR && tar czf - rel061-build.log pixi_install.log pixi_bootstrap.log apt.log rel061.exit 2>/dev/null; true" < /dev/null \
  | tar xzf - -C "$OUT" 2>/dev/null || true
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
if [ "${BUILD_EXIT:-}" != 0 ]; then
  log "remote work did not pass (exit ${BUILD_EXIT:-missing})"; exit 10
fi
grep -q '^admission=BUILT_NOT_INSTALLED ' "$STATE" && ! grep -q '^admission=REFUSED' "$STATE" || exit 10
log "done -- $OUT (leg.txt has the verdict; the VM is deleted by the EXIT trap next)"
