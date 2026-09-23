#!/usr/bin/env bash
# tools/vultr_leg.sh. ONE GUARDED VULTR AMD BARE METAL LEG THAT RUNS A LANE'S
# EXTRA BODY: price, create, install ROCm, ship the commit, pixi and the
# gates, the body, fetch, DELETE, verify gone.
#
#   MOJOLEARN_VULTR_TOKEN_FILE=$HOME/.mojolearn_vultr_token \
#   MOJOLEARN_GPU_ARCHS=gfx942 \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/attention_step_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-amd-mi300x8-vultr \
#   bash tools/vultr_leg.sh amd [--rent] [--minutes N] [--dry-run] [--skip-gates]
#   bash tools/vultr_leg.sh amd --segment-lease N --dollar-cap USD   # a lease above one hour, priced first
#   bash tools/vultr_leg.sh amd --plan vbm-256c-3072gb-8-mi325x-gpu --region ord ...   # another plan or region
#
#   amd   plan vbm-256c-2048gb-8-mi300x-gpu (8x MI300X, gfx942), Ubuntu 24.04 x64,
#         region: the first of the plan's locations whose availability lists it
#   --plan     vbm-256c-2048gb-8-mi300x-gpu (default) or vbm-256c-3072gb-8-mi325x-gpu
#   --region   a Vultr region id (ewr, ord, ...); default walks the plan's locations
#   --ubuntu   24.04 (default, the DigitalOcean image's release) or 22.04
#   --gpus N   gfx942 agents the box must show after ROCm is up (default 8)
#   --rent     the default; accepted so a caller can say it
#
# --dry-run rents nothing, needs no token and makes no API call. Exit 0 green,
# 1 this script is broken, 3 the world is not ready (dirty tree, oversized
# bundle). A real leg exits 3 when it refuses before the create (a live box,
# the lock, no stock, an account limit), 2 on a bad token file or argument.
#
# WHY THIS FILE EXISTS. The GPT-3 Small run driver (tools/lm_run_driver.py,
# _rent_amd_once, spec key amd_providers) walks AMD providers, and
# DigitalOcean allows one GPU droplet per account. Vultr sells 8x MI300X and
# 8x MI325X bare metal by the hour. This is tools/do_extra_leg.sh's contract
# on that box: the same body contract, the same output files under
# MOJOLEARN_GEMM_LEG_OUT, and the same guards.
#
# THE BODY CONTRACT IS tools/do_extra_leg.sh's, BYTE FOR BYTE WHERE IT CAN BE:
#   * the body is a local POSIX sh file, `sh -n` checked here, shipped to
#     /root/gemm_leg_extra.sh and copied to <leg out>/extra_body.sh;
#   * it runs as `sh /root/gemm_leg_extra.sh > /root/gemm_leg_out/extra.log
#     2>&1` with cwd /root/mojolearn, and its exit code lands in
#     /root/gemm_leg_out/leg.txt as extra_exit=;
#   * /root/mojolearn is `git archive` of the pinned commit, no .git;
#   * MOJOLEARN_GPU_ARCHS (required, gfx942) and MOJOLEARN_TARGET_COLUMN=amd
#     are exported to the body;
#   * everything under /root/gemm_leg_out comes home to <leg out>/remote/.
#
# WHERE IT DIFFERS FROM tools/do_extra_leg.sh, EACH ON PURPOSE:
#   1. ROCm IS INSTALLED BEFORE THE BODY. DigitalOcean's image 188571990
#      ships it; a Vultr Ubuntu bare metal does not. The DigitalOcean legs ran
#      Ubuntu 24.04 (kernel 6.8.0-59-generic, bench/results/lm_t2_2026-09-23/
#      A-2/uname.txt) with ROCm 6.4.0-47 and amdgpu 6.12.12
#      (bench/results/knn_amd_shared_tile_2026-09-21/digitalocean-mi325x-wide/
#      rocm_provenance.txt). The leg installs the same release here:
#      amdgpu-install 6.4.60400 from repo.radeon.com for the OS codename,
#      kernel headers, `amdgpu-install -y --usecase=rocm` (dkms driver
#      included), a reboot when the loaded amdgpu is not the dkms one, then
#      ssh again. /opt/rocm/.info/version must start with 6.4.0, /dev/kfd must
#      exist, rocm-smi must answer and rocminfo must list --gpus gfx942
#      agents, or the box is deleted unused. The install log comes home to
#      <leg out>/rocm/. Nothing of mojolearn is built by this step: the body
#      runs the same published wheel's gfx942 binding it runs on
#      DigitalOcean.
#   2. BARE METAL TAKES 10 TO 30 MINUTES TO PROVISION and the ROCm install
#      and reboot take more. All of it is inside the lease; size the lease
#      for it (the driver's vultr_extra_minutes).
#   3. THE ON-BOX DEAD-MAN SLEEPS TO AN ABSOLUTE DEADLINE and is also enabled
#      as a systemd unit, so the reboot does not disarm it. It is verified
#      again after the reboot.
#   4. THE PRICE IS READ EVERY TIME from GET /plans-metal (hourly_cost; when
#      a listing carries only monthly_cost it is divided by 672, Vultr's
#      monthly hour cap, which can only overstate the hourly price) and
#      written to lease.txt. --dollar-cap is enforced as on DigitalOcean.
#   5. NO STOCK AND ACCOUNT LIMITS ARE NAMED: "Vultr has NO STOCK" before the
#      create, "Vultr REFUSED the create on account limits or stock" after
#      it. tools/lm_run_driver.py reads both as busy and walks to the next
#      provider.
#
# THE GUARDS, THE SAME AS tools/do_extra_leg.sh (copied, not factored out,
# for the reason its header gives). Vultr bills until DELETE, so:
#   * the tree must be clean (minus bench/results) or nothing is rented;
#   * ONE VULTR GPU BARE METAL AT A TIME: the create is refused while any
#     bare metal on a GPU plan, or any bare metal tagged or labelled as this
#     leg, exists, and the refusal names them; and /tmp/mojolearn-vultr-gpu.lock
#     is held across sessions on this Mac;
#   * a DETACHED LOCAL DEAD-MAN is armed BEFORE the create, keyed by tag AND
#     label, at the lease deadline;
#   * a SECOND DEAD-MAN RUNS ON THE BOX, verified by process, by the id baked
#     in, and by a GET with the token that must return 200; if it cannot be
#     verified the box is deleted unused;
#   * an unreadable create response is ADOPTED by label rather than orphaned;
#   * DELETE is an EXIT trap, confirmed only by a follow-up GET returning 404
#     (written to teardown.txt); a box that cannot be reached is not gone
#     until the API says 404. The local dead-man is cancelled only after that.
#
# THE TOKEN. MOJOLEARN_VULTR_TOKEN_FILE (default ~/.mojolearn_vultr_token),
# one line, mode 600, outside the repository. A missing or empty file is
# refused by name. It is read once by the builtin `read`, written by the
# builtin `printf` into a 0600 curl config that every call reads with
# `curl -K`, never exported, never in an argv here or on the box, and it
# reaches the box on ssh STDIN (/root/.mojolearn-vultr.curlrc, 0600). Both
# process lists are then searched for it.
#
# TEST HOOKS (tools/tests/test_vultr_leg_shim.py): MOJOLEARN_VULTR_API (the
# API base), MOJOLEARN_VULTR_SSH_BIN and MOJOLEARN_VULTR_SCP_BIN (the ssh and
# scp programs), MOJOLEARN_VULTR_ROCM_DEB_URL (the amdgpu-install package),
# MOJOLEARN_VULTR_UPLINK_HOSTS, MOJOLEARN_VULTR_POLL_SECONDS (every poll
# interval), MOJOLEARN_VULTR_GPU_LOCK and MOJOLEARN_VULTR_SSH_PUBKEY.
set -uo pipefail

# RUN FROM AN IMMUTABLE SNAPSHOT (tools/do_extra_leg.sh does the same): bash
# reads a script lazily by offset, and a leg lasts long enough for someone to
# edit this file.
if [ "${MOJOLEARN_VULTR_LEG_FROZEN:-0}" != 1 ]; then
  _repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 2
  _snapdir="$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-vultr-snapshot.XXXXXX")" || exit 2
  cp "${BASH_SOURCE[0]}" "$_snapdir/vultr_leg.sh" || { rm -rf "$_snapdir"; exit 2; }
  chmod 555 "$_snapdir/vultr_leg.sh"
  trap 'rm -rf "$_snapdir"' EXIT
  MOJOLEARN_VULTR_LEG_FROZEN=1 MOJOLEARN_VULTR_LEG_REPO="$_repo" \
    bash "$_snapdir/vultr_leg.sh" "$@"
  exit $?
fi

REPO="${MOJOLEARN_VULTR_LEG_REPO:?}"
cd "$REPO" || exit 2
API="${MOJOLEARN_VULTR_API:-https://api.vultr.com/v2}"
API="${API%/}"
TAG=mojolearn-extra
FETCH_RESERVE="${FETCH_RESERVE:-420}"
MAX_BUNDLE_BYTES="${MOJOLEARN_VULTR_MAX_BYTES:-15000000}"
SSH_BIN="${MOJOLEARN_VULTR_SSH_BIN:-ssh}"
SCP_BIN="${MOJOLEARN_VULTR_SCP_BIN:-scp}"
UPLINK_HOSTS="${MOJOLEARN_VULTR_UPLINK_HOSTS:-https://pypi.org/ https://github.com/ https://www.google.com/}"
SSH_PUBKEY_FILE="${MOJOLEARN_VULTR_SSH_PUBKEY:-$HOME/.ssh/id_ed25519.pub}"
# ROCm 6.4.0, the release the DigitalOcean image ran (header, difference 1).
ROCM_WANT=6.4.0
ROCM_DEB_NAME=amdgpu-install_6.4.60400-1_all.deb
# Poll intervals. MOJOLEARN_VULTR_POLL_SECONDS replaces every one (the shim test).
_P="${MOJOLEARN_VULTR_POLL_SECONDS:-}"
case "$_P" in *[!0-9]*) echo "MOJOLEARN_VULTR_POLL_SECONDS must be whole seconds" >&2; exit 2 ;; esac
T_PROV="${_P:-20}"; T_SSH="${_P:-5}"; T_WORK="${_P:-30}"; T_VERIFY="${_P:-15}"
T_UPLINK="${_P:-7}"; T_REBOOT="${_P:-15}"; T_DELETE="${_P:-10}"
PROVISION_MAX_SECONDS="${MOJOLEARN_VULTR_PROVISION_SECONDS:-2700}"
ROCM_MAX_SECONDS="${MOJOLEARN_VULTR_ROCM_SECONDS:-3000}"
REBOOT_MAX_SECONDS="${MOJOLEARN_VULTR_REBOOT_SECONDS:-1800}"
# Vultr bills bare metal by the hour up to a monthly cap of 672 hours.
MONTH_HOURS=672

usage() {
  sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

VENDOR=""; MINUTES=60; DRY=0; GATES=1
SEGMENT_LEASE=""; DOLLAR_CAP=""; SEGMENT_CAP_MINUTES=2880
PLAN=vbm-256c-2048gb-8-mi300x-gpu; REGION_OVERRIDE=""; UBUNTU=24.04; GPUS=8
while [ $# -gt 0 ]; do
  case "$1" in
    amd)
      [ -z "$VENDOR" ] || { echo "one vendor per leg" >&2; exit 2; }
      VENDOR=$1 ;;
    nv|cpu-intel|cpu-amd) echo "tools/vultr_leg.sh rents AMD bare metal only; '$1' is tools/do_extra_leg.sh's" >&2; exit 2 ;;
    --rent) ;;
    --minutes) shift; MINUTES="${1:-}" ;;
    --minutes=*) MINUTES="${1#--minutes=}" ;;
    --segment-lease) shift; SEGMENT_LEASE="${1:-}" ;;
    --segment-lease=*) SEGMENT_LEASE="${1#--segment-lease=}" ;;
    --dollar-cap) shift; DOLLAR_CAP="${1:-}" ;;
    --dollar-cap=*) DOLLAR_CAP="${1#--dollar-cap=}" ;;
    --plan) shift; PLAN="${1:-}" ;;
    --plan=*) PLAN="${1#--plan=}" ;;
    --region) shift; REGION_OVERRIDE="${1:-}" ;;
    --region=*) REGION_OVERRIDE="${1#--region=}" ;;
    --ubuntu) shift; UBUNTU="${1:-}" ;;
    --ubuntu=*) UBUNTU="${1#--ubuntu=}" ;;
    --gpus) shift; GPUS="${1:-}" ;;
    --gpus=*) GPUS="${1#--gpus=}" ;;
    --dry-run) DRY=1 ;;
    --skip-gates) GATES=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
[ -n "$VENDOR" ] || { usage >&2; exit 2; }
case "$MINUTES" in ''|*[!0-9]*) echo "--minutes must be a whole number" >&2; exit 2 ;; esac
if [ -n "$SEGMENT_LEASE" ] || [ -n "$DOLLAR_CAP" ]; then
  { [ -n "$SEGMENT_LEASE" ] && [ -n "$DOLLAR_CAP" ]; } || { echo "--segment-lease and --dollar-cap go together" >&2; exit 2; }
  [ "$MINUTES" = 60 ] || { echo "--minutes and --segment-lease are two ways to say one thing; give one" >&2; exit 2; }
  case "$SEGMENT_LEASE" in ''|*[!0-9]*) echo "--segment-lease must be whole minutes" >&2; exit 2 ;; esac
  case "$DOLLAR_CAP" in ''|*[!0-9.]*|.|*.*.*) echo "--dollar-cap must be a dollar figure like 120 or 47.50" >&2; exit 2 ;; esac
  if [ "$SEGMENT_LEASE" -le 60 ] || [ "$SEGMENT_LEASE" -gt "$SEGMENT_CAP_MINUTES" ]; then
    echo "--segment-lease is for leases ABOVE one hour and at most $SEGMENT_CAP_MINUTES minutes (48 h); got $SEGMENT_LEASE" >&2; exit 2
  fi
  MINUTES=$SEGMENT_LEASE
elif [ "$MINUTES" -gt 60 ]; then
  echo "--minutes $MINUTES REFUSED: one hour is the hard cap for a rented GPU (a second leg, never an extension); a training segment names its lease with --segment-lease N --dollar-cap USD" >&2
  exit 2
fi
[ "$MINUTES" -ge 10 ] || { echo "--minutes must be at least 10" >&2; exit 2; }
case "$PLAN" in
  vbm-*-mi300x-gpu|vbm-*-mi325x-gpu) ;;
  *) echo "--plan must be a Vultr AMD Instinct bare metal plan (vbm-...-mi300x-gpu or vbm-...-mi325x-gpu), got '$PLAN'" >&2; exit 2 ;;
esac
case "$PLAN" in *[!a-z0-9-]*) echo "--plan: lowercase letters, digits and - only" >&2; exit 2 ;; esac
case "$REGION_OVERRIDE" in *[!a-z0-9-]*) echo "--region: lowercase letters, digits and - only" >&2; exit 2 ;; esac
case "$UBUNTU" in
  24.04) CODENAME=noble ;;
  22.04) CODENAME=jammy ;;
  *) echo "--ubuntu must be 24.04 or 22.04, got '$UBUNTU'" >&2; exit 2 ;;
esac
case "$GPUS" in ''|*[!0-9]*|0) echo "--gpus must be a whole number above 0" >&2; exit 2 ;; esac
ROCM_DEB_URL="${MOJOLEARN_VULTR_ROCM_DEB_URL:-https://repo.radeon.com/amdgpu-install/6.4/ubuntu/$CODENAME/$ROCM_DEB_NAME}"
case "$ROCM_DEB_URL" in *[!A-Za-z0-9_.:/%-]*) echo "MOJOLEARN_VULTR_ROCM_DEB_URL: letters, digits and _.:/%- only" >&2; exit 2 ;; esac

NAME=mojolearn-extra-amd-vultr
BODY_VENDOR=amd; COLUMN=amd
case "$PLAN" in *mi325x*) GPU_LABEL=amd-mi325x8-vultr ;; *) GPU_LABEL=amd-mi300x8-vultr ;; esac
SMI_CMD='rocm-smi --showproductname; echo "-- free VRAM at acceptance --"; rocm-smi --showmeminfo vram'

GPU_ARCHS="${MOJOLEARN_GPU_ARCHS:-}"
case "$GPU_ARCHS" in *[!A-Za-z0-9_]*)
  echo "MOJOLEARN_GPU_ARCHS='$GPU_ARCHS': exactly one architecture name (one mojo build is one GPU arch)" >&2; exit 2 ;;
esac
if [ -z "$GPU_ARCHS" ]; then
  echo "MOJOLEARN_GPU_ARCHS is required on amd (the MI300X and MI325X are gfx942). The bodies derive it from nvidia-smi, which this box does not have." >&2
  exit 2
fi
CARD_FULL="${MOJOLEARN_GEMM_CARD_FULL:-}"
LEG_DUMP="${MOJOLEARN_IDENTITY_TRACE_DUMP:-}"
for _v in "$CARD_FULL" "$LEG_DUMP"; do
  case "$_v" in *[!A-Za-z0-9_.,:-]*) echo "MOJOLEARN_GEMM_CARD_FULL / MOJOLEARN_IDENTITY_TRACE_DUMP: letters, digits and _.,:- only" >&2; exit 2 ;; esac
done

LEG_EXTRA="${MOJOLEARN_GEMM_LEG_EXTRA:-}"
[ -n "$LEG_EXTRA" ] || { echo "MOJOLEARN_GEMM_LEG_EXTRA=<body.sh> is required: running a body is this runner's whole job" >&2; exit 2; }
[ -f "$LEG_EXTRA" ] || { echo "MOJOLEARN_GEMM_LEG_EXTRA=$LEG_EXTRA does not exist" >&2; exit 2; }

TOKFILE="${MOJOLEARN_VULTR_TOKEN_FILE:-$HOME/.mojolearn_vultr_token}"
STAMP="$(date -u +%Y-%m-%d_%H%M%S)"
# THE VULTR GPU LOCK: one Vultr GPU bare metal at a time across every session
# on this Mac. A lock older than 100 minutes with no GPU or mojolearn bare
# metal live is an orphan and may be broken.
GPU_LOCK="${MOJOLEARN_VULTR_GPU_LOCK:-/tmp/mojolearn-vultr-gpu.lock}"
LOCK_STALE_SECONDS=6000
LOCK_LANE="${MOJOLEARN_VULTR_LOCK_LANE:-extra:$(basename "$LEG_EXTRA" .sh)}"
case "$LOCK_LANE" in *[!A-Za-z0-9_.,:-]*) echo "MOJOLEARN_VULTR_LOCK_LANE: letters, digits and _.,:- only" >&2; exit 2 ;; esac
LOCK_NONCE="$$-$STAMP"
LOCK_HELD=0

# Environment for the extra body, as tools/do_extra_leg.sh's
# MOJOLEARN_DO_EXTRA_ENV: NAME=value words, MOJOLEARN_ or MODULAR_ names,
# values of letters, digits and _.,:/=- only.
EXTRA_ENV="${MOJOLEARN_VULTR_EXTRA_ENV:-${MOJOLEARN_DO_EXTRA_ENV:-}}"
OUT="${MOJOLEARN_GEMM_LEG_OUT:-bench/results/e1g/${STAMP}-${GPU_LABEL}-extra}"
case "$OUT" in /*) ;; *) OUT="$REPO/$OUT" ;; esac
REAL_OUT="$OUT"
if [ "$DRY" = 1 ]; then
  OUT="$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-vultr-dryrun.XXXXXX")" || exit 2
fi

log() { printf '[%s %s/vultr] %s\n' "$(date +%T)" "$VENDOR" "$*"; }
die() { printf '\n%s\n' "$1" >&2; exit "${2:-1}"; }
sha256_of() {
  if command -v shasum > /dev/null 2>&1; then shasum -a 256 "$1"; else sha256sum "$1"; fi | awk '{print $1}'
}
utc_of() { date -u -r "$1" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d "@$1" +%Y-%m-%dT%H:%M:%SZ; }

# ------------------------------------------------------------------ state
TMPD=""; CURLRC=""; TOKPAT=""
BM_ID=""; IP=""; CREATE_ATTEMPTED=0; DESTROY_CONFIRMED=0
DEADMAN_PID=""; DEADMAN_DIR=""; DEADLINE_EPOCH=0; LEG_START=0
KEY_RED=0; FETCH_RED=0; BODY_STATE=not_started; SSH=("$SSH_BIN"); SSHN=("$SSH_BIN" -n)
REGION=""; OS_ID=""; SSHKEY_ID=""; PRICE_HOURLY=""; PRICE_SOURCE=""; MAX_COST=""

TMPD="$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-vultr.XXXXXX")" || exit 2
chmod 700 "$TMPD"

# shellcheck disable=SC2317
cancel_deadman() {
  [ -n "$DEADMAN_PID" ] || return 0
  pkill -P "$DEADMAN_PID" 2>/dev/null
  kill "$DEADMAN_PID" 2>/dev/null && log "local dead-man cancelled (pid $DEADMAN_PID)"
  [ -n "$DEADMAN_DIR" ] && rm -rf "$DEADMAN_DIR"
  echo "local_deadman=cancelled $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/deadman.txt"
  DEADMAN_PID=""; DEADMAN_DIR=""
}

# shellcheck disable=SC2317
release_lock() {
  [ "$LOCK_HELD" = 1 ] || return 0
  if grep -qx "nonce=$LOCK_NONCE" "$GPU_LOCK/owner" 2>/dev/null; then
    rm -rf "$GPU_LOCK" && log "released the Vultr GPU lock $GPU_LOCK"
    echo "gpu_lock=released $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/leg.txt"
  else
    log "!! $GPU_LOCK no longer carries this leg's nonce; left in place"
    echo "gpu_lock=NOT_OURS_AT_RELEASE" >> "$OUT/leg.txt"
  fi
  LOCK_HELD=0
}

lock_age() {
  local m
  m=$(stat -f %m "$GPU_LOCK" 2>/dev/null || stat -c %Y "$GPU_LOCK" 2>/dev/null) || return 1
  echo $(( $(date +%s) - m ))
}

take_lock() {
  mkdir "$GPU_LOCK" 2>/dev/null || return 1
  LOCK_HELD=1
  {
    echo "lane=$LOCK_LANE"
    echo "script=tools/vultr_leg.sh"
    echo "pid=$$"
    echo "nonce=$LOCK_NONCE"
    echo "label=$NAME"
    echo "plan=$PLAN"
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "out=$REAL_OUT"
  } > "$GPU_LOCK/owner"
}

# shellcheck disable=SC2317
http_code() {  # <method> <url> <body out>; prints the HTTP code, 000 on transport failure
  local c
  c=$(curl -K "$CURLRC" --max-time 30 -o "$3" -w '%{http_code}' -X "$1" "$2" 2>>"$TMPD/curl.err") || c=000
  printf '%s' "$c"
}

# shellcheck disable=SC2317
ids_by_name() {  # ids of bare metal tagged $TAG or labelled $NAME; returns 1 when the listing failed
  local c
  c=$(http_code GET "$API/bare-metals?per_page=500" "$TMPD/byname.json")
  [ "$c" = 200 ] || return 1
  python3 "$TMPD/byname.py" "$TMPD/byname.json" "$NAME" "$TAG"
}

# shellcheck disable=SC2317
destroy_bare_metal() {
  local ids id i c ok_all=1 gone
  ids="$BM_ID"
  if [ -z "$ids" ]; then
    if ! ids=$(ids_by_name); then
      log "!! could not list bare metal to find $NAME; deletion UNCONFIRMED"
      echo "sweep_by_name=LISTING_FAILED" >> "$OUT/teardown.txt"
      return 1
    fi
    if [ -z "$ids" ]; then
      echo "sweep_by_name=none (listing HTTP 200 shows no bare metal tagged $TAG or labelled $NAME)" >> "$OUT/teardown.txt"
      DESTROY_CONFIRMED=1
      return 0
    fi
  fi
  for id in $ids; do
    for i in 1 2 3 4 5 6; do
      c=$(http_code DELETE "$API/bare-metals/$id" /dev/null)
      log "DELETE bare metal $id -> HTTP $c"
      echo "delete $id attempt $i -> HTTP $c" >> "$OUT/teardown.txt"
      case "$c" in 204|404) break ;; esac
      sleep "$T_DELETE"
    done
    # DELETE 204 acknowledges the request. Only GET 404 proves absence.
    gone=0
    for i in $(seq 1 40); do
      c=$(http_code GET "$API/bare-metals/$id" /dev/null)
      log "post-delete GET bare metal $id -> HTTP $c"
      echo "verify $id attempt $i -> HTTP $c" >> "$OUT/teardown.txt"
      if [ "$c" = 404 ]; then
        gone=1
        echo "verified_gone=$id GET HTTP 404 at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/teardown.txt"
        break
      fi
      sleep "$T_VERIFY"
    done
    [ "$gone" = 1 ] || ok_all=0
  done
  [ "$ok_all" = 1 ] && DESTROY_CONFIRMED=1
  return 0
}

# shellcheck disable=SC2317
teardown() {
  local rc=$?
  trap - EXIT INT TERM
  if [ "$CREATE_ATTEMPTED" = 1 ]; then
    log "teardown (exit $rc)"
    echo "== teardown $(date -u +%Y-%m-%dT%H:%M:%SZ) exit=$rc bare_metal=${BM_ID:-unknown} ==" >> "$OUT/teardown.txt"
    destroy_bare_metal
    echo "destroy_confirmed=$DESTROY_CONFIRMED" >> "$OUT/teardown.txt"
  fi
  if [ "$CREATE_ATTEMPTED" = 0 ] || [ "$DESTROY_CONFIRMED" = 1 ]; then
    cancel_deadman
    release_lock
  else
    {
      echo
      echo "  ############################################################"
      echo "  # BARE METAL ${BM_ID:-<unknown id> labelled $NAME} MAY STILL BE BILLING."
      echo "  # The API did not confirm it is gone. BOTH dead-men are LEFT"
      echo "  # ARMED on purpose (local pid ${DEADMAN_PID:-none}, dir ${DEADMAN_DIR:-none};"
      echo "  # the on-box one fires by id). Delete it by hand now:"
      echo "  #   https://my.vultr.com/"
      echo "  # and only then: kill ${DEADMAN_PID:-<pid>}; rm -rf ${DEADMAN_DIR:-<dir>}"
      [ "$LOCK_HELD" = 1 ] && echo "  # The Vultr GPU lock stays HELD until then: rm -rf $GPU_LOCK"
      echo "  ############################################################"
    } | tee -a "$OUT/teardown.txt" >&2
    echo "local_deadman=LEFT_ARMED pid=$DEADMAN_PID" >> "$OUT/deadman.txt"
    [ "$rc" = 0 ] && rc=1
  fi
  [ "$CREATE_ATTEMPTED" = 1 ] && echo "exit=$rc" >> "$OUT/teardown.txt"
  [ -n "$TMPD" ] && rm -rf "$TMPD"
  exit "$rc"
}
trap teardown EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ------------------------------------------------------------- local checks
RED=0; BLOCK=0
rok()    { printf '  ok     %s\n' "$1"; }
rbad()   { RED=1;   printf '  FAIL   %s\n' "$1"; }
rblock() { BLOCK=1; printf '  BLOCK  %s\n' "$1"; }

COMMIT="$(git -C "$REPO" rev-parse HEAD)" || die "not a git checkout: $REPO" 2
COMMIT_LINE="$(git -C "$REPO" log -1 --format='%h parent %p' "$COMMIT")"
mkdir -p "$OUT" || die "cannot create $OUT" 2

echo "== vultr_leg: one Vultr AMD bare metal leg running an extra body =="
echo "   mode      $( [ "$DRY" = 1 ] && echo 'DRY RUN (nothing is rented, no API call)' || echo 'RENT' )"
echo "   commit    $COMMIT_LINE"
echo "   box       $NAME  plan=$PLAN region=${REGION_OVERRIDE:-<walk the plan locations>} ubuntu=$UBUNTU tag=$TAG"
echo "   rocm      $ROCM_WANT from $ROCM_DEB_URL ($GPUS gfx942 agents required)"
echo "   lease     $MINUTES minutes (local and on-box dead-men at that deadline)"
echo "   body      $LEG_EXTRA"
echo "   gates     $( [ "$GATES" = 1 ] && echo 'device check + card (RunPod order)' || echo 'SKIPPED (--skip-gates)' )"
echo "   archs     $GPU_ARCHS   column $COLUMN"
echo "   api       $API"
echo "   out       $REAL_OUT"
[ "$DRY" = 1 ] && echo "   dry out   $OUT"
echo

echo "== local checks =="
DIRTY="$(git -C "$REPO" status --porcelain -- . ':!bench/results' 2>/dev/null)"
if [ -n "$DIRTY" ]; then
  rblock "the tree is DIRTY (minus bench/results); a real leg refuses. Launch from git worktree add --detach:"
  printf '%s\n' "$DIRTY" | head -20 | sed 's/^/           /'
else
  rok "the tree is clean (minus bench/results)"
fi

if sh -n "$LEG_EXTRA" 2> "$TMPD/extra_syntax.err"; then
  rok "the extra body is valid sh: $LEG_EXTRA"
else
  rbad "the extra body is NOT valid sh: $(head -3 "$TMPD/extra_syntax.err")"
fi
cp "$LEG_EXTRA" "$OUT/extra_body.sh"
EXTRA_SHA="$(sha256_of "$OUT/extra_body.sh")"

{
  echo "# Generated by tools/vultr_leg.sh from MOJOLEARN_VULTR_EXTRA_ENV; sourced before the extra body."
  _env_ok=1
  for _w in $EXTRA_ENV; do
    case "$_w" in
      MOJOLEARN_DO_*=*|MOJOLEARN_VULTR_*=*) _env_ok=0; printf '# REFUSED (runner-only name): %s\n' "${_w%%=*}" ;;
      MOJOLEARN_[A-Z0-9_]*=*|MODULAR_[A-Z0-9_]*=*)
        _n=${_w%%=*}; _v=${_w#*=}
        case "$_n$_v" in
          *[!A-Za-z0-9_.,:/=-]*) _env_ok=0; printf '# REFUSED (characters): %s\n' "$_n" ;;
          *) printf "export %s='%s'\n" "$_n" "$_v" ;;
        esac ;;
      *) _env_ok=0; printf '# REFUSED (not NAME=value with a MOJOLEARN_ or MODULAR_ name): %s\n' "${_w%%=*}" ;;
    esac
  done
} > "$OUT/extra_env.sh"
if [ "${_env_ok:-1}" = 1 ] && sh -n "$OUT/extra_env.sh" 2>/dev/null; then
  rok "the extra body environment: $(grep -c '^export ' "$OUT/extra_env.sh" | tr -d ' ') export(s)$( [ -n "$EXTRA_ENV" ] && echo ": $EXTRA_ENV")"
else
  rbad "MOJOLEARN_VULTR_EXTRA_ENV is refused: $(grep '^# REFUSED' "$OUT/extra_env.sh" | tr '\n' ' ')"
fi
_env_ok=1

token_hygiene() {  # prints a reason and returns 1 when the file must not be used
  local perm
  [ -f "$TOKFILE" ] || { echo "token file $TOKFILE does not exist (create a Vultr API key and store it as one line in $TOKFILE, mode 600)"; return 1; }
  [ -s "$TOKFILE" ] || { echo "token file $TOKFILE is EMPTY (one line, the Vultr API key)"; return 1; }
  perm=$(stat -f '%OLp' "$TOKFILE" 2>/dev/null || stat -c '%a' "$TOKFILE" 2>/dev/null || echo '?')
  [ "$perm" = 600 ] || { echo "token file $TOKFILE is mode $perm, must be 600"; return 1; }
  case "$(cd "$(dirname "$TOKFILE")" && pwd)/" in "$REPO"/*) echo "token file $TOKFILE is INSIDE the repository"; return 1 ;; esac
  if git -C "$REPO" ls-files --error-unmatch "$TOKFILE" > /dev/null 2>&1; then
    echo "token file $TOKFILE is TRACKED BY GIT"; return 1
  fi
  return 0
}
if _why=$(token_hygiene); then
  rok "token file present, 0600, outside the repository (not read by a dry run)"
elif [ "$DRY" = 1 ]; then
  printf '  info   %s (a dry run does not need it)\n' "$_why"
else
  die "REFUSING to rent: $_why" 2
fi

if [ -f "$SSH_PUBKEY_FILE" ] && [ -s "$SSH_PUBKEY_FILE" ]; then
  rok "ssh public key $SSH_PUBKEY_FILE (matched against the account's keys, registered if absent)"
else
  rbad "no ssh public key at $SSH_PUBKEY_FILE (MOJOLEARN_VULTR_SSH_PUBKEY)"
fi

if _age=$(lock_age); then
  printf '  info   the Vultr GPU lock %s is HELD now (%ss old) by: %s; a real leg refuses unless it is over %ss old with no GPU bare metal live\n' \
    "$GPU_LOCK" "$_age" "$(tr '\n' ' ' < "$GPU_LOCK/owner" 2>/dev/null || echo 'no owner file')" "$LOCK_STALE_SECONDS"
else
  rok "the Vultr GPU lock $GPU_LOCK is free (a real leg takes it as lane $LOCK_LANE; a dry run never does)"
fi

# ---- the bundle ----
ARCHIVE_EXCLUDES=(':!bench/results' ':!mamba/corpus' ':!bench/oracle*' ':!bench/minentropy_oracle.txt' ':!*.bin')
if git -C "$REPO" archive --format=tar -o "$TMPD/src.tar" "$COMMIT" -- . "${ARCHIVE_EXCLUDES[@]}"; then
  gzip -9 -c "$TMPD/src.tar" > "$TMPD/src.tgz"
  tar tf "$TMPD/src.tar" | grep -v '/$' > "$OUT/bundle_files.txt"
  BUNDLE_BYTES=$(wc -c < "$TMPD/src.tgz" | tr -d ' ')
  BUNDLE_SHA=$(sha256_of "$TMPD/src.tgz")
  mkdir "$TMPD/archive" && tar xf "$TMPD/src.tar" -C "$TMPD/archive"
  rm -f "$TMPD/src.tar"
  if grep -Eq '(^bench/results/|^mamba/corpus/|\.bin$)' "$OUT/bundle_files.txt"; then
    rbad "the bundle carries an excluded path:"
    grep -E '(^bench/results/|^mamba/corpus/|\.bin$)' "$OUT/bundle_files.txt" | head -5 | sed 's/^/           /'
  fi
  if [ "$BUNDLE_BYTES" -gt "$MAX_BUNDLE_BYTES" ]; then
    rblock "the bundle is $BUNDLE_BYTES bytes gzipped, over the $MAX_BUNDLE_BYTES cap"
  else
    rok "bundle $BUNDLE_BYTES bytes gzipped, $(wc -l < "$OUT/bundle_files.txt" | tr -d ' ') files, sha256 ${BUNDLE_SHA:0:16}"
  fi
  REQUIRED="pixi.toml pixi.lock"
  [ "$GATES" = 1 ] && REQUIRED="$REQUIRED tools/with_identical_mode.sh tools/gemm_card.sh gemm/checks/gemm_device_check.mojo"
  for _need in $REQUIRED; do
    [ -f "$TMPD/archive/$_need" ] || rbad "the bundle does not contain $_need"
  done
else
  rbad "git archive of $COMMIT failed"
  BUNDLE_BYTES=0; BUNDLE_SHA=none
fi

source_sha_recipe() {
  ( cd "$1" && \
    { find . -name '*.mojo' -not -path './.pixi/*' -not -path './bench/results/*' \
        | LC_ALL=C sort | xargs shasum -a 256 2>/dev/null || \
      find . -name '*.mojo' -not -path './.pixi/*' -not -path './bench/results/*' \
        | LC_ALL=C sort | xargs sha256sum ; } \
    | { shasum -a 256 2>/dev/null || sha256sum ; } | awk '{print $1}' )
}
[ -d "$TMPD/archive" ] && source_sha_recipe "$TMPD/archive" > "$OUT/source_sha256_local.txt"

# ---- the remote body (tools/do_extra_leg.sh's, provider and plan named) ----
cat > "$OUT/remote_body.sh" <<'REMOTE_BODY'
#!/bin/sh
# Generated by tools/vultr_leg.sh. RUNS ON THE VULTR BARE METAL, after ROCm.
# The same steps, in the same order, as tools/do_extra_leg.sh's remote body,
# so a MOJOLEARN_GEMM_LEG_EXTRA body sees the same world.
#
# DELIBERATELY `set -u` AND NOT `set -e`. A gate that goes red is a RESULT
# and its log has to come home. POSIX sh only; Ubuntu's /bin/sh is dash.
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out
mkdir -p "$OUT"
cd "$ROOT" || exit 9

{
  echo "vendor=@VENDOR@"
  echo "commit=@COMMIT@"
  echo "card_full=@CARDFULL@"
  echo "trace_dump=@DUMP@"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "provider=vultr"
  echo "size=@PLAN@"
  echo "gates=@GATES@"
  echo "gpu_archs=@GPUARCHS@"
  echo "target_column=@COLUMN@"
  echo "rocm_version=$(cat /opt/rocm/.info/version 2>/dev/null)"
} > "$OUT/leg.txt"

MOJOLEARN_GPU_ARCHS="@GPUARCHS@"
if [ -n "$MOJOLEARN_GPU_ARCHS" ]; then export MOJOLEARN_GPU_ARCHS; else unset MOJOLEARN_GPU_ARCHS; fi
MOJOLEARN_TARGET_COLUMN="@COLUMN@"
export MOJOLEARN_TARGET_COLUMN

uname -a > "$OUT/uname.txt" 2>&1
@SMI@ > "$OUT/gpu.txt" 2>&1 || echo "no vendor smi tool answered" >> "$OUT/gpu.txt"
cp -r /root/vultr_rocm "$OUT/rocm" 2>/dev/null || true

{ find . -name '*.mojo' -not -path './.pixi/*' -not -path './bench/results/*' \
    | LC_ALL=C sort | xargs shasum -a 256 2>/dev/null || \
  find . -name '*.mojo' -not -path './.pixi/*' -not -path './bench/results/*' \
    | LC_ALL=C sort | xargs sha256sum ; } \
  | { shasum -a 256 2>/dev/null || sha256sum ; } \
  | awk '{print $1}' > "$OUT/source_sha256.txt"

if [ ! -x "$HOME/.pixi/bin/pixi" ] && ! command -v pixi > /dev/null 2>&1; then
    curl -fsSL https://pixi.sh/install.sh | sh > "$OUT/pixi_install.log" 2>&1
fi
PATH="$HOME/.pixi/bin:$PATH"
export PATH
command -v pixi > "$OUT/pixi_which.txt" 2>&1 || echo "NO PIXI" >> "$OUT/pixi_which.txt"

pixi install > "$OUT/pixi_env.log" 2>&1
echo "pixi_install_exit=$?" >> "$OUT/leg.txt"
pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1 || true

if [ "@GATES@" = "1" ]; then
    tools/with_identical_mode.sh pixi run mojo run -I . \
        gemm/checks/gemm_device_check.mojo > "$OUT/device_check.log" 2>&1
    echo "device_check_exit=$?" >> "$OUT/leg.txt"
    MOJOLEARN_GEMM_CARD_FULL="@CARDFULL@" MOJOLEARN_IDENTITY_TRACE_DUMP="@DUMP@" \
        sh tools/gemm_card.sh device "$OUT/@VENDOR@.card" > "$OUT/card_driver.log" 2>&1
    echo "card_exit=$?" >> "$OUT/leg.txt"
else
    echo "device_check_exit=SKIPPED" >> "$OUT/leg.txt"
    echo "card_exit=SKIPPED" >> "$OUT/leg.txt"
fi

if [ -f /root/gemm_leg_extra.sh ]; then
    (
        if [ -f /root/gemm_leg_extra_env.sh ]; then
            . /root/gemm_leg_extra_env.sh
        fi
        sh /root/gemm_leg_extra.sh
    ) > "$OUT/extra.log" 2>&1
    echo "extra_exit=$?" >> "$OUT/leg.txt"
    cp /root/gemm_leg_extra_env.sh "$OUT/extra_env.sh" 2>/dev/null || true
fi

echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/leg.txt"
: > /root/gemm_leg.done
echo REMOTE_BODY_DONE
REMOTE_BODY

subst() {  # <file>: replace every placeholder, then prove none survived
  sed -e "s|@VENDOR@|$BODY_VENDOR|g" \
      -e "s|@COMMIT@|$COMMIT|g" \
      -e "s|@CARDFULL@|$CARD_FULL|g" \
      -e "s|@DUMP@|$LEG_DUMP|g" \
      -e "s|@PLAN@|$PLAN|g" \
      -e "s|@GATES@|$GATES|g" \
      -e "s|@GPUARCHS@|$GPU_ARCHS|g" \
      -e "s|@COLUMN@|$COLUMN|g" \
      -e "s|@SMI@|$SMI_CMD|g" \
      -e "s|@WORK@|${WORK_SECONDS:-0}|g" \
      -e "s|@SHA@|${BUNDLE_SHA:-none}|g" \
      -e "s|@SECS@|${SUBST_SECS:-0}|g" \
      -e "s|@DEADLINE@|${SUBST_DEADLINE:-0}|g" \
      -e "s|@ID@|${SUBST_ID:-0}|g" \
      -e "s|@API@|$API|g" \
      -e "s|@TAG@|$TAG|g" \
      -e "s|@NAME@|$NAME|g" \
      -e "s|@CODENAME@|$CODENAME|g" \
      -e "s|@DEBURL@|$ROCM_DEB_URL|g" \
      -e "s|@ROCMWANT@|$ROCM_WANT|g" \
      -e "s|@GPUS@|$GPUS|g" \
      -e "s|@RECORD@|$OUT/deadman.txt|g" \
      "$1" > "$1.subst" && mv "$1.subst" "$1"
  if grep -q '@[A-Z][A-Z_]*@' "$1"; then
    grep -n '@[A-Z][A-Z_]*@' "$1" | sed 's/^/           /'
    return 1
  fi
  return 0
}

check_posix() {  # <file> <label>
  local f=$1 what=$2
  sh -n "$f" 2> "$TMPD/sh_n.err" || { rbad "$what is not valid sh: $(head -2 "$TMPD/sh_n.err")"; return 1; }
  if command -v dash > /dev/null 2>&1; then
    dash -n "$f" 2> "$TMPD/sh_n.err" || { rbad "$what is not valid dash: $(head -2 "$TMPD/sh_n.err")"; return 1; }
  fi
  awk '!/^[[:space:]]*#/ &&
       (/exec -a/ || /\[\[/ || /(^|[;{ \t])local / || /<\(/ ||
        /(^|[;{ \t])function / || /(^|[;{ \t])source / || /echo -e/) {
           print FNR ": " $0
       }' "$f" > "$TMPD/bashisms"
  if [ -s "$TMPD/bashisms" ]; then
    rbad "$what has a BASHISM (the box runs dash):"; sed 's/^/           /' "$TMPD/bashisms"; return 1
  fi
  return 0
}

if subst "$OUT/remote_body.sh"; then
  if check_posix "$OUT/remote_body.sh" "the remote body"; then
    if grep -q 'gemm_leg.done' "$OUT/remote_body.sh" && grep -q 'sh /root/gemm_leg_extra.sh' "$OUT/remote_body.sh"; then
      rok "the remote body substitutes cleanly, passes sh -n, dash -n and the bashism scan, runs the extra body and writes the sentinel"
    else
      rbad "the remote body lost its extra-body call or its sentinel"
    fi
  fi
else
  rbad "UNSUBSTITUTED PLACEHOLDER in the remote body"
fi

# ---- the local dead-man (composed now, armed only by a real leg) ----
write_local_deadman() {  # <dir> <seconds>
  local d=$1
  ( umask 077; mkdir -p "$d" )
  cat > "$d/deadman.sh" <<'LOCAL_DEADMAN'
#!/bin/sh
# Written by tools/vultr_leg.sh. DETACHED ON PURPOSE. Deletes the bare metal
# that leg created if that leg is no longer here to do it. Keyed by TAG AND
# LABEL, plus the id when the leg learned it. The token is in the 0600 curl
# config beside this file and in no argv.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
sleep @SECS@
L="$D/deadman.log"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) dead-man firing for tag @TAG@ label @NAME@" >> "$L"
ids=""
[ -s "$D/bare_metal_id.txt" ] && ids="$(cat "$D/bare_metal_id.txt")"
curl -K "$D/curlrc" --max-time 30 -o "$D/bare_metals.json" "@API@/bare-metals?per_page=500" >> "$L" 2>&1
ids="$ids $(python3 "$D/byname.py" "$D/bare_metals.json" "@NAME@" "@TAG@")"
for id in $ids; do
    c="$(curl -K "$D/curlrc" --max-time 30 -o /dev/null -w '%{http_code}' -X DELETE "@API@/bare-metals/$id" 2>> "$L")"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) DELETE $id -> $c" >> "$L"
    echo "local_deadman_fired $(date -u +%Y-%m-%dT%H:%M:%SZ) DELETE $id -> $c" >> "@RECORD@"
done
rm -f "$D/curlrc"
LOCAL_DEADMAN
  cp "$TMPD/byname.py" "$d/byname.py"
  SUBST_SECS=$2 subst "$d/deadman.sh" || return 1
  chmod 700 "$d/deadman.sh"
  return 0
}

cat > "$TMPD/byname.py" <<'BYNAME'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
name, tag = sys.argv[2], sys.argv[3]
print(" ".join(str(x.get("id")) for x in d.get("bare_metals", [])
               if x.get("label") == name or tag in (x.get("tags") or [])))
BYNAME

# ---- the on-box dead-man: an absolute deadline, so a reboot cannot reset it ----
cat > "$OUT/box_deadman.sh" <<'BOX_DEADMAN'
#!/bin/sh
# Written by tools/vultr_leg.sh. RUNS ON THE BARE METAL, DETACHED, and as the
# systemd unit mojolearn-selfkill.service after a reboot. The local dead-man
# dies with the Mac; this one does not. It sleeps to an absolute deadline.
# The token is in /root/.mojolearn-vultr.curlrc (0600, delivered on ssh
# stdin), never in an argv.
set -u
while [ "$(date +%s)" -lt @DEADLINE@ ]; do
    sleep 30
done
for attempt in 1 2 3 4 5; do
    code=$(curl -K /root/.mojolearn-vultr.curlrc --max-time 30 -o /root/selfkill.body -w '%{http_code}' \
        -X DELETE '@API@/bare-metals/@ID@')
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) DELETE @ID@ attempt $attempt -> $code" >> /root/selfkill.out
    case "$code" in 2*|404) break ;; esac
    sleep 10
done
BOX_DEADMAN

cat > "$OUT/remote_start.sh" <<'REMOTE_START'
#!/bin/sh
# Written by tools/vultr_leg.sh. Starts the remote body DETACHED under the
# work bound and prints the wrapper pid; the leg polls that pid and the
# sentinel. body_exit=124 is the bound firing.
set -u
rm -f /root/gemm_leg.done /root/gemm_leg_console.log
if command -v timeout > /dev/null 2>&1; then
    nohup sh -c 'timeout -k 30 @WORK@ sh /root/gemm_leg.sh; rc=$?; mkdir -p /root/gemm_leg_out; echo "body_exit=$rc" >> /root/gemm_leg_out/leg.txt' \
        > /root/gemm_leg_console.log 2>&1 < /dev/null &
else
    nohup sh -c 'sh /root/gemm_leg.sh; rc=$?; mkdir -p /root/gemm_leg_out; echo "body_exit=$rc (NO timeout(1): unbounded)" >> /root/gemm_leg_out/leg.txt' \
        > /root/gemm_leg_console.log 2>&1 < /dev/null &
fi
echo "REMOTE_PID=$!"
REMOTE_START

cat > "$OUT/remote_unpack.sh" <<'REMOTE_UNPACK'
#!/bin/sh
# Written by tools/vultr_leg.sh. The box recomputes the archive's sha256 and
# refuses a mismatch.
set -eu
cd /root
got=$(sha256sum extra_src.tgz | awk '{print $1}')
if [ "$got" != "@SHA@" ]; then echo "ARCHIVE SHA MISMATCH: sent @SHA@ got $got"; exit 9; fi
echo ARCHIVE-SHA-OK
rm -rf /root/mojolearn /root/gemm_leg_out /root/gemm_leg.done /root/gemm_leg_extra.sh
mkdir -p /root/mojolearn
tar -xzf extra_src.tgz -C /root/mojolearn
rm -f extra_src.tgz
echo "UNPACKED $(find /root/mojolearn -type f | wc -l) files"
REMOTE_UNPACK

# ---- ROCm: install (detached, polled), then check (after any reboot) ----
cat > "$OUT/remote_rocm_install.sh" <<'REMOTE_ROCM'
#!/bin/sh
# Written by tools/vultr_leg.sh. RUNS ON THE BARE METAL, DETACHED, before the
# body. Installs ROCm @ROCMWANT@ (the DigitalOcean image's release) with the
# dkms amdgpu driver. `set -u` and NOT `set -e`: every step's exit is a line
# in rocm.txt and the log comes home either way. Writes
# /root/vultr_rocm/done LAST.
set -u
D=/root/vultr_rocm
mkdir -p "$D"
rm -f "$D/done"
L="$D/install.log"
R="$D/rocm.txt"
DEBIAN_FRONTEND=noninteractive
export DEBIAN_FRONTEND
codename=unknown
if [ -r /etc/os-release ]; then
    codename=$(sed -n 's/^VERSION_CODENAME=//p' /etc/os-release | tr -d '"')
fi
{
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "os_codename=$codename"
  echo "os_codename_wanted=@CODENAME@"
  echo "kernel=$(uname -r)"
  echo "rocm_wanted=@ROCMWANT@"
  echo "deb_url=@DEBURL@"
} > "$R"
[ "$codename" = "@CODENAME@" ] || echo "os_codename_MISMATCH=1" >> "$R"
have=$(cat /opt/rocm/.info/version 2>/dev/null)
case "$have" in
  @ROCMWANT@*)
    echo "preinstalled=$have" >> "$R"
    echo "installer_exit=SKIPPED" >> "$R" ;;
  *)
    apt-get update > "$L" 2>&1
    echo "apt_update_exit=$?" >> "$R"
    apt-get install -y ca-certificates curl python3-setuptools python3-wheel \
        "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)" >> "$L" 2>&1
    rc=$?
    if [ "$rc" != 0 ]; then
        # linux-modules-extra does not exist for every kernel flavor; the headers are what dkms needs
        apt-get install -y ca-certificates curl python3-setuptools python3-wheel "linux-headers-$(uname -r)" >> "$L" 2>&1
        rc=$?
    fi
    echo "apt_prereq_exit=$rc" >> "$R"
    curl -fsSL -o "$D/amdgpu-install.deb" '@DEBURL@' >> "$L" 2>&1
    echo "deb_fetch_exit=$?" >> "$R"
    echo "deb_sha256=$(sha256sum "$D/amdgpu-install.deb" 2>/dev/null | awk '{print $1}')" >> "$R"
    apt-get install -y "$D/amdgpu-install.deb" >> "$L" 2>&1
    echo "deb_install_exit=$?" >> "$R"
    apt-get update >> "$L" 2>&1
    amdgpu-install -y --usecase=rocm >> "$L" 2>&1
    echo "installer_exit=$?" >> "$R" ;;
esac
dkms_ver=$(dpkg-query -W -f='${Version}' amdgpu-dkms 2>/dev/null)
loaded=$(cat /sys/module/amdgpu/version 2>/dev/null)
{
  echo "rocm_version=$(cat /opt/rocm/.info/version 2>/dev/null)"
  echo "amdgpu_dkms=$dkms_ver"
  echo "amdgpu_loaded=$loaded"
  echo "kfd=$( [ -e /dev/kfd ] && echo present || echo absent)"
} >> "$R"
# A reboot when the kfd node is missing, or the loaded amdgpu is not the dkms
# build (the in-tree module carries no version file, so it reads empty).
reboot_needed=0
[ -e /dev/kfd ] || reboot_needed=1
if [ -n "$dkms_ver" ]; then
    case "$dkms_ver" in *"$loaded"*) [ -n "$loaded" ] || reboot_needed=1 ;; *) reboot_needed=1 ;; esac
fi
echo "reboot_needed=$reboot_needed" >> "$R"
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$R"
: > "$D/done"
REMOTE_ROCM

cat > "$OUT/remote_rocm_check.sh" <<'REMOTE_ROCM_CHECK'
#!/bin/sh
# Written by tools/vultr_leg.sh. The box is ready for the body only when ROCm
# @ROCMWANT@ is installed, /dev/kfd exists, rocm-smi answers and rocminfo
# lists @GPUS@ gfx942 agents.
set -u
D=/root/vultr_rocm
mkdir -p "$D"
C="$D/check.txt"
rocm-smi --showproductname --showdriverversion > "$D/rocm_smi.txt" 2>&1
smi=$?
rocminfo > "$D/rocminfo.txt" 2>&1
agents=$(grep -c '^ *Name: *gfx942 *$' "$D/rocminfo.txt")
ver=$(cat /opt/rocm/.info/version 2>/dev/null)
{
  echo "checked=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "boot_id=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null)"
  echo "rocm_version=$ver"
  echo "amdgpu_loaded=$(cat /sys/module/amdgpu/version 2>/dev/null)"
  echo "kfd=$( [ -e /dev/kfd ] && echo present || echo absent)"
  echo "rocm_smi_exit=$smi"
  echo "gfx942_agents=$agents"
  echo "gfx942_agents_wanted=@GPUS@"
} > "$C"
ok=1
case "$ver" in @ROCMWANT@*) ;; *) ok=0 ;; esac
[ -e /dev/kfd ] || ok=0
[ "$smi" = 0 ] || ok=0
[ "$agents" = "@GPUS@" ] || ok=0
cat "$C"
if [ "$ok" = 1 ]; then echo ROCM_READY; else echo ROCM_NOT_READY; fi
REMOTE_ROCM_CHECK

if [ "$DRY" = 1 ]; then
  SUBST_ID=DRYRUN_ID SUBST_SECS=$((MINUTES * 60)) SUBST_DEADLINE=$(( $(date +%s) + MINUTES * 60 )) WORK_SECONDS=$((MINUTES * 60 - FETCH_RESERVE))
fi
for _f in box_deadman.sh remote_start.sh remote_unpack.sh; do
  cp "$OUT/$_f" "$TMPD/$_f.template"
done
if SUBST_ID="${SUBST_ID:-0}" SUBST_DEADLINE="${SUBST_DEADLINE:-0}" subst "$OUT/box_deadman.sh" \
   && subst "$OUT/remote_start.sh" && subst "$OUT/remote_unpack.sh" \
   && subst "$OUT/remote_rocm_install.sh" && subst "$OUT/remote_rocm_check.sh"; then
  _ok=1
  for _f in box_deadman.sh remote_start.sh remote_unpack.sh remote_rocm_install.sh remote_rocm_check.sh; do
    check_posix "$OUT/$_f" "$_f" || _ok=0
  done
  [ "$_ok" = 1 ] && rok "the on-box dead-man, start wrapper, unpack script and ROCm install and check substitute cleanly and pass sh -n, dash -n and the bashism scan"
else
  rbad "UNSUBSTITUTED PLACEHOLDER in a box script"
fi
if write_local_deadman "$TMPD/deadman-compose" $((MINUTES * 60)) && sh -n "$TMPD/deadman-compose/deadman.sh"; then
  rok "the local dead-man composes, substitutes cleanly and passes sh -n"
else
  rbad "the local dead-man does not compose"
fi
rm -rf "$TMPD/deadman-compose"

{
  echo "commit=$COMMIT_LINE"
  echo "commit_sha=$COMMIT"
  echo "provider=vultr"
  echo "vendor=$BODY_VENDOR"
  echo "name=$NAME"
  echo "tag=$TAG"
  echo "plan=$PLAN"
  echo "ubuntu=$UBUNTU"
  echo "rocm_wanted=$ROCM_WANT"
  echo "gpus_wanted=$GPUS"
  echo "minutes=$MINUTES"
  echo "gates=$GATES"
  echo "gpu_archs=$GPU_ARCHS"
  echo "target_column=$COLUMN"
  echo "extra=$LEG_EXTRA"
  echo "extra_sha256=$EXTRA_SHA"
  echo "bundle_bytes=$BUNDLE_BYTES"
  echo "bundle_sha256=$BUNDLE_SHA"
  echo "bundle_excludes=${ARCHIVE_EXCLUDES[*]}"
  echo "source_sha256_local=$(cat "$OUT/source_sha256_local.txt" 2>/dev/null)"
  echo "mode=$( [ "$DRY" = 1 ] && echo dry || echo rent )"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUT/leg.txt"
printf '%s\n' "$COMMIT" > "$OUT/commit.txt"

# ------------------------------------------------------------------ dry run
if [ "$DRY" = 1 ]; then
  echo
  echo "== the remote body (/root/gemm_leg.sh) =="
  cat "$OUT/remote_body.sh"
  echo
  echo "== the ROCm install (/root/vultr_rocm.sh) =="
  cat "$OUT/remote_rocm_install.sh"
  echo
  echo "== the on-box dead-man (/root/mojolearn-selfkill.sh; the id and deadline are filled at arm time) =="
  cat "$OUT/box_deadman.sh"
  echo
  echo "== what a real leg would do, in order =="
  echo "   1. refuse a dirty tree, a missing or empty token file, a broken script or an oversized bundle"
  echo "   2. GET $API/bare-metals: refuse while any GPU bare metal, or one tagged $TAG or"
  echo "      labelled $NAME, exists, naming each; take $GPU_LOCK; list again"
  echo "   3. GET /plans-metal: price $PLAN, refuse a lease over --dollar-cap; write lease.txt"
  echo "   4. pick the region: ${REGION_OVERRIDE:-the plan locations in order} whose availability lists $PLAN;"
  echo "      none: 'Vultr has NO STOCK' (the driver walks on)"
  echo "   5. GET /ssh-keys (POST the key if absent), GET /os (Ubuntu $UBUNTU x64), GET /account"
  echo "   6. three uplink probes; ARM THE LOCAL DEAD-MAN (tag $TAG + label $NAME, ${MINUTES}m)"
  echo "   7. POST /bare-metals   [THE BILL STARTS HERE]; adopt by label if unreadable"
  echo "   8. wait for active + main_ip (10 to 30 minutes), then three consecutive ssh successes"
  echo "   9. token to /root/.mojolearn-vultr.curlrc on stdin; arm the on-box dead-man by id and"
  echo "      as a systemd unit; verify process, id and a GET with the token = 200, else DELETE"
  echo "  10. install ROCm $ROCM_WANT (detached, polled), reboot if needed, verify the dead-man again,"
  echo "      check ROCm ($GPUS gfx942 agents), bring the logs home to rocm/"
  echo "  11. scp the bundle ($BUNDLE_BYTES bytes), verify sha256 on the box, extract fresh"
  echo "  12. ship the extra body and the remote body; start it detached under timeout(1)"
  echo "  13. fetch /root/gemm_leg_out -> $REAL_OUT/remote/; DELETE; GET until 404 (teardown.txt)"
  echo
  echo "   dry-run artifacts kept in $OUT"
  if [ "$RED" = 1 ]; then echo "DRY RUN: RED. This script is broken (a FAIL above). Nothing rented."; exit 1; fi
  if [ "$BLOCK" = 1 ]; then echo "DRY RUN: plumbing GREEN, and a real leg is BLOCKED (see BLOCK above). Nothing rented."; exit 3; fi
  echo "DRY RUN: GREEN. Nothing rented. This says the plumbing composes, nothing about the box."
  exit 0
fi

# ------------------------------------------------------ from here it bills
[ "$RED" = 1 ] && die "REFUSING to rent: a local check FAILED above." 1
[ -n "$DIRTY" ] && die "REFUSING to rent against a dirty tree. Commit, then launch from git worktree add --detach." 3
[ "$BUNDLE_BYTES" -gt "$MAX_BUNDLE_BYTES" ] && die "REFUSING to rent: the bundle is $BUNDLE_BYTES bytes (cap $MAX_BUNDLE_BYTES)." 3

TOK=""
IFS= read -r TOK < "$TOKFILE" || [ -n "$TOK" ] || die "REFUSING to rent: the token file $TOKFILE is empty." 2
TOK="${TOK//[$'\t\r\n ']/}"
[ -n "$TOK" ] || die "REFUSING to rent: the token file $TOKFILE is empty." 2
CURLRC="$TMPD/curlrc"
TOKPAT="$TMPD/token.pattern"
( umask 077
  printf 'header = "Authorization: Bearer %s"\nsilent\nshow-error\n' "$TOK" > "$CURLRC"
  printf '%s\n' "$TOK" > "$TOKPAT" )
unset TOK

redact() {  # <file>: the token, and a default_password, never reach the evidence
  [ -f "$1" ] || return 0
  python3 - "$TOKPAT" "$1" <<'PY'
import json, sys
tok = open(sys.argv[1]).read().strip()
p = sys.argv[2]
data = open(p, encoding="utf-8", errors="replace").read()
try:
    obj = json.loads(data)
    def scrub(o):
        if isinstance(o, dict):
            return {k: ("<redacted>" if k in ("default_password", "password") and v else scrub(v)) for k, v in o.items()}
        if isinstance(o, list):
            return [scrub(v) for v in o]
        return o
    data = json.dumps(scrub(obj), indent=1) + "\n"
except Exception:
    pass
if tok:
    data = data.replace(tok, "<redacted>")
open(p, "w").write(data)
PY
}

uplink_down() {
  local h
  for h in $UPLINK_HOSTS; do
    curl -s -o /dev/null --max-time 8 "$h" 2>/dev/null && return 1
  done
  return 0
}
uplink_stable() {
  local r=1
  while [ "$r" -le 3 ]; do
    if uplink_down; then log "uplink probe $r/3: NO neutral host answered"; return 1; fi
    [ "$r" -lt 3 ] && sleep "$T_UPLINK"
    r=$((r + 1))
  done
  return 0
}

echo
echo "== pre-flight =="
list_live() {  # every bare metal on a GPU plan, or tagged $TAG, or labelled $NAME; HTTP code and 1 when the listing failed
  local c
  c=$(http_code GET "$API/bare-metals?per_page=500" "$TMPD/all_bm.json")
  [ "$c" = 200 ] || { printf 'HTTP %s' "$c"; return 1; }
  python3 - "$TMPD/all_bm.json" "$NAME" "$TAG" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
name, tag = sys.argv[2], sys.argv[3]
for x in d.get("bare_metals", []):
    tags = x.get("tags") or []
    gpu = "gpu" in str(x.get("plan", "")).lower()
    if gpu or x.get("label") == name or tag in tags:
        print("  id=%s label=%s plan=%s region=%s status=%s tags=%s created=%s" % (
            x.get("id"), x.get("label"), x.get("plan"), x.get("region"), x.get("status"),
            ",".join(sorted(tags)), x.get("date_created")))
PY
}
refuse_live() {
  printf '%s\n' "$LIVE" > "$OUT/refused_live_bare_metal.txt"
  die "REFUSING to create $NAME: ONE Vultr GPU bare metal at a time on this account, and these already exist:
$LIVE
  Each is either a live leg or an orphan. Find out which, delete or wait, then re-run." 3
}
LIVE="$(list_live)" || die "REFUSING to rent: GET /bare-metals returned $LIVE (token or network). Nothing was created." 2
[ -z "$LIVE" ] || refuse_live

if ! take_lock; then
  _age=$(lock_age) || _age=""
  _owner=$(tr '\n' ' ' < "$GPU_LOCK/owner" 2>/dev/null)
  if [ -n "$_age" ] && [ "$_age" -gt "$LOCK_STALE_SECONDS" ]; then
    log "breaking a STALE Vultr GPU lock (${_age}s old, no GPU bare metal live): ${_owner:-no owner file}"
    echo "gpu_lock_broken_stale=age ${_age}s owner ${_owner:-none}" >> "$OUT/leg.txt"
    rm -rf "$GPU_LOCK"
    take_lock || die "REFUSING to create $NAME: another leg took $GPU_LOCK while the stale one was being broken. Nothing was created." 3
  else
    die "REFUSING to create $NAME: the Vultr GPU lock $GPU_LOCK is held (${_age:-?}s old) by: ${_owner:-no owner file}. One Vultr GPU bare metal at a time across every session. Nothing was created." 3
  fi
fi
echo "gpu_lock=taken $(date -u +%Y-%m-%dT%H:%M:%SZ) $GPU_LOCK lane=$LOCK_LANE" >> "$OUT/leg.txt"
log "Vultr GPU lock taken ($GPU_LOCK, lane $LOCK_LANE)"
LIVE="$(list_live)" || die "REFUSING to rent: GET /bare-metals returned $LIVE under the lock. Nothing was created." 2
[ -z "$LIVE" ] || refuse_live
log "API reachable, no GPU or mojolearn bare metal live"

# ---- the price, read live, and the cap, BEFORE anything is armed ----
c=$(http_code GET "$API/plans-metal?per_page=500" "$TMPD/plans.json")
[ "$c" = 200 ] || die "REFUSING to rent: GET /plans-metal answered HTTP $c; the lease cannot be priced. Nothing was created." 2
read -r PRICE_HOURLY PRICE_SOURCE PLAN_LOCATIONS < <(python3 - "$TMPD/plans.json" "$PLAN" "$MONTH_HOURS" <<'PY'
import json, sys
try:
    for p in json.load(open(sys.argv[1])).get("plans_metal", []):
        if p.get("id") == sys.argv[2]:
            locs = ",".join(p.get("locations") or []) or "-"
            if p.get("hourly_cost") not in (None, ""):
                print(float(p["hourly_cost"]), "hourly_cost", locs)
            elif p.get("monthly_cost") not in (None, ""):
                print(round(float(p["monthly_cost"]) / float(sys.argv[3]), 4), "monthly_cost/%s" % sys.argv[3], locs)
            break
except Exception:
    pass
PY
)
[ -n "${PRICE_HOURLY:-}" ] || die "REFUSING to rent: plan $PLAN is not in GET /plans-metal (or has no price); nothing was created." 2
MAX_COST=$(python3 -c "import sys; print('%.2f' % (float(sys.argv[1]) * int(sys.argv[2]) / 60.0))" "$PRICE_HOURLY" "$MINUTES")
{
  echo "provider=vultr"
  echo "plan=$PLAN"
  echo "minutes=$MINUTES"
  echo "price_hourly=$PRICE_HOURLY"
  echo "price_source=$PRICE_SOURCE"
  echo "max_cost=$MAX_COST"
  echo "dollar_cap=${DOLLAR_CAP:-<none: a lease of one hour or less>}"
  echo "plan_locations=$PLAN_LOCATIONS"
} > "$OUT/lease.txt"
if [ -n "$DOLLAR_CAP" ]; then
  if [ "$(python3 -c "import sys; print(1 if float(sys.argv[1]) > float(sys.argv[2]) else 0)" "$MAX_COST" "$DOLLAR_CAP")" = 1 ]; then
    echo "verdict=REFUSED_OVER_CAP" >> "$OUT/lease.txt"
    die "segment lease REFUSED: $MINUTES minutes of $PLAN at \$$PRICE_HOURLY/h ($PRICE_SOURCE) is up to \$$MAX_COST, above the --dollar-cap of \$$DOLLAR_CAP; nothing was created" 2
  fi
  log "segment lease: $MINUTES minutes of $PLAN at \$$PRICE_HOURLY/h is at most \$$MAX_COST, under the cap of \$$DOLLAR_CAP"
  { echo "segment_lease=$MINUTES"; echo "dollar_cap=$DOLLAR_CAP"; } >> "$OUT/leg.txt"
else
  log "lease: $MINUTES minutes of $PLAN at \$$PRICE_HOURLY/h is at most \$$MAX_COST"
fi
{ echo "price_hourly=$PRICE_HOURLY"; echo "price_source=$PRICE_SOURCE"; echo "max_cost=$MAX_COST"; } >> "$OUT/leg.txt"

# ---- the region: the first location whose availability lists the plan ----
if [ -n "$REGION_OVERRIDE" ]; then
  CANDIDATES="$REGION_OVERRIDE"
  case ",$PLAN_LOCATIONS," in *",$REGION_OVERRIDE,"*) ;; *)
    echo "verdict=NO_STOCK region $REGION_OVERRIDE not in the plan's locations" >> "$OUT/lease.txt"
    die "REFUSING to create $NAME: Vultr has NO STOCK of $PLAN in region $REGION_OVERRIDE (the plan's locations are $PLAN_LOCATIONS). Nothing was created." 3 ;;
  esac
else
  CANDIDATES="${PLAN_LOCATIONS//,/ }"
fi
: > "$OUT/availability.txt"
for _r in $CANDIDATES; do
  case "$_r" in *[!a-z0-9-]*|-) continue ;; esac
  c=$(http_code GET "$API/regions/$_r/availability" "$TMPD/avail.json")
  _has=$(python3 - "$TMPD/avail.json" "$PLAN" <<'PY'
import json, sys
try:
    print(1 if sys.argv[2] in (json.load(open(sys.argv[1])).get("available_plans") or []) else 0)
except Exception:
    print(0)
PY
)
  echo "region=$_r http=$c available=$_has" >> "$OUT/availability.txt"
  if [ "$c" = 200 ] && [ "$_has" = 1 ]; then REGION=$_r; break; fi
done
if [ -z "$REGION" ]; then
  echo "verdict=NO_STOCK" >> "$OUT/lease.txt"
  die "REFUSING to create $NAME: Vultr has NO STOCK of $PLAN in any of: $CANDIDATES (see availability.txt). Nothing was created." 3
fi
echo "region=$REGION" >> "$OUT/lease.txt"
log "region $REGION lists $PLAN as available"

# ---- the ssh key, the OS, the account ----
SSH_PUB="$(awk 'NF>=2 {print $1" "$2; exit}' "$SSH_PUBKEY_FILE")"
c=$(http_code GET "$API/ssh-keys?per_page=500" "$TMPD/keys.json")
[ "$c" = 200 ] || die "REFUSING to rent: GET /ssh-keys answered HTTP $c. Nothing was created." 2
SSHKEY_ID=$(python3 - "$TMPD/keys.json" "$SSH_PUB" <<'PY'
import json, sys
want = sys.argv[2].split()[:2]
for k in json.load(open(sys.argv[1])).get("ssh_keys", []):
    if str(k.get("ssh_key", "")).split()[:2] == want:
        print(k.get("id")); break
PY
)
if [ -z "$SSHKEY_ID" ]; then
  python3 -c 'import json,sys; print(json.dumps({"name": "mojolearn-leg", "ssh_key": sys.argv[1]}))' "$SSH_PUB" > "$TMPD/key_req.json"
  c=$(curl -K "$CURLRC" --max-time 30 -o "$TMPD/key_resp.json" -w '%{http_code}' -X POST \
        -H 'Content-Type: application/json' --data-binary "@$TMPD/key_req.json" "$API/ssh-keys" 2>>"$TMPD/curl.err") || c=000
  SSHKEY_ID=$(python3 -c 'import json,sys
try: print(json.load(open(sys.argv[1]))["ssh_key"]["id"])
except Exception: print("")' "$TMPD/key_resp.json")
  [ -n "$SSHKEY_ID" ] || die "REFUSING to rent: POST /ssh-keys answered HTTP $c and no key id. Nothing was created." 2
  log "registered $SSH_PUBKEY_FILE as Vultr ssh key $SSHKEY_ID"
fi
case "$SSHKEY_ID" in *[!A-Za-z0-9-]*) die "the ssh key id '$SSHKEY_ID' is not an id" 2 ;; esac
echo "sshkey_id=$SSHKEY_ID" >> "$OUT/leg.txt"

c=$(http_code GET "$API/os?per_page=500" "$TMPD/os.json")
[ "$c" = 200 ] || die "REFUSING to rent: GET /os answered HTTP $c. Nothing was created." 2
OS_ID=$(python3 - "$TMPD/os.json" "$UBUNTU" <<'PY'
import json, sys
for o in json.load(open(sys.argv[1])).get("os", []):
    n = str(o.get("name", ""))
    if str(o.get("family", "")).lower() == "ubuntu" and sys.argv[2] in n and "x64" in n and o.get("arch", "x64") == "x64":
        print(o.get("id")); break
PY
)
case "$OS_ID" in ''|*[!0-9]*) die "REFUSING to rent: GET /os lists no Ubuntu $UBUNTU x64. Nothing was created." 2 ;; esac
echo "os_id=$OS_ID" >> "$OUT/leg.txt"

c=$(http_code GET "$API/account" "$TMPD/account.json")
python3 - "$TMPD/account.json" "$c" >> "$OUT/leg.txt" <<'PY'
import json, sys
try:
    a = json.load(open(sys.argv[1])).get("account", {})
    print("account_http=%s account_balance=%s account_pending_charges=%s" % (sys.argv[2], a.get("balance"), a.get("pending_charges")))
except Exception:
    print("account_http=%s account=unreadable" % sys.argv[2])
PY

uplink_stable || die "REFUSING to create $NAME: this machine could not reach ANY neutral host. Nothing was created." 2
log "uplink up on all three probes"

# ---- the local dead-man, BEFORE the create ----
LEG_START=$(date +%s)
DEADMAN_SECONDS=$((MINUTES * 60))
DEADLINE_EPOCH=$((LEG_START + DEADMAN_SECONDS))
{ echo "lease_start=$(utc_of "$LEG_START")"; echo "deadline=$(utc_of "$DEADLINE_EPOCH")"; } >> "$OUT/lease.txt"
DEADMAN_DIR="${TMPDIR:-/tmp}/mojolearn-vultr-deadman-$$"
DEADMAN_DIR="${DEADMAN_DIR//\/\//\/}"
write_local_deadman "$DEADMAN_DIR" "$DEADMAN_SECONDS" || die "THE LOCAL DEAD-MAN DID NOT BUILD. Nothing was created." 1
( umask 077; cp "$CURLRC" "$DEADMAN_DIR/curlrc" )
nohup sh -c 'trap "" HUP INT; exec sh "$0"' "$DEADMAN_DIR/deadman.sh" > /dev/null 2>&1 < /dev/null &
DEADMAN_PID=$!
sleep 1
if ! kill -0 "$DEADMAN_PID" 2>/dev/null; then
  rm -rf "$DEADMAN_DIR"; DEADMAN_PID=""; DEADMAN_DIR=""
  die "THE LOCAL DEAD-MAN DID NOT START. Nothing was created." 1
fi
{
  echo "local_deadman_pid=$DEADMAN_PID"
  echo "local_deadman_dir=$DEADMAN_DIR"
  echo "local_deadman_seconds=$DEADMAN_SECONDS"
  echo "local_deadman_fires_at=$(utc_of "$DEADLINE_EPOCH")"
  echo "local_deadman_keyed_by=tag $TAG + label $NAME (+ id once known)"
} > "$OUT/deadman.txt"
log "local dead-man ARMED before the create: pid $DEADMAN_PID, fires at $(utc_of "$DEADLINE_EPOCH")"
if ps -axo command= 2>/dev/null | grep -q -F -f "$TOKPAT"; then
  KEY_RED=1; log "!! THE TOKEN IS VISIBLE IN THIS MACHINE'S PROCESS LIST. Rotate it after this leg."
  echo "local_key_in_ps=VISIBLE" >> "$OUT/leg.txt"
else
  echo "local_key_in_ps=not_visible" >> "$OUT/leg.txt"
fi

# ---- the create ----
python3 - "$REGION" "$PLAN" "$OS_ID" "$NAME" "$TAG" "$SSHKEY_ID" > "$OUT/create_request.json" <<'PY'
import json, sys
region, plan, os_id, name, tag, key = sys.argv[1:7]
print(json.dumps({"region": region, "plan": plan, "os_id": int(os_id), "label": name, "hostname": name,
                  "tags": [tag], "sshkey_id": [key], "activation_email": False, "enable_ipv6": False}, indent=1))
PY
echo
echo "== the bare metal =="
log "creating $NAME ($PLAN, $REGION, os $OS_ID)"
CREATE_ATTEMPTED=1
c=$(curl -K "$CURLRC" --max-time 90 -o "$OUT/create_response.json" -w '%{http_code}' \
      -X POST -H 'Content-Type: application/json' --data-binary "@$OUT/create_request.json" \
      "$API/bare-metals" 2>>"$TMPD/curl.err") || c=000
redact "$OUT/create_response.json"
echo "create_http=$c" >> "$OUT/leg.txt"
BM_ID=$(python3 - "$OUT/create_response.json" <<'PY'
import json, sys
try:
    print(json.load(open(sys.argv[1]))["bare_metal"]["id"])
except Exception:
    print("")
PY
)
if [ -z "$BM_ID" ]; then
  _err=$(python3 - "$OUT/create_response.json" <<'PY'
import json, sys
try:
    print(str(json.load(open(sys.argv[1])).get("error", ""))[:300])
except Exception:
    print(open(sys.argv[1], errors="replace").read()[:300].replace("\n", " "))
PY
)
  log "create returned no id (HTTP $c): $_err"
  sleep 5
  BM_ID=$(ids_by_name | awk '{print $1}')
  if [ -n "$BM_ID" ]; then
    log "ADOPTED bare metal $BM_ID found by label after an unreadable create response"
    echo "adopted_by_name=1" >> "$OUT/leg.txt"
  else
    case "$c" in
      4*)
        if printf '%s' "$_err" | grep -qiE 'limit|stock|capacity|unavailable|not available|quota|exceed|insufficient|balance|out of'; then
          echo "verdict=CREATE_REFUSED $_err" >> "$OUT/lease.txt"
          die "Vultr REFUSED the create on account limits or stock (HTTP $c): $_err. The teardown sweeps by label once more." 4
        fi ;;
    esac
    die "Vultr create FAILED (HTTP $c): $_err. No bare metal by that label either; the teardown sweeps by label once more." 4
  fi
fi
case "$BM_ID" in *[!A-Za-z0-9-]*) die "the bare metal id '$BM_ID' is not an id" 4 ;; esac
printf '%s\n' "$BM_ID" > "$OUT/bare_metal_id.txt"
printf '%s\n' "$BM_ID" > "$DEADMAN_DIR/bare_metal_id.txt"
echo "bare_metal=$BM_ID" >> "$OUT/leg.txt"
echo "bare_metal_id=$BM_ID" >> "$OUT/lease.txt"
log "bare metal id $BM_ID"

# ---- provisioning: 10 to 30 minutes on bare metal ----
: > "$OUT/status_poll.txt"
_prov_until=$(( $(date +%s) + PROVISION_MAX_SECONDS ))
while :; do
  [ "$(date +%s)" -lt $((DEADLINE_EPOCH - FETCH_RESERVE)) ] || break
  [ "$(date +%s)" -lt "$_prov_until" ] || break
  _hc=$(http_code GET "$API/bare-metals/$BM_ID" "$TMPD/bm.json")
  read -r _status IP < <(python3 - "$TMPD/bm.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))["bare_metal"]
    ip = d.get("main_ip") or ""
    print(d.get("status") or "unknown", "" if ip in ("", "0.0.0.0") else ip)
except Exception:
    print("unknown", "")
PY
)
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) http=$_hc status=$_status main_ip=${IP:-none}" >> "$OUT/status_poll.txt"
  [ "$_status" = active ] && [ -n "$IP" ] && break
  IP=""
  sleep "$T_PROV"
done
[ -n "$IP" ] || die "the bare metal never became active with a main_ip (status_poll.txt)" 5
case "$IP" in *[!0-9.]*) die "main_ip '$IP' is not an IPv4 address" 5 ;; esac
log "active at $IP after $(( $(date +%s) - LEG_START ))s"
echo "ip=$IP" >> "$OUT/leg.txt"

SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o "UserKnownHostsFile=$TMPD/known_hosts"
          -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3)
SSH=("$SSH_BIN" "${SSH_OPTS[@]}" "root@$IP")
SSHN=("$SSH_BIN" -n "${SSH_OPTS[@]}" "root@$IP")

wait_ssh() {  # three CONSECUTIVE successes: a fresh box answers once and resets while sshd settles
  local ok=0 i
  for i in $(seq 1 120); do
    if "${SSHN[@]}" true 2>/dev/null; then
      ok=$((ok + 1)); [ "$ok" -ge 3 ] && return 0
    else
      ok=0
    fi
    sleep "$T_SSH"
  done
  return 1
}
wait_ssh || die "ssh never settled on $IP" 5
log "ssh settled"

# ---- the on-box dead-man ----
SUBST_ID=$BM_ID
cp "$TMPD/box_deadman.sh.template" "$OUT/box_deadman.sh"
SUBST_ID=$SUBST_ID SUBST_DEADLINE=$DEADLINE_EPOCH subst "$OUT/box_deadman.sh" || die "the on-box dead-man did not substitute" 1
"${SSH[@]}" 'umask 077; cat > /root/.mojolearn-vultr.curlrc; chmod 600 /root/.mojolearn-vultr.curlrc' < "$CURLRC" \
  || die "could not deliver the token for the on-box dead-man; deleting unused" 6
"${SSH[@]}" 'umask 077; cat > /root/mojolearn-selfkill.sh; chmod 700 /root/mojolearn-selfkill.sh' < "$OUT/box_deadman.sh" \
  || die "could not deliver the on-box dead-man; deleting unused" 6

verify_box_deadman() {  # <label>: start it if absent, then prove process, id and token GET
  "${SSHN[@]}" "if ! pgrep -f 'mojolearn-[s]elfkill.sh' > /dev/null 2>&1; then
  nohup sh /root/mojolearn-selfkill.sh > /root/selfkill.log 2>&1 < /dev/null &
  sleep 1
fi
if pgrep -f 'mojolearn-[s]elfkill.sh' > /dev/null 2>&1; then echo ON_BOX_DEADMAN_ARMED; else echo ON_BOX_DEADMAN_FAILED; fi
echo ID_BAKED_IN=\$(grep -c 'bare-metals/$BM_ID' /root/mojolearn-selfkill.sh)
echo DEADLINE_BAKED_IN=\$(grep -c '$DEADLINE_EPOCH' /root/mojolearn-selfkill.sh)
echo TOKEN_GET_HTTP=\$(curl -K /root/.mojolearn-vultr.curlrc --max-time 20 -o /dev/null -w '%{http_code}' '$API/bare-metals/$BM_ID')" \
    > "$TMPD/arm.out" 2>&1
  sed 's/^/    /' "$TMPD/arm.out"
  grep -E '^(ON_BOX_DEADMAN_|ID_BAKED_IN=|DEADLINE_BAKED_IN=|TOKEN_GET_HTTP=)' "$TMPD/arm.out" | sed "s/^/on_box_${1}_/" >> "$OUT/deadman.txt"
  grep -q '^ON_BOX_DEADMAN_ARMED' "$TMPD/arm.out" \
    && grep -q '^ID_BAKED_IN=[1-9]' "$TMPD/arm.out" \
    && grep -q '^DEADLINE_BAKED_IN=[1-9]' "$TMPD/arm.out" \
    && grep -q '^TOKEN_GET_HTTP=200' "$TMPD/arm.out"
}
# The systemd unit re-arms it after the ROCm reboot (it sleeps to the same deadline).
"${SSHN[@]}" 'if command -v systemctl > /dev/null 2>&1; then
  mkdir -p /etc/systemd/system
  printf "[Unit]\nDescription=mojolearn Vultr lease dead-man\nAfter=network-online.target\nWants=network-online.target\n\n[Service]\nType=simple\nExecStart=/bin/sh /root/mojolearn-selfkill.sh\nRestart=on-failure\n\n[Install]\nWantedBy=multi-user.target\n" > /etc/systemd/system/mojolearn-selfkill.service
  systemctl daemon-reload && systemctl enable mojolearn-selfkill.service > /dev/null 2>&1 && echo REBOOT_REARM=systemd_enabled || echo REBOOT_REARM=systemd_FAILED
else
  echo REBOOT_REARM=NO_SYSTEMD
fi' > "$TMPD/unit.out" 2>&1
sed 's/^/    /' "$TMPD/unit.out"
grep '^REBOOT_REARM=' "$TMPD/unit.out" | sed 's/^/on_box_/' >> "$OUT/deadman.txt"
{
  echo "on_box_deadman_deadline=$DEADLINE_EPOCH"
  echo "on_box_deadman_fires_at=$(utc_of "$DEADLINE_EPOCH")"
} >> "$OUT/deadman.txt"
verify_box_deadman armed || die "THE ON-BOX DEAD-MAN COULD NOT BE VERIFIED (process, id, deadline, or token GET). A box that cannot guard itself is an orphan that has not happened yet. Deleting it unused." 6
log "on-box dead-man ARMED and verified (id $BM_ID, fires at $(utc_of "$DEADLINE_EPOCH"), token GET 200)"

"${SSH[@]}" 'umask 077; cat > /root/.mojolearn-vultr.key' < "$TOKPAT"
"${SSHN[@]}" 'ps -eo args= > /root/.mojolearn-ps.txt 2>/dev/null || ps ax > /root/.mojolearn-ps.txt
if grep -q -F -f /root/.mojolearn-vultr.key /root/.mojolearn-ps.txt; then echo KEY_VISIBLE_IN_PS; else echo KEY_NOT_IN_PS; fi
rm -f /root/.mojolearn-ps.txt /root/.mojolearn-vultr.key' > "$TMPD/ps.out" 2>&1
sed 's/^/    /' "$TMPD/ps.out"
if grep -q KEY_NOT_IN_PS "$TMPD/ps.out"; then
  echo "box_key_in_ps=not_visible" >> "$OUT/leg.txt"
else
  KEY_RED=1
  echo "box_key_in_ps=$(tr '\n' ' ' < "$TMPD/ps.out")" >> "$OUT/leg.txt"
  log "!! THE TOKEN IS VISIBLE (or unverifiable) IN THE BOX'S PROCESS LIST. Rotate it after this leg."
fi

with_deadline() {  # <seconds> <cmd...>: the command's status, or 124 at the deadline
  local secs=$1 pid waited=0
  shift
  "$@" &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$waited" -ge "$secs" ]; then
      pkill -P "$pid" 2>/dev/null; kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
      return 124
    fi
    sleep 1
    waited=$((waited + 1))
  done
  wait "$pid"
}

fetch_rocm_logs() {  # the install and check logs come home, whatever happened
  mkdir -p "$OUT/rocm"
  with_deadline 120 "${SSHN[@]}" 'cd /root/vultr_rocm 2>/dev/null && tar czf - --exclude=./amdgpu-install.deb .' > "$TMPD/rocm.tgz" 2>/dev/null \
    && tar xzf "$TMPD/rocm.tgz" -C "$OUT/rocm" 2>/dev/null
}
boot_id() { "${SSHN[@]}" 'cat /proc/sys/kernel/random/boot_id 2>/dev/null' 2>/dev/null | tr -d '\r\n '; }

# ---- ROCm, before the body ----
echo
echo "== ROCm $ROCM_WANT =="
"${SSH[@]}" 'umask 022; cat > /root/vultr_rocm.sh' < "$OUT/remote_rocm_install.sh" || die "could not ship the ROCm install" 6
"${SSH[@]}" 'umask 022; cat > /root/vultr_rocm_check.sh' < "$OUT/remote_rocm_check.sh" || die "could not ship the ROCm check" 6
"${SSHN[@]}" 'mkdir -p /root/vultr_rocm; rm -f /root/vultr_rocm/done; nohup sh /root/vultr_rocm.sh > /root/vultr_rocm/console.log 2>&1 < /dev/null & echo ROCM_PID=$!' > "$TMPD/rocm_start.out" 2>&1
grep -q '^ROCM_PID=[0-9]' "$TMPD/rocm_start.out" || die "THE ROCm INSTALL DID NOT START: $(tr '\n' ' ' < "$TMPD/rocm_start.out")" 6
_rocm_t0=$(date +%s)
_rocm_until=$(( _rocm_t0 + ROCM_MAX_SECONDS ))
[ "$_rocm_until" -lt $((DEADLINE_EPOCH - FETCH_RESERVE)) ] || _rocm_until=$((DEADLINE_EPOCH - FETCH_RESERVE))
log "ROCm install started (detached); polling for its sentinel, bound $(( _rocm_until - _rocm_t0 ))s"
_rocm_done=0
while [ "$(date +%s)" -lt "$_rocm_until" ]; do
  if "${SSHN[@]}" 'test -f /root/vultr_rocm/done' 2>/dev/null; then _rocm_done=1; break; fi
  sleep "$T_WORK"
done
fetch_rocm_logs
echo "rocm_install_seconds=$(( $(date +%s) - _rocm_t0 ))" >> "$OUT/leg.txt"
[ "$_rocm_done" = 1 ] || die "THE ROCm INSTALL DID NOT FINISH inside its bound; its log is in $OUT/rocm/. Deleting the box." 6
sed 's/^/    /' "$OUT/rocm/rocm.txt" 2>/dev/null
grep -E '^(installer_exit|rocm_version|amdgpu_dkms|reboot_needed)=' "$OUT/rocm/rocm.txt" 2>/dev/null | sed 's/^/rocm_/' >> "$OUT/leg.txt"

if grep -q '^reboot_needed=1' "$OUT/rocm/rocm.txt" 2>/dev/null; then
  _b0=$(boot_id)
  log "rebooting into the dkms amdgpu driver (boot_id ${_b0:-unknown})"
  "${SSHN[@]}" 'nohup sh -c "sleep 2; reboot" > /dev/null 2>&1 < /dev/null &' > /dev/null 2>&1
  _rb_until=$(( $(date +%s) + REBOOT_MAX_SECONDS ))
  _b1=""
  while [ "$(date +%s)" -lt "$_rb_until" ]; do
    sleep "$T_REBOOT"
    _b1=$(boot_id)
    [ -n "$_b1" ] && [ "$_b1" != "$_b0" ] && break
    _b1=""
  done
  [ -n "$_b1" ] || die "the box did not come back from the reboot with a new boot_id inside ${REBOOT_MAX_SECONDS}s" 6
  wait_ssh || die "ssh never settled on $IP after the reboot" 6
  echo "rebooted=1 boot_id_before=$_b0 boot_id_after=$_b1" >> "$OUT/leg.txt"
  log "back after the reboot (boot_id $_b1)"
  verify_box_deadman after_reboot || die "THE ON-BOX DEAD-MAN IS NOT ARMED AFTER THE REBOOT and could not be re-armed. Deleting the box." 6
  log "on-box dead-man verified again after the reboot"
else
  echo "rebooted=0" >> "$OUT/leg.txt"
fi

"${SSHN[@]}" 'sh /root/vultr_rocm_check.sh' > "$TMPD/rocm_check.out" 2>&1
sed 's/^/    /' "$TMPD/rocm_check.out"
fetch_rocm_logs
if ! grep -q '^ROCM_READY' "$TMPD/rocm_check.out"; then
  echo "rocm_ready=NO" >> "$OUT/leg.txt"
  die "ROCm $ROCM_WANT IS NOT READY on the box (version, /dev/kfd, rocm-smi or $GPUS gfx942 agents; $OUT/rocm/check.txt). A body without its runtime is not a result. Deleting the box." 6
fi
echo "rocm_ready=yes" >> "$OUT/leg.txt"
log "ROCm $ROCM_WANT ready, $GPUS gfx942 agents"
"${SSHN[@]}" "$SMI_CMD" > "$OUT/device.txt" 2>&1
sed 's/^/    [gpu] /' "$OUT/device.txt"

# ---- the source ----
echo
echo "== the source =="
log "scp the bundle ($BUNDLE_BYTES bytes, sha256 ${BUNDLE_SHA:0:16})"
_scp_ok=0
for _try in 1 2 3 4 5; do
  _left=$((DEADLINE_EPOCH - FETCH_RESERVE - $(date +%s)))
  [ "$_left" -gt 120 ] || break
  if with_deadline "$_left" "$SCP_BIN" -q "${SSH_OPTS[@]}" "$TMPD/src.tgz" "root@$IP:/root/extra_src.tgz"; then
    _scp_ok=1; break
  fi
  log "scp attempt $_try failed; retrying in 15s"
  sleep 15
done
[ "$_scp_ok" = 1 ] || die "the bundle upload failed" 7
log "uploaded after $(( $(date +%s) - LEG_START ))s of lease"
"${SSH[@]}" 'umask 022; cat > /root/gemm_leg_unpack.sh' < "$OUT/remote_unpack.sh"
"${SSHN[@]}" 'sh /root/gemm_leg_unpack.sh' > "$TMPD/unpack.out" 2>&1
sed 's/^/    /' "$TMPD/unpack.out"
grep -q '^ARCHIVE-SHA-OK' "$TMPD/unpack.out" && grep -q '^UNPACKED ' "$TMPD/unpack.out" \
  || die "the box refused or failed to unpack the bundle" 7
# DEVIATION 2704: datasets and corpora from R2, staged before the body runs.
sh tools/stage_from_r2.sh "${SSH_OPTS[*]} root@$IP" > "$OUT/stage.log" 2>&1 || true
log "$(tail -1 "$OUT/stage.log")"
if [ "${MOJOLEARN_BINCACHE:-0}" = 1 ]; then
  sh tools/bincache_leg.sh stage "${SSH_OPTS[*]} root@$IP" "vultr:ubuntu-$UBUNTU" > "$OUT/bincache_stage.log" 2>&1 || true
  log "$(tail -1 "$OUT/bincache_stage.log")"
fi

"${SSH[@]}" 'umask 022; cat > /root/gemm_leg_extra.sh' < "$OUT/extra_body.sh" || die "could not ship the extra body" 7
"${SSH[@]}" 'umask 022; cat > /root/gemm_leg_extra_env.sh' < "$OUT/extra_env.sh" || die "could not ship the extra body environment" 7
"${SSH[@]}" 'umask 022; cat > /root/gemm_leg.sh' < "$OUT/remote_body.sh" || die "could not ship the remote body" 7
log "shipped the extra body ($LEG_EXTRA, sha256 ${EXTRA_SHA:0:16}) and the remote body"

# ---- the work, detached and polled ----
echo
echo "== the work =="
WORK_SECONDS=$((DEADLINE_EPOCH - FETCH_RESERVE - $(date +%s)))
[ "$WORK_SECONDS" -ge 120 ] || die "only ${WORK_SECONDS}s of lease left for the work; not starting it" 8
cp "$TMPD/remote_start.sh.template" "$OUT/remote_start.sh"
subst "$OUT/remote_start.sh" || die "the start wrapper did not substitute" 1
echo "work_seconds=$WORK_SECONDS" >> "$OUT/leg.txt"
"${SSH[@]}" 'umask 022; cat > /root/gemm_leg_start.sh' < "$OUT/remote_start.sh"
"${SSHN[@]}" 'sh /root/gemm_leg_start.sh' > "$OUT/remote_start.log" 2>&1
RPID=$(sed -n 's/^REMOTE_PID=//p' "$OUT/remote_start.log" | tr -d '\r' | tail -1)
case "$RPID" in ''|*[!0-9]*) die "THE PAYLOAD DID NOT START (no pid). Read $OUT/remote_start.log." 8 ;; esac
BODY_STATE=running
log "remote pid $RPID, bound ${WORK_SECONDS}s; polling every ${T_WORK}s"

POLL_DEADLINE=$((DEADLINE_EPOCH - FETCH_RESERVE + 60))
_unreach=0
while :; do
  if [ "$(date +%s)" -ge "$POLL_DEADLINE" ]; then
    log "OUTER POLL DEADLINE reached. Fetching what exists."
    BODY_STATE=partial_deadline; FETCH_RED=1; break
  fi
  _st=$("${SSHN[@]}" "if [ -f /root/gemm_leg.done ]; then echo LEG_DONE; elif kill -0 $RPID 2>/dev/null; then echo LEG_RUNNING; else echo LEG_GONE; fi" 2>/dev/null) || _st=""
  case "$_st" in
    *LEG_DONE*)
      "${SSHN[@]}" "for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 $RPID 2>/dev/null || break; sleep 1; done" 2>/dev/null
      log "the body finished (sentinel on the box)"
      BODY_STATE="done"; break ;;
    *LEG_RUNNING*) _unreach=0 ;;
    *LEG_GONE*)
      log "THE BODY PROCESS IS GONE AND WROTE NO SENTINEL. Partial run; fetching it as one."
      BODY_STATE=partial_died; FETCH_RED=1; break ;;
    *)
      _unreach=$((_unreach + 1))
      [ "$_unreach" = 1 ] && log "poll: the box did not answer (the body is detached; retrying)"
      if [ "$_unreach" = 3 ] || [ "$_unreach" = 30 ] || [ "$_unreach" = 60 ]; then
        if uplink_down; then
          log "THIS MACHINE HAS NO UPLINK. The silence is here, not on the box; the on-box dead-man still ends it."
          echo "uplink_fault=1" >> "$OUT/leg.txt"
        else
          log "neutral hosts answer, so the BOX is the silent end"
        fi
      fi ;;
  esac
  sleep "$T_WORK"
done
echo "body=$BODY_STATE" >> "$OUT/leg.txt"

# ---- the fetch, bounded, and the DELETE runs whatever happens here ----
echo
echo "== fetch =="
_left=$((DEADLINE_EPOCH - $(date +%s) - 120))
FETCH_SECONDS=600
[ "$_left" -lt "$FETCH_SECONDS" ] && FETCH_SECONDS=$_left
[ "$FETCH_SECONDS" -ge 60 ] || FETCH_SECONDS=60
mkdir -p "$OUT/remote"
if with_deadline "$FETCH_SECONDS" "${SSHN[@]}" 'cd /root/gemm_leg_out && tar czf - --exclude=./tools-venv .' > "$TMPD/remote.tgz" \
   && tar xzf "$TMPD/remote.tgz" -C "$OUT/remote"; then
  log "fetched /root/gemm_leg_out -> $OUT/remote/ ($(wc -c < "$TMPD/remote.tgz" | tr -d ' ') bytes)"
else
  FETCH_RED=1
  log "FETCH FAILED or hit its ${FETCH_SECONDS}s bound; whatever arrived is in $OUT/remote/. Deleting regardless."
fi
rm -rf "$OUT/remote/tools-venv"
if [ "${MOJOLEARN_BINCACHE:-0}" = 1 ] && [ -f "$OUT/remote/bincache/uploads.tsv" ]; then
  sh tools/bincache_leg.sh promote "$OUT/remote/bincache" > "$OUT/bincache_promote.log" 2>&1 || true
  log "$(tail -1 "$OUT/bincache_promote.log")"
fi
with_deadline 60 "${SSHN[@]}" 'cat /root/gemm_leg_console.log' > "$OUT/remote_console.log" 2>/dev/null || true
tail -5 "$OUT/remote_console.log" 2>/dev/null | sed 's/^/    /'

_local_sha=$(cat "$OUT/source_sha256_local.txt" 2>/dev/null)
_remote_sha=$(cat "$OUT/remote/source_sha256.txt" 2>/dev/null)
if [ -n "$_local_sha" ] && [ "$_local_sha" = "$_remote_sha" ]; then
  echo "source_sha256_match=yes" >> "$OUT/leg.txt"
  log "source_sha256 agrees both ends (${_local_sha:0:16})"
else
  echo "source_sha256_match=NO local=$_local_sha remote=$_remote_sha" >> "$OUT/leg.txt"
  FETCH_RED=1
  log "!! source_sha256 does NOT agree: here ${_local_sha:-none}, there ${_remote_sha:-none}"
fi
for _k in pixi_install_exit device_check_exit card_exit extra_exit body_exit; do
  _v=$(sed -n "s/^$_k=//p" "$OUT/remote/leg.txt" 2>/dev/null | tail -1)
  echo "remote_$_k=${_v:-<absent>}" >> "$OUT/leg.txt"
  log "  $_k=${_v:-<absent>}"
done
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/leg.txt"
echo "lease_used_seconds=$(( $(date +%s) - LEG_START ))" >> "$OUT/leg.txt"

log "leg done; deleting (EXIT trap)"
[ "$FETCH_RED" = 1 ] && exit 1
[ "$KEY_RED" = 1 ] && exit 1
exit 0
