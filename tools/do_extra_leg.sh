#!/usr/bin/env bash
# tools/do_extra_leg.sh. ONE GUARDED DIGITALOCEAN LEG THAT RUNS A LANE'S
# EXTRA BODY: create, ship the commit, pixi and the gates, the body, fetch,
# DESTROY.
#
#   MOJOLEARN_DO_TOKEN_FILE=$HOME/.mojolearn_do_token \
#   MOJOLEARN_GPU_ARCHS=gfx942 \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/attention_step_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-amd-mi325x-attention-step \
#   bash tools/do_extra_leg.sh amd [--minutes N] [--dry-run] [--skip-gates]
#
#   amd   gpu-mi325x1-256gb, tor1, image 188571990
#   nv    gpu-h100x1-80gb,   nyc2, image 236925144
#
# --dry-run rents nothing, needs no token and makes no API call. It prints
# the create body, the bundle file list and size, the remote body and the
# on-droplet dead-man, and runs every local check a real leg runs before it
# creates anything. Exit 0 green, 1 this script is broken, 3 the world is
# not ready (dirty tree, oversized bundle).
#
# WHY THIS FILE EXISTS. Lanes write leg bodies for MOJOLEARN_GEMM_LEG_EXTRA
# (tools/attention_step_leg.sh is the working one) and only
# tools/gemm_remote_leg.sh runs them, on RunPod, which has no AMD stock. The
# DigitalOcean scripts either run the identity bootstrap
# (tools/e2_remote_leg.sh) or fixed speed families (tools/do_speed_leg.sh),
# and neither runs a body. Neural IDENTICAL speed tuning moved to the MI325X,
# so this is the missing runner.
#
# THE BODY CONTRACT, MIRRORED FROM tools/gemm_remote_leg.sh --payload gemm
# (leg_body_gemm and leg_ship_and_run), byte for byte where it can be:
#   * the body is a local POSIX sh file, `sh -n` checked here, shipped to
#     /root/gemm_leg_extra.sh and copied to <leg out>/extra_body.sh;
#   * it runs as `sh /root/gemm_leg_extra.sh > /root/gemm_leg_out/extra.log
#     2>&1` with cwd /root/mojolearn, and its exit code lands in
#     /root/gemm_leg_out/leg.txt as extra_exit=;
#   * /root/mojolearn is `git archive` of the pinned commit, no .git;
#   * PATH carries $HOME/.pixi/bin (exported), and `pixi install` of the
#     default environment has already run (pixi_install_exit= in leg.txt);
#   * /root/gemm_leg_out/leg.txt already holds vendor=, commit=, card_full=,
#     trace_dump=, started=, pixi_install_exit=, device_check_exit=,
#     card_exit=, and the dir holds uname.txt, gpu.txt, source_sha256.txt,
#     pixi_env.log, mojo_version.txt, device_check.log, card_driver.log and
#     <vendor>.card;
#   * everything under /root/gemm_leg_out comes home to <leg out>/remote/;
#   * nothing else is prebuilt: no bindings, no tools venv.
#
# WHERE IT DIFFERS FROM RUNPOD, EACH ON PURPOSE:
#   1. MOJOLEARN_GPU_ARCHS IS EXPORTED to the body when set, and REQUIRED for
#      amd. The RunPod gemm body never exports it, so a body there derives the
#      arch from nvidia-smi. The HIP equivalent does not exist in the bodies,
#      and an empty arch makes bindings/build.sh compile for "the build box's
#      device" with no arch read back. One mojo build is one GPU arch; say it.
#   2. MOJOLEARN_TARGET_COLUMN IS EXPORTED (amd or nvidia). Bodies default it
#      to nvidia (tools/attention_step_leg.sh, tools/lm_step_memory_probe.sh),
#      which is right on RunPod and a mislabelled kernel-matrix column on the
#      MI325X for every binding script that reads it (build_trees, build_tsa,
#      build_arima, build_preprocessing).
#   3. vendor=amd or vendor=nvidia and @SMI@ are the RunPod spellings, so a
#      body that greps leg.txt reads the same words. gpu.txt comes from
#      rocm-smi on amd.
#   4. THE GATES ARE OPTIONAL. --skip-gates records device_check_exit=SKIPPED
#      and card_exit=SKIPPED. Default runs them, as RunPod does. The Apple
#      card is NOT generated or diffed here; the AMD card comes home for a
#      later diff.
#   5. THE WHOLE BODY RUNS UNDER timeout(1) at the work bound (the lease minus
#      the fetch reserve), and body_exit= is appended to leg.txt. RunPod's gemm
#      body is bounded only by its poll deadline because RunPod has its own
#      on-pod lease; here the bound is what keeps the fetch alive.
#   6. THE BUNDLE ALSO EXCLUDES mamba/corpus, bench/oracle*,
#      bench/minentropy_oracle.txt and every *.bin (RunPod gemm excludes only
#      bench/results). About 9.7 MB gzipped against 21.8 MB, because this
#      desk's uplink is about 30 KB/s. The archive sha256 is verified on the
#      box before extraction, and source_sha256 is compared both ends.
#   7. THE DIRTY-TREE GATE IS THE WHOLE TREE minus bench/results, not a path
#      list: a body can reach any file, and legs launch from
#      `git worktree add --detach` where a clean tree costs nothing.
#   8. No ROCm-specific setup runs before the body. None is needed:
#      pixi.toml has no per-vendor feature, the default environment carries
#      numpy on linux-64, and the only HIP env the RunPod file sets
#      (MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_*) exists to fit its external
#      RSS guard in campaigns 4 to 6, which this runner does not use. The
#      ROCm image's bare python3 has no numpy; bodies use `pixi run python`.
#
# THE GUARDS ARE COPIED FROM tools/e2_remote_leg.sh AND tools/do_speed_leg.sh,
# NOT FACTORED OUT (do_speed_leg.sh's header says why: one shared file could
# break every lane's teardown at once). DigitalOcean bills until DESTROY, so:
#   * the uplink is probed three times before the create;
#   * ONE GPU DROPLET AT A TIME: the create is refused while any gpu-* droplet,
#     any droplet tagged e2/speed/rel061/extra, or any mojolearn-* droplet
#     exists, and the refusal names them;
#   * a DETACHED LOCAL DEAD-MAN is armed BEFORE the create, keyed by tag AND
#     name, capped at one hour;
#   * a SECOND DEAD-MAN RUNS ON THE DROPLET (the local one dies with this
#     Mac), verified by process, by the id baked in, and by a GET with the
#     token that must return 200; if it cannot be verified the box is
#     destroyed unused;
#   * the create is by tag and name, and an unreadable create response is
#     ADOPTED by name rather than orphaned;
#   * destroy is an EXIT trap, confirmed only by a follow-up GET returning
#     404; the local dead-man is cancelled only after that confirmation.
#
# THE TOKEN. It is read once, by the shell builtin `read`, from a 0600 file
# outside the repository, and written by the builtin `printf` into a 0600
# curl config that every call reads with `curl -K`. It is never exported,
# never in an argv here or on the droplet (tools/e2_remote_leg.sh and
# tools/do_speed_leg.sh both interpolate theirs into a `nohup bash -c`
# string; this file does not), and it reaches the droplet on ssh STDIN. Both
# process lists are then searched for it with `grep -F -f <pattern file>`.
set -uo pipefail

# RUN FROM AN IMMUTABLE SNAPSHOT (tools/gemm_remote_leg.sh DEVIATION 1882):
# bash reads a script lazily by offset, and a leg lasts long enough for
# someone to edit this file. The copy lives outside the tree so the
# dirty-tree gate does not see it.
if [ "${MOJOLEARN_DO_EXTRA_FROZEN:-0}" != 1 ]; then
  _repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 2
  _snapdir="$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-do-extra-snapshot.XXXXXX")" || exit 2
  cp "${BASH_SOURCE[0]}" "$_snapdir/do_extra_leg.sh" || { rm -rf "$_snapdir"; exit 2; }
  chmod 555 "$_snapdir/do_extra_leg.sh"
  trap 'rm -rf "$_snapdir"' EXIT
  MOJOLEARN_DO_EXTRA_FROZEN=1 MOJOLEARN_DO_EXTRA_REPO="$_repo" \
    bash "$_snapdir/do_extra_leg.sh" "$@"
  exit $?
fi

REPO="${MOJOLEARN_DO_EXTRA_REPO:?}"
cd "$REPO" || exit 2
API=https://api.digitalocean.com/v2
SSH_KEY_FP="df:f7:6b:0c:56:da:48:a5:6f:6d:ae:44:af:de:f3:0b"
TAG=extra
# Tags other DigitalOcean legs in this repository create droplets under.
LEG_TAGS="e2 speed rel061 extra"
FETCH_RESERVE="${FETCH_RESERVE:-420}"
MAX_BUNDLE_BYTES="${MOJOLEARN_DO_EXTRA_MAX_BYTES:-15000000}"

usage() {
  sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

VENDOR=""; MINUTES=60; DRY=0; GATES=1
while [ $# -gt 0 ]; do
  case "$1" in
    amd|nv)
      [ -z "$VENDOR" ] || { echo "one vendor per leg" >&2; exit 2; }
      VENDOR=$1 ;;
    --minutes) shift; MINUTES="${1:-}" ;;
    --minutes=*) MINUTES="${1#--minutes=}" ;;
    --dry-run) DRY=1 ;;
    --skip-gates) GATES=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
[ -n "$VENDOR" ] || { usage >&2; exit 2; }
case "$MINUTES" in ''|*[!0-9]*) echo "--minutes must be a whole number" >&2; exit 2 ;; esac
if [ "$MINUTES" -gt 60 ]; then
  echo "--minutes $MINUTES REFUSED: one hour is the hard cap for a rented GPU (a second leg, never an extension)" >&2
  exit 2
fi
[ "$MINUTES" -ge 10 ] || { echo "--minutes must be at least 10" >&2; exit 2; }

case "$VENDOR" in
  amd) NAME=mojolearn-extra-amd; REGION=tor1; SIZE=gpu-mi325x1-256gb; IMAGE=188571990
       BODY_VENDOR=amd;    COLUMN=amd;    GPU_LABEL=amd-mi325x
       SMI_CMD='rocm-smi --showproductname' ;;
  nv)  NAME=mojolearn-extra-nv;  REGION=nyc2; SIZE=gpu-h100x1-80gb;   IMAGE=236925144
       BODY_VENDOR=nvidia; COLUMN=nvidia; GPU_LABEL=nvidia-h100
       SMI_CMD='nvidia-smi --query-gpu=name,driver_version --format=csv,noheader' ;;
esac

GPU_ARCHS="${MOJOLEARN_GPU_ARCHS:-}"
case "$GPU_ARCHS" in *[!A-Za-z0-9_]*)
  echo "MOJOLEARN_GPU_ARCHS='$GPU_ARCHS': exactly one architecture name (one mojo build is one GPU arch)" >&2; exit 2 ;;
esac
if [ "$VENDOR" = amd ] && [ -z "$GPU_ARCHS" ]; then
  echo "MOJOLEARN_GPU_ARCHS is required on amd (the MI325X is gfx942). The bodies derive it from nvidia-smi, which this box does not have." >&2
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

TOKFILE="${MOJOLEARN_DO_TOKEN_FILE:-$HOME/.mojolearn_do_token}"
STAMP="$(date -u +%Y-%m-%d_%H%M%S)"
# THE SHARED GPU LOCK (ENGINEERING_RULES 10). A lock older than 100 minutes
# with no GPU or mojolearn droplet live is an orphan and may be broken.
GPU_LOCK="${MOJOLEARN_DO_GPU_LOCK:-/tmp/mojolearn-do-gpu.lock}"
LOCK_STALE_SECONDS=6000
LOCK_LANE="${MOJOLEARN_DO_LOCK_LANE:-extra:$(basename "$LEG_EXTRA" .sh)}"
case "$LOCK_LANE" in *[!A-Za-z0-9_.,:-]*) echo "MOJOLEARN_DO_LOCK_LANE: letters, digits and _.,:- only" >&2; exit 2 ;; esac
LOCK_NONCE="$$-$STAMP"
LOCK_HELD=0

# Environment for the extra body. The body runs on the droplet and inherits
# nothing from this shell, so a leg's knobs (MOJOLEARN_ATTN_LEG_ARMS and the
# like) ride MOJOLEARN_DO_EXTRA_ENV as space-separated NAME=value words. Names
# must start with MOJOLEARN_ or MODULAR_; values are letters, digits and
# _.,:/=- only, so they cannot carry shell syntax or a secret by accident.
EXTRA_ENV="${MOJOLEARN_DO_EXTRA_ENV:-}"
# Data files the box cannot fetch itself (the NYC TLC CloudFront refuses
# droplets by address, 2026-09-11) ride MOJOLEARN_DO_EXTRA_UPLOAD as
# space-separated ABSOLUTE local paths. Each is uploaded after the bundle to
# /root/gemm_leg_upload/<basename>, its sha256 is compared on the box against
# the one computed here, and the pair lands in leg.txt. Basenames: letters,
# digits and ._- only. Empty (the default) uploads nothing.
EXTRA_UPLOAD="${MOJOLEARN_DO_EXTRA_UPLOAD:-}"
UPLOAD_LIST=""
OUT="${MOJOLEARN_GEMM_LEG_OUT:-bench/results/e1g/${STAMP}-${GPU_LABEL}-extra}"
case "$OUT" in /*) ;; *) OUT="$REPO/$OUT" ;; esac
REAL_OUT="$OUT"
if [ "$DRY" = 1 ]; then
  # A dry run writes nothing under the checkout; its artifacts are kept here.
  OUT="$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-do-extra-dryrun.XXXXXX")" || exit 2
fi

log() { printf '[%s %s/extra] %s\n' "$(date +%T)" "$VENDOR" "$*"; }
die() { printf '\n%s\n' "$1" >&2; exit "${2:-1}"; }
sha256_of() {
  if command -v shasum > /dev/null 2>&1; then shasum -a 256 "$1"; else sha256sum "$1"; fi | awk '{print $1}'
}
utc_of() { date -u -r "$1" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d "@$1" +%Y-%m-%dT%H:%M:%SZ; }

# ------------------------------------------------------------------ state
TMPD=""; CURLRC=""; TOKPAT=""
DROPLET_ID=""; IP=""; CREATE_ATTEMPTED=0; DESTROY_CONFIRMED=0
DEADMAN_PID=""; DEADMAN_DIR=""; DEADLINE_EPOCH=0; LEG_START=0
KEY_RED=0; FETCH_RED=0; BODY_STATE=not_started; SSH=(ssh); SSHN=(ssh -n)

TMPD="$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-do-extra.XXXXXX")" || exit 2
chmod 700 "$TMPD"

# Every line below runs from the EXIT trap, so shellcheck calls it dead.
# shellcheck disable=SC2317
cancel_deadman() {
  [ -n "$DEADMAN_PID" ] || return 0
  # the wrapper AND its sleep child (seven orphan sleeps after the first E2 day)
  pkill -P "$DEADMAN_PID" 2>/dev/null
  kill "$DEADMAN_PID" 2>/dev/null && log "local dead-man cancelled (pid $DEADMAN_PID)"
  [ -n "$DEADMAN_DIR" ] && rm -rf "$DEADMAN_DIR"
  echo "local_deadman=cancelled $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/deadman.txt"
  DEADMAN_PID=""; DEADMAN_DIR=""
}

# THE SHARED GPU LOCK (ENGINEERING_RULES 10). One GPU droplet at a time across
# every session and lane on this Mac. Released only by the EXIT trap after the
# destroy is confirmed, and only when the owner file still carries our nonce.
# shellcheck disable=SC2317
release_lock() {
  [ "$LOCK_HELD" = 1 ] || return 0
  if grep -qx "nonce=$LOCK_NONCE" "$GPU_LOCK/owner" 2>/dev/null; then
    rm -rf "$GPU_LOCK" && log "released the shared GPU lock $GPU_LOCK"
    echo "gpu_lock=released $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/leg.txt"
  else
    log "!! $GPU_LOCK no longer carries this leg's nonce; left in place"
    echo "gpu_lock=NOT_OURS_AT_RELEASE" >> "$OUT/leg.txt"
  fi
  LOCK_HELD=0
}

lock_age() {  # seconds since the lock directory was made; returns 1 when it is absent
  local m
  m=$(stat -f %m "$GPU_LOCK" 2>/dev/null || stat -c %Y "$GPU_LOCK" 2>/dev/null) || return 1
  echo $(( $(date +%s) - m ))
}

take_lock() {  # mkdir is the atomic test-and-set; the owner file names the leg
  mkdir "$GPU_LOCK" 2>/dev/null || return 1
  LOCK_HELD=1
  {
    echo "lane=$LOCK_LANE"
    echo "script=tools/do_extra_leg.sh"
    echo "pid=$$"
    echo "nonce=$LOCK_NONCE"
    echo "droplet_name=$NAME"
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
ids_by_name() {  # prints ids tagged $TAG and named $NAME; returns 1 when the listing failed
  local c
  c=$(http_code GET "$API/droplets?tag_name=$TAG&per_page=200" "$TMPD/byname.json")
  [ "$c" = 200 ] || return 1
  python3 "$TMPD/byname.py" "$TMPD/byname.json" "$NAME"
}

# shellcheck disable=SC2317
destroy_droplet() {
  local ids id i c ok_all=1
  ids="$DROPLET_ID"
  if [ -z "$ids" ]; then
    # No id: sweep by tag and name, so an unreadable create cannot orphan a box.
    if ! ids=$(ids_by_name); then
      log "!! could not list droplets to find $NAME; destruction UNCONFIRMED"
      echo "sweep_by_name=LISTING_FAILED" >> "$OUT/teardown.txt"
      return 1
    fi
    if [ -z "$ids" ]; then
      echo "sweep_by_name=none (listing HTTP 200 shows no droplet tagged $TAG named $NAME)" >> "$OUT/teardown.txt"
      DESTROY_CONFIRMED=1
      return 0
    fi
  fi
  for id in $ids; do
    for i in 1 2 3 4 5 6; do
      c=$(http_code DELETE "$API/droplets/$id" /dev/null)
      log "DELETE droplet $id -> HTTP $c"
      echo "delete $id attempt $i -> HTTP $c" >> "$OUT/teardown.txt"
      case "$c" in 204|404) break ;; esac
      sleep 10
    done
    # DELETE 204 acknowledges an asynchronous request. Only GET 404 proves absence.
    local gone=0
    for i in 1 2 3 4 5 6 7 8; do
      c=$(http_code GET "$API/droplets/$id" /dev/null)
      log "post-destroy GET droplet $id -> HTTP $c"
      echo "verify $id attempt $i -> HTTP $c" >> "$OUT/teardown.txt"
      if [ "$c" = 404 ]; then gone=1; break; fi
      sleep 5
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
    echo "== teardown $(date -u +%Y-%m-%dT%H:%M:%SZ) exit=$rc droplet=${DROPLET_ID:-unknown} ==" >> "$OUT/teardown.txt"
    destroy_droplet
    echo "destroy_confirmed=$DESTROY_CONFIRMED" >> "$OUT/teardown.txt"
  fi
  if [ "$CREATE_ATTEMPTED" = 0 ] || [ "$DESTROY_CONFIRMED" = 1 ]; then
    cancel_deadman
    release_lock
  else
    {
      echo
      echo "  ############################################################"
      echo "  # DROPLET ${DROPLET_ID:-<unknown id> named $NAME} MAY STILL BE BILLING."
      echo "  # The API did not confirm it is gone. BOTH dead-men are LEFT"
      echo "  # ARMED on purpose (local pid ${DEADMAN_PID:-none}, dir ${DEADMAN_DIR:-none};"
      echo "  # the on-droplet one fires by id). Destroy it by hand now:"
      echo "  #   https://cloud.digitalocean.com/droplets"
      echo "  # and only then: kill ${DEADMAN_PID:-<pid>}; rm -rf ${DEADMAN_DIR:-<dir>}"
      [ "$LOCK_HELD" = 1 ] && echo "  # The shared GPU lock stays HELD until then: rm -rf $GPU_LOCK"
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

echo "== do_extra_leg: one DigitalOcean $VENDOR leg running an extra body =="
echo "   mode      $( [ "$DRY" = 1 ] && echo 'DRY RUN (nothing is rented, no API call)' || echo 'RENT' )"
echo "   commit    $COMMIT_LINE"
echo "   droplet   $NAME  size=$SIZE region=$REGION image=$IMAGE tag=$TAG"
echo "   lease     $MINUTES minutes (local and on-droplet dead-men at that deadline)"
echo "   body      $LEG_EXTRA"
echo "   gates     $( [ "$GATES" = 1 ] && echo 'device check + card (RunPod order)' || echo 'SKIPPED (--skip-gates)' )"
echo "   archs     ${GPU_ARCHS:-<unset: the body derives it>}   column $COLUMN"
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

# The extra body's environment, one export per word, shipped as
# /root/gemm_leg_extra_env.sh and kept as <leg out>/extra_env.sh.
{
  echo "# Generated by tools/do_extra_leg.sh from MOJOLEARN_DO_EXTRA_ENV; sourced before the extra body."
  _env_ok=1
  for _w in $EXTRA_ENV; do
    case "$_w" in
      MOJOLEARN_DO_*=*) _env_ok=0; printf '# REFUSED (runner-only name): %s\n' "${_w%%=*}" ;;
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
  rbad "MOJOLEARN_DO_EXTRA_ENV is refused: $(grep '^# REFUSED' "$OUT/extra_env.sh" | tr '\n' ' ')"
fi
_env_ok=1

token_hygiene() {  # prints a reason and returns 1 when the file must not be used
  local perm
  [ -f "$TOKFILE" ] || { echo "token file $TOKFILE does not exist"; return 1; }
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

if [ -n "$EXTRA_UPLOAD" ]; then
  : > "$OUT/upload_files.txt"
  _up_ok=1
  for _u in $EXTRA_UPLOAD; do
    _b=$(basename "$_u")
    case "$_u" in /*) ;; *) rbad "MOJOLEARN_DO_EXTRA_UPLOAD: $_u is not an absolute path"; _up_ok=0; continue ;; esac
    case "$_b" in .*|*[!A-Za-z0-9._-]*) rbad "MOJOLEARN_DO_EXTRA_UPLOAD: basename $_b (letters, digits and ._- only)"; _up_ok=0; continue ;; esac
    if [ ! -f "$_u" ] || [ ! -r "$_u" ]; then rbad "MOJOLEARN_DO_EXTRA_UPLOAD: $_u is not a readable file"; _up_ok=0; continue; fi
    printf '%s %s %s\n' "$(sha256_of "$_u")" "$(wc -c < "$_u" | tr -d ' ')" "$_u" >> "$OUT/upload_files.txt"
    UPLOAD_LIST="$UPLOAD_LIST $_u"
  done
  [ "$_up_ok" = 1 ] && rok "upload files: $(wc -l < "$OUT/upload_files.txt" | tr -d ' '), $(awk '{s+=$2} END {print s}' "$OUT/upload_files.txt") bytes, each sha256-checked on the box after upload"
fi

if _age=$(lock_age); then
  printf '  info   the shared GPU lock %s is HELD now (%ss old) by: %s; a real leg refuses unless it is over %ss old with no GPU droplet live\n' \
    "$GPU_LOCK" "$_age" "$(tr '\n' ' ' < "$GPU_LOCK/owner" 2>/dev/null || echo 'no owner file')" "$LOCK_STALE_SECONDS"
else
  rok "the shared GPU lock $GPU_LOCK is free (a real leg takes it as lane $LOCK_LANE; a dry run never does)"
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
    rblock "the bundle is $BUNDLE_BYTES bytes gzipped, over the $MAX_BUNDLE_BYTES cap (about 30 KB/s uplink)"
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

# The same recipe on both ends, byte for byte (tools/gemm_remote_leg.sh
# leg_source_sha_recipe), computed over the EXTRACTED ARCHIVE.
source_sha_recipe() {
  ( cd "$1" && \
    { find . -name '*.mojo' -not -path './.pixi/*' -not -path './bench/results/*' \
        | LC_ALL=C sort | xargs shasum -a 256 2>/dev/null || \
      find . -name '*.mojo' -not -path './.pixi/*' -not -path './bench/results/*' \
        | LC_ALL=C sort | xargs sha256sum ; } \
    | { shasum -a 256 2>/dev/null || sha256sum ; } | awk '{print $1}' )
}
[ -d "$TMPD/archive" ] && source_sha_recipe "$TMPD/archive" > "$OUT/source_sha256_local.txt"

# ---- the remote body ----
cat > "$OUT/remote_body.sh" <<'REMOTE_BODY'
#!/bin/sh
# Generated by tools/do_extra_leg.sh. RUNS ON THE DIGITALOCEAN DROPLET.
# The same steps, in the same order, as tools/gemm_remote_leg.sh
# leg_body_gemm, so a MOJOLEARN_GEMM_LEG_EXTRA body sees the same world.
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
  echo "provider=digitalocean"
  echo "size=@SIZE@"
  echo "gates=@GATES@"
  echo "gpu_archs=@GPUARCHS@"
  echo "target_column=@COLUMN@"
} > "$OUT/leg.txt"

# DIFFERENCES 1 AND 2 in tools/do_extra_leg.sh's header: the arch and the
# kernel-matrix column are this box's, not a body's nvidia defaults.
MOJOLEARN_GPU_ARCHS="@GPUARCHS@"
if [ -n "$MOJOLEARN_GPU_ARCHS" ]; then export MOJOLEARN_GPU_ARCHS; else unset MOJOLEARN_GPU_ARCHS; fi
MOJOLEARN_TARGET_COLUMN="@COLUMN@"
export MOJOLEARN_TARGET_COLUMN

uname -a > "$OUT/uname.txt" 2>&1
@SMI@ > "$OUT/gpu.txt" 2>&1 || echo "no vendor smi tool answered" >> "$OUT/gpu.txt"

# THE SOURCE HASH, computed BEFORE pixi installs anything, with the same
# recipe the Mac used on the extracted archive.
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
    # GATE 1: the device kernel's own invariance gates, on this silicon.
    tools/with_identical_mode.sh pixi run mojo run -I . \
        gemm/checks/gemm_device_check.mojo > "$OUT/device_check.log" 2>&1
    echo "device_check_exit=$?" >> "$OUT/leg.txt"
    # GATE 2: the card, through tools/gemm_card.sh as on every other column.
    MOJOLEARN_GEMM_CARD_FULL="@CARDFULL@" MOJOLEARN_IDENTITY_TRACE_DUMP="@DUMP@" \
        sh tools/gemm_card.sh device "$OUT/@VENDOR@.card" > "$OUT/card_driver.log" 2>&1
    echo "card_exit=$?" >> "$OUT/leg.txt"
else
    echo "device_check_exit=SKIPPED" >> "$OUT/leg.txt"
    echo "card_exit=SKIPPED" >> "$OUT/leg.txt"
fi

# MOJOLEARN_GEMM_LEG_EXTRA: the lane's own work, after the gates, same box,
# same lease. Bounded by the start wrapper's timeout(1).
if [ -f /root/gemm_leg_extra.sh ]; then
    # MOJOLEARN_DO_EXTRA_ENV, validated on the Mac; a subshell so the knobs
    # reach the extra body and nothing after it.
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
# THE COMPLETION SENTINEL, WRITTEN LAST. The poll loop tells finished,
# running and died-without-finishing apart with it.
: > /root/gemm_leg.done
echo REMOTE_BODY_DONE
REMOTE_BODY

subst() {  # <file>: replace every placeholder, then prove none survived
  sed -e "s|@VENDOR@|$BODY_VENDOR|g" \
      -e "s|@COMMIT@|$COMMIT|g" \
      -e "s|@CARDFULL@|$CARD_FULL|g" \
      -e "s|@DUMP@|$LEG_DUMP|g" \
      -e "s|@SIZE@|$SIZE|g" \
      -e "s|@GATES@|$GATES|g" \
      -e "s|@GPUARCHS@|$GPU_ARCHS|g" \
      -e "s|@COLUMN@|$COLUMN|g" \
      -e "s|@SMI@|$SMI_CMD|g" \
      -e "s|@WORK@|${WORK_SECONDS:-0}|g" \
      -e "s|@SHA@|${BUNDLE_SHA:-none}|g" \
      -e "s|@SECS@|${SUBST_SECS:-0}|g" \
      -e "s|@ID@|${SUBST_ID:-0}|g" \
      -e "s|@API@|$API|g" \
      -e "s|@TAG@|$TAG|g" \
      -e "s|@NAME@|$NAME|g" \
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
  # the bashism scan from tools/gemm_remote_leg.sh leg_check_remote_body
  awk '!/^[[:space:]]*#/ &&
       (/exec -a/ || /\[\[/ || /(^|[;{ \t])local / || /<\(/ ||
        /(^|[;{ \t])function / || /(^|[;{ \t])source / || /echo -e/) {
           print FNR ": " $0
       }' "$f" > "$TMPD/bashisms"
  if [ -s "$TMPD/bashisms" ]; then
    rbad "$what has a BASHISM (the droplet runs dash):"; sed 's/^/           /' "$TMPD/bashisms"; return 1
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
# Written by tools/do_extra_leg.sh. DETACHED ON PURPOSE. Ends the droplet that
# leg created if that leg is no longer here to do it. Keyed by TAG AND NAME,
# plus the id when the leg learned it, because the worst case is a create
# that succeeded and an id nobody parsed. The token is in the 0600 curl
# config beside this file and in no argv.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
sleep @SECS@
L="$D/deadman.log"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) dead-man firing for tag @TAG@ name @NAME@" >> "$L"
ids=""
[ -s "$D/droplet_id.txt" ] && ids="$(cat "$D/droplet_id.txt")"
curl -K "$D/curlrc" --max-time 30 -o "$D/droplets.json" "@API@/droplets?tag_name=@TAG@&per_page=200" >> "$L" 2>&1
ids="$ids $(python3 "$D/byname.py" "$D/droplets.json" "@NAME@")"
for id in $ids; do
    c="$(curl -K "$D/curlrc" --max-time 30 -o /dev/null -w '%{http_code}' -X DELETE "@API@/droplets/$id" 2>> "$L")"
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
print(" ".join(str(x.get("id")) for x in d.get("droplets", []) if x.get("name") == sys.argv[2]))
BYNAME

# ---- the on-droplet dead-man ----
cat > "$OUT/droplet_deadman.sh" <<'DROPLET_DEADMAN'
#!/bin/sh
# Written by tools/do_extra_leg.sh. RUNS ON THE DROPLET, DETACHED. The local
# dead-man dies with the Mac; this one does not. The token is in
# /root/.mojolearn-do.curlrc (0600, delivered on ssh stdin), never in an argv.
set -u
sleep @SECS@
for attempt in 1 2 3; do
    code=$(curl -K /root/.mojolearn-do.curlrc --max-time 30 -o /root/selfkill.body -w '%{http_code}' \
        -X DELETE '@API@/droplets/@ID@')
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) DELETE @ID@ attempt $attempt -> $code" >> /root/selfkill.out
    case "$code" in 2*|404) break ;; esac
    sleep 10
done
DROPLET_DEADMAN

cat > "$OUT/remote_start.sh" <<'REMOTE_START'
#!/bin/sh
# Written by tools/do_extra_leg.sh. Starts the remote body DETACHED under the
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
# Written by tools/do_extra_leg.sh. The integrity check that replaces a
# clone's: the box recomputes the archive's sha256 and refuses a mismatch.
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

if [ "$DRY" = 1 ]; then
  SUBST_ID=DRYRUN_ID SUBST_SECS=$((MINUTES * 60)) WORK_SECONDS=$((MINUTES * 60 - FETCH_RESERVE))
fi
for _f in droplet_deadman.sh remote_start.sh remote_unpack.sh; do
  cp "$OUT/$_f" "$TMPD/$_f.template"
done
if SUBST_ID="${SUBST_ID:-0}" SUBST_SECS="${SUBST_SECS:-60}" subst "$OUT/droplet_deadman.sh" \
   && subst "$OUT/remote_start.sh" && subst "$OUT/remote_unpack.sh"; then
  _ok=1
  for _f in droplet_deadman.sh remote_start.sh remote_unpack.sh; do
    check_posix "$OUT/$_f" "$_f" || _ok=0
  done
  [ "$_ok" = 1 ] && rok "the on-droplet dead-man, start wrapper and unpack script substitute cleanly and pass sh -n, dash -n and the bashism scan"
else
  rbad "UNSUBSTITUTED PLACEHOLDER in a droplet script"
fi
if write_local_deadman "$TMPD/deadman-compose" $((MINUTES * 60)) && sh -n "$TMPD/deadman-compose/deadman.sh"; then
  rok "the local dead-man composes, substitutes cleanly and passes sh -n"
else
  rbad "the local dead-man does not compose"
fi
rm -rf "$TMPD/deadman-compose"

CREATE_JSON="{\"name\":\"$NAME\",\"region\":\"$REGION\",\"size\":\"$SIZE\",\"image\":$IMAGE,\"ssh_keys\":[\"$SSH_KEY_FP\"],\"tags\":[\"$TAG\"]}"
printf '%s\n' "$CREATE_JSON" > "$OUT/create_request.json"
python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$OUT/create_request.json" \
  && rok "the create body is valid JSON" || rbad "the create body is not valid JSON"

{
  echo "commit=$COMMIT_LINE"
  echo "commit_sha=$COMMIT"
  echo "provider=digitalocean"
  echo "vendor=$BODY_VENDOR"
  echo "name=$NAME"
  echo "tag=$TAG"
  echo "size=$SIZE"
  echo "region=$REGION"
  echo "image=$IMAGE"
  echo "minutes=$MINUTES"
  echo "gates=$GATES"
  echo "gpu_archs=${GPU_ARCHS:-<unset>}"
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
  echo "== the create body (POST $API/droplets) =="
  python3 -m json.tool "$OUT/create_request.json"
  echo
  echo "== the bundle: git archive $COMMIT -- . ${ARCHIVE_EXCLUDES[*]} =="
  sed 's/^/   /' "$OUT/bundle_files.txt"
  echo "   ---- files by top-level path ----"
  awk -F/ '{print $1}' "$OUT/bundle_files.txt" | sort | uniq -c | sort -rn | head -40 | sed 's/^/   /'
  echo "   ---- $(wc -l < "$OUT/bundle_files.txt" | tr -d ' ') files, $BUNDLE_BYTES bytes gzipped (cap $MAX_BUNDLE_BYTES), sha256 $BUNDLE_SHA"
  echo
  echo "== the remote body (/root/gemm_leg.sh) =="
  cat "$OUT/remote_body.sh"
  echo
  echo "== the on-droplet dead-man (/root/mojolearn-selfkill.sh; the id and seconds are filled at arm time) =="
  cat "$OUT/droplet_deadman.sh"
  echo
  echo "== what a real leg would do, in order =="
  echo "   1. refuse a dirty tree, a bad token file, a broken script or an oversized bundle"
  echo "   2. GET $API/droplets: refuse while any gpu-* droplet, any droplet tagged"
  echo "      $LEG_TAGS, or any mojolearn-* droplet exists, naming each"
  echo "   3. three uplink probes against neutral hosts; any failure refuses"
  echo "   4. ARM THE LOCAL DEAD-MAN (tag $TAG + name $NAME, ${MINUTES}m) and read it back"
  echo "   5. POST the create body above   [THE BILL STARTS HERE]; adopt by name if unreadable"
  echo "   6. wait for active + IPv4, then three consecutive ssh successes"
  echo "   7. token to /root/.mojolearn-do.curlrc on stdin; arm the on-droplet dead-man by id;"
  echo "      verify the process, the id baked in, and a GET with the token = 200, else DESTROY"
  echo "   8. search both process lists for the token (grep -F -f)"
  echo "   9. scp the bundle ($BUNDLE_BYTES bytes), verify sha256 on the box, extract fresh"
  echo "  10. ship the extra body and the remote body; start it detached under timeout(1)"
  echo "      at the work bound; poll every 30 s for the sentinel or the pid"
  echo "  11. fetch /root/gemm_leg_out -> $REAL_OUT/remote/ (bounded; tools-venv dropped)"
  echo "  12. compare source_sha256 both ends; DELETE the droplet; GET until 404; cancel the"
  echo "      local dead-man only after the 404"
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

# THE TOKEN, READ BY A BUILTIN AND WRITTEN BY A BUILTIN. No process sees it.
TOK=""
IFS= read -r TOK < "$TOKFILE" || [ -n "$TOK" ] || die "REFUSING to rent: the token file is empty." 2
TOK="${TOK//[$'\t\r\n ']/}"
[ -n "$TOK" ] || die "REFUSING to rent: the token file is empty." 2
CURLRC="$TMPD/curlrc"
TOKPAT="$TMPD/token.pattern"
( umask 077
  printf 'header = "Authorization: Bearer %s"\nsilent\nshow-error\n' "$TOK" > "$CURLRC"
  printf '%s\n' "$TOK" > "$TOKPAT" )
unset TOK

redact() {  # <file>: replace any occurrence of the token (read from the pattern file)
  [ -f "$1" ] || return 0
  python3 - "$TOKPAT" "$1" <<'PY'
import sys
tok = open(sys.argv[1]).read().strip()
p = sys.argv[2]
data = open(p, encoding="utf-8", errors="replace").read()
if tok and tok in data:
    open(p, "w").write(data.replace(tok, "<redacted>"))
PY
}

uplink_down() {
  local h
  for h in https://pypi.org/ https://github.com/ https://www.google.com/; do
    curl -s -o /dev/null --max-time 8 "$h" 2>/dev/null && return 1
  done
  return 0
}
uplink_stable() {
  local r=1
  while [ "$r" -le 3 ]; do
    if uplink_down; then log "uplink probe $r/3: NO neutral host answered"; return 1; fi
    [ "$r" -lt 3 ] && sleep 7
    r=$((r + 1))
  done
  return 0
}

echo
echo "== pre-flight =="
list_live() {  # prints every GPU, leg-tagged or mojolearn-* droplet; prints the HTTP code and returns 1 when the listing failed
  local c
  c=$(http_code GET "$API/droplets?per_page=200" "$TMPD/all_droplets.json")
  [ "$c" = 200 ] || { printf 'HTTP %s' "$c"; return 1; }
  python3 - "$TMPD/all_droplets.json" "$LEG_TAGS" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
tags = set(sys.argv[2].split())
for x in d.get("droplets", []):
    t = set(x.get("tags") or [])
    if str(x.get("size_slug", "")).startswith("gpu-") or (t & tags) or str(x.get("name", "")).startswith("mojolearn-"):
        print("  id=%s name=%s size=%s region=%s tags=%s created=%s" % (
            x.get("id"), x.get("name"), x.get("size_slug"),
            (x.get("region") or {}).get("slug"), ",".join(sorted(t)), x.get("created_at")))
PY
}
refuse_live() {
  printf '%s\n' "$LIVE" > "$OUT/refused_live_droplets.txt"
  die "REFUSING to create $NAME: ONE GPU droplet at a time on this account, and these already exist:
$LIVE
  Each is either a live leg or an orphan. Find out which, destroy or wait, then re-run." 3
}
LIVE="$(list_live)" || die "REFUSING to rent: GET /droplets returned $LIVE (token or network). Nothing was created." 2
[ -z "$LIVE" ] || refuse_live

# THE SHARED GPU LOCK, before the uplink probes and the dead-man. A held lock
# refuses; a lock older than LOCK_STALE_SECONDS is broken only because the
# listing above shows no GPU or mojolearn droplet live.
if ! take_lock; then
  _age=$(lock_age) || _age=""
  _owner=$(tr '\n' ' ' < "$GPU_LOCK/owner" 2>/dev/null)
  if [ -n "$_age" ] && [ "$_age" -gt "$LOCK_STALE_SECONDS" ]; then
    log "breaking a STALE shared GPU lock (${_age}s old, no GPU droplet live): ${_owner:-no owner file}"
    echo "gpu_lock_broken_stale=age ${_age}s owner ${_owner:-none}" >> "$OUT/leg.txt"
    rm -rf "$GPU_LOCK"
    take_lock || die "REFUSING to rent: another leg took $GPU_LOCK while the stale one was being broken. Nothing was created." 3
  else
    die "REFUSING to create $NAME: the shared GPU lock $GPU_LOCK is held (${_age:-?}s old) by: ${_owner:-no owner file}. One GPU droplet at a time across every session (ENGINEERING_RULES 10). Nothing was created." 3
  fi
fi
echo "gpu_lock=taken $(date -u +%Y-%m-%dT%H:%M:%SZ) $GPU_LOCK lane=$LOCK_LANE" >> "$OUT/leg.txt"
log "shared GPU lock taken ($GPU_LOCK, lane $LOCK_LANE)"
# Listed again under the lock: a leg that created between the first listing
# and the mkdir shows here, and the EXIT trap releases the lock.
LIVE="$(list_live)" || die "REFUSING to rent: GET /droplets returned $LIVE under the lock. Nothing was created." 2
[ -z "$LIVE" ] || refuse_live
log "API reachable, no GPU or mojolearn droplet live"
uplink_stable || die "REFUSING to create $NAME: this machine could not reach ANY neutral host. Nothing was created." 2
log "uplink up on all three probes"

# ---- the local dead-man, BEFORE the create ----
LEG_START=$(date +%s)
DEADMAN_SECONDS=$((MINUTES * 60))
DEADLINE_EPOCH=$((LEG_START + DEADMAN_SECONDS))
DEADMAN_DIR="${TMPDIR:-/tmp}/mojolearn-do-extra-deadman-$$"
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
  echo "local_deadman_keyed_by=tag $TAG + name $NAME (+ id once known)"
} > "$OUT/deadman.txt"
log "local dead-man ARMED before the create: pid $DEADMAN_PID, fires at $(utc_of "$DEADLINE_EPOCH")"
if ps -axo command= 2>/dev/null | grep -q -F -f "$TOKPAT"; then
  KEY_RED=1; log "!! THE TOKEN IS VISIBLE IN THIS MACHINE'S PROCESS LIST. Rotate it after this leg."
  echo "local_key_in_ps=VISIBLE" >> "$OUT/leg.txt"
else
  echo "local_key_in_ps=not_visible" >> "$OUT/leg.txt"
fi

# ---- the create ----
echo
echo "== the droplet =="
log "creating $NAME ($SIZE, $REGION, image $IMAGE)"
CREATE_ATTEMPTED=1
c=$(curl -K "$CURLRC" --max-time 60 -o "$OUT/create_response.json" -w '%{http_code}' \
      -X POST -H 'Content-Type: application/json' --data-binary "@$OUT/create_request.json" \
      "$API/droplets" 2>>"$TMPD/curl.err") || c=000
redact "$OUT/create_response.json"
echo "create_http=$c" >> "$OUT/leg.txt"
DROPLET_ID=$(python3 - "$OUT/create_response.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(d["droplet"]["id"] if "droplet" in d else "")
except Exception:
    print("")
PY
)
if [ -z "$DROPLET_ID" ]; then
  log "create returned no id (HTTP $c): $(head -c 300 "$OUT/create_response.json" 2>/dev/null)"
  sleep 5
  DROPLET_ID=$(ids_by_name | awk '{print $1}')
  if [ -n "$DROPLET_ID" ]; then
    log "ADOPTED droplet $DROPLET_ID found by name after an unreadable create response"
    echo "adopted_by_name=1" >> "$OUT/leg.txt"
  else
    die "create FAILED (no droplet by that name either). The teardown sweeps by name once more." 4
  fi
fi
case "$DROPLET_ID" in *[!0-9]*) die "the droplet id '$DROPLET_ID' is not numeric" 4 ;; esac
printf '%s\n' "$DROPLET_ID" > "$OUT/droplet_id.txt"
printf '%s\n' "$DROPLET_ID" > "$DEADMAN_DIR/droplet_id.txt"
echo "droplet=$DROPLET_ID" >> "$OUT/leg.txt"
log "droplet id $DROPLET_ID"

for _i in $(seq 1 90); do
  [ "$(date +%s)" -lt $((DEADLINE_EPOCH - FETCH_RESERVE)) ] || break
  http_code GET "$API/droplets/$DROPLET_ID" "$TMPD/droplet.json" > /dev/null
  read -r _status IP < <(python3 - "$TMPD/droplet.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))["droplet"]
    ips = [n["ip_address"] for n in d["networks"].get("v4", []) if n["type"] == "public"]
    print(d["status"], ips[0] if ips else "")
except Exception:
    print("unknown", "")
PY
)
  [ "$_status" = active ] && [ -n "$IP" ] && break
  IP=""
  sleep 10
done
[ -n "$IP" ] || die "the droplet never became active with a public IPv4" 5
log "active at $IP"
echo "ip=$IP" >> "$OUT/leg.txt"

# A known_hosts of this leg's own: DigitalOcean recycles addresses, and a
# recycled address with a new host key would fail accept-new in ~/.ssh.
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o "UserKnownHostsFile=$TMPD/known_hosts"
          -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3)
SSH=(ssh "${SSH_OPTS[@]}" "root@$IP")
# -n BEFORE the host: stdin from /dev/null for every call that ships nothing.
SSHN=(ssh -n "${SSH_OPTS[@]}" "root@$IP")

# Three CONSECUTIVE successes: a droplet answers once and resets while sshd settles.
_ok=0
for _i in $(seq 1 60); do
  if "${SSHN[@]}" true 2>/dev/null; then
    _ok=$((_ok + 1)); [ "$_ok" -ge 3 ] && break
  else
    _ok=0
  fi
  sleep 5
done
[ "$_ok" -ge 3 ] || die "ssh never settled on $IP" 5
log "ssh settled"

# ---- the on-droplet dead-man ----
SUBST_SECS=$((DEADLINE_EPOCH - $(date +%s)))
[ "$SUBST_SECS" -ge 60 ] || SUBST_SECS=60
SUBST_ID=$DROPLET_ID
cp "$TMPD/droplet_deadman.sh.template" "$OUT/droplet_deadman.sh"
SUBST_ID=$SUBST_ID SUBST_SECS=$SUBST_SECS subst "$OUT/droplet_deadman.sh" || die "the on-droplet dead-man did not substitute" 1
"${SSH[@]}" 'umask 077; cat > /root/.mojolearn-do.curlrc; chmod 600 /root/.mojolearn-do.curlrc' < "$CURLRC" \
  || die "could not deliver the token for the on-droplet dead-man; destroying unused" 6
"${SSH[@]}" 'umask 077; cat > /root/mojolearn-selfkill.sh; chmod 700 /root/mojolearn-selfkill.sh' < "$OUT/droplet_deadman.sh" \
  || die "could not deliver the on-droplet dead-man; destroying unused" 6
"${SSHN[@]}" "nohup sh /root/mojolearn-selfkill.sh > /root/selfkill.log 2>&1 < /dev/null &
echo \$! > /root/selfkill.pid
sleep 1
kill -0 \"\$(cat /root/selfkill.pid)\" 2>/dev/null && echo ON_DROPLET_DEADMAN_ARMED || echo ON_DROPLET_DEADMAN_FAILED
echo ID_BAKED_IN=\$(grep -c 'droplets/$DROPLET_ID' /root/mojolearn-selfkill.sh)
echo TOKEN_GET_HTTP=\$(curl -K /root/.mojolearn-do.curlrc --max-time 20 -o /dev/null -w '%{http_code}' '$API/droplets/$DROPLET_ID')" \
  > "$TMPD/arm.out" 2>&1
sed 's/^/    /' "$TMPD/arm.out"
{
  echo "on_droplet_deadman_seconds=$SUBST_SECS"
  echo "on_droplet_deadman_fires_at=$(utc_of $(( $(date +%s) + SUBST_SECS )))"
  grep -E '^(ON_DROPLET_DEADMAN_|ID_BAKED_IN=|TOKEN_GET_HTTP=)' "$TMPD/arm.out" | sed 's/^/on_droplet_/'
} >> "$OUT/deadman.txt"
if ! grep -q '^ON_DROPLET_DEADMAN_ARMED' "$TMPD/arm.out" \
   || ! grep -q '^ID_BAKED_IN=[1-9]' "$TMPD/arm.out" \
   || ! grep -q '^TOKEN_GET_HTTP=200' "$TMPD/arm.out"; then
  die "THE ON-DROPLET DEAD-MAN COULD NOT BE VERIFIED (process, id, or token GET). A box that cannot guard itself is an orphan that has not happened yet. Destroying it unused." 6
fi
log "on-droplet dead-man ARMED and verified (id $DROPLET_ID, ${SUBST_SECS}s, token GET 200)"

"${SSH[@]}" 'umask 077; cat > /root/.mojolearn-do.key' < "$TOKPAT"
"${SSHN[@]}" 'ps -eo args= > /root/.mojolearn-ps.txt 2>/dev/null || ps ax > /root/.mojolearn-ps.txt
if grep -q -F -f /root/.mojolearn-do.key /root/.mojolearn-ps.txt; then echo KEY_VISIBLE_IN_PS; else echo KEY_NOT_IN_PS; fi
rm -f /root/.mojolearn-ps.txt /root/.mojolearn-do.key' > "$TMPD/ps.out" 2>&1
sed 's/^/    /' "$TMPD/ps.out"
if grep -q KEY_NOT_IN_PS "$TMPD/ps.out"; then
  echo "droplet_key_in_ps=not_visible" >> "$OUT/leg.txt"
else
  KEY_RED=1
  echo "droplet_key_in_ps=$(tr '\n' ' ' < "$TMPD/ps.out")" >> "$OUT/leg.txt"
  log "!! THE TOKEN IS VISIBLE (or unverifiable) IN THE DROPLET'S PROCESS LIST. Rotate it after this leg."
fi
"${SSHN[@]}" "$SMI_CMD" > "$OUT/device.txt" 2>&1
sed 's/^/    [gpu] /' "$OUT/device.txt"

# ---- the source ----
echo
echo "== the source =="
log "scp the bundle ($BUNDLE_BYTES bytes, sha256 ${BUNDLE_SHA:0:16})"
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
_scp_ok=0
for _try in 1 2 3 4 5; do
  _left=$((DEADLINE_EPOCH - FETCH_RESERVE - $(date +%s)))
  [ "$_left" -gt 120 ] || break
  if with_deadline "$_left" scp -q "${SSH_OPTS[@]}" "$TMPD/src.tgz" "root@$IP:/root/extra_src.tgz"; then
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

# MOJOLEARN_DO_EXTRA_UPLOAD: data the box cannot fetch, each file checked by
# sha256 on the box against the one computed before the create.
if [ -n "$UPLOAD_LIST" ]; then
  "${SSHN[@]}" 'mkdir -p /root/gemm_leg_upload' || die "could not create /root/gemm_leg_upload" 7
  for _u in $UPLOAD_LIST; do
    _b=$(basename "$_u")
    _want=$(awk -v p="$_u" '$3 == p { print $1 }' "$OUT/upload_files.txt")
    _bytes=$(awk -v p="$_u" '$3 == p { print $2 }' "$OUT/upload_files.txt")
    _left=$((DEADLINE_EPOCH - FETCH_RESERVE - $(date +%s)))
    [ "$_left" -gt 120 ] || die "no lease left to upload $_b" 7
    log "upload $_b ($_bytes bytes, sha256 ${_want:0:16})"
    with_deadline "$_left" scp -q "${SSH_OPTS[@]}" "$_u" "root@$IP:/root/gemm_leg_upload/$_b" \
      || die "the upload of $_b failed" 7
    _got=$("${SSHN[@]}" "sha256sum /root/gemm_leg_upload/$_b" 2>/dev/null | awk '{ print $1 }')
    [ -n "$_want" ] && [ "$_got" = "$_want" ] \
      || die "the upload of $_b does not match on the box (sha256 ${_got:0:16} there, ${_want:0:16} here)" 7
    echo "upload=$_b bytes=$_bytes sha256=$_want" >> "$OUT/leg.txt"
  done
  log "uploads verified by sha256 on the box after $(( $(date +%s) - LEG_START ))s of lease"
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
log "remote pid $RPID, bound ${WORK_SECONDS}s; polling every 30s"

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
      # let the wrapper append body_exit= before the fetch
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
          log "THIS MACHINE HAS NO UPLINK. The silence is here, not on the box; the on-droplet dead-man still ends it."
          echo "uplink_fault=1" >> "$OUT/leg.txt"
        else
          log "neutral hosts answer, so the BOX is the silent end"
        fi
      fi ;;
  esac
  sleep 30
done
echo "body=$BODY_STATE" >> "$OUT/leg.txt"

# ---- the fetch, bounded, and the destroy runs whatever happens here ----
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
  log "FETCH FAILED or hit its ${FETCH_SECONDS}s bound; whatever arrived is in $OUT/remote/. Destroying regardless."
fi
# Never commit a tools venv.
rm -rf "$OUT/remote/tools-venv"
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

log "leg done; destroying (EXIT trap)"
[ "$FETCH_RED" = 1 ] && exit 1
[ "$KEY_RED" = 1 ] && exit 1
exit 0
