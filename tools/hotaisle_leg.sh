#!/usr/bin/env bash
# tools/hotaisle_leg.sh. THE ONLY WAY ANY LANE RENTS AN AMD GPU ON HOT AISLE.
# One guarded 1x AMD MI300X VM: slot, create, tag, watchdog, ship the commit,
# pixi and the gates, the body, fetch, DELETE, verify gone.
#
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/attention_step_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-amd-mi300x-hotaisle-<lane> \
#   MOJOLEARN_HOTAISLE_EXTRA_ENV='MOJOLEARN_X=1 MODULAR_Y=2' \
#   bash tools/hotaisle_leg.sh amd [--rent | --probe] [--minutes N] [--skip-gates]
#   bash tools/hotaisle_leg.sh status        this team's VMs, descriptions, states, slots, balance
#   bash tools/hotaisle_leg.sh reap <vm>     DELETE ?force=true a mojolearn:* VM, verify gone
#
#   (no mode)  DRY RUN: no key read, no API call, nothing created
#   --probe    free GETs only: roles, VM limit, balance, stock and price, VMs, ssh key
#   --rent     BILLS: creates one VM
#
# ENVIRONMENT
#   MOJOLEARN_GEMM_LEG_EXTRA      the body (POSIX sh), run from /root/mojolearn on the VM
#   MOJOLEARN_GEMM_LEG_OUT        evidence dir; the VM's /root/gemm_leg_out lands in <out>/remote/
#   MOJOLEARN_HOTAISLE_EXTRA_ENV  NAME=value words for the body (MOJOLEARN_* or MODULAR_*, values [A-Za-z0-9_.,:/=-])
#   MOJOLEARN_GPU_ARCHS           one arch; unset reads it from rocminfo on the VM (the MI300X is gfx942)
#   MOJOLEARN_HOTAISLE_KEY_FILE   default ~/.mojolearn_hotaisle_key (0600, one line)
#   MOJOLEARN_HOTAISLE_SPEC       13core (default, comparable CPU opponent rows) or 8core
#   MOJOLEARN_HOTAISLE_LANE       tag in the VM description; default the body's basename
#   MOJOLEARN_HOTAISLE_RUNTIME    auto (default: docker, else podman, else native), docker, podman, native
#   MOJOLEARN_HOTAISLE_IMAGE      default rocm/dev-ubuntu-22.04:6.4.1-complete (the RunPod AMD image)
#   MOJOLEARN_HOTAISLE_TEAM       default andrews-team
# The body sees MOJOLEARN_TARGET_COLUMN=amd and MOJOLEARN_GPU_ARCHS exported,
# /root/mojolearn (git archive of HEAD, no .git), /root/gemm_leg_out, pixi on
# PATH with `pixi install` done, exactly as tools/do_extra_leg.sh provides.
# In a container runtime the body runs inside the image with /dev/kfd and
# /dev/dri passed through and host /root mounted at /root, so the paths are
# the same; the API key lives outside /root on the host and is not mounted.
#
# SAFETY, BAKED IN (none of it is optional)
#   a. SLOTS. At most min(3, the team's maximum_virtual_machines) VMs across
#      every session on this Mac: mkdir /tmp/mojolearn-hotaisle-slot.N, first
#      free, owner file (lane, pid, nonce, utc, VM ref). All busy: poll every
#      60 s. Released only after the delete is verified. A slot older than 100
#      minutes whose owner's VM is absent is logged as possibly broken, and
#      removed only when its owner pid is also dead.
#   b. Balance at or above 500 cents or the leg refuses by name. The chosen
#      spec must show Quantity > 0; otherwise wait and retry up to 30 minutes.
#      The 2x MI300X spec is never picked (exactly one GPU is matched).
#   c. A detached Mac-side dead-man armed BEFORE the create, keyed by the VM's
#      deployment_id as soon as the create returns it. At the deadline it
#      DELETEs with force and verifies, even if this script is gone.
#   d. An on-box watchdog armed before any work (setsid, root, key in a 0600
#      file under /var/lib/mojolearn-hotaisle). It sleeps to the deadline and
#      DELETEs its own VM with force. Verified by pid alive from a SECOND ssh
#      session, the ref baked in, and a GET of its own VM with the key that
#      returns 200 and this leg's description. Unverifiable means delete unused.
#   e. EXIT/INT/TERM trap: DELETE ?force=true, then poll until GET 404 or the
#      VM is absent from a 200 listing (not merely stopped). The verification
#      line is logged. Unverified: banner, dead-man left armed, slot kept.
#   f. --minutes defaults to 60 and 60 is the maximum. More is refused.
#   g. The description is PATCHed to mojolearn:<lane>:<utc> right after the
#      create. reap, status and every delete refuse a VM whose description is
#      not mojolearn:* (the leg's own VM may also be empty if the PATCH never
#      landed, since its id came from this leg's own create response).
#   Creates are serialized by /tmp/mojolearn-hotaisle-create.lock so an
#   unreadable create response is adopted by listing diff without ambiguity.
#   The key is read by the builtin `read`, written by the builtin `printf` into
#   a 0600 curl config read with `curl -K`, reaches the VM on ssh stdin, and is
#   in no argv on either machine. Both process lists are searched for it.
#
# TEST-ONLY FLAGS (they exist to prove the guards; a real leg never uses them)
#   --test-watchdog  ships nothing. The Mac issues no delete before deadline +
#                    10 min: the Mac dead-man moves there as a backstop and the
#                    trap leaves a verified watchdog alone. PASS = the VM is
#                    verified gone within 8 minutes after the deadline.
#   --bare           ships no source, no pixi, no gates, no container; the body
#                    runs natively from an empty /root/mojolearn. For the trap test.
#
# VERIFIED ON REAL VMs, 2026-09-11 (see RUNNER RESULTS below)
#
# THE HOT AISLE API (swagger https://admin.hotaisle.app/api/docs/swagger.json,
# read 2026-09-11, and live GETs)
#   Header `Authorization: Token <key>`. GET /teams/ gives effective_roles and
#   maximum_virtual_machines (2 on andrews-team). GET /teams/{t}/balance/:
#   available_balance in cents. GET .../virtual_machines/available/: Quantity,
#   OnDemandPrice (cents/hour), MinimumReservationMinutes, Specs. POST
#   .../virtual_machines/ with the Specs object: 200 VirtualMachineDetails,
#   401 team VM limit, 402 balance, 404 no stock, 428 no ssh key; provisioning
#   continues if the request is cancelled. GET/PATCH/DELETE .../{vm}/ where vm
#   is the name or deployment_id; PATCH {description} -> 204 (create has no
#   name field); DELETE ?force=true -> 204, needs operator, blocks until the
#   reset is complete. STOP KEEPS BILLING; only DELETE ends it. SSH user
#   `hotaisle`, port from ssh_access.port.
#
# RUNNER RESULTS
#   (filled in by the three tests below)
set -uo pipefail

# RUN FROM AN IMMUTABLE SNAPSHOT (tools/do_extra_leg.sh does the same): bash
# reads a script lazily by offset, and a leg lasts long enough for an edit.
if [ "${MOJOLEARN_HOTAISLE_FROZEN:-0}" != 1 ]; then
  _repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 2
  _snapdir="$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-hotaisle-snapshot.XXXXXX")" || exit 2
  cp "${BASH_SOURCE[0]}" "$_snapdir/hotaisle_leg.sh" || { rm -rf "$_snapdir"; exit 2; }
  chmod 555 "$_snapdir/hotaisle_leg.sh"
  trap 'rm -rf "$_snapdir"' EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  MOJOLEARN_HOTAISLE_FROZEN=1 MOJOLEARN_HOTAISLE_REPO="$_repo" \
    bash "$_snapdir/hotaisle_leg.sh" "$@" &
  _child=$!
  # Forward a TERM or INT to the leg AS TERM so its own trap runs the teardown
  # (a background child of a non-interactive shell ignores INT and cannot trap it).
  trap 'kill -TERM "$_child" 2>/dev/null; wait "$_child"; exit $?' TERM
  trap 'kill -TERM "$_child" 2>/dev/null; wait "$_child"; exit $?' INT
  wait "$_child"
  exit $?
fi

REPO="${MOJOLEARN_HOTAISLE_REPO:?}"
cd "$REPO" || exit 2
API=https://admin.hotaisle.app/api
TEAM="${MOJOLEARN_HOTAISLE_TEAM:-andrews-team}"
KEYFILE="${MOJOLEARN_HOTAISLE_KEY_FILE:-$HOME/.mojolearn_hotaisle_key}"
SPEC="${MOJOLEARN_HOTAISLE_SPEC:-13core}"
RUNTIME_WANT="${MOJOLEARN_HOTAISLE_RUNTIME:-auto}"
IMAGE="${MOJOLEARN_HOTAISLE_IMAGE:-rocm/dev-ubuntu-22.04:6.4.1-complete}"
SSH_KEY="$HOME/.ssh/id_ed25519"
SSH_KEY_FP="SHA256:pqDQ15Jijc636E5M3/2IvTGR7YKL11+JzQwGYtBvtQU"
SLOT_PREFIX=/tmp/mojolearn-hotaisle-slot
CREATE_LOCK=/tmp/mojolearn-hotaisle-create.lock
MAX_SLOTS=3
MIN_BALANCE_CENTS=500
SLOT_STALE_SECONDS=6000
SLOT_WAIT_MINUTES="${MOJOLEARN_HOTAISLE_SLOT_WAIT_MINUTES:-240}"
STOCK_WAIT_MINUTES=30
FETCH_RESERVE="${FETCH_RESERVE:-240}"
MAX_BUNDLE_BYTES="${MOJOLEARN_HOTAISLE_MAX_BYTES:-15000000}"
BOX_DIR=/var/lib/mojolearn-hotaisle
BOX_RC="$BOX_DIR/curlrc"

usage() { sed -n '2,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
log() { printf '[%s hotaisle] %s\n' "$(date +%T)" "$*"; }
die() { printf '\n%s\n' "$1" >&2; exit "${2:-1}"; }
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
utc_of() { date -u -r "$1" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d "@$1" +%Y-%m-%dT%H:%M:%SZ; }
nap() { sleep "$1" & wait $! 2>/dev/null; }  # a sleep a trapped signal interrupts
sha256_of() { if command -v shasum > /dev/null 2>&1; then shasum -a 256 "$1"; else sha256sum "$1"; fi | awk '{print $1}'; }

# ------------------------------------------------------------------ arguments
CMD=leg; MODE=dry; MINUTES=60; GATES=1; TEST_WATCHDOG=0; BARE=0; REAP_REF=""
while [ $# -gt 0 ]; do
  case "$1" in
    amd) ;;
    nv|nvidia) echo "Hot Aisle is AMD only; NVIDIA legs use RunPod" >&2; exit 2 ;;
    status) CMD=status ;;
    reap) CMD=reap; shift; REAP_REF="${1:-}" ;;
    --probe) MODE=probe ;;
    --rent) MODE=rent ;;
    --dry-run) MODE=dry ;;
    --minutes) shift; MINUTES="${1:-}" ;;
    --minutes=*) MINUTES="${1#--minutes=}" ;;
    --skip-gates) GATES=0 ;;
    --test-watchdog) TEST_WATCHDOG=1 ;;
    --bare) BARE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
case "$MINUTES" in ''|*[!0-9]*) echo "--minutes must be a whole number" >&2; exit 2 ;; esac
if [ "$MINUTES" -gt 60 ]; then
  echo "--minutes $MINUTES REFUSED: 60 is the maximum lease (a second leg, never an extension)" >&2; exit 2
fi
_min=10; [ "$BARE" = 1 ] && _min=5; [ "$TEST_WATCHDOG" = 1 ] && _min=3
[ "$MINUTES" -ge "$_min" ] || { echo "--minutes must be at least $_min for this mode" >&2; exit 2; }
[ "$TEST_WATCHDOG" = 1 ] && [ "$BARE" = 1 ] && { echo "--test-watchdog and --bare are separate tests" >&2; exit 2; }
case "$SPEC" in
  13core) SPEC_CORES=13 ;;
  8core)  SPEC_CORES=8 ;;
  *) echo "MOJOLEARN_HOTAISLE_SPEC='$SPEC': 13core or 8core (the 2x MI300X spec is never rented)" >&2; exit 2 ;;
esac
case "$RUNTIME_WANT" in auto|docker|podman|native) ;; *) echo "MOJOLEARN_HOTAISLE_RUNTIME: auto, docker, podman or native" >&2; exit 2 ;; esac
case "$IMAGE" in *[!A-Za-z0-9_.:/@-]*) echo "MOJOLEARN_HOTAISLE_IMAGE: letters, digits and _.:/@- only" >&2; exit 2 ;; esac
case "$TEAM" in ''|*[!A-Za-z0-9_-]*) echo "MOJOLEARN_HOTAISLE_TEAM: letters, digits, _ and - only" >&2; exit 2 ;; esac
GPU_ARCHS="${MOJOLEARN_GPU_ARCHS:-}"
case "$GPU_ARCHS" in *[!A-Za-z0-9_]*)
  echo "MOJOLEARN_GPU_ARCHS='$GPU_ARCHS': exactly one architecture name (one mojo build is one GPU arch)" >&2; exit 2 ;;
esac

# ------------------------------------------------------------------ state
TMPD="$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-hotaisle.XXXXXX")" || exit 2
chmod 700 "$TMPD"
CURLRC=""; TOKPAT=""; OUT="$TMPD/out"
VMREF=""; VMNAME=""; DEPLOY_ID=""; SSH_IP=""; SSH_PORT=22; DESC=""
CREATE_ATTEMPTED=0; CREATE_LOCK_HELD=0; DESTROY_CONFIRMED=0
DEADMAN_PID=""; DEADMAN_DIR=""; DEADLINE_EPOCH=0; LEG_START=0
SLOT=""; NONCE="$$-$(date -u +%Y%m%dT%H%M%SZ)"; WATCHDOG_OK=0; WDT_DONE=0
KEY_RED=0; FETCH_RED=0; BODY_STATE=not_started; SSH=(ssh); SSHN=(ssh -n)
BAL_BEFORE=""

# ---- JSON helper, one file, also copied beside the dead-man ----
cat > "$TMPD/j.py" <<'PY'
import json, sys
def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None
cmd, d = sys.argv[1], load(sys.argv[2])
a = sys.argv[3:]
if cmd == "teams":
    for t in d or []:
        if t.get("handle") == a[0]:
            print("yes" if "operator" in (t.get("effective_roles") or []) else "no",
                  t.get("maximum_virtual_machines", 0))
            break
    else:
        print("absent 0")
elif cmd == "balance":
    print((d or {}).get("available_balance", -1) if isinstance(d, dict) else -1)
elif cmd == "offers":
    for e in d or []:
        s = e.get("Specs") or {}
        g = s.get("gpus") or []
        print("  %sx %s  cpu_cores=%s ram=%dGB  quantity=%s  %s cents/h  min_reservation=%s min" % (
            sum(x.get("count", 0) for x in g), ",".join(x.get("model", "?") for x in g),
            s.get("cpu_cores"), (s.get("ram_capacity") or 0) // 2**30, e.get("Quantity"),
            e.get("OnDemandPrice"), e.get("MinimumReservationMinutes")))
elif cmd == "pick":
    cores, found = int(a[0]), []
    for e in d or []:
        s = e.get("Specs") or {}
        g = s.get("gpus") or []
        if (len(g) == 1 and g[0].get("count") == 1 and g[0].get("model") == "MI300X"
                and s.get("cpu_cores") == cores):
            found.append(e)
    if not found:
        print("none 0 0 0")
    else:
        found.sort(key=lambda e: -(e.get("Quantity") or 0))
        best = found[0]
        with open(a[1], "w") as f:
            json.dump(best["Specs"], f)
        print("found", best.get("Quantity", 0), best.get("OnDemandPrice", 0),
              best.get("MinimumReservationMinutes", 0))
elif cmd == "vm":
    v = d if isinstance(d, dict) else {}
    sa = v.get("ssh_access") or {}
    print("\t".join(str(x) for x in [v.get("name", ""), v.get("deployment_id", ""),
          sa.get("ip_address") or v.get("ip_address", ""), sa.get("port") or 22,
          v.get("description", "") or ""]))
elif cmd == "desc":
    print((d or {}).get("description", "") or "" if isinstance(d, dict) else "")
elif cmd == "state":
    print((d or {}).get("state", "unknown") if isinstance(d, dict) else "unknown")
elif cmd == "list":
    for v in d or []:
        print("\t".join(str(x) for x in [v.get("name", ""), v.get("deployment_id", ""),
              v.get("ip_address", ""), v.get("description", "") or ""]))
elif cmd == "count":
    print(len(d) if isinstance(d, list) else -1)
elif cmd == "inlist":
    if not isinstance(d, list):
        print("unknown")
    else:
        refs = set(x for x in a if x)
        print("yes" if any(v.get("name") in refs or v.get("deployment_id") in refs for v in d) else "no")
elif cmd == "ids":
    for v in d or []:
        print(v.get("deployment_id", ""), v.get("name", ""))
elif cmd == "sshkey":
    print("yes" if any(k.get("fingerprint") == a[0] for k in d or []) else "no")
PY
J() { python3 "$TMPD/j.py" "$@"; }

# ------------------------------------------------------------------ the key
key_hygiene() {
  local perm
  [ -f "$KEYFILE" ] || { echo "key file $KEYFILE does not exist"; return 1; }
  perm=$(stat -f '%OLp' "$KEYFILE" 2>/dev/null || stat -c '%a' "$KEYFILE" 2>/dev/null || echo '?')
  [ "$perm" = 600 ] || { echo "key file $KEYFILE is mode $perm, must be 600"; return 1; }
  case "$(cd "$(dirname "$KEYFILE")" && pwd)/" in "$REPO"/*) echo "key file $KEYFILE is INSIDE the repository"; return 1 ;; esac
  return 0
}
load_key() {
  local K="" why
  why=$(key_hygiene) || die "REFUSING: $why" 2
  IFS= read -r K < "$KEYFILE" || [ -n "$K" ] || die "REFUSING: the key file is empty." 2
  K="${K//[$'\t\r\n ']/}"
  case "$K" in ''|*[!A-Za-z0-9._-]*) die "REFUSING: the key file holds characters outside [A-Za-z0-9._-]." 2 ;; esac
  CURLRC="$TMPD/curlrc"; TOKPAT="$TMPD/key.pattern"
  ( umask 077
    printf 'header = "Authorization: Token %s"\nsilent\nshow-error\n' "$K" > "$CURLRC"
    printf '%s\n' "$K" > "$TOKPAT" )
  K=""
}
api() {  # <method> <path under /api/> <body out> [max-time] [json body file]; prints the HTTP code, 000 on transport failure
  local c mt="${4:-60}"
  if [ -n "${5:-}" ]; then
    c=$(curl -K "$CURLRC" --max-time "$mt" -o "$3" -w '%{http_code}' -X "$1" \
          -H 'Content-Type: application/json' --data-binary "@$5" "$API/$2" 2>>"$TMPD/curl.err") || c=000
  else
    c=$(curl -K "$CURLRC" --max-time "$mt" -o "$3" -w '%{http_code}' -X "$1" "$API/$2" 2>>"$TMPD/curl.err") || c=000
  fi
  printf '%s' "${c:-000}"
}
redact() {  # <file>: replace any occurrence of the key
  [ -f "$1" ] && [ -n "$TOKPAT" ] || return 0
  python3 - "$TOKPAT" "$1" <<'PY'
import sys
tok = open(sys.argv[1]).read().strip()
p = sys.argv[2]
data = open(p, encoding="utf-8", errors="replace").read()
if tok and tok in data:
    open(p, "w").write(data.replace(tok, "<redacted>"))
PY
}
balance_cents() {  # prints cents, or -1
  local c
  c=$(api GET "teams/$TEAM/balance/" "$TMPD/balance.json")
  [ "$c" = 200 ] || { echo -1; return; }
  J balance "$TMPD/balance.json"
}
dollars() { [ "$1" -ge 0 ] 2>/dev/null && printf '$%d.%02d' $(( $1 / 100 )) $(( $1 % 100 )) || printf 'unknown'; }
vm_desc_ok() { case "$1" in mojolearn:*) return 0 ;; esac; return 1; }

# gone = GET 404, or absent from a 200 team listing. Prints "yes get=.. listed=.. state=..".
gone_check() {  # <ref> [<other ref>]
  local g l inl st="n/a"
  g=$(api GET "teams/$TEAM/virtual_machines/$1/" "$TMPD/gone_vm.json")
  if [ "$g" = 200 ]; then
    [ "$(api GET "teams/$TEAM/virtual_machines/$1/state/" "$TMPD/gone_state.json")" = 200 ] \
      && st=$(J state "$TMPD/gone_state.json")
  fi
  l=$(api GET "teams/$TEAM/virtual_machines/" "$TMPD/gone_list.json")
  inl=unknown; [ "$l" = 200 ] && inl=$(J inlist "$TMPD/gone_list.json" "$1" "${2:-}")
  if [ "$g" = 404 ] || { [ "$l" = 200 ] && [ "$inl" = no ]; }; then
    printf 'yes get=%s list=%s listed=%s state=%s' "$g" "$l" "$inl" "$st"; return 0
  fi
  printf 'no get=%s list=%s listed=%s state=%s' "$g" "$l" "$inl" "$st"; return 1
}

# DELETE ?force=true, then verify gone for up to <verify seconds>. Sets GONE_LINE.
delete_and_verify() {  # <ref> <other ref or ""> <expected description or ""> <verify seconds> <record file>
  local ref=$1 other=$2 want=$3 secs=$4 rec=$5 c d i t0 line
  GONE_LINE=""
  if line=$(gone_check "$ref" "$other"); then
    GONE_LINE="verified_gone ref=$ref $line utc=$(utc) (already gone, no DELETE sent)"
    echo "$GONE_LINE" >> "$rec"; return 0
  fi
  c=$(api GET "teams/$TEAM/virtual_machines/$ref/" "$TMPD/del_vm.json")
  if [ "$c" = 200 ]; then
    d=$(J desc "$TMPD/del_vm.json")
    if [ -n "$d" ] && { ! vm_desc_ok "$d" || { [ -n "$want" ] && [ "$d" != "$want" ]; }; }; then
      echo "delete REFUSED ref=$ref: description '$d' is not this leg's ('${want:-mojolearn:*}')" >> "$rec"
      log "!! delete REFUSED: $ref has description '$d', not '${want:-mojolearn:*}'"
      return 2
    fi
  fi
  for i in 1 2 3 4; do
    c=$(api DELETE "teams/$TEAM/virtual_machines/$ref/?force=true" "$TMPD/del_body.txt" 900)
    log "DELETE $ref ?force=true -> HTTP $c"
    echo "delete ref=$ref attempt $i -> HTTP $c utc=$(utc) body=$(head -c 200 "$TMPD/del_body.txt" 2>/dev/null | tr '\n' ' ')" >> "$rec"
    case "$c" in 2*|404) break ;; esac
    nap 15
  done
  t0=$(date +%s)
  while :; do
    if line=$(gone_check "$ref" "$other"); then
      GONE_LINE="verified_gone ref=$ref $line utc=$(utc) after=$(( $(date +%s) - t0 ))s"
      echo "$GONE_LINE" >> "$rec"; log "$GONE_LINE"; return 0
    fi
    echo "verify ref=$ref $line utc=$(utc)" >> "$rec"
    [ $(( $(date +%s) - t0 )) -lt "$secs" ] || break
    nap 15
  done
  GONE_LINE="NOT_VERIFIED ref=$ref $line utc=$(utc)"
  echo "$GONE_LINE" >> "$rec"; log "!! $GONE_LINE"
  return 1
}

# ------------------------------------------------------------------ slots
slot_age() { local m; m=$(stat -f %m "$1" 2>/dev/null || stat -c %Y "$1" 2>/dev/null) || return 1; echo $(( $(date +%s) - m )); }
slot_field() { sed -n "s/^$2=//p" "$1/owner" 2>/dev/null | tail -1; }
slot_cap() {  # min(MAX_SLOTS, the team's maximum_virtual_machines)
  local cap=$MAX_SLOTS
  [ "${TEAM_MAX_VMS:-0}" -gt 0 ] 2>/dev/null && [ "$TEAM_MAX_VMS" -lt "$cap" ] && cap=$TEAM_MAX_VMS
  echo "$cap"
}
list_slots() {
  local n s
  for n in 1 2 3; do
    s="$SLOT_PREFIX.$n"
    if [ -d "$s" ]; then
      printf '  slot %s  HELD %ss  %s\n' "$n" "$(slot_age "$s")" "$(tr '\n' ' ' < "$s/owner" 2>/dev/null || echo 'no owner file')"
    else
      printf '  slot %s  free\n' "$n"
    fi
  done
}
try_take_slot() {  # <cap>
  local n
  for n in $(seq 1 "$1"); do
    if mkdir "$SLOT_PREFIX.$n" 2>/dev/null; then
      SLOT="$SLOT_PREFIX.$n"
      { echo "lane=$LANE"; echo "pid=$$"; echo "nonce=$NONCE"; echo "utc=$(utc)"; echo "out=$REAL_OUT"; } > "$SLOT/owner"
      return 0
    fi
  done
  return 1
}
check_stale_slots() {
  local n s age pid ref
  for n in 1 2 3; do
    s="$SLOT_PREFIX.$n"
    [ -d "$s" ] || continue
    age=$(slot_age "$s") || continue
    [ "$age" -gt "$SLOT_STALE_SECONDS" ] || continue
    pid=$(slot_field "$s" pid); ref=$(slot_field "$s" vm_ref)
    [ "$(api GET "teams/$TEAM/virtual_machines/" "$TMPD/stale_list.json")" = 200 ] || continue
    if [ -z "$ref" ] || [ "$(J inlist "$TMPD/stale_list.json" "$ref")" = no ]; then
      if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        log "slot $s is ${age}s old, its VM ${ref:-<none recorded>} is absent, owner pid $pid is ALIVE: may be broken, left alone"
      else
        log "slot $s is ${age}s old, its VM ${ref:-<none recorded>} is absent, owner pid ${pid:-?} is dead: may be broken, REMOVED"
        echo "slot_broken_stale=$s age=${age}s owner=$(tr '\n' ' ' < "$s/owner" 2>/dev/null)" >> "$OUT/leg.txt"
        rm -rf "$s"
      fi
    fi
  done
}
release_slot() {
  [ -n "$SLOT" ] || return 0
  if grep -qx "nonce=$NONCE" "$SLOT/owner" 2>/dev/null; then
    rm -rf "$SLOT" && log "released $SLOT"
    echo "slot=released $SLOT $(utc)" >> "$OUT/leg.txt"
  else
    log "!! $SLOT no longer carries this leg's nonce; left in place"
  fi
  SLOT=""
}

# ------------------------------------------------------------------ status / reap
if [ "$CMD" = status ] || [ "$CMD" = reap ]; then
  OUT="$TMPD"; : > "$OUT/leg.txt"
  trap 'rm -rf "$TMPD"' EXIT
  load_key
  _b=$(balance_cents)
  echo "team $TEAM balance $(dollars "$_b") ($_b cents)"
  if [ "$CMD" = status ]; then
    c=$(api GET "teams/$TEAM/virtual_machines/" "$TMPD/list.json")
    [ "$c" = 200 ] || die "GET virtual_machines -> HTTP $c" 2
    echo "VMs ($(J count "$TMPD/list.json")):"
    J list "$TMPD/list.json" | while IFS=$'\t' read -r n id ip d; do
      st=unknown
      [ "$(api GET "teams/$TEAM/virtual_machines/$id/state/" "$TMPD/st.json")" = 200 ] && st=$(J state "$TMPD/st.json")
      own="NOT OURS, never touched"; vm_desc_ok "$d" && own=ours
      printf '  %s  deployment_id=%s  ip=%s  state=%s  description=%s  (%s)\n' "$n" "$id" "$ip" "$st" "${d:-<empty>}" "$own"
    done
    echo "slots (/tmp, this Mac):"; list_slots
    exit 0
  fi
  case "$REAP_REF" in ''|*[!A-Za-z0-9_.-]*) die "reap <vm>: a VM name or deployment_id" 2 ;; esac
  c=$(api GET "teams/$TEAM/virtual_machines/$REAP_REF/" "$TMPD/reap.json")
  if [ "$c" = 200 ]; then
    d=$(J desc "$TMPD/reap.json")
    vm_desc_ok "$d" || die "REFUSING to reap $REAP_REF: its description '${d:-<empty>}' is not mojolearn:*" 3
    IFS=$'\t' read -r _n _id _ip _port _d < <(J vm "$TMPD/reap.json")
    delete_and_verify "$REAP_REF" "$_id" "" 600 "$TMPD/reap.txt"; rc=$?
  else
    delete_and_verify "$REAP_REF" "" "" 60 "$TMPD/reap.txt"; rc=$?
  fi
  cat "$TMPD/reap.txt"
  if [ "$rc" = 0 ]; then
    for _s in "$SLOT_PREFIX".1 "$SLOT_PREFIX".2 "$SLOT_PREFIX".3; do
      [ -d "$_s" ] || continue
      _r=$(slot_field "$_s" vm_ref); _p=$(slot_field "$_s" pid)
      if [ -n "$_r" ] && { [ "$_r" = "$REAP_REF" ] || [ "$_r" = "${_id:-}" ]; }; then
        if [ -n "$_p" ] && kill -0 "$_p" 2>/dev/null; then
          echo "slot $_s names this VM and its owner pid $_p is alive; left for its owner to release"
        else
          rm -rf "$_s" && echo "released $_s (owner pid ${_p:-?} dead, VM verified gone)"
        fi
      fi
    done
  fi
  echo "team $TEAM balance after $(dollars "$(balance_cents)")"
  exit "$rc"
fi

# ------------------------------------------------------------------ the leg
LEG_EXTRA="${MOJOLEARN_GEMM_LEG_EXTRA:-}"
if [ "$MODE" != probe ]; then
  [ -n "$LEG_EXTRA" ] || die "MOJOLEARN_GEMM_LEG_EXTRA=<body.sh> is required: running a body is this runner's whole job" 2
  [ -f "$LEG_EXTRA" ] || die "MOJOLEARN_GEMM_LEG_EXTRA=$LEG_EXTRA does not exist" 2
fi
LANE="${MOJOLEARN_HOTAISLE_LANE:-$(basename "${LEG_EXTRA:-probe}" .sh)}"
case "$LANE" in ''|*[!A-Za-z0-9_.-]*) die "MOJOLEARN_HOTAISLE_LANE='$LANE': letters, digits and _.- only (it goes in the VM description)" 2 ;; esac
STAMP="$(date -u +%Y-%m-%d_%H%M%S)"
CARD_FULL="${MOJOLEARN_GEMM_CARD_FULL:-}"
LEG_DUMP="${MOJOLEARN_IDENTITY_TRACE_DUMP:-}"
for _v in "$CARD_FULL" "$LEG_DUMP"; do
  case "$_v" in *[!A-Za-z0-9_.,:-]*) die "MOJOLEARN_GEMM_CARD_FULL / MOJOLEARN_IDENTITY_TRACE_DUMP: letters, digits and _.,:- only" 2 ;; esac
done
EXTRA_ENV="${MOJOLEARN_HOTAISLE_EXTRA_ENV:-}"
OUT="${MOJOLEARN_GEMM_LEG_OUT:-bench/results/e1g/${STAMP}-amd-mi300x-hotaisle-${LANE}}"
case "$OUT" in /*) ;; *) OUT="$REPO/$OUT" ;; esac
REAL_OUT="$OUT"
if [ "$MODE" != rent ]; then
  OUT="$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-hotaisle-$MODE.XXXXXX")" || exit 2
fi
mkdir -p "$OUT" || die "cannot create $OUT" 2
SIZE_LABEL="mi300x-${SPEC}"

# shellcheck disable=SC2317
cancel_deadman() {
  [ -n "$DEADMAN_PID" ] || return 0
  pkill -KILL -P "$DEADMAN_PID" 2>/dev/null
  kill -KILL "$DEADMAN_PID" 2>/dev/null && log "Mac dead-man cancelled (pid $DEADMAN_PID)"
  [ -n "$DEADMAN_DIR" ] && rm -rf "$DEADMAN_DIR"
  echo "mac_deadman=cancelled $(utc)" >> "$OUT/deadman.txt"
  DEADMAN_PID=""; DEADMAN_DIR=""
}
# shellcheck disable=SC2317
release_create_lock() {
  [ "$CREATE_LOCK_HELD" = 1 ] || return 0
  grep -qx "nonce=$NONCE" "$CREATE_LOCK/owner" 2>/dev/null && rm -rf "$CREATE_LOCK"
  CREATE_LOCK_HELD=0
}
# shellcheck disable=SC2317
adopt_new_vm() {  # sets VMREF from a VM absent from the pre-create snapshot; returns 1 when none or ambiguous
  local c new n
  c=$(api GET "teams/$TEAM/virtual_machines/" "$TMPD/adopt.json")
  [ "$c" = 200 ] || return 1
  new=$(J ids "$TMPD/adopt.json" | while read -r id nm; do
          grep -qx "$id" "$TMPD/pre_ids.txt" 2>/dev/null || echo "$id $nm"
        done)
  n=$(printf '%s' "$new" | grep -c . )
  [ "$n" = 1 ] || { [ "$n" -gt 1 ] && log "!! $n new VMs appear; adopting none: $(echo "$new" | tr '\n' ';')"; return 1; }
  DEPLOY_ID=${new%% *}; VMNAME=${new#* }; VMREF=$DEPLOY_ID
  return 0
}

# shellcheck disable=SC2317
teardown() {
  local rc=$? line
  trap '' INT TERM
  trap - EXIT
  if [ "$CREATE_ATTEMPTED" = 1 ]; then
    log "teardown (exit $rc)"
    echo "== teardown $(utc) exit=$rc vm=${VMREF:-unknown} ==" >> "$OUT/teardown.txt"
    if [ -z "$VMREF" ]; then
      for _i in 1 2 3 4 5 6 7 8 9 10 11 12; do adopt_new_vm && break; sleep 10; done
      if [ -n "$VMREF" ]; then
        log "ADOPTED $VMREF at teardown after an unreadable create"
        printf '%s\n' "$VMREF" > "$DEADMAN_DIR/vm_ref.txt" 2>/dev/null
      fi
    fi
    if [ -z "$VMREF" ]; then
      if [ "$(api GET "teams/$TEAM/virtual_machines/" "$TMPD/final.json")" = 200 ] \
         && [ -z "$(J ids "$TMPD/final.json" | while read -r id nm; do grep -qx "$id" "$TMPD/pre_ids.txt" || echo "$id"; done)" ]; then
        echo "verified_gone ref=none: the 200 listing shows no VM absent from the pre-create snapshot utc=$(utc)" >> "$OUT/teardown.txt"
        DESTROY_CONFIRMED=1
      fi
    elif [ "$TEST_WATCHDOG" = 1 ] && [ "$WATCHDOG_OK" = 1 ] && [ "$WDT_DONE" = 0 ]; then
      log "watchdog test interrupted before its verdict: the verified on-box watchdog and the late Mac backstop are LEFT to end $VMREF"
      echo "watchdog_test=INTERRUPTED; delete left to the on-box watchdog and the backstop" >> "$OUT/teardown.txt"
    elif delete_and_verify "$VMREF" "$VMNAME" "$DESC" 600 "$OUT/teardown.txt"; then
      DESTROY_CONFIRMED=1
    fi
    release_create_lock
    echo "destroy_confirmed=$DESTROY_CONFIRMED" >> "$OUT/teardown.txt"
  fi
  release_create_lock
  if [ "$CREATE_ATTEMPTED" = 0 ] || [ "$DESTROY_CONFIRMED" = 1 ]; then
    cancel_deadman
    release_slot
  else
    {
      echo
      echo "  ############################################################"
      echo "  # HOT AISLE VM ${VMREF:-<unknown>} ${VMNAME:+($VMNAME) }MAY STILL BE BILLING."
      echo "  # The API did not confirm it is gone. The Mac dead-man (pid ${DEADMAN_PID:-none})"
      echo "  # and the on-box watchdog are LEFT ARMED. End it by hand now:"
      echo "  #   bash tools/hotaisle_leg.sh reap ${VMREF:-<vm>}"
      echo "  # Slot ${SLOT:-none} stays HELD until then."
      echo "  ############################################################"
    } | tee -a "$OUT/teardown.txt" >&2
    echo "mac_deadman=LEFT_ARMED pid=$DEADMAN_PID" >> "$OUT/deadman.txt"
    [ "$rc" = 0 ] && rc=1
  fi
  if [ "$MODE" = rent ]; then
    line=$(balance_cents)
    log "team balance at the end $(dollars "$line") (at the start $(dollars "${BAL_BEFORE:--1}"))"
    echo "balance_after_cents=$line" >> "$OUT/leg.txt"
    echo "exit=$rc" >> "$OUT/teardown.txt"
  fi
  [ -n "$TMPD" ] && rm -rf "$TMPD"
  exit "$rc"
}
trap teardown EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ------------------------------------------------------------------ probe
if [ "$MODE" = probe ]; then
  load_key
  c=$(api GET "teams/" "$TMPD/teams.json"); echo "GET teams/ -> $c"
  read -r _op TEAM_MAX_VMS < <(J teams "$TMPD/teams.json" "$TEAM")
  echo "  team $TEAM operator=$_op maximum_virtual_machines=$TEAM_MAX_VMS (slots used: $(slot_cap))"
  _b=$(balance_cents); echo "  balance $(dollars "$_b") ($_b cents; floor $MIN_BALANCE_CENTS)"
  c=$(api GET "teams/$TEAM/virtual_machines/available/" "$TMPD/avail.json"); echo "GET available -> $c"
  J offers "$TMPD/avail.json"
  read -r _f _q _p _m < <(J pick "$TMPD/avail.json" "$SPEC_CORES" "$TMPD/spec.json")
  echo "  MOJOLEARN_HOTAISLE_SPEC=$SPEC -> $_f quantity=$_q price=$_p cents/h min_reservation=$_m"
  c=$(api GET "teams/$TEAM/virtual_machines/" "$TMPD/list.json"); echo "GET virtual_machines -> $c ($(J count "$TMPD/list.json") VMs)"
  J list "$TMPD/list.json" | sed 's/^/  /'
  c=$(api GET "user/ssh_keys/" "$TMPD/keys.json"); echo "GET user/ssh_keys -> $c; this Mac's key registered: $(J sshkey "$TMPD/keys.json" "$SSH_KEY_FP")"
  echo "slots:"; list_slots
  exit 0
fi

# ------------------------------------------------------------- local checks
RED=0; BLOCK=0
rok()    { printf '  ok     %s\n' "$1"; }
rbad()   { RED=1;   printf '  FAIL   %s\n' "$1"; }
rblock() { BLOCK=1; printf '  BLOCK  %s\n' "$1"; }
SHIPS_SOURCE=1; [ "$BARE" = 1 ] || [ "$TEST_WATCHDOG" = 1 ] && SHIPS_SOURCE=0

COMMIT="$(git -C "$REPO" rev-parse HEAD)" || die "not a git checkout: $REPO" 2
COMMIT_LINE="$(git -C "$REPO" log -1 --format='%h parent %p' "$COMMIT")"
echo "== hotaisle_leg: one Hot Aisle 1x MI300X leg running an extra body =="
echo "   mode      $MODE$( [ "$TEST_WATCHDOG" = 1 ] && echo ' TEST-WATCHDOG')$( [ "$BARE" = 1 ] && echo ' BARE')"
echo "   commit    $COMMIT_LINE"
echo "   spec      $SPEC (1x MI300X, $SPEC_CORES cores), team $TEAM"
echo "   lease     $MINUTES minutes (Mac dead-man and on-box watchdog at that deadline)"
echo "   body      $LEG_EXTRA   lane $LANE"
echo "   gates     $( [ "$GATES" = 1 ] && echo 'device check + card' || echo 'SKIPPED (--skip-gates)')"
echo "   archs     ${GPU_ARCHS:-<unset: read from rocminfo on the VM>}   column amd"
echo "   runtime   $RUNTIME_WANT (image $IMAGE)"
echo "   out       $REAL_OUT"
echo
echo "== local checks =="
DIRTY="$(git -C "$REPO" status --porcelain -- . ':!bench/results' 2>/dev/null)"
if [ -n "$DIRTY" ] && [ "$SHIPS_SOURCE" = 1 ]; then
  rblock "the tree is DIRTY (minus bench/results); a real leg refuses. Launch from git worktree add --detach:"
  printf '%s\n' "$DIRTY" | head -20 | sed 's/^/           /'
elif [ -n "$DIRTY" ]; then
  printf '  info   the tree is dirty; this test mode ships no source\n'
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
  echo "# Generated by tools/hotaisle_leg.sh from MOJOLEARN_HOTAISLE_EXTRA_ENV; sourced before the extra body."
  _env_ok=1
  for _w in $EXTRA_ENV; do
    case "$_w" in
      MOJOLEARN_HOTAISLE_*=*|MOJOLEARN_DO_*=*|MOJOLEARN_RUNPOD_*=*|MOJOLEARN_GEMM_LEG_*=*|MOJOLEARN_GPU_ARCHS=*|MOJOLEARN_TARGET_COLUMN=*)
        _env_ok=0; printf '# REFUSED (runner-owned name): %s\n' "${_w%%=*}" ;;
      MOJOLEARN_[A-Z0-9_]*=*|MODULAR_[A-Z0-9_]*=*)
        _n=${_w%%=*}; _v=${_w#*=}
        case "$_n" in *[!A-Z0-9_]*) _env_ok=0; printf '# REFUSED (name characters): %s\n' "$_n"; continue ;; esac
        case "$_v" in
          *[!A-Za-z0-9_.,:/=-]*) _env_ok=0; printf '# REFUSED (value characters): %s\n' "$_n" ;;
          *) printf "export %s='%s'\n" "$_n" "$_v" ;;
        esac ;;
      *) _env_ok=0; printf '# REFUSED (not NAME=value with a MOJOLEARN_ or MODULAR_ name): %s\n' "${_w%%=*}" ;;
    esac
  done
} > "$OUT/extra_env.sh"
if ! grep -q '^# REFUSED' "$OUT/extra_env.sh" && sh -n "$OUT/extra_env.sh" 2>/dev/null; then
  rok "the extra body environment: $(grep -c '^export ' "$OUT/extra_env.sh" | tr -d ' ') export(s)$( [ -n "$EXTRA_ENV" ] && echo ": $EXTRA_ENV")"
else
  rbad "MOJOLEARN_HOTAISLE_EXTRA_ENV is refused: $(grep '^# REFUSED' "$OUT/extra_env.sh" | tr '\n' ' ')"
fi

if _why=$(key_hygiene); then
  rok "key file present, 0600, outside the repository$( [ "$MODE" = dry ] && echo ' (not read by a dry run)')"
elif [ "$MODE" = dry ]; then
  printf '  info   %s (a dry run does not need it)\n' "$_why"
else
  die "REFUSING to rent: $_why" 2
fi
if ssh-keygen -lf "$SSH_KEY.pub" 2>/dev/null | grep -qF "$SSH_KEY_FP"; then
  rok "ssh key $SSH_KEY is the registered one ($SSH_KEY_FP)"
else
  rbad "ssh key $SSH_KEY.pub is missing or is not $SSH_KEY_FP"
fi
echo "  info   slots:"; list_slots | sed 's/^/       /'

# ---- the bundle ----
BUNDLE_BYTES=0; BUNDLE_SHA=none
if [ "$SHIPS_SOURCE" = 1 ]; then
  ARCHIVE_EXCLUDES=(':!bench/results' ':!mamba/corpus' ':!bench/oracle*' ':!bench/minentropy_oracle.txt' ':!*.bin')
  if git -C "$REPO" archive --format=tar -o "$TMPD/src.tar" "$COMMIT" -- . "${ARCHIVE_EXCLUDES[@]}"; then
    gzip -9 -c "$TMPD/src.tar" > "$TMPD/src.tgz"
    tar tf "$TMPD/src.tar" | grep -v '/$' > "$OUT/bundle_files.txt"
    BUNDLE_BYTES=$(wc -c < "$TMPD/src.tgz" | tr -d ' ')
    BUNDLE_SHA=$(sha256_of "$TMPD/src.tgz")
    mkdir "$TMPD/archive" && tar xf "$TMPD/src.tar" -C "$TMPD/archive"
    rm -f "$TMPD/src.tar"
    if grep -Eq '(^bench/results/|^mamba/corpus/|\.bin$)' "$OUT/bundle_files.txt"; then
      rbad "the bundle carries an excluded path"
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
    ( cd "$TMPD/archive" && find . -name '*.mojo' -not -path './.pixi/*' -not -path './bench/results/*' \
        | LC_ALL=C sort | xargs shasum -a 256 | shasum -a 256 | awk '{print $1}' ) > "$OUT/source_sha256_local.txt"
  else
    rbad "git archive of $COMMIT failed"
  fi
fi

# ---- the scripts that run on the VM ----
cat > "$TMPD/remote_body.sh.template" <<'REMOTE_BODY'
#!/bin/sh
# Generated by tools/hotaisle_leg.sh. RUNS ON THE HOT AISLE VM (in the
# container when runtime is docker or podman). The same steps, in the same
# order, as tools/do_extra_leg.sh's remote body, so a MOJOLEARN_GEMM_LEG_EXTRA
# body sees the same world. `set -u`, not `set -e`: a red gate is a result.
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out
mkdir -p "$OUT" "$ROOT"
cd "$ROOT" || exit 9
HOME=/root
export HOME

{
  echo "vendor=amd"
  echo "commit=@COMMIT@"
  echo "card_full=@CARDFULL@"
  echo "trace_dump=@DUMP@"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "provider=hotaisle"
  echo "size=@SIZE@"
  echo "runtime=@RUNTIME@"
  echo "image=@IMAGE@"
  echo "gates=@GATES@"
  echo "bare=@BARE@"
  echo "gpu_archs=@GPUARCHS@"
  echo "target_column=amd"
} > "$OUT/leg.txt"

MOJOLEARN_GPU_ARCHS="@GPUARCHS@"
export MOJOLEARN_GPU_ARCHS
MOJOLEARN_TARGET_COLUMN=amd
export MOJOLEARN_TARGET_COLUMN

uname -a > "$OUT/uname.txt" 2>&1
rocm-smi --showproductname > "$OUT/gpu.txt" 2>&1 || echo "no vendor smi tool answered" >> "$OUT/gpu.txt"

if [ "@BARE@" != "1" ]; then
    { find . -name '*.mojo' -not -path './.pixi/*' -not -path './bench/results/*' \
        | LC_ALL=C sort | xargs sha256sum ; } \
      | sha256sum | awk '{print $1}' > "$OUT/source_sha256.txt"

    if ! command -v curl > /dev/null 2>&1 && command -v apt-get > /dev/null 2>&1; then
        { apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl ca-certificates ; } \
            > "$OUT/apt_curl.log" 2>&1 < /dev/null
    fi
    if [ ! -x "$HOME/.pixi/bin/pixi" ] && ! command -v pixi > /dev/null 2>&1; then
        # No `< /dev/null` on this sh: its stdin IS the installer (a redirect
        # there made the first smoke install nothing, 2026-09-11).
        curl -fsSL https://pixi.sh/install.sh | sh > "$OUT/pixi_install.log" 2>&1
    fi
    PATH="$HOME/.pixi/bin:$PATH"
    export PATH
    command -v pixi > "$OUT/pixi_which.txt" 2>&1 || echo "NO PIXI" >> "$OUT/pixi_which.txt"

    t0=$(date +%s)
    pixi install > "$OUT/pixi_env.log" 2>&1 < /dev/null
    echo "pixi_install_exit=$?" >> "$OUT/leg.txt"
    echo "pixi_install_seconds=$(( $(date +%s) - t0 ))" >> "$OUT/leg.txt"
    pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1 < /dev/null || true

    if [ "@GATES@" = "1" ]; then
        tools/with_identical_mode.sh pixi run mojo run -I . \
            gemm/checks/gemm_device_check.mojo > "$OUT/device_check.log" 2>&1 < /dev/null
        echo "device_check_exit=$?" >> "$OUT/leg.txt"
        MOJOLEARN_GEMM_CARD_FULL="@CARDFULL@" MOJOLEARN_IDENTITY_TRACE_DUMP="@DUMP@" \
            sh tools/gemm_card.sh device "$OUT/amd.card" > "$OUT/card_driver.log" 2>&1 < /dev/null
        echo "card_exit=$?" >> "$OUT/leg.txt"
    else
        echo "device_check_exit=SKIPPED" >> "$OUT/leg.txt"
        echo "card_exit=SKIPPED" >> "$OUT/leg.txt"
    fi
fi

if [ -f /root/gemm_leg_extra.sh ]; then
    (
        if [ -f /root/gemm_leg_extra_env.sh ]; then
            . /root/gemm_leg_extra_env.sh
        fi
        sh /root/gemm_leg_extra.sh
    ) > "$OUT/extra.log" 2>&1 < /dev/null
    echo "extra_exit=$?" >> "$OUT/leg.txt"
    cp /root/gemm_leg_extra_env.sh "$OUT/extra_env.sh" 2>/dev/null || true
fi

echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/leg.txt"
: > /root/gemm_leg.done
echo REMOTE_BODY_DONE
REMOTE_BODY

cat > "$TMPD/watchdog.sh.template" <<'WATCHDOG'
#!/bin/sh
# Written by tools/hotaisle_leg.sh. RUNS ON THE VM AS ROOT, DETACHED (setsid).
# Sleeps to the lease deadline, then DELETEs THIS VM through the API with
# force. The key is in @BOXRC@ (0600), never in an argv.
set -u
trap '' HUP INT
echo $$ > @BOXDIR@/watchdog.pid
T=$(( $(date +%s) + @SECS@ ))
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) armed ref=@VMREF@ fires_in=@SECS@s" >> @BOXDIR@/watchdog.out
while [ "$(date +%s)" -lt "$T" ]; do sleep 10; done
n=1
while [ "$n" -le 20 ]; do
    code=$(curl -K @BOXRC@ --max-time 900 -o @BOXDIR@/watchdog.body -w '%{http_code}' \
        -X DELETE '@API@/teams/@TEAM@/virtual_machines/@VMREF@/?force=true')
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) DELETE @VMREF@ attempt $n -> $code" >> @BOXDIR@/watchdog.out
    case "$code" in 2*|404) exit 0 ;; esac
    sleep 15
    n=$((n + 1))
done
WATCHDOG

cat > "$TMPD/watchdog_arm.sh.template" <<'ARM'
set -u
if [ ! -s @BOXRC@ ]; then echo NO_CREDENTIAL_ON_BOX; exit 3; fi
chmod 600 @BOXRC@
chmod 700 @BOXDIR@ @BOXDIR@/watchdog.sh
rm -f @BOXDIR@/watchdog.pid
if command -v setsid > /dev/null 2>&1; then
    setsid nohup sh @BOXDIR@/watchdog.sh > @BOXDIR@/watchdog.log 2>&1 < /dev/null &
else
    nohup sh @BOXDIR@/watchdog.sh > @BOXDIR@/watchdog.log 2>&1 < /dev/null &
fi
i=0
while [ "$i" -lt 15 ] && [ ! -s @BOXDIR@/watchdog.pid ]; do sleep 1; i=$((i + 1)); done
pid=$(cat @BOXDIR@/watchdog.pid 2>/dev/null)
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo "WATCHDOG_ALIVE pid=$pid"; else echo WATCHDOG_DEAD; fi
echo "REF_BAKED_IN=$(grep -c 'virtual_machines/@VMREF@/?force=true' @BOXDIR@/watchdog.sh)"
echo "TOKEN_GET_HTTP=$(curl -K @BOXRC@ --max-time 30 -o @BOXDIR@/self.json -w '%{http_code}' '@API@/teams/@TEAM@/virtual_machines/@VMREF@/')"
if grep -q '"description": *"@DESC@"' @BOXDIR@/self.json; then echo DESC_MATCH; else echo DESC_MISMATCH; fi
ARM

cat > "$TMPD/device_probe.sh" <<'PROBE'
set -u
echo "== os";        . /etc/os-release 2>/dev/null && echo "OS=$PRETTY_NAME"; uname -r
echo "== cpu";       echo "NPROC=$(nproc)"; lscpu 2>/dev/null | grep -E '^(Model name|Socket|Core|Thread|CPU\(s\))'
echo "== memory";    free -g 2>/dev/null | head -2
echo "== disk";      df -h / /root 2>/dev/null
echo "== python";    python3 --version 2>&1
echo "== devices";   ls -l /dev/kfd /dev/dri 2>&1 | head -12
[ -e /dev/kfd ] && echo KFD_PRESENT || echo KFD_ABSENT
lsmod 2>/dev/null | grep -E '^amdgpu' | head -1
echo "== rocm";      ls -d /opt/rocm* 2>&1 | head -3
if command -v rocm-smi > /dev/null 2>&1; then echo ROCM_SMI_PRESENT; rocm-smi --showproductname 2>&1 | head -20; else echo ROCM_SMI_ABSENT; fi
if command -v rocminfo > /dev/null 2>&1; then
    echo ROCMINFO_PRESENT
    # The agent Name: field only. A bare `grep -Eo 'gfx[0-9a-f]+'` also matched
    # a stray "gfx9" on the MI300X (2026-09-11) and counted two archs.
    rocminfo 2>/dev/null | awk '$1 == "Name:" && $2 ~ /^gfx[0-9a-f]+$/ {print "GFX=" $2}' | sort -u
else
    echo ROCMINFO_ABSENT
fi
echo "== runtimes"
if command -v docker > /dev/null 2>&1 && docker info > /dev/null 2>&1; then echo DOCKER_OK; docker --version; fi
if command -v podman > /dev/null 2>&1 && podman info > /dev/null 2>&1; then echo PODMAN_OK; podman --version; fi
for t in setsid timeout curl tar gzip sha256sum; do echo "TOOL_$t=$(command -v "$t" || echo ABSENT)"; done
PROBE

cat > "$TMPD/remote_unpack.sh.template" <<'REMOTE_UNPACK'
set -eu
cd /root
got=$(sha256sum extra_src.tgz | awk '{print $1}')
if [ "$got" != "@SHA@" ]; then echo "ARCHIVE SHA MISMATCH: sent @SHA@ got $got"; exit 9; fi
echo ARCHIVE-SHA-OK
rm -rf /root/mojolearn /root/gemm_leg_out /root/gemm_leg.done
mkdir -p /root/mojolearn
tar -xzf extra_src.tgz -C /root/mojolearn
rm -f extra_src.tgz
echo "UNPACKED $(find /root/mojolearn -type f | wc -l) files"
REMOTE_UNPACK

cat > "$TMPD/remote_start.sh.template" <<'REMOTE_START'
set -u
rm -f /root/gemm_leg.done /root/gemm_leg_console.log
mkdir -p /root/gemm_leg_out /root/mojolearn
RT="@RUNTIME@"
if [ "$RT" = docker ] || [ "$RT" = podman ]; then
    "$RT" rm -f mojolearn-leg > /dev/null 2>&1
    setsid nohup sh -c '"$0" run --rm --name mojolearn-leg --device /dev/kfd --device /dev/dri \
        --security-opt seccomp=unconfined --ipc=host --network host \
        -e HOME=/root -v /root:/root -w /root/mojolearn @IMAGE@ \
        timeout -k 30 @WORK@ sh /root/gemm_leg.sh; rc=$?; "$0" rm -f mojolearn-leg > /dev/null 2>&1; \
        mkdir -p /root/gemm_leg_out; echo "body_exit=$rc" >> /root/gemm_leg_out/leg.txt' "$RT" \
        > /root/gemm_leg_console.log 2>&1 < /dev/null &
else
    setsid nohup sh -c 'timeout -k 30 @WORK@ sh /root/gemm_leg.sh; rc=$?; mkdir -p /root/gemm_leg_out; echo "body_exit=$rc" >> /root/gemm_leg_out/leg.txt' \
        > /root/gemm_leg_console.log 2>&1 < /dev/null &
fi
echo "REMOTE_PID=$!"
REMOTE_START

cat > "$TMPD/pull_start.sh.template" <<'PULL'
set -u
rm -f /root/mojolearn-pull.done
setsid nohup sh -c 't0=$(date +%s); "$0" pull @IMAGE@ > /root/mojolearn-pull.log 2>&1; rc=$?; echo "pull_exit=$rc pull_seconds=$(( $(date +%s) - t0 ))" > /root/mojolearn-pull.done' "@RUNTIME@" \
    > /dev/null 2>&1 < /dev/null &
echo PULL_STARTED
PULL

cat > "$TMPD/deadman.sh.template" <<'DEADMAN'
#!/bin/sh
# Written by tools/hotaisle_leg.sh. RUNS ON THIS MAC, DETACHED. Ends the VM
# that leg created if the leg is no longer here to do it. Keyed by the VM ref
# the leg writes beside this file as soon as the create returns it. The key
# is in the 0600 curl config beside this file and in no argv.
set -u
trap '' HUP INT TERM
D='@DMDIR@'
L="$D/deadman.log"
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
while [ "$(date +%s)" -lt @DEADLINE@ ]; do sleep 20; done
echo "$(now) dead-man firing" >> "$L"
ref=""
[ -s "$D/vm_ref.txt" ] && ref="$(cat "$D/vm_ref.txt")"
if [ -z "$ref" ]; then
    echo "mac_deadman_fired $(now) no VM ref was recorded; nothing to delete" >> '@RECORD@'
    rm -f "$D/curlrc"; exit 0
fi
c=$(curl -K "$D/curlrc" --max-time 60 -o "$D/vm.json" -w '%{http_code}' "@API@/teams/@TEAM@/virtual_machines/$ref/" 2>> "$L")
if [ "$c" = 200 ]; then
    d=$(python3 "$D/j.py" desc "$D/vm.json")
    if [ -n "$d" ] && [ "$d" != '@DESC@' ]; then
        echo "mac_deadman_fired $(now) REFUSED ref=$ref: description '$d' is not '@DESC@'" >> '@RECORD@'
        rm -f "$D/curlrc"; exit 0
    fi
    n=1
    while [ "$n" -le 6 ]; do
        c=$(curl -K "$D/curlrc" --max-time 900 -o /dev/null -w '%{http_code}' -X DELETE "@API@/teams/@TEAM@/virtual_machines/$ref/?force=true" 2>> "$L")
        echo "$(now) DELETE $ref attempt $n -> $c" >> "$L"
        case "$c" in 2*|404) break ;; esac
        sleep 15; n=$((n + 1))
    done
fi
gone=0; n=1; g=""; l=""; inl=""
while [ "$n" -le 40 ]; do
    g=$(curl -K "$D/curlrc" --max-time 60 -o /dev/null -w '%{http_code}' "@API@/teams/@TEAM@/virtual_machines/$ref/" 2>> "$L")
    l=$(curl -K "$D/curlrc" --max-time 60 -o "$D/list.json" -w '%{http_code}' "@API@/teams/@TEAM@/virtual_machines/" 2>> "$L")
    inl=$(python3 "$D/j.py" inlist "$D/list.json" "$ref")
    if [ "$g" = 404 ] || { [ "$l" = 200 ] && [ "$inl" = no ]; }; then gone=1; break; fi
    sleep 15; n=$((n + 1))
done
echo "mac_deadman_fired $(now) ref=$ref verified_gone=$gone get=$g list=$l listed=$inl" >> '@RECORD@'
if [ "$gone" = 1 ] && grep -qx 'nonce=@NONCE@' '@SLOT@/owner' 2>/dev/null && ! kill -0 @LEGPID@ 2>/dev/null; then
    rm -rf '@SLOT@' && echo "mac_deadman released @SLOT@" >> '@RECORD@'
fi
rm -f "$D/curlrc"
DEADMAN

subst() {  # <template> <out>: replace every placeholder, then prove none survived
  sed -e "s|@COMMIT@|$COMMIT|g" -e "s|@CARDFULL@|$CARD_FULL|g" -e "s|@DUMP@|$LEG_DUMP|g" \
      -e "s|@SIZE@|$SIZE_LABEL|g" -e "s|@RUNTIME@|${RUNTIME:-native}|g" -e "s|@IMAGE@|$IMAGE|g" \
      -e "s|@GATES@|$GATES|g" -e "s|@BARE@|$BARE|g" -e "s|@GPUARCHS@|${S_ARCHS:-$GPU_ARCHS}|g" \
      -e "s|@WORK@|${WORK_SECONDS:-0}|g" -e "s|@SHA@|$BUNDLE_SHA|g" -e "s|@SECS@|${S_SECS:-60}|g" \
      -e "s|@VMREF@|${VMREF:-DRYRUN_REF}|g" -e "s|@API@|$API|g" -e "s|@TEAM@|$TEAM|g" \
      -e "s|@DESC@|${DESC:-mojolearn:dry:run}|g" -e "s|@BOXRC@|$BOX_RC|g" -e "s|@BOXDIR@|$BOX_DIR|g" \
      -e "s|@DMDIR@|${DEADMAN_DIR:-/tmp/dry}|g" -e "s|@DEADLINE@|${S_DEADLINE:-0}|g" \
      -e "s|@RECORD@|$OUT/deadman.txt|g" -e "s|@NONCE@|$NONCE|g" -e "s|@SLOT@|${SLOT:-/tmp/dry-slot}|g" \
      -e "s|@LEGPID@|$$|g" \
      "$1" > "$2"
  if grep -q '@[A-Z][A-Z_]*@' "$2"; then grep -n '@[A-Z][A-Z_]*@' "$2" | sed 's/^/           /'; return 1; fi
  return 0
}
check_posix() {  # <file> <label>
  sh -n "$1" 2> "$TMPD/sh_n.err" || { rbad "$2 is not valid sh: $(head -2 "$TMPD/sh_n.err")"; return 1; }
  if command -v dash > /dev/null 2>&1; then
    dash -n "$1" 2> "$TMPD/sh_n.err" || { rbad "$2 is not valid dash: $(head -2 "$TMPD/sh_n.err")"; return 1; }
  fi
  awk '!/^[[:space:]]*#/ && (/exec -a/ || /\[\[/ || /(^|[;{ \t])local / || /<\(/ || /(^|[;{ \t])function / || /(^|[;{ \t])source / || /echo -e/) { print FNR ": " $0 }' "$1" > "$TMPD/bashisms"
  [ -s "$TMPD/bashisms" ] && { rbad "$2 has a BASHISM (the VM runs dash):"; sed 's/^/           /' "$TMPD/bashisms"; return 1; }
  return 0
}
_ok=1
for _t in remote_body watchdog watchdog_arm remote_unpack remote_start pull_start deadman; do
  if subst "$TMPD/$_t.sh.template" "$TMPD/check_$_t.sh"; then
    check_posix "$TMPD/check_$_t.sh" "$_t.sh" || _ok=0
  else
    rbad "UNSUBSTITUTED PLACEHOLDER in $_t.sh"; _ok=0
  fi
done
check_posix "$TMPD/device_probe.sh" device_probe.sh || _ok=0
[ "$_ok" = 1 ] && rok "the remote body, watchdog, arm, unpack, start, pull and Mac dead-man scripts substitute cleanly and pass sh -n, dash -n and the bashism scan"
cp "$TMPD/check_remote_body.sh" "$OUT/remote_body.sh"

{
  echo "commit=$COMMIT_LINE"
  echo "commit_sha=$COMMIT"
  echo "provider=hotaisle"
  echo "vendor=amd"
  echo "team=$TEAM"
  echo "spec=$SPEC"
  echo "lane=$LANE"
  echo "minutes=$MINUTES"
  echo "gates=$GATES"
  echo "bare=$BARE"
  echo "test_watchdog=$TEST_WATCHDOG"
  echo "gpu_archs_requested=${GPU_ARCHS:-<unset>}"
  echo "target_column=amd"
  echo "runtime_requested=$RUNTIME_WANT"
  echo "image=$IMAGE"
  echo "extra=$LEG_EXTRA"
  echo "extra_sha256=$EXTRA_SHA"
  echo "bundle_bytes=$BUNDLE_BYTES"
  echo "bundle_sha256=$BUNDLE_SHA"
  echo "source_sha256_local=$(cat "$OUT/source_sha256_local.txt" 2>/dev/null)"
  echo "mode=$MODE"
  echo "started=$(utc)"
} > "$OUT/leg.txt"

if [ "$MODE" = dry ]; then
  echo
  echo "== the remote body (/root/gemm_leg.sh) =="; cat "$OUT/remote_body.sh"
  echo; echo "== the on-box watchdog ($BOX_DIR/watchdog.sh; ref and seconds filled at arm time) =="; cat "$TMPD/check_watchdog.sh"
  echo; echo "== the start wrapper =="; cat "$TMPD/check_remote_start.sh"
  echo
  echo "== what --rent does, in order =="
  echo "   1. refuse a dirty tree (when source ships), a bad key file, a broken script, an oversized bundle"
  echo "   2. GET teams (operator role, VM limit), take a slot (/tmp/mojolearn-hotaisle-slot.N), balance >= $MIN_BALANCE_CENTS cents"
  echo "   3. wait for Quantity > 0 on the $SPEC spec (up to $STOCK_WAIT_MINUTES min); print price and balance"
  echo "   4. ARM THE MAC DEAD-MAN, then under the create lock: snapshot, POST   [THE BILL STARTS HERE]"
  echo "   5. PATCH description mojolearn:$LANE:<utc>, verify it; wait for running; ssh settle as hotaisle; sudo -n"
  echo "   6. key to $BOX_RC on stdin; arm the watchdog; verify pid (two sessions), ref, GET 200 + description"
  echo "   7. key-in-ps both ends; device probe; runtime (docker/podman/native); GPU arch from rocminfo"
  echo "   8. image pull in the background; stream the bundle over ssh stdin, sha256 check, unpack"
  echo "   9. body under timeout(1) at the work bound; poll 30 s; fetch /root/gemm_leg_out -> <out>/remote/"
  echo "  10. DELETE ?force=true; verify GET 404 or absent from the listing; cancel the dead-man; release the slot"
  echo "   dry-run artifacts kept in $OUT"
  [ "$RED" = 1 ] && { echo "DRY RUN: RED. This script is broken (a FAIL above). Nothing rented."; exit 1; }
  [ "$BLOCK" = 1 ] && { echo "DRY RUN: plumbing GREEN, and a real leg is BLOCKED (see BLOCK above). Nothing rented."; exit 3; }
  echo "DRY RUN: GREEN. Nothing rented."
  exit 0
fi

# ------------------------------------------------------ from here it can bill
[ "$RED" = 1 ] && die "REFUSING to rent: a local check FAILED above." 1
[ "$BLOCK" = 1 ] && die "REFUSING to rent: a local check BLOCKED above." 3
load_key

echo
echo "== pre-flight =="
c=$(api GET "teams/" "$TMPD/teams.json")
[ "$c" = 200 ] || die "REFUSING to rent: GET teams/ -> HTTP $c. Nothing was created." 2
read -r _op TEAM_MAX_VMS < <(J teams "$TMPD/teams.json" "$TEAM")
[ "$_op" = yes ] || die "REFUSING to rent: this key lacks the operator role on $TEAM (create and DELETE need it). Nothing was created." 2
BAL_BEFORE=$(balance_cents)
log "team $TEAM balance at the start $(dollars "$BAL_BEFORE") ($BAL_BEFORE cents); VM limit $TEAM_MAX_VMS, slots $(slot_cap)"
echo "balance_before_cents=$BAL_BEFORE" >> "$OUT/leg.txt"
[ "$BAL_BEFORE" -ge "$MIN_BALANCE_CENTS" ] 2>/dev/null \
  || die "REFUSED: balance $(dollars "$BAL_BEFORE") is below the \$5.00 floor ($MIN_BALANCE_CENTS cents). Nothing was created." 3
c=$(api GET "user/ssh_keys/" "$TMPD/keys.json")
[ "$c" = 200 ] && [ "$(J sshkey "$TMPD/keys.json" "$SSH_KEY_FP")" = yes ] \
  || die "REFUSING to rent: $SSH_KEY_FP is not registered on the account (HTTP $c); ssh would fail. Nothing was created." 2

# ---- a. the slot ----
_t0=$(date +%s); _said=0
while :; do
  if try_take_slot "$(slot_cap)"; then
    # Capacity is the team's, not only this Mac's: a VM outside the slots counts.
    if [ "$(api GET "teams/$TEAM/virtual_machines/" "$TMPD/cap.json")" = 200 ] \
       && [ "$(J count "$TMPD/cap.json")" -lt "$TEAM_MAX_VMS" ]; then
      break
    fi
    log "slot taken but the team already runs $(J count "$TMPD/cap.json") of $TEAM_MAX_VMS VMs; releasing and waiting"
    release_slot
  fi
  [ "$_said" = 0 ] && { log "all slots busy (or the team is at its VM limit); polling every 60 s"; list_slots; _said=1; }
  check_stale_slots
  [ $(( $(date +%s) - _t0 )) -lt $(( SLOT_WAIT_MINUTES * 60 )) ] \
    || die "REFUSED: no slot freed in $SLOT_WAIT_MINUTES minutes. Nothing was created." 3
  nap 60
done
echo "slot=$SLOT taken $(utc)" >> "$OUT/leg.txt"
log "slot $SLOT taken (lane $LANE)"

# ---- b. stock ----
_t0=$(date +%s)
while :; do
  c=$(api GET "teams/$TEAM/virtual_machines/available/" "$TMPD/avail.json")
  read -r _f _qty _price _minres < <(J pick "$TMPD/avail.json" "$SPEC_CORES" "$TMPD/create_request.json")
  if [ "$c" = 200 ] && [ "$_f" = found ] && [ "$_qty" -gt 0 ]; then break; fi
  [ $(( $(date +%s) - _t0 )) -lt $(( STOCK_WAIT_MINUTES * 60 )) ] \
    || die "REFUSED: the $SPEC 1x MI300X spec showed no stock for $STOCK_WAIT_MINUTES minutes (last HTTP $c, $_f, quantity ${_qty:-0}). Nothing was created." 3
  log "no stock on $SPEC (HTTP $c, $_f, quantity ${_qty:-0}); retrying in 60 s"
  nap 60
done
[ "$_minres" -le 10 ] || die "REFUSED: the $SPEC spec has MinimumReservationMinutes $_minres. Nothing was created." 3
cp "$TMPD/create_request.json" "$OUT/create_request.json"
cp "$TMPD/avail.json" "$OUT/offering.json"
log "spec $SPEC: quantity $_qty, $_price cents/hour, minimum reservation $_minres min; $MINUTES min costs at most $(dollars $(( _price * MINUTES / 60 + 1 )))"
echo "price_cents_per_hour=$_price min_reservation_minutes=$_minres" >> "$OUT/leg.txt"

# ---- c. the Mac dead-man, BEFORE the create ----
LEG_START=$(date +%s)
DEADLINE_EPOCH=$((LEG_START + MINUTES * 60))
S_DEADLINE=$DEADLINE_EPOCH
[ "$TEST_WATCHDOG" = 1 ] && S_DEADLINE=$((DEADLINE_EPOCH + 600))
DEADMAN_DIR="${TMPDIR:-/tmp}/mojolearn-hotaisle-deadman-$$"
DEADMAN_DIR="${DEADMAN_DIR//\/\//\/}"
DESC="mojolearn:$LANE:$(date -u +%Y%m%dT%H%M%SZ)"
( umask 077; mkdir -p "$DEADMAN_DIR" )
cp "$CURLRC" "$DEADMAN_DIR/curlrc"; chmod 600 "$DEADMAN_DIR/curlrc"
cp "$TMPD/j.py" "$DEADMAN_DIR/j.py"
subst "$TMPD/deadman.sh.template" "$DEADMAN_DIR/deadman.sh" || die "THE MAC DEAD-MAN DID NOT BUILD. Nothing was created." 1
nohup sh -c 'trap "" HUP INT TERM; exec sh "$0"' "$DEADMAN_DIR/deadman.sh" > /dev/null 2>&1 < /dev/null &
DEADMAN_PID=$!
disown "$DEADMAN_PID" 2>/dev/null  # no "Killed: 9" job line when a clean teardown cancels it
sleep 1
kill -0 "$DEADMAN_PID" 2>/dev/null || { DEADMAN_PID=""; die "THE MAC DEAD-MAN DID NOT START. Nothing was created." 1; }
{
  echo "mac_deadman_pid=$DEADMAN_PID"
  echo "mac_deadman_dir=$DEADMAN_DIR"
  echo "mac_deadman_fires_at=$(utc_of "$S_DEADLINE")$( [ "$TEST_WATCHDOG" = 1 ] && echo ' (TEST-WATCHDOG: late backstop, deadline + 600 s)')"
  echo "mac_deadman_keyed_by=vm_ref.txt (deployment_id from the create response) and description $DESC"
} > "$OUT/deadman.txt"
log "Mac dead-man ARMED before the create: pid $DEADMAN_PID, fires at $(utc_of "$S_DEADLINE")"
if ps -axo command= 2>/dev/null | grep -q -F -f "$TOKPAT"; then
  KEY_RED=1; log "!! THE KEY IS VISIBLE IN THIS MAC'S PROCESS LIST"; echo "local_key_in_ps=VISIBLE" >> "$OUT/leg.txt"
else
  echo "local_key_in_ps=not_visible" >> "$OUT/leg.txt"
fi

# ---- the create, serialized ----
_t0=$(date +%s)
until mkdir "$CREATE_LOCK" 2>/dev/null; do
  _age=$(slot_age "$CREATE_LOCK") || _age=0
  if [ "$_age" -gt 900 ]; then
    log "breaking a stale create lock (${_age}s): $(tr '\n' ' ' < "$CREATE_LOCK/owner" 2>/dev/null)"
    rm -rf "$CREATE_LOCK"; continue
  fi
  [ $(( $(date +%s) - _t0 )) -lt 900 ] || die "REFUSED: the create lock stayed held for 15 minutes. Nothing was created." 3
  nap 5
done
CREATE_LOCK_HELD=1
{ echo "nonce=$NONCE"; echo "pid=$$"; echo "lane=$LANE"; echo "utc=$(utc)"; } > "$CREATE_LOCK/owner"
c=$(api GET "teams/$TEAM/virtual_machines/" "$TMPD/pre.json")
[ "$c" = 200 ] || die "REFUSING to rent: the pre-create listing returned HTTP $c. Nothing was created." 2
J ids "$TMPD/pre.json" | awk '{print $1}' > "$TMPD/pre_ids.txt"

echo
echo "== the VM =="
log "creating 1x MI300X $SPEC"
CREATE_ATTEMPTED=1
c=$(api POST "teams/$TEAM/virtual_machines/" "$OUT/create_response.json" 300 "$TMPD/create_request.json")
redact "$OUT/create_response.json"
echo "create_http=$c create_utc=$(utc)" >> "$OUT/leg.txt"
case "$c" in
  401|402|403|404|428)
    log "create refused: HTTP $c: $(head -c 300 "$OUT/create_response.json")"
    nap 10
    adopt_new_vm || die "create REFUSED by the API (HTTP $c) and no new VM appears. Nothing is billing." 4 ;;
esac
if [ -z "$VMREF" ]; then
  IFS=$'\t' read -r VMNAME DEPLOY_ID _ip _port _d < <(J vm "$OUT/create_response.json")
  VMREF="$DEPLOY_ID"
fi
if [ -z "$VMREF" ]; then
  log "create returned no deployment_id (HTTP $c): $(head -c 300 "$OUT/create_response.json")"
  for _i in $(seq 1 18); do adopt_new_vm && break; nap 10; done
  [ -n "$VMREF" ] || die "create FAILED (no deployment_id and no new VM after 3 minutes). The teardown looks once more." 4
  log "ADOPTED $VMREF ($VMNAME) by listing diff after an unreadable create"
  echo "adopted_by_listing_diff=1" >> "$OUT/leg.txt"
fi
case "$VMREF" in *[!A-Za-z0-9_.-]*) die "the VM ref '$VMREF' has unexpected characters" 4 ;; esac
printf '%s\n' "$VMREF" > "$DEADMAN_DIR/vm_ref.txt"
printf 'vm_ref=%s\nvm_name=%s\n' "$VMREF" "$VMNAME" >> "$SLOT/owner"
printf 'vm_ref=%s\nvm_name=%s\n' "$VMREF" "$VMNAME" >> "$OUT/leg.txt"
log "VM $VMNAME deployment_id $VMREF (Mac dead-man now keyed by it)"

# {vm} is "name or deployment ID": prove the deployment_id form answers, else use the name.
c=$(api GET "teams/$TEAM/virtual_machines/$VMREF/" "$TMPD/vm.json")
if [ "$c" != 200 ] && [ -n "$VMNAME" ]; then
  c2=$(api GET "teams/$TEAM/virtual_machines/$VMNAME/" "$TMPD/vm.json")
  if [ "$c2" = 200 ]; then
    log "GET by deployment_id -> $c, by name -> 200: using the name as the ref"
    VMREF="$VMNAME"; printf '%s\n' "$VMREF" > "$DEADMAN_DIR/vm_ref.txt"; echo "vm_ref_is_name=1" >> "$OUT/leg.txt"
  fi
fi

# ---- g. the description ----
printf '{"description":"%s"}\n' "$DESC" > "$TMPD/patch.json"
_tagged=0
for _i in 1 2 3 4 5; do
  c=$(api PATCH "teams/$TEAM/virtual_machines/$VMREF/" "$TMPD/patch.out" 60 "$TMPD/patch.json")
  if [ "$(api GET "teams/$TEAM/virtual_machines/$VMREF/" "$TMPD/vm.json")" = 200 ] && [ "$(J desc "$TMPD/vm.json")" = "$DESC" ]; then
    _tagged=1; break
  fi
  log "PATCH description -> HTTP $c, not yet visible; retrying"
  nap 5
done
release_create_lock
[ "$_tagged" = 1 ] || die "the description PATCH never landed; deleting the VM unused" 5
echo "description=$DESC" >> "$OUT/leg.txt"
log "description $DESC"

_t0=$(date +%s); _state=unknown
while [ $(( $(date +%s) - _t0 )) -lt 600 ]; do
  [ "$(api GET "teams/$TEAM/virtual_machines/$VMREF/state/" "$TMPD/state.json")" = 200 ] && _state=$(J state "$TMPD/state.json")
  echo "$(utc) state=$_state" >> "$OUT/vm_states.txt"
  [ "$_state" = running ] && break
  nap 10
done
[ "$_state" = running ] || die "the VM never reached running (last state $_state)" 5
api GET "teams/$TEAM/virtual_machines/$VMREF/" "$TMPD/vm.json" > /dev/null
IFS=$'\t' read -r _n _id SSH_IP SSH_PORT _d < <(J vm "$TMPD/vm.json")
cp "$TMPD/vm.json" "$OUT/vm_details.json"; redact "$OUT/vm_details.json"
[ -n "$SSH_IP" ] || die "the VM has no ssh address" 5
case "$SSH_PORT" in ''|*[!0-9]*) SSH_PORT=22 ;; esac
log "running after $(( $(date +%s) - LEG_START ))s; ssh hotaisle@$SSH_IP -p $SSH_PORT"
echo "ssh=hotaisle@$SSH_IP:$SSH_PORT running_after_seconds=$(( $(date +%s) - LEG_START ))" >> "$OUT/leg.txt"

SSH_OPTS=(-p "$SSH_PORT" -i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new
          -o "UserKnownHostsFile=$TMPD/known_hosts" -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=4)
SSH=(ssh "${SSH_OPTS[@]}" "hotaisle@$SSH_IP")
SSHN=(ssh -n "${SSH_OPTS[@]}" "hotaisle@$SSH_IP")
_ok=0
for _i in $(seq 1 90); do
  if "${SSHN[@]}" true 2>/dev/null; then _ok=$((_ok + 1)); [ "$_ok" -ge 3 ] && break; else _ok=0; fi
  nap 5
done
[ "$_ok" -ge 3 ] || die "ssh never settled on $SSH_IP:$SSH_PORT" 5
log "ssh settled after $(( $(date +%s) - LEG_START ))s"
"${SSHN[@]}" 'sudo -n true && echo SUDO_OK' > "$TMPD/sudo.out" 2>&1
grep -q SUDO_OK "$TMPD/sudo.out" || die "passwordless sudo is not available for hotaisle ($(head -c 200 "$TMPD/sudo.out")); deleting unused" 6

# Remote helpers. rexec: the script is saved as the hotaisle user's mktemp file
# and run as root with stdin from /dev/null. rput: a file to a root path.
rexec() { "${SSH[@]}" 'f=$(mktemp) && cat > "$f" && sudo -n -H sh "$f" < /dev/null; rc=$?; rm -f "$f"; exit $rc' < "$1"; }
rput() { "${SSH[@]}" "sudo -n sh -c 'umask 077; mkdir -p \$(dirname $2); cat > $2 && chmod $3 $2'" < "$1"; }

# ---- d. the on-box watchdog, before any work ----
S_SECS=$((DEADLINE_EPOCH - $(date +%s)))
[ "$S_SECS" -ge 60 ] || S_SECS=60
subst "$TMPD/watchdog.sh.template" "$OUT/watchdog.sh" || die "the watchdog did not substitute" 1
subst "$TMPD/watchdog_arm.sh.template" "$TMPD/watchdog_arm.sh" || die "the watchdog arm did not substitute" 1
rput "$CURLRC" "$BOX_RC" 600 || die "could not deliver the key for the watchdog; deleting unused" 6
rput "$OUT/watchdog.sh" "$BOX_DIR/watchdog.sh" 700 || die "could not deliver the watchdog; deleting unused" 6
rexec "$TMPD/watchdog_arm.sh" > "$TMPD/arm.out" 2>&1
sed 's/^/    /' "$TMPD/arm.out"
_wpid=$(sed -n 's/^WATCHDOG_ALIVE pid=//p' "$TMPD/arm.out" | tr -d '\r')
nap 3
"${SSHN[@]}" "sudo -n sh -c 'kill -0 $_wpid 2>/dev/null && echo WATCHDOG_STILL_ALIVE_SECOND_SESSION'" > "$TMPD/arm2.out" 2>&1
sed 's/^/    /' "$TMPD/arm2.out"
{
  echo "watchdog_seconds=$S_SECS"
  echo "watchdog_fires_at=$(utc_of $(( $(date +%s) + S_SECS )))"
  grep -E '^(WATCHDOG_|REF_BAKED_IN=|TOKEN_GET_HTTP=|DESC_)' "$TMPD/arm.out" "$TMPD/arm2.out" | sed 's/^[^:]*://; s/^/watchdog_/'
} >> "$OUT/deadman.txt"
if ! grep -q '^WATCHDOG_ALIVE' "$TMPD/arm.out" || ! grep -q '^REF_BAKED_IN=[1-9]' "$TMPD/arm.out" \
   || ! grep -q '^TOKEN_GET_HTTP=200' "$TMPD/arm.out" || ! grep -q '^DESC_MATCH' "$TMPD/arm.out" \
   || ! grep -q WATCHDOG_STILL_ALIVE_SECOND_SESSION "$TMPD/arm2.out"; then
  die "THE ON-BOX WATCHDOG COULD NOT BE VERIFIED (pid, second session, ref, GET 200 or description). Deleting the VM unused." 6
fi
WATCHDOG_OK=1
log "on-box watchdog ARMED and verified (pid $_wpid alive in two sessions, ref $VMREF, GET 200, description matches, ${S_SECS}s)"

rput "$TOKPAT" "$BOX_DIR/key.pattern" 600
printf '%s\n' "ps -eo args= > $BOX_DIR/ps.txt 2>/dev/null || ps ax > $BOX_DIR/ps.txt" \
  "if grep -q -F -f $BOX_DIR/key.pattern $BOX_DIR/ps.txt; then echo KEY_VISIBLE_IN_PS; else echo KEY_NOT_IN_PS; fi" \
  "rm -f $BOX_DIR/ps.txt $BOX_DIR/key.pattern" > "$TMPD/ps.sh"
rexec "$TMPD/ps.sh" > "$TMPD/ps.out" 2>&1
if grep -q KEY_NOT_IN_PS "$TMPD/ps.out"; then
  echo "box_key_in_ps=not_visible" >> "$OUT/leg.txt"
else
  KEY_RED=1; echo "box_key_in_ps=$(tr '\n' ' ' < "$TMPD/ps.out")" >> "$OUT/leg.txt"
  log "!! THE KEY IS VISIBLE (or unverifiable) IN THE VM'S PROCESS LIST"
fi

# ---- device probe, runtime, arch ----
rexec "$TMPD/device_probe.sh" > "$OUT/device_probe.txt" 2>&1
sed 's/^/    [box] /' "$OUT/device_probe.txt"
RUNTIME=native
case "$RUNTIME_WANT" in
  auto) if grep -q '^DOCKER_OK' "$OUT/device_probe.txt"; then RUNTIME=docker
        elif grep -q '^PODMAN_OK' "$OUT/device_probe.txt"; then RUNTIME=podman; fi ;;
  docker|podman) grep -q "^$(echo "$RUNTIME_WANT" | tr a-z A-Z)_OK" "$OUT/device_probe.txt" \
                   || die "MOJOLEARN_HOTAISLE_RUNTIME=$RUNTIME_WANT but $RUNTIME_WANT does not answer on the VM" 6
                 RUNTIME=$RUNTIME_WANT ;;
esac
[ "$BARE" = 1 ] && RUNTIME=native
grep -q '^KFD_PRESENT' "$OUT/device_probe.txt" || die "/dev/kfd is absent on the VM: no AMD compute device. Deleting." 6
echo "runtime=$RUNTIME" >> "$OUT/leg.txt"
log "runtime $RUNTIME"
if [ "$TEST_WATCHDOG" = 0 ] && [ "$RUNTIME" != native ]; then
  subst "$TMPD/pull_start.sh.template" "$TMPD/pull_start.sh" && rexec "$TMPD/pull_start.sh" > "$TMPD/pull.out" 2>&1
  log "image pull started in the background: $(tr '\n' ' ' < "$TMPD/pull.out")"
fi
box_archs() { sed -n 's/^GFX=//p' "$1" | sort -u; }
BOX_ARCHS="$(box_archs "$OUT/device_probe.txt")"
echo "box_gfx=$(echo $BOX_ARCHS)" >> "$OUT/leg.txt"

# ---- --test-watchdog: nothing ships; the watchdog must end the VM ----
if [ "$TEST_WATCHDOG" = 1 ]; then
  echo
  echo "== TEST-WATCHDOG: the Mac sends no delete; waiting for the on-box watchdog =="
  while [ "$(date +%s)" -lt "$DEADLINE_EPOCH" ]; do nap 15; done
  log "deadline reached; polling for the VM to be gone (8 minutes grace)"
  _t0=$(date +%s)
  while :; do
    if _line=$(gone_check "$VMREF" "$VMNAME"); then
      echo "watchdog_test=PASS verified_gone ref=$VMREF $_line utc=$(utc) after_deadline=$(( $(date +%s) - DEADLINE_EPOCH ))s" | tee -a "$OUT/teardown.txt" "$OUT/leg.txt"
      DESTROY_CONFIRMED=1; WDT_DONE=1
      exit 0
    fi
    echo "$(utc) $_line" >> "$OUT/vm_states.txt"
    if [ $(( $(date +%s) - _t0 )) -ge 480 ]; then
      echo "watchdog_test=FAIL still present $_line utc=$(utc); the trap deletes it now" | tee -a "$OUT/teardown.txt" "$OUT/leg.txt"
      WDT_DONE=1
      exit 1
    fi
    nap 15
  done
fi

# ---- the arch ----
if [ -z "$GPU_ARCHS" ] && [ -z "$BOX_ARCHS" ] && [ "$RUNTIME" != native ]; then
  log "no host rocminfo; waiting for the image to read the arch from inside it"
  while [ "$(date +%s)" -lt $((DEADLINE_EPOCH - FETCH_RESERVE - 300)) ]; do
    "${SSHN[@]}" 'sudo -n test -f /root/mojolearn-pull.done && echo PULL_DONE' 2>/dev/null | grep -q PULL_DONE && break
    nap 15
  done
  printf '%s\n' "$RUNTIME run --rm --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined $IMAGE sh -c 'rocminfo 2>/dev/null | grep -Eo \"gfx[0-9a-f]+\" | sort -u | sed \"s/^/GFX=/\"'" > "$TMPD/arch.sh"
  rexec "$TMPD/arch.sh" > "$OUT/arch_from_image.txt" 2>&1
  BOX_ARCHS="$(box_archs "$OUT/arch_from_image.txt")"
fi
_n_archs=$(printf '%s\n' "$BOX_ARCHS" | grep -c .)
if [ -z "$GPU_ARCHS" ]; then
  [ "$_n_archs" = 1 ] || die "MOJOLEARN_GPU_ARCHS is unset and rocminfo on the VM gave $_n_archs gfx names ('$(echo $BOX_ARCHS)'); exactly one is required. Deleting." 6
  S_ARCHS="$BOX_ARCHS"
  log "MOJOLEARN_GPU_ARCHS=$S_ARCHS read from rocminfo"
else
  S_ARCHS="$GPU_ARCHS"
  if [ "$_n_archs" -ge 1 ] && ! printf '%s\n' "$BOX_ARCHS" | grep -qx "$GPU_ARCHS"; then
    die "MOJOLEARN_GPU_ARCHS=$GPU_ARCHS but rocminfo on the VM reports '$(echo $BOX_ARCHS)'. One mojo build is one GPU arch. Deleting." 6
  fi
fi
echo "gpu_archs=$S_ARCHS" >> "$OUT/leg.txt"

# ---- the source ----
with_deadline() {  # <seconds> <stdin file> <cmd...>: the command's status, or 124 at the deadline
  # stdin is an explicit redirect on the async command: without one, bash
  # gives a background job /dev/null, whatever the caller redirected.
  local secs=$1 infile=$2 pid waited=0
  shift 2
  "$@" < "$infile" &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$waited" -ge "$secs" ]; then pkill -P "$pid" 2>/dev/null; kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; return 124; fi
    sleep 1; waited=$((waited + 1))
  done
  wait "$pid"
}
if [ "$SHIPS_SOURCE" = 1 ]; then
  echo
  echo "== the source =="
  _up0=$(date +%s); _up_ok=0
  for _try in 1 2 3; do
    _left=$((DEADLINE_EPOCH - FETCH_RESERVE - $(date +%s)))
    [ "$_left" -gt 120 ] || break
    if with_deadline "$_left" "$TMPD/src.tgz" "${SSH[@]}" "sudo -n sh -c 'cat > /root/extra_src.tgz'"; then _up_ok=1; break; fi
    log "upload attempt $_try failed; retrying in 15 s"
    nap 15
  done
  [ "$_up_ok" = 1 ] || die "the bundle upload failed" 7
  _up_s=$(( $(date +%s) - _up0 ))
  log "uploaded $BUNDLE_BYTES bytes in ${_up_s}s over ssh stdin"
  echo "upload_seconds=$_up_s" >> "$OUT/leg.txt"
  subst "$TMPD/remote_unpack.sh.template" "$TMPD/remote_unpack.sh" || die "the unpack script did not substitute" 1
  rexec "$TMPD/remote_unpack.sh" > "$TMPD/unpack.out" 2>&1
  sed 's/^/    /' "$TMPD/unpack.out"
  grep -q '^ARCHIVE-SHA-OK' "$TMPD/unpack.out" && grep -q '^UNPACKED ' "$TMPD/unpack.out" || die "the VM refused or failed to unpack the bundle" 7
else
  printf 'rm -rf /root/mojolearn /root/gemm_leg_out /root/gemm_leg.done\nmkdir -p /root/mojolearn\n' > "$TMPD/bare_unpack.sh"
  rexec "$TMPD/bare_unpack.sh" > /dev/null 2>&1
fi
subst "$TMPD/remote_body.sh.template" "$OUT/remote_body.sh" || die "the remote body did not substitute" 1
rput "$OUT/extra_body.sh" /root/gemm_leg_extra.sh 644 || die "could not ship the extra body" 7
rput "$OUT/extra_env.sh" /root/gemm_leg_extra_env.sh 644 || die "could not ship the extra body environment" 7
rput "$OUT/remote_body.sh" /root/gemm_leg.sh 644 || die "could not ship the remote body" 7
log "shipped the extra body ($LEG_EXTRA, sha256 ${EXTRA_SHA:0:16}) and the remote body"

if [ "$RUNTIME" != native ]; then
  log "waiting for the image pull"
  while [ "$(date +%s)" -lt $((DEADLINE_EPOCH - FETCH_RESERVE - 180)) ]; do
    "${SSHN[@]}" 'sudo -n cat /root/mojolearn-pull.done 2>/dev/null' > "$TMPD/pull.done" 2>/dev/null
    grep -q pull_exit= "$TMPD/pull.done" && break
    nap 15
  done
  _pull="$(tr -d '\r' < "$TMPD/pull.done" 2>/dev/null)"
  echo "image_pull=${_pull:-NOT_DONE}" >> "$OUT/leg.txt"
  log "image pull: ${_pull:-NOT DONE}"
  case "$_pull" in
    *pull_exit=0*) ;;
    *) log "!! the image pull did not finish cleanly; falling back to native"; RUNTIME=native; echo "runtime=native (image pull failed)" >> "$OUT/leg.txt"
       subst "$TMPD/remote_body.sh.template" "$OUT/remote_body.sh" && rput "$OUT/remote_body.sh" /root/gemm_leg.sh 644 ;;
  esac
fi

# ---- the work, detached and polled ----
echo
echo "== the work =="
WORK_SECONDS=$((DEADLINE_EPOCH - FETCH_RESERVE - $(date +%s)))
[ "$WORK_SECONDS" -ge 120 ] || die "only ${WORK_SECONDS}s of lease left for the work; not starting it" 8
subst "$TMPD/remote_start.sh.template" "$OUT/remote_start.sh" || die "the start wrapper did not substitute" 1
echo "work_seconds=$WORK_SECONDS" >> "$OUT/leg.txt"
rexec "$OUT/remote_start.sh" > "$OUT/remote_start.log" 2>&1
RPID=$(sed -n 's/^REMOTE_PID=//p' "$OUT/remote_start.log" | tr -d '\r' | tail -1)
case "$RPID" in ''|*[!0-9]*) die "THE PAYLOAD DID NOT START (no pid). Read $OUT/remote_start.log." 8 ;; esac
BODY_STATE=running
log "remote pid $RPID ($RUNTIME), bound ${WORK_SECONDS}s; polling every 30 s"
POLL_DEADLINE=$((DEADLINE_EPOCH - FETCH_RESERVE + 60))
_unreach=0
while :; do
  if [ "$(date +%s)" -ge "$POLL_DEADLINE" ]; then
    log "OUTER POLL DEADLINE reached. Fetching what exists."; BODY_STATE=partial_deadline; FETCH_RED=1; break
  fi
  _st=$("${SSHN[@]}" "sudo -n sh -c 'if [ -f /root/gemm_leg.done ]; then echo LEG_DONE; elif kill -0 $RPID 2>/dev/null; then echo LEG_RUNNING; else echo LEG_GONE; fi'" 2>/dev/null) || _st=""
  case "$_st" in
    *LEG_DONE*)
      "${SSHN[@]}" "sudo -n sh -c 'for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 $RPID 2>/dev/null || break; sleep 1; done'" 2>/dev/null
      log "the body finished (sentinel on the VM)"; BODY_STATE="done"; break ;;
    *LEG_RUNNING*) _unreach=0 ;;
    *LEG_GONE*)
      log "THE BODY PROCESS IS GONE AND WROTE NO SENTINEL. Fetching a partial run."; BODY_STATE=partial_died; FETCH_RED=1; break ;;
    *)
      _unreach=$((_unreach + 1))
      [ "$_unreach" = 1 ] && log "poll: the VM did not answer (the body is detached; retrying)" ;;
  esac
  nap 30
done
echo "body=$BODY_STATE" >> "$OUT/leg.txt"

echo
echo "== fetch =="
_left=$((DEADLINE_EPOCH - $(date +%s) - 90))
FETCH_SECONDS=600
[ "$_left" -lt "$FETCH_SECONDS" ] && FETCH_SECONDS=$_left
[ "$FETCH_SECONDS" -ge 60 ] || FETCH_SECONDS=60
mkdir -p "$OUT/remote"
if with_deadline "$FETCH_SECONDS" /dev/null "${SSHN[@]}" "sudo -n sh -c 'cd /root/gemm_leg_out && tar czf - --exclude=./tools-venv .'" > "$TMPD/remote.tgz" \
   && tar xzf "$TMPD/remote.tgz" -C "$OUT/remote"; then
  log "fetched /root/gemm_leg_out -> $OUT/remote/ ($(wc -c < "$TMPD/remote.tgz" | tr -d ' ') bytes)"
else
  FETCH_RED=1; log "FETCH FAILED or hit its ${FETCH_SECONDS}s bound; deleting regardless"
fi
rm -rf "$OUT/remote/tools-venv"
with_deadline 60 /dev/null "${SSHN[@]}" 'sudo -n cat /root/gemm_leg_console.log /root/mojolearn-pull.done' > "$OUT/remote_console.log" 2>/dev/null || true
if [ "$SHIPS_SOURCE" = 1 ]; then
  _local_sha=$(cat "$OUT/source_sha256_local.txt" 2>/dev/null)
  _remote_sha=$(cat "$OUT/remote/source_sha256.txt" 2>/dev/null)
  if [ -n "$_local_sha" ] && [ "$_local_sha" = "$_remote_sha" ]; then
    echo "source_sha256_match=yes" >> "$OUT/leg.txt"
  else
    echo "source_sha256_match=NO local=$_local_sha remote=$_remote_sha" >> "$OUT/leg.txt"; FETCH_RED=1
  fi
fi
for _k in pixi_install_exit pixi_install_seconds device_check_exit card_exit extra_exit body_exit; do
  _v=$(sed -n "s/^$_k=//p" "$OUT/remote/leg.txt" 2>/dev/null | tail -1)
  echo "remote_$_k=${_v:-<absent>}" >> "$OUT/leg.txt"
  log "  $_k=${_v:-<absent>}"
done
echo "finished=$(utc)" >> "$OUT/leg.txt"
echo "lease_used_seconds=$(( $(date +%s) - LEG_START ))" >> "$OUT/leg.txt"
log "leg done; deleting (EXIT trap)"
[ "$FETCH_RED" = 1 ] && exit 1
[ "$KEY_RED" = 1 ] && exit 1
exit 0
