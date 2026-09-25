# tools/hotaisle_vm_lib.sh -- sourced, never run. ONE guarded Hot Aisle MI300X
# VM for the RELEASE: the AMD column (tools/release_wheel_smoke.sh --provider
# hotaisle) and the AMD build leg (tools/hotaisle_release_leg.sh), 2026-09-25.
# tools/hotaisle_leg.sh (the segment and lane leg) sources it too, for the
# shared settings (API, team, key file, ssh key, slot prefix, create lock,
# slot count, balance floor) and ha_dollars / ha_lease_cents, so every Hot
# Aisle runner counts the same slots and prices a lease the same way.
#
# The guards are tools/hotaisle_leg.sh's (read its header for the API and the
# runner results of 2026-09-11), re-stated here as functions so the two release
# runners share ONE copy instead of each retyping them:
#   a. SLOTS: the same /tmp/mojolearn-hotaisle-slot.N directories and create
#      lock as tools/hotaisle_leg.sh, so a release leg and any other Hot Aisle
#      leg on this Mac count against one another; min(3, the team's VM limit).
#   b. PRICED BEFORE THE CREATE: the offering's live OnDemandPrice (cents/h)
#      times the WHOLE horizon (the Mac dead-man's deadline, never less than
#      the offering's minimum reservation), rounded up to a cent. Above the
#      caller's dollar cap: refused. The balance is recorded, not enforced beyond
#      the 500-cent floor: Hot Aisle tops it up automatically (Andrew, 2026-09-25).
#   c. SPEC: 1gpu (one MI300X, any core count), 2gpu (the 2x MI300X VM, 60
#      minutes minimum) or auto (1gpu, and 2gpu only when no 1x VM is in
#      stock). On a 2gpu VM the release uses GPU 0 only; the caller pins it.
#   d. A Mac dead-man armed BEFORE the create, keyed by the deployment_id the
#      create returns (and the description, which it must carry to be deleted).
#   e. An on-box watchdog armed before any work: root, setsid, the key in a 0600
#      file in the caller's guard directory, sleeping to the lease and then
#      DELETEing its own VM with force. Verified by pid alive from a second ssh
#      session, the ref baked in, and a GET of its own VM with the key that
#      answers 200 with this VM's description. Unverifiable: delete unused.
#   f. The description is PATCHed to mojolearn:<lane>:<utc> right after the
#      create; a delete refuses a VM whose description is another's.
#   g. The teardown: DELETE ?force=true, then GET 404 or absent from a 200 team
#      listing. Only then are the dead-man cancelled and the slot released.
#      Unverified: a banner, the dead-man stays armed, the slot stays held.
# The key is read by the builtin `read`, written by `printf` into a 0600 curl
# config read with `curl -K`, reaches the VM on ssh stdin and is in no argv.
#
# The caller sets TMPD (a private temp dir) and defines die() and say(), and
# with_timeout (tools/runpod_pod_lib.sh's). Everything the caller reads back
# is an HA_* variable. MOJOLEARN_HOTAISLE_API, _SSH_KEY, _SSH_KEY_FP,
# _SLOT_PREFIX, _CREATE_LOCK and _POLL_SECONDS exist for the shim tests
# (tools/tests/test_hotaisle_release_shim.py), as do _VERIFY_SECONDS (how long
# a delete is verified, 600) and _RELEASE_SPEC (1gpu, 2gpu or auto, the one a
# release may set); a real run leaves the others unset.
HA_API=${MOJOLEARN_HOTAISLE_API:-https://admin.hotaisle.app/api}
HA_TEAM=${MOJOLEARN_HOTAISLE_TEAM:-andrews-team}
HA_KEYFILE=${MOJOLEARN_HOTAISLE_KEY_FILE:-$HOME/.mojolearn_hotaisle_key}
HA_SSH_KEY=${MOJOLEARN_HOTAISLE_SSH_KEY:-$HOME/.ssh/id_ed25519}
HA_SSH_KEY_FP=${MOJOLEARN_HOTAISLE_SSH_KEY_FP:-SHA256:pqDQ15Jijc636E5M3/2IvTGR7YKL11+JzQwGYtBvtQU}
HA_SLOT_PREFIX=${MOJOLEARN_HOTAISLE_SLOT_PREFIX:-/tmp/mojolearn-hotaisle-slot}
HA_CREATE_LOCK=${MOJOLEARN_HOTAISLE_CREATE_LOCK:-/tmp/mojolearn-hotaisle-create.lock}
HA_POLL=${MOJOLEARN_HOTAISLE_POLL_SECONDS:-5}
HA_SPEC_WANT=${MOJOLEARN_HOTAISLE_RELEASE_SPEC:-auto}
: "${HA_STOCK_WAIT_MINUTES:=${MOJOLEARN_HOTAISLE_STOCK_WAIT_MINUTES:-0}}"
: "${HA_SLOT_WAIT_MINUTES:=${MOJOLEARN_HOTAISLE_SLOT_WAIT_MINUTES:-0}}"
HA_MIN_BALANCE_CENTS=500
HA_MAX_SLOTS=3
HA_READY_SECONDS=${HA_READY_SECONDS:-600}
HA_VERIFY_SECONDS=${MOJOLEARN_HOTAISLE_VERIFY_SECONDS:-600}

HA_CURLRC=""; HA_TOKPAT=""; HA_CODE=""; HA_REFUSED=""
HA_VMREF=""; HA_VMNAME=""; HA_DESC=""; HA_SLOT=""; HA_NONCE="$$-$(date -u +%Y%m%dT%H%M%SZ)"
HA_DEADMAN_PID=""; HA_DEADMAN_DIR=""; HA_CREATE_ATTEMPTED=0; HA_CREATE_LOCK_HELD=0; HA_GONE=0
HA_SSH_IP=""; HA_SSH_PORT=22; HA_TARGET=""; HA_SPEC_USED=""; HA_PRICE=""; HA_MINRES=""; HA_CORES=""
HA_LEASE_CENTS=""; HA_BAL_BEFORE=""; HA_TEAM_MAX=0; HA_T_CREATE=""; HA_DEADLINE=0; HA_GFX=""
HA_WATCHDOG_OK=0; HA_GONE_LINE=""; HA_RECORD=/dev/null; HA_GUARD=""

ha_utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
ha_dollars() { [ "$1" -ge 0 ] 2>/dev/null && printf '$%d.%02d' $(( $1 / 100 )) $(( $1 % 100 )) || printf 'unknown'; }
ha_rec() { printf '%s\n' "$*" >> "$HA_RECORD"; }
ha_lease_cents() {  # <cents/hour> <minutes> <minimum reservation minutes>: the whole lease, rounded up to a cent
    _lc_bill=$2; [ "${3:-0}" -gt "$_lc_bill" ] 2>/dev/null && _lc_bill=$3
    echo $(( ($1 * _lc_bill + 59) / 60 ))
}

# ---------------------------------------------------------------- the key and the API
ha_key_hygiene() {  # a reason on stdout and 1 when the key file must not be used
    [ -f "$HA_KEYFILE" ] || { echo "key file $HA_KEYFILE does not exist"; return 1; }
    _p=$(stat -f '%OLp' "$HA_KEYFILE" 2>/dev/null || stat -c '%a' "$HA_KEYFILE" 2>/dev/null || echo '?')
    [ "$_p" = 600 ] || { echo "key file $HA_KEYFILE is mode $_p, must be 600"; return 1; }
    case "$(cd "$(dirname "$HA_KEYFILE")" && pwd)/" in "${ROOT:-/nonexistent}"/*) echo "key file $HA_KEYFILE is INSIDE the repository"; return 1 ;; esac
    return 0
}
ha_load_key() {  # read by a builtin, written by a builtin: no process sees the key
    _k=""
    IFS= read -r _k < "$HA_KEYFILE" || [ -n "$_k" ] || return 1
    _k="${_k//[$'\t\r\n ']/}"
    case "$_k" in ''|*[!A-Za-z0-9._-]*) _k=""; return 1 ;; esac
    HA_CURLRC="$TMPD/ha.curlrc"; HA_TOKPAT="$TMPD/ha.key.pattern"
    ( umask 077
      printf 'header = "Authorization: Token %s"\nsilent\nshow-error\n' "$_k" > "$HA_CURLRC"
      printf '%s\n' "$_k" > "$HA_TOKPAT" )
    _k=""
    return 0
}
ha_call() {  # METHOD PATH-under-/api/ [json file] [max-time]; sets HA_CODE, body in $TMPD/ha.body
    : > "$TMPD/ha.body"
    if [ -n "${3:-}" ]; then
        HA_CODE=$(curl -K "$HA_CURLRC" --max-time "${4:-60}" -o "$TMPD/ha.body" -w '%{http_code}' -X "$1" \
            -H 'Content-Type: application/json' --data-binary "@$3" "$HA_API/$2" 2>>"$TMPD/curl.err") || HA_CODE=000
    else
        HA_CODE=$(curl -K "$HA_CURLRC" --max-time "${4:-60}" -o "$TMPD/ha.body" -w '%{http_code}' -X "$1" "$HA_API/$2" 2>>"$TMPD/curl.err") || HA_CODE=000
    fi
    HA_CODE=${HA_CODE:-000}
}
ha_redact() {  # <file>: replace any occurrence of the key
    [ -f "$1" ] && [ -n "$HA_TOKPAT" ] && [ -f "$HA_TOKPAT" ] || return 0
    python3 - "$HA_TOKPAT" "$1" <<'PY'
import sys
tok = open(sys.argv[1]).read().strip()
p = sys.argv[2]
data = open(p, encoding="utf-8", errors="replace").read()
if tok and tok in data:
    open(p, "w").write(data.replace(tok, "<redacted>"))
PY
}
ha_py() {  # python over the last body; prints nothing on a parse failure
    python3 - "$TMPD/ha.body" "$@" <<'PYEOF' 2>/dev/null || true
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    d = None
what, a = sys.argv[2], sys.argv[3:]
if what == "teams":
    for t in d or []:
        if t.get("handle") == a[0]:
            print("yes" if "operator" in (t.get("effective_roles") or []) else "no", t.get("maximum_virtual_machines", 0))
            break
    else:
        print("absent 0")
elif what == "balance":
    print(d.get("available_balance", -1) if isinstance(d, dict) else -1)
elif what == "sshkey":
    print("yes" if any(k.get("fingerprint") == a[0] for k in d or []) else "no")
elif what == "offers":
    for e in d or []:
        s = e.get("Specs") or {}
        g = s.get("gpus") or []
        print("%sx %s cpu_cores=%s quantity=%s %s cents/h min_reservation=%s min" % (
            sum(x.get("count", 0) for x in g), ",".join(x.get("model", "?") for x in g),
            s.get("cpu_cores"), e.get("Quantity"), e.get("OnDemandPrice"), e.get("MinimumReservationMinutes")))
elif what == "pick":  # <want: 1gpu|2gpu|auto> <specs out>
    def gpus(e):
        g = (e.get("Specs") or {}).get("gpus") or []
        if g and all(x.get("model") == "MI300X" for x in g):
            return sum(x.get("count") or 0 for x in g)
        return 0
    order = {"1gpu": [1], "2gpu": [2], "auto": [1, 2]}[a[0]]
    for n in order:
        found = [e for e in d or [] if gpus(e) == n and (e.get("Quantity") or 0) > 0]
        if found:
            found.sort(key=lambda e: (-(e.get("Quantity") or 0), e.get("OnDemandPrice") or 0))
            best = found[0]
            json.dump(best["Specs"], open(a[1], "w"))
            print("found", "%dgpu" % n, best.get("Quantity") or 0, best.get("OnDemandPrice") or 0,
                  best.get("MinimumReservationMinutes") or 0, (best.get("Specs") or {}).get("cpu_cores") or 0)
            break
    else:
        print("none - 0 0 0 0")
elif what == "vm":
    v = d if isinstance(d, dict) else {}
    sa = v.get("ssh_access") or {}
    print("\t".join(str(x) for x in [v.get("name", "") or "-", v.get("deployment_id", "") or "-",
          sa.get("ip_address") or v.get("ip_address", "") or "-", sa.get("port") or 22]))
elif what == "desc":
    print((d.get("description") or "") if isinstance(d, dict) else "")
elif what == "state":
    print(d.get("state", "unknown") if isinstance(d, dict) else "unknown")
elif what == "count":
    print(len(d) if isinstance(d, list) else -1)
elif what == "inlist":
    if not isinstance(d, list):
        print("unknown")
    else:
        refs = set(x for x in a if x)
        print("yes" if any(v.get("name") in refs or v.get("deployment_id") in refs for v in d) else "no")
elif what == "ids":
    for v in d or []:
        print(v.get("deployment_id", ""), v.get("name", ""))
PYEOF
}

# ---------------------------------------------------------------- slots (tools/hotaisle_leg.sh's directories)
ha_slot_cap() {
    _c=$HA_MAX_SLOTS
    [ "${HA_TEAM_MAX:-0}" -gt 0 ] 2>/dev/null && [ "$HA_TEAM_MAX" -lt "$_c" ] && _c=$HA_TEAM_MAX
    echo "$_c"
}
ha_try_take_slot() {  # <lane> <out>
    for _n in $(seq 1 "$(ha_slot_cap)"); do
        if mkdir "$HA_SLOT_PREFIX.$_n" 2>/dev/null; then
            HA_SLOT="$HA_SLOT_PREFIX.$_n"
            { echo "lane=$1"; echo "pid=$$"; echo "nonce=$HA_NONCE"; echo "utc=$(ha_utc)"; echo "out=$2"; } > "$HA_SLOT/owner"
            return 0
        fi
    done
    return 1
}
ha_release_slot() {
    [ -n "$HA_SLOT" ] || return 0
    if grep -qx "nonce=$HA_NONCE" "$HA_SLOT/owner" 2>/dev/null; then
        rm -rf "$HA_SLOT" && say "released the Hot Aisle slot $HA_SLOT"
        ha_rec "slot=released $HA_SLOT $(ha_utc)"
    else
        say "!! $HA_SLOT no longer carries this run's nonce; left in place"
    fi
    HA_SLOT=""
}
ha_release_create_lock() {
    [ "$HA_CREATE_LOCK_HELD" = 1 ] || return 0
    grep -qx "nonce=$HA_NONCE" "$HA_CREATE_LOCK/owner" 2>/dev/null && rm -rf "$HA_CREATE_LOCK"
    HA_CREATE_LOCK_HELD=0
}

# ---------------------------------------------------------------- the Mac dead-man
ha_write_deadman() {  # <dir> <deadline epoch> <record file>; composes and checks, never arms
    ( umask 077; mkdir -p "$1"; if [ -f "$HA_CURLRC" ]; then cp "$HA_CURLRC" "$1/curlrc"; else : > "$1/curlrc"; fi )
    cat > "$1/deadman.sh" <<'DM_EOF'
#!/bin/sh
# tools/hotaisle_vm_lib.sh's Mac dead-man: ends the Hot Aisle VM if the runner
# is gone. Keyed by the deployment_id the runner writes beside this file as
# soon as the create returns it, and refused when the VM wears another
# description. The key is in the 0600 curl config beside this file.
set -u
trap '' HUP INT
D="$(cd "$(dirname "$0")" && pwd)"
while [ "$(date +%s)" -lt @DEADLINE@ ]; do sleep 20; done
L="$D/deadman.log"
echo "$(date -u +%FT%TZ) dead-man firing" >> "$L"
ref=""
[ -s "$D/vm_ref.txt" ] && ref="$(cat "$D/vm_ref.txt")"
if [ -z "$ref" ]; then echo "mac_deadman_fired no VM ref recorded" >> '@RECORD@'; rm -f "$D/curlrc"; exit 0; fi
want=""
[ -s "$D/desc.txt" ] && want="$(cat "$D/desc.txt")"
c=$(curl -K "$D/curlrc" --max-time 60 -o "$D/vm.json" -w '%{http_code}' "@API@/teams/@TEAM@/virtual_machines/$ref/" 2>> "$L")
if [ "$c" = 200 ]; then
    d=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("description") or "")' "$D/vm.json" 2>/dev/null)
    if [ -n "$d" ] && [ "$d" != "$want" ]; then
        echo "mac_deadman_fired REFUSED ref=$ref: description '$d' is not '$want'" >> '@RECORD@'; rm -f "$D/curlrc"; exit 0
    fi
    n=1
    while [ "$n" -le 6 ]; do
        c=$(curl -K "$D/curlrc" --max-time 900 -o /dev/null -w '%{http_code}' -X DELETE "@API@/teams/@TEAM@/virtual_machines/$ref/?force=true" 2>> "$L")
        echo "$(date -u +%FT%TZ) DELETE $ref attempt $n -> $c" >> "$L"
        case "$c" in 2*|404) break ;; esac
        sleep 15; n=$((n + 1))
    done
fi
echo "mac_deadman_fired $(date -u +%FT%TZ) ref=$ref last_http=$c" >> '@RECORD@'
rm -f "$D/curlrc"
DM_EOF
    sed -i.bak -e "s|@DEADLINE@|$2|g" -e "s|@RECORD@|$3|g" -e "s|@API@|$HA_API|g" -e "s|@TEAM@|$HA_TEAM|g" "$1/deadman.sh"
    rm -f "$1/deadman.sh.bak"
    if grep -q '@[A-Z0-9]*@' "$1/deadman.sh"; then return 1; fi
    sh -n "$1/deadman.sh"
}
ha_cancel_deadman() {
    [ -n "$HA_DEADMAN_PID" ] || return 0
    pkill -P "$HA_DEADMAN_PID" 2>/dev/null || true
    kill "$HA_DEADMAN_PID" 2>/dev/null && say "Hot Aisle dead-man cancelled (pid $HA_DEADMAN_PID)"
    [ -n "$HA_DEADMAN_DIR" ] && rm -rf "$HA_DEADMAN_DIR"
    ha_rec "mac_deadman=cancelled $(ha_utc)"
    HA_DEADMAN_PID=""; HA_DEADMAN_DIR=""
}

# ---------------------------------------------------------------- the on-box watchdog
ha_write_watchdog() {  # <file> <guard dir on the box> <seconds> <ref>; composed and checked
    cat > "$1" <<'WD_EOF'
#!/bin/sh
# tools/hotaisle_vm_lib.sh's ON-BOX watchdog. RUNS ON THE VM AS ROOT, DETACHED:
# sleeps to the lease and then DELETEs THIS VM through the API with force. The
# key is in the 0600 curl config beside this file (delivered on ssh stdin).
set -u
trap '' HUP INT
G='@GUARD@'
echo $$ > "$G/watchdog.pid"
T=$(( $(date +%s) + @SECS@ ))
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) armed ref=@REF@ fires_in=@SECS@s" >> "$G/watchdog.out"
while [ "$(date +%s)" -lt "$T" ]; do sleep 10; done
n=1
while [ "$n" -le 20 ]; do
    code=$(curl -K "$G/curlrc" --max-time 900 -o "$G/watchdog.body" -w '%{http_code}' \
        -X DELETE '@API@/teams/@TEAM@/virtual_machines/@REF@/?force=true')
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) DELETE @REF@ attempt $n -> $code" >> "$G/watchdog.out"
    case "$code" in 2*|404) exit 0 ;; esac
    sleep 15
    n=$((n + 1))
done
WD_EOF
    sed -i.bak -e "s|@GUARD@|$2|g" -e "s|@SECS@|$3|g" -e "s|@REF@|$4|g" -e "s|@API@|$HA_API|g" -e "s|@TEAM@|$HA_TEAM|g" "$1"
    rm -f "$1.bak"
    if grep -q '@[A-Z0-9]*@' "$1"; then return 1; fi
    sh -n "$1"
}

# ---------------------------------------------------------------- the box
ha_sq() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }
ha_root_cmd() { printf 'sudo -n -H bash -c %s' "$(ha_sq "$1")"; }   # a command, run as root on the VM
ha_ssh() {  # <seconds> <command run as root on the VM>; stdin passes through
    _s=$1; shift
    # shellcheck disable=SC2086
    with_timeout "$_s" ssh $HA_SSH_OPTS $HA_TARGET "$(ha_root_cmd "$1")"
}
HA_SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o BatchMode=yes"

# ---------------------------------------------------------------- gone, delete
ha_gone() {  # <ref> [<other ref>]: prints the reading; 0 when gone (GET 404, or absent from a 200 listing)
    ha_call GET "teams/$HA_TEAM/virtual_machines/$1/"; _g=$HA_CODE
    ha_call GET "teams/$HA_TEAM/virtual_machines/"; _l=$HA_CODE
    _in=unknown; [ "$_l" = 200 ] && _in=$(ha_py inlist "$1" "${2:-}")
    if [ "$_g" = 404 ] || { [ "$_l" = 200 ] && [ "$_in" = no ]; }; then printf 'yes get=%s list=%s listed=%s' "$_g" "$_l" "$_in"; return 0; fi
    printf 'no get=%s list=%s listed=%s' "$_g" "$_l" "$_in"; return 1
}
ha_delete_verify() {  # <ref> <other ref> <verify seconds>; sets HA_GONE_LINE; 0 gone, 1 unverified, 2 refused (not ours)
    HA_GONE_LINE=""
    if _line=$(ha_gone "$1" "$2"); then
        HA_GONE_LINE="verified_gone ref=$1 $_line utc=$(ha_utc) (already gone, no DELETE sent)"; ha_rec "$HA_GONE_LINE"; return 0
    fi
    ha_call GET "teams/$HA_TEAM/virtual_machines/$1/"
    if [ "$HA_CODE" = 200 ]; then
        _d=$(ha_py desc)
        if [ -n "$_d" ] && [ "$_d" != "$HA_DESC" ]; then
            ha_rec "delete REFUSED ref=$1: description '$_d' is not this run's ('$HA_DESC')"
            say "!! delete REFUSED: $1 has description '$_d', not '$HA_DESC'"
            return 2
        fi
    fi
    for _i in 1 2 3 4; do
        ha_call DELETE "teams/$HA_TEAM/virtual_machines/$1/?force=true" "" 900
        say "DELETE $1 ?force=true -> HTTP $HA_CODE"
        ha_rec "delete ref=$1 attempt $_i -> HTTP $HA_CODE utc=$(ha_utc)"
        case "$HA_CODE" in 2*|404) break ;; esac
        sleep 15
    done
    _t0=$(date +%s)
    while :; do
        if _line=$(ha_gone "$1" "$2"); then
            HA_GONE_LINE="verified_gone ref=$1 $_line utc=$(ha_utc) after=$(( $(date +%s) - _t0 ))s"
            ha_rec "$HA_GONE_LINE"; say "$HA_GONE_LINE"; return 0
        fi
        ha_rec "verify ref=$1 $_line utc=$(ha_utc)"
        [ $(( $(date +%s) - _t0 )) -lt "$3" ] || break
        sleep "$HA_POLL"
    done
    HA_GONE_LINE="NOT_VERIFIED ref=$1 $_line utc=$(ha_utc)"; ha_rec "$HA_GONE_LINE"; say "!! $HA_GONE_LINE"
    return 1
}
ha_adopt_new_vm() {  # a VM absent from the pre-create snapshot, when exactly one
    ha_call GET "teams/$HA_TEAM/virtual_machines/"
    [ "$HA_CODE" = 200 ] || return 1
    _new=$(ha_py ids | while read -r _id _nm; do grep -qx "$_id" "$TMPD/ha_pre_ids.txt" 2>/dev/null || echo "$_id $_nm"; done)
    [ "$(printf '%s' "$_new" | grep -c .)" = 1 ] || return 1
    HA_VMREF=${_new%% *}; HA_VMNAME=${_new#* }
    return 0
}

# ---------------------------------------------------------------- the rental
ha_refuse() {  # nothing was created: undo what was taken, say why, return 1
    HA_REFUSED="$1"
    ha_cancel_deadman
    ha_release_create_lock
    ha_release_slot
    ha_rec "hotaisle_refused=$1"
    say "Hot Aisle REFUSED: $1"
    return 1
}

# ha_rent <lane> <lease minutes> <cap cents> <record file> <guard dir on the box> <out dir>
# 0: a VM is up, ssh settled, the watchdog verified, HA_TARGET set.
# 1: NOTHING was created (HA_REFUSED says why): no key, no operator role, the
#    balance, the ssh key, no slot, no stock, the price over the cap, or a
#    create the API refused. The caller may walk to another provider.
# After a create, any failure calls die: the caller's EXIT trap runs ha_teardown.
ha_rent() {
    _lane=$1; _lease=$2; _cap=$3; HA_RECORD=$4; HA_GUARD=$5; _out=$6
    case "$_lane" in ''|*[!A-Za-z0-9_.-]*) die "Hot Aisle lane '$_lane': letters, digits and _.- only" ;; esac
    case "$HA_SPEC_WANT" in 1gpu|2gpu|auto) ;; *) die "MOJOLEARN_HOTAISLE_RELEASE_SPEC must be 1gpu, 2gpu or auto" ;; esac
    _why=$(ha_key_hygiene) || { ha_refuse "no usable Hot Aisle key: $_why"; return 1; }
    ha_load_key || { ha_refuse "the Hot Aisle key file $HA_KEYFILE is empty or malformed"; return 1; }
    ha_call GET "teams/"
    [ "$HA_CODE" = 200 ] || { ha_refuse "GET teams/ -> HTTP $HA_CODE"; return 1; }
    read -r _op HA_TEAM_MAX <<< "$(ha_py teams "$HA_TEAM")"
    [ "$_op" = yes ] || { ha_refuse "the key lacks the operator role on $HA_TEAM (create and DELETE need it)"; return 1; }
    ha_call GET "teams/$HA_TEAM/balance/"
    HA_BAL_BEFORE=$(ha_py balance); HA_BAL_BEFORE=${HA_BAL_BEFORE:--1}
    ha_rec "balance_before_cents=$HA_BAL_BEFORE"
    [ "$HA_BAL_BEFORE" -ge "$HA_MIN_BALANCE_CENTS" ] 2>/dev/null \
        || { ha_refuse "balance $(ha_dollars "$HA_BAL_BEFORE") is below the \$5.00 floor"; return 1; }
    ha_call GET "user/ssh_keys/"
    [ "$HA_CODE" = 200 ] && [ "$(ha_py sshkey "$HA_SSH_KEY_FP")" = yes ] \
        || { ha_refuse "the ssh key $HA_SSH_KEY_FP is not registered on the account (HTTP $HA_CODE)"; return 1; }
    # a. the slot
    _t0=$(date +%s)
    while :; do
        if ha_try_take_slot "$_lane" "$_out"; then
            ha_call GET "teams/$HA_TEAM/virtual_machines/"
            if [ "$HA_CODE" = 200 ] && [ "$(ha_py count)" -lt "$HA_TEAM_MAX" ]; then break; fi
            say "slot taken but the team already runs $(ha_py count) of $HA_TEAM_MAX VMs; releasing it"
            ha_release_slot
        fi
        [ $(( $(date +%s) - _t0 )) -lt $(( HA_SLOT_WAIT_MINUTES * 60 )) ] \
            || { ha_refuse "no Hot Aisle slot free in $HA_SLOT_WAIT_MINUTES minutes (the team VM limit is $HA_TEAM_MAX)"; return 1; }
        sleep 60
    done
    ha_rec "slot=$HA_SLOT taken $(ha_utc)"
    # b. stock, and the price of the whole horizon
    _t0=$(date +%s)
    while :; do
        ha_call GET "teams/$HA_TEAM/virtual_machines/available/"
        cp "$TMPD/ha.body" "$_out/hotaisle_offering.json" 2>/dev/null
        read -r _f HA_SPEC_USED _q HA_PRICE HA_MINRES HA_CORES <<< "$(ha_py pick "$HA_SPEC_WANT" "$TMPD/ha_create.json")"
        [ "$HA_CODE" = 200 ] && [ "$_f" = found ] && break
        [ $(( $(date +%s) - _t0 )) -lt $(( HA_STOCK_WAIT_MINUTES * 60 )) ] \
            || { ha_refuse "no stock: no MI300X VM of spec $HA_SPEC_WANT is available (HTTP $HA_CODE; offerings: $(ha_py offers | tr '\n' ';'))"; return 1; }
        sleep 60
    done
    case "$HA_PRICE" in ''|*[!0-9]*) HA_PRICE=0 ;; esac
    case "$HA_MINRES" in ''|*[!0-9]*) HA_MINRES=0 ;; esac
    [ "$HA_PRICE" -gt 0 ] || { ha_refuse "the $HA_SPEC_USED offering shows no OnDemandPrice, so the lease cannot be priced"; return 1; }
    _horizon=$(( _lease + HA_READY_SECONDS / 60 + 10 ))   # the Mac dead-man's deadline, the worst case billed
    _bill=$_horizon; [ "$HA_MINRES" -gt "$_bill" ] && _bill=$HA_MINRES
    HA_LEASE_CENTS=$(ha_lease_cents "$HA_PRICE" "$_horizon" "$HA_MINRES")
    ha_rec "spec=$HA_SPEC_USED cpu_cores=$HA_CORES price_cents_per_hour=$HA_PRICE min_reservation_minutes=$HA_MINRES"
    ha_rec "lease_minutes=$_lease horizon_minutes=$_horizon billed_minutes_at_most=$_bill max_cost_cents=$HA_LEASE_CENTS cap_cents=$_cap"
    [ "$HA_LEASE_CENTS" -le "$_cap" ] \
        || { ha_refuse "the $HA_SPEC_USED VM for up to $_bill minutes at $(ha_dollars "$HA_PRICE")/h is up to $(ha_dollars "$HA_LEASE_CENTS"), above the cap of $(ha_dollars "$_cap"); nothing was created"; return 1; }
    # THE BALANCE IS NOT A LIMIT (Andrew, 2026-09-25): Hot Aisle tops the team
    # balance up automatically, so a lease is never refused for exceeding it;
    # the balance is recorded before and after for the cost record, and only
    # an account below the $5.00 floor (a dead or empty account) refuses above.
    [ "$HA_BAL_BEFORE" -ge $(( HA_LEASE_CENTS + HA_MIN_BALANCE_CENTS )) ] \
        || say "note: balance $(ha_dollars "$HA_BAL_BEFORE") is below the whole lease ($(ha_dollars "$HA_LEASE_CENTS")) plus the floor; Hot Aisle tops up automatically, proceeding"
    say "Hot Aisle $HA_SPEC_USED MI300X VM ($HA_CORES cores) at $(ha_dollars "$HA_PRICE")/h; at most $(ha_dollars "$HA_LEASE_CENTS") for $_bill minutes; balance $(ha_dollars "$HA_BAL_BEFORE")"
    # d. the Mac dead-man, BEFORE the create
    HA_DEADLINE=$(( $(date +%s) + _horizon * 60 ))
    HA_DESC="mojolearn:$_lane:$(date -u +%Y%m%dT%H%M%SZ)"
    HA_DEADMAN_DIR="${TMPDIR:-/tmp}/mojolearn-hotaisle-release-deadman-$$"
    ha_write_deadman "$HA_DEADMAN_DIR" "$HA_DEADLINE" "$HA_RECORD" || { ha_refuse "the Mac dead-man did not compose"; return 1; }
    printf '%s\n' "$HA_DESC" > "$HA_DEADMAN_DIR/desc.txt"
    nohup sh -c 'trap "" HUP INT; exec sh "$0"' "$HA_DEADMAN_DIR/deadman.sh" > /dev/null 2>&1 < /dev/null &
    HA_DEADMAN_PID=$!
    disown "$HA_DEADMAN_PID" 2>/dev/null || true
    sleep 1
    kill -0 "$HA_DEADMAN_PID" 2>/dev/null || { HA_DEADMAN_PID=""; ha_refuse "the Mac dead-man did not start"; return 1; }
    ha_rec "mac_deadman_pid=$HA_DEADMAN_PID fires_at_epoch=$HA_DEADLINE"
    say "Hot Aisle dead-man ARMED before the create: pid $HA_DEADMAN_PID, fires in $(( HA_DEADLINE - $(date +%s) ))s"
    # the create, serialized with every Hot Aisle leg on this Mac
    _t0=$(date +%s)
    until mkdir "$HA_CREATE_LOCK" 2>/dev/null; do
        _m=$(stat -f %m "$HA_CREATE_LOCK" 2>/dev/null || stat -c %Y "$HA_CREATE_LOCK" 2>/dev/null || echo 0)
        if [ $(( $(date +%s) - _m )) -gt 900 ]; then rm -rf "$HA_CREATE_LOCK"; continue; fi
        [ $(( $(date +%s) - _t0 )) -lt 900 ] || { ha_refuse "the Hot Aisle create lock stayed held for 15 minutes"; return 1; }
        sleep 5
    done
    HA_CREATE_LOCK_HELD=1
    { echo "nonce=$HA_NONCE"; echo "pid=$$"; echo "lane=$_lane"; echo "utc=$(ha_utc)"; } > "$HA_CREATE_LOCK/owner"
    ha_call GET "teams/$HA_TEAM/virtual_machines/"
    [ "$HA_CODE" = 200 ] || { ha_refuse "the pre-create listing answered HTTP $HA_CODE"; return 1; }
    ha_py ids | awk '{print $1}' > "$TMPD/ha_pre_ids.txt"
    cp "$TMPD/ha_create.json" "$_out/hotaisle_create_request.json"
    say "creating the Hot Aisle $HA_SPEC_USED MI300X VM. THE BILL STARTS HERE."
    HA_CREATE_ATTEMPTED=1
    HA_T_CREATE=$(date +%s)
    ha_call POST "teams/$HA_TEAM/virtual_machines/" "$TMPD/ha_create.json" 300
    _cc=$HA_CODE
    cp "$TMPD/ha.body" "$_out/hotaisle_create_response.json"; ha_redact "$_out/hotaisle_create_response.json"
    ha_rec "create_http=$_cc create_utc=$(ha_utc)"
    IFS=$'\t' read -r HA_VMNAME HA_VMREF _ip _port <<< "$(ha_py vm)"
    [ "$HA_VMNAME" = - ] && HA_VMNAME=""; [ "$HA_VMREF" = - ] && HA_VMREF=""
    case "$_cc" in 2*) ;; *) HA_VMREF="" ;; esac
    if [ -z "$HA_VMREF" ]; then
        sleep 10
        if ! ha_adopt_new_vm; then
            case "$_cc" in
                401|402|403|404|428)
                    # the API refused the create (limit, balance, stock, ssh key); only a
                    # 200 listing with nothing new in it proves nothing is billing
                    ha_call GET "teams/$HA_TEAM/virtual_machines/"
                    if [ "$HA_CODE" = 200 ] && [ -z "$(ha_py ids | while read -r _id _nm; do grep -qx "$_id" "$TMPD/ha_pre_ids.txt" || echo "$_id"; done)" ]; then
                        HA_CREATE_ATTEMPTED=0
                        ha_refuse "the create was refused by the API (HTTP $_cc: $(head -c 200 "$_out/hotaisle_create_response.json")) and the listing shows no new VM"
                        return 1
                    fi ;;
            esac
            for _i in $(seq 1 18); do ha_adopt_new_vm && break; sleep 10; done
            [ -n "$HA_VMREF" ] || die "Hot Aisle create answered HTTP $_cc with no deployment_id and no new VM appeared; the teardown looks once more"
        fi
        say "ADOPTED $HA_VMREF by listing diff after an unreadable create (HTTP $_cc)"
        ha_rec "adopted_by_listing_diff=1"
    fi
    case "$HA_VMREF" in *[!A-Za-z0-9_.-]*) die "the Hot Aisle VM ref '$HA_VMREF' has unexpected characters" ;; esac
    printf '%s\n' "$HA_VMREF" > "$HA_DEADMAN_DIR/vm_ref.txt"
    printf 'vm_ref=%s\nvm_name=%s\n' "$HA_VMREF" "$HA_VMNAME" >> "$HA_SLOT/owner"
    ha_rec "vm_ref=$HA_VMREF"; ha_rec "vm_name=$HA_VMNAME"
    # f. the description
    printf '{"description":"%s"}\n' "$HA_DESC" > "$TMPD/ha_patch.json"
    _tagged=0
    for _i in 1 2 3 4 5; do
        ha_call PATCH "teams/$HA_TEAM/virtual_machines/$HA_VMREF/" "$TMPD/ha_patch.json"
        ha_call GET "teams/$HA_TEAM/virtual_machines/$HA_VMREF/"
        if [ "$HA_CODE" = 200 ] && [ "$(ha_py desc)" = "$HA_DESC" ]; then _tagged=1; break; fi
        sleep 5
    done
    ha_release_create_lock
    [ "$_tagged" = 1 ] || die "the Hot Aisle description PATCH never landed; deleting the VM unused"
    ha_rec "description=$HA_DESC"
    say "VM $HA_VMNAME deployment_id $HA_VMREF, description $HA_DESC"
    # running, then ssh as hotaisle, three consecutive successes, passwordless sudo
    _t0=$(date +%s); _st=unknown
    while [ $(( $(date +%s) - _t0 )) -lt "$HA_READY_SECONDS" ]; do
        ha_call GET "teams/$HA_TEAM/virtual_machines/$HA_VMREF/state/"
        [ "$HA_CODE" = 200 ] && _st=$(ha_py state)
        [ "$_st" = running ] && break
        sleep "$HA_POLL"
    done
    [ "$_st" = running ] || die "the Hot Aisle VM never reached running (last state $_st)"
    ha_call GET "teams/$HA_TEAM/virtual_machines/$HA_VMREF/"
    cp "$TMPD/ha.body" "$_out/hotaisle_vm.json"; ha_redact "$_out/hotaisle_vm.json"
    IFS=$'\t' read -r _n _id HA_SSH_IP HA_SSH_PORT <<< "$(ha_py vm)"
    [ -n "$HA_SSH_IP" ] && [ "$HA_SSH_IP" != - ] || die "the Hot Aisle VM has no ssh address"
    case "$HA_SSH_PORT" in ''|*[!0-9]*) HA_SSH_PORT=22 ;; esac
    HA_TARGET="-p $HA_SSH_PORT -i $HA_SSH_KEY -o IdentitiesOnly=yes hotaisle@$HA_SSH_IP"
    ha_rec "ssh=hotaisle@$HA_SSH_IP:$HA_SSH_PORT running_after_seconds=$(( $(date +%s) - HA_T_CREATE ))"
    _ok=0
    while [ $(( $(date +%s) - _t0 )) -lt "$HA_READY_SECONDS" ]; do
        # shellcheck disable=SC2086
        if with_timeout 40 ssh $HA_SSH_OPTS $HA_TARGET true < /dev/null 2>/dev/null; then _ok=$((_ok + 1)); [ "$_ok" -ge 3 ] && break; else _ok=0; fi
        sleep "$HA_POLL"
    done
    [ "$_ok" -ge 3 ] || die "ssh never settled on the Hot Aisle VM ($HA_SSH_IP:$HA_SSH_PORT)"
    ha_ssh 60 'echo SUDO_OK' < /dev/null 2>&1 | grep -q SUDO_OK || die "passwordless sudo is not available for hotaisle; deleting unused"
    say "ssh settled as hotaisle@$HA_SSH_IP:$HA_SSH_PORT after $(( $(date +%s) - HA_T_CREATE ))s"
    # e. the on-box watchdog, before any work
    _wd_secs=$(( HA_T_CREATE + _lease * 60 - $(date +%s) )); [ "$_wd_secs" -ge 120 ] || _wd_secs=120
    ha_write_watchdog "$TMPD/ha_watchdog.sh" "$HA_GUARD" "$_wd_secs" "$HA_VMREF" || die "the on-box watchdog did not compose; deleting unused"
    cp "$TMPD/ha_watchdog.sh" "$_out/hotaisle_watchdog.sh"
    ha_ssh 60 "umask 077; mkdir -p $HA_GUARD && chmod 700 $HA_GUARD && cat > $HA_GUARD/curlrc && chmod 600 $HA_GUARD/curlrc" < "$HA_CURLRC" \
        || die "could not deliver the key for the watchdog; deleting unused"
    ha_ssh 60 "umask 077; cat > $HA_GUARD/watchdog.sh && chmod 700 $HA_GUARD/watchdog.sh" < "$TMPD/ha_watchdog.sh" \
        || die "could not deliver the watchdog; deleting unused"
    ha_ssh 90 "rm -f $HA_GUARD/watchdog.pid
if command -v setsid > /dev/null 2>&1; then setsid nohup sh $HA_GUARD/watchdog.sh > $HA_GUARD/watchdog.log 2>&1 < /dev/null &
else nohup sh $HA_GUARD/watchdog.sh > $HA_GUARD/watchdog.log 2>&1 < /dev/null &
fi
i=0; while [ \$i -lt 15 ] && [ ! -s $HA_GUARD/watchdog.pid ]; do sleep 1; i=\$((i + 1)); done
p=\$(cat $HA_GUARD/watchdog.pid 2>/dev/null)
if [ -n \"\$p\" ] && kill -0 \"\$p\" 2>/dev/null; then echo WATCHDOG_ALIVE pid=\$p; else echo WATCHDOG_DEAD; fi
echo REF_BAKED_IN=\$(grep -c 'virtual_machines/$HA_VMREF/?force=true' $HA_GUARD/watchdog.sh)
echo TOKEN_GET_HTTP=\$(curl -K $HA_GUARD/curlrc --max-time 30 -o $HA_GUARD/self.json -w '%{http_code}' '$HA_API/teams/$HA_TEAM/virtual_machines/$HA_VMREF/')
if grep -q '\"description\": *\"$HA_DESC\"' $HA_GUARD/self.json; then echo DESC_MATCH; else echo DESC_MISMATCH; fi" \
        < /dev/null > "$_out/hotaisle_watchdog_check.txt" 2>&1
    _wpid=$(sed -n 's/^WATCHDOG_ALIVE pid=//p' "$_out/hotaisle_watchdog_check.txt" | tr -d '\r' | head -1)
    case "$_wpid" in ''|*[!0-9]*) _wpid=0 ;; esac
    sleep 2
    ha_ssh 60 "kill -0 $_wpid 2>/dev/null && echo WATCHDOG_STILL_ALIVE_SECOND_SESSION" < /dev/null >> "$_out/hotaisle_watchdog_check.txt" 2>&1
    sed 's/^/    /' "$_out/hotaisle_watchdog_check.txt"
    { grep -q '^WATCHDOG_ALIVE' "$_out/hotaisle_watchdog_check.txt" && grep -q '^REF_BAKED_IN=[1-9]' "$_out/hotaisle_watchdog_check.txt" \
      && grep -q '^TOKEN_GET_HTTP=200' "$_out/hotaisle_watchdog_check.txt" && grep -q '^DESC_MATCH' "$_out/hotaisle_watchdog_check.txt" \
      && grep -q WATCHDOG_STILL_ALIVE_SECOND_SESSION "$_out/hotaisle_watchdog_check.txt"; } \
        || die "THE ON-BOX WATCHDOG COULD NOT BE VERIFIED (pid, second session, ref, GET 200 or description); deleting the VM unused"
    HA_WATCHDOG_OK=1
    ha_rec "watchdog_seconds=$_wd_secs watchdog_verified=1"
    say "on-box watchdog ARMED and verified (pid $_wpid in two sessions, ref $HA_VMREF, GET 200, description matches, ${_wd_secs}s)"
    # the device: /dev/kfd and gfx942 from rocminfo's agent Name: field
    ha_ssh 90 'test -e /dev/kfd && echo KFD_PRESENT; rocminfo 2>/dev/null | awk '"'"'$1 == "Name:" && $2 ~ /^gfx[0-9a-f]+$/ {print "GFX=" $2}'"'"'; echo "GPU_AGENTS=$(rocminfo 2>/dev/null | awk '"'"'$1 == "Name:" && $2 ~ /^gfx[0-9a-f]+$/'"'"' | wc -l | tr -d " ")"; rocm-smi --showproductname 2>&1 | grep -i "card series" | head -4' \
        < /dev/null > "$_out/hotaisle_device.txt" 2>&1
    sed 's/^/    [box] /' "$_out/hotaisle_device.txt"
    grep -q '^KFD_PRESENT' "$_out/hotaisle_device.txt" || die "/dev/kfd is absent on the Hot Aisle VM; deleting"
    HA_GFX=$(sed -n 's/^GFX=//p' "$_out/hotaisle_device.txt" | sort -u | tr '\n' ' ' | sed 's/ $//')
    [ "$HA_GFX" = gfx942 ] || die "the Hot Aisle VM reads '$HA_GFX' from rocminfo, not gfx942; deleting"
    ha_rec "gfx=$HA_GFX gpu_agents=$(sed -n 's/^GPU_AGENTS=//p' "$_out/hotaisle_device.txt" | head -1)"
    return 0
}

# ha_teardown: 0 when nothing was created or the VM is verified gone (dead-man
# cancelled, slot released); 1 when it may still be billing (banner, dead-man
# and slot LEFT). Safe to call more than once.
ha_teardown() {
    if [ "$HA_CREATE_ATTEMPTED" = 1 ] && [ "$HA_GONE" = 0 ]; then
        if [ -z "$HA_VMREF" ]; then
            for _i in 1 2 3 4 5 6; do ha_adopt_new_vm && break; sleep 10; done
            [ -n "$HA_VMREF" ] && printf '%s\n' "$HA_VMREF" > "$HA_DEADMAN_DIR/vm_ref.txt" 2>/dev/null
        fi
        if [ -z "$HA_VMREF" ]; then
            ha_call GET "teams/$HA_TEAM/virtual_machines/"
            if [ "$HA_CODE" = 200 ] && [ -z "$(ha_py ids | while read -r _id _nm; do grep -qx "$_id" "$TMPD/ha_pre_ids.txt" || echo "$_id"; done)" ]; then
                ha_rec "verified_gone ref=none: the 200 listing shows no VM absent from the pre-create snapshot utc=$(ha_utc)"
                HA_GONE=1
            fi
        elif ha_delete_verify "$HA_VMREF" "$HA_VMNAME" "$HA_VERIFY_SECONDS"; then
            HA_GONE=1
        fi
        ha_rec "destroy_confirmed=$HA_GONE"
    fi
    ha_release_create_lock
    if [ "$HA_CREATE_ATTEMPTED" = 0 ] || [ "$HA_GONE" = 1 ]; then
        ha_cancel_deadman
        ha_release_slot
        return 0
    fi
    {
        echo "  ############################################################"
        echo "  # HOT AISLE VM ${HA_VMREF:-<unknown>} ${HA_VMNAME:+($HA_VMNAME) }MAY STILL BE BILLING."
        echo "  # The API did not confirm it is gone. The Mac dead-man (pid ${HA_DEADMAN_PID:-none})"
        echo "  # and the on-box watchdog are LEFT ARMED. End it by hand now:"
        echo "  #   bash tools/hotaisle_leg.sh reap ${HA_VMREF:-<vm>}"
        echo "  # Slot ${HA_SLOT:-none} stays HELD until then."
        echo "  ############################################################"
    } | tee -a "$HA_RECORD" >&2
    return 1
}
ha_spend() {  # a spend line from the price and the time since the create
    [ -n "$HA_T_CREATE" ] && [ -n "$HA_PRICE" ] || return 0
    _s=$(( $(date +%s) - HA_T_CREATE ))
    _b=$(( (_s + 59) / 60 )); [ "${HA_MINRES:-0}" -gt "$_b" ] 2>/dev/null && _b=$HA_MINRES
    printf 'spend_estimate=%s (%d s up, billed at least %d min at %s/h)\n' "$(ha_dollars $(( (HA_PRICE * _b + 59) / 60 )))" "$_s" "$_b" "$(ha_dollars "$HA_PRICE")"
}
