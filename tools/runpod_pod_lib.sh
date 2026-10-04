# tools/runpod_pod_lib.sh -- sourced, never run. The RunPod pod primitives of
# tools/runpod_cpu_leg.sh (2026-09-15), lifted verbatim so a second runner does
# not retype them: the key kept out of every argv, the REST call, the body
# reader, the DELETE that tries v1 then v2, the verify-gone that needs BOTH a
# 404/TERMINATED GET and absence from the listing, and the Mac dead-man.
# tools/release_wheel_smoke.sh uses it (2026-09-22). runpod_cpu_leg.sh still
# carries its own copy; it can source this file once the CPU build-box lane
# (lane/release-cpu-build-box), which edits that runner, has landed.
#
# The caller sets: TMPD (private temp dir), CURLRC (a path inside it),
# POD_NAME, and defines die(). Nothing here creates a pod.
RP=${RP:-https://rest.runpod.io/v1}
RP_V2=${RP_V2:-https://api.runpod.io/v2}

load_key() {
    _kf="${MOJOLEARN_RUNPOD_KEY_FILE:-$HOME/.mojolearn_runpod_key}"
    if [ -z "${RUNPOD_API_KEY:-}" ] && [ -f "$_kf" ]; then
        _perm=$(stat -f '%OLp' "$_kf" 2>/dev/null || stat -c '%a' "$_kf" 2>/dev/null)
        [ "$_perm" = 600 ] || die "key file $_kf is mode $_perm, must be 600"
        RUNPOD_API_KEY=$(cat "$_kf")
    fi
    [ -n "${RUNPOD_API_KEY:-}" ] || return 1
    export RUNPOD_API_KEY
    ( umask 077; printf 'header = "Authorization: Bearer %s"\nsilent\nshow-error\n' "$RUNPOD_API_KEY" > "$CURLRC" )
    return 0
}

rp_call() {  # METHOD URL [json file]; sets RP_CODE, body in $TMPD/rp.body
    : > "$TMPD/rp.body"
    if [ -n "${3:-}" ]; then
        RP_CODE=$(curl -K "$CURLRC" --max-time 60 -o "$TMPD/rp.body" -w '%{http_code}' -X "$1" \
            -H 'Content-Type: application/json' --data-binary "@$3" "$2" 2>>"$TMPD/curl.err") || RP_CODE=000
    else
        RP_CODE=$(curl -K "$CURLRC" --max-time 60 -o "$TMPD/rp.body" -w '%{http_code}' -X "$1" "$2" 2>>"$TMPD/curl.err") || RP_CODE=000
    fi
}

rp_py() {  # python over the parsed body `d` (a list becomes {"items": [...]})
    python3 - "$TMPD/rp.body" "$@" <<'PYEOF' 2>/dev/null || true
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
raw_pods = d if isinstance(d, list) else (d.get("items", d.get("pods")) if isinstance(d, dict) else None)
valid_pods = isinstance(raw_pods, list) and all(
    isinstance(p, dict) and isinstance(p.get("id"), str) and p["id"] for p in raw_pods)
if isinstance(d, list):
    d = {"items": d}
what = sys.argv[2]
pods = d.get("items") or d.get("pods") or []
if what == "id":
    print(d.get("id") or "")
elif what == "cost":
    print(d.get("costPerHr") or d.get("adjustedCostPerHr") or "")
elif what == "ssh":
    ip = d.get("publicIp") or ""
    pm = d.get("portMappings") or {}
    port = pm.get("22") if isinstance(pm, dict) else ""
    if ip and port:
        print("-p %s root@%s" % (port, ip))
elif what == "status":
    print(d.get("desiredStatus") or "")
elif what == "names":
    for p in pods:
        print("%s\t%s\t%s\t%s" % (p.get("id"), p.get("name"), p.get("desiredStatus"), p.get("costPerHr")))
elif what == "byname":
    print(" ".join(str(p.get("id")) for p in pods if p.get("name") == sys.argv[3]))
elif what == "listingvalid":
    print("yes" if valid_pods else "no")
elif what == "hasid":
    if valid_pods:
        print("yes" if any(p.get("id") == sys.argv[3] for p in raw_pods) else "no")
PYEOF
}

verify_gone() {  # pod id; 0 when the API says it is gone
    _i=1
    while [ "$_i" -le 8 ]; do
        rp_call GET "$RP/pods/$1"
        _code=$RP_CODE
        _st=$(rp_py status)
        rp_call GET "$RP/pods"
        _list_code=$RP_CODE
        _valid=$(rp_py listingvalid)
        _listed=$(rp_py hasid "$1")
        if { [ "$_code" = 404 ] || { [ "$_code" = 200 ] && [ "$_st" = TERMINATED ]; }; } &&
           { case "$_list_code" in 2*) true ;; *) false ;; esac; } &&
           [ "$_valid" = yes ] && [ "$_listed" = no ]; then
            echo "  VERIFIED: $1 is gone (GET pod HTTP $_code${_st:+ status $_st}; not in the pod listing)"
            return 0
        fi
        echo "  $1 not yet gone (GET HTTP $_code status '${_st:-?}', listed=$_listed), attempt $_i/8"
        sleep 10
        _i=$((_i + 1))
    done
    return 1
}

delete_pod() {
    for _u in "$RP/pods/$1" "$RP_V2/pods/$1"; do
        rp_call DELETE "$_u"
        echo "  DELETE $_u -> HTTP $RP_CODE"
        case "$RP_CODE" in 2*|404) break ;; esac
    done
}

write_deadman() {  # dir seconds; composes and checks, never arms
    ( umask 077; mkdir -p "$1"; if [ -f "$CURLRC" ]; then cp "$CURLRC" "$1/curlrc"; else : > "$1/curlrc"; fi )
    cat > "$1/deadman.sh" <<'DM_EOF'
#!/bin/sh
# tools/runpod_pod_lib.sh's Mac dead-man: ends the pod if the runner is gone.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
sleep @SECS@
echo "$(date -u +%FT%TZ) dead-man firing for @NAME@" >> "$D/deadman.log"
# Bounded requests and retries. On uncertainty preserve the credential for
# manual recovery and report failure; never call an unverified DELETE gone.
read_pods() {
    python3 - "$D/pods.json" "$1" "$2" <<'PODS'
import json, sys
value = json.load(open(sys.argv[1]))
pods = value if isinstance(value, list) else value.get('items', value.get('pods')) if isinstance(value, dict) else None
if not isinstance(pods, list) or not all(isinstance(p, dict) and isinstance(p.get('id'), str) and p['id'] for p in pods):
    raise SystemExit('invalid pod listing schema')
if sys.argv[2] == 'byname':
    print(' '.join(p['id'] for p in pods if p.get('name') == sys.argv[3]))
else:
    print('yes' if any(p['id'] == sys.argv[3] for p in pods) else 'no')
PODS
}
ids=""
[ -s "$D/pod_id.txt" ] && ids="$(cat "$D/pod_id.txt")"
attempt=0
while [ "$attempt" -lt 8 ]; do
    attempt=$((attempt + 1))
    code=$(curl -K "$D/curlrc" --connect-timeout 10 --max-time 30 -o "$D/pods.json" -w '%{http_code}' "@RP@/pods" 2>>"$D/deadman.log") || code=000
    case "$code" in 2*) ;; *) sleep 15; continue ;; esac
    matched=$(read_pods byname "@NAME@" 2>>"$D/deadman.log") || { sleep 15; continue; }
    [ -n "$ids" ] || ids="$matched"
    if [ -z "$ids" ]; then
        echo "$(date -u +%FT%TZ) no matching pod in valid listing after deadman deadline" >> "$D/deadman.log"
        rm -f "$D/curlrc"
        exit 0
    fi
    gone=1
    for id in $ids; do
        listed=$(read_pods hasid "$id" 2>>"$D/deadman.log") || listed=unknown
        state_code=$(curl -K "$D/curlrc" --connect-timeout 10 --max-time 30 -o "$D/pod.json" -w '%{http_code}' "@RP@/pods/$id" 2>>"$D/deadman.log") || state_code=000
        state=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("desiredStatus", ""))' "$D/pod.json" 2>/dev/null) || state=unknown
        if [ "$listed" = no ] && { [ "$state_code" = 404 ] || { [ "$state_code" = 200 ] && [ "$state" = TERMINATED ]; }; }; then
            continue
        fi
        gone=0
        for u in "@RP@/pods/$id" "@RPV2@/pods/$id"; do
            c=$(curl -K "$D/curlrc" --connect-timeout 10 --max-time 30 -o /dev/null -w '%{http_code}' -X DELETE "$u" 2>>"$D/deadman.log") || c=000
            echo "$(date -u +%FT%TZ) DELETE $u -> $c" >> "$D/deadman.log"
            case "$c" in 2*|404) break ;; esac
        done
    done
    if [ "$gone" = 1 ]; then
        echo "$(date -u +%FT%TZ) termination VERIFIED" >> "$D/deadman.log"
        rm -f "$D/curlrc"
        exit 0
    fi
    sleep 15
done
echo "$(date -u +%FT%TZ) TEARDOWN UNCONFIRMED after bounded retries; credential retained for recovery" >> "$D/deadman.log"
exit 1
DM_EOF
    sed -i.bak -e "s|@SECS@|$2|g" -e "s|@NAME@|$POD_NAME|g" -e "s|@RP@|$RP|g" -e "s|@RPV2@|$RP_V2|g" "$1/deadman.sh"
    rm -f "$1/deadman.sh.bak"
    if grep -q '@[A-Z0-9]*@' "$1/deadman.sh"; then return 1; fi
    sh -n "$1/deadman.sh"
}

# with_timeout SECONDS COMMAND...: the Mac has no `timeout`. perl's alarm
# survives exec, so the command itself is SIGALRM'd at the deadline (exit 142).
# 2026-09-22: a source upload over ssh stalled for 14 minutes with no bound.
with_timeout() {
    _secs=$1; shift
    perl -e 'alarm shift @ARGV; exec @ARGV or exit 127' "$_secs" "$@"
}
