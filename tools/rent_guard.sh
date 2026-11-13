#!/bin/bash
# tools/rent_guard.sh -- THE RUNPOD ACCOUNT CAP, checked before every create (2026-11-13).
#
#   bash tools/rent_guard.sh [--own-cap N]
#
# Counts the live pods on the RunPod account through the API (the key in ~/.mojolearn_runpod_key, or
# RUNPOD_API_KEY; query { myself { pods { id name } } }) and exits 0 only when the count is under
# MOJOLEARN_RUNPOD_CAP (default 8, Andrew's cap; the account is shared with other projects' fleets, so every pod
# counts, not only ours). --own-cap N also caps Mojolearn's own pods (named mojolearn-*) below N (tools/dev_pod.sh
# MAX_RUNPOD_PODS). Exit 3 = refused at or above a cap; exit 2 = the count could not be read (refused too: an
# unknown count never rents). It prints one line: "rent_guard: <live> live RunPod pods (<own> mojolearn-*), cap
# <cap>[, own cap <n>]: OK|REFUSED".
#
# Why: on 2026-11-13 the account had 13 live pods, above the cap of 8, and `tools/nvidia_central.sh up` still rented
# twice, because the only cap check was a hook that matched the devbox script's "up" in the command text. The cap
# now lives where the create happens: tools/dev_pod.sh (the only creator of RunPod pods) runs this guard before
# every create attempt (runpod_create), for NVIDIA and AMD alike, whoever called it.
#
# Tests (no API): MOJOLEARN_RENT_GUARD_TEST_PODS="<live> <own>" stands in for the API count;
# tools/test_rent_guard.sh.
set -uo pipefail
CAP="${MOJOLEARN_RUNPOD_CAP:-8}"
OWN_CAP=""
while [ $# -gt 0 ]; do
    case "$1" in
        --own-cap) OWN_CAP="${2:-}"; shift 2 || shift ;;
        --own-cap=*) OWN_CAP="${1#--own-cap=}"; shift ;;
        *) echo "rent_guard: usage: rent_guard.sh [--own-cap N]" >&2; exit 2 ;;
    esac
done
[[ "$CAP" =~ ^[0-9]+$ ]] || { echo "rent_guard: MOJOLEARN_RUNPOD_CAP must be a whole number, not '$CAP'; refused" >&2; exit 2; }
[ -z "$OWN_CAP" ] || [[ "$OWN_CAP" =~ ^[0-9]+$ ]] || { echo "rent_guard: --own-cap must be a whole number, not '$OWN_CAP'; refused" >&2; exit 2; }

count_pods() {  # prints "<live> <own>" or nothing
    if [ -n "${MOJOLEARN_RENT_GUARD_TEST_PODS:-}" ]; then
        echo "$MOJOLEARN_RENT_GUARD_TEST_PODS"; return
    fi
    local key="${RUNPOD_API_KEY:-}" kf="${MOJOLEARN_RUNPOD_KEY_FILE:-$HOME/.mojolearn_runpod_key}"
    [ -n "$key" ] || key=$(cat "$kf" 2> /dev/null) || true
    [ -n "$key" ] || { echo "rent_guard: no RunPod key ($kf or RUNPOD_API_KEY)" >&2; return; }
    # the key goes to curl on stdin (a config line), never on the command line
    printf 'header = "Authorization: Bearer %s"\n' "$key" \
        | curl -s -m 20 -K - -H "Content-Type: application/json" "${MOJOLEARN_RENT_GUARD_API:-https://api.runpod.io/graphql}" \
            -d '{"query":"query { myself { pods { id name } } }"}' \
        | python3 -c 'import sys, json
pods = json.load(sys.stdin)["data"]["myself"]["pods"]
print(len(pods), sum(1 for p in pods if str(p.get("name") or "").startswith("mojolearn-")))' 2> /dev/null
}

read -r LIVE OWN <<< "$(count_pods)"
if ! [[ "${LIVE:-}" =~ ^[0-9]+$ && "${OWN:-}" =~ ^[0-9]+$ ]]; then
    echo "rent_guard: could not count the live RunPod pods; refused (cap $CAP)" >&2
    exit 2
fi
msg="rent_guard: $LIVE live RunPod pods ($OWN mojolearn-*), cap $CAP${OWN_CAP:+, own cap $OWN_CAP}"
if [ "$LIVE" -ge "$CAP" ]; then
    echo "$msg: REFUSED. The account is at or above MOJOLEARN_RUNPOD_CAP; nothing is rented. Ask Andrew to bring the" \
        "live pods under $CAP or to approve this box explicitly." >&2
    exit 3
fi
if [ -n "$OWN_CAP" ] && [ "$OWN" -ge "$OWN_CAP" ]; then
    echo "$msg: REFUSED. $OWN mojolearn pods are live and the own cap is $OWN_CAP (MOJOLEARN_DEVPOD_MAX_PODS); share" \
        "a pod or bring one down." >&2
    exit 3
fi
echo "$msg: OK"
exit 0
