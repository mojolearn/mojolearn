#!/bin/bash
# tools/test_rent_guard.sh -- tools/rent_guard.sh with NO RunPod call: MOJOLEARN_RENT_GUARD_TEST_PODS stands in
# for the API count, and an unreachable local URL stands in for an API that does not answer. Also checks that
# tools/dev_pod.sh runs the guard in runpod_create before the dead-man is armed and the create is posted.
#   bash tools/test_rent_guard.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
G="$ROOT/tools/rent_guard.sh"
fails=0
check() {  # <name> <command>
    if bash -c "$2" > /dev/null 2>&1; then echo "ok   $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi
}
run() { MOJOLEARN_RENT_GUARD_TEST_PODS="$1" bash "$G" "${@:2}" 2>&1; echo "rc=$?"; }

out=$(run "13 0")
check "13 live pods (another project's), cap 8: refused with exit 3" "grep -q 'rc=3' <<< \"$out\" && grep -q REFUSED <<< \"$out\""
out=$(run "8 0")
check "exactly at the cap is refused" "grep -q 'rc=3' <<< \"$out\""
out=$(run "7 0")
check "under the cap passes" "grep -q 'rc=0' <<< \"$out\" && grep -q ': OK' <<< \"$out\""
out=$(MOJOLEARN_RUNPOD_CAP=3 MOJOLEARN_RENT_GUARD_TEST_PODS="5 0" bash "$G" 2>&1; echo "rc=$?")
check "MOJOLEARN_RUNPOD_CAP overrides the default" "grep -q 'rc=3' <<< \"$out\" && grep -q 'cap 3' <<< \"$out\""
out=$(run "4 3" --own-cap 3)
check "own mojolearn-* pods at --own-cap are refused" "grep -q 'rc=3' <<< \"$out\" && grep -q 'own cap is 3' <<< \"$out\""
out=$(run "4 2" --own-cap=3)
check "own pods under --own-cap pass" "grep -q 'rc=0' <<< \"$out\""
out=$(RUNPOD_API_KEY=fake MOJOLEARN_RENT_GUARD_API=http://127.0.0.1:9/graphql bash "$G" 2>&1; echo "rc=$?")
check "an API that does not answer refuses (exit 2): an unknown count never rents" "grep -q 'rc=2' <<< \"$out\""
out=$(MOJOLEARN_RUNPOD_CAP=x MOJOLEARN_RENT_GUARD_TEST_PODS="1 0" bash "$G" 2>&1; echo "rc=$?")
check "a malformed cap refuses" "grep -q 'rc=2' <<< \"$out\""

# dev_pod.sh: the guard runs inside runpod_create, before arm_mac_deadman and the POST
body=$(awk '/^runpod_create\(\)/,/^}/' "$ROOT/tools/dev_pod.sh")
g=$(grep -n 'tools/rent_guard.sh' <<< "$body" | head -1 | cut -d: -f1)
a=$(grep -n 'arm_mac_deadman' <<< "$body" | head -1 | cut -d: -f1)
p=$(grep -n 'rp_call POST' <<< "$body" | head -1 | cut -d: -f1)
check "dev_pod.sh runpod_create runs the guard before the dead-man and the create" \
    "[ -n '$g' ] && [ -n '$a' ] && [ -n '$p' ] && [ '$g' -lt '$a' ] && [ '$g' -lt '$p' ]"
check "dev_pod.sh dies when the guard refuses" "grep -q 'rent_guard.sh refused the create' '$ROOT/tools/dev_pod.sh'"
check "nvidia_central.sh up prints the create line" "grep -q 'CREATE: MOJOLEARN_DEVPOD_VIA_CENTRAL=1' '$ROOT/tools/nvidia_central.sh'"
check "do_amd_steward.sh up prints the create call" "grep -q 'CREATE: POST' '$ROOT/tools/do_amd_steward.sh'"

[ "$fails" = 0 ] && echo "rent_guard: all checks passed" || echo "rent_guard: $fails check(s) FAILED"
[ "$fails" = 0 ]
