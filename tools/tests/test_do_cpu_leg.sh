#!/bin/bash
# tools/tests/test_do_cpu_leg.sh -- argument parsing and refusal paths of
# tools/do_cpu_leg.sh with the DigitalOcean API mocked. No network, no provider.
# Run: bash tools/tests/test_do_cpu_leg.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; LEG="$HERE/../do_cpu_leg.sh"
W="$(mktemp -d "${TMPDIR:-/tmp}/test-do-cpu-leg.XXXXXX")"; trap 'rm -rf "$W"' EXIT
export MOJOLEARN_DO_CPU_STATE="$W/state" MOJOLEARN_STEWARD_DO_STATE="$W/steward" \
       MOJOLEARN_DO_TOKEN_FILE="$W/token" MOJOLEARN_DO_SELFKILL_TOKEN_FILE="$W/none" \
       MOJOLEARN_DO_API=https://mock.invalid/v2 MOJOLEARN_DO_CURL="$W/curl" MOCK="$W/mock"
mkdir -p "$W/state" "$W/steward" "$W/mock"
cat > "$W/curl" <<'M'
#!/bin/bash
# mock curl: logs "METHOD URL"; body from $MOCK/<METHOD>_<path with / ? & = -> _>, code from <same>.code
out=/dev/null; m=GET; url=""
while [ $# -gt 0 ]; do case "$1" in -o) out=$2; shift ;; -X) m=$2; shift ;; -K|-w|-H|--max-time|--data-binary) shift ;; http*) url=$1 ;; esac; shift; done
echo "$m $url" >> "$MOCK/calls.log"
k="${m}_$(echo "${url#https://mock.invalid/v2/}" | tr '/?&=' '____')"
if [ -f "$MOCK/$k" ]; then cat "$MOCK/$k" > "$out"; printf '%s' "$(cat "$MOCK/$k.code" 2>/dev/null || echo 200)"
else echo '{}' > "$out"; printf 404; fi
M
chmod +x "$W/curl"
P=0; F=0
check() {  # desc expected_rc pattern -- cmd...
    local desc=$1 erc=$2 pat=$3; shift 4
    out=$("$@" 2>&1); rc=$?
    if [ "$rc" = "$erc" ] && echo "$out" | grep -qE -- "$pat"; then P=$((P+1)); echo "ok   $desc"
    else F=$((F+1)); echo "FAIL $desc (rc=$rc want $erc; pattern '$pat')"; echo "$out" | tail -n 5 | sed 's/^/     /'; fi
}
check "no args prints usage"           2 'THE ONLY WAY'                  -- bash "$LEG"
check "unknown command"                1 "unknown command"               -- bash "$LEG" frob
check "up needs a name"                1 'usage: .* up <name>'           -- bash "$LEG" up
check "up rejects a bad name"          1 'name must be'                  -- bash "$LEG" up Bad_Name
check "up rejects minutes over max"    1 'hard maximum lease \(720\)'    -- bash "$LEG" up b1 721
check "up rejects non-numeric minutes" 1 'whole number'                  -- bash "$LEG" up b1 lots
check "extend rejects over max"        1 'hard maximum lease'            -- bash "$LEG" extend b1 9999
check "up refused without token"       1 'no DigitalOcean token'         -- bash "$LEG" up b1
echo tok > "$W/token"
mkdir -p "$W/state/b1"; echo DROPLET_ID=1 > "$W/state/b1/state.env"; echo $(( $(date +%s) + 9999 )) > "$W/state/b1/mac_deadline"; echo 1 > "$W/state/b1/droplet_id"
check "up refused when state exists"   1 'state file exists'             -- bash "$LEG" up b1
for n in b2 b3 b4; do cp -R "$W/state/b1" "$W/state/$n"; echo "${n#b}" > "$W/state/$n/droplet_id"; done
check "up refused at 4 live legs"      1 '4 CPU legs are live'           -- bash "$LEG" up b5
echo 0 > "$W/state/b4/mac_deadline"     # stale: no longer counts as live
python3 -c 'import json; print(json.dumps({"droplets":[{"id":i} for i in range(4)]}))' > "$MOCK/GET_droplets_tag_name_mojolearn-cpu-leg_per_page_200"
check "up refused at 4 tagged in account" 1 '4 droplets tagged'          -- bash "$LEG" up b5
: > "$MOCK/calls.log"
check "up --dry-run prints the body"   0 '"tags":\["mojolearn-cpu-leg"\]' -- bash "$LEG" up b5 60 --dry-run
check "dry-run says the bill line"     0 'THE BILL STARTS HERE'          -- bash "$LEG" --dry-run up b5
check "dry-run name carries deadline"  0 'mojolearn-cpu-b5-until-[0-9]{8}z' -- bash "$LEG" up b5 --dry-run
check "dry-run made no API call"       0 '^$'                            -- cat "$MOCK/calls.log"
check "dry-run left no state"          0 '^$'                            -- sh -c "test ! -e '$W/state/b5'"
check "status lists legs, stale flagged" 0 'b4 .*STALE'                  -- bash "$LEG" status
# sweep: 1 tracked (b1), 4 stale-tracked (b4 -> untracked), old cpu, young cpu, steward, foreign
old=$(python3 -c 'import time; print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time()-86400)))')
new=$(python3 -c 'import time; print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time()-600)))')
echo "DROPLET_ID='77'" > "$W/steward/state.env"
cat > "$MOCK/GET_droplets_per_page_200" <<J
{"droplets":[
 {"id":1,"name":"mojolearn-cpu-b1-until-10111200z","size_slug":"s-8vcpu-16gb-amd","created_at":"$old","tags":["mojolearn-cpu-leg"]},
 {"id":4,"name":"mojolearn-cpu-b4-until-10101200z","size_slug":"s-8vcpu-16gb-amd","created_at":"$old","tags":["mojolearn-cpu-leg"]},
 {"id":50,"name":"mojolearn-targeted-ab-amd-cpu-6fe3cfce38","size_slug":"s-8vcpu-16gb-amd","created_at":"$old","tags":["x"]},
 {"id":51,"name":"mojolearn-cpu-young","size_slug":"s-8vcpu-16gb-amd","created_at":"$new","tags":["mojolearn-cpu-leg"]},
 {"id":77,"name":"mojolearn-steward-do-amd","size_slug":"gpu-mi300x1-192gb","created_at":"$old","tags":["mojolearn-steward"]},
 {"id":78,"name":"mojolearn-steward-do-amd","size_slug":"gpu-mi300x1-192gb","created_at":"$old","tags":["mojolearn-steward"]},
 {"id":90,"name":"someone-else","size_slug":"s-1vcpu-1gb","created_at":"$old","tags":[]}]}
J
check "sweep skips tracked leg"        0 '^$'                            -- sh -c "bash '$LEG' sweep | grep 'UNTRACKED 1 ' || true"
check "sweep skips tracked steward"    0 '^$'                            -- sh -c "bash '$LEG' sweep | grep 'UNTRACKED 77 ' || true"
check "sweep lists stale leg"          0 'UNTRACKED 4 '                  -- bash "$LEG" sweep
check "sweep flags untracked steward"  0 'never auto-deleted'            -- bash "$LEG" sweep
check "sweep prints delete commands"   0 'DELETE https://mock.invalid/v2/droplets/90' -- bash "$LEG" sweep
: > "$MOCK/calls.log"
for id in 4 50; do echo '' > "$MOCK/DELETE_droplets_$id"; echo 204 > "$MOCK/DELETE_droplets_$id.code"; done
check "sweep --delete deletes old ours" 0 'sweep deleted 50 .*verified 404' -- bash "$LEG" sweep --delete
check "sweep --delete hit only 4,50"   0 '^DELETE .*/4 DELETE .*/50 $'   -- sh -c "grep DELETE '$MOCK/calls.log' | sort | tr '\n' ' '"
: > "$MOCK/calls.log"
check "sweep --delete --dry-run calls no DELETE" 0 'DRY-RUN DELETE'      -- bash "$LEG" sweep --delete --dry-run
check "dry sweep logged no DELETE"     0 '^$'                            -- sh -c "grep DELETE '$MOCK/calls.log' || true"
check "down needs a name"              1 'usage: .* down'                -- bash "$LEG" down
check "down unknown leg"               1 "no leg 'zz'"                   -- bash "$LEG" down zz
echo '' > "$MOCK/DELETE_droplets_1"; echo 204 > "$MOCK/DELETE_droplets_1.code"
check "down verifies 404 then releases" 0 'verified gone \(GET 404\)'    -- bash "$LEG" down b1
check "down removed state"             0 '^$'                            -- sh -c "test ! -e '$W/state/b1/state.env'"
echo "passed=$P failed=$F"; [ "$F" = 0 ]
