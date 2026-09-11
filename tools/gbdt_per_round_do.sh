#!/usr/bin/env bash
# tools/gbdt_per_round_do.sh -- ONE MI325X leg for lane gbdt-per-round, taken
# in the shared DigitalOcean lock's agreed order, run through
# tools/do_extra_leg.sh (all rental guards live there), lock released only
# after a verified destroy.
#
#   nohup bash tools/gbdt_per_round_do.sh > <log> 2>&1 &
#
# THE LOCK (ENGINEERING_RULES.md section 10; order agreed 2026-09-11 between
# sessions): release-0.8.1, then the amd trees leg, then the neural attention
# leg (owner contains "extra:"), then the GBDT A/B legs. This script takes
# `mkdir /tmp/mojolearn-do-gpu.lock` only after an "extra:" owner has held and
# released it after a trees owner, or after the lock has sat free for 300 s
# with no taker. It never deletes a lock it did not create, and it removes its
# own only when GET /v2/droplets shows no droplet named mojolearn-extra-amd.
# After the leg it waits 180 s before exiting, so a follow-up leg cannot
# re-take the lock at once.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK=/tmp/mojolearn-do-gpu.lock
ME="gbdt-per-round"
TOKFILE="${MOJOLEARN_DO_TOKEN_FILE:-$HOME/.mojolearn_do_token}"
STAMP="$(date -u +%Y-%m-%d_%H%M%S)"
OUTREL="${GPR_LEG_OUT:-bench/results/e1g/${STAMP}-amd-mi325x-gbdt-per-round}"
POLL=15

log() { printf '[%s gpr-do] %s\n' "$(date -u +%T)" "$*"; }
api_get() {  # <path>: body on stdout; the token rides curl's stdin config
    printf 'header = "Authorization: Bearer %s"\n' "$(cat "$TOKFILE")" \
        | curl -s --max-time 20 --config - "https://api.digitalocean.com/v2$1"
}

seen_trees=0; seen_extra=0; free_since=""; last_owner=""
log "waiting for the lock's turn (poll ${POLL}s)"
while :; do
    if [ -d "$LOCK" ]; then
        owner="$(cat "$LOCK/owner" 2>/dev/null || true)"
        free_since=""
        case "$owner" in *extra:*) [ "$seen_trees" = 1 ] && seen_extra=1 ;; esac
        case "$owner" in *trees*) seen_trees=1 ;; esac
        if [ "$owner" != "$last_owner" ]; then
            log "lock held by: ${owner:-<no owner file>} (seen_trees=$seen_trees seen_extra=$seen_extra)"
            last_owner="$owner"
        fi
    else
        now=$(date +%s)
        [ -n "$free_since" ] || { free_since=$now; log "lock free (seen_trees=$seen_trees seen_extra=$seen_extra)"; }
        if [ "$seen_extra" = 1 ] || [ $((now - free_since)) -ge 300 ]; then
            if mkdir "$LOCK" 2>/dev/null; then
                echo "$ME $(date -u +%FT%TZ)" > "$LOCK/owner"
                log "lock TAKEN: $(cat "$LOCK/owner")"
                break
            fi
        fi
    fi
    sleep "$POLL"
done

log "leg out: $OUTREL"
MOJOLEARN_DO_TOKEN_FILE="$TOKFILE" MOJOLEARN_GPU_ARCHS=gfx942 \
    MOJOLEARN_GEMM_LEG_EXTRA=tools/gbdt_per_round_leg.sh \
    MOJOLEARN_GEMM_LEG_OUT="$OUTREL" \
    bash "$REPO/tools/do_extra_leg.sh" amd --minutes 60 --skip-gates
rc=$?
log "do_extra_leg exit $rc"

verified=0
for attempt in 1 2 3 4 5 6 7 8 9 10; do
    body="$(api_get '/droplets?per_page=200')"
    count="$(printf '%s' "$body" | python3 -c 'import json,sys
try:
    d = json.load(sys.stdin)
except Exception:
    print("ERR"); sys.exit(0)
if "droplets" not in d:
    print("ERR"); sys.exit(0)
print(sum(1 for x in d["droplets"] if x.get("name") == "mojolearn-extra-amd"))')"
    log "verify attempt $attempt: droplets named mojolearn-extra-amd = $count"
    if [ "$count" = 0 ]; then verified=1; break; fi
    sleep 15
done
if [ "$verified" = 1 ] && [ "$(cat "$LOCK/owner" 2>/dev/null | cut -d' ' -f1)" = "$ME" ]; then
    rm -f "$LOCK/owner" && rmdir "$LOCK" && log "lock RELEASED after verified destroy"
else
    log "lock NOT released (verified=$verified, owner '$(cat "$LOCK/owner" 2>/dev/null)'): check the droplet list by hand"
fi
log "post-leg wait 180 s"
sleep 180
log "done (leg exit $rc)"
exit "$rc"
