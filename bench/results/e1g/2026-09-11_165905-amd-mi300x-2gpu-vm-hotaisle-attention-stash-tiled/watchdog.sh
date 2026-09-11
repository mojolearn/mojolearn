#!/bin/sh
# Written by tools/hotaisle_leg.sh. RUNS ON THE VM AS ROOT, DETACHED (setsid).
# Sleeps to the lease deadline, then DELETEs THIS VM through the API with
# force. The key is in /var/lib/mojolearn-hotaisle/curlrc (0600), never in an argv.
set -u
trap '' HUP INT
echo $$ > /var/lib/mojolearn-hotaisle/watchdog.pid
T=$(( $(date +%s) + 3572 ))
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) armed ref=210d0e97-ab28-40c4-8611-3f0682128fbd fires_in=3572s" >> /var/lib/mojolearn-hotaisle/watchdog.out
while [ "$(date +%s)" -lt "$T" ]; do sleep 10; done
n=1
while [ "$n" -le 20 ]; do
    code=$(curl -K /var/lib/mojolearn-hotaisle/curlrc --max-time 900 -o /var/lib/mojolearn-hotaisle/watchdog.body -w '%{http_code}' \
        -X DELETE 'https://admin.hotaisle.app/api/teams/andrews-team/virtual_machines/210d0e97-ab28-40c4-8611-3f0682128fbd/?force=true')
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) DELETE 210d0e97-ab28-40c4-8611-3f0682128fbd attempt $n -> $code" >> /var/lib/mojolearn-hotaisle/watchdog.out
    case "$code" in 2*|404) exit 0 ;; esac
    sleep 15
    n=$((n + 1))
done
