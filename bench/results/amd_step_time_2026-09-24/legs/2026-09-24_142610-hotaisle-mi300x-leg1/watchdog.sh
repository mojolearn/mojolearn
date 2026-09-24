#!/bin/sh
# Written by tools/hotaisle_leg.sh. RUNS ON THE VM AS ROOT, DETACHED (setsid).
# Sleeps to the lease deadline, then DELETEs THIS VM through the API with
# force. The key is in /var/lib/mojolearn-hotaisle/curlrc (0600), never in an argv.
set -u
trap '' HUP INT
echo $$ > /var/lib/mojolearn-hotaisle/watchdog.pid
T=$(( $(date +%s) + 3570 ))
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) armed ref=9ad27076-b752-4429-9262-c5f7348cff07 fires_in=3570s" >> /var/lib/mojolearn-hotaisle/watchdog.out
while [ "$(date +%s)" -lt "$T" ]; do sleep 10; done
n=1
while [ "$n" -le 20 ]; do
    code=$(curl -K /var/lib/mojolearn-hotaisle/curlrc --max-time 900 -o /var/lib/mojolearn-hotaisle/watchdog.body -w '%{http_code}' \
        -X DELETE 'https://admin.hotaisle.app/api/teams/andrews-team/virtual_machines/9ad27076-b752-4429-9262-c5f7348cff07/?force=true')
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) DELETE 9ad27076-b752-4429-9262-c5f7348cff07 attempt $n -> $code" >> /var/lib/mojolearn-hotaisle/watchdog.out
    case "$code" in 2*|404) exit 0 ;; esac
    sleep 15
    n=$((n + 1))
done
