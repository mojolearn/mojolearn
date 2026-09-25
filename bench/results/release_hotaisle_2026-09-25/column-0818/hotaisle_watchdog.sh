#!/bin/sh
# tools/hotaisle_vm_lib.sh's ON-BOX watchdog. RUNS ON THE VM AS ROOT, DETACHED:
# sleeps to the lease and then DELETEs THIS VM through the API with force. The
# key is in the 0600 curl config beside this file (delivered on ssh stdin).
set -u
trap '' HUP INT
G='/root/wheel-smoke-guard'
echo $$ > "$G/watchdog.pid"
T=$(( $(date +%s) + 2675 ))
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) armed ref=fa15f2e1-4ee2-4432-b705-92a10143d336 fires_in=2675s" >> "$G/watchdog.out"
while [ "$(date +%s)" -lt "$T" ]; do sleep 10; done
n=1
while [ "$n" -le 20 ]; do
    code=$(curl -K "$G/curlrc" --max-time 900 -o "$G/watchdog.body" -w '%{http_code}' \
        -X DELETE 'https://admin.hotaisle.app/api/teams/andrews-team/virtual_machines/fa15f2e1-4ee2-4432-b705-92a10143d336/?force=true')
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) DELETE fa15f2e1-4ee2-4432-b705-92a10143d336 attempt $n -> $code" >> "$G/watchdog.out"
    case "$code" in 2*|404) exit 0 ;; esac
    sleep 15
    n=$((n + 1))
done
