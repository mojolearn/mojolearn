#!/bin/sh
# Written by tools/do_extra_leg.sh. RUNS ON THE DROPLET, DETACHED. The local
# dead-man dies with the Mac; this one does not. The token is in
# /root/.mojolearn-do.curlrc (0600, delivered on ssh stdin), never in an argv.
set -u
sleep 3489
for attempt in 1 2 3; do
    code=$(curl -K /root/.mojolearn-do.curlrc --max-time 30 -o /root/selfkill.body -w '%{http_code}' \
        -X DELETE 'https://api.digitalocean.com/v2/droplets/599760325')
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) DELETE 599760325 attempt $attempt -> $code" >> /root/selfkill.out
    case "$code" in 2*|404) break ;; esac
    sleep 10
done
