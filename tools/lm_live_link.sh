#!/usr/bin/env bash
# tools/lm_live_link.sh: join a live coordinator box and a live worker box.
#
#   bash tools/lm_live_link.sh OUTDIR "<nvidia ssh target>" "<amd ssh target>" [--port 7777] [--ready-seconds 3600]
#
# The part of tools/lm_live_leg.sh that runs once both boxes exist: wait for
# /root/lm_segment_ready on each, make an ephemeral ssh key, authorize it on
# the coordinator's box, hand the private half to the worker's box over ssh
# stdin, start an ssh tunnel ON THE WORKER'S BOX to the coordinator's port,
# and write /root/live_peer.txt so the worker connects. Used on its own when
# one runner had to be restarted (a create refused for capacity) while the
# other box was already up. A target is the runner's ssh words, e.g.
# "-p 17494 root@1.2.3.4" (RunPod) or "root@1.2.3.4" (DigitalOcean).
set -u
OUT=${1:?usage: lm_live_link.sh OUTDIR "<coordinator target>" "<worker target>" [--port N] [--ready-seconds N]}; shift
NV=${1:?coordinator ssh target}; shift
AMD=${1:?worker ssh target}; shift
PORT=7777; READY_SECONDS=3600
while [ $# -gt 0 ]; do
    case "$1" in
        --port) shift; PORT="$1" ;;
        --ready-seconds) shift; READY_SECONDS="$1" ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
    shift
done
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
say() { echo "[$(date +%H:%M:%S) link] $*" | tee -a "$OUT/link.log"; }
SSH_BASE=(-o StrictHostKeyChecking=accept-new -o "UserKnownHostsFile=$OUT/known_hosts" -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30)
# shellcheck disable=SC2086  # a target is several ssh words on purpose
box() { ssh "${SSH_BASE[@]}" $1 "$2"; }

NV_READY=0; AMD_READY=0
deadline=$(( $(date +%s) + READY_SECONDS ))
while [ "$(date +%s)" -lt "$deadline" ]; do
    [ "$NV_READY" = 0 ] && box "$NV" 'test -e /root/lm_segment_ready' 2>/dev/null && NV_READY=1 && say "coordinator ready: $NV"
    [ "$AMD_READY" = 0 ] && box "$AMD" 'test -e /root/lm_segment_ready' 2>/dev/null && AMD_READY=1 && say "worker ready: $AMD"
    [ "$NV_READY" = 1 ] && [ "$AMD_READY" = 1 ] && break
    sleep 15
done
if [ "$NV_READY" != 1 ] || [ "$AMD_READY" != 1 ]; then
    say "NOT READY (coordinator $NV_READY, worker $AMD_READY) after ${READY_SECONDS}s; telling the worker to stand down"
    box "$AMD" 'echo "none 0" > /root/live_peer.txt' >/dev/null 2>&1
    exit 1
fi
NV_PORT=$(printf '%s' "$NV" | sed -nE 's/.*-p ([0-9]+).*/\1/p'); [ -n "$NV_PORT" ] || NV_PORT=22
NV_HOST=$(printf '%s' "$NV" | sed -E 's/.*root@([^ ]+).*/\1/')
[ -f "$OUT/live_key" ] || ssh-keygen -q -t ed25519 -N '' -f "$OUT/live_key" -C "mojolearn-live-$(date +%s)" < /dev/null
box "$NV" 'umask 077; mkdir -p ~/.ssh; cat >> ~/.ssh/authorized_keys' < "$OUT/live_key.pub" && say "key authorized on the coordinator's box"
box "$AMD" 'umask 077; cat > /root/live_key' < "$OUT/live_key" && say "private key on the worker's box"
box "$AMD" "nohup ssh -i /root/live_key -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -N -L $PORT:127.0.0.1:$PORT -p $NV_PORT root@$NV_HOST > /root/live_tunnel.log 2>&1 &
sleep 4; (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -q \":$PORT \" && echo TUNNEL_UP || { cat /root/live_tunnel.log; echo TUNNEL_DOWN; }" | tee -a "$OUT/link.log"
grep -q TUNNEL_UP "$OUT/link.log" || { say "the tunnel did not come up; the worker is told to stand down"; box "$AMD" 'echo "none 0" > /root/live_peer.txt'; exit 1; }
box "$AMD" "echo '127.0.0.1 $PORT' > /root/live_peer.txt" && say "peer file written; the worker connects through the tunnel"
exit 0
