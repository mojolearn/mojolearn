#!/usr/bin/env bash
# tools/live_xvendor_leg.sh: ONE live cross-vendor training run.
#
#   MOJOLEARN_PYTHON=.pixi/envs/default/bin/python \
#   bash tools/live_xvendor_leg.sh OUTDIR nvidia [amd]
#
# The coordinator and the Apple worker run on this Mac. Each named vendor is
# rented through its own guarded runner (nvidia: tools/gemm_remote_leg.sh on
# RunPod; amd: tools/hotaisle_leg.sh), with tools/live_xvendor_body.sh as the
# body, which builds and then waits. When a box reports ready, this script
# opens a reverse ssh tunnel (the box's 127.0.0.1:7777 -> this Mac's
# coordinator) and writes the box's shard list. K = 4 logical shards:
#   two workers   apple 0,1   other 2,3
#   three workers apple 0,1   nvidia 2   amd 3
# A box that is not ready within MOJOLEARN_LIVE_READY_SECONDS (default 2400)
# is left out, by name, and the run goes on with the others. Every step is
# held to the recorded one-process column (--expect). The runners keep their
# own dead-men, fetch and verified delete; nothing here rents or reaps.
# Host keys go to OUTDIR/known_hosts: cloud IPs are reused, and a stale key
# in ~/.ssh/known_hosts once kept a ready box from being seen at all.
set -u
OUT=${1:?usage: live_xvendor_leg.sh OUTDIR nvidia [amd]}; shift
[ $# -ge 1 ] || { echo "name at least one remote vendor: nvidia, amd" >&2; exit 2; }
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO" || exit 9
PY=${MOJOLEARN_PYTHON:-python3}
PORT=7777
EXPECT=bench/results/par_lm_xvendor/2026-09-21/apple-m4-1gpu.json
READY_SECONDS=${MOJOLEARN_LIVE_READY_SECONDS:-2400}
CARD=${MOJOLEARN_LIVE_APPLE_CARD:-$HOME/mojolearn-evidence/vendor-class-gaps-sep19/apple.card}
say() { echo "[$(date +%H:%M:%S) live] $*" | tee -a "$OUT/live.log"; }
say "commit $(git rev-parse HEAD) vendors: $*"

NV=0; AMD=0
for v in "$@"; do
    case "$v" in
        nvidia) NV=1
            MOJOLEARN_RUNPOD_KEY_FILE=${MOJOLEARN_RUNPOD_KEY_FILE:-$HOME/.mojolearn_runpod_key} \
            MOJOLEARN_GEMM_LEG_EXTRA=tools/live_xvendor_body.sh \
                sh tools/gemm_remote_leg.sh nvidia --rent --allow-concurrent --minutes 60 \
                --gpu "${MOJOLEARN_LIVE_NVIDIA_GPU:-NVIDIA H100 80GB HBM3}" --local-card "$CARD" \
                > "$OUT/nvidia-leg.log" 2>&1 &
            NV_PID=$! ;;
        amd) AMD=1
            MOJOLEARN_GEMM_LEG_EXTRA=tools/live_xvendor_body.sh MOJOLEARN_GEMM_LEG_OUT="$OUT/amd" \
                bash tools/hotaisle_leg.sh amd --rent --skip-gates > "$OUT/amd-leg.log" 2>&1 &
            AMD_PID=$! ;;
        *) echo "unknown vendor $v" >&2; exit 2 ;;
    esac
done

SSH_BASE=(-o StrictHostKeyChecking=accept-new -o "UserKnownHostsFile=$OUT/known_hosts" -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30)
nv_target() { grep -m1 'ssh target:' "$OUT/nvidia-leg.log" 2>/dev/null | sed 's/.*ssh target: //'; }
amd_target() {
    _l=$(grep -m1 'running after .*; ssh hotaisle@' "$OUT/amd-leg.log" 2>/dev/null) || return 0
    _ip=$(printf '%s' "$_l" | sed -E 's/.*ssh hotaisle@([^ ]+) -p ([0-9]+).*/\1/')
    _port=$(printf '%s' "$_l" | sed -E 's/.*ssh hotaisle@([^ ]+) -p ([0-9]+).*/\2/')
    echo "-p $_port -i $HOME/.ssh/id_ed25519 -o IdentitiesOnly=yes hotaisle@$_ip"
}
# shellcheck disable=SC2086  # a target is several ssh words on purpose
box() { ssh "${SSH_BASE[@]}" $1 "$2"; }

NV_READY=0; AMD_READY=0
deadline=$(( $(date +%s) + READY_SECONDS ))
while [ "$(date +%s)" -lt "$deadline" ]; do
    if [ "$NV" = 1 ] && [ "$NV_READY" = 0 ]; then
        t=$(nv_target); [ -n "$t" ] && box "$t" 'test -e /root/live_xvendor_ready || sudo -n test -e /root/live_xvendor_ready' 2>/dev/null && NV_READY=1 && say "nvidia ready: $t"
        kill -0 "$NV_PID" 2>/dev/null || { [ "$NV_READY" = 1 ] || { say "nvidia runner exited before ready"; NV=0; }; }
    fi
    if [ "$AMD" = 1 ] && [ "$AMD_READY" = 0 ]; then
        t=$(amd_target); [ -n "$t" ] && box "$t" 'test -e /root/live_xvendor_ready || sudo -n test -e /root/live_xvendor_ready' 2>/dev/null && AMD_READY=1 && say "amd ready: $t"
        kill -0 "$AMD_PID" 2>/dev/null || { [ "$AMD_READY" = 1 ] || { say "amd runner exited before ready"; AMD=0; }; }
    fi
    [ "$NV" = "$NV_READY" ] && [ "$AMD" = "$AMD_READY" ] && break
    sleep 15
done
[ "$NV" = 1 ] && [ "$NV_READY" = 0 ] && say "nvidia NOT ready after ${READY_SECONDS}s: left out"
[ "$AMD" = 1 ] && [ "$AMD_READY" = 0 ] && say "amd NOT ready after ${READY_SECONDS}s: left out"
WORKERS=$(( 1 + NV_READY + AMD_READY ))
if [ "$WORKERS" -lt 2 ]; then
    say "no remote worker came up; nothing live to run"
    for t in "$(nv_target)" "$(amd_target)"; do [ -n "$t" ] && box "$t" 'echo none > /root/live_xvendor_shards.txt || echo none | sudo -n tee /root/live_xvendor_shards.txt' >/dev/null 2>&1; done
    wait; exit 1
fi
if [ "$WORKERS" = 3 ]; then NV_SHARDS=2; AMD_SHARDS=3; else NV_SHARDS=2,3; AMD_SHARDS=2,3; fi

# Coordinator first, so every worker has something to reach.
MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$REPO/python" "$PY" tools/live_xvendor.py coordinator \
    --port "$PORT" --workers "$WORKERS" --out "$OUT/rows.json" --expect "$EXPECT" > "$OUT/coordinator.log" 2>&1 &
COORD_PID=$!
sleep 2
TUNNELS=""
open_box() {  # $1 target  $2 shards
    # shellcheck disable=SC2086
    ssh "${SSH_BASE[@]}" -N -o ExitOnForwardFailure=yes -R "$PORT:127.0.0.1:$PORT" $1 &
    TUNNELS="$TUNNELS $!"
    sleep 3
    box "$1" "echo $2 > /root/live_xvendor_shards.txt 2>/dev/null || echo $2 | sudo -n tee /root/live_xvendor_shards.txt > /dev/null"
}
[ "$NV_READY" = 1 ] && { open_box "$(nv_target)" "$NV_SHARDS"; say "nvidia shards $NV_SHARDS, tunnel up"; }
[ "$AMD_READY" = 1 ] && { open_box "$(amd_target)" "$AMD_SHARDS"; say "amd shards $AMD_SHARDS, tunnel up"; }

say "apple worker shards 0,1"
MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$REPO/python" "$PY" tools/mac_slot.py --timeout 3000 metal \
    "$PY" tools/live_xvendor.py worker --address "127.0.0.1:$PORT" --shards 0,1 --name apple-m4 \
    > "$OUT/apple-worker.log" 2>&1
say "apple worker exit=$?"
wait "$COORD_PID"; RC=$?
say "coordinator exit=$RC"
tail -12 "$OUT/coordinator.log" | tee -a "$OUT/live.log"
for p in $TUNNELS; do kill "$p" 2>/dev/null; done
say "waiting for the runners to fetch and delete their boxes"
wait
say "done"
exit "$RC"
