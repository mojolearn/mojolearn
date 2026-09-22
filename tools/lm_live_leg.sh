#!/usr/bin/env bash
# tools/lm_live_leg.sh: the multi-vendor segment on two rented boxes.
#
#   bash tools/lm_live_leg.sh OUTDIR NVIDIA_BODY AMD_BODY [--minutes N --dollar-cap USD] [--amd runpod|do]
#
# The NVIDIA box (tools/gemm_remote_leg.sh, RunPod) runs the rendered
# live-coordinator body: it builds, fetches the checkpoint, and starts
# `lm_segment.py run --live-role coordinator`, which listens for the group.
# The AMD box (tools/do_extra_leg.sh on DigitalOcean by default, or
# gemm_remote_leg.sh amd on RunPod) runs the live-worker body, which builds
# and then waits for /root/live_peer.txt. When both boxes report
# /root/lm_segment_ready, this script makes an ephemeral ssh key, authorizes
# it on the NVIDIA pod, hands the private half to the AMD box over ssh stdin,
# starts an ssh tunnel ON THE AMD BOX to the NVIDIA pod's coordinator port,
# and writes the peer file. The gradients cross the wide-area link inside
# that tunnel, box to box; nothing passes through this Mac. The runners keep
# their own dead-men, fetch and verified delete; nothing here rents or reaps.
#
# Both bodies come from tools/lm_segment_leg.py render (mode live-coordinator
# and live-worker, the same --from checkpoint, complementary --live-shards).
set -u
OUT=${1:?usage: lm_live_leg.sh OUTDIR NVIDIA_BODY AMD_BODY [--minutes N --dollar-cap USD] [--amd do|runpod]}; shift
NV_BODY=${1:?NVIDIA body}; shift
AMD_BODY=${1:?AMD body}; shift
MINUTES=120; CAP=""; AMD_PROVIDER=do; PORT=7777
while [ $# -gt 0 ]; do
    case "$1" in
        --minutes) shift; MINUTES="$1" ;;
        --dollar-cap) shift; CAP="$1" ;;
        --amd) shift; AMD_PROVIDER="$1" ;;
        --port) shift; PORT="$1" ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
    shift
done
[ -n "$CAP" ] || { echo "--dollar-cap is required (the segment lease needs it)" >&2; exit 2; }
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO" || exit 9
READY_SECONDS=${MOJOLEARN_LIVE_READY_SECONDS:-3600}
TOKENS=${MOJOLEARN_LIVE_STAGE_KEYS:-corpus/fineweb-edu-10BT/tokens/mojolearn-bpe-fineweb-edu-50257-v1}
say() { echo "[$(date +%H:%M:%S) live] $*" | tee -a "$OUT/live.log"; }
say "commit $(git rev-parse HEAD); nvidia body $NV_BODY; amd body $AMD_BODY ($AMD_PROVIDER); lease $MINUTES min, cap \$$CAP"

if [ "$MINUTES" -gt 60 ]; then LEASE_ARGS=(--segment-lease "$MINUTES" --dollar-cap "$CAP"); else LEASE_ARGS=(--minutes "$MINUTES"); fi
# The NVIDIA box: walk the GPU types in MOJOLEARN_LIVE_NVIDIA_GPUS (a | list)
# until one has capacity; a create refused for capacity fails in seconds and
# the next type is tried. Do not pin one spec (a leg once starved 30 minutes).
nvidia_walk() {
    IFS='|' read -r -a _gpus <<< "${MOJOLEARN_LIVE_NVIDIA_GPUS:-NVIDIA H100 80GB HBM3|NVIDIA H200|NVIDIA H100 PCIe|NVIDIA H100 NVL|NVIDIA A100 80GB PCIe}"
    for gpu in "${_gpus[@]}"; do
        slug=$(printf '%s' "$gpu" | tr ' ' '_' | tr -cd 'A-Za-z0-9_')
        say "nvidia: trying $gpu"
        MOJOLEARN_GEMM_LEG_EXTRA="$NV_BODY" MOJOLEARN_STAGE_KEYS="$TOKENS" MOJOLEARN_GEMM_LEG_OUT="$OUT/nvidia-$slug" \
            sh tools/gemm_remote_leg.sh nvidia --rent --allow-concurrent "${LEASE_ARGS[@]}" --gpu "$gpu" > "$OUT/nvidia-$slug.log" 2>&1
        rc=$?
        if grep -q "no instances currently available" "$OUT/nvidia-$slug/create_response.json" 2>/dev/null; then
            say "nvidia: no capacity for $gpu"; continue
        fi
        ln -sf "nvidia-$slug.log" "$OUT/nvidia-leg.log"
        return $rc
    done
    say "nvidia: NO CAPACITY on any type"; return 3
}
nvidia_walk &
NV_PID=$!
sleep 5
case "$AMD_PROVIDER" in
    do)
        MOJOLEARN_GEMM_LEG_EXTRA="$AMD_BODY" MOJOLEARN_GPU_ARCHS=gfx942 MOJOLEARN_STAGE_KEYS="$TOKENS" MOJOLEARN_GEMM_LEG_OUT="$OUT/amd" \
            bash tools/do_extra_leg.sh amd "${LEASE_ARGS[@]}" > "$OUT/amd-leg.log" 2>&1 &
        AMD_PID=$! ;;
    runpod)
        MOJOLEARN_GEMM_LEG_EXTRA="$AMD_BODY" MOJOLEARN_GPU_ARCHS=gfx942 MOJOLEARN_STAGE_KEYS="$TOKENS" MOJOLEARN_GEMM_LEG_OUT="$OUT/amd" \
            sh tools/gemm_remote_leg.sh amd --rent --allow-concurrent "${LEASE_ARGS[@]}" > "$OUT/amd-leg.log" 2>&1 &
        AMD_PID=$! ;;
    *) echo "--amd must be do or runpod" >&2; exit 2 ;;
esac

SSH_BASE=(-o StrictHostKeyChecking=accept-new -o "UserKnownHostsFile=$OUT/known_hosts" -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30)
nv_target() { cat "$OUT"/nvidia-*.log 2>/dev/null | grep -m1 'ssh target:' | sed 's/.*ssh target: //'; }
amd_target() {
    case "$AMD_PROVIDER" in
        do) _ip=$(grep -m1 'active at ' "$OUT/amd-leg.log" 2>/dev/null | sed 's/.*active at //'); [ -n "$_ip" ] && echo "root@$_ip" ;;
        runpod) grep -m1 'ssh target:' "$OUT/amd-leg.log" 2>/dev/null | sed 's/.*ssh target: //' ;;
    esac
}
# shellcheck disable=SC2086  # a target is several ssh words on purpose
box() { ssh "${SSH_BASE[@]}" $1 "$2"; }

NV_READY=0; AMD_READY=0
deadline=$(( $(date +%s) + READY_SECONDS ))
while [ "$(date +%s)" -lt "$deadline" ]; do
    if [ "$NV_READY" = 0 ]; then
        t=$(nv_target); [ -n "$t" ] && box "$t" 'test -e /root/lm_segment_ready' 2>/dev/null && NV_READY=1 && say "nvidia ready: $t"
        kill -0 "$NV_PID" 2>/dev/null || { [ "$NV_READY" = 1 ] || { say "nvidia runner exited before ready"; break; }; }
    fi
    if [ "$AMD_READY" = 0 ]; then
        t=$(amd_target); [ -n "$t" ] && box "$t" 'test -e /root/lm_segment_ready' 2>/dev/null && AMD_READY=1 && say "amd ready: $t"
        kill -0 "$AMD_PID" 2>/dev/null || { [ "$AMD_READY" = 1 ] || { say "amd runner exited before ready"; break; }; }
    fi
    [ "$NV_READY" = 1 ] && [ "$AMD_READY" = 1 ] && break
    sleep 15
done
if [ "$NV_READY" != 1 ] || [ "$AMD_READY" != 1 ]; then
    say "NOT READY (nvidia $NV_READY, amd $AMD_READY); telling the worker to stand down as soon as it can be reached"
    # the worker's box may still be coming up; it waits an hour for the peer
    # file, so keep trying to reach it rather than leave it waiting on the bill
    for _i in $(seq 1 40); do
        t=$(amd_target)
        if [ -n "$t" ] && box "$t" 'echo "none 0" > /root/live_peer.txt' >/dev/null 2>&1; then say "worker told to stand down"; break; fi
        sleep 15
    done
    wait; exit 1
fi

# The tunnel: an ephemeral key, authorized on the NVIDIA pod, private half on the AMD box.
NV=$(nv_target); AMD=$(amd_target)
NV_PORT=$(printf '%s' "$NV" | sed -E 's/.*-p ([0-9]+).*/\1/'); NV_HOST=$(printf '%s' "$NV" | sed -E 's/.*root@([^ ]+).*/\1/')
ssh-keygen -q -t ed25519 -N '' -f "$OUT/live_key" -C "mojolearn-live-$(date +%s)" < /dev/null
box "$NV" 'umask 077; mkdir -p ~/.ssh; cat >> ~/.ssh/authorized_keys' < "$OUT/live_key.pub" && say "key authorized on the nvidia pod"
box "$AMD" 'umask 077; cat > /root/live_key' < "$OUT/live_key" && say "private key on the amd box"
box "$AMD" "nohup ssh -i /root/live_key -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -N -L $PORT:127.0.0.1:$PORT -p $NV_PORT root@$NV_HOST > /root/live_tunnel.log 2>&1 &
sleep 4; ss -ltn 2>/dev/null | grep -q ':$PORT ' && echo TUNNEL_UP || (cat /root/live_tunnel.log; echo TUNNEL_DOWN)" | tee -a "$OUT/live.log"
box "$AMD" "echo '127.0.0.1 $PORT' > /root/live_peer.txt" && say "peer file written on the amd box; the worker connects through the tunnel"

say "waiting for both runners to finish, fetch and delete their boxes"
wait "$NV_PID"; NV_RC=$?
wait "$AMD_PID"; AMD_RC=$?
say "nvidia runner exit=$NV_RC, amd runner exit=$AMD_RC"
for d in nvidia amd; do
    st=$(find "$OUT/$d" -name status.txt 2>/dev/null | head -1)
    [ -n "$st" ] && { echo "== $d status =="; cat "$st"; } | tee -a "$OUT/live.log"
done
[ "$NV_RC" = 0 ] && [ "$AMD_RC" = 0 ]
