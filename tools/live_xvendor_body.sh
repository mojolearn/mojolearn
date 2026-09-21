#!/bin/sh
# tools/live_xvendor_body.sh: the on-box body of a live cross-vendor worker,
# for tools/gemm_remote_leg.sh (RunPod) or tools/hotaisle_leg.sh (AMD), driven
# by tools/live_xvendor_leg.sh on the Mac, which runs the coordinator.
#
# Builds the byte-LM binding, then waits for two things the Mac provides once
# it sees the box: /root/live_xvendor_shards.txt (this worker's shards, e.g.
# "2,3") and a reverse ssh tunnel that makes the Mac's coordinator reachable
# at 127.0.0.1:7777 on this box. Then one worker (tools/live_xvendor.py) runs
# until the coordinator says done or refuses. POSIX sh; set -u, not set -e.
set -u
cd /root/mojolearn || exit 9
OUT=/root/gemm_leg_out/live-xvendor
mkdir -p "$OUT/logs"
G="$OUT/gate.txt"
PATH="$PATH:/opt/rocm/bin"
export PATH
say() { echo "$@" >> "$G"; }
run() {
    _n=$1; shift
    _t0=$(date +%s)
    "$@" > "$OUT/logs/$_n.log" 2>&1
    _e=$?
    say "$_n exit=$_e secs=$(( $(date +%s) - _t0 ))"
    return "$_e"
}
say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ ! -s commit.txt ]; then
    _c=$(sed -n 's/^commit=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1 | awk '{print $1}')
    case "$_c" in [0-9a-f][0-9a-f]*) printf '%s\n' "$_c" > commit.txt ;; esac
fi
say "commit=$(head -1 commit.txt 2>/dev/null)"

if command -v nvidia-smi > /dev/null 2>&1; then
    VENDOR=nvidia
    if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
        _cc=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
        case "$_cc" in 9.0) MOJOLEARN_GPU_ARCHS=sm_90a ;; *) MOJOLEARN_GPU_ARCHS="sm_$(echo "$_cc" | tr -d .)" ;; esac
    fi
    nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version --format=csv > "$OUT/logs/device.txt" 2>&1
else
    VENDOR=amd
    [ -n "${MOJOLEARN_GPU_ARCHS:-}" ] || MOJOLEARN_GPU_ARCHS=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-z]+')
    { rocm-smi --showproductname --showuniqueid --showbus; cat /sys/module/amdgpu/version; } > "$OUT/logs/device.txt" 2>&1
fi
export MOJOLEARN_GPU_ARCHS
TAG="$VENDOR-$MOJOLEARN_GPU_ARCHS"
say "vendor=$VENDOR arch=$MOJOLEARN_GPU_ARCHS tag=$TAG"

BINCACHE="sh"
if command -v python3 > /dev/null 2>&1 && [ -f tools/bincache.py ]; then
    BINCACHE="python3 tools/bincache.py build"
fi
run portable_math env PYTHONPATH=/root/mojolearn/packaging/portable_math \
    pixi run python -c "import pathlib, stage; stage.build(pathlib.Path('/root/mojolearn/python/mojolearn/.libs/libMojolearnMath.so'))"
for b in build build_byte_lm; do
    # shellcheck disable=SC2086  # $BINCACHE is one word or three, split on purpose
    run "$b" env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 \
        MOJOLEARN_COMPILE_JOBS="${MOJOLEARN_COMPILE_JOBS:-8}" $BINCACHE "bindings/$b.sh"
done
sha256sum python/mojolearn/identical/_mojolearn_byte_lm.so >> "$G" 2>&1
pixi run python -c 'import numpy' > /dev/null 2>&1 || pixi run python -m pip install numpy > "$OUT/logs/numpy.log" 2>&1

# The build is done: say so where the Mac can see it, then wait for the shards.
touch /root/live_xvendor_ready
say "ready=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
_waited=0
while [ ! -s /root/live_xvendor_shards.txt ] && [ "$_waited" -lt 1800 ]; do
    sleep 5; _waited=$(( _waited + 5 ))
done
SHARDS=$(tr -cd '0-9,' < /root/live_xvendor_shards.txt 2>/dev/null)
if [ -z "$SHARDS" ]; then
    say "NO SHARDS after ${_waited}s: the Mac never assigned this worker; nothing run"
    exit 3
fi
say "shards=$SHARDS"
run worker env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
    pixi run python tools/live_xvendor.py worker --address 127.0.0.1:7777 --shards "$SHARDS" \
    --name "$TAG" --connect-timeout 600
tail -5 "$OUT/logs/worker.log" >> "$G"
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
