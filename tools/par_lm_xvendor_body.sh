#!/bin/sh
# tools/par_lm_xvendor_body.sh: the on-box body (MOJOLEARN_GEMM_LEG_EXTRA for
# tools/gemm_remote_leg.sh) of the small multi-GPU language model evidence leg.
#
#   MOJOLEARN_GEMM_LEG_GPU_COUNT=2 \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/par_lm_xvendor_body.sh \
#   sh tools/gemm_remote_leg.sh nvidia --rent --allow-concurrent      (or amd)
#
# Evidence, not a verifier lane. What runs is tools/par_lm_xvendor.py, whose
# docstring states the claim. On this box:
#   one      1 GPU, K=4 logical shards, 6 steps, shard gradients kept, state
#            after step 3 written as a handoff file
#   two      2 GPUs, the same K and recipe: must equal `one` bit for bit
#   two-from-one   2 GPUs resumed from `one`'s step 3 bytes (device-count handoff)
#   from-apple     1 GPU resumed from the M4's step 3 bytes, committed under
#                  bench/results/par_lm_xvendor/ (vendor handoff), if present
#   compare  on the box, all of the above, including every mixed assignment
#            of shard gradients across the runs
# Each run is ONCE. The whole body is minutes after the builds.
#
# POSIX sh (the pod's /bin/sh is dash); set -u and NOT set -e, so a failed
# phase is a finding whose log still comes home.
set -u
cd /root/mojolearn || exit 9
OUT=/root/gemm_leg_out/par-lm-xvendor
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
    echo "$_n	$_e	$(( $(date +%s) - _t0 ))" >> "$OUT/status.tsv"
    say "$_n exit=$_e secs=$(( $(date +%s) - _t0 ))"
    return "$_e"
}
say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# The commit, from the runner's own record. Never typed.
if [ ! -s commit.txt ]; then
    _c=$(sed -n 's/^commit=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1 | awk '{print $1}')
    case "$_c" in [0-9a-f][0-9a-f]*) printf '%s\n' "$_c" > commit.txt ;; esac
fi
say "commit=$(head -1 commit.txt 2>/dev/null)"

# ------------------------------------------------------------ vendor, arch, GPU count
if command -v nvidia-smi > /dev/null 2>&1; then
    VENDOR=nvidia
    NGPU=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')
    if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
        _cc=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
        case "$_cc" in 9.0) MOJOLEARN_GPU_ARCHS=sm_90a ;; *) MOJOLEARN_GPU_ARCHS="sm_$(echo "$_cc" | tr -d .)" ;; esac
    fi
    nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version --format=csv > "$OUT/logs/device.txt" 2>&1
else
    VENDOR=amd
    NGPU=$(rocminfo 2>/dev/null | grep -c -E '^ *Name: *gfx')
    [ -n "${MOJOLEARN_GPU_ARCHS:-}" ] || MOJOLEARN_GPU_ARCHS=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-z]+')
    { rocm-smi --showproductname --showuniqueid --showbus; cat /sys/module/amdgpu/version; } > "$OUT/logs/device.txt" 2>&1
fi
export MOJOLEARN_GPU_ARCHS
say "vendor=$VENDOR gpus=$NGPU arch=$MOJOLEARN_GPU_ARCHS"

# ------------------------------------------------------------ builds
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

PY="env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python pixi run python tools/par_lm_xvendor.py"
TAG="$VENDOR-$MOJOLEARN_GPU_ARCHS"

# ------------------------------------------------------------ runs, each once
run one $PY run --out "$OUT/$TAG-one.json" --devices 0 --grads "$OUT/$TAG-one.npz" --label "$TAG-1gpu"
RECS="$OUT/$TAG-one.json"
GRADS="$OUT/$TAG-one.npz"
if [ "$NGPU" -ge 2 ]; then
    run two $PY run --out "$OUT/$TAG-two.json" --devices 0,1 --grads "$OUT/$TAG-two.npz" --label "$TAG-2gpu"
    run two-from-one $PY run --out "$OUT/$TAG-two-from-one.json" --devices 0,1 \
        --resume "$OUT/$TAG-one.handoff.npz" --label "$TAG-2gpu-resumed-from-1gpu"
    RECS="$RECS $OUT/$TAG-two.json $OUT/$TAG-two-from-one.json"
    GRADS="$GRADS $OUT/$TAG-two.npz"
else
    say "ONE GPU VISIBLE: no two-device run; this leg then states nothing about multi-GPU"
fi
APPLE=$(ls bench/results/par_lm_xvendor/*/apple-m4-1gpu.handoff.npz 2>/dev/null | tail -1)
if [ -n "$APPLE" ]; then
    say "apple_handoff=$APPLE"
    run from-apple $PY run --out "$OUT/$TAG-from-apple.json" --devices 0 --resume "$APPLE" \
        --label "$TAG-1gpu-resumed-from-apple-m4"
    RECS="$RECS $OUT/$TAG-from-apple.json"
    cp "$(dirname "$APPLE")/apple-m4-1gpu.json" "$OUT/apple-m4-1gpu.json" 2>/dev/null \
        && RECS="$RECS $OUT/apple-m4-1gpu.json"
fi

# ------------------------------------------------------------ compare, on the box
# shellcheck disable=SC2086  # lists of paths, split on purpose
run compare $PY compare $RECS --grads $GRADS
tail -40 "$OUT/logs/compare.log" >> "$G"
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
