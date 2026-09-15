#!/bin/sh
# tools/hdbscan_soft_nvidia_leg.sh: the on-box body (MOJOLEARN_GEMM_LEG_EXTRA for
# tools/gemm_remote_leg.sh nvidia) of the HDBSCAN soft clustering leg, 2026-09-15.
#
#   MOJOLEARN_GEMM_LEG_GPU_NVIDIA='NVIDIA H100 80GB HBM3' \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/hdbscan_soft_nvidia_leg.sh \
#   sh tools/gemm_remote_leg.sh nvidia --rent
#
# On one NVIDIA box: build the hdbscan GPU binding (IDENTICAL, this device's
# architecture), run test_hdbscan_surface, run tools/identity_break.py on the
# hdbscan and hdbscan-leaf lanes (base, ties, dupes; two repeats) for this
# column's train, infer and batch hashes, then install cuML in a venv and run
# tools/hdbscan_soft_cuml_reference.py, which compares membership_vector and
# all_points_membership_vectors with cuML's own (DEVIATION 1616). Every phase
# runs even when an earlier one fails; status.tsv has each exit and seconds.
# POSIX sh: the pod's /bin/sh is dash.
set -u
cd /root/mojolearn || exit 9
OUT=/root/gemm_leg_out/hdbscan-soft
mkdir -p "$OUT/logs"
G="$OUT/gate.txt"
say() { echo "$@" >> "$G"; }
run() {
    _n=$1; shift
    _t0=$(date +%s)
    "$@" > "$OUT/logs/$_n.log" 2>&1
    _e=$?
    echo "$_n	$_e	$(( $(date +%s) - _t0 ))" >> "$OUT/status.tsv"
    return "$_e"
}
say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ -n "${MOJOLEARN_COMMIT:-}" ] && [ ! -s /root/mojolearn/commit.txt ]; then echo "$MOJOLEARN_COMMIT" > /root/mojolearn/commit.txt; fi
say "commit=$(cat /root/mojolearn/commit.txt 2>/dev/null | head -1)"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader > "$OUT/logs/device.txt" 2>&1
say "device=$(tr '\n' ' ' < "$OUT/logs/device.txt")"
if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
    _cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
    case "$_cc" in 9.0) MOJOLEARN_GPU_ARCHS=sm_90a ;; 8.9) MOJOLEARN_GPU_ARCHS=sm_89 ;; 8.6) MOJOLEARN_GPU_ARCHS=sm_86 ;; 8.0) MOJOLEARN_GPU_ARCHS=sm_80 ;; 12.0) MOJOLEARN_GPU_ARCHS=sm_120a ;; *) MOJOLEARN_GPU_ARCHS="sm_$(echo "$_cc" | tr -d .)" ;; esac
    export MOJOLEARN_GPU_ARCHS
fi
say "gpu_archs=$MOJOLEARN_GPU_ARCHS"
VENDOR_LABEL="nvidia-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr 'A-Z ' 'a-z-' | tr -cd 'a-z0-9-' | cut -c1-24)-$MOJOLEARN_GPU_ARCHS"
say "vendor_label=$VENDOR_LABEL"

run build-hdbscan env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 \
    MOJOLEARN_COMPILE_JOBS="${MOJOLEARN_COMPILE_JOBS:-8}" sh bindings/build_hdbscan.sh
sha256sum python/mojolearn/identical/_mojolearn_hdbscan.so >> "$G" 2>/dev/null

run surface env MOJOLEARN_NUMERIC_MODE=identical sh -c 'cd python && pixi run python -m mojolearn.tests.test_hdbscan_surface'
grep -E 'GREEN|RED|FAIL' "$OUT/logs/surface.log" | head -20 >> "$G"

run identity env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
    pixi run python tools/identity_break.py --lanes hdbscan,hdbscan-leaf --fixtures base,ties,dupes \
    --repeats 2 --vendor "$VENDOR_LABEL" --json "$OUT/$VENDOR_LABEL.json"
grep -E '^cells=|^summary|MOVED|DIVERGENT|REFUSED' "$OUT/logs/identity.log" | head -20 >> "$G"

VENV=/root/cuml-venv
run cuml-venv python3 -m venv --system-site-packages "$VENV"
run cuml-wheels "$VENV/bin/python" -m pip install --disable-pip-version-check --no-input --only-binary=:all: \
    numpy==2.4.6 cupy-cuda12x==14.2.0 cuml-cu12==26.8.0 --extra-index-url https://pypi.nvidia.com
run cuml-freeze "$VENV/bin/python" -m pip freeze
run cuml-reference env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
    "$VENV/bin/python" tools/hdbscan_soft_cuml_reference.py --out "$OUT/cuml-reference.json"
tail -8 "$OUT/logs/cuml-reference.log" >> "$G" 2>/dev/null
cat "$OUT/status.tsv" >> "$G"
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
