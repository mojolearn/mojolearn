#!/bin/sh
# tools/byte_lm_gpu_logits_leg.sh -- DEVIATION 2658: the byte LM's GPU logits
# against the CPU reference path, byte for byte, on a rented GPU. A
# vendor-agnostic body that runs ON THE BOX from /root/mojolearn with pixi on
# PATH; everything under /root/gemm_leg_out/gpu-logits/ comes home with the
# leg's fetch.
#
# AMD (DigitalOcean MI325X):
#   MOJOLEARN_DO_TOKEN_FILE=$HOME/.mojolearn_do_token \
#   MOJOLEARN_GPU_ARCHS=gfx942 \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/byte_lm_gpu_logits_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/<UTC stamp>-amd-mi325x-do-gpu-logits \
#   bash tools/do_extra_leg.sh amd --minutes 60 --skip-gates
#
# NVIDIA (RunPod H100):
#   MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
#   MOJOLEARN_GPU_ARCHS=sm_90a \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/byte_lm_gpu_logits_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/<UTC stamp>-nvidia-h100-gpu-logits \
#   sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 \
#       --gpu "NVIDIA H100 80GB HBM3" --local-card <existing card>
#
# Neither runner ships bench/results, and RunPod has no upload hook, so the
# three parameter states the sweep reads are fetched from GitHub at the leg's
# own commit (the `commit=` line of /root/gemm_leg_out/leg.txt) and verified
# against the SHA-256 their capture manifests record. The runner on RunPod
# exports neither the target column nor the GPU arch, so the body derives
# both from the box, as tools/step_breakdown_leg.sh does.
#
# PHASES, each with its exit code and seconds in status.tsv; a later phase
# runs even when an earlier one fails, because a red phase is a finding:
#   build-base        bindings/build.sh (the package import needs it)
#   build-byte-lm     bindings/build_byte_lm.sh for this box's GPU
#   build-host        bindings/build_byte_lm_host.sh (no GPU arch)
#   fetch-params      the three parameter states, verified
#   gpu-logits-sweep  tools/byte_lm_gpu_logits_sweep.py, GPU logits, next
#                     bytes and losses against LanguageModelInference with
#                     threaded=False, at every batch 1..8 and length 1..32
#   cpu-path-sweep    tools/byte_lm_host_path_sweep.py on this box's CPU
#
# POSIX sh only: RunPod's Ubuntu images link /bin/sh to dash.
set -u
ROOT=${MOJOLEARN_LOGITS_LEG_ROOT:-/root/mojolearn}
OUT=${MOJOLEARN_LOGITS_LEG_OUT:-/root/gemm_leg_out/gpu-logits}
STATELESS_EVERY=${MOJOLEARN_LOGITS_LEG_STATELESS_EVERY:-8}
REPO_RAW=https://raw.githubusercontent.com/mojolearn/mojolearn
CAPTURE=bench/results/resume/2026-09-07-root-byte-lm-three-vendor/apple/full128
mkdir -p "$OUT"
cd "$ROOT" || exit 9
export PATH="$HOME/.pixi/bin:$PATH"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export MOJOLEARN_NUMERIC_MODE=identical
STATUS="$OUT/status.tsv"
: > "$STATUS"

phase() {
    phase_name=$1
    shift
    phase_start=$(date +%s)
    "$@" > "$OUT/$phase_name.log" 2>&1
    phase_code=$?
    printf '%s\t%s\t%s\n' "$phase_name" "$phase_code" "$(( $(date +%s) - phase_start ))" >> "$STATUS"
    return $phase_code
}

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    VENDOR=nvidia
    nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader > "$OUT/gpu.txt" 2>&1
    if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
        cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d ' ')
        case "$cap" in
            9.0) MOJOLEARN_GPU_ARCHS=sm_90a ;;
            *) MOJOLEARN_GPU_ARCHS=sm_$(printf '%s' "$cap" | tr -d '.') ;;
        esac
    fi
    MOJOLEARN_TARGET_COLUMN=nvidia
elif [ -e /dev/kfd ] || command -v rocm-smi >/dev/null 2>&1; then
    VENDOR=amd
    (rocm-smi --showproductname 2>&1 || true) > "$OUT/gpu.txt"
    MOJOLEARN_TARGET_COLUMN=${MOJOLEARN_TARGET_COLUMN:-amd}
    if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
        echo 'amd: MOJOLEARN_GPU_ARCHS is required (gfx942 on an MI325X)' > "$OUT/refused.txt"
        exit 2
    fi
else
    echo 'no NVIDIA or AMD GPU detected' > "$OUT/refused.txt"
    exit 2
fi
export MOJOLEARN_GPU_ARCHS MOJOLEARN_TARGET_COLUMN

COMMIT=$(sed -n 's/^commit=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -n 1)
if [ -z "$COMMIT" ]; then
    COMMIT=${MOJOLEARN_LOGITS_LEG_COMMIT:-}
fi
export MOJOLEARN_GATE_COMMIT="$COMMIT"
printf 'vendor=%s\ngpu_archs=%s\ntarget_column=%s\ncommit=%s\n' \
    "$VENDOR" "$MOJOLEARN_GPU_ARCHS" "$MOJOLEARN_TARGET_COLUMN" "$COMMIT" > "$OUT/target.txt"
(lscpu 2>/dev/null || true) > "$OUT/host.txt"
(pixi run mojo --version 2>&1 || true) > "$OUT/mojo_version.txt"

fetch_params() {
    [ -n "$COMMIT" ] || { echo 'no commit to fetch the parameter states at'; return 5; }
    for item in \
        "initial/initial_p.f32 b87a6075597a82c94b68b810bef74f6331b86579d4d434d95455866ada3ba424" \
        "step000064/initial_p.f32 c95553ef5ae7b3715f72ca9b190310f1539e3916f5aa29d5b1277e2d42082f67" \
        "step000128/post_p.f32 e7c9c226dfa550446799ca930480f972ca8942c023df1e0c5c70a1526bfcd258"; do
        set -- $item
        mkdir -p "$CAPTURE/$(dirname "$1")"
        curl -fsSL --retry 3 -o "$CAPTURE/$1" "$REPO_RAW/$COMMIT/$CAPTURE/$1" || return 3
        got=$(sha256sum "$CAPTURE/$1" | cut -d' ' -f1)
        if [ "$got" != "$2" ]; then
            echo "sha256 mismatch for $1: $got"
            return 4
        fi
        echo "verified $1 $got"
    done
    return 0
}

phase build-base env MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh
phase build-byte-lm sh bindings/build_byte_lm.sh
phase build-host env -u MOJOLEARN_GPU_ARCHS sh bindings/build_byte_lm_host.sh
(find python/mojolearn -name '*.so' -exec sha256sum {} \; 2>/dev/null || true) > "$OUT/binaries.sha256"
phase fetch-params fetch_params
phase gpu-logits-sweep pixi run python tools/byte_lm_gpu_logits_sweep.py \
    --stateless-every "$STATELESS_EVERY" --report "$OUT/gpu_logits_sweep.json"
phase cpu-path-sweep pixi run python tools/byte_lm_host_path_sweep.py \
    --threads 1,2,3 --max-batch 8 --report "$OUT/cpu_path_sweep.json"
echo done > "$OUT/done.txt"
exit 0
