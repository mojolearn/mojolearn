#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# One isolated GPU: batched-scan correctness, forward/state/refusal outputs,
# and one selected backward case. Runner owns the bounded lease and teardown.
set -eu
cd /root/mojolearn
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MOJOLEARN_CPU_THREADS=1 MOJOLEARN_COMPILE_JOBS=2 MOJOLEARN_BUILD_JOBS=1
out=/root/gemm_leg_out/transformer-setup
mkdir -p "$out"
exec > "$out/checks.log" 2>&1
case "$MOJOLEARN_TARGET_COLUMN" in
    amd) backend=hip ;;
    nvidia) backend=cuda ;;
    *) echo 'explicit AMD or NVIDIA column required'; exit 2 ;;
esac
if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
    [ "$backend" = cuda ] || exit 2
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '. ')
    export MOJOLEARN_GPU_ARCHS="sm_$cc"
fi
sha256sum core/device_scan.mojo transformer/impl/llama/modeling_llama.mojo
# Compilation and each execution have independent hard limits; no retry loop.
timeout -k 5s 60s nice -n 19 pixi run mojo build -j 1 \
    --target-accelerator "$MOJOLEARN_GPU_ARCHS" \
    -D MOJOLEARN_STEP_PHASE_TIMERS=1 -I . core/device_scan_batch_check.mojo \
    -o /tmp/scan-batch-check
timeout -k 5s 60s nice -n 19 /tmp/scan-batch-check
timeout -k 5s 60s nice -n 19 pixi run mojo build -j 2 \
    --target-accelerator "$MOJOLEARN_GPU_ARCHS" \
    -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_STEP_PHASE_TIMERS=1 \
    -I . transformer/checks/weight_validation_check.mojo -o /tmp/weights-check
timeout -k 5s 60s nice -n 19 /tmp/weights-check
# Cold full bindings exceeded 60 seconds on both MI300X and H100. Allow
# two minutes for compilation only; every GPU execution still has one minute.
timeout -k 5s 120s nice -n 19 sh bindings/build_transformer.sh
binding=python/mojolearn/identical/_mojolearn_transformer.so
for group in outputs refusals; do
    timeout -k 5s 60s nice -n 19 pixi run python tools/transformer_setup_check.py \
        --binding "$binding" --backend "$backend" --group "$group" \
        --out "$out/$group.npz"
done
timeout -k 5s 60s nice -n 19 pixi run python tools/transformer_readback_check.py \
    --binding "$binding" --backend "$backend" --case small \
    --out "$out/backward.npz"
printf 'PASS transformer setup checks\n'
