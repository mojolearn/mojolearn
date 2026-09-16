#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Narrow native GPU checks for decode setup and the shared nonfinite scan.
set -eu
cd /root/mojolearn
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 MOJOLEARN_CPU_THREADS=1
export MOJOLEARN_COMPILE_JOBS=1 MOJOLEARN_BUILD_JOBS=1
out=/root/gemm_leg_out/neural-runtime
mkdir -p "$out"
exec > "$out/checks.log" 2>&1
printf 'GPU architecture: %s\n' "$MOJOLEARN_GPU_ARCHS"
sha256sum core/device_scan.mojo core/device_mutex.mojo \
    mamba/impl/modeling/modeling_mamba.mojo transformer/impl/llama/modeling_llama.mojo
for check in stage-init stage-init-poison device-scan device-mutex; do
    case "$check" in
        stage-init) source=training/checks/neural_stage_init.mojo; poison= ;;
        stage-init-poison) source=training/checks/neural_stage_init.mojo; poison='-D MOJOLEARN_MAMBA_POISON=1' ;;
        device-scan) source=core/device_scan_check.mojo; poison= ;;
        device-mutex) source=core/device_mutex_check.mojo; poison= ;;
    esac
    # The only optional word split is the fixed diagnostic define above.
    nice -n 19 pixi run mojo build -j 1 --target-accelerator "$MOJOLEARN_GPU_ARCHS" \
        -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_STEP_PHASE_TIMERS=1 \
        $poison -I . "$source" -o "/tmp/neural-$check"
    nice -n 19 "/tmp/neural-$check"
done
nice -n 19 sh bindings/build_transformer.sh
nice -n 19 sh bindings/build_mamba.sh
nice -n 19 pixi run python tools/bench_neural_decode.py \
    --bindings python/mojolearn/identical --reps 1 --out "$out/candidate.npz"
printf 'PASS neural GPU runtime checks\n'
