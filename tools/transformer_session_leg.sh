#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Narrow retained-setup qualification. Launch the guarded cloud runner with
# MOJOLEARN_STAGE_KEYS='' (these synthetic checks need no corpora).
set -eu
cd /root/mojolearn
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MOJOLEARN_CPU_THREADS=1 MOJOLEARN_COMPILE_JOBS=2 MOJOLEARN_BUILD_JOBS=1
out=/root/gemm_leg_out/transformer-session
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
sha256sum bindings/_mojolearn_transformer.mojo \
    transformer/impl/llama/modeling_llama.mojo python/mojolearn/_transformer_impl.py
# Compilation has its own cap; no GPU diagnostic may run beyond one minute.
timeout -k 5s 120s nice -n 19 sh bindings/build_transformer.sh
binding=python/mojolearn/identical/_mojolearn_transformer.so
for group in reuse refusals lifetime budget; do
    timeout -k 5s 60s nice -n 19 pixi run python tools/transformer_session_check.py \
        --binding "$binding" --backend "$backend" --group "$group" \
        --out "$out/native-$group.npz"
done
for group in state serialization threads; do
    # ROCm 6.4.1/MI300X crashes in HIP exit handlers when the FIRST GPU call
    # runs on a worker, including with legacy setup. Keep that cold-worker
    # diagnostic selectable; normal ownership qualification initializes on
    # main before dispatch.
    thread_init=worker
    if [ "$backend" = hip ]; then thread_init=main; fi
    thread_init=${MOJOLEARN_SESSION_THREAD_INIT:-$thread_init}
    timeout -k 5s 60s nice -n 19 pixi run python tools/transformer_session_surface_check.py \
        --binding "$binding" --backend "$backend" --group "$group" \
        --thread-init "$thread_init" \
        --out "$out/surface-$group.npz"
done
printf 'PASS transformer session checks\n'
