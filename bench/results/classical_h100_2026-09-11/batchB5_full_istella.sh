#!/bin/sh
# pod 2, step 5: kde and svc on Istella-S at the kernel-bound shape, 1 warm-up + 5 rounds
# Runs on the pod: nohup sh bench/results/classical_h100_2026-09-11/batchB5_full_istella.sh > /root/ctd_out/batchB5_full_istella.console 2>&1 &
cd /root/mojolearn || exit 9
mkdir -p /root/ctd_out
exec env MOJOLEARN_CTD_OUT=/root/ctd_out MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_CTD_BODY_SECONDS=4200 \
    MOJOLEARN_COMPILE_JOBS="$(nproc)" MOJOLEARN_CTD_LANES=kde,svc MOJOLEARN_CTD_PHASES="prep races" \
    MOJOLEARN_CTD_DATASETS=istella sh tools/classical_two_datasets_leg.sh
