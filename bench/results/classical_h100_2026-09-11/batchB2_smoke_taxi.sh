#!/bin/sh
# pod 2, step 2: kde and svc on taxi at the smoke shape, 1 round
# Runs on the pod: nohup sh bench/results/classical_h100_2026-09-11/batchB2_smoke_taxi.sh > /root/ctd_out/batchB2_smoke_taxi.console 2>&1 &
cd /root/mojolearn || exit 9
mkdir -p /root/ctd_out
exec env MOJOLEARN_CTD_OUT=/root/ctd_out MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_CTD_BODY_SECONDS=4200 \
    MOJOLEARN_COMPILE_JOBS="$(nproc)" MOJOLEARN_CTD_LANES=kde,svc MOJOLEARN_CTD_PHASES="prep races" \
    MOJOLEARN_CTD_DATASETS=taxi MOJOLEARN_CTD_SMOKE_ROWS=20000 sh tools/classical_two_datasets_leg.sh
