#!/bin/sh
# pod 2 (leg 2: kde, svc), step 1: env, pip (cuML 26.08), downloads, IDENTICAL bindings
# Runs on the pod: nohup sh bench/results/classical_h100_2026-09-11/batchB1_setup.sh > /root/ctd_out/batchB1_setup.console 2>&1 &
cd /root/mojolearn || exit 9
mkdir -p /root/ctd_out
exec env MOJOLEARN_CTD_OUT=/root/ctd_out MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_CTD_BODY_SECONDS=4200 \
    MOJOLEARN_COMPILE_JOBS="$(nproc)" MOJOLEARN_CTD_LANES=kde,svc MOJOLEARN_CTD_PHASES="setup" \
    MOJOLEARN_CTD_DATASETS=taxi sh tools/classical_two_datasets_leg.sh
