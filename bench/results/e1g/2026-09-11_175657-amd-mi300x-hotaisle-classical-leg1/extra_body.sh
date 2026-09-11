#!/bin/sh
# AMD classical ksplit A/B, leg 1 (svc, kmeans, pca, kde), for any AMD runner.
# tools/gemm_remote_leg.sh has no extra-env plumbing and does not export the GPU
# arch, so the lane list and the arch default are set here; the Hot Aisle and
# DigitalOcean runners keep whatever they already exported.
set -u
cd /root/mojolearn || exit 9
MOJOLEARN_GPU_ARCHS=${MOJOLEARN_GPU_ARCHS:-gfx942} \
MOJOLEARN_TARGET_COLUMN=amd \
MOJOLEARN_CLASSICAL_AB_LANES=${MOJOLEARN_CLASSICAL_AB_LANES:-svc,kmeans,pca,kde} \
    sh tools/gemm_ksplit_classical_amd_leg.sh
