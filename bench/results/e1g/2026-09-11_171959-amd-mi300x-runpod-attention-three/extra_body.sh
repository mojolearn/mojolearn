#!/bin/sh
# AMD attention three-way leg body (any AMD runner): baseline, stash_tiled and the
# round 3 winner on ONE clean 1x AMD box. The shared 2x Hot Aisle VM on
# 2026-09-11 (e1g/2026-09-11_165905-amd-mi300x-2gpu-vm-hotaisle-attention-stash-tiled)
# read stash_tiled at 1.16x baseline, and the AMD default was chosen from
# stash_tiled-relative legs, so the three need one box. gemm_remote_leg.sh does
# not export the GPU arch, so it defaults here (MI300X and MI325X are gfx942).
set -u
cd /root/mojolearn || exit 9
MOJOLEARN_GPU_ARCHS=${MOJOLEARN_GPU_ARCHS:-gfx942} \
MOJOLEARN_TARGET_COLUMN=amd \
MOJOLEARN_ATTN_BASELINE=baseline \
MOJOLEARN_ATTN_LEG_ARMS=stash_tiled,stash_tiled_fgrid_r32_qres_pf \
MOJOLEARN_ATTN_LEG_LM_ARMS=baseline,stash_tiled,stash_tiled_fgrid_r32_qres_pf \
MOJOLEARN_ATTN_LEG_SKIP_TIMERS=1 \
MOJOLEARN_COMPILE_JOBS=8 \
    sh tools/attention_step_leg.sh
