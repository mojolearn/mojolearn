#!/bin/sh
# AMD attention round 3 leg body for tools/gemm_remote_leg.sh (RunPod MI300X, 60-minute cap).
# tools/attention_round3_leg.sh keeps values already set, so this trims the LM
# arms to the reference, the H100 winner and _pf alone; the price still covers
# all five round 3 arms against stash_tiled. gemm_remote_leg.sh does not export
# the GPU arch or the column into the extra body, and the attention leg refuses
# AMD without the arch, so both are set here (MI300X = gfx942).
set -u
cd /root/mojolearn || exit 9
MOJOLEARN_GPU_ARCHS=${MOJOLEARN_GPU_ARCHS:-gfx942} \
MOJOLEARN_TARGET_COLUMN=amd \
MOJOLEARN_ATTN_LEG_LM_ARMS=stash_tiled,stash_tiled_fgrid_r32_qres_pf,stash_tiled_pf \
MOJOLEARN_COMPILE_JOBS=8 \
    sh tools/attention_round3_leg.sh
