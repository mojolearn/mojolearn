#!/usr/bin/env bash
set -euo pipefail
cd /root/mojolearn
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_ATTN_ARM=stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32
export MOJOLEARN_ATTN_BASELINE=stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32
export MOJOLEARN_ATTN_KINDS=hashed MOJOLEARN_ATTN_ORACLE=0 MOJOLEARN_ATTN_REACH=0
export MOJOLEARN_ATTN_RESOURCES=0 MOJOLEARN_ATTN_WARMUPS=3 MOJOLEARN_ATTN_ROUNDS=9
mkdir -p /root/jobs/packed-estash
for profile in estash packed recompute; do
  defs='-D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_ATTN_ARM_TRIAL=1'
  if [[ "$profile" == packed ]]; then defs="$defs -D MOJOLEARN_ATTN_V1_PACKED_ESTASH=1"; fi
  if [[ "$profile" == recompute ]]; then defs="$defs -D MOJOLEARN_ATTN_V1_RECOMPUTE_BACKWARD=1"; fi
  pixi run mojo build -j 2 $defs -I . bench/attention_step_price_main.mojo -o "/root/jobs/packed-estash/$profile"
  "/root/jobs/packed-estash/$profile" > "/root/jobs/packed-estash/$profile.log" 2>&1
done
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > /root/jobs/packed-estash/gpu.txt
sha256sum /root/jobs/packed-estash/*.log > /root/jobs/packed-estash/sha256.txt
