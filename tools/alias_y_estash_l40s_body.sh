#!/usr/bin/env bash
set -euo pipefail
cd /root/mojolearn
export PATH="$HOME/.pixi/bin:$PATH"
out=/root/jobs/alias-y-estash
mkdir -p "$out"

arm=stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32
common='-D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_ATTN_ARM_TRIAL=1'
pixi run mojo build -j 2 $common -I . bench/attention_step_price_main.mojo -o "$out/baseline"
pixi run mojo build -j 2 $common -D MOJOLEARN_ATTN_V1_ALIAS_Y_ESTASH=1 -I . bench/attention_step_price_main.mojo -o "$out/alias"

export MOJOLEARN_ATTN_ARM="$arm" MOJOLEARN_ATTN_BASELINE="$arm"
export MOJOLEARN_ATTN_KINDS=hashed MOJOLEARN_ATTN_ORACLE=1 MOJOLEARN_ATTN_REACH=0
export MOJOLEARN_ATTN_RESOURCES=0 MOJOLEARN_ATTN_WARMUPS=3 MOJOLEARN_ATTN_ROUNDS=9
for spec in b1-l2048-w0 b1-l1024-w0 b1-l2048-w512; do
  case "$spec" in
    b1-l2048-w0) b=1; l=2048; w=0 ;;
    b1-l1024-w0) b=1; l=1024; w=0 ;;
    b1-l2048-w512) b=1; l=2048; w=512 ;;
  esac
  export MOJOLEARN_ATTN_B="$b" MOJOLEARN_ATTN_L="$l" MOJOLEARN_ATTN_WINDOW="$w"
  for round in 0 1; do
    if (( round == 0 )); then order=(baseline alias); else order=(alias baseline); fi
    for profile in "${order[@]}"; do
      "$out/$profile" > "$out/${spec}-${round}-${profile}.log" 2>&1
    done
  done
done
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$out/gpu.txt"
sha256sum "$out"/*.log > "$out/sha256.txt"
