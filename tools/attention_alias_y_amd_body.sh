#!/bin/sh
set -eu
cd /root/mojolearn
export PATH="$HOME/.pixi/bin:$PATH"
out=/root/gemm_leg_out
mkdir -p "$out"

estash_arm=stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32
common='-D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_AMD=1 -D MOJOLEARN_ATTN_ARM_TRIAL=1'
pixi run mojo build -j 2 $common -I . bench/attention_step_price_main.mojo -o "$out/price"
pixi run mojo build -j 2 $common -D MOJOLEARN_ATTN_V1_ALIAS_Y_ESTASH=1 -I . bench/attention_step_price_main.mojo -o "$out/alias"
pixi run mojo build -j 2 $common -D MOJOLEARN_ATTN_PHASE_TIMERS=1 -I . bench/attention_step_price_main.mojo -o "$out/phases"

export MOJOLEARN_ATTN_KINDS=hashed MOJOLEARN_ATTN_ORACLE=1 MOJOLEARN_ATTN_REACH=0
export MOJOLEARN_ATTN_RESOURCES=0 MOJOLEARN_ATTN_B=1 MOJOLEARN_ATTN_L=2048
export MOJOLEARN_ATTN_NH=12 MOJOLEARN_ATTN_NKV=12 MOJOLEARN_ATTN_HD=64 MOJOLEARN_ATTN_WINDOW=0

# Serialized launch-level profile of the shipped production route first.
MOJOLEARN_ATTN_ARM="$estash_arm" MOJOLEARN_ATTN_BASELINE="$estash_arm" \
MOJOLEARN_ATTN_WARMUPS=0 MOJOLEARN_ATTN_ROUNDS=1 MOJOLEARN_TRANSFORMER_TIMING=1 \
  "$out/phases" > "$out/phases-default-b1-l2048.log" 2>&1

# Whole-call alternating process order. Both arms retain the eager oracle and
# seven output hashes; the candidate changes the route and aliases y into the
# consumed full exponent stash.
export MOJOLEARN_ATTN_WARMUPS=3 MOJOLEARN_ATTN_ROUNDS=9
for round in 0 1 2; do
  if [ $((round % 2)) -eq 0 ]; then order="default alias"; else order="alias default"; fi
  for profile in $order; do
    if [[ "$profile" == default ]]; then
      MOJOLEARN_ATTN_ARM="$estash_arm" MOJOLEARN_ATTN_BASELINE="$estash_arm" \
        "$out/price" > "$out/ab-${round}-${profile}.log" 2>&1
    else
      MOJOLEARN_ATTN_ARM="$estash_arm" MOJOLEARN_ATTN_BASELINE="$estash_arm" \
        "$out/alias" > "$out/ab-${round}-${profile}.log" 2>&1
    fi
  done
done

rocminfo > "$out/rocminfo.txt" 2>&1 || true
sha256sum "$out"/*.log > "$out/sha256.txt"
