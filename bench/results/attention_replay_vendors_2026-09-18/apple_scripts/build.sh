#!/bin/bash
# Build the Apple native gate binaries (CPU slot, one worker). No Metal work.
set -u
cd /Users/andrewhendel/mojolearn-wt/attention-replay-vendors
B=/Users/andrewhendel/mojolearn-evidence/attention-replay-vendors/apple-native/bin
mkdir -p "$B"
unset MACOSX_DEPLOYMENT_TARGET
BASE="-D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_APPLE"
b() { name=$1; src=$2; shift 2
  t0=$(date +%s)
  pixi run mojo build -j 1 $BASE "$@" -I . "$src" -o "$B/$name" > "$B/$name.build.log" 2>&1
  echo "$name exit=$? seconds=$(( $(date +%s)-t0 )) defines=$*" >> "$B/build_status.txt"
}
M=transformer/checks/attention_masked_tail_check.mojo
T=transformer/checks/attention_tail_guard_check.mojo
b masked_clean $M
b masked_rowoff $M -D MOJOLEARN_ATTN_LEGACY_CORNER=1
b masked_sab_z $M -D MOJOLEARN_ATTN_REPAIR_SAB_Z=1
b masked_sab_dq $M -D MOJOLEARN_ATTN_REPAIR_SAB_DQ=1
b masked_corrupt_z $M -D MOJOLEARN_CHECK_CORRUPT_Z_PRESERVE=1
b masked_corrupt_dq $M -D MOJOLEARN_CHECK_CORRUPT_DQ_PRESERVE=1
b tail_clean $T
b tail_sab $T -D MOJOLEARN_ATTN_TAIL_GUARD_SABOTAGE=1
b fused_default transformer/checks/transformer_fused_check.mojo
echo BUILD-DONE >> "$B/build_status.txt"
