#!/bin/sh
# Native identity only, never an AMD speed claim or a short training run.
set -eu
cd /root/mojolearn
OUT=/root/gemm_leg_out/lm-attention-amd-identity
mkdir -p "$OUT"
export PATH="$HOME/.pixi/bin:$PATH"
sh tools/fetch_corpus_enwik8.sh --check > "$OUT/corpus_verified.log"
DEFS="-D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_AMD -D MOJOLEARN_ATTN_REPAIR_MASKED_TAIL=1 -D MOJOLEARN_ATTN_DEFAULT_BSWZ_EVERY_COLUMN=1"
for site in Z DQ; do
    if pixi run mojo run -j 2 --target-accelerator gfx942 $DEFS -D MOJOLEARN_ATTN_REPAIR_SAB_$site=1 -I . transformer/checks/attention_masked_tail_check.mojo > "$OUT/sabotage_$site.log" 2>&1; then
        echo "BLIND repair sabotage $site"; exit 1
    fi
    if [ "$site" = Z ]; then pattern='zdot repair differs'; else pattern='dq repair differs'; fi
    grep "$pattern" "$OUT/sabotage_$site.log"
    if pixi run mojo run -j 2 --target-accelerator gfx942 $DEFS -D MOJOLEARN_CHECK_CORRUPT_${site}_PRESERVE=1 -I . transformer/checks/attention_masked_tail_check.mojo > "$OUT/preserve_sabotage_$site.log" 2>&1; then
        echo "BLIND preserve sabotage $site"; exit 1
    fi
    if [ "$site" = Z ]; then pattern='zdot preserve differs'; else pattern='dq preserve differs'; fi
    grep "$pattern" "$OUT/preserve_sabotage_$site.log"
done
pixi run mojo run -j 2 --target-accelerator gfx942 $DEFS -I . transformer/checks/attention_masked_tail_check.mojo > "$OUT/repair_gate.log" 2>&1
pixi run mojo run -j 2 --target-accelerator gfx942 $DEFS -I . transformer/checks/attention_tail_guard_check.mojo > "$OUT/tail_gate.log" 2>&1
pixi run mojo run -j 2 --target-accelerator gfx942 $DEFS -I . transformer/checks/transformer_fused_check.mojo > "$OUT/fused_gate.log" 2>&1
cat "$OUT/repair_gate.log" "$OUT/tail_gate.log" "$OUT/fused_gate.log"
