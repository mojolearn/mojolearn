#!/bin/sh
# Same-H100 batch multiplier and B4 legacy/default bit comparison, 700 each.
set -eu
cd /root/mojolearn
OUT=/root/gemm_leg_out/lm-attention-batch
mkdir -p "$OUT"
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_GPU_ARCHS=sm_90a
export PYTHONPATH="/root/mojolearn/python:/root/mojolearn"
CORPUS=training/corpus/enwik8/input.txt
sh tools/fetch_corpus_enwik8.sh --check > "$OUT/corpus_verified.log"
rm -f python/mojolearn/identical/_mojolearn.so python/mojolearn/identical/_mojolearn_byte_lm.so
MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh > "$OUT/build_base.log" 2>&1
MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_ATTN_LEGACY_CORNER=1" sh bindings/build_byte_lm.sh > "$OUT/build_legacy.log" 2>&1
pixi run python tools/lm_ce_alias_probe.py --out "$OUT/legacy4" --shape 4 2048 768 12 12 64 2048 12 50257 --steps 700 --tail 0 --corpus "$CORPUS" --smi-every 10 --witness-every 699 > "$OUT/legacy4.log" 2>&1
echo 'legacy4 complete' >> "$OUT/status.txt"
rm -f python/mojolearn/identical/_mojolearn_byte_lm.so
MOJOLEARN_BUILD_EXTRA_DEFINES="" sh bindings/build_byte_lm.sh > "$OUT/build_default.log" 2>&1
pixi run python tools/lm_ce_alias_probe.py --out "$OUT/default1" --steps 700 --tail 0 --corpus "$CORPUS" --smi-every 10 --witness-every 0 > "$OUT/default1.log" 2>&1
echo 'default1 complete' >> "$OUT/status.txt"
pixi run python tools/lm_ce_alias_probe.py --out "$OUT/default4" --shape 4 2048 768 12 12 64 2048 12 50257 --steps 700 --tail 0 --corpus "$CORPUS" --smi-every 10 --witness-every 699 > "$OUT/default4.log" 2>&1
echo 'default4 complete' >> "$OUT/status.txt"
pixi run python tools/lm_attention_batch_compare.py "$OUT" > "$OUT/verdict.log" 2>&1
cat "$OUT/verdict.log"
