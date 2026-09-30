#!/bin/sh
# Step 2 only: distinct build arms, exact bits, and 700-step measurements.
set -eu
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/lm-attention-tail
mkdir -p "$OUT"
cd "$ROOT"
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_TARGET_COLUMN=nvidia
export MOJOLEARN_GPU_ARCHS=sm_90a
export PYTHONPATH="$ROOT/python:$ROOT"
CORPUS=training/corpus/enwik8/input.txt
printf '%s  %s\n' 2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8 "$CORPUS" | sha256sum -c - > "$OUT/corpus_verified.log"
test "$(wc -c < "$CORPUS" | tr -d ' ')" = 100000000
TARGET="1 2048 768 12 12 64 2048 12 50257"
rm -f python/mojolearn/identical/_mojolearn.so python/mojolearn/identical/_mojolearn_byte_lm.so
MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh > "$OUT/build_base.log" 2>&1
# Run the deliberately wrong native arm FIRST, and require the exact failure.
DEFS="-D MOJOLEARN_ATTN_LEGACY_CORNER=1 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_NVIDIA -D MOJOLEARN_ATTN_EXACT_TAIL_GUARD=1"
if pixi run mojo run -j 2 --target-accelerator sm_90a $DEFS -D MOJOLEARN_ATTN_TAIL_GUARD_SABOTAGE=1 -I . transformer/checks/attention_tail_guard_check.mojo > "$OUT/sabotage.log" 2>&1; then
    echo 'BLIND: native negative-zero sabotage passed'; exit 1
fi
grep 'accepted dk differs' "$OUT/sabotage.log"
echo 'EXPECTED FAIL: native negative-zero canonicalization' > "$OUT/status.txt"
pixi run mojo run -j 2 --target-accelerator sm_90a $DEFS -I . transformer/checks/attention_tail_guard_check.mojo > "$OUT/tail_gate.log" 2>&1
pixi run mojo run -j 2 --target-accelerator sm_90a $DEFS -I . transformer/checks/transformer_fused_check.mojo > "$OUT/fused_gate.log" 2>&1
for arm in legacy guarded; do
    rm -f python/mojolearn/identical/_mojolearn_byte_lm.so
    defines="-D MOJOLEARN_ATTN_LEGACY_CORNER=1 -D MOJOLEARN_BYTE_LM_RETAIN_EAGER=1"
    if [ "$arm" = guarded ]; then defines="-D MOJOLEARN_ATTN_LEGACY_CORNER=1 -D MOJOLEARN_BYTE_LM_RETAIN_EAGER=1 -D MOJOLEARN_ATTN_EXACT_TAIL_GUARD=1"; fi
    MOJOLEARN_BUILD_EXTRA_DEFINES="$defines" sh bindings/build_byte_lm.sh > "$OUT/build_$arm.log" 2>&1
    pixi run python tools/lm_ce_alias_probe.py --out "$OUT/$arm" --shape $TARGET --steps 700 --tail 0 --corpus "$CORPUS" --smi-every 10 --witness-every 699 > "$OUT/$arm.log" 2>&1
    echo "$arm complete" >> "$OUT/status.txt"
done
pixi run python tools/lm_attention_tail_compare.py "$OUT" > "$OUT/verdict.log" 2>&1
cat "$OUT/verdict.log"
