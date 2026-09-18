#!/bin/sh
# Three target-shape 700-step arms: baseline, early eager routing, release.
set -eu
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/lm-attention-policy
mkdir -p "$OUT"
cd "$ROOT"
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_GPU_ARCHS=sm_90a
export PYTHONPATH="$ROOT/python:$ROOT"
CORPUS=training/corpus/enwik8/input.txt
printf '%s  %s\n' 2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8 "$CORPUS" | sha256sum -c - > "$OUT/corpus_verified.log"
test "$(wc -c < "$CORPUS" | tr -d ' ')" = 100000000
rm -f python/mojolearn/identical/_mojolearn.so python/mojolearn/identical/_mojolearn_byte_lm.so
MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh > "$OUT/build_base.log" 2>&1
for arm in baseline sticky released; do
    rm -f python/mojolearn/identical/_mojolearn_byte_lm.so
    defines=""
    if [ "$arm" = sticky ]; then defines="-D MOJOLEARN_BYTE_LM_STICKY_EAGER=1"; fi
    if [ "$arm" = released ]; then defines="-D MOJOLEARN_BYTE_LM_STICKY_EAGER=1 -D MOJOLEARN_BYTE_LM_RELEASE_EAGER=1"; fi
    MOJOLEARN_BUILD_EXTRA_DEFINES="$defines" sh bindings/build_byte_lm.sh > "$OUT/build_$arm.log" 2>&1
    pixi run python tools/lm_ce_alias_probe.py --out "$OUT/$arm" --steps 700 --tail 0 --corpus "$CORPUS" --smi-every 10 --witness-every 699 > "$OUT/$arm.log" 2>&1
    echo "$arm complete" >> "$OUT/status.txt"
done
pixi run python tools/lm_attention_policy_compare.py "$OUT" > "$OUT/verdict.log" 2>&1
cat "$OUT/verdict.log"
